import os
import re
import json
import math
import unicodedata
import logging
import datetime
import asyncio
from itertools import combinations
from typing import Any, Dict, List, Tuple

import httpx
from dateutil.relativedelta import relativedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from thefuzz import process

# Setup
load_dotenv(".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API keys and constants
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_KEY   = os.getenv("GROQ_API_KEY")
RAWG_KEY   = os.getenv("RAWG_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

if not GROQ_KEY:
    logger.error("Env - GROQ_API_KEY missing")
if not RAWG_KEY:
    logger.error("Env - RAWG_KEY missing")

# Mapping platform -> RAWG id
PLATFORM_IDS = {
    "pc": 4, "windows": 4,
    "ps5": 187, "playstation 5": 187,
    "ps4": 18, "playstation 4": 18,
    "switch": 7, "nintendo switch": 7,
    "xbox one": 1, "xbox series x": 186,
}

# caches
VALID_GENRES: Dict[str, str] = {}
VALID_TAGS:   Dict[str, str] = {}

# Helpers

# Normalize text into a slug for lookup
def slugify(text: str) -> str:
    txt = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", txt.lower()).strip("-")


# Filter a list of tokens against a valid-slug lookup
def filter_valid(tokens: List[str], lookup: Dict[str, str]) -> List[str]:
    seen, out = set(), []
    for t in tokens:
        slug = lookup.get(slugify(t))
        if slug and slug not in seen:
            seen.add(slug)
            out.append(slug)
    return out


# Call the Groq LLM endpoint, with up to 3 retries on 503
async def call_groq(system: str, user: str, expect_json: bool = True) -> Any:
    headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.0,
    }
    for attempt in range(3):
        async with httpx.AsyncClient(timeout=40) as cli:
            r = await cli.post(GROQ_URL, headers=headers, json=payload)
        if r.status_code == 503:
            logger.warning("Groq 503 – retry %d/3", attempt + 1)
            await asyncio.sleep(1 + attempt)
            continue
        if r.status_code != 200:
            logger.error("Groq error %s – %s", r.status_code, r.text)
            raise HTTPException(502, "LLM failure")
        content = r.json()["choices"][0]["message"]["content"]
        if expect_json:
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content).strip()
            return json.loads(content)
        return content.strip()
    raise HTTPException(502, "LLM failure after retries")


# Translate text into a specified language via the LLM
async def translate(text: str, to_lang: str) -> str:
    system = "You are a professional translator."
    prompt = f"Translate this into {to_lang} only:\n'''{text}'''"
    return await call_groq(system, prompt, expect_json=False)


# Build a date range filter from user input (text patterns first, then LLM hints)
def build_date_filter(norm_lower: str,
                      explicit: Dict[str, int] | None) -> Dict[str, int]:
    txt = norm_lower.lower()
    today = datetime.date.today()

    # last N years
    m = re.search(r"last\s+(\d+)\s+years?", txt)
    if m:
        yrs = int(m.group(1))
        start = today - relativedelta(years=yrs)
        return {"from": start.year, "to": today.year}

    # from XXXX to YYYY or between XXXX and YYYY
    m = re.search(r"(?:from|between)\s+(\d{4})\s+(?:to|and)\s+(\d{4})", txt)
    if m:
        y1, y2 = sorted(map(int, m.groups()))
        return {"from": y1, "to": y2}

    # after / since / later than / post YYYY
    m = re.search(r"(?:after|since|later than|post)(?:\s+the\s+year)?\s+(\d{4})", txt)
    if m:
        year = int(m.group(1))
        return {"from": year, "to": today.year}

    # before / until / prior to YYYY
    m = re.search(r"(?:before|until|prior to)\s+(\d{4})", txt)
    if m:
        return {"from": 1900, "to": int(m.group(1))}

    # Fallback to explicit JSON from LLM
    if explicit:
        frm = explicit.get("from")
        to  = explicit.get("to")
        if frm and to:
            return {"from": frm, "to": to}
        if frm:
            return {"from": frm, "to": today.year}
        if to:
            return {"from": 1900, "to": to}

    return {}


# Extract genre slugs from a game record
def all_genre_slugs_of_game(g: Dict[str, Any]) -> set[str]:
    return {x["slug"] for x in g.get("genres", [])}


# Extract tag slugs from a game record
def all_tag_slugs_of_game(g: Dict[str, Any]) -> set[str]:
    return {x["slug"] for x in g.get("tags", [])}

# FastAPI startup / caches
@app.on_event("startup")
async def startup() -> None:
    async with httpx.AsyncClient(timeout=15) as cli:
        resp = await cli.get("https://api.rawg.io/api/genres", params={"key": RAWG_KEY})
    if resp.status_code == 200:
        for g in resp.json()["results"]:
            VALID_GENRES[slugify(g["name"])] = g["slug"]
        logger.info("Loaded %d genres", len(VALID_GENRES))

    tag_file = os.path.join(os.path.dirname(__file__), "rawg_tags.json")
    with open(tag_file, encoding="utf-8") as f:
        VALID_TAGS.update(json.load(f))
    logger.info("Loaded %d tags from %s", len(VALID_TAGS), tag_file)

# Schemas
class ChatReq(BaseModel):
    message: str

# Endpoint
@app.post("/chat")
async def chat(req: ChatReq):
    original = req.message.strip()
    is_pt    = bool(re.search(r"[áéíóúãõ]", original.lower()))

    # 1) translate to EN for processing
    norm = await translate(original, "English") if is_pt else original
    logger.info("Query EN: %s", norm)

    # 2) detect reference game
    async with httpx.AsyncClient(timeout=10) as cli:
        r0 = await cli.get(
            "https://api.rawg.io/api/games",
            params={"key": RAWG_KEY, "search": norm, "page_size": 1}
        )
    base_game = (r0.json().get("results") or [None])[0]

    # 3) extract tags/genres via LLM
    base_ctx = ""
    if base_game:
        desc = (base_game.get("description_raw") or "")[:250].replace("\n", " ")
        genres_txt = ", ".join(g["name"] for g in base_game.get("genres", []))
        base_ctx = f"Reference game: {base_game['name']}. Genres: {genres_txt}. Short description: {desc}."
    system_prompt = (
        "You are a RAWG preference extractor. Return ONLY JSON with:\n"
        "- tags: array ordered by relevance (max 8)\n"
        "- genres: array ordered by relevance (max 5)\n"
        "- dates: {from:int,to:int} if user constrained years\n"
        "- platforms: list of platform names if user mentioned\n"
        + ("\n" + base_ctx if base_ctx else "")
    )
    llm = await call_groq(system_prompt, norm, expect_json=True)
    tags_raw       = llm.get("tags",   [])[:8]
    genres_raw     = llm.get("genres", [])[:5]
    date_filter    = build_date_filter(norm.lower(), llm.get("dates"))
    platform_names = llm.get("platforms", [])

    # 4) fuzzy-match platforms
    platform_ids = {
        PLATFORM_IDS[p.lower()]
        for p, score in process.extract(
            " ".join(platform_names) + " " + norm,
            PLATFORM_IDS.keys(),
            limit=8
        )
        if score >= 80
    }

    # 5) slugs & dynamic weights
    tag_slugs   = filter_valid(tags_raw,   VALID_TAGS)
    genre_slugs = filter_valid(genres_raw, VALID_GENRES)
    if not tag_slugs and base_game:
        tag_slugs = [t["slug"] for t in base_game.get("tags", [])][:4]
    if not genre_slugs and base_game:
        genre_slugs = [g["slug"] for g in base_game.get("genres", [])][:3]

    TAG_WEIGHTS: Dict[str, int] = {}
    n_tags = len(tag_slugs)
    if n_tags:
        chunk = math.ceil(n_tags / 3)
        for idx, slug in enumerate(tag_slugs):
            TAG_WEIGHTS[slug] = max(1, 3 - (idx // chunk))

    forbidden_slug = base_game["slug"] if base_game else ""

    # 6) generate combos and fetch results
    combos: List[Tuple[List[str], List[str]]] = []
    top_tags = tag_slugs[:4]
    for r in (2, 1):
        for tag_subset in combinations(top_tags, r):
            for k in range(min(3, len(genre_slugs)), 0, -1):
                combos.append((list(tag_subset), genre_slugs[:k]))

    gathered: Dict[str, Dict[str, Any]] = {}
    async with httpx.AsyncClient(timeout=25) as cli:
        for tags_c, genres_c in combos:
            params = {
                "key": RAWG_KEY,
                "ordering": "-rating,-added",
                "page_size": 40,
                "tags":   ",".join(tags_c),
                "genres": ",".join(genres_c),
            }
            if platform_ids:
                params["platforms"] = ",".join(map(str, platform_ids))
            if date_filter:
                params["dates"] = f"{date_filter['from']}-01-01,{date_filter['to']}-12-31"

            logger.info("RAWG combo params: %s", {k: v for k,v in params.items() if k!='key'})
            r = await cli.get("https://api.rawg.io/api/games", params=params)
            if r.status_code != 200:
                continue
            for g in r.json().get("results", []):
                slug = g["slug"]
                if slug == forbidden_slug or slug in gathered:
                    continue
                if set(genres_c).issubset(all_genre_slugs_of_game(g)) and \
                   set(tags_c).issubset(all_tag_slugs_of_game(g)):
                    gathered[slug] = g
            if len(gathered) >= 10:
                break

    # 7) fallback if no results
    if not gathered:
        params = {"key": RAWG_KEY, "ordering": "-rating,-added", "page_size": 40}
        if tag_slugs:
            params["tags"] = ",".join(tag_slugs)
        if genre_slugs:
            params["genres"] = ",".join(genre_slugs)
        if platform_ids:
            params["platforms"] = ",".join(map(str, platform_ids))
        if date_filter:
            params["dates"] = f"{date_filter['from']}-01-01,{date_filter['to']}-12-31"

        logger.info("RAWG fallback params: %s", {k: v for k,v in params.items() if k!='key'})
        async with httpx.AsyncClient(timeout=25) as cli:
            r = await cli.get("https://api.rawg.io/api/games", params=params)
        if r.status_code == 200:
            for g in r.json().get("results", []):
                if g["slug"] != forbidden_slug:
                    gathered.setdefault(g["slug"], g)

    # 8) scoring & select top6
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for g in gathered.values():
        tag_score   = sum(TAG_WEIGHTS.get(t, 0) for t in all_tag_slugs_of_game(g))
        genre_score = len(set(genre_slugs) & all_genre_slugs_of_game(g))
        scored.append((tag_score + genre_score, g))
    scored.sort(reverse=True, key=lambda x: x[0])
    games = [g for _, g in scored][:6]

    # 9) build reply lines
    lines = [
        f"{i}. **{g['name']}** "
        f"({(g.get('released') or '')[:4] or '????'}) – "
        f"Metacritic {g.get('metacritic') or '?'}"
        for i, g in enumerate(games, 1)
    ]
    reply = "\n".join(lines)

    # 10) expose filters
    filters_out: Dict[str, Any] = {}
    if genre_slugs:
        filters_out["genres"] = ",".join(genre_slugs)
    if tag_slugs:
        filters_out["tags"] = ",".join(tag_slugs)
    if platform_ids:
        filters_out["platforms"] = ",".join(map(str, platform_ids))
    if date_filter:
        filters_out["dates"] = f"{date_filter['from']}-01-01,{date_filter['to']}-12-31"

    return {
        "reply": reply,
        "games": games,
        "filters_applied": filters_out,
    }
