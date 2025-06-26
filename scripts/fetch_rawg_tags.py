#!/usr/bin/env python3
"""
scripts/fetch_rawg_tags.py

Standalone script to fetch all RAWG tags via their API and
incrementally save them to backend/rawg_tags.json, logging progress.
"""

import os
import re
import json
import asyncio
import logging
import unicodedata
import sys
from dotenv import load_dotenv
import httpx

# Load RAWG_KEY from environment or .env
load_dotenv()
RAWG_KEY = os.getenv("RAWG_KEY")
if not RAWG_KEY:
    raise RuntimeError("RAWG_KEY environment variable not set")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

def slugify(text: str) -> str:
    """Normalize text into a slug key."""
    txt = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", txt.lower()).strip("-")

async def fetch_all_tags():
    tags: dict[str, str] = {}
    base_url = "https://api.rawg.io/api/tags"
    params = {"key": RAWG_KEY, "page_size": 40}
    url = base_url
    page = 1

    # Determine output path: backend/rawg_tags.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.abspath(os.path.join(script_dir, "..", "rawg_tags.json"))

    # If the file already exists, ask before overwriting
    if os.path.exists(out_path):
        answer = input(f"The file '{out_path}' already exists. Overwrite it? (y/N): ")
        if answer.strip().lower() != 'y':
            logger.info("Aborting: user chose not to overwrite the file.")
            return

    async with httpx.AsyncClient(timeout=30) as client:
        while url:
            try:
                logger.info("Fetching page %d of tags...", page)
                resp = await client.get(url, params=params)
                resp.raise_for_status()
            except Exception as e:
                logger.error("Error fetching page %d: %s", page, e)
                break

            data = resp.json()
            results = data.get("results", [])
            if not results:
                logger.info("No more tags found on page %d.", page)
                break

            # Process and add to our dict
            for t in results:
                name = t.get("name", "")
                slug = t.get("slug")
                if name and slug:
                    key = slugify(name)
                    tags[key] = slug

            logger.info(
                "Page %d: fetched %d tags, total collected: %d",
                page, len(results), len(tags)
            )

            # Write current progress to file
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(tags, f, ensure_ascii=False, indent=2)
                logger.info("Wrote %d tags to %s", len(tags), out_path)
            except Exception as e:
                logger.error("Error writing to %s: %s", out_path, e)

            # Prepare next iteration
            url = data.get("next")
            params = None  # 'next' URL already contains query params
            page += 1

    logger.info("Finished fetching. Final tag count: %d", len(tags))

if __name__ == "__main__":
    asyncio.run(fetch_all_tags())
