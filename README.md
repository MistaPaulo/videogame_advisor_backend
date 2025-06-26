# Chatbot de Recomendação de Videojogos – Backend (FastAPI)

API baseada em **FastAPI** que alimenta o chatbot de recomendações de videojogos.

## Requisitos

* Python ≥ 3.11  
* Conta e chaves de API válidas  
  * `GROQ_API_KEY` — Groq / OpenRouter  
  * `RAWG_KEY` — RAWG.io

## Instalação

```bash
git clone https://github.com/MistaPaulo/videogame_advisor_backend.git
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuração

Cria um ficheiro `.env` na pasta **backend** com o seguinte conteúdo:

```
GROQ_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
RAWG_KEY=yyyyyyyyyyyyyyyyyyyyyyyyyyyy
```

> **Nota:** O ficheiro `.env` está incluído no `.gitignore` – não publiques as tuas chaves.

## Execução local (modo desenvolvimento)

```bash
uvicorn main:app --reload
```

* API disponível em <http://127.0.0.1:8000>  
* Documentação Swagger em <http://127.0.0.1:8000/docs>

## Script opcional

Atualizar a cache de *tags* da RAWG:

```bash
python scripts/fetch_rawg_tags.py
```

## Teste rápido

```bash
curl -X POST http://127.0.0.1:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message":"Recomenda-me RPG de ação parecido com Elden Ring"}'
```

## Deploy em produção

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
(Pode ser usado, por exemplo, no Render Web Service.)
