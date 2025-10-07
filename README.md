## CapillaryTech Docs RAG Chatbot

An end-to-end Retrieval Augmented Generation (RAG) chatbot that scrapes CapillaryTech public documentation, builds a local vector index, and serves a chat UI backed by a FastAPI API.

### Features
- Scrapes CapillaryTech docs (configurable allowed domains & start URLs)
- Cleans and chunks HTML → Markdown
- Embeds with `sentence-transformers` and stores in FAISS locally
- RAG pipeline with re-ranking for higher accuracy
- FastAPI backend with `/chat`, `/health`, `/ingest` endpoints
- Streamlit frontend with session history and citations

### Quickstart
1) Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
2) Copy env and set keys (optional for OpenAI/Anthropic; defaults to local models via `ollama` if available)
```bash
cp .env.example .env
```
3) Ingest docs (first run only)
```bash
python -m app.ingest
```
4) Run API
```bash
uvicorn app.api:app --reload --port 8000
```
5) Run UI
```bash
streamlit run app/ui.py
```

### Repo Structure
```
app/
  __init__.py
  config.py
  scraper.py
  ingest.py
  rag.py
  api.py
  ui.py
data/
  raw/
  index/
``` 

### Demo
- Record a short screen capture showing: ingest → ask 2-3 queries → citations popup.
- You can upload to Drive and share the link.

### Notes
- This project prefers local, offline-friendly defaults. If you set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, the backend will switch to those models automatically.

### Handy Commands
```bash
# 1) Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Environment
cp .env.example .env  # then edit if needed

# 3) One-shot crawl + index
python -m app.ingest

# 4) Run API and UI
uvicorn app.api:app --reload --port 8000
streamlit run app.ui.py

# 5) Use API directly
curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{"query":"What is CapillaryTech?"}'
```

LIVE LINK - https://jatinsingal-capillarytech-rag-chatbot-appui-xuux4f.streamlit.app/
