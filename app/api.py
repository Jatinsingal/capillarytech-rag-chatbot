from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import settings
from .ingest import build_index
from .rag import VectorStore, answer_query


app = FastAPI(title="CapillaryTech RAG API")
_store: VectorStore | None = None

# Enable CORS for local UI
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


class ChatRequest(BaseModel):
	query: str


class ChatResponse(BaseModel):
	answer: str
	citations: List[Dict[str, Any]]


@app.on_event("startup")
def _load_store() -> None:
	global _store
	try:
		_store = VectorStore()
		# Warmup: perform a tiny search to materialize models/indices
		_ = _store.search("warmup", k=1)
	except Exception:
		_store = None


@app.get("/health")
def health() -> Dict[str, str]:
	return {"status": "ok"}


@app.post("/ingest")
def ingest() -> Dict[str, str]:
	from .scraper import crawl
	import asyncio

	asyncio.run(crawl())
	build_index()
	global _store
	_store = VectorStore()
	return {"status": "ingested"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
	global _store
	if _store is None:
		_store = VectorStore()
	answer, docs = answer_query(req.query, _store)
	citations = [
		{"meta": d.meta, "score": d.score}
		for d in docs
	]
	return ChatResponse(answer=answer, citations=citations)


