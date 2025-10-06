import os
from dataclasses import dataclass
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from rank_bm25 import BM25Okapi

from .config import settings


@dataclass
class Document:
	text: str
	meta: str
	score: float


class VectorStore:
	def __init__(self) -> None:
		texts_path = os.path.join(settings.index_dir, "texts.txt")
		metas_path = os.path.join(settings.index_dir, "metas.txt")
		loaded_texts: List[str] = []
		loaded_metas: List[str] = []
		if os.path.exists(texts_path) and os.path.exists(metas_path):
			loaded_texts = _read_lines(texts_path)
			loaded_metas = _read_lines(metas_path)
		# Also pull any raw markdown files (seed/curated) and merge
		raw_dir = settings.raw_dir
		if os.path.isdir(raw_dir):
			for name in sorted(os.listdir(raw_dir)):
				if name.endswith('.md'):
					with open(os.path.join(raw_dir, name), 'r', encoding='utf-8') as f:
						loaded_texts.append(f.read())
						loaded_metas.append(name)
		if not loaded_texts:
			loaded_texts = ["CapillaryTech placeholder document. No docs ingested yet."]
			loaded_metas = ["placeholder.md"]
		# Clean boilerplate lines that say "Checking your browser"
		self.texts = [clean_text(t) for t in loaded_texts]
		self.metas = loaded_metas

		self.tfidf_mode = settings.embedding_model.lower() == "sklearn-tfidf"
		# Prefer BM25 by default for robustness on small/seeded corpora
		prefer_bm25 = os.getenv("FORCE_BM25", "true").lower() in ("1", "true", "yes")
		self.bm25_mode = False
		try:
			if prefer_bm25:
				tokenized = [t.split() for t in self.texts]
				self.bm25 = BM25Okapi(tokenized)
				self.bm25_mode = True
			elif self.tfidf_mode and os.path.exists(os.path.join(settings.index_dir, "tfidf_vectorizer.joblib")):
				self.vectorizer: TfidfVectorizer = joblib.load(os.path.join(settings.index_dir, "tfidf_vectorizer.joblib"))
				self.matrix = joblib.load(os.path.join(settings.index_dir, "tfidf_matrix.joblib"))
			elif os.path.exists(os.path.join(settings.index_dir, "faiss.index")):
				self.index = faiss.read_index(os.path.join(settings.index_dir, "faiss.index"))
				self.embed_model = SentenceTransformer(settings.embedding_model)
			else:
				# BM25 fallback entirely in-memory
				tokenized = [t.split() for t in self.texts]
				self.bm25 = BM25Okapi(tokenized)
				self.bm25_mode = True
		except Exception:
			tokenized = [t.split() for t in self.texts]
			self.bm25 = BM25Okapi(tokenized)
			self.bm25_mode = True

	def search(self, query: str, k: int = 8) -> List[Document]:
		if self.bm25_mode:
			scores_arr = np.array(self.bm25.get_scores(query.split()), dtype=float)
			idx_list = np.argsort(-scores_arr)[:k]
			scores = scores_arr[idx_list]
		elif self.tfidf_mode:
			q = self.vectorizer.transform([query])
			# cosine similarity
			scores_arr = (self.matrix @ q.T).toarray().ravel()
			indices = np.argsort(-scores_arr)[:k]
			scores = scores_arr[indices]
			idx_list = indices
		else:
			query_emb = (
				self.embed_model.encode([query], normalize_embeddings=True)
				.astype(np.float32)
			)
			scores, idx = self.index.search(query_emb, k)
			idx_list = idx[0]
			scores = scores[0]
		results: List[Document] = []
		for i, score in zip(idx_list, scores):
			if i < 0:
				continue
			results.append(
				Document(text=self.texts[i], meta=self.metas[i], score=float(score))
			)
		return results


def _read_lines(path: str) -> List[str]:
	with open(path, "r", encoding="utf-8") as f:
		return [line.rstrip("\n") for line in f]


def clean_text(text: str) -> str:
	bad_phrases = [
		"Checking your browser...",
		"This may take a few seconds.",
		"Click here if you are not redirected automatically",
	]
	for p in bad_phrases:
		text = text.replace(p, "")
	return text


def rerank(query: str, docs: List[Document], top_k: int = 4) -> List[Document]:
	if not docs:
		return []
	# Allow disabling rerank for speed
	if os.getenv("RERANK", "false").lower() not in ("1", "true", "yes"):
		return docs[:top_k]
	# Cache the cross-encoder to avoid reloading per request
	global _CROSS_ENCODER
	try:
		_CROSS_ENCODER
	except NameError:
		_CROSS_ENCODER = None
	if _CROSS_ENCODER is None:
		_CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
	pairs = [[query, d.text] for d in docs]
	scores = _CROSS_ENCODER.predict(pairs)
	for d, s in zip(docs, scores):
		d.score = float(s)
	return sorted(docs, key=lambda d: d.score, reverse=True)[:top_k]


def build_prompt(query: str, contexts: List[Document]) -> str:
	context_text = "\n\n".join(
		[f"Snippet {i+1} (source {d.meta}):\n{d.text}" for i, d in enumerate(contexts)]
	)
	return (
		"You are a helpful assistant answering based on CapillaryTech docs. "
		"If unsure, say you don't know.\n\n"
		f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer concisely with citations like [Snippet-#]."
	)


def call_llm(prompt: str, temperature: float = 0.2) -> str:
	provider = os.getenv("LLM_PROVIDER", "auto")
	if provider in ("auto", "openai") and os.getenv("OPENAI_API_KEY"):
		from openai import OpenAI

		client = OpenAI()
		resp = client.chat.completions.create(
			model="gpt-4o-mini",
			messages=[{"role": "user", "content": prompt}],
			temperature=temperature,
		)
		return resp.choices[0].message.content or ""
	if provider in ("auto", "anthropic") and os.getenv("ANTHROPIC_API_KEY"):
		import anthropic

		client = anthropic.Anthropic()
		msg = client.messages.create(
			model="claude-3-5-sonnet-latest",
			max_tokens=600,
			messages=[{"role": "user", "content": prompt}],
		)
		return "".join(block.text for block in msg.content if hasattr(block, "text"))
	# fallback to local ollama if available
	try:
		import ollama

		res = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])
		return res.get("message", {}).get("content", "")
	except Exception:
		return "Model not configured. Set OPENAI_API_KEY/ANTHROPIC_API_KEY or install ollama."


def answer_query(query: str, store: VectorStore, k: int = 8) -> Tuple[str, List[Document]]:
	# ultra-fast path for greetings and short small-talk
	qnorm = query.strip().lower()
	if qnorm in {"hi", "hello", "hey", "yo", "hola"} or len(qnorm.split()) <= 2:
		return (
			"Hello! I’m a CapillaryTech docs assistant. Ask me about products, loyalty, or how to get started.",
			[],
		)

	seed = store.search(query, k=min(k, 6))
	selected = rerank(query, seed, top_k=4)
	prompt = build_prompt(query, selected)
	text = call_llm(prompt)
	if not text or "Model not configured" in text:
		# Fast extractive fallback: return top snippets as the answer with citations
		snippets = []
		for i, d in enumerate(selected[:2]):
			snippet = d.text.strip().split("\n")
			snippet = " ".join(snippet)[:600]
			snippets.append(f"[Snippet-{i+1}] {snippet} (source {d.meta})")
		text = ("\n\n".join(snippets)) or "No context available yet. Please re-run ingestion."
	return text, selected


