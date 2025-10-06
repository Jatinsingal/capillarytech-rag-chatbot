import glob
import os
from dataclasses import dataclass
from typing import List, Tuple

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from .config import settings


def _iter_markdown_docs() -> List[Tuple[str, str]]:
	files = sorted(glob.glob(os.path.join(settings.raw_dir, "*.md")))
	docs: List[Tuple[str, str]] = []
	for path in files:
		with open(path, "r", encoding="utf-8") as f:
			text = f.read()
		docs.append((os.path.basename(path), text))
	return docs


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
	chunks: List[str] = []
	start = 0
	while start < len(text):
		end = min(len(text), start + chunk_size)
		chunk = text[start:end]
		if chunk.strip():
			chunks.append(chunk)
		start = end - overlap
		if start < 0:
			start = 0
		if start >= len(text):
			break
	return chunks


def build_index() -> None:
	os.makedirs(settings.index_dir, exist_ok=True)
	docs = _iter_markdown_docs()
	texts: List[str] = []
	metas: List[str] = []
	for name, content in docs:
		for chunk in _chunk_text(content, settings.chunk_size, settings.chunk_overlap):
			texts.append(chunk)
			metas.append(name)
	if not texts:
		raise RuntimeError("No text chunks found. Run scraper first.")

	# Lightweight fallback using TF-IDF
	if settings.embedding_model.lower() == "sklearn-tfidf":
		vectorizer = TfidfVectorizer(max_features=50000)
		X = vectorizer.fit_transform(texts)
		joblib.dump(vectorizer, os.path.join(settings.index_dir, "tfidf_vectorizer.joblib"))
		joblib.dump(X, os.path.join(settings.index_dir, "tfidf_matrix.joblib"))
		with open(os.path.join(settings.index_dir, "texts.txt"), "w", encoding="utf-8") as f:
			for t in texts:
				f.write(t.replace("\n", " ") + "\n")
		with open(os.path.join(settings.index_dir, "metas.txt"), "w", encoding="utf-8") as f:
			for m in metas:
				f.write(m + "\n")
		return
	# Heavy model import only when needed
	from sentence_transformers import SentenceTransformer
	emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
	index = faiss.IndexFlatIP(emb.shape[1])
	index.add(np.array(emb, dtype=np.float32))
	faiss.write_index(index, os.path.join(settings.index_dir, "faiss.index"))
	with open(os.path.join(settings.index_dir, "texts.txt"), "w", encoding="utf-8") as f:
		for t in texts:
			f.write(t.replace("\n", " ") + "\n")
	with open(os.path.join(settings.index_dir, "metas.txt"), "w", encoding="utf-8") as f:
		for m in metas:
			f.write(m + "\n")


if __name__ == "__main__":
	from .scraper import crawl

	import asyncio

	asyncio.run(crawl())
	build_index()


