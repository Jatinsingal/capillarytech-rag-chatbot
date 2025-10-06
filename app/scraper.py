import hashlib
import os
import re
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple

import httpx
from bs4 import BeautifulSoup
from trafilatura import extract
from readability import Document as ReadabilityDocument

from .config import settings


_USER_AGENT = (
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
	"AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)


def _is_allowed(url: str) -> bool:
	return any(domain in url for domain in settings.allowed_domains)


def _normalize(url: str) -> str:
	url = re.sub(r"[#?].*$", "", url)
	if url.endswith("/"):
		url = url[:-1]
	return url


def _hash(text: str) -> str:
	return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class ScrapedPage:
	url: str
	title: str
	content_md: str


async def _fetch(client: httpx.AsyncClient, url: str) -> Optional[str]:
	try:
		resp = await client.get(url, timeout=30)
		text = resp.text or ""
		return text if text.strip() else None
	except Exception:
		return None


def _extract_text(html: str, url: str) -> Tuple[str, str]:
    """Return title and markdown-ish text.
    Falls back to readability and plain text when needed.
    """
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title else ""
    cleaned = extract(html, include_comments=False, url=url) or ""
    if len(cleaned.strip()) < 200:
        try:
            doc = ReadabilityDocument(html)
            title = title or (doc.short_title() or "").strip()
            content_html = doc.summary(html_partial=True)
            cleaned = BeautifulSoup(content_html, "html.parser").get_text("\n")
        except Exception:
            pass
    if len(cleaned.strip()) < 100:
        cleaned = soup.get_text("\n")
    return title, cleaned


async def crawl(max_pages: int = 200) -> List[ScrapedPage]:
	os.makedirs(settings.raw_dir, exist_ok=True)
	seen: Set[str] = set()
	queue: deque[str] = deque(_normalize(u) for u in settings.start_urls)
	results: List[ScrapedPage] = []
	async with httpx.AsyncClient(headers={"User-Agent": _USER_AGENT}, follow_redirects=True) as client:
		while queue and len(results) < max_pages:
			url = queue.popleft()
			if not _is_allowed(url):
				continue
			if url in seen:
				continue
			seen.add(url)
			html = await _fetch(client, url)
			if not html:
				continue
			title, text_md = _extract_text(html, url)
			if len(text_md.strip()) < 50:
				continue
			results.append(ScrapedPage(url=url, title=title, content_md=text_md))
			# enqueue links
			soup = BeautifulSoup(html, "html.parser")
			for a in soup.find_all("a", href=True):
				href = a["href"].strip()
				if href.startswith("/") and settings.start_urls:
					base = settings.start_urls[0].rstrip("/")
					href = base + href
				href = _normalize(href)
				if _is_allowed(href) and href not in seen:
					queue.append(href)
	# write out
	for page in results:
		fname = os.path.join(settings.raw_dir, f"{_hash(page.url)}.md")
		with open(fname, "w", encoding="utf-8") as f:
			f.write(f"# {page.title}\n\nSource: {page.url}\n\n{page.content_md}\n")
	return results


async def fetch_seed(urls: List[str]) -> List[ScrapedPage]:
	"""Fetch a fixed list of URLs (useful when crawling is blocked)."""
	os.makedirs(settings.raw_dir, exist_ok=True)
	results: List[ScrapedPage] = []
	async with httpx.AsyncClient(headers={"User-Agent": _USER_AGENT}, follow_redirects=True) as client:
		for url in urls:
			if not _is_allowed(url):
				continue
			html = await _fetch(client, url)
			if not html:
				continue
			title, text_md = _extract_text(html, url)
			if len(text_md.strip()) < 50:
				continue
			results.append(ScrapedPage(url=url, title=title, content_md=text_md))
	for page in results:
		fname = os.path.join(settings.raw_dir, f"{_hash(page.url)}.md")
		with open(fname, "w", encoding="utf-8") as f:
			f.write(f"# {page.title}\n\nSource: {page.url}\n\n{page.content_md}\n")
	return results


