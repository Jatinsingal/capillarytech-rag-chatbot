import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv


load_dotenv()


def _csv_env(name: str, default: str = "") -> List[str]:
	value = os.getenv(name, default)
	items = [x.strip() for x in value.split(",") if x.strip()]
	return items


@dataclass
class Settings:
	llm_provider: str = os.getenv("LLM_PROVIDER", "auto")
	embedding_model: str = os.getenv(
		"EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
	)
	allowed_domains: List[str] = None  # type: ignore
	start_urls: List[str] = None  # type: ignore
	index_dir: str = os.getenv("INDEX_DIR", "data/index")
	raw_dir: str = os.getenv("RAW_DIR", "data/raw")
	chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
	chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

	def __post_init__(self) -> None:
		self.allowed_domains = _csv_env(
			"ALLOWED_DOMAINS",
			"capillarytech.com,help.capillarytech.com,documentation.capillarytech.com",
		)
		self.start_urls = _csv_env(
			"START_URLS",
			"https://www.capillarytech.com/resources/,https://www.capillarytech.com",
		)
		os.makedirs(self.index_dir, exist_ok=True)
		os.makedirs(self.raw_dir, exist_ok=True)


settings = Settings()


