"""Configuration loaded from env / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    openai_api_key: str | None
    chat_model: str
    summary_model: str
    embed_model: str
    store_dir: Path

    @classmethod
    def load(cls) -> "Config":
        store_dir = Path(os.getenv("FORM_AGENT_STORE_DIR", ".form_store")).resolve()
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            chat_model=os.getenv("FORM_AGENT_CHAT_MODEL", "gpt-4o-mini"),
            summary_model=os.getenv("FORM_AGENT_SUMMARY_MODEL", "gpt-4o-mini"),
            embed_model=os.getenv("FORM_AGENT_EMBED_MODEL", "text-embedding-3-small"),
            store_dir=store_dir,
        )


CONFIG = Config.load()
