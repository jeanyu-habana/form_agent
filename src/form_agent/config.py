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
    # Provider selection: "openai" (default) or "azure"
    llm_provider: str
    # Azure OpenAI settings (used when llm_provider == "azure")
    azure_openai_endpoint: str | None
    azure_openai_api_key: str | None
    azure_openai_api_version: str
    azure_chat_deployment: str | None
    azure_summary_deployment: str | None
    azure_embed_deployment: str | None

    @classmethod
    def load(cls) -> "Config":
        store_dir = Path(os.getenv("FORM_AGENT_STORE_DIR", ".form_store")).resolve()
        # Auto-detect Azure if endpoint is set and provider not explicitly chosen.
        provider = os.getenv("FORM_AGENT_LLM_PROVIDER")
        if not provider:
            provider = "azure" if os.getenv("AZURE_OPENAI_ENDPOINT") else "openai"
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            chat_model=os.getenv("FORM_AGENT_CHAT_MODEL", "gpt-4o-mini"),
            summary_model=os.getenv("FORM_AGENT_SUMMARY_MODEL", "gpt-4o-mini"),
            embed_model=os.getenv("FORM_AGENT_EMBED_MODEL", "text-embedding-3-small"),
            store_dir=store_dir,
            llm_provider=provider.lower(),
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            azure_chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            azure_summary_deployment=os.getenv("AZURE_OPENAI_SUMMARY_DEPLOYMENT"),
            azure_embed_deployment=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT"),
        )


CONFIG = Config.load()
