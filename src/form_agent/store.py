"""Persistent form storage (JSON) + vector store (Chroma) for cross-form retrieval."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .config import CONFIG
from .extraction import ExtractedField, ExtractionResult, FieldSpec
from .llm import LLMClient
from .parsing import ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class StoredForm:
    id: str
    source_path: str
    form_type: str
    pages: list[str]
    schema: list[dict]
    fields: dict[str, dict]
    metadata: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(f"[Page {i + 1}]\n{p}" for i, p in enumerate(self.pages))

    def field_objects(self) -> dict[str, ExtractedField]:
        return {k: ExtractedField(**v) for k, v in self.fields.items()}

    def schema_objects(self) -> list[FieldSpec]:
        return [FieldSpec(**s) for s in self.schema]


class FormStore:
    """JSON-on-disk store. One file per form under <store_dir>/forms/<id>.json."""

    def __init__(self, store_dir: Path | None = None) -> None:
        self.dir = (store_dir or CONFIG.store_dir) / "forms"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, form_id: str) -> Path:
        return self.dir / f"{form_id}.json"

    @staticmethod
    def make_id(source_path: str, text: str) -> str:
        h = hashlib.sha1(
            f"{source_path}|{text[:512]}".encode("utf-8"), usedforsecurity=False
        ).hexdigest()[:12]
        stem = Path(source_path).stem.lower().replace(" ", "_")[:32]
        return f"{stem}-{h}"

    def save(self, doc: ParsedDocument, result: ExtractionResult) -> StoredForm:
        form_id = self.make_id(doc.source_path, doc.text)
        stored = StoredForm(
            id=form_id,
            source_path=doc.source_path,
            form_type=result.form_type,
            pages=doc.pages,
            schema=[s.model_dump() for s in result.schema_],
            fields={k: v.model_dump() for k, v in result.fields.items()},
            metadata={"ocr_used": doc.ocr_used},
        )
        self._path(form_id).write_text(json.dumps(asdict(stored), indent=2, default=str))
        return stored

    def get(self, form_id: str) -> StoredForm | None:
        p = self._path(form_id)
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        return StoredForm(**data)

    def list_forms(self) -> list[StoredForm]:
        out = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                out.append(StoredForm(**json.loads(p.read_text())))
            except Exception as e:  # pragma: no cover
                logger.warning("Skipping bad form file %s: %s", p, e)
        return out

    def delete(self, form_id: str) -> bool:
        p = self._path(form_id)
        if p.exists():
            p.unlink()
            return True
        return False


class VectorStore:
    """Chroma-backed page-chunk store for cross-form RAG."""

    COLLECTION = "forms"

    def __init__(self, llm: LLMClient | None = None, store_dir: Path | None = None) -> None:
        import chromadb

        self.llm = llm
        path = (store_dir or CONFIG.store_dir) / "chroma"
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(self.COLLECTION)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.llm is None:
            raise RuntimeError("VectorStore needs an LLMClient for embeddings")
        return self.llm.embed(texts)

    def index_form(self, form: StoredForm) -> None:
        # Remove any existing chunks for this form first.
        self.delete_form(form.id)
        chunks: list[str] = []
        ids: list[str] = []
        metas: list[dict] = []
        for i, page in enumerate(form.pages):
            page = (page or "").strip()
            if not page:
                continue
            chunks.append(page)
            ids.append(f"{form.id}:p{i + 1}")
            metas.append({"form_id": form.id, "page": i + 1, "form_type": form.form_type,
                          "source_path": form.source_path})
        if not chunks:
            return
        embeddings = self._embed(chunks)
        self.collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metas)

    def delete_form(self, form_id: str) -> None:
        try:
            self.collection.delete(where={"form_id": form_id})
        except Exception:  # pragma: no cover
            pass

    def query(self, text: str, top_k: int = 6, form_ids: Iterable[str] | None = None) -> list[dict]:
        embedding = self._embed([text])[0]
        where = {"form_id": {"$in": list(form_ids)}} if form_ids else None
        res = self.collection.query(query_embeddings=[embedding], n_results=top_k, where=where)
        out = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[None] * len(docs)])[0]
        for d, m, dist in zip(docs, metas, dists):
            out.append({"document": d, "metadata": m, "distance": dist})
        return out


class BlobStore:
    """Optional Azure Blob Storage sink — archives extracted-form JSON on ingest.

    Silently disabled when ``AZURE_BLOB_CONNECTION_STRING`` is not configured.
    All upload errors are swallowed with a warning so ingestion is never blocked.
    """

    def __init__(self) -> None:
        self._enabled = bool(CONFIG.blob_connection_string)

    def upload_form(self, stored: StoredForm) -> None:
        """Upload ``stored`` as ``{form_id}.json`` into the configured container."""
        if not self._enabled:
            return
        try:
            from azure.storage.blob import BlobServiceClient

            blob_name = f"{stored.id}.json"
            payload = json.dumps(asdict(stored), indent=2, default=str).encode("utf-8")
            client = BlobServiceClient.from_connection_string(CONFIG.blob_connection_string)
            blob_client = client.get_blob_client(container=CONFIG.blob_container, blob=blob_name)
            blob_client.upload_blob(payload, overwrite=True)
            logger.info("Uploaded form JSON to blob %s/%s", CONFIG.blob_container, blob_name)
        except Exception as exc:  # pragma: no cover — network / auth errors at runtime
            logger.warning("Blob upload failed for %s: %s", stored.id, exc)
