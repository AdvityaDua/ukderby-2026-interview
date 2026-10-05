import os
import logging
import asyncio
import time
import pathlib
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from pypdf import PdfReader
from google import genai
from google.genai import types as genai_types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from app.core.key_manager import key_manager

logger = logging.getLogger(__name__)

# Persistent ChromaDB directory co-located with the project
_CHROMA_DIR = str(pathlib.Path(__file__).parent.parent.parent.parent / "chroma_db")

_EMBEDDING_MODEL = "models/gemini-embedding-001"


def _is_quota_error(exc: BaseException) -> bool:
    return "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)


def _get_genai_client() -> genai.Client:
    api_key = key_manager.get_gemini_key() or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set — cannot generate embeddings")
    return genai.Client(api_key=api_key)


class RAGService:
    """
    Local RAG service backed by ChromaDB + Google gemini-embedding-001.
    No Vertex AI dependency.  Each knowledge topic maps to one ChromaDB
    collection named  topic_<topic_id>.
    """

    def __init__(self):
        os.makedirs(_CHROMA_DIR, exist_ok=True)
        self._chroma = chromadb.PersistentClient(
            path=_CHROMA_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # In-process caches
        self._grounding_cache: Dict[str, tuple] = {}
        self._grounding_locks: Dict[str, asyncio.Lock] = {}
        self._grounding_ttl_seconds = int(os.getenv("RAG_GROUNDING_CACHE_TTL_SECONDS", "1800"))
        self._grounding_top_k = int(os.getenv("RAG_GROUNDING_TOP_K", "5"))  # default 5 chunks max
        self._grounding_chunk_cap = int(os.getenv("RAG_GROUNDING_CHUNK_CAP", "600"))  # max chars per chunk in prompt

    # ─────────────────────────────────────────────────────── embedding ──────

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed document chunks using google-genai gemini-embedding-001 (single batched API call)."""
        return self._embed_with_retry(texts)

    @retry(
        retry=retry_if_exception(_is_quota_error),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        client = _get_genai_client()
        result = client.models.embed_content(model=_EMBEDDING_MODEL, contents=texts)
        return [e.values for e in result.embeddings]

    def _embed_query(self, query: str) -> List[List[float]]:
        """Embed a single query string using google-genai gemini-embedding-001."""
        return self._embed_query_with_retry(query)

    @retry(
        retry=retry_if_exception(_is_quota_error),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _embed_query_with_retry(self, query: str) -> List[List[float]]:
        client = _get_genai_client()
        result = client.models.embed_content(model=_EMBEDDING_MODEL, contents=query)
        return [result.embeddings[0].values]

    # ─────────────────────────────────────────────────────── collection ─────

    def _collection(self, topic_id: str):
        """Get-or-create a ChromaDB collection for a topic."""
        name = f"topic_{topic_id}"
        return self._chroma.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    # ─────────────────────────────────────────────────────── indexing ───────

    async def index_pdf(self, file_path: str, topic_id: str) -> int:
        """
        Index a PDF **or** plain-text file into ChromaDB.
        Returns the number of chunks stored.
        """
        try:
            ext = pathlib.Path(file_path).suffix.lower()
            if ext == ".txt":
                text = await asyncio.to_thread(self._read_text_file, file_path)
            else:
                text = await asyncio.to_thread(self._extract_pdf_text, file_path)

            chunks = self._chunk_text(text, chunk_chars=1400, overlap_chars=220)
            if not chunks:
                logger.warning("No extractable text found in file: %s", file_path)
                return 0

            logger.info("Embedding %d chunks for topic %s …", len(chunks), topic_id)
            embeddings = await asyncio.to_thread(self._embed_texts, chunks)

            collection = self._collection(topic_id)
            base_name = pathlib.Path(file_path).stem

            ids = [f"{base_name}_c{i:04d}" for i in range(len(chunks))]
            metadatas = [{"source": os.path.basename(file_path), "chunk": i} for i in range(len(chunks))]

            # ChromaDB upsert handles re-indexing the same file idempotently
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )

            logger.info("Stored %d chunks for topic %s", len(chunks), topic_id)
            return len(chunks)

        except Exception as e:
            logger.error("Error indexing file %s for topic %s: %s", file_path, topic_id, e)
            raise

    # ─────────────────────────────────────────────────────── text helpers ───

    def _read_text_file(self, file_path: str) -> str:
        """Read a plain-text transcript file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract textual content from PDF pages with page headers."""
        reader = PdfReader(file_path)
        sections = []
        for idx, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            sections.append(f"[PAGE {idx}]\n{page_text}")
        return "\n\n".join(sections)

    def _chunk_text(self, text: str, chunk_chars: int = 1400, overlap_chars: int = 220) -> List[str]:
        """Split text into overlapping chunks while preserving paragraph boundaries."""
        normalized = "\n".join([line.rstrip() for line in text.splitlines()]).strip()
        if not normalized:
            return []

        paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
        chunks: List[str] = []
        current = ""

        for para in paragraphs:
            candidate = f"{current}\n\n{para}".strip() if current else para
            if len(candidate) <= chunk_chars:
                current = candidate
                continue

            if current:
                chunks.append(current)

            if len(para) <= chunk_chars:
                current = para
            else:
                start = 0
                step = max(chunk_chars - overlap_chars, 200)
                while start < len(para):
                    end = min(start + chunk_chars, len(para))
                    piece = para[start:end].strip()
                    if piece:
                        chunks.append(piece)
                    if end >= len(para):
                        break
                    start += step
                current = ""

        if current:
            chunks.append(current)

        deduped: List[str] = []
        seen: set = set()
        for chunk in chunks:
            key = chunk.strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(key)
        return deduped

    # ─────────────────────────────────────────────────────── querying ───────

    async def query_topic(self, topic_id: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the top-k most relevant chunks for a query from ChromaDB."""
        try:
            try:
                collection = self._chroma.get_collection(f"topic_{topic_id}")
            except Exception:
                logger.warning("No ChromaDB collection found for topic_%s", topic_id)
                return []

            if collection.count() == 0:
                return []

            query_embedding = await asyncio.to_thread(self._embed_query, query)
            results = await asyncio.to_thread(
                collection.query,
                query_embeddings=query_embedding,
                n_results=min(k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            output = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            for doc, meta in zip(docs, metas):
                output.append({"content": doc, "metadata": meta})
            return output
        except Exception as e:
            logger.error("Error querying ChromaDB for topic %s: %s", topic_id, e)
            return []

    async def delete_topic_collection(self, topic_id: str) -> bool:
        """Delete a topic's entire ChromaDB collection."""
        try:
            self._chroma.delete_collection(f"topic_{topic_id}")
            # Invalidate grounding cache for this topic
            keys_to_drop = [k for k in self._grounding_cache if k.startswith(f"{topic_id}:")]
            for k in keys_to_drop:
                del self._grounding_cache[k]
            logger.info("Deleted ChromaDB collection for topic_%s", topic_id)
            return True
        except Exception as e:
            logger.error("Error deleting ChromaDB collection for topic %s: %s", topic_id, e)
            return False

    async def get_topic_grounding(self, topic_id: str, limit: int = 20, company_name: str = "") -> str:
        """
        Fetch representative chunks from the knowledge base for LLM grounding.

        The returned string is injected verbatim into the interview system prompt so
        the LLM uses the uploaded transcript/document as its primary question source.
        """
        normalized_company = company_name.strip()
        cache_key = f"{topic_id}:{normalized_company.lower()}:{limit}"
        now = time.time()

        cached = self._grounding_cache.get(cache_key)
        if cached and cached[0] > now:
            return cached[1]

        lock = self._grounding_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            cached = self._grounding_cache.get(cache_key)
            now = time.time()
            if cached and cached[0] > now:
                return cached[1]

            effective_limit = max(1, min(limit, self._grounding_top_k))

            # Single focused query — one embedding API call per session.
            # If a company name is given, that query already covers the topic well.
            primary_query = (
                f"{normalized_company} interview questions and key topics"
                if normalized_company
                else "interview questions topics and key concepts"
            )

            seen_contents: set = set()
            results: List[Dict[str, Any]] = []

            batch = await self.query_topic(topic_id, query=primary_query, k=effective_limit)
            for item in batch:
                content = (item.get("content") or "").strip()
                if content and content not in seen_contents:
                    seen_contents.add(content)
                    results.append(item)

            if not results:
                logger.warning("No grounding documents found for topic_%s", topic_id)
                self._grounding_cache[cache_key] = (time.time() + self._grounding_ttl_seconds, "")
                return ""

            grounding_text = "\n\n".join(
                f"--- KNOWLEDGE CHUNK {i+1} ---\n{r['content'][:self._grounding_chunk_cap]}"
                for i, r in enumerate(results)
            )
            self._grounding_cache[cache_key] = (time.time() + self._grounding_ttl_seconds, grounding_text)
            return grounding_text


# Singleton instance
rag_service = RAGService()

