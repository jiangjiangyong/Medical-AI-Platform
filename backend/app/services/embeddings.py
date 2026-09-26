from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import settings


@dataclass(frozen=True)
class TextChunk:
    index: int
    content: str


def _split_long_piece(piece: str, max_chars: int, overlap: int) -> list[str]:
    if len(piece) <= max_chars:
        return [piece]
    result: list[str] = []
    start = 0
    step = max(1, max_chars - overlap)
    while start < len(piece):
        result.append(piece[start : start + max_chars].strip())
        start += step
    return [item for item in result if item]


def split_text(text: str, max_chars: int | None = None, overlap: int | None = None) -> list[TextChunk]:
    """Conservative splitter for BGE-large-zh-v1.5's 512-token request limit."""
    max_chars = max_chars or settings.embedding_max_chars
    overlap = overlap if overlap is not None else settings.embedding_chunk_overlap
    normalized = "\n".join(line.strip() for line in text.replace("\r\n", "\n").splitlines())
    normalized = normalized.strip()
    if not normalized:
        return []
    if overlap >= max_chars:
        overlap = max_chars // 8

    units: list[str] = []
    for paragraph in normalized.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentence_parts = [part.strip() for part in __import__("re").split(r"(?<=[。！？；;.!?])", paragraph) if part.strip()]
        for part in sentence_parts or [paragraph]:
            units.extend(_split_long_piece(part, max_chars, overlap))

    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
            continue
        candidate = f"{current}\n{unit}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current.strip())
            tail = current[-overlap:].strip() if overlap else ""
            current = f"{tail}\n{unit}".strip() if tail else unit
            if len(current) > max_chars:
                chunks.extend(_split_long_piece(current, max_chars, overlap)[:-1])
                current = _split_long_piece(current, max_chars, overlap)[-1]
    if current:
        chunks.append(current.strip())
    return [TextChunk(index=index, content=content) for index, content in enumerate(chunks) if content]


class SiliconFlowEmbeddingClient:
    def __init__(self) -> None:
        self.enabled = bool(settings.embedding_api_key)

    async def embed_texts(self, texts: list[str]) -> list[list[float] | None]:
        if not texts:
            return []
        if not self.enabled:
            return [None for _ in texts]
        results: list[list[float] | None] = []
        batch_size = max(1, settings.embedding_batch_size)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            results.extend(await self._embed_batch_with_split(batch))
        return results

    async def _embed_batch_with_split(self, texts: list[str]) -> list[list[float] | None]:
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    settings.embedding_url,
                    headers={
                        "Authorization": f"Bearer {settings.embedding_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings.embedding_model,
                        "input": texts,
                        "encoding_format": "float",
                    },
                )
                response.raise_for_status()
                payload: dict[str, Any] = response.json()
                data = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
                vectors = [item.get("embedding") for item in data]
                if len(vectors) == len(texts):
                    return vectors
                raise ValueError("Embedding response length does not match input length")
        except Exception:
            if len(texts) == 1:
                return [None]
            midpoint = max(1, len(texts) // 2)
            left, right = await asyncio.gather(
                self._embed_batch_with_split(texts[:midpoint]),
                self._embed_batch_with_split(texts[midpoint:]),
            )
            return left + right

    async def embed_query(self, text: str) -> list[float] | None:
        vectors = await self.embed_texts([text[: settings.embedding_max_chars]])
        return vectors[0] if vectors else None

