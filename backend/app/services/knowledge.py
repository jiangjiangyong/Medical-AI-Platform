from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.entities import KnowledgeChunk
from app.ml.rag import metadata_matches, rewrite_query
from app.ml.retrieval import bm25_score, filter_evidence, reciprocal_rank_fusion
from app.config import settings
from app.services.embeddings import SiliconFlowEmbeddingClient, split_text


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class KnowledgeService:
    def __init__(self) -> None:
        self.embedding_client = SiliconFlowEmbeddingClient()
        self._reranker = None
        self._reranker_load_attempted = False

    async def index_text(self, db: Session, source_name: str, content: str) -> dict:
        chunks = split_text(content)
        embeddings = await self.embedding_client.embed_texts([chunk.content for chunk in chunks])
        db.execute(delete(KnowledgeChunk).where(KnowledgeChunk.source_name == source_name))
        for chunk, embedding in zip(chunks, embeddings):
            db.add(
                KnowledgeChunk(
                    source_name=source_name,
                    chunk_index=chunk.index,
                    content=chunk.content,
                    embedding=embedding,
                    metadata_json={
                        "chunk_chars": len(chunk.content),
                        "embedding_ready": embedding is not None,
                        "knowledge_base_version": settings.knowledge_base_version,
                    },
                )
            )
        db.commit()
        return {
            "source_name": source_name,
            "chunks": len(chunks),
            "embedding_ready": sum(1 for item in embeddings if item is not None),
        }

    async def index_directory(self, db: Session, directory: Path) -> list[dict]:
        results = []
        for path in sorted(directory.glob("*.md")):
            results.append(await self.index_text(db, path.name, path.read_text(encoding="utf-8")))
        return results

    async def search(
        self,
        db: Session,
        query: str,
        limit: int = 4,
        metadata_filters: Mapping[str, Any] | None = None,
    ) -> list[dict]:
        query = rewrite_query(query)
        chunks = [
            chunk
            for chunk in db.scalars(select(KnowledgeChunk))
            if metadata_matches(chunk.metadata_json, metadata_filters)
        ]
        if not chunks:
            return []
        query_vector = await self.embedding_client.embed_query(query)
        corpus = [chunk.content for chunk in chunks]
        bm25_scores = {
            f"{chunk.source_name}#{chunk.chunk_index}": bm25_score(query, chunk.content, corpus)
            for chunk in chunks
        }
        lexical_ranking = [
            key for key, _ in sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True)
        ]

        dense_scores: dict[str, float] = {}
        if query_vector is not None:
            for chunk in chunks:
                if chunk.embedding:
                    dense_scores[f"{chunk.source_name}#{chunk.chunk_index}"] = cosine_similarity(
                        query_vector, chunk.embedding
                    )
        dense_ranking = [
            key for key, _ in sorted(dense_scores.items(), key=lambda item: item[1], reverse=True)
        ]
        rankings = {"bm25": lexical_ranking}
        if dense_ranking:
            rankings["dense"] = dense_ranking
        rrf_scores = reciprocal_rank_fusion(rankings, k=settings.rag_rrf_k)
        candidate_ids = set(lexical_ranking[: limit * settings.rag_candidate_multiplier])
        candidate_ids.update(dense_ranking[: limit * settings.rag_candidate_multiplier])
        chunk_by_id = {f"{chunk.source_name}#{chunk.chunk_index}": chunk for chunk in chunks}
        candidates = []
        for identifier in candidate_ids:
            chunk = chunk_by_id[identifier]
            rerank_score, reranker_name = self._rerank_score(query, chunk.content)
            candidates.append(
                {
                    "evidence_id": identifier,
                    "source_name": chunk.source_name,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "dense_score": round(dense_scores.get(identifier, 0.0), 6),
                    "bm25_score": round(bm25_scores.get(identifier, 0.0), 6),
                    "rrf_score": round(rrf_scores.get(identifier, 0.0), 6),
                    "rerank_score": round(rerank_score, 6),
                    "score": round(rerank_score + rrf_scores.get(identifier, 0.0), 6),
                    "reranker": reranker_name,
                    "retrieval_method": "dense_bm25_hybrid" if dense_ranking else "bm25_fallback",
                    "retrieval_version": settings.rag_version,
                }
            )
        return filter_evidence(
            candidates,
            limit=limit,
            min_score=settings.rag_min_evidence_score,
        )

    def _rerank_score(self, query: str, content: str) -> tuple[float, str]:
        if settings.rag_reranker_path and not self._reranker_load_attempted:
            self._reranker_load_attempted = True
            try:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(settings.rag_reranker_path)
            except Exception:
                self._reranker = None
        if self._reranker is not None:
            try:
                score = self._reranker.predict([(query, content)], show_progress_bar=False)[0]
                return float(score), "cross_encoder"
            except Exception:
                self._reranker = None
        return _lexical_rerank_score(query, content), "lexical_fallback"


def _lexical_rerank_score(query: str, content: str) -> float:
    query_terms = set(query.replace("，", " ").replace("。", " ").split())
    if not query_terms:
        return 0.0
    matched = sum(1 for term in query_terms if term and term in content)
    return matched / len(query_terms)
