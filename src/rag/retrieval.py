"""
RAG Retrieval Pipeline (Batch B)
================================

Hybrid retrieval combining:
- Vector search (pgvector cosine similarity)
- Lexical search (PostgreSQL full-text search)
- Reciprocal Rank Fusion (RRF) merging

Features:
- Configurable k values for each search type
- RRF merge with tunable k parameter
- Gating rules (similarity threshold, min chunks, recency)
- Full audit logging to rag_retrievals table

Usage:
    from src.rag.retrieval import RAGRetriever

    retriever = RAGRetriever()
    result = retriever.retrieve(
        query="What did the CEO say about AI demand?",
        ticker="MU",
        doc_types=['transcript']
    )

    if result.gating_passed:
        for chunk in result.chunks:
            print(f"[{chunk.chunk_id}] {chunk.text[:100]}...")

Author: HH Research Platform
"""

import os
import hashlib
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from psycopg2.extras import Json
from src.db.connection import get_connection
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import sentence-transformers for query embedding
try:
    from sentence_transformers import SentenceTransformer

    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")


class GatingDecision(Enum):
    """Gating decision outcomes."""
    PASS = "PASS"
    FAIL_SIMILARITY = "FAIL_SIMILARITY"  # Best similarity below threshold
    FAIL_MIN_CHUNKS = "FAIL_MIN_CHUNKS"  # Not enough chunks pass threshold
    FAIL_STALE = "FAIL_STALE"  # Documents too old
    FAIL_NO_RESULTS = "FAIL_NO_RESULTS"  # No results returned
    FAIL_NO_EMBEDDING = "FAIL_NO_EMBEDDING"  # Query embedding failed


@dataclass
class RetrievalConfig:
    """Configuration for retrieval pipeline."""

    # Vector search
    k_vector: int = 40  # Top-k for vector search

    # Lexical search
    k_lexical: int = 40  # Top-k for lexical search

    # RRF merge
    rrf_k: int = 60  # RRF smoothing parameter
    top_n_final: int = 12  # Final number of chunks to return

    # Gating thresholds
    min_cosine_similarity: float = 0.55  # Minimum similarity for top result
    min_chunks_above_threshold: int = 3  # Minimum chunks passing threshold
    chunk_similarity_threshold: float = 0.45  # Threshold for counting "good" chunks

    # Recency
    max_document_age_days: int = 365  # Documents older than this are stale

    # Embedding model
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024

    @classmethod
    def default(cls) -> 'RetrievalConfig':
        """Default configuration based on architecture decisions."""
        return cls()

    @classmethod
    def strict(cls) -> 'RetrievalConfig':
        """Stricter configuration for high-precision needs."""
        return cls(
            min_cosine_similarity=0.65,
            min_chunks_above_threshold=5,
            chunk_similarity_threshold=0.55,
        )

    @classmethod
    def loose(cls) -> 'RetrievalConfig':
        """Looser configuration for exploratory queries."""
        return cls(
            min_cosine_similarity=0.45,
            min_chunks_above_threshold=2,
            chunk_similarity_threshold=0.35,
            top_n_final=20,
        )


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with scores."""
    chunk_id: int
    doc_id: int
    ticker: str
    doc_type: str
    asof_ts_utc: datetime
    section: Optional[str]
    speaker: Optional[str]
    speaker_role: Optional[str]
    text: str
    char_start: int
    char_end: int

    # Scores
    vector_score: Optional[float] = None  # Cosine similarity (higher = better)
    lexical_score: Optional[float] = None  # ts_rank score
    rrf_score: Optional[float] = None  # Combined RRF score
    final_rank: int = 0

    # Source document info
    period_label: Optional[str] = None
    doc_title: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    query: str
    query_embedding: Optional[List[float]]

    # Retrieved chunks (after RRF merge and top-n selection)
    chunks: List[RetrievedChunk]

    # Scores for gating
    top_similarity: float
    chunks_above_threshold: int

    # Gating
    gating_decision: GatingDecision
    gating_details: Dict[str, Any]

    # Timing
    retrieval_time_ms: int

    # For audit logging
    retrieval_id: Optional[int] = None

    @property
    def gating_passed(self) -> bool:
        return self.gating_decision == GatingDecision.PASS


class RAGRetriever:
    """
    Hybrid retrieval with RRF merging and gating.

    Combines vector similarity search (pgvector) with lexical search
    (PostgreSQL full-text) using Reciprocal Rank Fusion.
    """

    def __init__(self, config: RetrievalConfig = None):
        """
        Initialize retriever.

        Args:
            config: Retrieval configuration
        """
        self.config = config or RetrievalConfig.default()
        self._embedding_model = None
        self._model_loaded = False

    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if not self._model_loaded:
            if not EMBEDDING_AVAILABLE:
                raise ImportError(
                    "sentence-transformers required for query embedding. "
                    "Install with: pip install sentence-transformers"
                )

            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self._embedding_model = SentenceTransformer(self.config.embedding_model)
            self._model_loaded = True
            logger.info("Embedding model loaded")

        return self._embedding_model

    def embed_query(self, query: str) -> Optional[np.ndarray]:
        """
        Generate embedding for query.

        Args:
            query: Query text

        Returns:
            Normalized embedding vector or None if failed
        """
        try:
            model = self._get_embedding_model()

            # Add query prefix for bge models (improves retrieval)
            query_with_prefix = f"Represent this sentence for searching relevant passages: {query}"

            embedding = model.encode(query_with_prefix, normalize_embeddings=True)

            # Validate dimension
            if embedding.shape[0] != self.config.embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: got {embedding.shape[0]}, "
                    f"expected {self.config.embedding_dim}"
                )

            return embedding

        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return None

    def _vector_search(
            self,
            conn,
            query_embedding: np.ndarray,
            ticker: Optional[str] = None,
            doc_types: Optional[List[str]] = None,
            min_date: Optional[datetime] = None,
    ) -> List[Tuple[int, float]]:
        """
        Vector similarity search using pgvector.

        Returns list of (chunk_id, similarity_score) tuples.
        Higher similarity = better match.
        """
        # Build WHERE clause
        conditions = ["c.embedding IS NOT NULL"]
        params = []

        if ticker:
            conditions.append("c.ticker = %s")
            params.append(ticker)

        if doc_types:
            conditions.append("c.doc_type = ANY(%s)")
            params.append(doc_types)

        if min_date:
            conditions.append("c.asof_ts_utc >= %s")
            params.append(min_date)

        where_clause = " AND ".join(conditions)

        # Convert embedding to PostgreSQL vector format
        embedding_str = "[" + ",".join(str(x) for x in query_embedding.tolist()) + "]"

        query = f"""
            SELECT 
                c.chunk_id,
                1 - (c.embedding <=> %s::vector) as similarity
            FROM rag.doc_chunks c
            WHERE {where_clause}
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """

        params = [embedding_str] + params + [embedding_str, self.config.k_vector]

        with conn.cursor() as cur:
            cur.execute(query, params)
            results = cur.fetchall()

        return [(row[0], row[1]) for row in results]

    def _lexical_search(
            self,
            conn,
            query: str,
            ticker: Optional[str] = None,
            doc_types: Optional[List[str]] = None,
            min_date: Optional[datetime] = None,
    ) -> List[Tuple[int, float]]:
        """
        Full-text search using PostgreSQL tsvector.

        Returns list of (chunk_id, rank_score) tuples.
        Higher rank = better match.
        """
        # Build WHERE clause
        conditions = ["c.text_tsv @@ plainto_tsquery('english', %s)"]
        params = [query]

        if ticker:
            conditions.append("c.ticker = %s")
            params.append(ticker)

        if doc_types:
            conditions.append("c.doc_type = ANY(%s)")
            params.append(doc_types)

        if min_date:
            conditions.append("c.asof_ts_utc >= %s")
            params.append(min_date)

        where_clause = " AND ".join(conditions)

        query_sql = f"""
            SELECT 
                c.chunk_id,
                ts_rank_cd(c.text_tsv, plainto_tsquery('english', %s), 32) as rank
            FROM rag.doc_chunks c
            WHERE {where_clause}
            ORDER BY rank DESC
            LIMIT %s
        """

        params = [query] + params + [self.config.k_lexical]

        with conn.cursor() as cur:
            cur.execute(query_sql, params)
            results = cur.fetchall()

        return [(row[0], row[1]) for row in results]

    def _rrf_merge(
            self,
            vector_results: List[Tuple[int, float]],
            lexical_results: List[Tuple[int, float]],
    ) -> List[Tuple[int, float, Optional[float], Optional[float]]]:
        """
        Merge results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) for each list containing the item

        Returns list of (chunk_id, rrf_score, vector_score, lexical_score) tuples.
        """
        k = self.config.rrf_k

        # Build lookup for scores
        vector_scores = {chunk_id: score for chunk_id, score in vector_results}
        lexical_scores = {chunk_id: score for chunk_id, score in lexical_results}

        # Build rank lookup (1-indexed)
        vector_ranks = {chunk_id: rank + 1 for rank, (chunk_id, _) in enumerate(vector_results)}
        lexical_ranks = {chunk_id: rank + 1 for rank, (chunk_id, _) in enumerate(lexical_results)}

        # Get all unique chunk IDs
        all_chunks = set(vector_ranks.keys()) | set(lexical_ranks.keys())

        # Calculate RRF scores
        rrf_scores = []
        for chunk_id in all_chunks:
            rrf_score = 0.0

            if chunk_id in vector_ranks:
                rrf_score += 1.0 / (k + vector_ranks[chunk_id])

            if chunk_id in lexical_ranks:
                rrf_score += 1.0 / (k + lexical_ranks[chunk_id])

            rrf_scores.append((
                chunk_id,
                rrf_score,
                vector_scores.get(chunk_id),
                lexical_scores.get(chunk_id),
            ))

        # Sort by RRF score (descending)
        rrf_scores.sort(key=lambda x: x[1], reverse=True)

        return rrf_scores[:self.config.top_n_final]

    def _fetch_chunks(
            self,
            conn,
            merged_results: List[Tuple[int, float, Optional[float], Optional[float]]],
    ) -> List[RetrievedChunk]:
        """Fetch full chunk data for merged results."""
        if not merged_results:
            return []

        chunk_ids = [r[0] for r in merged_results]
        scores_lookup = {r[0]: (r[1], r[2], r[3]) for r in merged_results}

        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    c.chunk_id, c.doc_id, c.ticker, c.doc_type, c.asof_ts_utc,
                    c.section, c.speaker, c.speaker_role, c.text,
                    c.char_start, c.char_end,
                    d.period_label, d.title
                FROM rag.doc_chunks c
                JOIN rag.documents d ON c.doc_id = d.doc_id
                WHERE c.chunk_id = ANY(%s)
            """, (chunk_ids,))

            rows = cur.fetchall()

        # Build chunks with scores
        chunks = []
        for row in rows:
            chunk_id = row[0]
            rrf_score, vector_score, lexical_score = scores_lookup[chunk_id]

            chunk = RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=row[1],
                ticker=row[2],
                doc_type=row[3],
                asof_ts_utc=row[4],
                section=row[5],
                speaker=row[6],
                speaker_role=row[7],
                text=row[8],
                char_start=row[9],
                char_end=row[10],
                period_label=row[11],
                doc_title=row[12],
                rrf_score=rrf_score,
                vector_score=vector_score,
                lexical_score=lexical_score,
            )
            chunks.append(chunk)

        # Sort by RRF score and assign ranks
        chunks.sort(key=lambda c: c.rrf_score or 0, reverse=True)
        for i, chunk in enumerate(chunks):
            chunk.final_rank = i + 1

        return chunks

    def _apply_gating(
            self,
            chunks: List[RetrievedChunk],
            config: RetrievalConfig,
    ) -> Tuple[GatingDecision, Dict[str, Any]]:
        """
        Apply gating rules to determine if retrieval is trustworthy.

        Returns (decision, details_dict)
        """
        details = {
            'total_chunks': len(chunks),
            'min_similarity_required': config.min_cosine_similarity,
            'min_chunks_required': config.min_chunks_above_threshold,
            'chunk_threshold': config.chunk_similarity_threshold,
        }

        # No results
        if not chunks:
            return GatingDecision.FAIL_NO_RESULTS, details

        # Get top similarity (from vector score)
        top_similarity = max(
            (c.vector_score for c in chunks if c.vector_score is not None),
            default=0.0
        )
        details['top_similarity'] = top_similarity

        # Check minimum similarity
        if top_similarity < config.min_cosine_similarity:
            details['reason'] = f"Top similarity {top_similarity:.3f} < {config.min_cosine_similarity}"
            return GatingDecision.FAIL_SIMILARITY, details

        # Count chunks above threshold
        chunks_above = sum(
            1 for c in chunks
            if c.vector_score is not None and c.vector_score >= config.chunk_similarity_threshold
        )
        details['chunks_above_threshold'] = chunks_above

        if chunks_above < config.min_chunks_above_threshold:
            details['reason'] = f"Only {chunks_above} chunks above threshold (need {config.min_chunks_above_threshold})"
            return GatingDecision.FAIL_MIN_CHUNKS, details

        # Check recency (if configured)
        if config.max_document_age_days:
            cutoff = datetime.now(timezone.utc) - timedelta(days=config.max_document_age_days)
            newest_doc = max(c.asof_ts_utc for c in chunks) if chunks else None

            if newest_doc and newest_doc < cutoff:
                details['newest_doc'] = newest_doc.isoformat()
                details['cutoff'] = cutoff.isoformat()
                details['reason'] = f"Newest document is older than {config.max_document_age_days} days"
                return GatingDecision.FAIL_STALE, details

        # All gates passed
        details['reason'] = "All gating criteria met"
        return GatingDecision.PASS, details

    def _log_retrieval(
            self,
            conn,
            snapshot_id: Optional[int],
            query: str,
            query_embedding: Optional[np.ndarray],
            result: 'RetrievalResult',
            filters: Dict[str, Any],
    ) -> int:
        """Log retrieval to rag_retrievals table."""

        embedding_list = query_embedding.tolist() if query_embedding is not None else None
        embedding_str = None
        if embedding_list:
            embedding_str = "[" + ",".join(str(x) for x in embedding_list) + "]"

        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO rag.rag_retrievals (
                    snapshot_id, query_text, query_embedding, query_embedding_model,
                    retrieval_method, k_vector, k_lexical, rrf_k, top_n_final,
                    Json(filters), retrieved_chunk_ids, scores, 
                    gating_decision, gating_details
                ) VALUES (
                    %s, %s, %s::vector, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s
                ) RETURNING retrieval_id
            """, (
                snapshot_id,
                query,
                embedding_str,
                self.config.embedding_model,
                'hybrid_rrf_v1',
                self.config.k_vector,
                self.config.k_lexical,
                self.config.rrf_k,
                self.config.top_n_final,
                filters,
                [c.chunk_id for c in result.chunks],
                Json({
                    'top_similarity': result.top_similarity,
                    'chunks_above_threshold': result.chunks_above_threshold,
                    'chunk_scores': [
                        {
                            'chunk_id': c.chunk_id,
                            'vector': c.vector_score,
                            'lexical': c.lexical_score,
                            'rrf': c.rrf_score,
                        }
                        for c in result.chunks
                    ]
                }),
                result.gating_decision.value,
                Json(result.gating_details),
            ))

            retrieval_id = cur.fetchone()[0]
            conn.commit()

        return retrieval_id

    def retrieve(
            self,
            query: str,
            ticker: Optional[str] = None,
            doc_types: Optional[List[str]] = None,
            min_date: Optional[datetime] = None,
            snapshot_id: Optional[int] = None,
            log_retrieval: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User's query
            ticker: Filter by ticker (optional)
            doc_types: Filter by document types (optional)
            min_date: Only consider documents after this date (optional)
            snapshot_id: Link to request_snapshots table (for audit)
            log_retrieval: Whether to log to rag_retrievals table

        Returns:
            RetrievalResult with chunks and gating decision
        """
        import time
        start_time = time.time()

        # Generate query embedding
        query_embedding = self.embed_query(query)

        if query_embedding is None:
            return RetrievalResult(
                query=query,
                query_embedding=None,
                chunks=[],
                top_similarity=0.0,
                chunks_above_threshold=0,
                gating_decision=GatingDecision.FAIL_NO_EMBEDDING,
                gating_details={'reason': 'Query embedding generation failed'},
                retrieval_time_ms=int((time.time() - start_time) * 1000),
            )

        filters = {
            'ticker': ticker,
            'doc_types': doc_types,
            'min_date': min_date.isoformat() if min_date else None,
        }

        with get_connection() as conn:
            # Vector search
            vector_results = self._vector_search(
                conn, query_embedding, ticker, doc_types, min_date
            )
            logger.debug(f"Vector search returned {len(vector_results)} results")

            # Lexical search
            lexical_results = self._lexical_search(
                conn, query, ticker, doc_types, min_date
            )
            logger.debug(f"Lexical search returned {len(lexical_results)} results")

            # RRF merge
            merged = self._rrf_merge(vector_results, lexical_results)
            logger.debug(f"RRF merge produced {len(merged)} results")

            # Fetch full chunk data
            chunks = self._fetch_chunks(conn, merged)

            # Apply gating
            gating_decision, gating_details = self._apply_gating(chunks, self.config)

            # Calculate summary stats
            top_similarity = max(
                (c.vector_score for c in chunks if c.vector_score is not None),
                default=0.0
            )
            chunks_above = sum(
                1 for c in chunks
                if c.vector_score is not None
                and c.vector_score >= self.config.chunk_similarity_threshold
            )

            retrieval_time_ms = int((time.time() - start_time) * 1000)

            result = RetrievalResult(
                query=query,
                query_embedding=query_embedding.tolist(),
                chunks=chunks,
                top_similarity=top_similarity,
                chunks_above_threshold=chunks_above,
                gating_decision=gating_decision,
                gating_details=gating_details,
                retrieval_time_ms=retrieval_time_ms,
            )

            # Log retrieval
            if log_retrieval:
                try:
                    retrieval_id = self._log_retrieval(
                        conn, snapshot_id, query, query_embedding, result, filters
                    )
                    result.retrieval_id = retrieval_id
                except Exception as e:
                    logger.error(f"Failed to log retrieval: {e}")

        logger.info(
            f"Retrieval complete: {len(chunks)} chunks, "
            f"top_sim={top_similarity:.3f}, gating={gating_decision.value}, "
            f"time={retrieval_time_ms}ms"
        )

        return result

    def retrieve_for_ticker_analysis(
            self,
            ticker: str,
            query: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Convenience method for retrieving context for ticker analysis.

        If no query provided, uses a default query for recent insights.
        """
        if query is None:
            query = f"Recent earnings guidance outlook demand trends for {ticker}"

        return self.retrieve(
            query=query,
            ticker=ticker,
            doc_types=['transcript', '10k', '10q', '8k'],
            min_date=datetime.now(timezone.utc) - timedelta(days=365),
        )


# Convenience function
def retrieve_chunks(
        query: str,
        ticker: Optional[str] = None,
        doc_types: Optional[List[str]] = None,
) -> RetrievalResult:
    """Quick function to retrieve relevant chunks."""
    retriever = RAGRetriever()
    return retriever.retrieve(query, ticker, doc_types)


if __name__ == "__main__":
    # Test retrieval
    print("Testing RAG Retriever...")

    retriever = RAGRetriever()
    print(f"Config: k_vector={retriever.config.k_vector}, k_lexical={retriever.config.k_lexical}")
    print(f"Gating: min_similarity={retriever.config.min_cosine_similarity}")

    # Test query (will only work if documents are ingested)
    result = retriever.retrieve(
        query="What did management say about AI demand?",
        ticker="MU",
    )

    print(f"\nResults:")
    print(f"  Chunks retrieved: {len(result.chunks)}")
    print(f"  Top similarity: {result.top_similarity:.3f}")
    print(f"  Gating: {result.gating_decision.value}")
    print(f"  Time: {result.retrieval_time_ms}ms")

    if result.chunks:
        print(f"\nTop chunk:")
        top = result.chunks[0]
        print(f"  ID: {top.chunk_id}")
        print(f"  Doc: {top.doc_type} / {top.period_label}")
        print(f"  Text: {top.text[:200]}...")