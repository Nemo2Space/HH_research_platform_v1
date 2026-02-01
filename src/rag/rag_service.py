"""
RAG Service (Batch B)
=====================

Complete RAG pipeline integrating:
- Hybrid retrieval (vector + lexical + RRF)
- Gating rules
- Prompt building with citations
- LLM generation
- Full audit logging

Usage:
    from src.rag.rag_service import RAGService

    service = RAGService()
    response = service.query(
        ticker="MU",
        query="What did management say about AI demand?",
        metrics=computed_metrics,
    )

    print(response.answer)
    print(response.citations)

Author: HH Research Platform
"""

import os
import hashlib
import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from src.db.connection import get_connection
from src.utils.logging import get_logger
from src.rag.retrieval import RAGRetriever, RetrievalConfig, RetrievalResult, GatingDecision
from src.rag.prompt_builder import RAGPromptBuilder, PromptConfig

logger = get_logger(__name__)

# Try to import OpenAI client for LLM
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class Citation:
    """A citation extracted from LLM response."""
    chunk_id: int
    text_span: str  # The text that was cited
    start_pos: int  # Position in response
    end_pos: int


@dataclass
class RAGResponse:
    """Complete RAG response with audit trail."""

    # Query
    query: str
    ticker: str

    # Answer
    answer: str
    citations: List[Citation]

    # Retrieval info
    retrieval_result: RetrievalResult
    chunks_used: int

    # Gating
    gating_passed: bool
    gating_reason: str

    # Model info
    model_name: str
    prompt_tokens: int
    response_tokens: int

    # Timing
    total_time_ms: int
    retrieval_time_ms: int
    generation_time_ms: int

    # Audit
    response_id: Optional[int] = None
    snapshot_id: Optional[int] = None
    retrieval_id: Optional[int] = None

    # Quality indicators
    unknown_claims: List[str] = field(default_factory=list)
    confidence_score: Optional[float] = None


@dataclass
class RAGServiceConfig:
    """Configuration for RAG service."""

    # LLM settings
    llm_base_url: str = "http://localhost:8090/v1"
    llm_model: str = "qwen3-32b"
    llm_api_key: str = "not-needed"
    llm_max_tokens: int = 1500
    llm_temperature: float = 0.3

    # Fallback LLM (if local fails)
    fallback_enabled: bool = True
    fallback_base_url: str = "https://api.openai.com/v1"
    fallback_model: str = "gpt-4o-mini"
    fallback_api_key: Optional[str] = None

    # Retrieval config
    retrieval_config: RetrievalConfig = field(default_factory=RetrievalConfig.default)

    # Prompt config
    prompt_config: PromptConfig = field(default_factory=PromptConfig)

    # Behavior when gating fails
    answer_on_gating_fail: bool = True  # Still generate but with warning
    gating_fail_prefix: str = "⚠️ Limited evidence available. "

    @classmethod
    def default(cls) -> 'RAGServiceConfig':
        return cls()

    @classmethod
    def from_env(cls) -> 'RAGServiceConfig':
        """Load config from environment variables."""
        return cls(
            llm_base_url=os.getenv('LLM_BASE_URL', 'http://localhost:8090/v1'),
            llm_model=os.getenv('LLM_MODEL', 'qwen3-32b'),
            llm_api_key=os.getenv('LLM_API_KEY', 'not-needed'),
            fallback_api_key=os.getenv('OPENAI_API_KEY'),
        )


class RAGService:
    """
    Complete RAG service with retrieval, generation, and audit.
    """

    def __init__(self, config: RAGServiceConfig = None):
        self.config = config or RAGServiceConfig.from_env()
        self.retriever = RAGRetriever(self.config.retrieval_config)
        self.prompt_builder = RAGPromptBuilder(self.config.prompt_config)
        self._llm_client = None
        self._fallback_client = None

    def _get_llm_client(self) -> 'OpenAI':
        """Get or create LLM client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required. Install with: pip install openai")

        if self._llm_client is None:
            self._llm_client = OpenAI(
                base_url=self.config.llm_base_url,
                api_key=self.config.llm_api_key,
            )
        return self._llm_client

    def _get_fallback_client(self) -> Optional['OpenAI']:
        """Get or create fallback LLM client."""
        if not self.config.fallback_enabled or not self.config.fallback_api_key:
            return None

        if not OPENAI_AVAILABLE:
            return None

        if self._fallback_client is None:
            self._fallback_client = OpenAI(
                base_url=self.config.fallback_base_url,
                api_key=self.config.fallback_api_key,
            )
        return self._fallback_client

    def _extract_citations(self, response_text: str) -> List[Citation]:
        """Extract [chunk:XXXXX] citations from response."""
        import re

        citations = []
        pattern = r'\[chunk:(\d+)\]'

        for match in re.finditer(pattern, response_text):
            chunk_id = int(match.group(1))

            # Find surrounding text (the claim being cited)
            start = max(0, match.start() - 100)
            end = match.start()

            # Find sentence boundary
            text_before = response_text[start:end]
            sentence_start = max(
                text_before.rfind('. ') + 2,
                text_before.rfind('\n') + 1,
                0
            )

            text_span = text_before[sentence_start:].strip()

            citations.append(Citation(
                chunk_id=chunk_id,
                text_span=text_span,
                start_pos=match.start(),
                end_pos=match.end(),
            ))

        return citations

    def _extract_unknown_claims(self, response_text: str) -> List[str]:
        """Extract any UNKNOWN statements from response."""
        import re

        unknowns = []
        patterns = [
            r'UNKNOWN[:\s-]+([^.]+)',
            r'no evidence available[:\s-]+([^.]+)',
            r'not (?:found|available) in (?:the )?(?:evidence|context)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, response_text, re.IGNORECASE):
                unknowns.append(match.group(0))

        return unknowns

    def _generate_response(
            self,
            system_prompt: str,
            user_prompt: str,
            use_fallback: bool = False,
    ) -> tuple:
        """
        Generate response from LLM.

        Returns (response_text, model_name, prompt_tokens, response_tokens)
        """
        client = self._get_fallback_client() if use_fallback else self._get_llm_client()
        model = self.config.fallback_model if use_fallback else self.config.llm_model

        if client is None:
            raise RuntimeError("No LLM client available")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
            )

            return (
                response.choices[0].message.content,
                model,
                response.usage.prompt_tokens if response.usage else 0,
                response.usage.completion_tokens if response.usage else 0,
            )

        except Exception as e:
            if not use_fallback and self._get_fallback_client():
                logger.warning(f"Primary LLM failed, trying fallback: {e}")
                return self._generate_response(system_prompt, user_prompt, use_fallback=True)
            raise

    def _log_response(
            self,
            conn,
            snapshot_id: Optional[int],
            retrieval_id: Optional[int],
            response: RAGResponse,
            prompt_hash: str,
    ) -> int:
        """Log response to ai_responses table."""

        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO rag.ai_responses (
                    snapshot_id, retrieval_id,
                    model_name, model_endpoint, prompt_version, prompt_hash,
                    prompt_tokens, context_tokens, response_text, response_tokens,
                    citations, unknown_claims, confidence_score, latency_ms
                ) VALUES (
                    %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s
                ) RETURNING response_id
            """, (
                snapshot_id,
                retrieval_id,
                response.model_name,
                self.config.llm_base_url,
                self.prompt_builder.config.prompt_version,
                prompt_hash,
                response.prompt_tokens,
                response.chunks_used * 500,  # Approximate context tokens
                response.answer,
                response.response_tokens,
                json.dumps([{
                    'chunk_id': c.chunk_id,
                    'text_span': c.text_span,
                } for c in response.citations]),
                json.dumps(response.unknown_claims) if response.unknown_claims else None,
                response.confidence_score,
                response.total_time_ms,
            ))

            response_id = cur.fetchone()[0]
            conn.commit()

        return response_id

    def query(
            self,
            ticker: str,
            query: str,
            metrics: Dict[str, Any] = None,
            doc_types: Optional[List[str]] = None,
            snapshot_id: Optional[int] = None,
            log_response: bool = True,
    ) -> RAGResponse:
        """
        Execute complete RAG query.

        Args:
            ticker: Stock ticker
            query: User's question
            metrics: Computed metrics dict (optional)
            doc_types: Document types to search (optional)
            snapshot_id: Link to request_snapshots (for audit)
            log_response: Whether to log to ai_responses table

        Returns:
            RAGResponse with answer, citations, and audit info
        """
        import time
        start_time = time.time()

        # Step 1: Retrieve relevant chunks
        retrieval_result = self.retriever.retrieve(
            query=query,
            ticker=ticker,
            doc_types=doc_types,
            snapshot_id=snapshot_id,
        )

        retrieval_time_ms = retrieval_result.retrieval_time_ms

        # Step 2: Check gating
        gating_passed = retrieval_result.gating_passed
        gating_reason = retrieval_result.gating_details.get('reason', 'Unknown')

        # Step 3: Build prompt
        prompt = self.prompt_builder.build_full_prompt(
            ticker=ticker,
            query=query,
            chunks=retrieval_result.chunks,
            metrics=metrics,
        )

        prompt_hash = hashlib.sha256(
            (prompt['system'] + prompt['user']).encode()
        ).hexdigest()[:16]

        # Step 4: Generate response
        generation_start = time.time()

        try:
            # Modify prompt if gating failed
            user_prompt = prompt['user']
            if not gating_passed and self.config.answer_on_gating_fail:
                user_prompt = (
                        f"NOTE: Evidence retrieval confidence is low ({retrieval_result.gating_decision.value}). "
                        f"Be extra cautious about unsupported claims.\n\n"
                        + user_prompt
                )

            answer, model_name, prompt_tokens, response_tokens = self._generate_response(
                prompt['system'],
                user_prompt,
            )

            # Add warning prefix if gating failed
            if not gating_passed and self.config.answer_on_gating_fail:
                answer = self.config.gating_fail_prefix + answer

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = f"Unable to generate response: {str(e)}"
            model_name = "error"
            prompt_tokens = 0
            response_tokens = 0

        generation_time_ms = int((time.time() - generation_start) * 1000)

        # Step 5: Extract citations and unknowns
        citations = self._extract_citations(answer)
        unknown_claims = self._extract_unknown_claims(answer)

        # Calculate confidence score
        confidence_score = None
        if retrieval_result.chunks:
            # Based on: gating pass, citation count, similarity
            base_score = 0.5 if gating_passed else 0.2
            citation_bonus = min(len(citations) * 0.05, 0.25)
            similarity_bonus = retrieval_result.top_similarity * 0.25
            confidence_score = min(base_score + citation_bonus + similarity_bonus, 1.0)

        total_time_ms = int((time.time() - start_time) * 1000)

        # Build response
        response = RAGResponse(
            query=query,
            ticker=ticker,
            answer=answer,
            citations=citations,
            retrieval_result=retrieval_result,
            chunks_used=len(retrieval_result.chunks),
            gating_passed=gating_passed,
            gating_reason=gating_reason,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            total_time_ms=total_time_ms,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            snapshot_id=snapshot_id,
            retrieval_id=retrieval_result.retrieval_id,
            unknown_claims=unknown_claims,
            confidence_score=confidence_score,
        )

        # Step 6: Log response
        if log_response:
            try:
                with get_connection() as conn:
                    response_id = self._log_response(
                        conn, snapshot_id, retrieval_result.retrieval_id,
                        response, prompt_hash
                    )
                    response.response_id = response_id
            except Exception as e:
                logger.error(f"Failed to log response: {e}")

        # Fixed format string - use (confidence_score or 0) instead of ternary in f-string
        conf_value = confidence_score if confidence_score is not None else 0
        logger.info(
            f"RAG query complete: {len(citations)} citations, "
            f"confidence={conf_value:.2f}, "
            f"time={total_time_ms}ms"
        )

        return response

    def query_simple(
            self,
            ticker: str,
            query: str,
    ) -> str:
        """
        Simple query interface - returns just the answer string.

        For quick integration without handling full response object.
        """
        response = self.query(ticker, query, log_response=False)
        return response.answer


# Convenience functions
def rag_query(
        ticker: str,
        query: str,
        metrics: Dict[str, Any] = None,
) -> RAGResponse:
    """Quick function to execute RAG query."""
    service = RAGService()
    return service.query(ticker, query, metrics)


def rag_answer(ticker: str, query: str) -> str:
    """Quick function to get just the answer."""
    service = RAGService()
    return service.query_simple(ticker, query)


if __name__ == "__main__":
    print("Testing RAG Service...")

    # Create service with config
    config = RAGServiceConfig(
        llm_base_url="http://localhost:8090/v1",
        llm_model="qwen3-32b",
    )

    service = RAGService(config)

    print(f"LLM: {config.llm_base_url} / {config.llm_model}")
    print(f"Retrieval: k_vector={config.retrieval_config.k_vector}")

    # Test query (will work if docs are ingested and LLM is running)
    try:
        response = service.query(
            ticker="MU",
            query="What did management say about AI demand?",
        )

        print(f"\nQuery: {response.query}")
        print(f"Gating: {'PASS' if response.gating_passed else 'FAIL'} ({response.gating_reason})")
        print(f"Chunks: {response.chunks_used}")
        print(f"Citations: {len(response.citations)}")
        conf_display = f"{response.confidence_score:.2f}" if response.confidence_score else "N/A"
        print(f"Confidence: {conf_display}")
        print(f"Time: {response.total_time_ms}ms")
        print(f"\nAnswer:\n{response.answer[:500]}...")

    except Exception as e:
        print(f"Test failed (expected if no docs or LLM): {e}")