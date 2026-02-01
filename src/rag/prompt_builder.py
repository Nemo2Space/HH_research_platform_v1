"""
RAG Prompt Builder (Batch B)
============================

Builds strict prompts with:
- COMPUTED_METRICS section (structured data)
- EVIDENCE section (retrieved chunks with citations)
- Mandatory citation format [chunk:12345]
- UNKNOWN handling for unsupported claims

Usage:
    from src.rag.prompt_builder import RAGPromptBuilder

    builder = RAGPromptBuilder()
    prompt = builder.build_analysis_prompt(
        ticker="MU",
        query="What's the outlook for AI demand?",
        chunks=retrieval_result.chunks,
        metrics=computed_metrics_dict,
    )

Author: HH Research Platform
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone

from src.rag.retrieval import RetrievedChunk
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PromptConfig:
    """Configuration for prompt building."""

    # Max tokens for evidence section
    max_evidence_tokens: int = 6000

    # Max chunks to include
    max_chunks: int = 12

    # Approximate chars per token
    chars_per_token: float = 4.0

    # Include speaker info for transcripts
    include_speaker: bool = True

    # Include section labels
    include_section: bool = True

    # Prompt version (for audit trail)
    prompt_version: str = "strict_citation_v1"


SYSTEM_PROMPT_TEMPLATE = """You are a financial analyst assistant for the HH Research Platform. 

CRITICAL RULES:
1. ONLY use information from the COMPUTED_METRICS and EVIDENCE sections below
2. EVERY factual claim MUST include a citation in format [chunk:XXXXX]
3. If information is not in COMPUTED_METRICS or EVIDENCE, respond with "UNKNOWN - no evidence available"
4. NEVER fabricate numbers, dates, or quotes
5. Distinguish between facts (cite) and your analysis (no cite needed)

CITATION FORMAT:
- Single source: "Revenue grew 15% [chunk:12345]"
- Multiple sources: "Both companies reported growth [chunk:12345][chunk:12346]"
- No citation for analysis: "This suggests strong momentum in the sector"

RESPONSE STRUCTURE:
1. Start with the direct answer to the question
2. Support with evidence from COMPUTED_METRICS and EVIDENCE
3. Note any limitations or missing information
4. Keep response focused and concise"""

ANALYSIS_PROMPT_TEMPLATE = """<COMPUTED_METRICS ticker="{ticker}">
{metrics_section}
</COMPUTED_METRICS>

<EVIDENCE>
{evidence_section}
</EVIDENCE>

<USER_QUERY>
{query}
</USER_QUERY>

Provide a response following the citation rules. If the evidence doesn't contain relevant information, state what's missing."""


class RAGPromptBuilder:
    """
    Builds prompts with strict citation requirements.

    Ensures LLM responses are grounded in provided evidence
    with mandatory citations for traceability.
    """

    def __init__(self, config: PromptConfig = None):
        self.config = config or PromptConfig()

    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format computed metrics as structured text."""
        if not metrics:
            return "No computed metrics available."

        lines = []

        # Price/market data
        if 'price' in metrics:
            lines.append(f"Current Price: ${metrics['price']:.2f}")
        if 'change_pct' in metrics:
            lines.append(f"Daily Change: {metrics['change_pct']:+.2f}%")
        if 'market_cap' in metrics:
            cap = metrics['market_cap']
            if cap >= 1e12:
                lines.append(f"Market Cap: ${cap / 1e12:.2f}T")
            elif cap >= 1e9:
                lines.append(f"Market Cap: ${cap / 1e9:.2f}B")
            else:
                lines.append(f"Market Cap: ${cap / 1e6:.2f}M")

        # Valuation
        if 'pe_ratio' in metrics and metrics['pe_ratio']:
            lines.append(f"P/E Ratio: {metrics['pe_ratio']:.2f}")
        if 'forward_pe' in metrics and metrics['forward_pe']:
            lines.append(f"Forward P/E: {metrics['forward_pe']:.2f}")
        if 'peg_ratio' in metrics and metrics['peg_ratio']:
            lines.append(f"PEG Ratio: {metrics['peg_ratio']:.2f}")

        # Growth
        if 'revenue_growth' in metrics and metrics['revenue_growth'] is not None:
            lines.append(f"Revenue Growth (YoY): {metrics['revenue_growth'] * 100:.1f}%")
        if 'earnings_growth' in metrics and metrics['earnings_growth'] is not None:
            lines.append(f"Earnings Growth (YoY): {metrics['earnings_growth'] * 100:.1f}%")

        # Signals
        if 'signal_type' in metrics:
            lines.append(f"Signal: {metrics['signal_type']}")
        if 'total_score' in metrics:
            lines.append(f"Composite Score: {metrics['total_score']:.1f}/100")

        # Component scores
        component_scores = [
            ('technical_score', 'Technical'),
            ('fundamental_score', 'Fundamental'),
            ('sentiment_score', 'Sentiment'),
            ('risk_score', 'Risk'),
        ]
        score_parts = []
        for key, label in component_scores:
            if key in metrics and metrics[key] is not None:
                score_parts.append(f"{label}: {metrics[key]:.1f}")
        if score_parts:
            lines.append(f"Component Scores: {', '.join(score_parts)}")

        # Options flow
        if 'put_call_ratio' in metrics and metrics['put_call_ratio'] is not None:
            lines.append(f"Put/Call Ratio: {metrics['put_call_ratio']:.2f}")
        if 'options_sentiment' in metrics:
            lines.append(f"Options Sentiment: {metrics['options_sentiment']}")

        # Data timestamp
        if 'data_as_of' in metrics:
            lines.append(f"Data as of: {metrics['data_as_of']}")

        return "\n".join(lines) if lines else "No computed metrics available."

    def _format_chunk(self, chunk: RetrievedChunk, index: int) -> str:
        """Format a single chunk for the evidence section."""
        lines = []

        # Header with chunk ID and metadata
        header_parts = [f"[chunk:{chunk.chunk_id}]"]

        if chunk.doc_type:
            header_parts.append(f"Type: {chunk.doc_type.upper()}")

        if chunk.period_label:
            header_parts.append(f"Period: {chunk.period_label}")

        if chunk.asof_ts_utc:
            header_parts.append(f"Date: {chunk.asof_ts_utc.strftime('%Y-%m-%d')}")

        lines.append(" | ".join(header_parts))

        # Section and speaker (for transcripts)
        if self.config.include_section and chunk.section:
            lines.append(f"Section: {chunk.section}")

        if self.config.include_speaker and chunk.speaker:
            speaker_info = chunk.speaker
            if chunk.speaker_role:
                speaker_info += f" ({chunk.speaker_role})"
            lines.append(f"Speaker: {speaker_info}")

        # Relevance score
        if chunk.vector_score is not None:
            lines.append(f"Relevance: {chunk.vector_score:.2%}")

        # Content
        lines.append(f"Content: {chunk.text}")

        return "\n".join(lines)

    def _format_evidence(self, chunks: List[RetrievedChunk]) -> str:
        """Format all chunks into evidence section."""
        if not chunks:
            return "No evidence retrieved."

        # Limit chunks
        chunks_to_use = chunks[:self.config.max_chunks]

        # Calculate token budget
        max_chars = int(self.config.max_evidence_tokens * self.config.chars_per_token)

        formatted_chunks = []
        total_chars = 0

        for i, chunk in enumerate(chunks_to_use):
            formatted = self._format_chunk(chunk, i)

            # Check if adding this chunk exceeds budget
            if total_chars + len(formatted) > max_chars and formatted_chunks:
                logger.debug(f"Evidence truncated at chunk {i} due to token budget")
                break

            formatted_chunks.append(formatted)
            total_chars += len(formatted)

        return "\n\n---\n\n".join(formatted_chunks)

    def build_system_prompt(self) -> str:
        """Build the system prompt with citation rules."""
        return SYSTEM_PROMPT_TEMPLATE

    def build_analysis_prompt(
            self,
            ticker: str,
            query: str,
            chunks: List[RetrievedChunk],
            metrics: Dict[str, Any] = None,
    ) -> str:
        """
        Build the analysis prompt with metrics and evidence.

        Args:
            ticker: Stock ticker
            query: User's question
            chunks: Retrieved chunks from RAG
            metrics: Computed metrics dict (optional)

        Returns:
            Formatted prompt string
        """
        metrics_section = self._format_metrics(metrics or {})
        evidence_section = self._format_evidence(chunks)

        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            ticker=ticker,
            metrics_section=metrics_section,
            evidence_section=evidence_section,
            query=query,
        )

        return prompt

    def build_full_prompt(
            self,
            ticker: str,
            query: str,
            chunks: List[RetrievedChunk],
            metrics: Dict[str, Any] = None,
    ) -> Dict[str, str]:
        """
        Build complete prompt with system and user messages.

        Returns dict with 'system' and 'user' keys for chat completion.
        """
        return {
            'system': self.build_system_prompt(),
            'user': self.build_analysis_prompt(ticker, query, chunks, metrics),
        }

    def get_prompt_metadata(self) -> Dict[str, Any]:
        """Get metadata about the prompt for audit logging."""
        import hashlib

        system_hash = hashlib.sha256(
            self.build_system_prompt().encode()
        ).hexdigest()[:16]

        return {
            'prompt_version': self.config.prompt_version,
            'system_prompt_hash': system_hash,
            'max_evidence_tokens': self.config.max_evidence_tokens,
            'max_chunks': self.config.max_chunks,
        }


# Convenience functions
def build_rag_prompt(
        ticker: str,
        query: str,
        chunks: List[RetrievedChunk],
        metrics: Dict[str, Any] = None,
) -> Dict[str, str]:
    """Quick function to build RAG prompt."""
    builder = RAGPromptBuilder()
    return builder.build_full_prompt(ticker, query, chunks, metrics)


if __name__ == "__main__":
    # Test prompt building
    from datetime import datetime, timezone

    print("Testing RAG Prompt Builder...")

    # Mock chunks
    mock_chunks = [
        RetrievedChunk(
            chunk_id=12345,
            doc_id=1,
            ticker="MU",
            doc_type="transcript",
            asof_ts_utc=datetime.now(timezone.utc),
            section="Prepared Remarks",
            speaker="Sanjay Mehrotra",
            speaker_role="CEO",
            text="We are seeing unprecedented demand for HBM driven by AI workloads. Our HBM3E is ramping significantly.",
            char_start=0,
            char_end=100,
            vector_score=0.85,
            lexical_score=0.7,
            rrf_score=0.02,
            period_label="2025Q3",
        ),
        RetrievedChunk(
            chunk_id=12346,
            doc_id=1,
            ticker="MU",
            doc_type="transcript",
            asof_ts_utc=datetime.now(timezone.utc),
            section="Q&A",
            speaker="Mark Murphy",
            speaker_role="CFO",
            text="We expect revenue to grow 20-25% sequentially, driven primarily by data center.",
            char_start=100,
            char_end=200,
            vector_score=0.78,
            lexical_score=0.65,
            rrf_score=0.018,
            period_label="2025Q3",
        ),
    ]

    # Mock metrics
    mock_metrics = {
        'price': 98.50,
        'change_pct': 2.3,
        'market_cap': 110e9,
        'pe_ratio': 25.4,
        'revenue_growth': 0.82,
        'signal_type': 'BUY',
        'total_score': 74.5,
        'technical_score': 68.0,
        'fundamental_score': 82.0,
        'sentiment_score': 71.0,
    }

    builder = RAGPromptBuilder()
    result = builder.build_full_prompt(
        ticker="MU",
        query="What's the outlook for AI-related memory demand?",
        chunks=mock_chunks,
        metrics=mock_metrics,
    )

    print("\n=== SYSTEM PROMPT ===")
    print(result['system'][:500] + "...")

    print("\n=== USER PROMPT ===")
    print(result['user'])

    print("\n=== METADATA ===")
    print(builder.get_prompt_metadata())