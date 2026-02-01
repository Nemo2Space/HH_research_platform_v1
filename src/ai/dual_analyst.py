"""
Dual Analyst Service - Fixed for Actual Schema
==============================================

Runs two independent analysts in parallel:
1. SQL Analyst - Uses structured data (prices, signals, fundamentals)
2. RAG Analyst - Uses SEC filings and extracted facts

Then evaluates both opinions for agreement/conflicts and synthesizes.

Usage:
    from src.ai.dual_analyst import DualAnalystService

    service = DualAnalystService()
    result = service.analyze("MU", "What's the outlook?")

Author: HH Research Platform
"""

import os
import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from src.db.connection import get_connection
from src.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class Sentiment(Enum):
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"
    UNKNOWN = "unknown"


@dataclass
class Opinion:
    """Opinion from a single analyst."""
    analyst_type: str
    ticker: str
    question: str
    summary: str
    sentiment: Sentiment
    confidence: float
    key_points: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    bullish_factors: List[str] = field(default_factory=list)
    bearish_factors: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""
    latency_ms: int = 0


@dataclass
class DualAnalysisResult:
    """Combined result from both analysts."""
    ticker: str
    question: str
    timestamp: datetime
    sql_opinion: Optional[Opinion] = None
    rag_opinion: Optional[Opinion] = None
    agreement_score: float = 0.0
    sentiment_match: bool = False
    conflicts: List[str] = field(default_factory=list)
    synthesis: str = ""
    final_sentiment: Sentiment = Sentiment.UNKNOWN
    final_confidence: float = 0.0
    combined_bullish: List[str] = field(default_factory=list)
    combined_bearish: List[str] = field(default_factory=list)
    combined_risks: List[str] = field(default_factory=list)
    total_latency_ms: int = 0


@dataclass
class DualAnalystConfig:
    llm_base_url: str = "http://localhost:8080/v1"
    llm_model: str = "qwen3-32b"
    llm_api_key: str = "not-needed"
    fallback_enabled: bool = True
    fallback_base_url: str = "https://api.openai.com/v1"
    fallback_model: str = "gpt-4o-mini"
    fallback_api_key: Optional[str] = None
    run_parallel: bool = True
    max_tokens: int = 1500
    temperature: float = 0.3

    @classmethod
    def from_env(cls) -> 'DualAnalystConfig':
        return cls(
            llm_base_url=os.getenv('LLM_BASE_URL', 'http://localhost:8080/v1'),
            llm_model=os.getenv('LLM_MODEL', 'qwen3-32b'),
            fallback_api_key=os.getenv('OPENAI_API_KEY'),
        )


# Prompts
SQL_ANALYST_SYSTEM = """You are a QUANTITATIVE ANALYST analyzing stocks using structured market data.
Your analysis is based ONLY on the numerical data provided. Be precise and data-driven."""

SQL_ANALYST_PROMPT = """Analyze {ticker} based on this QUANTITATIVE DATA:

{context}

Question: {question}

Return JSON:
{{
    "summary": "2-3 sentence executive summary",
    "sentiment": "very_bullish" | "bullish" | "neutral" | "bearish" | "very_bearish",
    "confidence": 0.0-1.0,
    "key_points": ["point 1", "point 2"],
    "bullish_factors": ["factor 1"],
    "bearish_factors": ["factor 1"],
    "risk_flags": ["risk 1"],
    "data_sources": ["prices", "signals"]
}}"""


RAG_ANALYST_SYSTEM = """You are a QUALITATIVE ANALYST analyzing stocks using SEC filings.
Focus on narrative insights: management guidance, risks, competitive landscape.
ALWAYS cite sources using [chunk:ID] format."""

RAG_ANALYST_PROMPT = """Analyze {ticker} based on SEC FILINGS and EXTRACTED FACTS:

DOCUMENT EXCERPTS:
{rag_context}

EXTRACTED FACTS:
{facts_context}

Question: {question}

Return JSON:
{{
    "summary": "2-3 sentence summary with [chunk:ID] citations",
    "sentiment": "very_bullish" | "bullish" | "neutral" | "bearish" | "very_bearish",
    "confidence": 0.0-1.0,
    "key_points": ["point 1 [chunk:ID]"],
    "bullish_factors": ["factor 1"],
    "bearish_factors": ["factor 1"],
    "risk_flags": ["risk [chunk:ID]"],
    "citations": [{{"chunk_id": 123, "relevance": "why"}}]
}}"""


EVALUATOR_PROMPT = """Evaluate two analyst opinions on {ticker}.

QUANTITATIVE ANALYST:
{sql_summary}
Sentiment: {sql_sentiment}

QUALITATIVE ANALYST:
{rag_summary}
Sentiment: {rag_sentiment}

Return JSON:
{{
    "agreement_score": 0.0-1.0,
    "sentiment_match": true/false,
    "conflicts": ["conflict if any"],
    "synthesis": "Combined analysis (3-4 sentences)",
    "final_sentiment": "very_bullish" | "bullish" | "neutral" | "bearish" | "very_bearish",
    "final_confidence": 0.0-1.0,
    "combined_bullish": ["merged factors"],
    "combined_bearish": ["merged factors"],
    "combined_risks": ["all risks"]
}}"""


class DualAnalystService:
    def __init__(self, config: DualAnalystConfig = None):
        self.config = config or DualAnalystConfig.from_env()
        self._llm_client = None
        self._fallback_client = None

    def _get_llm_client(self):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required")
        if self._llm_client is None:
            self._llm_client = OpenAI(
                base_url=self.config.llm_base_url,
                api_key=self.config.llm_api_key,
            )
        return self._llm_client

    def _get_fallback_client(self):
        if not self.config.fallback_enabled or not self.config.fallback_api_key:
            return None
        if self._fallback_client is None:
            self._fallback_client = OpenAI(
                base_url=self.config.fallback_base_url,
                api_key=self.config.fallback_api_key,
            )
        return self._fallback_client

    def _call_llm(self, system: str, prompt: str, use_fallback: bool = False) -> str:
        client = self._get_fallback_client() if use_fallback else self._get_llm_client()
        model = self.config.fallback_model if use_fallback else self.config.llm_model

        if client is None:
            raise RuntimeError("No LLM client available")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            if not use_fallback and self._get_fallback_client():
                logger.warning(f"Primary LLM failed, using fallback: {e}")
                return self._call_llm(system, prompt, use_fallback=True)
            raise

    def _parse_json(self, response: str) -> Dict[str, Any]:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        try:
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return {}

    # ============================================================
    # SQL ANALYST - Fixed for actual schema
    # ============================================================

    def _get_sql_context(self, ticker: str) -> str:
        """Get structured data context using actual table schema."""
        context_parts = []

        with get_connection() as conn:
            with conn.cursor() as cur:
                # Latest signals from screener_scores
                cur.execute("""
                    SELECT date, total_score, technical_score, fundamental_score,
                           sentiment_score, options_flow_score, composite_score
                    FROM screener_scores
                    WHERE ticker = %s
                    ORDER BY date DESC
                    LIMIT 1
                """, (ticker,))
                row = cur.fetchone()
                if row:
                    context_parts.append(f"""
SIGNALS (as of {row[0]}):
- Total Score: {row[1]}/100
- Technical: {row[2]}/100
- Fundamental: {row[3]}/100  
- Sentiment: {row[4]}/100
- Options Flow: {row[5]}/100
- Composite: {row[6]}/100
""")

                # Fundamentals
                cur.execute("""
                    SELECT pe_ratio, forward_pe, pb_ratio, profit_margin,
                           revenue_growth, earnings_growth, dividend_yield,
                           debt_to_equity, roe, market_cap
                    FROM fundamentals
                    WHERE ticker = %s
                    ORDER BY date DESC
                    LIMIT 1
                """, (ticker,))
                row = cur.fetchone()
                if row:
                    pe = f"{row[0]:.1f}" if row[0] else "N/A"
                    fwd_pe = f"{row[1]:.1f}" if row[1] else "N/A"
                    pb = f"{row[2]:.1f}" if row[2] else "N/A"
                    margin = f"{row[3]*100:.1f}%" if row[3] else "N/A"
                    rev_growth = f"{row[4]*100:.1f}%" if row[4] else "N/A"
                    earn_growth = f"{row[5]*100:.1f}%" if row[5] else "N/A"
                    div_yield = f"{row[6]*100:.2f}%" if row[6] else "N/A"
                    dte = f"{row[7]:.1f}" if row[7] else "N/A"
                    roe_val = f"{row[8]*100:.1f}%" if row[8] else "N/A"
                    mcap = f"${row[9]/1e9:.1f}B" if row[9] else "N/A"

                    context_parts.append(f"""
FUNDAMENTALS:
- Market Cap: {mcap}
- P/E Ratio: {pe}
- Forward P/E: {fwd_pe}
- Price/Book: {pb}
- Profit Margin: {margin}
- Revenue Growth: {rev_growth}
- Earnings Growth: {earn_growth}
- Dividend Yield: {div_yield}
- Debt/Equity: {dte}
- ROE: {roe_val}
""")

                # Latest price
                cur.execute("""
                    SELECT date, close, volume
                    FROM prices
                    WHERE ticker = %s
                    ORDER BY date DESC
                    LIMIT 1
                """, (ticker,))
                row = cur.fetchone()
                if row:
                    context_parts.append(f"""
PRICE DATA:
- Latest Price: ${row[1]:.2f} (as of {row[0]})
- Volume: {row[2]:,}
""")

                # Options flow from options_flow_daily
                cur.execute("""
                    SELECT scan_date, put_call_volume_ratio, overall_sentiment, 
                           sentiment_score, total_call_volume, total_put_volume
                    FROM options_flow_daily
                    WHERE ticker = %s
                    ORDER BY scan_date DESC
                    LIMIT 1
                """, (ticker,))
                row = cur.fetchone()
                if row:
                    context_parts.append(f"""
OPTIONS FLOW (as of {row[0]}):
- Put/Call Ratio: {row[1]:.2f}
- Overall Sentiment: {row[2]}
- Sentiment Score: {row[3]}/100
- Call Volume: {row[4]:,}
- Put Volume: {row[5]:,}
""")

        return "\n".join(context_parts) if context_parts else "No structured data available."

    def _run_sql_analyst(self, ticker: str, question: str) -> Opinion:
        import time
        start = time.time()

        context = self._get_sql_context(ticker)

        prompt = SQL_ANALYST_PROMPT.format(
            ticker=ticker,
            context=context,
            question=question,
        )

        try:
            response = self._call_llm(SQL_ANALYST_SYSTEM, prompt)
            data = self._parse_json(response)
        except Exception as e:
            logger.error(f"SQL analyst failed: {e}")
            return Opinion(
                analyst_type="sql",
                ticker=ticker,
                question=question,
                summary=f"Analysis failed: {e}",
                sentiment=Sentiment.UNKNOWN,
                confidence=0.0,
            )

        latency = int((time.time() - start) * 1000)

        return Opinion(
            analyst_type="sql",
            ticker=ticker,
            question=question,
            summary=data.get('summary', ''),
            sentiment=Sentiment(data.get('sentiment', 'unknown')),
            confidence=data.get('confidence', 0.5),
            key_points=data.get('key_points', []),
            bullish_factors=data.get('bullish_factors', []),
            bearish_factors=data.get('bearish_factors', []),
            risk_flags=data.get('risk_flags', []),
            data_sources=data.get('data_sources', []),
            raw_response=response,
            latency_ms=latency,
        )

    # ============================================================
    # RAG ANALYST - Fixed for actual class name
    # ============================================================

    def _get_rag_context(self, ticker: str, question: str) -> Tuple[str, List[Dict]]:
        """Get RAG context from SEC filings."""
        try:
            # Use RAGRetriever (correct class name)
            from src.rag.retrieval import RAGRetriever
            retriever = RAGRetriever()
            result = retriever.retrieve(ticker=ticker, query=question, k=8)

            if not result.chunks:
                return "No SEC filing data available.", []

            context_parts = []
            citations = []

            for chunk in result.chunks:
                context_parts.append(f"[chunk:{chunk.chunk_id}] {chunk.text[:500]}...")
                citations.append({
                    "chunk_id": chunk.chunk_id,
                    "similarity": chunk.similarity_score,
                })

            return "\n\n".join(context_parts), citations

        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return f"RAG retrieval error: {e}", []

    def _get_facts_context(self, ticker: str) -> str:
        """Get extracted facts context."""
        facts_parts = []

        with get_connection() as conn:
            with conn.cursor() as cur:
                # Filing facts
                cur.execute("""
                    SELECT filing_type, period_label, key_risks, 
                           china_exposure, material_litigation
                    FROM rag.filing_facts
                    WHERE ticker = %s
                    ORDER BY extracted_at_utc DESC
                    LIMIT 3
                """, (ticker,))

                for row in cur.fetchall():
                    risks = ', '.join(row[2]) if row[2] else 'None identified'
                    china = row[3] or 'Not mentioned'
                    litigation = 'Yes' if row[4] else 'No'
                    facts_parts.append(f"""
{row[0].upper()} ({row[1]}):
- Key Risks: {risks}
- China Exposure: {china[:100]}...
- Material Litigation: {litigation}
""")

                # Transcript facts if any
                cur.execute("""
                    SELECT period_label, guidance_direction, demand_tone,
                           ai_mentions_count, risk_themes
                    FROM rag.transcript_facts
                    WHERE ticker = %s
                    ORDER BY extracted_at_utc DESC
                    LIMIT 1
                """, (ticker,))
                row = cur.fetchone()
                if row:
                    themes = ', '.join(row[4]) if row[4] else 'None'
                    facts_parts.append(f"""
EARNINGS CALL ({row[0]}):
- Guidance: {row[1] or 'Not mentioned'}
- Demand Tone: {row[2] or 'Unknown'}
- AI Mentions: {row[3]}
- Risk Themes: {themes}
""")

        return "\n".join(facts_parts) if facts_parts else "No extracted facts available."

    def _run_rag_analyst(self, ticker: str, question: str) -> Opinion:
        import time
        start = time.time()

        rag_context, citations = self._get_rag_context(ticker, question)
        facts_context = self._get_facts_context(ticker)

        prompt = RAG_ANALYST_PROMPT.format(
            ticker=ticker,
            rag_context=rag_context,
            facts_context=facts_context,
            question=question,
        )

        try:
            response = self._call_llm(RAG_ANALYST_SYSTEM, prompt)
            data = self._parse_json(response)
        except Exception as e:
            logger.error(f"RAG analyst failed: {e}")
            return Opinion(
                analyst_type="rag",
                ticker=ticker,
                question=question,
                summary=f"Analysis failed: {e}",
                sentiment=Sentiment.UNKNOWN,
                confidence=0.0,
            )

        latency = int((time.time() - start) * 1000)

        return Opinion(
            analyst_type="rag",
            ticker=ticker,
            question=question,
            summary=data.get('summary', ''),
            sentiment=Sentiment(data.get('sentiment', 'unknown')),
            confidence=data.get('confidence', 0.5),
            key_points=data.get('key_points', []),
            bullish_factors=data.get('bullish_factors', []),
            bearish_factors=data.get('bearish_factors', []),
            risk_flags=data.get('risk_flags', []),
            citations=data.get('citations', citations),
            raw_response=response,
            latency_ms=latency,
        )

    # ============================================================
    # EVALUATOR
    # ============================================================

    def _evaluate_opinions(
        self,
        sql_opinion: Opinion,
        rag_opinion: Opinion,
        question: str,
    ) -> Dict[str, Any]:

        prompt = EVALUATOR_PROMPT.format(
            ticker=sql_opinion.ticker,
            question=question,
            sql_summary=sql_opinion.summary,
            sql_sentiment=sql_opinion.sentiment.value,
            rag_summary=rag_opinion.summary,
            rag_sentiment=rag_opinion.sentiment.value,
        )

        try:
            response = self._call_llm("You are an investment committee evaluator.", prompt)
            return self._parse_json(response)
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            sentiment_match = sql_opinion.sentiment == rag_opinion.sentiment
            return {
                "agreement_score": 0.8 if sentiment_match else 0.4,
                "sentiment_match": sentiment_match,
                "conflicts": [] if sentiment_match else ["Sentiment mismatch"],
                "synthesis": f"Quant: {sql_opinion.summary} Qual: {rag_opinion.summary}",
                "final_sentiment": sql_opinion.sentiment.value,
                "final_confidence": (sql_opinion.confidence + rag_opinion.confidence) / 2,
                "combined_bullish": list(set(sql_opinion.bullish_factors + rag_opinion.bullish_factors)),
                "combined_bearish": list(set(sql_opinion.bearish_factors + rag_opinion.bearish_factors)),
                "combined_risks": list(set(sql_opinion.risk_flags + rag_opinion.risk_flags)),
            }

    # ============================================================
    # MAIN ANALYZE
    # ============================================================

    def analyze(
        self,
        ticker: str,
        question: str = "Provide a comprehensive analysis",
        run_parallel: bool = None,
    ) -> DualAnalysisResult:
        import time
        start = time.time()

        if run_parallel is None:
            run_parallel = self.config.run_parallel

        logger.info(f"Starting dual analysis for {ticker}")

        if run_parallel:
            with ThreadPoolExecutor(max_workers=2) as executor:
                sql_future = executor.submit(self._run_sql_analyst, ticker, question)
                rag_future = executor.submit(self._run_rag_analyst, ticker, question)

                sql_opinion = sql_future.result()
                rag_opinion = rag_future.result()
        else:
            sql_opinion = self._run_sql_analyst(ticker, question)
            rag_opinion = self._run_rag_analyst(ticker, question)

        evaluation = self._evaluate_opinions(sql_opinion, rag_opinion, question)

        total_latency = int((time.time() - start) * 1000)

        result = DualAnalysisResult(
            ticker=ticker,
            question=question,
            timestamp=datetime.now(timezone.utc),
            sql_opinion=sql_opinion,
            rag_opinion=rag_opinion,
            agreement_score=evaluation.get('agreement_score', 0.5),
            sentiment_match=evaluation.get('sentiment_match', False),
            conflicts=evaluation.get('conflicts', []),
            synthesis=evaluation.get('synthesis', ''),
            final_sentiment=Sentiment(evaluation.get('final_sentiment', 'unknown')),
            final_confidence=evaluation.get('final_confidence', 0.5),
            combined_bullish=evaluation.get('combined_bullish', []),
            combined_bearish=evaluation.get('combined_bearish', []),
            combined_risks=evaluation.get('combined_risks', []),
            total_latency_ms=total_latency,
        )

        logger.info(f"Dual analysis complete: agreement={result.agreement_score:.0%}, "
                   f"sentiment={result.final_sentiment.value}, time={total_latency}ms")

        return result

    def analyze_for_display(self, ticker: str, question: str = None) -> Dict[str, Any]:
        if question is None:
            question = f"Provide a comprehensive investment analysis of {ticker}"

        result = self.analyze(ticker, question)

        return {
            "ticker": ticker,
            "timestamp": result.timestamp.isoformat(),
            "sql_analyst": {
                "summary": result.sql_opinion.summary if result.sql_opinion else "",
                "sentiment": result.sql_opinion.sentiment.value if result.sql_opinion else "unknown",
                "confidence": result.sql_opinion.confidence if result.sql_opinion else 0,
                "key_points": result.sql_opinion.key_points if result.sql_opinion else [],
                "bullish": result.sql_opinion.bullish_factors if result.sql_opinion else [],
                "bearish": result.sql_opinion.bearish_factors if result.sql_opinion else [],
                "risks": result.sql_opinion.risk_flags if result.sql_opinion else [],
                "latency_ms": result.sql_opinion.latency_ms if result.sql_opinion else 0,
            },
            "rag_analyst": {
                "summary": result.rag_opinion.summary if result.rag_opinion else "",
                "sentiment": result.rag_opinion.sentiment.value if result.rag_opinion else "unknown",
                "confidence": result.rag_opinion.confidence if result.rag_opinion else 0,
                "key_points": result.rag_opinion.key_points if result.rag_opinion else [],
                "bullish": result.rag_opinion.bullish_factors if result.rag_opinion else [],
                "bearish": result.rag_opinion.bearish_factors if result.rag_opinion else [],
                "risks": result.rag_opinion.risk_flags if result.rag_opinion else [],
                "citations": result.rag_opinion.citations if result.rag_opinion else [],
                "latency_ms": result.rag_opinion.latency_ms if result.rag_opinion else 0,
            },
            "evaluation": {
                "agreement_score": result.agreement_score,
                "sentiment_match": result.sentiment_match,
                "conflicts": result.conflicts,
            },
            "synthesis": {
                "summary": result.synthesis,
                "sentiment": result.final_sentiment.value,
                "confidence": result.final_confidence,
                "bullish_factors": result.combined_bullish,
                "bearish_factors": result.combined_bearish,
                "risk_flags": result.combined_risks,
            },
            "total_latency_ms": result.total_latency_ms,
        }


def dual_analyze(ticker: str, question: str = None) -> Dict[str, Any]:
    """Quick dual analysis for a ticker."""
    service = DualAnalystService()
    return service.analyze_for_display(ticker, question)


if __name__ == "__main__":
    print("Testing Dual Analyst Service...")

    service = DualAnalystService()
    result = service.analyze_for_display("MU", "What's the investment outlook?")

    print(f"\n{'='*60}")
    print(f"DUAL ANALYSIS: {result['ticker']}")
    print(f"{'='*60}")

    print(f"\n📊 QUANTITATIVE ANALYST:")
    print(f"   Sentiment: {result['sql_analyst']['sentiment']}")
    print(f"   Confidence: {result['sql_analyst']['confidence']:.0%}")
    print(f"   Summary: {result['sql_analyst']['summary'][:200]}...")

    print(f"\n📄 QUALITATIVE ANALYST:")
    print(f"   Sentiment: {result['rag_analyst']['sentiment']}")
    print(f"   Confidence: {result['rag_analyst']['confidence']:.0%}")
    print(f"   Summary: {result['rag_analyst']['summary'][:200]}...")

    print(f"\n🎯 SYNTHESIS:")
    print(f"   Agreement: {result['evaluation']['agreement_score']:.0%}")
    print(f"   Final Sentiment: {result['synthesis']['sentiment']}")
    print(f"   Confidence: {result['synthesis']['confidence']:.0%}")
    print(f"   Summary: {result['synthesis']['summary'][:300]}...")

    print(f"\n⏱️  Total Time: {result['total_latency_ms']}ms")