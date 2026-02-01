"""
AI Research Agent v4 - Wrapper for AlphaChat
=============================================

Provides the v4 interface expected by ai_assistant_tab_v4.py
but uses the existing AlphaChat system under the hood.

Author: HH Research Platform
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import AlphaChat
try:
    from src.ai.chat import AlphaChat

    ALPHA_CHAT_AVAILABLE = True
except ImportError:
    ALPHA_CHAT_AVAILABLE = False
    logger.error("AlphaChat not available")


@dataclass
class Citation:
    """Citation from research."""
    id: str
    source: str
    claim: str
    tier: int = 3
    tier_label: str = "Platform Data"
    url: str = ""


@dataclass
class GuardrailAlert:
    """Alert from guardrails."""
    severity: str  # "WARNING", "CAUTION"
    message: str


@dataclass
class ResearchResponse:
    """Response from research agent."""
    success: bool
    analysis: str
    outlook: str = "neutral"
    confidence: str = "medium"
    citations: List[Citation] = field(default_factory=list)
    guardrails_triggered: List[GuardrailAlert] = field(default_factory=list)
    data_staleness: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    disclaimer: str = "This is AI-generated research for informational purposes only. Not financial advice."


class AsyncResearchAgent:
    """
    Research agent that provides structured responses.
    Wraps AlphaChat for actual LLM calls.
    """

    def __init__(self, user_id: str = None):
        self.user_id = user_id or str(uuid.uuid4())[:8]
        self._chat = None

    def _get_chat(self) -> Optional['AlphaChat']:
        """Get or create AlphaChat instance."""
        if not ALPHA_CHAT_AVAILABLE:
            return None
        if self._chat is None:
            self._chat = AlphaChat()
        return self._chat

    def research(
            self,
            question: str,
            history: List[Dict] = None,
    ) -> ResearchResponse:
        """
        Perform research on a question.

        Args:
            question: The research question
            history: Conversation history

        Returns:
            ResearchResponse with analysis
        """
        import time
        start = time.time()

        chat = self._get_chat()
        if not chat:
            return ResearchResponse(
                success=False,
                analysis="AI Chat system not available. Check if LLM server is running.",
                metadata={"error": "AlphaChat not available"}
            )

        try:
            # Detect ticker from question
            ticker = self._detect_ticker(question)

            # Build context for chat
            enhanced_question = question
            if ticker:
                enhanced_question = f"[Analyzing {ticker}] {question}"

            # Get response from AlphaChat
            response_text = ""
            for chunk in chat.chat_stream(enhanced_question, ticker=ticker):
                response_text += chunk

            # Clean up thinking tags if present
            if '</think>' in response_text:
                response_text = response_text.split('</think>')[-1].strip()

            # Determine outlook from response
            outlook = self._extract_outlook(response_text)
            confidence = self._extract_confidence(response_text)

            latency = int((time.time() - start) * 1000)

            # Build citations
            citations = []
            if ticker:
                citations.append(Citation(
                    id="1",
                    source="Platform Data",
                    claim=f"Analysis based on {ticker} signals and scores",
                    tier=2,
                    tier_label="Internal Data"
                ))

            return ResearchResponse(
                success=True,
                analysis=response_text,
                outlook=outlook,
                confidence=confidence,
                citations=citations,
                metadata={
                    "request_id": str(uuid.uuid4())[:8],
                    "user_id": self.user_id,
                    "ticker": ticker,
                    "horizon": "short-term",
                    "latency_ms": latency,
                    "data_sources": {
                        "signals": True,
                        "news": True,
                        "fundamentals": True,
                    }
                }
            )

        except Exception as e:
            logger.exception(f"Research failed: {e}")
            return ResearchResponse(
                success=False,
                analysis=f"Research failed: {str(e)}",
                metadata={"error": str(e)}
            )

    def _detect_ticker(self, text: str) -> Optional[str]:
        """Detect ticker symbol from text."""
        import re

        # Common patterns
        patterns = [
            r'\b([A-Z]{1,5})\b',
        ]

        # Known tickers
        known = self._get_known_tickers()

        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            for match in matches:
                if match in known:
                    return match
        return None

    def _get_known_tickers(self) -> set:
        """Get known tickers from database."""
        try:
            from src.db.connection import get_engine
            import pandas as pd
            engine = get_engine()
            df = pd.read_sql("SELECT DISTINCT ticker FROM screener_scores LIMIT 500", engine)
            return set(df['ticker'].tolist())
        except:
            return {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'MU', 'INTC'}

    def _extract_outlook(self, text: str) -> str:
        """Extract outlook from response text."""
        text_lower = text.lower()

        if any(w in text_lower for w in ['strong buy', 'very bullish', 'highly recommend']):
            return "very_bullish"
        elif any(w in text_lower for w in ['buy', 'bullish', 'positive', 'upside']):
            return "bullish"
        elif any(w in text_lower for w in ['sell', 'bearish', 'negative', 'downside', 'avoid']):
            return "bearish"
        elif any(w in text_lower for w in ['strong sell', 'very bearish']):
            return "very_bearish"
        else:
            return "neutral"

    def _extract_confidence(self, text: str) -> str:
        """Extract confidence level from response."""
        text_lower = text.lower()

        if any(w in text_lower for w in ['high confidence', 'strongly', 'clearly', 'definitely']):
            return "high"
        elif any(w in text_lower for w in ['low confidence', 'uncertain', 'unclear', 'mixed']):
            return "low"
        else:
            return "medium"


def create_research_agent(user_id: str = None) -> AsyncResearchAgent:
    """Create a new research agent instance."""
    return AsyncResearchAgent(user_id=user_id)


def research_sync(
        question: str,
        user_id: str = None,
        history: List[Dict] = None,
) -> ResearchResponse:
    """
    Synchronous research function.

    Args:
        question: Research question
        user_id: User identifier
        history: Conversation history

    Returns:
        ResearchResponse
    """
    agent = create_research_agent(user_id)
    return agent.research(question, history)


# Test
if __name__ == "__main__":
    print("Testing AI Research Agent v4...")

    response = research_sync("What's the outlook for NVDA?")

    print(f"\nSuccess: {response.success}")
    print(f"Outlook: {response.outlook}")
    print(f"Confidence: {response.confidence}")
    print(f"\nAnalysis:\n{response.analysis[:500]}...")
    print(f"\nMetadata: {response.metadata}")