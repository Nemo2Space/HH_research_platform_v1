"""
Alpha Platform - Committee Agents

Multi-agent system for investment analysis.
Based on Project 2 agents.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentVote:
    """Agent vote for a stock."""
    role: str
    buy_prob: float = 0.5  # 0-1, probability of buy recommendation
    expected_alpha_bps: float = 0.0  # Expected alpha in basis points
    confidence: float = 0.5  # 0-1
    horizon_days: int = 63  # Trading days (~3 months)
    rationale: str = ""
    risks: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'role': self.role,
            'buy_prob': self.buy_prob,
            'expected_alpha_bps': self.expected_alpha_bps,
            'confidence': self.confidence,
            'horizon_days': self.horizon_days,
            'rationale': self.rationale,
            'risks': self.risks,
            'evidence_refs': self.evidence_refs
        }


@dataclass
class CommitteeDecision:
    """Final committee decision."""
    ticker: str
    asof: str
    verdict: str  # BUY, SELL, HOLD
    expected_alpha_bps: float
    confidence: float
    horizon_days: int
    conviction: int  # 0-100
    rationale: str
    risks: List[str]
    votes: List[AgentVote]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ticker': self.ticker,
            'asof': self.asof,
            'verdict': self.verdict,
            'expected_alpha_bps': self.expected_alpha_bps,
            'confidence': self.confidence,
            'horizon_days': self.horizon_days,
            'conviction': self.conviction,
            'rationale': self.rationale,
            'risks': self.risks,
            'votes': [v.to_dict() for v in self.votes]
        }


class BaseAgent(ABC):
    """Base class for all committee agents."""

    role: str = "base"

    def __init__(self, llm_client=None, config: Optional[Dict] = None):
        self.llm = llm_client
        self.config = config or {}
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def analyze(self, ticker: str, data: Dict[str, Any]) -> AgentVote:
        """Analyze a stock and return a vote."""
        pass

    def _call_llm(self, prompt: str, system: str = None) -> Optional[str]:
        """Call LLM with prompt."""
        if not self.llm:
            return None

        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = self.llm.chat.completions.create(
                model=self.config.get('model', 'Qwen3-32B-Q6_K.gguf'),
                messages=messages,
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return None

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        if not text:
            return {}

        # Find JSON block
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        return {}

    def _make_fallback_vote(self, rationale: str = "Analysis unavailable") -> AgentVote:
        """Create neutral fallback vote."""
        return AgentVote(
            role=self.role,
            buy_prob=0.5,
            expected_alpha_bps=0,
            confidence=0.2,
            horizon_days=63,
            rationale=rationale,
            risks=["Analysis incomplete"]
        )


class FundamentalAgent(BaseAgent):
    """Analyzes company fundamentals."""

    role = "fundamental"

    def analyze(self, ticker: str, data: Dict[str, Any]) -> AgentVote:
        """Analyze fundamentals and return vote."""
        fundamentals = data.get('fundamentals', {})

        if not fundamentals:
            return self._make_fallback_vote("No fundamental data available")

        # Calculate scores from data
        pe = fundamentals.get('pe_ratio')
        roe = fundamentals.get('roe')
        revenue_growth = fundamentals.get('revenue_growth')
        profit_margin = fundamentals.get('profit_margin')

        # Scoring logic
        score = 50
        reasons = []
        risks = []

        if pe:
            if pe < 15:
                score += 15
                reasons.append(f"Attractive P/E of {pe:.1f}")
            elif pe > 35:
                score -= 10
                risks.append(f"High P/E of {pe:.1f}")

        if roe and roe > 0.15:
            score += 10
            reasons.append(f"Strong ROE of {roe * 100:.1f}%")

        if revenue_growth:
            if revenue_growth > 0.15:
                score += 15
                reasons.append(f"Revenue growth of {revenue_growth * 100:.1f}%")
            elif revenue_growth < 0:
                score -= 10
                risks.append("Declining revenue")

        if profit_margin and profit_margin > 0.2:
            score += 10
            reasons.append(f"High profit margin of {profit_margin * 100:.1f}%")

        # Convert score to probability and alpha
        buy_prob = min(1.0, max(0.0, score / 100))
        alpha_bps = (score - 50) * 4  # Scale to basis points

        rationale = ". ".join(reasons) if reasons else "Mixed fundamentals"

        return AgentVote(
            role=self.role,
            buy_prob=buy_prob,
            expected_alpha_bps=alpha_bps,
            confidence=0.7 if len(reasons) >= 2 else 0.5,
            horizon_days=126,  # 6 months
            rationale=rationale,
            risks=risks
        )


class SentimentAgent(BaseAgent):
    """Analyzes news sentiment."""

    role = "sentiment"

    def analyze(self, ticker: str, data: Dict[str, Any]) -> AgentVote:
        """Analyze sentiment and return vote."""
        sentiment_score = data.get('sentiment_score')
        article_count = data.get('article_count', 0)

        # Convert sentiment to vote
        buy_prob = sentiment_score / 100
        alpha_bps = (sentiment_score - 50) * 3

        # Confidence based on article count
        if article_count >= 10:
            confidence = 0.7
        elif article_count >= 5:
            confidence = 0.5
        else:
            confidence = 0.3

        if sentiment_score >= 70:
            rationale = "Strong positive sentiment in recent news"
        elif sentiment_score >= 55:
            rationale = "Mildly positive news sentiment"
        elif sentiment_score <= 30:
            rationale = "Strong negative sentiment detected"
        elif sentiment_score <= 45:
            rationale = "Mildly negative news sentiment"
        else:
            rationale = "Neutral news sentiment"

        risks = []
        if article_count < 5:
            risks.append("Limited news coverage")
        if sentiment_score < 40:
            risks.append("Negative market perception")

        return AgentVote(
            role=self.role,
            buy_prob=buy_prob,
            expected_alpha_bps=alpha_bps,
            confidence=confidence,
            horizon_days=21,  # 1 month
            rationale=rationale,
            risks=risks
        )


class TechnicalAgent(BaseAgent):
    """Analyzes technical indicators."""

    role = "technical"

    def analyze(self, ticker: str, data: Dict[str, Any]) -> AgentVote:
        """Analyze technicals and return vote."""
        tech_score = data.get('technical_score')
        rsi = data.get('rsi', 50)
        trend = data.get('trend', 'neutral')
        momentum = data.get('momentum_5d', 0)

        # Build analysis
        reasons = []
        risks = []

        if trend == 'uptrend':
            reasons.append("Price in uptrend (above SMA20 and SMA50)")
        elif trend == 'downtrend':
            risks.append("Price in downtrend")

        if rsi > 70:
            risks.append(f"Overbought (RSI={rsi:.1f})")
        elif rsi < 30:
            reasons.append(f"Oversold (RSI={rsi:.1f})")

        if momentum > 5:
            reasons.append(f"Strong momentum (+{momentum:.1f}%)")
        elif momentum < -5:
            risks.append(f"Negative momentum ({momentum:.1f}%)")

        buy_prob = tech_score / 100
        alpha_bps = (tech_score - 50) * 3

        confidence = 0.6 if len(reasons) >= 2 else 0.4
        rationale = ". ".join(reasons) if reasons else "Mixed technical signals"

        return AgentVote(
            role=self.role,
            buy_prob=buy_prob,
            expected_alpha_bps=alpha_bps,
            confidence=confidence,
            horizon_days=21,
            rationale=rationale,
            risks=risks
        )


class ValuationAgent(BaseAgent):
    """Analyzes valuation and price targets."""

    role = "valuation"

    def analyze(self, ticker: str, data: Dict[str, Any]) -> AgentVote:
        """Analyze valuation and return vote."""
        target_upside = data.get('target_upside_pct', 0) or 0
        analyst_positivity = data.get('analyst_positivity', 50) or 50
        gap_score = data.get('gap_score', 50)

        reasons = []
        risks = []

        if target_upside > 20:
            reasons.append(f"Significant upside to target ({target_upside:.1f}%)")
        elif target_upside > 10:
            reasons.append(f"Upside to analyst targets ({target_upside:.1f}%)")
        elif target_upside < -10:
            risks.append(f"Trading above analyst targets ({target_upside:.1f}%)")

        if analyst_positivity > 70:
            reasons.append(f"Strong analyst consensus ({analyst_positivity:.0f}% positive)")
        elif analyst_positivity < 40:
            risks.append("Weak analyst consensus")

        # Calculate vote
        score = 50 + (target_upside / 2) + ((analyst_positivity - 50) / 5)
        buy_prob = min(1.0, max(0.0, score / 100))
        alpha_bps = target_upside * 10  # Direct mapping

        confidence = 0.6 if abs(target_upside) > 10 else 0.4
        rationale = ". ".join(reasons) if reasons else "Fair valuation"

        return AgentVote(
            role=self.role,
            buy_prob=buy_prob,
            expected_alpha_bps=alpha_bps,
            confidence=confidence,
            horizon_days=63,
            rationale=rationale,
            risks=risks
        )