"""
Portfolio Construction Engine V5 - Full AI Integration
=======================================================

Integrates ALL available AI data sources:
- ai_analysis: AI recommendations with bull/bear case
- ai_recommendations: AI probability scores
- committee_decisions: Committee verdict and conviction
- agent_votes: Individual agent buy probabilities
- alpha_predictions: ML model predictions
- enhanced_scores: Additional scoring factors
- trading_signals: Signal type and strength
- earnings_calendar: Upcoming earnings
- fda_calendar: FDA catalysts (biotech)

Author: HH Research Platform
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
import logging
import re
from datetime import datetime, date, timedelta
from decimal import Decimal

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class InvestmentStrategy(Enum):
    BIOTECH_GROWTH = "biotech_growth"
    TECH_GROWTH = "tech_growth"
    AGGRESSIVE_GROWTH = "aggressive_growth"
    VALUE = "value"
    DEEP_VALUE = "deep_value"
    INCOME = "income"
    DIVIDEND_GROWTH = "dividend_growth"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    ALL_WEATHER = "all_weather"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    SHARIAH = "shariah"


class RiskLevel(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class PortfolioObjective(Enum):
    BALANCED = "balanced"
    GROWTH = "growth"
    BIOTECH_GROWTH = "biotech_growth"
    TECH_GROWTH = "tech_growth"
    VALUE = "value"
    DEEP_VALUE = "deep_value"
    INCOME = "income"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    CONSERVATIVE = "conservative"
    SHARIAH = "shariah"
    CUSTOM = "custom"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ScoreDetail:
    """Detailed score with evidence."""
    name: str
    value: float
    weight: float
    weighted_contribution: float
    data_available: bool
    raw_data: Dict[str, Any]
    data_source: str
    missing_reason: Optional[str] = None


@dataclass
class AgentVote:
    """Individual AI agent vote."""
    agent_role: str  # fundamental, sentiment, technical, valuation
    buy_prob: float
    confidence: float
    rationale: str


@dataclass
class AIDecision:
    """Complete AI decision with all sources."""
    # Primary AI recommendation
    ai_action: Optional[str] = None
    ai_confidence: Optional[str] = None
    ai_probability: Optional[float] = None

    # Committee decision
    committee_verdict: Optional[str] = None
    committee_conviction: Optional[float] = None
    committee_rationale: Optional[str] = None

    # Individual agent votes
    agent_votes: List[AgentVote] = field(default_factory=list)
    avg_agent_buy_prob: Optional[float] = None

    # Alpha predictions
    alpha_probability: Optional[float] = None
    alpha_signal: Optional[str] = None

    # Enhanced scores
    insider_score: Optional[int] = None
    revision_score: Optional[int] = None
    earnings_surprise_score: Optional[int] = None

    # Trading signal
    signal_type: Optional[str] = None
    signal_strength: Optional[int] = None
    signal_reason: Optional[str] = None

    # AI reasoning
    bull_case: Optional[str] = None
    bear_case: Optional[str] = None
    key_risks: Optional[str] = None
    one_line_summary: Optional[str] = None

    # Coverage
    data_available: bool = False
    data_sources: List[str] = field(default_factory=list)


@dataclass
class CatalystInfo:
    """Catalyst information."""
    # Earnings
    earnings_date: Optional[date] = None
    days_to_earnings: Optional[int] = None
    eps_estimate: Optional[float] = None
    guidance_direction: Optional[str] = None

    # Catalyst (from fda_calendar - but may be clinical/regulatory/commercial)
    fda_date: Optional[date] = None
    days_to_fda: Optional[int] = None
    fda_drug: Optional[str] = None
    fda_catalyst_type: Optional[str] = None
    fda_priority: Optional[str] = None

    # Catalyst classification
    catalyst_class: Optional[str] = None  # REGULATORY / CLINICAL / COMMERCIAL / UNKNOWN
    catalyst_label: Optional[str] = None  # Human-readable label for display

    # Computed
    catalyst_score: float = 0  # Changed from 50 to 0 - no default assumption
    has_near_term_catalyst: bool = False
    catalyst_description: Optional[str] = None



@dataclass
class PortfolioHolding:
    """Enhanced holding with full AI data."""
    ticker: str
    company_name: str
    weight_pct: float
    shares: int
    value: float
    sector: str

    # Scores
    composite_score: float
    score_details: List[ScoreDetail]
    scores_used: List[str]
    scores_missing: List[str]

    # AI
    ai_decision: AIDecision

    # Catalysts
    catalyst_info: CatalystInfo

    # Conviction
    conviction: str
    conviction_rationale: str

    # Evidence
    bull_case: List[str]
    bear_case: List[str]
    key_catalysts: List[str]
    key_risks: List[str]

    # Raw metrics
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    revenue_growth: Optional[float] = None
    gross_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    options_sentiment: Optional[str] = None
    squeeze_risk: Optional[str] = None


@dataclass
class PortfolioIntent:
    """User intent."""
    objective: str = "balanced"
    risk_level: str = "moderate"
    portfolio_value: float = 100000.0

    sectors_include: List[str] = field(default_factory=list)
    sectors_exclude: List[str] = field(default_factory=list)
    tickers_include: List[str] = field(default_factory=list)
    tickers_exclude: List[str] = field(default_factory=list)

    min_holdings: Optional[int] = None
    max_holdings: Optional[int] = None
    max_position_pct: Optional[float] = None
    max_sector_pct: Optional[float] = None
    min_dividend_yield: Optional[float] = None

    # Cash buffer control
    cash_buffer_pct: Optional[float] = None  # Override strategy default
    fully_invested: bool = False  # If True, force cash_buffer_pct = 0

    # Advanced constraints for biotech/specialized portfolios
    min_subsectors: Optional[int] = None  # Minimum number of distinct subsectors
    max_binary_event_weight_pct: Optional[float] = None  # Max % in binary-event stocks
    min_established_weight_pct: Optional[float] = None  # Min % in established/commercial stocks
    max_established_weight_pct: Optional[float] = None  # Max % in established/commercial stocks

    equal_weight: bool = False
    shariah_compliant: bool = False
    restrict_to_tickers: bool = False

    require_ai_buy: bool = False
    min_ai_probability: Optional[float] = None
    require_catalyst: bool = False

    # Theme-based filtering (AI, Cybersecurity, Semiconductors, etc.)
    theme: Optional[str] = None  # "ai", "cybersecurity", "semiconductors", "fintech", etc.
    theme_mode: Optional[str] = None  # "builders" (companies building AI) vs "adopters" (companies using AI)
    require_theme_match: bool = False  # If True, only include stocks matching theme
    min_ai_exposure: Optional[float] = None  # Minimum ai_exposure_score (0-100) for AI theme

    @classmethod
    def from_json(cls, json_str: str) -> 'PortfolioIntent':
        data = json.loads(json_str)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PortfolioResult:
    """Complete portfolio result."""
    success: bool
    holdings: List[PortfolioHolding]

    total_value: float
    invested_value: float
    cash_value: float

    num_holdings: int
    num_sectors: int
    sectors: Dict[str, float]

    avg_score: float
    avg_ai_probability: Optional[float]
    avg_committee_conviction: Optional[float]

    strategy_name: str
    strategy_description: str

    # Transparency
    total_scores_available: int
    total_scores_missing: int
    ai_coverage_pct: float
    catalyst_coverage_pct: float

    portfolio_thesis: str
    key_risks: List[str]
    rebalance_triggers: List[str]

    warnings: List[str]
    errors: List[str]

    intent: PortfolioIntent
    constraints_used: Dict[str, Any]

    def to_markdown(self) -> str:
        """Generate comprehensive markdown report."""
        lines = []

        lines.append(f"# {self.strategy_name} Portfolio\n")
        lines.append(f"*{self.strategy_description}*\n")

        lines.append("## Investment Thesis\n")
        lines.append(f"{self.portfolio_thesis}\n")

        # Summary table
        lines.append("## Portfolio Summary\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Value | ${self.total_value:,.0f} |")
        lines.append(f"| Invested | ${self.invested_value:,.0f} ({self.invested_value/self.total_value*100:.1f}%) |")
        lines.append(f"| Cash Buffer | ${self.cash_value:,.0f} ({self.cash_value/self.total_value*100:.1f}%) |")
        lines.append(f"| Holdings | {self.num_holdings} |")
        lines.append(f"| Sectors | {self.num_sectors} |")
        lines.append(f"| Avg Score | {self.avg_score:.1f}/100 |")
        if self.avg_ai_probability:
            lines.append(f"| Avg AI Probability | {self.avg_ai_probability:.1f}% |")
        if self.avg_committee_conviction:
            lines.append(f"| Avg Committee Conviction | {self.avg_committee_conviction:.1f} |")
        lines.append(f"| AI Coverage | {self.ai_coverage_pct:.0f}% |")
        lines.append(f"| Catalyst Coverage | {self.catalyst_coverage_pct:.0f}% |")
        lines.append("")

        # Holdings table
        lines.append("## Holdings\n")
        lines.append("| # | Ticker | Weight | Score | AI Prob | Committee | Signal | Catalyst | Conviction |")
        lines.append("|---|--------|--------|-------|---------|-----------|--------|----------|------------|")

        for i, h in enumerate(sorted(self.holdings, key=lambda x: -x.weight_pct), 1):
            ai_prob = f"{h.ai_decision.ai_probability:.0f}%" if h.ai_decision.ai_probability else "N/A"
            committee = h.ai_decision.committee_verdict or "N/A"
            signal = h.ai_decision.signal_type or "N/A"
            catalyst = f"{h.catalyst_info.days_to_earnings}d" if h.catalyst_info.days_to_earnings else (
                f"FDA:{h.catalyst_info.days_to_fda}d" if h.catalyst_info.days_to_fda else "None"
            )
            lines.append(f"| {i} | **{h.ticker}** | {h.weight_pct:.1f}% | {h.composite_score:.0f} | {ai_prob} | {committee} | {signal} | {catalyst} | {h.conviction} |")
        lines.append("")

        # Detailed analysis for top 5
        lines.append("## Top Holdings Analysis\n")
        for h in sorted(self.holdings, key=lambda x: -x.weight_pct)[:5]:
            lines.append(f"### {h.ticker} - {h.company_name}\n")
            lines.append(f"**Weight:** {h.weight_pct:.1f}% | **Conviction:** {h.conviction}\n")

            # Key metrics
            lines.append("#### Metrics")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Score | {h.composite_score:.0f}/100 |")
            if h.market_cap:
                mc = f"${h.market_cap/1e9:.1f}B" if h.market_cap >= 1e9 else f"${h.market_cap/1e6:.0f}M"
                lines.append(f"| Market Cap | {mc} |")
            if h.options_sentiment:
                lines.append(f"| Options Flow | {h.options_sentiment} |")
            if h.squeeze_risk:
                lines.append(f"| Squeeze Risk | {h.squeeze_risk} |")
            lines.append("")

            # AI Analysis
            lines.append("#### AI Analysis")
            ai = h.ai_decision
            if ai.data_available:
                lines.append("| Source | Value |")
                lines.append("|--------|-------|")
                if ai.ai_action:
                    lines.append(f"| AI Action | **{ai.ai_action}** |")
                if ai.ai_confidence:
                    lines.append(f"| AI Confidence | {ai.ai_confidence} |")
                if ai.ai_probability:
                    lines.append(f"| AI Probability | {ai.ai_probability:.1f}% |")
                if ai.committee_verdict:
                    lines.append(f"| Committee Verdict | **{ai.committee_verdict}** |")
                if ai.committee_conviction:
                    lines.append(f"| Committee Conviction | {ai.committee_conviction:.0f} |")
                if ai.alpha_probability:
                    lines.append(f"| Alpha ML Probability | {ai.alpha_probability:.1f}% |")
                if ai.signal_type:
                    lines.append(f"| Trading Signal | {ai.signal_type} (strength: {ai.signal_strength}) |")
                lines.append("")

                # Agent votes
                if ai.agent_votes:
                    lines.append("**Agent Votes:**")
                    for vote in ai.agent_votes:
                        lines.append(f"- {vote.agent_role.title()}: {vote.buy_prob*100:.0f}% buy prob - {vote.rationale[:60]}...")
                    lines.append("")

                if ai.one_line_summary:
                    lines.append(f"**AI Summary:** {ai.one_line_summary}\n")

                if ai.bull_case:
                    lines.append(f"**AI Bull Case:** {ai.bull_case[:300]}...\n")
                if ai.bear_case:
                    lines.append(f"**AI Bear Case:** {ai.bear_case[:300]}...\n")
            else:
                lines.append("*No AI analysis available*\n")

            # Score breakdown
            lines.append("#### Score Breakdown")
            lines.append("| Component | Value | Weight | Contribution | Source |")
            lines.append("|-----------|-------|--------|--------------|--------|")
            for sd in sorted(h.score_details, key=lambda x: -x.weighted_contribution)[:10]:
                status = "✓" if sd.data_available else "✗"
                lines.append(f"| {status} {sd.name} | {sd.value:.1f} | {sd.weight:.0%} | {sd.weighted_contribution:.1f} | {sd.data_source[:25]} |")
            lines.append("")

            if h.scores_missing:
                lines.append(f"**Missing:** {', '.join(h.scores_missing)}\n")

            # Catalysts
            cat = h.catalyst_info
            if cat.has_near_term_catalyst:
                lines.append("#### Catalysts")
                if cat.days_to_earnings:
                    lines.append(f"- **Earnings:** {cat.days_to_earnings} days (EPS est: ${cat.eps_estimate:.2f})" if cat.eps_estimate else f"- **Earnings:** {cat.days_to_earnings} days")
                if cat.days_to_fda:
                    lines.append(f"- **FDA ({cat.fda_catalyst_type}):** {cat.days_to_fda} days - {cat.fda_drug} [{cat.fda_priority}]")
                if cat.catalyst_description:
                    lines.append(f"- {cat.catalyst_description}")
                lines.append("")

            # Bull/Bear
            if h.bull_case:
                lines.append("#### Bull Case")
                for p in h.bull_case[:4]:
                    lines.append(f"- ✓ {p}")
                lines.append("")

            if h.bear_case:
                lines.append("#### Bear Case")
                for p in h.bear_case[:4]:
                    lines.append(f"- ✗ {p}")
                lines.append("")

            lines.append("---\n")

        # Sectors
        lines.append("## Sector Allocation\n")
        lines.append("| Sector | Weight |")
        lines.append("|--------|--------|")
        for sector, weight in sorted(self.sectors.items(), key=lambda x: -x[1]):
            lines.append(f"| {sector} | {weight:.1f}% |")
        lines.append("")

        # Risks
        lines.append("## Key Risks\n")
        for r in self.key_risks:
            lines.append(f"- ⚠️ {r}")
        lines.append("")

        # Triggers
        lines.append("## Rebalance Triggers\n")
        for t in self.rebalance_triggers:
            lines.append(f"- {t}")

        return "\n".join(lines)


# =============================================================================
# STRATEGY MODELS
# =============================================================================

STRATEGY_SCORING_MODELS = {
    InvestmentStrategy.BIOTECH_GROWTH: {
        "name": "Biotech Growth",
        "description": "AI-driven catalyst selection with FDA/clinical focus",
        "weights": {
            "ai_committee_score": 20,    # Committee + agent votes
            "ai_probability": 15,        # AI recommendation probability
            "catalyst_score": 20,        # FDA + earnings catalysts
            "options_flow": 15,          # Smart money
            "fundamental_score": 10,     # Platform fundamentals
            "sentiment_score": 10,       # News sentiment
            "squeeze_potential": 5,      # Short squeeze
            "enhanced_score": 5,         # Insider + revisions
        },
        "constraints": {
            "min_market_cap": 200e6,
            "max_position_pct": 8,
            "cash_buffer_pct": 10,
            "min_holdings": 12,
            "max_holdings": 25,
        },
        "position_sizing": "conviction",
    },

    InvestmentStrategy.TECH_GROWTH: {
        "name": "Technology Growth",
        "description": "High-growth tech with AI validation",
        "weights": {
            "ai_committee_score": 20,
            "ai_probability": 15,
            "growth_score": 20,
            "fundamental_score": 15,
            "options_flow": 10,
            "sentiment_score": 10,
            "catalyst_score": 10,
        },
        "constraints": {
            "min_market_cap": 500e6,
            "max_position_pct": 8,
            "cash_buffer_pct": 5,
            "min_holdings": 12,
            "max_holdings": 25,
        },
        "position_sizing": "conviction",
    },

    InvestmentStrategy.VALUE: {
        "name": "Value Investing",
        "description": "AI-identified undervalued companies",
        "weights": {
            "valuation_score": 25,
            "fundamental_score": 25,
            "ai_committee_score": 15,
            "ai_probability": 10,
            "dividend_score": 10,
            "enhanced_score": 10,
            "sentiment_score": 5,
        },
        "constraints": {
            "min_market_cap": 1e9,
            "max_position_pct": 6,
            "cash_buffer_pct": 5,
            "min_holdings": 15,
            "max_holdings": 30,
        },
        "position_sizing": "fundamental",
    },

    InvestmentStrategy.BALANCED: {
        "name": "Balanced Portfolio",
        "description": "AI-optimized multi-factor approach",
        "weights": {
            "ai_committee_score": 15,
            "ai_probability": 15,
            "fundamental_score": 20,
            "growth_score": 15,
            "dividend_score": 10,
            "catalyst_score": 10,
            "sentiment_score": 10,
            "enhanced_score": 5,
        },
        "constraints": {
            "min_market_cap": 2e9,
            "max_position_pct": 5,
            "max_sector_pct": 20,
            "cash_buffer_pct": 5,
            "min_holdings": 20,
            "max_holdings": 30,
        },
        "position_sizing": "risk_parity",
    },

    InvestmentStrategy.MOMENTUM: {
        "name": "Momentum Strategy",
        "description": "AI-validated trend following",
        "weights": {
            "trading_signal_score": 25,
            "ai_committee_score": 15,
            "options_flow": 20,
            "sentiment_score": 15,
            "growth_score": 15,
            "catalyst_score": 10,
        },
        "constraints": {
            "min_market_cap": 1e9,
            "max_position_pct": 6,
            "cash_buffer_pct": 5,
            "min_holdings": 15,
            "max_holdings": 25,
        },
        "position_sizing": "momentum",
    },

    InvestmentStrategy.CONSERVATIVE: {
        "name": "Conservative Portfolio",
        "description": "Capital preservation with AI risk screening",
        "weights": {
            "fundamental_score": 30,
            "dividend_score": 25,
            "ai_committee_score": 15,
            "valuation_score": 15,
            "enhanced_score": 10,
            "sentiment_score": 5,
        },
        "constraints": {
            "min_market_cap": 10e9,
            "max_position_pct": 4,
            "max_sector_pct": 15,
            "cash_buffer_pct": 10,
            "min_holdings": 25,
            "max_holdings": 40,
        },
        "position_sizing": "equal",
    },

    InvestmentStrategy.INCOME: {
        "name": "Income Portfolio",
        "description": "Dividend-focused with AI quality validation",
        "weights": {
            "dividend_score": 30,
            "fundamental_score": 25,
            "ai_committee_score": 15,
            "valuation_score": 15,
            "enhanced_score": 10,
            "sentiment_score": 5,
        },
        "constraints": {
            "min_market_cap": 5e9,
            "min_dividend_yield": 2.0,
            "max_position_pct": 5,
            "max_sector_pct": 20,
            "cash_buffer_pct": 5,
            "min_holdings": 20,
            "max_holdings": 35,
        },
        "position_sizing": "yield",
    },

    InvestmentStrategy.QUALITY: {
        "name": "Quality Factor",
        "description": "High-quality companies with AI validation",
        "weights": {
            "fundamental_score": 30,
            "ai_committee_score": 20,
            "growth_score": 20,
            "enhanced_score": 15,
            "dividend_score": 10,
            "sentiment_score": 5,
        },
        "constraints": {
            "min_market_cap": 2e9,
            "max_position_pct": 5,
            "cash_buffer_pct": 5,
            "min_holdings": 20,
            "max_holdings": 30,
        },
        "position_sizing": "quality",
    },

    InvestmentStrategy.AGGRESSIVE_GROWTH: {
        "name": "Aggressive Growth",
        "description": "Maximum growth with AI momentum signals",
        "weights": {
            "ai_committee_score": 20,
            "ai_probability": 15,
            "growth_score": 20,
            "options_flow": 15,
            "catalyst_score": 15,
            "squeeze_potential": 10,
            "sentiment_score": 5,
        },
        "constraints": {
            "min_market_cap": 100e6,
            "max_position_pct": 10,
            "cash_buffer_pct": 0,
            "min_holdings": 10,
            "max_holdings": 20,
        },
        "position_sizing": "conviction",
    },

    InvestmentStrategy.SHARIAH: {
        "name": "Shariah Compliant",
        "description": "Islamic finance compliant with AI screening",
        "weights": {
            "fundamental_score": 30,
            "ai_committee_score": 20,
            "growth_score": 20,
            "valuation_score": 15,
            "sentiment_score": 10,
            "dividend_score": 5,
        },
        "constraints": {
            "min_market_cap": 1e9,
            "max_debt_ratio": 0.33,
            "max_position_pct": 5,
            "cash_buffer_pct": 5,
            "min_holdings": 20,
            "max_holdings": 30,
        },
        "position_sizing": "fundamental",
    },
}

# Add aliases for missing strategies
STRATEGY_SCORING_MODELS[InvestmentStrategy.DEEP_VALUE] = STRATEGY_SCORING_MODELS[InvestmentStrategy.VALUE].copy()
STRATEGY_SCORING_MODELS[InvestmentStrategy.DEEP_VALUE]["name"] = "Deep Value"
STRATEGY_SCORING_MODELS[InvestmentStrategy.DIVIDEND_GROWTH] = STRATEGY_SCORING_MODELS[InvestmentStrategy.INCOME].copy()
STRATEGY_SCORING_MODELS[InvestmentStrategy.DIVIDEND_GROWTH]["name"] = "Dividend Growth"
STRATEGY_SCORING_MODELS[InvestmentStrategy.ALL_WEATHER] = STRATEGY_SCORING_MODELS[InvestmentStrategy.BALANCED].copy()
STRATEGY_SCORING_MODELS[InvestmentStrategy.ALL_WEATHER]["name"] = "All-Weather"


# =============================================================================
# AI PORTFOLIO ENGINE V5
# =============================================================================

class AIPortfolioEngine:
    """Full AI-integrated portfolio engine."""

    def __init__(self, universe: pd.DataFrame):
        self.universe = universe.copy()
        self._prepare_universe()

    def _prepare_universe(self):
        """Ensure columns exist."""
        df = self.universe

        required = [
            'ticker', 'sector', 'market_cap', 'pe_ratio', 'dividend_yield',
            'sentiment_score', 'fundamental_score', 'growth_score',
            'technical_score', 'dividend_score', 'options_flow_score',
            'short_squeeze_score', 'options_sentiment', 'squeeze_risk',
            'days_to_earnings', 'target_upside_pct',
            # Theme-based columns
            'ai_exposure_score', 'is_ai_company', 'ai_category'
        ]

        for col in required:
            if col not in df.columns:
                df[col] = None

        self.universe = df

    def _classify_catalyst(self, catalyst_type: Optional[str]) -> Tuple[str, str]:
        """Classify catalyst type - NO DEFAULTS, NO FABRICATION.

        Returns:
            (catalyst_class, catalyst_label)
            catalyst_class: REGULATORY | CLINICAL | COMMERCIAL | UNKNOWN
            catalyst_label: Human-readable label for display
        """
        if not catalyst_type or pd.isna(catalyst_type):
            return ("UNKNOWN", "Unknown Catalyst")

        t = str(catalyst_type).strip().upper()

        # REGULATORY (actual FDA events)
        regulatory_keywords = ["PDUFA", "NDA", "BLA", "SNDA", "SBLA", "ADCOM", "FDA APPROVAL", "FDA DECISION"]
        if any(k in t for k in regulatory_keywords):
            return ("REGULATORY", "Regulatory")

        # CLINICAL (trials, data readouts)
        clinical_keywords = ["PHASE", "TOPLINE", "DATA", "READOUT", "ABSTRACT", "ASCO", "ESMO", "CLINICAL", "TRIAL"]
        if any(k in t for k in clinical_keywords):
            return ("CLINICAL", "Clinical")

        # COMMERCIAL (product launches, sales)
        commercial_keywords = ["LAUNCH", "PRODUCT", "COMMERCIAL", "SALES", "REVENUE", "MARKET"]
        if any(k in t for k in commercial_keywords):
            return ("COMMERCIAL", "Commercial")

        # If we can't classify it, don't guess
        return ("UNKNOWN", "Unknown Catalyst")

    def _is_binary_event_dominant(self, row: pd.Series, catalyst: CatalystInfo) -> bool:
        """Determine if stock is dominated by binary event risk - NO ASSUMPTIONS.

        A stock is binary-event-dominant if:
        - Has a near-term regulatory or clinical catalyst (within 6 months)
        - AND is not clearly revenue-established/commercial stage
        """
        # No catalyst data = not binary-event-dominant
        if catalyst.days_to_fda is None:
            return False

        # Only count catalysts within 180 days (6 months)
        if catalyst.days_to_fda > 180:
            return False

        # Only regulatory and clinical events are binary
        if catalyst.catalyst_class not in ("REGULATORY", "CLINICAL"):
            return False

        # Check if commercial stage (requires actual data, not assumptions)
        stage = row.get("stage")
        if pd.notna(stage) and str(stage).upper() == "COMMERCIAL":
            return False

        # Check revenue/profitability as proxy for commercial (only if data exists)
        revenue = row.get("revenue") or row.get("total_revenue")
        if pd.notna(revenue) and float(revenue) > 100_000_000:  # $100M+ revenue
            return False

        # Has near-term binary catalyst and not clearly commercial
        return True

    def _is_established(self, row: pd.Series) -> bool:
        """Determine if stock is established/lower volatility - ONLY FROM DATA.

        Established means:
        - Commercial stage (from actual stage data), OR
        - Significant revenue (>$500M), OR
        - Large market cap (>$5B)

        NO ASSUMPTIONS - if no data, return False
        """
        # Check stage column (if exists)
        stage = row.get("stage")
        if pd.notna(stage) and str(stage).upper() == "COMMERCIAL":
            return True

        # Check revenue
        revenue = row.get("revenue") or row.get("total_revenue")
        if pd.notna(revenue) and float(revenue) > 500_000_000:
            return True

        # Check market cap
        market_cap = row.get("market_cap")
        if pd.notna(market_cap) and float(market_cap) > 5_000_000_000:
            return True

        return False

    def _apply_theme_filter(
        self,
        df: pd.DataFrame,
        intent: PortfolioIntent,
        user_request: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Apply theme-based filtering to the universe.

        Supports themes: ai, semiconductors, cybersecurity, fintech, cloud, etc.

        Returns:
            (filtered_df, warnings)
        """
        warnings = []
        theme = (intent.theme or "").lower()

        if not theme:
            return df, warnings

        # Define known AI companies by ticker (core AI builders)
        AI_BUILDER_TICKERS = {
            # AI Infrastructure & Chips
            'NVDA', 'AMD', 'INTC', 'AVGO', 'QCOM', 'ARM', 'TSM', 'ASML', 'AMAT', 'LRCX', 'KLAC', 'MU', 'MRVL',
            # AI Cloud & Platform
            'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'ORCL', 'IBM', 'CRM', 'SNOW', 'PLTR', 'DDOG', 'MDB', 'NET',
            # AI Software & Tools
            'ADBE', 'NOW', 'PANW', 'CRWD', 'ZS', 'FTNT', 'SNPS', 'CDNS', 'ANSS', 'WDAY', 'TEAM', 'HUBS', 'DOCU',
            # AI Pure-Play
            'AI', 'PATH', 'UPST', 'S', 'BBAI', 'SOUN', 'AITX', 'PRCT', 'GFAI',
            # AI Robotics & Automation
            'ISRG', 'ROK', 'TER', 'CGNX', 'IRBT',
            # AI in Automotive
            'TSLA', 'MBLY', 'LAZR', 'INVZ', 'AEVA', 'OUST',
        }

        # Define semiconductor companies
        SEMICONDUCTOR_TICKERS = {
            'NVDA', 'AMD', 'INTC', 'AVGO', 'QCOM', 'TXN', 'ADI', 'MCHP', 'NXPI', 'ON', 'SWKS', 'QRVO',
            'TSM', 'ASML', 'AMAT', 'LRCX', 'KLAC', 'MU', 'MRVL', 'ARM', 'MPWR', 'ALGM', 'WOLF',
            'SMCI', 'DELL', 'HPE', 'ANET',  # AI infrastructure
        }

        # Define cybersecurity companies
        CYBERSECURITY_TICKERS = {
            'PANW', 'CRWD', 'ZS', 'FTNT', 'S', 'OKTA', 'CYBR', 'QLYS', 'TENB', 'RPD', 'VRNS', 'SAIL',
            'NET', 'AKAM', 'FFIV', 'NLOK', 'FEYE', 'MIME',
        }

        # Define fintech companies
        FINTECH_TICKERS = {
            'V', 'MA', 'PYPL', 'SQ', 'COIN', 'AFRM', 'SOFI', 'UPST', 'HOOD', 'NU', 'MELI',
            'FIS', 'FISV', 'GPN', 'ADP', 'INTU', 'BL', 'BILL', 'TOST',
        }

        # Map themes to ticker sets
        THEME_TICKERS = {
            'ai': AI_BUILDER_TICKERS,
            'artificial_intelligence': AI_BUILDER_TICKERS,
            'semiconductor': SEMICONDUCTOR_TICKERS,
            'semiconductors': SEMICONDUCTOR_TICKERS,
            'chips': SEMICONDUCTOR_TICKERS,
            'cybersecurity': CYBERSECURITY_TICKERS,
            'security': CYBERSECURITY_TICKERS,
            'fintech': FINTECH_TICKERS,
            'financial_technology': FINTECH_TICKERS,
        }

        # Check if we have ai_exposure_score column with data
        has_ai_exposure = 'ai_exposure_score' in df.columns and df['ai_exposure_score'].notna().any()
        has_is_ai_company = 'is_ai_company' in df.columns and df['is_ai_company'].notna().any()

        # First try: Filter by ai_exposure_score if available and threshold specified
        if theme == 'ai' and has_ai_exposure and intent.min_ai_exposure:
            min_exposure = intent.min_ai_exposure
            filtered = df[df['ai_exposure_score'] >= min_exposure]
            if len(filtered) >= 10:  # Enough stocks
                logger.info(f"AI theme filter: {len(filtered)} stocks with ai_exposure_score >= {min_exposure}")
                return filtered, warnings
            else:
                warnings.append(f"Only {len(filtered)} stocks have ai_exposure_score >= {min_exposure}, using fallback")

        # Second try: Filter by is_ai_company boolean if available
        if theme == 'ai' and has_is_ai_company:
            filtered = df[df['is_ai_company'] == True]
            if len(filtered) >= 10:
                logger.info(f"AI theme filter: {len(filtered)} stocks with is_ai_company=True")
                return filtered, warnings
            else:
                warnings.append(f"Only {len(filtered)} stocks marked as AI companies, using fallback")

        # Third try: Use predefined ticker lists
        if theme in THEME_TICKERS:
            theme_set = THEME_TICKERS[theme]
            filtered = df[df['ticker'].str.upper().isin(theme_set)]

            if len(filtered) >= 5:
                logger.info(f"Theme '{theme}' filter: {len(filtered)} stocks from predefined list")
                return filtered, warnings
            else:
                warnings.append(f"Only {len(filtered)} stocks found for theme '{theme}' in database")
                # Fall back to sector-based filtering

        # Fourth try: Sector-based filtering for themes
        THEME_SECTORS = {
            'ai': ['Technology', 'Communication Services'],
            'semiconductor': ['Technology'],
            'semiconductors': ['Technology'],
            'cybersecurity': ['Technology'],
            'fintech': ['Technology', 'Financial Services'],
            'tech': ['Technology'],
            'healthcare': ['Healthcare'],
            'biotech': ['Healthcare'],
        }

        if theme in THEME_SECTORS:
            sectors = THEME_SECTORS[theme]
            filtered = df[df['sector'].isin(sectors)]

            if len(filtered) >= 10:
                logger.info(f"Theme '{theme}' sector filter: {len(filtered)} stocks from sectors {sectors}")
                warnings.append(f"Using sector-based filtering for theme '{theme}'. For more precise results, add ai_exposure_score column to your database.")
                return filtered, warnings

        # If no matching filter, warn but return original
        warnings.append(f"Unknown theme '{theme}' - no filtering applied. Supported themes: ai, semiconductor, cybersecurity, fintech")
        return df, warnings

    def detect_strategy(self, user_request: str, objective: str = None) -> InvestmentStrategy:
        """Detect strategy from request."""
        req = (user_request or "").lower()
        obj = (objective or "").lower()

        if any(w in req for w in ['biotech', 'pharma', 'drug', 'fda', 'clinical']):
            return InvestmentStrategy.BIOTECH_GROWTH
        # Extended AI/Tech keywords for better detection
        if any(w in req for w in ['tech', 'software', 'saas', 'cloud', 'ai ', ' ai',
                                   'artificial intelligence', 'machine learning', 'ml ', ' ml',
                                   'llm', 'gpu', 'cuda', 'inference', 'data center', 'datacenter',
                                   'semiconductor', 'chip', 'nvidia', 'amd', 'intel']):
            return InvestmentStrategy.TECH_GROWTH
        if 'aggressive' in req:
            return InvestmentStrategy.AGGRESSIVE_GROWTH
        if 'deep value' in req:
            return InvestmentStrategy.DEEP_VALUE
        if any(w in req for w in ['value', 'undervalued', 'cheap']):
            return InvestmentStrategy.VALUE
        if any(w in req for w in ['income', 'dividend']):
            return InvestmentStrategy.INCOME
        if any(w in req for w in ['conservative', 'safe']):
            return InvestmentStrategy.CONSERVATIVE
        if any(w in req for w in ['momentum', 'trend']):
            return InvestmentStrategy.MOMENTUM
        if any(w in req for w in ['quality', 'moat']):
            return InvestmentStrategy.QUALITY
        if any(w in req for w in ['shariah', 'halal', 'islamic']):
            return InvestmentStrategy.SHARIAH

        if obj == 'biotech_growth':
            return InvestmentStrategy.BIOTECH_GROWTH
        if obj in ['growth', 'tech_growth']:
            return InvestmentStrategy.TECH_GROWTH
        if obj == 'value':
            return InvestmentStrategy.VALUE
        if obj == 'income':
            return InvestmentStrategy.INCOME
        if obj == 'momentum':
            return InvestmentStrategy.MOMENTUM

        return InvestmentStrategy.BALANCED

    def _scale_probability(self, val) -> Optional[float]:
        """Scale probability from 0-1 to 0-100 if needed."""
        if pd.isna(val) or val is None:
            return None
        val = float(val)
        if val <= 1.0 and val > 0:
            return val * 100
        return val

    def _get_score(self, row: pd.Series, col: str, default: float = 50) -> Tuple[float, bool, str]:
        """Get score value."""
        val = row.get(col)
        if pd.notna(val) and val is not None:
            return float(val), True, col
        return default, False, f"{col} (missing)"

    def _build_ai_decision(self, row: pd.Series) -> AIDecision:
        """Build comprehensive AI decision."""
        sources = []

        # Primary AI
        ai_action = row.get('ai_action')
        ai_conf = row.get('ai_confidence')
        ai_prob = self._scale_probability(row.get('ai_probability'))
        if ai_action:
            sources.append('ai_analysis')
        if ai_prob:
            sources.append('ai_recommendations')

        # Committee
        committee_verdict = row.get('committee_verdict')
        committee_conv = row.get('committee_conviction')
        committee_rationale = row.get('committee_rationale')
        if committee_verdict:
            sources.append('committee_decisions')

        # Agent votes
        agent_votes = []
        for role in ['fundamental', 'sentiment', 'technical', 'valuation']:
            prob_col = f'agent_{role}_buy_prob'
            rationale_col = f'agent_{role}_rationale'
            conf_col = f'agent_{role}_confidence'

            prob = row.get(prob_col)
            if pd.notna(prob):
                agent_votes.append(AgentVote(
                    agent_role=role,
                    buy_prob=float(prob),
                    confidence=float(row.get(conf_col, 0.5) or 0.5),
                    rationale=str(row.get(rationale_col, ''))[:100]
                ))

        if agent_votes:
            sources.append('agent_votes')

        avg_agent = np.mean([v.buy_prob for v in agent_votes]) * 100 if agent_votes else None

        # Alpha predictions
        alpha_prob = self._scale_probability(row.get('alpha_probability'))
        alpha_signal = row.get('alpha_pred_signal') or row.get('alpha_signal')
        if alpha_prob:
            sources.append('alpha_predictions')

        # Enhanced scores
        insider = row.get('insider_score')
        revision = row.get('revision_score')
        earnings_surprise = row.get('earnings_surprise_score')
        if any(pd.notna(x) for x in [insider, revision, earnings_surprise]):
            sources.append('enhanced_scores')

        # Trading signal
        signal_type = row.get('trading_signal_type') or row.get('signal_type')
        signal_strength = row.get('trading_signal_strength') or row.get('signal_strength')
        signal_reason = row.get('trading_signal_reason') or row.get('signal_reason')
        if signal_type:
            sources.append('trading_signals')

        return AIDecision(
            ai_action=ai_action,
            ai_confidence=ai_conf,
            ai_probability=ai_prob,
            committee_verdict=committee_verdict,
            committee_conviction=float(committee_conv) if pd.notna(committee_conv) else None,
            committee_rationale=committee_rationale,
            agent_votes=agent_votes,
            avg_agent_buy_prob=avg_agent,
            alpha_probability=alpha_prob,
            alpha_signal=alpha_signal,
            insider_score=int(insider) if pd.notna(insider) else None,
            revision_score=int(revision) if pd.notna(revision) else None,
            earnings_surprise_score=int(earnings_surprise) if pd.notna(earnings_surprise) else None,
            signal_type=signal_type,
            signal_strength=int(signal_strength) if pd.notna(signal_strength) else None,
            signal_reason=signal_reason,
            bull_case=row.get('ai_bull_case'),
            bear_case=row.get('ai_bear_case'),
            key_risks=row.get('ai_key_risks'),
            one_line_summary=row.get('one_line_summary'),
            data_available=len(sources) > 0,
            data_sources=sources,
        )

    def _build_catalyst_info(self, row: pd.Series) -> CatalystInfo:
        """Build catalyst info - NO DEFAULTS, NO FABRICATION."""
        today = date.today()

        # Earnings
        earnings_date = row.get('next_earnings_date') or row.get('earnings_date')
        days_to_earnings = None
        if pd.notna(earnings_date):
            try:
                if isinstance(earnings_date, str):
                    earnings_date = datetime.strptime(earnings_date, '%Y-%m-%d').date()
                if hasattr(earnings_date, 'date'):
                    earnings_date = earnings_date.date()
                days_to_earnings = (earnings_date - today).days if earnings_date >= today else None
            except:
                earnings_date = None
                days_to_earnings = None

        # Fallback to screener days_to_earnings only if no date
        if days_to_earnings is None:
            dte = row.get('days_to_earnings')
            if pd.notna(dte) and dte > 0:
                days_to_earnings = int(dte)

        # Catalyst date (from fda_calendar table - but may be clinical/regulatory/commercial)
        cat_date = row.get('fda_expected_date')
        days_to_cat = None
        if pd.notna(cat_date):
            try:
                if isinstance(cat_date, str):
                    cat_date = datetime.strptime(cat_date, '%Y-%m-%d').date()
                days_to_cat = (cat_date - today).days if cat_date >= today else None
            except:
                cat_date = None
                days_to_cat = None

        # Classify catalyst type
        cat_type = row.get('fda_catalyst_type')
        cat_class, cat_label = self._classify_catalyst(cat_type)

        # Calculate catalyst score - START FROM 0, not 50
        score = 0
        has_catalyst = False
        description_parts = []

        # Earnings contribution (only if actual date exists)
        if days_to_earnings is not None and days_to_earnings >= 0:
            has_catalyst = True
            if days_to_earnings <= 30:
                if days_to_earnings <= 7:
                    score += 30
                elif days_to_earnings <= 14:
                    score += 25
                else:
                    score += 15
                description_parts.append(f"Earnings in {days_to_earnings}d")

        # Catalyst contribution (regulatory/clinical/commercial)
        if days_to_cat is not None and days_to_cat >= 0:
            has_catalyst = True
            if days_to_cat <= 180:
                if days_to_cat <= 30:
                    score += 30
                elif days_to_cat <= 60:
                    score += 20
                elif days_to_cat <= 90:
                    score += 10
                else:
                    score += 5
                # Use correct label based on classification
                description_parts.append(f"{cat_label} catalyst in {days_to_cat}d")

        # Options sentiment (only if explicitly bullish)
        if row.get('options_sentiment') == 'BULLISH':
            score += 10

        # Target upside (only if data available and meaningful)
        upside = row.get('target_upside_pct')
        if pd.notna(upside) and upside > 30:
            score += 10

        return CatalystInfo(
            earnings_date=earnings_date if pd.notna(earnings_date) else None,
            days_to_earnings=days_to_earnings,
            eps_estimate=row.get('eps_estimate') if pd.notna(row.get('eps_estimate')) else None,
            guidance_direction=row.get('guidance_direction') if pd.notna(row.get('guidance_direction')) else None,
            fda_date=cat_date if pd.notna(cat_date) else None,
            days_to_fda=days_to_cat,
            fda_drug=row.get('fda_drug_name') if pd.notna(row.get('fda_drug_name')) else None,
            fda_catalyst_type=cat_type if pd.notna(cat_type) else None,
            fda_priority=row.get('fda_priority') if pd.notna(row.get('fda_priority')) else None,
            catalyst_class=cat_class,
            catalyst_label=cat_label,
            catalyst_score=min(100, score) if has_catalyst else 0,  # 0 if no catalyst, not 50
            has_near_term_catalyst=has_catalyst,
            catalyst_description="; ".join(description_parts) if description_parts else None,
        )

    def _calculate_ai_committee_score(self, row: pd.Series, ai: AIDecision) -> Tuple[float, bool, str]:
        """Calculate score from committee + agents."""
        sources = []
        scores = []

        # Committee conviction (0-100)
        if ai.committee_conviction:
            scores.append(ai.committee_conviction)
            sources.append('committee')

        # Committee verdict
        verdict = ai.committee_verdict
        if verdict:
            if verdict in ['BUY', 'STRONG BUY']:
                scores.append(75)
            elif verdict == 'WEAK BUY':
                scores.append(60)
            elif verdict == 'HOLD':
                scores.append(50)
            elif verdict in ['SELL', 'WEAK SELL']:
                scores.append(30)
            sources.append('verdict')

        # Agent average
        if ai.avg_agent_buy_prob:
            scores.append(ai.avg_agent_buy_prob)
            sources.append('agents')

        if scores:
            return np.mean(scores), True, ', '.join(sources)
        return 50, False, 'ai_committee (missing)'

    def _calculate_ai_probability_score(self, row: pd.Series, ai: AIDecision) -> Tuple[float, bool, str]:
        """Get best AI probability."""
        # Priority: ai_probability > alpha_probability > inferred from action
        if ai.ai_probability:
            return ai.ai_probability, True, 'ai_recommendations'
        if ai.alpha_probability:
            return ai.alpha_probability, True, 'alpha_predictions'

        # Infer from action
        action = ai.ai_action
        conf = (ai.ai_confidence or '').upper()
        if action == 'STRONG_BUY':
            return 85, True, 'inferred (STRONG_BUY)'
        elif action == 'BUY':
            return 75 if conf == 'HIGH' else 65, True, f'inferred (BUY+{conf})'
        elif action == 'HOLD':
            return 50, True, 'inferred (HOLD)'
        elif action in ['SELL', 'WEAK SELL']:
            return 30, True, 'inferred (SELL)'

        return 50, False, 'ai_probability (missing)'

    def _calculate_enhanced_score(self, row: pd.Series, ai: AIDecision) -> Tuple[float, bool, str]:
        """Calculate enhanced score from insider, revision, etc."""
        score = 50
        sources = []
        has_data = False

        # Insider score (-10 to +10 typically)
        if ai.insider_score is not None:
            score += ai.insider_score
            sources.append('insider')
            has_data = True

        # Revision score
        if ai.revision_score is not None:
            score += ai.revision_score
            sources.append('revision')
            has_data = True

        # Earnings surprise
        if ai.earnings_surprise_score is not None:
            score += ai.earnings_surprise_score
            sources.append('surprise')
            has_data = True

        return max(0, min(100, score)), has_data, ', '.join(sources) if sources else 'enhanced (missing)'

    def _calculate_trading_signal_score(self, row: pd.Series, ai: AIDecision) -> Tuple[float, bool, str]:
        """Calculate score from trading signals."""
        sig = ai.signal_type
        strength = ai.signal_strength or 50

        if not sig:
            return 50, False, 'trading_signal (missing)'

        base = 50
        if sig == 'STRONG_BUY' or sig == 'STRONG BUY':
            base = 85
        elif sig == 'BUY':
            base = 70
        elif sig == 'HOLD':
            base = 50
        elif sig == 'SELL':
            base = 30
        elif sig == 'STRONG_SELL' or sig == 'STRONG SELL':
            base = 15

        # Blend with strength
        final = (base + strength) / 2
        return final, True, 'trading_signals'

    def calculate_composite_score(
        self,
        row: pd.Series,
        strategy: InvestmentStrategy,
        ai: AIDecision,
        catalyst: CatalystInfo
    ) -> Tuple[float, List[ScoreDetail], List[str], List[str]]:
        """Calculate composite score with all sources."""

        model = STRATEGY_SCORING_MODELS.get(strategy, STRATEGY_SCORING_MODELS[InvestmentStrategy.BALANCED])
        weights = model['weights']

        details = []
        used = []
        missing = []

        total_weight = sum(weights.values())
        composite = 0

        for name, weight in weights.items():
            pct = weight / total_weight

            if name == 'ai_committee_score':
                val, avail, src = self._calculate_ai_committee_score(row, ai)
            elif name == 'ai_probability':
                val, avail, src = self._calculate_ai_probability_score(row, ai)
            elif name == 'catalyst_score':
                val, avail, src = catalyst.catalyst_score, catalyst.has_near_term_catalyst, 'catalyst_calendar'
            elif name == 'enhanced_score':
                val, avail, src = self._calculate_enhanced_score(row, ai)
            elif name == 'trading_signal_score':
                val, avail, src = self._calculate_trading_signal_score(row, ai)
            elif name == 'options_flow':
                val, avail, src = self._get_score(row, 'options_flow_score')
            elif name == 'squeeze_potential':
                val, avail, src = self._get_score(row, 'short_squeeze_score')
            elif name == 'valuation_score':
                val, avail, src = self._calculate_valuation_score(row)
            else:
                val, avail, src = self._get_score(row, name)

            val = max(0, min(100, val))
            contrib = val * pct
            composite += contrib

            details.append(ScoreDetail(
                name=name,
                value=val,
                weight=pct,
                weighted_contribution=contrib,
                data_available=avail,
                raw_data={},
                data_source=src,
            ))

            if avail:
                used.append(name)
            else:
                missing.append(name)

        return composite, details, used, missing

    def _calculate_valuation_score(self, row: pd.Series) -> Tuple[float, bool, str]:
        """Valuation score."""
        score = 50
        has_data = False
        sources = []

        pe = row.get('pe_ratio')
        if pd.notna(pe) and pe > 0:
            has_data = True
            sources.append('pe')
            if pe < 10:
                score += 25
            elif pe < 15:
                score += 15
            elif pe < 25:
                score += 5
            elif pe > 40:
                score -= 15

        pb = row.get('pb_ratio')
        if pd.notna(pb) and pb > 0:
            has_data = True
            sources.append('pb')
            if pb < 1:
                score += 15
            elif pb < 2:
                score += 5

        return max(0, min(100, score)), has_data, ', '.join(sources) if sources else 'valuation (missing)'

    def _determine_conviction(
        self,
        row: pd.Series,
        score: float,
        ai: AIDecision,
        catalyst: CatalystInfo,
        strategy: InvestmentStrategy
    ) -> Tuple[str, str]:
        """Determine conviction."""

        ai_bullish = ai.ai_action in ['BUY', 'STRONG_BUY', 'STRONG BUY']
        committee_bullish = ai.committee_verdict in ['BUY', 'STRONG BUY', 'WEAK BUY']
        ai_prob = ai.ai_probability or 0
        ai_prob_high = ai_prob >= 65
        options_bullish = row.get('options_sentiment') == 'BULLISH'
        has_catalyst = catalyst.has_near_term_catalyst

        if strategy in [InvestmentStrategy.BIOTECH_GROWTH, InvestmentStrategy.TECH_GROWTH, InvestmentStrategy.AGGRESSIVE_GROWTH]:
            if (ai_bullish or committee_bullish) and ai_prob_high and score >= 65:
                return "HIGH", f"AI/Committee BUY with {ai_prob:.0f}% probability"
            elif (ai_bullish or committee_bullish) and has_catalyst:
                return "HIGH", "AI BUY with near-term catalyst"
            elif ai_bullish or (options_bullish and score >= 60):
                return "MEDIUM", "AI support or bullish options flow"
            elif score >= 60:
                return "MEDIUM", "Solid composite score"
            else:
                return "LOW", "Limited conviction signals"

        elif strategy in [InvestmentStrategy.VALUE, InvestmentStrategy.CONSERVATIVE]:
            if score >= 70 and (ai_bullish or committee_bullish):
                return "HIGH", "Strong fundamentals with AI validation"
            elif score >= 65:
                return "MEDIUM", "Good fundamental score"
            else:
                return "LOW", "Below-average conviction"

        else:
            if (ai_bullish or committee_bullish) and score >= 65:
                return "HIGH", "AI/Committee buy with strong score"
            elif score >= 60 or ai_prob_high:
                return "MEDIUM", "Decent score or AI probability"
            else:
                return "LOW", "Below-average signals"

    def _generate_evidence(
        self,
        row: pd.Series,
        ai: AIDecision,
        catalyst: CatalystInfo,
        strategy: InvestmentStrategy
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Generate evidence."""
        bull = []
        bear = []
        catalysts = []
        risks = []

        # AI evidence
        if ai.ai_action in ['BUY', 'STRONG_BUY', 'STRONG BUY']:
            bull.append(f"AI recommends {ai.ai_action} ({ai.ai_confidence} confidence)")
        elif ai.ai_action in ['SELL', 'WEAK SELL']:
            bear.append(f"AI recommends {ai.ai_action}")

        if ai.committee_verdict in ['BUY', 'STRONG BUY']:
            bull.append(f"Committee verdict: {ai.committee_verdict} (conviction: {ai.committee_conviction:.0f})" if ai.committee_conviction else f"Committee verdict: {ai.committee_verdict}")
        elif ai.committee_verdict in ['SELL', 'WEAK SELL']:
            bear.append(f"Committee verdict: {ai.committee_verdict}")

        if ai.ai_probability and ai.ai_probability >= 65:
            bull.append(f"High AI probability ({ai.ai_probability:.0f}%)")
        elif ai.ai_probability and ai.ai_probability <= 40:
            bear.append(f"Low AI probability ({ai.ai_probability:.0f}%)")

        if ai.avg_agent_buy_prob and ai.avg_agent_buy_prob >= 70:
            bull.append(f"Agent consensus bullish ({ai.avg_agent_buy_prob:.0f}%)")

        # Enhanced scores
        if ai.insider_score and ai.insider_score > 5:
            bull.append(f"Positive insider activity (score: +{ai.insider_score})")
        elif ai.insider_score and ai.insider_score < -5:
            bear.append(f"Insider selling (score: {ai.insider_score})")

        if ai.earnings_surprise_score and ai.earnings_surprise_score > 5:
            bull.append(f"Recent earnings beat")

        # Market cap
        mc = row.get('market_cap', 0) or 0
        if mc >= 10e9:
            bull.append(f"Large cap (${mc/1e9:.1f}B) stability")
        elif mc >= 1e9:
            bull.append(f"Mid cap (${mc/1e9:.1f}B) growth potential")
        elif mc > 0:
            bear.append(f"Small cap (${mc/1e6:.0f}M) volatility risk")

        # Options
        if row.get('options_sentiment') == 'BULLISH':
            bull.append("Bullish options flow - smart money buying")
            catalysts.append("Institutional accumulation")
        elif row.get('options_sentiment') == 'BEARISH':
            bear.append("Bearish options flow")
            risks.append("Smart money selling")

        # Squeeze
        if row.get('squeeze_risk') == 'EXTREME':
            bull.append("Extreme short squeeze potential")
            catalysts.append("Potential squeeze event")

        # Catalysts
        if catalyst.days_to_earnings and catalyst.days_to_earnings <= 30:
            catalysts.append(f"Earnings in {catalyst.days_to_earnings} days")
            risks.append("Earnings volatility")

        if catalyst.days_to_fda and catalyst.days_to_fda <= 180:
            # Use correct catalyst label based on classification
            cat_desc = f"{catalyst.catalyst_label}"
            if catalyst.fda_catalyst_type:
                cat_desc += f" ({catalyst.fda_catalyst_type})"
            if catalyst.fda_drug:
                cat_desc += f": {catalyst.fda_drug}"
            cat_desc += f" in {catalyst.days_to_fda} days"
            if catalyst.fda_priority:
                cat_desc += f" [{catalyst.fda_priority}]"

            catalysts.append(cat_desc)

            # Risk depends on catalyst class
            if catalyst.catalyst_class == "REGULATORY":
                risks.append("Binary regulatory approval risk")
            elif catalyst.catalyst_class == "CLINICAL":
                risks.append("Binary clinical readout risk")
            elif catalyst.catalyst_class == "COMMERCIAL":
                risks.append("Commercial execution risk")
            else:
                risks.append("Event-driven catalyst risk")

        # Strategy risks - only for biotech and only if appropriate
        if strategy == InvestmentStrategy.BIOTECH_GROWTH:
            # Only add if we have actual clinical/regulatory events
            has_binary_risk = any("Binary" in r for r in risks)
            if not has_binary_risk and row.get('sector') == "Healthcare":
                risks.append("Clinical/regulatory risk")
            risks.append("Dilution risk")

        # NO DEFAULT FABRICATIONS - only add if actually empty
        if not bull:
            bull.append("Position included for diversification")
        if not bear:
            bear.append("General market risk")
        if not catalysts:
            catalysts.append("Operational execution")
        if not risks:
            risks.append("Market volatility")

        return bull, bear, catalysts, risks

    def _enforce_advanced_constraints(
        self,
        all_scored: List[Dict],
        initial_selected: List[Dict],
        intent: PortfolioIntent,
        warnings: List[str]
    ) -> List[Dict]:
        """Enforce advanced constraints - ONLY IF DATA EXISTS.

        Returns updated selection that satisfies:
        - min_subsectors (if subsector column exists)
        - max_binary_event_weight_pct (if stage/catalyst data exists)
        - min/max_established_weight_pct (if stage/revenue data exists)
        """
        selected = initial_selected.copy()

        # Check if required columns exist
        has_subsector = 'subsector' in self.universe.columns
        has_stage = 'stage' in self.universe.columns

        # 1. Subsector diversity constraint
        if intent.min_subsectors and has_subsector:
            subsectors = set()
            for s in selected:
                sub = s['row'].get('subsector')
                if pd.notna(sub):
                    subsectors.add(str(sub))

            if len(subsectors) < intent.min_subsectors:
                warnings.append(
                    f"Subsector diversity: {len(subsectors)} subsectors vs target {intent.min_subsectors}. "
                    f"Consider relaxing filters."
                )
        elif intent.min_subsectors and not has_subsector:
            warnings.append(
                "Cannot enforce min_subsectors constraint - 'subsector' column not in database. "
                "Add subsector classification to enable this feature."
            )

        # 2. Binary event concentration constraint
        if intent.max_binary_event_weight_pct:
            # Calculate equal weights to check constraint
            equal_wt = 100 / len(selected) if selected else 0
            binary_weight = 0

            for s in selected:
                if self._is_binary_event_dominant(s['row'], s['catalyst']):
                    binary_weight += equal_wt

            if binary_weight > intent.max_binary_event_weight_pct:
                warnings.append(
                    f"Binary event concentration: {binary_weight:.1f}% exceeds limit {intent.max_binary_event_weight_pct}%. "
                    f"Current portfolio has significant binary catalyst risk."
                )

        # 3. Established stocks bucket constraint
        if intent.min_established_weight_pct or intent.max_established_weight_pct:
            equal_wt = 100 / len(selected) if selected else 0
            established_weight = 0

            for s in selected:
                if self._is_established(s['row']):
                    established_weight += equal_wt

            if intent.min_established_weight_pct and established_weight < intent.min_established_weight_pct:
                warnings.append(
                    f"Established stocks: {established_weight:.1f}% below minimum {intent.min_established_weight_pct}%. "
                    f"Consider including more revenue-established names."
                )

            if intent.max_established_weight_pct and established_weight > intent.max_established_weight_pct:
                warnings.append(
                    f"Established stocks: {established_weight:.1f}% exceeds maximum {intent.max_established_weight_pct}%. "
                    f"Portfolio may be too conservative."
                )

        return selected

    def build_portfolio(
        self,
        intent: PortfolioIntent,
        user_request: str = ""
    ) -> PortfolioResult:
        """Build portfolio with full AI integration."""

        strategy = self.detect_strategy(user_request, intent.objective)
        model = STRATEGY_SCORING_MODELS.get(strategy, STRATEGY_SCORING_MODELS[InvestmentStrategy.BALANCED])

        constraints = model['constraints'].copy()
        if intent.max_holdings:
            constraints['max_holdings'] = intent.max_holdings
        if intent.max_position_pct:
            constraints['max_position_pct'] = intent.max_position_pct

        warnings = []
        errors = []

        # Filter
        df = self.universe.copy()

        # Handle ticker inclusion - if user specifies tickers, include ALL of them
        if intent.tickers_include and len(intent.tickers_include) > 0:
            whitelist = [t.upper() for t in intent.tickers_include]
            
            # If restrict_to_tickers OR user provided 5+ specific tickers, use ONLY those
            if intent.restrict_to_tickers or len(whitelist) >= 5:
                # First, diagnose why tickers might be missing BEFORE filtering
                requested = set(whitelist)
                available_in_df = set(df['ticker'].str.upper())
                
                # Check for tickers not in database at all
                not_in_db = requested - available_in_df
                if not_in_db:
                    warnings.append(f"⚠️ Tickers not found in database: {', '.join(sorted(not_in_db))}")
                
                # Check for tickers with missing critical data
                tickers_in_df = df[df['ticker'].str.upper().isin(whitelist)].copy()
                
                # Check market_cap issues
                if 'market_cap' in tickers_in_df.columns:
                    no_market_cap = tickers_in_df[tickers_in_df['market_cap'].isna()]['ticker'].tolist()
                    if no_market_cap:
                        warnings.append(f"⚠️ Tickers with no market cap data: {', '.join(no_market_cap)}")
                
                # Check fundamental_score issues
                if 'fundamental_score' in tickers_in_df.columns:
                    no_fundamental = tickers_in_df[tickers_in_df['fundamental_score'].isna()]['ticker'].tolist()
                    if no_fundamental:
                        warnings.append(f"⚠️ Tickers with no fundamental scores: {', '.join(no_fundamental)}")
                
                # Now filter to whitelist
                df = df[df['ticker'].str.upper().isin(whitelist)]
                
                # Override holdings limits to include ALL specified tickers
                available_count = len(df)
                if available_count > 0:
                    constraints['min_holdings'] = available_count
                    constraints['max_holdings'] = available_count
                    
                # Summary warning if not all tickers included
                if available_count < len(whitelist):
                    warnings.append(f"📊 Requested {len(whitelist)} tickers, only {available_count} available for portfolio")
            else:
                # Just ensure these tickers are included, but allow others too
                pass

        if intent.tickers_exclude:
            df = df[~df['ticker'].str.upper().isin([t.upper() for t in intent.tickers_exclude])]

        if intent.sectors_include:
            df = df[df['sector'].str.upper().isin([s.upper() for s in intent.sectors_include])]

        if not intent.restrict_to_tickers and constraints.get('min_market_cap'):
            df = df[df['market_cap'] >= constraints['min_market_cap']]

        if intent.require_ai_buy:
            df = df[df['ai_action'].isin(['BUY', 'STRONG_BUY', 'STRONG BUY'])]

        # Theme-based filtering (AI, Semiconductors, etc.)
        # SKIP theme filter if user explicitly specified tickers with restrict_to_tickers
        if intent.theme and intent.require_theme_match:
            if intent.restrict_to_tickers and intent.tickers_include and len(intent.tickers_include) >= 5:
                # User specified exact tickers - skip theme filter, respect their choices
                logger.info(f"Skipping theme filter - user specified {len(intent.tickers_include)} tickers with restrict_to_tickers=True")
            else:
                df, theme_warnings = self._apply_theme_filter(df, intent, user_request)
                warnings.extend(theme_warnings)

        if df.empty:
            return self._empty_result(intent, constraints, ["No stocks pass filters"], model)

        # Score all stocks
        scored = []
        total_avail = 0
        total_missing = 0
        ai_count = 0
        catalyst_count = 0

        for _, row in df.iterrows():
            ai = self._build_ai_decision(row)
            catalyst = self._build_catalyst_info(row)
            score, details, used, miss = self.calculate_composite_score(row, strategy, ai, catalyst)
            conv, rationale = self._determine_conviction(row, score, ai, catalyst, strategy)
            bull, bear, cats, risks = self._generate_evidence(row, ai, catalyst, strategy)

            total_avail += len(used)
            total_missing += len(miss)
            if ai.data_available:
                ai_count += 1
            if catalyst.has_near_term_catalyst:
                catalyst_count += 1

            scored.append({
                'row': row,
                'score': score,
                'details': details,
                'used': used,
                'missing': miss,
                'ai': ai,
                'catalyst': catalyst,
                'conviction': conv,
                'rationale': rationale,
                'bull': bull,
                'bear': bear,
                'cats': cats,
                'risks': risks,
            })

        scored.sort(key=lambda x: x['score'], reverse=True)

        # Select with min/max holdings
        min_h = intent.min_holdings or constraints.get('min_holdings', 8)
        max_h = intent.max_holdings or constraints.get('max_holdings', 25)

        # Ensure we have enough stocks
        if len(scored) < min_h:
            warnings.append(f"Only {len(scored)} stocks available, need at least {min_h}")
            selected = scored  # Use what we have
        else:
            selected = scored[:max_h]

        # Advanced constraint enforcement (if requested)
        if any([intent.min_subsectors, intent.max_binary_event_weight_pct,
                intent.min_established_weight_pct, intent.max_established_weight_pct]):
            selected = self._enforce_advanced_constraints(
                scored, selected, intent, warnings
            )

        # Weights - CRITICAL: Handle cash buffer override
        cash_buffer = constraints.get('cash_buffer_pct', 5)

        # Intent overrides (NO FABRICATION - use explicit settings only)
        if intent.fully_invested:
            cash_buffer = 0
        elif intent.cash_buffer_pct is not None:
            cash_buffer = float(intent.cash_buffer_pct)

        investable = 100 - cash_buffer
        max_pos = intent.max_position_pct or constraints.get('max_position_pct', 10)

        total_score = sum(s['score'] for s in selected)
        sizing = model.get('position_sizing', 'score')

        holdings = []
        for s in selected:
            row = s['row']

            if sizing == 'equal' or intent.equal_weight:
                raw = investable / len(selected)
            elif sizing == 'conviction':
                mult = {'HIGH': 1.4, 'MEDIUM': 1.0, 'LOW': 0.7}
                adj = s['score'] * mult.get(s['conviction'], 1.0)
                adj_total = sum(ss['score'] * mult.get(ss['conviction'], 1.0) for ss in selected)
                raw = (adj / adj_total) * investable if adj_total > 0 else investable / len(selected)
            else:
                raw = (s['score'] / total_score) * investable if total_score > 0 else investable / len(selected)

            weight = min(raw, max_pos)
            value = weight / 100 * intent.portfolio_value
            price = row.get('price', 100) or 100
            shares = int(value / price)

            holdings.append(PortfolioHolding(
                ticker=row['ticker'],
                company_name=row.get('company_name', row['ticker']),
                weight_pct=weight,
                shares=shares,
                value=value,
                sector=row.get('sector', 'Unknown') or 'Unknown',
                composite_score=s['score'],
                score_details=s['details'],
                scores_used=s['used'],
                scores_missing=s['missing'],
                ai_decision=s['ai'],
                catalyst_info=s['catalyst'],
                conviction=s['conviction'],
                conviction_rationale=s['rationale'],
                bull_case=s['bull'],
                bear_case=s['bear'],
                key_catalysts=s['cats'],
                key_risks=s['risks'],
                market_cap=row.get('market_cap'),
                pe_ratio=row.get('pe_ratio'),
                dividend_yield=row.get('dividend_yield'),
                revenue_growth=row.get('revenue_growth'),
                gross_margin=row.get('gross_margin'),
                debt_to_equity=row.get('debt_to_equity'),
                options_sentiment=row.get('options_sentiment'),
                squeeze_risk=row.get('squeeze_risk'),
            ))

        # Normalize
        total_w = sum(h.weight_pct for h in holdings)
        if total_w > investable:
            scale = investable / total_w
            for h in holdings:
                h.weight_pct *= scale
                h.value = h.weight_pct / 100 * intent.portfolio_value

        # Metrics
        sectors = {}
        for h in holdings:
            sectors[h.sector] = sectors.get(h.sector, 0) + h.weight_pct

        invested_pct = sum(h.weight_pct for h in holdings)
        invested_val = invested_pct / 100 * intent.portfolio_value
        cash_val = intent.portfolio_value - invested_val

        ai_probs = [h.ai_decision.ai_probability for h in holdings if h.ai_decision.ai_probability]
        avg_ai = np.mean(ai_probs) if ai_probs else None

        convictions = [h.ai_decision.committee_conviction for h in holdings if h.ai_decision.committee_conviction]
        avg_conv = np.mean(convictions) if convictions else None

        ai_cov = (ai_count / len(df) * 100) if len(df) > 0 else 0
        cat_cov = (catalyst_count / len(df) * 100) if len(df) > 0 else 0

        thesis = self._generate_thesis(strategy, holdings, model, avg_ai, avg_conv)
        key_risks = self._generate_risks(strategy, holdings, sectors)
        triggers = self._generate_triggers(strategy)

        return PortfolioResult(
            success=True,
            holdings=holdings,
            total_value=intent.portfolio_value,
            invested_value=invested_val,
            cash_value=cash_val,
            num_holdings=len(holdings),
            num_sectors=len(sectors),
            sectors=sectors,
            avg_score=np.mean([h.composite_score for h in holdings]),
            avg_ai_probability=avg_ai,
            avg_committee_conviction=avg_conv,
            strategy_name=model['name'],
            strategy_description=model['description'],
            total_scores_available=total_avail,
            total_scores_missing=total_missing,
            ai_coverage_pct=ai_cov,
            catalyst_coverage_pct=cat_cov,
            portfolio_thesis=thesis,
            key_risks=key_risks,
            rebalance_triggers=triggers,
            warnings=warnings,
            errors=errors,
            intent=intent,
            constraints_used=constraints,
        )

    def _empty_result(self, intent, constraints, errors, model):
        return PortfolioResult(
            success=False, holdings=[], total_value=intent.portfolio_value,
            invested_value=0, cash_value=intent.portfolio_value,
            num_holdings=0, num_sectors=0, sectors={},
            avg_score=0, avg_ai_probability=None, avg_committee_conviction=None,
            strategy_name=model['name'], strategy_description=model['description'],
            total_scores_available=0, total_scores_missing=0,
            ai_coverage_pct=0, catalyst_coverage_pct=0,
            portfolio_thesis="", key_risks=[], rebalance_triggers=[],
            warnings=[], errors=errors, intent=intent, constraints_used=constraints,
        )

    def _generate_thesis(self, strategy, holdings, model, avg_ai, avg_conv):
        high = len([h for h in holdings if h.conviction == 'HIGH'])
        med = len([h for h in holdings if h.conviction == 'MEDIUM'])
        ai_buys = len([h for h in holdings if h.ai_decision.ai_action in ['BUY', 'STRONG_BUY', 'STRONG BUY']])
        comm_buys = len([h for h in holdings if h.ai_decision.committee_verdict in ['BUY', 'STRONG BUY']])

        ai_info = ""
        if avg_ai:
            ai_info = f"\n\nAI Analysis: {ai_buys}/{len(holdings)} AI BUY, {comm_buys}/{len(holdings)} Committee BUY. Avg AI probability: {avg_ai:.0f}%."
        if avg_conv:
            ai_info += f" Avg committee conviction: {avg_conv:.0f}."

        return f"""This {model['name']} portfolio contains {len(holdings)} positions with {high} high-conviction and {med} medium-conviction holdings.
{ai_info}

Strategy: {model['description']}

Scoring integrates: AI recommendations, committee decisions, agent votes, enhanced scores, catalysts, and platform fundamentals."""

    def _generate_risks(self, strategy, holdings, sectors):
        risks = []
        max_sec = max(sectors.values()) if sectors else 0
        if max_sec > 30:
            top = max(sectors.items(), key=lambda x: x[1])[0]
            risks.append(f"Sector concentration: {top} at {max_sec:.0f}%")

        no_ai = len([h for h in holdings if not h.ai_decision.data_available])
        if no_ai > len(holdings) * 0.3:
            risks.append(f"{no_ai}/{len(holdings)} holdings lack AI data")

        if strategy == InvestmentStrategy.BIOTECH_GROWTH:
            # Check if we actually have binary events in the portfolio
            has_regulatory = any(h.catalyst_info.catalyst_class == "REGULATORY" for h in holdings)
            has_clinical = any(h.catalyst_info.catalyst_class == "CLINICAL" for h in holdings)

            if has_regulatory and has_clinical:
                risks.append("Binary regulatory/clinical risk")
            elif has_regulatory:
                risks.append("Binary regulatory approval risk")
            elif has_clinical:
                risks.append("Binary clinical readout risk")
            else:
                risks.append("Clinical/regulatory risk")

            risks.append("Dilution risk")
        if strategy == InvestmentStrategy.MOMENTUM:
            risks.append("Momentum reversal risk")

        risks.append("General market risk")
        return risks[:6]

    def _generate_triggers(self, strategy):
        triggers = [
            "Monthly position review",
            "AI signal change (BUY→HOLD/SELL)",
            "Committee verdict change",
            "Position exceeds max weight",
        ]
        if strategy == InvestmentStrategy.BIOTECH_GROWTH:
            triggers.extend(["FDA catalyst event", "Cash runway < 12 months"])
        return triggers[:6]


# =============================================================================
# BACKWARD COMPATIBILITY CLASSES
# =============================================================================

class PortfolioEngine:
    """Backward-compatible wrapper that passes user_request for theme detection."""
    def __init__(self, universe: pd.DataFrame):
        self.engine = AIPortfolioEngine(universe)

    def build_portfolio(self, intent: PortfolioIntent, user_request: str = "") -> PortfolioResult:
        """Build portfolio - now accepts user_request for proper theme detection."""
        return self.engine.build_portfolio(intent, user_request)


class IntelligentPortfolioEngine:
    def __init__(self, universe: pd.DataFrame):
        self.engine = AIPortfolioEngine(universe)
    def build_portfolio(self, intent: PortfolioIntent, user_request: str = "") -> PortfolioResult:
        return self.engine.build_portfolio(intent, user_request)



# =============================================================================
# REQUIRED BY portfolio_builder.py - DO NOT REMOVE
# =============================================================================

# Risk constraints mapping
RISK_CONSTRAINTS = {
    RiskLevel.CONSERVATIVE: {
        'max_position_pct': 5,
        'max_sector_pct': 25,
        'min_holdings': 20,
        'max_holdings': 40,
        'cash_buffer_pct': 10,
    },
    RiskLevel.MODERATE: {
        'max_position_pct': 8,
        'max_sector_pct': 30,
        'min_holdings': 15,
        'max_holdings': 30,
        'cash_buffer_pct': 5,
    },
    RiskLevel.AGGRESSIVE: {
        'max_position_pct': 12,
        'max_sector_pct': 40,
        'min_holdings': 10,
        'max_holdings': 25,
        'cash_buffer_pct': 3,
    },
}

# Objective weights mapping
OBJECTIVE_WEIGHTS = {
    'balanced': {'fundamental': 0.25, 'technical': 0.25, 'sentiment': 0.25, 'options': 0.25},
    'growth': {'fundamental': 0.20, 'technical': 0.30, 'sentiment': 0.30, 'options': 0.20},
    'tech_growth': {'ai_committee': 0.20, 'ai_probability': 0.15, 'growth': 0.20, 'fundamental': 0.15, 'options': 0.15, 'sentiment': 0.10, 'catalyst': 0.05},
    'value': {'fundamental': 0.40, 'technical': 0.20, 'sentiment': 0.20, 'options': 0.20},
    'income': {'fundamental': 0.35, 'technical': 0.15, 'sentiment': 0.20, 'dividend': 0.30},
    'momentum': {'fundamental': 0.15, 'technical': 0.40, 'sentiment': 0.30, 'options': 0.15},
    'biotech_growth': {'ai_committee': 0.20, 'ai_probability': 0.15, 'catalyst': 0.20, 'options': 0.15, 'fundamental': 0.10, 'sentiment': 0.10, 'squeeze': 0.05, 'enhanced': 0.05},
    'aggressive': {'ai_committee': 0.20, 'ai_probability': 0.15, 'growth': 0.20, 'options': 0.15, 'catalyst': 0.15, 'squeeze': 0.10, 'sentiment': 0.05},
    'conservative': {'fundamental': 0.30, 'dividend': 0.25, 'ai_committee': 0.15, 'valuation': 0.15, 'enhanced': 0.10, 'sentiment': 0.05},
    'quality': {'fundamental': 0.30, 'ai_committee': 0.20, 'growth': 0.20, 'enhanced': 0.15, 'dividend': 0.10, 'sentiment': 0.05},
    'shariah': {'fundamental': 0.30, 'ai_committee': 0.20, 'growth': 0.20, 'valuation': 0.15, 'sentiment': 0.10, 'dividend': 0.05},
}


def get_intent_extraction_prompt(user_request: str, sectors: list = None, tickers: list = None, available_tickers: list = None) -> str:
    """Generate the prompt for LLM to extract portfolio intent."""
    ticker_hint = ""
    sector_hint = ""

    # Use tickers parameter if provided, otherwise fall back to available_tickers
    available_tickers_list = tickers or available_tickers

    if available_tickers_list:
        ticker_hint = f"\nAvailable tickers in database: {', '.join(available_tickers_list[:50])}..."

    if sectors:
        sector_hint = f"\nAvailable sectors: {', '.join(sectors)}"

    return f"""Extract portfolio construction intent from this user request.

User Request: "{user_request}"
{ticker_hint}{sector_hint}

Return ONLY valid JSON with these fields:
{{
    "objective": "balanced|growth|value|income|momentum|biotech_growth|tech_growth|shariah|conservative",
    "risk_level": "conservative|moderate|aggressive",
    "portfolio_value": <number or null>,
    "min_holdings": <number or null>,
    "max_holdings": <number or null>,
    "max_position_pct": <number or null>,
    "max_sector_pct": <number or null>,
    "min_dividend_yield": <number or null>,
    "cash_buffer_pct": <number 0-100 or null>,
    "fully_invested": <true to invest 100% with no cash buffer>,
    "min_subsectors": <minimum number of distinct subsectors or null>,
    "max_binary_event_weight_pct": <max % in binary-event stocks or null>,
    "min_established_weight_pct": <min % in established/commercial stocks or null>,
    "max_established_weight_pct": <max % in established/commercial stocks or null>,
    "sectors_include": [<list of sectors to include>],
    "sectors_exclude": [<list of sectors to exclude>],
    "tickers_include": [<list of specific tickers to include>],
    "tickers_exclude": [<list of tickers to exclude>],
    "restrict_to_tickers": <true if ONLY use specified tickers>,
    "equal_weight": <true for equal weighting>,
    "shariah_compliant": <true for Islamic finance compliance>,
    "require_ai_buy": <true to require AI BUY signal>,
    "min_ai_probability": <minimum AI probability 0-100>,
    "require_catalyst": <true to require near-term catalyst>,
    "theme": <"ai"|"semiconductor"|"cybersecurity"|"fintech"|"cloud"|null - industry/technology theme>,
    "theme_mode": <"builders"|"adopters"|null - for AI: companies building AI vs using AI>,
    "require_theme_match": <true to only include stocks matching the theme>,
    "min_ai_exposure": <minimum AI exposure score 0-100 for AI theme, or null>
}}

Rules:
- Extract numeric values where mentioned
- If user says "total weights = 100%" or "fully invested", set fully_invested=true
- If user specifies subsector diversity (e.g. "at least 8 subsectors"), set min_subsectors
- If user specifies binary event limits (e.g. "no more than 30% in binary events"), set max_binary_event_weight_pct
- If user wants established names (e.g. "10-20% in revenue-established"), set min/max_established_weight_pct
- Default risk_level to "moderate" if not specified
- Set restrict_to_tickers=true if user says "only these tickers" or provides a specific list
- For biotech requests, use objective="biotech_growth"
- For AI/ML/LLM/GPU/semiconductor/chip requests, set theme="ai" or theme="semiconductor" AND require_theme_match=true
- When user mentions "AI companies", "AI products", "artificial intelligence", "machine learning", set theme="ai" and require_theme_match=true
- When user mentions "chip companies", "semiconductor", "GPU", set theme="semiconductor" and require_theme_match=true
- Return ONLY the JSON, no explanation"""


def parse_llm_intent(llm_response: str, valid_tickers: list = None, valid_sectors: list = None) -> Tuple[PortfolioIntent, List[str]]:
    """Parse LLM response into PortfolioIntent.

    Args:
        llm_response: JSON string from LLM
        valid_tickers: List of valid tickers to validate against
        valid_sectors: List of valid sectors to validate against
    """
    errors = []

    # Try to extract JSON from response
    json_match = re.search(r'\{[^{}]*\}', llm_response, re.DOTALL)

    if not json_match:
        errors.append("No JSON found in LLM output")
        return PortfolioIntent(), errors

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        errors.append(f"JSON parse error: {e}")
        return PortfolioIntent(), errors

    # Validate tickers if provided
    if valid_tickers:
        valid_tickers_upper = [t.upper() for t in valid_tickers]

        # Validate tickers_include
        tickers_include = data.get('tickers_include', [])
        if tickers_include:
            invalid_tickers = [t for t in tickers_include if t.upper() not in valid_tickers_upper]
            if invalid_tickers:
                errors.append(f"Invalid tickers in include list: {', '.join(invalid_tickers[:10])}")

        # Validate tickers_exclude
        tickers_exclude = data.get('tickers_exclude', [])
        if tickers_exclude:
            invalid_tickers = [t for t in tickers_exclude if t.upper() not in valid_tickers_upper]
            if invalid_tickers:
                errors.append(f"Invalid tickers in exclude list: {', '.join(invalid_tickers[:10])}")

    # Validate sectors if provided
    if valid_sectors:
        valid_sectors_lower = [s.lower() for s in valid_sectors]

        # Validate sectors_include
        sectors_include = data.get('sectors_include', [])
        if sectors_include:
            invalid_sectors = [s for s in sectors_include if s.lower() not in valid_sectors_lower]
            if invalid_sectors:
                errors.append(f"Invalid sectors in include list: {', '.join(invalid_sectors)}")

        # Validate sectors_exclude
        sectors_exclude = data.get('sectors_exclude', [])
        if sectors_exclude:
            invalid_sectors = [s for s in sectors_exclude if s.lower() not in valid_sectors_lower]
            if invalid_sectors:
                errors.append(f"Invalid sectors in exclude list: {', '.join(invalid_sectors)}")

    # Map to PortfolioIntent
    try:
        intent = PortfolioIntent(
            objective=data.get('objective', 'balanced'),
            risk_level=data.get('risk_level', 'moderate'),
            portfolio_value=data.get('portfolio_value', 100000) or 100000,
            min_holdings=data.get('min_holdings'),
            max_holdings=data.get('max_holdings'),
            max_position_pct=data.get('max_position_pct'),
            max_sector_pct=data.get('max_sector_pct'),
            min_dividend_yield=data.get('min_dividend_yield'),
            cash_buffer_pct=data.get('cash_buffer_pct'),
            fully_invested=data.get('fully_invested', False),
            min_subsectors=data.get('min_subsectors'),
            max_binary_event_weight_pct=data.get('max_binary_event_weight_pct'),
            min_established_weight_pct=data.get('min_established_weight_pct'),
            max_established_weight_pct=data.get('max_established_weight_pct'),
            sectors_include=data.get('sectors_include', []),
            sectors_exclude=data.get('sectors_exclude', []),
            tickers_include=data.get('tickers_include', []),
            tickers_exclude=data.get('tickers_exclude', []),
            restrict_to_tickers=data.get('restrict_to_tickers', False),
            equal_weight=data.get('equal_weight', False),
            shariah_compliant=data.get('shariah_compliant', False),
            require_ai_buy=data.get('require_ai_buy', False),
            min_ai_probability=data.get('min_ai_probability'),
            require_catalyst=data.get('require_catalyst', False),
            # Theme-based fields
            theme=data.get('theme'),
            theme_mode=data.get('theme_mode'),
            require_theme_match=data.get('require_theme_match', False),
            min_ai_exposure=data.get('min_ai_exposure'),
        )
    except Exception as e:
        errors.append(f"Intent creation error: {e}")
        return PortfolioIntent(), errors

    return intent, errors


def check_shariah_compliance_from_data(row: dict) -> Tuple[bool, List[str]]:
    """Check if a stock is Shariah compliant based on available data."""
    violations = []

    sector = str(row.get('sector', '')).lower()
    industry = str(row.get('industry', '')).lower()

    # Sector exclusions
    haram_sectors = ['financial services', 'banks', 'insurance', 'alcohol',
                     'gambling', 'tobacco', 'adult entertainment', 'weapons']

    for haram in haram_sectors:
        if haram in sector or haram in industry:
            violations.append(f"Excluded sector: {sector}")
            break

    # Financial ratios (simplified)
    debt_to_equity = row.get('debt_to_equity')
    if debt_to_equity and debt_to_equity > 33:
        violations.append(f"Debt/Equity {debt_to_equity:.1f}% > 33%")

    is_compliant = len(violations) == 0
    return is_compliant, violations