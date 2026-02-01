"""
Earnings Reaction Analyzer V2 - Enhanced Post-Earnings Analysis

This module extends the existing Earnings Intelligence System (IES/EQS/ECS) 
with QUANTITATIVE analysis to answer:

1. WHY did the stock move? (Extract from actual news, not assumptions)
2. Is the reaction OVERSOLD or JUSTIFIED? (Quantitative analysis)
3. What should you DO? (Specific recommendation with price levels)

QUANTITATIVE FACTORS:
- Implied move vs actual move (options were pricing X%, got Y%)
- Historical earnings reactions (stock typically moves Z% on beats)
- Sector comparison (sector moved X%, stock moved Y%)
- Technical levels (RSI, support/resistance)
- ECS clearance (beat expectations vs priced-in expectations)

Usage:
    from src.analytics.earnings_intelligence.reaction_analyzer import analyze_post_earnings

    result = analyze_post_earnings("NKE")
    print(result.get_summary())

Author: Alpha Research Platform
"""

import re
import json
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class ReactionAssessment(Enum):
    """Assessment of whether price reaction is appropriate."""
    STRONGLY_OVERSOLD = "STRONGLY_OVERSOLD"  # Major overreaction, strong buy
    OVERSOLD = "OVERSOLD"                     # Overreaction, potential buy
    SLIGHTLY_OVERSOLD = "SLIGHTLY_OVERSOLD"   # Minor overreaction
    JUSTIFIED = "JUSTIFIED"                   # Reaction matches news
    SLIGHTLY_OVERBOUGHT = "SLIGHTLY_OVERBOUGHT"
    OVERBOUGHT = "OVERBOUGHT"
    STRONGLY_OVERBOUGHT = "STRONGLY_OVERBOUGHT"
    UNCERTAIN = "UNCERTAIN"


class Recommendation(Enum):
    """Actionable recommendation."""
    STRONG_BUY = "STRONG_BUY"      # Heavily oversold, high conviction
    BUY_DIP = "BUY_DIP"            # Good entry point
    NIBBLE = "NIBBLE"              # Small position, wait for confirmation
    WAIT = "WAIT"                  # Wait for stabilization
    AVOID = "AVOID"                # Stay away, more downside
    SELL = "SELL"                  # If long, exit
    TAKE_PROFITS = "TAKE_PROFITS"  # If long, reduce position


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class QuantitativeMetrics:
    """Quantitative metrics for reaction analysis."""

    # Options-based
    implied_move_pct: Optional[float] = None      # What options priced
    actual_move_pct: Optional[float] = None       # What happened
    move_surprise_pct: Optional[float] = None     # actual - implied
    move_vs_implied_ratio: Optional[float] = None # actual / implied

    # Historical comparison
    avg_earnings_reaction: Optional[float] = None  # Historical avg move
    max_earnings_drop: Optional[float] = None      # Worst historical drop
    reaction_percentile: Optional[float] = None    # Where this ranks

    # Sector comparison
    sector_move_pct: Optional[float] = None        # How sector moved
    relative_to_sector: Optional[float] = None     # Stock vs sector

    # Technical
    rsi_14: Optional[float] = None
    distance_to_support_pct: Optional[float] = None
    distance_to_52w_low_pct: Optional[float] = None

    # ECS integration
    ecs_category: Optional[str] = None
    clearance_margin: Optional[float] = None

    def get_oversold_score(self) -> float:
        """
        Calculate oversold score (0-100).
        Higher = more oversold = better buy opportunity
        """
        score = 50.0  # Start neutral

        # Move vs implied: If dropped more than expected, oversold
        if self.move_vs_implied_ratio is not None and self.actual_move_pct and self.actual_move_pct < 0:
            if self.move_vs_implied_ratio > 1.5:  # Dropped 50%+ more than implied
                score += 15
            elif self.move_vs_implied_ratio > 1.2:
                score += 10
            elif self.move_vs_implied_ratio < 0.8:
                score -= 10  # Dropped less than expected

        # Historical: If this is a historically large drop
        if self.reaction_percentile is not None:
            if self.reaction_percentile > 90:  # Top 10% worst drops
                score += 15
            elif self.reaction_percentile > 75:
                score += 10
            elif self.reaction_percentile < 25:  # Mild drop historically
                score -= 10

        # RSI: If oversold technically
        if self.rsi_14 is not None:
            if self.rsi_14 < 30:
                score += 15
            elif self.rsi_14 < 40:
                score += 8
            elif self.rsi_14 > 70:
                score -= 10

        # Near support/52w low
        if self.distance_to_52w_low_pct is not None:
            if self.distance_to_52w_low_pct < 5:  # Within 5% of 52w low
                score += 10
            elif self.distance_to_52w_low_pct < 10:
                score += 5

        # Sector: If stock dropped more than sector
        if self.relative_to_sector is not None:
            if self.relative_to_sector < -5:  # 5%+ underperformance
                score += 5

        return max(0, min(100, score))


@dataclass
class ReactionAnalysis:
    """Complete post-earnings reaction analysis with quantitative support."""

    ticker: str
    analysis_date: date = field(default_factory=date.today)

    # From ECS/EQS (existing system)
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    eps_surprise_pct: Optional[float] = None
    eps_beat: Optional[bool] = None  # None = unknown, True = beat, False = miss
    revenue_surprise_pct: Optional[float] = None
    eqs: Optional[float] = None
    ecs_category: Optional[str] = None
    ies: Optional[float] = None  # Pre-earnings expectations

    # Price reaction
    price_before: Optional[float] = None
    price_after: Optional[float] = None
    price_current: Optional[float] = None
    reaction_pct: Optional[float] = None

    # Quantitative analysis
    quant_metrics: QuantitativeMetrics = field(default_factory=QuantitativeMetrics)
    oversold_score: float = 50.0  # 0-100, higher = more oversold

    # AI-extracted analysis
    drop_reasons: List[str] = field(default_factory=list)
    primary_reason: str = "Unknown"
    sentiment_summary: str = ""

    # Assessment with degree
    reaction_assessment: ReactionAssessment = ReactionAssessment.UNCERTAIN
    confidence: float = 50.0

    # Specific recommendation
    recommendation: Recommendation = Recommendation.WAIT
    recommendation_reason: str = ""

    # Price targets
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    risk_reward_ratio: Optional[float] = None

    # Position sizing
    suggested_position_pct: float = 0.0  # % of normal position

    # Supporting data
    key_headlines: List[str] = field(default_factory=list)
    analyst_actions: List[str] = field(default_factory=list)

    # Metadata
    data_quality: str = "MEDIUM"
    computed_at: datetime = field(default_factory=datetime.now)

    @property
    def earnings_result(self) -> str:
        """Get earnings result as string (BEAT/MISS/INLINE/N/A).

        Priority order:
        1. eps_beat flag (set from actual EPS vs estimate comparison)
        2. eps_surprise_pct (calculated surprise percentage)
        3. ECS category (but only for STRONG_BEAT/STRONG_MISS qualifiers)
        """
        # Method 1: Use explicit eps_beat flag (most reliable - from actual vs estimate)
        if self.eps_beat is not None:
            return "BEAT" if self.eps_beat else "MISS"

        # Method 2: Calculate from surprise percentage
        if self.eps_surprise_pct is not None:
            if self.eps_surprise_pct > 1:  # >1% beat
                return "BEAT"
            elif self.eps_surprise_pct < -1:  # >1% miss
                return "MISS"
            else:
                return "INLINE"

        # Method 3: Calculate from actual vs estimate
        if self.eps_actual is not None and self.eps_estimate is not None:
            if self.eps_actual > self.eps_estimate * 1.01:  # >1% beat
                return "BEAT"
            elif self.eps_actual < self.eps_estimate * 0.99:  # >1% miss
                return "MISS"
            else:
                return "INLINE"

        # Method 4: Use ECS category only as last resort (it measures "cleared expectations" not beat/miss)
        if self.ecs_category:
            # Only use for STRONG qualifiers, otherwise it can be misleading
            if 'STRONG_BEAT' in self.ecs_category.upper():
                return "BEAT"
            elif 'STRONG_MISS' in self.ecs_category.upper():
                return "MISS"

        return "N/A"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'ticker': self.ticker,
            'analysis_date': self.analysis_date.isoformat() if self.analysis_date else None,
            'eps_actual': self.eps_actual,
            'eps_estimate': self.eps_estimate,
            'eps_surprise_pct': self.eps_surprise_pct,
            'eps_beat': self.eps_beat,
            'revenue_surprise_pct': self.revenue_surprise_pct,
            'eqs': self.eqs,
            'ecs_category': self.ecs_category,
            'ies': self.ies,
            'price_before': self.price_before,
            'price_after': self.price_after,
            'price_current': self.price_current,
            'reaction_pct': self.reaction_pct,
            'oversold_score': self.oversold_score,
            'drop_reasons': self.drop_reasons,
            'primary_reason': self.primary_reason,
            'reaction_assessment': self.reaction_assessment.value,
            'confidence': self.confidence,
            'recommendation': self.recommendation.value,
            'recommendation_reason': self.recommendation_reason,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'target_price': self.target_price,
            'risk_reward_ratio': self.risk_reward_ratio,
            'suggested_position_pct': self.suggested_position_pct,
            'key_headlines': self.key_headlines[:5],
            'data_quality': self.data_quality,
            # Quant metrics
            'implied_move_pct': self.quant_metrics.implied_move_pct,
            'move_vs_implied_ratio': self.quant_metrics.move_vs_implied_ratio,
            'rsi_14': self.quant_metrics.rsi_14,
            'sector_move_pct': self.quant_metrics.sector_move_pct,
        }

    def get_summary(self) -> str:
        """Generate formatted summary."""
        lines = [
            f"\n{'='*70}",
            f"POST-EARNINGS REACTION ANALYSIS: {self.ticker}",
            f"{'='*70}",
        ]

        # Earnings results - use earnings_result property for consistency
        result_str = self.earnings_result
        if result_str == "BEAT":
            beat_miss = "✅ BEAT"
        elif result_str == "MISS":
            beat_miss = "❌ MISSED"
        elif result_str == "INLINE":
            beat_miss = "➡️ INLINE"
        else:
            beat_miss = "❓ N/A"
        lines.append(f"\n📊 EARNINGS: {beat_miss}")
        if self.eps_actual is not None and self.eps_estimate is not None:
            surprise_pct = self.eps_surprise_pct if self.eps_surprise_pct is not None else 0
            lines.append(f"   EPS: ${self.eps_actual:.2f} vs ${self.eps_estimate:.2f} ({surprise_pct:+.1f}%)")
        if self.revenue_surprise_pct is not None:
            lines.append(f"   Revenue Surprise: {self.revenue_surprise_pct:+.1f}%")
        if self.eqs:
            lines.append(f"   Earnings Quality Score: {self.eqs:.0f}/100")
        if self.ecs_category:
            lines.append(f"   Expectations Clearance: {self.ecs_category}")
        if self.ies:
            lines.append(f"   Pre-Earnings IES: {self.ies:.0f}/100")

        # Price reaction
        if self.reaction_pct is not None:
            direction = "📉 DOWN" if self.reaction_pct < 0 else "📈 UP"
            lines.append(f"\n💰 PRICE REACTION: {direction} {abs(self.reaction_pct):.1f}%")
            if self.price_before and self.price_current:
                lines.append(f"   ${self.price_before:.2f} → ${self.price_current:.2f}")

        # Quantitative analysis
        lines.append(f"\n📐 QUANTITATIVE ANALYSIS:")
        qm = self.quant_metrics

        if qm.implied_move_pct is not None and qm.actual_move_pct is not None:
            implied_str = f"{qm.implied_move_pct:+.1f}%" if qm.implied_move_pct else "N/A"
            actual_str = f"{qm.actual_move_pct:+.1f}%"
            ratio_str = f"{qm.move_vs_implied_ratio:.2f}x" if qm.move_vs_implied_ratio else ""

            if qm.move_vs_implied_ratio and qm.move_vs_implied_ratio > 1.2:
                verdict = "⚠️ MOVED MORE THAN EXPECTED"
            elif qm.move_vs_implied_ratio and qm.move_vs_implied_ratio < 0.8:
                verdict = "✅ Moved less than expected"
            else:
                verdict = "≈ In line with expectations"

            lines.append(f"   Options implied: ±{abs(qm.implied_move_pct):.1f}% | Actual: {actual_str} ({ratio_str})")
            lines.append(f"   {verdict}")

        if qm.avg_earnings_reaction is not None:
            lines.append(f"   Historical avg reaction: {qm.avg_earnings_reaction:+.1f}%")
            if qm.reaction_percentile:
                lines.append(f"   This reaction percentile: {qm.reaction_percentile:.0f}th (vs history)")

        if qm.rsi_14 is not None:
            rsi_status = "🔴 OVERSOLD" if qm.rsi_14 < 30 else "🟢 OVERBOUGHT" if qm.rsi_14 > 70 else "neutral"
            lines.append(f"   RSI(14): {qm.rsi_14:.0f} ({rsi_status})")

        if qm.sector_move_pct is not None and qm.relative_to_sector is not None:
            lines.append(f"   Sector moved: {qm.sector_move_pct:+.1f}% | Stock vs sector: {qm.relative_to_sector:+.1f}%")

        if qm.distance_to_52w_low_pct is not None:
            lines.append(f"   Distance to 52w low: {qm.distance_to_52w_low_pct:.1f}%")

        # Oversold score
        lines.append(f"\n   📊 OVERSOLD SCORE: {self.oversold_score:.0f}/100")
        if self.oversold_score >= 70:
            lines.append(f"   → Strong oversold signal")
        elif self.oversold_score >= 60:
            lines.append(f"   → Moderately oversold")
        elif self.oversold_score <= 40:
            lines.append(f"   → Not oversold, reaction seems fair")

        # Why it moved
        lines.append(f"\n🔍 WHY DID IT MOVE?")
        lines.append(f"   Primary: {self.primary_reason}")
        if self.drop_reasons:
            lines.append("   All Factors:")
            for reason in self.drop_reasons[:5]:
                lines.append(f"   • {reason}")

        # Assessment
        emoji_map = {
            ReactionAssessment.STRONGLY_OVERSOLD: "🟢🟢",
            ReactionAssessment.OVERSOLD: "🟢",
            ReactionAssessment.SLIGHTLY_OVERSOLD: "🟢",
            ReactionAssessment.JUSTIFIED: "🟡",
            ReactionAssessment.SLIGHTLY_OVERBOUGHT: "🟠",
            ReactionAssessment.OVERBOUGHT: "🔴",
            ReactionAssessment.STRONGLY_OVERBOUGHT: "🔴🔴",
            ReactionAssessment.UNCERTAIN: "⚪",
        }
        emoji = emoji_map.get(self.reaction_assessment, "⚪")
        lines.append(f"\n📋 ASSESSMENT: {emoji} {self.reaction_assessment.value}")
        lines.append(f"   Confidence: {self.confidence:.0f}%")

        # Recommendation with details
        rec_emoji = {
            Recommendation.STRONG_BUY: "🟢🟢",
            Recommendation.BUY_DIP: "🟢",
            Recommendation.NIBBLE: "🟢",
            Recommendation.WAIT: "🟡",
            Recommendation.AVOID: "🔴",
            Recommendation.SELL: "🔴🔴",
            Recommendation.TAKE_PROFITS: "🟠",
        }
        lines.append(f"\n🎯 RECOMMENDATION: {rec_emoji.get(self.recommendation, '⚪')} {self.recommendation.value}")
        lines.append(f"   {self.recommendation_reason}")

        # Price targets if available
        if self.entry_price or self.stop_loss or self.target_price:
            lines.append(f"\n💵 LEVELS:")
            if self.entry_price:
                lines.append(f"   Entry:  ${self.entry_price:.2f}")
            if self.stop_loss:
                lines.append(f"   Stop:   ${self.stop_loss:.2f}")
            if self.target_price:
                lines.append(f"   Target: ${self.target_price:.2f}")
            if self.risk_reward_ratio:
                lines.append(f"   R/R:    {self.risk_reward_ratio:.1f}:1")

        if self.suggested_position_pct > 0:
            lines.append(f"\n   Suggested position: {self.suggested_position_pct:.0f}% of normal size")

        # Key headlines
        if self.key_headlines:
            lines.append(f"\n📰 KEY HEADLINES:")
            for h in self.key_headlines[:3]:
                lines.append(f"   • {h[:65]}...")

        lines.append(f"\n{'='*70}")

        return "\n".join(lines)

    def get_deep_dive_dict(self) -> Dict[str, Any]:
        """Get data formatted for Deep Dive integration."""
        return {
            'ticker': self.ticker,
            'section': 'post_earnings_reaction',

            # Summary
            'headline': f"{self.recommendation.value}: {self.primary_reason}",
            'assessment': self.reaction_assessment.value,
            'recommendation': self.recommendation.value,
            'confidence': self.confidence,

            # Earnings
            'eps_beat': self.eps_beat,
            'eps_surprise_pct': self.eps_surprise_pct,
            'ecs_category': self.ecs_category,

            # Reaction
            'reaction_pct': self.reaction_pct,
            'oversold_score': self.oversold_score,

            # Quant
            'implied_move_pct': self.quant_metrics.implied_move_pct,
            'actual_vs_implied': self.quant_metrics.move_vs_implied_ratio,
            'rsi': self.quant_metrics.rsi_14,

            # Reasons
            'primary_reason': self.primary_reason,
            'all_reasons': self.drop_reasons,

            # Recommendation details
            'recommendation_reason': self.recommendation_reason,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'target_price': self.target_price,
            'position_size_pct': self.suggested_position_pct,

            # Headlines
            'key_headlines': self.key_headlines[:5],
        }


# ============================================================================
# MAIN ANALYZER CLASS
# ============================================================================

class EarningsReactionAnalyzer:
    """
    Enhanced post-earnings reaction analyzer with quantitative analysis.
    """

    def __init__(self):
        self.llm_client = None
        self._init_llm()

    def _init_llm(self):
        """Initialize LLM client for analysis."""
        import os
        from dotenv import load_dotenv
        load_dotenv()

        try:
            from openai import OpenAI
            base_url = os.getenv("LLM_QWEN_BASE_URL", "http://172.23.193.91:8090/v1")

            self.llm_client = OpenAI(
                base_url=base_url,
                api_key="not-needed",
                timeout=120
            )
            self.llm_client.models.list()
            logger.info(f"Reaction Analyzer LLM connected: {base_url}")
        except Exception as e:
            logger.warning(f"LLM not available for reaction analysis: {e}")

    def analyze(self, ticker: str, earnings_date: Optional[date] = None) -> ReactionAnalysis:
        """
        Complete post-earnings reaction analysis with quantitative support.
        """
        logger.info(f"{ticker}: Starting enhanced post-earnings reaction analysis")

        result = ReactionAnalysis(ticker=ticker)

        # 1. Get earnings data from existing system
        self._load_earnings_data(result, earnings_date)

        # DEBUG: Log what we got
        logger.info(f"{ticker}: After _load_earnings_data -> eps_beat={result.eps_beat}, eps_actual={result.eps_actual}, eps_estimate={result.eps_estimate}, eps_surprise_pct={result.eps_surprise_pct}")

        # 2. Get price reaction
        self._load_price_reaction(result)

        # DEBUG: Log after price reaction (may infer beat/miss)
        logger.info(f"{ticker}: After _load_price_reaction -> eps_beat={result.eps_beat}, reaction_pct={result.reaction_pct}")

        # 3. Get quantitative metrics
        self._calculate_quant_metrics(result)

        # 4. Get news headlines
        headlines = self._get_headlines(ticker)
        result.key_headlines = headlines[:10]

        # 5. AI analysis - extract reasons
        if self.llm_client and headlines:
            self._ai_analyze(result, headlines)
        else:
            self._extract_reasons_from_headlines(result, headlines)

        # 6. Make quantitative assessment
        self._make_assessment(result)

        # 7. Generate specific recommendation
        self._generate_recommendation(result)

        # 8. Calculate price targets
        self._calculate_price_targets(result)

        logger.info(f"{ticker}: Analysis complete - {result.recommendation.value} ({result.confidence:.0f}%)")

        return result

    def _load_earnings_data(self, result: ReactionAnalysis, earnings_date: Optional[date] = None):
        """Load earnings data from existing ECS/EQS/IES system."""
        ecs_loaded = False

        try:
            from src.analytics.earnings_intelligence.ecs_calculator import calculate_ecs
            from src.analytics.earnings_intelligence.ies_calculator import calculate_ies

            # Get ECS
            ecs_result = calculate_ecs(result.ticker, earnings_date)
            result.ecs_category = ecs_result.ecs_category.value if ecs_result.ecs_category else None
            result.eqs = ecs_result.eqs
            result.eps_surprise_pct = ecs_result.eps_surprise_pct
            result.revenue_surprise_pct = ecs_result.revenue_surprise_pct

            # Set eps_beat based on ECS category or surprise percentage
            if result.ecs_category:
                if 'BEAT' in result.ecs_category.upper():
                    result.eps_beat = True
                    logger.info(f"{result.ticker}: ECS category {result.ecs_category} -> BEAT")
                elif 'MISS' in result.ecs_category.upper():
                    result.eps_beat = False
                    logger.info(f"{result.ticker}: ECS category {result.ecs_category} -> MISS")
                elif result.eps_surprise_pct is not None:
                    result.eps_beat = result.eps_surprise_pct > 0
                    logger.info(f"{result.ticker}: EPS surprise {result.eps_surprise_pct:.1f}% -> {'BEAT' if result.eps_beat else 'MISS'}")
            elif result.eps_surprise_pct is not None:
                result.eps_beat = result.eps_surprise_pct > 0
                logger.info(f"{result.ticker}: EPS surprise {result.eps_surprise_pct:.1f}% -> {'BEAT' if result.eps_beat else 'MISS'}")

            # Also get actual/estimate from ECS if available
            if hasattr(ecs_result, 'eps_actual') and ecs_result.eps_actual:
                result.eps_actual = ecs_result.eps_actual
            if hasattr(ecs_result, 'eps_estimate') and ecs_result.eps_estimate:
                result.eps_estimate = ecs_result.eps_estimate

            # Store clearance margin for quant analysis
            result.quant_metrics.ecs_category = result.ecs_category
            result.quant_metrics.clearance_margin = ecs_result.clearance_margin

            # Get IES (pre-earnings expectations)
            ies_result = calculate_ies(result.ticker)
            result.ies = ies_result.ies
            result.quant_metrics.implied_move_pct = ies_result.implied_move_pct

            logger.debug(f"{result.ticker}: Loaded ECS/IES data")
            ecs_loaded = True

        except ImportError:
            logger.debug(f"{result.ticker}: ECS/IES not available, using yfinance")
        except Exception as e:
            logger.warning(f"{result.ticker}: Error loading ECS/IES: {e}")

        # ALWAYS try yfinance to get accurate Surprise(%) data
        # This will override eps_beat if yfinance has better data
        self._load_earnings_from_yfinance(result, override_if_better=True)

    def _load_earnings_from_yfinance(self, result: ReactionAnalysis, override_if_better: bool = False):
        """Load/verify earnings from yfinance with improved BEAT/MISS detection.

        Args:
            result: ReactionAnalysis object to populate
            override_if_better: If True, override existing eps_beat if yfinance has Surprise(%) column
        """
        try:
            import yfinance as yf
            import pandas as pd

            stock = yf.Ticker(result.ticker)

            ed = stock.earnings_dates
            if ed is not None and not ed.empty:
                logger.debug(f"{result.ticker}: yfinance earnings_dates columns: {ed.columns.tolist()}")

                for idx in ed.index:
                    d = idx.date() if hasattr(idx, 'date') else idx
                    if d <= date.today():
                        row = ed.loc[idx]

                        # Get EPS values - handle NaN properly
                        eps_actual = row.get('Reported EPS')
                        eps_estimate = row.get('EPS Estimate')
                        surprise_pct = row.get('Surprise(%)')

                        # Convert to float and check for NaN
                        if pd.notna(eps_actual):
                            yf_eps_actual = float(eps_actual)
                            if result.eps_actual is None or override_if_better:
                                result.eps_actual = yf_eps_actual
                        if pd.notna(eps_estimate):
                            yf_eps_estimate = float(eps_estimate)
                            if result.eps_estimate is None or override_if_better:
                                result.eps_estimate = yf_eps_estimate

                        # Method 1: Use Surprise(%) column directly if available (MOST RELIABLE)
                        # This is the authoritative source from Yahoo Finance
                        if pd.notna(surprise_pct):
                            yf_surprise = float(surprise_pct)
                            yf_beat = yf_surprise > 0

                            # Always use Surprise(%) if available - it's the most reliable
                            if override_if_better or result.eps_beat is None:
                                result.eps_surprise_pct = yf_surprise
                                result.eps_beat = yf_beat
                                logger.info(f"{result.ticker}: yfinance Surprise(%) = {yf_surprise:.2f}% -> {'BEAT' if yf_beat else 'MISS'}")
                            break

                        # Method 2: Calculate from actual vs estimate
                        if pd.notna(eps_actual) and pd.notna(eps_estimate):
                            yf_eps_actual = float(eps_actual)
                            yf_eps_estimate = float(eps_estimate)

                            if abs(yf_eps_estimate) > 0.001:  # Avoid division by zero
                                yf_surprise = ((yf_eps_actual - yf_eps_estimate) / abs(yf_eps_estimate)) * 100
                                yf_beat = yf_eps_actual > yf_eps_estimate

                                if result.eps_beat is None:  # Only set if not already set
                                    result.eps_surprise_pct = yf_surprise
                                    result.eps_beat = yf_beat
                                    logger.info(f"{result.ticker}: yfinance calculated surprise = {yf_surprise:.2f}% -> {'BEAT' if yf_beat else 'MISS'}")
                            else:
                                # If estimate is ~0, use actual sign
                                if result.eps_beat is None:
                                    result.eps_beat = float(eps_actual) > 0
                                    logger.info(f"{result.ticker}: yfinance estimate near zero, using actual sign -> {'BEAT' if result.eps_beat else 'MISS'}")
                            break

                        # If we found a row but couldn't determine beat/miss, continue to next row
                        logger.debug(f"{result.ticker}: Row {d} has incomplete data, trying next")
                        continue

                # Method 3: If still no beat/miss determined, try earnings_history
                if result.eps_beat is None:
                    try:
                        hist = stock.earnings_history
                        if hist is not None and not hist.empty:
                            latest = hist.iloc[0]
                            eps_actual = latest.get('epsActual')
                            eps_estimate = latest.get('epsEstimate')

                            if pd.notna(eps_actual) and pd.notna(eps_estimate):
                                result.eps_actual = float(eps_actual)
                                result.eps_estimate = float(eps_estimate)
                                if abs(result.eps_estimate) > 0.001:
                                    result.eps_surprise_pct = ((result.eps_actual - result.eps_estimate) / abs(result.eps_estimate)) * 100
                                result.eps_beat = result.eps_actual > result.eps_estimate
                                logger.info(f"{result.ticker}: From yfinance earnings_history -> {'BEAT' if result.eps_beat else 'MISS'}")
                    except Exception as e:
                        logger.debug(f"{result.ticker}: earnings_history error: {e}")

        except Exception as e:
            logger.warning(f"{result.ticker}: Error loading yfinance earnings: {e}")

        # Method 4: Fallback - infer from price reaction if we have it
        # A stock up >5% after earnings likely beat; down >5% likely missed
        if result.eps_beat is None and result.reaction_pct is not None:
            if result.reaction_pct > 5:
                result.eps_beat = True
                logger.info(f"{result.ticker}: Inferred BEAT from +{result.reaction_pct:.1f}% reaction")
            elif result.reaction_pct < -5:
                result.eps_beat = False
                logger.info(f"{result.ticker}: Inferred MISS from {result.reaction_pct:.1f}% reaction")

    def _load_price_reaction(self, result: ReactionAnalysis):
        """Load price reaction data."""
        try:
            import yfinance as yf

            stock = yf.Ticker(result.ticker)
            info = stock.info
            hist = stock.history(period="10d")

            if len(hist) >= 2:
                result.price_before = float(hist['Close'].iloc[-2])
                result.price_after = float(hist['Close'].iloc[-1])
                result.price_current = result.price_after
                result.reaction_pct = ((result.price_after - result.price_before) / result.price_before) * 100

            # Check pre-market
            pre = info.get('preMarketPrice')
            if pre:
                result.price_current = float(pre)
                prev_close = info.get('previousClose', result.price_before)
                if prev_close:
                    result.reaction_pct = ((pre - prev_close) / prev_close) * 100

            result.quant_metrics.actual_move_pct = result.reaction_pct

        except Exception as e:
            logger.warning(f"{result.ticker}: Error loading price reaction: {e}")

    def _calculate_quant_metrics(self, result: ReactionAnalysis):
        """Calculate quantitative metrics for oversold/overbought assessment."""
        try:
            import yfinance as yf

            stock = yf.Ticker(result.ticker)
            hist = stock.history(period="1y")
            info = stock.info

            qm = result.quant_metrics

            # 1. Move vs implied
            if qm.implied_move_pct and qm.actual_move_pct:
                qm.move_surprise_pct = abs(qm.actual_move_pct) - qm.implied_move_pct
                if qm.implied_move_pct > 0:
                    qm.move_vs_implied_ratio = abs(qm.actual_move_pct) / qm.implied_move_pct

            # 2. Historical earnings reactions (estimate from volatility around earnings)
            if len(hist) > 60:
                # Calculate daily returns
                returns = hist['Close'].pct_change().dropna() * 100

                # Get larger moves (likely earnings-related)
                big_moves = returns[abs(returns) > returns.std() * 2]
                if len(big_moves) > 0:
                    qm.avg_earnings_reaction = abs(big_moves).mean()

                    # Calculate percentile of current reaction
                    if qm.actual_move_pct:
                        qm.reaction_percentile = (abs(returns) < abs(qm.actual_move_pct)).mean() * 100

            # 3. RSI calculation
            if len(hist) > 14:
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                qm.rsi_14 = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else None

            # 4. Distance to 52-week low
            week_52_low = info.get('fiftyTwoWeekLow')
            current = result.price_current or result.price_after
            if week_52_low and current:
                qm.distance_to_52w_low_pct = ((current - week_52_low) / week_52_low) * 100

            # 5. Sector comparison
            sector = info.get('sector', '')
            sector_etf = self._get_sector_etf(sector)
            if sector_etf:
                try:
                    etf = yf.Ticker(sector_etf)
                    etf_hist = etf.history(period="5d")
                    if len(etf_hist) >= 2:
                        etf_return = ((etf_hist['Close'].iloc[-1] - etf_hist['Close'].iloc[-2]) / etf_hist['Close'].iloc[-2]) * 100
                        qm.sector_move_pct = etf_return
                        if qm.actual_move_pct is not None:
                            qm.relative_to_sector = qm.actual_move_pct - etf_return
                except:
                    pass

            # Calculate oversold score
            result.oversold_score = qm.get_oversold_score()

        except Exception as e:
            logger.warning(f"{result.ticker}: Error calculating quant metrics: {e}")

    def _get_sector_etf(self, sector: str) -> Optional[str]:
        """Get sector ETF for comparison."""
        sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financials': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Consumer Cyclical': 'XLY',
            'Consumer Staples': 'XLP',
            'Consumer Defensive': 'XLP',
            'Energy': 'XLE',
            'Industrials': 'XLI',
            'Materials': 'XLB',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Communication Services': 'XLC',
        }
        return sector_etfs.get(sector, 'SPY')

    def _get_headlines(self, ticker: str) -> List[str]:
        """Get recent news headlines from database."""
        try:
            from src.db.connection import get_engine
            import pandas as pd

            engine = get_engine()

            df = pd.read_sql(f"""
                SELECT headline, source, published_at, relevance_score
                FROM news_articles 
                WHERE ticker = '{ticker}'
                AND headline IS NOT NULL AND headline != ''
                ORDER BY COALESCE(published_at, created_at) DESC
                LIMIT 30
            """, engine)

            if df.empty:
                return []

            # Filter out junk
            junk = ['stock price', 'stock quote', 'google finance', 'yahoo finance quote']
            df = df[~df['headline'].str.lower().str.contains('|'.join(junk), na=False)]

            return df['headline'].tolist()

        except Exception as e:
            logger.warning(f"{ticker}: Error loading headlines: {e}")
            return []

    def _ai_analyze(self, result: ReactionAnalysis, headlines: List[str]):
        """Use AI to extract reasons from headlines."""

        headlines_text = "\n".join([f"- {h}" for h in headlines[:20]])

        eps_beat = "BEAT" if result.eps_beat else "MISSED"
        reaction = result.reaction_pct or 0
        eps_surprise = result.eps_surprise_pct if result.eps_surprise_pct is not None else 0

        prompt = f"""/no_think
Analyze this post-earnings stock reaction for {result.ticker}:

EARNINGS: EPS {eps_beat} ({eps_surprise:+.1f}% surprise)
STOCK REACTION: {reaction:+.1f}%
EXPECTATIONS CLEARANCE: {result.ecs_category or 'N/A'}

NEWS HEADLINES:
{headlines_text}

Extract the SPECIFIC reasons for the stock move from the headlines.
Look for: guidance, revenue issues, regional problems (China, Europe), margins, competition, inventory, macro, analyst actions.

Return ONLY JSON:
{{
    "reasons": ["reason 1", "reason 2", ...],
    "primary_reason": "main driver in one sentence",
    "sentiment_summary": "overall tone",
    "analyst_actions": ["upgrades/downgrades if any"]
}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen3-32B-Q6_K.gguf",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800
            )

            result_text = response.choices[0].message.content.strip()

            if '</think>' in result_text:
                result_text = result_text.split('</think>')[-1].strip()

            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(0))
                result.drop_reasons = analysis.get('reasons', [])
                result.primary_reason = analysis.get('primary_reason', 'Unknown')
                result.sentiment_summary = analysis.get('sentiment_summary', '')
                result.analyst_actions = analysis.get('analyst_actions', [])

        except Exception as e:
            logger.error(f"{result.ticker}: AI analysis error: {e}")
            self._extract_reasons_from_headlines(result, headlines)

    def _extract_reasons_from_headlines(self, result: ReactionAnalysis, headlines: List[str]):
        """Simple keyword extraction from headlines."""
        reasons = []

        keywords = {
            'china': 'China weakness',
            'guidance': 'Guidance concerns',
            'margin': 'Margin pressure',
            'inventory': 'Inventory issues',
            'revenue': 'Revenue concerns',
            'demand': 'Demand weakness',
            'competition': 'Competitive pressure',
            'tariff': 'Tariff impact',
            'downgrade': 'Analyst downgrade',
            'upgrade': 'Analyst upgrade',
            'miss': 'Missed expectations',
            'beat': 'Beat expectations',
            'weak': 'Weak outlook',
        }

        for headline in headlines[:10]:
            hl_lower = headline.lower()
            for keyword, reason in keywords.items():
                if keyword in hl_lower and reason not in reasons:
                    reasons.append(reason)

        result.drop_reasons = reasons[:5]
        result.primary_reason = reasons[0] if reasons else "Market reaction to earnings"

    def _make_assessment(self, result: ReactionAnalysis):
        """Make quantitative assessment of reaction appropriateness."""

        score = result.oversold_score
        reaction = result.reaction_pct or 0
        eps_beat = result.eps_beat
        ecs = result.ecs_category

        # Determine assessment based on multiple factors
        if reaction < 0:  # Stock dropped
            if score >= 75:
                result.reaction_assessment = ReactionAssessment.STRONGLY_OVERSOLD
                result.confidence = min(90, 50 + score/2)
            elif score >= 65:
                result.reaction_assessment = ReactionAssessment.OVERSOLD
                result.confidence = min(80, 45 + score/2)
            elif score >= 55:
                result.reaction_assessment = ReactionAssessment.SLIGHTLY_OVERSOLD
                result.confidence = min(70, 40 + score/2)
            elif score <= 40:
                result.reaction_assessment = ReactionAssessment.JUSTIFIED
                result.confidence = 60
            else:
                result.reaction_assessment = ReactionAssessment.UNCERTAIN
                result.confidence = 50
        else:  # Stock rose
            if score <= 30:
                result.reaction_assessment = ReactionAssessment.OVERBOUGHT
                result.confidence = 60
            elif score >= 60:
                result.reaction_assessment = ReactionAssessment.JUSTIFIED
                result.confidence = 65
            else:
                result.reaction_assessment = ReactionAssessment.UNCERTAIN
                result.confidence = 50

        # Adjust for ECS
        if ecs == 'STRONG_MISS' and result.reaction_assessment in [ReactionAssessment.OVERSOLD, ReactionAssessment.STRONGLY_OVERSOLD]:
            # If truly missed expectations, maybe not oversold
            result.confidence -= 15
        elif ecs == 'BEAT' and reaction < -5:
            # Beat but dropped a lot - likely oversold
            result.confidence += 10

    def _generate_recommendation(self, result: ReactionAnalysis):
        """Generate specific recommendation based on assessment."""

        assessment = result.reaction_assessment
        score = result.oversold_score
        ecs = result.ecs_category
        reaction = result.reaction_pct or 0

        if assessment == ReactionAssessment.STRONGLY_OVERSOLD:
            if ecs in ['BEAT', 'STRONG_BEAT', 'INLINE']:
                result.recommendation = Recommendation.BUY_DIP
                result.recommendation_reason = f"Heavily oversold (score: {score:.0f}) despite clearing expectations. High probability of bounce."
                result.suggested_position_pct = 75
            else:
                result.recommendation = Recommendation.NIBBLE
                result.recommendation_reason = f"Oversold but missed expectations. Small position, wait for confirmation."
                result.suggested_position_pct = 30

        elif assessment == ReactionAssessment.OVERSOLD:
            if ecs in ['BEAT', 'STRONG_BEAT']:
                result.recommendation = Recommendation.BUY_DIP
                result.recommendation_reason = f"Oversold reaction to an earnings beat. Good entry opportunity."
                result.suggested_position_pct = 50
            else:
                result.recommendation = Recommendation.NIBBLE
                result.recommendation_reason = f"Moderately oversold. Consider small position with tight stop."
                result.suggested_position_pct = 25

        elif assessment == ReactionAssessment.SLIGHTLY_OVERSOLD:
            result.recommendation = Recommendation.WAIT
            result.recommendation_reason = f"Slightly oversold but not compelling. Wait for better entry or stabilization."
            result.suggested_position_pct = 0

        elif assessment == ReactionAssessment.JUSTIFIED:
            if reaction < 0:
                result.recommendation = Recommendation.AVOID
                result.recommendation_reason = f"Reaction appears justified given the news. Risk of further downside."
                result.suggested_position_pct = 0
            else:
                result.recommendation = Recommendation.WAIT
                result.recommendation_reason = f"Rally is justified but wait for pullback to enter."
                result.suggested_position_pct = 0

        elif assessment in [ReactionAssessment.OVERBOUGHT, ReactionAssessment.STRONGLY_OVERBOUGHT]:
            result.recommendation = Recommendation.TAKE_PROFITS
            result.recommendation_reason = f"Stock may be overbought after earnings. Consider taking profits if long."
            result.suggested_position_pct = 0

        else:  # UNCERTAIN
            result.recommendation = Recommendation.WAIT
            result.recommendation_reason = f"Mixed signals. Wait for more clarity before taking a position."
            result.suggested_position_pct = 0

    def _calculate_price_targets(self, result: ReactionAnalysis):
        """Calculate entry, stop, and target prices."""

        current = result.price_current or result.price_after
        if not current:
            return

        reaction = result.reaction_pct or 0

        if result.recommendation in [Recommendation.BUY_DIP, Recommendation.STRONG_BUY, Recommendation.NIBBLE]:
            # Entry: current price or slightly below
            result.entry_price = round(current * 0.99, 2)  # 1% below current

            # Stop: below recent low or 5-8% below entry
            if result.recommendation == Recommendation.NIBBLE:
                stop_pct = 0.08  # Tighter stop for nibble
            else:
                stop_pct = 0.05
            result.stop_loss = round(result.entry_price * (1 - stop_pct), 2)

            # Target: recover some/all of the drop
            if reaction < -10:
                # Big drop - target 50% recovery
                recovery = abs(reaction) * 0.5
            elif reaction < -5:
                # Moderate drop - target 70% recovery
                recovery = abs(reaction) * 0.7
            else:
                # Small drop - target full recovery
                recovery = abs(reaction)

            result.target_price = round(result.entry_price * (1 + recovery/100), 2)

            # Risk/reward
            if result.stop_loss and result.target_price:
                risk = result.entry_price - result.stop_loss
                reward = result.target_price - result.entry_price
                if risk > 0:
                    result.risk_reward_ratio = round(reward / risk, 2)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def analyze_post_earnings(ticker: str, earnings_date: Optional[date] = None) -> ReactionAnalysis:
    """Analyze post-earnings reaction for a ticker."""
    analyzer = EarningsReactionAnalyzer()
    return analyzer.analyze(ticker, earnings_date)


def get_reaction_summary(ticker: str) -> str:
    """Get formatted reaction summary for display."""
    result = analyze_post_earnings(ticker)
    return result.get_summary()


def get_reaction_for_deep_dive(ticker: str) -> Dict[str, Any]:
    """Get reaction analysis formatted for Deep Dive tab integration."""
    result = analyze_post_earnings(ticker)
    return result.get_deep_dive_dict()


def get_reaction_for_chat(ticker: str) -> Dict[str, Any]:
    """Get reaction analysis formatted for AI chat context."""
    result = analyze_post_earnings(ticker)
    return result.to_dict()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.analytics.earnings_intelligence.reaction_analyzer TICKER")
        print("       python -m src.analytics.earnings_intelligence.reaction_analyzer NKE")
        sys.exit(1)

    ticker = sys.argv[1].upper()

    print(f"\nAnalyzing post-earnings reaction for {ticker}...")
    result = analyze_post_earnings(ticker)
    print(result.get_summary())