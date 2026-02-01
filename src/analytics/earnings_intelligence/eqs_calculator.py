"""
EQS Calculator - Earnings Quality Score

Measures the QUALITY of actual earnings results AFTER the earnings report.
This is the post-earnings counterpart to IES (pre-earnings expectations).

EQS Components:
- eps_z: EPS surprise z-score (vs historical surprises)
- rev_z: Revenue surprise z-score
- guidance_score: Quality of forward guidance
- margin_score: Margin trend vs expectations
- tone_score: Management tone from call/transcript

EQS feeds into ECS (Expectations Clearance Score) which determines:
- Did results CLEAR the bar set by IES?
- What's the expected price reaction?

Usage:
    from src.analytics.earnings_intelligence import calculate_eqs

    result = calculate_eqs("NVDA", earnings_date=date(2024, 11, 20))
    print(f"EQS: {result.eqs}/100 - Event Z: {result.event_z}")

Author: Alpha Research Platform
Phase: 8 of Earnings Intelligence System
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import json
import os

from src.utils.logging import get_logger

# Import from models
from src.analytics.earnings_intelligence.models import (
    EQSComponents,
    ZScoreComponents,
    GuidanceDirection,
    DataQuality,
    EQS_WEIGHTS,
    EVENT_Z_WEIGHTS,
)

logger = get_logger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Database availability flag
try:
    from src.db.connection import get_connection, get_engine

    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.info("Database not available - using yfinance fallback for earnings data")

# Historical surprise statistics (for z-score calculation)
# These are market-wide averages - ideally would be ticker-specific
DEFAULT_EPS_SURPRISE_MEAN = 4.5  # Average EPS beat is ~4.5%
DEFAULT_EPS_SURPRISE_STD = 12.0  # Standard deviation
DEFAULT_REV_SURPRISE_MEAN = 1.5  # Average revenue beat is ~1.5%
DEFAULT_REV_SURPRISE_STD = 5.0  # Standard deviation


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EarningsData:
    """Raw earnings data from various sources."""
    ticker: str
    earnings_date: date

    # EPS
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    eps_surprise_pct: Optional[float] = None

    # Revenue
    revenue_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_surprise_pct: Optional[float] = None

    # Guidance
    guidance_direction: Optional[GuidanceDirection] = None
    guidance_summary: Optional[str] = None

    # Qualitative (from transcript analysis)
    management_tone: Optional[str] = None
    sentiment_score: Optional[int] = None
    key_highlights: Optional[List[str]] = None
    concerns: Optional[List[str]] = None

    # Margins
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    gross_margin_change: Optional[float] = None
    operating_margin_change: Optional[float] = None

    # Source tracking
    data_source: str = 'unknown'


@dataclass
class EQSCalculationResult:
    """Complete EQS calculation result."""

    # Core outputs
    ticker: str
    earnings_date: date
    eqs: Optional[float]  # Final EQS score (0-100)
    event_z: Optional[float]  # Blended event z-score

    # Z-score components
    eps_z: Optional[float]
    rev_z: Optional[float]
    guidance_z: Optional[float]

    # EQS component scores (all 0-100)
    eps_score: Optional[float]
    rev_score: Optional[float]
    guidance_score: Optional[float]
    margin_score: Optional[float]
    tone_score: Optional[float]

    # Raw values
    eps_surprise_pct: Optional[float]
    revenue_surprise_pct: Optional[float]
    guidance_direction: Optional[GuidanceDirection]
    management_tone: Optional[str]

    # Data quality
    data_quality: DataQuality
    missing_inputs: List[str]
    input_count: int
    max_inputs: int = 5

    # Metadata
    computed_at: datetime = field(default_factory=datetime.now)
    data_source: str = 'unknown'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'ticker': self.ticker,
            'earnings_date': self.earnings_date.isoformat() if self.earnings_date else None,
            'eqs': self.eqs,
            'event_z': self.event_z,

            'eps_z': self.eps_z,
            'rev_z': self.rev_z,
            'guidance_z': self.guidance_z,

            'eps_score': self.eps_score,
            'rev_score': self.rev_score,
            'guidance_score': self.guidance_score,
            'margin_score': self.margin_score,
            'tone_score': self.tone_score,

            'eps_surprise_pct': self.eps_surprise_pct,
            'revenue_surprise_pct': self.revenue_surprise_pct,
            'guidance_direction': self.guidance_direction.value if self.guidance_direction else None,
            'management_tone': self.management_tone,

            'data_quality': self.data_quality.value if self.data_quality else None,
            'missing_inputs': self.missing_inputs,
            'input_count': self.input_count,

            'computed_at': self.computed_at.isoformat() if self.computed_at else None,
            'data_source': self.data_source,
        }

    def get_ai_summary(self) -> str:
        """Generate formatted summary for AI context."""
        lines = [
            f"\n{'=' * 55}",
            f"EARNINGS QUALITY SCORE (EQS): {self.ticker}",
            f"{'=' * 55}",
        ]

        if self.earnings_date:
            lines.append(f"📅 Earnings Date: {self.earnings_date}")

        # Main scores
        if self.eqs is not None:
            eqs_level = self._get_eqs_level()
            lines.append(f"\n📊 EQS: {self.eqs:.0f}/100 ({eqs_level})")
        else:
            lines.append("\n📊 EQS: Unable to calculate")

        if self.event_z is not None:
            lines.append(f"📈 Event Z-Score: {self.event_z:+.2f}")

        # Surprise breakdown
        lines.append("\n📋 EARNINGS SURPRISES:")

        if self.eps_surprise_pct is not None:
            eps_emoji = "✅" if self.eps_surprise_pct > 0 else "❌"
            lines.append(
                f"   {eps_emoji} EPS: {self.eps_surprise_pct:+.1f}% (z={self.eps_z:+.2f})" if self.eps_z else f"   {eps_emoji} EPS: {self.eps_surprise_pct:+.1f}%")
        else:
            lines.append("   EPS: N/A")

        if self.revenue_surprise_pct is not None:
            rev_emoji = "✅" if self.revenue_surprise_pct > 0 else "❌"
            lines.append(
                f"   {rev_emoji} Revenue: {self.revenue_surprise_pct:+.1f}% (z={self.rev_z:+.2f})" if self.rev_z else f"   {rev_emoji} Revenue: {self.revenue_surprise_pct:+.1f}%")
        else:
            lines.append("   Revenue: N/A")

        if self.guidance_direction:
            guide_emoji = "📈" if "RAISED" in self.guidance_direction.value else "📉" if "LOWERED" in self.guidance_direction.value else "➡️"
            lines.append(f"   {guide_emoji} Guidance: {self.guidance_direction.value}")
        else:
            lines.append("   Guidance: N/A")

        # Component scores
        lines.append("\n📊 COMPONENT SCORES:")

        if self.eps_score is not None:
            lines.append(f"   EPS Score:      {self.eps_score:.0f}/100 (25%)")
        if self.rev_score is not None:
            lines.append(f"   Revenue Score:  {self.rev_score:.0f}/100 (25%)")
        if self.guidance_score is not None:
            lines.append(f"   Guidance Score: {self.guidance_score:.0f}/100 (30%)")
        if self.margin_score is not None:
            lines.append(f"   Margin Score:   {self.margin_score:.0f}/100 (10%)")
        if self.tone_score is not None:
            lines.append(f"   Tone Score:     {self.tone_score:.0f}/100 (10%)")

        # Management tone
        if self.management_tone:
            lines.append(f"\n🎙️ Management Tone: {self.management_tone}")

        # Data quality
        lines.append(f"\n📊 Data Quality: {self.data_quality.value} ({self.input_count}/{self.max_inputs} inputs)")
        if self.missing_inputs:
            lines.append(f"   Missing: {', '.join(self.missing_inputs)}")

        # Interpretation
        lines.append("\n💡 INTERPRETATION:")
        lines.extend(self._get_interpretation())

        return "\n".join(lines)

    def _get_eqs_level(self) -> str:
        """Get human-readable EQS level."""
        if self.eqs is None:
            return "Unknown"
        if self.eqs >= 80:
            return "EXCEPTIONAL - Blowout quarter"
        elif self.eqs >= 65:
            return "STRONG - Solid beat"
        elif self.eqs >= 50:
            return "GOOD - Above expectations"
        elif self.eqs >= 35:
            return "MIXED - Inline to slight miss"
        elif self.eqs >= 20:
            return "WEAK - Below expectations"
        else:
            return "POOR - Significant miss"

    def _get_interpretation(self) -> List[str]:
        """Generate interpretation bullets."""
        interp = []

        if self.eqs is None:
            return ["   • Insufficient data for analysis"]

        if self.eqs >= 70:
            interp.append("   • STRONG earnings quality")
            interp.append("   • Results exceeded consensus meaningfully")
        elif self.eqs >= 50:
            interp.append("   • SOLID earnings quality")
            interp.append("   • Results met or modestly beat consensus")
        else:
            interp.append("   • WEAK earnings quality")
            interp.append("   • Results disappointed vs consensus")

        # Guidance interpretation
        if self.guidance_direction:
            if "RAISED" in self.guidance_direction.value:
                interp.append("   • ✅ Forward guidance RAISED - bullish signal")
            elif "LOWERED" in self.guidance_direction.value:
                interp.append("   • ⚠️ Forward guidance LOWERED - bearish signal")

        return interp


# ============================================================================
# DATA RETRIEVAL FUNCTIONS
# ============================================================================

def get_earnings_data_from_db(ticker: str, earnings_date: Optional[date] = None) -> Optional[EarningsData]:
    """
    Get earnings data from YOUR earnings_analysis table.

    Args:
        ticker: Stock symbol
        earnings_date: Specific earnings date (optional, defaults to most recent)

    Returns:
        EarningsData or None if not found
    """
    if not DB_AVAILABLE:
        return None

    try:
        if earnings_date:
            query = """
                    SELECT filing_date, \
                           eps_actual, \
                           eps_estimate, \
                           eps_surprise_pct,
                           revenue_actual, \
                           revenue_estimate, \
                           revenue_surprise_pct,
                           guidance_direction, \
                           guidance_summary, \
                           overall_sentiment,
                           sentiment_score, \
                           management_tone, \
                           key_highlights, \
                           concerns
                    FROM earnings_analysis
                    WHERE ticker = %s \
                      AND filing_date = %s \
                    """
            params = (ticker, earnings_date)
        else:
            query = """
                    SELECT filing_date, \
                           eps_actual, \
                           eps_estimate, \
                           eps_surprise_pct,
                           revenue_actual, \
                           revenue_estimate, \
                           revenue_surprise_pct,
                           guidance_direction, \
                           guidance_summary, \
                           overall_sentiment,
                           sentiment_score, \
                           management_tone, \
                           key_highlights, \
                           concerns
                    FROM earnings_analysis
                    WHERE ticker = %s
                    ORDER BY filing_date DESC LIMIT 1 \
                    """
            params = (ticker,)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()

        if row:
            # Parse guidance direction
            guidance_dir = None
            if row[7]:
                try:
                    guidance_dir = GuidanceDirection(row[7])
                except ValueError:
                    # Try mapping common variations
                    guidance_map = {
                        'RAISED': GuidanceDirection.RAISED,
                        'LOWERED': GuidanceDirection.LOWERED,
                        'MAINTAINED': GuidanceDirection.MAINTAINED,
                        'NOT_PROVIDED': GuidanceDirection.NOT_PROVIDED,
                        'RAISED_STRONG': GuidanceDirection.RAISED_STRONG,
                        'LOWERED_STRONG': GuidanceDirection.LOWERED_STRONG,
                    }
                    guidance_dir = guidance_map.get(row[7].upper())

            return EarningsData(
                ticker=ticker,
                earnings_date=row[0],
                eps_actual=float(row[1]) if row[1] else None,
                eps_estimate=float(row[2]) if row[2] else None,
                eps_surprise_pct=float(row[3]) if row[3] else None,
                revenue_actual=float(row[4]) if row[4] else None,
                revenue_estimate=float(row[5]) if row[5] else None,
                revenue_surprise_pct=float(row[6]) if row[6] else None,
                guidance_direction=guidance_dir,
                guidance_summary=row[8],
                sentiment_score=int(row[10]) if row[10] else None,
                management_tone=row[11],
                key_highlights=row[12] if row[12] else None,
                concerns=row[13] if row[13] else None,
                data_source='database',
            )

        return None

    except Exception as e:
        logger.warning(f"{ticker}: Error getting earnings from DB: {e}")
        return None


def get_earnings_data_from_yfinance(ticker: str, earnings_date: Optional[date] = None) -> Optional[EarningsData]:
    """
    Get earnings data from yfinance as fallback.

    Args:
        ticker: Stock symbol
        earnings_date: Specific earnings date (optional)

    Returns:
        EarningsData or None if not found
    """
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)

        # Get earnings history
        earnings = stock.earnings_history

        if earnings is None or earnings.empty:
            logger.debug(f"{ticker}: No earnings history in yfinance")
            return None

        # Find the relevant earnings record
        if earnings_date:
            # Find closest match
            earnings['date'] = pd.to_datetime(earnings.index).date
            matches = earnings[earnings['date'] == earnings_date]
            if matches.empty:
                # Try within 7 days
                target = pd.Timestamp(earnings_date)
                earnings['days_diff'] = abs((pd.to_datetime(earnings.index) - target).days)
                closest = earnings[earnings['days_diff'] <= 7]
                if closest.empty:
                    return None
                row = closest.iloc[0]
            else:
                row = matches.iloc[0]
        else:
            # Get most recent
            row = earnings.iloc[0]

        # Extract data
        eps_actual = row.get('epsActual')
        eps_estimate = row.get('epsEstimate')

        # Calculate surprise
        eps_surprise_pct = None
        if eps_actual is not None and eps_estimate is not None and eps_estimate != 0:
            eps_surprise_pct = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100

        # Get revenue from quarterly financials
        revenue_actual = None
        revenue_estimate = None
        revenue_surprise_pct = None

        try:
            quarterly = stock.quarterly_income_stmt
            if quarterly is not None and not quarterly.empty:
                if 'Total Revenue' in quarterly.index:
                    revenue_actual = float(quarterly.loc['Total Revenue'].iloc[0])
        except:
            pass

        # Determine earnings date from index
        if hasattr(row, 'name'):
            actual_date = pd.to_datetime(row.name).date()
        else:
            actual_date = earnings_date or date.today()

        return EarningsData(
            ticker=ticker,
            earnings_date=actual_date,
            eps_actual=float(eps_actual) if eps_actual else None,
            eps_estimate=float(eps_estimate) if eps_estimate else None,
            eps_surprise_pct=float(eps_surprise_pct) if eps_surprise_pct else None,
            revenue_actual=revenue_actual,
            revenue_estimate=revenue_estimate,
            revenue_surprise_pct=revenue_surprise_pct,
            guidance_direction=None,  # yfinance doesn't provide guidance
            data_source='yfinance',
        )

    except Exception as e:
        logger.warning(f"{ticker}: Error getting earnings from yfinance: {e}")
        return None


def get_earnings_data(ticker: str, earnings_date: Optional[date] = None) -> Optional[EarningsData]:
    """
    Get earnings data from best available source.

    Priority:
    1. Your database (earnings_analysis table)
    2. yfinance fallback

    Args:
        ticker: Stock symbol
        earnings_date: Specific earnings date (optional)

    Returns:
        EarningsData or None if not found
    """
    # Try database first
    data = get_earnings_data_from_db(ticker, earnings_date)

    if data:
        logger.debug(f"{ticker}: Got earnings data from database")
        return data

    # Fallback to yfinance
    data = get_earnings_data_from_yfinance(ticker, earnings_date)

    if data:
        logger.debug(f"{ticker}: Got earnings data from yfinance")

    return data


def get_historical_surprises(ticker: str, num_quarters: int = 8) -> Dict[str, Any]:
    """
    Get historical surprise statistics for z-score calculation.

    Args:
        ticker: Stock symbol
        num_quarters: Number of quarters to analyze

    Returns:
        Dict with mean/std for EPS and revenue surprises
    """
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        earnings = stock.earnings_history

        if earnings is None or len(earnings) < 4:
            return {
                'eps_mean': DEFAULT_EPS_SURPRISE_MEAN,
                'eps_std': DEFAULT_EPS_SURPRISE_STD,
                'rev_mean': DEFAULT_REV_SURPRISE_MEAN,
                'rev_std': DEFAULT_REV_SURPRISE_STD,
            }

        # Calculate EPS surprises
        surprises = []
        for _, row in earnings.head(num_quarters).iterrows():
            actual = row.get('epsActual')
            estimate = row.get('epsEstimate')
            if actual is not None and estimate is not None and estimate != 0:
                surprise = ((actual - estimate) / abs(estimate)) * 100
                surprises.append(surprise)

        if len(surprises) >= 4:
            eps_mean = np.mean(surprises)
            eps_std = np.std(surprises) if np.std(surprises) > 0 else DEFAULT_EPS_SURPRISE_STD
        else:
            eps_mean = DEFAULT_EPS_SURPRISE_MEAN
            eps_std = DEFAULT_EPS_SURPRISE_STD

        return {
            'eps_mean': eps_mean,
            'eps_std': eps_std,
            'rev_mean': DEFAULT_REV_SURPRISE_MEAN,
            'rev_std': DEFAULT_REV_SURPRISE_STD,
            'num_quarters': len(surprises),
        }

    except Exception as e:
        logger.debug(f"{ticker}: Error getting historical surprises: {e}")
        return {
            'eps_mean': DEFAULT_EPS_SURPRISE_MEAN,
            'eps_std': DEFAULT_EPS_SURPRISE_STD,
            'rev_mean': DEFAULT_REV_SURPRISE_MEAN,
            'rev_std': DEFAULT_REV_SURPRISE_STD,
        }


# ============================================================================
# Z-SCORE CALCULATION FUNCTIONS
# ============================================================================

def calculate_eps_z(eps_surprise_pct: float,
                    historical_mean: float = DEFAULT_EPS_SURPRISE_MEAN,
                    historical_std: float = DEFAULT_EPS_SURPRISE_STD) -> float:
    """
    Calculate EPS surprise z-score.

    Z = (surprise - historical_mean) / historical_std

    A z-score of 0 means the surprise matched historical average.
    Positive z = beat more than usual, Negative z = beat less than usual

    Args:
        eps_surprise_pct: EPS surprise percentage
        historical_mean: Historical average surprise
        historical_std: Historical standard deviation

    Returns:
        Z-score
    """
    if historical_std == 0:
        historical_std = DEFAULT_EPS_SURPRISE_STD

    return (eps_surprise_pct - historical_mean) / historical_std


def calculate_rev_z(rev_surprise_pct: float,
                    historical_mean: float = DEFAULT_REV_SURPRISE_MEAN,
                    historical_std: float = DEFAULT_REV_SURPRISE_STD) -> float:
    """
    Calculate revenue surprise z-score.

    Args:
        rev_surprise_pct: Revenue surprise percentage
        historical_mean: Historical average surprise
        historical_std: Historical standard deviation

    Returns:
        Z-score
    """
    if historical_std == 0:
        historical_std = DEFAULT_REV_SURPRISE_STD

    return (rev_surprise_pct - historical_mean) / historical_std


def calculate_guidance_z(guidance: GuidanceDirection) -> float:
    """
    Calculate guidance z-score from direction.

    Maps guidance direction to approximate z-score:
    - RAISED_STRONG: +2.0
    - RAISED: +1.0
    - MAINTAINED: 0.0
    - LOWERED: -1.0
    - LOWERED_STRONG: -2.0

    Args:
        guidance: GuidanceDirection enum

    Returns:
        Z-score
    """
    return guidance.numeric_value


def calculate_event_z(eps_z: Optional[float],
                      rev_z: Optional[float],
                      guidance_z: Optional[float]) -> Optional[float]:
    """
    Calculate blended event z-score.

    Weights: EPS 35%, Revenue 30%, Guidance 35%

    Args:
        eps_z: EPS surprise z-score
        rev_z: Revenue surprise z-score
        guidance_z: Guidance z-score

    Returns:
        Blended event z-score or None
    """
    components = {
        'eps_z': eps_z,
        'rev_z': rev_z,
        'guidance_z': guidance_z,
    }

    available = {k: v for k, v in components.items() if v is not None}

    if not available:
        return None

    # Get weights for available components
    weights = {k: EVENT_Z_WEIGHTS[k] for k in available.keys()}
    total_weight = sum(weights.values())

    # Normalize weights
    normalized = {k: v / total_weight for k, v in weights.items()}

    # Calculate weighted average
    event_z = sum(available[k] * normalized[k] for k in available.keys())

    return event_z


# ============================================================================
# SCORE CONVERSION FUNCTIONS
# ============================================================================

def z_to_score(z: float, center: float = 50, scale: float = 15) -> float:
    """
    Convert z-score to 0-100 score.

    Uses linear mapping: score = center + (z * scale)
    Clamped to [0, 100]

    Args:
        z: Z-score
        center: Score for z=0 (default 50)
        scale: Points per z (default 15)

    Returns:
        Score 0-100
    """
    score = center + (z * scale)
    return max(0, min(100, score))


def guidance_to_score(guidance: Optional[GuidanceDirection]) -> float:
    """
    Convert guidance direction to 0-100 score.

    Args:
        guidance: GuidanceDirection enum

    Returns:
        Score 0-100
    """
    if guidance is None:
        return 50.0  # Neutral

    return guidance.eqs_score


def tone_to_score(tone: Optional[str], sentiment_score: Optional[int] = None) -> float:
    """
    Convert management tone to 0-100 score.

    Args:
        tone: Management tone string
        sentiment_score: Optional sentiment score (0-100)

    Returns:
        Score 0-100
    """
    if sentiment_score is not None:
        return float(sentiment_score)

    if tone is None:
        return 50.0

    tone_scores = {
        'CONFIDENT': 75,
        'OPTIMISTIC': 70,
        'CAUTIOUS': 45,
        'DEFENSIVE': 30,
        'NEUTRAL': 50,
    }

    return tone_scores.get(tone.upper(), 50.0)


def margin_to_score(margin_change: Optional[float]) -> float:
    """
    Convert margin change to 0-100 score.

    Args:
        margin_change: Change in margin (percentage points)

    Returns:
        Score 0-100
    """
    if margin_change is None:
        return 50.0

    # Map -5% to +5% margin change to 0-100
    # -5% -> 0, 0% -> 50, +5% -> 100
    score = 50 + (margin_change * 10)
    return max(0, min(100, score))


# ============================================================================
# MAIN CALCULATION FUNCTION
# ============================================================================

def calculate_eqs(ticker: str,
                  earnings_date: Optional[date] = None,
                  earnings_data: Optional[EarningsData] = None) -> EQSCalculationResult:
    """
    Calculate the complete Earnings Quality Score for a ticker.

    This is the main entry point for Phase 8. It:
    1. Retrieves earnings data (DB or yfinance)
    2. Calculates z-scores for surprises
    3. Converts to component scores
    4. Computes weighted EQS

    Args:
        ticker: Stock symbol
        earnings_date: Specific earnings date (optional)
        earnings_data: Pre-fetched earnings data (optional)

    Returns:
        EQSCalculationResult with all components
    """
    logger.info(f"{ticker}: Calculating EQS...")

    missing_inputs = []

    # =========================================================================
    # STEP 1: Get earnings data
    # =========================================================================

    if earnings_data is None:
        earnings_data = get_earnings_data(ticker, earnings_date)

    if earnings_data is None:
        logger.warning(f"{ticker}: No earnings data available")
        return EQSCalculationResult(
            ticker=ticker,
            earnings_date=earnings_date or date.today(),
            eqs=None,
            event_z=None,
            eps_z=None,
            rev_z=None,
            guidance_z=None,
            eps_score=None,
            rev_score=None,
            guidance_score=None,
            margin_score=None,
            tone_score=None,
            eps_surprise_pct=None,
            revenue_surprise_pct=None,
            guidance_direction=None,
            management_tone=None,
            data_quality=DataQuality.LOW,
            missing_inputs=['earnings_data'],
            input_count=0,
            data_source='none',
        )

    # =========================================================================
    # STEP 2: Get historical statistics for z-scores
    # =========================================================================

    historical = get_historical_surprises(ticker)

    # =========================================================================
    # STEP 3: Calculate z-scores
    # =========================================================================

    # EPS z-score
    eps_z = None
    if earnings_data.eps_surprise_pct is not None:
        eps_z = calculate_eps_z(
            earnings_data.eps_surprise_pct,
            historical['eps_mean'],
            historical['eps_std']
        )
    else:
        missing_inputs.append('eps_surprise')

    # Revenue z-score
    rev_z = None
    if earnings_data.revenue_surprise_pct is not None:
        rev_z = calculate_rev_z(
            earnings_data.revenue_surprise_pct,
            historical['rev_mean'],
            historical['rev_std']
        )
    else:
        missing_inputs.append('revenue_surprise')

    # Guidance z-score
    guidance_z = None
    if earnings_data.guidance_direction:
        guidance_z = calculate_guidance_z(earnings_data.guidance_direction)
    else:
        missing_inputs.append('guidance')

    # Event z-score (blended)
    event_z = calculate_event_z(eps_z, rev_z, guidance_z)

    # =========================================================================
    # STEP 4: Convert to component scores (0-100)
    # =========================================================================

    eps_score = z_to_score(eps_z) if eps_z is not None else None
    rev_score = z_to_score(rev_z) if rev_z is not None else None
    guidance_score = guidance_to_score(earnings_data.guidance_direction)

    # Margin score
    margin_score = None
    if earnings_data.operating_margin_change is not None:
        margin_score = margin_to_score(earnings_data.operating_margin_change)
    elif earnings_data.gross_margin_change is not None:
        margin_score = margin_to_score(earnings_data.gross_margin_change)
    else:
        margin_score = 50.0  # Default neutral
        missing_inputs.append('margin_data')

    # Tone score
    tone_score = tone_to_score(
        earnings_data.management_tone,
        earnings_data.sentiment_score
    )
    if earnings_data.management_tone is None and earnings_data.sentiment_score is None:
        missing_inputs.append('tone_data')

    # =========================================================================
    # STEP 5: Calculate weighted EQS
    # =========================================================================

    scores = {
        'eps_z': eps_score,
        'rev_z': rev_score,
        'guidance_score': guidance_score,
        'margin_score': margin_score,
        'tone_score': tone_score,
    }

    eqs, input_count = _calculate_weighted_eqs(scores)

    # =========================================================================
    # STEP 6: Assess data quality
    # =========================================================================

    data_quality = _assess_eqs_data_quality(missing_inputs, input_count)

    # =========================================================================
    # STEP 7: Build result
    # =========================================================================

    result = EQSCalculationResult(
        ticker=ticker,
        earnings_date=earnings_data.earnings_date,
        eqs=eqs,
        event_z=event_z,
        eps_z=eps_z,
        rev_z=rev_z,
        guidance_z=guidance_z,
        eps_score=eps_score,
        rev_score=rev_score,
        guidance_score=guidance_score,
        margin_score=margin_score,
        tone_score=tone_score,
        eps_surprise_pct=earnings_data.eps_surprise_pct,
        revenue_surprise_pct=earnings_data.revenue_surprise_pct,
        guidance_direction=earnings_data.guidance_direction,
        management_tone=earnings_data.management_tone,
        data_quality=data_quality,
        missing_inputs=missing_inputs,
        input_count=input_count,
        data_source=earnings_data.data_source,
    )

    eqs_str = f"{eqs:.0f}" if eqs is not None else "N/A"
    event_z_str = f"{event_z:.2f}" if event_z is not None else "N/A"
    logger.info(f"{ticker}: EQS={eqs_str}, Event_Z={event_z_str}, Quality={data_quality.value}")

    return result


def _calculate_weighted_eqs(scores: Dict[str, Optional[float]]) -> Tuple[Optional[float], int]:
    """
    Calculate weighted EQS from component scores.

    Args:
        scores: Dict of component name -> score (0-100 or None)

    Returns:
        Tuple of (final_eqs, input_count)
    """
    # Map EQS_WEIGHTS keys to our score keys
    weight_mapping = {
        'eps_z': 'eps_z',
        'rev_z': 'rev_z',
        'guidance_score': 'guidance_score',
        'margin_score': 'margin_score',
        'tone_score': 'tone_score',
    }

    available = {k: v for k, v in scores.items() if v is not None}
    input_count = len(available)

    if input_count == 0:
        return None, 0

    # Get weights for available components
    weights = {k: EQS_WEIGHTS[weight_mapping.get(k, k)] for k in available.keys()}
    total_weight = sum(weights.values())

    if total_weight == 0:
        return None, 0

    # Normalize weights
    normalized = {k: v / total_weight for k, v in weights.items()}

    # Calculate weighted average
    eqs = sum(available[k] * normalized[k] for k in available.keys())

    return max(0, min(100, eqs)), input_count


def _assess_eqs_data_quality(missing_inputs: List[str], input_count: int) -> DataQuality:
    """
    Assess EQS data quality.

    Args:
        missing_inputs: List of missing inputs
        input_count: Number of available inputs

    Returns:
        DataQuality level
    """
    # Critical: EPS and revenue data
    critical_missing = any(x in missing_inputs for x in ['eps_surprise', 'earnings_data'])

    if input_count >= 4:
        return DataQuality.HIGH
    elif input_count >= 3 and not critical_missing:
        return DataQuality.MEDIUM
    elif input_count >= 2:
        return DataQuality.MEDIUM
    else:
        return DataQuality.LOW


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_eqs_summary_for_ai(ticker: str, earnings_date: Optional[date] = None) -> str:
    """
    Get EQS summary formatted for AI context.

    Args:
        ticker: Stock symbol
        earnings_date: Specific earnings date (optional)

    Returns:
        Formatted string for AI consumption
    """
    result = calculate_eqs(ticker, earnings_date)
    return result.get_ai_summary()


def get_eqs_for_screener(ticker: str, earnings_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Get EQS data formatted for screener integration.

    Args:
        ticker: Stock symbol
        earnings_date: Specific earnings date (optional)

    Returns:
        Dict with screener-relevant fields
    """
    result = calculate_eqs(ticker, earnings_date)

    return {
        'eqs': result.eqs,
        'event_z': result.event_z,
        'eps_surprise_pct': result.eps_surprise_pct,
        'revenue_surprise_pct': result.revenue_surprise_pct,
        'guidance_direction': result.guidance_direction.value if result.guidance_direction else None,
        'data_quality': result.data_quality.value if result.data_quality else 'LOW',
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("Phase 8: EQS Calculator Test")
    print("=" * 60)

    test_tickers = ["AAPL", "NVDA", "TSLA"]

    for ticker in test_tickers:
        print(f"\n--- {ticker} ---")
        result = calculate_eqs(ticker)

        print(f"EQS: {result.eqs:.0f}/100" if result.eqs else "EQS: N/A")
        print(f"Event Z: {result.event_z:+.2f}" if result.event_z else "Event Z: N/A")
        print(f"EPS Surprise: {result.eps_surprise_pct:+.1f}%" if result.eps_surprise_pct else "EPS Surprise: N/A")
        print(f"Data Quality: {result.data_quality.value}")

    # Print full summary for one ticker
    print("\n" + "=" * 60)
    print("Full AI Summary for AAPL:")
    print("=" * 60)
    print(get_eqs_summary_for_ai("AAPL"))