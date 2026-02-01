"""
ECS Calculator - Expectations Clearance Score

Combines IES (pre-earnings expectations) + EQS (post-earnings quality) to determine
if earnings results CLEARED the expectations bar.

The key insight: It's not just about beating consensus, it's about beating
what was PRICED IN before earnings.

Formula:
    required_z = REQUIRED_Z_BASE + (implied_move_pctl / 50)
    required_z = clamp(required_z, REQUIRED_Z_FLOOR, REQUIRED_Z_CEILING)

    diff = event_z - required_z

    ECS Categories:
    - STRONG_BEAT: diff > 0.5
    - BEAT: diff >= 0
    - INLINE: diff >= -0.3
    - MISS: diff >= -1.0
    - STRONG_MISS: diff < -1.0

Usage:
    from src.analytics.earnings_intelligence import calculate_ecs

    result = calculate_ecs("NVDA", earnings_date=date(2024, 11, 20))
    print(f"ECS: {result.ecs_category.value} - Cleared: {result.cleared_bar}")

Author: Alpha Research Platform
Phase: 9 of Earnings Intelligence System
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import os

from src.utils.logging import get_logger

# Import from models
from src.analytics.earnings_intelligence.models import (
    ECSCategory,
    ExpectationsRegime,
    DataQuality,
    PositionScaling,
    EarningsIntelligenceResult,
    REQUIRED_Z_BASE,
    REQUIRED_Z_SLOPE,
    REQUIRED_Z_FLOOR,
    REQUIRED_Z_CEILING,
)

# Import from previous phases
from src.analytics.earnings_intelligence.ies_calculator import (
    calculate_ies,
    IESCalculationResult,
)

from src.analytics.earnings_intelligence.eqs_calculator import (
    calculate_eqs,
    EQSCalculationResult,
    get_earnings_data,
)

logger = get_logger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ECSCalculationResult:
    """Complete ECS calculation result combining IES + EQS."""

    # Core outputs
    ticker: str
    earnings_date: Optional[date]

    # ECS determination
    ecs_category: ECSCategory
    cleared_bar: bool                       # Did results clear expectations?

    # The key comparison
    event_z: Optional[float]                # What they delivered (from EQS)
    required_z: Optional[float]             # What was required (from IES)
    clearance_margin: Optional[float]       # event_z - required_z

    # Pre-earnings context (IES)
    ies: Optional[float]
    regime: ExpectationsRegime
    implied_move_pctl: Optional[float]

    # Post-earnings results (EQS)
    eqs: Optional[float]
    eps_surprise_pct: Optional[float]
    revenue_surprise_pct: Optional[float]

    # Position sizing
    position_scale: float
    score_adjustment: int                   # For screener integration

    # Data quality
    data_quality: DataQuality
    missing_inputs: List[str]

    # Component results (for debugging)
    ies_result: Optional[IESCalculationResult] = None
    eqs_result: Optional[EQSCalculationResult] = None

    # Metadata
    computed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'ticker': self.ticker,
            'earnings_date': self.earnings_date.isoformat() if self.earnings_date else None,

            'ecs_category': self.ecs_category.value if self.ecs_category else None,
            'cleared_bar': self.cleared_bar,

            'event_z': self.event_z,
            'required_z': self.required_z,
            'clearance_margin': self.clearance_margin,

            'ies': self.ies,
            'regime': self.regime.value if self.regime else None,
            'implied_move_pctl': self.implied_move_pctl,

            'eqs': self.eqs,
            'eps_surprise_pct': self.eps_surprise_pct,
            'revenue_surprise_pct': self.revenue_surprise_pct,

            'position_scale': self.position_scale,
            'score_adjustment': self.score_adjustment,

            'data_quality': self.data_quality.value if self.data_quality else None,
            'missing_inputs': self.missing_inputs,

            'computed_at': self.computed_at.isoformat() if self.computed_at else None,
        }

    def get_ai_summary(self) -> str:
        """Generate formatted summary for AI context."""
        lines = [
            f"\n{'='*60}",
            f"EXPECTATIONS CLEARANCE SCORE (ECS): {self.ticker}",
            f"{'='*60}",
        ]

        if self.earnings_date:
            lines.append(f"📅 Earnings Date: {self.earnings_date}")

        # Main ECS result
        lines.append("")
        emoji = self._get_ecs_emoji()
        lines.append(f"{emoji} ECS: {self.ecs_category.value}")
        lines.append(f"{'✅' if self.cleared_bar else '❌'} Cleared Expectations Bar: {'YES' if self.cleared_bar else 'NO'}")

        # The key comparison
        lines.append("")
        lines.append("📊 THE KEY COMPARISON:")
        if self.event_z is not None:
            lines.append(f"   Event Z (delivered):  {self.event_z:+.2f}")
        if self.required_z is not None:
            lines.append(f"   Required Z (bar):     {self.required_z:+.2f}")
        if self.clearance_margin is not None:
            margin_emoji = "✅" if self.clearance_margin >= 0 else "❌"
            lines.append(f"   Clearance Margin:     {margin_emoji} {self.clearance_margin:+.2f}")

        # Pre-earnings setup
        lines.append("")
        lines.append("📈 PRE-EARNINGS (IES):")
        if self.ies is not None:
            lines.append(f"   IES: {self.ies:.0f}/100")
        lines.append(f"   Regime: {self.regime.value}")
        if self.implied_move_pctl is not None:
            lines.append(f"   Implied Move Pctl: {self.implied_move_pctl:.0f}th")

        # Post-earnings results
        lines.append("")
        lines.append("📉 POST-EARNINGS (EQS):")
        if self.eqs is not None:
            lines.append(f"   EQS: {self.eqs:.0f}/100")
        if self.eps_surprise_pct is not None:
            eps_emoji = "✅" if self.eps_surprise_pct > 0 else "❌"
            lines.append(f"   {eps_emoji} EPS Surprise: {self.eps_surprise_pct:+.1f}%")
        if self.revenue_surprise_pct is not None:
            rev_emoji = "✅" if self.revenue_surprise_pct > 0 else "❌"
            lines.append(f"   {rev_emoji} Revenue Surprise: {self.revenue_surprise_pct:+.1f}%")

        # Trading implications
        lines.append("")
        lines.append("💼 TRADING IMPLICATIONS:")
        lines.append(f"   Position Scale: {self.position_scale:.0%}")
        lines.append(f"   Score Adjustment: {self.score_adjustment:+d}")

        # Data quality
        lines.append(f"\n📊 Data Quality: {self.data_quality.value}")
        if self.missing_inputs:
            lines.append(f"   Missing: {', '.join(self.missing_inputs[:3])}")

        # Interpretation
        lines.append("")
        lines.append("💡 INTERPRETATION:")
        lines.extend(self._get_interpretation())

        return "\n".join(lines)

    def _get_ecs_emoji(self) -> str:
        """Get emoji for ECS category."""
        emojis = {
            ECSCategory.STRONG_BEAT: "🚀",
            ECSCategory.BEAT: "✅",
            ECSCategory.INLINE: "➡️",
            ECSCategory.MISS: "⚠️",
            ECSCategory.STRONG_MISS: "🔴",
            ECSCategory.UNKNOWN: "❓",
        }
        return emojis.get(self.ecs_category, "❓")

    def _get_interpretation(self) -> List[str]:
        """Generate interpretation bullets."""
        interp = []

        if self.ecs_category == ECSCategory.UNKNOWN:
            return ["   • Insufficient data for analysis"]

        # ECS interpretation
        if self.ecs_category == ECSCategory.STRONG_BEAT:
            interp.append("   • 🚀 STRONG BEAT - Results significantly exceeded priced-in expectations")
            interp.append("   • Likely positive price reaction")
            interp.append("   • Consider adding to position")
        elif self.ecs_category == ECSCategory.BEAT:
            interp.append("   • ✅ BEAT - Results cleared the expectations bar")
            interp.append("   • Modest positive reaction expected")
        elif self.ecs_category == ECSCategory.INLINE:
            interp.append("   • ➡️ INLINE - Results roughly matched expectations")
            interp.append("   • Neutral to slightly negative reaction")
            interp.append("   • Hold existing position")
        elif self.ecs_category == ECSCategory.MISS:
            interp.append("   • ⚠️ MISS - Results fell short of priced-in expectations")
            interp.append("   • Negative reaction likely")
            interp.append("   • Consider reducing position")
        elif self.ecs_category == ECSCategory.STRONG_MISS:
            interp.append("   • 🔴 STRONG MISS - Results significantly disappointed")
            interp.append("   • Sharp negative reaction expected")
            interp.append("   • Review thesis, consider exit")

        # Regime context
        if self.regime == ExpectationsRegime.HYPED:
            if self.ecs_category.is_positive:
                interp.append("   • ⭐ Beat in HYPED regime - very impressive")
            else:
                interp.append("   • Even a beat was required in this HYPED setup")
        elif self.regime == ExpectationsRegime.FEARED:
            if self.ecs_category.is_negative:
                interp.append("   • Failed to clear even a LOW bar")
            else:
                interp.append("   • Cleared a LOW bar as expected")

        return interp


# ============================================================================
# REQUIRED Z CALCULATION
# ============================================================================

def calculate_required_z(ies: Optional[float] = None,
                         implied_move_pctl: Optional[float] = None) -> float:
    """
    Calculate the required z-score threshold based on IES/implied move.

    Higher expectations (higher IES/implied_move) = higher bar to clear.

    Formula:
        required_z = BASE + (implied_move_pctl / 50)
        required_z = clamp(required_z, FLOOR, CEILING)

    Args:
        ies: Implied Expectations Score (0-100)
        implied_move_pctl: Implied move percentile (0-100)

    Returns:
        Required z-score threshold
    """
    # Use implied_move_pctl if available, otherwise derive from IES
    if implied_move_pctl is not None:
        base_input = implied_move_pctl
    elif ies is not None:
        # IES is correlated with implied move, use as proxy
        base_input = ies
    else:
        # Default to median
        base_input = 50.0

    # Calculate required z
    required_z = REQUIRED_Z_BASE + (base_input * REQUIRED_Z_SLOPE)

    # Clamp to floor/ceiling
    required_z = max(REQUIRED_Z_FLOOR, min(REQUIRED_Z_CEILING, required_z))

    return required_z


def calculate_clearance_margin(event_z: Optional[float],
                               required_z: Optional[float]) -> Optional[float]:
    """
    Calculate the clearance margin (how much above/below the bar).

    Args:
        event_z: Actual event z-score
        required_z: Required z-score threshold

    Returns:
        Clearance margin or None
    """
    if event_z is None or required_z is None:
        return None

    return event_z - required_z


# ============================================================================
# MAIN CALCULATION FUNCTION
# ============================================================================

def calculate_ecs(ticker: str,
                  earnings_date: Optional[date] = None,
                  ies_result: Optional[IESCalculationResult] = None,
                  eqs_result: Optional[EQSCalculationResult] = None) -> ECSCalculationResult:
    """
    Calculate the complete Expectations Clearance Score.

    This is the main entry point for Phase 9. It:
    1. Gets/uses IES (pre-earnings expectations)
    2. Gets/uses EQS (post-earnings quality)
    3. Calculates required_z (the bar)
    4. Compares event_z vs required_z
    5. Determines ECS category

    Args:
        ticker: Stock symbol
        earnings_date: Specific earnings date (optional)
        ies_result: Pre-calculated IES result (optional)
        eqs_result: Pre-calculated EQS result (optional)

    Returns:
        ECSCalculationResult with all components
    """
    logger.info(f"{ticker}: Calculating ECS...")

    missing_inputs = []

    # =========================================================================
    # STEP 1: Get/calculate IES (pre-earnings)
    # =========================================================================

    if ies_result is None:
        ies_result = calculate_ies(ticker)

    ies = ies_result.ies
    regime = ies_result.regime
    implied_move_pctl = ies_result.implied_move_pctl
    position_scale = ies_result.position_scale

    if ies is None:
        missing_inputs.append('ies')

    # =========================================================================
    # STEP 2: Get/calculate EQS (post-earnings)
    # =========================================================================

    if eqs_result is None:
        eqs_result = calculate_eqs(ticker, earnings_date)

    eqs = eqs_result.eqs
    event_z = eqs_result.event_z
    eps_surprise_pct = eqs_result.eps_surprise_pct
    revenue_surprise_pct = eqs_result.revenue_surprise_pct
    actual_earnings_date = eqs_result.earnings_date

    if event_z is None:
        missing_inputs.append('event_z')

    # =========================================================================
    # STEP 3: Calculate required_z (the bar)
    # =========================================================================

    required_z = calculate_required_z(ies, implied_move_pctl)

    # =========================================================================
    # STEP 4: Calculate clearance margin
    # =========================================================================

    clearance_margin = calculate_clearance_margin(event_z, required_z)

    # =========================================================================
    # STEP 5: Determine ECS category
    # =========================================================================

    ecs_category = ECSCategory.from_event_z(event_z, required_z)
    cleared_bar = ecs_category.is_positive or ecs_category == ECSCategory.INLINE

    # =========================================================================
    # STEP 6: Get score adjustment for screener
    # =========================================================================

    score_adjustment = ecs_category.score_adjustment

    # =========================================================================
    # STEP 7: Assess combined data quality
    # =========================================================================

    data_quality = _assess_ecs_data_quality(
        ies_result.data_quality,
        eqs_result.data_quality,
        missing_inputs
    )

    # Combine missing inputs from both
    all_missing = list(set(
        missing_inputs +
        ies_result.missing_inputs +
        eqs_result.missing_inputs
    ))

    # =========================================================================
    # STEP 8: Build result
    # =========================================================================

    result = ECSCalculationResult(
        ticker=ticker,
        earnings_date=actual_earnings_date,

        ecs_category=ecs_category,
        cleared_bar=cleared_bar,

        event_z=event_z,
        required_z=required_z,
        clearance_margin=clearance_margin,

        ies=ies,
        regime=regime,
        implied_move_pctl=implied_move_pctl,

        eqs=eqs,
        eps_surprise_pct=eps_surprise_pct,
        revenue_surprise_pct=revenue_surprise_pct,

        position_scale=position_scale,
        score_adjustment=score_adjustment,

        data_quality=data_quality,
        missing_inputs=all_missing,

        ies_result=ies_result,
        eqs_result=eqs_result,
    )

    event_z_str = f"{event_z:.2f}" if event_z is not None else "N/A"
    logger.info(f"{ticker}: ECS={ecs_category.value}, Event_Z={event_z_str}, Required_Z={required_z:.2f}, Cleared={cleared_bar}")

    return result


def _assess_ecs_data_quality(ies_quality: DataQuality,
                              eqs_quality: DataQuality,
                              missing_inputs: List[str]) -> DataQuality:
    """
    Assess combined ECS data quality.

    Args:
        ies_quality: IES data quality
        eqs_quality: EQS data quality
        missing_inputs: List of missing critical inputs

    Returns:
        DataQuality level
    """
    # Both HIGH -> HIGH
    if ies_quality == DataQuality.HIGH and eqs_quality == DataQuality.HIGH:
        return DataQuality.HIGH

    # Either LOW -> LOW
    if ies_quality == DataQuality.LOW or eqs_quality == DataQuality.LOW:
        return DataQuality.LOW

    # Missing critical inputs -> LOW
    if 'event_z' in missing_inputs or 'ies' in missing_inputs:
        return DataQuality.LOW

    # Default to MEDIUM
    return DataQuality.MEDIUM


# ============================================================================
# BATCH CALCULATION
# ============================================================================

def calculate_ecs_batch(tickers: List[str],
                        progress_callback=None) -> Dict[str, ECSCalculationResult]:
    """
    Calculate ECS for multiple tickers.

    Args:
        tickers: List of ticker symbols
        progress_callback: Optional callback(current, total, ticker)

    Returns:
        Dict of ticker -> ECSCalculationResult
    """
    results = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        try:
            results[ticker] = calculate_ecs(ticker)
        except Exception as e:
            logger.error(f"{ticker}: Error calculating ECS: {e}")
            # Create minimal result with error
            results[ticker] = ECSCalculationResult(
                ticker=ticker,
                earnings_date=None,
                ecs_category=ECSCategory.UNKNOWN,
                cleared_bar=False,
                event_z=None,
                required_z=None,
                clearance_margin=None,
                ies=None,
                regime=ExpectationsRegime.NORMAL,
                implied_move_pctl=None,
                eqs=None,
                eps_surprise_pct=None,
                revenue_surprise_pct=None,
                position_scale=1.0,
                score_adjustment=0,
                data_quality=DataQuality.LOW,
                missing_inputs=['error'],
            )

        if progress_callback:
            progress_callback(i + 1, total, ticker)

    return results


# ============================================================================
# DATABASE INTEGRATION
# ============================================================================

def save_ecs_to_db(result: ECSCalculationResult) -> bool:
    """
    Save ECS result to earnings_intelligence table.

    Args:
        result: ECSCalculationResult to save

    Returns:
        True if successful
    """
    try:
        from src.db.connection import get_connection

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE earnings_intelligence SET
                                                     event_z               = %s,
                                                     required_z            = %s,
                                                     ecs                   = %s,
                                                     ecs_compute_timestamp = %s,
                                                     eqs                   = %s,
                                                     eps_actual            = %s,
                                                     eps_surprise_pct      = %s,
                                                     revenue_surprise_pct  = %s,
                                                     updated_at            = NOW()
                    WHERE ticker = %s AND earnings_date = %s
                """, (
                    result.event_z,
                    result.required_z,
                    result.ecs_category.value if result.ecs_category else None,
                    result.computed_at,
                    result.eqs,
                    result.eqs_result.eps_surprise_pct if result.eqs_result else None,
                    result.eps_surprise_pct,
                    result.revenue_surprise_pct,
                    result.ticker,
                    result.earnings_date,
                ))

        logger.info(f"{result.ticker}: Saved ECS to database")
        return True

    except Exception as e:
        logger.error(f"{result.ticker}: Error saving ECS to database: {e}")
        return False


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_ecs_summary_for_ai(ticker: str, earnings_date: Optional[date] = None) -> str:
    """
    Get ECS summary formatted for AI context.

    Args:
        ticker: Stock symbol
        earnings_date: Specific earnings date (optional)

    Returns:
        Formatted string for AI consumption
    """
    result = calculate_ecs(ticker, earnings_date)
    return result.get_ai_summary()


def get_ecs_for_screener(ticker: str, earnings_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Get ECS data formatted for screener integration.

    Args:
        ticker: Stock symbol
        earnings_date: Specific earnings date (optional)

    Returns:
        Dict with screener-relevant fields
    """
    result = calculate_ecs(ticker, earnings_date)

    return {
        'ecs_category': result.ecs_category.value if result.ecs_category else 'UNKNOWN',
        'cleared_bar': result.cleared_bar,
        'event_z': result.event_z,
        'required_z': result.required_z,
        'clearance_margin': result.clearance_margin,
        'score_adjustment': result.score_adjustment,
        'ies': result.ies,
        'eqs': result.eqs,
        'regime': result.regime.value if result.regime else 'NORMAL',
        'data_quality': result.data_quality.value if result.data_quality else 'LOW',
    }


def get_full_earnings_analysis(ticker: str,
                                earnings_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Get complete earnings intelligence analysis.

    Combines IES, EQS, and ECS into a single comprehensive result.

    Args:
        ticker: Stock symbol
        earnings_date: Specific earnings date (optional)

    Returns:
        Dict with complete analysis
    """
    # Calculate all components
    ies_result = calculate_ies(ticker)
    eqs_result = calculate_eqs(ticker, earnings_date)
    ecs_result = calculate_ecs(ticker, earnings_date, ies_result, eqs_result)

    return {
        'ticker': ticker,
        'earnings_date': ecs_result.earnings_date.isoformat() if ecs_result.earnings_date else None,

        # Pre-earnings
        'pre_earnings': {
            'ies': ies_result.ies,
            'regime': ies_result.regime.value if ies_result.regime else None,
            'implied_move_pctl': ies_result.implied_move_pctl,
            'position_scale': ies_result.position_scale,
            'drift_20d': ies_result.drift_20d,
            'iv_pctl': ies_result.iv_pctl,
        },

        # Post-earnings
        'post_earnings': {
            'eqs': eqs_result.eqs,
            'event_z': eqs_result.event_z,
            'eps_surprise_pct': eqs_result.eps_surprise_pct,
            'revenue_surprise_pct': eqs_result.revenue_surprise_pct,
            'guidance_direction': eqs_result.guidance_direction.value if eqs_result.guidance_direction else None,
        },

        # Clearance
        'clearance': {
            'ecs_category': ecs_result.ecs_category.value if ecs_result.ecs_category else None,
            'cleared_bar': ecs_result.cleared_bar,
            'required_z': ecs_result.required_z,
            'clearance_margin': ecs_result.clearance_margin,
            'score_adjustment': ecs_result.score_adjustment,
        },

        # Quality
        'data_quality': ecs_result.data_quality.value if ecs_result.data_quality else 'LOW',
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("Phase 9: ECS Calculator Test")
    print("=" * 60)

    test_tickers = ["NVDA", "AAPL", "TSLA"]

    for ticker in test_tickers:
        print(f"\n--- {ticker} ---")
        result = calculate_ecs(ticker)

        print(f"ECS: {result.ecs_category.value}")
        print(f"Cleared Bar: {'YES' if result.cleared_bar else 'NO'}")
        print(f"Event Z: {result.event_z:+.2f}" if result.event_z else "Event Z: N/A")
        print(f"Required Z: {result.required_z:+.2f}")
        print(f"Margin: {result.clearance_margin:+.2f}" if result.clearance_margin else "Margin: N/A")
        print(f"Score Adjustment: {result.score_adjustment:+d}")

    # Print full summary for one ticker
    print("\n" + "=" * 60)
    print("Full AI Summary for NVDA:")
    print("=" * 60)
    print(get_ecs_summary_for_ai("NVDA"))