"""
IES Calculator - Implied Expectations Score

Combines all pre-earnings inputs into a single IES (0-100) that estimates
what expectations are priced into the stock before earnings.

IES Components:
- Phase 4: Price Drift (drift_20d, relative_drift) - 30%
- Phase 5: Options (implied_move_pctl, iv_pctl, skew_shift) - 45%
- Phase 6: Analyst (revision_score, estimate_momentum) - 25%

Higher IES = Higher expectations priced in = Harder to beat

Usage:
    from src.analytics.earnings_intelligence import calculate_ies

    result = calculate_ies("NVDA")
    print(f"IES: {result.ies}/100 - Regime: {result.regime}")

Author: Alpha Research Platform
Phase: 7 of Earnings Intelligence System
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import os

from src.utils.logging import get_logger

# Import from previous phases
from src.analytics.earnings_intelligence.models import (
    IESComponents,
    ExpectationsRegime,
    DataQuality,
    PositionScaling,
    EarningsIntelligenceResult,
    IES_WEIGHTS,
    COMPUTE_WINDOW_START,
    COMPUTE_WINDOW_END,
    ACTION_WINDOW_START,
    ACTION_WINDOW_END,
)

from src.analytics.earnings_intelligence.windows import (
    get_earnings_info,
    is_in_compute_window,
    is_in_action_window,
)

from src.analytics.earnings_intelligence.drift import (
    calculate_all_drift_metrics,
    normalize_drift_to_score,
)

from src.analytics.earnings_intelligence.options_inputs import (
    calculate_all_options_inputs,
    normalize_implied_move_to_score,
    normalize_iv_to_score,
    normalize_skew_to_score,
)

from src.analytics.earnings_intelligence.analyst_inputs import (
    calculate_all_analyst_inputs,
    normalize_revision_to_score,
    normalize_momentum_to_score,
)

logger = get_logger(__name__)


# ============================================================================
# UPDATED WEIGHTS (Phase 7)
# ============================================================================

# These weights are updated from the original IES_WEIGHTS to reflect
# the redesigned Phase 6 (analyst inputs instead of sentiment/news)
IES_WEIGHTS_V2 = {
    # Phase 4: Price Drift (30%)
    'drift_20d': 0.15,
    'rel_drift_20d': 0.15,

    # Phase 5: Options (45%)
    'implied_move_pctl': 0.20,
    'iv_pctl': 0.15,
    'skew_shift': 0.10,

    # Phase 6: Analyst Expectations (25%)
    'revision_score': 0.15,
    'estimate_momentum': 0.10,
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class IESCalculationResult:
    """Complete IES calculation result with all components."""

    # Core outputs
    ticker: str
    ies: Optional[float]                    # Final IES score (0-100)
    regime: ExpectationsRegime              # HYPED, FEARED, VOLATILE, NORMAL
    position_scale: float                   # Recommended position size (0.2-1.0)

    # Component scores (all 0-100)
    drift_score: Optional[float]
    rel_drift_score: Optional[float]
    implied_move_score: Optional[float]
    iv_score: Optional[float]
    skew_score: Optional[float]
    revision_score: Optional[float]
    momentum_score: Optional[float]

    # Raw values (for debugging/display)
    drift_20d: Optional[float]              # Raw drift %
    rel_drift_20d: Optional[float]          # Raw relative drift %
    implied_move_pct: Optional[float]       # Raw implied move %
    implied_move_pctl: Optional[float]      # Implied move percentile
    iv_pctl: Optional[float]                # IV percentile
    skew_shift: Optional[float]             # Raw skew shift
    analyst_revision: Optional[float]       # Raw revision score
    analyst_momentum: Optional[float]       # Raw momentum score

    # Earnings window info
    earnings_date: Optional[date]
    days_to_earnings: Optional[int]
    in_compute_window: bool
    in_action_window: bool

    # Data quality
    data_quality: DataQuality
    missing_inputs: List[str]
    input_count: int                        # Number of inputs available
    max_inputs: int = 7                     # Total possible inputs

    # Metadata
    computed_at: datetime = field(default_factory=datetime.now)
    data_sources: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'ticker': self.ticker,
            'ies': self.ies,
            'regime': self.regime.value if self.regime else None,
            'position_scale': self.position_scale,

            'drift_score': self.drift_score,
            'rel_drift_score': self.rel_drift_score,
            'implied_move_score': self.implied_move_score,
            'iv_score': self.iv_score,
            'skew_score': self.skew_score,
            'revision_score': self.revision_score,
            'momentum_score': self.momentum_score,

            'drift_20d': self.drift_20d,
            'rel_drift_20d': self.rel_drift_20d,
            'implied_move_pct': self.implied_move_pct,
            'implied_move_pctl': self.implied_move_pctl,
            'iv_pctl': self.iv_pctl,
            'skew_shift': self.skew_shift,
            'analyst_revision': self.analyst_revision,
            'analyst_momentum': self.analyst_momentum,

            'earnings_date': self.earnings_date.isoformat() if self.earnings_date else None,
            'days_to_earnings': self.days_to_earnings,
            'in_compute_window': self.in_compute_window,
            'in_action_window': self.in_action_window,

            'data_quality': self.data_quality.value if self.data_quality else None,
            'missing_inputs': self.missing_inputs,
            'input_count': self.input_count,
            'max_inputs': self.max_inputs,

            'computed_at': self.computed_at.isoformat() if self.computed_at else None,
            'data_sources': self.data_sources,
        }

    def get_ai_summary(self) -> str:
        """Generate formatted summary for AI context."""
        lines = [
            f"\n{'='*55}",
            f"IMPLIED EXPECTATIONS SCORE (IES): {self.ticker}",
            f"{'='*55}",
        ]

        # Main score
        if self.ies is not None:
            ies_level = self._get_ies_level()
            lines.append(f"📊 IES: {self.ies:.0f}/100 ({ies_level})")
            lines.append(f"📈 Regime: {self.regime.value}")
            lines.append(f"📉 Position Scale: {self.position_scale:.0%}")
        else:
            lines.append("📊 IES: Unable to calculate (insufficient data)")

        # Earnings window
        lines.append("")
        if self.earnings_date:
            lines.append(f"📅 Earnings Date: {self.earnings_date}")
            if self.days_to_earnings is not None:
                if self.days_to_earnings > 0:
                    lines.append(f"   Days Until: {self.days_to_earnings}")
                elif self.days_to_earnings == 0:
                    lines.append("   ⚠️ EARNINGS TODAY")
                else:
                    lines.append(f"   Days Since: {abs(self.days_to_earnings)}")

            if self.in_action_window:
                lines.append("   🎯 IN ACTION WINDOW")
            elif self.in_compute_window:
                lines.append("   📊 In Compute Window")
        else:
            lines.append("📅 No upcoming earnings found")

        # Component breakdown
        lines.append("")
        lines.append("📋 COMPONENT SCORES:")

        # Price Drift (30%)
        lines.append("   [Price Drift - 30%]")
        if self.drift_score is not None:
            lines.append(f"     Drift 20d: {self.drift_score:.0f}/100 ({self.drift_20d:+.1%})")
        else:
            lines.append("     Drift 20d: N/A")
        if self.rel_drift_score is not None:
            lines.append(f"     Rel Drift: {self.rel_drift_score:.0f}/100 ({self.rel_drift_20d:+.1%})")
        else:
            lines.append("     Rel Drift: N/A")

        # Options (45%)
        lines.append("   [Options - 45%]")
        if self.implied_move_score is not None:
            lines.append(f"     Impl Move: {self.implied_move_score:.0f}/100 ({self.implied_move_pctl:.0f}th pctl)")
        else:
            lines.append("     Impl Move: N/A")
        if self.iv_score is not None:
            lines.append(f"     IV Pctl:   {self.iv_score:.0f}/100 ({self.iv_pctl:.0f}th pctl)")
        else:
            lines.append("     IV Pctl: N/A")
        if self.skew_score is not None:
            lines.append(f"     Skew:      {self.skew_score:.0f}/100 ({self.skew_shift:+.3f})")
        else:
            lines.append("     Skew: N/A")

        # Analyst (25%)
        lines.append("   [Analyst - 25%]")
        if self.revision_score is not None:
            lines.append(f"     Revision:  {self.revision_score:.0f}/100")
        else:
            lines.append("     Revision: N/A")
        if self.momentum_score is not None:
            lines.append(f"     Momentum:  {self.momentum_score:.0f}/100")
        else:
            lines.append("     Momentum: N/A")

        # Data quality
        lines.append("")
        lines.append(f"📊 Data Quality: {self.data_quality.value} ({self.input_count}/{self.max_inputs} inputs)")
        if self.missing_inputs:
            lines.append(f"   Missing: {', '.join(self.missing_inputs)}")

        # Interpretation
        lines.append("")
        lines.append("💡 INTERPRETATION:")
        lines.extend(self._get_interpretation())

        return "\n".join(lines)

    def _get_ies_level(self) -> str:
        """Get human-readable IES level."""
        if self.ies is None:
            return "Unknown"
        if self.ies >= 80:
            return "EXTREME - Very hard to beat"
        elif self.ies >= 65:
            return "HIGH - Hard to beat"
        elif self.ies >= 50:
            return "ELEVATED - Above average bar"
        elif self.ies >= 35:
            return "NORMAL - Standard expectations"
        elif self.ies >= 20:
            return "LOW - Easy to beat"
        else:
            return "VERY LOW - Very easy to beat"

    def _get_interpretation(self) -> List[str]:
        """Generate interpretation bullets."""
        interp = []

        if self.ies is None:
            return ["   • Insufficient data for analysis"]

        # Main IES interpretation
        if self.ies >= 70:
            interp.append("   • HIGH expectations priced in")
            interp.append("   • Stock needs exceptional results to move higher")
            interp.append("   • Consider reducing position size or waiting")
        elif self.ies >= 50:
            interp.append("   • MODERATE expectations priced in")
            interp.append("   • Beat consensus to see positive reaction")
        else:
            interp.append("   • LOW expectations priced in")
            interp.append("   • Easier to surprise positively")
            interp.append("   • Potential for strong reaction on good results")

        # Regime-specific advice
        if self.regime == ExpectationsRegime.HYPED:
            interp.append("   • ⚠️ HYPED: Even a beat may disappoint")
        elif self.regime == ExpectationsRegime.FEARED:
            interp.append("   • 📉 FEARED: Low bar, but sentiment is negative")
        elif self.regime == ExpectationsRegime.VOLATILE:
            interp.append("   • ⚡ VOLATILE: Big move expected, direction uncertain")

        return interp


# ============================================================================
# MAIN CALCULATION FUNCTIONS
# ============================================================================

def calculate_ies(ticker: str) -> IESCalculationResult:
    """
    Calculate the complete Implied Expectations Score for a ticker.

    This is the main entry point for Phase 7. It:
    1. Gathers all inputs from Phases 4-6
    2. Normalizes each to 0-100 scale
    3. Applies weights to compute final IES
    4. Classifies the expectations regime
    5. Computes recommended position scale

    Args:
        ticker: Stock symbol

    Returns:
        IESCalculationResult with all components and final IES
    """
    logger.info(f"{ticker}: Calculating IES...")

    # Initialize tracking
    missing_inputs = []
    data_sources = {}

    # =========================================================================
    # STEP 1: Get earnings window info
    # =========================================================================
    earnings_info = get_earnings_info(ticker)

    earnings_date = earnings_info.earnings_date if earnings_info else None
    days_to_earnings = earnings_info.days_to_earnings if earnings_info else None
    in_compute = is_in_compute_window(ticker)
    in_action = is_in_action_window(ticker)

    # =========================================================================
    # STEP 2: Gather all inputs from previous phases
    # =========================================================================

    # Phase 4: Drift metrics
    drift_inputs = calculate_all_drift_metrics(ticker)

    drift_20d = drift_inputs.get('drift_20d')
    rel_drift_20d = drift_inputs.get('relative_drift')

    if drift_20d is None:
        missing_inputs.append('drift_20d')
    if rel_drift_20d is None:
        missing_inputs.append('rel_drift_20d')

    data_sources['drift'] = 'yfinance' if drift_20d else 'none'

    # Phase 5: Options metrics
    options_inputs = calculate_all_options_inputs(ticker)

    implied_move_pct = options_inputs.get('implied_move_pct')
    implied_move_pctl = options_inputs.get('implied_move_pctl')
    iv_pctl = options_inputs.get('iv_pctl')
    skew_shift = options_inputs.get('skew_shift')

    if implied_move_pctl is None:
        missing_inputs.append('implied_move_pctl')
    if iv_pctl is None:
        missing_inputs.append('iv_pctl')
    if skew_shift is None:
        missing_inputs.append('skew_shift')

    data_sources['options'] = 'yfinance' if implied_move_pctl else 'none'

    # Phase 6: Analyst metrics
    analyst_inputs = calculate_all_analyst_inputs(ticker)

    analyst_revision = analyst_inputs.get('revision_score')
    analyst_momentum = analyst_inputs.get('estimate_momentum_score')

    if analyst_revision is None:
        missing_inputs.append('revision_score')
    if analyst_momentum is None:
        missing_inputs.append('estimate_momentum')

    data_sources['analyst'] = analyst_inputs.get('data_source', 'yfinance')

    # =========================================================================
    # STEP 3: Normalize all inputs to 0-100 scores
    # =========================================================================

    # Drift scores (higher drift = higher expectations = higher score)
    drift_score = normalize_drift_to_score(drift_20d) if drift_20d is not None else None
    rel_drift_score = normalize_drift_to_score(rel_drift_20d) if rel_drift_20d is not None else None

    # Options scores
    implied_move_score = normalize_implied_move_to_score(implied_move_pctl) if implied_move_pctl is not None else None
    iv_score = normalize_iv_to_score(iv_pctl) if iv_pctl is not None else None
    skew_score = normalize_skew_to_score(skew_shift) if skew_shift is not None else None

    # Analyst scores (already 0-100)
    revision_score = normalize_revision_to_score(analyst_revision) if analyst_revision is not None else None

    # Momentum score: INVERT it for IES
    # High momentum score = easier to beat = LOWER expectations
    # So for IES: 100 - momentum_score
    momentum_score = None
    if analyst_momentum is not None:
        momentum_score = 100 - normalize_momentum_to_score(analyst_momentum)

    # =========================================================================
    # STEP 4: Calculate weighted IES
    # =========================================================================

    scores = {
        'drift_20d': drift_score,
        'rel_drift_20d': rel_drift_score,
        'implied_move_pctl': implied_move_score,
        'iv_pctl': iv_score,
        'skew_shift': skew_score,
        'revision_score': revision_score,
        'estimate_momentum': momentum_score,
    }

    ies, input_count = _calculate_weighted_ies(scores, IES_WEIGHTS_V2)

    # =========================================================================
    # STEP 5: Classify regime
    # =========================================================================

    regime = ExpectationsRegime.classify(
        ies=ies,
        implied_move_pctl=implied_move_pctl or 50,
        drift_20d=drift_20d or 0
    )

    # =========================================================================
    # STEP 6: Calculate position scale
    # =========================================================================

    position_scaling = PositionScaling.calculate(
        ies=ies or 50,
        implied_move_pctl=implied_move_pctl or 50
    )
    position_scale = position_scaling.final_scale

    # =========================================================================
    # STEP 7: Assess data quality
    # =========================================================================

    data_quality = _assess_ies_data_quality(missing_inputs, input_count)

    # =========================================================================
    # STEP 8: Build result
    # =========================================================================

    result = IESCalculationResult(
        ticker=ticker,
        ies=ies,
        regime=regime,
        position_scale=position_scale,

        drift_score=drift_score,
        rel_drift_score=rel_drift_score,
        implied_move_score=implied_move_score,
        iv_score=iv_score,
        skew_score=skew_score,
        revision_score=revision_score,
        momentum_score=momentum_score,

        drift_20d=drift_20d,
        rel_drift_20d=rel_drift_20d,
        implied_move_pct=implied_move_pct,
        implied_move_pctl=implied_move_pctl,
        iv_pctl=iv_pctl,
        skew_shift=skew_shift,
        analyst_revision=analyst_revision,
        analyst_momentum=analyst_momentum,

        earnings_date=earnings_date,
        days_to_earnings=days_to_earnings,
        in_compute_window=in_compute,
        in_action_window=in_action,

        data_quality=data_quality,
        missing_inputs=missing_inputs,
        input_count=input_count,

        data_sources=data_sources,
    )

    ies_str = f"{ies:.0f}" if ies is not None else "N/A"
    logger.info(f"{ticker}: IES={ies_str}, Regime={regime.value}, Quality={data_quality.value}")

    return result


def _calculate_weighted_ies(scores: Dict[str, Optional[float]],
                            weights: Dict[str, float]) -> Tuple[Optional[float], int]:
    """
    Calculate weighted IES from component scores.

    Handles missing inputs by redistributing weights proportionally.

    Args:
        scores: Dict of component name -> score (0-100 or None)
        weights: Dict of component name -> weight (sum to 1.0)

    Returns:
        Tuple of (final_ies, input_count)
    """
    available_scores = {k: v for k, v in scores.items() if v is not None}
    input_count = len(available_scores)

    if input_count == 0:
        return None, 0

    # Redistribute weights for available inputs
    available_weights = {k: weights[k] for k in available_scores.keys()}
    total_available_weight = sum(available_weights.values())

    if total_available_weight == 0:
        return None, 0

    # Normalize weights to sum to 1.0
    normalized_weights = {k: v / total_available_weight for k, v in available_weights.items()}

    # Calculate weighted average
    ies = sum(available_scores[k] * normalized_weights[k] for k in available_scores.keys())

    return max(0, min(100, ies)), input_count


def _assess_ies_data_quality(missing_inputs: List[str], input_count: int) -> DataQuality:
    """
    Assess IES data quality based on available inputs.

    Args:
        missing_inputs: List of missing input names
        input_count: Number of available inputs

    Returns:
        DataQuality level
    """
    # Critical: Options data (most predictive)
    options_missing = any(x in missing_inputs for x in ['implied_move_pctl', 'iv_pctl'])

    # Important: Drift data
    drift_missing = 'drift_20d' in missing_inputs

    if input_count >= 6:
        return DataQuality.HIGH
    elif input_count >= 4 and not options_missing:
        return DataQuality.MEDIUM
    elif input_count >= 3:
        return DataQuality.MEDIUM
    else:
        return DataQuality.LOW


# ============================================================================
# BATCH CALCULATION
# ============================================================================

def calculate_ies_batch(tickers: List[str],
                        progress_callback=None) -> Dict[str, IESCalculationResult]:
    """
    Calculate IES for multiple tickers.

    Args:
        tickers: List of ticker symbols
        progress_callback: Optional callback(current, total, ticker)

    Returns:
        Dict of ticker -> IESCalculationResult
    """
    results = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        try:
            results[ticker] = calculate_ies(ticker)
        except Exception as e:
            logger.error(f"{ticker}: Error calculating IES: {e}")
            # Create minimal result with error
            results[ticker] = IESCalculationResult(
                ticker=ticker,
                ies=None,
                regime=ExpectationsRegime.NORMAL,
                position_scale=1.0,
                drift_score=None,
                rel_drift_score=None,
                implied_move_score=None,
                iv_score=None,
                skew_score=None,
                revision_score=None,
                momentum_score=None,
                drift_20d=None,
                rel_drift_20d=None,
                implied_move_pct=None,
                implied_move_pctl=None,
                iv_pctl=None,
                skew_shift=None,
                analyst_revision=None,
                analyst_momentum=None,
                earnings_date=None,
                days_to_earnings=None,
                in_compute_window=False,
                in_action_window=False,
                data_quality=DataQuality.LOW,
                missing_inputs=['error'],
                input_count=0,
            )

        if progress_callback:
            progress_callback(i + 1, total, ticker)

    return results


# ============================================================================
# DATABASE INTEGRATION
# ============================================================================

def save_ies_to_db(result: IESCalculationResult) -> bool:
    """
    Save IES result to earnings_intelligence table.

    Args:
        result: IESCalculationResult to save

    Returns:
        True if successful
    """
    try:
        from src.db.connection import get_connection

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO earnings_intelligence (
                        ticker, earnings_date, 
                        drift_20d, rel_drift_20d, iv_pctl, implied_move_pct, 
                        implied_move_pctl, skew_shift, revision_score, 
                        ies, ies_compute_timestamp, regime,
                        position_scale, data_quality, missing_inputs
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (ticker, earnings_date) DO UPDATE SET
                        drift_20d = EXCLUDED.drift_20d,
                        rel_drift_20d = EXCLUDED.rel_drift_20d,
                        iv_pctl = EXCLUDED.iv_pctl,
                        implied_move_pct = EXCLUDED.implied_move_pct,
                        implied_move_pctl = EXCLUDED.implied_move_pctl,
                        skew_shift = EXCLUDED.skew_shift,
                        revision_score = EXCLUDED.revision_score,
                        ies = EXCLUDED.ies,
                        ies_compute_timestamp = EXCLUDED.ies_compute_timestamp,
                        regime = EXCLUDED.regime,
                        position_scale = EXCLUDED.position_scale,
                        data_quality = EXCLUDED.data_quality,
                        missing_inputs = EXCLUDED.missing_inputs,
                        updated_at = NOW()
                """, (
                    result.ticker,
                    result.earnings_date,
                    result.drift_20d,
                    result.rel_drift_20d,
                    result.iv_pctl,
                    result.implied_move_pct,
                    result.implied_move_pctl,
                    result.skew_shift,
                    result.analyst_revision,
                    result.ies,
                    result.computed_at,
                    result.regime.value if result.regime else None,
                    result.position_scale,
                    result.data_quality.value if result.data_quality else None,
                    result.missing_inputs,
                ))

        logger.info(f"{result.ticker}: Saved IES to database")
        return True

    except Exception as e:
        logger.error(f"{result.ticker}: Error saving IES to database: {e}")
        return False


def save_ies_to_cache(result: IESCalculationResult) -> bool:
    """
    Save IES result to ies_cache table for quick lookups.

    Args:
        result: IESCalculationResult to cache

    Returns:
        True if successful
    """
    try:
        from src.db.connection import get_connection

        # Cache expires in 4 hours (refresh before next trading session)
        expires_at = datetime.now() + timedelta(hours=4)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ies_cache (
                        ticker, ies, regime, implied_move_pctl, position_scale,
                        data_quality, earnings_date, days_to_earnings,
                        in_compute_window, in_action_window, computed_at, expires_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (ticker) DO UPDATE SET
                        ies = EXCLUDED.ies,
                        regime = EXCLUDED.regime,
                        implied_move_pctl = EXCLUDED.implied_move_pctl,
                        position_scale = EXCLUDED.position_scale,
                        data_quality = EXCLUDED.data_quality,
                        earnings_date = EXCLUDED.earnings_date,
                        days_to_earnings = EXCLUDED.days_to_earnings,
                        in_compute_window = EXCLUDED.in_compute_window,
                        in_action_window = EXCLUDED.in_action_window,
                        computed_at = EXCLUDED.computed_at,
                        expires_at = EXCLUDED.expires_at
                """, (
                    result.ticker,
                    result.ies,
                    result.regime.value if result.regime else None,
                    result.implied_move_pctl,
                    result.position_scale,
                    result.data_quality.value if result.data_quality else None,
                    result.earnings_date,
                    result.days_to_earnings,
                    result.in_compute_window,
                    result.in_action_window,
                    result.computed_at,
                    expires_at,
                ))

        return True

    except Exception as e:
        logger.error(f"{result.ticker}: Error saving IES to cache: {e}")
        return False


def get_ies_from_cache(ticker: str) -> Optional[IESCalculationResult]:
    """
    Get IES from cache if available and not expired.

    Args:
        ticker: Stock symbol

    Returns:
        IESCalculationResult if cached and valid, else None
    """
    try:
        from src.db.connection import get_connection

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ies, regime, implied_move_pctl, position_scale,
                           data_quality, earnings_date, days_to_earnings,
                           in_compute_window, in_action_window, computed_at
                    FROM ies_cache
                    WHERE ticker = %s AND expires_at > NOW()
                """, (ticker,))

                row = cur.fetchone()

                if row:
                    return IESCalculationResult(
                        ticker=ticker,
                        ies=row[0],
                        regime=ExpectationsRegime(row[1]) if row[1] else ExpectationsRegime.NORMAL,
                        position_scale=row[3] or 1.0,
                        drift_score=None,
                        rel_drift_score=None,
                        implied_move_score=None,
                        iv_score=None,
                        skew_score=None,
                        revision_score=None,
                        momentum_score=None,
                        drift_20d=None,
                        rel_drift_20d=None,
                        implied_move_pct=None,
                        implied_move_pctl=row[2],
                        iv_pctl=None,
                        skew_shift=None,
                        analyst_revision=None,
                        analyst_momentum=None,
                        earnings_date=row[5],
                        days_to_earnings=row[6],
                        in_compute_window=row[7] or False,
                        in_action_window=row[8] or False,
                        data_quality=DataQuality(row[4]) if row[4] else DataQuality.LOW,
                        missing_inputs=[],
                        input_count=0,
                        computed_at=row[9],
                        data_sources={'source': 'cache'},
                    )

        return None

    except Exception as e:
        logger.debug(f"{ticker}: Cache lookup failed: {e}")
        return None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_ies_summary_for_ai(ticker: str) -> str:
    """
    Get IES summary formatted for AI context.

    Args:
        ticker: Stock symbol

    Returns:
        Formatted string for AI consumption
    """
    result = calculate_ies(ticker)
    return result.get_ai_summary()


def get_ies_for_screener(ticker: str) -> Dict[str, Any]:
    """
    Get IES data formatted for screener integration.

    Args:
        ticker: Stock symbol

    Returns:
        Dict with screener-relevant fields
    """
    result = calculate_ies(ticker)

    return {
        'ies': result.ies,
        'ies_level': result._get_ies_level() if result.ies else 'Unknown',
        'regime': result.regime.value if result.regime else 'NORMAL',
        'position_scale': result.position_scale,
        'days_to_earnings': result.days_to_earnings,
        'in_action_window': result.in_action_window,
        'data_quality': result.data_quality.value if result.data_quality else 'LOW',
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("Phase 7: IES Calculator Test")
    print("=" * 60)

    test_tickers = ["NVDA", "AAPL", "TSLA"]

    for ticker in test_tickers:
        print(f"\n--- {ticker} ---")
        result = calculate_ies(ticker)

        print(f"IES: {result.ies:.0f}/100" if result.ies else "IES: N/A")
        print(f"Regime: {result.regime.value}")
        print(f"Position Scale: {result.position_scale:.0%}")
        print(f"Data Quality: {result.data_quality.value} ({result.input_count}/7 inputs)")

        if result.in_action_window:
            print(f"⚠️ IN ACTION WINDOW - {result.days_to_earnings} days to earnings")

    # Print full summary for one ticker
    print("\n" + "=" * 60)
    print("Full AI Summary for NVDA:")
    print("=" * 60)
    print(get_ies_summary_for_ai("NVDA"))