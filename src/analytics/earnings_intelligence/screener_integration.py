"""
Earnings Intelligence - Screener Integration

Integrates the Earnings Intelligence System (IES/EQS/ECS) into the daily
screener workflow. Provides:

1. Pre-screener enrichment: Add IES data to stocks approaching earnings
2. Score adjustments: Apply ECS-based score adjustments
3. Risk flags: Flag stocks in earnings windows
4. Position sizing: Provide recommended position scales

Usage:
    from src.analytics.earnings_intelligence.screener_integration import (
        enrich_screener_with_earnings,
        get_earnings_adjustments,
        get_tickers_in_earnings_window,
    )

    # During screener run
    adjustments = get_earnings_adjustments(tickers)
    for ticker, adj in adjustments.items():
        total_score += adj['score_adjustment']

Author: Alpha Research Platform
Phase: 10 of Earnings Intelligence System
"""

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from src.utils.logging import get_logger

# Import from previous phases
from src.analytics.earnings_intelligence.windows import (
    get_earnings_info,
    is_in_compute_window,
    is_in_action_window,
    get_days_to_earnings,
)

from src.analytics.earnings_intelligence.ies_calculator import (
    calculate_ies,
    IESCalculationResult,
)

from src.analytics.earnings_intelligence.eqs_calculator import (
    calculate_eqs,
    EQSCalculationResult,
)

from src.analytics.earnings_intelligence.ecs_calculator import (
    calculate_ecs,
    ECSCalculationResult,
    get_full_earnings_analysis,
)

from src.analytics.earnings_intelligence.models import (
    ECSCategory,
    ExpectationsRegime,
    DataQuality,
)

logger = get_logger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Score adjustment weights
SCORE_ADJUSTMENTS = {
    ECSCategory.STRONG_BEAT: 18,
    ECSCategory.BEAT: 8,
    ECSCategory.INLINE: 0,
    ECSCategory.MISS: -12,
    ECSCategory.STRONG_MISS: -22,
    ECSCategory.UNKNOWN: 0,
}

# Pre-earnings adjustments (caution flags)
PRE_EARNINGS_ADJUSTMENT = {
    'action_window': -5,  # Within 5 days of earnings
    'earnings_day': -10,  # On earnings day
    'high_ies': -5,  # IES > 70 (high expectations)
    'hyped_regime': -8,  # HYPED regime
}

# Position scale thresholds
POSITION_SCALE_THRESHOLDS = {
    'full': 1.0,  # Normal position
    'reduced': 0.7,  # Reduced due to earnings
    'minimal': 0.5,  # High risk setup
    'avoid': 0.2,  # Very high risk
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EarningsEnrichment:
    """Earnings intelligence data for screener enrichment."""
    ticker: str

    # Window status
    has_upcoming_earnings: bool
    days_to_earnings: Optional[int]
    earnings_date: Optional[date]
    in_action_window: bool
    in_compute_window: bool

    # Pre-earnings (IES)
    ies: Optional[float]
    regime: Optional[ExpectationsRegime]
    implied_move_pctl: Optional[float]

    # Post-earnings (EQS/ECS) - if applicable
    eqs: Optional[float]
    ecs_category: Optional[ECSCategory]
    cleared_bar: Optional[bool]

    # Adjustments
    score_adjustment: int
    position_scale: float
    risk_flags: List[str]

    # Data quality
    data_quality: DataQuality

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'ticker': self.ticker,
            'has_upcoming_earnings': self.has_upcoming_earnings,
            'days_to_earnings': self.days_to_earnings,
            'earnings_date': self.earnings_date.isoformat() if self.earnings_date else None,
            'in_action_window': self.in_action_window,
            'in_compute_window': self.in_compute_window,
            'ies': self.ies,
            'regime': self.regime.value if self.regime else None,
            'implied_move_pctl': self.implied_move_pctl,
            'eqs': self.eqs,
            'ecs_category': self.ecs_category.value if self.ecs_category else None,
            'cleared_bar': self.cleared_bar,
            'score_adjustment': self.score_adjustment,
            'position_scale': self.position_scale,
            'risk_flags': self.risk_flags,
            'data_quality': self.data_quality.value if self.data_quality else None,
        }


# ============================================================================
# MAIN INTEGRATION FUNCTIONS
# ============================================================================

def enrich_screener_with_earnings(ticker: str) -> EarningsEnrichment:
    """
    Enrich a single ticker with earnings intelligence data.

    This is the main function called during screener processing.

    Args:
        ticker: Stock symbol

    Returns:
        EarningsEnrichment with all relevant data
    """
    risk_flags = []
    score_adjustment = 0
    position_scale = 1.0

    # Get earnings window info
    earnings_info = get_earnings_info(ticker)

    if earnings_info is None:
        # No earnings data available
        return EarningsEnrichment(
            ticker=ticker,
            has_upcoming_earnings=False,
            days_to_earnings=None,
            earnings_date=None,
            in_action_window=False,
            in_compute_window=False,
            ies=None,
            regime=None,
            implied_move_pctl=None,
            eqs=None,
            ecs_category=None,
            cleared_bar=None,
            score_adjustment=0,
            position_scale=1.0,
            risk_flags=[],
            data_quality=DataQuality.LOW,
        )

    # Extract window info
    days_to_earnings = earnings_info.days_to_earnings
    earnings_date = earnings_info.earnings_date
    in_action = is_in_action_window(ticker)
    in_compute = is_in_compute_window(ticker)

    # Determine if we should calculate IES/ECS
    ies_result = None
    eqs_result = None
    ecs_result = None

    if in_compute:
        # In compute window - calculate IES
        try:
            ies_result = calculate_ies(ticker)
        except Exception as e:
            logger.warning(f"{ticker}: Error calculating IES: {e}")

    # Check if we have recent earnings to evaluate (post-earnings)
    if days_to_earnings is not None and days_to_earnings < 0 and days_to_earnings >= -5:
        # Within 5 days after earnings - calculate ECS
        try:
            ecs_result = calculate_ecs(ticker, earnings_date)
            eqs_result = ecs_result.eqs_result
        except Exception as e:
            logger.warning(f"{ticker}: Error calculating ECS: {e}")

    # Calculate score adjustments

    # Pre-earnings adjustments
    if days_to_earnings is not None and days_to_earnings >= 0:
        if days_to_earnings == 0:
            # Earnings TODAY
            score_adjustment += PRE_EARNINGS_ADJUSTMENT['earnings_day']
            risk_flags.append("EARNINGS_TODAY")
        elif in_action:
            # In action window (within 5 days)
            score_adjustment += PRE_EARNINGS_ADJUSTMENT['action_window']
            risk_flags.append("EARNINGS_SOON")

        # IES-based adjustments
        if ies_result and ies_result.ies is not None:
            if ies_result.ies >= 70:
                score_adjustment += PRE_EARNINGS_ADJUSTMENT['high_ies']
                risk_flags.append("HIGH_EXPECTATIONS")

            if ies_result.regime == ExpectationsRegime.HYPED:
                score_adjustment += PRE_EARNINGS_ADJUSTMENT['hyped_regime']
                risk_flags.append("HYPED_REGIME")

            # Use IES position scale
            position_scale = ies_result.position_scale

    # Post-earnings adjustments
    if ecs_result and ecs_result.ecs_category != ECSCategory.UNKNOWN:
        score_adjustment += ecs_result.score_adjustment

        if ecs_result.ecs_category == ECSCategory.STRONG_MISS:
            risk_flags.append("EARNINGS_MISS")
        elif ecs_result.ecs_category == ECSCategory.STRONG_BEAT:
            risk_flags.append("EARNINGS_BEAT")

    # Build result
    return EarningsEnrichment(
        ticker=ticker,
        has_upcoming_earnings=earnings_date is not None,
        days_to_earnings=days_to_earnings,
        earnings_date=earnings_date,
        in_action_window=in_action,
        in_compute_window=in_compute,
        ies=ies_result.ies if ies_result else None,
        regime=ies_result.regime if ies_result else None,
        implied_move_pctl=ies_result.implied_move_pctl if ies_result else None,
        eqs=eqs_result.eqs if eqs_result else None,
        ecs_category=ecs_result.ecs_category if ecs_result else None,
        cleared_bar=ecs_result.cleared_bar if ecs_result else None,
        score_adjustment=score_adjustment,
        position_scale=position_scale,
        risk_flags=risk_flags,
        data_quality=ies_result.data_quality if ies_result else DataQuality.LOW,
    )


def get_earnings_adjustments(tickers: List[str],
                             progress_callback=None) -> Dict[str, Dict[str, Any]]:
    """
    Get earnings adjustments for multiple tickers.

    Returns a dict with score_adjustment and position_scale for each ticker.

    Args:
        tickers: List of ticker symbols
        progress_callback: Optional callback(current, total, ticker)

    Returns:
        Dict of ticker -> {score_adjustment, position_scale, risk_flags, ...}
    """
    results = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        try:
            enrichment = enrich_screener_with_earnings(ticker)
            results[ticker] = {
                'score_adjustment': enrichment.score_adjustment,
                'position_scale': enrichment.position_scale,
                'risk_flags': enrichment.risk_flags,
                'days_to_earnings': enrichment.days_to_earnings,
                'in_action_window': enrichment.in_action_window,
                'ies': enrichment.ies,
                'ecs_category': enrichment.ecs_category.value if enrichment.ecs_category else None,
            }
        except Exception as e:
            logger.error(f"{ticker}: Error getting earnings adjustment: {e}")
            results[ticker] = {
                'score_adjustment': 0,
                'position_scale': 1.0,
                'risk_flags': [],
                'days_to_earnings': None,
                'in_action_window': False,
                'ies': None,
                'ecs_category': None,
            }

        if progress_callback:
            progress_callback(i + 1, total, ticker)

    return results


def get_tickers_in_earnings_window(tickers: List[str],
                                   window_type: str = 'action') -> List[str]:
    """
    Filter tickers to those in an earnings window.

    Args:
        tickers: List of ticker symbols
        window_type: 'action' (5 days) or 'compute' (10 days)

    Returns:
        List of tickers in the specified window
    """
    in_window = []

    for ticker in tickers:
        try:
            if window_type == 'action':
                if is_in_action_window(ticker):
                    in_window.append(ticker)
            else:
                if is_in_compute_window(ticker):
                    in_window.append(ticker)
        except Exception as e:
            logger.debug(f"{ticker}: Error checking earnings window: {e}")

    return in_window


def get_upcoming_earnings(tickers: List[str],
                          days_ahead: int = 14) -> List[Dict[str, Any]]:
    """
    Get list of upcoming earnings for a set of tickers.

    Args:
        tickers: List of ticker symbols
        days_ahead: How many days to look ahead

    Returns:
        List of dicts with ticker, earnings_date, days_to_earnings
    """
    upcoming = []

    for ticker in tickers:
        try:
            earnings_info = get_earnings_info(ticker)

            if earnings_info and earnings_info.days_to_earnings is not None:
                if 0 <= earnings_info.days_to_earnings <= days_ahead:
                    upcoming.append({
                        'ticker': ticker,
                        'earnings_date': earnings_info.earnings_date,
                        'days_to_earnings': earnings_info.days_to_earnings,
                    })
        except Exception as e:
            logger.debug(f"{ticker}: Error getting earnings info: {e}")

    # Sort by days to earnings
    upcoming.sort(key=lambda x: x['days_to_earnings'])

    return upcoming


# ============================================================================
# BATCH PROCESSING FOR SCREENER
# ============================================================================

def process_earnings_for_screener(tickers: List[str],
                                  fast_mode: bool = True,
                                  progress_callback=None) -> pd.DataFrame:
    """
    Process earnings intelligence for all screener tickers.

    In fast_mode, only processes tickers in earnings windows.

    Args:
        tickers: List of ticker symbols
        fast_mode: Only process tickers near earnings
        progress_callback: Optional callback(current, total, ticker)

    Returns:
        DataFrame with earnings data for all tickers
    """
    results = []

    if fast_mode:
        # First pass: identify tickers in earnings windows
        tickers_to_process = get_tickers_in_earnings_window(tickers, 'compute')
        logger.info(f"Fast mode: Processing {len(tickers_to_process)}/{len(tickers)} tickers in earnings windows")
    else:
        tickers_to_process = tickers

    total = len(tickers_to_process)

    for i, ticker in enumerate(tickers_to_process):
        try:
            enrichment = enrich_screener_with_earnings(ticker)
            results.append(enrichment.to_dict())
        except Exception as e:
            logger.error(f"{ticker}: Error processing earnings: {e}")
            results.append({
                'ticker': ticker,
                'has_upcoming_earnings': False,
                'score_adjustment': 0,
                'position_scale': 1.0,
                'risk_flags': [],
                'data_quality': 'LOW',
            })

        if progress_callback:
            progress_callback(i + 1, total, ticker)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Add tickers not processed (fast_mode)
    if fast_mode:
        missing = set(tickers) - set(tickers_to_process)
        for ticker in missing:
            results.append({
                'ticker': ticker,
                'has_upcoming_earnings': False,
                'days_to_earnings': None,
                'earnings_date': None,
                'in_action_window': False,
                'in_compute_window': False,
                'ies': None,
                'regime': None,
                'implied_move_pctl': None,
                'eqs': None,
                'ecs_category': None,
                'cleared_bar': None,
                'score_adjustment': 0,
                'position_scale': 1.0,
                'risk_flags': [],
                'data_quality': None,
            })
        df = pd.DataFrame(results)

    return df


# ============================================================================
# DATABASE INTEGRATION
# ============================================================================

def save_earnings_enrichment_to_db(enrichments: List[EarningsEnrichment]) -> int:
    """
    Save earnings enrichment data to database.

    Updates the screener_scores table with earnings adjustments.

    Args:
        enrichments: List of EarningsEnrichment objects

    Returns:
        Number of records updated
    """
    try:
        from src.db.connection import get_connection

        count = 0
        today = date.today()

        with get_connection() as conn:
            with conn.cursor() as cur:
                for e in enrichments:
                    if e.score_adjustment != 0:
                        # Update screener_scores with earnings adjustment
                        cur.execute("""
                                    UPDATE screener_scores
                                    SET earnings_adjustment = %s,
                                        total_score         = total_score + %s
                                    WHERE ticker = %s
                                      AND date = %s
                                    """, (
                                        e.score_adjustment,
                                        e.score_adjustment,
                                        e.ticker,
                                        today,
                                    ))
                        count += cur.rowcount

        logger.info(f"Updated {count} screener records with earnings adjustments")
        return count

    except Exception as e:
        logger.error(f"Error saving earnings enrichments: {e}")
        return 0


def get_earnings_summary_for_date(score_date: date = None) -> Dict[str, Any]:
    """
    Get summary of earnings intelligence for a given date.

    Args:
        score_date: Date to summarize (default: today)

    Returns:
        Summary dict with counts and highlights
    """
    if score_date is None:
        score_date = date.today()

    try:
        from src.db.connection import get_engine

        engine = get_engine()

        # Get IES cache data
        query = """
                SELECT ticker, \
                       ies, \
                       regime, \
                       days_to_earnings, \
                       in_action_window,
                       position_scale, \
                       data_quality
                FROM ies_cache
                WHERE expires_at > NOW() \
                """

        df = pd.read_sql(query, engine)

        if df.empty:
            return {
                'date': score_date.isoformat(),
                'tickers_in_window': 0,
                'tickers_in_action': 0,
                'high_ies_count': 0,
                'hyped_count': 0,
                'highlights': [],
            }

        # Calculate summary
        in_action = df[df['in_action_window'] == True]
        high_ies = df[df['ies'] >= 70]
        hyped = df[df['regime'] == 'HYPED']

        # Get highlights (tickers to watch)
        highlights = []

        for _, row in in_action.head(10).iterrows():
            highlights.append({
                'ticker': row['ticker'],
                'days_to_earnings': row['days_to_earnings'],
                'ies': row['ies'],
                'regime': row['regime'],
            })

        return {
            'date': score_date.isoformat(),
            'tickers_in_window': len(df),
            'tickers_in_action': len(in_action),
            'high_ies_count': len(high_ies),
            'hyped_count': len(hyped),
            'highlights': highlights,
        }

    except Exception as e:
        logger.error(f"Error getting earnings summary: {e}")
        return {
            'date': score_date.isoformat(),
            'tickers_in_window': 0,
            'error': str(e),
        }


# ============================================================================
# HELPER FUNCTIONS FOR SCREENER WORKER
# ============================================================================

def apply_earnings_adjustment(base_score: int, ticker: str) -> Tuple[int, List[str]]:
    """
    Apply earnings adjustment to a base score.

    Simple helper for screener integration.

    Args:
        base_score: Original score
        ticker: Stock symbol

    Returns:
        Tuple of (adjusted_score, risk_flags)
    """
    try:
        enrichment = enrich_screener_with_earnings(ticker)
        adjusted = base_score + enrichment.score_adjustment
        return adjusted, enrichment.risk_flags
    except Exception as e:
        logger.debug(f"{ticker}: Error applying earnings adjustment: {e}")
        return base_score, []


def get_position_scale(ticker: str) -> float:
    """
    Get recommended position scale for a ticker.

    Args:
        ticker: Stock symbol

    Returns:
        Position scale (0.2 to 1.0)
    """
    try:
        enrichment = enrich_screener_with_earnings(ticker)
        return enrichment.position_scale
    except Exception as e:
        logger.debug(f"{ticker}: Error getting position scale: {e}")
        return 1.0


def should_flag_for_earnings(ticker: str) -> bool:
    """
    Check if ticker should be flagged for earnings risk.

    Args:
        ticker: Stock symbol

    Returns:
        True if ticker has earnings-related risk
    """
    try:
        return is_in_action_window(ticker)
    except Exception as e:
        return False


# ============================================================================
# SCREENER REPORT GENERATION
# ============================================================================

def generate_earnings_report(tickers: List[str]) -> str:
    """
    Generate a formatted earnings report for screener results.

    Args:
        tickers: List of ticker symbols

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "EARNINGS INTELLIGENCE REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
    ]

    # Get upcoming earnings
    upcoming = get_upcoming_earnings(tickers, days_ahead=14)

    if upcoming:
        lines.append("📅 UPCOMING EARNINGS (Next 14 Days)")
        lines.append("-" * 40)
        for item in upcoming[:20]:
            lines.append(f"  {item['ticker']:6s} | {item['earnings_date']} | {item['days_to_earnings']} days")
        lines.append("")

    # Get tickers in action window
    in_action = get_tickers_in_earnings_window(tickers, 'action')

    if in_action:
        lines.append("⚠️ IN ACTION WINDOW (Within 5 Days)")
        lines.append("-" * 40)

        for ticker in in_action[:15]:
            try:
                enrichment = enrich_screener_with_earnings(ticker)
                ies_str = f"IES={enrichment.ies:.0f}" if enrichment.ies else "IES=N/A"
                regime_str = enrichment.regime.value if enrichment.regime else "N/A"
                adj_str = f"Adj={enrichment.score_adjustment:+d}"

                lines.append(f"  {ticker:6s} | {ies_str:10s} | {regime_str:10s} | {adj_str}")
            except Exception:
                lines.append(f"  {ticker:6s} | Error")

        lines.append("")

    # Summary
    lines.append("📊 SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Tickers analyzed: {len(tickers)}")
    lines.append(f"  Upcoming earnings (14d): {len(upcoming)}")
    lines.append(f"  In action window: {len(in_action)}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("Phase 10: Screener Integration Test")
    print("=" * 60)

    test_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]

    print("\n1. Testing single ticker enrichment...")
    for ticker in test_tickers[:2]:
        enrichment = enrich_screener_with_earnings(ticker)
        print(f"\n  {ticker}:")
        print(f"    Days to Earnings: {enrichment.days_to_earnings}")
        print(f"    In Action Window: {enrichment.in_action_window}")
        print(f"    IES: {enrichment.ies:.0f}" if enrichment.ies else "    IES: N/A")
        print(f"    Score Adjustment: {enrichment.score_adjustment:+d}")
        print(f"    Position Scale: {enrichment.position_scale:.0%}")
        print(f"    Risk Flags: {enrichment.risk_flags}")

    print("\n2. Testing batch adjustments...")
    adjustments = get_earnings_adjustments(test_tickers)
    for ticker, adj in adjustments.items():
        print(f"  {ticker}: adj={adj['score_adjustment']:+d}, scale={adj['position_scale']:.0%}")

    print("\n3. Testing earnings report...")
    report = generate_earnings_report(test_tickers)
    print(report)