"""
Earnings Intelligence - Backtesting Module

Validates the Earnings Intelligence System against historical earnings data.
Measures predictive accuracy of IES/EQS/ECS signals.

Key Metrics:
1. ECS Accuracy: Did BEAT/MISS correctly predict price direction?
2. Regime Performance: How did each regime (HYPED/FEARED/NORMAL) perform?
3. Position Scaling: Did reduced positions avoid losses?
4. Score Adjustment Impact: Did adjustments improve signal quality?

Usage:
    from src.analytics.earnings_intelligence.backtesting import (
        run_backtest,
        BacktestResult,
    )

    result = run_backtest(tickers=["AAPL", "NVDA", "TSLA"], lookback_quarters=8)
    print(result.summary())

Author: Alpha Research Platform
Phase: 12 of Earnings Intelligence System
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from src.utils.logging import get_logger

# Import from previous phases
from src.analytics.earnings_intelligence.models import (
    ECSCategory,
    ExpectationsRegime,
    DataQuality,
)

from src.analytics.earnings_intelligence.eqs_calculator import (
    calculate_eps_z,
    calculate_rev_z,
    calculate_event_z,
    DEFAULT_EPS_SURPRISE_MEAN,
    DEFAULT_EPS_SURPRISE_STD,
    DEFAULT_REV_SURPRISE_MEAN,
    DEFAULT_REV_SURPRISE_STD,
)

from src.analytics.earnings_intelligence.ecs_calculator import (
    calculate_required_z,
)

logger = get_logger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EarningsEvent:
    """Single historical earnings event for backtesting."""
    ticker: str
    earnings_date: date

    # Surprise data
    eps_surprise_pct: Optional[float]
    revenue_surprise_pct: Optional[float]

    # Price reaction (the outcome we're predicting)
    pre_close: Optional[float]          # Close before earnings
    post_open: Optional[float]          # Open after earnings
    post_close: Optional[float]         # Close after earnings
    gap_reaction: Optional[float]       # (post_open - pre_close) / pre_close
    total_reaction: Optional[float]     # (post_close - pre_close) / pre_close

    # Calculated scores
    eps_z: Optional[float] = None
    rev_z: Optional[float] = None
    event_z: Optional[float] = None
    required_z: Optional[float] = None
    ecs_category: Optional[ECSCategory] = None

    # Prediction accuracy
    predicted_direction: Optional[str] = None  # 'positive', 'negative', 'neutral'
    actual_direction: Optional[str] = None
    correct_prediction: Optional[bool] = None


@dataclass
class BacktestResult:
    """Complete backtest results."""

    # Overview
    tickers_tested: List[str]
    total_events: int
    valid_events: int
    date_range: Tuple[date, date]

    # ECS Accuracy
    ecs_accuracy: float                 # Overall accuracy
    ecs_by_category: Dict[str, Dict]    # Accuracy by BEAT/MISS/etc

    # Direction Prediction
    direction_accuracy: float           # Did we predict up/down correctly?
    direction_by_category: Dict[str, float]

    # Regime Performance
    regime_performance: Dict[str, Dict] # Avg return by regime

    # Position Scaling Impact
    scaling_impact: Dict[str, float]    # Return with vs without scaling

    # Detailed events
    events: List[EarningsEvent] = field(default_factory=list)

    # Metadata
    computed_at: datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            "=" * 60,
            "EARNINGS INTELLIGENCE BACKTEST RESULTS",
            "=" * 60,
            "",
            f"Period: {self.date_range[0]} to {self.date_range[1]}",
            f"Tickers: {len(self.tickers_tested)}",
            f"Total Events: {self.total_events}",
            f"Valid Events: {self.valid_events}",
            "",
            "─" * 40,
            "ECS CATEGORY ACCURACY",
            "─" * 40,
        ]

        for cat, stats in self.ecs_by_category.items():
            count = stats.get('count', 0)
            accuracy = stats.get('accuracy', 0)
            avg_return = stats.get('avg_return', 0)
            lines.append(f"  {cat:15s}: {count:3d} events, {accuracy:5.1f}% accurate, {avg_return:+.2f}% avg return")

        lines.extend([
            "",
            f"Overall ECS Accuracy: {self.ecs_accuracy:.1f}%",
            f"Direction Accuracy: {self.direction_accuracy:.1f}%",
            "",
            "─" * 40,
            "REGIME PERFORMANCE",
            "─" * 40,
        ])

        for regime, stats in self.regime_performance.items():
            count = stats.get('count', 0)
            avg_return = stats.get('avg_return', 0)
            win_rate = stats.get('win_rate', 0)
            lines.append(f"  {regime:10s}: {count:3d} events, {avg_return:+.2f}% avg, {win_rate:.1f}% win rate")

        lines.extend([
            "",
            "─" * 40,
            "POSITION SCALING IMPACT",
            "─" * 40,
            f"  Unscaled Return: {self.scaling_impact.get('unscaled', 0):+.2f}%",
            f"  Scaled Return:   {self.scaling_impact.get('scaled', 0):+.2f}%",
            f"  Improvement:     {self.scaling_impact.get('improvement', 0):+.2f}%",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'tickers_tested': self.tickers_tested,
            'total_events': self.total_events,
            'valid_events': self.valid_events,
            'date_range': [d.isoformat() for d in self.date_range],
            'ecs_accuracy': self.ecs_accuracy,
            'ecs_by_category': self.ecs_by_category,
            'direction_accuracy': self.direction_accuracy,
            'direction_by_category': self.direction_by_category,
            'regime_performance': self.regime_performance,
            'scaling_impact': self.scaling_impact,
            'computed_at': self.computed_at.isoformat(),
        }


# ============================================================================
# DATA RETRIEVAL
# ============================================================================

def get_historical_earnings_yfinance(ticker: str,
                                      lookback_quarters: int = 8) -> List[EarningsEvent]:
    """
    Get historical earnings data from yfinance.

    Args:
        ticker: Stock symbol
        lookback_quarters: Number of quarters to look back

    Returns:
        List of EarningsEvent objects
    """
    events = []

    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)

        # Get earnings history
        earnings = stock.earnings_history

        if earnings is None or earnings.empty:
            logger.debug(f"{ticker}: No earnings history available")
            return events

        # Get price history for reaction calculation
        # Need at least 2 years of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_quarters * 100)

        prices = stock.history(start=start_date, end=end_date)

        if prices.empty:
            logger.debug(f"{ticker}: No price history available")
            return events

        # Process each earnings event
        for idx in range(min(lookback_quarters, len(earnings))):
            try:
                row = earnings.iloc[idx]

                # Get earnings date
                if hasattr(row, 'name'):
                    earn_date = pd.to_datetime(row.name).date()
                else:
                    continue

                # Skip future dates
                if earn_date > date.today():
                    continue

                # Calculate surprise
                eps_actual = row.get('epsActual')
                eps_estimate = row.get('epsEstimate')

                eps_surprise_pct = None
                if eps_actual is not None and eps_estimate is not None and eps_estimate != 0:
                    eps_surprise_pct = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100

                # Get price reaction
                pre_close, post_open, post_close = _get_price_reaction(
                    prices, earn_date
                )

                gap_reaction = None
                total_reaction = None

                if pre_close and post_open:
                    gap_reaction = ((post_open - pre_close) / pre_close) * 100

                if pre_close and post_close:
                    total_reaction = ((post_close - pre_close) / pre_close) * 100

                event = EarningsEvent(
                    ticker=ticker,
                    earnings_date=earn_date,
                    eps_surprise_pct=eps_surprise_pct,
                    revenue_surprise_pct=None,  # yfinance doesn't provide this easily
                    pre_close=pre_close,
                    post_open=post_open,
                    post_close=post_close,
                    gap_reaction=gap_reaction,
                    total_reaction=total_reaction,
                )

                events.append(event)

            except Exception as e:
                logger.debug(f"{ticker}: Error processing earnings event: {e}")
                continue

    except Exception as e:
        logger.warning(f"{ticker}: Error getting historical earnings: {e}")

    return events


def _get_price_reaction(prices: pd.DataFrame,
                        earnings_date: date) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Get price reaction around earnings date.

    Returns:
        Tuple of (pre_close, post_open, post_close)
    """
    try:
        # Convert to datetime for comparison
        prices_idx = pd.to_datetime(prices.index).date

        # Find the closest trading day before earnings
        pre_dates = [d for d in prices_idx if d < earnings_date]
        if not pre_dates:
            return None, None, None
        pre_date = max(pre_dates)

        # Find the closest trading day after earnings
        post_dates = [d for d in prices_idx if d > earnings_date]
        if not post_dates:
            return None, None, None
        post_date = min(post_dates)

        # Get prices
        pre_row = prices[pd.to_datetime(prices.index).date == pre_date]
        post_row = prices[pd.to_datetime(prices.index).date == post_date]

        pre_close = float(pre_row['Close'].iloc[0]) if not pre_row.empty else None
        post_open = float(post_row['Open'].iloc[0]) if not post_row.empty else None
        post_close = float(post_row['Close'].iloc[0]) if not post_row.empty else None

        return pre_close, post_open, post_close

    except Exception as e:
        logger.debug(f"Error getting price reaction: {e}")
        return None, None, None


# ============================================================================
# BACKTEST CALCULATIONS
# ============================================================================

def calculate_event_scores(event: EarningsEvent,
                           implied_move_pctl: float = 50.0) -> EarningsEvent:
    """
    Calculate z-scores and ECS for a historical event.

    Args:
        event: EarningsEvent to score
        implied_move_pctl: Assumed implied move percentile (for required_z)

    Returns:
        Updated EarningsEvent with scores
    """
    # Calculate EPS z-score
    if event.eps_surprise_pct is not None:
        event.eps_z = calculate_eps_z(event.eps_surprise_pct)

    # Calculate revenue z-score
    if event.revenue_surprise_pct is not None:
        event.rev_z = calculate_rev_z(event.revenue_surprise_pct)

    # Calculate event z (blended)
    event.event_z = calculate_event_z(event.eps_z, event.rev_z, None)

    # Calculate required z
    event.required_z = calculate_required_z(implied_move_pctl=implied_move_pctl)

    # Determine ECS category
    if event.event_z is not None and event.required_z is not None:
        event.ecs_category = ECSCategory.from_event_z(event.event_z, event.required_z)

    # Determine predicted direction
    if event.ecs_category:
        if event.ecs_category in (ECSCategory.STRONG_BEAT, ECSCategory.BEAT):
            event.predicted_direction = 'positive'
        elif event.ecs_category in (ECSCategory.MISS, ECSCategory.STRONG_MISS):
            event.predicted_direction = 'negative'
        else:
            event.predicted_direction = 'neutral'

    # Determine actual direction
    if event.total_reaction is not None:
        if event.total_reaction > 1.0:
            event.actual_direction = 'positive'
        elif event.total_reaction < -1.0:
            event.actual_direction = 'negative'
        else:
            event.actual_direction = 'neutral'

    # Check prediction accuracy
    if event.predicted_direction and event.actual_direction:
        # For neutral predictions, we're "correct" if the move was small
        if event.predicted_direction == 'neutral':
            event.correct_prediction = event.actual_direction == 'neutral'
        else:
            event.correct_prediction = event.predicted_direction == event.actual_direction

    return event


def calculate_backtest_metrics(events: List[EarningsEvent]) -> Dict[str, Any]:
    """
    Calculate backtest metrics from scored events.

    Args:
        events: List of scored EarningsEvent objects

    Returns:
        Dict with all metrics
    """
    # Filter to valid events
    valid_events = [e for e in events if e.ecs_category and e.total_reaction is not None]

    if not valid_events:
        return {
            'ecs_accuracy': 0.0,
            'direction_accuracy': 0.0,
            'ecs_by_category': {},
            'direction_by_category': {},
            'regime_performance': {},
            'scaling_impact': {},
        }

    # ECS Category Analysis
    ecs_by_category = {}
    for cat in ECSCategory:
        cat_events = [e for e in valid_events if e.ecs_category == cat]
        if cat_events:
            correct = sum(1 for e in cat_events if e.correct_prediction)
            avg_return = np.mean([e.total_reaction for e in cat_events])
            win_rate = sum(1 for e in cat_events if e.total_reaction > 0) / len(cat_events) * 100

            ecs_by_category[cat.value] = {
                'count': len(cat_events),
                'accuracy': correct / len(cat_events) * 100 if cat_events else 0,
                'avg_return': avg_return,
                'win_rate': win_rate,
            }

    # Overall ECS accuracy
    correct_predictions = sum(1 for e in valid_events if e.correct_prediction)
    ecs_accuracy = correct_predictions / len(valid_events) * 100 if valid_events else 0

    # Direction accuracy
    direction_events = [e for e in valid_events if e.predicted_direction and e.actual_direction]
    direction_correct = sum(1 for e in direction_events if e.correct_prediction)
    direction_accuracy = direction_correct / len(direction_events) * 100 if direction_events else 0

    # Direction by category
    direction_by_category = {}
    for direction in ['positive', 'negative', 'neutral']:
        dir_events = [e for e in direction_events if e.predicted_direction == direction]
        if dir_events:
            correct = sum(1 for e in dir_events if e.correct_prediction)
            direction_by_category[direction] = correct / len(dir_events) * 100

    # Position scaling simulation
    # Compare returns with vs without scaling
    unscaled_return = np.mean([e.total_reaction for e in valid_events])

    # Simulate scaling: reduce position for MISS/STRONG_MISS
    scaled_returns = []
    for e in valid_events:
        scale = 1.0
        if e.ecs_category == ECSCategory.STRONG_MISS:
            scale = 0.3  # Heavily reduced
        elif e.ecs_category == ECSCategory.MISS:
            scale = 0.6  # Reduced
        elif e.ecs_category == ECSCategory.STRONG_BEAT:
            scale = 1.2  # Slightly increased

        scaled_returns.append(e.total_reaction * scale)

    scaled_return = np.mean(scaled_returns) if scaled_returns else 0

    scaling_impact = {
        'unscaled': unscaled_return,
        'scaled': scaled_return,
        'improvement': scaled_return - unscaled_return,
    }

    # Regime performance (simulated - we don't have historical IES)
    # Group by ECS category as proxy
    regime_performance = {
        'BEAT': {},
        'MISS': {},
        'NEUTRAL': {},
    }

    beat_events = [e for e in valid_events if e.ecs_category in (ECSCategory.STRONG_BEAT, ECSCategory.BEAT)]
    miss_events = [e for e in valid_events if e.ecs_category in (ECSCategory.MISS, ECSCategory.STRONG_MISS)]
    neutral_events = [e for e in valid_events if e.ecs_category == ECSCategory.INLINE]

    for regime, regime_events in [('BEAT', beat_events), ('MISS', miss_events), ('NEUTRAL', neutral_events)]:
        if regime_events:
            regime_performance[regime] = {
                'count': len(regime_events),
                'avg_return': np.mean([e.total_reaction for e in regime_events]),
                'win_rate': sum(1 for e in regime_events if e.total_reaction > 0) / len(regime_events) * 100,
            }

    return {
        'ecs_accuracy': ecs_accuracy,
        'direction_accuracy': direction_accuracy,
        'ecs_by_category': ecs_by_category,
        'direction_by_category': direction_by_category,
        'regime_performance': regime_performance,
        'scaling_impact': scaling_impact,
    }


# ============================================================================
# MAIN BACKTEST FUNCTION
# ============================================================================

def run_backtest(tickers: List[str],
                 lookback_quarters: int = 8,
                 implied_move_pctl: float = 50.0,
                 progress_callback=None) -> BacktestResult:
    """
    Run backtest on historical earnings data.

    Args:
        tickers: List of ticker symbols
        lookback_quarters: Number of quarters to analyze per ticker
        implied_move_pctl: Assumed implied move percentile for required_z
        progress_callback: Optional callback(current, total, ticker)

    Returns:
        BacktestResult with all metrics
    """
    logger.info(f"Running backtest on {len(tickers)} tickers, {lookback_quarters} quarters each...")

    all_events = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        try:
            # Get historical events
            events = get_historical_earnings_yfinance(ticker, lookback_quarters)

            # Score each event
            for event in events:
                event = calculate_event_scores(event, implied_move_pctl)
                all_events.append(event)

            logger.debug(f"{ticker}: {len(events)} events processed")

        except Exception as e:
            logger.warning(f"{ticker}: Error in backtest: {e}")

        if progress_callback:
            progress_callback(i + 1, total, ticker)

    # Calculate metrics
    metrics = calculate_backtest_metrics(all_events)

    # Determine date range
    valid_dates = [e.earnings_date for e in all_events if e.earnings_date]
    date_range = (min(valid_dates), max(valid_dates)) if valid_dates else (date.today(), date.today())

    # Build result
    result = BacktestResult(
        tickers_tested=tickers,
        total_events=len(all_events),
        valid_events=len([e for e in all_events if e.ecs_category and e.total_reaction is not None]),
        date_range=date_range,
        ecs_accuracy=metrics['ecs_accuracy'],
        ecs_by_category=metrics['ecs_by_category'],
        direction_accuracy=metrics['direction_accuracy'],
        direction_by_category=metrics['direction_by_category'],
        regime_performance=metrics['regime_performance'],
        scaling_impact=metrics['scaling_impact'],
        events=all_events,
    )

    logger.info(f"Backtest complete: {result.valid_events} valid events, {result.ecs_accuracy:.1f}% accuracy")

    return result


def run_quick_backtest(tickers: List[str] = None) -> BacktestResult:
    """
    Run a quick backtest with default settings.

    Args:
        tickers: List of tickers (defaults to popular stocks)

    Returns:
        BacktestResult
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "META", "AMZN", "AMD"]

    return run_backtest(tickers, lookback_quarters=4)


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_backtest_report(result: BacktestResult) -> str:
    """
    Generate a detailed backtest report.

    Args:
        result: BacktestResult object

    Returns:
        Formatted report string
    """
    return result.summary()


def save_backtest_to_csv(result: BacktestResult, filepath: str) -> bool:
    """
    Save backtest events to CSV for analysis.

    Args:
        result: BacktestResult object
        filepath: Output file path

    Returns:
        True if successful
    """
    try:
        rows = []
        for event in result.events:
            rows.append({
                'ticker': event.ticker,
                'earnings_date': event.earnings_date,
                'eps_surprise_pct': event.eps_surprise_pct,
                'eps_z': event.eps_z,
                'event_z': event.event_z,
                'required_z': event.required_z,
                'ecs_category': event.ecs_category.value if event.ecs_category else None,
                'gap_reaction': event.gap_reaction,
                'total_reaction': event.total_reaction,
                'predicted_direction': event.predicted_direction,
                'actual_direction': event.actual_direction,
                'correct_prediction': event.correct_prediction,
            })

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)

        logger.info(f"Backtest saved to {filepath}")
        return True

    except Exception as e:
        logger.error(f"Error saving backtest: {e}")
        return False


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("Phase 12: Backtesting Module Test")
    print("=" * 60)

    test_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL"]

    def progress(current, total, ticker):
        print(f"  [{current}/{total}] {ticker}")

    print("\nRunning backtest...")
    result = run_backtest(test_tickers, lookback_quarters=4, progress_callback=progress)

    print("\n" + result.summary())

    # Print sample events
    print("\n" + "─" * 40)
    print("SAMPLE EVENTS")
    print("─" * 40)

    for event in result.events[:5]:
        if event.ecs_category and event.total_reaction is not None:
            print(f"  {event.ticker} {event.earnings_date}: "
                  f"EPS={event.eps_surprise_pct:+.1f}% "
                  f"ECS={event.ecs_category.value} "
                  f"Reaction={event.total_reaction:+.1f}% "
                  f"Correct={event.correct_prediction}")