"""
Alpha Platform - Signal Generation

Generates trading signals based on multiple factors.
Based on Project 1 signal_generation.py
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Signal:
    """Trading signal data."""
    type: str
    strength: int
    reasons: List[str]
    color: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'strength': self.strength,
            'reasons': self.reasons,
            'color': self.color,
            'timestamp': self.timestamp
        }


# Signal type definitions with colors
SIGNAL_TYPES = {
    'STRONG BUY':  {'strength': 5,  'color': '#22c55e'},  # Bright green
    'BUY':         {'strength': 4,  'color': '#4ade80'},  # Light green
    'WEAK BUY':    {'strength': 3,  'color': '#86efac'},  # Very light green
    'INCOME BUY':  {'strength': 3,  'color': '#818cf8'},  # Purple-blue
    'GROWTH BUY':  {'strength': 3,  'color': '#2dd4bf'},  # Teal
    'NEUTRAL+':    {'strength': 1,  'color': '#bef264'},  # Pale green
    'NEUTRAL':     {'strength': 0,  'color': '#d1d5db'},  # Light gray
    'NEUTRAL-':    {'strength': -1, 'color': '#fecaca'},  # Pale red
    'WEAK SELL':   {'strength': -3, 'color': '#fca5a5'},  # Very light red
    'SELL':        {'strength': -4, 'color': '#f87171'},  # Light red
    'STRONG SELL': {'strength': -5, 'color': '#ef4444'},  # Bright red
}


def generate_trading_signal(scores: Dict[str, Any]) -> Signal:
    """
    Generate trading signal based on multiple factors.

    Args:
        scores: Dict containing:
            - sentiment_score (0-100)
            - fundamental_score (0-100)
            - growth_score (0-100)
            - dividend_score (0-100)
            - gap_score (0-100)
            - likelihood_score (0-100)
            - analyst_positivity (0-100)
            - total_score (0-100)

    Returns:
        Signal object with type, strength, reasons, color
    """
    # Extract scores — None means "not available"
    def _score(key: str):
        v = scores.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    sentiment = _score('sentiment_score')
    likelihood = _score('likelihood_score')
    gap_type = _score('gap_score')
    fundamental_score = _score('fundamental_score')
    growth_score = _score('growth_score')
    dividend_score = _score('dividend_score')
    analyst_positivity = _score('analyst_positivity')
    total_score = _score('total_score')

    # Initialize
    signal_type = "NEUTRAL"
    signal_strength = 0
    signal_reasons = []
    signal_color = "#d1d5db"

    ticker = scores.get('ticker', 'UNKNOWN')

    # None-safe comparisons: missing data → condition fails (can't confirm signal)
    def _gt(v, t): return v is not None and v > t
    def _lt(v, t): return v is not None and v < t
    def _gte(v, t): return v is not None and v >= t

    # Count available scores for confidence
    available = sum(1 for v in [sentiment, likelihood, gap_type, fundamental_score] if v is not None)

    if available == 0:
        # No data at all — cannot generate signal
        return Signal(
            type="NEUTRAL",
            strength=0,
            reasons=["Insufficient data for signal generation"],
            color="#d1d5db",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    # Signal generation logic

    # 1. Strong buy signals
    if (_gt(likelihood, 75) and _gt(sentiment, 60) and _gt(gap_type, 60) and
            (_gt(fundamental_score, 70) or _gt(analyst_positivity, 80))):
        signal_type = "STRONG BUY"
        signal_strength = 5
        signal_reasons.append("Multiple indicators strongly positive")
        signal_color = "#22c55e"

    # 2. Buy signals
    elif (_gt(likelihood, 60) and _gt(sentiment, 50) and
          (_gt(gap_type, 50) or _gt(fundamental_score, 60))):
        signal_type = "BUY"
        signal_strength = 4
        signal_reasons.append("Positive outlook with technical support")
        signal_color = "#4ade80"

    # 3. Weak buy signals
    elif (_gt(likelihood, 55) and (_gt(sentiment, 45) or _gt(gap_type, 45)) and _gt(total_score, 50)):
        signal_type = "WEAK BUY"
        signal_strength = 3
        signal_reasons.append("Modestly positive indicators")
        signal_color = "#86efac"

    # 4. Strong sell signals
    elif (_lt(likelihood, 25) and _lt(sentiment, 40) and _lt(gap_type, 40) and
          (_lt(fundamental_score, 30) or _lt(analyst_positivity, 20))):
        signal_type = "STRONG SELL"
        signal_strength = -5
        signal_reasons.append("Multiple indicators strongly negative")
        signal_color = "#ef4444"

    # 5. Sell signals
    elif (_lt(likelihood, 40) and _lt(sentiment, 50) and
          (_lt(gap_type, 50) or _lt(fundamental_score, 40))):
        signal_type = "SELL"
        signal_strength = -4
        signal_reasons.append("Negative outlook with technical weakness")
        signal_color = "#f87171"

    # 6. Weak sell signals
    elif (_lt(likelihood, 45) and (_lt(sentiment, 55) or _lt(gap_type, 55)) and _lt(total_score, 50)):
        signal_type = "WEAK SELL"
        signal_strength = -3
        signal_reasons.append("Modestly negative indicators")
        signal_color = "#fca5a5"

    # 7. Neutral with positive bias
    elif _gt(likelihood, 55) or _gt(sentiment, 55) or _gt(total_score, 55):
        signal_type = "NEUTRAL+"
        signal_strength = 1
        signal_reasons.append("Slightly positive indicators")
        signal_color = "#bef264"

    # 8. Neutral with negative bias
    elif _lt(likelihood, 45) or _lt(sentiment, 45) or _lt(total_score, 45):
        signal_type = "NEUTRAL-"
        signal_strength = -1
        signal_reasons.append("Slightly negative indicators")
        signal_color = "#fecaca"

    # 9. Fully neutral
    else:
        signal_type = "NEUTRAL"
        signal_strength = 0
        signal_reasons.append("Balanced indicators")
        signal_color = "#d1d5db"

    # Special case: High dividend stocks with good fundamentals
    if _gt(dividend_score, 70) and _gt(fundamental_score, 60) and signal_strength >= 0:
        if signal_strength < 3:
            signal_type = "INCOME BUY"
            signal_strength = 3
            signal_reasons = ["High dividend potential with solid fundamentals"]
            signal_color = "#818cf8"

    # Special case: High growth stocks with positive sentiment
    if _gt(growth_score, 75) and _gt(sentiment, 60) and signal_strength >= 0:
        if signal_strength < 3:
            signal_type = "GROWTH BUY"
            signal_strength = 3
            signal_reasons = ["Strong growth potential with positive sentiment"]
            signal_color = "#2dd4bf"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.debug(f"{ticker}: {signal_type} (strength={signal_strength})")

    return Signal(
        type=signal_type,
        strength=signal_strength,
        reasons=signal_reasons,
        color=signal_color,
        timestamp=timestamp
    )


def calculate_composite_score(scores: Dict[str, Any],
                              weights: Optional[Dict[str, float]] = None) -> Optional[int]:
    """
    Calculate composite score from individual scores.
    Only includes components with real data; normalizes weights dynamically.

    NOTE (F-03 fix): 'likelihood' removed — it's derived from sentiment +
    fundamental + analyst, so including it double-counts those components.
    """
    default_weights = {
        'sentiment': 0.25,
        'fundamental': 0.30,
        'growth': 0.15,
        'dividend': 0.05,
        'gap': 0.10,
        'analyst': 0.15,
    }

    w = weights or default_weights

    # Map weight keys to score keys
    score_keys = {
        'sentiment': 'sentiment_score',
        'fundamental': 'fundamental_score',
        'growth': 'growth_score',
        'dividend': 'dividend_score',
        'gap': 'gap_score',
        'analyst': 'analyst_positivity',
    }

    # Collect available components
    weighted_total = 0.0
    weight_sum = 0.0

    for weight_key, score_key in score_keys.items():
        val = scores.get(score_key)
        if val is not None:
            try:
                weighted_total += float(val) * w.get(weight_key, 0)
                weight_sum += w.get(weight_key, 0)
            except (TypeError, ValueError):
                continue

    if weight_sum == 0:
        return None  # Cannot score — no data

    result = int(weighted_total / weight_sum)
    return max(0, min(100, result))

def calculate_likelihood_score(scores: Dict[str, Any]) -> Optional[int]:
    """
    Calculate likelihood score (probability of positive outcome).
    Based on sentiment, fundamentals, and analyst consensus.
    Only uses components with real data.
    """
    component_weights = {
        'sentiment_score': 0.4,
        'fundamental_score': 0.35,
        'analyst_positivity': 0.25,
    }

    weighted_total = 0.0
    weight_sum = 0.0

    for key, weight in component_weights.items():
        val = scores.get(key)
        if val is not None:
            try:
                weighted_total += float(val) * weight
                weight_sum += weight
            except (TypeError, ValueError):
                continue

    if weight_sum == 0:
        return None  # Cannot compute — no input data

    likelihood = weighted_total / weight_sum
    return int(max(0, min(100, likelihood)))


class SignalGenerator:
    """Generates trading signals for stocks."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights

    def generate_signal(self, scores: Dict[str, Any]) -> Signal:
        """Generate signal for a single stock."""
        return generate_trading_signal(scores)

    def generate_batch(self, scores_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate signals for a batch of stocks.

        Returns list of scores dicts with signal info added.
        """
        results = []
        for scores in scores_list:
            signal = self.generate_signal(scores)

            result = scores.copy()
            result['signal_type'] = signal.type
            result['signal_strength'] = signal.strength
            result['signal_color'] = signal.color
            result['signal_reason'] = signal.reasons[0] if signal.reasons else ""
            result['signal_timestamp'] = signal.timestamp

            results.append(result)

        return results