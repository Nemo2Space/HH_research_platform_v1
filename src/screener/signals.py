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
    # Extract scores with defaults
    sentiment = float(scores.get('sentiment_score', 50) or 50)
    likelihood = float(scores.get('likelihood_score', 50) or 50)
    gap_type = float(scores.get('gap_score', 50) or 50)
    fundamental_score = float(scores.get('fundamental_score', 50) or 50)
    growth_score = float(scores.get('growth_score', 50) or 50)
    dividend_score = float(scores.get('dividend_score', 50) or 50)
    analyst_positivity = float(scores.get('analyst_positivity', 50) or 50)
    total_score = float(scores.get('total_score', 50) or 50)

    # Initialize
    signal_type = "NEUTRAL"
    signal_strength = 0
    signal_reasons = []
    signal_color = "#d1d5db"

    ticker = scores.get('ticker', 'UNKNOWN')

    # Signal generation logic (from Project 1)

    # 1. Strong buy signals
    if (likelihood > 75 and sentiment > 60 and gap_type > 60 and
            (fundamental_score > 70 or analyst_positivity > 80)):
        signal_type = "STRONG BUY"
        signal_strength = 5
        signal_reasons.append("Multiple indicators strongly positive")
        signal_color = "#22c55e"

    # 2. Buy signals
    elif (likelihood > 60 and sentiment > 50 and
          (gap_type > 50 or fundamental_score > 60)):
        signal_type = "BUY"
        signal_strength = 4
        signal_reasons.append("Positive outlook with technical support")
        signal_color = "#4ade80"

    # 3. Weak buy signals
    elif (likelihood > 55 and (sentiment > 45 or gap_type > 45) and total_score > 50):
        signal_type = "WEAK BUY"
        signal_strength = 3
        signal_reasons.append("Modestly positive indicators")
        signal_color = "#86efac"

    # 4. Strong sell signals
    elif (likelihood < 25 and sentiment < 40 and gap_type < 40 and
          (fundamental_score < 30 or analyst_positivity < 20)):
        signal_type = "STRONG SELL"
        signal_strength = -5
        signal_reasons.append("Multiple indicators strongly negative")
        signal_color = "#ef4444"

    # 5. Sell signals
    elif (likelihood < 40 and sentiment < 50 and
          (gap_type < 50 or fundamental_score < 40)):
        signal_type = "SELL"
        signal_strength = -4
        signal_reasons.append("Negative outlook with technical weakness")
        signal_color = "#f87171"

    # 6. Weak sell signals
    elif (likelihood < 45 and (sentiment < 55 or gap_type < 55) and total_score < 50):
        signal_type = "WEAK SELL"
        signal_strength = -3
        signal_reasons.append("Modestly negative indicators")
        signal_color = "#fca5a5"

    # 7. Neutral with positive bias
    elif likelihood > 55 or sentiment > 55 or total_score > 55:
        signal_type = "NEUTRAL+"
        signal_strength = 1
        signal_reasons.append("Slightly positive indicators")
        signal_color = "#bef264"

    # 8. Neutral with negative bias
    elif likelihood < 45 or sentiment < 45 or total_score < 45:
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
    if dividend_score > 70 and fundamental_score > 60 and signal_strength >= 0:
        if signal_strength < 3:
            signal_type = "INCOME BUY"
            signal_strength = 3
            signal_reasons = ["High dividend potential with solid fundamentals"]
            signal_color = "#818cf8"

    # Special case: High growth stocks with positive sentiment
    if growth_score > 75 and sentiment > 60 and signal_strength >= 0:
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
                              weights: Optional[Dict[str, float]] = None) -> int:
    """
    Calculate composite score from individual scores.
    """
    default_weights = {
        'sentiment': 0.20,
        'fundamental': 0.25,
        'growth': 0.15,
        'dividend': 0.05,
        'gap': 0.10,
        'analyst': 0.10,
        'likelihood': 0.15,
    }

    w = weights or default_weights

    composite = 0.0
    composite += float(scores.get('sentiment_score', 50) or 50) * w.get('sentiment', 0.20)
    composite += float(scores.get('fundamental_score', 50) or 50) * w.get('fundamental', 0.25)
    composite += float(scores.get('growth_score', 50) or 50) * w.get('growth', 0.15)
    composite += float(scores.get('dividend_score', 50) or 50) * w.get('dividend', 0.05)
    composite += float(scores.get('gap_score', 50) or 50) * w.get('gap', 0.10)
    composite += float(scores.get('analyst_positivity', 50) or 50) * w.get('analyst', 0.10)
    composite += float(scores.get('likelihood_score', 50) or 50) * w.get('likelihood', 0.15)

    result = int(composite)
    return max(0, min(100, result))

def calculate_likelihood_score(scores: Dict[str, Any]) -> int:
    """
    Calculate likelihood score (probability of positive outcome).

    Based on sentiment, fundamentals, and analyst consensus.
    """
    sentiment = float(scores.get('sentiment_score', 50) or 50)
    fundamental = float(scores.get('fundamental_score', 50) or 50)
    analyst = float(scores.get('analyst_positivity', 50) or 50)

    # Simple weighted average
    likelihood = (sentiment * 0.4) + (fundamental * 0.35) + (analyst * 0.25)

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