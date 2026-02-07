"""
Earnings Whisper Module - Phase 3

Predicts earnings surprises BEFORE the announcement.

Analyzes:
- Analyst estimate revisions (are they quietly raising/lowering?)
- Pre-earnings options positioning (smart money bets)
- Historical beat/miss patterns
- Guidance trends (sandbagging vs aggressive)
- Sector earnings momentum

Returns probability of beat/miss and expected surprise magnitude.

Author: Alpha Research Platform
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from src.utils.logging import get_logger

logger = get_logger(__name__)


class EarningsPrediction(Enum):
    """Earnings prediction outcome."""
    STRONG_BEAT = "STRONG_BEAT"      # >10% surprise
    BEAT = "BEAT"                     # 2-10% surprise
    INLINE = "INLINE"                 # -2% to +2%
    MISS = "MISS"                     # -2% to -10%
    STRONG_MISS = "STRONG_MISS"       # <-10% surprise


class WhisperSignal(Enum):
    """Trading signal based on earnings whisper."""
    STRONG_BULLISH = "STRONG_BULLISH"   # High confidence beat
    BULLISH = "BULLISH"                  # Likely beat
    NEUTRAL = "NEUTRAL"                  # Uncertain
    BEARISH = "BEARISH"                  # Likely miss
    STRONG_BEARISH = "STRONG_BEARISH"    # High confidence miss


@dataclass
class EstimateRevisions:
    """Analyst estimate revision tracking."""
    ticker: str

    # Current estimates
    eps_estimate: float = 0.0
    revenue_estimate: float = 0.0

    # Revision counts (last 30 days)
    eps_up_revisions: int = 0
    eps_down_revisions: int = 0
    revenue_up_revisions: int = 0
    revenue_down_revisions: int = 0

    # Revision magnitude
    eps_revision_pct_30d: float = 0.0   # % change in consensus
    eps_revision_pct_7d: float = 0.0
    revenue_revision_pct_30d: float = 0.0

    # Trend
    revision_trend: str = "STABLE"  # RISING, FALLING, STABLE
    analyst_sentiment: str = "NEUTRAL"  # BULLISH, NEUTRAL, BEARISH


@dataclass
class OptionsPositioning:
    """Pre-earnings options positioning."""
    ticker: str

    # Overall positioning
    put_call_ratio: float = 1.0
    call_volume_ratio: float = 1.0  # vs 20-day avg
    put_volume_ratio: float = 1.0

    # Skew analysis
    iv_skew: float = 0.0  # Put IV - Call IV
    iv_rank: float = 50.0  # IV percentile

    # Expected move
    implied_move_pct: float = 0.0  # Market's expected move
    straddle_price_pct: float = 0.0  # ATM straddle as % of stock

    # Directional signals
    unusual_calls: bool = False
    unusual_puts: bool = False
    smart_money_direction: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL


@dataclass
class HistoricalPattern:
    """Historical earnings beat/miss patterns."""
    ticker: str

    # Win rate
    beat_rate_4q: float = 0.0    # Last 4 quarters
    beat_rate_8q: float = 0.0    # Last 8 quarters
    beat_rate_12q: float = 0.0   # Last 12 quarters

    # Magnitude
    avg_surprise_pct: float = 0.0
    avg_beat_magnitude: float = 0.0
    avg_miss_magnitude: float = 0.0

    # Consistency
    consecutive_beats: int = 0
    consecutive_misses: int = 0

    # Post-earnings reaction
    avg_post_earnings_move: float = 0.0
    beat_reaction_avg: float = 0.0    # Avg move on beats
    miss_reaction_avg: float = 0.0    # Avg move on misses

    # Pattern
    tends_to_beat: bool = False
    sandbagging_score: float = 0.0  # 0-100, higher = more sandbagging


@dataclass
class EarningsWhisper:
    """Complete earnings whisper analysis."""
    ticker: str
    as_of_date: date
    earnings_date: str = ""
    days_to_earnings: int = 999

    # Predictions
    prediction: EarningsPrediction = EarningsPrediction.INLINE
    beat_probability: float = 50.0   # 0-100%
    expected_surprise_pct: float = 0.0

    # Confidence
    confidence: float = 50.0  # 0-100
    signal: WhisperSignal = WhisperSignal.NEUTRAL
    signal_strength: int = 50

    # Components
    estimate_revisions: EstimateRevisions = None
    options_positioning: OptionsPositioning = None
    historical_pattern: HistoricalPattern = None

    # Component scores (0-100, >50 = bullish)
    revision_score: int = 50
    options_score: int = 50
    historical_score: int = 50

    # Risks
    high_expectations: bool = False
    whisper_vs_consensus: float = 0.0  # Whisper number vs consensus

    # Warnings
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ticker': self.ticker,
            'earnings_date': self.earnings_date,
            'days_to_earnings': self.days_to_earnings,
            'prediction': self.prediction.value,
            'beat_probability': round(self.beat_probability, 1),
            'expected_surprise_pct': round(self.expected_surprise_pct, 1),
            'confidence': round(self.confidence, 1),
            'signal': self.signal.value,
            'signal_strength': self.signal_strength,
            'revision_score': self.revision_score,
            'options_score': self.options_score,
            'historical_score': self.historical_score,
            'high_expectations': self.high_expectations,
            'warnings': self.warnings,
        }

    def get_summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"{'=' * 50}",
            f"EARNINGS WHISPER: {self.ticker}",
            f"{'=' * 50}",
            f"Earnings Date: {self.earnings_date} ({self.days_to_earnings} days)",
            f"",
            f"PREDICTION: {self.prediction.value}",
            f"  Beat Probability: {self.beat_probability:.0f}%",
            f"  Expected Surprise: {self.expected_surprise_pct:+.1f}%",
            f"  Confidence: {self.confidence:.0f}%",
            f"",
            f"SIGNAL: {self.signal.value} (strength: {self.signal_strength})",
            f"",
            f"Component Scores:",
            f"  Estimate Revisions: {self.revision_score}/100",
            f"  Options Positioning: {self.options_score}/100",
            f"  Historical Pattern: {self.historical_score}/100",
        ]

        if self.warnings:
            lines.append(f"\n⚠️ Warnings:")
            for w in self.warnings:
                lines.append(f"  • {w}")

        lines.append("=" * 50)
        return "\n".join(lines)


class EarningsWhisperAnalyzer:
    """
    Analyzes multiple factors to predict earnings outcomes.
    """

    def __init__(self):
        self._cache = {}
        self._cache_time = {}
        self._cache_duration = timedelta(hours=1)

    def analyze(self, ticker: str) -> EarningsWhisper:
        """
        Complete earnings whisper analysis for a ticker.

        Args:
            ticker: Stock symbol

        Returns:
            EarningsWhisper with prediction and component analysis
        """
        whisper = EarningsWhisper(
            ticker=ticker,
            as_of_date=date.today(),
        )

        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)

            # Get earnings date
            self._get_earnings_date(whisper, stock)

            # If no upcoming earnings, limited analysis
            if whisper.days_to_earnings > 60:
                whisper.warnings.append("No earnings within 60 days")
                return whisper

            # Analyze each component
            whisper.estimate_revisions = self._analyze_revisions(ticker, stock)
            whisper.options_positioning = self._analyze_options(ticker, stock)
            whisper.historical_pattern = self._analyze_history(ticker, stock)

            # Calculate component scores
            whisper.revision_score = self._score_revisions(whisper.estimate_revisions)
            whisper.options_score = self._score_options(whisper.options_positioning)
            whisper.historical_score = self._score_history(whisper.historical_pattern)

            # Combine into prediction
            self._generate_prediction(whisper)

        except Exception as e:
            logger.error(f"{ticker}: Earnings whisper analysis error: {e}")
            whisper.warnings.append(f"Analysis error: {str(e)}")

        return whisper

    def _get_earnings_date(self, whisper: EarningsWhisper, stock) -> None:
        """Get next earnings date."""
        try:
            calendar = stock.calendar

            if calendar is not None:
                if isinstance(calendar, dict):
                    earnings_dates = calendar.get('Earnings Date', [])
                    if earnings_dates:
                        next_date = earnings_dates[0]
                        if hasattr(next_date, 'date'):
                            next_date = next_date.date()
                        elif isinstance(next_date, str):
                            next_date = datetime.strptime(next_date[:10], '%Y-%m-%d').date()

                        whisper.earnings_date = str(next_date)
                        whisper.days_to_earnings = (next_date - date.today()).days
                elif hasattr(calendar, 'iloc'):
                    # DataFrame format
                    if 'Earnings Date' in calendar.columns:
                        dates = calendar['Earnings Date'].dropna()
                        if len(dates) > 0:
                            next_date = dates.iloc[0]
                            if hasattr(next_date, 'date'):
                                next_date = next_date.date()
                            whisper.earnings_date = str(next_date)
                            whisper.days_to_earnings = (next_date - date.today()).days
        except Exception as e:
            logger.debug(f"Error getting earnings date: {e}")
            whisper.earnings_date = "Unknown"

    def _analyze_revisions(self, ticker: str, stock) -> EstimateRevisions:
        """Analyze analyst estimate revisions."""
        revisions = EstimateRevisions(ticker=ticker)

        try:
            # Get analyst info
            info = stock.info

            # Current estimates
            revisions.eps_estimate = info.get('targetMeanPrice', 0) or 0  # This is price target, not EPS

            # Try to get actual EPS estimates from earnings
            earnings = stock.earnings_dates
            if earnings is not None and not earnings.empty:
                if 'EPS Estimate' in earnings.columns:
                    upcoming = earnings[earnings.index >= datetime.now()]
                    if not upcoming.empty:
                        revisions.eps_estimate = upcoming['EPS Estimate'].iloc[0]

            # Analyst recommendations give us sentiment
            recommendations = stock.recommendations
            if recommendations is not None and not recommendations.empty:
                recent = recommendations.tail(30)

                # Count upgrades vs downgrades
                if 'To Grade' in recent.columns and 'From Grade' in recent.columns:
                    for _, row in recent.iterrows():
                        to_grade = str(row.get('To Grade', '')).lower()
                        from_grade = str(row.get('From Grade', '')).lower()

                        buy_terms = ['buy', 'outperform', 'overweight', 'strong buy']
                        sell_terms = ['sell', 'underperform', 'underweight', 'strong sell']

                        to_bullish = any(t in to_grade for t in buy_terms)
                        to_bearish = any(t in to_grade for t in sell_terms)
                        from_bullish = any(t in from_grade for t in buy_terms)
                        from_bearish = any(t in from_grade for t in sell_terms)

                        if to_bullish and not from_bullish:
                            revisions.eps_up_revisions += 1
                        elif to_bearish and not from_bearish:
                            revisions.eps_down_revisions += 1

            # Determine trend
            net_revisions = revisions.eps_up_revisions - revisions.eps_down_revisions
            if net_revisions >= 2:
                revisions.revision_trend = "RISING"
                revisions.analyst_sentiment = "BULLISH"
            elif net_revisions <= -2:
                revisions.revision_trend = "FALLING"
                revisions.analyst_sentiment = "BEARISH"
            else:
                revisions.revision_trend = "STABLE"
                revisions.analyst_sentiment = "NEUTRAL"

        except Exception as e:
            logger.debug(f"{ticker}: Error analyzing revisions: {e}")

        return revisions

    def _analyze_options(self, ticker: str, stock) -> OptionsPositioning:
        """Analyze pre-earnings options positioning."""
        positioning = OptionsPositioning(ticker=ticker)

        try:
            # Get options chain
            expirations = stock.options

            if not expirations:
                return positioning

            # Find first expiration after earnings
            current_price = stock.history(period="1d")['Close'].iloc[-1]

            # Get nearest expiration
            exp = expirations[0]
            opt_chain = stock.option_chain(exp)

            calls = opt_chain.calls
            puts = opt_chain.puts

            # Put/Call ratio by volume
            total_call_vol = calls['volume'].sum()
            total_put_vol = puts['volume'].sum()

            if total_call_vol > 0:
                positioning.put_call_ratio = total_put_vol / total_call_vol

            # IV analysis
            atm_calls = calls[
                (calls['strike'] >= current_price * 0.95) &
                (calls['strike'] <= current_price * 1.05)
            ]
            atm_puts = puts[
                (puts['strike'] >= current_price * 0.95) &
                (puts['strike'] <= current_price * 1.05)
            ]

            if not atm_calls.empty and not atm_puts.empty:
                if 'impliedVolatility' in atm_calls.columns:
                    call_iv = atm_calls['impliedVolatility'].mean()
                    put_iv = atm_puts['impliedVolatility'].mean()

                    positioning.iv_skew = (put_iv - call_iv) * 100

                    # Estimate expected move from straddle
                    # ATM call + ATM put price = expected move
                    atm_strike = round(current_price / 5) * 5  # Round to nearest 5

                    atm_call = calls[calls['strike'] == atm_strike]
                    atm_put = puts[puts['strike'] == atm_strike]

                    if not atm_call.empty and not atm_put.empty:
                        straddle_price = atm_call['lastPrice'].iloc[0] + atm_put['lastPrice'].iloc[0]
                        positioning.straddle_price_pct = (straddle_price / current_price) * 100
                        positioning.implied_move_pct = positioning.straddle_price_pct * 0.85  # ~85% of straddle

            # Detect unusual activity
            if 'volume' in calls.columns and 'openInterest' in calls.columns:
                call_vol_oi = (calls['volume'] / calls['openInterest'].replace(0, 1)).mean()
                put_vol_oi = (puts['volume'] / puts['openInterest'].replace(0, 1)).mean()

                positioning.unusual_calls = call_vol_oi > 0.5
                positioning.unusual_puts = put_vol_oi > 0.5

            # Smart money direction
            if positioning.put_call_ratio < 0.7 and positioning.unusual_calls:
                positioning.smart_money_direction = "BULLISH"
            elif positioning.put_call_ratio > 1.3 and positioning.unusual_puts:
                positioning.smart_money_direction = "BEARISH"
            elif positioning.put_call_ratio < 0.8:
                positioning.smart_money_direction = "BULLISH"
            elif positioning.put_call_ratio > 1.2:
                positioning.smart_money_direction = "BEARISH"

        except Exception as e:
            logger.debug(f"{ticker}: Error analyzing options: {e}")

        return positioning

    def _analyze_history(self, ticker: str, stock) -> HistoricalPattern:
        """Analyze historical earnings beat/miss patterns."""
        pattern = HistoricalPattern(ticker=ticker)

        try:
            # Get earnings history
            earnings = stock.earnings_dates

            if earnings is None or earnings.empty:
                return pattern

            # Filter to past earnings
            past_earnings = earnings[earnings.index < datetime.now()].head(12)

            if past_earnings.empty:
                return pattern

            # Check for surprise data
            if 'Surprise(%)' in past_earnings.columns:
                surprises = past_earnings['Surprise(%)'].dropna()
            elif 'EPS Estimate' in past_earnings.columns and 'Reported EPS' in past_earnings.columns:
                # Calculate surprise
                est = past_earnings['EPS Estimate']
                actual = past_earnings['Reported EPS']
                surprises = ((actual - est) / est.abs().replace(0, 0.01)) * 100
                surprises = surprises.dropna()
            else:
                surprises = pd.Series()

            if len(surprises) > 0:
                # Beat rates
                beats = (surprises > 2).sum()  # >2% = beat

                if len(surprises) >= 4:
                    pattern.beat_rate_4q = beats / min(4, len(surprises)) * 100
                if len(surprises) >= 8:
                    recent_8 = surprises.head(8)
                    pattern.beat_rate_8q = (recent_8 > 2).sum() / len(recent_8) * 100
                if len(surprises) >= 12:
                    pattern.beat_rate_12q = beats / len(surprises) * 100

                # Average surprise
                pattern.avg_surprise_pct = surprises.mean()

                beat_surprises = surprises[surprises > 2]
                miss_surprises = surprises[surprises < -2]

                if len(beat_surprises) > 0:
                    pattern.avg_beat_magnitude = beat_surprises.mean()
                if len(miss_surprises) > 0:
                    pattern.avg_miss_magnitude = miss_surprises.mean()

                # Consecutive beats/misses
                pattern.consecutive_beats = 0
                pattern.consecutive_misses = 0

                for s in surprises:
                    if s > 2:
                        if pattern.consecutive_misses == 0:
                            pattern.consecutive_beats += 1
                        else:
                            break
                    elif s < -2:
                        if pattern.consecutive_beats == 0:
                            pattern.consecutive_misses += 1
                        else:
                            break
                    else:
                        break

                # Sandbagging score (companies that consistently beat by small amounts)
                if pattern.beat_rate_4q > 70 and pattern.avg_beat_magnitude < 10:
                    pattern.sandbagging_score = 70 + (pattern.beat_rate_4q - 70)
                else:
                    pattern.sandbagging_score = max(0, pattern.beat_rate_4q - 20)

                pattern.tends_to_beat = pattern.beat_rate_4q >= 60

        except Exception as e:
            logger.debug(f"{ticker}: Error analyzing history: {e}")

        return pattern

    def _score_revisions(self, revisions: EstimateRevisions) -> int:
        """Score revisions (0-100, >50 = bullish)."""
        if revisions is None:
            return 50

        score = 50

        # Net revision direction
        net = revisions.eps_up_revisions - revisions.eps_down_revisions
        score += net * 8  # Each net revision = 8 points

        # Trend
        if revisions.revision_trend == "RISING":
            score += 10
        elif revisions.revision_trend == "FALLING":
            score -= 10

        return max(0, min(100, score))

    def _score_options(self, options: OptionsPositioning) -> int:
        """Score options positioning (0-100, >50 = bullish)."""
        if options is None:
            return 50

        score = 50

        # Put/Call ratio (lower = more bullish)
        if options.put_call_ratio < 0.7:
            score += 20
        elif options.put_call_ratio < 0.9:
            score += 10
        elif options.put_call_ratio > 1.3:
            score -= 20
        elif options.put_call_ratio > 1.1:
            score -= 10

        # IV skew (high put IV = fear = contrarian bullish sometimes)
        if options.iv_skew > 10:
            score -= 5  # Elevated fear
        elif options.iv_skew < -5:
            score += 5  # Complacency

        # Unusual activity
        if options.unusual_calls and not options.unusual_puts:
            score += 15
        elif options.unusual_puts and not options.unusual_calls:
            score -= 15

        # Smart money
        if options.smart_money_direction == "BULLISH":
            score += 10
        elif options.smart_money_direction == "BEARISH":
            score -= 10

        return max(0, min(100, score))

    def _score_history(self, pattern: HistoricalPattern) -> int:
        """Score historical pattern (0-100, >50 = bullish)."""
        if pattern is None:
            return 50

        score = 50

        # Beat rate
        if pattern.beat_rate_4q >= 75:
            score += 25
        elif pattern.beat_rate_4q >= 50:
            score += 10
        elif pattern.beat_rate_4q < 25:
            score -= 20

        # Consecutive beats/misses
        score += pattern.consecutive_beats * 5
        score -= pattern.consecutive_misses * 5

        # Sandbagging (positive - they tend to beat)
        if pattern.sandbagging_score > 50:
            score += 10

        return max(0, min(100, score))

    def _generate_prediction(self, whisper: EarningsWhisper) -> None:
        """Generate final prediction from component scores."""

        # Weighted combination
        weights = {
            'revision': 0.35,
            'options': 0.35,
            'historical': 0.30,
        }

        composite = (
            whisper.revision_score * weights['revision'] +
            whisper.options_score * weights['options'] +
            whisper.historical_score * weights['historical']
        )

        # Convert to beat probability (composite maps roughly to probability)
        whisper.beat_probability = composite

        # Expected surprise (rough estimate)
        if whisper.historical_pattern and whisper.historical_pattern.avg_surprise_pct != 0:
            base_surprise = whisper.historical_pattern.avg_surprise_pct
        else:
            base_surprise = 0.0  # No history = no directional assumption

        # Adjust based on current signals
        adjustment = (composite - 50) / 10  # -5 to +5
        whisper.expected_surprise_pct = base_surprise + adjustment

        # Determine prediction
        if whisper.beat_probability >= 75:
            if whisper.expected_surprise_pct >= 8:
                whisper.prediction = EarningsPrediction.STRONG_BEAT
            else:
                whisper.prediction = EarningsPrediction.BEAT
        elif whisper.beat_probability >= 55:
            whisper.prediction = EarningsPrediction.BEAT
        elif whisper.beat_probability >= 45:
            whisper.prediction = EarningsPrediction.INLINE
        elif whisper.beat_probability >= 25:
            whisper.prediction = EarningsPrediction.MISS
        else:
            whisper.prediction = EarningsPrediction.STRONG_MISS

        # Confidence based on agreement between components
        scores = [whisper.revision_score, whisper.options_score, whisper.historical_score]
        std_dev = np.std(scores)
        whisper.confidence = max(30, 100 - std_dev)

        # Signal
        if whisper.beat_probability >= 70 and whisper.confidence >= 60:
            whisper.signal = WhisperSignal.STRONG_BULLISH
            whisper.signal_strength = int(whisper.beat_probability)
        elif whisper.beat_probability >= 55:
            whisper.signal = WhisperSignal.BULLISH
            whisper.signal_strength = int(whisper.beat_probability)
        elif whisper.beat_probability <= 30 and whisper.confidence >= 60:
            whisper.signal = WhisperSignal.STRONG_BEARISH
            whisper.signal_strength = int(100 - whisper.beat_probability)
        elif whisper.beat_probability <= 45:
            whisper.signal = WhisperSignal.BEARISH
            whisper.signal_strength = int(100 - whisper.beat_probability)
        else:
            whisper.signal = WhisperSignal.NEUTRAL
            whisper.signal_strength = 50

        # Warnings
        if whisper.options_positioning:
            if whisper.options_positioning.implied_move_pct > 10:
                whisper.warnings.append(
                    f"High implied move ({whisper.options_positioning.implied_move_pct:.1f}%) - volatile reaction expected"
                )

        if whisper.historical_pattern:
            if whisper.historical_pattern.consecutive_beats >= 4:
                whisper.warnings.append(
                    f"{whisper.historical_pattern.consecutive_beats} consecutive beats - high bar"
                )
                whisper.high_expectations = True

            if whisper.historical_pattern.beat_rate_4q >= 75 and whisper.options_score < 50:
                whisper.warnings.append("Strong beat history but options positioning bearish")

        # Check for divergence
        if abs(whisper.revision_score - whisper.options_score) > 30:
            whisper.warnings.append("Divergence between analyst sentiment and options positioning")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_analyzer = None

def get_whisper_analyzer() -> EarningsWhisperAnalyzer:
    """Get singleton analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = EarningsWhisperAnalyzer()
    return _analyzer


def get_earnings_whisper(ticker: str) -> EarningsWhisper:
    """
    Get complete earnings whisper analysis for a ticker.

    Usage:
        whisper = get_earnings_whisper('NVDA')
        print(f"Prediction: {whisper.prediction.value}")
        print(f"Beat Probability: {whisper.beat_probability:.0f}%")
        print(f"Signal: {whisper.signal.value}")
    """
    analyzer = get_whisper_analyzer()
    return analyzer.analyze(ticker)


def get_whisper_table(tickers: List[str]) -> pd.DataFrame:
    """
    Get earnings whisper for multiple tickers as DataFrame.

    Usage:
        df = get_whisper_table(['AAPL', 'NVDA', 'MSFT'])
        print(df)
    """
    analyzer = get_whisper_analyzer()

    data = []
    for ticker in tickers:
        try:
            whisper = analyzer.analyze(ticker)
            data.append({
                'Ticker': ticker,
                'Earnings': whisper.earnings_date,
                'Days': whisper.days_to_earnings,
                'Prediction': whisper.prediction.value,
                'Beat Prob': f"{whisper.beat_probability:.0f}%",
                'Exp Surprise': f"{whisper.expected_surprise_pct:+.1f}%",
                'Rev Score': whisper.revision_score,
                'Opt Score': whisper.options_score,
                'Hist Score': whisper.historical_score,
                'Signal': whisper.signal.value,
            })
        except Exception as e:
            logger.debug(f"{ticker}: Whisper analysis failed: {e}")

    return pd.DataFrame(data)


def screen_for_beats(tickers: List[str],
                     min_probability: float = 65) -> List[EarningsWhisper]:
    """
    Screen for stocks likely to beat earnings.

    Usage:
        likely_beats = screen_for_beats(universe, min_probability=70)
        for w in likely_beats:
            print(f"{w.ticker}: {w.beat_probability:.0f}% probability")
    """
    analyzer = get_whisper_analyzer()

    results = []
    for ticker in tickers:
        try:
            whisper = analyzer.analyze(ticker)
            if whisper.beat_probability >= min_probability:
                results.append(whisper)
        except Exception:
            pass

    # Sort by probability
    results.sort(key=lambda x: x.beat_probability, reverse=True)
    return results


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Earnings Whisper...")

    whisper = get_earnings_whisper('NVDA')
    print(whisper.get_summary())

    print("\nComponent Details:")
    if whisper.estimate_revisions:
        print(f"  Revisions: {whisper.estimate_revisions.eps_up_revisions} up, "
              f"{whisper.estimate_revisions.eps_down_revisions} down")

    if whisper.options_positioning:
        print(f"  P/C Ratio: {whisper.options_positioning.put_call_ratio:.2f}")
        print(f"  Smart Money: {whisper.options_positioning.smart_money_direction}")

    if whisper.historical_pattern:
        print(f"  Beat Rate (4Q): {whisper.historical_pattern.beat_rate_4q:.0f}%")
        print(f"  Consecutive Beats: {whisper.historical_pattern.consecutive_beats}")