"""
Alpha Platform - Gap Analysis

Comprehensive technical gap analysis including:
- Price gap identification (actual open vs previous close)
- Technical indicators (SMA, RSI, MACD, Bollinger Bands)
- Gap classification (Breakaway, Continuation, Exhaustion)
- Trend probability scoring

Based on proven gap analysis methodology.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from src.db.repository import Repository
from src.utils.logging import get_logger

logger = get_logger(__name__)


class GapType(Enum):
    """Gap classification types."""
    STRONG_BULLISH = "STRONG_BULLISH"  # Gap up with all technicals aligned
    BULLISH = "BULLISH"                 # Gap up with some confirmation
    BREAKAWAY_UP = "BREAKAWAY_UP"       # Gap up breaking resistance (start of trend)
    CONTINUATION_UP = "CONTINUATION_UP" # Gap up in existing uptrend
    EXHAUSTION_UP = "EXHAUSTION_UP"     # Gap up at end of uptrend (reversal warning)
    NEUTRAL = "NEUTRAL"                 # No significant gap
    EXHAUSTION_DOWN = "EXHAUSTION_DOWN" # Gap down at end of downtrend
    CONTINUATION_DOWN = "CONTINUATION_DOWN"  # Gap down in existing downtrend
    BREAKAWAY_DOWN = "BREAKAWAY_DOWN"   # Gap down breaking support
    BEARISH = "BEARISH"                 # Gap down with some confirmation
    STRONG_BEARISH = "STRONG_BEARISH"   # Gap down with all technicals aligned


@dataclass
class GapAnalysisResult:
    """Result of gap analysis for a ticker."""
    ticker: str
    gap_score: int              # 0-100 trend probability
    gap_type: str               # GapType value
    gap_pct: float              # Most recent gap percentage

    # Technical indicators
    rsi: float
    above_sma20: bool
    above_sma50: bool
    above_sma200: bool
    macd_bullish: bool
    bb_position: float          # 0-1, position within Bollinger Bands

    # Recent gaps
    recent_gaps: List[Dict]     # Last few gaps with details

    # Price trend
    price_change_10d: float     # 10-day price change %
    price_change_20d: float     # 20-day price change %

    summary: str                # Human-readable summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            'gap_score': self.gap_score,
            'gap_type': self.gap_type,
            'gap_pct': self.gap_pct,
            'rsi': self.rsi,
            'above_sma20': self.above_sma20,
            'above_sma50': self.above_sma50,
            'above_sma200': self.above_sma200,
            'macd_bullish': self.macd_bullish,
            'bb_position': self.bb_position,
            'price_change_10d': self.price_change_10d,
            'price_change_20d': self.price_change_20d,
            'summary': self.summary
        }


class GapAnalyzer:
    """
    Comprehensive gap and technical analysis.
    """

    def __init__(self, repository: Optional[Repository] = None):
        self.repo = repository or Repository()

    def get_price_history(self, ticker: str, days: int = 200) -> pd.DataFrame:
        """Get historical price data from database."""
        query = f"""
            SELECT date, open, high, low, close, volume
            FROM prices
            WHERE ticker = %(ticker)s
            AND date >= NOW() - INTERVAL '{days} days'
            ORDER BY date ASC
        """
        try:
            df = pd.read_sql(query, self.repo.engine, params={"ticker": ticker})
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            return df
        except Exception as e:
            logger.error(f"{ticker}: Failed to get price history - {e}")
            return pd.DataFrame()


    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators.

        Adds columns: SMA20, SMA50, SMA200, RSI, MACD, MACD_SIGNAL, BB_UPPER, BB_MIDDLE, BB_LOWER
        """
        if df.empty or len(df) < 20:
            return df

        prices = df['close'].values

        # Simple Moving Averages
        df['SMA20'] = pd.Series(prices).rolling(window=20).mean().values
        df['SMA50'] = pd.Series(prices).rolling(window=50).mean().values
        df['SMA200'] = pd.Series(prices).rolling(window=200).mean().values

        # RSI (14-period)
        df['RSI'] = self._calculate_rsi(prices, period=14)

        # MACD (12, 26, 9)
        macd, signal, hist = self._calculate_macd(prices)
        df['MACD'] = macd
        df['MACD_SIGNAL'] = signal
        df['MACD_HIST'] = hist

        # Bollinger Bands (20-period, 2 std)
        upper, middle, lower = self._calculate_bollinger_bands(prices)
        df['BB_UPPER'] = upper
        df['BB_MIDDLE'] = middle
        df['BB_LOWER'] = lower

        return df

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI."""
        deltas = np.diff(prices, prepend=prices[0])

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = pd.Series(gains).rolling(window=period).mean().values
        avg_loss = pd.Series(losses).rolling(window=period).mean().values

        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        rsi = np.nan_to_num(rsi, nan=50.0)
        return rsi

    def _calculate_macd(self, prices: np.ndarray,
                        fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD."""
        ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean().values
        ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean().values

        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _calculate_bollinger_bands(self, prices: np.ndarray,
                                    window: int = 20, num_std: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands."""
        rolling_mean = pd.Series(prices).rolling(window=window).mean()
        rolling_std = pd.Series(prices).rolling(window=window).std()

        upper = (rolling_mean + (rolling_std * num_std)).values
        middle = rolling_mean.values
        lower = (rolling_mean - (rolling_std * num_std)).values

        return upper, middle, lower

    def identify_gaps(self, df: pd.DataFrame, min_gap_pct: float = 1.0) -> List[Dict]:
        """
        Identify price gaps (open vs previous close).

        Args:
            df: DataFrame with OHLC data
            min_gap_pct: Minimum gap size as percentage

        Returns:
            List of gap dictionaries with date, gap_pct, direction, etc.
        """
        if df.empty or len(df) < 2:
            return []

        gaps = []

        for i in range(1, len(df)):
            prev_close = df['close'].iloc[i-1]
            curr_open = df['open'].iloc[i]
            curr_close = df['close'].iloc[i]

            if prev_close <= 0:
                continue

            gap_pct = ((curr_open - prev_close) / prev_close) * 100

            if abs(gap_pct) >= min_gap_pct:
                # Check if gap was filled during the day
                if gap_pct > 0:  # Gap up
                    filled = df['low'].iloc[i] <= prev_close
                else:  # Gap down
                    filled = df['high'].iloc[i] >= prev_close

                gaps.append({
                    'date': df.index[i],
                    'gap_pct': round(gap_pct, 2),
                    'direction': 'up' if gap_pct > 0 else 'down',
                    'open': curr_open,
                    'prev_close': prev_close,
                    'close': curr_close,
                    'filled': filled,
                    'filled_same_day': filled
                })

        return gaps

    def classify_gap(self, gap: Dict, df: pd.DataFrame, indicators: pd.DataFrame) -> str:
        """
        Classify a gap as Breakaway, Continuation, or Exhaustion.

        Args:
            gap: Gap dictionary
            df: Price DataFrame
            indicators: DataFrame with technical indicators

        Returns:
            GapType string
        """
        gap_date = gap['date']
        direction = gap['direction']
        gap_pct = abs(gap['gap_pct'])

        # Get indicator values at gap date
        try:
            if gap_date in indicators.index:
                row = indicators.loc[gap_date]
                rsi = row.get('RSI', 50)
                above_sma20 = gap['close'] > row.get('SMA20', gap['close'])
                above_sma50 = gap['close'] > row.get('SMA50', gap['close'])
                macd_bullish = row.get('MACD', 0) > row.get('MACD_SIGNAL', 0)
            else:
                rsi = 50
                above_sma20 = True
                above_sma50 = True
                macd_bullish = True
        except:
            rsi = 50
            above_sma20 = True
            above_sma50 = True
            macd_bullish = True

        # Get trend context (price change over last 20 days before gap)
        gap_idx = df.index.get_loc(gap_date)
        if gap_idx >= 20:
            price_20d_ago = df['close'].iloc[gap_idx - 20]
            trend_pct = ((gap['prev_close'] - price_20d_ago) / price_20d_ago) * 100
        else:
            trend_pct = 0

        if direction == 'up':
            # Strong bullish: All technicals aligned
            if above_sma20 and above_sma50 and macd_bullish and rsi < 70:
                return GapType.STRONG_BULLISH.value

            # Exhaustion gap: Gap up but already overbought or extended
            if rsi > 70 or trend_pct > 15:
                return GapType.EXHAUSTION_UP.value

            # Breakaway: Gap up breaking above resistance
            if not above_sma50 and gap['close'] > indicators.loc[gap_date].get('SMA50', 0):
                return GapType.BREAKAWAY_UP.value

            # Continuation: Gap up in existing uptrend
            if above_sma20 and above_sma50 and trend_pct > 5:
                return GapType.CONTINUATION_UP.value

            return GapType.BULLISH.value

        else:  # direction == 'down'
            # Strong bearish: All technicals aligned
            if not above_sma20 and not above_sma50 and not macd_bullish and rsi > 30:
                return GapType.STRONG_BEARISH.value

            # Exhaustion gap: Gap down but already oversold
            if rsi < 30 or trend_pct < -15:
                return GapType.EXHAUSTION_DOWN.value

            # Breakaway: Gap down breaking below support
            if above_sma50 and gap['close'] < indicators.loc[gap_date].get('SMA50', float('inf')):
                return GapType.BREAKAWAY_DOWN.value

            # Continuation: Gap down in existing downtrend
            if not above_sma20 and not above_sma50 and trend_pct < -5:
                return GapType.CONTINUATION_DOWN.value

            return GapType.BEARISH.value

    def calculate_trend_probability(self, ticker: str, df: pd.DataFrame,
                                     indicators: pd.DataFrame,
                                     recent_gaps: List[Dict]) -> int:
        """
        Calculate trend probability score (0-100).

        Scoring breakdown:
        - Price vs moving averages: 30 points
        - RSI: 10 points
        - MACD: 15 points
        - Bollinger Bands: 15 points
        - Recent gaps: 10 points
        - 10-day price trend: 20 points

        Returns:
            Score from 0 (extremely bearish) to 100 (extremely bullish)
        """
        score = 50  # Start neutral

        if indicators.empty or len(indicators) < 20:
            return score

        try:
            # Get latest values
            latest = indicators.iloc[-1]
            current_price = latest['close'] if 'close' in latest else df['close'].iloc[-1]

            # 1. Price vs Moving Averages (30 points)
            sma20 = latest.get('SMA20')
            sma50 = latest.get('SMA50')
            sma200 = latest.get('SMA200')

            if pd.notna(sma20):
                score += 5 if current_price > sma20 else -5
            if pd.notna(sma50):
                score += 10 if current_price > sma50 else -10
            if pd.notna(sma200):
                score += 15 if current_price > sma200 else -15

            # 2. RSI (10 points)
            rsi = latest.get('RSI', 50)
            if pd.notna(rsi):
                if rsi > 70:
                    score -= 5  # Overbought - bearish short term
                elif rsi < 30:
                    score += 5  # Oversold - bullish short term
                elif rsi > 50:
                    score += 5  # Above 50 generally bullish
                else:
                    score -= 5  # Below 50 generally bearish

            # 3. MACD (15 points)
            macd = latest.get('MACD')
            macd_signal = latest.get('MACD_SIGNAL')

            if pd.notna(macd) and pd.notna(macd_signal):
                if macd > macd_signal:
                    score += 10  # MACD above signal is bullish
                else:
                    score -= 10

                if macd > 0:
                    score += 5  # MACD above zero is bullish
                else:
                    score -= 5

            # 4. Bollinger Bands (15 points)
            bb_upper = latest.get('BB_UPPER')
            bb_middle = latest.get('BB_MIDDLE')
            bb_lower = latest.get('BB_LOWER')

            if pd.notna(bb_upper) and pd.notna(bb_lower) and pd.notna(bb_middle):
                bb_range = bb_upper - bb_lower
                if bb_range > 0:
                    bb_position = (current_price - bb_lower) / bb_range

                    if bb_position > 0.8:
                        score -= 5  # Near upper band - overbought
                    elif bb_position < 0.2:
                        score += 5  # Near lower band - oversold

                    if current_price > bb_middle:
                        score += 10
                    else:
                        score -= 10

            # 5. Recent Gaps (10 points)
            if recent_gaps:
                recent_gap = recent_gaps[-1]  # Most recent
                gap_pct = recent_gap.get('gap_pct', 0)

                if gap_pct > 0:
                    score += min(10, gap_pct)  # Gap up adds points (capped at 10)
                else:
                    score -= min(10, abs(gap_pct))  # Gap down subtracts

            # 6. 10-day Price Trend (20 points)
            if len(df) >= 10:
                price_10d_ago = df['close'].iloc[-10]
                if price_10d_ago > 0:
                    price_change_pct = ((current_price - price_10d_ago) / price_10d_ago) * 100
                    trend_impact = min(20, abs(price_change_pct))

                    if price_change_pct > 0:
                        score += trend_impact
                    else:
                        score -= trend_impact

        except Exception as e:
            logger.error(f"{ticker}: Error calculating trend probability - {e}")

        # Ensure score is between 0 and 100
        return max(0, min(100, int(score)))

    def analyze(self, ticker: str, days: int = 200, min_gap_pct: float = 1.0) -> GapAnalysisResult:
        """
        Run complete gap analysis for a ticker.

        Args:
            ticker: Stock ticker symbol
            days: Days of history to analyze
            min_gap_pct: Minimum gap size as percentage

        Returns:
            GapAnalysisResult with all analysis data
        """
        # Get price data
        df = self.get_price_history(ticker, days)

        if df.empty or len(df) < 20:
            logger.warning(f"{ticker}: Insufficient price data for gap analysis")
            return GapAnalysisResult(
                ticker=ticker,
                gap_score=50,
                gap_type=GapType.NEUTRAL.value,
                gap_pct=0,
                rsi=50,
                above_sma20=False,
                above_sma50=False,
                above_sma200=False,
                macd_bullish=False,
                bb_position=0.5,
                recent_gaps=[],
                price_change_10d=0,
                price_change_20d=0,
                summary="Insufficient data"
            )

        # Calculate indicators
        indicators = self.calculate_indicators(df.copy())

        # Identify gaps
        gaps = self.identify_gaps(df, min_gap_pct)

        # Classify recent gaps
        for gap in gaps[-5:]:  # Last 5 gaps
            gap['type'] = self.classify_gap(gap, df, indicators)

        # Calculate trend probability
        gap_score = self.calculate_trend_probability(ticker, df, indicators, gaps)

        # Get latest indicator values
        latest = indicators.iloc[-1]
        current_price = df['close'].iloc[-1]

        rsi = latest.get('RSI', 50)
        sma20 = latest.get('SMA20', current_price)
        sma50 = latest.get('SMA50', current_price)
        sma200 = latest.get('SMA200', current_price)
        macd = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_SIGNAL', 0)
        bb_upper = latest.get('BB_UPPER', current_price)
        bb_lower = latest.get('BB_LOWER', current_price)

        above_sma20 = current_price > sma20 if pd.notna(sma20) else False
        above_sma50 = current_price > sma50 if pd.notna(sma50) else False
        above_sma200 = current_price > sma200 if pd.notna(sma200) else False
        macd_bullish = macd > macd_signal if pd.notna(macd) and pd.notna(macd_signal) else False

        # Bollinger Band position
        bb_range = bb_upper - bb_lower if pd.notna(bb_upper) and pd.notna(bb_lower) else 1
        bb_position = (current_price - bb_lower) / bb_range if bb_range > 0 else 0.5

        # Price changes
        price_10d = ((current_price - df['close'].iloc[-10]) / df['close'].iloc[-10] * 100) if len(df) >= 10 else 0
        price_20d = ((current_price - df['close'].iloc[-20]) / df['close'].iloc[-20] * 100) if len(df) >= 20 else 0

        # Determine overall gap type
        if gaps:
            recent_gap = gaps[-1]
            gap_type = recent_gap.get('type', GapType.NEUTRAL.value)
            gap_pct = recent_gap.get('gap_pct', 0)
        else:
            gap_type = GapType.NEUTRAL.value
            gap_pct = 0

        # Generate summary
        trend = "Bullish" if gap_score >= 60 else "Bearish" if gap_score <= 40 else "Neutral"
        ma_status = []
        if above_sma20:
            ma_status.append("SMA20")
        if above_sma50:
            ma_status.append("SMA50")
        if above_sma200:
            ma_status.append("SMA200")

        ma_text = f"Above {', '.join(ma_status)}" if ma_status else "Below all MAs"
        summary = f"{trend} ({gap_score}/100). RSI: {rsi:.1f}. {ma_text}. MACD: {'Bullish' if macd_bullish else 'Bearish'}."

        if gaps:
            summary += f" Last gap: {gap_pct:+.1f}% ({gap_type})"

        logger.info(f"{ticker}: Gap analysis complete - Score={gap_score}, Type={gap_type}")

        return GapAnalysisResult(
            ticker=ticker,
            gap_score=gap_score,
            gap_type=gap_type,
            gap_pct=gap_pct,
            rsi=round(rsi, 1) if pd.notna(rsi) else 50,
            above_sma20=above_sma20,
            above_sma50=above_sma50,
            above_sma200=above_sma200,
            macd_bullish=macd_bullish,
            bb_position=round(bb_position, 2),
            recent_gaps=gaps[-5:] if gaps else [],
            price_change_10d=round(price_10d, 2),
            price_change_20d=round(price_20d, 2),
            summary=summary
        )


def test_gap_analysis():
    """Test gap analysis."""
    analyzer = GapAnalyzer()

    tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA']

    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"{ticker} Gap Analysis")
        print('='*60)

        result = analyzer.analyze(ticker)

        print(f"Gap Score: {result.gap_score}/100")
        print(f"Gap Type: {result.gap_type}")
        print(f"Last Gap: {result.gap_pct:+.2f}%")
        print(f"RSI: {result.rsi}")
        print(f"Above SMA20: {result.above_sma20}")
        print(f"Above SMA50: {result.above_sma50}")
        print(f"Above SMA200: {result.above_sma200}")
        print(f"MACD Bullish: {result.macd_bullish}")
        print(f"BB Position: {result.bb_position}")
        print(f"10D Change: {result.price_change_10d:+.2f}%")
        print(f"20D Change: {result.price_change_20d:+.2f}%")
        print(f"\nSummary: {result.summary}")

        if result.recent_gaps:
            print(f"\nRecent Gaps:")
            for gap in result.recent_gaps[-3:]:
                print(f"  {gap['date'].strftime('%Y-%m-%d')}: {gap['gap_pct']:+.2f}% ({gap.get('type', 'N/A')})")


if __name__ == "__main__":
    test_gap_analysis()