"""
Alpha Platform - Technical Analysis

Calculates technical indicators: RSI, MACD, moving averages, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import date, timedelta

from src.db.repository import Repository
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TechnicalAnalyzer:
    """Calculates technical indicators for stocks."""

    def __init__(self, repository: Optional[Repository] = None):
        self.repo = repository or Repository()

    def get_price_history(self, ticker: str, days: int = 60) -> pd.DataFrame:
        """Get price history for a ticker."""
        query = """
            SELECT date, open, high, low, close, volume
            FROM prices
            WHERE ticker = %(ticker)s
            ORDER BY date DESC
            LIMIT %(days)s
        """
        df = pd.read_sql(query, self.repo.engine, params={"ticker": ticker, "days": days})

        if len(df) > 0:
            df = df.sort_values('date').reset_index(drop=True)

        return df

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """
        Calculate Relative Strength Index.

        Returns:
            RSI value (0-100), or 50 if insufficient data
        """
        if len(prices) < period + 1:
            return 50.0

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        latest_rsi = rsi.iloc[-1]

        if pd.isna(latest_rsi):
            return 50.0

        return float(latest_rsi)

    def calculate_macd(self, prices: pd.Series,
                       fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        if len(prices) < slow + signal:
            return 0.0, 0.0, 0.0

        exp_fast = prices.ewm(span=fast, adjust=False).mean()
        exp_slow = prices.ewm(span=slow, adjust=False).mean()

        macd_line = exp_fast - exp_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return (
            float(macd_line.iloc[-1]),
            float(signal_line.iloc[-1]),
            float(histogram.iloc[-1])
        )

    def calculate_moving_averages(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate various moving averages."""
        result = {}

        current_price = float(prices.iloc[-1]) if len(prices) > 0 else 0
        result['current_price'] = current_price

        for period in [5, 10, 20, 50]:
            if len(prices) >= period:
                ma = float(prices.tail(period).mean())
                result[f'sma_{period}'] = ma
                result[f'sma_{period}_pct'] = ((current_price - ma) / ma * 100) if ma > 0 else 0
            else:
                result[f'sma_{period}'] = current_price
                result[f'sma_{period}_pct'] = 0

        return result

    def calculate_bollinger_bands(self, prices: pd.Series,
                                   period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            current = float(prices.iloc[-1]) if len(prices) > 0 else 0
            return {'bb_upper': current, 'bb_middle': current, 'bb_lower': current, 'bb_width': 0}

        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        bb_middle = float(middle.iloc[-1])
        bb_upper = float(upper.iloc[-1])
        bb_lower = float(lower.iloc[-1])

        bb_width = ((bb_upper - bb_lower) / bb_middle * 100) if bb_middle > 0 else 0

        return {
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'bb_width': bb_width
        }

    def calculate_volatility(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate historical volatility (annualized)."""
        if len(prices) < period:
            return 0.0

        returns = prices.pct_change().dropna()

        if len(returns) < period:
            return 0.0

        volatility = returns.tail(period).std() * np.sqrt(252) * 100  # Annualized

        return float(volatility)

    def calculate_momentum(self, prices: pd.Series, periods: list = [5, 10, 20]) -> Dict[str, float]:
        """Calculate price momentum over various periods."""
        result = {}
        current = float(prices.iloc[-1]) if len(prices) > 0 else 0

        for period in periods:
            if len(prices) > period:
                past = float(prices.iloc[-period-1])
                momentum = ((current - past) / past * 100) if past > 0 else 0
                result[f'momentum_{period}d'] = momentum
            else:
                result[f'momentum_{period}d'] = 0

        return result

    def calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-based indicators."""
        if len(df) < 20:
            return {'volume_sma_ratio': 1.0, 'volume_trend': 0}

        volume = df['volume']

        # Volume vs 20-day average
        vol_sma = volume.tail(20).mean()
        current_vol = float(volume.iloc[-1])
        vol_ratio = current_vol / vol_sma if vol_sma > 0 else 1.0

        # Volume trend (5-day vs 20-day average)
        vol_5d = volume.tail(5).mean()
        vol_trend = ((vol_5d - vol_sma) / vol_sma * 100) if vol_sma > 0 else 0

        return {
            'volume_sma_ratio': float(vol_ratio),
            'volume_trend': float(vol_trend)
        }

    def analyze_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Run full technical analysis on a ticker.

        Returns:
            Dict with all technical indicators and scores
        """
        df = self.get_price_history(ticker, days=60)

        if len(df) < 5:
            logger.warning(f"{ticker}: Insufficient price data for technical analysis")
            return {
                'ticker': ticker,
                'technical_score': 50,
                'rsi': 50,
                'macd_signal': 'neutral',
                'trend': 'neutral'
            }

        prices = df['close']

        # Calculate all indicators
        rsi = self.calculate_rsi(prices)
        macd_line, signal_line, histogram = self.calculate_macd(prices)
        ma_data = self.calculate_moving_averages(prices)
        bb_data = self.calculate_bollinger_bands(prices)
        volatility = self.calculate_volatility(prices)
        momentum = self.calculate_momentum(prices)
        volume = self.calculate_volume_indicators(df)

        # Determine signals
        # RSI signal
        if rsi > 70:
            rsi_signal = 'overbought'
            rsi_score = 30
        elif rsi < 30:
            rsi_signal = 'oversold'
            rsi_score = 70
        elif rsi > 60:
            rsi_signal = 'bullish'
            rsi_score = 60
        elif rsi < 40:
            rsi_signal = 'bearish'
            rsi_score = 40
        else:
            rsi_signal = 'neutral'
            rsi_score = 50

        # MACD signal
        if histogram > 0 and macd_line > signal_line:
            macd_signal = 'bullish'
            macd_score = 65
        elif histogram < 0 and macd_line < signal_line:
            macd_signal = 'bearish'
            macd_score = 35
        else:
            macd_signal = 'neutral'
            macd_score = 50

        # Trend based on moving averages
        current_price = ma_data['current_price']
        sma_20 = ma_data.get('sma_20', current_price)
        sma_50 = ma_data.get('sma_50', current_price)

        if current_price > sma_20 > sma_50:
            trend = 'uptrend'
            trend_score = 70
        elif current_price < sma_20 < sma_50:
            trend = 'downtrend'
            trend_score = 30
        elif current_price > sma_20:
            trend = 'bullish'
            trend_score = 60
        elif current_price < sma_20:
            trend = 'bearish'
            trend_score = 40
        else:
            trend = 'neutral'
            trend_score = 50

        # Momentum score
        mom_5d = momentum.get('momentum_5d', 0)
        if mom_5d > 5:
            momentum_score = 70
        elif mom_5d > 2:
            momentum_score = 60
        elif mom_5d < -5:
            momentum_score = 30
        elif mom_5d < -2:
            momentum_score = 40
        else:
            momentum_score = 50

        # Composite technical score
        technical_score = int(
            rsi_score * 0.25 +
            macd_score * 0.25 +
            trend_score * 0.30 +
            momentum_score * 0.20
        )

        return {
            'ticker': ticker,
            'technical_score': technical_score,

            # RSI
            'rsi': round(rsi, 2),
            'rsi_signal': rsi_signal,

            # MACD
            'macd_line': round(macd_line, 4),
            'macd_signal_line': round(signal_line, 4),
            'macd_histogram': round(histogram, 4),
            'macd_signal': macd_signal,

            # Moving averages
            'sma_5': round(ma_data.get('sma_5', 0), 2),
            'sma_20': round(ma_data.get('sma_20', 0), 2),
            'sma_50': round(ma_data.get('sma_50', 0), 2),
            'price_vs_sma20': round(ma_data.get('sma_20_pct', 0), 2),

            # Bollinger Bands
            'bb_upper': round(bb_data['bb_upper'], 2),
            'bb_lower': round(bb_data['bb_lower'], 2),
            'bb_width': round(bb_data['bb_width'], 2),

            # Volatility & Momentum
            'volatility': round(volatility, 2),
            'momentum_5d': round(momentum.get('momentum_5d', 0), 2),
            'momentum_20d': round(momentum.get('momentum_20d', 0), 2),

            # Volume
            'volume_ratio': round(volume['volume_sma_ratio'], 2),

            # Overall signals
            'trend': trend,
        }

    def analyze_universe(self, progress_callback=None) -> Dict[str, Dict]:
        """Analyze all tickers in universe."""
        tickers = self.repo.get_universe()
        total = len(tickers)

        results = {}

        for i, ticker in enumerate(tickers):
            if progress_callback:
                progress_callback(i + 1, total, ticker)

            results[ticker] = self.analyze_ticker(ticker)

        return results