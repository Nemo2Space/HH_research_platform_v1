"""
Earnings Intelligence System - Price Drift Calculator

Calculates price drift metrics for IES (Implied Expectations Score):
- drift_20d: 20-day price return before earnings
- rel_drift_20d: Relative performance vs sector ETF

These measure how much the stock has run up (or down) into earnings,
which indicates market expectations.

Author: Alpha Research Platform
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np
import yfinance as yf

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Sector ETF mappings (from technical_analysis.py)
SECTOR_ETFS = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financials': 'XLF',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Industrials': 'XLI',
    'Materials': 'XLB',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Communication Services': 'XLC',
    # Aliases
    'Information Technology': 'XLK',
    'Consumer Cyclical': 'XLY',
    'Consumer Defensive': 'XLP',
}

# Default ETF for unknown sectors
DEFAULT_SECTOR_ETF = 'SPY'

# Cache for price data to reduce API calls
_price_cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
CACHE_EXPIRY_MINUTES = 60


def _get_cached_prices(ticker: str, period: str = "3mo") -> Optional[pd.DataFrame]:
    """
    Get price data with caching.

    Args:
        ticker: Stock symbol
        period: yfinance period string

    Returns:
        DataFrame with OHLCV data or None
    """
    cache_key = f"{ticker}_{period}"

    # Check cache
    if cache_key in _price_cache:
        df, cached_at = _price_cache[cache_key]
        age_minutes = (datetime.now() - cached_at).total_seconds() / 60
        if age_minutes < CACHE_EXPIRY_MINUTES:
            return df

    # Fetch from yfinance
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df is not None and not df.empty:
            _price_cache[cache_key] = (df, datetime.now())
            return df
        else:
            logger.warning(f"{ticker}: No price data returned")
            return None

    except Exception as e:
        logger.error(f"{ticker}: Error fetching prices: {e}")
        return None


def get_sector_etf(sector: str) -> str:
    """
    Get the sector ETF for a given sector name.

    Args:
        sector: Sector name (e.g., 'Technology', 'Healthcare')

    Returns:
        ETF symbol (e.g., 'XLK', 'XLV')
    """
    if not sector:
        return DEFAULT_SECTOR_ETF

    return SECTOR_ETFS.get(sector, DEFAULT_SECTOR_ETF)


def get_ticker_sector(ticker: str) -> Optional[str]:
    """
    Get the sector for a ticker from yfinance.

    Args:
        ticker: Stock symbol

    Returns:
        Sector name or None
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('sector')
    except Exception as e:
        logger.debug(f"{ticker}: Could not get sector: {e}")
        return None


def calculate_return(prices: pd.DataFrame, days: int) -> Optional[float]:
    """
    Calculate return over N trading days.

    Args:
        prices: DataFrame with 'Close' column
        days: Number of trading days

    Returns:
        Return as decimal (e.g., 0.15 for 15%) or None
    """
    if prices is None or prices.empty:
        return None

    if len(prices) < days + 1:
        logger.debug(f"Insufficient price data: {len(prices)} rows, need {days + 1}")
        return None

    try:
        # Get most recent close and close N days ago
        current_close = prices['Close'].iloc[-1]
        past_close = prices['Close'].iloc[-(days + 1)]

        if past_close == 0 or pd.isna(past_close) or pd.isna(current_close):
            return None

        return (current_close - past_close) / past_close

    except Exception as e:
        logger.debug(f"Error calculating return: {e}")
        return None


def calculate_drift_20d(ticker: str) -> Optional[float]:
    """
    Calculate 20-day price drift (return) for a ticker.

    This measures how much the stock has moved in the 20 trading days
    leading up to the current date (or earnings date if specified).

    A high positive drift suggests the market is optimistic and
    expectations may be elevated.

    Args:
        ticker: Stock symbol

    Returns:
        20-day return as decimal (e.g., 0.15 for +15%), or None if unavailable
    """
    prices = _get_cached_prices(ticker, period="3mo")

    if prices is None:
        return None

    drift = calculate_return(prices, 20)

    if drift is not None:
        logger.debug(f"{ticker}: 20-day drift = {drift:.2%}")

    return drift


def calculate_relative_drift(ticker: str, sector: str = None) -> Optional[float]:
    """
    Calculate relative 20-day drift vs sector ETF.

    This measures outperformance/underperformance relative to the sector,
    isolating stock-specific expectations from sector-wide moves.

    Args:
        ticker: Stock symbol
        sector: Sector name (will be fetched if not provided)

    Returns:
        Relative drift as decimal (e.g., 0.05 for +5% outperformance), or None
    """
    # Get ticker drift
    ticker_drift = calculate_drift_20d(ticker)
    if ticker_drift is None:
        return None

    # Get sector if not provided
    if not sector:
        sector = get_ticker_sector(ticker)

    # Get sector ETF
    sector_etf = get_sector_etf(sector)

    # Get ETF drift
    etf_prices = _get_cached_prices(sector_etf, period="3mo")
    if etf_prices is None:
        logger.warning(f"Could not get prices for sector ETF {sector_etf}")
        return None

    etf_drift = calculate_return(etf_prices, 20)
    if etf_drift is None:
        return None

    # Calculate relative drift
    rel_drift = ticker_drift - etf_drift

    logger.debug(f"{ticker}: rel_drift = {rel_drift:.2%} (stock: {ticker_drift:.2%}, {sector_etf}: {etf_drift:.2%})")

    return rel_drift


def calculate_drift_for_date(ticker: str, target_date: date, lookback_days: int = 20) -> Optional[float]:
    """
    Calculate drift as of a specific date (for historical analysis).

    Args:
        ticker: Stock symbol
        target_date: Date to calculate drift as of
        lookback_days: Number of trading days to look back

    Returns:
        Drift as decimal, or None
    """
    try:
        # Fetch enough history
        start_date = target_date - timedelta(days=lookback_days * 2)  # Extra buffer for weekends
        end_date = target_date + timedelta(days=1)

        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df is None or df.empty:
            return None

        # Filter to dates <= target_date
        df = df[df.index.date <= target_date]

        if len(df) < lookback_days + 1:
            logger.debug(f"{ticker}: Insufficient data for {target_date}")
            return None

        return calculate_return(df, lookback_days)

    except Exception as e:
        logger.error(f"{ticker}: Error calculating historical drift: {e}")
        return None


def calculate_all_drift_metrics(ticker: str, sector: str = None) -> Dict[str, Optional[float]]:
    """
    Calculate all drift metrics for a ticker.

    Args:
        ticker: Stock symbol
        sector: Sector name (optional)

    Returns:
        Dict with drift_20d and rel_drift_20d
    """
    # Get sector if not provided
    if not sector:
        sector = get_ticker_sector(ticker)

    return {
        'drift_20d': calculate_drift_20d(ticker),
        'rel_drift_20d': calculate_relative_drift(ticker, sector),
        'sector': sector,
        'sector_etf': get_sector_etf(sector),
    }


def normalize_drift_to_score(drift: float,
                              low_threshold: float = -0.10,
                              high_threshold: float = 0.20) -> float:
    """
    Normalize drift to 0-100 score for IES calculation.

    Higher drift → Higher expectations → Higher score

    Args:
        drift: Drift as decimal (e.g., 0.15 for 15%)
        low_threshold: Drift value that maps to score 0
        high_threshold: Drift value that maps to score 100

    Returns:
        Score from 0-100
    """
    if drift is None:
        return 50.0  # Neutral if unknown

    # Linear interpolation
    if drift <= low_threshold:
        return 0.0
    elif drift >= high_threshold:
        return 100.0
    else:
        # Linear scale between thresholds
        range_size = high_threshold - low_threshold
        position = (drift - low_threshold) / range_size
        return position * 100.0


def clear_price_cache():
    """Clear the price cache."""
    global _price_cache
    _price_cache = {}
    logger.debug("Price cache cleared")


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 4 TEST: Price Drift Calculator")
    print("=" * 60)

    test_tickers = ["NVDA", "AAPL", "XOM", "JNJ", "JPM"]

    print("\nCalculating drift metrics...")
    print("-" * 60)

    for ticker in test_tickers:
        metrics = calculate_all_drift_metrics(ticker)

        drift_20d = metrics['drift_20d']
        rel_drift = metrics['rel_drift_20d']
        sector = metrics['sector']
        etf = metrics['sector_etf']

        drift_str = f"{drift_20d:+.2%}" if drift_20d is not None else "N/A"
        rel_str = f"{rel_drift:+.2%}" if rel_drift is not None else "N/A"

        print(f"{ticker}:")
        print(f"  Sector: {sector} ({etf})")
        print(f"  20-day Drift: {drift_str}")
        print(f"  Relative Drift: {rel_str}")

        if drift_20d is not None:
            score = normalize_drift_to_score(drift_20d)
            print(f"  Drift Score: {score:.0f}/100")
        print()

    # Test score normalization
    print("-" * 60)
    print("Testing score normalization:")
    test_drifts = [-0.15, -0.05, 0.0, 0.10, 0.25, 0.50]
    for d in test_drifts:
        score = normalize_drift_to_score(d)
        print(f"  Drift {d:+.0%} -> Score {score:.0f}")

    print("\n" + "=" * 60)
    print("[OK] Phase 4 tests complete!")
    print("=" * 60)