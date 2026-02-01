"""
Options-Based IES Inputs Calculator

Computes options-derived inputs for the Implied Expectations Score (IES):
- implied_move_pct: Expected move from ATM straddle / stock price
- implied_move_pctl: Percentile vs historical earnings
- iv_pctl: IV percentile vs historical range
- skew_shift: Call IV vs Put IV directional bias

Author: Alpha Research Platform
Phase: 5 of Earnings Intelligence System
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yfinance as yf
import json
import os

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class OptionsMetrics:
    """Options-derived metrics for a ticker."""
    ticker: str
    stock_price: float

    # ATM Straddle metrics
    atm_call_price: float
    atm_put_price: float
    atm_straddle_price: float
    atm_strike: float

    # Implied move
    implied_move_pct: float  # ATM straddle / stock price (e.g., 0.08 = 8%)
    implied_move_pctl: Optional[float]  # Percentile vs history (0-100)

    # IV metrics
    atm_call_iv: float  # ATM call IV (decimal, e.g., 0.45 = 45%)
    atm_put_iv: float   # ATM put IV
    avg_iv: float       # Average of call and put IV
    iv_pctl: Optional[float]  # Percentile vs history (0-100)

    # Skew metrics
    iv_skew: float           # Put IV - Call IV (positive = bearish skew)
    skew_shift: Optional[float]  # Change in skew over 5-10 days

    # Meta
    expiry_used: str
    days_to_expiry: int
    timestamp: datetime


@dataclass
class HistoricalOptionsData:
    """Historical options data for percentile calculations."""
    ticker: str
    earnings_date: str
    implied_move_pct: float
    avg_iv: float
    iv_skew: float


# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

# In-memory cache for options data
_options_cache: Dict[str, Tuple[OptionsMetrics, datetime]] = {}
_OPTIONS_CACHE_EXPIRY_MINUTES = 30  # Refresh every 30 minutes

# Historical data cache file
_HISTORICAL_CACHE_FILE = "data/earnings_options_history.json"
_HISTORICAL_CACHE_EXPIRY_DAYS = 7  # Refresh historical data weekly


def clear_options_cache():
    """Clear the in-memory options cache."""
    global _options_cache
    _options_cache.clear()
    logger.debug("Options cache cleared")


# ============================================================================
# CORE OPTIONS DATA RETRIEVAL
# ============================================================================

def _get_nearest_expiry_after_date(ticker: str, target_date: date) -> Optional[str]:
    """
    Get the nearest options expiry date that falls after the target date.
    Used to find options that capture the earnings event.

    Args:
        ticker: Stock symbol
        target_date: Date to look for expiry after (usually earnings date)

    Returns:
        Expiry date string (YYYY-MM-DD) or None if not found
    """
    try:
        stock = yf.Ticker(ticker)
        expiries = stock.options

        if not expiries:
            return None

        for expiry in expiries:
            exp_date = datetime.strptime(expiry, '%Y-%m-%d').date()
            if exp_date >= target_date:
                return expiry

        return None

    except Exception as e:
        logger.error(f"Error getting expiries for {ticker}: {e}")
        return None


def _interpolate_atm_options(
    options_df: pd.DataFrame,
    stock_price: float,
    option_type: str = 'call'
) -> Tuple[float, float, float]:
    """
    Interpolate ATM option price and IV between two nearest strikes.

    Options rarely have exact ATM strikes. This function finds the strikes
    immediately above and below the current stock price and linearly
    interpolates to estimate the true ATM values.

    Args:
        options_df: DataFrame with strike, lastPrice, impliedVolatility columns
        stock_price: Current stock price
        option_type: 'call' or 'put' for logging

    Returns:
        Tuple of (interpolated_price, interpolated_iv, atm_strike)
    """
    if options_df.empty:
        return 0.0, 0.0, stock_price

    # Get strikes above and below stock price
    strikes = options_df['strike'].values
    below_strikes = strikes[strikes <= stock_price]
    above_strikes = strikes[strikes >= stock_price]

    if len(below_strikes) == 0 or len(above_strikes) == 0:
        # Edge case: stock price outside strike range
        # Use nearest available strike
        nearest_idx = (np.abs(strikes - stock_price)).argmin()
        row = options_df.iloc[nearest_idx]
        price = row.get('lastPrice', 0) or row.get('ask', 0) or 0
        iv = row.get('impliedVolatility', 0) or 0
        return float(price), float(iv), float(row['strike'])

    strike_below = below_strikes.max()
    strike_above = above_strikes.min()

    if strike_below == strike_above:
        # Exact ATM strike exists
        row = options_df[options_df['strike'] == strike_below].iloc[0]
        price = row.get('lastPrice', 0) or row.get('ask', 0) or 0
        iv = row.get('impliedVolatility', 0) or 0
        return float(price), float(iv), float(strike_below)

    # Get data for both strikes
    row_below = options_df[options_df['strike'] == strike_below].iloc[0]
    row_above = options_df[options_df['strike'] == strike_above].iloc[0]

    price_below = row_below.get('lastPrice', 0) or row_below.get('ask', 0) or 0
    price_above = row_above.get('lastPrice', 0) or row_above.get('ask', 0) or 0
    iv_below = row_below.get('impliedVolatility', 0) or 0
    iv_above = row_above.get('impliedVolatility', 0) or 0

    # Linear interpolation weight
    weight = (stock_price - strike_below) / (strike_above - strike_below)

    interp_price = price_below + weight * (price_above - price_below)
    interp_iv = iv_below + weight * (iv_above - iv_below)
    atm_strike = strike_below + weight * (strike_above - strike_below)

    return float(interp_price), float(interp_iv), float(atm_strike)


def get_options_metrics(ticker: str, earnings_date: Optional[date] = None) -> Optional[OptionsMetrics]:
    """
    Get comprehensive options metrics for a ticker.

    Fetches options data for the expiry immediately following earnings (or nearest
    expiry if no earnings date provided) and computes:
    - ATM straddle price and implied move
    - Average IV and IV skew

    Args:
        ticker: Stock symbol
        earnings_date: Optional earnings date to find relevant expiry

    Returns:
        OptionsMetrics dataclass or None if data unavailable
    """
    global _options_cache

    # Check cache
    cache_key = f"{ticker}_{earnings_date or 'default'}"
    if cache_key in _options_cache:
        cached_data, cached_time = _options_cache[cache_key]
        if datetime.now() - cached_time < timedelta(minutes=_OPTIONS_CACHE_EXPIRY_MINUTES):
            return cached_data

    try:
        stock = yf.Ticker(ticker)

        # Get stock price
        info = stock.info
        stock_price = (
            info.get('currentPrice') or
            info.get('regularMarketPrice') or
            info.get('previousClose', 0)
        )

        if not stock_price:
            hist = stock.history(period='1d')
            if not hist.empty:
                stock_price = hist['Close'].iloc[-1]

        if not stock_price or stock_price <= 0:
            logger.warning(f"Cannot get stock price for {ticker}")
            return None

        # Determine which expiry to use
        if earnings_date:
            expiry = _get_nearest_expiry_after_date(ticker, earnings_date)
        else:
            expiries = stock.options
            expiry = expiries[0] if expiries else None

        if not expiry:
            logger.warning(f"No valid expiry found for {ticker}")
            return None

        # Get options chain for this expiry
        opt_chain = stock.option_chain(expiry)
        calls = opt_chain.calls
        puts = opt_chain.puts

        if calls.empty or puts.empty:
            logger.warning(f"Empty options chain for {ticker} expiry {expiry}")
            return None

        # Filter out options with invalid IV (>200% or 0)
        calls = calls[(calls['impliedVolatility'] > 0) & (calls['impliedVolatility'] < 2.0)]
        puts = puts[(puts['impliedVolatility'] > 0) & (puts['impliedVolatility'] < 2.0)]

        if calls.empty or puts.empty:
            logger.warning(f"No valid IV data for {ticker}")
            return None

        # Calculate ATM interpolated values
        call_price, call_iv, atm_strike = _interpolate_atm_options(calls, stock_price, 'call')
        put_price, put_iv, _ = _interpolate_atm_options(puts, stock_price, 'put')

        # Calculate straddle price and implied move
        straddle_price = call_price + put_price
        implied_move_pct = straddle_price / stock_price if stock_price > 0 else 0

        # Calculate average IV and skew
        avg_iv = (call_iv + put_iv) / 2 if (call_iv > 0 and put_iv > 0) else max(call_iv, put_iv)
        iv_skew = put_iv - call_iv  # Positive = bearish skew

        # Calculate days to expiry
        exp_date = datetime.strptime(expiry, '%Y-%m-%d')
        days_to_expiry = (exp_date - datetime.now()).days

        metrics = OptionsMetrics(
            ticker=ticker,
            stock_price=stock_price,
            atm_call_price=call_price,
            atm_put_price=put_price,
            atm_straddle_price=straddle_price,
            atm_strike=atm_strike,
            implied_move_pct=implied_move_pct,
            implied_move_pctl=None,  # Calculated separately
            atm_call_iv=call_iv,
            atm_put_iv=put_iv,
            avg_iv=avg_iv,
            iv_pctl=None,  # Calculated separately
            iv_skew=iv_skew,
            skew_shift=None,  # Calculated separately
            expiry_used=expiry,
            days_to_expiry=days_to_expiry,
            timestamp=datetime.now()
        )

        # Cache result
        _options_cache[cache_key] = (metrics, datetime.now())

        return metrics

    except Exception as e:
        logger.error(f"Error getting options metrics for {ticker}: {e}")
        return None


# ============================================================================
# HISTORICAL DATA FOR PERCENTILE CALCULATIONS
# ============================================================================

def _load_historical_cache() -> Dict[str, List[HistoricalOptionsData]]:
    """Load historical options data from cache file."""
    if not os.path.exists(_HISTORICAL_CACHE_FILE):
        return {}

    try:
        with open(_HISTORICAL_CACHE_FILE, 'r') as f:
            data = json.load(f)

        # Convert to dataclass objects
        result = {}
        for ticker, history in data.get('history', {}).items():
            result[ticker] = [
                HistoricalOptionsData(**h) for h in history
            ]
        return result

    except Exception as e:
        logger.error(f"Error loading historical cache: {e}")
        return {}


def _save_historical_cache(cache: Dict[str, List[HistoricalOptionsData]]) -> None:
    """Save historical options data to cache file."""
    try:
        os.makedirs(os.path.dirname(_HISTORICAL_CACHE_FILE), exist_ok=True)

        # Convert to JSON-serializable format
        data = {
            'updated': datetime.now().isoformat(),
            'history': {
                ticker: [
                    {
                        'ticker': h.ticker,
                        'earnings_date': h.earnings_date,
                        'implied_move_pct': h.implied_move_pct,
                        'avg_iv': h.avg_iv,
                        'iv_skew': h.iv_skew
                    }
                    for h in history
                ]
                for ticker, history in cache.items()
            }
        }

        with open(_HISTORICAL_CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        logger.error(f"Error saving historical cache: {e}")


def get_historical_options_data(ticker: str, num_quarters: int = 8) -> List[HistoricalOptionsData]:
    """
    Get historical options metrics for past earnings events.

    This data is used to calculate percentile ranks for current implied move
    and IV versus historical ranges.

    Args:
        ticker: Stock symbol
        num_quarters: Number of historical quarters to retrieve

    Returns:
        List of HistoricalOptionsData for past earnings
    """
    # First try to load from cache
    cache = _load_historical_cache()

    if ticker in cache and len(cache[ticker]) >= num_quarters:
        return cache[ticker][:num_quarters]

    # Need to fetch historical data
    # NOTE: This is a simplified implementation. In production, you would:
    # 1. Query your database for historical earnings dates
    # 2. Use stored options snapshots from before each earnings
    #
    # For now, we'll estimate based on typical quarterly patterns
    historical_data = []

    try:
        stock = yf.Ticker(ticker)

        # Get earnings dates from yfinance
        earnings = stock.earnings_dates

        if earnings is not None and not earnings.empty:
            # Get current time - make it timezone-aware if earnings index is tz-aware
            now = datetime.now()

            # Check if earnings index is timezone-aware
            if earnings.index.tz is not None:
                # Make now timezone-aware to match
                import pytz
                now = now.replace(tzinfo=pytz.UTC)

            # Get past earnings dates
            past_earnings = earnings[earnings.index < now].head(num_quarters)

            # For each past earnings, we'd ideally have stored options data
            # Since we don't, we'll estimate based on historical price action
            hist = stock.history(period='2y')

            for earnings_date in past_earnings.index:
                try:
                    # Convert to date object for comparison
                    if hasattr(earnings_date, 'date'):
                        ed = earnings_date.date()
                    else:
                        ed = earnings_date

                    # Look for price data around this earnings
                    # Find the actual price reaction to estimate what was implied
                    start_date = ed - timedelta(days=5)
                    end_date = ed + timedelta(days=3)

                    # hist.index may also be timezone-aware, convert to date for comparison
                    hist_dates = hist.index.date if hasattr(hist.index, 'date') else hist.index
                    period_data = hist[(hist_dates >= start_date) & (hist_dates <= end_date)]

                    if len(period_data) >= 2:
                        # Estimate implied move from actual move (rough approximation)
                        pre_earnings_price = period_data.iloc[0]['Close']
                        post_earnings_price = period_data.iloc[-1]['Close']
                        actual_move = abs(post_earnings_price - pre_earnings_price) / pre_earnings_price

                        # Historical implied moves tend to be slightly higher than actual
                        # Using 1.2x multiplier as a heuristic
                        estimated_implied = actual_move * 1.2

                        # Estimate IV (typically 30-80% for most stocks around earnings)
                        base_iv = 0.35  # 35% baseline
                        iv_scaling = 1 + (estimated_implied * 2)  # Higher move = higher IV
                        estimated_iv = min(base_iv * iv_scaling, 1.5)  # Cap at 150%

                        historical_data.append(HistoricalOptionsData(
                            ticker=ticker,
                            earnings_date=str(ed),
                            implied_move_pct=estimated_implied,
                            avg_iv=estimated_iv,
                            iv_skew=0.02  # Slight bearish skew as default
                        ))

                except Exception as e:
                    logger.debug(f"Error processing historical earnings for {ticker}: {e}")
                    continue

        # Cache the results
        if historical_data:
            cache[ticker] = historical_data
            _save_historical_cache(cache)

    except Exception as e:
        logger.error(f"Error fetching historical options data for {ticker}: {e}")

    return historical_data


# ============================================================================
# IES INPUT CALCULATIONS
# ============================================================================

def calculate_implied_move_pct(ticker: str, earnings_date: Optional[date] = None) -> Optional[float]:
    """
    Calculate the implied move percentage from ATM straddle.

    Formula: implied_move_pct = ATM_straddle_price / stock_price

    Example: Stock at $100, ATM straddle at $8 → 8% implied move

    Args:
        ticker: Stock symbol
        earnings_date: Optional earnings date to find relevant expiry

    Returns:
        Implied move as decimal (e.g., 0.08 = 8%) or None if unavailable
    """
    metrics = get_options_metrics(ticker, earnings_date)

    if metrics is None:
        return None

    return metrics.implied_move_pct


def calculate_implied_move_pctl(ticker: str, earnings_date: Optional[date] = None) -> Optional[float]:
    """
    Calculate the percentile rank of current implied move vs historical earnings.

    Example: If current implied move of 8% is higher than 7 out of 8 historical
    earnings cycles, the percentile is 87.5.

    Args:
        ticker: Stock symbol
        earnings_date: Optional earnings date for context

    Returns:
        Percentile (0-100) or None if insufficient history
    """
    current_metrics = get_options_metrics(ticker, earnings_date)

    if current_metrics is None:
        return None

    current_im = current_metrics.implied_move_pct

    # Get historical data
    history = get_historical_options_data(ticker)

    if len(history) < 4:
        # Insufficient history for reliable percentile
        logger.debug(f"Insufficient history for {ticker} implied move percentile")
        return None

    historical_ims = [h.implied_move_pct for h in history]

    # Calculate percentile
    count_below = sum(1 for h in historical_ims if h < current_im)
    percentile = (count_below / len(historical_ims)) * 100

    # Clamp to 5-95 range per spec
    percentile = max(5, min(95, percentile))

    return percentile


def calculate_iv_pctl(ticker: str, earnings_date: Optional[date] = None) -> Optional[float]:
    """
    Calculate the percentile rank of current IV vs historical earnings.

    High IV percentile indicates the options market expects larger-than-normal
    volatility around this earnings event.

    Args:
        ticker: Stock symbol
        earnings_date: Optional earnings date for context

    Returns:
        Percentile (0-100) or None if insufficient history
    """
    current_metrics = get_options_metrics(ticker, earnings_date)

    if current_metrics is None:
        return None

    current_iv = current_metrics.avg_iv

    # Get historical data
    history = get_historical_options_data(ticker)

    if len(history) < 4:
        logger.debug(f"Insufficient history for {ticker} IV percentile")
        return None

    historical_ivs = [h.avg_iv for h in history]

    # Calculate percentile
    count_below = sum(1 for h in historical_ivs if h < current_iv)
    percentile = (count_below / len(historical_ivs)) * 100

    # Clamp to 5-95 range
    percentile = max(5, min(95, percentile))

    return percentile


def calculate_skew_shift(
    ticker: str,
    lookback_days: int = 5,
    earnings_date: Optional[date] = None
) -> Optional[float]:
    """
    Calculate the change in IV skew over recent days.

    Skew shift measures whether call IV has increased relative to put IV
    (bullish positioning) or vice versa (bearish positioning).

    Formula: skew_shift = (current_call_iv - current_put_iv) - (prior_call_iv - prior_put_iv)

    Positive skew_shift = calls getting bid up = bullish positioning
    Negative skew_shift = puts getting bid up = bearish positioning

    Args:
        ticker: Stock symbol
        lookback_days: Days to look back for prior skew
        earnings_date: Optional earnings date for context

    Returns:
        Skew shift value or None if unavailable

    Note:
        This is a simplified implementation. Full production version would
        track daily IV snapshots and compute actual shift.
    """
    current_metrics = get_options_metrics(ticker, earnings_date)

    if current_metrics is None:
        return None

    # Current skew (negative of iv_skew since iv_skew = put_iv - call_iv)
    current_call_vs_put = -current_metrics.iv_skew

    # For a proper implementation, we'd need historical IV snapshots
    # As a heuristic, we'll use the historical average skew as the baseline
    history = get_historical_options_data(ticker)

    if len(history) < 2:
        # If no history, return 0 (neutral)
        return 0.0

    # Historical average skew
    avg_historical_skew = np.mean([h.iv_skew for h in history])

    # Current skew vs historical baseline
    skew_shift = current_call_vs_put - (-avg_historical_skew)

    return skew_shift


def normalize_implied_move_to_score(
    implied_move_pctl: Optional[float],
    default: float = 50.0
) -> float:
    """
    Normalize implied move percentile to a 0-100 score for IES.

    The implied_move_pctl is already 0-100, but we may want to apply
    additional transformations for IES weighting.

    Args:
        implied_move_pctl: Percentile (0-100)
        default: Default value if percentile is None

    Returns:
        Score (0-100)
    """
    if implied_move_pctl is None:
        return default

    # Clamp and return as-is (percentile already suitable for IES)
    return max(0, min(100, implied_move_pctl))


def normalize_iv_to_score(
    iv_pctl: Optional[float],
    default: float = 50.0
) -> float:
    """
    Normalize IV percentile to a 0-100 score for IES.

    Args:
        iv_pctl: IV percentile (0-100)
        default: Default value if percentile is None

    Returns:
        Score (0-100)
    """
    if iv_pctl is None:
        return default

    return max(0, min(100, iv_pctl))


def normalize_skew_to_score(
    skew_shift: Optional[float],
    default: float = 50.0
) -> float:
    """
    Normalize skew shift to a 0-100 score for IES.

    Skew shift interpretation:
    - Positive (calls bid) = bullish = higher expectations
    - Negative (puts bid) = bearish = lower expectations

    Scoring:
    - skew_shift < -0.05: 0-30 (bearish)
    - skew_shift ~ 0: 40-60 (neutral)
    - skew_shift > +0.05: 70-100 (bullish)

    Args:
        skew_shift: Change in call IV vs put IV
        default: Default value if skew_shift is None

    Returns:
        Score (0-100)
    """
    if skew_shift is None:
        return default

    # Linear scaling with clamping
    # -0.10 maps to 0, +0.10 maps to 100, 0 maps to 50
    score = 50 + (skew_shift / 0.10) * 50

    return max(0, min(100, score))


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def calculate_all_options_inputs(
    ticker: str,
    earnings_date: Optional[date] = None
) -> Dict[str, Optional[float]]:
    """
    Calculate all options-based IES inputs at once.

    Args:
        ticker: Stock symbol
        earnings_date: Optional earnings date

    Returns:
        Dictionary with all options inputs:
        - implied_move_pct: Raw implied move (decimal)
        - implied_move_pctl: Percentile vs history (0-100)
        - implied_move_score: Normalized score for IES (0-100)
        - iv_pctl: IV percentile vs history (0-100)
        - iv_score: Normalized score for IES (0-100)
        - skew_shift: Raw skew shift value
        - skew_score: Normalized score for IES (0-100)
        - avg_iv: Current average IV (decimal)
        - stock_price: Current stock price
        - straddle_price: ATM straddle price
        - expiry_used: Options expiry used
        - days_to_expiry: Days until expiry
    """
    metrics = get_options_metrics(ticker, earnings_date)

    if metrics is None:
        return {
            'implied_move_pct': None,
            'implied_move_pctl': None,
            'implied_move_score': 50.0,
            'iv_pctl': None,
            'iv_score': 50.0,
            'skew_shift': None,
            'skew_score': 50.0,
            'avg_iv': None,
            'stock_price': None,
            'straddle_price': None,
            'expiry_used': None,
            'days_to_expiry': None
        }

    # Calculate percentiles
    implied_move_pctl = calculate_implied_move_pctl(ticker, earnings_date)
    iv_pctl = calculate_iv_pctl(ticker, earnings_date)
    skew_shift = calculate_skew_shift(ticker, earnings_date=earnings_date)

    return {
        'implied_move_pct': metrics.implied_move_pct,
        'implied_move_pctl': implied_move_pctl,
        'implied_move_score': normalize_implied_move_to_score(implied_move_pctl),
        'iv_pctl': iv_pctl,
        'iv_score': normalize_iv_to_score(iv_pctl),
        'skew_shift': skew_shift,
        'skew_score': normalize_skew_to_score(skew_shift),
        'avg_iv': metrics.avg_iv,
        'stock_price': metrics.stock_price,
        'straddle_price': metrics.atm_straddle_price,
        'expiry_used': metrics.expiry_used,
        'days_to_expiry': metrics.days_to_expiry
    }


def get_options_summary_for_ai(ticker: str, earnings_date: Optional[date] = None) -> str:
    """
    Get a formatted summary of options inputs for AI context.

    Args:
        ticker: Stock symbol
        earnings_date: Optional earnings date

    Returns:
        Formatted string for AI consumption
    """
    inputs = calculate_all_options_inputs(ticker, earnings_date)

    lines = [
        f"\n{'='*50}",
        f"OPTIONS-BASED EXPECTATIONS: {ticker}",
        f"{'='*50}",
    ]

    if inputs['implied_move_pct'] is not None:
        lines.extend([
            f"Stock Price: ${inputs['stock_price']:.2f}",
            f"ATM Straddle: ${inputs['straddle_price']:.2f}",
            f"Expiry Used: {inputs['expiry_used']} ({inputs['days_to_expiry']} days)",
            f"",
            f"📊 IMPLIED MOVE:",
            f"   Current: {inputs['implied_move_pct']*100:.1f}%",
            f"   Percentile: {inputs['implied_move_pctl']:.0f}th" if inputs['implied_move_pctl'] else "   Percentile: N/A (insufficient history)",
            f"   IES Score: {inputs['implied_move_score']:.0f}/100",
            f"",
            f"📈 IMPLIED VOLATILITY:",
            f"   Current IV: {inputs['avg_iv']*100:.0f}%",
            f"   Percentile: {inputs['iv_pctl']:.0f}th" if inputs['iv_pctl'] else "   Percentile: N/A (insufficient history)",
            f"   IES Score: {inputs['iv_score']:.0f}/100",
            f"",
            f"⚖️ SKEW:",
            f"   Shift: {inputs['skew_shift']:+.3f}" if inputs['skew_shift'] else "   Shift: N/A",
            f"   IES Score: {inputs['skew_score']:.0f}/100",
        ])
    else:
        lines.append("❌ Options data unavailable for this ticker")

    return "\n".join(lines)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("Phase 5 Options Inputs Test")
    print("=" * 60)

    test_tickers = ["AAPL", "TSLA", "NVDA"]

    for ticker in test_tickers:
        print(f"\n--- {ticker} ---")
        inputs = calculate_all_options_inputs(ticker)

        print(f"Implied Move: {inputs['implied_move_pct']*100:.1f}%" if inputs['implied_move_pct'] else "Implied Move: N/A")
        print(f"IM Percentile: {inputs['implied_move_pctl']:.0f}th" if inputs['implied_move_pctl'] else "IM Percentile: N/A")
        print(f"IV Percentile: {inputs['iv_pctl']:.0f}th" if inputs['iv_pctl'] else "IV Percentile: N/A")
        print(f"Skew Shift: {inputs['skew_shift']:.3f}" if inputs['skew_shift'] else "Skew Shift: N/A")
        print(f"Expiry: {inputs['expiry_used']}")