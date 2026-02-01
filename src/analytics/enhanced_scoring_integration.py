"""
Enhanced Scoring Integration for Signals Tab

This module provides easy integration of enhanced scoring into the signals table.
It fetches additional data (price history, volume, fundamentals) and applies
enhanced scoring adjustments.

    from src.analytics.enhanced_scoring_integration import (
        apply_enhanced_scores_to_dataframe,
        get_enhanced_total_score,
    )

    # After loading df from database:
    df = apply_enhanced_scores_to_dataframe(df)

Author: Alpha Research Platform
Version: 2024-12-23
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Import enhanced scoring module
try:
    from src.analytics.enhanced_scoring import (
        get_enhanced_scores,
        get_enhanced_score_adjustment,
        calculate_enhanced_total_score,
        score_pe_relative,
        score_price_target,
        score_macd,
        score_volume_profile,
        score_insider_trading,
        score_analyst_revisions,
        score_earnings_surprise,
        SECTOR_PE_MEDIANS,
    )
    ENHANCED_SCORING_AVAILABLE = True
except ImportError:
    ENHANCED_SCORING_AVAILABLE = False
    logger.warning("Enhanced scoring module not available")

# Database
try:
    from src.db.connection import get_engine, get_connection
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

# yfinance for price/volume history
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available for price history")


# =============================================================================
# PRICE/VOLUME HISTORY RETRIEVAL
# =============================================================================

def get_price_volume_history(ticker: str, days: int = 60) -> Tuple[pd.Series, pd.Series]:
    """
    Get price and volume history for a ticker.

    Args:
        ticker: Stock ticker symbol
        days: Number of days of history to retrieve

    Returns:
        Tuple of (price_series, volume_series)
    """
    if not YFINANCE_AVAILABLE:
        return None, None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{days}d")

        if hist.empty:
            return None, None

        return hist['Close'], hist['Volume']

    except Exception as e:
        logger.debug(f"Error getting price history for {ticker}: {e}")
        return None, None


def get_price_volume_history_batch(tickers: List[str], days: int = 60) -> Dict[str, Tuple[pd.Series, pd.Series]]:
    """
    Get price/volume history for multiple tickers efficiently.
    """
    results = {}

    if not YFINANCE_AVAILABLE:
        return results

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Download all at once
            data = yf.download(
                tickers,
                period=f"{days}d",
                group_by='ticker',
                progress=False,
                threads=True
            )

        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    prices = data['Close']
                    volumes = data['Volume']
                else:
                    prices = data[ticker]['Close'] if ticker in data.columns.get_level_values(0) else None
                    volumes = data[ticker]['Volume'] if ticker in data.columns.get_level_values(0) else None

                if prices is not None and not prices.empty:
                    results[ticker] = (prices.dropna(), volumes.dropna() if volumes is not None else None)
            except:
                pass

    except Exception as e:
        logger.debug(f"Error in batch price history: {e}")

    return results


# =============================================================================
# ENHANCED SCORE CALCULATION
# =============================================================================

def get_enhanced_total_score(
    ticker: str,
    base_score: int,
    row_data: Dict[str, Any],
    price_history: pd.Series = None,
    volume_history: pd.Series = None,
) -> Tuple[int, int, Dict[str, Any]]:
    """
    Calculate enhanced total score with all adjustments.

    Args:
        ticker: Stock ticker
        base_score: Original total_score from weighted components
        row_data: Dict with fundamentals (PE, sector, price target, etc.)
        price_history: Price series for MACD (optional, will fetch if None)
        volume_history: Volume series for accumulation/distribution

    Returns:
        Tuple of (enhanced_score, adjustment, details_dict)
    """
    if not ENHANCED_SCORING_AVAILABLE:
        return base_score, 0, {'error': 'Enhanced scoring not available'}

    # Fetch price/volume history if not provided
    if price_history is None:
        price_history, volume_history = get_price_volume_history(ticker, days=60)

    # Prepare row data with consistent keys
    normalized_data = {
        'price': row_data.get('price') or row_data.get('Price') or row_data.get('close'),
        'target_mean': row_data.get('target_mean') or row_data.get('TargetPrice') or row_data.get('target_price'),
        'pe_ratio': row_data.get('pe_ratio') or row_data.get('PE') or row_data.get('pe'),
        'forward_pe': row_data.get('forward_pe') or row_data.get('ForwardPE'),
        'peg_ratio': row_data.get('peg_ratio') or row_data.get('PEG'),
        'sector': row_data.get('sector') or row_data.get('Sector'),
        'buy_count': row_data.get('buy_count') or row_data.get('BuyRatings') or row_data.get('analyst_buy') or 0,
        'total_ratings': row_data.get('total_ratings') or row_data.get('TotalRatings') or row_data.get('analyst_total') or 0,
    }

    try:
        adjustment, details = get_enhanced_score_adjustment(
            ticker=ticker,
            current_price=normalized_data['price'],
            target_price=normalized_data['target_mean'],
            pe_ratio=normalized_data['pe_ratio'],
            forward_pe=normalized_data['forward_pe'],
            peg_ratio=normalized_data['peg_ratio'],
            sector=normalized_data['sector'],
            buy_count=int(normalized_data['buy_count'] or 0),
            total_ratings=int(normalized_data['total_ratings'] or 0),
            price_history=price_history,
            volume_history=volume_history,
            max_adjustment=30,
        )

        enhanced_score = max(0, min(100, base_score + adjustment))

        return enhanced_score, adjustment, details

    except Exception as e:
        logger.warning(f"Error calculating enhanced score for {ticker}: {e}")
        return base_score, 0, {'error': str(e)}


# =============================================================================
# DATAFRAME ENHANCEMENT
# =============================================================================

def apply_enhanced_scores_to_dataframe(
    df: pd.DataFrame,
    score_column: str = 'total_score',
    enhanced_column: str = 'enhanced_score',
    adjustment_column: str = 'score_adjustment',
    max_workers: int = 5,
    fetch_price_history: bool = True,
) -> pd.DataFrame:
    """
    Apply enhanced scoring to a dataframe of signals.

    This adds:
    - enhanced_score: The new total score with all adjustments
    - score_adjustment: The adjustment that was applied (+/-)
    - enhancement_details: JSON string with component breakdown

    Args:
        df: DataFrame with signals data
        score_column: Name of the base score column
        enhanced_column: Name for the new enhanced score column
        adjustment_column: Name for the adjustment column
        max_workers: Number of parallel workers for fetching data
        fetch_price_history: Whether to fetch price/volume history

    Returns:
        DataFrame with enhanced scores added
    """
    if df.empty:
        return df

    if not ENHANCED_SCORING_AVAILABLE:
        logger.warning("Enhanced scoring not available, returning original dataframe")
        df[enhanced_column] = df[score_column]
        df[adjustment_column] = 0
        return df

    # Identify ticker column
    ticker_col = 'ticker' if 'ticker' in df.columns else 'Ticker'

    tickers = df[ticker_col].unique().tolist()

    # Batch fetch price/volume history if requested
    price_volume_data = {}
    if fetch_price_history and YFINANCE_AVAILABLE:
        try:
            logger.info(f"Fetching price history for {len(tickers)} tickers...")
            price_volume_data = get_price_volume_history_batch(tickers, days=60)
            logger.info(f"Got price history for {len(price_volume_data)} tickers")
        except Exception as e:
            logger.warning(f"Error fetching batch price history: {e}")

    # Calculate enhanced scores
    enhanced_scores = []
    adjustments = []
    details_list = []

    for idx, row in df.iterrows():
        ticker = row[ticker_col]
        base_score = row.get(score_column, 50) or 50

        # Get price/volume history for this ticker
        price_hist, vol_hist = price_volume_data.get(ticker, (None, None))

        # Convert row to dict
        row_data = row.to_dict()

        # Calculate enhanced score
        enhanced_score, adjustment, details = get_enhanced_total_score(
            ticker=ticker,
            base_score=int(base_score),
            row_data=row_data,
            price_history=price_hist,
            volume_history=vol_hist,
        )

        enhanced_scores.append(enhanced_score)
        adjustments.append(adjustment)
        details_list.append(details)

    # Add columns to dataframe
    df[enhanced_column] = enhanced_scores
    df[adjustment_column] = adjustments

    # Re-calculate signal type based on enhanced score
    df['enhanced_signal'] = df[enhanced_column].apply(lambda x:
        'STRONG BUY' if x >= 80 else
        'BUY' if x >= 65 else
        'WEAK BUY' if x >= 55 else
        'STRONG SELL' if x <= 20 else
        'SELL' if x <= 35 else
        'WEAK SELL' if x <= 45 else
        'HOLD'
    )

    logger.info(f"Enhanced scoring applied: avg adjustment = {np.mean(adjustments):.1f}")

    return df


# =============================================================================
# SINGLE TICKER ENHANCEMENT (for real-time analysis)
# =============================================================================

# Cache for single ticker enhancements (avoid repeated fetches)
_enhancement_cache = {}
_enhancement_cache_ttl = 300  # 5 minutes

def get_single_ticker_enhancement(ticker: str, row_data: Dict = None, use_cache: bool = True, force_recalc: bool = False) -> Dict[str, Any]:
    """
    Get detailed enhanced scoring breakdown for a single ticker.

    Priority:
    1. Load from database (pre-computed during batch analysis) - FAST
    2. Check in-memory cache
    3. Compute fresh (slow, only if force_recalc=True or no stored data)

    Returns dict with all component scores and reasons.
    """
    import time

    if not ENHANCED_SCORING_AVAILABLE:
        return {'error': 'Enhanced scoring not available'}

    # PRIORITY 1: Load from database (computed during batch analysis)
    if not force_recalc:
        try:
            from src.analytics.enhanced_scores_db import load_enhanced_scores
            db_scores = load_enhanced_scores(ticker, max_age_days=1)
            if db_scores:
                logger.debug(f"{ticker}: Loaded enhanced scores from database")
                return db_scores
        except ImportError:
            pass  # DB module not available
        except Exception as e:
            logger.debug(f"{ticker}: Could not load from DB: {e}")

    # PRIORITY 2: Check in-memory cache
    cache_key = ticker
    if use_cache and not force_recalc and cache_key in _enhancement_cache:
        cached_time, cached_result = _enhancement_cache[cache_key]
        if time.time() - cached_time < _enhancement_cache_ttl:
            return cached_result

    # PRIORITY 3: Compute fresh (slow)
    logger.info(f"{ticker}: Computing enhanced scores fresh (no cached data)")

    # Fetch fresh data if row_data not provided
    if row_data is None:
        row_data = _fetch_ticker_fundamentals(ticker)

    # Get price history (this is the slow part)
    price_hist, vol_hist = get_price_volume_history(ticker, days=60)

    # Get full enhanced scores object
    scores = get_enhanced_scores(
        ticker=ticker,
        current_price=row_data.get('price'),
        target_price=row_data.get('target_mean'),
        pe_ratio=row_data.get('pe_ratio'),
        forward_pe=row_data.get('forward_pe'),
        peg_ratio=row_data.get('peg_ratio'),
        sector=row_data.get('sector'),
        buy_count=int(row_data.get('buy_count') or 0),
        total_ratings=int(row_data.get('total_ratings') or 0),
        price_history=price_hist,
        volume_history=vol_hist,
    )

    result = {
        'ticker': ticker,
        'pe_relative': {'score': scores.pe_relative_score, 'reason': scores.details.get('pe_relative', '')},
        'peg': {'score': scores.peg_score, 'reason': scores.details.get('peg', '')},
        'price_target': {'score': scores.price_target_score, 'reason': scores.details.get('price_target', '')},
        'macd': {'score': scores.macd_score, 'reason': scores.details.get('macd', '')},
        'volume': {'score': scores.volume_score, 'reason': scores.details.get('volume', '')},
        'insider': {'score': scores.insider_score, 'reason': scores.details.get('insider', '')},
        'revisions': {'score': scores.revision_score, 'reason': scores.details.get('revisions', '')},
        'earnings_surprise': {'score': scores.earnings_surprise_score, 'reason': scores.details.get('earnings_surprise', '')},
        'total_adjustment': scores.enhancement_total,
        'from_cache': False,
    }

    # Cache the result
    _enhancement_cache[cache_key] = (time.time(), result)

    # Also save to database for future fast loading
    try:
        from src.analytics.enhanced_scores_db import save_enhanced_scores_from_dict
        save_enhanced_scores_from_dict(ticker, result)
    except:
        pass

    return result


def _fetch_ticker_fundamentals(ticker: str) -> Dict[str, Any]:
    """Fetch fundamental data for a ticker from database."""
    if not DB_AVAILABLE:
        return {}

    try:
        query = """
            SELECT 
                f.sector,
                f.pe_ratio,
                f.forward_pe,
                f.peg_ratio,
                ar.analyst_buy as buy_count,
                ar.analyst_total as total_ratings,
                pt.target_mean,
                p.close as price
            FROM fundamentals f
            LEFT JOIN analyst_ratings ar ON f.ticker = ar.ticker
            LEFT JOIN price_targets pt ON f.ticker = pt.ticker
            LEFT JOIN (
                SELECT ticker, close 
                FROM prices 
                WHERE ticker = %s
                ORDER BY date DESC LIMIT 1
            ) p ON f.ticker = p.ticker
            WHERE f.ticker = %s
        """

        df = pd.read_sql(query, get_engine(), params=(ticker, ticker))

        if df.empty:
            return {}

        return df.iloc[0].to_dict()

    except Exception as e:
        logger.warning(f"Error fetching fundamentals for {ticker}: {e}")
        return {}


# =============================================================================
# DISPLAY HELPERS FOR STREAMLIT
# =============================================================================

def render_enhancement_breakdown(ticker: str, row_data: Dict = None):
    """
    Render enhanced scoring breakdown in Streamlit.
    Call this in the detailed view panel.
    """
    import streamlit as st

    with st.spinner(f"Loading enhanced scores for {ticker}..."):
        enhancement = get_single_ticker_enhancement(ticker, row_data)

    if 'error' in enhancement:
        st.warning(f"Enhanced scoring unavailable: {enhancement['error']}")
        return

    # Show if loaded from cache or computed fresh
    if enhancement.get('from_cache'):
        st.caption("✅ Loaded from database (pre-computed during analysis)")
    else:
        st.caption("⚡ Computed fresh (will be cached for next time)")

    components = [
        ('PE Relative', 'pe_relative', '📉'),
        ('PEG Ratio', 'peg', '📈'),
        ('Price Target', 'price_target', '🎯'),
        ('MACD Momentum', 'macd', '📊'),
        ('Volume Profile', 'volume', '📶'),
        ('Insider Trading', 'insider', '👔'),
        ('Analyst Revisions', 'revisions', '📝'),
        ('Earnings Surprise', 'earnings_surprise', '💰'),
    ]

    cols = st.columns(4)

    for i, (label, key, emoji) in enumerate(components):
        data = enhancement.get(key, {})
        score = data.get('score', 0)
        reason = data.get('reason', 'N/A')

        col = cols[i % 4]

        # Color based on score
        if score > 5:
            color = "#00C853"  # Green
        elif score < -5:
            color = "#FF1744"  # Red
        else:
            color = "#888888"  # Gray

        with col:
            st.markdown(f"""
                <div style="text-align: center; padding: 5px; margin: 2px; border-radius: 5px; background: #1E1E1E;">
                    <span style="font-size: 1.2em;">{emoji}</span><br>
                    <span style="font-size: 0.8em; color: #AAA;">{label}</span><br>
                    <span style="font-size: 1.3em; font-weight: bold; color: {color};">{score:+d}</span>
                </div>
            """, unsafe_allow_html=True)

            if reason and reason != 'N/A':
                st.caption(reason[:50])

    # Total adjustment
    total = enhancement.get('total_adjustment', 0)
    total_color = "#00C853" if total > 0 else "#FF1744" if total < 0 else "#888888"

    st.markdown(f"""
        <div style="text-align: center; padding: 10px; margin-top: 10px; border-radius: 5px; background: #2E2E2E;">
            <span style="font-size: 1em; color: #AAA;">Total Enhancement Adjustment:</span>
            <span style="font-size: 1.5em; font-weight: bold; color: {total_color};"> {total:+d}</span>
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SQL QUERY ENHANCEMENT
# =============================================================================

def get_enhanced_signals_query():
    """
    Returns the SQL query for loading signals with additional fields
    needed for enhanced scoring.

    This adds: forward_pe, peg_ratio to the standard query.
    """
    return """
        WITH latest_prices AS (
            SELECT DISTINCT ON (ticker) ticker, close as price
            FROM prices
            ORDER BY ticker, date DESC
        )
        SELECT 
            s.ticker,
            s.date as score_date,
            f.sector,
            CASE 
                WHEN s.total_score >= 80 THEN 'STRONG BUY'
                WHEN s.total_score >= 65 THEN 'BUY'
                WHEN s.total_score >= 55 THEN 'WEAK BUY'
                WHEN s.total_score <= 20 THEN 'STRONG SELL'
                WHEN s.total_score <= 35 THEN 'SELL'
                WHEN s.total_score <= 45 THEN 'WEAK SELL'
                ELSE 'HOLD'
            END as signal_type,
            s.total_score,
            s.sentiment_score,
            s.options_flow_score,
            s.short_squeeze_score,
            s.fundamental_score,
            s.growth_score,
            s.dividend_score,
            s.technical_score,
            s.gap_score,
            CASE 
                WHEN s.gap_score >= 70 THEN 'BEARISH'
                WHEN s.gap_score <= 30 THEN 'BULLISH'
                ELSE 'None'
            END as gap_type,
            s.likelihood_score,
            s.article_count,
            lp.price,
            pt.target_mean,
            -- Additional fields for enhanced scoring
            f.pe_ratio,
            f.forward_pe,
            f.peg_ratio,
            f.roe,
            f.dividend_yield,
            ar.analyst_buy as buy_count,
            ar.analyst_total as total_ratings,
            ar.analyst_positivity,
            -- Earnings
            COALESCE(
                (SELECT earnings_date FROM earnings_calendar 
                 WHERE ticker = s.ticker AND earnings_date >= CURRENT_DATE 
                 ORDER BY earnings_date ASC LIMIT 1),
                (SELECT earnings_date FROM earnings_calendar 
                 WHERE ticker = s.ticker AND earnings_date >= CURRENT_DATE - INTERVAL '14 days' AND earnings_date < CURRENT_DATE
                 ORDER BY earnings_date DESC LIMIT 1)
            ) as earnings_date,
            f.ex_dividend_date,
            CASE WHEN lp.price > 0 AND pt.target_mean > 0 
                 THEN ROUND(((pt.target_mean - lp.price) / lp.price * 100)::numeric, 2)
                 ELSE NULL END as target_upside_pct
        FROM screener_scores s
        LEFT JOIN fundamentals f ON s.ticker = f.ticker
        LEFT JOIN analyst_ratings ar ON s.ticker = ar.ticker
        LEFT JOIN price_targets pt ON s.ticker = pt.ticker
        LEFT JOIN latest_prices lp ON s.ticker = lp.ticker
        WHERE s.date = (
            SELECT MAX(date) FROM screener_scores ss WHERE ss.ticker = s.ticker
        )
        ORDER BY s.total_score DESC NULLS LAST
    """


# =============================================================================
# UTILITY: Update Total Score in Batch Analysis
# =============================================================================

def compute_enhanced_total_score(
    ticker: str,
    sentiment: float,
    fundamental: float,
    technical: float,
    options: float,
    squeeze: float,
    row_data: Dict[str, Any],
    base_weights: Dict[str, float] = None,
) -> Tuple[int, int, Dict]:
    """
    Compute enhanced total score for use in batch analysis.

    This is the function to use when creating new screener_scores rows.

    Args:
        ticker: Stock ticker
        sentiment: Sentiment score (0-100)
        fundamental: Fundamental score (0-100)
        technical: Technical score (0-100)
        options: Options flow score (0-100)
        squeeze: Short squeeze score (0-100)
        row_data: Dict with PE, sector, price target, etc.
        base_weights: Optional custom weights (default: sent=25%, fund=25%, tech=25%, opts=15%, sqz=10%)

    Returns:
        Tuple of (enhanced_total_score, adjustment, details)
    """
    # Default weights
    if base_weights is None:
        base_weights = {
            'sentiment': 0.25,
            'fundamental': 0.25,
            'technical': 0.25,
            'options': 0.15,
            'squeeze': 0.10,
        }

    # Calculate base score
    base_score = int(
        (sentiment or 50) * base_weights['sentiment'] +
        (fundamental or 50) * base_weights['fundamental'] +
        (technical or 50) * base_weights['technical'] +
        (options or 50) * base_weights['options'] +
        (squeeze or 50) * base_weights['squeeze']
    )

    # Apply enhanced scoring
    if ENHANCED_SCORING_AVAILABLE:
        enhanced_score, adjustment, details = get_enhanced_total_score(
            ticker=ticker,
            base_score=base_score,
            row_data=row_data,
        )
        return enhanced_score, adjustment, details
    else:
        return base_score, 0, {}


# Test function
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = "AAPL"

    print(f"\n{'='*60}")
    print(f"Enhanced Scoring Integration Test: {ticker}")
    print(f"{'='*60}")

    # Test single ticker enhancement
    result = get_single_ticker_enhancement(ticker)

    print("\nComponent Scores:")
    for key, value in result.items():
        if isinstance(value, dict):
            print(f"  {key}: {value.get('score', 0):+3d}  {value.get('reason', '')}")
        elif key != 'ticker':
            print(f"  {key}: {value}")

    print(f"\n{'='*60}\n")