"""
Analyst Expectations IES Inputs Calculator (Redesigned)

Computes FORWARD-LOOKING analyst-derived inputs for the Implied Expectations Score (IES):
- revision_score: Overall analyst sentiment (consensus, positivity, target upside)
- estimate_momentum_score: Direction of EPS estimate revisions (last 30/60/90 days)

These measure EXPECTATIONS before earnings, not backward-looking news.

Key Insight:
- Rising estimates = Higher bar to beat = Lower IES (harder to surprise)
- Falling estimates = Lower bar = Higher IES (easier to surprise)

Data Sources:
1. analyst_ratings table - buy/hold/sell counts, positivity %
2. price_targets table - target price upside
3. yfinance - EPS estimate trends, revision data

Author: Alpha Research Platform
Phase: 6 of Earnings Intelligence System (Redesigned)
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
# CONFIGURATION
# ============================================================================

# Database availability flag
try:
    from src.db.connection import get_connection, get_engine
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.info("Database not available - using yfinance fallback for analyst data")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AnalystMetrics:
    """Analyst ratings and target price metrics."""
    ticker: str

    # Ratings breakdown
    buy_count: int
    hold_count: int
    sell_count: int
    total_analysts: int

    # Derived metrics
    buy_pct: float          # % of analysts with BUY
    positivity_pct: float   # (buy - sell) / total normalized
    consensus: str          # BUY, HOLD, SELL, STRONG_BUY, STRONG_SELL

    # Price targets
    target_mean: Optional[float]
    target_high: Optional[float]
    target_low: Optional[float]
    current_price: Optional[float]
    target_upside_pct: Optional[float]

    # Computed score
    revision_score: float   # 0-100 for IES

    timestamp: datetime


@dataclass
class EstimateMetrics:
    """EPS estimate revision metrics - FORWARD LOOKING."""
    ticker: str

    # Current estimates
    current_eps_estimate: Optional[float]
    next_quarter_estimate: Optional[float]
    current_year_estimate: Optional[float]
    next_year_estimate: Optional[float]

    # Estimate changes (positive = raised, negative = lowered)
    eps_revision_7d: Optional[float]    # % change in last 7 days
    eps_revision_30d: Optional[float]   # % change in last 30 days
    eps_revision_90d: Optional[float]   # % change in last 90 days

    # Revision counts
    num_revisions_up: int
    num_revisions_down: int

    # Surprise history (how often company beats)
    beat_rate: Optional[float]          # % of quarters beating estimates
    avg_surprise_pct: Optional[float]   # Average surprise magnitude

    # Computed score
    estimate_momentum_score: float      # 0-100 for IES

    timestamp: datetime


# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

_analyst_cache: Dict[str, Tuple[AnalystMetrics, datetime]] = {}
_estimate_cache: Dict[str, Tuple[EstimateMetrics, datetime]] = {}
_CACHE_EXPIRY_HOURS = 6  # Refresh every 6 hours


def clear_analyst_cache():
    """Clear both analyst and estimate caches."""
    global _analyst_cache, _estimate_cache
    _analyst_cache.clear()
    _estimate_cache.clear()
    logger.debug("Analyst caches cleared")


# ============================================================================
# ANALYST DATA FUNCTIONS
# ============================================================================

def get_analyst_data_from_db(ticker: str) -> Optional[Dict]:
    """
    Get analyst ratings and price targets from database.

    Args:
        ticker: Stock symbol

    Returns:
        Dict with analyst data or None if unavailable
    """
    if not DB_AVAILABLE:
        return None

    try:
        # Get analyst ratings
        ratings_query = """
            SELECT consensus_rating, analyst_buy, analyst_hold, analyst_sell, 
                   analyst_total, analyst_positivity
            FROM analyst_ratings
            WHERE ticker = %s
            ORDER BY date DESC
            LIMIT 1
        """

        # Get price targets
        targets_query = """
            SELECT current_price, target_high, target_low, target_mean, 
                   target_upside_pct, analyst_count
            FROM price_targets
            WHERE ticker = %s
            ORDER BY date DESC
            LIMIT 1
        """

        with get_connection() as conn:
            with conn.cursor() as cur:
                # Ratings
                cur.execute(ratings_query, (ticker,))
                ratings_row = cur.fetchone()

                # Targets
                cur.execute(targets_query, (ticker,))
                targets_row = cur.fetchone()

        if ratings_row:
            return {
                'consensus': ratings_row[0],
                'buy_count': ratings_row[1] or 0,
                'hold_count': ratings_row[2] or 0,
                'sell_count': ratings_row[3] or 0,
                'total_analysts': ratings_row[4] or 0,
                'positivity_pct': float(ratings_row[5] or 50),
                'target_mean': float(targets_row[3]) if targets_row and targets_row[3] else None,
                'target_high': float(targets_row[1]) if targets_row and targets_row[1] else None,
                'target_low': float(targets_row[2]) if targets_row and targets_row[2] else None,
                'current_price': float(targets_row[0]) if targets_row and targets_row[0] else None,
                'target_upside_pct': float(targets_row[4]) if targets_row and targets_row[4] else None,
            }

    except Exception as e:
        logger.error(f"Error getting analyst data from DB for {ticker}: {e}")

    return None


def get_analyst_data_from_yfinance(ticker: str) -> Optional[Dict]:
    """
    Get analyst data from yfinance as fallback.

    Args:
        ticker: Stock symbol

    Returns:
        Dict with analyst data or None if unavailable
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get current price
        current_price = (
            info.get('currentPrice') or
            info.get('regularMarketPrice') or
            info.get('previousClose', 0)
        )

        # Get recommendations summary
        rec = info.get('recommendationKey', 'hold')

        # Map recommendation to consensus
        consensus_map = {
            'strong_buy': 'STRONG_BUY',
            'buy': 'BUY',
            'hold': 'HOLD',
            'sell': 'SELL',
            'strong_sell': 'STRONG_SELL',
        }
        consensus = consensus_map.get(rec, 'HOLD')

        # Get analyst counts from recommendations
        recommendations = stock.recommendations
        if recommendations is not None and not recommendations.empty:
            # Get most recent recommendation counts
            recent = recommendations.tail(1).iloc[0] if len(recommendations) > 0 else None
            if recent is not None:
                # yfinance columns vary, try different names
                buy_count = int(recent.get('strongBuy', 0) or 0) + int(recent.get('buy', 0) or 0)
                hold_count = int(recent.get('hold', 0) or 0)
                sell_count = int(recent.get('sell', 0) or 0) + int(recent.get('strongSell', 0) or 0)
                total = buy_count + hold_count + sell_count
            else:
                buy_count = hold_count = sell_count = total = 0
        else:
            # Estimate from numberOfAnalystOpinions
            total = info.get('numberOfAnalystOpinions', 0)
            # Use recommendation to estimate distribution
            if consensus in ['STRONG_BUY', 'BUY']:
                buy_count = int(total * 0.7)
                hold_count = int(total * 0.2)
                sell_count = total - buy_count - hold_count
            elif consensus == 'HOLD':
                buy_count = int(total * 0.3)
                hold_count = int(total * 0.5)
                sell_count = total - buy_count - hold_count
            else:
                buy_count = int(total * 0.1)
                hold_count = int(total * 0.3)
                sell_count = total - buy_count - hold_count

        # Get price targets
        target_mean = info.get('targetMeanPrice')
        target_high = info.get('targetHighPrice')
        target_low = info.get('targetLowPrice')

        # Calculate upside
        target_upside_pct = None
        if target_mean and current_price and current_price > 0:
            target_upside_pct = ((target_mean - current_price) / current_price) * 100

        # Calculate positivity
        positivity_pct = 50.0
        if total > 0:
            positivity_pct = ((buy_count - sell_count) / total + 1) * 50  # Normalize to 0-100

        return {
            'consensus': consensus,
            'buy_count': buy_count,
            'hold_count': hold_count,
            'sell_count': sell_count,
            'total_analysts': total,
            'positivity_pct': positivity_pct,
            'target_mean': float(target_mean) if target_mean else None,
            'target_high': float(target_high) if target_high else None,
            'target_low': float(target_low) if target_low else None,
            'current_price': float(current_price) if current_price else None,
            'target_upside_pct': float(target_upside_pct) if target_upside_pct else None,
        }

    except Exception as e:
        logger.error(f"Error getting analyst data from yfinance for {ticker}: {e}")

    return None


def get_analyst_metrics(ticker: str) -> Optional[AnalystMetrics]:
    """
    Get comprehensive analyst metrics for a ticker.

    Tries database first, falls back to yfinance.

    Args:
        ticker: Stock symbol

    Returns:
        AnalystMetrics dataclass or None if unavailable
    """
    global _analyst_cache

    # Check cache
    if ticker in _analyst_cache:
        cached_data, cached_time = _analyst_cache[ticker]
        if datetime.now() - cached_time < timedelta(hours=_CACHE_EXPIRY_HOURS):
            return cached_data

    # Try database first
    data = get_analyst_data_from_db(ticker)

    # Fallback to yfinance
    if data is None:
        data = get_analyst_data_from_yfinance(ticker)

    if data is None:
        return None

    # Calculate revision score
    revision_score = calculate_revision_score_from_data(data)

    # Calculate buy percentage - ensure float conversion
    total = int(data['total_analysts'] or 1)
    buy_pct = (float(data['buy_count']) / total) * 100 if total > 0 else 50.0

    metrics = AnalystMetrics(
        ticker=ticker,
        buy_count=int(data['buy_count'] or 0),
        hold_count=int(data['hold_count'] or 0),
        sell_count=int(data['sell_count'] or 0),
        total_analysts=total,
        buy_pct=buy_pct,
        positivity_pct=float(data['positivity_pct'] or 50),
        consensus=data['consensus'],
        target_mean=data['target_mean'],
        target_high=data['target_high'],
        target_low=data['target_low'],
        current_price=data['current_price'],
        target_upside_pct=data['target_upside_pct'],
        revision_score=revision_score,
        timestamp=datetime.now()
    )

    # Cache result
    _analyst_cache[ticker] = (metrics, datetime.now())

    return metrics


def calculate_revision_score_from_data(data: Dict) -> float:
    """
    Calculate revision score (0-100) from analyst data.

    Components:
    - 40% from consensus (STRONG_BUY=100, BUY=80, HOLD=50, SELL=20, STRONG_SELL=0)
    - 30% from positivity percentage
    - 30% from target upside (clamped to -30% to +50%)

    Higher score = more bullish analyst sentiment = higher expectations

    Args:
        data: Dict with analyst metrics

    Returns:
        Score 0-100
    """
    # Consensus component (40%)
    consensus_scores = {
        'STRONG_BUY': 100,
        'BUY': 80,
        'HOLD': 50,
        'SELL': 20,
        'STRONG_SELL': 0,
    }
    consensus_score = consensus_scores.get(data.get('consensus', 'HOLD'), 50)

    # Positivity component (30%) - ensure float conversion
    positivity_score = float(data.get('positivity_pct', 50) or 50)
    positivity_score = max(0, min(100, positivity_score))

    # Target upside component (30%) - ensure float conversion
    target_upside = float(data.get('target_upside_pct', 0) or 0)
    # Map -30% to +50% range to 0-100
    # -30% -> 0, 0% -> 37.5, +50% -> 100
    upside_normalized = (target_upside + 30) / 80 * 100
    upside_score = max(0, min(100, upside_normalized))

    # Weighted average
    score = (
        consensus_score * 0.40 +
        positivity_score * 0.30 +
        upside_score * 0.30
    )

    return max(0, min(100, score))


# ============================================================================
# ESTIMATE MOMENTUM FUNCTIONS (FORWARD-LOOKING)
# ============================================================================

def get_estimate_momentum_from_db(ticker: str, days_back: int = 90) -> Optional[Dict]:
    """
    Calculate estimate momentum from YOUR historical analyst_ratings data.

    Compares current analyst ratings to historical data to detect trends:
    - Are buy counts increasing? (bullish momentum)
    - Are sell counts increasing? (bearish momentum)
    - Is target upside changing?

    Args:
        ticker: Stock symbol
        days_back: Days to look back for comparison

    Returns:
        Dict with momentum metrics or None if insufficient data
    """
    if not DB_AVAILABLE:
        return None

    try:
        engine = get_engine()

        # Get historical analyst ratings
        query = f"""
            SELECT date, analyst_buy, analyst_hold, analyst_sell, analyst_total,
                   analyst_positivity, consensus_rating
            FROM analyst_ratings
            WHERE ticker = %(ticker)s
              AND date >= CURRENT_DATE - INTERVAL '{days_back} days'
            ORDER BY date DESC
        """

        df = pd.read_sql(query, engine, params={'ticker': ticker})

        if df.empty or len(df) < 2:
            logger.debug(f"{ticker}: Insufficient analyst history for momentum calculation")
            return None

        # Get current (most recent) and historical data points
        current = df.iloc[0]

        # Compare to 30, 60, 90 days ago
        results = {
            'current_buy': int(current['analyst_buy'] or 0),
            'current_sell': int(current['analyst_sell'] or 0),
            'current_total': int(current['analyst_total'] or 0),
            'current_positivity': float(current['analyst_positivity'] or 50),
        }

        # Find data points at different intervals
        for days, label in [(7, '7d'), (30, '30d'), (60, '60d'), (90, '90d')]:
            # Find closest data point to target date
            target_date = pd.Timestamp.now() - pd.Timedelta(days=days)
            historical = df[pd.to_datetime(df['date']) <= target_date]

            if not historical.empty:
                hist_row = historical.iloc[0]

                # Calculate changes
                buy_change = int(current['analyst_buy'] or 0) - int(hist_row['analyst_buy'] or 0)
                sell_change = int(current['analyst_sell'] or 0) - int(hist_row['analyst_sell'] or 0)
                positivity_change = float(current['analyst_positivity'] or 50) - float(hist_row['analyst_positivity'] or 50)

                results[f'buy_change_{label}'] = buy_change
                results[f'sell_change_{label}'] = sell_change
                results[f'positivity_change_{label}'] = positivity_change
                results[f'net_change_{label}'] = buy_change - sell_change  # Net upgrade/downgrade

        # Get price target changes
        target_query = f"""
            SELECT date, target_mean, target_upside_pct
            FROM price_targets
            WHERE ticker = %(ticker)s
              AND date >= CURRENT_DATE - INTERVAL '{days_back} days'
            ORDER BY date DESC
        """

        target_df = pd.read_sql(target_query, engine, params={'ticker': ticker})

        if not target_df.empty and len(target_df) >= 2:
            current_target = target_df.iloc[0]
            results['current_target_upside'] = float(current_target['target_upside_pct'] or 0)

            for days, label in [(30, '30d'), (90, '90d')]:
                target_date = pd.Timestamp.now() - pd.Timedelta(days=days)
                historical = target_df[pd.to_datetime(target_df['date']) <= target_date]

                if not historical.empty:
                    hist_target = historical.iloc[0]
                    upside_change = float(current_target['target_upside_pct'] or 0) - float(hist_target['target_upside_pct'] or 0)
                    results[f'target_upside_change_{label}'] = upside_change

        # Calculate revision counts (net upgrades vs downgrades)
        # Positive net_change = more upgrades = analysts getting more bullish
        net_30d = results.get('net_change_30d', 0)
        net_90d = results.get('net_change_90d', 0)

        # Count as "revisions"
        if net_30d > 0:
            results['num_revisions_up'] = abs(net_30d)
            results['num_revisions_down'] = 0
        elif net_30d < 0:
            results['num_revisions_up'] = 0
            results['num_revisions_down'] = abs(net_30d)
        else:
            results['num_revisions_up'] = 0
            results['num_revisions_down'] = 0

        results['data_source'] = 'database'

        logger.info(f"{ticker}: Got momentum from DB - Net 30d: {net_30d}, Net 90d: {net_90d}")
        return results

    except Exception as e:
        logger.warning(f"Error getting estimate momentum from DB for {ticker}: {e}")
        return None


def get_estimate_data_from_yfinance(ticker: str) -> Optional[Dict]:
    """
    Get EPS estimate and revision data from yfinance.

    This captures FORWARD-LOOKING expectations:
    - Current EPS estimates
    - Estimate trends (are analysts raising or lowering?)
    - Historical beat rate

    Args:
        ticker: Stock symbol

    Returns:
        Dict with estimate data or None if unavailable
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Current estimates
        current_eps = info.get('trailingEps')
        forward_eps = info.get('forwardEps')

        # Get earnings estimates if available
        earnings_estimate = None
        try:
            # yfinance has earnings_dates with estimates
            earnings = stock.earnings_dates
            if earnings is not None and not earnings.empty:
                # Get future earnings (estimates)
                now = datetime.now()
                if hasattr(earnings.index, 'tz') and earnings.index.tz is not None:
                    import pytz
                    now = now.replace(tzinfo=pytz.UTC)

                future_earnings = earnings[earnings.index > now]
                if not future_earnings.empty:
                    next_earning = future_earnings.iloc[0]
                    earnings_estimate = next_earning.get('EPS Estimate')
        except Exception as e:
            logger.debug(f"{ticker}: Could not get earnings estimates: {e}")

        # Get analyst trends (estimate revisions)
        eps_trend = None
        try:
            # yfinance provides earnings_trend
            trends = stock.earnings_trend
            if trends is not None and not trends.empty:
                eps_trend = trends.to_dict()
        except:
            pass

        # Get earnings history for beat rate calculation
        earnings_history = None
        beat_count = 0
        total_quarters = 0
        avg_surprise = 0.0

        try:
            hist = stock.earnings_history
            if hist is not None and not hist.empty:
                earnings_history = hist

                # Calculate beat rate
                for _, row in hist.iterrows():
                    actual = row.get('epsActual')
                    estimate = row.get('epsEstimate')

                    if actual is not None and estimate is not None:
                        total_quarters += 1
                        if actual > estimate:
                            beat_count += 1

                        # Calculate surprise %
                        if estimate != 0:
                            surprise = ((actual - estimate) / abs(estimate)) * 100
                            avg_surprise += surprise

                if total_quarters > 0:
                    avg_surprise = avg_surprise / total_quarters
        except Exception as e:
            logger.debug(f"{ticker}: Could not get earnings history: {e}")

        # Calculate beat rate
        beat_rate = None
        if total_quarters >= 4:
            beat_rate = (beat_count / total_quarters) * 100

        # Get estimate revisions from info
        # yfinance provides some revision data in info
        eps_revision_7d = None
        eps_revision_30d = None
        eps_revision_90d = None

        # Try to extract from earnings_trend if available
        try:
            if eps_trend:
                # Look for growth estimates which indicate trend direction
                growth = eps_trend.get('growth')
                if growth:
                    # This is a rough proxy - actual revision data would be better
                    pass
        except:
            pass

        # Revision counts (from recommendations changes)
        num_up = 0
        num_down = 0

        try:
            upgrades = stock.upgrades_downgrades
            if upgrades is not None and not upgrades.empty:
                # Look at recent 90 days
                cutoff = datetime.now() - timedelta(days=90)

                # Handle timezone
                if hasattr(upgrades.index, 'tz') and upgrades.index.tz is not None:
                    import pytz
                    cutoff = cutoff.replace(tzinfo=pytz.UTC)

                recent = upgrades[upgrades.index > cutoff]

                for _, row in recent.iterrows():
                    action = str(row.get('Action', '')).lower()
                    if 'up' in action or 'raise' in action or 'init' in action and 'buy' in str(row.get('ToGrade', '')).lower():
                        num_up += 1
                    elif 'down' in action or 'lower' in action:
                        num_down += 1
        except Exception as e:
            logger.debug(f"{ticker}: Could not get upgrades/downgrades: {e}")

        return {
            'current_eps_estimate': float(forward_eps) if forward_eps else None,
            'trailing_eps': float(current_eps) if current_eps else None,
            'next_quarter_estimate': float(earnings_estimate) if earnings_estimate else None,
            'eps_revision_7d': eps_revision_7d,
            'eps_revision_30d': eps_revision_30d,
            'eps_revision_90d': eps_revision_90d,
            'num_revisions_up': num_up,
            'num_revisions_down': num_down,
            'beat_rate': beat_rate,
            'avg_surprise_pct': avg_surprise if total_quarters > 0 else None,
            'total_quarters_analyzed': total_quarters,
        }

    except Exception as e:
        logger.error(f"Error getting estimate data from yfinance for {ticker}: {e}")

    return None


def get_estimate_metrics(ticker: str) -> Optional[EstimateMetrics]:
    """
    Get comprehensive EPS estimate metrics for a ticker.

    This is the FORWARD-LOOKING component of Phase 6:
    - Are analysts raising or lowering estimates?
    - How often does this company beat?
    - What's the estimate momentum?

    Data Priority:
    1. YOUR database (analyst_ratings history) - for momentum
    2. yfinance - for beat rate, EPS estimates

    Args:
        ticker: Stock symbol

    Returns:
        EstimateMetrics dataclass or None if unavailable
    """
    global _estimate_cache

    # Check cache
    if ticker in _estimate_cache:
        cached_data, cached_time = _estimate_cache[ticker]
        if datetime.now() - cached_time < timedelta(hours=_CACHE_EXPIRY_HOURS):
            return cached_data

    # Try database first for momentum data (YOUR historical data)
    db_data = get_estimate_momentum_from_db(ticker)

    # Get yfinance data for EPS estimates and beat rate
    yf_data = get_estimate_data_from_yfinance(ticker)

    if db_data is None and yf_data is None:
        return None

    # Merge data sources - prefer DB for momentum, yfinance for estimates
    data = {}

    if yf_data:
        data.update(yf_data)

    if db_data:
        # Override revision counts with DB data (more reliable)
        data['num_revisions_up'] = db_data.get('num_revisions_up', data.get('num_revisions_up', 0))
        data['num_revisions_down'] = db_data.get('num_revisions_down', data.get('num_revisions_down', 0))
        data['data_source'] = 'database'

        # Add detailed momentum data
        data['positivity_change_30d'] = db_data.get('positivity_change_30d')
        data['positivity_change_90d'] = db_data.get('positivity_change_90d')
        data['target_upside_change_30d'] = db_data.get('target_upside_change_30d')
        data['target_upside_change_90d'] = db_data.get('target_upside_change_90d')
        data['net_change_30d'] = db_data.get('net_change_30d', 0)
        data['net_change_90d'] = db_data.get('net_change_90d', 0)

    # Calculate estimate momentum score
    momentum_score = calculate_estimate_momentum_score(data)

    metrics = EstimateMetrics(
        ticker=ticker,
        current_eps_estimate=data.get('current_eps_estimate'),
        next_quarter_estimate=data.get('next_quarter_estimate'),
        current_year_estimate=data.get('current_eps_estimate'),  # Approximate
        next_year_estimate=None,
        eps_revision_7d=data.get('eps_revision_7d'),
        eps_revision_30d=data.get('eps_revision_30d'),
        eps_revision_90d=data.get('eps_revision_90d'),
        num_revisions_up=data.get('num_revisions_up', 0),
        num_revisions_down=data.get('num_revisions_down', 0),
        beat_rate=data.get('beat_rate'),
        avg_surprise_pct=data.get('avg_surprise_pct'),
        estimate_momentum_score=momentum_score,
        timestamp=datetime.now()
    )

    # Cache result
    _estimate_cache[ticker] = (metrics, datetime.now())

    return metrics


def calculate_estimate_momentum_score(data: Dict) -> float:
    """
    Calculate estimate momentum score (0-100) for IES.

    KEY INSIGHT FOR IES:
    - Rising estimates (analysts raising targets) = HIGHER bar to beat = LOWER IES
    - Falling estimates = LOWER bar = HIGHER IES (easier to surprise positively)

    This is INVERTED from typical bullish/bearish thinking!

    Components:
    - 40% from revision direction (up vs down count OR net change)
    - 30% from beat rate (companies that beat often have higher bars)
    - 30% from positivity/target changes (if available from DB)

    Args:
        data: Dict with estimate metrics

    Returns:
        Score 0-100 (higher = easier to beat expectations)
    """
    # Revision direction component (40%)
    # More upgrades = higher expectations = LOWER score (harder to beat)

    # Check if we have detailed DB data
    net_change_30d = data.get('net_change_30d')
    net_change_90d = data.get('net_change_90d')

    if net_change_30d is not None or net_change_90d is not None:
        # Use database-derived momentum (more reliable)
        # Weight recent changes more heavily
        net_30 = net_change_30d or 0
        net_90 = net_change_90d or 0

        # Weighted net change (recent matters more)
        weighted_net = net_30 * 0.7 + net_90 * 0.3

        # Map to score: +5 net upgrades -> 0 (very high bar)
        #               0 net -> 50 (neutral)
        #               -5 net downgrades -> 100 (low bar)
        revision_score = 50 - (weighted_net * 10)
        revision_score = max(0, min(100, revision_score))
    else:
        # Fallback to simple up/down counts
        num_up = data.get('num_revisions_up', 0)
        num_down = data.get('num_revisions_down', 0)
        total_revisions = num_up + num_down

        if total_revisions > 0:
            net_ratio = (num_up - num_down) / total_revisions
            revision_score = (1 - net_ratio) * 50  # 0 to 100
        else:
            revision_score = 50  # Neutral

    # Beat rate component (30%)
    beat_rate = data.get('beat_rate')

    if beat_rate is not None:
        # High beat rate = analysts may have already raised estimates = harder to beat again
        # 100% beat rate -> harder -> lower score
        # Use moderate inversion: 75% beat rate is neutral
        beat_score = max(0, min(100, 100 - (beat_rate - 50)))
    else:
        beat_score = 50  # Neutral

    # Third component (30%): Use positivity change OR surprise magnitude
    positivity_change = data.get('positivity_change_30d')
    target_change = data.get('target_upside_change_30d')
    avg_surprise = data.get('avg_surprise_pct')

    if positivity_change is not None:
        # Rising positivity = analysts getting more bullish = harder to beat
        # +10% positivity change -> lower score
        # -10% positivity change -> higher score
        third_score = 50 - (positivity_change * 2.5)
        third_score = max(0, min(100, third_score))
    elif target_change is not None:
        # Rising targets = higher expectations
        # +20% target change -> lower score
        third_score = 50 - (target_change * 2)
        third_score = max(0, min(100, third_score))
    elif avg_surprise is not None:
        # Large positive surprises historically = bar raised
        third_score = max(0, min(100, 50 - avg_surprise * 2))
    else:
        third_score = 50  # Neutral

    # Weighted average
    score = (
        revision_score * 0.40 +
        beat_score * 0.30 +
        third_score * 0.30
    )

    return max(0, min(100, score))


# ============================================================================
# SCORE NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_revision_to_score(revision_score: Optional[float], default: float = 50.0) -> float:
    """
    Normalize revision score for IES (already 0-100).

    Args:
        revision_score: Raw revision score
        default: Default if None

    Returns:
        Score 0-100
    """
    if revision_score is None:
        return default
    return max(0, min(100, revision_score))


def normalize_momentum_to_score(momentum_score: Optional[float], default: float = 50.0) -> float:
    """
    Normalize estimate momentum score for IES (already 0-100).

    Args:
        momentum_score: Raw momentum score
        default: Default if None

    Returns:
        Score 0-100
    """
    if momentum_score is None:
        return default
    return max(0, min(100, momentum_score))


# ============================================================================
# IES INPUT FUNCTIONS
# ============================================================================

def calculate_revision_score(ticker: str) -> Optional[float]:
    """
    Calculate analyst revision score (0-100) for IES.

    Higher score = more bullish analyst ratings = higher expectations

    Args:
        ticker: Stock symbol

    Returns:
        Score 0-100 or None if data unavailable
    """
    metrics = get_analyst_metrics(ticker)

    if metrics is None:
        return None

    return metrics.revision_score


def calculate_estimate_momentum(ticker: str) -> Optional[float]:
    """
    Calculate estimate momentum score (0-100) for IES.

    Higher score = falling/stable estimates = easier to beat
    Lower score = rising estimates = harder to beat

    Args:
        ticker: Stock symbol

    Returns:
        Score 0-100 or None if data unavailable
    """
    metrics = get_estimate_metrics(ticker)

    if metrics is None:
        return None

    return metrics.estimate_momentum_score


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def calculate_all_analyst_inputs(ticker: str) -> Dict[str, Optional[float]]:
    """
    Calculate all analyst-based IES inputs at once.

    Args:
        ticker: Stock symbol

    Returns:
        Dictionary with all analyst inputs:
        - revision_score: Analyst sentiment score (0-100)
        - estimate_momentum_score: Estimate revision direction (0-100)
        - consensus: Analyst consensus rating
        - buy_pct: Percentage of buy ratings
        - target_upside_pct: Price target upside %
        - beat_rate: Historical earnings beat rate
        - num_revisions_up/down: Recent estimate revision counts
        - positivity_change_30d/90d: Change in analyst positivity (from YOUR DB)
    """
    analyst_metrics = get_analyst_metrics(ticker)
    estimate_metrics = get_estimate_metrics(ticker)

    # Also get detailed DB momentum if available
    db_momentum = get_estimate_momentum_from_db(ticker)

    return {
        # Primary IES inputs
        'revision_score': analyst_metrics.revision_score if analyst_metrics else None,
        'estimate_momentum_score': estimate_metrics.estimate_momentum_score if estimate_metrics else None,

        # Normalized scores for IES
        'revision_score_normalized': normalize_revision_to_score(
            analyst_metrics.revision_score if analyst_metrics else None
        ),
        'momentum_score_normalized': normalize_momentum_to_score(
            estimate_metrics.estimate_momentum_score if estimate_metrics else None
        ),

        # Analyst details
        'consensus': analyst_metrics.consensus if analyst_metrics else None,
        'buy_pct': analyst_metrics.buy_pct if analyst_metrics else None,
        'target_upside_pct': analyst_metrics.target_upside_pct if analyst_metrics else None,
        'total_analysts': analyst_metrics.total_analysts if analyst_metrics else 0,
        'positivity_pct': analyst_metrics.positivity_pct if analyst_metrics else None,
        'current_price': analyst_metrics.current_price if analyst_metrics else None,
        'target_mean': analyst_metrics.target_mean if analyst_metrics else None,

        # Estimate details
        'current_eps_estimate': estimate_metrics.current_eps_estimate if estimate_metrics else None,
        'next_quarter_estimate': estimate_metrics.next_quarter_estimate if estimate_metrics else None,
        'beat_rate': estimate_metrics.beat_rate if estimate_metrics else None,
        'avg_surprise_pct': estimate_metrics.avg_surprise_pct if estimate_metrics else None,
        'num_revisions_up': estimate_metrics.num_revisions_up if estimate_metrics else 0,
        'num_revisions_down': estimate_metrics.num_revisions_down if estimate_metrics else 0,

        # Detailed momentum from YOUR database
        'positivity_change_30d': db_momentum.get('positivity_change_30d') if db_momentum else None,
        'positivity_change_90d': db_momentum.get('positivity_change_90d') if db_momentum else None,
        'net_change_30d': db_momentum.get('net_change_30d') if db_momentum else None,
        'net_change_90d': db_momentum.get('net_change_90d') if db_momentum else None,
        'target_upside_change_30d': db_momentum.get('target_upside_change_30d') if db_momentum else None,
        'data_source': db_momentum.get('data_source', 'yfinance') if db_momentum else 'yfinance',
    }


def get_analyst_summary_for_ai(ticker: str) -> str:
    """
    Get a formatted summary of analyst inputs for AI context.

    Args:
        ticker: Stock symbol

    Returns:
        Formatted string for AI consumption
    """
    inputs = calculate_all_analyst_inputs(ticker)

    lines = [
        f"\n{'='*50}",
        f"ANALYST EXPECTATIONS: {ticker}",
        f"{'='*50}",
        f"Data Source: {inputs.get('data_source', 'yfinance').upper()}",
    ]

    # Analyst ratings section
    if inputs['consensus'] is not None:
        lines.extend([
            f"\n📊 ANALYST RATINGS:",
            f"   Consensus: {inputs['consensus']}",
            f"   Buy %: {inputs['buy_pct']:.0f}%",
            f"   Total Analysts: {inputs['total_analysts']}",
            f"   Positivity: {inputs['positivity_pct']:.0f}%",
        ])

        if inputs['target_upside_pct'] is not None:
            lines.append(f"   Target Upside: {inputs['target_upside_pct']:+.1f}%")

        lines.append(f"   → Revision Score: {inputs['revision_score']:.0f}/100")
    else:
        lines.append("\n📊 ANALYST RATINGS: Not available")

    # Estimate momentum section
    lines.append("\n📈 ESTIMATE MOMENTUM (Forward-Looking):")

    if inputs['current_eps_estimate'] is not None:
        lines.append(f"   Current EPS Estimate: ${inputs['current_eps_estimate']:.2f}")

    if inputs['next_quarter_estimate'] is not None:
        lines.append(f"   Next Quarter Estimate: ${inputs['next_quarter_estimate']:.2f}")

    # Show detailed DB momentum if available
    if inputs.get('data_source') == 'database':
        if inputs.get('net_change_30d') is not None:
            net_30 = inputs['net_change_30d']
            direction = "↑ upgrades" if net_30 > 0 else "↓ downgrades" if net_30 < 0 else "unchanged"
            lines.append(f"   Net 30-Day Change: {abs(net_30)} {direction}")

        if inputs.get('net_change_90d') is not None:
            net_90 = inputs['net_change_90d']
            direction = "↑ upgrades" if net_90 > 0 else "↓ downgrades" if net_90 < 0 else "unchanged"
            lines.append(f"   Net 90-Day Change: {abs(net_90)} {direction}")

        if inputs.get('positivity_change_30d') is not None:
            pos_change = inputs['positivity_change_30d']
            lines.append(f"   Positivity Change (30d): {pos_change:+.1f}%")

        if inputs.get('target_upside_change_30d') is not None:
            target_change = inputs['target_upside_change_30d']
            lines.append(f"   Target Upside Change (30d): {target_change:+.1f}%")
    else:
        lines.append(f"   Recent Upgrades: {inputs['num_revisions_up']}")
        lines.append(f"   Recent Downgrades: {inputs['num_revisions_down']}")

    if inputs['beat_rate'] is not None:
        lines.append(f"   Historical Beat Rate: {inputs['beat_rate']:.0f}%")

    if inputs['avg_surprise_pct'] is not None:
        lines.append(f"   Avg Surprise: {inputs['avg_surprise_pct']:+.1f}%")

    if inputs['estimate_momentum_score'] is not None:
        lines.append(f"   → Momentum Score: {inputs['estimate_momentum_score']:.0f}/100")

    # Interpretation
    lines.append("\n📋 IES INTERPRETATION:")

    rev_score = inputs.get('revision_score_normalized', 50)
    mom_score = inputs.get('momentum_score_normalized', 50)

    if rev_score > 70:
        lines.append("   • High analyst expectations (harder to beat)")
    elif rev_score < 30:
        lines.append("   • Low analyst expectations (easier to beat)")
    else:
        lines.append("   • Moderate analyst expectations")

    if mom_score > 60:
        lines.append("   • Estimates falling/stable (favorable for surprise)")
    elif mom_score < 40:
        lines.append("   • Estimates rising (bar getting higher)")
    else:
        lines.append("   • Estimate momentum neutral")

    return "\n".join(lines)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("Phase 6 Analyst Expectations Test (Redesigned)")
    print("=" * 60)

    test_tickers = ["AAPL", "TSLA", "NVDA"]

    for ticker in test_tickers:
        print(f"\n--- {ticker} ---")
        inputs = calculate_all_analyst_inputs(ticker)

        print(f"Revision Score: {inputs['revision_score']:.0f}" if inputs['revision_score'] else "Revision Score: N/A")
        print(f"Consensus: {inputs['consensus']}")
        print(f"Target Upside: {inputs['target_upside_pct']:+.1f}%" if inputs['target_upside_pct'] else "Target Upside: N/A")
        print(f"Momentum Score: {inputs['estimate_momentum_score']:.0f}" if inputs['estimate_momentum_score'] else "Momentum Score: N/A")
        print(f"Beat Rate: {inputs['beat_rate']:.0f}%" if inputs['beat_rate'] else "Beat Rate: N/A")
        print(f"Revisions: ↑{inputs['num_revisions_up']} ↓{inputs['num_revisions_down']}")