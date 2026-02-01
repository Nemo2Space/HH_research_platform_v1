"""
Smart Data Refresh System

Tracks data freshness per ticker and data type.
Only refreshes stale data to avoid unnecessary API calls.

Key Concepts:
- Each data type has a different TTL (time-to-live)
- Critical data (news, earnings) refreshes more often
- Stable data (fundamentals, 13F) refreshes weekly
- Earnings proximity triggers more frequent updates

Author: Alpha Research Platform
Version: 2024-12-29
"""

import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.db.connection import get_engine, get_connection
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataType(Enum):
    """Types of data we track for freshness."""
    PRICE = "price"
    NEWS = "news"
    SENTIMENT = "sentiment"
    EARNINGS = "earnings"
    OPTIONS_FLOW = "options_flow"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    INSIDER = "insider"
    INSTITUTIONAL = "institutional"
    SHORT_SQUEEZE = "short_squeeze"


@dataclass
class RefreshConfig:
    """Configuration for data refresh intervals."""

    # TTL in hours for each data type
    ttl_hours: Dict[DataType, int] = field(default_factory=lambda: {
        DataType.PRICE: 1,           # Refresh hourly during market
        DataType.NEWS: 4,            # Every 4 hours
        DataType.SENTIMENT: 4,       # Every 4 hours
        DataType.EARNINGS: 24,       # Daily (check for upcoming)
        DataType.OPTIONS_FLOW: 4,    # Every 4 hours
        DataType.TECHNICAL: 24,      # Daily (end of day)
        DataType.FUNDAMENTAL: 168,   # Weekly (7 days)
        DataType.INSIDER: 168,       # Weekly
        DataType.INSTITUTIONAL: 168, # Weekly
        DataType.SHORT_SQUEEZE: 24,  # Daily
    })

    # Near-earnings multiplier (refresh more often when earnings approaching)
    earnings_proximity_days: int = 7
    earnings_ttl_multiplier: float = 0.25  # 4x more frequent near earnings

    # Weekend handling
    skip_weekends: bool = True

    def get_ttl(self, data_type: DataType, days_to_earnings: Optional[int] = None) -> timedelta:
        """Get TTL for a data type, considering earnings proximity."""
        base_hours = self.ttl_hours.get(data_type, 24)

        # If near earnings, refresh more frequently for relevant data
        if days_to_earnings is not None and days_to_earnings <= self.earnings_proximity_days:
            if data_type in [DataType.NEWS, DataType.SENTIMENT, DataType.OPTIONS_FLOW]:
                base_hours = int(base_hours * self.earnings_ttl_multiplier)

        return timedelta(hours=base_hours)


# ============================================================================
# DATABASE TRACKING
# ============================================================================

def ensure_freshness_table():
    """Create data_freshness table to track when each data type was last updated."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS data_freshness (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(20) NOT NULL,
        data_type VARCHAR(30) NOT NULL,
        last_updated TIMESTAMP NOT NULL,
        next_update TIMESTAMP NOT NULL,
        update_count INTEGER DEFAULT 1,
        last_status VARCHAR(20) DEFAULT 'SUCCESS',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, data_type)
    );
    
    CREATE INDEX IF NOT EXISTS idx_freshness_ticker ON data_freshness(ticker);
    CREATE INDEX IF NOT EXISTS idx_freshness_next ON data_freshness(next_update);
    CREATE INDEX IF NOT EXISTS idx_freshness_type ON data_freshness(data_type);
    """

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(create_sql)
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error creating data_freshness table: {e}")
        return False


def mark_updated(ticker: str, data_type: DataType, config: RefreshConfig = None,
                 days_to_earnings: int = None, status: str = "SUCCESS"):
    """Mark a data type as freshly updated for a ticker."""
    ensure_freshness_table()

    if config is None:
        config = RefreshConfig()

    now = datetime.now()
    ttl = config.get_ttl(data_type, days_to_earnings)
    next_update = now + ttl

    sql = """
    INSERT INTO data_freshness (ticker, data_type, last_updated, next_update, last_status)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (ticker, data_type) 
    DO UPDATE SET 
        last_updated = EXCLUDED.last_updated,
        next_update = EXCLUDED.next_update,
        update_count = data_freshness.update_count + 1,
        last_status = EXCLUDED.last_status
    """

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (ticker, data_type.value, now, next_update, status))
            conn.commit()
    except Exception as e:
        logger.error(f"Error marking {ticker}/{data_type.value} as updated: {e}")


def mark_batch_updated(tickers: List[str], data_type: DataType, config: RefreshConfig = None):
    """Mark multiple tickers as updated for a data type."""
    for ticker in tickers:
        mark_updated(ticker, data_type, config)


def get_stale_tickers(data_type: DataType, tickers: List[str] = None) -> List[str]:
    """
    Get list of tickers that need refresh for a given data type.

    Args:
        data_type: Which data type to check
        tickers: Optional list to filter (if None, checks all)

    Returns:
        List of tickers that need updating
    """
    ensure_freshness_table()

    now = datetime.now()

    if tickers:
        # Check specific tickers
        placeholders = ','.join(['%s'] * len(tickers))
        sql = f"""
        SELECT ticker FROM (
            SELECT unnest(ARRAY[{placeholders}]) as ticker
        ) t
        WHERE ticker NOT IN (
            SELECT ticker FROM data_freshness 
            WHERE data_type = %s AND next_update > %s
        )
        """
        params = tuple(tickers) + (data_type.value, now)
    else:
        # Get all stale tickers
        sql = """
        SELECT DISTINCT ticker FROM data_freshness
        WHERE data_type = %s AND next_update <= %s
        """
        params = (data_type.value, now)

    try:
        df = pd.read_sql(sql, get_engine(), params=params)
        return df['ticker'].tolist() if not df.empty else []
    except Exception as e:
        logger.error(f"Error getting stale tickers for {data_type.value}: {e}")
        return tickers or []  # If error, assume all need refresh


def get_freshness_status(tickers: List[str] = None) -> pd.DataFrame:
    """
    Get freshness status for all data types.

    Returns DataFrame with columns:
    - ticker, data_type, last_updated, next_update, is_stale
    """
    ensure_freshness_table()

    now = datetime.now()

    if tickers:
        placeholders = ','.join(['%s'] * len(tickers))
        sql = f"""
        SELECT 
            ticker, 
            data_type, 
            last_updated, 
            next_update,
            CASE WHEN next_update <= %s THEN true ELSE false END as is_stale,
            update_count,
            last_status
        FROM data_freshness
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, data_type
        """
        params = (now,) + tuple(tickers)
    else:
        sql = """
        SELECT 
            ticker, 
            data_type, 
            last_updated, 
            next_update,
            CASE WHEN next_update <= %s THEN true ELSE false END as is_stale,
            update_count,
            last_status
        FROM data_freshness
        ORDER BY ticker, data_type
        """
        params = (now,)

    try:
        return pd.read_sql(sql, get_engine(), params=params)
    except Exception as e:
        logger.error(f"Error getting freshness status: {e}")
        return pd.DataFrame()


# ============================================================================
# SMART REFRESH LOGIC
# ============================================================================

@dataclass
class RefreshPlan:
    """Plan for what needs to be refreshed."""
    tickers_by_type: Dict[DataType, List[str]] = field(default_factory=dict)
    skip_count: int = 0
    refresh_count: int = 0

    def add(self, data_type: DataType, tickers: List[str]):
        """Add tickers that need refresh for a data type."""
        if tickers:
            self.tickers_by_type[data_type] = tickers
            self.refresh_count += len(tickers)

    def get_tickers_for(self, data_type: DataType) -> List[str]:
        """Get tickers needing refresh for a data type."""
        return self.tickers_by_type.get(data_type, [])

    def summary(self) -> str:
        """Get human-readable summary of the plan."""
        lines = ["📊 Smart Refresh Plan:"]
        for dt, tickers in self.tickers_by_type.items():
            lines.append(f"  • {dt.value}: {len(tickers)} tickers")
        lines.append(f"  → Total refreshes: {self.refresh_count}")
        lines.append(f"  → Skipped (fresh): {self.skip_count}")
        return "\n".join(lines)


def create_refresh_plan(tickers: List[str], config: RefreshConfig = None,
                        earnings_dates: Dict[str, date] = None) -> RefreshPlan:
    """
    Create a smart refresh plan based on data freshness.

    Args:
        tickers: List of tickers to consider
        config: Refresh configuration
        earnings_dates: Dict of ticker -> next earnings date (optional)

    Returns:
        RefreshPlan with tickers organized by what needs refreshing
    """
    if config is None:
        config = RefreshConfig()

    plan = RefreshPlan()
    total_checks = len(tickers) * len(DataType)

    for data_type in DataType:
        stale = get_stale_tickers(data_type, tickers)

        # If we have earnings dates, check proximity
        if earnings_dates and data_type in [DataType.NEWS, DataType.SENTIMENT, DataType.OPTIONS_FLOW]:
            # Force refresh for tickers near earnings even if not technically stale
            today = date.today()
            near_earnings = [
                t for t in tickers
                if t in earnings_dates and
                (earnings_dates[t] - today).days <= config.earnings_proximity_days
            ]
            stale = list(set(stale) | set(near_earnings))

        plan.add(data_type, stale)

    plan.skip_count = total_checks - plan.refresh_count

    return plan


def get_earnings_proximity(tickers: List[str]) -> Dict[str, int]:
    """
    Get days until next earnings for each ticker.

    Returns:
        Dict of ticker -> days_to_earnings (None if unknown)
    """
    try:
        sql = """
        SELECT ticker, earnings_date
        FROM earnings_calendar
        WHERE ticker = ANY(%s)
          AND earnings_date >= CURRENT_DATE
        ORDER BY ticker, earnings_date
        """

        df = pd.read_sql(sql, get_engine(), params=(tickers,))

        if df.empty:
            return {}

        today = date.today()
        result = {}

        for ticker in df['ticker'].unique():
            ticker_dates = df[df['ticker'] == ticker]['earnings_date']
            if not ticker_dates.empty:
                next_date = ticker_dates.iloc[0]
                if isinstance(next_date, datetime):
                    next_date = next_date.date()
                result[ticker] = (next_date - today).days

        return result

    except Exception as e:
        logger.debug(f"Could not get earnings proximity: {e}")
        return {}


# ============================================================================
# SMART SCANNER INTEGRATION
# ============================================================================

class SmartScanner:
    """
    Scanner that only refreshes stale data.

    Usage:
        scanner = SmartScanner()
        plan = scanner.create_plan(tickers)
        print(plan.summary())
        scanner.execute(plan)
    """

    def __init__(self, config: RefreshConfig = None):
        self.config = config or RefreshConfig()
        ensure_freshness_table()

    def create_plan(self, tickers: List[str]) -> RefreshPlan:
        """Create a refresh plan for the given tickers."""
        # Get earnings dates for proximity check
        earnings_dates = {}
        try:
            proximity = get_earnings_proximity(tickers)
            today = date.today()
            for ticker, days in proximity.items():
                earnings_dates[ticker] = today + timedelta(days=days)
        except:
            pass

        return create_refresh_plan(tickers, self.config, earnings_dates)

    def get_fresh_tickers(self, tickers: List[str], data_type: DataType) -> List[str]:
        """Get tickers that DON'T need refresh (already fresh)."""
        stale = set(get_stale_tickers(data_type, tickers))
        return [t for t in tickers if t not in stale]

    def get_stale_tickers(self, tickers: List[str], data_type: DataType) -> List[str]:
        """Get tickers that DO need refresh."""
        return get_stale_tickers(data_type, tickers)

    def mark_complete(self, ticker: str, data_type: DataType, success: bool = True):
        """Mark a refresh as complete."""
        mark_updated(ticker, data_type, self.config,
                    status="SUCCESS" if success else "FAILED")

    def get_status_summary(self, tickers: List[str] = None) -> Dict[str, any]:
        """Get summary of data freshness."""
        df = get_freshness_status(tickers)

        if df.empty:
            return {
                "total_entries": 0,
                "stale_count": 0,
                "fresh_count": 0,
                "by_type": {}
            }

        stale_count = df['is_stale'].sum() if 'is_stale' in df.columns else 0

        by_type = {}
        for dt in DataType:
            type_df = df[df['data_type'] == dt.value]
            if not type_df.empty:
                by_type[dt.value] = {
                    "total": len(type_df),
                    "stale": type_df['is_stale'].sum() if 'is_stale' in type_df.columns else 0,
                    "fresh": len(type_df) - (type_df['is_stale'].sum() if 'is_stale' in type_df.columns else 0)
                }

        return {
            "total_entries": len(df),
            "stale_count": int(stale_count),
            "fresh_count": len(df) - int(stale_count),
            "by_type": by_type
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def should_refresh(ticker: str, data_type: DataType) -> bool:
    """Quick check if a single ticker/type needs refresh."""
    stale = get_stale_tickers(data_type, [ticker])
    return ticker in stale


def get_refresh_stats() -> Dict[str, any]:
    """Get overall refresh statistics."""
    scanner = SmartScanner()
    return scanner.get_status_summary()


def reset_freshness(tickers: List[str] = None, data_types: List[DataType] = None):
    """
    Reset freshness tracking (force refresh on next scan).

    Args:
        tickers: Specific tickers to reset (None = all)
        data_types: Specific data types to reset (None = all)
    """
    ensure_freshness_table()

    conditions = []
    params = []

    if tickers:
        placeholders = ','.join(['%s'] * len(tickers))
        conditions.append(f"ticker IN ({placeholders})")
        params.extend(tickers)

    if data_types:
        type_values = [dt.value for dt in data_types]
        placeholders = ','.join(['%s'] * len(type_values))
        conditions.append(f"data_type IN ({placeholders})")
        params.extend(type_values)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    sql = f"DELETE FROM data_freshness WHERE {where_clause}"

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()
        logger.info(f"Reset freshness tracking: {len(tickers or [])} tickers, {len(data_types or [])} types")
    except Exception as e:
        logger.error(f"Error resetting freshness: {e}")