"""
Earnings Intelligence System - Earnings Calendar & Window Detection

Handles:
- Fetching earnings dates from yfinance
- Determining if a ticker is within compute window (-10 to +2 days)
- Determining if a ticker is within action window (-5 to +2 days)
- Caching earnings dates to reduce API calls

Author: Alpha Research Platform
"""

import os
import json
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import yfinance as yf
import pandas as pd

from src.utils.logging import get_logger
from src.analytics.earnings_intelligence.models import (
    COMPUTE_WINDOW_START,
    COMPUTE_WINDOW_END,
    ACTION_WINDOW_START,
    ACTION_WINDOW_END,
)

logger = get_logger(__name__)

# Cache settings
CACHE_FILE = "data/earnings_calendar_intelligence.json"
CACHE_EXPIRY_HOURS = 24


@dataclass
class EarningsDateInfo:
    """Information about a ticker's earnings date."""
    ticker: str
    earnings_date: Optional[date] = None
    earnings_timestamp: Optional[datetime] = None
    time_of_day: str = "UNKNOWN"  # BMO (Before Market Open), AMC (After Market Close), UNKNOWN

    # Window status
    days_to_earnings: Optional[int] = None
    in_compute_window: bool = False
    in_action_window: bool = False

    # Data freshness
    fetched_at: Optional[datetime] = None
    source: str = "yfinance"

    def to_dict(self) -> dict:
        return {
            'ticker': self.ticker,
            'earnings_date': self.earnings_date.isoformat() if self.earnings_date else None,
            'earnings_timestamp': self.earnings_timestamp.isoformat() if self.earnings_timestamp else None,
            'time_of_day': self.time_of_day,
            'days_to_earnings': self.days_to_earnings,
            'in_compute_window': self.in_compute_window,
            'in_action_window': self.in_action_window,
            'fetched_at': self.fetched_at.isoformat() if self.fetched_at else None,
            'source': self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'EarningsDateInfo':
        return cls(
            ticker=data['ticker'],
            earnings_date=date.fromisoformat(data['earnings_date']) if data.get('earnings_date') else None,
            earnings_timestamp=datetime.fromisoformat(data['earnings_timestamp']) if data.get('earnings_timestamp') else None,
            time_of_day=data.get('time_of_day', 'UNKNOWN'),
            days_to_earnings=data.get('days_to_earnings'),
            in_compute_window=data.get('in_compute_window', False),
            in_action_window=data.get('in_action_window', False),
            fetched_at=datetime.fromisoformat(data['fetched_at']) if data.get('fetched_at') else None,
            source=data.get('source', 'yfinance'),
        )


class EarningsCalendarManager:
    """
    Manages earnings calendar data with caching.

    Fetches earnings dates from yfinance and determines window status.
    """

    def __init__(self, cache_file: str = None):
        self.cache_file = cache_file or CACHE_FILE
        self._cache: Dict[str, EarningsDateInfo] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cache from file."""
        try:
            cache_path = Path(self.cache_file)
            if cache_path.exists():
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for ticker, info in data.get('tickers', {}).items():
                    self._cache[ticker] = EarningsDateInfo.from_dict(info)

                logger.debug(f"Loaded {len(self._cache)} tickers from earnings cache")
        except Exception as e:
            logger.warning(f"Could not load earnings cache: {e}")
            self._cache = {}

    def _save_cache(self):
        """Save cache to file."""
        try:
            cache_path = Path(self.cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'updated_at': datetime.now().isoformat(),
                'tickers': {ticker: info.to_dict() for ticker, info in self._cache.items()}
            }

            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Could not save earnings cache: {e}")

    def _is_cache_valid(self, info: EarningsDateInfo) -> bool:
        """Check if cached info is still valid."""
        if not info.fetched_at:
            return False

        age_hours = (datetime.now() - info.fetched_at).total_seconds() / 3600

        # Basic age check
        if age_hours >= CACHE_EXPIRY_HOURS:
            return False

        # If earnings date is far in the past (>14 days ago), refresh to get next date
        if info.earnings_date:
            days_since = (date.today() - info.earnings_date).days
            if days_since > 14:
                logger.debug(f"{info.ticker}: Cache stale - earnings {days_since} days ago, refreshing")
                return False

        return True

    def _fetch_earnings_date(self, ticker: str) -> EarningsDateInfo:
        """
        Fetch earnings date from yfinance.

        Args:
            ticker: Stock symbol

        Returns:
            EarningsDateInfo with earnings date or None if not found
        """
        info = EarningsDateInfo(
            ticker=ticker,
            fetched_at=datetime.now(),
            source='yfinance'
        )

        try:
            stock = yf.Ticker(ticker)

            # Method 1: Try calendar property (most reliable for upcoming)
            try:
                calendar = stock.calendar
                if calendar is not None and not calendar.empty:
                    # calendar can be DataFrame or dict depending on yfinance version
                    if isinstance(calendar, pd.DataFrame):
                        if 'Earnings Date' in calendar.index:
                            earnings_dates = calendar.loc['Earnings Date']
                            if len(earnings_dates) > 0:
                                ed = earnings_dates.iloc[0]
                                if pd.notna(ed):
                                    if isinstance(ed, (datetime, pd.Timestamp)):
                                        info.earnings_date = ed.date() if hasattr(ed, 'date') else ed
                                        info.earnings_timestamp = ed if isinstance(ed, datetime) else None
                    elif isinstance(calendar, dict):
                        if 'Earnings Date' in calendar:
                            ed = calendar['Earnings Date']
                            if isinstance(ed, list) and len(ed) > 0:
                                ed = ed[0]
                            if ed and pd.notna(ed):
                                if isinstance(ed, (datetime, pd.Timestamp)):
                                    info.earnings_date = ed.date() if hasattr(ed, 'date') else ed
            except Exception as e:
                logger.debug(f"{ticker}: Calendar method failed: {e}")

            # Method 2: Try earnings_dates property
            if not info.earnings_date:
                try:
                    earnings_dates = stock.earnings_dates
                    if earnings_dates is not None and not earnings_dates.empty:
                        # Get the next upcoming earnings (future dates)
                        today = datetime.now()
                        future_dates = earnings_dates[earnings_dates.index > today]

                        if not future_dates.empty:
                            next_date = future_dates.index[0]
                            info.earnings_date = next_date.date() if hasattr(next_date, 'date') else next_date
                            info.earnings_timestamp = next_date if isinstance(next_date, datetime) else None
                        else:
                            # If no future dates, get the most recent past date
                            past_dates = earnings_dates[earnings_dates.index <= today]
                            if not past_dates.empty:
                                last_date = past_dates.index[-1]
                                info.earnings_date = last_date.date() if hasattr(last_date, 'date') else last_date
                except Exception as e:
                    logger.debug(f"{ticker}: earnings_dates method failed: {e}")

            # Method 3: Try info dict
            if not info.earnings_date:
                try:
                    stock_info = stock.info
                    if stock_info:
                        # Check for earnings date in info
                        for key in ['earningsDate', 'nextEarningsDate', 'mostRecentQuarter']:
                            if key in stock_info and stock_info[key]:
                                val = stock_info[key]
                                if isinstance(val, (int, float)):
                                    # Unix timestamp
                                    info.earnings_date = datetime.fromtimestamp(val).date()
                                    break
                                elif isinstance(val, (datetime, date)):
                                    info.earnings_date = val if isinstance(val, date) else val.date()
                                    break
                except Exception as e:
                    logger.debug(f"{ticker}: info method failed: {e}")

            # Calculate window status
            if info.earnings_date:
                self._update_window_status(info)
                logger.debug(f"{ticker}: Earnings date = {info.earnings_date}, days = {info.days_to_earnings}")
            else:
                logger.debug(f"{ticker}: No earnings date found")

        except Exception as e:
            logger.error(f"{ticker}: Error fetching earnings date: {e}")

        return info

    def _update_window_status(self, info: EarningsDateInfo):
        """Update the window status based on earnings date."""
        if not info.earnings_date:
            info.days_to_earnings = None
            info.in_compute_window = False
            info.in_action_window = False
            return

        today = date.today()
        delta = (info.earnings_date - today).days
        info.days_to_earnings = delta

        # delta: positive = earnings in future, negative = earnings in past
        # Compute window: from 10 days before earnings to 2 days after
        # This means delta should be between -2 (2 days past) and +10 (10 days ahead)
        info.in_compute_window = COMPUTE_WINDOW_START <= delta <= COMPUTE_WINDOW_END

        # Action window: from 5 days before earnings to 2 days after
        # This means delta should be between -2 (2 days past) and +5 (5 days ahead)
        info.in_action_window = ACTION_WINDOW_START <= delta <= ACTION_WINDOW_END

    def get_earnings_info(self, ticker: str, force_refresh: bool = False) -> EarningsDateInfo:
        """
        Get earnings date info for a ticker.

        Args:
            ticker: Stock symbol
            force_refresh: Force fetch from API even if cached

        Returns:
            EarningsDateInfo
        """
        ticker = ticker.upper()

        # Check cache
        if not force_refresh and ticker in self._cache:
            cached = self._cache[ticker]
            if self._is_cache_valid(cached):
                # Update window status (might have changed since caching)
                self._update_window_status(cached)
                return cached

        # Fetch from API
        info = self._fetch_earnings_date(ticker)

        # Update cache
        self._cache[ticker] = info
        self._save_cache()

        return info

    def get_batch_earnings_info(self, tickers: List[str],
                                force_refresh: bool = False) -> Dict[str, EarningsDateInfo]:
        """
        Get earnings info for multiple tickers.

        Args:
            tickers: List of stock symbols
            force_refresh: Force fetch from API

        Returns:
            Dict mapping ticker to EarningsDateInfo
        """
        results = {}

        for ticker in tickers:
            try:
                results[ticker.upper()] = self.get_earnings_info(ticker, force_refresh)
            except Exception as e:
                logger.error(f"{ticker}: Error getting earnings info: {e}")
                results[ticker.upper()] = EarningsDateInfo(ticker=ticker.upper())

        return results

    def get_tickers_in_window(self, tickers: List[str],
                              window_type: str = 'action') -> List[str]:
        """
        Filter tickers to those within specified window.

        Args:
            tickers: List of stock symbols
            window_type: 'compute' or 'action'

        Returns:
            List of tickers in the window
        """
        in_window = []

        for ticker in tickers:
            info = self.get_earnings_info(ticker)

            if window_type == 'compute' and info.in_compute_window:
                in_window.append(ticker.upper())
            elif window_type == 'action' and info.in_action_window:
                in_window.append(ticker.upper())

        return in_window

    def clear_cache(self):
        """Clear the earnings cache."""
        self._cache = {}
        try:
            cache_path = Path(self.cache_file)
            if cache_path.exists():
                cache_path.unlink()
        except Exception as e:
            logger.warning(f"Could not delete cache file: {e}")


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

_manager: Optional[EarningsCalendarManager] = None


def get_manager() -> EarningsCalendarManager:
    """Get singleton manager instance."""
    global _manager
    if _manager is None:
        _manager = EarningsCalendarManager()
    return _manager


def get_earnings_date(ticker: str) -> Optional[date]:
    """
    Get the earnings date for a ticker.

    Args:
        ticker: Stock symbol

    Returns:
        Earnings date or None if not found
    """
    info = get_manager().get_earnings_info(ticker)
    return info.earnings_date


def get_days_to_earnings(ticker: str) -> Optional[int]:
    """
    Get days until earnings for a ticker.

    Args:
        ticker: Stock symbol

    Returns:
        Days to earnings (negative = past), or None if unknown
    """
    info = get_manager().get_earnings_info(ticker)
    return info.days_to_earnings


def is_in_compute_window(ticker: str) -> bool:
    """
    Check if ticker is in compute window (-10 to +2 days from earnings).

    Args:
        ticker: Stock symbol

    Returns:
        True if in compute window
    """
    info = get_manager().get_earnings_info(ticker)
    return info.in_compute_window


def is_in_action_window(ticker: str) -> bool:
    """
    Check if ticker is in action window (-5 to +2 days from earnings).

    Args:
        ticker: Stock symbol

    Returns:
        True if in action window
    """
    info = get_manager().get_earnings_info(ticker)
    return info.in_action_window


def get_earnings_info(ticker: str) -> EarningsDateInfo:
    """
    Get full earnings info for a ticker.

    Args:
        ticker: Stock symbol

    Returns:
        EarningsDateInfo with all details
    """
    return get_manager().get_earnings_info(ticker)


def get_upcoming_earnings(tickers: List[str], days_ahead: int = 14) -> List[EarningsDateInfo]:
    """
    Get tickers with earnings in the next N days.

    Args:
        tickers: List of stock symbols
        days_ahead: How many days to look ahead

    Returns:
        List of EarningsDateInfo for tickers with upcoming earnings, sorted by date
    """
    manager = get_manager()
    upcoming = []

    for ticker in tickers:
        info = manager.get_earnings_info(ticker)
        if info.earnings_date and info.days_to_earnings is not None:
            if 0 <= info.days_to_earnings <= days_ahead:
                upcoming.append(info)

    # Sort by earnings date
    upcoming.sort(key=lambda x: x.earnings_date or date.max)

    return upcoming


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 3 TEST: Earnings Calendar & Window Detection")
    print("=" * 60)

    # Test with some well-known tickers
    test_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "META"]

    print("\nFetching earnings dates...")
    manager = EarningsCalendarManager()

    for ticker in test_tickers:
        info = manager.get_earnings_info(ticker)

        print(f"\n{ticker}:")
        print(f"  Earnings Date: {info.earnings_date}")
        print(f"  Days to Earnings: {info.days_to_earnings}")
        print(f"  In Compute Window: {info.in_compute_window}")
        print(f"  In Action Window: {info.in_action_window}")

    # Test convenience functions
    print("\n" + "-" * 40)
    print("Testing convenience functions...")

    ticker = "AAPL"
    print(f"\nget_earnings_date('{ticker}'): {get_earnings_date(ticker)}")
    print(f"get_days_to_earnings('{ticker}'): {get_days_to_earnings(ticker)}")
    print(f"is_in_compute_window('{ticker}'): {is_in_compute_window(ticker)}")
    print(f"is_in_action_window('{ticker}'): {is_in_action_window(ticker)}")

    # Test upcoming earnings
    print("\n" + "-" * 40)
    print("Testing upcoming earnings (next 30 days)...")

    upcoming = get_upcoming_earnings(test_tickers, days_ahead=30)
    if upcoming:
        for info in upcoming:
            print(f"  {info.ticker}: {info.earnings_date} ({info.days_to_earnings} days)")
    else:
        print("  No upcoming earnings found in next 30 days")

    print("\n" + "=" * 60)
    print("[OK] Phase 3 tests complete!")
    print("=" * 60)