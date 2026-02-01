"""
Fast Scanning Configuration

Optimizes signal generation for speed while respecting API rate limits.

Rate Limits (as of Dec 2024):
- Yahoo Finance: ~360/hour (~6/min), unstable - can block aggressively
- Finnhub: 60 calls/minute (free tier)
- SEC EDGAR: 10 requests/second
- Alpha Vantage: 25 calls/day (very limited)

Usage:
    from src.core.fast_scan_config import FastScanConfig, get_scan_config
    config = get_scan_config()

Author: Alpha Research Platform
Version: 2024-12-28
"""

import os
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict
from datetime import datetime, timedelta
from collections import deque

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    Thread-safe implementation.
    """
    calls_per_minute: int = 60
    burst_limit: int = 10

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _call_times: deque = field(default_factory=lambda: deque(maxlen=1000), repr=False)
    _last_call: float = field(default=0.0, repr=False)

    def wait_if_needed(self) -> float:
        """
        Wait if we're over the rate limit.
        Returns the time waited in seconds.
        """
        with self._lock:
            now = time.time()

            # Clean old entries (older than 1 minute)
            cutoff = now - 60
            while self._call_times and self._call_times[0] < cutoff:
                self._call_times.popleft()

            # Check if we're over the limit
            if len(self._call_times) >= self.calls_per_minute:
                # Wait until oldest call expires
                wait_time = self._call_times[0] + 60 - now + 0.1
                if wait_time > 0:
                    logger.debug(f"Rate limit: waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    now = time.time()

            # Enforce minimum delay between calls (burst protection)
            min_delay = 60.0 / self.calls_per_minute / 2  # Half the average rate
            if self._last_call > 0:
                elapsed = now - self._last_call
                if elapsed < min_delay:
                    time.sleep(min_delay - elapsed)
                    now = time.time()

            # Record this call
            self._call_times.append(now)
            self._last_call = now

            return 0.0

    def get_calls_remaining(self) -> int:
        """Get number of calls remaining in current minute."""
        with self._lock:
            now = time.time()
            cutoff = now - 60
            while self._call_times and self._call_times[0] < cutoff:
                self._call_times.popleft()
            return max(0, self.calls_per_minute - len(self._call_times))


# Global rate limiters for each API
_rate_limiters: Dict[str, RateLimiter] = {}


def get_rate_limiter(api_name: str) -> RateLimiter:
    """Get or create rate limiter for an API."""
    if api_name not in _rate_limiters:
        # Default limits based on research
        limits = {
            'yfinance': RateLimiter(calls_per_minute=30, burst_limit=5),  # Conservative
            'finnhub': RateLimiter(calls_per_minute=55, burst_limit=10),  # Under 60 limit
            'sec': RateLimiter(calls_per_minute=120, burst_limit=10),  # 10/sec = 600/min, be conservative
            'default': RateLimiter(calls_per_minute=30, burst_limit=5),
        }
        _rate_limiters[api_name] = limits.get(api_name, limits['default'])
    return _rate_limiters[api_name]


@dataclass
class FastScanConfig:
    """
    Configuration for fast scanning mode.

    Balances speed with API rate limits to avoid getting blocked.
    """

    # Parallel processing (conservative to respect rate limits)
    max_workers: int = 5  # Keep low to avoid rate limit issues

    # Feature toggles (set False to skip slow/rate-limited features)
    enable_insider_tracking: bool = True  # SEC Form 4 - slow
    enable_13f_tracking: bool = True  # SEC 13F - slow
    enable_earnings_whisper: bool = True
    enable_news_sentiment: bool = True
    enable_options_flow: bool = True
    enable_short_squeeze: bool = True
    enable_technical: bool = True
    enable_fundamental: bool = True

    # Caching
    use_db_cache: bool = True
    cache_ttl_hours: int = 4

    # Rate limiting (calls per minute)
    yfinance_rpm: int = 30  # Very conservative for Yahoo
    finnhub_rpm: int = 55  # Under 60 limit
    sec_rpm: int = 120  # Conservative

    # Delays between API calls (seconds)
    yfinance_delay: float = 0.5  # 500ms between Yahoo calls
    finnhub_delay: float = 0.1  # 100ms between Finnhub calls
    sec_delay: float = 0.2  # 200ms between SEC calls

    # Timeouts (seconds)
    api_timeout: int = 15

    # Mode presets
    fast_mode: bool = False

    def __post_init__(self):
        """Apply mode preset if enabled."""
        if self.fast_mode:
            self.enable_fast_mode()

    def enable_fast_mode(self):
        """
        Enable fast mode - use DB cache only, skip slow API calls.
        This is 10-100x faster than full mode.
        """
        logger.info("🚀 Fast scan mode ENABLED - using cached data only")

        # Use more workers since we're hitting DB not APIs
        self.max_workers = 20

        # Disable slow SEC calls
        self.enable_insider_tracking = False
        self.enable_13f_tracking = False

        # Keep fast features
        self.enable_earnings_whisper = True
        self.enable_news_sentiment = True
        self.enable_options_flow = True
        self.enable_short_squeeze = True
        self.enable_technical = True
        self.enable_fundamental = True

        # Aggressive caching
        self.use_db_cache = True
        self.cache_ttl_hours = 24

    def enable_balanced_mode(self):
        """
        Balanced mode - moderate speed with some API calls.
        Good for daily refreshes.
        """
        logger.info("⚖️ Balanced scan mode ENABLED")

        self.max_workers = 5

        # Enable most features except slowest
        self.enable_insider_tracking = False  # Skip (slow)
        self.enable_13f_tracking = False  # Skip (slow)
        self.enable_earnings_whisper = True
        self.enable_news_sentiment = True
        self.enable_options_flow = True
        self.enable_short_squeeze = True
        self.enable_technical = True
        self.enable_fundamental = True

        self.use_db_cache = True
        self.cache_ttl_hours = 4

    def enable_full_mode(self):
        """
        Full mode - all features, slower but complete.
        Use for weekend/overnight full refreshes.
        """
        logger.info("📊 Full scan mode ENABLED (slower, complete)")

        self.max_workers = 3  # Very conservative

        # Enable all features
        self.enable_insider_tracking = True
        self.enable_13f_tracking = True
        self.enable_earnings_whisper = True
        self.enable_news_sentiment = True
        self.enable_options_flow = True
        self.enable_short_squeeze = True
        self.enable_technical = True
        self.enable_fundamental = True

        self.use_db_cache = True
        self.cache_ttl_hours = 1

    def estimate_scan_time(self, num_tickers: int) -> str:
        """Estimate scan time based on current settings."""
        if self.use_db_cache and not self.enable_insider_tracking:
            # Fast mode: ~0.1s per ticker (DB only)
            seconds = num_tickers * 0.1 / self.max_workers
        elif not self.enable_insider_tracking:
            # Balanced: ~2s per ticker
            seconds = num_tickers * 2 / self.max_workers
        else:
            # Full: ~15s per ticker (SEC calls)
            seconds = num_tickers * 15 / self.max_workers

        if seconds < 60:
            return f"~{int(seconds)} seconds"
        elif seconds < 3600:
            return f"~{int(seconds/60)} minutes"
        else:
            return f"~{seconds/3600:.1f} hours"


# Global config instance
_scan_config: Optional[FastScanConfig] = None


def get_scan_config() -> FastScanConfig:
    """Get the global scan configuration."""
    global _scan_config
    if _scan_config is None:
        fast_mode = os.getenv('FAST_SCAN_MODE', 'true').lower() == 'true'
        _scan_config = FastScanConfig(fast_mode=fast_mode)
    return _scan_config


def set_scan_mode(mode: str = 'fast') -> FastScanConfig:
    """
    Set scan mode globally.

    Args:
        mode: 'fast', 'balanced', or 'full'
    """
    global _scan_config
    _scan_config = FastScanConfig()

    if mode == 'fast':
        _scan_config.enable_fast_mode()
    elif mode == 'balanced':
        _scan_config.enable_balanced_mode()
    elif mode == 'full':
        _scan_config.enable_full_mode()

    return _scan_config


def rate_limited_call(api_name: str, func, *args, **kwargs):
    """
    Execute a function with rate limiting.

    Args:
        api_name: Name of API ('yfinance', 'finnhub', 'sec')
        func: Function to call
        *args, **kwargs: Arguments to pass to function

    Returns:
        Result of function call
    """
    limiter = get_rate_limiter(api_name)
    limiter.wait_if_needed()
    return func(*args, **kwargs)