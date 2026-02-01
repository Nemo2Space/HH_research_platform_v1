"""
Compliance-Grade HTTP Rate Limiter
===================================

Provides robust HTTP handling for external APIs (especially SEC EDGAR):
- Configurable rate limiting
- Exponential backoff with jitter
- Disk caching with sidecar metadata
- Atomic file writes to prevent corruption
- Proper error handling for 429/503/5xx

Usage:
    from src.rag.rate_limiter import RateLimiter, RateLimiterConfig

    config = RateLimiterConfig.for_sec()
    limiter = RateLimiter(config)

    response = limiter.get(session, url)

Author: HH Research Platform
"""

import os
import time
import json
import random
import hashlib
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import requests

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimiterConfig:
    """Configuration for rate limiter."""

    # Rate limiting
    requests_per_second: float = 2.0  # Target rate (conservative for SEC)
    min_interval: float = 0.3  # Minimum time between requests

    # Retry configuration
    max_retries: int = 5
    base_delay: float = 1.0  # Initial retry delay (seconds)
    max_delay: float = 60.0  # Maximum retry delay

    # Jitter
    jitter_factor: float = 0.3  # ±30% random jitter

    # Timeouts
    connect_timeout: float = 10.0
    read_timeout: float = 30.0

    # Caching
    cache_enabled: bool = True
    cache_dir: str = field(default_factory=lambda: os.getenv('SEC_CACHE_DIR', '.sec_cache'))
    cache_ttl_seconds: int = 86400 * 365  # 1 year for immutable content (SEC filings)
    metadata_ttl_seconds: int = 86400  # 24 hours for filing lists

    @classmethod
    def for_sec(cls) -> 'RateLimiterConfig':
        """Configuration optimized for SEC EDGAR."""
        return cls(
            requests_per_second=2.0,  # Well below 10/sec limit
            min_interval=0.4,
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
            jitter_factor=0.3,
            cache_enabled=True,
        )

    @classmethod
    def for_testing(cls) -> 'RateLimiterConfig':
        """Fast configuration for testing."""
        return cls(
            requests_per_second=10.0,
            min_interval=0.05,
            max_retries=2,
            base_delay=0.1,
            cache_enabled=False,
        )


@dataclass
class CachedResponse:
    """Represents a cached HTTP response."""
    url: str
    status_code: int
    content: bytes
    headers: Dict[str, str]
    fetched_at_utc: str
    content_sha256: str

    def to_dict(self) -> Dict:
        return {
            'url': self.url,
            'status_code': self.status_code,
            'headers': self.headers,
            'fetched_at_utc': self.fetched_at_utc,
            'content_sha256': self.content_sha256,
        }

    @classmethod
    def from_response(cls, url: str, response: requests.Response) -> 'CachedResponse':
        content = response.content
        return cls(
            url=url,
            status_code=response.status_code,
            content=content,
            headers=dict(response.headers),
            fetched_at_utc=datetime.utcnow().isoformat() + 'Z',
            content_sha256=hashlib.sha256(content).hexdigest(),
        )


class RateLimiter:
    """
    Compliance-grade rate limiter with caching.

    Features:
    - Enforces rate limits with jitter
    - Exponential backoff on failures
    - Disk caching with metadata
    - Atomic file writes
    """

    def __init__(self, config: RateLimiterConfig = None):
        self.config = config or RateLimiterConfig.for_sec()
        self.last_request_time: float = 0
        self._request_count: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # Initialize cache directory
        if self.config.cache_enabled:
            self.cache_dir = Path(self.config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / 'content').mkdir(exist_ok=True)
            (self.cache_dir / 'metadata').mkdir(exist_ok=True)

    def _get_cache_paths(self, url: str) -> Tuple[Path, Path]:
        """Get cache file paths for a URL."""
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        content_file = self.cache_dir / 'content' / f'{url_hash}.bin'
        meta_file = self.cache_dir / 'metadata' / f'{url_hash}.json'
        return content_file, meta_file

    def _is_cache_valid(self, meta_file: Path, ttl_seconds: int) -> bool:
        """Check if cached item is still valid."""
        if not meta_file.exists():
            return False

        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)

            fetched_at = datetime.fromisoformat(meta['fetched_at_utc'].rstrip('Z'))
            age = (datetime.utcnow() - fetched_at).total_seconds()
            return age < ttl_seconds

        except Exception as e:
            logger.debug(f"Cache validation error: {e}")
            return False

    def _get_cached(self, url: str, ttl_seconds: int = None) -> Optional[CachedResponse]:
        """Get cached response if valid."""
        if not self.config.cache_enabled:
            return None

        ttl = ttl_seconds or self.config.cache_ttl_seconds
        content_file, meta_file = self._get_cache_paths(url)

        if not self._is_cache_valid(meta_file, ttl):
            return None

        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)

            with open(content_file, 'rb') as f:
                content = f.read()

            # Verify content integrity
            if hashlib.sha256(content).hexdigest() != meta.get('content_sha256'):
                logger.warning(f"Cache integrity check failed for {url}")
                return None

            self._cache_hits += 1
            logger.debug(f"Cache hit: {url}")

            return CachedResponse(
                url=meta['url'],
                status_code=meta['status_code'],
                content=content,
                headers=meta.get('headers', {}),
                fetched_at_utc=meta['fetched_at_utc'],
                content_sha256=meta['content_sha256'],
            )

        except Exception as e:
            logger.debug(f"Cache read error: {e}")
            return None

    def _save_to_cache(self, url: str, cached: CachedResponse):
        """Save response to cache with atomic writes."""
        if not self.config.cache_enabled:
            return

        content_file, meta_file = self._get_cache_paths(url)

        try:
            # Write content atomically
            with tempfile.NamedTemporaryFile(
                    dir=self.cache_dir / 'content',
                    delete=False,
                    mode='wb'
            ) as tmp:
                tmp.write(cached.content)
                tmp_content = tmp.name

            shutil.move(tmp_content, content_file)

            # Write metadata atomically
            with tempfile.NamedTemporaryFile(
                    dir=self.cache_dir / 'metadata',
                    delete=False,
                    mode='w',
                    suffix='.json'
            ) as tmp:
                json.dump(cached.to_dict(), tmp, indent=2)
                tmp_meta = tmp.name

            shutil.move(tmp_meta, meta_file)

            logger.debug(f"Cached: {url}")

        except Exception as e:
            logger.warning(f"Cache write error: {e}")
            # Clean up partial files
            for f in [tmp_content, tmp_meta]:
                try:
                    if f and os.path.exists(f):
                        os.unlink(f)
                except:
                    pass

    def _wait_for_rate_limit(self):
        """Wait to maintain rate limit with jitter."""
        elapsed = time.time() - self.last_request_time
        target_interval = 1.0 / self.config.requests_per_second

        # Add jitter
        jitter = random.uniform(-self.config.jitter_factor, self.config.jitter_factor)
        interval = max(self.config.min_interval, target_interval * (1 + jitter))

        if elapsed < interval:
            sleep_time = interval - elapsed
            logger.debug(f"Rate limit wait: {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate backoff delay with jitter."""
        delay = min(
            self.config.base_delay * (2 ** attempt),
            self.config.max_delay
        )
        # Add jitter to backoff
        jitter = random.uniform(0, self.config.jitter_factor * delay)
        return delay + jitter

    def get(
            self,
            session: requests.Session,
            url: str,
            ttl_seconds: int = None,
            skip_cache: bool = False,
    ) -> requests.Response:
        """
        GET request with rate limiting, retries, and caching.

        Args:
            session: requests.Session with appropriate headers
            url: URL to fetch
            ttl_seconds: Cache TTL override (None = use default)
            skip_cache: Force fresh fetch

        Returns:
            requests.Response (or raises exception after max retries)
        """
        # Check cache first
        if not skip_cache:
            cached = self._get_cached(url, ttl_seconds)
            if cached:
                # Return a mock response object
                response = requests.Response()
                response.status_code = cached.status_code
                response._content = cached.content
                response.headers.update(cached.headers)
                response.url = cached.url
                return response

        self._cache_misses += 1

        # Retry loop
        last_error = None
        for attempt in range(self.config.max_retries):
            self._wait_for_rate_limit()
            self._request_count += 1

            try:
                response = session.get(
                    url,
                    timeout=(self.config.connect_timeout, self.config.read_timeout)
                )

                if response.status_code == 200:
                    # Cache successful response
                    cached = CachedResponse.from_response(url, response)
                    self._save_to_cache(url, cached)
                    return response

                elif response.status_code in (429, 503):
                    # Rate limited or service unavailable
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        f"Rate limited ({response.status_code}) on attempt {attempt + 1}, "
                        f"backing off {delay:.1f}s: {url}"
                    )
                    time.sleep(delay)
                    continue

                elif response.status_code >= 500:
                    # Server error
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        f"Server error ({response.status_code}) on attempt {attempt + 1}, "
                        f"retry in {delay:.1f}s: {url}"
                    )
                    time.sleep(delay)
                    continue

                elif response.status_code == 404:
                    # Not found - don't retry
                    logger.warning(f"Not found (404): {url}")
                    response.raise_for_status()

                else:
                    # Other client error - don't retry
                    logger.error(f"Client error ({response.status_code}): {url}")
                    response.raise_for_status()

            except requests.exceptions.Timeout as e:
                delay = self._calculate_backoff(attempt)
                logger.warning(
                    f"Timeout on attempt {attempt + 1}, retry in {delay:.1f}s: {url}"
                )
                last_error = e
                time.sleep(delay)

            except requests.exceptions.ConnectionError as e:
                delay = self._calculate_backoff(attempt)
                logger.warning(
                    f"Connection error on attempt {attempt + 1}, retry in {delay:.1f}s: {url}"
                )
                last_error = e
                time.sleep(delay)

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
                raise

        # All retries exhausted
        error_msg = f"Failed after {self.config.max_retries} attempts: {url}"
        if last_error:
            error_msg += f" (last error: {last_error})"
        raise requests.exceptions.RequestException(error_msg)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            'requests': self._request_count,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0 else 0
            ),
        }

    def clear_cache(self, older_than_days: int = None):
        """
        Clear cache files.

        Args:
            older_than_days: Only clear files older than this (None = clear all)
        """
        if not self.config.cache_enabled:
            return

        cutoff = None
        if older_than_days is not None:
            cutoff = datetime.utcnow() - timedelta(days=older_than_days)

        cleared = 0
        for subdir in ['content', 'metadata']:
            dir_path = self.cache_dir / subdir
            if not dir_path.exists():
                continue

            for file_path in dir_path.iterdir():
                if cutoff:
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mtime >= cutoff:
                        continue

                try:
                    file_path.unlink()
                    cleared += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")

        logger.info(f"Cleared {cleared} cache files")
        return cleared


# Convenience function for creating a configured session
def create_sec_session(user_agent: str = None) -> Tuple[requests.Session, RateLimiter]:
    """
    Create a session and rate limiter configured for SEC EDGAR.

    Args:
        user_agent: SEC-compliant User-Agent string

    Returns:
        (session, rate_limiter)
    """
    if user_agent is None:
        user_agent = os.getenv(
            'SEC_USER_AGENT',
            'HHResearch/1.0 (contact@example.com)'
        )

    session = requests.Session()
    session.headers.update({
        'User-Agent': user_agent,
        'Accept-Encoding': 'gzip, deflate',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    })

    limiter = RateLimiter(RateLimiterConfig.for_sec())

    return session, limiter


if __name__ == "__main__":
    # Test the rate limiter
    print("Testing rate limiter...")

    session, limiter = create_sec_session()

    # Test with SEC company tickers endpoint
    url = "https://www.sec.gov/files/company_tickers.json"

    print(f"Fetching: {url}")
    response = limiter.get(session, url, ttl_seconds=3600)
    print(f"Status: {response.status_code}")
    print(f"Content length: {len(response.content)}")
    print(f"Stats: {limiter.get_stats()}")

    # Second request should be cached
    print("\nFetching again (should be cached)...")
    response = limiter.get(session, url, ttl_seconds=3600)
    print(f"Stats: {limiter.get_stats()}")