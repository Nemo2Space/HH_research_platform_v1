"""
Bond Market Intelligence Module - LIVE DATA ONLY

Fetches REAL, LIVE data from:
- FRED API: Fed funds rate, economic indicators
- Yahoo Finance: Treasury yields, bond ETFs
- Web Search: Latest Fed decisions, rate probabilities, news

NO HARDCODED DATA. Everything is fetched dynamically.

Location: src/analytics/bond_market_intelligence.py
Author: Alpha Research Platform
"""

import os
import json
import math
import requests
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import yfinance as yf

from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# MOVE INDEX FETCHER - Proper validation
# =============================================================================

def fetch_move_index() -> Tuple[Optional[float], str, str]:
    """
    Fetch MOVE Index (bond market volatility) with proper validation.

    Returns:
        Tuple of (move_value, regime_label, data_status)
        - move_value: The MOVE index value, or None if unavailable
        - regime_label: "LOW", "NORMAL", "ELEVATED", "HIGH", or "UNKNOWN"
        - data_status: "LIVE", "STALE", or "ERROR"
    """
    try:
        # Fetch from Yahoo Finance
        data = yf.download('^MOVE', period='10d', progress=False, auto_adjust=False)

        if data.empty:
            logger.warning("MOVE index: No data returned from Yahoo")
            return None, "UNKNOWN", "ERROR"

        # Get the most recent value
        latest_close = data['Close'].iloc[-1]
        move_value = float(latest_close.iloc[0]) if hasattr(latest_close, 'iloc') else float(latest_close)

        # Get the date of the most recent data
        latest_date = data.index[-1]
        if hasattr(latest_date, 'date'):
            latest_date = latest_date.date()

        # Check staleness (data should be within 3 trading days)
        days_old = (date.today() - latest_date).days
        if days_old > 5:
            logger.warning(f"MOVE index data is {days_old} days old - may be stale")
            data_status = "STALE"
        else:
            data_status = "LIVE"

        # Sanity checks
        if move_value < 20 or move_value > 250:
            logger.warning(f"MOVE index {move_value} outside reasonable range (20-250)")
            return None, "UNKNOWN", "ERROR"

        # Check for suspicious jumps (compare to previous day)
        if len(data) >= 2:
            prev_close = data['Close'].iloc[-2]
            prev_value = float(prev_close.iloc[0]) if hasattr(prev_close, 'iloc') else float(prev_close)
            pct_change = abs(move_value - prev_value) / prev_value * 100
            if pct_change > 30:
                logger.warning(f"MOVE index jumped {pct_change:.1f}% in one day - verify data")

        # Determine regime
        # Historical MOVE context:
        # - Below 60: Low volatility (calm markets)
        # - 60-80: Normal
        # - 80-100: Elevated
        # - 100-120: High
        # - Above 120: Crisis levels (e.g., March 2020 hit 163)
        if move_value < 60:
            regime = "LOW"
        elif move_value < 80:
            regime = "NORMAL"
        elif move_value < 100:
            regime = "ELEVATED"
        else:
            regime = "HIGH"

        logger.info(f"MOVE Index: {move_value:.2f} ({regime}) - {data_status}")
        return move_value, regime, data_status

    except Exception as e:
        logger.error(f"Error fetching MOVE index: {e}")
        return None, "UNKNOWN", "ERROR"


@dataclass
class FedPolicyState:
    """Current Federal Reserve policy state - LIVE DATA."""
    current_rate_lower: float = 0.0
    current_rate_upper: float = 0.0
    effective_rate: float = 0.0

    last_decision: str = ""  # CUT, HIKE, HOLD
    last_decision_date: str = ""
    last_decision_bps: int = 0

    # From web search
    recent_fed_news: List[str] = field(default_factory=list)
    rate_outlook: str = ""

    # Data source
    data_source: str = ""
    fetch_time: str = ""


@dataclass
class RateProbabilities:
    """Rate probabilities - from web search."""
    next_meeting_date: str = ""
    next_meeting_probs: Dict[str, float] = field(default_factory=dict)

    # From CME FedWatch via web search
    cuts_priced_in_2025: float = 0.0
    cuts_priced_in_2026: float = 0.0

    # Market expectations
    market_expectation_summary: str = ""

    data_source: str = ""


@dataclass
class YieldData:
    """Live yield data with audit/quality metadata."""
    yield_3m: float = 0.0
    yield_2y: float = 0.0
    yield_5y: float = 0.0
    yield_10y: float = 0.0
    yield_30y: float = 0.0

    # Spreads
    spread_10y_2y: float = 0.0
    spread_10y_3m: float = 0.0
    spread_10y_fed: float = 0.0

    # Term premium - None means cannot calculate without rate expectations data
    term_premium_10y: Optional[float] = None

    fetch_time: str = ""

    # Audit / quality metadata
    data_quality: str = "OK"  # OK | DEGRADED
    sources: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class EconomicData:
    """Economic indicators from FRED."""
    # Inflation
    cpi_yoy: float = 0.0
    cpi_date: str = ""
    pce_yoy: float = 0.0
    pce_date: str = ""

    # Employment
    unemployment_rate: float = 0.0
    unemployment_date: str = ""

    # Growth
    gdp_growth: float = 0.0
    gdp_date: str = ""

    data_source: str = "FRED"


@dataclass
class BondMarketIntelligence:
    """Complete bond market intelligence - ALL LIVE DATA."""
    timestamp: datetime

    fed_policy: FedPolicyState
    rate_probabilities: RateProbabilities
    yields: YieldData
    economic: EconomicData

    # MOVE Index (volatility)
    move_index: Optional[float] = None
    move_regime: str = "UNKNOWN"
    move_data_status: str = "UNKNOWN"

    # AI-ready context
    key_facts: List[str] = field(default_factory=list)
    trading_implications: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)

    ai_context: str = ""


class BondMarketIntelligenceGatherer:
    """Gathers LIVE bond market intelligence from multiple sources."""

    def __init__(self):
        self._cache = None
        self._cache_time = None
        self._cache_duration = timedelta(hours=2)

        # API keys
        self.fred_api_key = os.getenv('FRED_API_KEY', '')
        self.tool_server_url = os.getenv('TOOL_SERVER_URL', '')

        logger.info(f"BondMarketIntelligence: FRED={'✓' if self.fred_api_key else '✗'}, "
                    f"ToolServer={'✓' if self.tool_server_url else '✗'}")

    def gather_intelligence(self, force_refresh: bool = False) -> BondMarketIntelligence:
        """Gather ALL live bond market intelligence."""

        # Check cache
        if not force_refresh and self._cache and self._cache_time:
            if datetime.now() - self._cache_time < self._cache_duration:
                logger.debug("Using cached bond intelligence")
                return self._cache

        logger.info("Fetching LIVE bond market intelligence...")

        # Gather from all sources
        fed_policy = self._fetch_fed_policy()
        rate_probs = self._fetch_rate_probabilities()
        yields = self._fetch_yields()
        economic = self._fetch_economic_data()

        # Fetch MOVE Index (volatility) with validation
        move_value, move_regime, move_status = fetch_move_index()

        # Calculate spreads - DO NOT call this "term premium"
        # Term premium requires a model (e.g., NY Fed ACM, Kim-Wright)
        # This is just the YIELD SPREAD vs Fed funds
        if yields.yield_10y > 0 and fed_policy.effective_rate > 0:
            # This is the 10Y-Fed spread (NOT term premium)
            yields.spread_10y_fed = yields.yield_10y - fed_policy.effective_rate

            # NOTE: True term premium would require:
            # - NY Fed ACM term premium series
            # - FRED "Term Premium on a 10 Year Zero Coupon Bond" (Kim-Wright)
            # Set to None to avoid mislabeling
            yields.term_premium_10y = None  # Cannot calculate without model data

        # Build intelligence
        intel = BondMarketIntelligence(
            timestamp=datetime.now(),
            fed_policy=fed_policy,
            rate_probabilities=rate_probs,
            yields=yields,
            economic=economic,
            move_index=move_value,
            move_regime=move_regime,
            move_data_status=move_status,
        )

        # Generate insights
        intel.key_facts = self._generate_key_facts(intel)
        intel.trading_implications = self._generate_implications(intel)
        intel.risks = self._generate_risks(intel)
        intel.ai_context = self._build_ai_context(intel)

        # Cache
        self._cache = intel
        self._cache_time = datetime.now()

        return intel

    def _fetch_fed_policy(self) -> FedPolicyState:
        """Fetch current Fed policy from FRED and web search."""
        policy = FedPolicyState()
        policy.fetch_time = datetime.now().isoformat()

        # 1. Get Fed Funds TARGET RANGE from FRED (official series)
        if self.fred_api_key:
            try:
                # DFEDTARL = Lower bound of target range
                # DFEDTARU = Upper bound of target range
                # EFFR = Effective Federal Funds Rate

                base_url = "https://api.stlouisfed.org/fred/series/observations"

                # Fetch lower bound
                params_lower = {
                    'series_id': 'DFEDTARL',
                    'api_key': self.fred_api_key,
                    'file_type': 'json',
                    'limit': 5,
                    'sort_order': 'desc'
                }
                resp_lower = requests.get(base_url, params=params_lower, timeout=10)

                # Fetch upper bound
                params_upper = {
                    'series_id': 'DFEDTARU',
                    'api_key': self.fred_api_key,
                    'file_type': 'json',
                    'limit': 5,
                    'sort_order': 'desc'
                }
                resp_upper = requests.get(base_url, params=params_upper, timeout=10)

                # Fetch effective rate
                params_effr = {
                    'series_id': 'EFFR',
                    'api_key': self.fred_api_key,
                    'file_type': 'json',
                    'limit': 5,
                    'sort_order': 'desc'
                }
                resp_effr = requests.get(base_url, params=params_effr, timeout=10)

                # Parse responses
                if resp_lower.status_code == 200:
                    data = resp_lower.json()
                    if data.get('observations'):
                        # Get most recent non-empty value
                        for obs in data['observations']:
                            if obs.get('value') and obs['value'] != '.':
                                policy.current_rate_lower = float(obs['value'])
                                break

                if resp_upper.status_code == 200:
                    data = resp_upper.json()
                    if data.get('observations'):
                        for obs in data['observations']:
                            if obs.get('value') and obs['value'] != '.':
                                policy.current_rate_upper = float(obs['value'])
                                break

                if resp_effr.status_code == 200:
                    data = resp_effr.json()
                    if data.get('observations'):
                        for obs in data['observations']:
                            if obs.get('value') and obs['value'] != '.':
                                policy.effective_rate = float(obs['value'])
                                break

                # Validate data
                if policy.current_rate_lower > 0 and policy.current_rate_upper > 0:
                    policy.data_source = "FRED (DFEDTARL/DFEDTARU/EFFR)"
                    logger.info(f"FRED: Fed funds target = {policy.current_rate_lower:.2f}%-{policy.current_rate_upper:.2f}%, effective = {policy.effective_rate:.2f}%")
                elif policy.effective_rate > 0:
                    # Fallback: derive range from effective (less accurate)
                    policy.current_rate_upper = round(policy.effective_rate + 0.11, 2)  # EFFR typically ~11bps below upper
                    policy.current_rate_lower = policy.current_rate_upper - 0.25
                    policy.data_source = "FRED EFFR (range derived)"
                    logger.warning(f"FRED: Target range series unavailable, using EFFR-derived range")

            except Exception as e:
                logger.warning(f"FRED API error: {e}")

        # 2. Fallback: Yahoo Finance 3M T-Bill (UNRELIABLE - mark clearly)
        # NOTE: This is NOT the Fed Funds rate - it's a proxy that can be off by 10-20bps
        if policy.effective_rate == 0:
            try:
                data = yf.download('^IRX', period='5d', progress=False, auto_adjust=False)
                if not data.empty:
                    val = data['Close'].iloc[-1]
                    rate_3m = float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)
                    # WARNING: 3M T-bill is NOT Fed Funds - just a rough proxy
                    policy.effective_rate = round(rate_3m, 2)
                    policy.current_rate_upper = round(rate_3m + 0.15, 2)  # Very rough estimate
                    policy.current_rate_lower = policy.current_rate_upper - 0.25
                    policy.data_source = "⚠️ Yahoo Finance 3M T-Bill (PROXY - may be inaccurate)"
                    logger.warning(f"Using 3M T-Bill as Fed Funds proxy: {rate_3m:.2f}% - THIS IS NOT ACCURATE")
            except Exception as e:
                logger.warning(f"Yahoo Finance fallback error: {e}")

        # 3. Get latest Fed news via web search
        if self.tool_server_url:
            try:
                response = requests.post(
                    f"{self.tool_server_url}/search",
                    json={"query": "Federal Reserve interest rate decision latest FOMC", "max_results": 5},
                    timeout=15
                )
                if response.status_code == 200:
                    results = response.json().get('results', [])
                    for item in results[:3]:
                        title = item.get('title', '')
                        snippet = item.get('snippet', '')
                        policy.recent_fed_news.append(f"{title}: {snippet[:150]}")

                        # Try to extract decision info
                        text = (title + ' ' + snippet).lower()
                        if 'cut' in text and ('25' in text or 'basis' in text or 'quarter' in text):
                            if not policy.last_decision:
                                policy.last_decision = "CUT"
                                policy.last_decision_bps = -25
                        elif 'hike' in text or 'raise' in text:
                            policy.last_decision = "HIKE"
                            policy.last_decision_bps = 25
                        elif 'hold' in text or 'unchanged' in text or 'pause' in text:
                            policy.last_decision = "HOLD"
                            policy.last_decision_bps = 0

            except Exception as e:
                logger.warning(f"Web search for Fed news error: {e}")

        return policy

    def _fetch_rate_probabilities(self) -> RateProbabilities:
        """Fetch rate probabilities from web search (CME FedWatch)."""
        probs = RateProbabilities()

        if not self.tool_server_url:
            return probs

        try:
            # Search for CME FedWatch probabilities
            response = requests.post(
                f"{self.tool_server_url}/search",
                json={"query": "CME FedWatch Fed rate probability next meeting", "max_results": 5},
                timeout=15
            )

            if response.status_code == 200:
                results = response.json().get('results', [])

                for item in results:
                    snippet = item.get('snippet', '').lower()
                    title = item.get('title', '')

                    probs.market_expectation_summary += f"{title[:60]}... "

                    # Try to extract probabilities from text
                    import re

                    # Look for percentage patterns
                    # e.g., "85% chance of hold", "72% probability of cut"
                    hold_match = re.search(r'(\d+(?:\.\d+)?)\s*%?\s*(?:chance|probability|odds)?\s*(?:of\s+)?(?:hold|unchanged|no\s*change)', snippet)
                    cut_match = re.search(r'(\d+(?:\.\d+)?)\s*%?\s*(?:chance|probability|odds)?\s*(?:of\s+)?(?:cut|lower|reduction)', snippet)

                    if hold_match:
                        probs.next_meeting_probs['hold'] = float(hold_match.group(1)) / 100
                    if cut_match:
                        probs.next_meeting_probs['cut'] = float(cut_match.group(1)) / 100

                    # Look for basis points of cuts priced
                    bps_match = re.search(r'(\d+)\s*(?:basis\s*points?|bps)\s*(?:of\s+)?cuts?\s*(?:priced|expected|by)', snippet)
                    if bps_match:
                        probs.cuts_priced_in_2025 = float(bps_match.group(1))

                probs.data_source = "Web Search (CME FedWatch)"

        except Exception as e:
            logger.warning(f"Rate probabilities search error: {e}")

        return probs

    def _fetch_yields(self) -> YieldData:
        """Fetch current Treasury yields with robust handling.

        Preference order:
          1) FRED (official): DGS3MO, DGS2, DGS5, DGS10, DGS30
          2) Yahoo Finance fallback: ^IRX, ^FVX, ^TNX, ^TYX (with scaling sanity checks),
             plus 2YY=F for 2Y when available.

        Notes:
          - Yahoo's ^TNX/^TYX/^FVX are often quoted as "yield * 10" (e.g., 41.7 == 4.17%).
            We normalize any suspicious values.
          - If core yields cannot be fetched reliably, we mark data_quality=DEGRADED and
            keep yields at 0.0 rather than fabricating plausible numbers.
        """
        yields = YieldData()
        yields.fetch_time = datetime.now().isoformat()
        yields.data_quality = "OK"
        yields.sources = {}
        yields.warnings = []

        def _to_float(x: str) -> Optional[float]:
            try:
                v = float(x)
                if math.isnan(v) or math.isinf(v):
                    return None
                return v
            except Exception:
                return None

        def _normalize_yahoo(ticker: str, raw: Optional[float]) -> Optional[float]:
            """Normalize Yahoo yield data - they often quote as yield*10"""
            if raw is None:
                return None
            v = float(raw)
            # Common Yahoo yield tickers are in "index points" (yield * 10)
            if v > 20.0:
                v = v / 10.0
            # Basic sanity (yields outside 0-20% are almost certainly bad data)
            if v < 0.0 or v > 20.0:
                return None
            return round(v, 2)

        def _fred_latest(series_id: str) -> Tuple[Optional[float], Optional[str]]:
            if not self.fred_api_key:
                return None, None
            try:
                base_url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    "series_id": series_id,
                    "api_key": self.fred_api_key,
                    "file_type": "json",
                    "limit": 20,
                    "sort_order": "desc",
                }
                resp = requests.get(base_url, params=params, timeout=10)
                if resp.status_code != 200:
                    return None, None
                data = resp.json()
                obs = data.get("observations") or []
                for o in obs:
                    val = o.get("value")
                    if val in (None, "", "."):
                        continue
                    v = _to_float(val)
                    if v is None:
                        continue
                    # FRED yields are already in percent
                    if 0.0 <= v <= 20.0:
                        return round(v, 2), o.get("date")
                return None, None
            except Exception:
                return None, None

        # --- 1) FRED (preferred) ---
        fred_map = {
            "DGS3MO": "yield_3m",
            "DGS2": "yield_2y",
            "DGS5": "yield_5y",
            "DGS10": "yield_10y",
            "DGS30": "yield_30y",
        }

        fred_ok = False
        if self.fred_api_key:
            for series_id, attr in fred_map.items():
                v, d = _fred_latest(series_id)
                if v is not None:
                    setattr(yields, attr, v)
                    yields.sources[attr] = f"FRED:{series_id} ({d})"
                    fred_ok = True
                else:
                    yields.warnings.append(f"FRED {series_id} unavailable.")

        # --- 2) Yahoo fallback (only for missing fields) ---
        try:
            yahoo_tickers = {
                "^IRX": "yield_3m",
                "^FVX": "yield_5y",
                "^TNX": "yield_10y",
                "^TYX": "yield_30y",
            }

            need_yahoo = (not fred_ok) or any(getattr(yields, a) == 0.0 for a in yahoo_tickers.values())
            if need_yahoo:
                data = yf.download(list(yahoo_tickers.keys()), period="5d", progress=False, auto_adjust=False)
                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        latest = data["Close"].iloc[-1]
                        for tkr, attr in yahoo_tickers.items():
                            if getattr(yields, attr) != 0.0:
                                continue
                            if tkr in latest.index:
                                v = _normalize_yahoo(tkr, _to_float(str(latest[tkr])))
                                if v is not None:
                                    setattr(yields, attr, v)
                                    yields.sources[attr] = f"Yahoo:{tkr}"
        except Exception as e:
            yields.warnings.append(f"Yahoo yields fetch failed: {e}")

        # --- 3) 2Y yield: prefer FRED; only then try Yahoo futures; finally estimate ---
        if yields.yield_2y == 0.0:
            try:
                data_2y = yf.download("2YY=F", period="5d", progress=False, auto_adjust=False)
                if not data_2y.empty:
                    v = _to_float(str(data_2y["Close"].iloc[-1]))
                    v = _normalize_yahoo("2YY=F", v)
                    if v is not None:
                        yields.yield_2y = v
                        yields.sources["yield_2y"] = "Yahoo:2YY=F"
            except Exception as e:
                yields.warnings.append(f"2YY=F fetch failed: {e}")

        if yields.yield_2y == 0.0 and yields.yield_3m > 0.0 and yields.yield_5y > 0.0:
            # Last-resort estimate (documented)
            yields.yield_2y = round((yields.yield_3m * 0.4 + yields.yield_5y * 0.6), 2)
            yields.sources["yield_2y"] = "Estimated(0.4*3M + 0.6*5Y)"
            yields.warnings.append("2Y yield estimated from 3M and 5Y (no direct source).")

        # --- Spreads ---
        if yields.yield_10y > 0.0 and yields.yield_2y > 0.0:
            yields.spread_10y_2y = round(yields.yield_10y - yields.yield_2y, 2)
        if yields.yield_10y > 0.0 and yields.yield_3m > 0.0:
            yields.spread_10y_3m = round(yields.yield_10y - yields.yield_3m, 2)

        # Data quality
        core_ok = (yields.yield_10y > 0.0 and yields.yield_30y > 0.0 and yields.yield_2y > 0.0)
        if not core_ok:
            yields.data_quality = "DEGRADED"
            yields.warnings.append("Missing one or more core yields (2Y/10Y/30Y).")

        logger.info(
            f"Yields[{yields.data_quality}]: 3M={yields.yield_3m}%, 2Y={yields.yield_2y}%, "
            f"5Y={yields.yield_5y}%, 10Y={yields.yield_10y}%, 30Y={yields.yield_30y}%, "
            f"10Y-2Y={yields.spread_10y_2y}%"
        )
        return yields

    def _fetch_economic_data(self) -> EconomicData:
        """Fetch economic indicators from FRED."""
        econ = EconomicData()

        if not self.fred_api_key:
            return econ

        # FRED series to fetch
        series = {
            'CPIAUCSL': ('cpi_yoy', 'cpi_date'),      # CPI
            'PCEPI': ('pce_yoy', 'pce_date'),          # PCE
            'UNRATE': ('unemployment_rate', 'unemployment_date'),  # Unemployment
        }

        for series_id, (value_attr, date_attr) in series.items():
            try:
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    'series_id': series_id,
                    'api_key': self.fred_api_key,
                    'file_type': 'json',
                    'limit': 13,  # Get enough for YoY calculation
                    'sort_order': 'desc'
                }

                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    obs = data.get('observations', [])

                    if obs:
                        latest = float(obs[0]['value']) if obs[0]['value'] != '.' else 0
                        latest_date = obs[0]['date']

                        # For CPI and PCE, calculate YoY change
                        if series_id in ['CPIAUCSL', 'PCEPI'] and len(obs) >= 13:
                            year_ago = float(obs[12]['value']) if obs[12]['value'] != '.' else latest
                            if year_ago > 0:
                                yoy_change = ((latest / year_ago) - 1) * 100
                                setattr(econ, value_attr, round(yoy_change, 1))
                        else:
                            setattr(econ, value_attr, round(latest, 1))

                        setattr(econ, date_attr, latest_date)

            except Exception as e:
                logger.debug(f"FRED {series_id} error: {e}")

        return econ

    def _generate_key_facts(self, intel: BondMarketIntelligence) -> List[str]:
        """Generate key facts from live data."""
        facts = []

        # Fed rate fact
        if intel.fed_policy.effective_rate > 0:
            facts.append(
                f"Fed funds rate: {intel.fed_policy.current_rate_lower:.2f}%-{intel.fed_policy.current_rate_upper:.2f}% "
                f"(effective: {intel.fed_policy.effective_rate:.2f}%) [Source: {intel.fed_policy.data_source}]"
            )

        # Yield facts
        if intel.yields.yield_10y > 0:
            facts.append(f"10-Year Treasury: {intel.yields.yield_10y:.2f}%")
            facts.append(f"30-Year Treasury: {intel.yields.yield_30y:.2f}%")

            if intel.yields.spread_10y_2y < 0:
                facts.append(f"⚠️ Yield curve INVERTED: 10Y-2Y spread = {intel.yields.spread_10y_2y:.2f}%")
            else:
                facts.append(f"Yield curve normal: 10Y-2Y spread = +{intel.yields.spread_10y_2y:.2f}%")

        # Term premium - only show if we have data
        if intel.yields.term_premium_10y is not None:
            if intel.yields.term_premium_10y > 0.3:
                facts.append(f"Term premium ELEVATED: +{intel.yields.term_premium_10y:.2f}% - long bonds may offer value")
            elif intel.yields.term_premium_10y < -0.2:
                facts.append(f"Term premium COMPRESSED: {intel.yields.term_premium_10y:.2f}% - flight to safety")
        else:
            # Use 10Y-Fed spread as proxy indicator
            if intel.yields.spread_10y_fed > 0.5:
                facts.append(f"10Y trading {intel.yields.spread_10y_fed:.2f}% ABOVE Fed funds - may offer value")
            elif intel.yields.spread_10y_fed < -0.3:
                facts.append(f"10Y trading {abs(intel.yields.spread_10y_fed):.2f}% BELOW Fed funds - cuts heavily priced")

        # Economic facts
        if intel.economic.cpi_yoy > 0:
            facts.append(f"CPI inflation: {intel.economic.cpi_yoy:.1f}% YoY (as of {intel.economic.cpi_date})")

        if intel.economic.unemployment_rate > 0:
            facts.append(f"Unemployment: {intel.economic.unemployment_rate:.1f}% (as of {intel.economic.unemployment_date})")

        return facts

    def _generate_implications(self, intel: BondMarketIntelligence) -> List[str]:
        """Generate trading implications."""
        implications = []

        # Based on yield levels
        if intel.yields.yield_10y > 4.5:
            implications.append(
                f"10Y at {intel.yields.yield_10y:.2f}% - historically attractive for long bonds. "
                f"Consider TLT/ZROZ on further weakness."
            )
        elif intel.yields.yield_10y < 3.5:
            implications.append(
                f"10Y at {intel.yields.yield_10y:.2f}% - yields compressed. "
                f"Long bonds less attractive at current levels."
            )
        else:
            implications.append(
                f"10Y at {intel.yields.yield_10y:.2f}% - moderate yield level. "
                f"Evaluate based on rate outlook."
            )

        # Based on spread to Fed funds
        if intel.yields.spread_10y_fed > 0.5:
            implications.append(
                f"10Y trading {intel.yields.spread_10y_fed:.2f}% ABOVE Fed funds. "
                f"If Fed cuts materialize, long bonds could rally."
            )
        elif intel.yields.spread_10y_fed < 0:
            implications.append(
                f"10Y trading {abs(intel.yields.spread_10y_fed):.2f}% BELOW Fed funds. "
                f"Rate cuts heavily priced in - limited upside unless more cuts expected."
            )
        else:
            implications.append(
                f"10Y close to Fed funds (spread: {intel.yields.spread_10y_fed:+.2f}%). "
                f"Some cuts priced in."
            )

        # Based on curve shape
        if intel.yields.spread_10y_2y < -0.3:
            implications.append(
                "Deeply inverted curve suggests recession fears. "
                "Long bonds may rally on flight to safety."
            )
        elif intel.yields.spread_10y_2y > 0.5:
            implications.append(
                f"Steep curve ({intel.yields.spread_10y_2y:+.2f}%). "
                f"Long bonds offer yield pickup but more rate risk."
            )

        return implications

    def _generate_risks(self, intel: BondMarketIntelligence) -> List[str]:
        """Generate risks."""
        risks = []

        # Inflation risk
        if intel.economic.cpi_yoy > 3.0:
            risks.append(
                f"Inflation elevated at {intel.economic.cpi_yoy:.1f}% - "
                f"Fed may pause/reverse cuts if inflation re-accelerates."
            )
        elif intel.economic.cpi_yoy > 2.5:
            risks.append(
                f"Inflation at {intel.economic.cpi_yoy:.1f}% - above Fed's 2% target. "
                f"Watch for sticky inflation risk."
            )

        # Duration risk
        if intel.yields.yield_30y > 4.5:
            risks.append(
                f"30Y at {intel.yields.yield_30y:.2f}% - ultra-long bonds (ZROZ, EDV) "
                f"highly sensitive to rate moves. Limit position size."
            )

        # If yields very low
        if intel.yields.yield_10y < 3.5:
            risks.append(
                f"10Y yield at {intel.yields.yield_10y:.2f}% - limited room for further rally. "
                f"Risk/reward less favorable for new long positions."
            )

        return risks

    def _build_ai_context(self, intel: BondMarketIntelligence) -> str:
        """Build comprehensive AI context from LIVE data."""

        today = datetime.now().strftime('%Y-%m-%d')

        context = f"""
{'='*70}
🏦 BOND MARKET INTELLIGENCE - LIVE DATA
{'='*70}
Generated: {intel.timestamp.strftime('%Y-%m-%d %H:%M')}
TODAY'S DATE: {today}
Data Sources: FRED API, Yahoo Finance, Web Search

⚠️ IMPORTANT: Use ONLY the data below. Do NOT use training data for rates/yields.

{'='*70}
📊 CURRENT FED POLICY (LIVE from {intel.fed_policy.data_source or 'FRED/YF'})
{'='*70}
Fed Funds Target Range: {intel.fed_policy.current_rate_lower:.2f}% - {intel.fed_policy.current_rate_upper:.2f}%
Effective Fed Funds Rate: {intel.fed_policy.effective_rate:.2f}%
Data fetched: {intel.fed_policy.fetch_time}
"""

        if intel.fed_policy.last_decision:
            context += f"""
Last Decision: {intel.fed_policy.last_decision} ({intel.fed_policy.last_decision_bps:+d}bps)
"""

        if intel.fed_policy.recent_fed_news:
            context += "\nLatest Fed News (from web search):\n"
            for news in intel.fed_policy.recent_fed_news[:3]:
                context += f"  • {news[:200]}...\n"

        context += f"""
{'='*70}
📈 TREASURY YIELDS (LIVE from Yahoo Finance)
{'='*70}
Fetched: {intel.yields.fetch_time}

3-Month T-Bill:  {intel.yields.yield_3m:.2f}%
2-Year Note:     {intel.yields.yield_2y:.2f}%
5-Year Note:     {intel.yields.yield_5y:.2f}%
10-Year Note:    {intel.yields.yield_10y:.2f}%
30-Year Bond:    {intel.yields.yield_30y:.2f}%

YIELD CURVE ANALYSIS:
  10Y - 2Y Spread: {intel.yields.spread_10y_2y:+.2f}% {"⚠️ INVERTED" if intel.yields.spread_10y_2y < 0 else "(Normal)"}
  10Y - 3M Spread: {intel.yields.spread_10y_3m:+.2f}%
  10Y - Fed Funds Spread: {intel.yields.spread_10y_fed:+.2f}%

POLICY SPREAD ANALYSIS (NOT term premium):
  10Y vs Effective Fed Funds: {intel.yields.spread_10y_fed:+.2f}%
  Note: This is a YIELD SPREAD, not term premium.
  True term premium requires NY Fed ACM or Kim-Wright model data.
  Interpretation: {"10Y trading ABOVE Fed = may offer value if cuts materialize" if intel.yields.spread_10y_fed > 0.3 else "10Y near/below Fed = rate cuts largely priced in" if intel.yields.spread_10y_fed < 0 else "10Y modestly above Fed"}
"""

        if intel.rate_probabilities.market_expectation_summary:
            context += f"""
{'='*70}
📊 MARKET RATE EXPECTATIONS (from Web Search)
{'='*70}
{intel.rate_probabilities.market_expectation_summary[:500]}
"""
            if intel.rate_probabilities.next_meeting_probs:
                context += "\nExtracted Probabilities:\n"
                for action, prob in intel.rate_probabilities.next_meeting_probs.items():
                    context += f"  • {action.upper()}: {prob*100:.0f}%\n"
            if intel.rate_probabilities.cuts_priced_in_2025 > 0:
                context += f"  • Cuts priced for 2025: ~{intel.rate_probabilities.cuts_priced_in_2025:.0f}bps\n"

        if intel.economic.cpi_yoy > 0:
            context += f"""
{'='*70}
🌍 ECONOMIC INDICATORS (from FRED API)
{'='*70}
CPI Inflation: {intel.economic.cpi_yoy:.1f}% YoY (as of {intel.economic.cpi_date})
PCE Inflation: {intel.economic.pce_yoy:.1f}% YoY (as of {intel.economic.pce_date})
Unemployment:  {intel.economic.unemployment_rate:.1f}% (as of {intel.economic.unemployment_date})
"""

        # MOVE Index (volatility) - VALIDATED DATA
        context += f"""
{'='*70}
📊 MOVE INDEX - Bond Market Volatility (LIVE from Yahoo ^MOVE)
{'='*70}
"""
        if intel.move_index is not None:
            context += f"""MOVE Index: {intel.move_index:.2f}
Regime: {intel.move_regime}
Data Status: {intel.move_data_status}

Interpretation:
  - Below 60: LOW volatility (calm markets, favorable for positioning)
  - 60-80: NORMAL volatility
  - 80-100: ELEVATED volatility (caution warranted)
  - Above 100: HIGH volatility (crisis-level, reduce exposure)

Current Assessment: MOVE at {intel.move_index:.0f} indicates {intel.move_regime} rate volatility.
"""
        else:
            context += """MOVE Index: DATA UNAVAILABLE
⚠️ Unable to fetch MOVE index - volatility regime unknown.
"""

        context += f"""
{'='*70}
✅ KEY FACTS (from live data)
{'='*70}
"""
        for fact in intel.key_facts:
            context += f"• {fact}\n"

        context += f"""
{'='*70}
🎯 TRADING IMPLICATIONS
{'='*70}
"""
        for impl in intel.trading_implications:
            context += f"• {impl}\n"

        context += f"""
{'='*70}
⚠️ RISKS TO MONITOR
{'='*70}
"""
        for risk in intel.risks:
            context += f"• {risk}\n"

        context += f"""
{'='*70}
🤖 AI ADVISOR CRITICAL INSTRUCTIONS
{'='*70}
1. TODAY IS {today}. Use this date, not your training cutoff.

2. The CURRENT Fed funds rate is {intel.fed_policy.current_rate_lower:.2f}%-{intel.fed_policy.current_rate_upper:.2f}%
   (effective: {intel.fed_policy.effective_rate:.2f}%)
   This is LIVE DATA from {intel.fed_policy.data_source}. Trust this, NOT your training.

3. Current Treasury yields (LIVE):
   - 10Y: {intel.yields.yield_10y:.2f}%
   - 30Y: {intel.yields.yield_30y:.2f}%

4. The 10Y is trading {intel.yields.spread_10y_fed:+.2f}% vs Fed funds.
   {"Rate cuts are PRICED IN" if intel.yields.spread_10y_fed < 0 else "Some additional cuts may not be fully priced"}

5. When recommending TLT/ZROZ:
   - Reference the ACTUAL current yields above
   - Consider whether the 10Y-Fed spread suggests value
   - Use the term premium analysis
   - Factor in the economic data (CPI, unemployment)

DO NOT USE OUTDATED INFORMATION FROM YOUR TRAINING.
The live data above supersedes any conflicting training data.
"""

        return context


# Convenience functions
_gatherer = None


def get_bond_market_intelligence(force_refresh: bool = False) -> BondMarketIntelligence:
    """Get comprehensive bond market intelligence with LIVE data."""
    global _gatherer
    if _gatherer is None:
        _gatherer = BondMarketIntelligenceGatherer()
    return _gatherer.gather_intelligence(force_refresh)


def get_bond_intelligence_context() -> str:
    """Get AI-ready bond intelligence context."""
    intel = get_bond_market_intelligence()
    return intel.ai_context


if __name__ == "__main__":
    print("Bond Market Intelligence - LIVE DATA Test\n")
    print("=" * 70)

    intel = get_bond_market_intelligence()

    print(f"Timestamp: {intel.timestamp}")
    print(f"\nFed Funds: {intel.fed_policy.current_rate_lower:.2f}%-{intel.fed_policy.current_rate_upper:.2f}%")
    print(f"Effective Rate: {intel.fed_policy.effective_rate:.2f}%")
    print(f"Data Source: {intel.fed_policy.data_source}")

    print(f"\nYields (LIVE):")
    print(f"  2Y:  {intel.yields.yield_2y:.2f}%")
    print(f"  10Y: {intel.yields.yield_10y:.2f}%")
    print(f"  30Y: {intel.yields.yield_30y:.2f}%")
    print(f"  10Y-2Y Spread: {intel.yields.spread_10y_2y:+.2f}%")
    print(f"  10Y-Fed Spread: {intel.yields.spread_10y_fed:+.2f}%")

    print(f"\nEconomic (FRED):")
    print(f"  CPI: {intel.economic.cpi_yoy:.1f}% ({intel.economic.cpi_date})")
    print(f"  Unemployment: {intel.economic.unemployment_rate:.1f}% ({intel.economic.unemployment_date})")

    print("\n" + "=" * 70)
    print("KEY FACTS:")
    for fact in intel.key_facts:
        print(f"  • {fact}")

    print("\n" + "=" * 70)
    print("TRADING IMPLICATIONS:")
    for impl in intel.trading_implications:
        print(f"  • {impl}")