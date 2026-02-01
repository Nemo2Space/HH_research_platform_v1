"""
Bond/Rates Trading Module - WITH PREDICTION ENGINE INTEGRATION

Professional-grade bond trading signals based on:
- Treasury yield movements (10Y, 30Y)
- Fed policy expectations
- Inflation data (CPI, PCE, breakevens)
- Real yields
- Yield curve shape
- Risk regime integration
- Duration/Convexity analysis
- Economic Calendar Events (CPI, FOMC, Jobs)
- NEW: Dynamic yield forecasts from BondPredictionEngine
- NEW: Calendar effects (first day of year, month-end, ex-dividend, etc.)
- NEW: Seasonality patterns
- NEW: Cross-asset signals

Instruments: TLT, ZROZ, EDV, TMF, ZB futures

Author: Alpha Research Platform
Location: src/analytics/bond_signals_analytics.py
"""

import os
import json
import math
import requests
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

import pandas as pd
import numpy as np
import yfinance as yf

from src.utils.logging import get_logger
from src.db.connection import get_engine, get_connection

logger = get_logger(__name__)

# ============================================================
# Import Economic Calendar (optional)
# ============================================================
try:
    from src.analytics.economic_calendar import (
        EconomicCalendarFetcher,
        EconomicCalendar,
        EconomicEvent,
    )
    ECONOMIC_CALENDAR_AVAILABLE = True
except ImportError:
    ECONOMIC_CALENDAR_AVAILABLE = False
    logger.debug("Economic calendar module not available")

# ============================================================
# Import Bond Prediction Engine (NEW)
# ============================================================
try:
    from src.analytics.bond_prediction_engine import (
        BondPredictionEngine,
        get_bond_prediction_engine,
        YieldMomentum,
        CalendarAnalysis,
        CrossAssetSignals,
    )
    PREDICTION_ENGINE_AVAILABLE = True
except ImportError:
    PREDICTION_ENGINE_AVAILABLE = False
    logger.debug("Bond prediction engine not available")


class BondSignal(Enum):
    """Bond trading signal types."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class YieldTrend(Enum):
    """Yield trend direction."""
    FALLING_FAST = "FALLING_FAST"
    FALLING = "FALLING"
    STABLE = "STABLE"
    RISING = "RISING"
    RISING_FAST = "RISING_FAST"


class CurveShape(Enum):
    """Yield curve shape."""
    STEEP = "STEEP"
    NORMAL = "NORMAL"
    FLAT = "FLAT"
    INVERTED = "INVERTED"
    DEEPLY_INVERTED = "DEEPLY_INVERTED"


@dataclass
class YieldData:
    """Treasury yield data with audit/quality metadata."""
    date: date
    yield_2y: float
    yield_10y: float
    yield_30y: float
    spread_10y_2y: float
    spread_30y_10y: float
    real_yield_10y: float = None
    breakeven_10y: float = None

    # As-of dates for each yield (from FRED observation dates)
    yield_2y_date: str = ""
    yield_10y_date: str = ""
    yield_30y_date: str = ""
    spread_10y_2y_date: str = ""  # Date of T10Y2Y FRED series

    # Audit / quality metadata
    fetched_at: str = ""
    data_quality: str = "OK"  # OK | DEGRADED
    sources: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class BondInstrument:
    """Bond ETF/Future instrument data."""
    ticker: str
    name: str
    instrument_type: str
    duration: float
    expense_ratio: float = 0
    leverage: float = 1.0
    price: float = 0
    change_1d: float = 0
    change_5d: float = 0
    change_20d: float = 0
    volume: int = 0
    avg_volume: int = 0
    dv01: float = 0
    convexity: float = 0


@dataclass
class EconomicImpactOnBonds:
    """Economic calendar impact on bond trading."""
    has_high_impact_today: bool = False
    has_high_impact_this_week: bool = False
    today_events: List[Dict] = field(default_factory=list)
    today_summary: str = ""
    today_bond_impact: str = "NEUTRAL"
    today_score_adjustment: int = 0
    week_events: List[Dict] = field(default_factory=list)
    cpi_today: bool = False
    cpi_result: str = ""
    cpi_bond_impact: str = ""
    fomc_today: bool = False
    fomc_this_week: bool = False
    days_to_fomc: int = 999
    jobs_today: bool = False
    jobs_result: str = ""
    ai_analysis: str = ""
    ai_trading_implications: str = ""
    # NEW: Calendar effects
    calendar_alerts: List[str] = field(default_factory=list)
    is_first_trading_day_year: bool = False
    is_first_trading_day_month: bool = False
    is_month_end: bool = False
    is_quarter_end: bool = False
    seasonality_bias: str = "NEUTRAL"
    expected_volatility: str = "NORMAL"


@dataclass
class BondSignalResult:
    """Complete bond trading signal."""
    ticker: str
    signal: BondSignal
    score: int
    confidence: float
    current_price: float
    target_price: float
    stop_loss: float
    upside_pct: float
    downside_pct: float
    risk_reward: float
    yield_trend: YieldTrend
    yield_trend_score: int
    curve_shape: CurveShape
    curve_score: int
    fed_policy_score: int
    inflation_score: int
    regime_score: int
    technical_score: int
    economic_score: int = 50
    # NEW: Additional scores from prediction engine
    calendar_score: int = 50
    seasonality_score: int = 50
    cross_asset_score: int = 50
    yield_momentum_forecast_bps: Optional[float] = None
    economic_impact: Optional[EconomicImpactOnBonds] = None
    bull_factors: List[str] = field(default_factory=list)
    bear_factors: List[str] = field(default_factory=list)
    recommendation: str = ""
    analysis_date: datetime = field(default_factory=datetime.now)
    # NEW: Prediction engine data
    predicted_return_5d: Optional[float] = None
    risk_events_ahead: List[str] = field(default_factory=list)
    # NEW: Inflation data with sourcing
    inflation_data: Dict = field(default_factory=dict)


# Bond instruments
BOND_INSTRUMENTS = {
    'TLT': BondInstrument(
        ticker='TLT',
        name='iShares 20+ Year Treasury Bond ETF',
        instrument_type='ETF',
        duration=17.5,
        expense_ratio=0.15,
    ),
    'ZROZ': BondInstrument(
        ticker='ZROZ',
        name='PIMCO 25+ Year Zero Coupon Treasury',
        instrument_type='ETF',
        duration=27.0,
        expense_ratio=0.15,
    ),
    'EDV': BondInstrument(
        ticker='EDV',
        name='Vanguard Extended Duration Treasury',
        instrument_type='ETF',
        duration=24.5,
        expense_ratio=0.06,
    ),
    'TMF': BondInstrument(
        ticker='TMF',
        name='Direxion Daily 20+ Year Treasury Bull 3X',
        instrument_type='ETF',
        duration=17.5,
        expense_ratio=1.01,
        leverage=3.0,
    ),
    'TBT': BondInstrument(
        ticker='TBT',
        name='ProShares UltraShort 20+ Year Treasury',
        instrument_type='ETF',
        duration=17.5,
        expense_ratio=0.90,
        leverage=-2.0,
    ),
    'SHY': BondInstrument(
        ticker='SHY',
        name='iShares 1-3 Year Treasury Bond ETF',
        instrument_type='ETF',
        duration=1.9,
        expense_ratio=0.15,
    ),
    'IEF': BondInstrument(
        ticker='IEF',
        name='iShares 7-10 Year Treasury Bond ETF',
        instrument_type='ETF',
        duration=7.5,
        expense_ratio=0.15,
    ),
}


class BondSignalGenerator:
    """
    Generates trading signals for bond ETFs.
    With economic calendar integration and prediction engine.
    """

    def __init__(self):
        self._yields_cache = None
        self._yields_cache_time = None
        self._cache_duration = timedelta(minutes=15)
        self._economic_calendar = None
        self._prediction_engine = None

        if ECONOMIC_CALENDAR_AVAILABLE:
            try:
                self._economic_calendar = EconomicCalendarFetcher()
                logger.info("Economic calendar integration enabled")
            except Exception as e:
                logger.debug(f"Could not initialize economic calendar: {e}")

        # NEW: Initialize prediction engine
        if PREDICTION_ENGINE_AVAILABLE:
            try:
                self._prediction_engine = get_bond_prediction_engine()
                logger.info("Bond prediction engine integration enabled")
            except Exception as e:
                logger.debug(f"Could not initialize prediction engine: {e}")

    def _fred_latest(self, series_id: str, limit: int = 10) -> tuple:
        """
        Return (value, obs_date_iso) for the latest non-missing FRED observation.
        Returns (None, None) if unavailable.
        """
        try:
            fred_api_key = os.getenv("FRED_API_KEY", "") or getattr(self, "fred_api_key", "")
            if not fred_api_key:
                return None, None

            base_url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": fred_api_key,
                "file_type": "json",
                "limit": limit,
                "sort_order": "desc",
            }
            r = requests.get(base_url, params=params, timeout=12)
            if r.status_code != 200:
                return None, None

            obs = r.json().get("observations", []) or []
            for row in obs:
                v = row.get("value")
                d = row.get("date")
                if v and v != ".":
                    try:
                        return float(v), str(d)
                    except Exception:
                        continue
            return None, None
        except Exception:
            return None, None

    def _fred_history(self, series_id: str, days: int = 90) -> pd.DataFrame:
        """
        Return a DataFrame with columns: ['date','value'] for the last N calendar days.
        Missing values are dropped.
        """
        try:
            fred_api_key = os.getenv("FRED_API_KEY", "") or getattr(self, "fred_api_key", "")
            if not fred_api_key:
                return pd.DataFrame(columns=["date", "value"])

            end = date.today()
            start = end - timedelta(days=days)

            base_url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": fred_api_key,
                "file_type": "json",
                "observation_start": start.isoformat(),
                "observation_end": end.isoformat(),
                "sort_order": "asc",
            }
            r = requests.get(base_url, params=params, timeout=15)
            if r.status_code != 200:
                return pd.DataFrame(columns=["date", "value"])

            rows = r.json().get("observations", []) or []
            out = []
            for row in rows:
                v = row.get("value")
                d = row.get("date")
                if not v or v == ".":
                    continue
                try:
                    out.append((pd.to_datetime(d), float(v)))
                except Exception:
                    continue

            if not out:
                return pd.DataFrame(columns=["date", "value"])

            df = pd.DataFrame(out, columns=["date", "value"]).sort_values("date")
            return df
        except Exception:
            return pd.DataFrame(columns=["date", "value"])

    def get_treasury_yields(self) -> YieldData:
        """
        Get current treasury yields with strong guarantees:
        - Prefer FRED DGS* series for 2Y/10Y/30Y/3M/5Y
        - NEVER fabricate defaults (no "reasonable default yields")
        - If inputs missing: mark data_quality=DEGRADED and do not pretend precision
        - Attach audit metadata: fetched_at, data_quality, sources, warnings
        """
        # Cache (keep, but do not allow stale beyond duration)
        if self._yields_cache and self._yields_cache_time:
            if datetime.now() - self._yields_cache_time < self._cache_duration:
                return self._yields_cache

        warnings = []
        sources = {}
        fetched_at = datetime.now().isoformat()

        # 1) Try FRED first (official)
        y3m, d3m = self._fred_latest("DGS3MO")
        y2y, d2y = self._fred_latest("DGS2")
        y5y, d5y = self._fred_latest("DGS5")
        y10y, d10y = self._fred_latest("DGS10")
        y30y, d30y = self._fred_latest("DGS30")

        if y3m is not None:
            sources["yield_3m"] = f"FRED:DGS3MO:{d3m}"
        if y2y is not None:
            sources["yield_2y"] = f"FRED:DGS2:{d2y}"
        if y5y is not None:
            sources["yield_5y"] = f"FRED:DGS5:{d5y}"
        if y10y is not None:
            sources["yield_10y"] = f"FRED:DGS10:{d10y}"
        if y30y is not None:
            sources["yield_30y"] = f"FRED:DGS30:{d30y}"

        # 2) If FRED missing and ONLY then: Yahoo fallback for 3M/5Y/10Y/30Y (NOT 2Y)
        # IMPORTANT: Yahoo ^TNX/^TYX/^FVX are typically *10 (e.g., 41.9 => 4.19)
        def _yf_scale(val: float) -> float:
            try:
                v = float(val)
                return v / 10.0 if v > 20 else v
            except Exception:
                return None

        if (y10y is None) or (y30y is None) or (y3m is None) or (y5y is None):
            try:
                tickers = ["^IRX", "^FVX", "^TNX", "^TYX"]
                data = yf.download(tickers, period="7d", progress=False, auto_adjust=False)
                if not data.empty:
                    close = data["Close"].iloc[-1] if "Close" in data else None
                    if close is not None:
                        if y3m is None and "^IRX" in close.index and not pd.isna(close["^IRX"]):
                            y3m = _yf_scale(close["^IRX"])
                            sources["yield_3m"] = f"YF:^IRX:{str(data.index[-1].date())}"
                        if y5y is None and "^FVX" in close.index and not pd.isna(close["^FVX"]):
                            y5y = _yf_scale(close["^FVX"])
                            sources["yield_5y"] = f"YF:^FVX:{str(data.index[-1].date())}"
                        if y10y is None and "^TNX" in close.index and not pd.isna(close["^TNX"]):
                            y10y = _yf_scale(close["^TNX"])
                            sources["yield_10y"] = f"YF:^TNX:{str(data.index[-1].date())}"
                        if y30y is None and "^TYX" in close.index and not pd.isna(close["^TYX"]):
                            y30y = _yf_scale(close["^TYX"])
                            sources["yield_30y"] = f"YF:^TYX:{str(data.index[-1].date())}"
            except Exception as e:
                warnings.append(f"Yahoo yields fallback error: {e}")

        # 3) DO NOT estimate 2Y, DO NOT fabricate defaults
        data_quality = "OK"
        if y2y is None or y10y is None or y30y is None:
            data_quality = "DEGRADED"
            if y2y is None:
                warnings.append("Missing 2Y yield (DGS2). Spread 10Y-2Y may be unavailable.")
            if y10y is None:
                warnings.append("Missing 10Y yield (DGS10/^TNX).")
            if y30y is None:
                warnings.append("Missing 30Y yield (DGS30/^TYX).")

        # 4) Compute spreads only if inputs exist
        spread_10y_2y = (y10y - y2y) if (y10y is not None and y2y is not None) else np.nan
        spread_30y_10y = (y30y - y10y) if (y30y is not None and y10y is not None) else np.nan

        yields = YieldData(
            date=date.today(),
            yield_2y=float(y2y) if y2y is not None else np.nan,
            yield_10y=float(y10y) if y10y is not None else np.nan,
            yield_30y=float(y30y) if y30y is not None else np.nan,
            spread_10y_2y=float(spread_10y_2y) if not pd.isna(spread_10y_2y) else np.nan,
            spread_30y_10y=float(spread_30y_10y) if not pd.isna(spread_30y_10y) else np.nan,
        )

        # Attach audit metadata dynamically (dataclass is not slotted)
        yields.fetched_at = fetched_at
        yields.data_quality = data_quality
        yields.sources = sources
        yields.warnings = warnings

        self._yields_cache = yields
        self._yields_cache_time = datetime.now()
        return yields

    def get_treasury_yields(self) -> YieldData:
        """Get current Treasury yields (robust, audit-friendly).

        Preference order:
          1) FRED (official): DGS2, DGS10, DGS30 (+ optional DGS3MO)
          2) Yahoo fallback: ^TNX, ^TYX (normalized if quoted as *10)

        Output:
          - Uses NaN when values cannot be fetched (never fabricates plausible yields).
          - Adds data_quality/sources/warnings for downstream confidence gating.
        """
        if self._yields_cache and self._yields_cache_time:
            if datetime.now() - self._yields_cache_time < self._cache_duration:
                return self._yields_cache

        fetched_at = datetime.now().isoformat()
        warnings: List[str] = []
        sources: Dict[str, str] = {}
        data_quality = "OK"

        def _nan() -> float:
            return float("nan")

        def _is_finite(x: float) -> bool:
            return isinstance(x, (int, float)) and math.isfinite(float(x))

        def _to_float(x: str) -> Optional[float]:
            try:
                v = float(x)
                if not math.isfinite(v):
                    return None
                return v
            except Exception:
                return None

        def _normalize_yahoo(ticker: str, raw: Optional[float]) -> Optional[float]:
            """Normalize Yahoo yield data - they often quote as yield*10 (e.g., 41.7 == 4.17%)"""
            if raw is None:
                return None
            v = float(raw)
            # Yahoo's ^TNX/^TYX are frequently quoted as yield*10
            if v > 20.0:
                v = v / 10.0
            if v < 0.0 or v > 20.0:
                return None
            return round(v, 4)

        def _fred_latest(series_id: str, api_key: str) -> Tuple[Optional[float], Optional[str]]:
            try:
                base_url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    "series_id": series_id,
                    "api_key": api_key,
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
                    if 0.0 <= v <= 20.0:
                        return round(v, 4), o.get("date")
                return None, None
            except Exception:
                return None, None

        fred_api_key = os.getenv("FRED_API_KEY", "") or ""
        y2 = _nan()
        y10 = _nan()
        y30 = _nan()
        y3m = _nan()

        # --- FRED first (preferred, official data) ---
        if fred_api_key:
            v, d = _fred_latest("DGS2", fred_api_key)
            if v is not None:
                y2 = v
                sources["yield_2y"] = f"FRED:DGS2 ({d})"
            else:
                warnings.append("FRED DGS2 unavailable.")

            v, d = _fred_latest("DGS10", fred_api_key)
            if v is not None:
                y10 = v
                sources["yield_10y"] = f"FRED:DGS10 ({d})"
            else:
                warnings.append("FRED DGS10 unavailable.")

            v, d = _fred_latest("DGS30", fred_api_key)
            if v is not None:
                y30 = v
                sources["yield_30y"] = f"FRED:DGS30 ({d})"
            else:
                warnings.append("FRED DGS30 unavailable.")

            v, d = _fred_latest("DGS3MO", fred_api_key)
            if v is not None:
                y3m = v
                sources["yield_3m"] = f"FRED:DGS3MO ({d})"

        # --- Yahoo fallback (only for missing fields) ---
        try:
            need_yahoo = (not _is_finite(y10)) or (not _is_finite(y30))
            if need_yahoo:
                data = yf.download(["^TNX", "^TYX"], period="5d", progress=False, auto_adjust=False)
                if not data.empty and "Close" in data:
                    if isinstance(data.columns, pd.MultiIndex):
                        latest = data["Close"].iloc[-1]
                        if not _is_finite(y10) and "^TNX" in latest.index:
                            v = _normalize_yahoo("^TNX", _to_float(str(latest["^TNX"])))
                            if v is not None:
                                y10 = v
                                sources["yield_10y"] = "Yahoo:^TNX"
                        if not _is_finite(y30) and "^TYX" in latest.index:
                            v = _normalize_yahoo("^TYX", _to_float(str(latest["^TYX"])))
                            if v is not None:
                                y30 = v
                                sources["yield_30y"] = "Yahoo:^TYX"
        except Exception as e:
            warnings.append(f"Yahoo fallback failed: {e}")

        # --- 2Y fallback: try Yahoo futures if FRED unavailable ---
        if not _is_finite(y2):
            try:
                data_2y = yf.download("2YY=F", period="5d", progress=False, auto_adjust=False)
                if not data_2y.empty:
                    v = _to_float(str(data_2y["Close"].iloc[-1]))
                    v = _normalize_yahoo("2YY=F", v)
                    if v is not None:
                        y2 = v
                        sources["yield_2y"] = "Yahoo:2YY=F"
            except Exception as e:
                warnings.append(f"2YY=F fetch failed: {e}")

        # --- Last resort: estimate 2Y from 3M and 5Y ---
        if not _is_finite(y2) and _is_finite(y3m) and _is_finite(y10):
            # Documented estimation (not fabrication)
            y5_est = (y3m + y10) / 2  # rough 5Y estimate
            y2 = round((y3m * 0.4 + y5_est * 0.6), 4)
            sources["yield_2y"] = "Estimated(0.4*3M + 0.6*5Y_est)"
            warnings.append("2Y yield estimated from 3M and 10Y (no direct source).")

        # --- Spreads ---
        # PREFER T10Y2Y from FRED (official spread series) over calculated spread
        spread_10y_2y = _nan()
        spread_30y_10y = _nan()

        # Try to get official T10Y2Y spread from FRED
        if fred_api_key:
            v, d = _fred_latest("T10Y2Y", fred_api_key)
            if v is not None:
                spread_10y_2y = v
                sources["spread_10y_2y"] = f"FRED:T10Y2Y ({d})"

        # Fallback: calculate spread if T10Y2Y unavailable
        if not _is_finite(spread_10y_2y) and _is_finite(y10) and _is_finite(y2):
            spread_10y_2y = round(y10 - y2, 4)
            # Use the later of the two dates for calculated spread
            y10_date = sources.get("yield_10y", "").split("(")[-1].rstrip(")") if "yield_10y" in sources else ""
            y2_date = sources.get("yield_2y", "").split("(")[-1].rstrip(")") if "yield_2y" in sources else ""
            calc_date = max(y10_date, y2_date) if y10_date and y2_date else ""
            sources["spread_10y_2y"] = f"Calculated:DGS10-DGS2 ({calc_date})" if calc_date else "Calculated:DGS10-DGS2"
            warnings.append("Using calculated 10Y-2Y spread (T10Y2Y series unavailable).")

        if _is_finite(y30) and _is_finite(y10):
            spread_30y_10y = round(y30 - y10, 4)

        # Data quality classification
        if not (_is_finite(y2) and _is_finite(y10) and _is_finite(y30)):
            data_quality = "DEGRADED"
            warnings.append("Missing one or more core yields (2Y/10Y/30Y).")

        # Extract individual dates from sources
        def _extract_date_from_source(src: str) -> str:
            """Extract date like '2026-01-09' from 'FRED:DGS10 (2026-01-09)'"""
            try:
                if "(" in src and ")" in src:
                    return src.split("(")[-1].rstrip(")")
                return ""
            except Exception:
                return ""

        y2_date = _extract_date_from_source(sources.get("yield_2y", ""))
        y10_date = _extract_date_from_source(sources.get("yield_10y", ""))
        y30_date = _extract_date_from_source(sources.get("yield_30y", ""))
        spread_date = _extract_date_from_source(sources.get("spread_10y_2y", ""))

        yields = YieldData(
            date=date.today(),
            yield_2y=float(y2),
            yield_10y=float(y10),
            yield_30y=float(y30),
            spread_10y_2y=float(spread_10y_2y),
            spread_30y_10y=float(spread_30y_10y),
            yield_2y_date=y2_date,
            yield_10y_date=y10_date,
            yield_30y_date=y30_date,
            spread_10y_2y_date=spread_date,
            fetched_at=fetched_at,
            data_quality=data_quality,
            sources=sources,
            warnings=warnings,
        )

        self._yields_cache = yields
        self._yields_cache_time = datetime.now()

        logger.info(
            "Yields[%s]: 2Y=%s, 10Y=%s, 30Y=%s, 10Y-2Y=%s, sources=%s",
            yields.data_quality,
            yields.yield_2y,
            yields.yield_10y,
            yields.yield_30y,
            yields.spread_10y_2y,
            yields.sources,
        )
        return yields

    def _default_yields(self) -> YieldData:
        """Return a degraded YieldData placeholder (never fabricate plausible yields)."""
        return YieldData(
            date=date.today(),
            yield_2y=float("nan"),
            yield_10y=float("nan"),
            yield_30y=float("nan"),
            spread_10y_2y=float("nan"),
            spread_30y_10y=float("nan"),
            fetched_at=datetime.now().isoformat(),
            data_quality="DEGRADED",
            sources={},
            warnings=["Default yields used (no data)."],
        )

    def get_move_index(self) -> dict:
        """
        Fetch MOVE index from Yahoo Finance (^MOVE) with sanity checks.
        Returns dict with value, asof, source, quality, warning.

        The MOVE index measures Treasury volatility (like VIX for bonds).
        Typical range: 50-150. Spikes above 150 indicate high volatility.
        """
        out = {
            "value": None,
            "asof": None,
            "source": None,
            "quality": "MISSING",
            "warning": None
        }
        try:
            df = yf.download("^MOVE", period="7d", interval="1d", progress=False, auto_adjust=False)
            if df is None or df.empty:
                out["warning"] = "MOVE data empty from Yahoo"
                return out

            # Get the last valid close
            close_col = df["Close"] if "Close" in df.columns else df[("Close", "^MOVE")] if ("Close", "^MOVE") in df.columns else None
            if close_col is None or close_col.dropna().empty:
                out["warning"] = "MOVE close column not found or empty"
                return out

            val = float(close_col.dropna().iloc[-1])
            asof = str(df.index[-1].date())

            # Sanity check: MOVE rarely > 200 or < 30
            if val <= 30 or val > 250:
                out["value"] = val
                out["asof"] = asof
                out["source"] = "YahooFinance:^MOVE"
                out["quality"] = "SUSPICIOUS"
                out["warning"] = f"MOVE out of expected range (30-200): {val:.1f}"
                return out

            out["value"] = val
            out["asof"] = asof
            out["source"] = "YahooFinance:^MOVE"
            out["quality"] = "OK"
            return out
        except Exception as e:
            out["warning"] = f"MOVE fetch error: {e}"
            return out

    def get_yield_history(self, days: int = 60) -> pd.DataFrame:
        """Get historical yield data (official FRED series).

        Uses:
          - DGS2, DGS10, DGS30 (and optionally DGS3MO if needed elsewhere)
        Returns a DataFrame indexed by date with:
          - yield_2y, yield_10y, yield_30y, spread_10y_2y
        """
        fred_api_key = os.getenv("FRED_API_KEY", "") or ""
        if not fred_api_key:
            logger.warning("FRED_API_KEY not set; falling back to Yahoo for yield history.")
            return self._get_yield_history_yahoo(days)

        start_date = (date.today() - timedelta(days=max(5, days * 2))).isoformat()

        def _fred_series(series_id: str) -> pd.Series:
            base_url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": fred_api_key,
                "file_type": "json",
                "observation_start": start_date,
                "sort_order": "asc",
            }
            resp = requests.get(base_url, params=params, timeout=15)
            if resp.status_code != 200:
                raise RuntimeError(f"FRED {series_id} HTTP {resp.status_code}")
            data = resp.json()
            obs = data.get("observations") or []
            records = {}
            for o in obs:
                dt = o.get("date")
                val = o.get("value")
                if not dt or val in (None, "", "."):
                    continue
                try:
                    v = float(val)
                    if not math.isfinite(v):
                        continue
                    if 0.0 <= v <= 20.0:
                        records[dt] = v
                except Exception:
                    continue
            if not records:
                return pd.Series(dtype=float)
            s = pd.Series(records, dtype=float)
            s.index = pd.to_datetime(s.index)
            return s

        try:
            s2 = _fred_series("DGS2")
            s10 = _fred_series("DGS10")
            s30 = _fred_series("DGS30")
        except Exception as e:
            logger.error(f"Error fetching yield history from FRED: {e}")
            return self._get_yield_history_yahoo(days)

        df = pd.DataFrame(
            {
                "yield_2y": s2,
                "yield_10y": s10,
                "yield_30y": s30,
            }
        ).dropna()

        if df.empty:
            return df

        df["spread_10y_2y"] = df["yield_10y"] - df["yield_2y"]

        # Keep the most recent `days` rows (trading days)
        if len(df) > days:
            df = df.iloc[-days:]

        return df

    def _get_yield_history_yahoo(self, days: int = 60) -> pd.DataFrame:
        """Fallback: Get historical yield data from Yahoo (less reliable)."""
        try:
            tickers = ['^TNX', '^TYX']
            data = yf.download(tickers, period=f'{days}d', progress=False, auto_adjust=True)

            if data.empty:
                return pd.DataFrame()

            def _normalize(series: pd.Series) -> pd.Series:
                """Normalize Yahoo yields (they may be *10)"""
                return series.apply(lambda x: x / 10.0 if x > 20.0 else x)

            df = pd.DataFrame({
                'yield_10y': _normalize(data['Close']['^TNX']) if '^TNX' in data['Close'].columns else None,
                'yield_30y': _normalize(data['Close']['^TYX']) if '^TYX' in data['Close'].columns else None,
            })

            # Note: Yahoo doesn't have 2Y, so we can't calculate accurate spread
            df['yield_2y'] = float('nan')
            df['spread_10y_2y'] = float('nan')

            return df.dropna(subset=['yield_10y', 'yield_30y'])

        except Exception as e:
            logger.error(f"Error fetching yield history from Yahoo: {e}")
            return pd.DataFrame()

    def analyze_yield_trend(self, yield_hist: pd.DataFrame) -> Tuple[YieldTrend, int, Dict]:
        """Analyze yield trend direction."""
        if yield_hist.empty or len(yield_hist) < 5:
            return YieldTrend.STABLE, 50, {}

        y30 = yield_hist['yield_30y']
        change_5d = y30.iloc[-1] - y30.iloc[-5] if len(y30) >= 5 else 0
        change_20d = y30.iloc[-1] - y30.iloc[-20] if len(y30) >= 20 else change_5d

        if change_20d < -0.3:
            trend = YieldTrend.FALLING_FAST
            score = 80
        elif change_20d < -0.1:
            trend = YieldTrend.FALLING
            score = 65
        elif change_20d > 0.3:
            trend = YieldTrend.RISING_FAST
            score = 20
        elif change_20d > 0.1:
            trend = YieldTrend.RISING
            score = 35
        else:
            trend = YieldTrend.STABLE
            score = 50

        details = {
            'change_5d': change_5d,
            'change_20d': change_20d,
            'current': y30.iloc[-1],
            'trend': trend.value,
        }

        return trend, score, details

    def analyze_curve_shape(self, yields: YieldData) -> Tuple[CurveShape, int, Dict]:
        """Analyze yield curve shape."""
        spread = yields.spread_10y_2y

        if spread < -0.5:
            shape = CurveShape.DEEPLY_INVERTED
            score = 85
            interpretation = "Strongly bullish for bonds - recession signal"
        elif spread < 0:
            shape = CurveShape.INVERTED
            score = 70
            interpretation = "Bullish for bonds - Fed pivot expected"
        elif spread < 0.5:
            shape = CurveShape.FLAT
            score = 55
            interpretation = "Neutral to slightly bullish"
        elif spread < 1.5:
            shape = CurveShape.NORMAL
            score = 45
            interpretation = "Neutral curve shape"
        else:
            shape = CurveShape.STEEP
            score = 30
            interpretation = "Bearish for long bonds - growth expected"

        details = {
            'spread_10y_2y': spread,
            'interpretation': interpretation,
        }

        return shape, score, details

    def get_economic_impact(self) -> EconomicImpactOnBonds:
        """Get economic calendar impact on bonds - ENHANCED with prediction engine."""
        impact = EconomicImpactOnBonds()

        # NEW: Get calendar effects from prediction engine
        if self._prediction_engine:
            try:
                calendar = self._prediction_engine.analyze_calendar_effects()

                impact.calendar_alerts = calendar.alerts
                impact.is_first_trading_day_year = calendar.is_first_trading_day_year
                impact.is_first_trading_day_month = calendar.is_first_trading_day_month
                impact.is_month_end = calendar.is_last_trading_day_month
                impact.is_quarter_end = calendar.is_quarter_end
                impact.seasonality_bias = calendar.seasonality_bias
                impact.expected_volatility = calendar.expected_volatility
                impact.days_to_fomc = calendar.days_to_fomc
                impact.fomc_this_week = calendar.is_fomc_week
                impact.fomc_today = calendar.is_fomc_day

                # Add calendar alerts to summary
                if calendar.alerts:
                    impact.today_summary = " | ".join(calendar.alerts[:3])
                    impact.has_high_impact_today = True

            except Exception as e:
                logger.debug(f"Error getting calendar effects: {e}")

        if not self._economic_calendar:
            return impact

        try:
            # Get today's events
            events = self._economic_calendar.get_today_events()

            for event in events:
                name = event.get('name', '').lower()
                impact.today_events.append(event)

                # Check for CPI
                if 'cpi' in name or 'inflation' in name:
                    impact.cpi_today = True
                    impact.has_high_impact_today = True

                    actual = event.get('actual')
                    forecast = event.get('forecast')
                    if actual and forecast:
                        try:
                            if float(actual) < float(forecast):
                                impact.cpi_result = "MISS"
                                impact.cpi_bond_impact = "BULLISH"
                                impact.today_score_adjustment += 10
                            elif float(actual) > float(forecast):
                                impact.cpi_result = "BEAT"
                                impact.cpi_bond_impact = "BEARISH"
                                impact.today_score_adjustment -= 10
                            else:
                                impact.cpi_result = "IN-LINE"
                        except:
                            pass

                # Check for FOMC
                if 'fomc' in name or 'fed' in name:
                    impact.fomc_today = True
                    impact.fomc_this_week = True
                    impact.days_to_fomc = 0
                    impact.has_high_impact_today = True

                # Check for jobs
                if 'payroll' in name or 'employment' in name or 'jobs' in name or 'claims' in name:
                    impact.jobs_today = True
                    impact.has_high_impact_today = True

                    actual = event.get('actual')
                    forecast = event.get('forecast')
                    if actual and forecast:
                        try:
                            # For jobless claims, lower is stronger economy
                            if 'claims' in name:
                                if float(actual) < float(forecast):
                                    impact.jobs_result = "BEAT (lower claims)"
                                    impact.today_score_adjustment -= 5  # Bearish for bonds
                                elif float(actual) > float(forecast):
                                    impact.jobs_result = "MISS (higher claims)"
                                    impact.today_score_adjustment += 5  # Bullish for bonds
                            else:
                                # For payrolls, higher is stronger economy
                                if float(actual) < float(forecast):
                                    impact.jobs_result = "MISS"
                                    impact.today_score_adjustment += 5
                                elif float(actual) > float(forecast):
                                    impact.jobs_result = "BEAT"
                                    impact.today_score_adjustment -= 5
                        except:
                            pass

            # Build summary
            if impact.today_events and not impact.today_summary:
                event_names = [e.get('name', 'Unknown')[:30] for e in impact.today_events[:3]]
                impact.today_summary = f"Today: {', '.join(event_names)}"
            elif not impact.today_summary:
                impact.today_summary = "No high-impact events today"

        except Exception as e:
            logger.debug(f"Error getting economic impact: {e}")

        return impact

    def _get_yield_forecast(self, yield_trend: YieldTrend) -> Tuple[float, Optional[float]]:
        """
        Get dynamic yield forecast from prediction engine.
        Returns (expected_yield_change, forecast_bps).
        """
        # NEW: Use prediction engine for dynamic forecast
        if self._prediction_engine:
            try:
                yield_momentum = self._prediction_engine.analyze_yield_momentum()

                if yield_momentum.forecast_5d_bps:
                    # Convert bps to percent (e.g., 10bps = 0.10%)
                    expected_yield_change = yield_momentum.forecast_5d_bps / 100
                    return expected_yield_change, yield_momentum.forecast_5d_bps
            except Exception as e:
                logger.debug(f"Error getting yield forecast: {e}")

        # Fallback to static estimates based on trend
        if yield_trend in [YieldTrend.FALLING, YieldTrend.FALLING_FAST]:
            if yield_trend == YieldTrend.FALLING_FAST:
                return -0.30, -30.0
            return -0.15, -15.0
        elif yield_trend in [YieldTrend.RISING, YieldTrend.RISING_FAST]:
            if yield_trend == YieldTrend.RISING_FAST:
                return 0.30, 30.0
            return 0.15, 15.0

        return 0, 0

    def _get_fed_policy_score(self) -> int:
        """Estimate Fed policy stance score."""
        # Try to get from prediction engine
        if self._prediction_engine:
            try:
                rate_signals = self._prediction_engine.get_rate_signal_for_alpha()
                # Convert rate direction to score
                direction = rate_signals.get('rate_direction', 'STABLE')
                if 'FALLING' in direction:
                    return 65  # Dovish = bullish for bonds
                elif 'RISING' in direction:
                    return 35  # Hawkish = bearish for bonds
            except:
                pass
        return 50

    def _get_inflation_score(self) -> Tuple[int, Dict]:
        """Get inflation expectations score from FRED data.

        Uses:
        - CPIAUCSL: CPI All Urban Consumers (YoY calculation)
        - PCEPILFE: Core PCE Price Index (YoY calculation)

        Returns:
            Tuple of (score, metadata dict with values and dates)
        """
        fred_api_key = os.getenv("FRED_API_KEY", "") or ""

        inflation_data = {
            'cpi_yoy': None,
            'cpi_date': None,
            'core_pce_yoy': None,
            'core_pce_date': None,
            'data_available': False,
        }

        if not fred_api_key:
            logger.warning("FRED_API_KEY not set - inflation score unavailable")
            return 50, inflation_data

        def _fred_yoy(series_id: str) -> Tuple[Optional[float], Optional[str]]:
            """Fetch series and compute YoY change."""
            try:
                base_url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    "series_id": series_id,
                    "api_key": fred_api_key,
                    "file_type": "json",
                    "limit": 15,  # Get ~15 months for YoY
                    "sort_order": "desc",
                }
                resp = requests.get(base_url, params=params, timeout=10)
                if resp.status_code != 200:
                    return None, None
                data = resp.json()
                obs = data.get("observations") or []

                # Need at least 13 observations for YoY
                values = []
                for o in obs:
                    val = o.get("value")
                    if val in (None, "", "."):
                        continue
                    try:
                        v = float(val)
                        if math.isfinite(v):
                            values.append((o.get("date"), v))
                    except:
                        continue

                if len(values) < 13:
                    return None, None

                # Most recent vs 12 months ago
                latest_date, latest_val = values[0]
                _, year_ago_val = values[12]

                if year_ago_val > 0:
                    yoy = ((latest_val / year_ago_val) - 1) * 100
                    return round(yoy, 2), latest_date
                return None, None
            except Exception as e:
                logger.warning(f"FRED {series_id} fetch error: {e}")
                return None, None

        # Fetch CPI YoY
        cpi_yoy, cpi_date = _fred_yoy("CPIAUCSL")
        if cpi_yoy is not None:
            inflation_data['cpi_yoy'] = cpi_yoy
            inflation_data['cpi_date'] = cpi_date
            inflation_data['data_available'] = True

        # Fetch Core PCE YoY
        pce_yoy, pce_date = _fred_yoy("PCEPILFE")
        if pce_yoy is not None:
            inflation_data['core_pce_yoy'] = pce_yoy
            inflation_data['core_pce_date'] = pce_date
            inflation_data['data_available'] = True

        # Score based on inflation level (if available)
        # Lower inflation = more bullish for bonds (higher score)
        score = 50  # Default neutral
        if cpi_yoy is not None:
            if cpi_yoy < 2.0:
                score = 70  # Bullish - below target
            elif cpi_yoy < 2.5:
                score = 60  # Moderately bullish
            elif cpi_yoy < 3.0:
                score = 50  # Neutral
            elif cpi_yoy < 4.0:
                score = 40  # Moderately bearish
            else:
                score = 30  # Bearish - high inflation

        return score, inflation_data

    def _get_regime_score(self) -> int:
        """Get macro regime score for bonds."""
        # First try prediction engine for cross-asset
        if self._prediction_engine:
            try:
                cross_asset = self._prediction_engine.analyze_cross_assets()
                return cross_asset.cross_asset_score
            except:
                pass

        # Fallback to macro regime module
        try:
            from src.analytics.macro_regime import get_current_regime
            regime = get_current_regime()

            # Risk-off regimes are bullish for bonds
            if regime and hasattr(regime, 'name'):
                if 'RISK_OFF' in regime.name.upper():
                    return 70
                elif 'RISK_ON' in regime.name.upper():
                    return 35
        except:
            pass
        return 50

    def _get_technical_score(self, ticker: str) -> Tuple[int, float, Dict]:
        """Get technical analysis score for bond ETF."""
        try:
            data = yf.download(ticker, period='60d', progress=False, auto_adjust=True)

            if data.empty:
                return 50, 0, {}

            close = data['Close']
            # Handle both Series and DataFrame cases
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            current = float(close.iloc[-1])

            # Moving averages
            ma_20 = float(close.rolling(20).mean().iloc[-1])
            ma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else ma_20

            score = 50

            # Above MAs is bullish
            if current > ma_20:
                score += 10
            else:
                score -= 10

            if current > ma_50:
                score += 10
            else:
                score -= 10

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = float(100 - (100 / (1 + rs.iloc[-1])))

            if rsi < 30:
                score += 15  # Oversold = bullish
            elif rsi > 70:
                score -= 15  # Overbought = bearish

            # Momentum
            if len(close) >= 20:
                momentum_20d = float((close.iloc[-1] / close.iloc[-20] - 1) * 100)
                if momentum_20d > 3:
                    score += 10
                elif momentum_20d < -3:
                    score -= 10

            details = {
                'current_price': current,
                'ma_20': ma_20,
                'ma_50': ma_50,
                'rsi': rsi,
                'above_ma20': current > ma_20,
                'above_ma50': current > ma_50,
            }

            return max(0, min(100, score)), current, details

        except Exception as e:
            logger.debug(f"Error in technical analysis: {e}")
            return 50, 0, {}

    def _get_calendar_score(self) -> int:
        """Get calendar effects score from prediction engine."""
        if self._prediction_engine:
            try:
                calendar = self._prediction_engine.analyze_calendar_effects()
                return calendar.calendar_score
            except:
                pass
        return 50

    def _get_seasonality_score(self) -> int:
        """Get seasonality score from prediction engine."""
        if self._prediction_engine:
            try:
                # Get current month seasonality
                from src.analytics.bond_prediction_engine import BOND_SEASONALITY
                month = date.today().month
                season = BOND_SEASONALITY.get(month, {})

                if season.get('bias') == 'BULLISH':
                    return 70
                elif season.get('bias') == 'SLIGHT_BULLISH':
                    return 60
                elif season.get('bias') == 'BEARISH':
                    return 30
                elif season.get('bias') == 'SLIGHT_BEARISH':
                    return 40
            except:
                pass
        return 50

    def generate_signal(self, ticker: str) -> BondSignalResult:
        """Generate trading signal for a bond ETF - ENHANCED with prediction engine."""
        if ticker not in BOND_INSTRUMENTS:
            raise ValueError(f"Unknown bond instrument: {ticker}")

        instrument = BOND_INSTRUMENTS[ticker]
        leverage_factor = abs(instrument.leverage)

        # Get all analysis components
        yields = self.get_treasury_yields()
        yield_hist = self.get_yield_history(60)
        yield_trend, yield_score, yield_details = self.analyze_yield_trend(yield_hist)
        curve_shape, curve_score, curve_details = self.analyze_curve_shape(yields)
        economic_impact = self.get_economic_impact()
        fed_policy_score = self._get_fed_policy_score()
        inflation_score, inflation_data = self._get_inflation_score()
        regime_score = self._get_regime_score()
        technical_score, current_price, tech_details = self._get_technical_score(ticker)
        calendar_score = self._get_calendar_score()
        seasonality_score = self._get_seasonality_score()

        # Apply economic impact adjustment
        economic_score = 50 + economic_impact.today_score_adjustment

        # Composite score with enhanced weighting
        weights = {
            'yield': 0.25,
            'curve': 0.10,
            'economic': 0.15,
            'fed': 0.10,
            'regime': 0.10,
            'technical': 0.10,
            'calendar': 0.10,
            'seasonality': 0.10,
        }

        score = int(
            yield_score * weights['yield'] +
            curve_score * weights['curve'] +
            economic_score * weights['economic'] +
            fed_policy_score * weights['fed'] +
            regime_score * weights['regime'] +
            technical_score * weights['technical'] +
            calendar_score * weights['calendar'] +
            seasonality_score * weights['seasonality']
        )

        # Determine signal
        if score >= 70:
            signal = BondSignal.STRONG_BUY
        elif score >= 55:
            signal = BondSignal.BUY
        elif score >= 45:
            signal = BondSignal.HOLD
        elif score >= 30:
            signal = BondSignal.SELL
        else:
            signal = BondSignal.STRONG_SELL

        # Reverse for inverse ETFs
        if instrument.leverage < 0:
            score = 100 - score
            signal_map = {
                BondSignal.STRONG_BUY: BondSignal.STRONG_SELL,
                BondSignal.BUY: BondSignal.SELL,
                BondSignal.HOLD: BondSignal.HOLD,
                BondSignal.SELL: BondSignal.BUY,
                BondSignal.STRONG_SELL: BondSignal.STRONG_BUY,
            }
            signal = signal_map[signal]

        # Price targets based on duration and DYNAMIC yield expectations
        duration = instrument.duration * leverage_factor

        # NEW: Get dynamic yield forecast instead of static 25bp
        expected_yield_change, forecast_bps = self._get_yield_forecast(yield_trend)

        # Duration approximation: % price change ≈ -duration * yield change
        expected_return = -duration * expected_yield_change / 100

        if current_price > 0:
            target_price = current_price * (1 + expected_return)
            stop_loss = current_price * (1 - 0.05 / leverage_factor)  # 5% adjusted for leverage
        else:
            target_price = 0
            stop_loss = 0

        upside_pct = ((target_price / current_price) - 1) * 100 if current_price > 0 else 0
        downside_pct = ((current_price - stop_loss) / current_price) * 100 if current_price > 0 else 5
        risk_reward = abs(upside_pct / downside_pct) if downside_pct > 0 else 1

        # Confidence based on score extremity and component agreement
        confidence = min(0.9, 0.5 + abs(score - 50) / 100)

        # Reduce confidence for high volatility events
        if economic_impact.expected_volatility == "EXTREME":
            confidence *= 0.7
        elif economic_impact.expected_volatility == "HIGH":
            confidence *= 0.85

        # Bull/Bear factors - ENHANCED
        bull_factors = []
        bear_factors = []

        if yield_score >= 60:
            bull_factors.append(f"Yields {yield_trend.value.lower().replace('_', ' ')}")
        elif yield_score <= 40:
            bear_factors.append(f"Yields {yield_trend.value.lower().replace('_', ' ')}")

        if curve_score >= 60:
            bull_factors.append(f"Curve {curve_shape.value.lower().replace('_', ' ')}")
        elif curve_score <= 40:
            bear_factors.append(f"Curve {curve_shape.value.lower().replace('_', ' ')}")

        if technical_score >= 60:
            bull_factors.append("Price above moving averages")
        elif technical_score <= 40:
            bear_factors.append("Price below moving averages")

        if economic_impact.cpi_bond_impact == "BULLISH":
            bull_factors.append(f"CPI {economic_impact.cpi_result} - bullish for bonds")
        elif economic_impact.cpi_bond_impact == "BEARISH":
            bear_factors.append(f"CPI {economic_impact.cpi_result} - bearish for bonds")

        # NEW: Calendar factors
        if calendar_score >= 60:
            bull_factors.append(f"Seasonally bullish ({economic_impact.seasonality_bias})")
        elif calendar_score <= 40:
            bear_factors.append(f"Seasonally weak ({economic_impact.seasonality_bias})")

        if economic_impact.is_first_trading_day_year:
            bull_factors.append("First trading day of year - rebalancing flows")

        if regime_score >= 60:
            bull_factors.append("Risk-off environment - flight to safety")
        elif regime_score <= 40:
            bear_factors.append("Risk-on environment - rotation out of bonds")

        # Recommendation - ENHANCED
        if signal in [BondSignal.STRONG_BUY, BondSignal.BUY]:
            recommendation = f"Consider buying {ticker}. Duration: {duration:.1f}y. Target: ${target_price:.2f} ({upside_pct:+.1f}%)"
            if forecast_bps:
                recommendation += f" | Yield forecast: {forecast_bps:+.0f}bps"
        elif signal in [BondSignal.STRONG_SELL, BondSignal.SELL]:
            recommendation = f"Avoid or reduce {ticker}. Consider shorter duration alternatives."
            if forecast_bps:
                recommendation += f" | Yield forecast: {forecast_bps:+.0f}bps"
        else:
            recommendation = f"Hold {ticker}. Wait for clearer direction in yields."

        return BondSignalResult(
            ticker=ticker,
            signal=signal,
            score=score,
            confidence=confidence,
            current_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            upside_pct=upside_pct,
            downside_pct=downside_pct,
            risk_reward=risk_reward,
            yield_trend=yield_trend,
            yield_trend_score=yield_score,
            curve_shape=curve_shape,
            curve_score=curve_score,
            fed_policy_score=fed_policy_score,
            inflation_score=inflation_score,
            inflation_data=inflation_data,
            regime_score=regime_score,
            technical_score=technical_score,
            economic_score=economic_score,
            calendar_score=calendar_score,
            seasonality_score=seasonality_score,
            cross_asset_score=regime_score,
            yield_momentum_forecast_bps=forecast_bps,
            economic_impact=economic_impact,
            bull_factors=bull_factors,
            bear_factors=bear_factors,
            recommendation=recommendation,
            predicted_return_5d=expected_return * 100 if expected_return else None,
            risk_events_ahead=economic_impact.calendar_alerts,
        )

    def generate_all_signals(self) -> Dict[str, BondSignalResult]:
        """Generate signals for all bond instruments."""
        signals = {}
        for ticker in BOND_INSTRUMENTS:
            try:
                signals[ticker] = self.generate_signal(ticker)
            except Exception as e:
                logger.error(f"Error generating signal for {ticker}: {e}")
        return signals

    def get_bond_context_for_ai(self, ticker: str = None) -> str:
        """Get bond analysis formatted for AI Chat - ENHANCED."""
        yields = self.get_treasury_yields()
        yield_hist = self.get_yield_history(60)
        yield_trend, yield_score, yield_details = self.analyze_yield_trend(yield_hist)
        curve_shape, curve_score, curve_details = self.analyze_curve_shape(yields)
        economic_impact = self.get_economic_impact()

        context = f"""
🏦 BOND MARKET ANALYSIS
{'=' * 50}

📊 TREASURY YIELDS:
   30-Year: {yields.yield_30y:.2f}%
   10-Year: {yields.yield_10y:.2f}%
   2-Year:  {yields.yield_2y:.2f}%

📈 YIELD TREND: {yield_trend.value}
   5-Day Change: {yield_details.get('change_5d', 0):+.2f}%
   20-Day Change: {yield_details.get('change_20d', 0):+.2f}%

📉 YIELD CURVE: {curve_shape.value}
   10Y-2Y Spread: {yields.spread_10y_2y:+.2f}%
   Interpretation: {curve_details.get('interpretation', '')}
"""

        # NEW: Add calendar alerts
        if economic_impact.calendar_alerts:
            context += f"""
📅 CALENDAR ALERTS:
"""
            for alert in economic_impact.calendar_alerts[:5]:
                context += f"   {alert}\n"

        if economic_impact.has_high_impact_today or economic_impact.today_events:
            context += f"""
{'=' * 50}
📅 TODAY'S ECONOMIC EVENTS:
   {economic_impact.today_summary}
"""
            if economic_impact.cpi_today:
                context += f"""
   📊 CPI Report: {economic_impact.cpi_result or 'PENDING'}
      Bond Impact: {economic_impact.cpi_bond_impact or 'TBD'}
"""
            if economic_impact.fomc_today:
                context += """
   🏛️ FOMC DECISION TODAY - Expect high volatility!
"""

        # NEW: Seasonality
        context += f"""
📆 SEASONALITY:
   Bias: {economic_impact.seasonality_bias}
   Expected Volatility: {economic_impact.expected_volatility}
   Days to FOMC: {economic_impact.days_to_fomc}
"""

        context += f"""
💡 IMPLICATION:
   {"Yields falling = BULLISH for bond ETFs (TLT, ZROZ)" if yield_score > 50 else "Yields rising = BEARISH for bond ETFs"}
"""

        if ticker:
            try:
                signal = self.generate_signal(ticker)
                context += f"""
{'=' * 50}
📊 {ticker} SIGNAL: {signal.signal.value}
   Score: {signal.score}/100
   Current Price: ${signal.current_price:.2f}
   Target: ${signal.target_price:.2f} ({signal.upside_pct:+.1f}%)
   Predicted 5D Return: {signal.predicted_return_5d:+.1f}%
   Confidence: {signal.confidence:.0%}
   Recommendation: {signal.recommendation}

Bull Factors:
"""
                for factor in signal.bull_factors:
                    context += f"   ✅ {factor}\n"

                context += "\nBear Factors:\n"
                for factor in signal.bear_factors:
                    context += f"   ⚠️ {factor}\n"

            except Exception as e:
                logger.error(f"Error getting signal for {ticker}: {e}")

        return context


# ============================================================
# Convenience Functions
# ============================================================

_generator = None


def get_bond_generator() -> BondSignalGenerator:
    """Get singleton generator instance."""
    global _generator
    if _generator is None:
        _generator = BondSignalGenerator()
    return _generator


def get_bond_signal(ticker: str) -> BondSignalResult:
    """Get bond signal for a ticker."""
    return get_bond_generator().generate_signal(ticker)


def get_all_bond_signals() -> Dict[str, BondSignalResult]:
    """Get signals for all bond instruments."""
    return get_bond_generator().generate_all_signals()


def get_bond_context_for_ai(ticker: str = None) -> str:
    """Get bond analysis for AI chat."""
    return get_bond_generator().get_bond_context_for_ai(ticker)


def get_treasury_yields() -> YieldData:
    """Get current treasury yields."""
    return get_bond_generator().get_treasury_yields()


# NEW: Integration with alpha model
def get_rate_signals_for_alpha() -> Dict[str, Any]:
    """Get rate signals for integration with main alpha model."""
    try:
        from src.analytics.bond_prediction_engine import get_rate_signals
        return get_rate_signals()
    except ImportError:
        # Fallback
        generator = get_bond_generator()
        yields = generator.get_treasury_yields()
        yield_hist = generator.get_yield_history(20)
        yield_trend, _, details = generator.analyze_yield_trend(yield_hist)

        return {
            'rate_direction': yield_trend.value,
            'yield_30y': yields.yield_30y,
            'yield_10y': yields.yield_10y,
            'yield_change_5d_bps': details.get('change_5d', 0) * 100,
            'bond_signal': (50 - details.get('change_5d', 0) * 200) / 50,  # Rough conversion
        }


if __name__ == "__main__":
    print("Bond Signal Generator Test (Enhanced)\n")
    print("=" * 60)

    generator = BondSignalGenerator()
    yields = generator.get_treasury_yields()

    print(f"30Y Yield: {yields.yield_30y:.2f}%")
    print(f"10Y Yield: {yields.yield_10y:.2f}%")
    print(f"10Y-2Y Spread: {yields.spread_10y_2y:+.2f}%")

    # Economic impact
    print("\n" + "=" * 60)
    print("ECONOMIC IMPACT:")
    impact = generator.get_economic_impact()
    print(f"  First day of year: {impact.is_first_trading_day_year}")
    print(f"  Seasonality: {impact.seasonality_bias}")
    print(f"  Expected volatility: {impact.expected_volatility}")
    print(f"  Days to FOMC: {impact.days_to_fomc}")
    if impact.calendar_alerts:
        print("  Alerts:")
        for alert in impact.calendar_alerts:
            print(f"    {alert}")

    print("\n" + "=" * 60)
    print("BOND SIGNALS:")

    for ticker in ['TLT', 'ZROZ', 'EDV', 'TBT']:
        signal = generator.generate_signal(ticker)
        print(f"\n{ticker}: {signal.signal.value} (Score: {signal.score})")
        print(f"  Price: ${signal.current_price:.2f}")
        print(f"  Target: ${signal.target_price:.2f} ({signal.upside_pct:+.1f}%)")
        print(f"  Predicted 5D: {signal.predicted_return_5d:+.1f}%" if signal.predicted_return_5d else "  Predicted: N/A")
        print(f"  Confidence: {signal.confidence:.0%}")
        print(f"  Yield Forecast: {signal.yield_momentum_forecast_bps:+.0f}bps" if signal.yield_momentum_forecast_bps else "")
        if signal.bull_factors:
            print(f"  Bull: {', '.join(signal.bull_factors[:2])}")
        if signal.bear_factors:
            print(f"  Bear: {', '.join(signal.bear_factors[:2])}")