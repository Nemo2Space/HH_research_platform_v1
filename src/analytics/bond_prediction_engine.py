"""
Bond Prediction Engine - Comprehensive Forecasting Module

Predicts bond ETF moves based on:
1. Yield Momentum (rate of change, acceleration)
2. Economic Data Surprises (jobless claims, CPI, NFP, GDP)
3. Calendar Effects (month-end, quarter-end, year-end, ex-dividend)
4. Seasonality Patterns (January effect, summer doldrums, etc.)
5. Fed Policy Expectations
6. Cross-Asset Signals (VIX, DXY, Gold)
7. Auction Calendar
8. Technical Momentum

Instruments: TLT, ZROZ, EDV, TMF, TBT, IEF, SHY

Author: Alpha Research Platform
Location: src/analytics/bond_prediction_engine.py
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import calendar as cal_module
import requests

try:
    from src.utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

try:
    from src.db.connection import get_engine
except ImportError:
    get_engine = None


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class PredictionSignal(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class YieldDirection(Enum):
    RISING_FAST = "RISING_FAST"      # >5bp/day momentum
    RISING = "RISING"                 # 2-5bp/day
    STABLE = "STABLE"                 # -2 to +2bp/day
    FALLING = "FALLING"               # -5 to -2bp/day
    FALLING_FAST = "FALLING_FAST"     # <-5bp/day


class EconomicSurpriseType(Enum):
    HAWKISH_SURPRISE = "HAWKISH_SURPRISE"      # Data stronger than expected
    SLIGHT_HAWKISH = "SLIGHT_HAWKISH"
    IN_LINE = "IN_LINE"
    SLIGHT_DOVISH = "SLIGHT_DOVISH"
    DOVISH_SURPRISE = "DOVISH_SURPRISE"        # Data weaker than expected


class CalendarEffect(Enum):
    FIRST_TRADING_DAY_YEAR = "FIRST_TRADING_DAY_YEAR"
    FIRST_TRADING_DAY_MONTH = "FIRST_TRADING_DAY_MONTH"
    LAST_TRADING_DAY_MONTH = "LAST_TRADING_DAY_MONTH"
    QUARTER_END = "QUARTER_END"
    YEAR_END = "YEAR_END"
    EX_DIVIDEND = "EX_DIVIDEND"
    FOMC_DAY = "FOMC_DAY"
    FOMC_WEEK = "FOMC_WEEK"
    NFP_DAY = "NFP_DAY"            # Non-Farm Payrolls (first Friday)
    CPI_DAY = "CPI_DAY"
    AUCTION_DAY = "AUCTION_DAY"
    HOLIDAY_WEEK = "HOLIDAY_WEEK"
    SUMMER_DOLDRUMS = "SUMMER_DOLDRUMS"  # July-August
    NORMAL = "NORMAL"


# Bond ETF Definitions with enhanced metadata
BOND_ETFS = {
    'TLT': {
        'name': 'iShares 20+ Year Treasury Bond ETF',
        'duration': 17.5,
        'benchmark_yield': '30y',
        'expense_ratio': 0.15,
        'ex_div_months': [1, 4, 7, 10],  # Quarterly
        'avg_yield': 3.5,
    },
    'ZROZ': {
        'name': 'PIMCO 25+ Year Zero Coupon Treasury ETF',
        'duration': 27.0,
        'benchmark_yield': '30y',
        'expense_ratio': 0.15,
        'ex_div_months': [3, 6, 9, 12],
        'avg_yield': 0,  # Zero coupon
    },
    'EDV': {
        'name': 'Vanguard Extended Duration Treasury ETF',
        'duration': 24.5,
        'benchmark_yield': '30y',
        'expense_ratio': 0.06,
        'ex_div_months': [3, 6, 9, 12],
        'avg_yield': 3.8,
    },
    'TMF': {
        'name': 'Direxion Daily 20+ Year Treasury Bull 3X',
        'duration': 52.5,  # 3x leveraged
        'benchmark_yield': '30y',
        'expense_ratio': 1.01,
        'leverage': 3.0,
        'ex_div_months': [3, 6, 9, 12],
        'avg_yield': 0,
    },
    'TBT': {
        'name': 'ProShares UltraShort 20+ Year Treasury',
        'duration': -35.0,  # Inverse 2x
        'benchmark_yield': '30y',
        'expense_ratio': 0.90,
        'leverage': -2.0,
        'ex_div_months': [],
        'avg_yield': 0,
    },
    'IEF': {
        'name': 'iShares 7-10 Year Treasury Bond ETF',
        'duration': 7.5,
        'benchmark_yield': '10y',
        'expense_ratio': 0.15,
        'ex_div_months': [1, 4, 7, 10],
        'avg_yield': 3.2,
    },
    'SHY': {
        'name': 'iShares 1-3 Year Treasury Bond ETF',
        'duration': 1.9,
        'benchmark_yield': '2y',
        'expense_ratio': 0.15,
        'ex_div_months': [1, 4, 7, 10],
        'avg_yield': 4.5,
    },
}

# High-impact economic events and their bond impact
ECONOMIC_EVENTS = {
    'NFP': {
        'name': 'Non-Farm Payrolls',
        'frequency': 'monthly',
        'day': 'first_friday',
        'impact': 'HIGH',
        'bond_sensitivity': -0.8,  # Stronger jobs = bearish bonds
    },
    'CPI': {
        'name': 'Consumer Price Index',
        'frequency': 'monthly',
        'impact': 'HIGH',
        'bond_sensitivity': -1.0,  # Higher inflation = very bearish bonds
    },
    'FOMC': {
        'name': 'FOMC Rate Decision',
        'frequency': '8x_yearly',
        'impact': 'EXTREME',
        'bond_sensitivity': -0.5,  # Depends on decision
    },
    'JOBLESS_CLAIMS': {
        'name': 'Initial Jobless Claims',
        'frequency': 'weekly',
        'day': 'thursday',
        'impact': 'MEDIUM',
        'bond_sensitivity': 0.6,  # Higher claims = bullish bonds
    },
    'GDP': {
        'name': 'GDP Growth',
        'frequency': 'quarterly',
        'impact': 'HIGH',
        'bond_sensitivity': -0.7,  # Stronger growth = bearish bonds
    },
    'PCE': {
        'name': 'PCE Price Index',
        'frequency': 'monthly',
        'impact': 'HIGH',
        'bond_sensitivity': -0.9,  # Fed's preferred inflation gauge
    },
    'ISM_MFG': {
        'name': 'ISM Manufacturing',
        'frequency': 'monthly',
        'day': 'first_business',
        'impact': 'MEDIUM',
        'bond_sensitivity': -0.5,
    },
    'RETAIL_SALES': {
        'name': 'Retail Sales',
        'frequency': 'monthly',
        'impact': 'MEDIUM',
        'bond_sensitivity': -0.4,
    },
    'HOUSING_STARTS': {
        'name': 'Housing Starts',
        'frequency': 'monthly',
        'impact': 'LOW',
        'bond_sensitivity': -0.3,
    },
    'TREASURY_AUCTION': {
        'name': 'Treasury Auction',
        'frequency': 'varies',
        'impact': 'MEDIUM',
        'bond_sensitivity': 0.0,  # Depends on demand
    },
}

# 2026 FOMC Meeting Dates (approximate)
FOMC_DATES_2026 = [
    date(2026, 1, 28), date(2026, 1, 29),    # Jan
    date(2026, 3, 17), date(2026, 3, 18),    # Mar
    date(2026, 5, 5), date(2026, 5, 6),      # May
    date(2026, 6, 16), date(2026, 6, 17),    # Jun
    date(2026, 7, 28), date(2026, 7, 29),    # Jul
    date(2026, 9, 15), date(2026, 9, 16),    # Sep
    date(2026, 11, 3), date(2026, 11, 4),    # Nov
    date(2026, 12, 15), date(2026, 12, 16),  # Dec
]

# Seasonality patterns (average monthly returns for TLT historically)
BOND_SEASONALITY = {
    1: {'avg_return': 0.5, 'volatility': 'HIGH', 'bias': 'SLIGHT_BULLISH',
        'note': 'January effect - portfolio rebalancing'},
    2: {'avg_return': -0.3, 'volatility': 'NORMAL', 'bias': 'SLIGHT_BEARISH',
        'note': 'Post-January selling'},
    3: {'avg_return': 0.2, 'volatility': 'NORMAL', 'bias': 'NEUTRAL',
        'note': 'Quarter-end rebalancing'},
    4: {'avg_return': 0.4, 'volatility': 'LOW', 'bias': 'SLIGHT_BULLISH',
        'note': 'Tax season flows'},
    5: {'avg_return': 0.3, 'volatility': 'NORMAL', 'bias': 'NEUTRAL',
        'note': 'Sell in May uncertainty'},
    6: {'avg_return': 0.6, 'volatility': 'NORMAL', 'bias': 'BULLISH',
        'note': 'Mid-year safety rotation'},
    7: {'avg_return': 0.1, 'volatility': 'LOW', 'bias': 'NEUTRAL',
        'note': 'Summer doldrums begin'},
    8: {'avg_return': 0.4, 'volatility': 'LOW', 'bias': 'SLIGHT_BULLISH',
        'note': 'Summer doldrums - flight to safety'},
    9: {'avg_return': -0.5, 'volatility': 'HIGH', 'bias': 'BEARISH',
        'note': 'September effect - worst month historically'},
    10: {'avg_return': -0.2, 'volatility': 'HIGH', 'bias': 'SLIGHT_BEARISH',
         'note': 'Volatility continues'},
    11: {'avg_return': 0.3, 'volatility': 'NORMAL', 'bias': 'SLIGHT_BULLISH',
         'note': 'Year-end positioning begins'},
    12: {'avg_return': 0.1, 'volatility': 'LOW', 'bias': 'NEUTRAL',
         'note': 'Holiday trading, low volume'},
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class YieldMomentum:
    """Yield momentum analysis."""
    current_yield_10y: float
    current_yield_30y: float

    # Changes
    change_1d_10y: Optional[float] = None
    change_1d_30y: Optional[float] = None
    change_5d_10y: Optional[float] = None
    change_5d_30y: Optional[float] = None
    change_20d_10y: Optional[float] = None
    change_20d_30y: Optional[float] = None

    # Momentum metrics
    momentum_score: int = 50  # 0-100, >50 = yields rising
    acceleration: Optional[float] = None  # Rate of change of change
    direction: YieldDirection = YieldDirection.STABLE

    # Forecast
    forecast_1d_bps: Optional[float] = None  # Expected 1-day change in bps
    forecast_5d_bps: Optional[float] = None
    confidence: float = 0.5

    reasoning: List[str] = field(default_factory=list)


@dataclass
class EconomicSurprise:
    """Economic data surprise analysis."""
    event_type: str
    event_name: str
    release_date: Optional[date] = None
    release_time: Optional[str] = None

    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None

    surprise_pct: Optional[float] = None  # (actual - forecast) / forecast
    surprise_type: EconomicSurpriseType = EconomicSurpriseType.IN_LINE

    bond_impact_score: int = 0  # -100 to +100, positive = bullish bonds
    bond_impact_label: str = "NEUTRAL"

    notes: str = ""


@dataclass
class CalendarAnalysis:
    """Calendar effects analysis."""
    today: date

    # Calendar flags
    is_first_trading_day_year: bool = False
    is_first_trading_day_month: bool = False
    is_last_trading_day_month: bool = False
    is_quarter_end: bool = False
    is_year_end: bool = False
    is_fomc_day: bool = False
    is_fomc_week: bool = False
    is_nfp_day: bool = False
    is_cpi_week: bool = False
    is_auction_day: bool = False
    is_ex_dividend_week: Dict[str, bool] = field(default_factory=dict)

    # Days to events
    days_to_fomc: int = 999
    days_to_nfp: int = 999
    days_to_cpi: int = 999
    days_to_quarter_end: int = 999

    # Active effects
    active_effects: List[CalendarEffect] = field(default_factory=list)

    # Seasonality
    month: int = 1
    seasonality_bias: str = "NEUTRAL"
    seasonality_note: str = ""
    expected_volatility: str = "NORMAL"

    # Combined calendar score
    calendar_score: int = 50  # 0-100, >50 = bullish for bonds
    calendar_adjustment: int = 0  # -20 to +20 score adjustment

    alerts: List[str] = field(default_factory=list)


@dataclass
class CrossAssetSignals:
    """Cross-asset correlation signals."""
    # VIX
    vix_level: Optional[float] = None
    vix_change_1d: Optional[float] = None
    vix_signal: str = "NEUTRAL"  # HIGH_FEAR, ELEVATED, NORMAL, COMPLACENT

    # Dollar
    dxy_level: Optional[float] = None
    dxy_change_5d: Optional[float] = None
    dxy_signal: str = "NEUTRAL"

    # Gold
    gold_price: Optional[float] = None
    gold_change_5d: Optional[float] = None
    gold_signal: str = "NEUTRAL"

    # Equity
    spy_change_5d: Optional[float] = None
    equity_bond_correlation: str = "NORMAL"  # POSITIVE, NEGATIVE, DECORRELATED

    # Combined
    cross_asset_score: int = 50
    risk_regime: str = "NEUTRAL"  # RISK_ON, RISK_OFF, NEUTRAL


@dataclass
class BondPrediction:
    """Complete bond prediction with all factors."""
    ticker: str
    name: str
    prediction_time: datetime

    # Current state
    current_price: float
    duration: float

    # Predictions
    predicted_return_1d: Optional[float] = None
    predicted_return_5d: Optional[float] = None
    predicted_return_20d: Optional[float] = None

    predicted_price_1d: Optional[float] = None
    predicted_price_5d: Optional[float] = None

    # Signal
    signal: PredictionSignal = PredictionSignal.HOLD
    confidence: float = 0.5

    # Component scores (0-100, >50 = bullish)
    yield_momentum_score: int = 50
    economic_score: int = 50
    calendar_score: int = 50
    seasonality_score: int = 50
    cross_asset_score: int = 50
    technical_score: int = 50

    # Composite
    composite_score: int = 50

    # Components
    yield_momentum: Optional[YieldMomentum] = None
    economic_surprises: List[EconomicSurprise] = field(default_factory=list)
    calendar_analysis: Optional[CalendarAnalysis] = None
    cross_asset: Optional[CrossAssetSignals] = None

    # Risk
    expected_volatility: str = "NORMAL"
    risk_events_ahead: List[str] = field(default_factory=list)

    # Reasoning
    bull_factors: List[str] = field(default_factory=list)
    bear_factors: List[str] = field(default_factory=list)
    key_drivers: List[str] = field(default_factory=list)
    recommendation: str = ""

    # For alpha model integration
    alpha_signal: Optional[float] = None  # -1 to +1


# =============================================================================
# BOND PREDICTION ENGINE
# =============================================================================

class BondPredictionEngine:
    """
    Comprehensive bond prediction engine.

    Analyzes:
    - Yield momentum and direction
    - Economic data surprises
    - Calendar effects (month-end, FOMC, ex-div, etc.)
    - Seasonality patterns
    - Cross-asset signals
    """

    def __init__(self):
        self._yield_cache = {}
        self._yield_cache_time = None
        self._cache_duration = timedelta(minutes=15)

        # API endpoints
        self.fred_api_key = os.getenv('FRED_API_KEY', '')
        self.tool_server_url = os.getenv('TOOL_SERVER_URL', '')

        logger.info(f"BondPredictionEngine initialized: FRED={'✓' if self.fred_api_key else '✗'}")

    # =========================================================================
    # YIELD MOMENTUM ANALYSIS
    # =========================================================================

    def analyze_yield_momentum(self) -> YieldMomentum:
        """Analyze yield momentum and forecast direction."""

        momentum = YieldMomentum(
            current_yield_10y=0,
            current_yield_30y=0,
        )

        try:
            # Fetch yield history
            tickers = ['^TNX', '^TYX']  # 10Y, 30Y
            data = yf.download(tickers, period='60d', progress=False, auto_adjust=True)

            if data.empty:
                logger.warning("No yield data available")
                return momentum

            # Current yields
            close = data['Close']
            momentum.current_yield_10y = float(close['^TNX'].iloc[-1])
            momentum.current_yield_30y = float(close['^TYX'].iloc[-1])

            # Calculate changes
            y10 = close['^TNX']
            y30 = close['^TYX']

            if len(y30) >= 2:
                momentum.change_1d_10y = float(y10.iloc[-1] - y10.iloc[-2])
                momentum.change_1d_30y = float(y30.iloc[-1] - y30.iloc[-2])

            if len(y30) >= 5:
                momentum.change_5d_10y = float(y10.iloc[-1] - y10.iloc[-5])
                momentum.change_5d_30y = float(y30.iloc[-1] - y30.iloc[-5])

            if len(y30) >= 20:
                momentum.change_20d_10y = float(y10.iloc[-1] - y10.iloc[-20])
                momentum.change_20d_30y = float(y30.iloc[-1] - y30.iloc[-20])

            # Calculate momentum score (0-100)
            # Uses weighted average of recent changes
            score = 50  # Start neutral

            if momentum.change_1d_30y:
                # 1-day change: weight 40%
                # +10bp = +20 score, -10bp = -20 score
                score += momentum.change_1d_30y * 200 * 0.4

            if momentum.change_5d_30y:
                # 5-day change: weight 35%
                score += momentum.change_5d_30y * 100 * 0.35

            if momentum.change_20d_30y:
                # 20-day change: weight 25%
                score += momentum.change_20d_30y * 50 * 0.25

            momentum.momentum_score = int(max(0, min(100, score)))

            # Determine direction
            if momentum.momentum_score >= 70:
                momentum.direction = YieldDirection.RISING_FAST
            elif momentum.momentum_score >= 58:
                momentum.direction = YieldDirection.RISING
            elif momentum.momentum_score <= 30:
                momentum.direction = YieldDirection.FALLING_FAST
            elif momentum.momentum_score <= 42:
                momentum.direction = YieldDirection.FALLING
            else:
                momentum.direction = YieldDirection.STABLE

            # Calculate acceleration (2nd derivative)
            if len(y30) >= 10:
                recent_changes = y30.diff().dropna()
                if len(recent_changes) >= 5:
                    recent_avg = recent_changes.iloc[-5:].mean()
                    prior_avg = recent_changes.iloc[-10:-5].mean()
                    momentum.acceleration = float(recent_avg - prior_avg)

            # Forecast based on momentum
            if momentum.change_1d_30y and momentum.change_5d_30y:
                # Simple momentum-based forecast
                avg_daily = momentum.change_5d_30y / 5
                momentum.forecast_1d_bps = avg_daily * 100  # Convert to bps
                momentum.forecast_5d_bps = avg_daily * 5 * 100

                # Adjust for acceleration
                if momentum.acceleration:
                    momentum.forecast_1d_bps += momentum.acceleration * 50
                    momentum.forecast_5d_bps += momentum.acceleration * 150

            # Confidence based on consistency
            if momentum.change_1d_30y and momentum.change_5d_30y:
                # Same direction = higher confidence
                if (momentum.change_1d_30y > 0) == (momentum.change_5d_30y > 0):
                    momentum.confidence = 0.7
                else:
                    momentum.confidence = 0.4

            # Build reasoning
            if momentum.direction in [YieldDirection.RISING, YieldDirection.RISING_FAST]:
                momentum.reasoning.append(f"30Y yield up {momentum.change_5d_30y*100:.0f}bps in 5 days")
                momentum.reasoning.append("Rising yields = BEARISH for bond prices")
            elif momentum.direction in [YieldDirection.FALLING, YieldDirection.FALLING_FAST]:
                momentum.reasoning.append(f"30Y yield down {abs(momentum.change_5d_30y)*100:.0f}bps in 5 days")
                momentum.reasoning.append("Falling yields = BULLISH for bond prices")

            if momentum.acceleration and abs(momentum.acceleration) > 0.01:
                if momentum.acceleration > 0:
                    momentum.reasoning.append("Yield rise is ACCELERATING")
                else:
                    momentum.reasoning.append("Yield rise is DECELERATING")

        except Exception as e:
            logger.error(f"Error analyzing yield momentum: {e}")

        return momentum

    # =========================================================================
    # ECONOMIC SURPRISE ANALYSIS
    # =========================================================================

    def analyze_economic_surprises(self, days_back: int = 7) -> List[EconomicSurprise]:
        """Analyze recent economic data surprises and their bond impact."""

        surprises = []

        # Try to get from economic calendar if available
        try:
            from src.analytics.economic_calendar import EconomicCalendarFetcher

            fetcher = EconomicCalendarFetcher()
            calendar_data = fetcher.get_calendar(days_back=days_back, days_ahead=0)

            if calendar_data and calendar_data.events:
                for event in calendar_data.events:
                    if event.importance != 'HIGH':
                        continue

                    surprise = self._analyze_single_event(event)
                    if surprise:
                        surprises.append(surprise)

        except ImportError:
            logger.debug("Economic calendar not available, using web search fallback")
        except Exception as e:
            logger.warning(f"Error getting economic calendar: {e}")

        # Fallback: Search for recent economic data
        if not surprises and self.tool_server_url:
            surprises = self._search_recent_economic_data()

        return surprises

    def _analyze_single_event(self, event) -> Optional[EconomicSurprise]:
        """Analyze a single economic event for surprise factor."""

        try:
            actual = getattr(event, 'actual', None)
            forecast = getattr(event, 'forecast', None)
            previous = getattr(event, 'previous', None)
            name = getattr(event, 'event', '') or getattr(event, 'name', '')

            if actual is None or forecast is None:
                return None

            # Parse values
            try:
                actual_val = float(str(actual).replace('%', '').replace('K', '000').replace('M', '000000'))
                forecast_val = float(str(forecast).replace('%', '').replace('K', '000').replace('M', '000000'))
            except:
                return None

            # Calculate surprise
            if forecast_val != 0:
                surprise_pct = (actual_val - forecast_val) / abs(forecast_val) * 100
            else:
                surprise_pct = 0

            # Determine surprise type
            if surprise_pct > 5:
                surprise_type = EconomicSurpriseType.HAWKISH_SURPRISE
            elif surprise_pct > 1:
                surprise_type = EconomicSurpriseType.SLIGHT_HAWKISH
            elif surprise_pct < -5:
                surprise_type = EconomicSurpriseType.DOVISH_SURPRISE
            elif surprise_pct < -1:
                surprise_type = EconomicSurpriseType.SLIGHT_DOVISH
            else:
                surprise_type = EconomicSurpriseType.IN_LINE

            # Determine bond impact based on event type
            bond_sensitivity = self._get_event_bond_sensitivity(name)
            bond_impact_score = int(-surprise_pct * bond_sensitivity * 2)
            bond_impact_score = max(-100, min(100, bond_impact_score))

            if bond_impact_score > 20:
                bond_impact_label = "BULLISH"
            elif bond_impact_score > 5:
                bond_impact_label = "SLIGHT_BULLISH"
            elif bond_impact_score < -20:
                bond_impact_label = "BEARISH"
            elif bond_impact_score < -5:
                bond_impact_label = "SLIGHT_BEARISH"
            else:
                bond_impact_label = "NEUTRAL"

            return EconomicSurprise(
                event_type=self._classify_event_type(name),
                event_name=name,
                actual=actual_val,
                forecast=forecast_val,
                previous=float(str(previous).replace('%', '').replace('K', '000').replace('M', '000000')) if previous else None,
                surprise_pct=surprise_pct,
                surprise_type=surprise_type,
                bond_impact_score=bond_impact_score,
                bond_impact_label=bond_impact_label,
                notes=f"Actual {actual} vs Forecast {forecast}",
            )

        except Exception as e:
            logger.debug(f"Error analyzing event: {e}")
            return None

    def _get_event_bond_sensitivity(self, event_name: str) -> float:
        """Get bond sensitivity for an economic event."""
        name_lower = event_name.lower()

        if 'cpi' in name_lower or 'inflation' in name_lower:
            return -1.0
        elif 'pce' in name_lower:
            return -0.9
        elif 'payroll' in name_lower or 'nfp' in name_lower or 'employment' in name_lower:
            return -0.8
        elif 'gdp' in name_lower:
            return -0.7
        elif 'jobless' in name_lower or 'claims' in name_lower:
            return 0.6  # Higher claims = weaker economy = bullish bonds
        elif 'retail' in name_lower:
            return -0.4
        elif 'ism' in name_lower or 'pmi' in name_lower:
            return -0.5
        elif 'housing' in name_lower:
            return -0.3
        else:
            return -0.3

    def _classify_event_type(self, event_name: str) -> str:
        """Classify event type from name."""
        name_lower = event_name.lower()

        if 'cpi' in name_lower:
            return 'CPI'
        elif 'pce' in name_lower:
            return 'PCE'
        elif 'payroll' in name_lower or 'nfp' in name_lower:
            return 'NFP'
        elif 'jobless' in name_lower or 'claims' in name_lower:
            return 'JOBLESS_CLAIMS'
        elif 'gdp' in name_lower:
            return 'GDP'
        elif 'fomc' in name_lower or 'fed' in name_lower:
            return 'FOMC'
        elif 'ism' in name_lower:
            return 'ISM_MFG'
        elif 'retail' in name_lower:
            return 'RETAIL_SALES'
        else:
            return 'OTHER'

    def _search_recent_economic_data(self) -> List[EconomicSurprise]:
        """Search for recent economic data via web search."""
        surprises = []

        if not self.tool_server_url:
            return surprises

        try:
            # Search for recent jobless claims
            response = requests.post(
                f"{self.tool_server_url}/search",
                json={"query": "initial jobless claims this week actual vs forecast", "max_results": 3},
                timeout=15
            )

            if response.status_code == 200:
                results = response.json().get('results', [])
                for item in results:
                    snippet = item.get('snippet', '').lower()

                    # Try to parse numbers from snippet
                    import re
                    numbers = re.findall(r'(\d{3})[,\s]*000', snippet)

                    if len(numbers) >= 2:
                        # Assume format like "actual 219,000 vs forecast 225,000"
                        surprises.append(EconomicSurprise(
                            event_type='JOBLESS_CLAIMS',
                            event_name='Initial Jobless Claims',
                            notes=snippet[:200],
                        ))
                        break

        except Exception as e:
            logger.debug(f"Error searching economic data: {e}")

        return surprises

    # =========================================================================
    # CALENDAR EFFECTS ANALYSIS
    # =========================================================================

    def analyze_calendar_effects(self, target_date: date = None) -> CalendarAnalysis:
        """Analyze calendar effects for trading."""

        if target_date is None:
            target_date = date.today()

        analysis = CalendarAnalysis(today=target_date, month=target_date.month)

        # First trading day of year
        if target_date.month == 1 and target_date.day <= 3:
            # Check if it's actually the first trading day
            if target_date.weekday() < 5:  # Weekday
                jan1 = date(target_date.year, 1, 1)
                if jan1.weekday() >= 5:  # Jan 1 is weekend
                    first_trading = jan1 + timedelta(days=(7 - jan1.weekday()))
                else:
                    first_trading = jan1

                if target_date == first_trading or (target_date - first_trading).days <= 1:
                    analysis.is_first_trading_day_year = True
                    analysis.active_effects.append(CalendarEffect.FIRST_TRADING_DAY_YEAR)
                    analysis.alerts.append("🎆 First trading day of year - expect volatility, rebalancing flows")
                    analysis.calendar_adjustment += 5  # Slight bullish bias historically

        # First trading day of month
        if target_date.day <= 3:
            first_of_month = date(target_date.year, target_date.month, 1)
            if first_of_month.weekday() >= 5:
                first_trading = first_of_month + timedelta(days=(7 - first_of_month.weekday()))
            else:
                first_trading = first_of_month

            if target_date == first_trading:
                analysis.is_first_trading_day_month = True
                analysis.active_effects.append(CalendarEffect.FIRST_TRADING_DAY_MONTH)
                analysis.alerts.append("📅 First trading day of month - pension/401k inflows")

        # Last trading day of month
        last_day = cal_module.monthrange(target_date.year, target_date.month)[1]
        last_of_month = date(target_date.year, target_date.month, last_day)
        while last_of_month.weekday() >= 5:
            last_of_month -= timedelta(days=1)

        if target_date == last_of_month:
            analysis.is_last_trading_day_month = True
            analysis.active_effects.append(CalendarEffect.LAST_TRADING_DAY_MONTH)
            analysis.alerts.append("📅 Last trading day of month - window dressing")
            analysis.calendar_adjustment += 3

        # Quarter end
        if target_date.month in [3, 6, 9, 12] and target_date == last_of_month:
            analysis.is_quarter_end = True
            analysis.active_effects.append(CalendarEffect.QUARTER_END)
            analysis.alerts.append("📊 Quarter-end - significant rebalancing expected")
            analysis.calendar_adjustment += 5

        # Calculate days to quarter end
        current_quarter = (target_date.month - 1) // 3 + 1
        quarter_end_month = current_quarter * 3
        quarter_end_day = cal_module.monthrange(target_date.year, quarter_end_month)[1]
        quarter_end = date(target_date.year, quarter_end_month, quarter_end_day)
        analysis.days_to_quarter_end = (quarter_end - target_date).days

        # Year end
        if target_date.month == 12 and target_date.day >= 28:
            analysis.is_year_end = True
            analysis.active_effects.append(CalendarEffect.YEAR_END)
            analysis.alerts.append("🎄 Year-end - tax loss selling, low liquidity")

        # FOMC
        for fomc_date in FOMC_DATES_2026:
            days_to = (fomc_date - target_date).days
            if days_to >= 0 and days_to < analysis.days_to_fomc:
                analysis.days_to_fomc = days_to

            if days_to == 0:
                analysis.is_fomc_day = True
                analysis.active_effects.append(CalendarEffect.FOMC_DAY)
                analysis.alerts.append("🏛️ FOMC DAY - EXTREME volatility expected!")
                analysis.expected_volatility = "EXTREME"
            elif 0 <= days_to <= 3:
                analysis.is_fomc_week = True
                analysis.active_effects.append(CalendarEffect.FOMC_WEEK)
                analysis.alerts.append(f"🏛️ FOMC in {days_to} days - elevated volatility")
                analysis.expected_volatility = "HIGH"

        # NFP (First Friday of month)
        first_of_month = date(target_date.year, target_date.month, 1)
        first_friday = first_of_month + timedelta(days=(4 - first_of_month.weekday() + 7) % 7)

        days_to_nfp = (first_friday - target_date).days
        if days_to_nfp < 0:
            # Next month
            next_month = target_date.month + 1 if target_date.month < 12 else 1
            next_year = target_date.year if target_date.month < 12 else target_date.year + 1
            first_of_next = date(next_year, next_month, 1)
            first_friday = first_of_next + timedelta(days=(4 - first_of_next.weekday() + 7) % 7)
            days_to_nfp = (first_friday - target_date).days

        analysis.days_to_nfp = days_to_nfp

        if target_date == first_friday:
            analysis.is_nfp_day = True
            analysis.active_effects.append(CalendarEffect.NFP_DAY)
            analysis.alerts.append("📊 NFP DAY - Major market mover!")
            analysis.expected_volatility = "HIGH"

        # Summer doldrums
        if target_date.month in [7, 8]:
            analysis.active_effects.append(CalendarEffect.SUMMER_DOLDRUMS)
            analysis.expected_volatility = "LOW"
            analysis.alerts.append("☀️ Summer doldrums - low volume, reduced volatility")

        # Ex-dividend analysis
        for ticker, info in BOND_ETFS.items():
            ex_div_months = info.get('ex_div_months', [])
            if target_date.month in ex_div_months:
                # Ex-div is usually around 1st week of the month
                if 1 <= target_date.day <= 7:
                    analysis.is_ex_dividend_week[ticker] = True
                    analysis.alerts.append(f"💰 {ticker} ex-dividend week")

        # Seasonality
        season = BOND_SEASONALITY.get(target_date.month, {})
        analysis.seasonality_bias = season.get('bias', 'NEUTRAL')
        analysis.seasonality_note = season.get('note', '')

        if season.get('volatility') == 'HIGH':
            if analysis.expected_volatility != "EXTREME":
                analysis.expected_volatility = "HIGH"

        # Calculate calendar score
        base_score = 50

        # Seasonality adjustment
        if analysis.seasonality_bias == 'BULLISH':
            base_score += 10
        elif analysis.seasonality_bias == 'SLIGHT_BULLISH':
            base_score += 5
        elif analysis.seasonality_bias == 'BEARISH':
            base_score -= 10
        elif analysis.seasonality_bias == 'SLIGHT_BEARISH':
            base_score -= 5

        # Event adjustments
        base_score += analysis.calendar_adjustment

        analysis.calendar_score = max(0, min(100, base_score))

        return analysis

    # =========================================================================
    # CROSS-ASSET ANALYSIS
    # =========================================================================

    def analyze_cross_assets(self) -> CrossAssetSignals:
        """Analyze cross-asset signals for bond direction."""

        signals = CrossAssetSignals()

        try:
            # Fetch data
            tickers = ['^VIX', 'DX-Y.NYB', 'GC=F', 'SPY']
            data = yf.download(tickers, period='20d', progress=False, auto_adjust=True)

            if data.empty:
                return signals

            close = data['Close']

            # VIX
            if '^VIX' in close.columns:
                vix = close['^VIX']
                signals.vix_level = float(vix.iloc[-1])
                if len(vix) >= 2:
                    signals.vix_change_1d = float(vix.iloc[-1] - vix.iloc[-2])

                if signals.vix_level >= 30:
                    signals.vix_signal = "HIGH_FEAR"
                elif signals.vix_level >= 20:
                    signals.vix_signal = "ELEVATED"
                elif signals.vix_level <= 12:
                    signals.vix_signal = "COMPLACENT"
                else:
                    signals.vix_signal = "NORMAL"

            # Dollar Index
            if 'DX-Y.NYB' in close.columns:
                dxy = close['DX-Y.NYB']
                signals.dxy_level = float(dxy.iloc[-1])
                if len(dxy) >= 5:
                    signals.dxy_change_5d = float(dxy.iloc[-1] - dxy.iloc[-5])

                if signals.dxy_change_5d:
                    if signals.dxy_change_5d > 1:
                        signals.dxy_signal = "STRONG_DOLLAR"
                    elif signals.dxy_change_5d < -1:
                        signals.dxy_signal = "WEAK_DOLLAR"

            # Gold
            if 'GC=F' in close.columns:
                gold = close['GC=F']
                signals.gold_price = float(gold.iloc[-1])
                if len(gold) >= 5:
                    signals.gold_change_5d = float((gold.iloc[-1] / gold.iloc[-5] - 1) * 100)

                if signals.gold_change_5d:
                    if signals.gold_change_5d > 2:
                        signals.gold_signal = "RISK_OFF"
                    elif signals.gold_change_5d < -2:
                        signals.gold_signal = "RISK_ON"

            # SPY
            if 'SPY' in close.columns:
                spy = close['SPY']
                if len(spy) >= 5:
                    signals.spy_change_5d = float((spy.iloc[-1] / spy.iloc[-5] - 1) * 100)

            # Determine risk regime
            risk_off_count = 0
            risk_on_count = 0

            if signals.vix_signal in ["HIGH_FEAR", "ELEVATED"]:
                risk_off_count += 1
            elif signals.vix_signal == "COMPLACENT":
                risk_on_count += 1

            if signals.gold_signal == "RISK_OFF":
                risk_off_count += 1
            elif signals.gold_signal == "RISK_ON":
                risk_on_count += 1

            if signals.spy_change_5d and signals.spy_change_5d < -3:
                risk_off_count += 1
            elif signals.spy_change_5d and signals.spy_change_5d > 3:
                risk_on_count += 1

            if risk_off_count >= 2:
                signals.risk_regime = "RISK_OFF"
            elif risk_on_count >= 2:
                signals.risk_regime = "RISK_ON"
            else:
                signals.risk_regime = "NEUTRAL"

            # Calculate cross-asset score
            # Risk-off = bullish bonds, Risk-on = bearish bonds
            score = 50

            if signals.risk_regime == "RISK_OFF":
                score += 15
            elif signals.risk_regime == "RISK_ON":
                score -= 15

            if signals.vix_signal == "HIGH_FEAR":
                score += 10
            elif signals.vix_signal == "COMPLACENT":
                score -= 5

            signals.cross_asset_score = max(0, min(100, score))

        except Exception as e:
            logger.error(f"Error analyzing cross-assets: {e}")

        return signals

    # =========================================================================
    # MAIN PREDICTION
    # =========================================================================

    def generate_prediction(self, ticker: str) -> BondPrediction:
        """Generate comprehensive prediction for a bond ETF."""

        if ticker not in BOND_ETFS:
            raise ValueError(f"Unknown ticker: {ticker}")

        etf_info = BOND_ETFS[ticker]
        duration = etf_info['duration']
        leverage = etf_info.get('leverage', 1.0)

        # Get current price
        try:
            data = yf.download(ticker, period='5d', progress=False, auto_adjust=True)
            current_price = float(data['Close'].iloc[-1].iloc[0]) if not data.empty else 0
        except:
            current_price = 0

        prediction = BondPrediction(
            ticker=ticker,
            name=etf_info['name'],
            prediction_time=datetime.now(),
            current_price=current_price,
            duration=duration,
        )

        # Analyze all components
        yield_momentum = self.analyze_yield_momentum()
        economic_surprises = self.analyze_economic_surprises()
        calendar_analysis = self.analyze_calendar_effects()
        cross_asset = self.analyze_cross_assets()

        prediction.yield_momentum = yield_momentum
        prediction.economic_surprises = economic_surprises
        prediction.calendar_analysis = calendar_analysis
        prediction.cross_asset = cross_asset

        # Convert yield momentum score to bond score (inverse)
        # High yield momentum = bearish bonds
        prediction.yield_momentum_score = 100 - yield_momentum.momentum_score

        # Economic score from surprises
        if economic_surprises:
            avg_impact = sum(s.bond_impact_score for s in economic_surprises) / len(economic_surprises)
            prediction.economic_score = int(50 + avg_impact / 2)

        prediction.calendar_score = calendar_analysis.calendar_score
        prediction.cross_asset_score = cross_asset.cross_asset_score

        # Seasonality score
        season = BOND_SEASONALITY.get(date.today().month, {})
        if season.get('bias') == 'BULLISH':
            prediction.seasonality_score = 70
        elif season.get('bias') == 'SLIGHT_BULLISH':
            prediction.seasonality_score = 60
        elif season.get('bias') == 'BEARISH':
            prediction.seasonality_score = 30
        elif season.get('bias') == 'SLIGHT_BEARISH':
            prediction.seasonality_score = 40
        else:
            prediction.seasonality_score = 50

        # Calculate composite score
        weights = {
            'yield_momentum': 0.35,
            'economic': 0.20,
            'calendar': 0.10,
            'seasonality': 0.10,
            'cross_asset': 0.15,
            'technical': 0.10,
        }

        prediction.composite_score = int(
            prediction.yield_momentum_score * weights['yield_momentum'] +
            prediction.economic_score * weights['economic'] +
            prediction.calendar_score * weights['calendar'] +
            prediction.seasonality_score * weights['seasonality'] +
            prediction.cross_asset_score * weights['cross_asset'] +
            prediction.technical_score * weights['technical']
        )

        # Adjust for leverage
        if leverage < 0:  # Inverse ETF
            prediction.composite_score = 100 - prediction.composite_score

        # Determine signal
        if prediction.composite_score >= 70:
            prediction.signal = PredictionSignal.STRONG_BUY
        elif prediction.composite_score >= 58:
            prediction.signal = PredictionSignal.BUY
        elif prediction.composite_score <= 30:
            prediction.signal = PredictionSignal.STRONG_SELL
        elif prediction.composite_score <= 42:
            prediction.signal = PredictionSignal.SELL
        else:
            prediction.signal = PredictionSignal.HOLD

        # Calculate predicted returns
        if yield_momentum.forecast_5d_bps:
            # Price change ≈ -duration * yield change
            expected_yield_change_pct = yield_momentum.forecast_5d_bps / 100

            # For regular bond ETFs: rising yields = falling prices
            # base_return = -duration * yield_change
            # Example: TLT with duration=17.5, yields up 0.07% → -17.5 * 0.07 = -1.225%
            #
            # For inverse ETFs like TBT:
            # Duration is ALREADY negative (-35 for 2x inverse)
            # So: -(-35) * 0.07 = +2.45% (correctly positive when yields rise)
            #
            # Leverage is applied on top:
            # TBT leverage is -2, but since duration already reflects the inverse nature,
            # we use absolute leverage for the multiplier

            base_return = -duration * expected_yield_change_pct

            # Apply leverage multiplier (always use absolute value since direction
            # is already embedded in the duration sign)
            leverage = etf_info.get('leverage', 1.0)
            leverage_multiplier = abs(leverage) if leverage != 0 else 1.0

            # For leveraged products, the duration already includes leverage effect
            # So we don't multiply again - just use base_return directly
            # The -35 duration for TBT already reflects 2x inverse exposure
            prediction.predicted_return_5d = base_return
            prediction.predicted_return_1d = prediction.predicted_return_5d / 5

            if current_price > 0:
                prediction.predicted_price_5d = current_price * (1 + prediction.predicted_return_5d / 100)
                prediction.predicted_price_1d = current_price * (1 + prediction.predicted_return_1d / 100)

        # Confidence
        prediction.confidence = yield_momentum.confidence * 0.5 + 0.3
        if calendar_analysis.expected_volatility == "EXTREME":
            prediction.confidence *= 0.7
        elif calendar_analysis.expected_volatility == "HIGH":
            prediction.confidence *= 0.85

        prediction.expected_volatility = calendar_analysis.expected_volatility

        # Build reasoning
        # Bull factors
        if prediction.yield_momentum_score >= 60:
            prediction.bull_factors.append(f"Yields {yield_momentum.direction.value} - bullish for bonds")
        if prediction.economic_score >= 60:
            prediction.bull_factors.append("Economic data supportive for bonds")
        if cross_asset.risk_regime == "RISK_OFF":
            prediction.bull_factors.append("Risk-off environment - flight to safety")
        if prediction.seasonality_score >= 60:
            prediction.bull_factors.append(f"Seasonally bullish month ({season.get('note', '')})")

        # Bear factors
        if prediction.yield_momentum_score <= 40:
            prediction.bear_factors.append(f"Yields {yield_momentum.direction.value} - bearish for bonds")
        if prediction.economic_score <= 40:
            prediction.bear_factors.append("Strong economic data pressuring bonds")
        if cross_asset.risk_regime == "RISK_ON":
            prediction.bear_factors.append("Risk-on environment - rotation out of bonds")
        if prediction.seasonality_score <= 40:
            prediction.bear_factors.append(f"Seasonally weak month ({season.get('note', '')})")

        # Risk events
        prediction.risk_events_ahead = calendar_analysis.alerts

        # Key drivers
        if abs(prediction.yield_momentum_score - 50) > 15:
            prediction.key_drivers.append(f"Yield momentum: {yield_momentum.direction.value}")
        if economic_surprises:
            for s in economic_surprises[:2]:
                if abs(s.bond_impact_score) > 10:
                    prediction.key_drivers.append(f"{s.event_name}: {s.bond_impact_label}")

        # Recommendation
        if prediction.signal == PredictionSignal.STRONG_BUY:
            prediction.recommendation = f"STRONG BUY {ticker}. Duration {duration:.1f}y. Expected +{prediction.predicted_return_5d:.1f}% in 5 days."
        elif prediction.signal == PredictionSignal.BUY:
            prediction.recommendation = f"BUY {ticker}. Yields showing downward momentum."
        elif prediction.signal == PredictionSignal.STRONG_SELL:
            prediction.recommendation = f"AVOID {ticker}. Rising yield environment. Consider TBT or reduce duration."
        elif prediction.signal == PredictionSignal.SELL:
            prediction.recommendation = f"REDUCE {ticker}. Yields trending higher."
        else:
            prediction.recommendation = f"HOLD {ticker}. Mixed signals - wait for clarity."

        # Alpha signal for integration (-1 to +1)
        prediction.alpha_signal = (prediction.composite_score - 50) / 50

        return prediction

    def generate_all_predictions(self) -> Dict[str, BondPrediction]:
        """Generate predictions for all bond ETFs."""
        predictions = {}

        for ticker in BOND_ETFS:
            try:
                predictions[ticker] = self.generate_prediction(ticker)
            except Exception as e:
                logger.error(f"Error generating prediction for {ticker}: {e}")

        return predictions

    # =========================================================================
    # INTEGRATION WITH ALPHA MODEL
    # =========================================================================

    def get_rate_signal_for_alpha(self) -> Dict[str, Any]:
        """
        Get rate/bond signal to feed into the main alpha model.

        Returns dict with:
        - rate_direction: RISING, FALLING, STABLE
        - rate_momentum_score: 0-100
        - bond_signal: -1 to +1 (positive = rates falling = bullish stocks generally)
        - duration_recommendation: SHORT, NEUTRAL, LONG
        - risk_regime: RISK_ON, RISK_OFF, NEUTRAL
        """

        yield_momentum = self.analyze_yield_momentum()
        cross_asset = self.analyze_cross_assets()
        calendar = self.analyze_calendar_effects()

        # Rate direction affects different sectors differently
        result = {
            'rate_direction': yield_momentum.direction.value,
            'rate_momentum_score': yield_momentum.momentum_score,
            'yield_30y': yield_momentum.current_yield_30y,
            'yield_10y': yield_momentum.current_yield_10y,
            'yield_change_5d_bps': (yield_momentum.change_5d_30y or 0) * 100,

            # Bond signal: positive = bullish for bonds (rates falling)
            'bond_signal': (50 - yield_momentum.momentum_score) / 50,

            # Duration recommendation
            'duration_recommendation': 'SHORT' if yield_momentum.momentum_score > 60 else 'LONG' if yield_momentum.momentum_score < 40 else 'NEUTRAL',

            # Risk regime
            'risk_regime': cross_asset.risk_regime,
            'vix_level': cross_asset.vix_level,

            # Calendar
            'is_fomc_week': calendar.is_fomc_week,
            'days_to_fomc': calendar.days_to_fomc,
            'expected_volatility': calendar.expected_volatility,

            # Sector implications
            'sector_impacts': self._calculate_sector_impacts(yield_momentum),
        }

        return result

    def _calculate_sector_impacts(self, yield_momentum: YieldMomentum) -> Dict[str, float]:
        """Calculate how rate moves affect different sectors."""

        # Sector sensitivity to rising rates
        # Positive = benefits from rising rates, Negative = hurt by rising rates
        SECTOR_RATE_SENSITIVITY = {
            'Financials': 0.6,       # Banks benefit from higher rates
            'Real Estate': -0.7,     # REITs hurt by higher rates
            'Utilities': -0.6,       # Bond proxies hurt
            'Consumer Staples': -0.3,
            'Technology': -0.2,      # Growth hurt by higher discount rate
            'Healthcare': -0.1,
            'Industrials': 0.1,
            'Materials': 0.2,
            'Energy': 0.3,
            'Consumer Discretionary': -0.2,
            'Communication Services': -0.1,
        }

        # Rate direction score: >50 = rising, <50 = falling
        rate_factor = (yield_momentum.momentum_score - 50) / 50  # -1 to +1

        impacts = {}
        for sector, sensitivity in SECTOR_RATE_SENSITIVITY.items():
            # Impact = rate_direction * sector_sensitivity
            # Positive impact = bullish for sector
            impacts[sector] = round(rate_factor * sensitivity, 2)

        return impacts


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_engine_instance = None

def get_bond_prediction_engine() -> BondPredictionEngine:
    """Get singleton instance of BondPredictionEngine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = BondPredictionEngine()
    return _engine_instance


def predict_bond(ticker: str) -> BondPrediction:
    """Quick function to predict a single bond ETF."""
    engine = get_bond_prediction_engine()
    return engine.generate_prediction(ticker)


def get_rate_signals() -> Dict[str, Any]:
    """Get rate signals for alpha model integration."""
    engine = get_bond_prediction_engine()
    return engine.get_rate_signal_for_alpha()


def get_calendar_alerts() -> List[str]:
    """Get today's calendar alerts for bonds."""
    engine = get_bond_prediction_engine()
    calendar = engine.analyze_calendar_effects()
    return calendar.alerts


# =============================================================================
# MAIN / TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BOND PREDICTION ENGINE - TEST")
    print("=" * 70)

    engine = BondPredictionEngine()

    # Test yield momentum
    print("\n📈 YIELD MOMENTUM:")
    momentum = engine.analyze_yield_momentum()
    print(f"  30Y Yield: {momentum.current_yield_30y:.2f}%")
    print(f"  Direction: {momentum.direction.value}")
    print(f"  Momentum Score: {momentum.momentum_score}/100")
    print(f"  5D Change: {(momentum.change_5d_30y or 0)*100:.0f}bps")
    print(f"  Forecast 5D: {momentum.forecast_5d_bps:.0f}bps" if momentum.forecast_5d_bps else "  Forecast: N/A")

    # Test calendar
    print("\n📅 CALENDAR EFFECTS:")
    calendar = engine.analyze_calendar_effects()
    print(f"  Today: {calendar.today}")
    print(f"  Seasonality: {calendar.seasonality_bias} - {calendar.seasonality_note}")
    print(f"  Days to FOMC: {calendar.days_to_fomc}")
    print(f"  Expected Vol: {calendar.expected_volatility}")
    if calendar.alerts:
        print("  Alerts:")
        for alert in calendar.alerts:
            print(f"    {alert}")

    # Test cross-asset
    print("\n🌐 CROSS-ASSET:")
    cross = engine.analyze_cross_assets()
    print(f"  VIX: {cross.vix_level:.1f} ({cross.vix_signal})" if cross.vix_level else "  VIX: N/A")
    print(f"  Risk Regime: {cross.risk_regime}")

    # Test predictions
    print("\n🔮 PREDICTIONS:")
    for ticker in ['TLT', 'ZROZ', 'TBT']:
        try:
            pred = engine.generate_prediction(ticker)
            signal_emoji = {
                PredictionSignal.STRONG_BUY: "🟢🟢",
                PredictionSignal.BUY: "🟢",
                PredictionSignal.HOLD: "🟡",
                PredictionSignal.SELL: "🔴",
                PredictionSignal.STRONG_SELL: "🔴🔴",
            }
            print(f"\n  {ticker}:")
            print(f"    Price: ${pred.current_price:.2f}")
            print(f"    Signal: {signal_emoji.get(pred.signal, '')} {pred.signal.value}")
            print(f"    Score: {pred.composite_score}/100")
            print(f"    Predicted 5D: {pred.predicted_return_5d:+.1f}%" if pred.predicted_return_5d else "    Predicted: N/A")
            print(f"    Confidence: {pred.confidence:.0%}")
            print(f"    Recommendation: {pred.recommendation}")
        except Exception as e:
            print(f"  {ticker}: Error - {e}")

    # Test alpha integration
    print("\n🔗 ALPHA MODEL INTEGRATION:")
    rate_signal = engine.get_rate_signal_for_alpha()
    print(f"  Rate Direction: {rate_signal['rate_direction']}")
    print(f"  Bond Signal: {rate_signal['bond_signal']:+.2f}")
    print(f"  Duration Rec: {rate_signal['duration_recommendation']}")
    print(f"  Risk Regime: {rate_signal['risk_regime']}")
    print("  Sector Impacts:")
    for sector, impact in sorted(rate_signal['sector_impacts'].items(), key=lambda x: x[1], reverse=True):
        emoji = "📈" if impact > 0 else "📉" if impact < 0 else "➡️"
        print(f"    {emoji} {sector}: {impact:+.2f}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)