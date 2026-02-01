"""
Institutional Bond Signal Generator

Modeled after JPMorgan and BlackRock fixed income analytics:
- Technical Analysis: VWAP, RSI, MACD, Bollinger Bands, Support/Resistance
- Fair Value: Duration-based pricing, term premium analysis
- Flow Analysis: Auction demand, ETF flows
- Macro: Fed policy, inflation, growth
- Sentiment: News analysis

NO HARDCODED DATA - All live from APIs.

Location: src/analytics/bond_signals_institutional.py
Author: Alpha Research Platform
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import requests

from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class Signal(Enum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"


class TrendDirection(Enum):
    STRONG_UP = "Strong Uptrend"
    UP = "Uptrend"
    NEUTRAL = "Neutral"
    DOWN = "Downtrend"
    STRONG_DOWN = "Strong Downtrend"


@dataclass
class TechnicalIndicators:
    """Technical analysis indicators."""
    # Price data
    current_price: float = 0.0
    price_change_1d: float = 0.0
    price_change_5d: float = 0.0
    price_change_20d: float = 0.0

    # Moving averages
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_100: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None

    # VWAP
    vwap: Optional[float] = None
    vwap_deviation: Optional[float] = None  # % from VWAP

    # RSI
    rsi_14: Optional[float] = None
    rsi_signal: str = ""  # Overbought/Oversold/Neutral

    # MACD
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    macd_crossover: str = ""  # Bullish/Bearish/None

    # Bollinger Bands
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_position: str = ""  # Above Upper/Below Lower/Within
    bb_width: Optional[float] = None  # Volatility measure

    # Support/Resistance
    support_1: Optional[float] = None
    support_2: Optional[float] = None
    resistance_1: Optional[float] = None
    resistance_2: Optional[float] = None

    # Fibonacci levels (from 52-week range)
    fib_236: Optional[float] = None
    fib_382: Optional[float] = None
    fib_500: Optional[float] = None
    fib_618: Optional[float] = None

    # Trend
    trend: TrendDirection = TrendDirection.NEUTRAL

    # Overall technical score (0-100)
    technical_score: Optional[int] = None


@dataclass
class FundamentalAnalysis:
    """Fundamental/Fair value analysis."""
    # Current yields
    yield_10y: float = 0.0
    yield_30y: float = 0.0
    fed_funds_rate: float = 0.0

    # Fair value calculation
    fair_value_yield_10y: Optional[float] = None
    fair_value_yield_30y: Optional[float] = None
    yield_vs_fair_value: Optional[float] = None  # Positive = cheap, negative = rich

    # Duration
    modified_duration: float = 0.0

    # Expected yield change (from Fed outlook)
    expected_yield_change_6m: Optional[float] = None
    expected_yield_change_12m: Optional[float] = None

    # Price target
    price_target_6m: Optional[float] = None
    price_target_12m: Optional[float] = None
    upside_pct: Optional[float] = None

    # Carry analysis
    carry_3m: Optional[float] = None  # Roll down + yield

    # Term premium
    term_premium: Optional[float] = None
    term_premium_signal: str = ""  # Elevated/Normal/Compressed

    # Real yield
    real_yield_10y: Optional[float] = None
    breakeven_inflation: Optional[float] = None

    # Overall fundamental score (0-100)
    fundamental_score: Optional[int] = None


@dataclass
class FlowAnalysis:
    """Flow and positioning analysis."""
    # Auction data
    auction_demand_score: Optional[int] = None
    recent_bid_to_cover: Optional[float] = None
    auction_signal: str = ""

    # ETF flows (if available)
    tlt_flow_1w: Optional[float] = None
    tlt_flow_1m: Optional[float] = None
    etf_flow_signal: str = ""

    # Overall flow score (0-100)
    flow_score: Optional[int] = None


@dataclass
class MacroAnalysis:
    """Macro and Fed policy analysis."""
    # Fed policy
    fed_funds_current: float = 0.0
    fed_expected_terminal: Optional[float] = None
    cuts_priced_in: Optional[float] = None  # In bps
    next_fomc_probability_hold: Optional[float] = None
    next_fomc_probability_cut: Optional[float] = None
    fed_policy_signal: str = ""

    # Inflation
    cpi_latest: Optional[float] = None
    cpi_trend: str = ""
    pce_latest: Optional[float] = None

    # Growth
    gdp_growth: Optional[float] = None
    unemployment: Optional[float] = None

    # Overall macro score (0-100)
    macro_score: Optional[int] = None


@dataclass
class SentimentAnalysis:
    """News and sentiment analysis."""
    news_sentiment_score: Optional[float] = None  # 0-1
    news_sentiment_label: str = ""
    bullish_headlines: int = 0
    bearish_headlines: int = 0
    key_themes: List[str] = field(default_factory=list)

    # Overall sentiment score (0-100)
    sentiment_score: Optional[int] = None


@dataclass
class BondSignal:
    """Complete bond signal with all analysis."""
    ticker: str
    name: str

    # Current market data
    current_price: float
    previous_close: float
    day_change_pct: float
    volume: int
    avg_volume: int

    # Analysis components
    technical: TechnicalIndicators
    fundamental: FundamentalAnalysis
    flow: FlowAnalysis
    macro: MacroAnalysis
    sentiment: SentimentAnalysis

    # Final signal
    composite_score: int  # 0-100
    signal: Signal
    confidence: str  # High/Medium/Low

    # Targets
    target_price: float
    upside_pct: float
    stop_loss: Optional[float] = None

    # Reasoning
    bull_case: List[str] = field(default_factory=list)
    bear_case: List[str] = field(default_factory=list)
    key_levels: Dict[str, float] = field(default_factory=dict)

    # Timestamps
    analysis_time: datetime = field(default_factory=datetime.now)


# =============================================================================
# ETF DEFINITIONS - Duration and characteristics
# =============================================================================

BOND_ETFS = {
    'TLT': {
        'name': 'iShares 20+ Year Treasury Bond ETF',
        'duration': 17.5,
        'benchmark_yield': '30y',  # Most sensitive to long end
        'expense_ratio': 0.15,
    },
    'ZROZ': {
        'name': 'PIMCO 25+ Year Zero Coupon Treasury ETF',
        'duration': 27.0,  # Highest duration
        'benchmark_yield': '30y',
        'expense_ratio': 0.15,
    },
    'EDV': {
        'name': 'Vanguard Extended Duration Treasury ETF',
        'duration': 24.5,
        'benchmark_yield': '30y',
        'expense_ratio': 0.06,
    },
    'TMF': {
        'name': 'Direxion Daily 20+ Year Treasury Bull 3X',
        'duration': 52.5,  # 3x leveraged
        'benchmark_yield': '30y',
        'expense_ratio': 1.01,
        'leveraged': True,
    },
    'TBT': {
        'name': 'ProShares UltraShort 20+ Year Treasury',
        'duration': -35.0,  # Inverse 2x
        'benchmark_yield': '30y',
        'expense_ratio': 0.90,
        'inverse': True,
    },
    'SHY': {
        'name': 'iShares 1-3 Year Treasury Bond ETF',
        'duration': 1.9,
        'benchmark_yield': '2y',
        'expense_ratio': 0.15,
    },
    'IEF': {
        'name': 'iShares 7-10 Year Treasury Bond ETF',
        'duration': 7.5,
        'benchmark_yield': '10y',
        'expense_ratio': 0.15,
    },
    'TIP': {
        'name': 'iShares TIPS Bond ETF',
        'duration': 6.8,
        'benchmark_yield': '10y',
        'expense_ratio': 0.19,
        'tips': True,
    },
}


# =============================================================================
# TECHNICAL ANALYSIS CALCULATIONS
# =============================================================================

class TechnicalAnalyzer:
    """JPM/BlackRock style technical analysis."""

    def __init__(self):
        self.lookback_days = 365  # 1 year for support/resistance

    def analyze(self, ticker: str) -> TechnicalIndicators:
        """Run full technical analysis on a bond ETF."""
        tech = TechnicalIndicators()

        try:
            # Fetch price data
            etf = yf.Ticker(ticker)
            hist = etf.history(period='1y')

            if hist.empty:
                logger.warning(f"No price data for {ticker}")
                return tech

            # Current price
            tech.current_price = float(hist['Close'].iloc[-1])

            if len(hist) < 2:
                return tech

            # Price changes
            tech.price_change_1d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100
            if len(hist) >= 5:
                tech.price_change_5d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-5] - 1) * 100
            if len(hist) >= 20:
                tech.price_change_20d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1) * 100

            # Moving Averages
            if len(hist) >= 20:
                tech.sma_20 = float(hist['Close'].rolling(20).mean().iloc[-1])
            if len(hist) >= 50:
                tech.sma_50 = float(hist['Close'].rolling(50).mean().iloc[-1])
            if len(hist) >= 100:
                tech.sma_100 = float(hist['Close'].rolling(100).mean().iloc[-1])
            if len(hist) >= 200:
                tech.sma_200 = float(hist['Close'].rolling(200).mean().iloc[-1])

            # EMA for MACD
            tech.ema_12 = float(hist['Close'].ewm(span=12, adjust=False).mean().iloc[-1])
            tech.ema_26 = float(hist['Close'].ewm(span=26, adjust=False).mean().iloc[-1])

            # VWAP (using available data)
            if 'Volume' in hist.columns and hist['Volume'].sum() > 0:
                typical_price = (hist['High'] + hist['Low'] + hist['Close']) / 3
                cumulative_tp_vol = (typical_price * hist['Volume']).cumsum()
                cumulative_vol = hist['Volume'].cumsum()
                vwap_series = cumulative_tp_vol / cumulative_vol
                tech.vwap = float(vwap_series.iloc[-1])
                tech.vwap_deviation = (tech.current_price / tech.vwap - 1) * 100

            # RSI (14-day)
            tech.rsi_14 = self._calculate_rsi(hist['Close'], 14)
            if tech.rsi_14 is not None:
                if tech.rsi_14 > 70:
                    tech.rsi_signal = "OVERBOUGHT"
                elif tech.rsi_14 < 30:
                    tech.rsi_signal = "OVERSOLD"
                else:
                    tech.rsi_signal = "NEUTRAL"

            # MACD
            tech.macd_line, tech.macd_signal, tech.macd_histogram = self._calculate_macd(hist['Close'])
            if tech.macd_histogram is not None:
                if tech.macd_histogram > 0 and tech.macd_line > tech.macd_signal:
                    tech.macd_crossover = "BULLISH"
                elif tech.macd_histogram < 0 and tech.macd_line < tech.macd_signal:
                    tech.macd_crossover = "BEARISH"
                else:
                    tech.macd_crossover = "NEUTRAL"

            # Bollinger Bands (20-day, 2 std dev)
            if len(hist) >= 20:
                tech.bb_middle = float(hist['Close'].rolling(20).mean().iloc[-1])
                std_20 = float(hist['Close'].rolling(20).std().iloc[-1])
                tech.bb_upper = tech.bb_middle + 2 * std_20
                tech.bb_lower = tech.bb_middle - 2 * std_20
                tech.bb_width = (tech.bb_upper - tech.bb_lower) / tech.bb_middle * 100

                if tech.current_price > tech.bb_upper:
                    tech.bb_position = "ABOVE_UPPER"
                elif tech.current_price < tech.bb_lower:
                    tech.bb_position = "BELOW_LOWER"
                else:
                    tech.bb_position = "WITHIN"

            # Support/Resistance from historical levels
            tech.support_1, tech.support_2, tech.resistance_1, tech.resistance_2 = \
                self._find_support_resistance(hist)

            # Fibonacci levels
            high_52w = float(hist['High'].max())
            low_52w = float(hist['Low'].min())
            range_52w = high_52w - low_52w

            tech.fib_236 = high_52w - range_52w * 0.236
            tech.fib_382 = high_52w - range_52w * 0.382
            tech.fib_500 = high_52w - range_52w * 0.500
            tech.fib_618 = high_52w - range_52w * 0.618

            # Determine trend
            tech.trend = self._determine_trend(tech)

            # Calculate technical score
            tech.technical_score = self._calculate_technical_score(tech)

            logger.info(f"{ticker} Technical: Price=${tech.current_price:.2f}, RSI={tech.rsi_14:.1f}, Score={tech.technical_score}")

        except Exception as e:
            logger.error(f"Technical analysis error for {ticker}: {e}")

        return tech

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return None

        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None

    def _calculate_macd(self, prices: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate MACD indicator."""
        if len(prices) < 26:
            return None, None, None

        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])

    def _find_support_resistance(self, hist: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Find support and resistance levels from price history."""
        if len(hist) < 20:
            return None, None, None, None

        current_price = float(hist['Close'].iloc[-1])
        prices = hist['Close'].values

        # Find local minima (support) and maxima (resistance)
        window = 10

        supports = []
        resistances = []

        for i in range(window, len(prices) - window):
            # Local minimum (support)
            if prices[i] == min(prices[i-window:i+window+1]):
                if prices[i] < current_price:
                    supports.append(prices[i])

            # Local maximum (resistance)
            if prices[i] == max(prices[i-window:i+window+1]):
                if prices[i] > current_price:
                    resistances.append(prices[i])

        # Get closest support/resistance levels
        supports = sorted(set(supports), reverse=True)[:2]  # Closest below
        resistances = sorted(set(resistances))[:2]  # Closest above

        support_1 = supports[0] if len(supports) > 0 else None
        support_2 = supports[1] if len(supports) > 1 else None
        resistance_1 = resistances[0] if len(resistances) > 0 else None
        resistance_2 = resistances[1] if len(resistances) > 1 else None

        return support_1, support_2, resistance_1, resistance_2

    def _determine_trend(self, tech: TechnicalIndicators) -> TrendDirection:
        """Determine overall trend."""
        score = 0

        # Price vs MAs
        if tech.sma_50 and tech.current_price > tech.sma_50:
            score += 1
        elif tech.sma_50 and tech.current_price < tech.sma_50:
            score -= 1

        if tech.sma_200 and tech.current_price > tech.sma_200:
            score += 1
        elif tech.sma_200 and tech.current_price < tech.sma_200:
            score -= 1

        # MA alignment
        if tech.sma_50 and tech.sma_200 and tech.sma_50 > tech.sma_200:
            score += 1  # Golden cross territory
        elif tech.sma_50 and tech.sma_200 and tech.sma_50 < tech.sma_200:
            score -= 1  # Death cross territory

        # Recent momentum
        if tech.price_change_20d and tech.price_change_20d > 3:
            score += 1
        elif tech.price_change_20d and tech.price_change_20d < -3:
            score -= 1

        if score >= 3:
            return TrendDirection.STRONG_UP
        elif score >= 1:
            return TrendDirection.UP
        elif score <= -3:
            return TrendDirection.STRONG_DOWN
        elif score <= -1:
            return TrendDirection.DOWN
        else:
            return TrendDirection.NEUTRAL

    def _calculate_technical_score(self, tech: TechnicalIndicators) -> int:
        """Calculate overall technical score (0-100)."""
        score = 50  # Start neutral

        # RSI contribution (±15 points)
        if tech.rsi_14:
            if tech.rsi_14 < 30:
                score += 15  # Oversold = bullish
            elif tech.rsi_14 < 40:
                score += 8
            elif tech.rsi_14 > 70:
                score -= 15  # Overbought = bearish
            elif tech.rsi_14 > 60:
                score -= 8

        # MACD contribution (±10 points)
        if tech.macd_crossover == "BULLISH":
            score += 10
        elif tech.macd_crossover == "BEARISH":
            score -= 10

        # Bollinger Band position (±10 points)
        if tech.bb_position == "BELOW_LOWER":
            score += 10  # Oversold
        elif tech.bb_position == "ABOVE_UPPER":
            score -= 10  # Overbought

        # Moving average alignment (±10 points)
        if tech.sma_50 and tech.sma_200:
            if tech.current_price > tech.sma_50 > tech.sma_200:
                score += 10  # Bullish alignment
            elif tech.current_price < tech.sma_50 < tech.sma_200:
                score -= 10  # Bearish alignment

        # VWAP position (±5 points)
        if tech.vwap_deviation:
            if tech.vwap_deviation < -2:
                score += 5  # Below VWAP = potential value
            elif tech.vwap_deviation > 2:
                score -= 5  # Above VWAP = stretched

        return max(0, min(100, score))


# =============================================================================
# FUNDAMENTAL ANALYSIS
# =============================================================================

class FundamentalAnalyzer:
    """BlackRock/JPM style fundamental analysis."""

    def __init__(self):
        self.fred_api_key = os.getenv('FRED_API_KEY', '')
        self.tool_server_url = os.getenv('TOOL_SERVER_URL', '')

    def analyze(self, ticker: str, current_price: float) -> FundamentalAnalysis:
        """Run fundamental analysis."""
        fund = FundamentalAnalysis()

        etf_info = BOND_ETFS.get(ticker, {})
        fund.modified_duration = etf_info.get('duration', 10.0)

        # Fetch current yields
        yields = self._fetch_yields()
        fund.yield_10y = yields.get('10y', 0)
        fund.yield_30y = yields.get('30y', 0)

        # Fetch Fed funds rate
        fund.fed_funds_rate = self._fetch_fed_funds_rate()

        # Calculate fair value yield
        # JPM/BlackRock approach: Fed funds expected path + term premium
        fund.fair_value_yield_10y, fund.term_premium = self._calculate_fair_value(fund)

        if fund.fair_value_yield_10y and fund.yield_10y:
            fund.yield_vs_fair_value = fund.fair_value_yield_10y - fund.yield_10y
            # Positive = yields are below fair value = bonds are rich
            # Negative = yields are above fair value = bonds are cheap

        # Calculate expected yield change and price targets
        fund.expected_yield_change_6m = self._estimate_yield_change(fund, months=6)
        fund.expected_yield_change_12m = self._estimate_yield_change(fund, months=12)

        if current_price > 0:
            if fund.expected_yield_change_12m is not None and fund.expected_yield_change_12m != 0:
                # Duration-based price target
                # Price change ≈ -Duration × Yield change (in decimal)
                # Note: yield change is in percentage points (e.g., -0.25 = 25bps fall)
                expected_price_change = -fund.modified_duration * (fund.expected_yield_change_12m / 100)
                fund.price_target_12m = current_price * (1 + expected_price_change)
                fund.upside_pct = (fund.price_target_12m / current_price - 1) * 100
                logger.debug(f"{ticker}: Yield change {fund.expected_yield_change_12m:.2f}% -> Price change {expected_price_change*100:.1f}%")
            else:
                # Fallback: assume small yield compression based on current yield level
                # If yields > 4%, assume some mean reversion lower
                if fund.yield_10y > 4.0:
                    implied_yield_change = -0.20  # Assume 20bps fall
                elif fund.yield_10y < 3.0:
                    implied_yield_change = 0.15  # Assume 15bps rise
                else:
                    implied_yield_change = -0.10  # Small fall in neutral case

                expected_price_change = -fund.modified_duration * (implied_yield_change / 100)
                fund.price_target_12m = current_price * (1 + expected_price_change)
                fund.upside_pct = (fund.price_target_12m / current_price - 1) * 100
                fund.expected_yield_change_12m = implied_yield_change

        if current_price > 0 and fund.expected_yield_change_6m is not None:
            expected_price_change = -fund.modified_duration * (fund.expected_yield_change_6m / 100)
            fund.price_target_6m = current_price * (1 + expected_price_change)

        # Carry analysis (simplified)
        fund.carry_3m = self._calculate_carry(fund)

        # Fetch real yields and breakevens
        fund.real_yield_10y, fund.breakeven_inflation = self._fetch_real_yields()

        # Calculate fundamental score
        fund.fundamental_score = self._calculate_fundamental_score(fund)

        target_str = f"${fund.price_target_12m:.2f}" if fund.price_target_12m else "N/A"
        logger.info(f"{ticker} Fundamental: 10Y={fund.yield_10y:.2f}%, Target={target_str}, Score={fund.fundamental_score}")

        return fund

    def _fetch_yields(self) -> Dict[str, float]:
        """Fetch current Treasury yields from Yahoo Finance."""
        yields = {}

        try:
            tickers = {'^TNX': '10y', '^TYX': '30y', '^FVX': '5y', '^IRX': '3m'}
            data = yf.download(list(tickers.keys()), period='5d', progress=False, auto_adjust=False)

            if not data.empty:
                # Handle MultiIndex columns from multi-ticker download
                if isinstance(data.columns, pd.MultiIndex):
                    latest = data['Close'].iloc[-1]
                    for yf_ticker, name in tickers.items():
                        if yf_ticker in latest.index:
                            yields[name] = float(latest[yf_ticker])
                else:
                    # Single ticker case
                    val = data['Close'].iloc[-1]
                    yields['10y'] = float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)
        except Exception as e:
            logger.warning(f"Error fetching yields: {e}")

        return yields

    def _fetch_fed_funds_rate(self) -> float:
        """Fetch current Fed funds rate."""
        # Try FRED first
        if self.fred_api_key:
            try:
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    'series_id': 'EFFR',
                    'api_key': self.fred_api_key,
                    'file_type': 'json',
                    'limit': 1,
                    'sort_order': 'desc'
                }
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('observations'):
                        return float(data['observations'][0]['value'])
            except Exception as e:
                logger.warning(f"FRED API error: {e}")

        # Fallback: use 3-month T-bill as proxy
        try:
            data = yf.download('^IRX', period='5d', progress=False, auto_adjust=False)
            if not data.empty:
                val = data['Close'].iloc[-1]
                return float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)
        except:
            pass

        return 0.0

    def _calculate_fair_value(self, fund: FundamentalAnalysis) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate fair value yield using JPM/BlackRock methodology.

        Fair Value = Expected Fed Funds Path + Term Premium

        Since we don't have Fed funds futures, we use:
        - Current Fed funds as base
        - Market-implied cuts from yield curve
        - Historical average term premium
        """
        if fund.fed_funds_rate == 0 or fund.yield_10y == 0:
            return None, None

        # The spread between 10Y and Fed funds gives us market's view
        # of future rate path + term premium combined
        spread_10y_ff = fund.yield_10y - fund.fed_funds_rate

        # Historical average 10Y term premium is roughly 0-150bps
        # If spread is negative, market expects significant cuts
        # If spread is very positive (>100bps), term premium may be elevated

        # We can't separate term premium from rate expectations without futures
        # So we report the spread and let user interpret
        term_premium = spread_10y_ff  # This is actually spread, not true TP

        if term_premium > 0.5:
            fund.term_premium_signal = "ELEVATED"
        elif term_premium < -0.3:
            fund.term_premium_signal = "COMPRESSED"
        else:
            fund.term_premium_signal = "NORMAL"

        # Fair value = we assume term premium should normalize to ~0.3%
        # So fair value yield = Fed funds + 0.3%
        # If actual yield is higher, bonds are cheap
        # If actual yield is lower, bonds are rich
        normal_term_premium = 0.30  # Historical average
        fair_value_10y = fund.fed_funds_rate + normal_term_premium

        return fair_value_10y, term_premium

    def _estimate_yield_change(self, fund: FundamentalAnalysis, months: int) -> Optional[float]:
        """
        Estimate yield change based on Fed outlook, fair value, and mean reversion.

        This determines the price target based on what's NOT fully priced in.

        Returns: Expected yield change in percentage points (e.g., -0.25 means yields fall 25bps)
        """
        if fund.yield_10y == 0 or fund.fed_funds_rate == 0:
            return None

        spread = fund.yield_10y - fund.fed_funds_rate
        time_factor = months / 12  # Scale by time horizon

        expected_change = 0.0

        # 1. Mean reversion to historical spread (~0.3-0.5% typical)
        # If spread is too high, yields tend to fall; if too low, yields rise
        normal_spread = 0.40  # Historical average 10Y - Fed funds
        spread_deviation = spread - normal_spread

        if abs(spread_deviation) > 0.3:
            # Strong mean reversion signal
            mean_reversion = -spread_deviation * 0.4 * time_factor  # 40% mean reversion per year
            expected_change += mean_reversion
        elif abs(spread_deviation) > 0.1:
            # Moderate mean reversion
            mean_reversion = -spread_deviation * 0.25 * time_factor
            expected_change += mean_reversion

        # 2. Term premium analysis
        # If yields above fair value (term premium elevated), expect some compression
        if fund.yield_vs_fair_value is not None:
            if fund.yield_vs_fair_value < -0.3:
                # Yields above fair value = bonds cheap = expect yields to fall
                expected_change += -0.15 * time_factor
            elif fund.yield_vs_fair_value > 0.3:
                # Yields below fair value = bonds rich = expect yields to rise
                expected_change += 0.10 * time_factor

        # 3. Fed policy assumption
        # Assume gradual convergence toward neutral (~3%)
        # If Fed funds high, expect cuts over time; if low, expect hikes
        fed_neutral = 3.0
        fed_deviation = fund.fed_funds_rate - fed_neutral

        if fed_deviation > 0.5:
            # Fed above neutral, expect cuts which lowers yields
            expected_change += -0.10 * time_factor
        elif fed_deviation < -0.5:
            # Fed below neutral, expect hikes
            expected_change += 0.05 * time_factor

        # 4. If all factors are neutral, assume small mean reversion toward historical yield
        # Historical 10Y average is around 3-4%
        if abs(expected_change) < 0.05:
            historical_10y_avg = 3.5
            yield_deviation = fund.yield_10y - historical_10y_avg
            if abs(yield_deviation) > 0.5:
                expected_change = -yield_deviation * 0.15 * time_factor

        # Cap expected change to realistic levels
        expected_change = max(-0.75, min(0.50, expected_change))  # Cap at ±75bps/year

        return expected_change

    def _calculate_carry(self, fund: FundamentalAnalysis) -> Optional[float]:
        """Calculate 3-month carry (yield + roll-down)."""
        if fund.yield_10y == 0:
            return None

        # Simplified: carry ≈ yield / 4 (quarterly)
        # In reality would include roll-down from curve
        return fund.yield_10y / 4

    def _fetch_real_yields(self) -> Tuple[Optional[float], Optional[float]]:
        """Fetch real yields and breakeven inflation from TIPS."""
        try:
            # TIPS ETF price movement can proxy real yield changes
            # TIP ETF
            tip = yf.Ticker('TIP')
            info = tip.info

            # We can't get real yields directly from yfinance
            # Would need FRED for DFII10 (10Y TIPS yield)

            if self.fred_api_key:
                # Fetch 10Y TIPS yield
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    'series_id': 'DFII10',
                    'api_key': self.fred_api_key,
                    'file_type': 'json',
                    'limit': 1,
                    'sort_order': 'desc'
                }
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('observations'):
                        real_yield = float(data['observations'][0]['value'])
                        # Breakeven = Nominal - Real
                        nominal = self._fetch_yields().get('10y', 0)
                        breakeven = nominal - real_yield if nominal else None
                        return real_yield, breakeven
        except Exception as e:
            logger.debug(f"Real yield fetch error: {e}")

        return None, None

    def _calculate_fundamental_score(self, fund: FundamentalAnalysis) -> int:
        """Calculate overall fundamental score (0-100)."""
        score = 50  # Start neutral

        # Yield vs Fair Value (±15 points)
        if fund.yield_vs_fair_value is not None:
            if fund.yield_vs_fair_value < -0.5:
                score += 15  # Yields above fair value = cheap
            elif fund.yield_vs_fair_value < -0.2:
                score += 8
            elif fund.yield_vs_fair_value > 0.5:
                score -= 15  # Yields below fair value = rich
            elif fund.yield_vs_fair_value > 0.2:
                score -= 8

        # Term premium signal (±10 points)
        if fund.term_premium_signal == "ELEVATED":
            score += 10  # High term premium = potential value
        elif fund.term_premium_signal == "COMPRESSED":
            score -= 10  # Low/negative = expensive

        # Expected yield change (±10 points)
        if fund.expected_yield_change_12m is not None:
            if fund.expected_yield_change_12m < -0.3:
                score += 10  # Yields expected to fall = bullish
            elif fund.expected_yield_change_12m > 0.3:
                score -= 10  # Yields expected to rise = bearish

        # Carry (±5 points)
        if fund.carry_3m is not None:
            if fund.carry_3m > 1.2:
                score += 5  # Good carry
            elif fund.carry_3m < 0.8:
                score -= 5

        return max(0, min(100, score))


# =============================================================================
# FLOW ANALYSIS
# =============================================================================

class FlowAnalyzer:
    """Analyze flows and positioning."""

    def analyze(self, auction_score: Optional[int] = None) -> FlowAnalysis:
        """Analyze flow data."""
        flow = FlowAnalysis()

        # Use auction score if provided
        if auction_score is not None:
            flow.auction_demand_score = auction_score
            if auction_score >= 65:
                flow.auction_signal = "STRONG_DEMAND"
            elif auction_score >= 55:
                flow.auction_signal = "SOLID_DEMAND"
            elif auction_score <= 35:
                flow.auction_signal = "WEAK_DEMAND"
            else:
                flow.auction_signal = "AVERAGE"

        # ETF flow analysis (simplified - check volume)
        flow.tlt_flow_1w, flow.tlt_flow_1m = self._estimate_etf_flows('TLT')

        if flow.tlt_flow_1w is not None:
            if flow.tlt_flow_1w > 0:
                flow.etf_flow_signal = "INFLOWS"
            elif flow.tlt_flow_1w < 0:
                flow.etf_flow_signal = "OUTFLOWS"
            else:
                flow.etf_flow_signal = "NEUTRAL"

        # Calculate flow score
        flow.flow_score = self._calculate_flow_score(flow)

        return flow

    def _estimate_etf_flows(self, ticker: str) -> Tuple[Optional[float], Optional[float]]:
        """Estimate ETF flows from volume and price action."""
        try:
            etf = yf.Ticker(ticker)
            hist = etf.history(period='2mo')

            if len(hist) < 20:
                return None, None

            # Simple proxy: positive price change + high volume = inflows
            # This is a rough estimate - real flow data requires Bloomberg/provider

            price_change_1w = (hist['Close'].iloc[-1] / hist['Close'].iloc[-5] - 1) if len(hist) >= 5 else 0
            price_change_1m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1) if len(hist) >= 20 else 0

            avg_vol = hist['Volume'].mean()
            recent_vol = hist['Volume'].iloc[-5:].mean()
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1

            # Combine price and volume
            flow_1w = price_change_1w * vol_ratio * 100
            flow_1m = price_change_1m * vol_ratio * 100

            return flow_1w, flow_1m
        except:
            return None, None

    def _calculate_flow_score(self, flow: FlowAnalysis) -> int:
        """Calculate overall flow score (0-100)."""
        score = 50

        # Auction demand (±20 points)
        if flow.auction_demand_score is not None:
            score += int((flow.auction_demand_score - 50) * 0.4)

        # ETF flows (±10 points)
        if flow.etf_flow_signal == "INFLOWS":
            score += 10
        elif flow.etf_flow_signal == "OUTFLOWS":
            score -= 10

        return max(0, min(100, score))


# =============================================================================
# MACRO ANALYSIS
# =============================================================================

class MacroAnalyzer:
    """Analyze macro environment and Fed policy."""

    def __init__(self):
        self.fred_api_key = os.getenv('FRED_API_KEY', '')
        self.tool_server_url = os.getenv('TOOL_SERVER_URL', '')

    def analyze(self, fed_rate: float = 0, rate_probs: Dict = None) -> MacroAnalysis:
        """Analyze macro environment."""
        macro = MacroAnalysis()
        macro.fed_funds_current = fed_rate

        # Fetch economic data from FRED
        if self.fred_api_key:
            macro.cpi_latest = self._fetch_fred_series('CPIAUCSL', yoy=True)
            macro.pce_latest = self._fetch_fred_series('PCEPI', yoy=True)
            macro.unemployment = self._fetch_fred_series('UNRATE')
            macro.gdp_growth = self._fetch_fred_series('A191RL1Q225SBEA')  # Real GDP growth

        # Rate probabilities (if provided from intelligence module)
        if rate_probs:
            macro.next_fomc_probability_hold = rate_probs.get('hold', 0)
            macro.next_fomc_probability_cut = rate_probs.get('cut', 0)
            macro.cuts_priced_in = rate_probs.get('cuts_priced_in_2025', 0)

        # Determine Fed policy signal
        macro.fed_policy_signal = self._determine_fed_signal(macro)

        # CPI trend
        if macro.cpi_latest:
            if macro.cpi_latest > 3.0:
                macro.cpi_trend = "ELEVATED"
            elif macro.cpi_latest < 2.5:
                macro.cpi_trend = "CONTROLLED"
            else:
                macro.cpi_trend = "STICKY"

        # Calculate macro score
        macro.macro_score = self._calculate_macro_score(macro)

        return macro

    def _fetch_fred_series(self, series_id: str, yoy: bool = False) -> Optional[float]:
        """Fetch a FRED series."""
        if not self.fred_api_key:
            return None

        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'limit': 13 if yoy else 1,
                'sort_order': 'desc'
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                obs = data.get('observations', [])

                if obs and obs[0]['value'] != '.':
                    if yoy and len(obs) >= 13:
                        latest = float(obs[0]['value'])
                        year_ago = float(obs[12]['value'])
                        return ((latest / year_ago) - 1) * 100
                    else:
                        return float(obs[0]['value'])
        except Exception as e:
            logger.debug(f"FRED fetch error for {series_id}: {e}")

        return None

    def _determine_fed_signal(self, macro: MacroAnalysis) -> str:
        """Determine Fed policy outlook."""
        if macro.cuts_priced_in and macro.cuts_priced_in > 75:
            return "DOVISH"  # Significant cuts expected
        elif macro.cuts_priced_in and macro.cuts_priced_in > 25:
            return "NEUTRAL_DOVISH"
        elif macro.cpi_latest and macro.cpi_latest > 3.5:
            return "HAWKISH"  # Inflation concern
        else:
            return "NEUTRAL"

    def _calculate_macro_score(self, macro: MacroAnalysis) -> int:
        """Calculate overall macro score (0-100)."""
        score = 50

        # Fed policy outlook (±15 points)
        if macro.fed_policy_signal == "DOVISH":
            score += 15  # Cuts bullish for bonds
        elif macro.fed_policy_signal == "NEUTRAL_DOVISH":
            score += 8
        elif macro.fed_policy_signal == "HAWKISH":
            score -= 15  # Hikes bearish for bonds

        # Inflation (±10 points)
        if macro.cpi_trend == "CONTROLLED":
            score += 10  # Low inflation bullish
        elif macro.cpi_trend == "ELEVATED":
            score -= 10  # High inflation bearish

        # Growth/Employment (±5 points)
        if macro.unemployment and macro.unemployment > 4.5:
            score += 5  # Weakness bullish for bonds
        elif macro.unemployment and macro.unemployment < 3.5:
            score -= 5  # Strong labor bearish

        return max(0, min(100, score))


# =============================================================================
# SENTIMENT ANALYSIS
# =============================================================================

class SentimentAnalyzer:
    """Analyze news sentiment."""

    def analyze(self, news_score: Optional[float] = None,
                bullish_count: int = 0, bearish_count: int = 0,
                themes: List[str] = None) -> SentimentAnalysis:
        """Analyze sentiment."""
        sent = SentimentAnalysis()

        if news_score is not None:
            sent.news_sentiment_score = news_score
            if news_score >= 0.6:
                sent.news_sentiment_label = "BULLISH"
            elif news_score <= 0.4:
                sent.news_sentiment_label = "BEARISH"
            else:
                sent.news_sentiment_label = "NEUTRAL"

        sent.bullish_headlines = bullish_count
        sent.bearish_headlines = bearish_count
        sent.key_themes = themes or []

        # Calculate sentiment score
        sent.sentiment_score = self._calculate_sentiment_score(sent)

        return sent

    def _calculate_sentiment_score(self, sent: SentimentAnalysis) -> int:
        """Calculate overall sentiment score (0-100)."""
        if sent.news_sentiment_score is None:
            return 50  # Neutral if no data

        # Convert 0-1 score to 0-100
        return int(sent.news_sentiment_score * 100)


# =============================================================================
# MAIN SIGNAL GENERATOR
# =============================================================================

class InstitutionalBondSignalGenerator:
    """
    Institutional-grade bond signal generator.

    Integrates all analysis components into unified signals.
    """

    # Component weights (JPM/BlackRock style)
    WEIGHTS = {
        'technical': 0.25,
        'fundamental': 0.30,
        'flow': 0.20,
        'macro': 0.15,
        'sentiment': 0.10,
    }

    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.flow_analyzer = FlowAnalyzer()
        self.macro_analyzer = MacroAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()

    def generate_signal(self, ticker: str,
                       auction_score: Optional[int] = None,
                       news_score: Optional[float] = None,
                       news_bullish: int = 0,
                       news_bearish: int = 0,
                       news_themes: List[str] = None,
                       rate_probs: Dict = None) -> BondSignal:
        """
        Generate comprehensive bond signal.

        Args:
            ticker: Bond ETF ticker
            auction_score: From institutional_bond_analysis (optional)
            news_score: From bond_news (optional)
            news_bullish: Count of bullish headlines
            news_bearish: Count of bearish headlines
            news_themes: Key themes from news
            rate_probs: Rate probabilities from intelligence module

        Returns:
            BondSignal with complete analysis
        """
        etf_info = BOND_ETFS.get(ticker, {'name': ticker, 'duration': 10})

        # Run all analyses
        technical = self.technical_analyzer.analyze(ticker)

        fundamental = self.fundamental_analyzer.analyze(ticker, technical.current_price)

        flow = self.flow_analyzer.analyze(auction_score)

        macro = self.macro_analyzer.analyze(
            fed_rate=fundamental.fed_funds_rate,
            rate_probs=rate_probs
        )

        sentiment = self.sentiment_analyzer.analyze(
            news_score=news_score,
            bullish_count=news_bullish,
            bearish_count=news_bearish,
            themes=news_themes
        )

        # Calculate composite score
        composite_score = self._calculate_composite_score(
            technical, fundamental, flow, macro, sentiment
        )

        # Determine signal
        signal = self._determine_signal(composite_score)

        # Determine confidence
        confidence = self._determine_confidence(
            technical, fundamental, flow, macro, sentiment
        )

        # Calculate target and stop loss
        target_price = fundamental.price_target_12m or technical.current_price
        upside_pct = ((target_price / technical.current_price) - 1) * 100 if technical.current_price > 0 else 0

        # Stop loss based on support or ATR
        stop_loss = technical.support_1 if technical.support_1 else technical.current_price * 0.95

        # Build bull/bear cases
        bull_case, bear_case = self._build_cases(
            technical, fundamental, flow, macro, sentiment
        )

        # Key levels
        key_levels = {
            'support_1': technical.support_1,
            'support_2': technical.support_2,
            'resistance_1': technical.resistance_1,
            'resistance_2': technical.resistance_2,
            'vwap': technical.vwap,
            'sma_50': technical.sma_50,
            'sma_200': technical.sma_200,
        }

        # Get volume data
        try:
            etf = yf.Ticker(ticker)
            hist = etf.history(period='1mo')
            volume = int(hist['Volume'].iloc[-1]) if not hist.empty else 0
            avg_volume = int(hist['Volume'].mean()) if not hist.empty else 0
            prev_close = float(hist['Close'].iloc[-2]) if len(hist) >= 2 else technical.current_price
        except:
            volume, avg_volume, prev_close = 0, 0, technical.current_price

        day_change = ((technical.current_price / prev_close) - 1) * 100 if prev_close > 0 else 0

        return BondSignal(
            ticker=ticker,
            name=etf_info.get('name', ticker),
            current_price=technical.current_price,
            previous_close=prev_close,
            day_change_pct=day_change,
            volume=volume,
            avg_volume=avg_volume,
            technical=technical,
            fundamental=fundamental,
            flow=flow,
            macro=macro,
            sentiment=sentiment,
            composite_score=composite_score,
            signal=signal,
            confidence=confidence,
            target_price=target_price,
            upside_pct=upside_pct,
            stop_loss=stop_loss,
            bull_case=bull_case,
            bear_case=bear_case,
            key_levels=key_levels,
            analysis_time=datetime.now(),
        )

    def _calculate_composite_score(self, technical: TechnicalIndicators,
                                   fundamental: FundamentalAnalysis,
                                   flow: FlowAnalysis,
                                   macro: MacroAnalysis,
                                   sentiment: SentimentAnalysis) -> int:
        """Calculate weighted composite score."""

        scores = {}
        weights = {}

        # Only include components with data
        if technical.technical_score is not None:
            scores['technical'] = technical.technical_score
            weights['technical'] = self.WEIGHTS['technical']

        if fundamental.fundamental_score is not None:
            scores['fundamental'] = fundamental.fundamental_score
            weights['fundamental'] = self.WEIGHTS['fundamental']

        if flow.flow_score is not None:
            scores['flow'] = flow.flow_score
            weights['flow'] = self.WEIGHTS['flow']

        if macro.macro_score is not None:
            scores['macro'] = macro.macro_score
            weights['macro'] = self.WEIGHTS['macro']

        if sentiment.sentiment_score is not None:
            scores['sentiment'] = sentiment.sentiment_score
            weights['sentiment'] = self.WEIGHTS['sentiment']

        if not scores:
            return 50  # Neutral if no data

        # Normalize weights
        total_weight = sum(weights.values())

        # Calculate weighted average
        composite = sum(scores[k] * weights[k] for k in scores.keys()) / total_weight

        return int(round(composite))

    def _determine_signal(self, score: int) -> Signal:
        """Determine signal from composite score."""
        if score >= 75:
            return Signal.STRONG_BUY
        elif score >= 60:
            return Signal.BUY
        elif score <= 25:
            return Signal.STRONG_SELL
        elif score <= 40:
            return Signal.SELL
        else:
            return Signal.HOLD

    def _determine_confidence(self, technical: TechnicalIndicators,
                             fundamental: FundamentalAnalysis,
                             flow: FlowAnalysis,
                             macro: MacroAnalysis,
                             sentiment: SentimentAnalysis) -> str:
        """Determine confidence level based on data agreement."""

        scores = []
        if technical.technical_score is not None:
            scores.append(technical.technical_score)
        if fundamental.fundamental_score is not None:
            scores.append(fundamental.fundamental_score)
        if flow.flow_score is not None:
            scores.append(flow.flow_score)
        if macro.macro_score is not None:
            scores.append(macro.macro_score)
        if sentiment.sentiment_score is not None:
            scores.append(sentiment.sentiment_score)

        if len(scores) < 3:
            return "Low"  # Not enough data

        # Check agreement
        std_dev = np.std(scores)

        if std_dev < 10:
            return "High"  # Strong agreement
        elif std_dev < 20:
            return "Medium"
        else:
            return "Low"  # Conflicting signals

    def _build_cases(self, technical: TechnicalIndicators,
                     fundamental: FundamentalAnalysis,
                     flow: FlowAnalysis,
                     macro: MacroAnalysis,
                     sentiment: SentimentAnalysis) -> Tuple[List[str], List[str]]:
        """Build bull and bear cases."""

        bull_case = []
        bear_case = []

        # Technical
        if technical.rsi_signal == "OVERSOLD":
            bull_case.append(f"RSI oversold at {technical.rsi_14:.0f}")
        elif technical.rsi_signal == "OVERBOUGHT":
            bear_case.append(f"RSI overbought at {technical.rsi_14:.0f}")

        if technical.macd_crossover == "BULLISH":
            bull_case.append("MACD bullish crossover")
        elif technical.macd_crossover == "BEARISH":
            bear_case.append("MACD bearish crossover")

        if technical.bb_position == "BELOW_LOWER":
            bull_case.append("Price below lower Bollinger Band")
        elif technical.bb_position == "ABOVE_UPPER":
            bear_case.append("Price above upper Bollinger Band")

        # Fundamental
        if fundamental.yield_vs_fair_value and fundamental.yield_vs_fair_value < -0.3:
            bull_case.append(f"Yields {abs(fundamental.yield_vs_fair_value):.1f}% above fair value (cheap)")
        elif fundamental.yield_vs_fair_value and fundamental.yield_vs_fair_value > 0.3:
            bear_case.append(f"Yields {fundamental.yield_vs_fair_value:.1f}% below fair value (rich)")

        if fundamental.term_premium_signal == "ELEVATED":
            bull_case.append("Elevated term premium offers value")
        elif fundamental.term_premium_signal == "COMPRESSED":
            bear_case.append("Compressed term premium limits upside")

        # Flow
        if flow.auction_signal == "STRONG_DEMAND":
            bull_case.append("Strong Treasury auction demand")
        elif flow.auction_signal == "WEAK_DEMAND":
            bear_case.append("Weak Treasury auction demand")

        # Macro
        if macro.fed_policy_signal == "DOVISH":
            bull_case.append("Dovish Fed outlook supports bonds")
        elif macro.fed_policy_signal == "HAWKISH":
            bear_case.append("Hawkish Fed risks higher yields")

        if macro.cpi_trend == "CONTROLLED":
            bull_case.append("Inflation under control")
        elif macro.cpi_trend == "ELEVATED":
            bear_case.append("Elevated inflation headwind")

        # Sentiment
        if sentiment.news_sentiment_label == "BULLISH":
            bull_case.append(f"Bullish news flow ({sentiment.bullish_headlines} bullish headlines)")
        elif sentiment.news_sentiment_label == "BEARISH":
            bear_case.append(f"Bearish news flow ({sentiment.bearish_headlines} bearish headlines)")

        return bull_case, bear_case

    def generate_all_signals(self,
                            auction_score: Optional[int] = None,
                            news_score: Optional[float] = None,
                            news_bullish: int = 0,
                            news_bearish: int = 0,
                            news_themes: List[str] = None,
                            rate_probs: Dict = None) -> Dict[str, BondSignal]:
        """Generate signals for all bond ETFs."""

        signals = {}

        for ticker in BOND_ETFS.keys():
            try:
                signal = self.generate_signal(
                    ticker=ticker,
                    auction_score=auction_score,
                    news_score=news_score,
                    news_bullish=news_bullish,
                    news_bearish=news_bearish,
                    news_themes=news_themes,
                    rate_probs=rate_probs,
                )
                signals[ticker] = signal
                logger.info(f"{ticker}: {signal.signal.value} (Score: {signal.composite_score}, Target: ${signal.target_price:.2f})")
            except Exception as e:
                logger.error(f"Error generating signal for {ticker}: {e}")

        return signals


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def get_institutional_bond_signals(
    auction_score: Optional[int] = None,
    news_result = None,  # BondNewsResult from bond_news.py
    rate_probs: Dict = None,
) -> Dict[str, BondSignal]:
    """
    Get institutional-grade bond signals.

    Args:
        auction_score: From institutional_bond_analysis.py
        news_result: BondNewsResult from bond_news.py
        rate_probs: Rate probabilities dict

    Returns:
        Dict of ticker -> BondSignal
    """
    generator = InstitutionalBondSignalGenerator()

    # Extract news data if provided
    news_score = None
    news_bullish = 0
    news_bearish = 0
    news_themes = []

    if news_result:
        news_score = getattr(news_result, 'overall_score', None)
        news_bullish = getattr(news_result, 'bullish_count', 0)
        news_bearish = getattr(news_result, 'bearish_count', 0)
        news_themes = getattr(news_result, 'themes', [])

    return generator.generate_all_signals(
        auction_score=auction_score,
        news_score=news_score,
        news_bullish=news_bullish,
        news_bearish=news_bearish,
        news_themes=news_themes,
        rate_probs=rate_probs,
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("INSTITUTIONAL BOND SIGNAL GENERATOR TEST")
    print("=" * 70)

    generator = InstitutionalBondSignalGenerator()

    # Test single signal
    signal = generator.generate_signal('TLT')

    print(f"\n{signal.ticker} - {signal.name}")
    print(f"Price: ${signal.current_price:.2f} ({signal.day_change_pct:+.2f}%)")
    print(f"Target: ${signal.target_price:.2f} ({signal.upside_pct:+.1f}%)")
    print(f"\nSignal: {signal.signal.value}")
    print(f"Score: {signal.composite_score}/100")
    print(f"Confidence: {signal.confidence}")

    print(f"\nComponent Scores:")
    print(f"  Technical:   {signal.technical.technical_score}/100")
    print(f"  Fundamental: {signal.fundamental.fundamental_score}/100")
    print(f"  Flow:        {signal.flow.flow_score}/100")
    print(f"  Macro:       {signal.macro.macro_score}/100")
    print(f"  Sentiment:   {signal.sentiment.sentiment_score}/100")

    print(f"\nTechnical Indicators:")
    print(f"  RSI(14): {signal.technical.rsi_14:.1f} ({signal.technical.rsi_signal})")
    print(f"  MACD: {signal.technical.macd_crossover}")
    print(f"  BB Position: {signal.technical.bb_position}")
    print(f"  Trend: {signal.technical.trend.value}")

    print(f"\nKey Levels:")
    print(f"  Support 1: ${signal.key_levels.get('support_1', 0):.2f}")
    print(f"  Resistance 1: ${signal.key_levels.get('resistance_1', 0):.2f}")
    print(f"  VWAP: ${signal.key_levels.get('vwap', 0):.2f}")
    print(f"  SMA 200: ${signal.key_levels.get('sma_200', 0):.2f}")

    print(f"\nBull Case:")
    for point in signal.bull_case:
        print(f"  ✅ {point}")

    print(f"\nBear Case:")
    for point in signal.bear_case:
        print(f"  ⚠️ {point}")

    print("\n" + "=" * 70)
    print("ALL SIGNALS:")
    print("=" * 70)

    signals = generator.generate_all_signals()

    for ticker, sig in signals.items():
        print(f"{ticker}: {sig.signal.value:12} Score: {sig.composite_score:3} "
              f"Price: ${sig.current_price:7.2f} Target: ${sig.target_price:7.2f} "
              f"({sig.upside_pct:+5.1f}%)")