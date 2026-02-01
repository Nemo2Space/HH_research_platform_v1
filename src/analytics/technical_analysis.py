

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Sector mapping for stocks
SECTOR_ETFS = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financials': 'XLF',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Industrials': 'XLI',
    'Materials': 'XLB',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Communication Services': 'XLC',
    'Information Technology': 'XLK',  # Alias
    'Consumer Cyclical': 'XLY',  # Alias
}


@dataclass
class TechnicalLevels:
    """Key technical levels for a stock."""
    ticker: str
    current_price: float = 0

    # Support/Resistance
    support_1: float = 0  # Nearest support
    support_2: float = 0  # Secondary support
    resistance_1: float = 0  # Nearest resistance
    resistance_2: float = 0  # Secondary resistance

    # Distance to levels (%)
    distance_to_support_pct: float = 0
    distance_to_resistance_pct: float = 0

    # 52-week range
    high_52w: float = 0
    low_52w: float = 0
    pct_from_52w_high: float = 0
    pct_from_52w_low: float = 0

    # Moving averages
    ma_20: float = 0
    ma_50: float = 0
    ma_200: float = 0
    above_20ma: bool = False
    above_50ma: bool = False
    above_200ma: bool = False

    # Indicators
    rsi_14: float = 50
    rsi_status: str = "NEUTRAL"  # OVERSOLD, NEUTRAL, OVERBOUGHT

    # Trend
    trend_5d: str = "NEUTRAL"  # UP, DOWN, NEUTRAL
    trend_20d: str = "NEUTRAL"

    # Risk/Reward based on levels
    risk_to_support_pct: float = 0
    reward_to_resistance_pct: float = 0
    risk_reward_ratio: float = 0


@dataclass
class RelativeStrength:
    """Relative strength vs benchmarks."""
    ticker: str

    # vs SPY
    vs_spy_5d: float = 0
    vs_spy_20d: float = 0
    vs_spy_60d: float = 0
    outperforming_spy: bool = False

    # vs Sector ETF
    sector: str = ""
    sector_etf: str = ""
    vs_sector_5d: float = 0
    vs_sector_20d: float = 0
    outperforming_sector: bool = False

    # Sector momentum
    sector_momentum: str = "NEUTRAL"  # HOT, NEUTRAL, COLD

    # Overall RS rating
    rs_rating: int = 50  # 0-100, like IBD RS rating
    rs_status: str = "NEUTRAL"  # STRONG, NEUTRAL, WEAK


@dataclass
class LiquidityScore:
    """Liquidity analysis for a stock."""
    ticker: str

    # Volume metrics
    avg_volume_20d: int = 0
    avg_volume_50d: int = 0
    avg_dollar_volume: float = 0  # Price * Volume

    # Today's volume
    volume_today: int = 0
    relative_volume: float = 0  # vs 20d avg

    # Liquidity classification
    liquidity_score: str = "MEDIUM"  # HIGH, MEDIUM, LOW, ILLIQUID

    # Position sizing guidance
    max_position_1pct_volume: float = 0  # Max $ to trade without moving market

    # Spread estimate (if available)
    avg_spread_pct: float = 0


@dataclass
class TechnicalAnalysis:
    """Complete technical analysis for a stock."""
    ticker: str
    current_price: float = 0

    # Components
    levels: TechnicalLevels = None
    relative_strength: RelativeStrength = None
    liquidity: LiquidityScore = None

    # Overall assessment
    technical_rating: str = "NEUTRAL"  # BULLISH, NEUTRAL, BEARISH
    technical_score: int = 50  # 0-100


class TechnicalAnalyzer:
    """
    Comprehensive technical analysis for stocks.
    """

    def __init__(self):
        self._price_cache = {}
        self._spy_data = None
        self._sector_data = {}

    def analyze_ticker(self, ticker: str, sector: str = None) -> TechnicalAnalysis:
        """
        Run complete technical analysis on a ticker.

        Args:
            ticker: Stock symbol
            sector: Stock's sector (optional, will try to fetch)

        Returns:
            TechnicalAnalysis with all components
        """
        ticker = ticker.upper()

        analysis = TechnicalAnalysis(ticker=ticker)

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get price history
            hist = stock.history(period="1y")
            if len(hist) < 20:
                logger.warning(f"Insufficient data for {ticker}")
                return analysis

            current_price = hist['Close'].iloc[-1]
            analysis.current_price = current_price

            # Get sector if not provided
            if not sector:
                sector = info.get('sector', '')

            # ============================================================
            # 1. TECHNICAL LEVELS
            # ============================================================
            analysis.levels = self._calculate_levels(ticker, hist, info)

            # ============================================================
            # 2. RELATIVE STRENGTH
            # ============================================================
            analysis.relative_strength = self._calculate_relative_strength(ticker, hist, sector)

            # ============================================================
            # 3. LIQUIDITY
            # ============================================================
            analysis.liquidity = self._calculate_liquidity(ticker, hist, info)

            # ============================================================
            # 4. OVERALL RATING
            # ============================================================
            analysis.technical_score, analysis.technical_rating = self._calculate_overall_rating(analysis)

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")

        return analysis

    def _calculate_levels(self, ticker: str, hist: pd.DataFrame, info: dict) -> TechnicalLevels:
        """Calculate support/resistance and technical levels."""
        levels = TechnicalLevels(ticker=ticker)

        current = hist['Close'].iloc[-1]
        levels.current_price = current

        # 52-week high/low
        levels.high_52w = hist['High'].max()
        levels.low_52w = hist['Low'].min()
        levels.pct_from_52w_high = ((levels.high_52w - current) / levels.high_52w) * 100
        levels.pct_from_52w_low = ((current - levels.low_52w) / levels.low_52w) * 100

        # Moving averages
        if len(hist) >= 20:
            levels.ma_20 = hist['Close'].iloc[-20:].mean()
            levels.above_20ma = current > levels.ma_20

        if len(hist) >= 50:
            levels.ma_50 = hist['Close'].iloc[-50:].mean()
            levels.above_50ma = current > levels.ma_50

        if len(hist) >= 200:
            levels.ma_200 = hist['Close'].iloc[-200:].mean()
            levels.above_200ma = current > levels.ma_200

        # RSI
        levels.rsi_14 = self._calculate_rsi(hist['Close'], 14)
        if levels.rsi_14 < 30:
            levels.rsi_status = "OVERSOLD"
        elif levels.rsi_14 > 70:
            levels.rsi_status = "OVERBOUGHT"
        else:
            levels.rsi_status = "NEUTRAL"

        # Trend
        if len(hist) >= 5:
            change_5d = (current - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5] * 100
            levels.trend_5d = "UP" if change_5d > 2 else "DOWN" if change_5d < -2 else "NEUTRAL"

        if len(hist) >= 20:
            change_20d = (current - hist['Close'].iloc[-20]) / hist['Close'].iloc[-20] * 100
            levels.trend_20d = "UP" if change_20d > 5 else "DOWN" if change_20d < -5 else "NEUTRAL"

        # Support/Resistance using pivot points and recent swing highs/lows
        supports, resistances = self._find_support_resistance(hist, current)

        if supports:
            levels.support_1 = supports[0]
            if len(supports) > 1:
                levels.support_2 = supports[1]
            levels.distance_to_support_pct = ((current - levels.support_1) / current) * 100

        if resistances:
            levels.resistance_1 = resistances[0]
            if len(resistances) > 1:
                levels.resistance_2 = resistances[1]
            levels.distance_to_resistance_pct = ((levels.resistance_1 - current) / current) * 100

        # Risk/Reward
        if levels.support_1 > 0:
            levels.risk_to_support_pct = levels.distance_to_support_pct
        if levels.resistance_1 > 0:
            levels.reward_to_resistance_pct = levels.distance_to_resistance_pct

        if levels.risk_to_support_pct > 0:
            levels.risk_reward_ratio = levels.reward_to_resistance_pct / levels.risk_to_support_pct

        return levels

    def _find_support_resistance(self, hist: pd.DataFrame, current_price: float) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels using swing points."""
        supports = []
        resistances = []

        # Use last 60 days
        recent = hist.tail(60)

        if len(recent) < 10:
            return supports, resistances

        highs = recent['High'].values
        lows = recent['Low'].values
        closes = recent['Close'].values

        # Find swing lows (potential support)
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
                if lows[i] < current_price:  # Must be below current price
                    supports.append(lows[i])

        # Find swing highs (potential resistance)
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and highs[i] > highs[i + 1] and highs[i] > highs[
                i + 2]:
                if highs[i] > current_price:  # Must be above current price
                    resistances.append(highs[i])

        # Add moving averages as potential levels
        if len(hist) >= 50:
            ma_50 = closes[-50:].mean()
            if ma_50 < current_price:
                supports.append(ma_50)
            else:
                resistances.append(ma_50)

        if len(hist) >= 200:
            ma_200 = closes[-200:].mean()
            if ma_200 < current_price:
                supports.append(ma_200)
            else:
                resistances.append(ma_200)

        # Sort and get closest levels
        supports = sorted(set(supports), reverse=True)[:2]  # Closest supports first
        resistances = sorted(set(resistances))[:2]  # Closest resistances first

        return supports, resistances

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        except:
            return 50

    def _calculate_relative_strength(self, ticker: str, hist: pd.DataFrame, sector: str) -> RelativeStrength:
        """Calculate relative strength vs SPY and sector."""
        rs = RelativeStrength(ticker=ticker, sector=sector)

        # Get SPY data
        try:
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="3mo")

            if len(spy_hist) >= 5 and len(hist) >= 5:
                # 5-day relative strength
                stock_5d = (hist['Close'].iloc[-1] - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5] * 100
                spy_5d = (spy_hist['Close'].iloc[-1] - spy_hist['Close'].iloc[-5]) / spy_hist['Close'].iloc[-5] * 100
                rs.vs_spy_5d = stock_5d - spy_5d

            if len(spy_hist) >= 20 and len(hist) >= 20:
                # 20-day relative strength
                stock_20d = (hist['Close'].iloc[-1] - hist['Close'].iloc[-20]) / hist['Close'].iloc[-20] * 100
                spy_20d = (spy_hist['Close'].iloc[-1] - spy_hist['Close'].iloc[-20]) / spy_hist['Close'].iloc[-20] * 100
                rs.vs_spy_20d = stock_20d - spy_20d

            if len(spy_hist) >= 60 and len(hist) >= 60:
                # 60-day relative strength
                stock_60d = (hist['Close'].iloc[-1] - hist['Close'].iloc[-60]) / hist['Close'].iloc[-60] * 100
                spy_60d = (spy_hist['Close'].iloc[-1] - spy_hist['Close'].iloc[-60]) / spy_hist['Close'].iloc[-60] * 100
                rs.vs_spy_60d = stock_60d - spy_60d

            rs.outperforming_spy = rs.vs_spy_20d > 0

        except Exception as e:
            logger.debug(f"Error getting SPY data: {e}")

        # Get sector ETF data
        sector_etf = SECTOR_ETFS.get(sector, '')
        rs.sector_etf = sector_etf

        if sector_etf:
            try:
                etf = yf.Ticker(sector_etf)
                etf_hist = etf.history(period="3mo")

                if len(etf_hist) >= 5 and len(hist) >= 5:
                    stock_5d = (hist['Close'].iloc[-1] - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5] * 100
                    etf_5d = (etf_hist['Close'].iloc[-1] - etf_hist['Close'].iloc[-5]) / etf_hist['Close'].iloc[
                        -5] * 100
                    rs.vs_sector_5d = stock_5d - etf_5d

                if len(etf_hist) >= 20 and len(hist) >= 20:
                    stock_20d = (hist['Close'].iloc[-1] - hist['Close'].iloc[-20]) / hist['Close'].iloc[-20] * 100
                    etf_20d = (etf_hist['Close'].iloc[-1] - etf_hist['Close'].iloc[-20]) / etf_hist['Close'].iloc[
                        -20] * 100
                    rs.vs_sector_20d = stock_20d - etf_20d

                    # Sector momentum
                    if etf_20d > 3:
                        rs.sector_momentum = "HOT"
                    elif etf_20d < -3:
                        rs.sector_momentum = "COLD"
                    else:
                        rs.sector_momentum = "NEUTRAL"

                rs.outperforming_sector = rs.vs_sector_20d > 0

            except Exception as e:
                logger.debug(f"Error getting sector ETF data: {e}")

        # Calculate RS Rating (0-100, like IBD)
        # Based on 20d and 60d relative performance
        rs_score = 50
        rs_score += min(25, max(-25, rs.vs_spy_20d * 2))  # 20d weight
        rs_score += min(25, max(-25, rs.vs_spy_60d))  # 60d weight
        rs.rs_rating = int(max(0, min(100, rs_score)))

        if rs.rs_rating >= 70:
            rs.rs_status = "STRONG"
        elif rs.rs_rating <= 30:
            rs.rs_status = "WEAK"
        else:
            rs.rs_status = "NEUTRAL"

        return rs

    def _calculate_liquidity(self, ticker: str, hist: pd.DataFrame, info: dict) -> LiquidityScore:
        """Calculate liquidity metrics."""
        liq = LiquidityScore(ticker=ticker)

        current_price = hist['Close'].iloc[-1]

        # Volume metrics
        if len(hist) >= 20:
            liq.avg_volume_20d = int(hist['Volume'].iloc[-20:].mean())
        if len(hist) >= 50:
            liq.avg_volume_50d = int(hist['Volume'].iloc[-50:].mean())

        liq.volume_today = int(hist['Volume'].iloc[-1])

        # Average dollar volume
        if liq.avg_volume_20d > 0:
            liq.avg_dollar_volume = liq.avg_volume_20d * current_price

        # Relative volume
        if liq.avg_volume_20d > 0:
            liq.relative_volume = liq.volume_today / liq.avg_volume_20d

        # Liquidity classification
        if liq.avg_dollar_volume >= 100_000_000:  # $100M+ daily
            liq.liquidity_score = "HIGH"
        elif liq.avg_dollar_volume >= 10_000_000:  # $10M+ daily
            liq.liquidity_score = "MEDIUM"
        elif liq.avg_dollar_volume >= 1_000_000:  # $1M+ daily
            liq.liquidity_score = "LOW"
        else:
            liq.liquidity_score = "ILLIQUID"

        # Max position without moving market (1% of daily volume)
        liq.max_position_1pct_volume = liq.avg_dollar_volume * 0.01

        return liq

    def _calculate_overall_rating(self, analysis: TechnicalAnalysis) -> Tuple[int, str]:
        """Calculate overall technical score and rating."""
        score = 50  # Start neutral

        levels = analysis.levels
        rs = analysis.relative_strength
        liq = analysis.liquidity

        if levels:
            # Trend (+/- 15)
            if levels.trend_20d == "UP":
                score += 15
            elif levels.trend_20d == "DOWN":
                score -= 15

            # Moving averages (+/- 10)
            if levels.above_50ma and levels.above_200ma:
                score += 10
            elif not levels.above_50ma and not levels.above_200ma:
                score -= 10

            # RSI (+/- 5)
            if levels.rsi_status == "OVERSOLD":
                score += 5  # Potential bounce
            elif levels.rsi_status == "OVERBOUGHT":
                score -= 5

            # Near 52w high (+5)
            if levels.pct_from_52w_high < 10:
                score += 5

        if rs:
            # Relative strength (+/- 15)
            if rs.rs_status == "STRONG":
                score += 15
            elif rs.rs_status == "WEAK":
                score -= 15

            # Sector momentum (+/- 5)
            if rs.sector_momentum == "HOT":
                score += 5
            elif rs.sector_momentum == "COLD":
                score -= 5

        if liq:
            # Liquidity (-10 if illiquid)
            if liq.liquidity_score == "ILLIQUID":
                score -= 10

        # Clamp to 0-100
        score = max(0, min(100, score))

        # Rating
        if score >= 65:
            rating = "BULLISH"
        elif score <= 35:
            rating = "BEARISH"
        else:
            rating = "NEUTRAL"

        return score, rating

    def get_analysis_for_ai(self, ticker: str, sector: str = None) -> str:
        """Get technical analysis formatted for AI."""
        analysis = self.analyze_ticker(ticker, sector)

        lines = [
            f"\n{'=' * 50}",
            f"📈 TECHNICAL ANALYSIS: {ticker}",
            f"{'=' * 50}",
            f"",
            f"💰 PRICE: ${analysis.current_price:.2f}",
            f"📊 TECHNICAL RATING: {analysis.technical_score}/100 ({analysis.technical_rating})",
        ]

        if analysis.levels:
            lvl = analysis.levels
            lines.extend([
                f"",
                f"📍 KEY LEVELS:",
                f"   Support 1: ${lvl.support_1:.2f} ({lvl.distance_to_support_pct:.1f}% below)",
                f"   Resistance 1: ${lvl.resistance_1:.2f} ({lvl.distance_to_resistance_pct:.1f}% above)",
                f"   52W High: ${lvl.high_52w:.2f} ({lvl.pct_from_52w_high:.1f}% from high)",
                f"   52W Low: ${lvl.low_52w:.2f}",
                f"   Risk/Reward: {lvl.risk_reward_ratio:.1f}:1",
                f"",
                f"📊 MOVING AVERAGES:",
                f"   Above 20 MA: {'✅' if lvl.above_20ma else '❌'} (${lvl.ma_20:.2f})",
                f"   Above 50 MA: {'✅' if lvl.above_50ma else '❌'} (${lvl.ma_50:.2f})",
                f"   Above 200 MA: {'✅' if lvl.above_200ma else '❌'} (${lvl.ma_200:.2f})",
                f"",
                f"📈 MOMENTUM:",
                f"   RSI(14): {lvl.rsi_14:.0f} ({lvl.rsi_status})",
                f"   5-Day Trend: {lvl.trend_5d}",
                f"   20-Day Trend: {lvl.trend_20d}",
            ])

        if analysis.relative_strength:
            rs = analysis.relative_strength
            lines.extend([
                f"",
                f"💪 RELATIVE STRENGTH:",
                f"   RS Rating: {rs.rs_rating}/100 ({rs.rs_status})",
                f"   vs SPY (5d): {rs.vs_spy_5d:+.1f}%",
                f"   vs SPY (20d): {rs.vs_spy_20d:+.1f}% {'✅ Outperforming' if rs.outperforming_spy else '❌ Underperforming'}",
                f"   vs Sector (20d): {rs.vs_sector_20d:+.1f}% ({rs.sector_etf})",
                f"   Sector Momentum: {rs.sector_momentum}",
            ])

        if analysis.liquidity:
            liq = analysis.liquidity
            lines.extend([
                f"",
                f"💧 LIQUIDITY:",
                f"   Avg Daily Volume: {liq.avg_volume_20d:,}",
                f"   Avg Dollar Volume: ${liq.avg_dollar_volume / 1e6:.1f}M",
                f"   Today's Volume: {liq.relative_volume:.1f}x average",
                f"   Liquidity Score: {liq.liquidity_score}",
            ])

        return "\n".join(lines)

    def scan_universe(self, tickers: List[str], sectors: Dict[str, str] = None,
                      max_workers: int = 5) -> List[TechnicalAnalysis]:
        """
        Scan multiple tickers for technical analysis.

        Args:
            tickers: List of ticker symbols
            sectors: Dict mapping ticker to sector
            max_workers: Parallel threads

        Returns:
            List of TechnicalAnalysis sorted by score
        """
        results = []
        sectors = sectors or {}

        logger.info(f"Running technical analysis on {len(tickers)} tickers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.analyze_ticker, t, sectors.get(t)): t
                for t in tickers
            }

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result.current_price > 0:
                        results.append(result)
                except Exception as e:
                    logger.debug(f"Error analyzing {ticker}: {e}")

        # Sort by technical score
        results.sort(key=lambda x: x.technical_score, reverse=True)

        return results


# Convenience functions
_tech_analyzer = None


def get_technical_analyzer() -> TechnicalAnalyzer:
    """Get singleton instance."""
    global _tech_analyzer
    if _tech_analyzer is None:
        _tech_analyzer = TechnicalAnalyzer()
    return _tech_analyzer


def analyze_technicals(ticker: str, sector: str = None) -> TechnicalAnalysis:
    """Quick access to technical analysis."""
    return get_technical_analyzer().analyze_ticker(ticker, sector)


def get_technicals_for_ai(ticker: str, sector: str = None) -> str:
    """Get technical analysis formatted for AI."""
    return get_technical_analyzer().get_analysis_for_ai(ticker, sector)


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    analyzer = TechnicalAnalyzer()

    # Test single ticker
    print(analyzer.get_analysis_for_ai("AAPL", "Technology"))

    # Test scan
    results = analyzer.scan_universe(["AAPL", "MSFT", "NVDA", "AMD"])
    print(f"\nTop by Technical Score:")
    for r in results:
        print(f"  {r.ticker}: {r.technical_score} ({r.technical_rating})")