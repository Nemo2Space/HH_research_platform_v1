"""
Market Context Module

Provides market-wide context for trading decisions:
1. Economic Calendar (Fed meetings, CPI, Jobs reports) - LIVE DATA
2. Sector Performance & Momentum
3. Market Regime (Risk-On/Risk-Off)

Author: Alpha Research Platform
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import yfinance as yf

from src.utils.logging import get_logger

# Import economic calendar
try:
    from src.analytics.economic_calendar import (
        EconomicCalendarFetcher,
        get_economic_calendar,
        get_calendar_summary,
        is_high_impact_day
    )

    ECONOMIC_CALENDAR_AVAILABLE = True
except ImportError:
    ECONOMIC_CALENDAR_AVAILABLE = False

# Import economic news analyzer
try:
    from src.analytics.economic_news_analyzer import (
        get_news_analyzer,
        analyze_all_economic_events,
        get_market_quick_summary
    )

    NEWS_ANALYZER_AVAILABLE = True
except ImportError:
    NEWS_ANALYZER_AVAILABLE = False

logger = get_logger(__name__)


# Note: Economic calendar data is now fetched from src.analytics.economic_calendar module


@dataclass
class MarketContext:
    """Overall market context."""
    # Economic calendar
    next_fed_meeting: Optional[date] = None
    last_fed_meeting: Optional[date] = None
    days_to_fed: int = 999
    fed_meeting_this_week: bool = False

    next_cpi: Optional[date] = None
    days_to_cpi: int = 999

    next_jobs: Optional[date] = None
    days_to_jobs: int = 999

    high_impact_today: bool = False
    high_impact_this_week: bool = False
    today_events: List[str] = field(default_factory=list)  # Event names

    # Market regime
    spy_trend: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    vix_level: float = 0
    vix_regime: str = "NORMAL"  # LOW, NORMAL, HIGH, EXTREME
    market_regime: str = "NEUTRAL"  # RISK_ON, RISK_OFF, NEUTRAL

    # Sector performance (5-day)
    hot_sectors: List[str] = field(default_factory=list)
    cold_sectors: List[str] = field(default_factory=list)
    sector_performance: Dict[str, float] = field(default_factory=dict)

    # Calendar summary for display
    calendar_summary: str = ""

    # AI Analysis of economic news
    economic_news_analysis: str = ""
    economic_signal: str = ""  # BULLISH, BEARISH, MIXED, PENDING


class MarketContextAnalyzer:
    """
    Analyzes overall market context.
    """

    # Sector ETFs for tracking
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
    }

    def __init__(self):
        self._cache = {}
        self._cache_time = None
        self._cache_duration = timedelta(hours=1)

        # Initialize economic calendar fetcher
        if ECONOMIC_CALENDAR_AVAILABLE:
            self._calendar_fetcher = EconomicCalendarFetcher()
        else:
            self._calendar_fetcher = None
        self._cache_time = None
        self._cache_duration = timedelta(hours=1)

    def get_market_context(self, force_refresh: bool = False, run_ai_analysis: bool = True) -> MarketContext:
        """
        Get current market context.

        Args:
            force_refresh: Force refresh from cache
            run_ai_analysis: Run AI analysis on economic news (takes 30-60s)

        Returns:
            MarketContext with economic calendar and market regime
        """
        # Check cache
        if not force_refresh and self._cache_time:
            if datetime.now() - self._cache_time < self._cache_duration:
                return self._cache.get('context', MarketContext())

        context = MarketContext()
        today = date.today()

        # ============================================================
        # 1. ECONOMIC CALENDAR (using live data)
        # ============================================================
        if self._calendar_fetcher:
            try:
                econ_cal = self._calendar_fetcher.get_calendar()

                # Copy data from economic calendar
                context.next_fed_meeting = econ_cal.next_fed_meeting
                context.last_fed_meeting = econ_cal.last_fed_meeting
                context.days_to_fed = econ_cal.days_to_fed
                context.fed_meeting_this_week = econ_cal.fed_meeting_this_week
                context.next_cpi = econ_cal.next_cpi
                context.days_to_cpi = econ_cal.days_to_cpi
                context.next_jobs = econ_cal.next_jobs
                context.days_to_jobs = econ_cal.days_to_jobs
                context.high_impact_today = econ_cal.high_impact_today
                context.high_impact_this_week = econ_cal.high_impact_this_week
                context.today_events = [e.event_name for e in econ_cal.today_events]
                context.calendar_summary = self._calendar_fetcher.get_calendar_summary()

                # ============================================================
                # 1b. AI ANALYSIS OF ECONOMIC NEWS (if data released)
                # ============================================================
                if run_ai_analysis and NEWS_ANALYZER_AVAILABLE and econ_cal.today_events:
                    try:
                        # Convert events to format for analyzer
                        events_for_analysis = [
                            {
                                'name': e.event_name,
                                'time': e.event_time,
                                'actual': e.actual,
                                'forecast': e.forecast,
                                'previous': e.previous
                            }
                            for e in econ_cal.today_events
                        ]

                        # Check if any data has been released
                        released_events = [e for e in events_for_analysis if e.get('actual')]

                        if released_events:
                            # Get quick summary first (fast)
                            context.economic_signal = get_market_quick_summary(events_for_analysis)

                            # Get full AI analysis (slower, so we cache it)
                            logger.info(f"Running AI analysis on {len(released_events)} released events...")
                            analysis = analyze_all_economic_events(events_for_analysis)

                            if analysis and analysis.full_analysis:
                                context.economic_news_analysis = analysis.full_analysis
                                context.economic_signal = f"{analysis.overall_signal}"

                        else:
                            context.economic_signal = "PENDING"
                            context.economic_news_analysis = "⏳ Economic data not yet released. Analysis available after releases."

                    except Exception as e:
                        logger.warning(f"Error running AI news analysis: {e}")
                        context.economic_news_analysis = f"Analysis unavailable: {str(e)}"

            except Exception as e:
                logger.warning(f"Error getting economic calendar: {e}")

        # ============================================================
        # 2. MARKET REGIME (VIX, SPY trend)
        # ============================================================
        try:
            # Get VIX
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="5d")
            if len(vix_data) > 0:
                context.vix_level = vix_data['Close'].iloc[-1]

                if context.vix_level < 15:
                    context.vix_regime = "LOW"
                elif context.vix_level < 20:
                    context.vix_regime = "NORMAL"
                elif context.vix_level < 30:
                    context.vix_regime = "HIGH"
                else:
                    context.vix_regime = "EXTREME"

            # Get SPY trend
            spy = yf.Ticker("SPY")
            spy_data = spy.history(period="1mo")
            if len(spy_data) >= 20:
                current = spy_data['Close'].iloc[-1]
                ma_20 = spy_data['Close'].iloc[-20:].mean()
                change_5d = (current - spy_data['Close'].iloc[-5]) / spy_data['Close'].iloc[-5] * 100

                if current > ma_20 and change_5d > 1:
                    context.spy_trend = "BULLISH"
                elif current < ma_20 and change_5d < -1:
                    context.spy_trend = "BEARISH"
                else:
                    context.spy_trend = "NEUTRAL"

            # Determine market regime
            if context.vix_regime in ["LOW", "NORMAL"] and context.spy_trend == "BULLISH":
                context.market_regime = "RISK_ON"
            elif context.vix_regime in ["HIGH", "EXTREME"] or context.spy_trend == "BEARISH":
                context.market_regime = "RISK_OFF"
            else:
                context.market_regime = "NEUTRAL"

        except Exception as e:
            logger.debug(f"Error getting market regime: {e}")

        # ============================================================
        # 3. SECTOR PERFORMANCE
        # ============================================================
        context.sector_performance = self._get_sector_performance()

        # Identify hot and cold sectors
        sorted_sectors = sorted(context.sector_performance.items(), key=lambda x: x[1], reverse=True)
        context.hot_sectors = [s[0] for s in sorted_sectors[:3] if s[1] > 1]  # > 1% gain
        context.cold_sectors = [s[0] for s in sorted_sectors[-3:] if s[1] < -1]  # > 1% loss

        # Cache results
        self._cache['context'] = context
        self._cache_time = datetime.now()

        return context

    def _get_sector_performance(self, period: str = "5d") -> Dict[str, float]:
        """Get sector ETF performance."""
        performance = {}

        try:
            etf_symbols = list(self.SECTOR_ETFS.values())
            data = yf.download(etf_symbols, period=period, progress=False, auto_adjust=True)['Close']

            if len(data) >= 2:
                for sector, etf in self.SECTOR_ETFS.items():
                    if etf in data.columns:
                        start_price = data[etf].iloc[0]
                        end_price = data[etf].iloc[-1]
                        if start_price > 0:
                            pct_change = ((end_price - start_price) / start_price) * 100
                            performance[sector] = pct_change
        except Exception as e:
            logger.debug(f"Error getting sector performance: {e}")

        return performance

    def get_context_for_ai(self) -> str:
        """Get market context formatted for AI."""
        ctx = self.get_market_context()

        lines = [
            f"\n{'=' * 50}",
            f"📅 MARKET CONTEXT - {date.today()}",
            f"{'=' * 50}",
        ]

        # Use calendar summary if available
        if ctx.calendar_summary:
            lines.append(ctx.calendar_summary)
        else:
            # Fallback to basic format
            lines.append(f"")
            lines.append(f"🏛️ ECONOMIC CALENDAR:")

            if ctx.last_fed_meeting:
                days_since = (date.today() - ctx.last_fed_meeting).days
                lines.append(f"   Last Fed Meeting: {ctx.last_fed_meeting} ({days_since} days ago)")

            if ctx.next_fed_meeting:
                if ctx.days_to_fed <= 7:
                    lines.append(f"   ⚠️ Next Fed Meeting: {ctx.next_fed_meeting} ({ctx.days_to_fed} days)")
                else:
                    lines.append(f"   Next Fed Meeting: {ctx.next_fed_meeting} ({ctx.days_to_fed} days)")

            if ctx.days_to_cpi <= 7:
                lines.append(f"   ⚠️ CPI REPORT: {ctx.next_cpi} ({ctx.days_to_cpi} days)")

            if ctx.days_to_jobs <= 7:
                lines.append(f"   ⚠️ JOBS REPORT: {ctx.next_jobs} ({ctx.days_to_jobs} days)")

            if ctx.high_impact_today:
                lines.append(f"   🔴 HIGH IMPACT EVENT TODAY - Consider reducing position sizes")
            elif ctx.high_impact_this_week:
                lines.append(f"   🟡 High impact event this week - Be cautious")

        # Add AI analysis of economic news if available
        if ctx.economic_news_analysis:
            lines.extend([
                f"",
                f"{'=' * 50}",
                f"🤖 AI ECONOMIC NEWS ANALYSIS",
                f"{'=' * 50}",
                f"Signal: {ctx.economic_signal}",
                f"",
                ctx.economic_news_analysis,
            ])
        elif ctx.economic_signal:
            lines.extend([
                f"",
                f"🤖 Economic Signal: {ctx.economic_signal}",
            ])

        lines.extend([
            f"",
            f"📊 MARKET REGIME:",
            f"   VIX: {ctx.vix_level:.1f} ({ctx.vix_regime})",
            f"   SPY Trend: {ctx.spy_trend}",
            f"   Regime: {ctx.market_regime}",
        ])

        if ctx.hot_sectors:
            lines.append(f"")
            lines.append(f"🔥 HOT SECTORS (5d): {', '.join(ctx.hot_sectors)}")

        if ctx.cold_sectors:
            lines.append(f"❄️ COLD SECTORS (5d): {', '.join(ctx.cold_sectors)}")

        return "\n".join(lines)


# Convenience functions
_market_analyzer = None


def get_market_context(run_ai_analysis: bool = True) -> MarketContext:
    """Get current market context."""
    global _market_analyzer
    if _market_analyzer is None:
        _market_analyzer = MarketContextAnalyzer()
    return _market_analyzer.get_market_context(run_ai_analysis=run_ai_analysis)


def get_market_context_for_ai(run_ai_analysis: bool = True) -> str:
    """Get market context formatted for AI (includes AI analysis if run_ai_analysis=True)."""
    global _market_analyzer
    if _market_analyzer is None:
        _market_analyzer = MarketContextAnalyzer()
    # Force refresh to include AI analysis
    _market_analyzer.get_market_context(force_refresh=True, run_ai_analysis=run_ai_analysis)
    return _market_analyzer.get_context_for_ai()


def is_high_impact_day() -> bool:
    """Check if today has high impact events."""
    ctx = get_market_context()
    return ctx.high_impact_today


def get_sector_momentum(sector: str) -> str:
    """Get momentum for a specific sector."""
    ctx = get_market_context()
    if sector in ctx.hot_sectors:
        return "HOT"
    elif sector in ctx.cold_sectors:
        return "COLD"
    return "NEUTRAL"


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    analyzer = MarketContextAnalyzer()
    print(analyzer.get_context_for_ai())

    ctx = analyzer.get_market_context()
    print(f"\nSector Performance:")
    for sector, perf in sorted(ctx.sector_performance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {sector}: {perf:+.2f}%")