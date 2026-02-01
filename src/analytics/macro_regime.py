"""
Macro Regime Detection

Detects market regime (Risk-On vs Risk-Off) using multiple indicators:
- VIX (Fear Index)
- Treasury Yield Spread (10Y-2Y)
- Credit Spreads (High Yield)
- Stock vs Bond performance
- Dollar Index
- Sector Leadership

Used to adjust trading signals based on market environment.

Author: Alpha Research Platform
"""

import os
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

import pandas as pd
import numpy as np
import yfinance as yf

from src.utils.logging import get_logger
from src.db.connection import get_engine

logger = get_logger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    STRONG_RISK_ON = "STRONG_RISK_ON"  # Score >= 70
    RISK_ON = "RISK_ON"  # Score 55-70
    NEUTRAL = "NEUTRAL"  # Score 45-55
    RISK_OFF = "RISK_OFF"  # Score 30-45
    STRONG_RISK_OFF = "STRONG_RISK_OFF"  # Score < 30


@dataclass
class RegimeIndicator:
    """Individual regime indicator."""
    name: str
    value: float
    signal: str  # RISK_ON, NEUTRAL, RISK_OFF
    score: int  # 0-100 contribution
    description: str
    weight: float = 1.0


@dataclass
class MacroRegimeResult:
    """Complete macro regime analysis."""
    regime: MarketRegime
    regime_score: int  # 0-100 (higher = more risk-on)
    confidence: float  # 0-1

    # Individual indicators
    vix: RegimeIndicator = None
    yield_spread: RegimeIndicator = None
    credit_spread: RegimeIndicator = None
    stock_vs_bond: RegimeIndicator = None
    dollar_index: RegimeIndicator = None
    sector_leadership: RegimeIndicator = None
    market_breadth: RegimeIndicator = None

    # Summary
    risk_on_factors: List[str] = field(default_factory=list)
    risk_off_factors: List[str] = field(default_factory=list)

    # Signal adjustments
    growth_adjustment: int = 0  # Points to add/subtract for growth stocks
    defensive_adjustment: int = 0  # Points to add/subtract for defensive stocks

    # Metadata
    analysis_date: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        result = {
            'regime': self.regime.value,
            'regime_score': self.regime_score,
            'confidence': self.confidence,
            'risk_on_factors': self.risk_on_factors,
            'risk_off_factors': self.risk_off_factors,
            'growth_adjustment': self.growth_adjustment,
            'defensive_adjustment': self.defensive_adjustment,
            'analysis_date': self.analysis_date.isoformat(),
        }

        for indicator in ['vix', 'yield_spread', 'credit_spread',
                          'stock_vs_bond', 'dollar_index', 'sector_leadership',
                          'market_breadth']:
            ind = getattr(self, indicator)
            if ind:
                result[indicator] = asdict(ind)

        return result


class MacroRegimeDetector:
    """
    Detects market regime using multiple indicators.
    """

    # Sector classifications
    GROWTH_SECTORS = ['Technology', 'Consumer Cyclical', 'Communication Services']
    DEFENSIVE_SECTORS = ['Utilities', 'Consumer Defensive', 'Healthcare']

    # Sector ETFs
    SECTOR_ETFS = {
        'XLK': 'Technology',
        'XLY': 'Consumer Cyclical',
        'XLC': 'Communication Services',
        'XLF': 'Financials',
        'XLE': 'Energy',
        'XLI': 'Industrials',
        'XLB': 'Materials',
        'XLU': 'Utilities',
        'XLP': 'Consumer Defensive',
        'XLV': 'Healthcare',
        'XLRE': 'Real Estate',
    }

    def __init__(self):
        """Initialize detector."""
        self.engine = get_engine()
        self._cache = {}
        self._cache_time = None

    def detect_regime(self, use_cache: bool = True) -> MacroRegimeResult:
        """
        Detect current market regime.

        Args:
            use_cache: Use cached result if < 1 hour old

        Returns:
            MacroRegimeResult with full analysis
        """
        # Check cache
        if use_cache and self._cache_time:
            if (datetime.now() - self._cache_time).total_seconds() < 3600:
                logger.info("Using cached regime detection")
                return self._cache.get('result')

        logger.info("Detecting macro regime...")

        indicators = []
        risk_on_factors = []
        risk_off_factors = []

        # 1. VIX Analysis
        vix_indicator = self._analyze_vix()
        if vix_indicator:
            indicators.append(vix_indicator)
            if vix_indicator.signal == 'RISK_ON':
                risk_on_factors.append(vix_indicator.description)
            elif vix_indicator.signal == 'RISK_OFF':
                risk_off_factors.append(vix_indicator.description)

        # 2. Treasury Yield Spread (10Y-2Y)
        yield_indicator = self._analyze_yield_spread()
        if yield_indicator:
            indicators.append(yield_indicator)
            if yield_indicator.signal == 'RISK_ON':
                risk_on_factors.append(yield_indicator.description)
            elif yield_indicator.signal == 'RISK_OFF':
                risk_off_factors.append(yield_indicator.description)

        # 3. Stock vs Bond Performance
        svb_indicator = self._analyze_stock_vs_bond()
        if svb_indicator:
            indicators.append(svb_indicator)
            if svb_indicator.signal == 'RISK_ON':
                risk_on_factors.append(svb_indicator.description)
            elif svb_indicator.signal == 'RISK_OFF':
                risk_off_factors.append(svb_indicator.description)

        # 4. Dollar Index
        dxy_indicator = self._analyze_dollar()
        if dxy_indicator:
            indicators.append(dxy_indicator)
            if dxy_indicator.signal == 'RISK_ON':
                risk_on_factors.append(dxy_indicator.description)
            elif dxy_indicator.signal == 'RISK_OFF':
                risk_off_factors.append(dxy_indicator.description)

        # 5. Sector Leadership
        sector_indicator = self._analyze_sector_leadership()
        if sector_indicator:
            indicators.append(sector_indicator)
            if sector_indicator.signal == 'RISK_ON':
                risk_on_factors.append(sector_indicator.description)
            elif sector_indicator.signal == 'RISK_OFF':
                risk_off_factors.append(sector_indicator.description)

        # 6. Market Breadth (Advance/Decline)
        breadth_indicator = self._analyze_market_breadth()
        if breadth_indicator:
            indicators.append(breadth_indicator)
            if breadth_indicator.signal == 'RISK_ON':
                risk_on_factors.append(breadth_indicator.description)
            elif breadth_indicator.signal == 'RISK_OFF':
                risk_off_factors.append(breadth_indicator.description)

        # Calculate weighted regime score
        if indicators:
            total_weight = sum(ind.weight for ind in indicators)
            regime_score = sum(ind.score * ind.weight for ind in indicators) / total_weight
            regime_score = int(regime_score)
        else:
            regime_score = 50  # Neutral if no data

        # Determine regime
        if regime_score >= 70:
            regime = MarketRegime.STRONG_RISK_ON
        elif regime_score >= 55:
            regime = MarketRegime.RISK_ON
        elif regime_score >= 45:
            regime = MarketRegime.NEUTRAL
        elif regime_score >= 30:
            regime = MarketRegime.RISK_OFF
        else:
            regime = MarketRegime.STRONG_RISK_OFF

        # Calculate confidence based on indicator agreement
        if indicators:
            signals = [ind.signal for ind in indicators]
            most_common = max(set(signals), key=signals.count)
            agreement = signals.count(most_common) / len(signals)
            confidence = agreement
        else:
            confidence = 0.5

        # Calculate signal adjustments
        growth_adj, defensive_adj = self._calculate_adjustments(regime_score)

        # Build result
        result = MacroRegimeResult(
            regime=regime,
            regime_score=regime_score,
            confidence=confidence,
            risk_on_factors=risk_on_factors,
            risk_off_factors=risk_off_factors,
            growth_adjustment=growth_adj,
            defensive_adjustment=defensive_adj,
        )

        # Assign indicators
        for ind in indicators:
            if ind.name == 'VIX':
                result.vix = ind
            elif ind.name == 'Yield Spread':
                result.yield_spread = ind
            elif ind.name == 'Stock vs Bond':
                result.stock_vs_bond = ind
            elif ind.name == 'Dollar Index':
                result.dollar_index = ind
            elif ind.name == 'Sector Leadership':
                result.sector_leadership = ind
            elif ind.name == 'Market Breadth':
                result.market_breadth = ind

        # Cache result
        self._cache['result'] = result
        self._cache_time = datetime.now()

        logger.info(f"Regime: {regime.value} (score: {regime_score}, confidence: {confidence:.0%})")

        return result

    def _analyze_vix(self) -> Optional[RegimeIndicator]:
        """Analyze VIX (Fear Index)."""
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="5d")

            if hist.empty:
                return None

            current_vix = hist['Close'].iloc[-1]
            vix_20d_avg = vix.history(period="1mo")['Close'].mean()

            # Score: Low VIX = Risk-On, High VIX = Risk-Off
            if current_vix < 15:
                score = 85
                signal = 'RISK_ON'
                desc = f"VIX very low ({current_vix:.1f}) - extreme complacency"
            elif current_vix < 20:
                score = 70
                signal = 'RISK_ON'
                desc = f"VIX low ({current_vix:.1f}) - bullish sentiment"
            elif current_vix < 25:
                score = 50
                signal = 'NEUTRAL'
                desc = f"VIX moderate ({current_vix:.1f}) - normal conditions"
            elif current_vix < 30:
                score = 35
                signal = 'RISK_OFF'
                desc = f"VIX elevated ({current_vix:.1f}) - increasing fear"
            else:
                score = 15
                signal = 'RISK_OFF'
                desc = f"VIX high ({current_vix:.1f}) - extreme fear"

            return RegimeIndicator(
                name='VIX',
                value=current_vix,
                signal=signal,
                score=score,
                description=desc,
                weight=1.5  # VIX is important
            )

        except Exception as e:
            logger.error(f"Error analyzing VIX: {e}")
            return None

    def _analyze_yield_spread(self) -> Optional[RegimeIndicator]:
        """Analyze 10Y-2Y Treasury spread."""
        try:
            # Get treasury yields
            tnx = yf.Ticker("^TNX")  # 10-year
            twy = yf.Ticker("^IRX")  # 3-month (proxy for short-term)

            tnx_hist = tnx.history(period="5d")

            if tnx_hist.empty:
                return None

            yield_10y = tnx_hist['Close'].iloc[-1]

            # Try to get 2-year or use approximation
            try:
                twy_hist = twy.history(period="5d")
                yield_short = twy_hist['Close'].iloc[-1] if not twy_hist.empty else yield_10y - 0.5
            except:
                yield_short = yield_10y - 0.5

            spread = yield_10y - yield_short

            # Positive spread = normal = Risk-On
            # Negative spread = inverted = Risk-Off (recession signal)
            if spread > 1.0:
                score = 75
                signal = 'RISK_ON'
                desc = f"Yield curve steep (+{spread:.2f}%) - healthy economy"
            elif spread > 0.5:
                score = 65
                signal = 'RISK_ON'
                desc = f"Yield curve positive (+{spread:.2f}%) - normal"
            elif spread > 0:
                score = 50
                signal = 'NEUTRAL'
                desc = f"Yield curve flat (+{spread:.2f}%) - watch closely"
            elif spread > -0.5:
                score = 35
                signal = 'RISK_OFF'
                desc = f"Yield curve inverted ({spread:.2f}%) - recession risk"
            else:
                score = 20
                signal = 'RISK_OFF'
                desc = f"Yield curve deeply inverted ({spread:.2f}%) - high recession risk"

            return RegimeIndicator(
                name='Yield Spread',
                value=spread,
                signal=signal,
                score=score,
                description=desc,
                weight=1.2
            )

        except Exception as e:
            logger.error(f"Error analyzing yield spread: {e}")
            return None

    def _analyze_stock_vs_bond(self) -> Optional[RegimeIndicator]:
        """Analyze SPY vs TLT performance (stocks vs bonds)."""
        try:
            spy = yf.Ticker("SPY")
            tlt = yf.Ticker("TLT")

            spy_hist = spy.history(period="1mo")
            tlt_hist = tlt.history(period="1mo")

            if spy_hist.empty or tlt_hist.empty:
                return None

            # Calculate 20-day returns
            spy_return = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0] - 1) * 100
            tlt_return = (tlt_hist['Close'].iloc[-1] / tlt_hist['Close'].iloc[0] - 1) * 100

            relative = spy_return - tlt_return

            # Stocks outperforming = Risk-On
            if relative > 5:
                score = 80
                signal = 'RISK_ON'
                desc = f"Stocks crushing bonds (SPY {spy_return:+.1f}% vs TLT {tlt_return:+.1f}%)"
            elif relative > 2:
                score = 65
                signal = 'RISK_ON'
                desc = f"Stocks beating bonds (SPY {spy_return:+.1f}% vs TLT {tlt_return:+.1f}%)"
            elif relative > -2:
                score = 50
                signal = 'NEUTRAL'
                desc = f"Stocks & bonds even (SPY {spy_return:+.1f}% vs TLT {tlt_return:+.1f}%)"
            elif relative > -5:
                score = 35
                signal = 'RISK_OFF'
                desc = f"Bonds beating stocks (TLT {tlt_return:+.1f}% vs SPY {spy_return:+.1f}%)"
            else:
                score = 20
                signal = 'RISK_OFF'
                desc = f"Flight to safety - bonds crushing stocks"

            return RegimeIndicator(
                name='Stock vs Bond',
                value=relative,
                signal=signal,
                score=score,
                description=desc,
                weight=1.3
            )

        except Exception as e:
            logger.error(f"Error analyzing stock vs bond: {e}")
            return None

    def _analyze_dollar(self) -> Optional[RegimeIndicator]:
        """Analyze Dollar Index trend."""
        try:
            # UUP is Dollar Index ETF
            uup = yf.Ticker("UUP")
            hist = uup.history(period="1mo")

            if hist.empty or len(hist) < 5:
                return None

            current = hist['Close'].iloc[-1]
            month_ago = hist['Close'].iloc[0]
            week_ago = hist['Close'].iloc[-5] if len(hist) >= 5 else month_ago

            monthly_change = (current / month_ago - 1) * 100
            weekly_change = (current / week_ago - 1) * 100

            # Falling dollar = Risk-On (money flowing to risk assets)
            # Rising dollar = Risk-Off (flight to safety)
            if monthly_change < -2:
                score = 75
                signal = 'RISK_ON'
                desc = f"Dollar falling ({monthly_change:+.1f}% monthly) - risk appetite"
            elif monthly_change < -0.5:
                score = 60
                signal = 'RISK_ON'
                desc = f"Dollar weakening ({monthly_change:+.1f}% monthly)"
            elif monthly_change < 0.5:
                score = 50
                signal = 'NEUTRAL'
                desc = f"Dollar stable ({monthly_change:+.1f}% monthly)"
            elif monthly_change < 2:
                score = 40
                signal = 'RISK_OFF'
                desc = f"Dollar strengthening ({monthly_change:+.1f}% monthly)"
            else:
                score = 25
                signal = 'RISK_OFF'
                desc = f"Dollar surging ({monthly_change:+.1f}% monthly) - flight to safety"

            return RegimeIndicator(
                name='Dollar Index',
                value=monthly_change,
                signal=signal,
                score=score,
                description=desc,
                weight=0.8
            )

        except Exception as e:
            logger.error(f"Error analyzing dollar: {e}")
            return None

    def _analyze_sector_leadership(self) -> Optional[RegimeIndicator]:
        """Analyze which sectors are leading (growth vs defensive)."""
        try:
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="1mo")

            if spy_hist.empty:
                return None

            spy_return = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0] - 1) * 100

            growth_returns = []
            defensive_returns = []

            for etf, sector in self.SECTOR_ETFS.items():
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(period="1mo")
                    if not hist.empty:
                        ret = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
                        rel_ret = ret - spy_return  # Relative to SPY

                        if sector in self.GROWTH_SECTORS:
                            growth_returns.append(rel_ret)
                        elif sector in self.DEFENSIVE_SECTORS:
                            defensive_returns.append(rel_ret)
                except:
                    continue

            if not growth_returns or not defensive_returns:
                return None

            avg_growth = np.mean(growth_returns)
            avg_defensive = np.mean(defensive_returns)
            leadership = avg_growth - avg_defensive

            # Growth leading = Risk-On
            if leadership > 3:
                score = 80
                signal = 'RISK_ON'
                desc = f"Growth crushing defensive ({leadership:+.1f}% spread)"
            elif leadership > 1:
                score = 65
                signal = 'RISK_ON'
                desc = f"Growth leading ({leadership:+.1f}% spread)"
            elif leadership > -1:
                score = 50
                signal = 'NEUTRAL'
                desc = f"No clear sector leadership ({leadership:+.1f}%)"
            elif leadership > -3:
                score = 35
                signal = 'RISK_OFF'
                desc = f"Defensive leading ({abs(leadership):.1f}% spread)"
            else:
                score = 20
                signal = 'RISK_OFF'
                desc = f"Flight to defensive sectors ({abs(leadership):.1f}% spread)"

            return RegimeIndicator(
                name='Sector Leadership',
                value=leadership,
                signal=signal,
                score=score,
                description=desc,
                weight=1.2
            )

        except Exception as e:
            logger.error(f"Error analyzing sector leadership: {e}")
            return None

    def _analyze_market_breadth(self) -> Optional[RegimeIndicator]:
        """Analyze market breadth using advance/decline proxy."""
        try:
            # Use RSP (equal-weight S&P) vs SPY (cap-weight) as breadth proxy
            rsp = yf.Ticker("RSP")
            spy = yf.Ticker("SPY")

            rsp_hist = rsp.history(period="1mo")
            spy_hist = spy.history(period="1mo")

            if rsp_hist.empty or spy_hist.empty:
                return None

            rsp_return = (rsp_hist['Close'].iloc[-1] / rsp_hist['Close'].iloc[0] - 1) * 100
            spy_return = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0] - 1) * 100

            breadth = rsp_return - spy_return

            # RSP outperforming = broad participation = Risk-On
            # SPY outperforming = narrow leadership = potentially Risk-Off
            if breadth > 2:
                score = 75
                signal = 'RISK_ON'
                desc = f"Broad market participation (RSP +{rsp_return:.1f}% vs SPY +{spy_return:.1f}%)"
            elif breadth > 0:
                score = 60
                signal = 'RISK_ON'
                desc = f"Healthy breadth (equal-weight keeping up)"
            elif breadth > -2:
                score = 45
                signal = 'NEUTRAL'
                desc = f"Narrow leadership (mega-caps leading)"
            else:
                score = 30
                signal = 'RISK_OFF'
                desc = f"Very narrow market - only mega-caps holding up"

            return RegimeIndicator(
                name='Market Breadth',
                value=breadth,
                signal=signal,
                score=score,
                description=desc,
                weight=1.0
            )

        except Exception as e:
            logger.error(f"Error analyzing market breadth: {e}")
            return None

    def _calculate_adjustments(self, regime_score: int) -> Tuple[int, int]:
        """
        Calculate signal adjustments based on regime.

        Returns:
            (growth_adjustment, defensive_adjustment)
        """
        if regime_score >= 70:
            # Strong Risk-On: Boost growth, reduce defensive
            return (15, -5)
        elif regime_score >= 55:
            # Risk-On: Slight boost to growth
            return (10, 0)
        elif regime_score >= 45:
            # Neutral: No adjustments
            return (0, 0)
        elif regime_score >= 30:
            # Risk-Off: Boost defensive, reduce growth
            return (-10, 10)
        else:
            # Strong Risk-Off: Strong defensive bias
            return (-15, 15)

    def get_stock_adjustment(self, ticker: str, sector: str) -> int:
        """
        Get signal adjustment for a specific stock based on regime.

        Args:
            ticker: Stock ticker
            sector: Stock's sector

        Returns:
            Points to add/subtract from signal
        """
        result = self.detect_regime()

        if sector in self.GROWTH_SECTORS:
            return result.growth_adjustment
        elif sector in self.DEFENSIVE_SECTORS:
            return result.defensive_adjustment
        else:
            # Other sectors: smaller adjustment
            return result.growth_adjustment // 2

    def get_regime_context_for_ai(self) -> str:
        """Get regime analysis formatted for AI Chat."""
        result = self.detect_regime()

        context = f"""
🌍 MACRO REGIME ANALYSIS
{'=' * 50}

📊 CURRENT REGIME: {result.regime.value}
   Regime Score: {result.regime_score}/100 (higher = more risk-on)
   Confidence: {result.confidence:.0%}

📈 SIGNAL ADJUSTMENTS:
   Growth Stocks: {result.growth_adjustment:+d} points
   Defensive Stocks: {result.defensive_adjustment:+d} points

"""

        if result.risk_on_factors:
            context += "✅ RISK-ON FACTORS:\n"
            for factor in result.risk_on_factors:
                context += f"   • {factor}\n"
            context += "\n"

        if result.risk_off_factors:
            context += "⚠️ RISK-OFF FACTORS:\n"
            for factor in result.risk_off_factors:
                context += f"   • {factor}\n"
            context += "\n"

        # Add indicator details
        context += "📉 INDICATOR DETAILS:\n"
        for ind_name in ['vix', 'yield_spread', 'stock_vs_bond', 'dollar_index',
                         'sector_leadership', 'market_breadth']:
            ind = getattr(result, ind_name)
            if ind:
                emoji = "🟢" if ind.signal == 'RISK_ON' else "🔴" if ind.signal == 'RISK_OFF' else "🟡"
                context += f"   {emoji} {ind.name}: {ind.description}\n"

        return context


# ============================================================
# File-based caching for persistence
# ============================================================

def save_regime_to_cache(result: MacroRegimeResult):
    """Save regime result to file cache."""
    try:
        import os
        os.makedirs("data", exist_ok=True)

        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'result': result.to_dict()
        }

        with open("data/macro_regime_cache.json", 'w') as f:
            json.dump(cache_data, f, indent=2)

    except Exception as e:
        logger.error(f"Error saving regime cache: {e}")


def load_regime_from_cache(max_age_hours: int = 4) -> Optional[MacroRegimeResult]:
    """Load regime result from file cache if fresh."""
    try:
        cache_file = "data/macro_regime_cache.json"

        if not os.path.exists(cache_file):
            return None

        with open(cache_file, 'r') as f:
            cache_data = json.load(f)

        cache_time = datetime.fromisoformat(cache_data['timestamp'])

        if (datetime.now() - cache_time).total_seconds() > max_age_hours * 3600:
            return None

        # Reconstruct result
        data = cache_data['result']

        result = MacroRegimeResult(
            regime=MarketRegime(data['regime']),
            regime_score=data['regime_score'],
            confidence=data['confidence'],
            risk_on_factors=data.get('risk_on_factors', []),
            risk_off_factors=data.get('risk_off_factors', []),
            growth_adjustment=data.get('growth_adjustment', 0),
            defensive_adjustment=data.get('defensive_adjustment', 0),
        )

        # Reconstruct indicators
        for ind_name in ['vix', 'yield_spread', 'stock_vs_bond', 'dollar_index',
                         'sector_leadership', 'market_breadth']:
            ind_data = data.get(ind_name)
            if ind_data:
                setattr(result, ind_name, RegimeIndicator(**ind_data))

        logger.info(f"Loaded regime from cache ({(datetime.now() - cache_time).total_seconds() / 3600:.1f}h old)")
        return result

    except Exception as e:
        logger.error(f"Error loading regime cache: {e}")
        return None


# ============================================================
# Convenience Functions
# ============================================================

_detector = None


def get_detector() -> MacroRegimeDetector:
    """Get singleton detector instance."""
    global _detector
    if _detector is None:
        _detector = MacroRegimeDetector()
    return _detector


def get_current_regime() -> MacroRegimeResult:
    """Get current macro regime (with caching)."""
    # Try file cache first
    cached = load_regime_from_cache(max_age_hours=4)
    if cached:
        return cached

    # Calculate fresh
    detector = get_detector()
    result = detector.detect_regime()

    # Save to cache
    save_regime_to_cache(result)

    return result


def get_regime_adjustment(ticker: str, sector: str) -> int:
    """Get signal adjustment for a stock based on regime."""
    detector = get_detector()
    return detector.get_stock_adjustment(ticker, sector)


def get_regime_for_ai() -> str:
    """Get regime analysis for AI chat."""
    detector = get_detector()
    return detector.get_regime_context_for_ai()


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    print("Detecting macro regime...\n")

    detector = MacroRegimeDetector()
    result = detector.detect_regime(use_cache=False)

    print(f"Regime: {result.regime.value}")
    print(f"Score: {result.regime_score}/100")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"\nGrowth Adjustment: {result.growth_adjustment:+d}")
    print(f"Defensive Adjustment: {result.defensive_adjustment:+d}")

    print("\n" + "=" * 50)
    print(detector.get_regime_context_for_ai())