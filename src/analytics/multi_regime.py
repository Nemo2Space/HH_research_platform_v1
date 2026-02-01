"""
Multi-Dimensional Regime Module - Phase 1

Combines multiple market dimensions to identify the current market regime:
- Interest Rate Regime (rising/falling/stable)
- Dollar Regime (strong/weak/neutral)
- Breadth Regime (healthy/deteriorating/crisis)
- Volatility Regime (low/normal/high/crisis)
- Credit Regime (tight/normal/easy)
- Liquidity Regime (abundant/normal/scarce)

This provides context for when different strategies work:
- Momentum works in low vol, easy credit
- Value works in high vol recoveries
- Quality works in tightening credit

Author: Alpha Research Platform
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from src.utils.logging import get_logger

logger = get_logger(__name__)


class RegimeState(Enum):
    """Generic regime states."""
    VERY_BULLISH = "VERY_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    VERY_BEARISH = "VERY_BEARISH"


class RateRegime(Enum):
    """Interest rate environment."""
    FALLING_FAST = "FALLING_FAST"  # Aggressive cuts
    FALLING = "FALLING"  # Cutting cycle
    STABLE_LOW = "STABLE_LOW"  # Low and stable
    STABLE = "STABLE"  # Normal
    RISING = "RISING"  # Hiking cycle
    RISING_FAST = "RISING_FAST"  # Aggressive hikes


class DollarRegime(Enum):
    """USD strength environment."""
    VERY_WEAK = "VERY_WEAK"  # DXY < 95
    WEAK = "WEAK"  # DXY 95-100
    NEUTRAL = "NEUTRAL"  # DXY 100-105
    STRONG = "STRONG"  # DXY 105-110
    VERY_STRONG = "VERY_STRONG"  # DXY > 110


class BreadthRegime(Enum):
    """Market breadth environment."""
    STRONG = "STRONG"  # >70% above 200 DMA
    HEALTHY = "HEALTHY"  # 50-70% above 200 DMA
    MIXED = "MIXED"  # 30-50% above 200 DMA
    WEAK = "WEAK"  # 10-30% above 200 DMA
    CRISIS = "CRISIS"  # <10% above 200 DMA


class VolRegime(Enum):
    """Volatility environment."""
    VERY_LOW = "VERY_LOW"  # VIX < 12
    LOW = "LOW"  # VIX 12-16
    NORMAL = "NORMAL"  # VIX 16-22
    ELEVATED = "ELEVATED"  # VIX 22-30
    HIGH = "HIGH"  # VIX 30-40
    CRISIS = "CRISIS"  # VIX > 40


class CreditRegime(Enum):
    """Credit spread environment."""
    VERY_TIGHT = "VERY_TIGHT"  # Spreads very narrow
    TIGHT = "TIGHT"  # Risk-on
    NORMAL = "NORMAL"  # Average
    WIDE = "WIDE"  # Risk-off
    CRISIS = "CRISIS"  # Blowout


@dataclass
class RegimeSnapshot:
    """Complete multi-dimensional regime snapshot."""
    as_of_date: date

    # Individual regimes
    rate_regime: RateRegime = RateRegime.STABLE
    dollar_regime: DollarRegime = DollarRegime.NEUTRAL
    breadth_regime: BreadthRegime = BreadthRegime.HEALTHY
    vol_regime: VolRegime = VolRegime.NORMAL
    credit_regime: CreditRegime = CreditRegime.NORMAL

    # Composite
    overall_regime: RegimeState = RegimeState.NEUTRAL
    regime_score: int = 50  # 0-100, higher = more bullish

    # Raw values
    fed_funds_rate: float = 0.0
    rate_change_3m: float = 0.0
    dxy_level: float = 100.0
    dxy_change_3m: float = 0.0
    pct_above_200dma: float = 50.0
    advance_decline_ratio: float = 1.0
    vix_level: float = 18.0
    vix_percentile: float = 50.0
    credit_spread: float = 3.5
    yield_curve_slope: float = 0.0

    # Strategy implications
    favored_strategies: List[str] = field(default_factory=list)
    avoid_strategies: List[str] = field(default_factory=list)

    # Signals
    risk_on: bool = True
    regime_change_warning: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'as_of_date': str(self.as_of_date),
            'overall_regime': self.overall_regime.value,
            'regime_score': self.regime_score,
            'rate_regime': self.rate_regime.value,
            'dollar_regime': self.dollar_regime.value,
            'breadth_regime': self.breadth_regime.value,
            'vol_regime': self.vol_regime.value,
            'credit_regime': self.credit_regime.value,
            'vix_level': round(self.vix_level, 1),
            'dxy_level': round(self.dxy_level, 1),
            'pct_above_200dma': round(self.pct_above_200dma, 1),
            'risk_on': self.risk_on,
            'favored_strategies': self.favored_strategies,
            'avoid_strategies': self.avoid_strategies,
            'warnings': self.warnings,
        }

    def get_summary(self) -> str:
        """Human-readable regime summary."""
        lines = [
            "=" * 55,
            "MULTI-DIMENSIONAL REGIME ANALYSIS",
            "=" * 55,
            f"Date: {self.as_of_date}",
            f"Overall: {self.overall_regime.value} (Score: {self.regime_score}/100)",
            "",
            "Dimension Breakdown:",
            f"  Rate Environment:    {self.rate_regime.value}",
            f"  Dollar Strength:     {self.dollar_regime.value} (DXY: {self.dxy_level:.1f})",
            f"  Market Breadth:      {self.breadth_regime.value} ({self.pct_above_200dma:.0f}% > 200 DMA)",
            f"  Volatility:          {self.vol_regime.value} (VIX: {self.vix_level:.1f})",
            f"  Credit Conditions:   {self.credit_regime.value}",
            "",
            f"Risk Appetite: {'🟢 RISK-ON' if self.risk_on else '🔴 RISK-OFF'}",
        ]

        if self.favored_strategies:
            lines.append(f"\n✅ Favored Strategies: {', '.join(self.favored_strategies)}")
        if self.avoid_strategies:
            lines.append(f"❌ Avoid Strategies: {', '.join(self.avoid_strategies)}")

        if self.warnings:
            lines.append("\n⚠️ Warnings:")
            for w in self.warnings:
                lines.append(f"  • {w}")

        lines.append("=" * 55)
        return "\n".join(lines)


class RegimeDataProvider:
    """
    Fetches market data for regime detection.
    """

    def __init__(self):
        self._cache = {}
        self._cache_time = {}
        self._cache_duration = timedelta(minutes=30)

    def get_vix(self) -> Dict[str, float]:
        """Get VIX data."""
        if self._is_cached('vix'):
            return self._cache['vix']

        try:
            import yfinance as yf

            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1y")

            if hist.empty:
                return {'level': 18.0, 'percentile': 50.0, 'change_1m': 0.0}

            current = hist['Close'].iloc[-1]

            # Calculate percentile
            percentile = (hist['Close'] < current).mean() * 100

            # 1-month change
            if len(hist) >= 21:
                change_1m = current - hist['Close'].iloc[-21]
            else:
                change_1m = 0.0

            result = {
                'level': float(current),
                'percentile': float(percentile),
                'change_1m': float(change_1m),
            }

            self._cache['vix'] = result
            self._cache_time['vix'] = datetime.now()
            return result

        except Exception as e:
            logger.error(f"Error fetching VIX: {e}")
            return {'level': 18.0, 'percentile': 50.0, 'change_1m': 0.0}

    def get_dxy(self) -> Dict[str, float]:
        """Get Dollar Index data."""
        if self._is_cached('dxy'):
            return self._cache['dxy']

        try:
            import yfinance as yf

            # DXY ticker
            dxy = yf.Ticker("DX-Y.NYB")
            hist = dxy.history(period="6mo")

            if hist.empty:
                # Try UUP as proxy
                uup = yf.Ticker("UUP")
                hist = uup.history(period="6mo")
                if hist.empty:
                    return {'level': 103.0, 'change_3m': 0.0}
                # Scale UUP to approximate DXY
                current = hist['Close'].iloc[-1] * 3.8  # Rough conversion
            else:
                current = hist['Close'].iloc[-1]

            # 3-month change
            if len(hist) >= 63:
                prev = hist['Close'].iloc[-63]
                change_3m = ((current / prev) - 1) * 100 if prev > 0 else 0
            else:
                change_3m = 0.0

            result = {
                'level': float(current),
                'change_3m': float(change_3m),
            }

            self._cache['dxy'] = result
            self._cache_time['dxy'] = datetime.now()
            return result

        except Exception as e:
            logger.error(f"Error fetching DXY: {e}")
            return {'level': 103.0, 'change_3m': 0.0}

    def get_treasury_yields(self) -> Dict[str, float]:
        """Get Treasury yield data."""
        if self._is_cached('yields'):
            return self._cache['yields']

        try:
            import yfinance as yf

            # Fetch yields
            tickers = {
                '2y': '^IRX',  # 3-month (proxy for short end)
                '10y': '^TNX',  # 10-year
                '30y': '^TYX',  # 30-year
            }

            yields = {}
            for name, ticker in tickers.items():
                try:
                    t = yf.Ticker(ticker)
                    hist = t.history(period="5d")
                    if not hist.empty:
                        yields[name] = float(hist['Close'].iloc[-1])
                except:
                    pass

            # Calculate slope (10y - 2y proxy)
            if '10y' in yields and '2y' in yields:
                slope = yields['10y'] - yields['2y']
            elif '10y' in yields:
                slope = yields['10y'] - 4.5  # Assume ~4.5% short rate
            else:
                slope = 0.0

            result = {
                'yield_2y': yields.get('2y', 4.5),
                'yield_10y': yields.get('10y', 4.2),
                'yield_30y': yields.get('30y', 4.5),
                'slope': slope,
            }

            self._cache['yields'] = result
            self._cache_time['yields'] = datetime.now()
            return result

        except Exception as e:
            logger.error(f"Error fetching yields: {e}")
            return {'yield_10y': 4.2, 'slope': 0.0}

    def get_market_breadth(self) -> Dict[str, float]:
        """
        Get market breadth data.
        Uses SPY components as proxy.
        """
        if self._is_cached('breadth'):
            return self._cache['breadth']

        try:
            import yfinance as yf

            # Get S&P 500 advance/decline data via breadth ETFs
            # SPXU for rough approximation

            # Check SPY 200 DMA
            spy = yf.Ticker("SPY")
            hist = spy.history(period="1y")

            if hist.empty:
                return {'pct_above_200dma': 50.0, 'ad_ratio': 1.0}

            # Calculate SPY's position vs 200 DMA
            sma_200 = hist['Close'].rolling(200).mean()
            current = hist['Close'].iloc[-1]
            sma_200_val = sma_200.iloc[-1]

            spy_above_200 = current > sma_200_val

            # Estimate breadth from SPY momentum
            # In reality, would use actual breadth data
            if spy_above_200:
                pct_above_200dma = 55 + (current / sma_200_val - 1) * 100
            else:
                pct_above_200dma = 45 - (1 - current / sma_200_val) * 100

            pct_above_200dma = max(5, min(95, pct_above_200dma))

            # A/D ratio estimate
            returns_20d = hist['Close'].pct_change(20).iloc[-1]
            ad_ratio = 1.0 + returns_20d * 2  # Rough proxy

            result = {
                'pct_above_200dma': float(pct_above_200dma),
                'ad_ratio': float(ad_ratio),
                'spy_vs_200dma': float((current / sma_200_val - 1) * 100),
            }

            self._cache['breadth'] = result
            self._cache_time['breadth'] = datetime.now()
            return result

        except Exception as e:
            logger.error(f"Error fetching breadth: {e}")
            return {'pct_above_200dma': 50.0, 'ad_ratio': 1.0}

    def get_credit_spreads(self) -> Dict[str, float]:
        """
        Get credit spread data.
        Uses HYG/LQD ratio as proxy for credit conditions.
        """
        if self._is_cached('credit'):
            return self._cache['credit']

        try:
            import yfinance as yf

            # HYG (high yield) vs LQD (investment grade)
            hyg = yf.Ticker("HYG")
            lqd = yf.Ticker("LQD")

            hyg_hist = hyg.history(period="6mo")
            lqd_hist = lqd.history(period="6mo")

            if hyg_hist.empty or lqd_hist.empty:
                return {'spread_proxy': 3.5, 'spread_percentile': 50.0}

            # Calculate ratio (higher = tighter spreads = risk-on)
            current_ratio = hyg_hist['Close'].iloc[-1] / lqd_hist['Close'].iloc[-1]

            # Historical percentile
            hyg_aligned = hyg_hist['Close'].reindex(lqd_hist.index, method='ffill')
            ratio_hist = hyg_aligned / lqd_hist['Close']
            percentile = (ratio_hist < current_ratio).mean() * 100

            # Invert to spread proxy (lower ratio = wider spreads)
            spread_proxy = 5.0 - (current_ratio - 0.7) * 10
            spread_proxy = max(2.0, min(8.0, spread_proxy))

            result = {
                'spread_proxy': float(spread_proxy),
                'spread_percentile': float(percentile),
                'hyg_lqd_ratio': float(current_ratio),
            }

            self._cache['credit'] = result
            self._cache_time['credit'] = datetime.now()
            return result

        except Exception as e:
            logger.error(f"Error fetching credit spreads: {e}")
            return {'spread_proxy': 3.5, 'spread_percentile': 50.0}

    def _is_cached(self, key: str) -> bool:
        if key not in self._cache or key not in self._cache_time:
            return False
        age = datetime.now() - self._cache_time[key]
        return age < self._cache_duration


class MultiDimensionalRegime:
    """
    Main class for multi-dimensional regime detection.
    """

    def __init__(self, data_provider: RegimeDataProvider = None):
        self.data_provider = data_provider or RegimeDataProvider()

    def get_current_regime(self) -> RegimeSnapshot:
        """
        Get current multi-dimensional regime snapshot.
        """
        snapshot = RegimeSnapshot(as_of_date=date.today())

        # Fetch all data
        vix_data = self.data_provider.get_vix()
        dxy_data = self.data_provider.get_dxy()
        yield_data = self.data_provider.get_treasury_yields()
        breadth_data = self.data_provider.get_market_breadth()
        credit_data = self.data_provider.get_credit_spreads()

        # Store raw values
        snapshot.vix_level = vix_data.get('level', 18.0)
        snapshot.vix_percentile = vix_data.get('percentile', 50.0)
        snapshot.dxy_level = dxy_data.get('level', 103.0)
        snapshot.dxy_change_3m = dxy_data.get('change_3m', 0.0)
        snapshot.pct_above_200dma = breadth_data.get('pct_above_200dma', 50.0)
        snapshot.advance_decline_ratio = breadth_data.get('ad_ratio', 1.0)
        snapshot.yield_curve_slope = yield_data.get('slope', 0.0)
        snapshot.credit_spread = credit_data.get('spread_proxy', 3.5)

        # Determine individual regimes

        # 1. Volatility Regime
        vix = snapshot.vix_level
        if vix < 12:
            snapshot.vol_regime = VolRegime.VERY_LOW
        elif vix < 16:
            snapshot.vol_regime = VolRegime.LOW
        elif vix < 22:
            snapshot.vol_regime = VolRegime.NORMAL
        elif vix < 30:
            snapshot.vol_regime = VolRegime.ELEVATED
        elif vix < 40:
            snapshot.vol_regime = VolRegime.HIGH
        else:
            snapshot.vol_regime = VolRegime.CRISIS

        # 2. Dollar Regime
        dxy = snapshot.dxy_level
        if dxy < 95:
            snapshot.dollar_regime = DollarRegime.VERY_WEAK
        elif dxy < 100:
            snapshot.dollar_regime = DollarRegime.WEAK
        elif dxy < 105:
            snapshot.dollar_regime = DollarRegime.NEUTRAL
        elif dxy < 110:
            snapshot.dollar_regime = DollarRegime.STRONG
        else:
            snapshot.dollar_regime = DollarRegime.VERY_STRONG

        # 3. Rate Regime (based on yield curve)
        slope = snapshot.yield_curve_slope
        if slope < -0.5:
            snapshot.rate_regime = RateRegime.RISING_FAST  # Inverted = tight
        elif slope < 0:
            snapshot.rate_regime = RateRegime.RISING
        elif slope < 0.5:
            snapshot.rate_regime = RateRegime.STABLE
        elif slope < 1.5:
            snapshot.rate_regime = RateRegime.STABLE_LOW
        else:
            snapshot.rate_regime = RateRegime.FALLING

        # 4. Breadth Regime
        breadth = snapshot.pct_above_200dma
        if breadth > 70:
            snapshot.breadth_regime = BreadthRegime.STRONG
        elif breadth > 50:
            snapshot.breadth_regime = BreadthRegime.HEALTHY
        elif breadth > 30:
            snapshot.breadth_regime = BreadthRegime.MIXED
        elif breadth > 10:
            snapshot.breadth_regime = BreadthRegime.WEAK
        else:
            snapshot.breadth_regime = BreadthRegime.CRISIS

        # 5. Credit Regime
        spread = snapshot.credit_spread
        if spread < 2.5:
            snapshot.credit_regime = CreditRegime.VERY_TIGHT
        elif spread < 3.5:
            snapshot.credit_regime = CreditRegime.TIGHT
        elif spread < 5.0:
            snapshot.credit_regime = CreditRegime.NORMAL
        elif spread < 7.0:
            snapshot.credit_regime = CreditRegime.WIDE
        else:
            snapshot.credit_regime = CreditRegime.CRISIS

        # Calculate overall regime score
        scores = {
            'vol': self._vol_to_score(snapshot.vol_regime),
            'breadth': self._breadth_to_score(snapshot.breadth_regime),
            'credit': self._credit_to_score(snapshot.credit_regime),
            'dollar': self._dollar_to_score(snapshot.dollar_regime),
            'rate': self._rate_to_score(snapshot.rate_regime),
        }

        # Weighted average (breadth and vol most important)
        weights = {'vol': 0.25, 'breadth': 0.30, 'credit': 0.20, 'dollar': 0.10, 'rate': 0.15}
        snapshot.regime_score = int(sum(scores[k] * weights[k] for k in scores))

        # Determine overall regime
        if snapshot.regime_score >= 75:
            snapshot.overall_regime = RegimeState.VERY_BULLISH
        elif snapshot.regime_score >= 60:
            snapshot.overall_regime = RegimeState.BULLISH
        elif snapshot.regime_score >= 40:
            snapshot.overall_regime = RegimeState.NEUTRAL
        elif snapshot.regime_score >= 25:
            snapshot.overall_regime = RegimeState.BEARISH
        else:
            snapshot.overall_regime = RegimeState.VERY_BEARISH

        # Risk on/off
        snapshot.risk_on = snapshot.regime_score >= 45

        # Strategy implications
        self._set_strategy_implications(snapshot)

        # Warnings
        self._add_warnings(snapshot)

        return snapshot

    def _vol_to_score(self, regime: VolRegime) -> int:
        mapping = {
            VolRegime.VERY_LOW: 85,
            VolRegime.LOW: 75,
            VolRegime.NORMAL: 55,
            VolRegime.ELEVATED: 35,
            VolRegime.HIGH: 20,
            VolRegime.CRISIS: 5,
        }
        return mapping.get(regime, 50)

    def _breadth_to_score(self, regime: BreadthRegime) -> int:
        mapping = {
            BreadthRegime.STRONG: 90,
            BreadthRegime.HEALTHY: 70,
            BreadthRegime.MIXED: 45,
            BreadthRegime.WEAK: 25,
            BreadthRegime.CRISIS: 5,
        }
        return mapping.get(regime, 50)

    def _credit_to_score(self, regime: CreditRegime) -> int:
        mapping = {
            CreditRegime.VERY_TIGHT: 85,
            CreditRegime.TIGHT: 70,
            CreditRegime.NORMAL: 50,
            CreditRegime.WIDE: 30,
            CreditRegime.CRISIS: 10,
        }
        return mapping.get(regime, 50)

    def _dollar_to_score(self, regime: DollarRegime) -> int:
        # Strong dollar is slightly negative for US equities
        mapping = {
            DollarRegime.VERY_WEAK: 65,
            DollarRegime.WEAK: 60,
            DollarRegime.NEUTRAL: 50,
            DollarRegime.STRONG: 40,
            DollarRegime.VERY_STRONG: 35,
        }
        return mapping.get(regime, 50)

    def _rate_to_score(self, regime: RateRegime) -> int:
        mapping = {
            RateRegime.FALLING_FAST: 80,
            RateRegime.FALLING: 65,
            RateRegime.STABLE_LOW: 55,
            RateRegime.STABLE: 50,
            RateRegime.RISING: 35,
            RateRegime.RISING_FAST: 20,
        }
        return mapping.get(regime, 50)

    def _set_strategy_implications(self, snapshot: RegimeSnapshot):
        """Set favored and avoid strategies based on regime."""

        # Low vol + tight credit = momentum works
        if snapshot.vol_regime in [VolRegime.VERY_LOW, VolRegime.LOW]:
            if snapshot.credit_regime in [CreditRegime.VERY_TIGHT, CreditRegime.TIGHT]:
                snapshot.favored_strategies.append("Momentum")
                snapshot.favored_strategies.append("Growth")

        # High vol = quality and low vol strategies work
        if snapshot.vol_regime in [VolRegime.HIGH, VolRegime.CRISIS]:
            snapshot.favored_strategies.append("Quality")
            snapshot.favored_strategies.append("Low Volatility")
            snapshot.avoid_strategies.append("Momentum")

        # Weak breadth = defensive
        if snapshot.breadth_regime in [BreadthRegime.WEAK, BreadthRegime.CRISIS]:
            snapshot.favored_strategies.append("Defensive")
            snapshot.avoid_strategies.append("Small Cap")
            snapshot.avoid_strategies.append("High Beta")

        # Strong breadth = risk-on
        if snapshot.breadth_regime == BreadthRegime.STRONG:
            snapshot.favored_strategies.append("Small Cap")
            snapshot.favored_strategies.append("Cyclicals")

        # Rising rates = value, avoid duration
        if snapshot.rate_regime in [RateRegime.RISING, RateRegime.RISING_FAST]:
            snapshot.favored_strategies.append("Value")
            snapshot.avoid_strategies.append("Long Duration")
            snapshot.avoid_strategies.append("High P/E Growth")

        # Strong dollar = domestic over international
        if snapshot.dollar_regime in [DollarRegime.STRONG, DollarRegime.VERY_STRONG]:
            snapshot.favored_strategies.append("Domestic")
            snapshot.avoid_strategies.append("International")
            snapshot.avoid_strategies.append("Emerging Markets")

        # Remove duplicates
        snapshot.favored_strategies = list(set(snapshot.favored_strategies))
        snapshot.avoid_strategies = list(set(snapshot.avoid_strategies))

    def _add_warnings(self, snapshot: RegimeSnapshot):
        """Add regime-specific warnings."""

        if snapshot.vol_regime == VolRegime.CRISIS:
            snapshot.warnings.append("VIX in crisis territory - expect large swings")

        if snapshot.breadth_regime in [BreadthRegime.WEAK, BreadthRegime.CRISIS]:
            snapshot.warnings.append("Weak breadth - market rally not broad-based")

        if snapshot.credit_regime in [CreditRegime.WIDE, CreditRegime.CRISIS]:
            snapshot.warnings.append("Credit spreads widening - risk-off environment")

        if snapshot.yield_curve_slope < -0.5:
            snapshot.warnings.append("Inverted yield curve - recession indicator")

        if snapshot.vix_level < 12:
            snapshot.warnings.append("VIX extremely low - complacency risk")

        # Regime change detection (would need historical data)
        if snapshot.vix_percentile > 80:
            snapshot.regime_change_warning = True
            snapshot.warnings.append("VIX elevated vs history - potential regime shift")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_regime_analyzer = None


def get_regime_analyzer() -> MultiDimensionalRegime:
    """Get singleton analyzer instance."""
    global _regime_analyzer
    if _regime_analyzer is None:
        _regime_analyzer = MultiDimensionalRegime()
    return _regime_analyzer


def get_current_regime() -> RegimeSnapshot:
    """
    Get current multi-dimensional regime.

    Usage:
        regime = get_current_regime()
        print(regime.get_summary())
        print(f"Favored: {regime.favored_strategies}")
    """
    analyzer = get_regime_analyzer()
    return analyzer.get_current_regime()


def get_regime_score() -> int:
    """
    Get simple regime score (0-100).

    Usage:
        score = get_regime_score()
        if score > 60:
            print("Risk-on environment")
    """
    regime = get_current_regime()
    return regime.regime_score


def is_risk_on() -> bool:
    """
    Simple risk on/off check.

    Usage:
        if is_risk_on():
            # Favor growth, momentum
        else:
            # Favor quality, defensive
    """
    regime = get_current_regime()
    return regime.risk_on


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Multi-Dimensional Regime...")

    analyzer = MultiDimensionalRegime()
    regime = analyzer.get_current_regime()

    print(regime.get_summary())
    print("\nRaw data:")
    print(f"  VIX: {regime.vix_level:.1f}")
    print(f"  DXY: {regime.dxy_level:.1f}")
    print(f"  Breadth: {regime.pct_above_200dma:.1f}%")
    print(f"  Yield Curve Slope: {regime.yield_curve_slope:.2f}%")