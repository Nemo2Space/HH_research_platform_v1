"""
Cross-Asset Signals Module - Phase 2

Analyzes relationships between equities and other asset classes
to generate trading signals.

Key relationships:
- Bonds (TLT/IEF): Rising rates hurt growth, help financials
- Dollar (DXY/UUP): Strong dollar hurts multinationals, helps domestic
- Gold (GLD): Risk-off indicator, inflation hedge
- Oil (USO/CL): Energy correlation, inflation input
- VIX: Fear gauge, mean-reverting
- Credit (HYG/LQD): Risk appetite indicator
- Copper (COPX): Economic growth proxy

Signals:
- Divergences between assets and equities
- Correlation breakdowns
- Intermarket confirmation/divergence
- Sector rotation signals

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


class AssetClass(Enum):
    """Asset classes for cross-asset analysis."""
    EQUITY = "EQUITY"
    BONDS = "BONDS"
    DOLLAR = "DOLLAR"
    GOLD = "GOLD"
    OIL = "OIL"
    VOLATILITY = "VOLATILITY"
    CREDIT = "CREDIT"
    COPPER = "COPPER"


class CrossAssetSignal(Enum):
    """Cross-asset signal types."""
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    DIVERGENCE = "DIVERGENCE"
    CONFIRMATION = "CONFIRMATION"
    ROTATION = "ROTATION"
    NEUTRAL = "NEUTRAL"


class SectorRotation(Enum):
    """Sector rotation signals."""
    EARLY_CYCLE = "EARLY_CYCLE"      # Recovery - favor cyclicals
    MID_CYCLE = "MID_CYCLE"          # Expansion - favor growth
    LATE_CYCLE = "LATE_CYCLE"        # Peak - favor defensives
    RECESSION = "RECESSION"          # Contraction - favor bonds/gold


@dataclass
class AssetMetrics:
    """Metrics for a single asset."""
    ticker: str
    asset_class: AssetClass
    current_price: float = 0.0
    change_1d: float = 0.0
    change_5d: float = 0.0
    change_20d: float = 0.0
    change_60d: float = 0.0
    percentile_52w: float = 50.0     # Where in 52-week range
    trend: str = "NEUTRAL"           # UP, DOWN, NEUTRAL
    momentum: float = 0.0            # Rate of change
    relative_strength: float = 0.0   # vs SPY


@dataclass
class CrossAssetAnalysis:
    """Complete cross-asset analysis."""
    as_of_date: date

    # Asset metrics
    assets: Dict[str, AssetMetrics] = field(default_factory=dict)

    # Correlations
    stock_bond_corr: float = 0.0     # SPY vs TLT
    stock_dollar_corr: float = 0.0   # SPY vs DXY
    stock_gold_corr: float = 0.0     # SPY vs GLD
    stock_oil_corr: float = 0.0      # SPY vs USO

    # Divergences
    divergences: List[str] = field(default_factory=list)

    # Overall signals
    primary_signal: CrossAssetSignal = CrossAssetSignal.NEUTRAL
    signal_strength: int = 50
    risk_appetite: str = "NEUTRAL"   # RISK_ON, RISK_OFF, NEUTRAL

    # Sector rotation
    cycle_phase: SectorRotation = SectorRotation.MID_CYCLE
    favored_sectors: List[str] = field(default_factory=list)
    avoid_sectors: List[str] = field(default_factory=list)

    # Specific signals
    bond_signal: str = "NEUTRAL"     # Bonds signaling for equities
    dollar_signal: str = "NEUTRAL"   # Dollar signaling for equities
    commodity_signal: str = "NEUTRAL"
    credit_signal: str = "NEUTRAL"

    # Warnings
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'as_of_date': str(self.as_of_date),
            'primary_signal': self.primary_signal.value,
            'signal_strength': self.signal_strength,
            'risk_appetite': self.risk_appetite,
            'cycle_phase': self.cycle_phase.value,
            'bond_signal': self.bond_signal,
            'dollar_signal': self.dollar_signal,
            'commodity_signal': self.commodity_signal,
            'credit_signal': self.credit_signal,
            'favored_sectors': self.favored_sectors,
            'avoid_sectors': self.avoid_sectors,
            'divergences': self.divergences,
            'warnings': self.warnings,
        }

    def get_summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 55,
            "CROSS-ASSET ANALYSIS",
            "=" * 55,
            f"Date: {self.as_of_date}",
            f"Primary Signal: {self.primary_signal.value} (strength: {self.signal_strength})",
            f"Risk Appetite: {self.risk_appetite}",
            f"Cycle Phase: {self.cycle_phase.value}",
            "",
            "Asset Signals:",
            f"  Bonds:      {self.bond_signal}",
            f"  Dollar:     {self.dollar_signal}",
            f"  Commodities: {self.commodity_signal}",
            f"  Credit:     {self.credit_signal}",
        ]

        if self.favored_sectors:
            lines.append(f"\n✅ Favored Sectors: {', '.join(self.favored_sectors)}")
        if self.avoid_sectors:
            lines.append(f"❌ Avoid Sectors: {', '.join(self.avoid_sectors)}")

        if self.divergences:
            lines.append("\n⚠️ Divergences:")
            for d in self.divergences:
                lines.append(f"  • {d}")

        if self.warnings:
            lines.append("\n🚨 Warnings:")
            for w in self.warnings:
                lines.append(f"  • {w}")

        lines.append("=" * 55)
        return "\n".join(lines)


class CrossAssetAnalyzer:
    """
    Analyzes cross-asset relationships for trading signals.
    """

    # Asset tickers to track
    ASSETS = {
        'SPY': AssetClass.EQUITY,
        'QQQ': AssetClass.EQUITY,
        'IWM': AssetClass.EQUITY,
        'TLT': AssetClass.BONDS,      # Long-term bonds
        'IEF': AssetClass.BONDS,      # Intermediate bonds
        'SHY': AssetClass.BONDS,      # Short-term bonds
        'UUP': AssetClass.DOLLAR,     # Dollar bull
        'GLD': AssetClass.GOLD,
        'USO': AssetClass.OIL,
        'VXX': AssetClass.VOLATILITY, # VIX ETN
        'HYG': AssetClass.CREDIT,     # High yield
        'LQD': AssetClass.CREDIT,     # Investment grade
        'COPX': AssetClass.COPPER,    # Copper miners
    }

    # Sector ETFs for rotation
    SECTORS = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLE': 'Energy',
        'XLV': 'Healthcare',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLI': 'Industrials',
        'XLB': 'Materials',
        'XLU': 'Utilities',
        'XLRE': 'Real Estate',
    }

    def __init__(self):
        self._cache = {}
        self._cache_time = {}
        self._cache_duration = timedelta(minutes=15)

    def analyze(self) -> CrossAssetAnalysis:
        """
        Perform complete cross-asset analysis.

        Returns:
            CrossAssetAnalysis with all signals
        """
        analysis = CrossAssetAnalysis(as_of_date=date.today())

        try:
            # Fetch all asset data
            asset_data = self._fetch_asset_data()

            if not asset_data:
                analysis.warnings.append("Could not fetch asset data")
                return analysis

            # Calculate metrics for each asset
            for ticker, df in asset_data.items():
                if ticker in self.ASSETS:
                    metrics = self._calculate_asset_metrics(ticker, df)
                    analysis.assets[ticker] = metrics

            # Calculate correlations
            self._calculate_correlations(analysis, asset_data)

            # Analyze each relationship
            self._analyze_bond_signal(analysis)
            self._analyze_dollar_signal(analysis)
            self._analyze_commodity_signal(analysis)
            self._analyze_credit_signal(analysis)

            # Detect divergences
            self._detect_divergences(analysis)

            # Determine cycle phase
            self._determine_cycle_phase(analysis)

            # Generate overall signal
            self._generate_overall_signal(analysis)

        except Exception as e:
            logger.error(f"Cross-asset analysis error: {e}")
            analysis.warnings.append(f"Analysis error: {str(e)}")

        return analysis

    def _fetch_asset_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch price data for all tracked assets."""

        cache_key = 'asset_data'
        if self._is_cached(cache_key):
            return self._cache[cache_key]

        try:
            import yfinance as yf

            tickers = list(self.ASSETS.keys()) + list(self.SECTORS.keys())

            # Fetch 6 months of data
            data = yf.download(
                tickers,
                period="6mo",
                progress=False,
                group_by='ticker'
            )

            result = {}
            for ticker in tickers:
                try:
                    if ticker in data.columns.get_level_values(0):
                        df = data[ticker].copy()
                        if not df.empty:
                            result[ticker] = df
                except:
                    pass

            # If multi-ticker download failed, try individual
            if len(result) < len(tickers) / 2:
                for ticker in tickers:
                    if ticker not in result:
                        try:
                            t = yf.Ticker(ticker)
                            df = t.history(period="6mo")
                            if not df.empty:
                                result[ticker] = df
                        except:
                            pass

            self._cache[cache_key] = result
            self._cache_time[cache_key] = datetime.now()

            return result

        except Exception as e:
            logger.error(f"Error fetching asset data: {e}")
            return {}

    def _calculate_asset_metrics(self, ticker: str,
                                  df: pd.DataFrame) -> AssetMetrics:
        """Calculate metrics for a single asset."""

        metrics = AssetMetrics(
            ticker=ticker,
            asset_class=self.ASSETS.get(ticker, AssetClass.EQUITY),
        )

        if df.empty:
            return metrics

        close = df['Close'] if 'Close' in df.columns else df['Adj Close']

        metrics.current_price = float(close.iloc[-1])

        # Price changes
        if len(close) >= 2:
            metrics.change_1d = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
        if len(close) >= 6:
            metrics.change_5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100)
        if len(close) >= 21:
            metrics.change_20d = float((close.iloc[-1] / close.iloc[-21] - 1) * 100)
        if len(close) >= 61:
            metrics.change_60d = float((close.iloc[-1] / close.iloc[-61] - 1) * 100)

        # 52-week percentile
        high_52w = close.tail(252).max() if len(close) >= 252 else close.max()
        low_52w = close.tail(252).min() if len(close) >= 252 else close.min()

        if high_52w > low_52w:
            metrics.percentile_52w = float(
                (metrics.current_price - low_52w) / (high_52w - low_52w) * 100
            )

        # Trend
        if len(close) >= 50:
            sma_50 = close.rolling(50).mean().iloc[-1]
            sma_20 = close.rolling(20).mean().iloc[-1]

            if metrics.current_price > sma_50 and sma_20 > sma_50:
                metrics.trend = "UP"
            elif metrics.current_price < sma_50 and sma_20 < sma_50:
                metrics.trend = "DOWN"

        # Momentum (20-day rate of change)
        if len(close) >= 21:
            metrics.momentum = float((close.iloc[-1] / close.iloc[-21] - 1) * 100)

        return metrics

    def _calculate_correlations(self, analysis: CrossAssetAnalysis,
                                 asset_data: Dict[str, pd.DataFrame]):
        """Calculate rolling correlations between assets."""

        # Build returns DataFrame
        returns = pd.DataFrame()

        for ticker, df in asset_data.items():
            if 'Close' in df.columns:
                returns[ticker] = df['Close'].pct_change()
            elif 'Adj Close' in df.columns:
                returns[ticker] = df['Adj Close'].pct_change()

        if returns.empty:
            return

        # 20-day rolling correlations
        if 'SPY' in returns.columns:
            if 'TLT' in returns.columns:
                corr = returns['SPY'].tail(20).corr(returns['TLT'].tail(20))
                analysis.stock_bond_corr = float(corr) if not pd.isna(corr) else 0

            if 'UUP' in returns.columns:
                corr = returns['SPY'].tail(20).corr(returns['UUP'].tail(20))
                analysis.stock_dollar_corr = float(corr) if not pd.isna(corr) else 0

            if 'GLD' in returns.columns:
                corr = returns['SPY'].tail(20).corr(returns['GLD'].tail(20))
                analysis.stock_gold_corr = float(corr) if not pd.isna(corr) else 0

            if 'USO' in returns.columns:
                corr = returns['SPY'].tail(20).corr(returns['USO'].tail(20))
                analysis.stock_oil_corr = float(corr) if not pd.isna(corr) else 0

    def _analyze_bond_signal(self, analysis: CrossAssetAnalysis):
        """Analyze bond market signal for equities."""

        tlt = analysis.assets.get('TLT')
        ief = analysis.assets.get('IEF')

        if not tlt:
            return

        # Rising bonds (falling yields) = typically bullish for growth
        # Falling bonds (rising yields) = typically bearish for growth, good for value

        if tlt.change_20d > 3:
            analysis.bond_signal = "BULLISH_GROWTH"
            analysis.favored_sectors.extend(['Technology', 'Consumer Discretionary'])
        elif tlt.change_20d < -3:
            analysis.bond_signal = "BULLISH_VALUE"
            analysis.favored_sectors.extend(['Financials', 'Energy'])
        else:
            analysis.bond_signal = "NEUTRAL"

        # Yield curve (TLT vs IEF spread)
        if tlt and ief:
            spread_change = tlt.change_20d - ief.change_20d
            if spread_change > 2:
                analysis.warnings.append("Yield curve steepening - growth expectations rising")
            elif spread_change < -2:
                analysis.warnings.append("Yield curve flattening - growth concerns")

    def _analyze_dollar_signal(self, analysis: CrossAssetAnalysis):
        """Analyze dollar signal for equities."""

        uup = analysis.assets.get('UUP')

        if not uup:
            return

        # Strong dollar = headwind for multinationals, tailwind for domestic
        if uup.change_20d > 2:
            analysis.dollar_signal = "BEARISH_MULTINATIONALS"
            analysis.avoid_sectors.append('Technology')  # High foreign revenue
            analysis.favored_sectors.append('Utilities')  # Domestic focused
        elif uup.change_20d < -2:
            analysis.dollar_signal = "BULLISH_MULTINATIONALS"
            analysis.favored_sectors.append('Technology')
        else:
            analysis.dollar_signal = "NEUTRAL"

    def _analyze_commodity_signal(self, analysis: CrossAssetAnalysis):
        """Analyze commodity signals."""

        gld = analysis.assets.get('GLD')
        uso = analysis.assets.get('USO')
        copx = analysis.assets.get('COPX')

        signals = []

        # Gold - risk-off / inflation hedge
        if gld:
            if gld.change_20d > 5:
                signals.append("RISK_OFF")
                analysis.warnings.append("Gold surging - risk-off signal")
            elif gld.change_20d < -5:
                signals.append("RISK_ON")

        # Oil - inflation / energy sector
        if uso:
            if uso.change_20d > 10:
                analysis.favored_sectors.append('Energy')
                analysis.avoid_sectors.append('Consumer Discretionary')
            elif uso.change_20d < -10:
                analysis.avoid_sectors.append('Energy')
                analysis.favored_sectors.append('Consumer Discretionary')

        # Copper - economic growth proxy
        if copx:
            if copx.change_20d > 5:
                signals.append("GROWTH")
                analysis.favored_sectors.extend(['Industrials', 'Materials'])
            elif copx.change_20d < -5:
                signals.append("SLOWDOWN")
                analysis.avoid_sectors.extend(['Industrials', 'Materials'])

        # Aggregate commodity signal
        if "RISK_OFF" in signals:
            analysis.commodity_signal = "RISK_OFF"
        elif "GROWTH" in signals:
            analysis.commodity_signal = "BULLISH"
        elif "SLOWDOWN" in signals:
            analysis.commodity_signal = "BEARISH"
        else:
            analysis.commodity_signal = "NEUTRAL"

    def _analyze_credit_signal(self, analysis: CrossAssetAnalysis):
        """Analyze credit market signal."""

        hyg = analysis.assets.get('HYG')
        lqd = analysis.assets.get('LQD')

        if not hyg or not lqd:
            return

        # HYG/LQD ratio = risk appetite
        # Rising ratio = risk-on
        # Falling ratio = risk-off

        hyg_vs_lqd = hyg.change_20d - lqd.change_20d

        if hyg_vs_lqd > 2:
            analysis.credit_signal = "RISK_ON"
        elif hyg_vs_lqd < -2:
            analysis.credit_signal = "RISK_OFF"
            analysis.warnings.append("Credit spreads widening - risk-off")
        else:
            analysis.credit_signal = "NEUTRAL"

    def _detect_divergences(self, analysis: CrossAssetAnalysis):
        """Detect divergences between asset classes."""

        spy = analysis.assets.get('SPY')
        tlt = analysis.assets.get('TLT')
        hyg = analysis.assets.get('HYG')

        if not spy:
            return

        # Stock/Bond divergence
        if tlt:
            if spy.change_20d > 3 and tlt.change_20d > 3:
                analysis.divergences.append(
                    "Stocks and bonds both rising - unusual, watch for reversal"
                )
            elif spy.change_20d < -3 and tlt.change_20d < -3:
                analysis.divergences.append(
                    "Stocks and bonds both falling - liquidity crisis risk"
                )

        # Stock/Credit divergence
        if hyg:
            if spy.change_20d > 3 and hyg.change_20d < -1:
                analysis.divergences.append(
                    "Stocks up but credit weak - bearish divergence"
                )
            elif spy.change_20d < -3 and hyg.change_20d > 1:
                analysis.divergences.append(
                    "Stocks down but credit holding - bullish divergence"
                )

    def _determine_cycle_phase(self, analysis: CrossAssetAnalysis):
        """Determine economic cycle phase from asset behavior."""

        spy = analysis.assets.get('SPY')
        tlt = analysis.assets.get('TLT')
        copx = analysis.assets.get('COPX')

        # Simple cycle model based on asset trends
        stocks_up = spy and spy.trend == "UP"
        bonds_up = tlt and tlt.trend == "UP"
        copper_up = copx and copx.trend == "UP"

        if stocks_up and copper_up and not bonds_up:
            analysis.cycle_phase = SectorRotation.MID_CYCLE
            analysis.favored_sectors.extend(['Technology', 'Industrials'])
        elif stocks_up and bonds_up:
            analysis.cycle_phase = SectorRotation.EARLY_CYCLE
            analysis.favored_sectors.extend(['Financials', 'Consumer Discretionary'])
        elif not stocks_up and bonds_up:
            analysis.cycle_phase = SectorRotation.RECESSION
            analysis.favored_sectors.extend(['Utilities', 'Consumer Staples', 'Healthcare'])
        elif not stocks_up and not bonds_up:
            analysis.cycle_phase = SectorRotation.LATE_CYCLE
            analysis.favored_sectors.extend(['Energy', 'Materials'])

        # Deduplicate
        analysis.favored_sectors = list(set(analysis.favored_sectors))
        analysis.avoid_sectors = list(set(analysis.avoid_sectors))

        # Remove conflicts
        analysis.favored_sectors = [
            s for s in analysis.favored_sectors if s not in analysis.avoid_sectors
        ]

    def _generate_overall_signal(self, analysis: CrossAssetAnalysis):
        """Generate overall cross-asset signal."""

        score = 50

        # Bond signal impact
        if analysis.bond_signal == "BULLISH_GROWTH":
            score += 10
        elif analysis.bond_signal == "BULLISH_VALUE":
            score += 5

        # Credit signal impact
        if analysis.credit_signal == "RISK_ON":
            score += 15
        elif analysis.credit_signal == "RISK_OFF":
            score -= 15

        # Commodity signal impact
        if analysis.commodity_signal == "BULLISH":
            score += 10
        elif analysis.commodity_signal == "RISK_OFF":
            score -= 10
        elif analysis.commodity_signal == "BEARISH":
            score -= 5

        # Divergence penalty
        score -= len(analysis.divergences) * 5

        analysis.signal_strength = max(0, min(100, score))

        # Determine primary signal
        if score >= 70:
            analysis.primary_signal = CrossAssetSignal.RISK_ON
            analysis.risk_appetite = "RISK_ON"
        elif score >= 55:
            analysis.primary_signal = CrossAssetSignal.CONFIRMATION
            analysis.risk_appetite = "RISK_ON"
        elif score <= 30:
            analysis.primary_signal = CrossAssetSignal.RISK_OFF
            analysis.risk_appetite = "RISK_OFF"
        elif score <= 45:
            analysis.primary_signal = CrossAssetSignal.DIVERGENCE
            analysis.risk_appetite = "NEUTRAL"
        else:
            analysis.primary_signal = CrossAssetSignal.NEUTRAL
            analysis.risk_appetite = "NEUTRAL"

    def _is_cached(self, key: str) -> bool:
        if key not in self._cache or key not in self._cache_time:
            return False
        age = datetime.now() - self._cache_time[key]
        return age < self._cache_duration


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_analyzer = None

def get_cross_asset_analyzer() -> CrossAssetAnalyzer:
    """Get singleton analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = CrossAssetAnalyzer()
    return _analyzer


def get_cross_asset_signals() -> CrossAssetAnalysis:
    """
    Get complete cross-asset analysis.

    Usage:
        analysis = get_cross_asset_signals()
        print(analysis.get_summary())
        print(f"Risk Appetite: {analysis.risk_appetite}")
    """
    analyzer = get_cross_asset_analyzer()
    return analyzer.analyze()


def get_sector_rotation_signal() -> Tuple[str, List[str], List[str]]:
    """
    Get sector rotation signal.

    Returns:
        Tuple of (cycle_phase, favored_sectors, avoid_sectors)

    Usage:
        phase, favor, avoid = get_sector_rotation_signal()
        print(f"Cycle: {phase}")
        print(f"Favor: {favor}")
    """
    analysis = get_cross_asset_signals()
    return (
        analysis.cycle_phase.value,
        analysis.favored_sectors,
        analysis.avoid_sectors
    )


def is_risk_on_cross_asset() -> bool:
    """
    Simple risk-on/off check from cross-asset analysis.

    Usage:
        if is_risk_on_cross_asset():
            # Favor growth, high beta
        else:
            # Favor defensive, quality
    """
    analysis = get_cross_asset_signals()
    return analysis.risk_appetite == "RISK_ON"


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Cross-Asset Analysis...")

    analysis = get_cross_asset_signals()
    print(analysis.get_summary())