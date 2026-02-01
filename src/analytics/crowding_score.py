"""
Crowding Score Module - Phase 1

Detects when stocks are "crowded" - owned by too many funds chasing the same trade.
Crowded stocks tend to:
- Move together (correlation spikes)
- Crash harder in selloffs (forced selling)
- Have lower future returns (alpha already extracted)

Inputs:
- Institutional ownership concentration (13F data)
- Short interest levels
- ETF ownership overlap
- Options flow concentration
- Analyst coverage density

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


class CrowdingLevel(Enum):
    """Crowding severity levels."""
    LOW = "LOW"  # Safe - not crowded
    MODERATE = "MODERATE"  # Some concentration
    HIGH = "HIGH"  # Crowded - caution
    EXTREME = "EXTREME"  # Very crowded - high risk


@dataclass
class CrowdingMetrics:
    """Crowding metrics for a single stock."""
    ticker: str
    as_of_date: date

    # Core crowding scores (0-100, higher = more crowded)
    institutional_crowding: float = 50.0  # % owned by top funds
    hedge_fund_crowding: float = 50.0  # Hedge fund concentration
    etf_crowding: float = 50.0  # ETF ownership overlap
    short_interest_score: float = 50.0  # Short interest level
    analyst_crowding: float = 50.0  # Analyst coverage density
    options_crowding: float = 50.0  # Options flow concentration

    # Composite
    total_crowding_score: float = 50.0  # Weighted average
    crowding_level: CrowdingLevel = CrowdingLevel.MODERATE

    # Risk indicators
    momentum_crowding: bool = False  # Is this a momentum darling?
    value_trap_risk: bool = False  # Crowded value stock?
    short_squeeze_risk: bool = False  # High short + high crowding

    # Raw data
    institutional_ownership_pct: float = 0.0
    top_10_holders_pct: float = 0.0
    short_interest_pct: float = 0.0
    days_to_cover: float = 0.0
    analyst_count: int = 0
    etf_count: int = 0

    # Warnings
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ticker': self.ticker,
            'total_crowding_score': round(self.total_crowding_score, 1),
            'crowding_level': self.crowding_level.value,
            'institutional_crowding': round(self.institutional_crowding, 1),
            'hedge_fund_crowding': round(self.hedge_fund_crowding, 1),
            'etf_crowding': round(self.etf_crowding, 1),
            'short_interest_score': round(self.short_interest_score, 1),
            'momentum_crowding': self.momentum_crowding,
            'short_squeeze_risk': self.short_squeeze_risk,
            'institutional_ownership_pct': round(self.institutional_ownership_pct, 1),
            'short_interest_pct': round(self.short_interest_pct, 1),
            'warnings': self.warnings,
        }


class CrowdingDataProvider:
    """
    Fetches data needed for crowding calculations.
    Uses yfinance as primary source, can extend to other APIs.
    """

    def __init__(self):
        self._cache = {}
        self._cache_time = {}
        self._cache_duration = timedelta(hours=6)

    def get_institutional_holders(self, ticker: str) -> Dict[str, Any]:
        """Get institutional ownership data."""
        cache_key = f"inst_{ticker}"
        if self._is_cached(cache_key):
            return self._cache[cache_key]

        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)

            # Get institutional holders
            inst_holders = stock.institutional_holders
            major_holders = stock.major_holders

            result = {
                'institutional_ownership_pct': 0.0,
                'top_10_holders_pct': 0.0,
                'holder_count': 0,
                'top_holders': [],
            }

            # Parse major holders
            if major_holders is not None and not major_holders.empty:
                for _, row in major_holders.iterrows():
                    val = row.iloc[0] if len(row) > 0 else ''
                    label = row.iloc[1] if len(row) > 1 else ''

                    if isinstance(val, str) and '%' in val:
                        pct = float(val.replace('%', ''))
                        if 'institution' in str(label).lower():
                            result['institutional_ownership_pct'] = pct
                        elif 'insider' in str(label).lower():
                            result['insider_ownership_pct'] = pct

            # Parse institutional holders for concentration
            if inst_holders is not None and not inst_holders.empty:
                result['holder_count'] = len(inst_holders)

                # Top 10 concentration
                if 'pctHeld' in inst_holders.columns:
                    top_10_pct = inst_holders['pctHeld'].head(10).sum() * 100
                    result['top_10_holders_pct'] = float(top_10_pct)
                elif '% Out' in inst_holders.columns:
                    top_10_pct = inst_holders['% Out'].head(10).sum()
                    result['top_10_holders_pct'] = float(top_10_pct)

                # Get top holder names
                if 'Holder' in inst_holders.columns:
                    result['top_holders'] = inst_holders['Holder'].head(5).tolist()

            self._cache[cache_key] = result
            self._cache_time[cache_key] = datetime.now()
            return result

        except Exception as e:
            logger.debug(f"{ticker}: Error fetching institutional data: {e}")
            return {
                'institutional_ownership_pct': 0.0,
                'top_10_holders_pct': 0.0,
                'holder_count': 0,
            }

    def get_short_interest(self, ticker: str) -> Dict[str, Any]:
        """Get short interest data."""
        cache_key = f"short_{ticker}"
        if self._is_cached(cache_key):
            return self._cache[cache_key]

        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            info = stock.info

            result = {
                'short_interest': info.get('sharesShort', 0) or 0,
                'short_interest_pct': info.get('shortPercentOfFloat', 0) or 0,
                'short_ratio': info.get('shortRatio', 0) or 0,  # Days to cover
                'shares_outstanding': info.get('sharesOutstanding', 0) or 0,
            }

            # Convert to percentage if needed
            if result['short_interest_pct'] > 0 and result['short_interest_pct'] < 1:
                result['short_interest_pct'] *= 100

            self._cache[cache_key] = result
            self._cache_time[cache_key] = datetime.now()
            return result

        except Exception as e:
            logger.debug(f"{ticker}: Error fetching short interest: {e}")
            return {
                'short_interest_pct': 0.0,
                'short_ratio': 0.0,
            }

    def get_analyst_coverage(self, ticker: str) -> Dict[str, Any]:
        """Get analyst coverage data."""
        cache_key = f"analyst_{ticker}"
        if self._is_cached(cache_key):
            return self._cache[cache_key]

        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            info = stock.info

            result = {
                'analyst_count': info.get('numberOfAnalystOpinions', 0) or 0,
                'target_mean_price': info.get('targetMeanPrice', 0) or 0,
                'recommendation': info.get('recommendationKey', 'none'),
            }

            self._cache[cache_key] = result
            self._cache_time[cache_key] = datetime.now()
            return result

        except Exception as e:
            logger.debug(f"{ticker}: Error fetching analyst data: {e}")
            return {'analyst_count': 0}

    def get_etf_exposure(self, ticker: str) -> Dict[str, Any]:
        """
        Estimate ETF exposure for a stock.
        Uses market cap as proxy - larger stocks are in more ETFs.
        """
        cache_key = f"etf_{ticker}"
        if self._is_cached(cache_key):
            return self._cache[cache_key]

        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            info = stock.info

            market_cap = info.get('marketCap', 0) or 0

            # Estimate ETF count based on market cap
            # Mega-cap (>200B): ~400+ ETFs
            # Large-cap (10-200B): ~200 ETFs
            # Mid-cap (2-10B): ~100 ETFs
            # Small-cap (<2B): ~30 ETFs

            if market_cap > 200e9:
                etf_count_estimate = 400
                etf_weight_estimate = 2.0  # ~2% of major ETFs
            elif market_cap > 50e9:
                etf_count_estimate = 300
                etf_weight_estimate = 1.0
            elif market_cap > 10e9:
                etf_count_estimate = 150
                etf_weight_estimate = 0.5
            elif market_cap > 2e9:
                etf_count_estimate = 75
                etf_weight_estimate = 0.2
            else:
                etf_count_estimate = 30
                etf_weight_estimate = 0.1

            result = {
                'etf_count_estimate': etf_count_estimate,
                'avg_etf_weight_estimate': etf_weight_estimate,
                'market_cap': market_cap,
            }

            self._cache[cache_key] = result
            self._cache_time[cache_key] = datetime.now()
            return result

        except Exception as e:
            logger.debug(f"{ticker}: Error fetching ETF exposure: {e}")
            return {'etf_count_estimate': 100, 'avg_etf_weight_estimate': 0.5}

    def _is_cached(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache or key not in self._cache_time:
            return False
        age = datetime.now() - self._cache_time[key]
        return age < self._cache_duration


class CrowdingScoreCalculator:
    """
    Calculates crowding scores for stocks.
    """

    # Thresholds for scoring
    INST_OWNERSHIP_HIGH = 85  # % institutional ownership considered high
    TOP_10_CONCENTRATION_HIGH = 40  # Top 10 holders own >40% is concentrated
    SHORT_INTEREST_HIGH = 15  # >15% short interest is high
    ANALYST_COUNT_HIGH = 30  # >30 analysts is heavily covered
    ETF_COUNT_HIGH = 300  # In >300 ETFs is saturated

    def __init__(self, data_provider: CrowdingDataProvider = None):
        self.data_provider = data_provider or CrowdingDataProvider()

    def calculate_crowding(self, ticker: str) -> CrowdingMetrics:
        """
        Calculate comprehensive crowding score for a ticker.
        """
        metrics = CrowdingMetrics(
            ticker=ticker,
            as_of_date=date.today(),
        )

        # Fetch all data
        inst_data = self.data_provider.get_institutional_holders(ticker)
        short_data = self.data_provider.get_short_interest(ticker)
        analyst_data = self.data_provider.get_analyst_coverage(ticker)
        etf_data = self.data_provider.get_etf_exposure(ticker)

        # Store raw data
        metrics.institutional_ownership_pct = inst_data.get('institutional_ownership_pct', 0)
        metrics.top_10_holders_pct = inst_data.get('top_10_holders_pct', 0)
        metrics.short_interest_pct = short_data.get('short_interest_pct', 0)
        metrics.days_to_cover = short_data.get('short_ratio', 0)
        metrics.analyst_count = analyst_data.get('analyst_count', 0)
        metrics.etf_count = etf_data.get('etf_count_estimate', 0)

        # Calculate component scores (0-100)

        # 1. Institutional Crowding
        inst_own = metrics.institutional_ownership_pct
        top_10 = metrics.top_10_holders_pct

        # High institutional ownership = more crowded
        inst_score = min(100, (inst_own / self.INST_OWNERSHIP_HIGH) * 70)
        # High concentration in top 10 = more crowded
        conc_score = min(100, (top_10 / self.TOP_10_CONCENTRATION_HIGH) * 100)
        metrics.institutional_crowding = (inst_score * 0.6 + conc_score * 0.4)

        # 2. Hedge Fund Crowding (proxy: high turnover + high inst ownership)
        # Without 13F data, use institutional as proxy
        metrics.hedge_fund_crowding = metrics.institutional_crowding * 0.8

        # 3. ETF Crowding
        etf_count = metrics.etf_count
        metrics.etf_crowding = min(100, (etf_count / self.ETF_COUNT_HIGH) * 100)

        # 4. Short Interest Score
        short_pct = metrics.short_interest_pct
        if short_pct > 0:
            metrics.short_interest_score = min(100, (short_pct / self.SHORT_INTEREST_HIGH) * 100)
        else:
            metrics.short_interest_score = 30  # Default if no data

        # 5. Analyst Crowding
        analyst_count = metrics.analyst_count
        if analyst_count > 0:
            metrics.analyst_crowding = min(100, (analyst_count / self.ANALYST_COUNT_HIGH) * 100)
        else:
            metrics.analyst_crowding = 50  # Default if no data

        # 6. Options Crowding (would need options data - use proxy)
        metrics.options_crowding = (metrics.institutional_crowding + metrics.analyst_crowding) / 2

        # Calculate total crowding score
        weights = {
            'institutional': 0.30,
            'hedge_fund': 0.15,
            'etf': 0.20,
            'short': 0.15,
            'analyst': 0.10,
            'options': 0.10,
        }

        metrics.total_crowding_score = (
                metrics.institutional_crowding * weights['institutional'] +
                metrics.hedge_fund_crowding * weights['hedge_fund'] +
                metrics.etf_crowding * weights['etf'] +
                metrics.short_interest_score * weights['short'] +
                metrics.analyst_crowding * weights['analyst'] +
                metrics.options_crowding * weights['options']
        )

        # Determine crowding level
        if metrics.total_crowding_score >= 80:
            metrics.crowding_level = CrowdingLevel.EXTREME
        elif metrics.total_crowding_score >= 65:
            metrics.crowding_level = CrowdingLevel.HIGH
        elif metrics.total_crowding_score >= 45:
            metrics.crowding_level = CrowdingLevel.MODERATE
        else:
            metrics.crowding_level = CrowdingLevel.LOW

        # Check for specific risk patterns

        # Momentum crowding: high inst + high analyst + recent outperformance
        if metrics.institutional_crowding > 70 and metrics.analyst_crowding > 70:
            metrics.momentum_crowding = True
            metrics.warnings.append("Momentum darling - vulnerable to rotation")

        # Short squeeze risk: high short + high crowding
        if metrics.short_interest_pct > 20 and metrics.institutional_crowding > 60:
            metrics.short_squeeze_risk = True
            metrics.warnings.append(f"Short squeeze risk - {metrics.short_interest_pct:.1f}% short")

        # General warnings
        if metrics.crowding_level == CrowdingLevel.EXTREME:
            metrics.warnings.append("Extremely crowded - high drawdown risk in selloffs")
        elif metrics.crowding_level == CrowdingLevel.HIGH:
            metrics.warnings.append("Crowded trade - consider position sizing")

        if metrics.top_10_holders_pct > 50:
            metrics.warnings.append(f"Top 10 holders own {metrics.top_10_holders_pct:.0f}%")

        return metrics

    def calculate_portfolio_crowding(self, positions: List[Dict]) -> Dict[str, Any]:
        """
        Calculate aggregate crowding for a portfolio.

        Args:
            positions: List of {'ticker': str, 'weight': float}

        Returns:
            Portfolio-level crowding metrics
        """
        if not positions:
            return {'portfolio_crowding_score': 50, 'crowded_positions': []}

        crowded_positions = []
        total_crowding = 0.0
        total_weight = 0.0

        for pos in positions:
            ticker = pos.get('ticker') or pos.get('symbol')
            weight = pos.get('weight', 1.0 / len(positions))

            if not ticker:
                continue

            metrics = self.calculate_crowding(ticker)

            total_crowding += metrics.total_crowding_score * weight
            total_weight += weight

            if metrics.crowding_level in [CrowdingLevel.HIGH, CrowdingLevel.EXTREME]:
                crowded_positions.append({
                    'ticker': ticker,
                    'weight': weight,
                    'crowding_score': metrics.total_crowding_score,
                    'level': metrics.crowding_level.value,
                    'warnings': metrics.warnings,
                })

        portfolio_score = total_crowding / total_weight if total_weight > 0 else 50

        return {
            'portfolio_crowding_score': round(portfolio_score, 1),
            'crowded_positions': crowded_positions,
            'crowded_weight': sum(p['weight'] for p in crowded_positions),
            'position_count': len(positions),
            'crowded_count': len(crowded_positions),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_calculator = None


def get_crowding_calculator() -> CrowdingScoreCalculator:
    """Get singleton calculator instance."""
    global _calculator
    if _calculator is None:
        _calculator = CrowdingScoreCalculator()
    return _calculator


def get_crowding_score(ticker: str) -> CrowdingMetrics:
    """
    Get crowding score for a single ticker.

    Usage:
        metrics = get_crowding_score('NVDA')
        print(f"Crowding: {metrics.total_crowding_score} ({metrics.crowding_level.value})")
    """
    calc = get_crowding_calculator()
    return calc.calculate_crowding(ticker)


def get_portfolio_crowding(positions: List[Dict]) -> Dict[str, Any]:
    """
    Get crowding analysis for a portfolio.

    Usage:
        positions = [{'ticker': 'AAPL', 'weight': 0.10}, ...]
        result = get_portfolio_crowding(positions)
        print(f"Portfolio crowding: {result['portfolio_crowding_score']}")
    """
    calc = get_crowding_calculator()
    return calc.calculate_portfolio_crowding(positions)


def get_crowding_table(tickers: List[str]) -> pd.DataFrame:
    """
    Get crowding scores as a DataFrame.

    Usage:
        df = get_crowding_table(['AAPL', 'NVDA', 'TSLA'])
        print(df)
    """
    calc = get_crowding_calculator()

    data = []
    for ticker in tickers:
        metrics = calc.calculate_crowding(ticker)
        data.append({
            'Ticker': ticker,
            'Crowding Score': round(metrics.total_crowding_score, 1),
            'Level': metrics.crowding_level.value,
            'Inst. Own %': round(metrics.institutional_ownership_pct, 1),
            'Top 10 %': round(metrics.top_10_holders_pct, 1),
            'Short %': round(metrics.short_interest_pct, 1),
            'Analysts': metrics.analyst_count,
            'Squeeze Risk': '⚠️' if metrics.short_squeeze_risk else '',
        })

    return pd.DataFrame(data)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Crowding Score...")

    test_tickers = ['AAPL', 'NVDA', 'TSLA', 'GME', 'AMC']

    for ticker in test_tickers:
        metrics = get_crowding_score(ticker)
        print(f"\n{ticker}:")
        print(f"  Crowding Score: {metrics.total_crowding_score:.1f} ({metrics.crowding_level.value})")
        print(f"  Inst. Ownership: {metrics.institutional_ownership_pct:.1f}%")
        print(f"  Short Interest: {metrics.short_interest_pct:.1f}%")
        if metrics.warnings:
            print(f"  Warnings: {', '.join(metrics.warnings)}")