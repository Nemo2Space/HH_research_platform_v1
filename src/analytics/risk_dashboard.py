

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import yfinance as yf

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a portfolio."""
    # Concentration
    top_10_concentration: float  # % of portfolio in top 10 holdings
    hhi_index: float  # Herfindahl-Hirschman Index (0-10000)
    largest_position: str
    largest_position_pct: float

    # Sector exposure
    sector_weights: Dict[str, float] = field(default_factory=dict)
    largest_sector: str = ""
    largest_sector_pct: float = 0.0

    # Correlation
    avg_correlation: float = 0.0
    max_correlation_pair: Tuple[str, str] = ("", "")
    max_correlation: float = 0.0

    # Value at Risk
    var_95_daily: float = 0.0  # 95% VaR as % of portfolio
    var_99_daily: float = 0.0  # 99% VaR
    var_95_dollar: float = 0.0  # 95% VaR in dollars

    # Volatility
    portfolio_volatility: float = 0.0  # Annualized
    portfolio_beta: float = 0.0  # vs SPY

    # Drawdown
    max_drawdown_30d: float = 0.0
    current_drawdown: float = 0.0

    # Diversification
    effective_n_stocks: float = 0.0  # 1/HHI normalized
    diversification_score: float = 0.0  # 0-100


class RiskDashboard:
    """
    Portfolio risk analysis dashboard.

    Provides:
    - Concentration analysis
    - Correlation matrix
    - VaR calculations
    - Sector exposure
    - Beta and volatility
    """

    def __init__(self, positions: List[Dict], total_value: float = None):
        """
        Initialize with portfolio positions.

        Args:
            positions: List of position dicts with 'symbol', 'marketValue', 'weight'
            total_value: Total portfolio value (calculated if not provided)
        """
        self.positions = positions

        # Calculate total value if not provided
        if total_value:
            self.total_value = total_value
        else:
            self.total_value = sum(p.get('marketValue', p.get('market_value', 0)) for p in positions)

        # Extract symbols and weights
        self.symbols = []
        self.weights = []
        self.values = []

        for p in positions:
            symbol = p.get('symbol', '')
            if not symbol or symbol in ['USD', 'CASH']:
                continue

            value = p.get('marketValue', p.get('market_value', 0))
            if value <= 0:
                continue

            self.symbols.append(symbol)
            self.values.append(value)

            weight = p.get('weight', 0)
            if weight == 0 and self.total_value > 0:
                weight = (value / self.total_value) * 100
            self.weights.append(weight)

        # Convert to numpy arrays
        self.weights = np.array(self.weights)
        self.values = np.array(self.values)

        # Price data cache
        self._price_data = None
        self._returns_data = None

    def _fetch_price_data(self, days: int = 252) -> pd.DataFrame:
        """Fetch historical price data for all positions."""
        if self._price_data is not None:
            return self._price_data

        if not self.symbols:
            return pd.DataFrame()

        logger.info(f"Fetching price data for {len(self.symbols)} symbols...")

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 30)  # Extra buffer

            # Add SPY for beta calculation
            symbols_with_spy = self.symbols + ['SPY']

            # yfinance >= 0.2.50: auto_adjust=True is default, so 'Close' is already adjusted
            # No more 'Adj Close' column - use 'Close' directly
            data = yf.download(
                symbols_with_spy,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True  # Explicit for clarity
            )['Close']

            if isinstance(data, pd.Series):
                data = data.to_frame(name=symbols_with_spy[0])

            self._price_data = data
            self._returns_data = data.pct_change().dropna()

            return self._price_data

        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            return pd.DataFrame()

    def calculate_concentration(self) -> Dict[str, any]:
        """Calculate concentration metrics."""
        if len(self.weights) == 0:
            return {}

        # Sort by weight
        sorted_indices = np.argsort(self.weights)[::-1]
        sorted_weights = self.weights[sorted_indices]
        sorted_symbols = [self.symbols[i] for i in sorted_indices]

        # Top 10 concentration
        top_10_pct = sorted_weights[:10].sum()

        # HHI Index (sum of squared weights)
        weight_fractions = self.weights / 100  # Convert to fractions
        hhi = (weight_fractions ** 2).sum() * 10000  # Scale to 0-10000

        # Effective number of stocks (inverse HHI)
        effective_n = 1 / (weight_fractions ** 2).sum() if (weight_fractions ** 2).sum() > 0 else 0

        # Diversification score (0-100, higher is better)
        # Based on effective N vs actual N
        actual_n = len(self.symbols)
        div_score = min(100, (effective_n / actual_n) * 100) if actual_n > 0 else 0

        return {
            'top_10_concentration': top_10_pct,
            'top_10_holdings': list(zip(sorted_symbols[:10], sorted_weights[:10])),
            'hhi_index': hhi,
            'effective_n_stocks': effective_n,
            'diversification_score': div_score,
            'largest_position': sorted_symbols[0] if sorted_symbols else "",
            'largest_position_pct': sorted_weights[0] if len(sorted_weights) > 0 else 0
        }

    def calculate_correlation_matrix(self) -> Tuple[pd.DataFrame, Dict]:
        """Calculate correlation matrix and metrics."""
        self._fetch_price_data()

        if self._returns_data is None or self._returns_data.empty:
            return pd.DataFrame(), {}

        # Get returns for portfolio symbols only (not SPY)
        portfolio_returns = self._returns_data[[s for s in self.symbols if s in self._returns_data.columns]]

        if portfolio_returns.empty or len(portfolio_returns.columns) < 2:
            return pd.DataFrame(), {}

        # Calculate correlation matrix
        corr_matrix = portfolio_returns.corr()

        # Find average correlation (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_corr = corr_matrix.where(mask)
        avg_corr = upper_corr.stack().mean()

        # Find max correlation pair
        max_corr = 0
        max_pair = ("", "")
        for i, sym1 in enumerate(corr_matrix.columns):
            for j, sym2 in enumerate(corr_matrix.columns):
                if i < j:
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > abs(max_corr):
                        max_corr = corr_val
                        max_pair = (sym1, sym2)

        metrics = {
            'avg_correlation': avg_corr,
            'max_correlation': max_corr,
            'max_correlation_pair': max_pair
        }

        return corr_matrix, metrics

    def calculate_var(self, confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, float]:
        """
        Calculate Value at Risk using historical simulation.

        Args:
            confidence_levels: List of confidence levels (e.g., [0.95, 0.99])

        Returns:
            Dict with VaR values
        """
        self._fetch_price_data()

        if self._returns_data is None or self._returns_data.empty:
            return {}

        # Get portfolio returns (weighted average of position returns)
        portfolio_returns = []

        for date in self._returns_data.index:
            daily_return = 0
            total_weight = 0

            for i, symbol in enumerate(self.symbols):
                if symbol in self._returns_data.columns:
                    ret = self._returns_data.loc[date, symbol]
                    if not pd.isna(ret):
                        weight = self.weights[i] / 100
                        daily_return += ret * weight
                        total_weight += weight

            if total_weight > 0:
                portfolio_returns.append(daily_return / total_weight * total_weight)  # Normalize

        if not portfolio_returns:
            return {}

        portfolio_returns = np.array(portfolio_returns)

        results = {}
        for conf in confidence_levels:
            var_pct = np.percentile(portfolio_returns, (1 - conf) * 100)
            results[f'var_{int(conf * 100)}_daily'] = abs(var_pct) * 100  # As percentage
            results[f'var_{int(conf * 100)}_dollar'] = abs(var_pct) * self.total_value

        # Portfolio volatility (annualized)
        results['portfolio_volatility'] = np.std(portfolio_returns) * np.sqrt(252) * 100

        return results

    def calculate_beta(self) -> float:
        """Calculate portfolio beta vs SPY."""
        self._fetch_price_data()

        if self._returns_data is None or 'SPY' not in self._returns_data.columns:
            return 1.0

        spy_returns = self._returns_data['SPY'].dropna()

        # Calculate weighted portfolio returns
        portfolio_returns = []

        for date in spy_returns.index:
            if date not in self._returns_data.index:
                continue

            daily_return = 0
            total_weight = 0

            for i, symbol in enumerate(self.symbols):
                if symbol in self._returns_data.columns:
                    ret = self._returns_data.loc[date, symbol]
                    if not pd.isna(ret):
                        weight = self.weights[i] / 100
                        daily_return += ret * weight
                        total_weight += weight

            if total_weight > 0:
                portfolio_returns.append((date, daily_return))

        if len(portfolio_returns) < 30:
            return 1.0

        port_df = pd.DataFrame(portfolio_returns, columns=['date', 'return']).set_index('date')
        aligned = pd.concat([port_df['return'], spy_returns], axis=1, join='inner')
        aligned.columns = ['portfolio', 'spy']

        # Calculate beta using covariance/variance
        cov = aligned['portfolio'].cov(aligned['spy'])
        var = aligned['spy'].var()

        beta = cov / var if var > 0 else 1.0

        return beta

    def calculate_drawdown(self, days: int = 30) -> Dict[str, float]:
        """Calculate drawdown metrics."""
        self._fetch_price_data()

        if self._price_data is None or self._price_data.empty:
            return {}

        # Calculate portfolio value over time
        portfolio_values = []

        for date in self._price_data.index[-days:]:
            daily_value = 0

            for i, symbol in enumerate(self.symbols):
                if symbol in self._price_data.columns:
                    price = self._price_data.loc[date, symbol]
                    if not pd.isna(price):
                        # Use current weight to estimate historical value
                        daily_value += (self.weights[i] / 100) * self.total_value

            if daily_value > 0:
                portfolio_values.append(daily_value)

        if len(portfolio_values) < 2:
            return {}

        portfolio_values = np.array(portfolio_values)

        # Calculate running maximum
        running_max = np.maximum.accumulate(portfolio_values)

        # Drawdown series
        drawdowns = (portfolio_values - running_max) / running_max * 100

        return {
            'max_drawdown_30d': abs(drawdowns.min()),
            'current_drawdown': abs(drawdowns[-1])
        }

    def get_sector_exposure(self) -> Dict[str, float]:
        """Get sector exposure from positions."""
        sector_weights = {}

        for i, p in enumerate(self.positions):
            sector = p.get('sector', 'Unknown')
            if not sector:
                sector = 'Unknown'

            weight = self.weights[i] if i < len(self.weights) else 0
            sector_weights[sector] = sector_weights.get(sector, 0) + weight

        return dict(sorted(sector_weights.items(), key=lambda x: x[1], reverse=True))

    def get_full_risk_metrics(self) -> RiskMetrics:
        """Calculate all risk metrics."""

        # Concentration
        concentration = self.calculate_concentration()

        # Correlation
        _, corr_metrics = self.calculate_correlation_matrix()

        # VaR
        var_metrics = self.calculate_var()

        # Beta
        beta = self.calculate_beta()

        # Drawdown
        drawdown = self.calculate_drawdown()

        # Sector exposure
        sector_weights = self.get_sector_exposure()
        largest_sector = list(sector_weights.keys())[0] if sector_weights else ""
        largest_sector_pct = list(sector_weights.values())[0] if sector_weights else 0

        return RiskMetrics(
            # Concentration
            top_10_concentration=concentration.get('top_10_concentration', 0),
            hhi_index=concentration.get('hhi_index', 0),
            largest_position=concentration.get('largest_position', ''),
            largest_position_pct=concentration.get('largest_position_pct', 0),

            # Sector
            sector_weights=sector_weights,
            largest_sector=largest_sector,
            largest_sector_pct=largest_sector_pct,

            # Correlation
            avg_correlation=corr_metrics.get('avg_correlation', 0),
            max_correlation_pair=corr_metrics.get('max_correlation_pair', ('', '')),
            max_correlation=corr_metrics.get('max_correlation', 0),

            # VaR
            var_95_daily=var_metrics.get('var_95_daily', 0),
            var_99_daily=var_metrics.get('var_99_daily', 0),
            var_95_dollar=var_metrics.get('var_95_dollar', 0),

            # Volatility
            portfolio_volatility=var_metrics.get('portfolio_volatility', 0),
            portfolio_beta=beta,

            # Drawdown
            max_drawdown_30d=drawdown.get('max_drawdown_30d', 0),
            current_drawdown=drawdown.get('current_drawdown', 0),

            # Diversification
            effective_n_stocks=concentration.get('effective_n_stocks', 0),
            diversification_score=concentration.get('diversification_score', 0)
        )

    def get_risk_summary_text(self) -> str:
        """Get a text summary of risk metrics."""
        metrics = self.get_full_risk_metrics()

        lines = [
            "=" * 50,
            "PORTFOLIO RISK SUMMARY",
            "=" * 50,
            "",
            f"📊 CONCENTRATION:",
            f"   Top 10 holdings: {metrics.top_10_concentration:.1f}% of portfolio",
            f"   Largest position: {metrics.largest_position} ({metrics.largest_position_pct:.1f}%)",
            f"   Effective # stocks: {metrics.effective_n_stocks:.1f}",
            f"   Diversification score: {metrics.diversification_score:.0f}/100",
            "",
            f"📈 VOLATILITY & RISK:",
            f"   Portfolio volatility: {metrics.portfolio_volatility:.1f}% (annualized)",
            f"   Portfolio beta: {metrics.portfolio_beta:.2f}",
            f"   VaR 95% (daily): {metrics.var_95_daily:.2f}% (${metrics.var_95_dollar:,.0f})",
            f"   VaR 99% (daily): {metrics.var_99_daily:.2f}%",
            "",
            f"📉 DRAWDOWN:",
            f"   Max drawdown (30d): {metrics.max_drawdown_30d:.2f}%",
            f"   Current drawdown: {metrics.current_drawdown:.2f}%",
            "",
            f"🔗 CORRELATION:",
            f"   Average correlation: {metrics.avg_correlation:.2f}",
            f"   Most correlated: {metrics.max_correlation_pair[0]}/{metrics.max_correlation_pair[1]} ({metrics.max_correlation:.2f})",
            "",
            f"🏭 SECTOR EXPOSURE:",
        ]

        for sector, weight in list(metrics.sector_weights.items())[:5]:
            lines.append(f"   {sector}: {weight:.1f}%")

        return "\n".join(lines)


def get_risk_dashboard(positions: List[Dict], total_value: float = None) -> RiskDashboard:
    """Create risk dashboard instance."""
    return RiskDashboard(positions, total_value)


if __name__ == "__main__":
    # Test with mock data
    mock_positions = [
        {"symbol": "AAPL", "marketValue": 50000, "sector": "Technology"},
        {"symbol": "MSFT", "marketValue": 45000, "sector": "Technology"},
        {"symbol": "GOOGL", "marketValue": 40000, "sector": "Technology"},
        {"symbol": "AMZN", "marketValue": 35000, "sector": "Consumer"},
        {"symbol": "JPM", "marketValue": 30000, "sector": "Financial"},
        {"symbol": "JNJ", "marketValue": 25000, "sector": "Healthcare"},
        {"symbol": "XOM", "marketValue": 20000, "sector": "Energy"},
        {"symbol": "PG", "marketValue": 15000, "sector": "Consumer"},
    ]

    dashboard = RiskDashboard(mock_positions)
    print(dashboard.get_risk_summary_text())