"""
Portfolio Optimization Module

Implements institutional-grade portfolio optimization:
- Mean-Variance Optimization (Markowitz)
- Risk Parity
- Maximum Sharpe Ratio
- Minimum Volatility
- Black-Litterman (with views)

All calculations use REAL market data from yfinance.
NO hardcoded or fake data.

Author: Alpha Research Platform
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import yfinance as yf

from src.utils.logging import get_logger

logger = get_logger(__name__)


class OptimizationType(Enum):
    """Portfolio optimization strategies."""
    MAX_SHARPE = "MAX_SHARPE"  # Maximum Sharpe Ratio
    MIN_VOLATILITY = "MIN_VOLATILITY"  # Minimum Volatility
    RISK_PARITY = "RISK_PARITY"  # Equal Risk Contribution
    MAX_RETURN = "MAX_RETURN"  # Maximum Return (for given risk)
    EFFICIENT_RETURN = "EFFICIENT_RETURN"  # Target return with min risk
    BLACK_LITTERMAN = "BLACK_LITTERMAN"  # With investor views


@dataclass
class PortfolioStats:
    """Statistics for a portfolio."""
    expected_return: float  # Annualized
    volatility: float  # Annualized
    sharpe_ratio: float
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0  # 95% Value at Risk
    cvar_95: float = 0.0  # Conditional VaR


@dataclass
class OptimizationResult:
    """Result from portfolio optimization."""
    success: bool
    optimization_type: OptimizationType

    # Optimal weights
    weights: Dict[str, float]

    # Portfolio statistics
    expected_return: float
    volatility: float
    sharpe_ratio: float

    # Risk metrics
    var_95: float = 0.0
    cvar_95: float = 0.0
    max_drawdown: float = 0.0

    # Comparison to current
    current_weights: Dict[str, float] = field(default_factory=dict)
    current_return: float = 0.0
    current_volatility: float = 0.0
    current_sharpe: float = 0.0

    # Trade suggestions
    trades_needed: Dict[str, float] = field(default_factory=dict)  # ticker -> weight change

    # Efficient frontier
    frontier_returns: List[float] = field(default_factory=list)
    frontier_volatilities: List[float] = field(default_factory=list)

    # Metadata
    risk_free_rate: float = 0.05
    lookback_days: int = 252
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'optimization_type': self.optimization_type.value,
            'weights': self.weights,
            'expected_return': self.expected_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'var_95': self.var_95,
            'trades_needed': self.trades_needed,
        }


class PortfolioOptimizer:
    """
    Portfolio optimization engine using modern portfolio theory.

    All data is fetched from yfinance - no hardcoded or fake data.
    """

    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize optimizer.

        Args:
            risk_free_rate: Annual risk-free rate (default 5% for current T-bill rates)
        """
        self.risk_free_rate = risk_free_rate
        self._returns_cache = {}
        self._cache_date = None

    def optimize(self,
                 tickers: List[str],
                 optimization_type: OptimizationType = OptimizationType.MAX_SHARPE,
                 current_weights: Optional[Dict[str, float]] = None,
                 constraints: Optional[Dict] = None,
                 target_return: Optional[float] = None,
                 views: Optional[Dict[str, float]] = None,
                 lookback_days: int = 252) -> OptimizationResult:
        """
        Optimize portfolio weights.

        Args:
            tickers: List of stock tickers
            optimization_type: Which optimization to run
            current_weights: Current portfolio weights (for comparison)
            constraints: Optional constraints dict with:
                - max_weight: Maximum weight per stock (default 0.25)
                - min_weight: Minimum weight per stock (default 0.0)
                - sector_limits: Dict of sector -> max weight
            target_return: For EFFICIENT_RETURN, the target annual return
            views: For BLACK_LITTERMAN, dict of ticker -> expected return view
            lookback_days: Days of historical data to use

        Returns:
            OptimizationResult with optimal weights and statistics
        """
        # Default constraints
        if constraints is None:
            constraints = {}
        max_weight = constraints.get('max_weight', 0.25)
        min_weight = constraints.get('min_weight', 0.0)

        # Get returns data
        returns_df = self._get_returns(tickers, lookback_days)

        if returns_df is None or returns_df.empty:
            return OptimizationResult(
                success=False,
                optimization_type=optimization_type,
                weights={},
                expected_return=0,
                volatility=0,
                sharpe_ratio=0,
            )

        # Filter to tickers we have data for
        valid_tickers = [t for t in tickers if t in returns_df.columns]
        if len(valid_tickers) < 2:
            logger.warning("Need at least 2 tickers with valid data")
            return OptimizationResult(
                success=False,
                optimization_type=optimization_type,
                weights={},
                expected_return=0,
                volatility=0,
                sharpe_ratio=0,
            )

        returns_df = returns_df[valid_tickers]

        # Calculate expected returns and covariance
        mean_returns = returns_df.mean() * 252  # Annualize
        cov_matrix = returns_df.cov() * 252  # Annualize

        n_assets = len(valid_tickers)

        # Run optimization based on type
        if optimization_type == OptimizationType.MAX_SHARPE:
            weights = self._optimize_max_sharpe(mean_returns, cov_matrix, min_weight, max_weight)
        elif optimization_type == OptimizationType.MIN_VOLATILITY:
            weights = self._optimize_min_volatility(cov_matrix, min_weight, max_weight)
        elif optimization_type == OptimizationType.RISK_PARITY:
            weights = self._optimize_risk_parity(cov_matrix)
        elif optimization_type == OptimizationType.EFFICIENT_RETURN:
            if target_return is None:
                target_return = mean_returns.mean()  # Use average if not specified
            weights = self._optimize_efficient_return(mean_returns, cov_matrix, target_return, min_weight, max_weight)
        elif optimization_type == OptimizationType.BLACK_LITTERMAN:
            weights = self._optimize_black_litterman(mean_returns, cov_matrix, views or {}, min_weight, max_weight)
        else:
            weights = np.ones(n_assets) / n_assets  # Equal weight fallback

        # Create weights dict
        weights_dict = {valid_tickers[i]: float(weights[i]) for i in range(n_assets)}

        # Calculate portfolio statistics
        port_return = np.sum(mean_returns.values * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
        port_sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        # Calculate VaR and CVaR
        portfolio_returns = (returns_df * weights).sum(axis=1)
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)  # Annualized
        cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * np.sqrt(252)

        # Calculate max drawdown
        cum_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Calculate trades needed if current weights provided
        trades_needed = {}
        current_return = 0
        current_vol = 0
        current_sharpe = 0

        if current_weights:
            # Normalize current weights to valid tickers
            current_valid = {t: current_weights.get(t, 0) for t in valid_tickers}
            total = sum(current_valid.values())
            if total > 0:
                current_valid = {t: w / total for t, w in current_valid.items()}

            for ticker in valid_tickers:
                diff = weights_dict.get(ticker, 0) - current_valid.get(ticker, 0)
                if abs(diff) > 0.01:  # Only show meaningful changes
                    trades_needed[ticker] = diff

            # Calculate current portfolio stats
            current_w = np.array([current_valid.get(t, 0) for t in valid_tickers])
            if sum(current_w) > 0:
                current_return = np.sum(mean_returns.values * current_w)
                current_vol = np.sqrt(np.dot(current_w.T, np.dot(cov_matrix.values, current_w)))
                current_sharpe = (current_return - self.risk_free_rate) / current_vol if current_vol > 0 else 0

        # Generate efficient frontier
        frontier_returns, frontier_vols = self._calculate_efficient_frontier(mean_returns, cov_matrix, min_weight,
                                                                             max_weight)

        return OptimizationResult(
            success=True,
            optimization_type=optimization_type,
            weights=weights_dict,
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=port_sharpe,
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            current_weights=current_weights or {},
            current_return=current_return,
            current_volatility=current_vol,
            current_sharpe=current_sharpe,
            trades_needed=trades_needed,
            frontier_returns=frontier_returns,
            frontier_volatilities=frontier_vols,
            risk_free_rate=self.risk_free_rate,
            lookback_days=lookback_days,
        )

    def _get_returns(self, tickers: List[str], lookback_days: int) -> Optional[pd.DataFrame]:
        """Fetch historical returns from yfinance."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 30)  # Extra buffer

            # Fetch data
            data = yf.download(
                tickers,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True
            )

            if data.empty:
                logger.warning("No data returned from yfinance")
                return None

            # Handle single ticker case
            if len(tickers) == 1:
                prices = data['Close'].to_frame(tickers[0])
            else:
                prices = data['Close']

            # Calculate daily returns
            returns = prices.pct_change().dropna()

            # Keep only last lookback_days
            returns = returns.tail(lookback_days)

            return returns

        except Exception as e:
            logger.error(f"Error fetching returns: {e}")
            return None

    def _optimize_max_sharpe(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                             min_weight: float, max_weight: float) -> np.ndarray:
        """Maximize Sharpe Ratio."""
        n = len(mean_returns)

        def neg_sharpe(weights):
            port_return = np.sum(mean_returns.values * weights)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
            return -(port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        bounds = tuple((min_weight, max_weight) for _ in range(n))

        # Initial guess (equal weight)
        x0 = np.ones(n) / n

        result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        return result.x if result.success else x0

    def _optimize_min_volatility(self, cov_matrix: pd.DataFrame,
                                 min_weight: float, max_weight: float) -> np.ndarray:
        """Minimize portfolio volatility."""
        n = len(cov_matrix)

        def portfolio_vol(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        bounds = tuple((min_weight, max_weight) for _ in range(n))

        x0 = np.ones(n) / n

        result = minimize(portfolio_vol, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        return result.x if result.success else x0

    def _optimize_risk_parity(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Risk parity - equal risk contribution from each asset."""
        n = len(cov_matrix)

        def risk_parity_objective(weights):
            # Portfolio volatility
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))

            # Marginal risk contribution
            marginal_contrib = np.dot(cov_matrix.values, weights)

            # Risk contribution
            risk_contrib = weights * marginal_contrib / port_vol

            # Target: equal risk contribution
            target = port_vol / n

            # Minimize squared difference from target
            return np.sum((risk_contrib - target) ** 2)

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        bounds = tuple((0.01, 0.5) for _ in range(n))  # Risk parity allows larger weights

        x0 = np.ones(n) / n

        result = minimize(risk_parity_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        return result.x if result.success else x0

    def _optimize_efficient_return(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                                   target_return: float, min_weight: float, max_weight: float) -> np.ndarray:
        """Minimize risk for a target return."""
        n = len(mean_returns)

        def portfolio_vol(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(mean_returns.values * x) - target_return}
        ]
        bounds = tuple((min_weight, max_weight) for _ in range(n))

        x0 = np.ones(n) / n

        result = minimize(portfolio_vol, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            return result.x
        else:
            # If can't hit target, fall back to max sharpe
            return self._optimize_max_sharpe(mean_returns, cov_matrix, min_weight, max_weight)

    def _optimize_black_litterman(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                                  views: Dict[str, float], min_weight: float, max_weight: float) -> np.ndarray:
        """
        Black-Litterman optimization with investor views.

        Views dict maps ticker -> expected return view (e.g., {'AAPL': 0.15} means you expect 15% return)
        """
        n = len(mean_returns)
        tickers = mean_returns.index.tolist()

        # Market implied returns (using CAPM as prior)
        # Simplified: use historical mean as prior
        prior_returns = mean_returns.values

        # If we have views, adjust returns
        if views:
            tau = 0.05  # Scaling factor for uncertainty in prior

            # Build view matrix
            P = np.zeros((len(views), n))
            Q = np.zeros(len(views))

            for i, (ticker, view) in enumerate(views.items()):
                if ticker in tickers:
                    idx = tickers.index(ticker)
                    P[i, idx] = 1
                    Q[i] = view

            if np.any(P):
                # Omega: uncertainty in views (diagonal)
                omega = np.diag([0.1] * len(views))  # 10% uncertainty in views

                # Black-Litterman formula
                cov = cov_matrix.values
                tau_cov = tau * cov

                try:
                    # Posterior expected returns
                    inv_tau_cov = np.linalg.inv(tau_cov)
                    inv_omega = np.linalg.inv(omega)

                    M = np.linalg.inv(inv_tau_cov + P.T @ inv_omega @ P)
                    posterior_returns = M @ (inv_tau_cov @ prior_returns + P.T @ inv_omega @ Q)

                    # Use posterior returns for optimization
                    adjusted_returns = pd.Series(posterior_returns, index=mean_returns.index)
                    return self._optimize_max_sharpe(adjusted_returns, cov_matrix, min_weight, max_weight)
                except np.linalg.LinAlgError:
                    logger.warning("Black-Litterman matrix inversion failed, using standard optimization")

        # Fallback to max sharpe with original returns
        return self._optimize_max_sharpe(mean_returns, cov_matrix, min_weight, max_weight)

    def _calculate_efficient_frontier(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                                      min_weight: float, max_weight: float, n_points: int = 50) -> Tuple[
        List[float], List[float]]:
        """Calculate the efficient frontier."""
        returns = []
        volatilities = []

        # Get min and max possible returns
        min_vol_weights = self._optimize_min_volatility(cov_matrix, min_weight, max_weight)
        min_ret = np.sum(mean_returns.values * min_vol_weights)
        max_ret = mean_returns.max()

        target_returns = np.linspace(min_ret, max_ret * 0.95, n_points)

        for target in target_returns:
            try:
                weights = self._optimize_efficient_return(mean_returns, cov_matrix, target, min_weight, max_weight)
                port_ret = np.sum(mean_returns.values * weights)
                port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))

                returns.append(port_ret)
                volatilities.append(port_vol)
            except:
                continue

        return returns, volatilities

    def get_correlation_matrix(self, tickers: List[str], lookback_days: int = 252) -> pd.DataFrame:
        """Get correlation matrix for tickers."""
        returns_df = self._get_returns(tickers, lookback_days)
        if returns_df is not None:
            return returns_df.corr()
        return pd.DataFrame()

    def get_portfolio_stats(self, weights: Dict[str, float], lookback_days: int = 252) -> PortfolioStats:
        """Calculate statistics for a given portfolio."""
        tickers = list(weights.keys())
        returns_df = self._get_returns(tickers, lookback_days)

        if returns_df is None or returns_df.empty:
            return PortfolioStats(0, 0, 0)

        # Calculate weighted returns
        weight_array = np.array([weights.get(t, 0) for t in returns_df.columns])
        portfolio_returns = (returns_df * weight_array).sum(axis=1)

        # Statistics
        ann_return = portfolio_returns.mean() * 252
        ann_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = (ann_return - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0

        # Sortino (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else ann_vol
        sortino = (ann_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0

        # VaR and CVaR
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
        cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * np.sqrt(252)

        # Max Drawdown
        cum_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_dd = drawdowns.min()

        return PortfolioStats(
            expected_return=ann_return,
            volatility=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            var_95=var_95,
            cvar_95=cvar_95,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_optimizer = None


def get_optimizer(risk_free_rate: float = 0.05) -> PortfolioOptimizer:
    """Get singleton optimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = PortfolioOptimizer(risk_free_rate)
    return _optimizer


def optimize_portfolio(tickers: List[str],
                       optimization_type: str = "MAX_SHARPE",
                       current_weights: Optional[Dict[str, float]] = None,
                       max_weight: float = 0.25) -> OptimizationResult:
    """
    Optimize a portfolio.

    Usage:
        result = optimize_portfolio(
            tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            optimization_type='MAX_SHARPE',
            max_weight=0.30
        )
        print(f"Optimal weights: {result.weights}")
        print(f"Expected return: {result.expected_return:.1%}")
        print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
    """
    optimizer = get_optimizer()

    opt_type = OptimizationType[optimization_type.upper()]

    return optimizer.optimize(
        tickers=tickers,
        optimization_type=opt_type,
        current_weights=current_weights,
        constraints={'max_weight': max_weight},
    )


def get_risk_parity_weights(tickers: List[str]) -> Dict[str, float]:
    """
    Get risk parity weights for a set of tickers.

    Usage:
        weights = get_risk_parity_weights(['AAPL', 'MSFT', 'GLD', 'TLT'])
        print(weights)
    """
    optimizer = get_optimizer()
    result = optimizer.optimize(
        tickers=tickers,
        optimization_type=OptimizationType.RISK_PARITY,
    )
    return result.weights if result.success else {}


def get_min_volatility_weights(tickers: List[str], max_weight: float = 0.25) -> Dict[str, float]:
    """
    Get minimum volatility portfolio weights.

    Usage:
        weights = get_min_volatility_weights(['AAPL', 'MSFT', 'JNJ', 'PG', 'KO'])
        print(weights)
    """
    optimizer = get_optimizer()
    result = optimizer.optimize(
        tickers=tickers,
        optimization_type=OptimizationType.MIN_VOLATILITY,
        constraints={'max_weight': max_weight},
    )
    return result.weights if result.success else {}


def optimize_with_views(tickers: List[str],
                        views: Dict[str, float],
                        max_weight: float = 0.30) -> OptimizationResult:
    """
    Black-Litterman optimization with investor views.

    Usage:
        # You expect AAPL +20%, MSFT +15%, GOOGL +10%
        result = optimize_with_views(
            tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            views={'AAPL': 0.20, 'MSFT': 0.15, 'GOOGL': 0.10},
        )
        print(result.weights)
    """
    optimizer = get_optimizer()
    return optimizer.optimize(
        tickers=tickers,
        optimization_type=OptimizationType.BLACK_LITTERMAN,
        views=views,
        constraints={'max_weight': max_weight},
    )


def compare_portfolio_to_optimal(current_weights: Dict[str, float]) -> OptimizationResult:
    """
    Compare current portfolio to optimal allocation.

    Usage:
        current = {'AAPL': 0.25, 'MSFT': 0.25, 'GOOGL': 0.20, 'AMZN': 0.15, 'META': 0.15}
        result = compare_portfolio_to_optimal(current)
        print(f"Current Sharpe: {result.current_sharpe:.2f}")
        print(f"Optimal Sharpe: {result.sharpe_ratio:.2f}")
        print(f"Trades needed: {result.trades_needed}")
    """
    optimizer = get_optimizer()
    tickers = list(current_weights.keys())

    return optimizer.optimize(
        tickers=tickers,
        optimization_type=OptimizationType.MAX_SHARPE,
        current_weights=current_weights,
    )


def get_correlation_heatmap_data(tickers: List[str]) -> pd.DataFrame:
    """
    Get correlation matrix for visualization.

    Usage:
        corr = get_correlation_heatmap_data(['AAPL', 'MSFT', 'GOOGL', 'GLD', 'TLT'])
        # Use with seaborn or plotly for heatmap
    """
    optimizer = get_optimizer()
    return optimizer.get_correlation_matrix(tickers)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Portfolio Optimizer...")
    print("=" * 60)

    # Test tickers (diversified portfolio)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'JNJ', 'PG', 'XOM', 'GLD', 'TLT']

    # Test Max Sharpe
    print("\n1. MAX SHARPE OPTIMIZATION")
    result = optimize_portfolio(tickers, 'MAX_SHARPE', max_weight=0.25)
    print(f"   Success: {result.success}")
    print(f"   Expected Return: {result.expected_return:.1%}")
    print(f"   Volatility: {result.volatility:.1%}")
    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"   Weights: {result.weights}")

    # Test Min Volatility
    print("\n2. MIN VOLATILITY OPTIMIZATION")
    result = optimize_portfolio(tickers, 'MIN_VOLATILITY', max_weight=0.25)
    print(f"   Expected Return: {result.expected_return:.1%}")
    print(f"   Volatility: {result.volatility:.1%}")
    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")

    # Test Risk Parity
    print("\n3. RISK PARITY")
    weights = get_risk_parity_weights(tickers)
    print(f"   Weights: {weights}")

    # Test comparison
    print("\n4. COMPARE TO EQUAL WEIGHT")
    equal_weights = {t: 1 / len(tickers) for t in tickers}
    result = compare_portfolio_to_optimal(equal_weights)
    print(f"   Current Sharpe: {result.current_sharpe:.2f}")
    print(f"   Optimal Sharpe: {result.sharpe_ratio:.2f}")
    print(f"   Improvement: {(result.sharpe_ratio - result.current_sharpe):.2f}")

    # Test Black-Litterman with views
    print("\n5. BLACK-LITTERMAN WITH VIEWS")
    views = {'AAPL': 0.20, 'MSFT': 0.15}  # Bullish on tech
    result = optimize_with_views(tickers, views)
    print(f"   Weights: {result.weights}")
    print(f"   Sharpe: {result.sharpe_ratio:.2f}")

    print("\n" + "=" * 60)
    print("All tests complete!")