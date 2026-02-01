"""
Alpha Platform - Backtest Metrics

Additional performance metrics and analysis functions.

Usage:
    from src.backtest.metrics import calculate_all_metrics, format_metrics_report
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class MetricsReport:
    """Comprehensive metrics report."""
    # Core metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float

    # Risk metrics
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: int  # Days
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR

    # Trade metrics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    expectancy: float

    # Streak metrics
    max_consecutive_wins: int
    max_consecutive_losses: int

    # Time metrics
    avg_holding_period: float
    best_month: float
    worst_month: float


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.05,
                           periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Array of period returns (as percentages)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Sharpe ratio
    """
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0

    # Convert to decimal returns
    decimal_returns = returns / 100

    # Annualize
    mean_return = np.mean(decimal_returns) * periods_per_year
    std_return = np.std(decimal_returns) * np.sqrt(periods_per_year)

    if std_return == 0:
        return 0.0

    return (mean_return - risk_free_rate) / std_return


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.05,
                            periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio (uses downside deviation).

    Args:
        returns: Array of period returns (as percentages)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    decimal_returns = returns / 100
    downside_returns = decimal_returns[decimal_returns < 0]

    if len(downside_returns) == 0:
        return 10.0  # No downside

    mean_return = np.mean(decimal_returns) * periods_per_year
    downside_std = np.std(downside_returns) * np.sqrt(periods_per_year)

    if downside_std == 0:
        return 0.0

    return (mean_return - risk_free_rate) / downside_std


def calculate_max_drawdown(returns: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown and its duration.

    Args:
        returns: Array of period returns (as percentages)

    Returns:
        Tuple of (max_drawdown_pct, peak_idx, trough_idx)
    """
    if len(returns) == 0:
        return 0.0, 0, 0

    # Build equity curve
    equity = np.cumprod(1 + returns / 100)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak

    max_dd = np.min(drawdown) * 100
    trough_idx = np.argmin(drawdown)
    peak_idx = np.argmax(equity[:trough_idx + 1]) if trough_idx > 0 else 0

    return max_dd, peak_idx, trough_idx


def calculate_calmar_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).
    """
    if len(returns) < 2:
        return 0.0

    annual_return = np.mean(returns / 100) * periods_per_year * 100
    max_dd, _, _ = calculate_max_drawdown(returns)

    if max_dd == 0:
        return 10.0

    return annual_return / abs(max_dd)


def calculate_profit_factor(returns: np.ndarray) -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    """
    profits = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    if losses == 0:
        return 10.0 if profits > 0 else 0.0

    return profits / losses


def calculate_expectancy(returns: np.ndarray) -> float:
    """
    Calculate expectancy per trade.

    Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
    """
    if len(returns) == 0:
        return 0.0

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_rate = len(wins) / len(returns)
    loss_rate = len(losses) / len(returns)

    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0

    return (win_rate * avg_win) - (loss_rate * avg_loss)


def calculate_consecutive_streaks(returns: np.ndarray) -> Tuple[int, int]:
    """
    Calculate max consecutive wins and losses.

    Returns:
        Tuple of (max_wins, max_losses)
    """
    if len(returns) == 0:
        return 0, 0

    wins = returns > 0

    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0

    for is_win in wins:
        if is_win:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)

    return max_wins, max_losses


def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk at specified confidence level.

    Returns the worst expected return at the given confidence level.
    """
    if len(returns) == 0:
        return 0.0

    return float(np.percentile(returns, (1 - confidence) * 100))


def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate Conditional VaR (Expected Shortfall).

    Average of returns below VaR threshold.
    """
    if len(returns) == 0:
        return 0.0

    var = calculate_var(returns, confidence)
    below_var = returns[returns <= var]

    if len(below_var) == 0:
        return var

    return float(np.mean(below_var))


def calculate_information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray,
                                periods_per_year: int = 252) -> float:
    """
    Calculate Information Ratio (active return / tracking error).
    """
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        return 0.0

    active_returns = returns - benchmark_returns

    if np.std(active_returns) == 0:
        return 0.0

    return (np.mean(active_returns) * periods_per_year) / (np.std(active_returns) * np.sqrt(periods_per_year))


def calculate_all_metrics(returns: np.ndarray, benchmark_returns: np.ndarray = None,
                          holding_period: int = 10) -> Dict[str, float]:
    """
    Calculate all performance metrics.

    Args:
        returns: Array of trade returns (as percentages)
        benchmark_returns: Optional benchmark returns for comparison
        holding_period: Average holding period in days

    Returns:
        Dict with all metrics
    """
    periods_per_year = 252 / holding_period

    if len(returns) == 0:
        return {
            'total_return': 0,
            'avg_return': 0,
            'win_rate': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'profit_factor': 0,
            'expectancy': 0,
        }

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    max_dd, _, _ = calculate_max_drawdown(returns)
    max_wins, max_losses = calculate_consecutive_streaks(returns)

    metrics = {
        # Returns
        'total_return': float(np.sum(returns)),
        'avg_return': float(np.mean(returns)),
        'median_return': float(np.median(returns)),
        'std_return': float(np.std(returns)),
        'annualized_return': float(np.mean(returns) * periods_per_year),

        # Win/Loss
        'win_rate': len(wins) / len(returns) if len(returns) > 0 else 0,
        'total_trades': len(returns),
        'winning_trades': len(wins),
        'losing_trades': len(losses),
        'avg_win': float(np.mean(wins)) if len(wins) > 0 else 0,
        'avg_loss': float(np.mean(losses)) if len(losses) > 0 else 0,
        'best_trade': float(np.max(returns)),
        'worst_trade': float(np.min(returns)),

        # Risk-adjusted
        'sharpe_ratio': calculate_sharpe_ratio(returns, periods_per_year=int(periods_per_year)),
        'sortino_ratio': calculate_sortino_ratio(returns, periods_per_year=int(periods_per_year)),
        'calmar_ratio': calculate_calmar_ratio(returns, int(periods_per_year)),

        # Risk
        'max_drawdown': max_dd,
        'var_95': calculate_var(returns, 0.95),
        'cvar_95': calculate_cvar(returns, 0.95),
        'volatility': float(np.std(returns) * np.sqrt(periods_per_year)),

        # Trade quality
        'profit_factor': calculate_profit_factor(returns),
        'expectancy': calculate_expectancy(returns),
        'win_loss_ratio': abs(np.mean(wins) / np.mean(losses)) if len(losses) > 0 and np.mean(losses) != 0 else 10,

        # Streaks
        'max_consecutive_wins': max_wins,
        'max_consecutive_losses': max_losses,
    }

    # Add benchmark comparison if available
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        bench_return = float(np.sum(benchmark_returns))
        metrics['benchmark_return'] = bench_return
        metrics['alpha'] = metrics['total_return'] - bench_return
        metrics['information_ratio'] = calculate_information_ratio(returns, benchmark_returns, int(periods_per_year))

    return metrics


def format_metrics_report(metrics: Dict[str, float]) -> str:
    """Format metrics as readable report."""

    report = []
    report.append("=" * 50)
    report.append("BACKTEST PERFORMANCE REPORT")
    report.append("=" * 50)

    report.append("\n📊 RETURNS")
    report.append(f"  Total Return:       {metrics.get('total_return', 0):+.2f}%")
    report.append(f"  Average Return:     {metrics.get('avg_return', 0):+.2f}%")
    report.append(f"  Annualized Return:  {metrics.get('annualized_return', 0):+.2f}%")
    report.append(f"  Best Trade:         {metrics.get('best_trade', 0):+.2f}%")
    report.append(f"  Worst Trade:        {metrics.get('worst_trade', 0):+.2f}%")

    report.append("\n📈 TRADE STATISTICS")
    report.append(f"  Total Trades:       {metrics.get('total_trades', 0)}")
    report.append(f"  Win Rate:           {metrics.get('win_rate', 0):.1%}")
    report.append(f"  Avg Win:            {metrics.get('avg_win', 0):+.2f}%")
    report.append(f"  Avg Loss:           {metrics.get('avg_loss', 0):+.2f}%")
    report.append(f"  Win/Loss Ratio:     {metrics.get('win_loss_ratio', 0):.2f}")

    report.append("\n📉 RISK METRICS")
    report.append(f"  Sharpe Ratio:       {metrics.get('sharpe_ratio', 0):.2f}")
    report.append(f"  Sortino Ratio:      {metrics.get('sortino_ratio', 0):.2f}")
    report.append(f"  Max Drawdown:       {metrics.get('max_drawdown', 0):.2f}%")
    report.append(f"  Volatility:         {metrics.get('volatility', 0):.2f}%")
    report.append(f"  VaR (95%):          {metrics.get('var_95', 0):.2f}%")

    report.append("\n🎯 QUALITY METRICS")
    report.append(f"  Profit Factor:      {metrics.get('profit_factor', 0):.2f}")
    report.append(f"  Expectancy:         {metrics.get('expectancy', 0):+.2f}%")
    report.append(f"  Max Win Streak:     {metrics.get('max_consecutive_wins', 0)}")
    report.append(f"  Max Loss Streak:    {metrics.get('max_consecutive_losses', 0)}")

    if 'benchmark_return' in metrics:
        report.append("\n🏆 VS BENCHMARK")
        report.append(f"  Benchmark Return:   {metrics.get('benchmark_return', 0):+.2f}%")
        report.append(f"  Alpha:              {metrics.get('alpha', 0):+.2f}%")
        report.append(f"  Information Ratio:  {metrics.get('information_ratio', 0):.2f}")

    report.append("\n" + "=" * 50)

    return "\n".join(report)


def compare_strategies(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare multiple backtest results side by side.

    Args:
        results: List of dicts with strategy results

    Returns:
        DataFrame with comparison
    """
    rows = []

    for r in results:
        rows.append({
            'Strategy': r.get('strategy_name', 'Unknown'),
            'Trades': r.get('total_trades', 0),
            'Win Rate': f"{r.get('win_rate', 0):.1%}",
            'Avg Return': f"{r.get('avg_return', 0):+.2f}%",
            'Total Return': f"{r.get('total_return', 0):+.2f}%",
            'Sharpe': f"{r.get('sharpe_ratio', 0):.2f}",
            'Max DD': f"{r.get('max_drawdown', 0):.2f}%",
            'Alpha': f"{r.get('alpha', 0):+.2f}%",
        })

    df = pd.DataFrame(rows)

    return df


def rank_strategies(results: List[Dict[str, Any]],
                    weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
    """
    Rank strategies by weighted score.

    Args:
        results: List of strategy results
        weights: Metric weights (default: sharpe=0.3, win_rate=0.2, alpha=0.3, max_dd=0.2)

    Returns:
        Sorted list with rankings
    """
    if not weights:
        weights = {
            'sharpe_ratio': 0.30,
            'win_rate': 0.20,
            'alpha': 0.30,
            'max_drawdown': 0.20,  # Lower is better - will invert
        }

    for r in results:
        score = 0

        # Normalize and weight each metric
        sharpe = r.get('sharpe_ratio', 0)
        score += min(sharpe / 2, 1) * weights.get('sharpe_ratio', 0) * 100

        win_rate = r.get('win_rate', 0)
        score += win_rate * weights.get('win_rate', 0) * 100

        alpha = r.get('alpha', 0)
        score += min(alpha / 20, 1) * weights.get('alpha', 0) * 100

        max_dd = abs(r.get('max_drawdown', 0))
        dd_score = max(0, 1 - max_dd / 30)  # Penalize drawdowns > 30%
        score += dd_score * weights.get('max_drawdown', 0) * 100

        r['composite_score'] = round(score, 2)

    # Sort by composite score
    ranked = sorted(results, key=lambda x: x['composite_score'], reverse=True)

    # Add rank
    for i, r in enumerate(ranked, 1):
        r['rank'] = i

    return ranked