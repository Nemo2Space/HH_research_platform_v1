"""Alpha Platform - Backtest Module"""

from .engine import BacktestEngine
from .strategies import STRATEGIES, get_strategy, HOLDING_PERIODS, BENCHMARKS
from .metrics import calculate_all_metrics, format_metrics_report

__all__ = [
    'BacktestEngine',
    'STRATEGIES',
    'get_strategy',
    'HOLDING_PERIODS',
    'BENCHMARKS',
    'calculate_all_metrics',
    'format_metrics_report',
]