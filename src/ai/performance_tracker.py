"""
Alpha Platform - Performance Tracker

Captures daily portfolio snapshots and calculates performance metrics.

Location: src/ai/performance_tracker.py
"""

import yfinance as yf
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd

from src.db.repository import Repository
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PerformanceTracker:
    """
    Tracks portfolio performance over time.

    Usage:
        tracker = PerformanceTracker()
        tracker.capture_snapshot(ibkr_positions, ibkr_summary)
    """

    def __init__(self, benchmark: str = "SPY"):
        self.repo = Repository()
        self.benchmark = benchmark

    def get_benchmark_price(self, ticker: str = None, date_val: date = None) -> Optional[float]:
        """Get benchmark price for a specific date."""
        ticker = ticker or self.benchmark
        date_val = date_val or date.today()

        try:
            bench = yf.Ticker(ticker)
            # Get last 5 days to ensure we get data
            hist = bench.history(start=date_val - timedelta(days=5), end=date_val + timedelta(days=1))
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            logger.warning(f"Could not get benchmark price: {e}")
        return None

    def capture_snapshot(self,
                         positions: List[Dict[str, Any]],
                         account_summary: Dict[str, Any],
                         notes: str = None) -> int:
        """
        Capture a portfolio snapshot.

        Args:
            positions: List of position dicts from IBKR
            account_summary: Account summary from IBKR
            notes: Optional notes for this snapshot

        Returns:
            Snapshot ID
        """
        today = date.today()
        account_id = account_summary.get('account_id', 'default')

        # Current values
        net_liquidation = account_summary.get('net_liquidation', 0)
        total_cash = account_summary.get('total_cash', 0)
        gross_position_value = account_summary.get('gross_position_value', 0)

        # Calculate unrealized P&L from positions
        unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in positions)

        # Get previous snapshot for daily calculations
        prev_snapshot = self.repo.get_latest_snapshot(account_id)
        first_snapshot = self.repo.get_first_snapshot(account_id)

        # Calculate daily P&L and return
        daily_pnl = 0
        daily_return_pct = 0
        if prev_snapshot and prev_snapshot.get('net_liquidation'):
            prev_value = float(prev_snapshot['net_liquidation'])
            daily_pnl = net_liquidation - prev_value
            daily_return_pct = (daily_pnl / prev_value * 100) if prev_value > 0 else 0

        # Calculate cumulative return
        cumulative_return_pct = 0
        if first_snapshot and first_snapshot.get('net_liquidation'):
            first_value = float(first_snapshot['net_liquidation'])
            cumulative_return_pct = ((net_liquidation - first_value) / first_value * 100) if first_value > 0 else 0
        elif not first_snapshot:
            # This is the first snapshot
            cumulative_return_pct = 0

        # Get benchmark data
        benchmark_value = self.get_benchmark_price()
        benchmark_return_pct = 0
        alpha_vs_benchmark = 0

        if first_snapshot and first_snapshot.get('benchmark_value') and benchmark_value:
            first_bench = float(first_snapshot['benchmark_value'])
            benchmark_return_pct = ((benchmark_value - first_bench) / first_bench * 100) if first_bench > 0 else 0
            alpha_vs_benchmark = cumulative_return_pct - benchmark_return_pct

        # Build snapshot
        snapshot = {
            'snapshot_date': today,
            'account_id': account_id,
            'net_liquidation': net_liquidation,
            'total_cash': total_cash,
            'gross_position_value': gross_position_value,
            'realized_pnl': None,  # Would need IBKR trade history
            'unrealized_pnl': unrealized_pnl,
            'daily_pnl': round(daily_pnl, 2),
            'daily_return_pct': round(daily_return_pct, 4),
            'cumulative_return_pct': round(cumulative_return_pct, 4),
            'benchmark_value': benchmark_value,
            'benchmark_return_pct': round(benchmark_return_pct, 4) if benchmark_return_pct else None,
            'alpha_vs_benchmark': round(alpha_vs_benchmark, 4) if alpha_vs_benchmark else None,
            'position_count': len(positions),
            'notes': notes
        }

        snapshot_id = self.repo.save_portfolio_snapshot(snapshot)
        logger.info(f"Captured snapshot {snapshot_id}: ${net_liquidation:,.2f} ({daily_return_pct:+.2f}% today)")

        return snapshot_id

    def get_performance_chart_data(self, days: int = 30) -> pd.DataFrame:
        """Get data formatted for performance chart."""
        snapshots = self.repo.get_portfolio_snapshots(days=days)

        if snapshots.empty:
            return pd.DataFrame()

        # Normalize to percentage returns from first day
        first_value = float(snapshots.iloc[0]['net_liquidation'])
        first_bench = float(snapshots.iloc[0]['benchmark_value']) if snapshots.iloc[0]['benchmark_value'] else None

        snapshots['portfolio_return'] = (snapshots['net_liquidation'].astype(float) - first_value) / first_value * 100

        if first_bench:
            snapshots['benchmark_return'] = (snapshots['benchmark_value'].astype(
                float) - first_bench) / first_bench * 100
        else:
            snapshots['benchmark_return'] = 0

        return snapshots[['snapshot_date', 'net_liquidation', 'portfolio_return', 'benchmark_return', 'daily_pnl',
                          'daily_return_pct']]

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return self.repo.get_performance_summary()


def capture_daily_snapshot(positions: List[Dict], account_summary: Dict, notes: str = None) -> int:
    """Convenience function to capture a snapshot."""
    tracker = PerformanceTracker()
    return tracker.capture_snapshot(positions, account_summary, notes)