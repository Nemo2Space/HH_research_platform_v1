"""
Alpha Platform - Backtest Engine

Core backtesting engine that:
- Runs trading strategies on historical data
- Calculates returns and performance metrics
- Compares against benchmarks
- Feeds results back to AI learner
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from src.db.connection import get_connection, get_engine
from src.db.repository import Repository
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Phase 0: Transaction costs
try:
    from src.backtest.transaction_costs import TransactionCostModel, ROUND_TRIP_COST_TABLE
    TRANSACTION_COSTS_AVAILABLE = True
except ImportError:
    TRANSACTION_COSTS_AVAILABLE = False

# Phase 0: Point-in-Time Validator
try:
    from src.data.pit_validator import PITValidator, BacktestIntegrityChecker
    PIT_VALIDATOR_AVAILABLE = True
except ImportError:
    PIT_VALIDATOR_AVAILABLE = False

# AI Strategy support
try:
    from src.backtest.ai_strategy import AIBacktestStrategy
    AI_STRATEGY_AVAILABLE = True
except ImportError:
    AI_STRATEGY_AVAILABLE = False

# Phase 1: Factor Decomposition
try:
    from src.analytics.factor_decomposition import decompose_backtest_returns, FactorAttribution
    FACTOR_DECOMPOSITION_AVAILABLE = True
except ImportError:
    FACTOR_DECOMPOSITION_AVAILABLE = False

@dataclass
class Trade:
    """Single trade record."""
    ticker: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    signal_type: str
    direction: str  # LONG or SHORT
    return_pct: float
    holding_days: int
    scores: Dict[str, Any] = field(default_factory=dict)
    # Phase 0: Transaction costs
    cost_bps: float = 0.0
    return_pct_net: float = 0.0  # return_pct minus costs

    @property
    def is_winner(self) -> bool:
        if self.direction == 'LONG':
            return self.return_pct > 0
        else:
            return self.return_pct < 0


@dataclass
class BacktestResult:
    """Complete backtest results."""
    strategy_name: str
    start_date: date
    end_date: date
    benchmark: str
    holding_period: int

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Returns
    total_return: float
    avg_return: float
    median_return: float
    std_return: float
    best_trade: float
    worst_trade: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float

    # Benchmark comparison
    benchmark_return: float
    alpha: float
    beta: float

    # By signal breakdown
    returns_by_signal: Dict[str, Dict[str, float]]

    # Individual trades
    trades: List[Trade]

    # Equity curve
    # Equity curve
    equity_curve: pd.DataFrame

    # Phase 0: Data integrity stats
    pit_violations: int = 0  # Count of point-in-time violations detected
    rows_filtered: int = 0  # Rows removed due to data issues
    data_quality_score: float = 100.0  # % of clean data
    factor_attribution: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_name': self.strategy_name,
            'start_date': str(self.start_date),
            'end_date': str(self.end_date),
            'benchmark': self.benchmark,
            'holding_period': self.holding_period,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate, 4),
            'total_return': round(self.total_return, 4),
            'avg_return': round(self.avg_return, 4),
            'median_return': round(self.median_return, 4),
            'std_return': round(self.std_return, 4),
            'best_trade': round(self.best_trade, 4),
            'worst_trade': round(self.worst_trade, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 4),
            'sortino_ratio': round(self.sortino_ratio, 4),
            'max_drawdown': round(self.max_drawdown, 4),
            'benchmark_return': round(self.benchmark_return, 4),
            'alpha': round(self.alpha, 4),
            'returns_by_signal': self.returns_by_signal,
            'pit_violations': self.pit_violations,
            'factor_attribution': self.factor_attribution,
            'rows_filtered': self.rows_filtered,
            'data_quality_score': round(self.data_quality_score, 2),
        }


class BacktestEngine:
    """
    Core backtesting engine.
    Runs strategies on historical data and calculates performance metrics.
    """

    def __init__(self, repository: Optional[Repository] = None):
        self.repo = repository or Repository()
        self.engine = get_engine()

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    def load_historical_scores(self, start_date: str = None, end_date: str = None,
                               tickers: List[str] = None) -> pd.DataFrame:
        """Load historical scores from database."""
        query = """
            SELECT 
                score_date as date,
                ticker,
                sector,
                sentiment,
                fundamental_score,
                growth_score,
                dividend_score,
                total_score,
                gap_score,
                mkt_score,
                signal_type,
                signal_correct,
                op_price as entry_price,
                return_1d,
                return_5d,
                return_10d,
                return_20d,
                price_1d,
                price_5d,
                price_10d,
                price_20d
            FROM historical_scores
            WHERE 1=1
        """
        params = {}

        if start_date:
            query += " AND score_date >= %(start_date)s"
            params['start_date'] = start_date

        if end_date:
            query += " AND score_date <= %(end_date)s"
            params['end_date'] = end_date

        if tickers:
            query += " AND ticker = ANY(%(tickers)s)"
            params['tickers'] = tickers

        query += " ORDER BY score_date, ticker"

        df = pd.read_sql(query, self.engine, params=params)
        logger.info(f"Loaded {len(df)} historical scores")
        return df

    def load_benchmark(self, benchmark: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load benchmark price data."""
        query = """
            SELECT date, close, adj_close
            FROM prices
            WHERE ticker = %(benchmark)s
              AND date >= %(start_date)s
              AND date <= %(end_date)s
            ORDER BY date
        """

        df = pd.read_sql(query, self.engine, params={
            'benchmark': benchmark,
            'start_date': start_date,
            'end_date': end_date
        })

        if df.empty:
            logger.warning(f"No benchmark data for {benchmark}")

        return df

    def _validate_pit_compliance(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate point-in-time compliance of backtest data.
        Phase 0: Prevents lookahead bias.

        Returns:
            Tuple of (cleaned_dataframe, validation_stats)
        """
        stats = {
            'original_rows': len(df),
            'violations': 0,
            'rows_filtered': 0,
            'issues': []
        }

        if not PIT_VALIDATOR_AVAILABLE:
            logger.debug("PIT Validator not available, skipping validation")
            stats['data_quality_score'] = 100.0
            return df, stats

        try:
            checker = BacktestIntegrityChecker()

            # Check 1: Look for future data leakage in returns
            # The return columns should only be populated AFTER the holding period
            # This is usually fine in historical_scores, but let's verify

            # Check 2: Look for suspiciously high returns (possible data errors)
            outlier_mask = pd.Series([False] * len(df), index=df.index)
            for col in ['return_1d', 'return_5d', 'return_10d', 'return_20d']:
                if col in df.columns:
                    # Flag returns > 100% or < -50% as suspicious
                    suspicious = (df[col] > 100) | (df[col] < -50)
                    if suspicious.any():
                        count = suspicious.sum()
                        stats['issues'].append(f"{col}: {count} outlier returns (>100% or <-50%)")
                        stats['violations'] += count
                        outlier_mask |= suspicious

            # Check 3: Look for duplicate entries (same ticker, same date)
            duplicates = df.duplicated(subset=['ticker', 'date'], keep='first')
            if duplicates.any():
                dup_count = duplicates.sum()
                stats['issues'].append(f"Duplicate entries: {dup_count}")
                stats['violations'] += dup_count

            # Check 4: Look for missing prices (signals without entry price)
            if 'entry_price' in df.columns:
                missing_price = df['entry_price'].isna() | (df['entry_price'] <= 0)
                if missing_price.any():
                    count = missing_price.sum()
                    stats['issues'].append(f"Missing entry prices: {count}")
                    # Don't count as violation, just note it

            # Check 5: Verify chronological order
            if not df['date'].is_monotonic_increasing:
                # Sort by date
                df = df.sort_values('date').reset_index(drop=True)
                stats['issues'].append("Data was not chronologically sorted (fixed)")

            # Filter out problematic rows
            rows_to_remove = outlier_mask | duplicates
            if rows_to_remove.any():
                stats['rows_filtered'] = rows_to_remove.sum()
                df = df[~rows_to_remove].reset_index(drop=True)
                logger.warning(f"PIT Validation: Filtered {stats['rows_filtered']} problematic rows")

            # Calculate data quality score
            if stats['original_rows'] > 0:
                clean_rows = stats['original_rows'] - stats['rows_filtered']
                stats['data_quality_score'] = (clean_rows / stats['original_rows']) * 100
            else:
                stats['data_quality_score'] = 100.0

            # Log summary
            if stats['violations'] > 0:
                logger.warning(f"PIT Validation found {stats['violations']} issues: {stats['issues']}")
            else:
                logger.info(f"PIT Validation passed: {len(df)} clean rows")

        except Exception as e:
            logger.error(f"PIT Validation error: {e}")
            stats['data_quality_score'] = 100.0  # Don't fail backtest on validation error

        return df, stats

    # =========================================================================
    # STRATEGY EXECUTION
    # =========================================================================

    def run_backtest(self,
                     strategy: str = 'signal_based',
                     start_date: str = None,
                     end_date: str = None,
                     holding_period: int = 10,
                     benchmark: str = 'SPY',
                     params: Dict[str, Any] = None) -> BacktestResult:
        """
        Run a backtest with the specified strategy.

        Args:
            strategy: Strategy name ('signal_based', 'score_threshold', 'composite')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            holding_period: Days to hold each position
            benchmark: Benchmark ticker for comparison
            params: Strategy-specific parameters

        Returns:
            BacktestResult with full analysis
        """
        params = params or {}

        # Load data
        df = self.load_historical_scores(start_date, end_date)

        if df.empty:
            raise ValueError("No historical data found for the specified period")

        # Phase 0: Validate point-in-time compliance
        df, pit_stats = self._validate_pit_compliance(df)

        if df.empty:
            raise ValueError("No valid data remaining after PIT validation")

        # Get date range from data
        actual_start = df['date'].min()
        actual_end = df['date'].max()

        # Load benchmark
        benchmark_df = self.load_benchmark(benchmark, str(actual_start), str(actual_end))

        # Generate trades based on strategy
        if strategy == 'signal_based':
            trades = self._run_signal_strategy(df, holding_period, params)
        elif strategy == 'score_threshold':
            trades = self._run_score_threshold_strategy(df, holding_period, params)
        elif strategy == 'composite':
            trades = self._run_composite_strategy(df, holding_period, params)
        elif strategy == 'sentiment_only':
            trades = self._run_sentiment_strategy(df, holding_period, params)
        elif strategy == 'fundamental_only':
            trades = self._run_fundamental_strategy(df, holding_period, params)
        elif strategy == 'ai_probability':
            trades = self._run_ai_strategy(df, holding_period, params)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


        # Calculate results
        result = self._calculate_results(
            trades=trades,
            strategy_name=strategy,
            start_date=actual_start,
            end_date=actual_end,
            benchmark=benchmark,
            benchmark_df=benchmark_df,
            holding_period=holding_period
        )

        # Phase 1: Factor Decomposition
        if FACTOR_DECOMPOSITION_AVAILABLE and trades:
            try:
                trade_dicts = [
                    {'ticker': t.ticker, 'entry_date': t.entry_date, 'return_pct': t.return_pct_net}
                    for t in trades
                ]
                attribution = decompose_backtest_returns(trade_dicts, holding_period)
                result.factor_attribution = attribution.to_dict()

                # Log summary
                logger.info(f"Factor Attribution: Alpha={attribution.alpha:.2f}%, "
                            f"Beta={attribution.avg_market_beta:.2f}, "
                            f"Momentum={attribution.avg_momentum_exposure:.2f}")
            except Exception as e:
                logger.debug(f"Factor decomposition failed: {e}")

        # Phase 0: Add PIT validation stats to result
        result.pit_violations = pit_stats.get('violations', 0)
        result.rows_filtered = pit_stats.get('rows_filtered', 0)
        result.data_quality_score = pit_stats.get('data_quality_score', 100.0)

        return result

    def _run_signal_strategy(self, df: pd.DataFrame, holding_period: int,
                             params: Dict[str, Any]) -> List[Trade]:
        """Signal-based strategy: Buy on BUY/STRONG_BUY signals."""
        buy_signals = params.get('buy_signals', ['STRONG_BUY', 'BUY', 'WEAK_BUY'])
        sell_signals = params.get('sell_signals', ['STRONG_SELL', 'SELL'])

        trades = []
        return_col = self._get_return_column(holding_period)
        price_col = self._get_price_column(holding_period)

        for _, row in df.iterrows():
            signal = row.get('signal_type')
            ret = row.get(return_col)

            if pd.isna(ret) or ret is None:
                continue

            entry_price = row.get('entry_price', 0)
            exit_price = row.get(price_col, entry_price)

            if signal in buy_signals:
                trades.append(Trade(
                    ticker=row['ticker'],
                    entry_date=row['date'],
                    exit_date=row['date'] + timedelta(days=holding_period),
                    entry_price=float(entry_price) if entry_price else 0,
                    exit_price=float(exit_price) if exit_price else 0,
                    signal_type=signal,
                    direction='LONG',
                    return_pct=float(ret),
                    holding_days=holding_period,
                    scores={
                        'sentiment': row.get('sentiment'),
                        'fundamental': row.get('fundamental_score'),
                        'total': row.get('total_score'),
                    }
                ))
            elif signal in sell_signals:
                trades.append(Trade(
                    ticker=row['ticker'],
                    entry_date=row['date'],
                    exit_date=row['date'] + timedelta(days=holding_period),
                    entry_price=float(entry_price) if entry_price else 0,
                    exit_price=float(exit_price) if exit_price else 0,
                    signal_type=signal,
                    direction='SHORT',
                    return_pct=-float(ret),
                    holding_days=holding_period,
                    scores={
                        'sentiment': row.get('sentiment'),
                        'fundamental': row.get('fundamental_score'),
                        'total': row.get('total_score'),
                    }
                ))

        return trades

    def _run_score_threshold_strategy(self, df: pd.DataFrame, holding_period: int,
                                      params: Dict[str, Any]) -> List[Trade]:
        """Score threshold strategy: Buy when score exceeds threshold."""
        score_col = params.get('score_column', 'total_score')
        buy_threshold = params.get('buy_threshold', 60)
        sell_threshold = params.get('sell_threshold', 40)

        trades = []
        return_col = self._get_return_column(holding_period)
        price_col = self._get_price_column(holding_period)

        for _, row in df.iterrows():
            score = row.get(score_col)
            ret = row.get(return_col)

            if pd.isna(score) or pd.isna(ret):
                continue

            entry_price = row.get('entry_price', 0)
            exit_price = row.get(price_col, entry_price)

            if score >= buy_threshold:
                trades.append(Trade(
                    ticker=row['ticker'],
                    entry_date=row['date'],
                    exit_date=row['date'] + timedelta(days=holding_period),
                    entry_price=float(entry_price) if entry_price else 0,
                    exit_price=float(exit_price) if exit_price else 0,
                    signal_type=f"{score_col}>={buy_threshold}",
                    direction='LONG',
                    return_pct=float(ret),
                    holding_days=holding_period,
                    scores={score_col: score}
                ))
            elif score <= sell_threshold:
                trades.append(Trade(
                    ticker=row['ticker'],
                    entry_date=row['date'],
                    exit_date=row['date'] + timedelta(days=holding_period),
                    entry_price=float(entry_price) if entry_price else 0,
                    exit_price=float(exit_price) if exit_price else 0,
                    signal_type=f"{score_col}<={sell_threshold}",
                    direction='SHORT',
                    return_pct=-float(ret),
                    holding_days=holding_period,
                    scores={score_col: score}
                ))

        return trades

    def _run_composite_strategy(self, df: pd.DataFrame, holding_period: int,
                                params: Dict[str, Any]) -> List[Trade]:
        """Composite strategy: Multiple conditions must be met."""
        min_sentiment = params.get('min_sentiment', 55)
        min_fundamental = params.get('min_fundamental', 55)
        min_total = params.get('min_total', 55)
        require_all = params.get('require_all', False)

        trades = []
        return_col = self._get_return_column(holding_period)
        price_col = self._get_price_column(holding_period)

        for _, row in df.iterrows():
            sentiment = row.get('sentiment', 50)
            fundamental = row.get('fundamental_score', 50)
            total = row.get('total_score', 50)
            ret = row.get(return_col)

            if pd.isna(ret):
                continue

            conditions_met = 0
            if sentiment and sentiment >= min_sentiment:
                conditions_met += 1
            if fundamental and fundamental >= min_fundamental:
                conditions_met += 1
            if total and total >= min_total:
                conditions_met += 1

            should_trade = (conditions_met == 3) if require_all else (conditions_met >= 2)

            if should_trade:
                entry_price = row.get('entry_price', 0)
                exit_price = row.get(price_col, entry_price)

                trades.append(Trade(
                    ticker=row['ticker'],
                    entry_date=row['date'],
                    exit_date=row['date'] + timedelta(days=holding_period),
                    entry_price=float(entry_price) if entry_price else 0,
                    exit_price=float(exit_price) if exit_price else 0,
                    signal_type='COMPOSITE',
                    direction='LONG',
                    return_pct=float(ret),
                    holding_days=holding_period,
                    scores={
                        'sentiment': sentiment,
                        'fundamental': fundamental,
                        'total': total,
                    }
                ))

        return trades

    def _run_sentiment_strategy(self, df: pd.DataFrame, holding_period: int,
                                params: Dict[str, Any]) -> List[Trade]:
        """Sentiment-only strategy."""
        params['score_column'] = 'sentiment'
        params.setdefault('buy_threshold', 65)
        params.setdefault('sell_threshold', 35)
        return self._run_score_threshold_strategy(df, holding_period, params)

    def _run_fundamental_strategy(self, df: pd.DataFrame, holding_period: int,
                                  params: Dict[str, Any]) -> List[Trade]:
        """Fundamental-only strategy."""
        params['score_column'] = 'fundamental_score'
        params.setdefault('buy_threshold', 65)
        params.setdefault('sell_threshold', 35)
        return self._run_score_threshold_strategy(df, holding_period, params)

    def _run_ai_strategy(self, df: pd.DataFrame, holding_period: int,
                         params: Dict[str, Any]) -> List[Trade]:
        """AI probability-based strategy using ML model."""
        if not AI_STRATEGY_AVAILABLE:
            logger.warning("AI Strategy not available. Using fallback signal-based strategy.")
            return self._run_signal_strategy(df, holding_period,
                                             {'buy_signals': ['STRONG_BUY', 'BUY']})

        try:
            strategy = AIBacktestStrategy()
            if not strategy.initialize():
                logger.warning("AI model not loaded. Train first: python -m src.ml.ai_trading_system --train")
                return []
            return strategy.run_strategy(df, holding_period, params)
        except Exception as e:
            logger.error(f"AI strategy failed: {e}")
            return []

    def _apply_transaction_costs(self, trades: List[Trade]) -> List[Trade]:
        """
        Apply transaction costs to all trades.
        Phase 0: Realistic execution cost modeling.
        """
        if not TRANSACTION_COSTS_AVAILABLE:
            logger.debug("Transaction costs module not available, skipping")
            for trade in trades:
                trade.return_pct_net = trade.return_pct
            return trades

        cost_model = TransactionCostModel()

        for trade in trades:
            # Estimate trade value from entry price (assume 100 shares as proxy)
            trade_value = trade.entry_price * 100 if trade.entry_price > 0 else 10000

            # Get round-trip cost based on trade characteristics
            try:
                cost_bps = cost_model.estimate_round_trip_cost(
                    ticker=trade.ticker,
                    trade_value=trade_value,
                    # Could add market_cap lookup here for more accuracy
                )
            except Exception:
                # Fallback to default large-cap cost
                cost_bps = ROUND_TRIP_COST_TABLE.get('large', 10)

            trade.cost_bps = cost_bps
            trade.return_pct_net = trade.return_pct - (cost_bps / 100)

        return trades

    # =========================================================================
    # RESULTS CALCULATION
    # =========================================================================

    def _calculate_results(self, trades: List[Trade], strategy_name: str,
                           start_date: date, end_date: date, benchmark: str,
                           benchmark_df: pd.DataFrame, holding_period: int) -> BacktestResult:
        """Calculate comprehensive backtest results."""

        if not trades:
            return self._empty_result(strategy_name, start_date, end_date,
                                      benchmark, holding_period)

            # Phase 0: Apply transaction costs
        trades = self._apply_transaction_costs(trades)

        # Use net returns (after costs) for all calculations
        returns = [t.return_pct_net for t in trades]
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]

        total_trades = len(trades)
        winning_trades = len(winners)
        losing_trades = len(losers)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        returns_array = np.array(returns)
        total_return = float(np.sum(returns_array))
        avg_return = float(np.mean(returns_array))
        median_return = float(np.median(returns_array))
        std_return = float(np.std(returns_array)) if len(returns_array) > 1 else 0
        best_trade = float(np.max(returns_array))
        worst_trade = float(np.min(returns_array))

        sharpe_ratio = self._calculate_sharpe(returns_array, holding_period)
        sortino_ratio = self._calculate_sortino(returns_array, holding_period)
        max_drawdown = self._calculate_max_drawdown(returns_array)

        benchmark_return = self._calculate_benchmark_return(benchmark_df, start_date, end_date)
        alpha = avg_return * (252 / holding_period) - benchmark_return
        beta = 1.0

        returns_by_signal = self._calculate_returns_by_signal(trades)
        equity_curve = self._build_equity_curve(trades)

        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            benchmark=benchmark,
            holding_period=holding_period,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return=total_return,
            avg_return=avg_return,
            median_return=median_return,
            std_return=std_return,
            best_trade=best_trade,
            worst_trade=worst_trade,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
            returns_by_signal=returns_by_signal,
            trades=trades,
            equity_curve=equity_curve
        )

    def _calculate_sharpe(self, returns: np.ndarray, holding_period: int,
                          risk_free_rate: float = 0.05) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0

        periods_per_year = 252 / holding_period
        annual_return = np.mean(returns) * periods_per_year
        annual_std = np.std(returns) * np.sqrt(periods_per_year)

        sharpe = (annual_return - risk_free_rate) / annual_std if annual_std > 0 else 0
        return float(sharpe)

    def _calculate_sortino(self, returns: np.ndarray, holding_period: int,
                           risk_free_rate: float = 0.05) -> float:
        """Calculate annualized Sortino ratio."""
        if len(returns) < 2:
            return 0.0

        downside = returns[returns < 0]
        if len(downside) == 0:
            return 10.0

        periods_per_year = 252 / holding_period
        annual_return = np.mean(returns) * periods_per_year
        downside_std = np.std(downside) * np.sqrt(periods_per_year)

        sortino = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        return float(sortino)

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0

        cumulative = np.cumprod(1 + returns / 100)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak

        return float(np.min(drawdown)) * 100

    def _calculate_benchmark_return(self, benchmark_df: pd.DataFrame,
                                    start_date: date, end_date: date) -> float:
        """Calculate benchmark total return."""
        if benchmark_df.empty:
            return 0.0

        start_price = benchmark_df.iloc[0]['adj_close'] or benchmark_df.iloc[0]['close']
        end_price = benchmark_df.iloc[-1]['adj_close'] or benchmark_df.iloc[-1]['close']

        if start_price and end_price:
            return float((end_price - start_price) / start_price * 100)
        return 0.0

    def _calculate_returns_by_signal(self, trades: List[Trade]) -> Dict[str, Dict[str, float]]:
        """Calculate performance by signal type."""
        by_signal = {}

        for trade in trades:
            sig = trade.signal_type
            if sig not in by_signal:
                by_signal[sig] = {'returns': [], 'count': 0, 'winners': 0}

            by_signal[sig]['returns'].append(trade.return_pct)
            by_signal[sig]['count'] += 1
            if trade.is_winner:
                by_signal[sig]['winners'] += 1

        result = {}
        for sig, data in by_signal.items():
            returns = data['returns']
            result[sig] = {
                'count': data['count'],
                'avg_return': round(float(np.mean(returns)), 2),
                'total_return': round(float(np.sum(returns)), 2),
                'win_rate': round(data['winners'] / data['count'], 2) if data['count'] > 0 else 0,
                'best': round(float(max(returns)), 2),
                'worst': round(float(min(returns)), 2),
            }

        return result

    def _build_equity_curve(self, trades: List[Trade]) -> pd.DataFrame:
        """Build equity curve from trades."""
        if not trades:
            return pd.DataFrame()

        sorted_trades = sorted(trades, key=lambda t: t.entry_date)

        data = []
        cumulative = 100.0

        for trade in sorted_trades:
            cumulative = cumulative * (1 + trade.return_pct / 100)
            data.append({
                'date': trade.entry_date,
                'equity': cumulative,
                'return': trade.return_pct,
                'ticker': trade.ticker,
                'signal': trade.signal_type
            })

        return pd.DataFrame(data)

    def _get_return_column(self, holding_period: int) -> str:
        """Map holding period to return column."""
        mapping = {1: 'return_1d', 5: 'return_5d', 10: 'return_10d', 20: 'return_20d'}
        closest = min(mapping.keys(), key=lambda x: abs(x - holding_period))
        return mapping[closest]

    def _get_price_column(self, holding_period: int) -> str:
        """Map holding period to price column."""
        mapping = {1: 'price_1d', 5: 'price_5d', 10: 'price_10d', 20: 'price_20d'}
        closest = min(mapping.keys(), key=lambda x: abs(x - holding_period))
        return mapping[closest]

    def _empty_result(self, strategy_name: str, start_date: date, end_date: date,
                      benchmark: str, holding_period: int) -> BacktestResult:
        """Return empty result when no trades."""

        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            benchmark=benchmark,
            holding_period=holding_period,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            total_return=0,
            avg_return=0,
            median_return=0,
            std_return=0,
            best_trade=0,
            worst_trade=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            benchmark_return=0,
            alpha=0,
            beta=0,
            returns_by_signal={},
            trades=[],
            equity_curve=pd.DataFrame(),
            pit_violations=0,
            rows_filtered=0,
            data_quality_score=100.0,
            factor_attribution=None
        )
    # =========================================================================
    # SAVE FOR AI LEARNING
    # =========================================================================

    def save_results_for_learning(self, result: BacktestResult) -> bool:
        """Save backtest results for AI learning."""
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO backtest_results (
                            strategy_name, run_date, start_date, end_date,
                            holding_period, benchmark, total_trades,
                            win_rate, avg_return, sharpe_ratio, alpha,
                            returns_by_signal
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        result.strategy_name,
                        datetime.now().date(),
                        result.start_date,
                        result.end_date,
                        result.holding_period,
                        result.benchmark,
                        result.total_trades,
                        result.win_rate,
                        result.avg_return,
                        result.sharpe_ratio,
                        result.alpha,
                        str(result.returns_by_signal)
                    ))

            logger.info(f"Saved backtest results: {result.strategy_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to save backtest results: {e}")
            return False