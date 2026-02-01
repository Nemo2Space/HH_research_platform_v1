"""
Portfolio Backtesting Module - FIXED VERSION V2
================================================

Correct backtest implementation with:
- NO look-ahead bias (ffill only, no bfill)
- Start date where ALL tickers have valid prices
- Proper cash accounting with transaction costs deducted from NAV
- Monthly rebalance on first trading day of new month
- Clean price index (no duplicates)

Fixes the "cliff drop" bug at the beginning of backtests.

Author: HH Research Platform (Fixed)
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Database connection
try:
    from src.db.connection import get_connection
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

# Yahoo Finance for historical data
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    portfolio_id: int  # Use -1 for CSV portfolios
    portfolio_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    benchmark: str
    rebalance_frequency: str  # 'never', 'monthly', 'quarterly', 'yearly'
    transaction_cost_pct: float = 0.0  # % per trade
    risk_free_rate: float = 0.05  # Annual risk-free rate for Sharpe/alpha calculations


@dataclass
class BacktestResult:
    """Backtesting results."""
    portfolio_returns: pd.Series
    benchmark_returns: pd.Series
    portfolio_value: pd.Series
    benchmark_value: pd.Series
    holdings_over_time: pd.DataFrame

    # Performance metrics
    total_return_pct: float
    annualized_return_pct: float
    volatility_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float

    # Benchmark comparison
    benchmark_total_return_pct: float
    benchmark_annualized_return_pct: float
    benchmark_volatility_pct: float
    benchmark_sharpe_ratio: float
    benchmark_max_drawdown_pct: float

    alpha: float
    beta: float

    # Trade statistics
    num_rebalances: int
    total_transaction_costs: float

    # Universe hygiene report
    dropped_tickers_df: Optional[pd.DataFrame] = None  # Tickers dropped due to insufficient data


class PortfolioBacktester:
    """Backtest saved portfolios or CSV portfolios with historical data."""

    def __init__(self):
        self.price_cache = {}
        self.last_dropped_tickers = None  # Store dropped tickers report for UI display

    def filter_universe_by_price_availability(
        self,
        prices_filled: pd.DataFrame,
        target_weights: pd.Series,
        min_history_days: int = 60,
        max_missing_ratio: float = 0.05
    ) -> Tuple[Optional[pd.Series], pd.DataFrame, Optional[pd.Timestamp]]:
        """
        Drops tickers that don't have enough valid history or have too many missing values.
        This is standard "universe hygiene" used by hedge funds.

        Args:
            prices_filled: Forward-filled price DataFrame
            target_weights: Series of ticker weights (index=ticker, values=weight as decimal)
            min_history_days: Minimum required trading days of valid data (default 60 = ~3 months)
            max_missing_ratio: Maximum allowed missing data ratio after ffill (default 5%)

        Returns:
            (kept_weights, dropped_df, start_date) or (None, dropped_df, None) if all dropped
        """
        tickers = [t for t in target_weights.index if t in prices_filled.columns]
        if not tickers:
            return None, pd.DataFrame([{"ticker": "ALL", "reason": "No tickers found in price columns"}]), None

        df = prices_filled[tickers].copy()

        report = []
        for t in tickers:
            s = df[t]
            valid = s.notna()
            valid_count = int(valid.sum())
            total_count = int(len(s))
            missing_ratio = float(1.0 - (valid_count / total_count)) if total_count > 0 else 1.0
            first_valid = s.dropna().index.min() if valid_count > 0 else None

            reason = None
            if valid_count < min_history_days:
                reason = f"Insufficient history: {valid_count} < {min_history_days} days"
            elif missing_ratio > max_missing_ratio:
                reason = f"Too many missing values: {missing_ratio:.2%} > {max_missing_ratio:.2%}"

            report.append({
                "ticker": t,
                "valid_days": valid_count,
                "total_days": total_count,
                "missing_ratio": f"{missing_ratio:.2%}",
                "first_valid_date": first_valid.strftime("%Y-%m-%d") if first_valid else "N/A",
                "status": "DROP" if reason else "KEEP",
                "reason": reason if reason else ""
            })

        rep_df = pd.DataFrame(report).sort_values(["status", "valid_days"], ascending=[True, True])

        kept = rep_df[rep_df["status"] == "KEEP"]["ticker"].tolist()
        dropped_df = rep_df[rep_df["status"] == "DROP"].copy()

        if not kept:
            return None, dropped_df, None

        kept_weights = target_weights.loc[kept].copy()
        kept_weights = kept_weights / kept_weights.sum()

        # Start date: first date where all kept tickers are non-NaN
        valid_mask = prices_filled[kept].notna().all(axis=1)
        start_date = prices_filled.index[valid_mask.argmax()] if valid_mask.any() else None

        return kept_weights, dropped_df, start_date

    def get_saved_portfolios(self) -> pd.DataFrame:
        """Get list of saved portfolios."""
        if not DB_AVAILABLE:
            return pd.DataFrame()

        try:
            with get_connection() as conn:
                query = """
                    SELECT 
                        id,
                        name,
                        description,
                        created_at,
                        num_holdings,
                        total_value,
                        avg_score
                    FROM saved_portfolios
                    ORDER BY created_at DESC
                """
                return pd.read_sql(query, conn)
        except Exception as e:
            st.error(f"Error loading portfolios: {e}")
            return pd.DataFrame()

    def load_portfolio_holdings(self, portfolio_id: int) -> pd.DataFrame:
        """Load holdings for a saved portfolio."""
        if not DB_AVAILABLE:
            return pd.DataFrame()

        try:
            with get_connection() as conn:
                query = """
                    SELECT 
                        ticker,
                        weight_pct,
                        value,
                        shares,
                        score,
                        conviction
                    FROM saved_portfolio_holdings
                    WHERE portfolio_id = %s
                    ORDER BY weight_pct DESC
                """
                return pd.read_sql(query, conn, params=(portfolio_id,))
        except Exception as e:
            st.error(f"Error loading holdings: {e}")
            return pd.DataFrame()

    def fetch_historical_prices(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch historical adjusted close prices (clean index; no duplicates)."""
        if not YF_AVAILABLE:
            st.error("yfinance not available. Install: pip install yfinance")
            return pd.DataFrame()

        try:
            cache_key = f"{','.join(sorted(tickers))}_{start_date}_{end_date}"
            if cache_key in self.price_cache:
                return self.price_cache[cache_key]

            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )

            if data.empty:
                return pd.DataFrame()

            if len(tickers) == 1:
                prices = data[['Close']].copy()
                prices.columns = [tickers[0]]
            else:
                prices = data['Close'].copy()

            # --- Critical cleaning ---
            prices = prices.sort_index()
            prices = prices[~prices.index.duplicated(keep='first')]
            prices = prices.astype(float)

            self.price_cache[cache_key] = prices
            return prices

        except Exception as e:
            st.error(f"Error fetching prices: {e}")
            return pd.DataFrame()

    def calculate_portfolio_value(
        self,
        prices: pd.DataFrame,
        holdings: pd.DataFrame,
        initial_capital: float,
        rebalance_frequency: str = 'never',
        transaction_cost_pct: float = 0.0
    ) -> Tuple[pd.Series, pd.DataFrame, int, float]:
        """
        Correct portfolio backtest with:
        - NO look-ahead fill (ffill only)
        - Start date where all included tickers have valid prices
        - Monthly rebalance on first trading day of new month
        - Transaction costs deducted via a cash account
        """
        if prices is None or prices.empty or holdings is None or holdings.empty:
            return pd.Series(dtype=float), pd.DataFrame(), 0, 0.0

        # --- Clean / align prices ---
        prices = prices.sort_index()
        prices = prices[~prices.index.duplicated(keep='first')]

        # Forward fill ONLY (no bfill -> avoids look-ahead)
        prices_filled = prices.ffill()

        # Target weights
        target_weights = holdings.set_index('ticker')['weight_pct'].astype(float) / 100.0
        target_weights.index = target_weights.index.astype(str)

        # Keep only tickers we actually have columns for
        tickers_in_data = [t for t in target_weights.index if t in prices_filled.columns]
        if not tickers_in_data:
            st.error("None of the portfolio tickers exist in price data columns.")
            return pd.Series(index=prices_filled.index, dtype=float), pd.DataFrame(), 0, 0.0

        target_weights = target_weights.loc[tickers_in_data]

        # --- Filter universe by availability (hedge fund standard "universe hygiene") ---
        # This drops tickers with insufficient data and renormalizes remaining weights
        kept_weights, dropped_df, start_date = self.filter_universe_by_price_availability(
            prices_filled=prices_filled,
            target_weights=target_weights,
            min_history_days=60,       # ~3 months trading days (adjust as needed)
            max_missing_ratio=0.05     # 5% missing allowed (adjust as needed)
        )

        # Store dropped tickers for UI display
        self.last_dropped_tickers = dropped_df if not dropped_df.empty else None

        if kept_weights is None or start_date is None:
            st.error("No date where all included tickers have valid prices after forward-fill.")
            if dropped_df is not None and not dropped_df.empty:
                st.warning(f"⚠️ {len(dropped_df)} tickers dropped due to insufficient price data:")
                st.dataframe(dropped_df[["ticker", "valid_days", "first_valid_date", "reason"]], hide_index=True)
            return pd.Series(index=prices_filled.index, dtype=float), pd.DataFrame(), 0, 0.0

        # Use filtered universe
        tickers_in_data = kept_weights.index.tolist()
        target_weights = kept_weights

        # Notify user of dropped tickers (if any)
        if dropped_df is not None and not dropped_df.empty:
            st.warning(f"⚠️ {len(dropped_df)} tickers dropped due to insufficient price data: "
                      f"{', '.join(dropped_df['ticker'].tolist()[:5])}" +
                      (f" and {len(dropped_df)-5} more" if len(dropped_df) > 5 else ""))

        # Start from the computed start_date
        prices_filled = prices_filled.loc[start_date:].copy()

        # --- Build rebalance dates (first trading day of period) ---
        rebalance_dates = set()
        if rebalance_frequency != 'never':
            dates = prices_filled.index.to_list()
            for i in range(1, len(dates)):
                d = dates[i]
                p = dates[i - 1]

                if rebalance_frequency == 'monthly':
                    if d.month != p.month:
                        rebalance_dates.add(d)
                elif rebalance_frequency == 'quarterly':
                    if d.month in (1, 4, 7, 10) and d.month != p.month:
                        rebalance_dates.add(d)
                elif rebalance_frequency == 'yearly':
                    if d.year != p.year:
                        rebalance_dates.add(d)

        # --- Portfolio state ---
        current_shares: Dict[str, float] = {t: 0.0 for t in tickers_in_data}
        cash = float(initial_capital)
        total_costs = 0.0
        num_rebalances = 0

        portfolio_value = pd.Series(index=prices_filled.index, dtype=float)
        holdings_over_time = []

        # Helper: current holdings value
        def holdings_value_at(row) -> float:
            return float(sum(current_shares[t] * row[t] for t in tickers_in_data))

        # --- Initial buy (deduct costs correctly) ---
        first_row = prices_filled.iloc[0]

        # Investable notional so that: invest + cost = initial_capital
        # Total cost at init = investable * tc (because sum weights = 1)
        investable = cash / (1.0 + transaction_cost_pct) if transaction_cost_pct > 0 else cash

        for t in tickers_in_data:
            target_value = investable * float(target_weights[t])
            price = float(first_row[t])
            shares = target_value / price
            current_shares[t] = shares

            cost = target_value * transaction_cost_pct
            total_costs += cost
            cash -= (target_value + cost)

        # Small numeric drift guard
        if abs(cash) < 1e-8:
            cash = 0.0

        # --- Backtest loop ---
        for date, row in prices_filled.iterrows():
            # Rebalance
            if date in rebalance_dates:
                equity = cash + holdings_value_at(row)
                desired_values = {t: equity * float(target_weights[t]) for t in tickers_in_data}

                # Compute trades
                trades = {}
                for t in tickers_in_data:
                    cur_val = current_shares[t] * float(row[t])
                    trades[t] = desired_values[t] - cur_val

                # Transaction costs on turnover
                turnover = sum(abs(v) for v in trades.values())
                est_cost = turnover * transaction_cost_pct

                # If we don't have enough cash to pay costs (or buys), scale buys down.
                # This keeps the simulation feasible without creating fake negative cash.
                if transaction_cost_pct > 0:
                    # Cash impact = sum(buys) + cost - sum(sells)
                    buys = sum(max(0.0, v) for v in trades.values())
                    sells = sum(max(0.0, -v) for v in trades.values())
                    net_cash_needed = buys - sells + est_cost

                    if net_cash_needed > cash and buys > 0:
                        # Scale only the buy side
                        scale = max(0.0, (cash + sells - est_cost) / buys)
                        for t in tickers_in_data:
                            if trades[t] > 0:
                                trades[t] *= scale
                        # Recompute turnover/cost after scaling
                        turnover = sum(abs(v) for v in trades.values())
                        est_cost = turnover * transaction_cost_pct

                # Execute trades
                for t in tickers_in_data:
                    trade_val = trades[t]
                    price = float(row[t])
                    delta_shares = trade_val / price
                    current_shares[t] += delta_shares
                    cash -= trade_val

                # Deduct transaction costs
                cash -= est_cost
                total_costs += est_cost
                num_rebalances += 1

            # Daily valuation (prices are forward-filled, so no artificial zeros)
            equity = cash + holdings_value_at(row)
            portfolio_value.loc[date] = equity

            snapshot = {"date": date, "cash": cash}
            for t in tickers_in_data:
                snapshot[t] = current_shares[t]
            holdings_over_time.append(snapshot)

        holdings_df = pd.DataFrame(holdings_over_time).set_index("date")
        return portfolio_value, holdings_df, num_rebalances, total_costs

    def calculate_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series = None,
        risk_free_rate: float = 0.05
    ) -> Dict:
        """Calculate performance metrics."""

        # Remove any NaN or inf values
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns) == 0:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
            }

        # Basic metrics
        total_return = (1 + returns).prod() - 1
        trading_days = len(returns)
        years = trading_days / 252

        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        volatility = returns.std() * np.sqrt(252)

        # Sharpe ratio
        excess_returns = returns - risk_free_rate / 252
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
        }

        # Benchmark metrics
        if benchmark_returns is not None:
            benchmark_returns = benchmark_returns.replace([np.inf, -np.inf], np.nan).dropna()

            if len(benchmark_returns) > 0:
                bench_total = (1 + benchmark_returns).prod() - 1
                bench_years = len(benchmark_returns) / 252
                bench_annualized = (1 + bench_total) ** (1 / bench_years) - 1 if bench_years > 0 else 0
                bench_volatility = benchmark_returns.std() * np.sqrt(252)
                bench_excess = benchmark_returns - risk_free_rate / 252
                bench_sharpe = bench_excess.mean() / bench_excess.std() * np.sqrt(252) if bench_excess.std() > 0 else 0

                bench_cumulative = (1 + benchmark_returns).cumprod()
                bench_running_max = bench_cumulative.expanding().max()
                bench_drawdown = (bench_cumulative - bench_running_max) / bench_running_max
                bench_max_dd = bench_drawdown.min()

                metrics['benchmark_total_return'] = bench_total
                metrics['benchmark_annualized_return'] = bench_annualized
                metrics['benchmark_volatility'] = bench_volatility
                metrics['benchmark_sharpe_ratio'] = bench_sharpe
                metrics['benchmark_max_drawdown'] = bench_max_dd

                # Alpha and Beta - align the series first
                aligned_returns, aligned_bench = returns.align(benchmark_returns, join='inner')

                if len(aligned_returns) > 1:
                    covariance = aligned_returns.cov(aligned_bench)
                    benchmark_variance = aligned_bench.var()
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                    alpha = annualized_return - (risk_free_rate + beta * (bench_annualized - risk_free_rate))
                else:
                    alpha = 0
                    beta = 0

                metrics['alpha'] = alpha
                metrics['beta'] = beta

        return metrics

    def _run_backtest_core(
        self,
        holdings: pd.DataFrame,
        config: BacktestConfig
    ) -> Optional[BacktestResult]:
        """
        Core backtest logic used by both run_backtest and run_backtest_from_holdings.

        Args:
            holdings: DataFrame with columns: ticker, weight_pct (and optionally others)
            config: BacktestConfig object

        Returns:
            BacktestResult or None if failed
        """
        if holdings.empty:
            st.error("No holdings provided for backtesting")
            return None

        # Get all tickers (portfolio + benchmark)
        portfolio_tickers = holdings['ticker'].tolist()
        all_tickers = portfolio_tickers + [config.benchmark]

        # Fetch historical prices
        with st.spinner(f"Fetching historical data for {len(all_tickers)} tickers..."):
            prices = self.fetch_historical_prices(
                all_tickers,
                config.start_date,
                config.end_date
            )

        if prices.empty:
            st.error("No price data available for the selected period")
            return None

        # Check which tickers have data
        available_tickers = [t for t in portfolio_tickers if t in prices.columns]
        missing_tickers = [t for t in portfolio_tickers if t not in prices.columns]

        if missing_tickers:
            st.warning(f"⚠️ No price data for: {', '.join(missing_tickers[:10])}" +
                      (f" and {len(missing_tickers)-10} more" if len(missing_tickers) > 10 else ""))

        if not available_tickers:
            st.error("No tickers have available price data for the selected period")
            return None

        # Filter holdings to only those with price data and renormalize weights
        filtered_holdings = holdings[holdings['ticker'].isin(available_tickers)].copy()
        total_weight = filtered_holdings['weight_pct'].sum()
        if total_weight > 0:
            filtered_holdings['weight_pct'] = filtered_holdings['weight_pct'] / total_weight * 100

        # Calculate portfolio value
        portfolio_value, holdings_df, num_rebalances, total_costs = self.calculate_portfolio_value(
            prices[available_tickers],
            filtered_holdings,
            config.initial_capital,
            config.rebalance_frequency,
            config.transaction_cost_pct
        )

        # Handle empty portfolio value (edge case)
        if portfolio_value.empty or portfolio_value.isna().all():
            st.error("Could not calculate portfolio value - check ticker data availability")
            return None

        # --- DEBUG: Uncomment these lines to verify no artificial cliff ---
        # st.write("Backtest start:", portfolio_value.index.min())
        # st.write("Backtest end:", portfolio_value.index.max())
        # st.write("First 5 NAV points:", portfolio_value.head())
        # st.write("Biggest daily drop (%):", f"{portfolio_value.pct_change().min() * 100:.2f}%")
        # -------------------------------------------------------------------

        # Calculate benchmark value (aligned to portfolio dates)
        if config.benchmark in prices.columns:
            benchmark_prices = prices[config.benchmark].ffill()
            # Align benchmark to portfolio dates
            benchmark_prices = benchmark_prices.reindex(portfolio_value.index).ffill()
            first_valid_bench = benchmark_prices.first_valid_index()
            if first_valid_bench is not None:
                benchmark_value = (benchmark_prices / benchmark_prices.loc[first_valid_bench]) * config.initial_capital
            else:
                benchmark_value = pd.Series(index=portfolio_value.index, data=config.initial_capital)
        else:
            st.warning(f"Benchmark {config.benchmark} not available")
            benchmark_value = pd.Series(index=portfolio_value.index, data=config.initial_capital)

        # Calculate returns (let NaN remain on first day - metrics will handle it)
        portfolio_returns = portfolio_value.pct_change()
        benchmark_returns = benchmark_value.pct_change()

        # Calculate metrics (use configurable risk-free rate)
        metrics = self.calculate_metrics(
            portfolio_returns,
            benchmark_returns,
            risk_free_rate=config.risk_free_rate
        )

        # Build result
        result = BacktestResult(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            portfolio_value=portfolio_value,
            benchmark_value=benchmark_value,
            holdings_over_time=holdings_df,
            total_return_pct=metrics['total_return'] * 100,
            annualized_return_pct=metrics['annualized_return'] * 100,
            volatility_pct=metrics['volatility'] * 100,
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown_pct=metrics['max_drawdown'] * 100,
            benchmark_total_return_pct=metrics.get('benchmark_total_return', 0) * 100,
            benchmark_annualized_return_pct=metrics.get('benchmark_annualized_return', 0) * 100,
            benchmark_volatility_pct=metrics.get('benchmark_volatility', 0) * 100,
            benchmark_sharpe_ratio=metrics.get('benchmark_sharpe_ratio', 0),
            benchmark_max_drawdown_pct=metrics.get('benchmark_max_drawdown', 0) * 100,
            alpha=metrics.get('alpha', 0) * 100,
            beta=metrics.get('beta', 0),
            num_rebalances=num_rebalances,
            total_transaction_costs=total_costs,
            dropped_tickers_df=self.last_dropped_tickers
        )

        return result

    def run_backtest(self, config: BacktestConfig) -> Optional[BacktestResult]:
        """
        Run backtest for a saved portfolio (loaded from database).

        Args:
            config: BacktestConfig with portfolio_id pointing to saved portfolio

        Returns:
            BacktestResult or None if failed
        """
        # Load portfolio holdings from database
        holdings = self.load_portfolio_holdings(config.portfolio_id)
        if holdings.empty:
            st.error("No holdings found for this portfolio")
            return None

        return self._run_backtest_core(holdings, config)

    def run_backtest_from_holdings(
        self,
        holdings: pd.DataFrame,
        config: BacktestConfig
    ) -> Optional[BacktestResult]:
        """
        Run backtest from a DataFrame of holdings (e.g., from CSV upload).

        Args:
            holdings: DataFrame with at least 'ticker' and 'weight_pct' columns
            config: BacktestConfig (portfolio_id can be -1 for CSV portfolios)

        Returns:
            BacktestResult or None if failed
        """
        # Validate required columns
        if 'ticker' not in holdings.columns or 'weight_pct' not in holdings.columns:
            st.error("Holdings DataFrame must have 'ticker' and 'weight_pct' columns")
            return None

        return self._run_backtest_core(holdings, config)


def create_performance_chart(result: BacktestResult, config: BacktestConfig) -> go.Figure:
    """Create interactive performance chart."""

    fig = go.Figure()

    # Portfolio line
    fig.add_trace(go.Scatter(
        x=result.portfolio_value.index,
        y=result.portfolio_value.values,
        name=config.portfolio_name,
        line=dict(color='#2E86AB', width=2),
        hovertemplate='%{y:$,.0f}<extra></extra>'
    ))

    # Benchmark line
    fig.add_trace(go.Scatter(
        x=result.benchmark_value.index,
        y=result.benchmark_value.values,
        name=config.benchmark,
        line=dict(color='#A23B72', width=2, dash='dash'),
        hovertemplate='%{y:$,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Portfolio Performance: {config.portfolio_name} vs {config.benchmark}",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def create_drawdown_chart(result: BacktestResult) -> go.Figure:
    """Create drawdown chart."""

    # Calculate drawdowns
    port_cumulative = (1 + result.portfolio_returns).cumprod()
    port_running_max = port_cumulative.expanding().max()
    port_drawdown = (port_cumulative - port_running_max) / port_running_max * 100

    bench_cumulative = (1 + result.benchmark_returns).cumprod()
    bench_running_max = bench_cumulative.expanding().max()
    bench_drawdown = (bench_cumulative - bench_running_max) / bench_running_max * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=port_drawdown.index,
        y=port_drawdown.values,
        name='Portfolio DD',
        fill='tozeroy',
        line=dict(color='#2E86AB'),
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=bench_drawdown.index,
        y=bench_drawdown.values,
        name='Benchmark DD',
        line=dict(color='#A23B72', dash='dash'),
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title="Drawdown Analysis",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig


def create_returns_distribution(result: BacktestResult) -> go.Figure:
    """Create returns distribution chart."""

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=result.portfolio_returns * 100,
        name='Portfolio',
        opacity=0.7,
        nbinsx=50,
        marker_color='#2E86AB'
    ))

    fig.add_trace(go.Histogram(
        x=result.benchmark_returns * 100,
        name='Benchmark',
        opacity=0.7,
        nbinsx=50,
        marker_color='#A23B72'
    ))

    fig.update_layout(
        title="Daily Returns Distribution",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        barmode='overlay',
        template='plotly_white',
        height=400
    )

    return fig


# Export for use in main app
__all__ = [
    'PortfolioBacktester',
    'BacktestConfig',
    'BacktestResult',
    'create_performance_chart',
    'create_drawdown_chart',
    'create_returns_distribution'
]