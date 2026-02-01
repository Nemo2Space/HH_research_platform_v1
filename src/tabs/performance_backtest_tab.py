"""
Performance & Backtesting - Unified Analytics Page

Combines:
1. Signal Performance Tracking - How accurate are our signals?
2. Strategy Backtesting - Test strategies on historical data
3. Signal Combination Analysis - Which signal combos work best?
4. Trade Attribution - Link trades to signals

Location: dashboard/tabs/performance_backtest_tab.py or src/tabs/performance_backtest_tab.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
import sys
import os

# Add project root to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Try to import logger, fall back to basic logging
try:
    from src.utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Import components - try multiple paths
try:
    from src.analytics.backtesting.engine import BacktestEngine, BacktestResult
    from src.analytics.backtesting.strategies import (
        STRATEGIES, HOLDING_PERIODS, BENCHMARKS,
        get_strategy, get_all_strategies
    )
    BACKTEST_AVAILABLE = True
except ImportError:
    try:
        from src.backtest.engine import BacktestEngine, BacktestResult
        from src.backtest.strategies import (
            STRATEGIES, HOLDING_PERIODS, BENCHMARKS,
            get_strategy, get_all_strategies
        )
        BACKTEST_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Backtesting not available: {e}")
        BACKTEST_AVAILABLE = False

try:
    from src.analytics.signal_performance import SignalPerformanceTracker
    SIGNAL_PERF_AVAILABLE = True
except ImportError:
    try:
        from src.ai.signal_performance import SignalPerformanceTracker
        SIGNAL_PERF_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Signal performance not available: {e}")
        SIGNAL_PERF_AVAILABLE = False


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_performance_backtest_tab():
    """Render the unified Performance & Backtesting page."""

    st.title("📊 Performance & Backtesting")
    st.caption("Analyze signal accuracy, backtest strategies, and find what works best")

    # Create sub-tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Signal Performance",
        "🔬 Strategy Backtester",
        "🎯 Signal Combinations",
        "📋 Trade Journal"
    ])

    # =========================================================================
    # TAB 1: SIGNAL PERFORMANCE
    # =========================================================================
    with tab1:
        _render_signal_performance_section()

    # =========================================================================
    # TAB 2: STRATEGY BACKTESTER
    # =========================================================================
    with tab2:
        _render_backtester_section()

    # =========================================================================
    # TAB 3: SIGNAL COMBINATIONS
    # =========================================================================
    with tab3:
        _render_signal_combinations_section()

    # =========================================================================
    # TAB 4: TRADE JOURNAL
    # =========================================================================
    with tab4:
        _render_trade_journal_section()


# =============================================================================
# TAB 1: SIGNAL PERFORMANCE
# =============================================================================

def _render_signal_performance_section():
    """Render signal performance tracking section."""

    st.subheader("📈 Signal Performance Tracker")
    st.caption("How accurate are our trading signals over time?")

    if not SIGNAL_PERF_AVAILABLE:
        st.warning("Signal Performance Tracker not available. Check imports.")
        return

    # Controls
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        days_back = st.selectbox(
            "Lookback Period",
            options=[30, 60, 90, 180],
            index=2,
            format_func=lambda x: f"{x} Days",
            key="perf_days_back"
        )

    with col2:
        if st.button("🔄 Refresh Data", key="refresh_perf"):
            st.cache_data.clear()
            st.rerun()

    # Load data
    with st.spinner("Loading signal performance data..."):
        tracker = SignalPerformanceTracker()
        summary = tracker.get_performance_summary(days_back=days_back)

    if summary['total_signals'] == 0:
        st.info("No signals found in the selected period. Run analysis first to generate signals.")
        return

    # =========================================================================
    # OVERVIEW METRICS
    # =========================================================================
    st.markdown("---")
    st.markdown("### 📊 Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Signals",
            f"{summary['total_signals']:,}",
            help="Total signals generated in period"
        )

    with col2:
        st.metric(
            "With Returns",
            f"{summary['signals_with_returns']:,}",
            help="Signals with calculated returns (need 5+ days)"
        )

    with col3:
        win_rate = summary['overall_win_rate']
        win_color = "🟢" if win_rate >= 55 else "🟡" if win_rate >= 50 else "🔴"
        st.metric(
            "Win Rate (10d)",
            f"{win_color} {win_rate:.1f}%",
            help="% of BUY signals that were profitable after 10 days"
        )

    with col4:
        avg_ret = summary['overall_avg_return']
        ret_color = "🟢" if avg_ret > 0 else "🔴"
        st.metric(
            "Avg Return (10d)",
            f"{ret_color} {avg_ret:+.2f}%",
            help="Average return after 10 days"
        )

    with col5:
        if summary.get('date_range'):
            st.metric(
                "Period",
                f"{summary['date_range']['start'][:5]} - {summary['date_range']['end'][:5]}",
                help="Date range of signals"
            )

    # =========================================================================
    # PERFORMANCE BY SIGNAL TYPE
    # =========================================================================
    st.markdown("---")
    st.markdown("### 📋 Performance by Signal Type")

    by_signal = summary.get('by_signal_type', {})

    if by_signal:
        # Create table data
        table_data = []
        for signal_type, perf in by_signal.items():
            # Determine if this is a good performer
            is_good = perf.win_rate_10d >= 55 and perf.avg_return_10d > 0

            table_data.append({
                'Signal': signal_type,
                'Count': perf.total_signals,
                'Win% 5d': perf.win_rate_5d,
                'Win% 10d': perf.win_rate_10d,
                'Win% 30d': perf.win_rate_30d,
                'Avg Ret 5d': perf.avg_return_5d,
                'Avg Ret 10d': perf.avg_return_10d,
                'Avg Ret 30d': perf.avg_return_30d,
                'Best': f"{perf.best_ticker} ({perf.best_return:+.1f}%)",
                'Worst': f"{perf.worst_ticker} ({perf.worst_return:+.1f}%)",
                'Status': '✅ Edge' if is_good else '⚠️ Review'
            })

        df = pd.DataFrame(table_data)

        # Sort by win rate 10d
        df = df.sort_values('Win% 10d', ascending=False)

        # Column config for formatting while keeping sortable
        signal_column_config = {
            'Signal': st.column_config.TextColumn('Signal'),
            'Count': st.column_config.NumberColumn('Count', format='%d'),
            'Win% 5d': st.column_config.NumberColumn('Win% 5d', format='%.1f%%'),
            'Win% 10d': st.column_config.NumberColumn('Win% 10d', format='%.1f%%'),
            'Win% 30d': st.column_config.NumberColumn('Win% 30d', format='%.1f%%'),
            'Avg Ret 5d': st.column_config.NumberColumn('Avg Ret 5d', format='%+.2f%%'),
            'Avg Ret 10d': st.column_config.NumberColumn('Avg Ret 10d', format='%+.2f%%'),
            'Avg Ret 30d': st.column_config.NumberColumn('Avg Ret 30d', format='%+.2f%%'),
        }

        st.dataframe(df, use_container_width=True, hide_index=True, column_config=signal_column_config)

        # Visualization
        st.markdown("#### Win Rate Comparison")

        fig = go.Figure()

        signals = [d['Signal'] for d in table_data]
        win_5d = [d['Win% 5d'] for d in table_data]
        win_10d = [d['Win% 10d'] for d in table_data]
        win_30d = [d['Win% 30d'] for d in table_data]

        fig.add_trace(go.Bar(name='5 Day', x=signals, y=win_5d, marker_color='#636EFA'))
        fig.add_trace(go.Bar(name='10 Day', x=signals, y=win_10d, marker_color='#00CC96'))
        fig.add_trace(go.Bar(name='30 Day', x=signals, y=win_30d, marker_color='#EF553B'))

        # Add 50% reference line
        fig.add_hline(y=50, line_dash="dash", line_color="gray",
                      annotation_text="50% (No Edge)")

        fig.update_layout(
            barmode='group',
            title='Win Rate by Signal Type and Holding Period',
            yaxis_title='Win Rate %',
            xaxis_title='Signal Type',
            height=400,
            template='plotly_dark'
        )

        st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # RECENT SIGNALS TABLE
    # =========================================================================
    st.markdown("---")
    st.markdown("### 📜 Recent Signals & Outcomes")

    try:
        recent_df = tracker.get_recent_signals_performance(limit=50)

        if not recent_df.empty:
            # Format for display
            display_df = recent_df.copy()

            # Format date
            if 'signal_date' in display_df.columns:
                display_df['signal_date'] = pd.to_datetime(display_df['signal_date']).dt.strftime('%m-%d')

            # Format returns
            for col in ['return_2d', 'return_5d', 'return_10d', 'return_30d']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(
                        lambda x: f"{x:+.1f}%" if pd.notna(x) else "-"
                    )

            st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)
        else:
            st.info("No recent signals with return data.")

    except Exception as e:
        st.warning(f"Could not load recent signals: {e}")

    # =========================================================================
    # KEY INSIGHTS
    # =========================================================================
    st.markdown("---")
    st.markdown("### 💡 Key Insights")

    if by_signal:
        # Find best and worst performers
        best_signal = max(by_signal.items(), key=lambda x: x[1].win_rate_10d)
        worst_signal = min(by_signal.items(), key=lambda x: x[1].win_rate_10d)

        col1, col2 = st.columns(2)

        with col1:
            st.success(f"""
            **🏆 Best Performing Signal: {best_signal[0]}**
            - Win Rate (10d): {best_signal[1].win_rate_10d:.1f}%
            - Avg Return: {best_signal[1].avg_return_10d:+.2f}%
            - Best Trade: {best_signal[1].best_ticker} ({best_signal[1].best_return:+.1f}%)
            """)

        with col2:
            if worst_signal[1].win_rate_10d < 50:
                st.error(f"""
                **⚠️ Underperforming Signal: {worst_signal[0]}**
                - Win Rate (10d): {worst_signal[1].win_rate_10d:.1f}%
                - Avg Return: {worst_signal[1].avg_return_10d:+.2f}%
                - Consider reducing reliance on this signal
                """)
            else:
                st.info(f"""
                **📊 Lowest Performer: {worst_signal[0]}**
                - Win Rate (10d): {worst_signal[1].win_rate_10d:.1f}%
                - Still above 50%, showing some edge
                """)


# =============================================================================
# TAB 2: STRATEGY BACKTESTER
# =============================================================================

def _render_backtester_section():
    """Render strategy backtesting section."""

    st.subheader("🔬 Strategy Backtester")
    st.caption("Test trading strategies on historical data")

    if not BACKTEST_AVAILABLE:
        st.warning("Backtesting engine not available. Check imports.")
        _show_backtest_setup_instructions()
        return

    # =========================================================================
    # STRATEGY SELECTION
    # =========================================================================
    st.markdown("### ⚙️ Configure Backtest")

    col1, col2 = st.columns(2)

    with col1:
        # Strategy selector
        strategy_options = {s.name: s.display_name for s in get_all_strategies()}
        selected_strategy = st.selectbox(
            "Strategy",
            options=list(strategy_options.keys()),
            format_func=lambda x: strategy_options[x],
            key="bt_strategy"
        )

        # Show strategy description
        strategy_config = get_strategy(selected_strategy)
        st.caption(f"📝 {strategy_config.description}")

        # Holding period
        holding_period = st.selectbox(
            "Holding Period",
            options=[h['days'] for h in HOLDING_PERIODS],
            format_func=lambda x: next(h['label'] for h in HOLDING_PERIODS if h['days'] == x),
            index=1,  # Default to 5 days
            key="bt_holding"
        )

    with col2:
        # Date range
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input(
                "Start Date",
                value=date.today() - timedelta(days=180),
                key="bt_start"
            )
        with col_end:
            end_date = st.date_input(
                "End Date",
                value=date.today() - timedelta(days=5),
                key="bt_end"
            )

        # Benchmark
        benchmark = st.selectbox(
            "Benchmark",
            options=[b['ticker'] for b in BENCHMARKS],
            format_func=lambda x: next(b['name'] for b in BENCHMARKS if b['ticker'] == x),
            key="bt_benchmark"
        )

    # Strategy parameters (if customizable)
    if strategy_config.param_options:
        st.markdown("#### 🎛️ Strategy Parameters")
        params = {}

        cols = st.columns(len(strategy_config.param_options))
        for i, (param_name, options) in enumerate(strategy_config.param_options.items()):
            with cols[i]:
                default_val = strategy_config.default_params.get(param_name)
                if isinstance(options[0], list):
                    # List of lists - use selectbox
                    params[param_name] = st.selectbox(
                        param_name.replace('_', ' ').title(),
                        options=options,
                        index=options.index(default_val) if default_val in options else 0,
                        key=f"bt_param_{param_name}"
                    )
                elif isinstance(options[0], bool):
                    params[param_name] = st.checkbox(
                        param_name.replace('_', ' ').title(),
                        value=default_val,
                        key=f"bt_param_{param_name}"
                    )
                else:
                    params[param_name] = st.selectbox(
                        param_name.replace('_', ' ').title(),
                        options=options,
                        index=options.index(default_val) if default_val in options else 0,
                        key=f"bt_param_{param_name}"
                    )
    else:
        params = strategy_config.default_params

    # =========================================================================
    # RUN BACKTEST
    # =========================================================================
    st.markdown("---")

    col_run, col_compare = st.columns(2)

    with col_run:
        run_single = st.button("🚀 Run Backtest", type="primary", key="run_backtest")

    with col_compare:
        run_compare = st.button("📊 Compare All Strategies", key="compare_all")

    if run_single:
        with st.spinner("Running backtest... This may take a minute."):
            try:
                engine = BacktestEngine()
                result = engine.run_backtest(
                    strategy=strategy_config.strategy_type,  # Use strategy_type, not name
                    start_date=str(start_date),
                    end_date=str(end_date),
                    holding_period=holding_period,
                    benchmark=benchmark,
                    params=params
                )

                # Override strategy name for display
                result.strategy_name = selected_strategy

                # Store result in session state
                st.session_state['backtest_result'] = result
                st.success("✅ Backtest complete!")

                # Auto-save for AI learning
                try:
                    from src.backtest.backtest_learning import auto_save_backtest_result
                    if auto_save_backtest_result(result):
                        st.caption("📚 Saved to AI learning database")
                except Exception:
                    pass  # Learning is optional

            except Exception as e:
                st.error(f"Backtest failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # =========================================================================
    # COMPARE ALL STRATEGIES
    # =========================================================================
    if run_compare:
        with st.spinner("Running all strategies... This may take a few minutes."):
            comparison_results = []
            engine = BacktestEngine()

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, (strat_name, strat_config) in enumerate(STRATEGIES.items()):
                try:
                    status_text.text(f"Running {strat_config.display_name}...")

                    result = engine.run_backtest(
                        strategy=strat_config.strategy_type,
                        start_date=str(start_date),
                        end_date=str(end_date),
                        holding_period=holding_period,
                        benchmark=benchmark,
                        params=strat_config.default_params
                    )
                    result.strategy_name = strat_name

                    comparison_results.append({
                        'Strategy': strat_config.display_name,
                        'Type': strat_config.strategy_type,
                        'Trades': result.total_trades,
                        'Win Rate': result.win_rate,
                        'Avg Return': result.avg_return,
                        'Total Return': getattr(result, 'total_return', result.avg_return * result.total_trades),
                        'Sharpe': result.sharpe_ratio,
                        'Alpha': result.alpha,
                        'Max DD': getattr(result, 'max_drawdown', 0),
                        '_result': result  # Keep for later use
                    })

                    # Auto-save for AI learning
                    try:
                        from src.backtest.backtest_learning import auto_save_backtest_result
                        auto_save_backtest_result(result)
                    except Exception:
                        pass

                except Exception as e:
                    logger.warning(f"Strategy {strat_name} failed: {e}")

                progress_bar.progress((i + 1) / len(STRATEGIES))

            progress_bar.empty()
            status_text.empty()

            if comparison_results:
                st.session_state['comparison_results'] = comparison_results
                st.success(f"✅ Compared {len(comparison_results)} strategies!")

    # Display comparison results
    if 'comparison_results' in st.session_state and st.session_state['comparison_results']:
        _render_strategy_comparison(st.session_state['comparison_results'])

    # =========================================================================
    # DISPLAY RESULTS
    # =========================================================================
    if 'backtest_result' in st.session_state:
        result = st.session_state['backtest_result']
        _render_backtest_results(result)


def _render_strategy_comparison(comparison_results: list):
    """Render strategy comparison table with best strategy highlighting."""
    import pandas as pd

    st.markdown("---")
    st.markdown("### 📊 Strategy Comparison")

    # Create DataFrame for display
    df = pd.DataFrame(comparison_results)
    display_df = df.drop(columns=['_result'], errors='ignore').copy()

    # Convert Win Rate from decimal to percentage value (e.g., 0.55 -> 55.0)
    # Keep as numeric for proper sorting
    display_df['Win Rate'] = display_df['Win Rate'] * 100  # Now 55.0 instead of 0.55

    # Use column_config for proper formatting while keeping numeric values sortable
    column_config = {
        'Strategy': st.column_config.TextColumn('Strategy', width='medium'),
        'Type': st.column_config.TextColumn('Type', width='small'),
        'Trades': st.column_config.NumberColumn('Trades', format='%d'),
        'Win Rate': st.column_config.NumberColumn('↓ Win Rate', format='%.1f%%'),
        'Avg Return': st.column_config.NumberColumn('Avg Return', format='%+.2f%%'),
        'Total Return': st.column_config.NumberColumn('↓ Total Return', format='%+.1f%%'),
        'Sharpe': st.column_config.NumberColumn('Sharpe', format='%.2f'),
        'Alpha': st.column_config.NumberColumn('Alpha', format='%+.2f%%'),
        'Max DD': st.column_config.NumberColumn('Max DD', format='%.1f%%'),
    }

    # Show table with sortable numeric columns
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config
    )

    # Find best strategies by different metrics (use original df with decimals)
    st.markdown("### 🏆 Best Strategies")

    col1, col2, col3 = st.columns(3)

    # Best by Sharpe
    best_sharpe_idx = df['Sharpe'].idxmax()
    best_sharpe = df.loc[best_sharpe_idx]
    with col1:
        st.markdown("**Best Risk-Adjusted (Sharpe)**")
        st.success(f"🥇 {best_sharpe['Strategy']}")
        st.caption(f"Sharpe: {best_sharpe['Sharpe']:.2f} | Win: {best_sharpe['Win Rate']:.1%}")

    # Best by Win Rate
    best_wr_idx = df['Win Rate'].idxmax()
    best_wr = df.loc[best_wr_idx]
    with col2:
        st.markdown("**Best Win Rate**")
        st.success(f"🥇 {best_wr['Strategy']}")
        st.caption(f"Win: {best_wr['Win Rate']:.1%} | Trades: {best_wr['Trades']}")

    # Best by Total Return
    best_ret_idx = df['Total Return'].idxmax()
    best_ret = df.loc[best_ret_idx]
    with col3:
        st.markdown("**Best Total Return**")
        st.success(f"🥇 {best_ret['Strategy']}")
        st.caption(f"Return: {best_ret['Total Return']:+.1f}% | Trades: {best_ret['Trades']}")

    # AI vs Traditional comparison
    ai_strategies = df[df['Type'] == 'ai_probability']
    trad_strategies = df[df['Type'] != 'ai_probability']

    if not ai_strategies.empty and not trad_strategies.empty:
        st.markdown("### 🤖 AI vs Traditional Strategies")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**AI Strategies**")
            ai_avg_sharpe = ai_strategies['Sharpe'].mean()
            ai_avg_wr = ai_strategies['Win Rate'].mean()
            ai_trades = ai_strategies['Trades'].sum()
            st.metric("Avg Sharpe", f"{ai_avg_sharpe:.2f}")
            st.metric("Avg Win Rate", f"{ai_avg_wr:.1%}")
            st.metric("Total Trades", ai_trades)

        with col2:
            st.markdown("**Traditional Strategies**")
            trad_avg_sharpe = trad_strategies['Sharpe'].mean()
            trad_avg_wr = trad_strategies['Win Rate'].mean()
            trad_trades = trad_strategies['Trades'].sum()
            sharpe_delta = ai_avg_sharpe - trad_avg_sharpe
            wr_delta = ai_avg_wr - trad_avg_wr
            st.metric("Avg Sharpe", f"{trad_avg_sharpe:.2f}", delta=f"{sharpe_delta:+.2f} vs AI")
            st.metric("Avg Win Rate", f"{trad_avg_wr:.1%}", delta=f"{wr_delta:+.1%} vs AI")
            st.metric("Total Trades", trad_trades)

    # Recommendations
    st.markdown("### 💡 Recommendations")

    recommendations = []

    # High Sharpe strategies
    high_sharpe = df[df['Sharpe'] >= 1.0]
    if not high_sharpe.empty:
        strategies_list = ', '.join(high_sharpe['Strategy'].tolist())
        recommendations.append(f"✅ **High quality strategies (Sharpe ≥ 1.0):** {strategies_list}")

    # Good win rate with volume
    good_wr = df[(df['Win Rate'] >= 0.55) & (df['Trades'] >= 50)]
    if not good_wr.empty:
        strategies_list = ', '.join(good_wr['Strategy'].tolist())
        recommendations.append(f"✅ **Reliable strategies (55%+ win rate, 50+ trades):** {strategies_list}")

    # Positive alpha
    positive_alpha = df[df['Alpha'] > 0]
    if not positive_alpha.empty:
        strategies_list = ', '.join(positive_alpha['Strategy'].tolist())
        recommendations.append(f"✅ **Outperforming benchmark:** {strategies_list}")

    # Warning for low trade count
    low_trades = df[df['Trades'] < 20]
    if not low_trades.empty:
        strategies_list = ', '.join(low_trades['Strategy'].tolist())
        recommendations.append(f"⚠️ **Insufficient data (<20 trades):** {strategies_list}")

    for rec in recommendations:
        st.markdown(rec)

    # Option to use best strategy
    st.markdown("---")
    if st.button("📥 Load Best Strategy (by Sharpe)", key="load_best"):
        st.session_state['backtest_result'] = best_sharpe['_result']
        st.success(f"Loaded {best_sharpe['Strategy']} results!")
        st.rerun()


def _render_backtest_results(result: 'BacktestResult'):
    """Render backtest results."""

    st.markdown("---")
    st.markdown("### 📊 Backtest Results")

    # =========================================================================
    # KEY METRICS
    # =========================================================================
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Trades", result.total_trades)

    with col2:
        win_color = "🟢" if result.win_rate >= 0.55 else "🟡" if result.win_rate >= 0.5 else "🔴"
        st.metric("Win Rate", f"{win_color} {result.win_rate:.1%}")

    with col3:
        ret_color = "🟢" if result.avg_return > 0 else "🔴"
        st.metric("Avg Return", f"{ret_color} {result.avg_return:.2f}%")

    with col4:
        sharpe_color = "🟢" if result.sharpe_ratio >= 1 else "🟡" if result.sharpe_ratio >= 0.5 else "🔴"
        st.metric("Sharpe Ratio", f"{sharpe_color} {result.sharpe_ratio:.2f}")

    with col5:
        alpha_color = "🟢" if result.alpha > 0 else "🔴"
        st.metric("Alpha vs Benchmark", f"{alpha_color} {result.alpha:.2f}%")

    # Second row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Return", f"{result.total_return:.2f}%")

    with col2:
        st.metric("Sortino Ratio", f"{result.sortino_ratio:.2f}")

    with col3:
        st.metric("Max Drawdown", f"{result.max_drawdown:.2f}%")

    with col4:
        st.metric("Best Trade", f"{result.best_trade:+.2f}%")

    with col5:
        st.metric("Worst Trade", f"{result.worst_trade:+.2f}%")

    # =========================================================================
    # EQUITY CURVE
    # =========================================================================
    st.markdown("---")
    st.markdown("### 📈 Equity Curve")

    if not result.equity_curve.empty:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=result.equity_curve['date'],
            y=result.equity_curve['equity'],
            mode='lines',
            name='Strategy',
            line=dict(color='#00CC96', width=2)
        ))

        # Add benchmark line (normalized to 100)
        fig.add_hline(y=100, line_dash="dash", line_color="gray",
                      annotation_text="Starting Value")

        fig.update_layout(
            title=f'{result.strategy_name} - Equity Curve',
            xaxis_title='Date',
            yaxis_title='Equity (Started at 100)',
            height=400,
            template='plotly_dark'
        )

        st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # RETURNS BY SIGNAL
    # =========================================================================
    st.markdown("---")
    st.markdown("### 📋 Returns by Signal Type")

    if result.returns_by_signal:
        signal_data = []
        for signal_type, stats in result.returns_by_signal.items():
            signal_data.append({
                'Signal': signal_type,
                'Trades': stats['count'],
                'Win Rate': f"{stats['win_rate']:.0%}",
                'Avg Return': f"{stats['avg_return']:+.2f}%",
                'Total Return': f"{stats['total_return']:+.2f}%",
                'Best': f"{stats['best']:+.2f}%",
                'Worst': f"{stats['worst']:+.2f}%"
            })

        st.dataframe(pd.DataFrame(signal_data), use_container_width=True, hide_index=True)

    # =========================================================================
    # TRADE LIST
    # =========================================================================
    with st.expander("📜 Individual Trades", expanded=False):
        if result.trades:
            # Sort by entry date descending (most recent first)
            sorted_trades = sorted(result.trades, key=lambda t: t.entry_date, reverse=True)

            trade_data = []
            for t in sorted_trades[:100]:  # Limit to 100 most recent
                trade_data.append({
                    'Ticker': t.ticker,
                    'Entry': str(t.entry_date),
                    '↓ Exit': str(t.exit_date),
                    'Signal': t.signal_type,
                    'Direction': t.direction,
                    'Return': t.return_pct,
                    'Result': '✅' if t.is_winner else '❌'
                })

            trade_df = pd.DataFrame(trade_data)

            # Column config for sortable Return column
            trade_col_config = {
                'Return': st.column_config.NumberColumn('Return', format='%+.2f%%'),
            }

            st.dataframe(trade_df, use_container_width=True, hide_index=True, height=400, column_config=trade_col_config)

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Save for AI Learning", key="save_backtest"):
            try:
                from src.backtest.backtest_learning import auto_save_backtest_result
                if auto_save_backtest_result(result):
                    st.success("Results saved for AI learning!")
                else:
                    st.warning("Could not save results")
            except Exception as e:
                st.error(f"Failed to save: {e}")

    with col2:
        # Download as CSV
        if result.trades:
            trade_df = pd.DataFrame([{
                'ticker': t.ticker,
                'entry_date': t.entry_date,
                'exit_date': t.exit_date,
                'signal_type': t.signal_type,
                'return_pct': t.return_pct
            } for t in result.trades])

            csv = trade_df.to_csv(index=False)
            st.download_button(
                "📥 Download Trades CSV",
                csv,
                f"backtest_{result.strategy_name}_{result.start_date}.csv",
                "text/csv",
                key="download_backtest"
            )


def _show_backtest_setup_instructions():
    """Show setup instructions if backtesting not available."""
    st.markdown("""
    ### Setup Required
    
    To enable backtesting, ensure these files exist:
    
    ```
    src/analytics/backtesting/
    ├── __init__.py
    ├── engine.py        # BacktestEngine class
    └── strategies.py    # Strategy definitions
    ```
    
    Also ensure you have historical data in the `historical_scores` table.
    """)


# =============================================================================
# TAB 3: SIGNAL COMBINATIONS
# =============================================================================

def _render_signal_combinations_section():
    """Render signal combination analysis section."""

    st.subheader("🎯 Signal Combination Analysis")
    st.caption("Discover which combinations of signals work best together")

    # Load data
    try:
        from src.db.connection import get_engine
        engine = get_engine()

        # Get historical data with all score components
        query = """
            SELECT 
                ticker,
                date,
                sentiment_score,
                fundamental_score,
                technical_score,
                options_flow_score,
                short_squeeze_score,
                total_score,
                gap_score
            FROM screener_scores
            WHERE date >= CURRENT_DATE - INTERVAL '90 days'
              AND date <= CURRENT_DATE - INTERVAL '5 days'
            ORDER BY date DESC
        """

        scores_df = pd.read_sql(query, engine)

        if scores_df.empty:
            st.info("Not enough historical data for combination analysis. Need at least 5 days of data.")
            return

    except Exception as e:
        st.warning(f"Could not load data: {e}")
        _render_combination_analysis_placeholder()
        return

    st.markdown("---")

    # =========================================================================
    # DEFINE COMBINATIONS
    # =========================================================================
    st.markdown("### 🔧 Define Signal Combinations")

    col1, col2, col3 = st.columns(3)

    with col1:
        sentiment_threshold = st.slider(
            "Min Sentiment Score",
            min_value=40, max_value=90, value=65,
            key="combo_sentiment"
        )

    with col2:
        fundamental_threshold = st.slider(
            "Min Fundamental Score",
            min_value=40, max_value=90, value=60,
            key="combo_fundamental"
        )

    with col3:
        options_threshold = st.slider(
            "Min Options Flow Score",
            min_value=40, max_value=90, value=55,
            key="combo_options"
        )

    # =========================================================================
    # ANALYZE COMBINATIONS
    # =========================================================================
    if st.button("🔍 Analyze Combinations", key="analyze_combos"):
        with st.spinner("Analyzing signal combinations..."):
            results = _analyze_signal_combinations(
                scores_df,
                sentiment_threshold,
                fundamental_threshold,
                options_threshold
            )

            st.session_state['combo_results'] = results

    # Display results
    if 'combo_results' in st.session_state:
        _display_combination_results(st.session_state['combo_results'])


def _analyze_signal_combinations(df: pd.DataFrame, sent_th: int, fund_th: int, opt_th: int) -> Dict:
    """Analyze different signal combinations."""

    # We need price data to calculate returns
    # For now, use total_score as a proxy for "good outcome"
    # In production, this should use actual returns

    results = {}

    # Define combinations to test
    combinations = [
        {
            'name': 'High Sentiment Only',
            'filter': lambda d: d['sentiment_score'] >= sent_th,
            'description': f'Sentiment >= {sent_th}'
        },
        {
            'name': 'High Fundamental Only',
            'filter': lambda d: d['fundamental_score'] >= fund_th,
            'description': f'Fundamental >= {fund_th}'
        },
        {
            'name': 'High Options Only',
            'filter': lambda d: d['options_flow_score'] >= opt_th,
            'description': f'Options >= {opt_th}'
        },
        {
            'name': 'Sentiment + Fundamental',
            'filter': lambda d: (d['sentiment_score'] >= sent_th) & (d['fundamental_score'] >= fund_th),
            'description': f'Sentiment >= {sent_th} AND Fundamental >= {fund_th}'
        },
        {
            'name': 'Sentiment + Options',
            'filter': lambda d: (d['sentiment_score'] >= sent_th) & (d['options_flow_score'] >= opt_th),
            'description': f'Sentiment >= {sent_th} AND Options >= {opt_th}'
        },
        {
            'name': 'Fundamental + Options',
            'filter': lambda d: (d['fundamental_score'] >= fund_th) & (d['options_flow_score'] >= opt_th),
            'description': f'Fundamental >= {fund_th} AND Options >= {opt_th}'
        },
        {
            'name': 'Triple Screen (All 3)',
            'filter': lambda d: (d['sentiment_score'] >= sent_th) &
                               (d['fundamental_score'] >= fund_th) &
                               (d['options_flow_score'] >= opt_th),
            'description': f'All three signals >= thresholds'
        },
        {
            'name': 'High Total Score',
            'filter': lambda d: d['total_score'] >= 65,
            'description': 'Total Score >= 65'
        },
    ]

    for combo in combinations:
        try:
            filtered = df[combo['filter'](df)]

            if len(filtered) > 0:
                # Calculate metrics
                # Using total_score change as proxy for performance
                avg_total = filtered['total_score'].mean()
                count = len(filtered)

                # High score rate (>= 60 total score)
                high_score_rate = (filtered['total_score'] >= 60).mean() * 100

                results[combo['name']] = {
                    'description': combo['description'],
                    'count': count,
                    'avg_total_score': avg_total,
                    'high_score_rate': high_score_rate,
                    'unique_tickers': filtered['ticker'].nunique()
                }
            else:
                results[combo['name']] = {
                    'description': combo['description'],
                    'count': 0,
                    'avg_total_score': 0,
                    'high_score_rate': 0,
                    'unique_tickers': 0
                }

        except Exception as e:
            logger.warning(f"Error analyzing {combo['name']}: {e}")

    return results


def _display_combination_results(results: Dict):
    """Display combination analysis results."""

    st.markdown("---")
    st.markdown("### 📊 Combination Results")

    # Create table
    table_data = []
    for name, data in results.items():
        table_data.append({
            'Combination': name,
            'Signals': data['count'],
            'Tickers': data['unique_tickers'],
            'Avg Score': f"{data['avg_total_score']:.1f}",
            'High Score %': f"{data['high_score_rate']:.1f}%",
            'Quality': '🟢 High' if data['high_score_rate'] >= 70 else '🟡 Medium' if data['high_score_rate'] >= 50 else '🔴 Low'
        })

    df = pd.DataFrame(table_data)
    df = df.sort_values('High Score %', ascending=False, key=lambda x: x.str.replace('%', '').astype(float))

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Visualization
    st.markdown("#### High Score Rate by Combination")

    fig = go.Figure()

    names = [d['Combination'] for d in table_data]
    rates = [float(d['High Score %'].replace('%', '')) for d in table_data]
    counts = [d['Signals'] for d in table_data]

    # Sort by rate
    sorted_data = sorted(zip(names, rates, counts), key=lambda x: x[1], reverse=True)
    names, rates, counts = zip(*sorted_data)

    colors = ['#00CC96' if r >= 70 else '#FECB52' if r >= 50 else '#EF553B' for r in rates]

    fig.add_trace(go.Bar(
        x=list(names),
        y=list(rates),
        marker_color=colors,
        text=[f"{r:.0f}% ({c})" for r, c in zip(rates, counts)],
        textposition='outside'
    ))

    fig.add_hline(y=50, line_dash="dash", line_color="gray",
                  annotation_text="50% (No Edge)")

    fig.update_layout(
        title='Signal Combination Quality',
        yaxis_title='High Score Rate %',
        xaxis_title='Combination',
        height=400,
        template='plotly_dark'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Key insight
    best_combo = max(results.items(), key=lambda x: x[1]['high_score_rate'])
    if best_combo[1]['high_score_rate'] >= 60:
        st.success(f"""
        **🏆 Best Combination: {best_combo[0]}**
        
        - {best_combo[1]['description']}
        - High Score Rate: {best_combo[1]['high_score_rate']:.1f}%
        - Based on {best_combo[1]['count']} signals
        
        **Recommendation:** Focus on signals that meet this combination's criteria.
        """)


def _render_combination_analysis_placeholder():
    """Placeholder when data not available."""
    st.markdown("""
    ### How Signal Combination Analysis Works
    
    This tool analyzes which combinations of signals produce the best results:
    
    1. **Single Signals**: High Sentiment alone, High Fundamental alone, etc.
    2. **Double Combinations**: Sentiment + Fundamental, Sentiment + Options, etc.
    3. **Triple Screen**: All three signals must be strong
    
    To use this feature:
    - Run analysis on stocks for several days to build historical data
    - Return here to see which combinations have the highest win rates
    """)


# =============================================================================
# TAB 4: TRADE JOURNAL
# =============================================================================

def _render_trade_journal_section():
    """Render trade journal section."""

    st.subheader("📋 Trade Journal")
    st.caption("Track your trades and link them to signals")

    # Check if trade journal table exists
    try:
        from src.db.connection import get_engine
        engine = get_engine()

        # Try to load existing trades
        trades_df = pd.read_sql("""
            SELECT * FROM trade_journal 
            ORDER BY entry_date DESC 
            LIMIT 100
        """, engine)

        has_journal = True

    except Exception as e:
        has_journal = False
        trades_df = pd.DataFrame()

    # =========================================================================
    # ADD NEW TRADE
    # =========================================================================
    st.markdown("### ➕ Log New Trade")

    with st.form("new_trade_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            ticker = st.text_input("Ticker", placeholder="AAPL").upper()
            entry_date = st.date_input("Entry Date", value=date.today())
            entry_price = st.number_input("Entry Price", min_value=0.01, value=100.00, step=0.01)

        with col2:
            direction = st.selectbox("Direction", ["LONG", "SHORT"])
            shares = st.number_input("Shares", min_value=1, value=100)
            exit_date = st.date_input("Exit Date (if closed)", value=None)

        with col3:
            exit_price = st.number_input("Exit Price (if closed)", min_value=0.0, value=0.0, step=0.01)

            # Signal attribution
            entry_signals = st.multiselect(
                "Entry Signals",
                ["High Sentiment", "Committee BUY", "Options Bullish",
                 "Strong Fundamentals", "Technical Breakout", "Earnings Play", "Other"]
            )

        notes = st.text_area("Trade Notes", placeholder="Why did you take this trade?")

        submitted = st.form_submit_button("💾 Save Trade")

        if submitted:
            if not ticker:
                st.error("Please enter a ticker symbol")
            elif not has_journal:
                st.warning("Trade journal table not found. Creating it...")
                _create_trade_journal_table()
                st.rerun()
            else:
                _save_trade(
                    ticker=ticker,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    direction=direction,
                    shares=shares,
                    exit_date=exit_date if exit_price > 0 else None,
                    exit_price=exit_price if exit_price > 0 else None,
                    entry_signals=entry_signals,
                    notes=notes
                )
                st.success(f"✅ Trade logged: {direction} {shares} {ticker} @ ${entry_price:.2f}")
                st.rerun()

    # =========================================================================
    # EXISTING TRADES
    # =========================================================================
    st.markdown("---")
    st.markdown("### 📜 Trade History")

    if not trades_df.empty:
        # Calculate P&L for closed trades
        display_df = trades_df.copy()

        if 'exit_price' in display_df.columns and 'entry_price' in display_df.columns:
            display_df['P&L %'] = display_df.apply(
                lambda r: ((r['exit_price'] - r['entry_price']) / r['entry_price'] * 100
                          * (1 if r.get('direction', 'LONG') == 'LONG' else -1))
                          if pd.notna(r.get('exit_price')) and r.get('exit_price', 0) > 0 else None,
                axis=1
            )

        # Column config for Trade History
        trade_column_config = {
            'P&L %': st.column_config.NumberColumn('P&L %', format='%+.2f%%'),
            'entry_price': st.column_config.NumberColumn('Entry Price', format='$%.2f'),
            'exit_price': st.column_config.NumberColumn('Exit Price', format='$%.2f'),
        }

        st.dataframe(display_df, use_container_width=True, hide_index=True, height=400, column_config=trade_column_config)

        # Summary stats
        if 'exit_price' in trades_df.columns:
            closed = trades_df[trades_df['exit_price'].notna() & (trades_df['exit_price'] > 0)]
            if len(closed) > 0:
                st.markdown("#### 📊 Summary")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Trades", len(trades_df))
                with col2:
                    st.metric("Closed", len(closed))
                with col3:
                    st.metric("Open", len(trades_df) - len(closed))
                with col4:
                    # Win rate
                    wins = sum(1 for _, r in closed.iterrows()
                              if (r['exit_price'] > r['entry_price'] and r.get('direction', 'LONG') == 'LONG')
                              or (r['exit_price'] < r['entry_price'] and r.get('direction') == 'SHORT'))
                    win_rate = wins / len(closed) * 100 if len(closed) > 0 else 0
                    st.metric("Win Rate", f"{win_rate:.1f}%")
    else:
        st.info("No trades logged yet. Use the form above to start tracking your trades.")

        if not has_journal:
            if st.button("🔧 Create Trade Journal Table"):
                _create_trade_journal_table()
                st.success("Trade journal table created!")
                st.rerun()


def _create_trade_journal_table():
    """Create trade journal table if it doesn't exist."""
    try:
        from src.db.connection import get_connection

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trade_journal (
                        id SERIAL PRIMARY KEY,
                        ticker VARCHAR(10) NOT NULL,
                        entry_date DATE NOT NULL,
                        entry_price DECIMAL(12, 4) NOT NULL,
                        direction VARCHAR(10) DEFAULT 'LONG',
                        shares INTEGER DEFAULT 1,
                        exit_date DATE,
                        exit_price DECIMAL(12, 4),
                        entry_signals TEXT[],
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
            conn.commit()

        logger.info("Created trade_journal table")
        return True

    except Exception as e:
        logger.error(f"Failed to create trade_journal table: {e}")
        return False


def _save_trade(ticker: str, entry_date, entry_price: float, direction: str,
                shares: int, exit_date, exit_price, entry_signals: List[str], notes: str):
    """Save a trade to the journal."""
    try:
        from src.db.connection import get_connection

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trade_journal 
                    (ticker, entry_date, entry_price, direction, shares, exit_date, exit_price, entry_signals, notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    ticker, entry_date, entry_price, direction, shares,
                    exit_date, exit_price, entry_signals, notes
                ))
            conn.commit()

        logger.info(f"Saved trade: {direction} {shares} {ticker}")
        return True

    except Exception as e:
        logger.error(f"Failed to save trade: {e}")
        st.error(f"Failed to save trade: {e}")
        return False


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    st.set_page_config(
        page_title="Performance & Backtesting",
        page_icon="📊",
        layout="wide"
    )
    render_performance_backtest_tab()