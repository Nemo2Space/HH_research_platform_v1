"""
Analytics Tab - Dashboard Component (FIXED VERSION)

Integrates:
- Signal Performance Tracker
- Risk Dashboard
- Portfolio Optimizer
- Options Flow
- Short Squeeze (FIXED: handles None values)

FIXES APPLIED:
- Added fmt() helper to safely display None values
- Short squeeze metrics show "N/A" when data unavailable
- Universe scan handles stocks with missing data

Place this file at: dashboard/analytics_tab.py
Import and call render_analytics_tab() in your main app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from typing import Any, Optional


# =============================================================================
# DISPLAY HELPER FUNCTIONS - Handle None values safely
# =============================================================================

def fmt(val: Any, spec: str = ".0f", prefix: str = "", suffix: str = "", na: str = "N/A") -> str:
    """
    Safely format a value for display.

    Args:
        val: Value to format (can be None)
        spec: Format specification (e.g., ".0f", ".2f", "+.1f")
        prefix: Prefix to add (e.g., "$")
        suffix: Suffix to add (e.g., "%")
        na: String to show when value is None

    Returns:
        Formatted string
    """
    if val is None:
        return na
    try:
        return f"{prefix}{val:{spec}}{suffix}"
    except (ValueError, TypeError):
        return na


def fmt_bool(val: Optional[bool], true_text: str = "Yes", false_text: str = "No", na_text: str = "N/A") -> str:
    """Format boolean value safely."""
    if val is None:
        return na_text
    return true_text if val else false_text


def safe_val(val: Any, default: float = 0) -> float:
    """Return value or default if None (for calculations)."""
    return val if val is not None else default


# Economic Calendar UI
try:
    from src.components.economic_calendar_ui import render_economic_calendar

    ECON_CALENDAR_AVAILABLE = True
except ImportError:
    ECON_CALENDAR_AVAILABLE = False


def render_analytics_tab(positions: list = None, account_summary: dict = None):
    """
    Render the Analytics tab with Signal Performance, Risk Dashboard, Optimizer, and Options Flow.

    Args:
        positions: List of portfolio positions from IBKR
        account_summary: Account summary dict from IBKR
    """

    st.subheader("📊 Analytics & Optimization")

    # Create sub-tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Signal Performance",
        "⚠️ Risk Dashboard",
        "🎯 Portfolio Optimizer",
        "🔮 Options Flow",
        "🩳 Short Squeeze",
        "📊 Earnings Analysis",
        "🌍 Macro Regime",
        "📅 Economic Calendar"
    ])

    # =========================================================================
    # TAB 1: SIGNAL PERFORMANCE
    # =========================================================================
    with tab1:
        render_signal_performance_tab()

    # =========================================================================
    # TAB 2: RISK DASHBOARD
    # =========================================================================
    with tab2:
        render_risk_dashboard_tab(positions, account_summary)

    # =========================================================================
    # TAB 3: PORTFOLIO OPTIMIZER
    # =========================================================================
    with tab3:
        render_optimizer_tab(positions, account_summary)

    # =========================================================================
    # TAB 4: OPTIONS FLOW
    # =========================================================================
    with tab4:
        render_options_flow_tab(positions)

    # =========================================================================
    # TAB 5: SHORT SQUEEZE
    # =========================================================================
    with tab5:
        render_short_squeeze_tab(positions)

    # =========================================================================
    # TAB 6: EARNINGS ANALYSIS
    # =========================================================================
    with tab6:
        try:
            from dashboard.earnings_tab import render_earnings_tab
            render_earnings_tab(positions, account_summary)
        except ImportError as e:
            st.warning("📊 Earnings Analysis not available")
            st.info(f"Make sure dashboard/earnings_tab.py exists\n\nError: {e}")

    # =========================================================================
    # TAB 7: MACRO REGIME
    # =========================================================================
    with tab7:
        try:
            from dashboard.macro_regime_tab import render_macro_regime_tab
            render_macro_regime_tab()
        except ImportError as e:
            st.warning("🌍 Macro Regime not available")
            st.info(f"Make sure dashboard/macro_regime_tab.py exists\n\nError: {e}")

    # =========================================================================
    # TAB 8: ECONOMIC CALENDAR
    # =========================================================================
    with tab8:
        if ECON_CALENDAR_AVAILABLE:
            render_economic_calendar()
        else:
            st.warning("📅 Economic Calendar not available")
            st.info(
                "Make sure these files exist:\n- src/components/economic_calendar_ui.py\n- src/analytics/economic_calendar.py\n- src/analytics/economic_news_analyzer.py")
            st.code("pip install investpy", language="bash")


def render_signal_performance_tab():
    """Render Signal Performance Tracker."""

    st.markdown("### 📈 Signal Performance Tracker")
    st.caption("Track how accurate your trading signals are over time")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        days_back = st.slider("Analysis Period (days)", 30, 180, 90, key="perf_days")

    with col2:
        if st.button("📥 Load Analysis", key="load_signal_perf", type="primary"):
            st.session_state.signal_perf_loaded = True
            # Clear old cache for new load
            for key in list(st.session_state.keys()):
                if key.startswith('signal_perf_') and key != 'signal_perf_loaded':
                    del st.session_state[key]

    with col3:
        if st.button("🔄 Refresh", key="refresh_signal_perf"):
            # Clear cache for this function
            for key in list(st.session_state.keys()):
                if key.startswith('signal_perf_'):
                    del st.session_state[key]
            st.session_state.signal_perf_loaded = True

    # Only load if user clicked Load or Refresh, or already loaded
    if not st.session_state.get('signal_perf_loaded', False):
        st.info("📊 Click 'Load Analysis' to calculate signal performance metrics.")
        return

    try:
        from src.analytics.signal_performance import SignalPerformanceTracker

        # Always create tracker instance
        tracker = SignalPerformanceTracker()

        # Use session state caching for the expensive summary calculation
        cache_key = f"signal_perf_{days_back}"

        if cache_key not in st.session_state:
            with st.spinner("Calculating signal performance..."):
                summary = tracker.get_performance_summary(days_back=days_back)
            st.session_state[cache_key] = summary
        else:
            summary = st.session_state[cache_key]

        if summary['total_signals'] == 0:
            st.warning("No signals found in the selected period. Run the screener to generate signals.")
            return

        # Overview metrics
        st.markdown("#### Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Signals", summary['total_signals'])
        with col2:
            st.metric("With Returns", summary['signals_with_returns'])
        with col3:
            win_rate = summary['overall_win_rate']
            st.metric("Win Rate (10d)", f"{win_rate:.1f}%",
                      delta="Good" if win_rate > 50 else "Needs Work",
                      delta_color="normal" if win_rate > 50 else "inverse")
        with col4:
            avg_ret = summary['overall_avg_return']
            st.metric("Avg Return (10d)", f"{avg_ret:+.2f}%",
                      delta_color="normal" if avg_ret > 0 else "inverse")

        st.markdown("---")

        # Performance by signal type
        st.markdown("#### Performance by Signal Type")

        by_type = summary.get('by_signal_type', {})

        if by_type:
            # Create comparison chart
            perf_data = []
            for sig_type, perf in by_type.items():
                perf_data.append({
                    'Signal Type': sig_type,
                    'Count': perf.total_signals,
                    'Win Rate 5d': perf.win_rate_5d,
                    'Win Rate 10d': perf.win_rate_10d,
                    'Win Rate 30d': perf.win_rate_30d,
                    'Avg Return 5d': perf.avg_return_5d,
                    'Avg Return 10d': perf.avg_return_10d,
                    'Avg Return 30d': perf.avg_return_30d,
                    'Best': f"{perf.best_ticker} ({perf.best_return:+.1f}%)",
                    'Worst': f"{perf.worst_ticker} ({perf.worst_return:+.1f}%)"
                })

            perf_df = pd.DataFrame(perf_data)

            # Win rate chart
            fig = go.Figure()

            for sig_type in perf_df['Signal Type']:
                row = perf_df[perf_df['Signal Type'] == sig_type].iloc[0]
                fig.add_trace(go.Bar(
                    name=sig_type,
                    x=['5 Day', '10 Day', '30 Day'],
                    y=[row['Win Rate 5d'], row['Win Rate 10d'], row['Win Rate 30d']],
                    text=[f"{row['Win Rate 5d']:.0f}%", f"{row['Win Rate 10d']:.0f}%", f"{row['Win Rate 30d']:.0f}%"],
                    textposition='auto'
                ))

            fig.update_layout(
                title="Win Rate by Signal Type and Horizon",
                yaxis_title="Win Rate (%)",
                barmode='group',
                height=400
            )
            fig.add_hline(y=50, line_dash="dash", line_color="gray",
                          annotation_text="50% (Random)")

            st.plotly_chart(fig, width='stretch')

            # Detailed table
            st.markdown("#### Detailed Statistics")
            display_df = perf_df[['Signal Type', 'Count', 'Win Rate 10d', 'Avg Return 10d', 'Best', 'Worst']]
            display_df.columns = ['Signal', 'Count', 'Win Rate', 'Avg Return', 'Best Performer', 'Worst Performer']

            st.dataframe(display_df, width='stretch', hide_index=True)

        # Recent signals with results
        st.markdown("---")
        st.markdown("#### Recent Signals Performance")

        recent_df = tracker.get_recent_signals_performance(limit=30)

        if not recent_df.empty:
            recent_df['signal_date'] = pd.to_datetime(recent_df['signal_date']).dt.strftime('%Y-%m-%d')

            # Format returns
            recent_df['return_2d'] = recent_df['return_2d'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
            recent_df['return_5d'] = recent_df['return_5d'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
            recent_df['return_10d'] = recent_df['return_10d'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")

            # Add result_2d column
            def get_result_2d(row):
                if row.get('return_2d') == 'N/A':
                    return "⏳ Pending"
                try:
                    ret_val = float(row['return_2d'].replace('%', '').replace('+', ''))
                    signal = row.get('signal_type', '').upper()
                    if 'BUY' in signal or 'INCOME' in signal:
                        return "✅ Win" if ret_val > 0 else "❌ Loss"
                    elif 'SELL' in signal:
                        return "✅ Win" if ret_val < 0 else "❌ Loss"
                    else:
                        return "➖ Neutral"
                except:
                    return "⏳ Pending"

            recent_df['result_2d'] = recent_df.apply(get_result_2d, axis=1)

            # Display columns
            display_cols = ['ticker', 'signal_date', 'signal_type', 'return_2d', 'return_5d', 'return_10d', 'result_2d']
            display_cols = [c for c in display_cols if c in recent_df.columns]

            st.dataframe(
                recent_df[display_cols],
                width='stretch',
                hide_index=True,
                height=400
            )
        else:
            st.info("No recent signals with price data found.")

    except ImportError as e:
        st.error(f"Module not found: {e}")
        st.info("Make sure signal_performance.py is in src/analytics/")
    except Exception as e:
        st.error(f"Error loading signal performance: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_risk_dashboard_tab(positions: list, account_summary: dict):
    """Render Risk Dashboard."""

    st.markdown("### ⚠️ Risk Dashboard")
    st.caption("Comprehensive portfolio risk analysis")

    if not positions:
        st.warning("No portfolio data available. Go to Portfolio tab to load positions, then return here.")
        return

    try:
        from src.analytics.risk_dashboard import RiskDashboard

        total_value = account_summary.get('net_liquidation', 0) if account_summary else None

        with st.spinner("Analyzing portfolio risk..."):
            dashboard = RiskDashboard(positions, total_value)
            metrics = dashboard.get_full_risk_metrics()

        # Risk Score Card
        st.markdown("#### Risk Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            div_score = metrics.diversification_score
            color = "normal" if div_score > 60 else "inverse"
            st.metric("Diversification Score", f"{div_score:.0f}/100",
                      delta="Good" if div_score > 60 else "Low",
                      delta_color=color)

        with col2:
            st.metric("Portfolio Beta", f"{metrics.portfolio_beta:.2f}",
                      delta="Market-like" if 0.8 < metrics.portfolio_beta < 1.2 else "Off-market")

        with col3:
            st.metric("Volatility (Ann.)", f"{metrics.portfolio_volatility:.1f}%")

        with col4:
            st.metric("VaR 95% Daily", f"${metrics.var_95_dollar:,.0f}",
                      delta=f"{metrics.var_95_daily:.2f}%")

        st.markdown("---")

        # Two column layout
        col_left, col_right = st.columns(2)

        with col_left:
            # Concentration Analysis
            st.markdown("#### Concentration Risk")

            st.metric("Top 10 Holdings", f"{metrics.top_10_concentration:.1f}%",
                      delta="High" if metrics.top_10_concentration > 60 else "OK",
                      delta_color="inverse" if metrics.top_10_concentration > 60 else "off")

            st.metric("Largest Position",
                      f"{metrics.largest_position} ({metrics.largest_position_pct:.1f}%)")

            st.metric("Effective # of Stocks", f"{metrics.effective_n_stocks:.1f}",
                      help="How many equally-weighted stocks your portfolio behaves like")

            # Sector pie chart
            st.markdown("#### Sector Allocation")
            if metrics.sector_weights:
                sector_df = pd.DataFrame([
                    {'Sector': k, 'Weight': v}
                    for k, v in metrics.sector_weights.items()
                ])

                fig = px.pie(sector_df, values='Weight', names='Sector',
                             title='Sector Breakdown')
                fig.update_layout(height=350)
                st.plotly_chart(fig, width='stretch')

        with col_right:
            # Drawdown & VaR
            st.markdown("#### Drawdown Analysis")

            col_dd1, col_dd2 = st.columns(2)
            with col_dd1:
                st.metric("Max Drawdown (30d)", f"{metrics.max_drawdown_30d:.2f}%")
            with col_dd2:
                st.metric("Current Drawdown", f"{metrics.current_drawdown:.2f}%")

            # Correlation info
            st.markdown("#### Correlation Risk")

            st.metric("Avg Correlation", f"{metrics.avg_correlation:.2f}",
                      delta="High" if metrics.avg_correlation > 0.6 else "OK",
                      delta_color="inverse" if metrics.avg_correlation > 0.6 else "off")

            if metrics.max_correlation_pair[0]:
                st.metric("Most Correlated Pair",
                          f"{metrics.max_correlation_pair[0]} / {metrics.max_correlation_pair[1]}",
                          delta=f"{metrics.max_correlation:.2f}")

            # VaR gauge
            st.markdown("#### Value at Risk")

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=metrics.var_95_daily,
                title={'text': "Daily VaR 95%"},
                delta={'reference': 2.0},
                gauge={
                    'axis': {'range': [0, 5]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 1.5], 'color': "lightgreen"},
                        {'range': [1.5, 3], 'color': "yellow"},
                        {'range': [3, 5], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 3
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, width='stretch')

        # Correlation Matrix (expandable)
        st.markdown("---")
        with st.expander("📊 View Correlation Matrix"):
            corr_matrix, _ = dashboard.calculate_correlation_matrix()
            if not corr_matrix.empty:
                # Limit to top 20 holdings for readability
                if len(corr_matrix) > 20:
                    top_symbols = list(metrics.sector_weights.keys())[
                        :20] if metrics.sector_weights else corr_matrix.columns[:20].tolist()
                    valid_symbols = [s for s in top_symbols if s in corr_matrix.columns]
                    corr_matrix = corr_matrix.loc[valid_symbols, valid_symbols]

                fig = px.imshow(corr_matrix,
                                labels=dict(color="Correlation"),
                                color_continuous_scale="RdBu_r",
                                aspect="auto")
                fig.update_layout(height=500)
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("Insufficient data to calculate correlation matrix")

    except ImportError as e:
        st.error(f"Module not found: {e}")
        st.info("Make sure risk_dashboard.py is in src/analytics/")
    except Exception as e:
        st.error(f"Error loading risk dashboard: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_optimizer_tab(positions: list, account_summary: dict):
    """Render Portfolio Optimizer."""

    st.markdown("### 🎯 Portfolio Optimizer")
    st.caption("Optimize your portfolio using Modern Portfolio Theory")

    if not positions:
        st.warning("No portfolio data available. Go to Portfolio tab to load positions, then return here.")
        return

    # Optimization settings
    col1, col2, col3 = st.columns(3)

    with col1:
        strategy = st.selectbox(
            "Optimization Strategy",
            options=['max_sharpe', 'min_volatility', 'risk_parity', 'max_diversification'],
            format_func=lambda x: {
                'max_sharpe': '⚡ Maximum Sharpe Ratio',
                'min_volatility': '🛡️ Minimum Volatility',
                'risk_parity': '⚖️ Risk Parity',
                'max_diversification': '🎯 Maximum Diversification'
            }.get(x, x),
            key="opt_strategy"
        )

    with col2:
        max_weight = st.slider("Max Position Weight", 5, 30, 15, key="opt_max_weight") / 100

    with col3:
        min_positions = st.slider("Min Positions", 5, 30, 10, key="opt_min_pos")

    # Run optimization
    if st.button("🚀 Run Optimization", type="primary", key="run_optimizer"):
        try:
            from src.analytics.portfolio_optimizer import PortfolioOptimizer, OptimizationConstraints

            # Extract symbols and current weights
            symbols = []
            current_weights = {}
            total_value = sum(p.get('market_value', p.get('marketValue', 0)) for p in positions)

            for p in positions:
                symbol = p.get('symbol', '')
                if not symbol or symbol in ['USD', 'CASH', '']:
                    continue
                value = p.get('market_value', p.get('marketValue', 0))
                if value > 0:
                    symbols.append(symbol)
                    current_weights[symbol] = value / total_value if total_value > 0 else 0

            if len(symbols) < 5:
                st.error("Need at least 5 positions for optimization")
                return

            with st.spinner(f"Running {strategy} optimization..."):
                constraints = OptimizationConstraints(
                    min_weight=0.0,
                    max_weight=max_weight,
                    min_positions=min_positions
                )

                optimizer = PortfolioOptimizer(symbols, current_weights)
                result = optimizer.optimize(strategy, constraints)

            if result is None:
                st.error("Optimization failed. Try different parameters.")
                return

            # Store result in session state
            st.session_state.optimization_result = result
            st.session_state.optimizer = optimizer

            st.success("✅ Optimization complete!")

        except ImportError as e:
            st.error(f"Module not found: {e}")
            st.info("Make sure portfolio_optimizer.py is in src/analytics/")
        except Exception as e:
            st.error(f"Optimization error: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Display results if available
    if 'optimization_result' in st.session_state and st.session_state.optimization_result:
        result = st.session_state.optimization_result

        st.markdown("---")
        st.markdown("#### Optimization Results")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Expected Return", f"{result.expected_return * 100:.1f}%",
                      help="Annualized expected return")
        with col2:
            st.metric("Expected Volatility", f"{result.expected_volatility * 100:.1f}%",
                      help="Annualized volatility")
        with col3:
            st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
        with col4:
            st.metric("Positions", len(result.weights))

        st.markdown("---")

        # Two columns: weights and changes
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Optimal Weights")

            weights_df = pd.DataFrame([
                {'Symbol': k, 'Weight': f"{v * 100:.1f}%", 'Weight_val': v * 100}
                for k, v in sorted(result.weights.items(), key=lambda x: x[1], reverse=True)
            ])

            if not weights_df.empty:
                # Bar chart
                fig = px.bar(weights_df.head(20), x='Symbol', y='Weight_val',
                             title='Top 20 Holdings (Optimal)',
                             labels={'Weight_val': 'Weight (%)'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')

        with col_right:
            st.markdown("#### Suggested Changes")

            if result.weight_changes:
                changes_df = pd.DataFrame([
                    {
                        'Symbol': k,
                        'Change': f"{v * 100:+.1f}%",
                        'Change_val': v * 100,
                        'Action': '🟢 BUY' if v > 0 else '🔴 SELL'
                    }
                    for k, v in sorted(result.weight_changes.items(),
                                       key=lambda x: abs(x[1]), reverse=True)
                ])

                # Show buys and sells
                buys = changes_df[changes_df['Change_val'] > 0.5]
                sells = changes_df[changes_df['Change_val'] < -0.5]

                st.markdown("**🟢 Increase:**")
                if not buys.empty:
                    for _, row in buys.head(10).iterrows():
                        st.write(f"  {row['Symbol']}: {row['Change']}")
                else:
                    st.write("  None")

                st.markdown("**🔴 Decrease:**")
                if not sells.empty:
                    for _, row in sells.head(10).iterrows():
                        st.write(f"  {row['Symbol']}: {row['Change']}")
                else:
                    st.write("  None")

        # Full weights table (expandable)
        with st.expander("📋 View Full Optimal Weights"):
            full_df = pd.DataFrame([
                {
                    'Symbol': k,
                    'Current': f"{result.current_weights.get(k, 0) * 100:.1f}%",
                    'Optimal': f"{v * 100:.1f}%",
                    'Change': f"{result.weight_changes.get(k, 0) * 100:+.1f}%"
                }
                for k, v in sorted(result.weights.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(full_df, width='stretch', hide_index=True)

        # Compare strategies
        st.markdown("---")
        with st.expander("📊 Compare All Strategies"):
            if st.button("Run All Strategies", key="compare_all"):
                try:
                    optimizer = st.session_state.get('optimizer')
                    if optimizer:
                        with st.spinner("Comparing strategies..."):
                            comparisons = optimizer.compare_strategies()

                        compare_data = []
                        for strat, res in comparisons.items():
                            compare_data.append({
                                'Strategy': strat,
                                'Expected Return': f"{res.expected_return * 100:.1f}%",
                                'Volatility': f"{res.expected_volatility * 100:.1f}%",
                                'Sharpe': f"{res.sharpe_ratio:.2f}",
                                'Positions': len(res.weights)
                            })

                        st.dataframe(pd.DataFrame(compare_data), width='stretch', hide_index=True)
                except Exception as e:
                    st.error(f"Comparison error: {e}")


def render_options_flow_tab(positions: list):
    """Render Options Flow analyzer with database integration."""

    st.markdown("### 🔮 Options Flow Analyzer")
    st.caption("Detect unusual options activity that may signal institutional moves")

    # Sub-tabs for different views
    flow_tab1, flow_tab2, flow_tab3 = st.tabs([
        "🔍 Analyze",
        "📋 Recent Alerts",
        "📈 History"
    ])

    # =================================================================
    # TAB 1: ANALYZE
    # =================================================================
    with flow_tab1:
        # Load universe tickers into session state (once)
        if 'universe_tickers' not in st.session_state:
            try:
                query = """
                        SELECT DISTINCT ticker \
                        FROM screener_scores
                        WHERE date >= CURRENT_DATE - INTERVAL '7 days'
                        ORDER BY ticker \
                        """
                from src.db.connection import get_engine
                df = pd.read_sql(query, get_engine())
                st.session_state.universe_tickers = df['ticker'].tolist()
            except Exception as e:
                st.session_state.universe_tickers = []

        universe_tickers = st.session_state.universe_tickers

        # Get portfolio tickers
        portfolio_tickers = []
        if positions:
            portfolio_tickers = [p.get('symbol', '') for p in positions
                                 if p.get('symbol') and p.get('symbol') not in ['USD', 'CASH']]

        # Callback to load all tickers (runs BEFORE widget renders on next rerun)
        def load_all_options_tickers():
            st.session_state.options_tickers_text = ", ".join(st.session_state.universe_tickers)

        # Initialize ticker value in session state
        if 'options_tickers_text' not in st.session_state:
            if universe_tickers:
                st.session_state.options_tickers_text = ", ".join(universe_tickers[:30])
            elif portfolio_tickers:
                st.session_state.options_tickers_text = ", ".join(portfolio_tickers[:30])
            else:
                st.session_state.options_tickers_text = "AAPL, MSFT, NVDA, TSLA, AMD, AMZN, GOOGL, META, SPY, QQQ"

        # Input section
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            ticker_input = st.text_area(
                "Tickers to analyze (comma-separated)",
                height=80,
                help=f"Universe: {len(universe_tickers)} tickers | Portfolio: {len(portfolio_tickers)} tickers",
                key="options_tickers_text"
            )

        with col2:
            st.write("")  # Spacing
            st.button(
                f"📥 Load All ({len(universe_tickers)})",
                key="load_full_universe",
                type="secondary",
                on_click=load_all_options_tickers
            )

        with col3:
            st.write("")
            use_portfolio = st.checkbox("Include portfolio", value=False, key="options_use_portfolio")

        with col4:
            st.write("")
            save_to_db = st.checkbox("💾 Save to DB", value=True, key="options_save_db")

        # Parse tickers directly from widget return value
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        if use_portfolio and portfolio_tickers:
            tickers = list(set(tickers + portfolio_tickers))

        # Show count
        st.caption(f"📊 **{len(tickers)} tickers** ready to scan")

        # Single ticker analysis
        st.markdown("---")
        st.markdown("#### 🔍 Single Ticker Analysis")

        col_single1, col_single2 = st.columns([3, 1])

        with col_single1:
            single_ticker = st.selectbox(
                "Select ticker for detailed analysis",
                options=tickers if tickers else ["AAPL"],
                key="options_single_ticker"
            )

        with col_single2:
            analyze_single = st.button("🔍 Analyze", key="analyze_single_options", type="primary")

        if analyze_single:
            try:
                from src.analytics.options_flow import OptionsFlowAnalyzer

                with st.spinner(f"Analyzing {single_ticker} options flow..."):
                    analyzer = OptionsFlowAnalyzer()
                    if save_to_db:
                        summary = analyzer.analyze_and_save(single_ticker)
                        st.toast(f"✅ Saved {len(summary.alerts)} alerts to database")
                    else:
                        summary = analyzer.analyze_ticker(single_ticker)
                    st.session_state.options_single_result = summary

            except Exception as e:
                st.error(f"Analysis error: {e}")

        # Display results if available
        if st.session_state.get('options_single_result'):
            summary = st.session_state.options_single_result

            if summary:
                # Key metrics row
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Stock Price", f"${summary.stock_price:.2f}")
                with col2:
                    st.metric("Max Pain", f"${summary.max_pain_price:.2f}",
                              delta=f"{((summary.max_pain_price - summary.stock_price) / summary.stock_price * 100):.1f}%" if summary.stock_price > 0 else None)
                with col3:
                    sentiment_color = "🟢" if summary.overall_sentiment == "BULLISH" else "🔴" if summary.overall_sentiment == "BEARISH" else "⚪"
                    st.metric("Sentiment", f"{sentiment_color} {summary.overall_sentiment}",
                              delta=f"Score: {summary.sentiment_score:.0f}")
                with col4:
                    st.metric("Put/Call Ratio", f"{summary.put_call_volume_ratio:.2f}",
                              delta="Bullish" if summary.put_call_volume_ratio < 0.7 else "Bearish" if summary.put_call_volume_ratio > 1.3 else "Neutral")

                # Volume and OI
                st.markdown("---")
                col_vol1, col_vol2 = st.columns(2)

                with col_vol1:
                    st.markdown("##### 📊 Volume")
                    vol_data = pd.DataFrame({
                        'Type': ['Calls', 'Puts'],
                        'Volume': [summary.total_call_volume, summary.total_put_volume]
                    })
                    fig = px.bar(vol_data, x='Type', y='Volume', color='Type',
                                 color_discrete_map={'Calls': 'green', 'Puts': 'red'})
                    fig.update_layout(height=250, showlegend=False)
                    st.plotly_chart(fig, width='stretch')

                with col_vol2:
                    st.markdown("##### 📈 Open Interest")
                    oi_data = pd.DataFrame({
                        'Type': ['Calls', 'Puts'],
                        'Open Interest': [summary.total_call_oi, summary.total_put_oi]
                    })
                    fig = px.bar(oi_data, x='Type', y='Open Interest', color='Type',
                                 color_discrete_map={'Calls': 'green', 'Puts': 'red'})
                    fig.update_layout(height=250, showlegend=False)
                    st.plotly_chart(fig, width='stretch')

                # IV metrics
                st.markdown("##### 📉 Implied Volatility")
                col_iv1, col_iv2, col_iv3 = st.columns(3)
                with col_iv1:
                    st.metric("Avg Call IV", f"{summary.avg_call_iv * 100:.1f}%")
                with col_iv2:
                    st.metric("Avg Put IV", f"{summary.avg_put_iv * 100:.1f}%")
                with col_iv3:
                    st.metric("IV Skew", f"{summary.iv_skew * 100:.1f}%",
                              help="Put IV - Call IV. Positive = more fear/hedging")

                # Unusual activity alerts
                if summary.alerts:
                    st.markdown("---")
                    st.markdown(f"##### ⚠️ Unusual Activity ({len(summary.alerts)} alerts)")

                    alerts_data = []
                    for alert in summary.alerts[:15]:
                        alerts_data.append({
                            'Severity': alert.severity,
                            'Type': alert.option_type,
                            'Strike': f"${alert.strike:.0f}",
                            'Expiry': alert.expiry,
                            'Volume': f"{alert.volume:,}",
                            'OI': f"{alert.open_interest:,}",
                            'Vol/OI': f"{alert.volume_oi_ratio:.1f}x" if alert.volume_oi_ratio < 100 else ">100x",
                            'IV': f"{alert.implied_volatility * 100:.0f}%",
                            'Direction': alert.direction,
                        })

                    if alerts_data:
                        alerts_df = pd.DataFrame(alerts_data)
                        st.dataframe(alerts_df, width='stretch', hide_index=True)
                else:
                    st.success("✅ No unusual options activity detected")

        # Universe scan section
        st.markdown("---")
        st.markdown("#### 🌐 Universe Scan")

        col_scan1, col_scan2 = st.columns([3, 1])

        with col_scan1:
            st.caption(f"Scan {len(tickers)} tickers for unusual activity")

        with col_scan2:
            scan_btn = st.button("🚀 Scan All", key="scan_options_universe", type="primary")

        if scan_btn:
            if not tickers:
                st.warning("Enter at least one ticker to scan")
            else:
                try:
                    from src.analytics.options_flow import OptionsFlowAnalyzer

                    with st.spinner(f"Scanning {len(tickers)} tickers..."):
                        analyzer = OptionsFlowAnalyzer()
                        if save_to_db:
                            results = analyzer.scan_and_save(tickers, max_workers=3)
                            st.toast(f"✅ Scan complete - results saved to database")
                        else:
                            results = analyzer.scan_universe(tickers, max_workers=3)

                    if results:
                        st.success(f"Found {len(results)} tickers with notable activity")

                        summary_data = []
                        for r in results[:20]:
                            summary_data.append({
                                'Ticker': r.ticker,
                                'Price': f"${r.stock_price:.2f}",
                                'Sentiment': r.overall_sentiment,
                                'Score': f"{r.sentiment_score:.0f}",
                                'P/C Ratio': f"{r.put_call_volume_ratio:.2f}",
                                'Call Vol': f"{r.total_call_volume:,}",
                                'Put Vol': f"{r.total_put_volume:,}",
                                'Alerts': len(r.alerts)
                            })

                        st.dataframe(pd.DataFrame(summary_data), width='stretch', hide_index=True)
                    else:
                        st.info("No significant unusual activity found")

                except Exception as e:
                    st.error(f"Scan error: {e}")

    # =================================================================
    # TAB 2: RECENT ALERTS FROM DATABASE
    # =================================================================
    with flow_tab2:
        st.markdown("#### 📋 Recent Alerts from Database")

        col_filter1, col_filter2, col_filter3 = st.columns(3)

        with col_filter1:
            days_back = st.selectbox("Days back", [1, 3, 7, 14, 30], index=2, key="alerts_days")
        with col_filter2:
            severity_filter = st.selectbox("Severity", ["All", "HIGH", "MEDIUM", "LOW"], key="alerts_severity")
        with col_filter3:
            ticker_filter = st.text_input("Ticker (optional)", key="alerts_ticker_filter")

        if st.button("🔄 Load Alerts", key="load_alerts"):
            try:
                from src.analytics.options_flow import OptionsFlowAnalyzer

                sev = severity_filter if severity_filter != "All" else None
                tick = ticker_filter.upper() if ticker_filter else None

                alerts_df = OptionsFlowAnalyzer.get_recent_alerts(days=days_back, severity=sev, ticker=tick)

                if not alerts_df.empty:
                    st.success(f"Found {len(alerts_df)} alerts")
                    st.dataframe(alerts_df, width='stretch', hide_index=True, height=500)
                else:
                    st.info("No alerts found. Run a scan to populate the database.")

            except Exception as e:
                st.error(f"Error loading alerts: {e}")
                st.info("Make sure you've run the SQL migration to create the options_flow_alerts table.")

    # =================================================================
    # TAB 3: HISTORICAL SENTIMENT
    # =================================================================
    with flow_tab3:
        st.markdown("#### 📈 Historical Options Sentiment")

        col_hist1, col_hist2 = st.columns([3, 1])

        with col_hist1:
            history_ticker = st.text_input("Ticker", value="AAPL", key="history_ticker")
        with col_hist2:
            history_days = st.selectbox("Days", [7, 14, 30, 60, 90], index=2, key="history_days")

        if st.button("📊 Load History", key="load_history"):
            try:
                from src.analytics.options_flow import OptionsFlowAnalyzer

                history_df = OptionsFlowAnalyzer.get_sentiment_history(history_ticker, history_days)

                if not history_df.empty:
                    st.success(f"Found {len(history_df)} days of data")

                    # Sentiment chart
                    fig = go.Figure()

                    # Sentiment score line
                    fig.add_trace(go.Scatter(
                        x=history_df['scan_date'],
                        y=history_df['sentiment_score'],
                        mode='lines+markers',
                        name='Sentiment Score',
                        line=dict(color='blue', width=2)
                    ))

                    # Add horizontal lines for reference
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig.add_hline(y=50, line_dash="dot", line_color="green", annotation_text="Bullish")
                    fig.add_hline(y=-50, line_dash="dot", line_color="red", annotation_text="Bearish")

                    fig.update_layout(
                        title=f"{history_ticker} Options Sentiment Over Time",
                        yaxis_title="Sentiment Score",
                        height=400
                    )
                    st.plotly_chart(fig, width='stretch')

                    # Put/Call ratio chart
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=history_df['scan_date'],
                        y=history_df['put_call_volume_ratio'],
                        mode='lines+markers',
                        name='Put/Call Ratio',
                        line=dict(color='purple', width=2)
                    ))
                    fig2.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Neutral")
                    fig2.update_layout(
                        title="Put/Call Volume Ratio",
                        yaxis_title="Ratio",
                        height=300
                    )
                    st.plotly_chart(fig2, width='stretch')

                    # Data table
                    with st.expander("📋 View Raw Data"):
                        st.dataframe(history_df, width='stretch', hide_index=True)
                else:
                    st.info(
                        f"No historical data found for {history_ticker}. Run scans over multiple days to build history.")

            except Exception as e:
                st.error(f"Error loading history: {e}")
                st.info("Make sure you've run the SQL migration to create the options_flow_daily table.")


def render_short_squeeze_tab(positions: list):
    """Render Short Squeeze detector."""

    st.markdown("### 🩳 Short Squeeze Detector")
    st.caption("Identify stocks with high short squeeze potential")

    # Info box explaining squeeze mechanics
    with st.expander("ℹ️ How Short Squeezes Work"):
        st.markdown("""
        **What is a Short Squeeze?**

        A short squeeze occurs when a heavily shorted stock rises in price, forcing short sellers to buy shares to cover their positions, which drives the price even higher.

        **Key Indicators:**
        - **Short % of Float > 20%**: Heavy short interest
        - **Days to Cover > 5**: Takes long time for shorts to exit
        - **Bullish Options Flow**: Call buying adds gamma squeeze pressure
        - **Rising Price + Volume**: Momentum forcing shorts to cover

        **Famous Examples:** GameStop (GME) 2021, Volkswagen 2008
        """)

    # Input section - use same universe from session state
    if 'universe_tickers' not in st.session_state:
        try:
            query = """
                    SELECT DISTINCT ticker \
                    FROM screener_scores
                    WHERE date >= CURRENT_DATE - INTERVAL '7 days'
                    ORDER BY ticker \
                    """
            from src.db.connection import get_engine
            df = pd.read_sql(query, get_engine())
            st.session_state.universe_tickers = df['ticker'].tolist()
        except:
            st.session_state.universe_tickers = []

    universe_tickers = st.session_state.universe_tickers

    # Get portfolio tickers
    portfolio_tickers = []
    if positions:
        portfolio_tickers = [p.get('symbol', '') for p in positions
                             if p.get('symbol') and p.get('symbol') not in ['USD', 'CASH']]

    # Callback to load all tickers
    def load_all_squeeze_tickers():
        st.session_state.squeeze_tickers_text = ", ".join(st.session_state.universe_tickers)

    # Initialize ticker value in session state
    if 'squeeze_tickers_text' not in st.session_state:
        if universe_tickers:
            st.session_state.squeeze_tickers_text = ", ".join(universe_tickers[:30])
        elif portfolio_tickers:
            st.session_state.squeeze_tickers_text = ", ".join(portfolio_tickers[:30])
        else:
            st.session_state.squeeze_tickers_text = "GME, AMC, CVNA, MARA, RIOT, AAPL, TSLA, NVDA, AMD, SPY"

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        ticker_input = st.text_area(
            "Tickers to analyze (comma-separated)",
            height=80,
            help=f"Universe: {len(universe_tickers)} tickers | Portfolio: {len(portfolio_tickers)} tickers",
            key="squeeze_tickers_text"
        )

    with col2:
        st.write("")
        st.button(
            f"📥 Load All ({len(universe_tickers)})",
            key="load_squeeze_universe",
            type="secondary",
            on_click=load_all_squeeze_tickers
        )

    with col3:
        st.write("")
        include_portfolio = st.checkbox("Include portfolio", value=False, key="squeeze_include_portfolio")

    # Parse tickers directly from widget return value
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    if include_portfolio and portfolio_tickers:
        tickers = list(set(tickers + portfolio_tickers))

    # Show count
    st.caption(f"📊 **{len(tickers)} tickers** ready to scan")

    # Single ticker analysis
    st.markdown("---")
    st.markdown("#### 🔍 Single Stock Analysis")

    col_single1, col_single2 = st.columns([3, 1])

    with col_single1:
        single_ticker = st.selectbox(
            "Select ticker for detailed analysis",
            options=tickers if tickers else ["GME"],
            key="squeeze_single_ticker"
        )

    with col_single2:
        analyze_btn = st.button("🔍 Analyze", key="analyze_squeeze", type="primary")

    if analyze_btn:
        try:
            from src.analytics.short_squeeze import ShortSqueezeDetector

            with st.spinner(f"Analyzing {single_ticker} for squeeze potential..."):
                detector = ShortSqueezeDetector()
                data = detector.analyze_ticker(single_ticker)

            # Display results
            if data:
                # Squeeze score gauge
                col_score, col_risk = st.columns([2, 1])

                with col_score:
                    # Color based on risk level - FIXED: Handle None and NOT_ANALYZED
                    risk_colors = {
                        "EXTREME": "🔴",
                        "HIGH": "🟠",
                        "MEDIUM": "🟡",
                        "LOW": "🟢",
                        "UNKNOWN": "⚪",
                        "NOT_ANALYZED": "⚪",
                    }
                    risk_label = data.squeeze_risk if data.squeeze_risk else "UNKNOWN"
                    st.metric(
                        "Squeeze Score",
                        fmt(data.squeeze_score, suffix="/100"),
                        delta=f"{risk_colors.get(risk_label, '⚪')} {risk_label}"
                    )

                with col_risk:
                    st.metric("Current Price", fmt(data.current_price, ".2f", "$"))

                # Key metrics
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    short_pct = data.short_percent_of_float
                    st.metric(
                        "Short % of Float",
                        fmt(short_pct, ".1f", suffix="%"),
                        delta="High" if short_pct and short_pct > 15 else "Normal" if short_pct is not None else None
                    )

                with col2:
                    dtc = data.days_to_cover
                    st.metric(
                        "Days to Cover",
                        fmt(dtc, ".1f"),
                        delta="Long" if dtc and dtc > 5 else "Short" if dtc is not None else None
                    )

                with col3:
                    mom = data.short_change_mom
                    st.metric(
                        "Short Change MoM",
                        fmt(mom, "+.1f", suffix="%"),
                        delta="Increasing" if mom and mom > 0 else "Decreasing" if mom is not None else None
                    )

                with col4:
                    st.metric(
                        "Options Sentiment",
                        data.options_sentiment if data.options_sentiment and data.options_sentiment != "NOT_ANALYZED" else "N/A",
                        delta=f"P/C: {fmt(data.put_call_ratio, '.2f')}" if data.put_call_ratio else None
                    )

                # Additional metrics row
                col5, col6, col7, col8 = st.columns(4)

                with col5:
                    st.metric("5-Day Change", fmt(data.price_change_5d, "+.1f", suffix="%"))

                with col6:
                    st.metric("Relative Volume", fmt(data.relative_volume, ".1f", suffix="x"))

                with col7:
                    st.metric("RSI(14)", fmt(data.rsi_14))

                with col8:
                    st.metric(
                        "Unusual Calls",
                        fmt_bool(data.unusual_call_activity, "YES 🚀", "No")
                    )

                # Squeeze factors and warnings
                col_factors, col_warnings = st.columns(2)

                with col_factors:
                    st.markdown("##### ✅ Squeeze Factors")
                    if data.squeeze_factors:
                        for factor in data.squeeze_factors:
                            st.write(f"• {factor}")
                    else:
                        st.write("• No significant squeeze factors")

                with col_warnings:
                    st.markdown("##### ⚠️ Warnings")
                    if data.warning_factors:
                        for warning in data.warning_factors:
                            st.write(f"• {warning}")
                    else:
                        st.write("• No warnings")

                # Raw data - FIXED: Handle None values
                with st.expander("📋 View All Data"):
                    data_dict = {
                        "Metric": [
                            "Shares Short", "Float Shares", "Shares Outstanding",
                            "Avg Daily Volume", "1-Month Change", "Above 20 MA", "Above 50 MA"
                        ],
                        "Value": [
                            f"{data.short_interest:,}" if data.short_interest else "N/A",
                            f"{data.float_shares:,}" if data.float_shares else "N/A",
                            f"{data.shares_outstanding:,}" if data.shares_outstanding else "N/A",
                            f"{data.avg_volume:,}" if data.avg_volume else "N/A",
                            fmt(data.price_change_1m, "+.1f", suffix="%"),
                            fmt_bool(data.above_20ma) if hasattr(data, 'above_20ma') else "N/A",
                            fmt_bool(data.above_50ma) if hasattr(data, 'above_50ma') else "N/A"
                        ]
                    }
                    st.dataframe(pd.DataFrame(data_dict), width='stretch', hide_index=True)

        except Exception as e:
            st.error(f"Error analyzing: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Universe scan
    st.markdown("---")
    st.markdown("#### 🌐 Universe Scan")

    col_scan1, col_scan2 = st.columns([3, 1])

    with col_scan1:
        st.caption(f"Scan {len(tickers)} tickers for squeeze potential")

    with col_scan2:
        scan_btn = st.button("🚀 Scan All", key="scan_squeeze", type="primary")

    if scan_btn:
        if not tickers:
            st.warning("Enter at least one ticker to scan")
        else:
            try:
                from src.analytics.short_squeeze import ShortSqueezeDetector

                with st.spinner(f"Scanning {len(tickers)} tickers..."):
                    detector = ShortSqueezeDetector()
                    results = detector.scan_universe(tickers, max_workers=3)

                if results:
                    # Filter to show only those with some short interest - FIXED: Handle None
                    results = [r for r in results if r.short_percent_of_float and r.short_percent_of_float > 1]

                    if results:
                        st.success(f"Found {len(results)} tickers with short data")

                        # Summary table - FIXED: Handle None values
                        summary_data = []
                        for r in results:
                            risk_emoji = {
                                "EXTREME": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢",
                                "UNKNOWN": "⚪", "NOT_ANALYZED": "⚪"
                            }.get(r.squeeze_risk, "⚪")
                            summary_data.append({
                                'Ticker': r.ticker,
                                'Score': fmt(r.squeeze_score),
                                'Risk': f"{risk_emoji} {r.squeeze_risk or 'N/A'}",
                                'Short %': fmt(r.short_percent_of_float, ".1f", suffix="%"),
                                'Days Cover': fmt(r.days_to_cover, ".1f"),
                                'Options': r.options_sentiment if r.options_sentiment and r.options_sentiment != "NOT_ANALYZED" else "N/A",
                                '5D Chg': fmt(r.price_change_5d, "+.1f", suffix="%"),
                                'Vol': fmt(r.relative_volume, ".1f", suffix="x")
                            })

                        st.dataframe(
                            pd.DataFrame(summary_data),
                            width='stretch',
                            hide_index=True,
                            height=400
                        )
                    else:
                        st.info("No tickers with significant short interest found")
                else:
                    st.info("No results returned. Some tickers may not have short data available.")

            except Exception as e:
                st.error(f"Scan error: {e}")


# Standalone test
if __name__ == "__main__":
    st.set_page_config(page_title="Analytics Test", layout="wide")
    st.title("🧪 Analytics Tab Test")

    # Mock positions for testing
    mock_positions = [
        {"symbol": "AAPL", "market_value": 50000, "sector": "Technology"},
        {"symbol": "MSFT", "market_value": 45000, "sector": "Technology"},
        {"symbol": "GOOGL", "market_value": 40000, "sector": "Technology"},
        {"symbol": "AMZN", "market_value": 35000, "sector": "Consumer"},
        {"symbol": "JPM", "market_value": 30000, "sector": "Financial"},
    ]

    render_analytics_tab(mock_positions, {"net_liquidation": 200000})