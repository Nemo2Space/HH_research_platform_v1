"""
Portfolio Backtest Tab UI
=========================

Streamlit UI for backtesting saved portfolios or uploaded CSV/JSON portfolios.

Features:
- Load saved portfolios from database
- Upload CSV portfolio (requires: Ticker, Weight, Name columns)
- Upload JSON portfolio (requires: symbol, weight, name fields)
- Historical performance simulation
- Benchmark comparison

Author: HH Research Platform
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Optional, Tuple

try:
    from dashboard.portfolio_backtester import (
        PortfolioBacktester,
        BacktestConfig,
        create_performance_chart,
        create_drawdown_chart,
        create_returns_distribution
    )
    BACKTESTER_AVAILABLE = True
except ImportError:
    BACKTESTER_AVAILABLE = False


def validate_portfolio_data(df: pd.DataFrame) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Validate and normalize portfolio data from CSV or JSON.

    Supports multiple column naming conventions:
    - CSV style: Ticker, Weight, Name
    - JSON style: symbol, weight, name

    Returns:
        (is_valid, error_message, normalized_dataframe)
    """
    if df.empty:
        return False, "Uploaded file is empty", None

    # Normalize column names to lowercase for matching
    df.columns = df.columns.str.strip()
    col_map = {col.lower(): col for col in df.columns}

    # Find ticker column (supports: ticker, symbol)
    ticker_col = None
    for candidate in ['ticker', 'symbol']:
        if candidate in col_map:
            ticker_col = col_map[candidate]
            break

    # Find weight column (supports: weight, weight_pct)
    weight_col = None
    for candidate in ['weight', 'weight_pct']:
        if candidate in col_map:
            weight_col = col_map[candidate]
            break

    # Find name column
    name_col = col_map.get('name')

    # Check required columns
    missing = []
    if ticker_col is None:
        missing.append("Ticker/Symbol")
    if weight_col is None:
        missing.append("Weight")
    if name_col is None:
        missing.append("Name")

    if missing:
        return False, f"Missing required columns: {', '.join(missing)}. Found columns: {list(df.columns)}", None

    # Normalize to standard column names
    normalized_df = pd.DataFrame()
    normalized_df['ticker'] = df[ticker_col].astype(str).str.strip().str.upper()
    normalized_df['weight_pct'] = pd.to_numeric(df[weight_col], errors='coerce')
    normalized_df['name'] = df[name_col].astype(str).str.strip()

    # Optional: capture sector if available
    if 'sector' in col_map:
        normalized_df['sector'] = df[col_map['sector']].astype(str).str.strip()
    else:
        normalized_df['sector'] = ''

    # Optional: capture security type if available
    if 'sectype' in col_map:
        normalized_df['sec_type'] = df[col_map['sectype']].astype(str).str.strip()
    else:
        normalized_df['sec_type'] = 'STK'

    # Check for invalid weights
    if normalized_df['weight_pct'].isna().any():
        return False, "Some weight values are not valid numbers", None

    # Check weights sum (allow some tolerance)
    total_weight = normalized_df['weight_pct'].sum()
    if total_weight < 1:
        # Assume weights are in decimal form (0.10 = 10%)
        normalized_df['weight_pct'] = normalized_df['weight_pct'] * 100
        total_weight = normalized_df['weight_pct'].sum()

    if abs(total_weight - 100) > 5:
        return False, f"Weights sum to {total_weight:.2f}%, expected ~100%", None

    # Check for empty tickers
    if (normalized_df['ticker'] == '').any():
        return False, "Some tickers are empty", None

    # Check for duplicates
    duplicates = normalized_df[normalized_df['ticker'].duplicated()]['ticker'].tolist()
    if duplicates:
        return False, f"Duplicate tickers found: {', '.join(duplicates[:5])}", None

    # Add placeholder columns for compatibility
    normalized_df['value'] = 0  # Will be calculated based on initial capital
    normalized_df['shares'] = 0
    normalized_df['score'] = 0
    normalized_df['conviction'] = None

    return True, "", normalized_df


def parse_uploaded_file(uploaded_file) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Parse uploaded file (CSV or JSON) and return normalized DataFrame.

    Returns:
        (is_valid, error_message, normalized_dataframe)
    """
    filename = uploaded_file.name.lower()

    try:
        if filename.endswith('.json'):
            # Parse JSON
            content = uploaded_file.read().decode('utf-8')
            data = json.loads(content)

            # Handle both array of objects and object with array
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Try to find an array in the dict
                for key, value in data.items():
                    if isinstance(value, list):
                        df = pd.DataFrame(value)
                        break
                else:
                    return False, "JSON must contain an array of holdings", None
            else:
                return False, "Invalid JSON structure", None

        elif filename.endswith('.csv'):
            # Parse CSV
            df = pd.read_csv(uploaded_file)
        else:
            return False, f"Unsupported file type: {filename}. Use .csv or .json", None

        # Validate and normalize
        return validate_portfolio_data(df)

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON format: {e}", None
    except pd.errors.EmptyDataError:
        return False, "File is empty", None
    except Exception as e:
        return False, f"Error reading file: {e}", None


def render_backtest_tab():
    """Render the Portfolio Backtest tab."""

    st.markdown("### 📈 Portfolio Backtesting")
    st.caption("Backtest saved portfolios or upload a CSV portfolio")

    if not BACKTESTER_AVAILABLE:
        st.error("""
        Portfolio Backtester not available.
        
        **Required:**
        1. Install yfinance: `pip install yfinance`
        2. Place `portfolio_backtester.py` in `dashboard/` folder
        """)
        return

    # Initialize backtester
    backtester = PortfolioBacktester()

    # Session state
    if 'backtest_result' not in st.session_state:
        st.session_state.backtest_result = None
    if 'backtest_config' not in st.session_state:
        st.session_state.backtest_config = None
    if 'csv_holdings' not in st.session_state:
        st.session_state.csv_holdings = None
    if 'csv_portfolio_name' not in st.session_state:
        st.session_state.csv_portfolio_name = None

    # ===================================================================
    # SECTION 1: Portfolio Source Selection
    # ===================================================================
    st.markdown("#### 1️⃣ Select Portfolio Source")

    source_type = st.radio(
        "Portfolio Source",
        options=["📁 Saved Portfolio", "📄 Upload File (CSV/JSON)"],
        horizontal=True,
        key="portfolio_source_type"
    )

    holdings_df = None
    portfolio_name = None
    initial_value = 100000  # Default

    # ===================================================================
    # SAVED PORTFOLIO SELECTION
    # ===================================================================
    if source_type == "📁 Saved Portfolio":
        # Clear CSV state
        st.session_state.csv_holdings = None
        st.session_state.csv_portfolio_name = None

        # Load saved portfolios
        portfolios_df = backtester.get_saved_portfolios()

        if portfolios_df.empty:
            st.warning("""
            No saved portfolios found.
            
            **To create a portfolio:**
            1. Go to "AI Portfolio Manager" tab
            2. Build a portfolio
            3. Save it using the 💾 Save Portfolio button
            
            **Or:** Select "Upload CSV" to backtest a custom portfolio.
            """)
            return

        # Portfolio selection
        col1, col2 = st.columns([2, 1])

        with col1:
            portfolio_options = {
                f"{row['name']} (ID: {row['id']}, {row['num_holdings']} holdings)": row['id']
                for _, row in portfolios_df.iterrows()
            }

            selected_name = st.selectbox(
                "Choose a saved portfolio",
                options=list(portfolio_options.keys()),
                key='backtest_portfolio_select'
            )

            selected_id = portfolio_options[selected_name]

        with col2:
            # Show portfolio details
            port_info = portfolios_df[portfolios_df['id'] == selected_id].iloc[0]
            st.metric("Holdings", f"{port_info['num_holdings']}")
            st.metric("Original Value", f"${port_info['total_value']:,.0f}")
            initial_value = int(port_info['total_value'])
            portfolio_name = port_info['name']

        # Show holdings
        with st.expander("📋 View Portfolio Holdings"):
            holdings_df = backtester.load_portfolio_holdings(selected_id)
            if not holdings_df.empty:
                st.dataframe(
                    holdings_df[['ticker', 'weight_pct', 'value', 'score', 'conviction']],
                    width='stretch',
                    hide_index=True
                )
            else:
                st.warning("No holdings found")
                return

    # ===================================================================
    # CSV/JSON UPLOAD SELECTION
    # ===================================================================
    else:  # Upload CSV/JSON
        st.markdown("""
        **Supported File Formats:**
        - **CSV**: Columns `Ticker`, `Weight`, `Name` (or `Symbol` instead of `Ticker`)
        - **JSON**: Array of objects with `symbol`, `weight`, `name` fields
        
        Weights can be percentage (e.g., 10.5) or decimal (e.g., 0.105) and should sum to ~100%
        """)

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Upload Portfolio (CSV or JSON)",
                type=['csv', 'json'],
                key='portfolio_file_upload'
            )

        with col2:
            csv_portfolio_name = st.text_input(
                "Portfolio Name",
                value="Uploaded Portfolio",
                key='uploaded_portfolio_name_input'
            )

        if uploaded_file is not None:
            try:
                # Parse file (CSV or JSON)
                is_valid, error_msg, normalized_df = parse_uploaded_file(uploaded_file)

                if not is_valid:
                    st.error(f"❌ Invalid file: {error_msg}")
                    return

                # Store in session state
                st.session_state.csv_holdings = normalized_df
                st.session_state.csv_portfolio_name = csv_portfolio_name

                holdings_df = normalized_df
                portfolio_name = csv_portfolio_name

                # Show success and preview
                file_type = "JSON" if uploaded_file.name.lower().endswith('.json') else "CSV"
                st.success(f"✅ Loaded {len(holdings_df)} holdings from {file_type}")

                col1, col2 = st.columns([2, 1])
                with col1:
                    with st.expander("📋 View Portfolio Holdings", expanded=True):
                        display_cols = ['ticker', 'weight_pct', 'name']
                        if 'sector' in holdings_df.columns and holdings_df['sector'].any():
                            display_cols.append('sector')
                        display_df = holdings_df[display_cols].copy()
                        display_df.columns = ['Ticker', 'Weight %', 'Name'] + (['Sector'] if 'sector' in display_cols else [])
                        st.dataframe(display_df, width='stretch', hide_index=True)

                with col2:
                    st.metric("Holdings", len(holdings_df))
                    st.metric("Total Weight", f"{holdings_df['weight_pct'].sum():.2f}%")
                    if 'sector' in holdings_df.columns:
                        unique_sectors = holdings_df[holdings_df['sector'] != '']['sector'].nunique()
                        if unique_sectors > 0:
                            st.metric("Sectors", unique_sectors)

            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
                return

        elif st.session_state.csv_holdings is not None:
            # Use previously loaded CSV
            holdings_df = st.session_state.csv_holdings
            portfolio_name = st.session_state.csv_portfolio_name

            st.info(f"Using previously loaded portfolio: {portfolio_name} ({len(holdings_df)} holdings)")

            with st.expander("📋 View Portfolio Holdings"):
                display_df = holdings_df[['ticker', 'weight_pct', 'name']].copy()
                display_df.columns = ['Ticker', 'Weight %', 'Name']
                st.dataframe(display_df, width='stretch', hide_index=True)
        else:
            st.info("👆 Upload a CSV file to continue")
            return

    # Ensure we have holdings to proceed
    if holdings_df is None or holdings_df.empty:
        st.warning("No holdings available for backtesting")
        return

    # ===================================================================
    # SECTION 2: Backtest Configuration
    # ===================================================================
    st.markdown("---")
    st.markdown("#### 2️⃣ Backtest Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=10_000_000,
            value=initial_value,
            step=1000,
            help="Starting portfolio value"
        )

        benchmark = st.selectbox(
            "Benchmark",
            options=['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'AGG'],
            index=0,
            help="Index to compare against"
        )

    with col2:
        # Date range
        default_start = datetime.now() - timedelta(days=365)
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            max_value=datetime.now(),
            help="Backtest start date"
        )

        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            min_value=start_date,
            max_value=datetime.now(),
            help="Backtest end date"
        )

    with col3:
        rebalance_freq = st.selectbox(
            "Rebalancing",
            options=['never', 'monthly', 'quarterly', 'yearly'],
            index=0,
            help="How often to rebalance to target weights"
        )

        transaction_cost = st.number_input(
            "Transaction Cost (%)",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.01,
            help="% cost per trade (e.g., 0.1 = 0.1%)"
        )

    # ===================================================================
    # SECTION 3: Run Backtest
    # ===================================================================
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("🚀 Run Backtest", type="primary", width='stretch'):

            # Validate dates
            if start_date >= end_date:
                st.error("Start date must be before end date")
                return

            # Create config (use -1 for uploaded file portfolios since they don't have an ID)
            config = BacktestConfig(
                portfolio_id=-1 if source_type == "📄 Upload File (CSV/JSON)" else selected_id,
                portfolio_name=portfolio_name,
                start_date=datetime.combine(start_date, datetime.min.time()),
                end_date=datetime.combine(end_date, datetime.min.time()),
                initial_capital=initial_capital,
                benchmark=benchmark,
                rebalance_frequency=rebalance_freq,
                transaction_cost_pct=transaction_cost / 100
            )

            # Run backtest
            with st.spinner("Running backtest... This may take a minute..."):
                if source_type == "📄 Upload File (CSV/JSON)":
                    # Use the new method for uploaded portfolios
                    result = backtester.run_backtest_from_holdings(holdings_df, config)
                else:
                    result = backtester.run_backtest(config)

            if result:
                st.session_state.backtest_result = result
                st.session_state.backtest_config = config
                st.success("✅ Backtest complete!")
                st.rerun()

    # ===================================================================
    # SECTION 4: Display Results
    # ===================================================================

    if st.session_state.backtest_result is None:
        st.info("👆 Configure parameters above and click 'Run Backtest' to see results")
        return

    result = st.session_state.backtest_result
    config = st.session_state.backtest_config

    st.markdown("---")
    st.markdown("### 📊 Backtest Results")
    st.caption(f"{config.portfolio_name} • {config.start_date.strftime('%Y-%m-%d')} to {config.end_date.strftime('%Y-%m-%d')}")

    # ===================================================================
    # PERFORMANCE METRICS
    # ===================================================================
    st.markdown("#### 💰 Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Return",
            f"{result.total_return_pct:+.2f}%",
            delta=f"{result.total_return_pct - result.benchmark_total_return_pct:+.2f}% vs {config.benchmark}"
        )

        st.metric(
            "Annualized Return",
            f"{result.annualized_return_pct:+.2f}%",
            delta=f"{result.annualized_return_pct - result.benchmark_annualized_return_pct:+.2f}% vs {config.benchmark}"
        )

    with col2:
        st.metric(
            "Volatility (Annual)",
            f"{result.volatility_pct:.2f}%",
            delta=f"{result.volatility_pct - result.benchmark_volatility_pct:+.2f}% vs {config.benchmark}",
            delta_color="inverse"
        )

        st.metric(
            "Sharpe Ratio",
            f"{result.sharpe_ratio:.2f}",
            delta=f"{result.sharpe_ratio - result.benchmark_sharpe_ratio:+.2f} vs {config.benchmark}"
        )

    with col3:
        st.metric(
            "Max Drawdown",
            f"{result.max_drawdown_pct:.2f}%",
            delta=f"{result.max_drawdown_pct - result.benchmark_max_drawdown_pct:+.2f}% vs {config.benchmark}",
            delta_color="inverse"
        )

        st.metric(
            "Alpha",
            f"{result.alpha:+.2f}%",
            help="Excess return vs benchmark"
        )

    with col4:
        st.metric(
            "Beta",
            f"{result.beta:.2f}",
            help="Correlation with benchmark"
        )

        st.metric(
            "Rebalances",
            f"{result.num_rebalances}",
            help=f"Transaction costs: ${result.total_transaction_costs:,.2f}"
        )

    # ===================================================================
    # BENCHMARK COMPARISON TABLE
    # ===================================================================
    st.markdown("#### 📊 Portfolio vs Benchmark")

    comparison_data = {
        'Metric': [
            'Total Return',
            'Annualized Return',
            'Volatility',
            'Sharpe Ratio',
            'Max Drawdown',
            'Final Value'
        ],
        'Portfolio': [
            f"{result.total_return_pct:+.2f}%",
            f"{result.annualized_return_pct:+.2f}%",
            f"{result.volatility_pct:.2f}%",
            f"{result.sharpe_ratio:.2f}",
            f"{result.max_drawdown_pct:.2f}%",
            f"${result.portfolio_value.iloc[-1]:,.0f}"
        ],
        config.benchmark: [
            f"{result.benchmark_total_return_pct:+.2f}%",
            f"{result.benchmark_annualized_return_pct:+.2f}%",
            f"{result.benchmark_volatility_pct:.2f}%",
            f"{result.benchmark_sharpe_ratio:.2f}",
            f"{result.benchmark_max_drawdown_pct:.2f}%",
            f"${result.benchmark_value.iloc[-1]:,.0f}"
        ],
        'Difference': [
            f"{result.total_return_pct - result.benchmark_total_return_pct:+.2f}%",
            f"{result.annualized_return_pct - result.benchmark_annualized_return_pct:+.2f}%",
            f"{result.volatility_pct - result.benchmark_volatility_pct:+.2f}%",
            f"{result.sharpe_ratio - result.benchmark_sharpe_ratio:+.2f}",
            f"{result.max_drawdown_pct - result.benchmark_max_drawdown_pct:+.2f}%",
            f"${result.portfolio_value.iloc[-1] - result.benchmark_value.iloc[-1]:+,.0f}"
        ]
    }

    st.dataframe(
        pd.DataFrame(comparison_data),
        width='stretch',
        hide_index=True
    )

    # ===================================================================
    # PERFORMANCE CHART
    # ===================================================================
    st.markdown("#### 📈 Portfolio Performance Over Time")

    fig = create_performance_chart(result, config)
    st.plotly_chart(fig, width='stretch')

    # ===================================================================
    # ADDITIONAL CHARTS
    # ===================================================================
    st.markdown("#### 📉 Risk Analysis")

    tab1, tab2, tab3 = st.tabs(["Drawdown", "Returns Distribution", "Monthly Returns"])

    with tab1:
        dd_fig = create_drawdown_chart(result)
        st.plotly_chart(dd_fig, width='stretch')

        st.caption("""
        **Drawdown** shows the peak-to-trough decline during the backtest period.
        Lower drawdown indicates better downside protection.
        """)

    with tab2:
        dist_fig = create_returns_distribution(result)
        st.plotly_chart(dist_fig, width='stretch')

        st.caption("""
        **Returns Distribution** shows the frequency of daily returns.
        Wider distribution = higher volatility.
        """)

    with tab3:
        # Monthly returns table
        monthly_returns = result.portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        monthly_bench = result.benchmark_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100

        monthly_df = pd.DataFrame({
            'Month': monthly_returns.index.strftime('%Y-%m'),
            'Portfolio': monthly_returns.values,
            config.benchmark: monthly_bench.values,
            'Outperformance': (monthly_returns - monthly_bench).values
        })

        st.dataframe(
            monthly_df.style.format({
                'Portfolio': '{:+.2f}%',
                config.benchmark: '{:+.2f}%',
                'Outperformance': '{:+.2f}%'
            }),
            width='stretch',
            hide_index=True,
            height=400
        )

    # ===================================================================
    # EXPORT OPTIONS
    # ===================================================================
    st.markdown("---")
    st.markdown("#### 💾 Export Results")

    # Prepare comprehensive time series data
    timeseries_df = pd.DataFrame({
        'Date': result.portfolio_value.index.strftime('%Y-%m-%d'),
        'Portfolio_Value': result.portfolio_value.values,
        'Benchmark_Value': result.benchmark_value.values,
        'Portfolio_Daily_Return': result.portfolio_returns.values,
        'Benchmark_Daily_Return': result.benchmark_returns.values,
    })

    # Add cumulative returns
    timeseries_df['Portfolio_Cumulative_Return'] = (1 + result.portfolio_returns).cumprod().values - 1
    timeseries_df['Benchmark_Cumulative_Return'] = (1 + result.benchmark_returns).cumprod().values - 1
    timeseries_df['Excess_Return'] = timeseries_df['Portfolio_Cumulative_Return'] - timeseries_df['Benchmark_Cumulative_Return']

    # Add drawdown
    port_cumulative = (1 + result.portfolio_returns).cumprod()
    port_running_max = port_cumulative.expanding().max()
    timeseries_df['Portfolio_Drawdown'] = ((port_cumulative - port_running_max) / port_running_max).values

    bench_cumulative = (1 + result.benchmark_returns).cumprod()
    bench_running_max = bench_cumulative.expanding().max()
    timeseries_df['Benchmark_Drawdown'] = ((bench_cumulative - bench_running_max) / bench_running_max).values

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Export to CSV
        csv = timeseries_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"{config.portfolio_name.replace(' ', '_')}_backtest_{config.start_date.strftime('%Y%m%d')}.csv",
            "text/csv",
            width='stretch'
        )

    with col2:
        # Export to Excel
        try:
            from io import BytesIO

            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                # Time series sheet
                timeseries_df.to_excel(writer, sheet_name='Time_Series', index=False)

                # Summary metrics sheet
                summary_data = {
                    'Metric': [
                        'Portfolio Name',
                        'Benchmark',
                        'Start Date',
                        'End Date',
                        'Initial Capital',
                        'Final Portfolio Value',
                        'Final Benchmark Value',
                        'Total Return (%)',
                        'Benchmark Total Return (%)',
                        'Excess Return (%)',
                        'Annualized Return (%)',
                        'Benchmark Annualized Return (%)',
                        'Volatility (%)',
                        'Benchmark Volatility (%)',
                        'Sharpe Ratio',
                        'Benchmark Sharpe Ratio',
                        'Max Drawdown (%)',
                        'Benchmark Max Drawdown (%)',
                        'Alpha (%)',
                        'Beta',
                        'Rebalances',
                        'Transaction Costs ($)'
                    ],
                    'Value': [
                        config.portfolio_name,
                        config.benchmark,
                        config.start_date.strftime('%Y-%m-%d'),
                        config.end_date.strftime('%Y-%m-%d'),
                        config.initial_capital,
                        result.portfolio_value.iloc[-1],
                        result.benchmark_value.iloc[-1],
                        round(result.total_return_pct, 2),
                        round(result.benchmark_total_return_pct, 2),
                        round(result.total_return_pct - result.benchmark_total_return_pct, 2),
                        round(result.annualized_return_pct, 2),
                        round(result.benchmark_annualized_return_pct, 2),
                        round(result.volatility_pct, 2),
                        round(result.benchmark_volatility_pct, 2),
                        round(result.sharpe_ratio, 2),
                        round(result.benchmark_sharpe_ratio, 2),
                        round(result.max_drawdown_pct, 2),
                        round(result.benchmark_max_drawdown_pct, 2),
                        round(result.alpha, 2),
                        round(result.beta, 2),
                        result.num_rebalances,
                        round(result.total_transaction_costs, 2)
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

                # Monthly returns sheet
                monthly_returns = result.portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                monthly_bench = result.benchmark_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                monthly_df = pd.DataFrame({
                    'Month': monthly_returns.index.strftime('%Y-%m'),
                    'Portfolio_Return': monthly_returns.values,
                    'Benchmark_Return': monthly_bench.values,
                    'Excess_Return': (monthly_returns - monthly_bench).values
                })
                monthly_df.to_excel(writer, sheet_name='Monthly_Returns', index=False)

            excel_buffer.seek(0)

            st.download_button(
                "📥 Download Excel",
                excel_buffer,
                f"{config.portfolio_name.replace(' ', '_')}_backtest_{config.start_date.strftime('%Y%m%d')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width='stretch'
            )
        except ImportError:
            st.button("📥 Excel (install openpyxl)", disabled=True, width='stretch')

    with col3:
        # Export metrics as JSON
        import json

        metrics_dict = {
            'portfolio_name': config.portfolio_name,
            'benchmark': config.benchmark,
            'start_date': config.start_date.strftime('%Y-%m-%d'),
            'end_date': config.end_date.strftime('%Y-%m-%d'),
            'initial_capital': config.initial_capital,
            'final_value': round(result.portfolio_value.iloc[-1], 2),
            'total_return_pct': round(result.total_return_pct, 2),
            'annualized_return_pct': round(result.annualized_return_pct, 2),
            'volatility_pct': round(result.volatility_pct, 2),
            'sharpe_ratio': round(result.sharpe_ratio, 2),
            'max_drawdown_pct': round(result.max_drawdown_pct, 2),
            'alpha': round(result.alpha, 2),
            'beta': round(result.beta, 2),
            'benchmark_total_return_pct': round(result.benchmark_total_return_pct, 2),
            'excess_return_pct': round(result.total_return_pct - result.benchmark_total_return_pct, 2),
            'num_rebalances': result.num_rebalances,
            'transaction_costs': round(result.total_transaction_costs, 2)
        }

        json_str = json.dumps(metrics_dict, indent=2)

        st.download_button(
            "📥 Download JSON",
            json_str,
            f"{config.portfolio_name.replace(' ', '_')}_metrics.json",
            "application/json",
            width='stretch'
        )

    with col4:
        if st.button("🔄 New Backtest", width='stretch'):
            st.session_state.backtest_result = None
            st.session_state.backtest_config = None
            st.rerun()


# Export for main app
__all__ = ['render_backtest_tab']