"""
Signals Tab - Table View

Signals table rendering, prediction tracker, smart refresh, batch analysis.
"""
from .analysis import _run_single_analysis
from .deep_dive import _render_deep_dive
from .earnings_views import _display_news_items
from .shared import (
    st, pd, np, logger, datetime, date, timedelta, time, json,
    Dict, List, Optional, _to_native, _get_ai_probabilities_for_table, text,
    get_engine, get_connection,
    SIGNAL_HUB_AVAILABLE, DB_AVAILABLE, AI_SYSTEM_AVAILABLE,
    ENHANCED_SCORING_AVAILABLE, ENHANCED_SCORES_DB_AVAILABLE,
    DUAL_ANALYST_AVAILABLE, DualAnalystService,
    INSTITUTIONAL_SIGNALS_AVAILABLE,
    UnifiedSignal, AI_PROB_AVAILABLE,
)

from .job_manager import (
    _get_job_status, _ensure_job_tables, _start_job, _stop_job,
    _complete_job, _reset_job, _get_job_log, _get_job_stats, _get_processed_tickers,
)

if SIGNAL_HUB_AVAILABLE:
    from .shared import get_signal_engine, get_market_overview, SignalStrength, RiskLevel

if ENHANCED_SCORING_AVAILABLE:
    from .shared import apply_enhanced_scores_to_dataframe

def _test_ai_connection():
    """Test if AI chat assistant is working."""
    st.markdown("### 🔧 AI Connection Test")

    with st.spinner("Testing AI connection..."):
        try:
            # Test direct OpenAI client (what batch analysis uses)
            st.write("Testing direct OpenAI client connection...")

            from openai import OpenAI
            import os
            from dotenv import load_dotenv

            load_dotenv()

            base_url = os.getenv("LLM_QWEN_BASE_URL", "http://172.23.193.91:8090/v1")
            model = os.getenv("LLM_QWEN_MODEL", "Qwen3-32B-Q6_K.gguf")

            st.write(f"   - Base URL: {base_url}")
            st.write(f"   - Model: {model}")

            client = OpenAI(
                base_url=base_url,
                api_key="not-needed"
            )

            st.write("✅ OpenAI client created")

            # Try a simple test message
            st.write("📡 Sending test message...")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say 'AI connection OK' and nothing else."}
                ],
                temperature=0.1,
                max_tokens=50
            )

            result = response.choices[0].message.content
            st.write(f"✅ Response received: {result}")

            # Also test AlphaChat if available
            st.markdown("---")
            st.write("Testing AlphaChat (for comparison)...")
            try:
                from src.ai.chat import AlphaChat
                assistant = AlphaChat()
                st.write(f"✅ AlphaChat created, available={assistant.available}")
            except Exception as e:
                st.warning(f"AlphaChat not available: {e}")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())


def _test_single_ticker_analysis():
    """Test AI analysis on a single ticker with verbose output."""
    st.markdown("### 🔧 Single Ticker Test (AAPL)")

    ticker = "AAPL"  # Use a common ticker for testing

    with st.spinner(f"Testing analysis for {ticker}..."):
        try:
            from src.ai.batch_ai_analysis import BatchAIAnalyzer, STRUCTURED_ANALYSIS_PROMPT

            st.write("✅ BatchAIAnalyzer imported")

            analyzer = BatchAIAnalyzer()
            st.write("✅ Analyzer created")

            # Test context building
            st.write(f"📄 Building context for {ticker}...")
            context = analyzer._build_context_for_ticker(ticker)

            if context:
                st.write(f"✅ Context built, length: {len(context)} chars")
                with st.expander("View Context", expanded=False):
                    st.code(context[:2000] + "..." if len(context) > 2000 else context)
            else:
                st.error("❌ Context is empty!")
                return

            # Test direct AI call
            st.write("🤖 Testing direct AI call...")

            prompt = STRUCTURED_ANALYSIS_PROMPT.format(context=context)
            st.write(f"📝 Prompt length: {len(prompt)} chars")

            # Call AI directly
            st.write("📡 Calling AI (direct)...")
            response = analyzer._call_ai_direct(prompt)

            if response:
                st.write(f"✅ Response received, length: {len(response)} chars")
                with st.expander("View Raw Response", expanded=True):
                    st.code(response[:3000] + "..." if len(response) > 3000 else response)

                # Try to parse
                st.write("🔍 Parsing JSON...")
                parsed = analyzer._parse_json_response(response, ticker)

                if parsed:
                    st.write("✅ JSON parsed successfully!")
                    st.json(parsed)
                else:
                    st.error("❌ Failed to parse JSON from response")
            else:
                st.error("❌ Empty response from AI!")

        except ImportError as e:
            st.error(f"❌ Import error: {e}")
            st.info("Make sure batch_ai_analysis.py is in src/ai/ folder")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())


def _render_prediction_tracker():
    """
    Render the Prediction Tracker dashboard showing ML learning progress.
    """
    st.markdown("""
    **Monitor your ML system's learning progress.** Track predictions, outcomes, and accuracy over time.
    """)

    try:
        from src.ml.prediction_tracker import (
            get_prediction_stats,
            get_recent_predictions,
            update_outcomes,
            ensure_predictions_table,
            reset_predictions_table
        )

        # Ensure table exists
        ensure_predictions_table()

        # Get stats
        stats = get_prediction_stats()

        # =====================================================================
        # ML PROGRESS SECTION
        # =====================================================================
        st.markdown("### 🎯 ML Reliability Progress")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Predictions", stats.total_predictions)
        with col2:
            st.metric("With Outcomes", stats.predictions_with_outcome)
        with col3:
            st.metric("Pending", stats.pending_outcomes)
        with col4:
            # Color code ML status
            status_colors = {
                "BLOCKED": "🔴",
                "DEGRADED": "🟡",
                "TRADABLE": "🟢"
            }
            status_icon = status_colors.get(stats.ml_gate_status, "⚪")
            st.metric("ML Status", f"{status_icon} {stats.ml_gate_status}")

        # Progress bars
        st.markdown("#### Progress to ML Reliability Thresholds")

        col1, col2 = st.columns(2)

        with col1:
            progress_to_degraded = min(stats.samples_for_ml / 40, 1.0)
            st.progress(progress_to_degraded, text=f"DEGRADED: {stats.samples_for_ml}/40 samples")
            if stats.samples_to_degraded > 0:
                st.caption(f"Need {stats.samples_to_degraded} more samples")
            else:
                st.caption("✅ Threshold reached!")

        with col2:
            progress_to_tradable = min(stats.samples_for_ml / 80, 1.0)
            st.progress(progress_to_tradable, text=f"TRADABLE: {stats.samples_for_ml}/80 samples")
            if stats.samples_to_tradable > 0:
                st.caption(f"Need {stats.samples_to_tradable} more samples + >55% accuracy")
            else:
                if stats.direction_accuracy >= 0.55:
                    st.caption("✅ Threshold reached!")
                else:
                    st.caption(f"⚠️ Accuracy {stats.direction_accuracy:.1%} < 55% required")

        # =====================================================================
        # ACCURACY METRICS
        # =====================================================================
        if stats.predictions_with_outcome > 0:
            st.markdown("### 📈 Accuracy Metrics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                acc_color = "🟢" if stats.direction_accuracy >= 0.55 else "🔴"
                st.metric("Direction Accuracy", f"{acc_color} {stats.direction_accuracy:.1%}")
            with col2:
                st.metric("Win Rate", f"{stats.win_rate_overall:.1%}")
            with col3:
                st.metric("Mean Abs Error", f"{stats.mean_absolute_error:.2%}")
            with col4:
                st.metric("Last 30d Accuracy", f"{stats.accuracy_last_30d:.1%}" if stats.accuracy_last_30d else "N/A")

        # =====================================================================
        # UPDATE OUTCOMES BUTTON
        # =====================================================================
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("🔄 Update Outcomes", key="update_outcomes_btn"):
                with st.spinner("Fetching actual returns..."):
                    updated = update_outcomes(days_back=30)
                    if updated > 0:
                        st.success(f"✅ Updated {updated} predictions with actual outcomes!")
                        st.rerun()
                    else:
                        st.info("No predictions ready for outcome update yet.")

        with col2:
            if st.button("🔧 Fix/Reset Table", key="reset_pred_table_btn"):
                with st.spinner("Resetting predictions table..."):
                    if reset_predictions_table():
                        st.success("✅ Table reset successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to reset table")

        with col3:
            st.caption("🔄 Updates outcomes from Yahoo Finance | 🔧 Resets table if schema issues")

        # =====================================================================
        # RECENT PREDICTIONS TABLE
        # =====================================================================
        st.markdown("### 📋 Recent Predictions")

        recent_df = get_recent_predictions(limit=20)

        if recent_df.empty:
            st.info("No predictions recorded yet. Predictions are saved when you analyze stocks with the Alpha Model.")
        else:
            # Format for display
            display_df = recent_df.copy()

            # Format columns
            if 'prediction_date' in display_df.columns:
                display_df['prediction_date'] = pd.to_datetime(display_df['prediction_date']).dt.strftime('%m-%d')

            if 'predicted_return_5d' in display_df.columns:
                display_df['predicted_return_5d'] = display_df['predicted_return_5d'].apply(
                    lambda x: f"{x:+.2%}" if pd.notna(x) else "N/A"
                )

            if 'actual_return_5d' in display_df.columns:
                display_df['actual_return_5d'] = display_df['actual_return_5d'].apply(
                    lambda x: f"{x:+.2%}" if pd.notna(x) else "Pending"
                )

            if 'predicted_probability' in display_df.columns:
                display_df['predicted_probability'] = display_df['predicted_probability'].apply(
                    lambda x: f"{x:.0%}" if pd.notna(x) else "N/A"
                )

            if 'price_at_prediction' in display_df.columns:
                display_df['price_at_prediction'] = display_df['price_at_prediction'].apply(
                    lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
                )

            # Rename columns
            col_rename = {
                'ticker': 'Ticker',
                'prediction_date': 'Date',
                'alpha_signal': 'Signal',
                'predicted_return_5d': 'Pred 5d',
                'predicted_probability': 'P(Win)',
                'actual_return_5d': 'Actual 5d',
                'result': 'Result'
            }
            display_df = display_df.rename(columns={k: v for k, v in col_rename.items() if k in display_df.columns})

            # Select columns to show
            show_cols = ['Ticker', 'Date', 'Signal', 'Pred 5d', 'P(Win)', 'Actual 5d', 'Result']
            show_cols = [c for c in show_cols if c in display_df.columns]
            display_df = display_df[show_cols]

            # Color code results
            def color_result(val):
                if val == 'WIN':
                    return 'background-color: #69F0AE; color: black'
                elif val == 'LOSS':
                    return 'background-color: #FF8A80; color: black'
                elif val == 'PENDING':
                    return 'background-color: #FFE082; color: black'
                return ''

            styled = display_df.style
            if 'Result' in display_df.columns:
                styled = styled.applymap(color_result, subset=['Result'])

            st.dataframe(styled, width='stretch', height=300)

            # Summary
            if 'result' in recent_df.columns:
                wins = len(recent_df[recent_df['result'] == 'WIN'])
                losses = len(recent_df[recent_df['result'] == 'LOSS'])
                pending = len(recent_df[recent_df['result'] == 'PENDING'])
                st.caption(f"Last 20: 🟢 {wins} wins | 🔴 {losses} losses | 🟡 {pending} pending")

    except ImportError as e:
        st.warning(f"Prediction Tracker module not found: {e}")
        st.info("Place `prediction_tracker.py` in `src/ml/` folder")
    except Exception as e:
        st.error(f"Error loading prediction tracker: {e}")
        import traceback
        st.code(traceback.format_exc())


def _render_smart_refresh_status():
    """
    Render the Smart Refresh status showing data freshness.
    """
    st.markdown("""
    **Avoid duplicate work!** Only refresh data that's actually stale.
    Different data types have different refresh intervals based on how often they change.
    """)

    try:
        from src.core.smart_refresh import (
            SmartScanner, DataType, RefreshConfig,
            get_refresh_stats, reset_freshness
        )

        scanner = SmartScanner()
        stats = scanner.get_status_summary()

        # =====================================================================
        # OVERVIEW METRICS
        # =====================================================================
        st.markdown("### 📊 Data Freshness Overview")

        col1, col2, col3, col4 = st.columns(4)

        total = stats.get('total_entries', 0)
        fresh = stats.get('fresh_count', 0)
        stale = stats.get('stale_count', 0)

        with col1:
            st.metric("Total Tracked", total)
        with col2:
            st.metric("🟢 Fresh", fresh)
        with col3:
            st.metric("🟡 Stale", stale)
        with col4:
            pct = (fresh / total * 100) if total > 0 else 0
            st.metric("Fresh %", f"{pct:.0f}%")

        # =====================================================================
        # REFRESH INTERVALS
        # =====================================================================
        st.markdown("### ⏱️ Refresh Intervals")

        config = RefreshConfig()

        intervals_data = []
        for dt in DataType:
            ttl_hours = config.ttl_hours.get(dt, 24)
            type_stats = stats.get('by_type', {}).get(dt.value, {})

            if ttl_hours >= 168:
                interval_str = f"{ttl_hours // 24} days"
            elif ttl_hours >= 24:
                interval_str = f"{ttl_hours // 24} day{'s' if ttl_hours >= 48 else ''}"
            else:
                interval_str = f"{ttl_hours} hours"

            intervals_data.append({
                "Data Type": dt.value.replace('_', ' ').title(),
                "Interval": interval_str,
                "Fresh": type_stats.get('fresh', 0),
                "Stale": type_stats.get('stale', 0),
            })

        intervals_df = pd.DataFrame(intervals_data)

        # Color code the rows
        def color_stale(val):
            if isinstance(val, (int, float)) and val > 0:
                return 'background-color: #FFCDD2'
            return ''

        styled = intervals_df.style.applymap(color_stale, subset=['Stale'])
        st.dataframe(styled, width='stretch', hide_index=True)

        # =====================================================================
        # EXPLANATION
        # =====================================================================
        st.markdown("### 💡 How It Works")

        st.markdown("""
        | Data Type | Why This Interval |
        |-----------|-------------------|
        | **News/Sentiment** | Can change rapidly, especially near earnings |
        | **Options Flow** | Intraday shifts, refreshed every 4 hours |
        | **Technical** | Based on daily prices, refreshed daily |
        | **Fundamentals** | Only changes on earnings, refreshed weekly |
        | **Insider/13F** | SEC filings are slow, refreshed weekly |
        
        **Near Earnings**: When a stock is within 7 days of earnings, news/sentiment/options refresh 4x more frequently!
        """)

        # =====================================================================
        # CONTROLS
        # =====================================================================
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("🔄 Force Full Refresh", key="force_refresh_btn"):
                with st.spinner("Resetting freshness tracking..."):
                    reset_freshness()
                    st.success("✅ All data marked for refresh on next scan!")
                    st.rerun()

        with col2:
            refresh_type = st.selectbox(
                "Reset specific type",
                options=[""] + [dt.value for dt in DataType],
                key="reset_type_select"
            )
            if refresh_type and st.button("Reset Type", key="reset_type_btn"):
                dt = DataType(refresh_type)
                reset_freshness(data_types=[dt])
                st.success(f"✅ {refresh_type} marked for refresh!")

        with col3:
            st.caption("""
            **Force Full Refresh**: Clears all freshness tracking - everything will refresh on next scan.
            **Reset Type**: Only reset a specific data type (e.g., news only).
            """)

    except ImportError as e:
        st.warning(f"Smart Refresh module not found: {e}")
        st.info("Place `smart_refresh.py` in `src/core/` folder")
    except Exception as e:
        st.error(f"Error loading smart refresh: {e}")
        import traceback
        st.code(traceback.format_exc())


def _render_ai_batch_analysis(filtered_df: pd.DataFrame):
    """
    Render the AI Batch Analysis UI section.
    Allows running AI analysis on top N signals and displays results.
    """

    # Initialize session state
    if 'ai_batch_running' not in st.session_state:
        st.session_state.ai_batch_running = False
    if 'ai_batch_progress' not in st.session_state:
        st.session_state.ai_batch_progress = {'current': 0, 'total': 0, 'ticker': ''}
    if 'ai_batch_results' not in st.session_state:
        st.session_state.ai_batch_results = None

    st.markdown("""
    **Run AI deep analysis on your top signals.** The AI will analyze each ticker and provide:
    - Action recommendation (BUY/SELL/HOLD)
    - Confidence level and position sizing
    - Entry/Exit prices and stop loss
    - Bull/Bear case and key risks
    """)

    # Speed mode info
    st.info("""💡 **Fast Mode Enabled**: Using cached data only (no API calls).  
    ⏱️ Estimated time: ~5-10 seconds per ticker (AI processing only).  
    📊 For 10 stocks ≈ 1-2 minutes | For 20 stocks ≈ 2-4 minutes""")

    # Controls row
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        top_n = st.selectbox(
            "Analyze Top N",
            options=[5, 10, 15, 20, 30],
            index=1,  # Default to 10
            key="ai_batch_top_n"
        )

    with col2:
        sort_options = ['Total', 'Sentiment', 'OptFlow', 'Fundamental', 'Technical']
        sort_by = st.selectbox(
            "Sort By",
            options=sort_options,
            key="ai_batch_sort_by"
        )

    with col3:
        filter_options = ['All', 'BUY Only', 'SELL Only', 'BUY+SELL']
        signal_filter = st.selectbox(
            "Signal Filter",
            options=filter_options,
            key="ai_batch_signal_filter"
        )

    with col4:
        st.write("")  # Spacer
        st.write("")
        run_clicked = st.button(
            "🤖 Run AI Analysis",
            type="primary",
            width='stretch',
            disabled=st.session_state.ai_batch_running,
            key="ai_batch_run_btn"
        )

    # Debug button row
    debug_col1, debug_col2 = st.columns(2)
    with debug_col1:
        if st.button("🔧 Test AI Connection", key="ai_test_connection"):
            _test_ai_connection()
    with debug_col2:
        if st.button("🔧 Test Single Ticker", key="ai_test_single"):
            _test_single_ticker_analysis()

    # Process if button clicked
    if run_clicked and not st.session_state.ai_batch_running:
        st.session_state.ai_batch_running = True
        st.session_state.ai_batch_results = None

        # Get tickers to analyze
        df_to_analyze = filtered_df.copy()

        # Apply signal filter
        if signal_filter == 'BUY Only' and 'Signal' in df_to_analyze.columns:
            df_to_analyze = df_to_analyze[df_to_analyze['Signal'].str.contains('BUY', na=False)]
        elif signal_filter == 'SELL Only' and 'Signal' in df_to_analyze.columns:
            df_to_analyze = df_to_analyze[df_to_analyze['Signal'].str.contains('SELL', na=False)]
        elif signal_filter == 'BUY+SELL' and 'Signal' in df_to_analyze.columns:
            df_to_analyze = df_to_analyze[
                df_to_analyze['Signal'].str.contains('BUY', na=False) |
                df_to_analyze['Signal'].str.contains('SELL', na=False)
            ]

        # Sort
        if sort_by in df_to_analyze.columns:
            df_to_analyze = df_to_analyze.sort_values(sort_by, ascending=False)

        # Get top N tickers
        tickers = df_to_analyze.head(top_n)['Ticker'].tolist() if 'Ticker' in df_to_analyze.columns else []

        if not tickers:
            st.warning("No tickers match the filter criteria")
            st.session_state.ai_batch_running = False
        else:
            # Build signal data dict
            signal_data = {}
            for _, row in df_to_analyze.iterrows():
                ticker = row.get('Ticker')
                if ticker:
                    signal_data[ticker] = {
                        # FIX: Use None for missing scores, not 50
                        'total_score': row.get('Total') if pd.notna(row.get('Total')) else None,
                        'signal_type': row.get('Signal', 'HOLD'),
                        'technical_score': row.get('Technical') if pd.notna(row.get('Technical')) else None,
                        'fundamental_score': row.get('Fundamental') if pd.notna(row.get('Fundamental')) else None,
                        'sentiment_score': row.get('Sentiment') if pd.notna(row.get('Sentiment')) else None,
                    }

            # Run analysis
            try:
                from src.ai.batch_ai_analysis import BatchAIAnalyzer
                import time

                analyzer = BatchAIAnalyzer()
                results = []
                errors = []

                progress_bar = st.progress(0, text="Starting AI analysis...")
                status_text = st.empty()
                eta_text = st.empty()
                error_container = st.empty()

                start_time = time.time()

                for i, ticker in enumerate(tickers):
                    ticker_start = time.time()

                    # Calculate ETA
                    elapsed = time.time() - start_time
                    if i > 0:
                        avg_per_ticker = elapsed / i
                        remaining = (len(tickers) - i) * avg_per_ticker
                        if remaining < 60:
                            eta_str = f"~{remaining:.0f} seconds remaining"
                        else:
                            eta_str = f"~{remaining/60:.1f} minutes remaining"
                    else:
                        eta_str = "Calculating ETA..."

                    progress = (i + 1) / len(tickers)
                    progress_bar.progress(progress, text=f"Analyzing {ticker}... ({i+1}/{len(tickers)})")
                    status_text.info(f"🤖 Processing: {ticker}")
                    eta_text.caption(f"⏱️ {eta_str} | Elapsed: {elapsed:.0f}s")

                    try:
                        # Use fast_mode=True for DB-only context (much faster)
                        result = analyzer.analyze_ticker(ticker, signal_data.get(ticker, {}), fast_mode=True)
                        ticker_elapsed = time.time() - ticker_start

                        if result:
                            save_ok = analyzer.save_result(result)
                            if save_ok:
                                results.append(result)
                                logger.info(f"✅ {ticker}: {result.ai_action} ({result.ai_confidence}) - {ticker_elapsed:.1f}s")
                            else:
                                errors.append(f"{ticker}: Failed to save")
                        else:
                            errors.append(f"{ticker}: No result returned")
                            logger.warning(f"❌ {ticker}: analyze_ticker returned None - {ticker_elapsed:.1f}s")
                    except Exception as e:
                        error_msg = f"{ticker}: {str(e)}"
                        errors.append(error_msg)
                        logger.error(f"Error analyzing {ticker}: {e}")
                        continue

                total_time = time.time() - start_time
                progress_bar.progress(1.0, text="Complete!")
                eta_text.empty()

                if results:
                    status_text.success(f"✅ Analyzed {len(results)} of {len(tickers)} tickers in {total_time:.0f}s")
                else:
                    status_text.warning(f"⚠️ No results obtained from {len(tickers)} tickers")

                # Show errors if any
                if errors:
                    with error_container.expander(f"⚠️ {len(errors)} errors occurred", expanded=False):
                        for err in errors:
                            st.text(err)

                st.session_state.ai_batch_results = results
                st.session_state.ai_batch_running = False

                # Force refresh to show results
                if results:
                    st.rerun()

            except ImportError as e:
                st.error(f"❌ Batch AI Analysis module not available: {e}")
                st.info("Make sure `batch_ai_analysis.py` is in `src/ai/` folder")
                st.session_state.ai_batch_running = False
            except Exception as e:
                st.error(f"❌ Analysis error: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.session_state.ai_batch_running = False

    # Display results
    st.markdown("---")
    st.markdown("### 📊 AI Analysis Results")

    # Load from database
    try:
        from src.ai.batch_ai_analysis import get_ai_analysis_for_display
        ai_df = get_ai_analysis_for_display()

        if ai_df.empty:
            st.info("No AI analysis results yet. Click **Run AI Analysis** to analyze top signals.")
        else:
            # Format for display
            display_cols = {
                'ticker': 'Ticker',
                'ai_action': 'AI Action',
                'ai_confidence': 'Confidence',
                'trade_allowed': 'Trade OK',
                'entry_price': 'Entry',
                'stop_loss': 'Stop',
                'target_price': 'Target',
                'position_size': 'Size',
                'time_horizon': 'Horizon',
                'platform_score': 'Score',
                'analysis_date': 'Analyzed'
            }

            # Select and rename columns
            available_cols = [c for c in display_cols.keys() if c in ai_df.columns]
            ai_display = ai_df[available_cols].copy()
            ai_display = ai_display.rename(columns=display_cols)

            # Format date
            if 'Analyzed' in ai_display.columns:
                ai_display['Analyzed'] = pd.to_datetime(ai_display['Analyzed']).dt.strftime('%m-%d %H:%M')

            # Format prices
            for col in ['Entry', 'Stop', 'Target']:
                if col in ai_display.columns:
                    ai_display[col] = ai_display[col].apply(
                        lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
                    )

            # Color code AI Action
            def color_action(val):
                if pd.isna(val):
                    return ''
                val = str(val).upper()
                if 'BUY' in val:
                    return 'background-color: #69F0AE; color: black'
                elif 'SELL' in val:
                    return 'background-color: #FF8A80; color: black'
                elif 'WAIT' in val:
                    return 'background-color: #FFE082; color: black'
                return ''

            def color_confidence(val):
                if pd.isna(val):
                    return ''
                val = str(val).upper()
                if val == 'HIGH':
                    return 'background-color: #00C853; color: white'
                elif val == 'MEDIUM':
                    return 'background-color: #FFD54F; color: black'
                return 'background-color: #BDBDBD'

            def color_trade_ok(val):
                if val == True:
                    return 'background-color: #69F0AE'
                elif val == False:
                    return 'background-color: #FF8A80'
                return ''

            # Apply styling
            styled = ai_display.style
            if 'AI Action' in ai_display.columns:
                styled = styled.applymap(color_action, subset=['AI Action'])
            if 'Confidence' in ai_display.columns:
                styled = styled.applymap(color_confidence, subset=['Confidence'])
            if 'Trade OK' in ai_display.columns:
                styled = styled.applymap(color_trade_ok, subset=['Trade OK'])

            # Stats row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                buy_count = len(ai_df[ai_df['ai_action'] == 'BUY']) if 'ai_action' in ai_df.columns else 0
                st.metric("🟢 BUY", buy_count)
            with col2:
                sell_count = len(ai_df[ai_df['ai_action'] == 'SELL']) if 'ai_action' in ai_df.columns else 0
                st.metric("🔴 SELL", sell_count)
            with col3:
                hold_count = len(ai_df[ai_df['ai_action'] == 'HOLD']) if 'ai_action' in ai_df.columns else 0
                st.metric("🟡 HOLD", hold_count)
            with col4:
                high_conf = len(ai_df[ai_df['ai_confidence'] == 'HIGH']) if 'ai_confidence' in ai_df.columns else 0
                st.metric("⭐ High Conf", high_conf)

            # Display table
            st.dataframe(styled, width='stretch', height=300)

            # Filter controls for AI results
            col1, col2 = st.columns(2)
            with col1:
                ai_action_filter = st.multiselect(
                    "Filter by AI Action",
                    options=['BUY', 'SELL', 'HOLD', 'WAIT_FOR_PULLBACK'],
                    key="ai_result_action_filter"
                )
            with col2:
                ai_conf_filter = st.multiselect(
                    "Filter by Confidence",
                    options=['HIGH', 'MEDIUM', 'LOW'],
                    key="ai_result_conf_filter"
                )

            # Show filtered actionable signals
            if ai_action_filter or ai_conf_filter:
                filtered_ai = ai_df.copy()
                if ai_action_filter:
                    filtered_ai = filtered_ai[filtered_ai['ai_action'].isin(ai_action_filter)]
                if ai_conf_filter:
                    filtered_ai = filtered_ai[filtered_ai['ai_confidence'].isin(ai_conf_filter)]

                if not filtered_ai.empty:
                    st.markdown(f"**Filtered Results: {len(filtered_ai)} signals**")
                    st.dataframe(filtered_ai[['ticker', 'ai_action', 'ai_confidence', 'entry_price', 'target_price']])

    except ImportError:
        st.warning("AI Batch Analysis module not installed. Place `batch_ai_analysis.py` in `src/ai/` folder.")
    except Exception as e:
        st.error(f"Error loading AI results: {e}")
        logger.error(f"AI results display error: {e}")


def _render_signals_table_view():
    """Render the signals table (Universe-style) with dropdown for deep dive."""

    # =========================================================================
    # MARKET OVERVIEW (SPY, QQQ, VIX)
    # =========================================================================
    try:
        overview = get_market_overview()
        if overview:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                c = "🟢" if overview.spy_change >= 0 else "🔴"
                st.metric("SPY", f"{c} {overview.spy_change:+.2f}%")
            with col2:
                c = "🟢" if overview.qqq_change >= 0 else "🔴"
                st.metric("QQQ", f"{c} {overview.qqq_change:+.2f}%")
            with col3:
                st.metric("VIX", f"{overview.vix:.1f}")
            with col4:
                st.metric("Regime", overview.regime)
    except Exception as e:
        logger.debug(f"Could not load market overview: {e}")

    # =========================================================================
    # LOAD DATA FROM DATABASE
    # =========================================================================
    try:
        engine = get_engine()
        df = pd.read_sql("""
                    WITH latest_prices AS (
                        SELECT DISTINCT ON (ticker) ticker, close as price
                        FROM prices
                        ORDER BY ticker, date DESC
                    ),
                    latest_fundamentals AS (
                        SELECT DISTINCT ON (ticker) *
                        FROM fundamentals
                        ORDER BY ticker, date DESC
                    ),
                    latest_analyst_ratings AS (
                        SELECT DISTINCT ON (ticker) *
                        FROM analyst_ratings
                        ORDER BY ticker, date DESC
                    ),
                    latest_price_targets AS (
                        SELECT DISTINCT ON (ticker) *
                        FROM price_targets
                        ORDER BY ticker, date DESC
                    )
                    SELECT 
                        s.ticker,
                        s.date as score_date,
                        f.sector,
                        f.market_cap,
                        CASE 
                            WHEN s.total_score >= 80 THEN 'STRONG BUY'
                            WHEN s.total_score >= 65 THEN 'BUY'
                            WHEN s.total_score >= 55 THEN 'WEAK BUY'
                            WHEN s.total_score <= 20 THEN 'STRONG SELL'
                            WHEN s.total_score <= 35 THEN 'SELL'
                            WHEN s.total_score <= 45 THEN 'WEAK SELL'
                            ELSE 'HOLD'
                        END as signal_type,
                        s.total_score,
                        s.sentiment_score,
                        s.options_flow_score,
                        s.short_squeeze_score,
                        s.fundamental_score,
                        s.growth_score,
                        s.dividend_score,
                        s.technical_score,
                        s.gap_score,
                        CASE 
                            WHEN s.gap_score >= 70 THEN 'BEARISH'
                            WHEN s.gap_score <= 30 THEN 'BULLISH'
                            ELSE 'None'
                        END as gap_type,
                        s.likelihood_score,
                        s.article_count,
                        lp.price,
                        pt.target_mean,
                        -- Get next upcoming earnings OR most recent past earnings (within 14 days)
                        COALESCE(
                            (SELECT earnings_date FROM earnings_calendar 
                             WHERE ticker = s.ticker AND earnings_date >= CURRENT_DATE 
                             ORDER BY earnings_date ASC LIMIT 1),
                            (SELECT earnings_date FROM earnings_calendar 
                             WHERE ticker = s.ticker AND earnings_date >= CURRENT_DATE - INTERVAL '14 days' AND earnings_date < CURRENT_DATE
                             ORDER BY earnings_date DESC LIMIT 1)
                        ) as earnings_date,
                        f.ex_dividend_date,
                        CASE WHEN lp.price > 0 AND pt.target_mean > 0 
                             THEN ROUND(((pt.target_mean - lp.price) / lp.price * 100)::numeric, 2)
                             ELSE NULL END as target_upside_pct,
                        ar.analyst_positivity,
                        ar.analyst_buy as buy_count,
                        ar.analyst_total as total_ratings,
                        f.dividend_yield,
                        f.pe_ratio,
                        f.forward_pe,
                        f.peg_ratio,
                        f.roe
                    FROM screener_scores s
                    LEFT JOIN latest_fundamentals f ON s.ticker = f.ticker
                    LEFT JOIN latest_analyst_ratings ar ON s.ticker = ar.ticker
                    LEFT JOIN latest_price_targets pt ON s.ticker = pt.ticker
                    LEFT JOIN latest_prices lp ON s.ticker = lp.ticker
                    WHERE s.date = (
                        SELECT MAX(date) FROM screener_scores ss WHERE ss.ticker = s.ticker
                    )
                    ORDER BY s.total_score DESC NULLS LAST
                """, engine)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    if df.empty:
        st.warning("No signals data. Run Analysis first.")
        return

    # =========================================================================
    # ENHANCED SCORING - OPTIONAL (expensive operation)
    # =========================================================================
    # Enhanced scoring is computed during batch analysis and stored.
    # This checkbox allows on-demand recalculation for display only.
    col_enh1, col_enh2 = st.columns([3, 1])
    with col_enh2:
        apply_enhanced = st.checkbox(
            "🔬 Live Enhanced Scoring",
            value=False,
            help="Fetches live data (MACD, earnings, PEG) - slower but most current. Leave OFF for fast loading."
        )

    if ENHANCED_SCORING_AVAILABLE and apply_enhanced:
        try:
            with st.spinner("Fetching live enhanced scores (this may take 30-60 seconds)..."):
                df = apply_enhanced_scores_to_dataframe(
                    df,
                    score_column='total_score',
                    enhanced_column='enhanced_score',
                    adjustment_column='score_adjustment',
                    fetch_price_history=True,
                )
                logger.info(f"Enhanced scoring applied to {len(df)} tickers")

                # Use enhanced score as primary if available
                if 'enhanced_score' in df.columns:
                    df['original_total'] = df['total_score']
                    df['total_score'] = df['enhanced_score']

                    # Recalculate signal type based on enhanced score
                    df['signal_type'] = df['total_score'].apply(lambda x:
                        'STRONG BUY' if x >= 80 else
                        'BUY' if x >= 65 else
                        'WEAK BUY' if x >= 55 else
                        'STRONG SELL' if x <= 20 else
                        'SELL' if x <= 35 else
                        'WEAK SELL' if x <= 45 else
                        'HOLD'
                    )
        except Exception as e:
            logger.warning(f"Enhanced scoring failed, using base scores: {e}")

    # Column mapping
    display_cols = {
        'ticker': 'Ticker',
        'score_date': 'Date',
        'sector': 'Sector',
        'market_cap': 'Cap',
        'signal_type': 'Signal',
        'total_score': 'Total',
        'score_adjustment': 'Adj',  # Enhanced scoring adjustment
        'sentiment_score': 'Sentiment',
        'options_flow_score': 'OptFlow',
        'short_squeeze_score': 'Squeeze',
        'fundamental_score': 'Fundamental',
        'growth_score': 'Growth',
        'dividend_score': 'Dividend',
        'technical_score': 'Technical',
        'gap_score': 'Gap',
        'gap_type': 'GapType',
        'likelihood_score': 'Likelihood',
        'article_count': 'Articles',
        'price': 'Price',
        'target_mean': 'TargetPrice',
        'earnings_date': 'Earnings',
        'ex_dividend_date': 'Ex-Div',
        'target_upside_pct': 'Upside%',
        'analyst_positivity': 'AnalystPos%',
        'buy_count': 'BuyRatings',
        'total_ratings': 'TotalRatings',
        'dividend_yield': 'DivYield%',
        'pe_ratio': 'PE',
        'forward_pe': 'FwdPE',
        'peg_ratio': 'PEG',
        'roe': 'ROE',
    }

    available_cols = [c for c in display_cols.keys() if c in df.columns]
    display_df = df[available_cols].copy()
    display_df = display_df.rename(columns={k: v for k, v in display_cols.items() if k in display_df.columns})

    # Format dates
    if 'Date' in display_df.columns:
        def format_score_date(val):
            if pd.isna(val) or val == '' or val is None:
                return ''
            try:
                if hasattr(val, 'strftime'):
                    return val.strftime('%m-%d')
                return str(val)[5:10] if len(str(val)) >= 10 else str(val)
            except:
                return ''
        display_df['Date'] = display_df['Date'].apply(format_score_date)

    # Format market cap as human-readable (B/T)
    if 'Cap' in display_df.columns:
        def format_market_cap(val):
            if pd.isna(val) or val == 0:
                return ''
            try:
                v = float(val)
                if v >= 1e12:
                    return f"${v / 1e12:.1f}T"
                elif v >= 1e9:
                    return f"${v / 1e9:.0f}B"
                elif v >= 1e6:
                    return f"${v / 1e6:.0f}M"
                else:
                    return f"${v:,.0f}"
            except:
                return ''

        display_df['Cap'] = display_df['Cap'].apply(format_market_cap)

    if 'Earnings' in display_df.columns:
        def format_earnings_date(val):
            if pd.isna(val) or val == '' or val is None:
                return ''
            try:
                from datetime import date
                if hasattr(val, 'date'):
                    dt = val.date() if hasattr(val, 'date') else val
                elif hasattr(val, 'strftime'):
                    dt = val
                else:
                    dt = pd.to_datetime(str(val)).date()

                today = date.today()

                # Format: MM-DD with indicator
                if dt < today:
                    # Past earnings (already reported) - add ✓
                    if val.year != today.year:
                        return f"✓{dt.strftime('%m-%d-%y')}"
                    return f"✓{dt.strftime('%m-%d')}"
                else:
                    # Future earnings (upcoming)
                    if val.year != today.year:
                        return dt.strftime('%m-%d-%y')
                    return dt.strftime('%m-%d')
            except:
                return str(val)[:10] if len(str(val)) >= 10 else str(val)
        display_df['Earnings'] = display_df['Earnings'].apply(format_earnings_date)

    if 'Ex-Div' in display_df.columns:
        display_df['Ex-Div'] = display_df['Ex-Div'].apply(lambda x: x.strftime('%m-%d') if hasattr(x, 'strftime') else '')

    # Format numeric columns - KEEP AS NUMBERS for proper sorting
    # FIX: Score columns should show empty/- for NULL, not 0
    # Separate score columns (where NULL = missing data) from count columns (where 0 is valid)
    count_columns = ['BuyRatings', 'TotalRatings', 'Articles']  # 0 is valid here
    score_columns = ['Sentiment', 'Fundamental', 'Growth', 'Dividend', 'Technical',
                     'Gap', 'Likelihood', 'Total', 'OptFlow', 'Squeeze']  # NULL = missing

    for col in count_columns:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce').fillna(0).astype(int)

    for col in score_columns:
        if col in display_df.columns:
            # Keep as numeric for sorting, but NaN will display as empty
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
            # Round first, then convert to nullable int - NaN stays as NaN for display
            display_df[col] = display_df[col].round(0).astype('Int64')  # Nullable integer type

    # Convert decimal columns to numeric (NOT strings) - required for proper sorting
    # Fill NaN with -999 for sorting purposes (will sort to bottom in descending order)
    decimal_columns = ['Price', 'TargetPrice', 'Upside%', 'AnalystPos%', 'DivYield%', 'PE', 'ROE']
    for col in decimal_columns:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
            # Round for display but keep as float
            display_df[col] = display_df[col].round(2)

    # =========================================================================
    # ADD AI WIN PROBABILITY COLUMN
    # =========================================================================
    if AI_PROB_AVAILABLE and 'Ticker' in display_df.columns:
        try:
            # Get AI probabilities for all tickers (cached for 5 min)
            ticker_scores = {}
            for _, row in display_df.iterrows():
                ticker_scores[row['Ticker']] = {
                    # FIX: Use None as default, not 50 - AI model should handle missing data
                    'sentiment_score': row.get('Sentiment') if pd.notna(row.get('Sentiment')) else None,
                    'fundamental_score': row.get('Fundamental') if pd.notna(row.get('Fundamental')) else None,
                    'technical_score': row.get('Technical') if pd.notna(row.get('Technical')) else None,
                    'options_flow_score': row.get('OptFlow') if pd.notna(row.get('OptFlow')) else None,
                    'short_squeeze_score': row.get('Squeeze') if pd.notna(row.get('Squeeze')) else None,
                    'total_score': row.get('Total') if pd.notna(row.get('Total')) else None,
                }

            ai_probs = _get_ai_probabilities_for_table(ticker_scores)

            # Add AI Prob column
            display_df['AI Prob'] = display_df['Ticker'].map(
                lambda t: ai_probs.get(t, {}).get('prob', None)
            )
            # Convert to percentage for display (keep as float for sorting)
            display_df['AI Prob'] = pd.to_numeric(display_df['AI Prob'], errors='coerce')

            # Reorder columns to put AI Prob after Total
            if 'AI Prob' in display_df.columns and 'Total' in display_df.columns:
                cols = display_df.columns.tolist()
                if 'AI Prob' in cols:
                    cols.remove('AI Prob')
                    total_idx = cols.index('Total') if 'Total' in cols else 0
                    cols.insert(total_idx + 1, 'AI Prob')
                    display_df = display_df[cols]

        except Exception as e:
            logger.debug(f"AI Prob column not added: {e}")

    # =========================================================================
    # FILTERS ROW
    # =========================================================================
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 2, 2])

    with col1:
        signal_options = ['STRONG BUY', 'BUY', 'WEAK BUY', 'HOLD', 'WEAK SELL', 'SELL', 'STRONG SELL']
        signal_filter = st.multiselect("Filter by Signal", options=signal_options, key="signal_filter")

    with col2:
        min_total = st.slider("Min Total", 0, 100, 0, key="min_total_slider")

    with col3:
        min_sentiment = st.slider("Min Sent", 0, 100, 0, key="min_sentiment_slider")

    with col4:
        # AI Strategy Filter
        ai_filter_options = [
            "All Signals",
            "🤖 AI Probability (≥55%)",
            "🤖 AI Conservative (≥60%)",
            "🤖 AI High Conviction (≥65%)"
        ]
        ai_filter = st.selectbox("AI Strategy", ai_filter_options, key="ai_filter_select")

    with col5:
        sort_options = ['Total', 'AI Prob', 'Sentiment', 'OptFlow', 'Squeeze', 'Fundamental', 'Technical', 'Upside%',
                        'Price', 'PE']
        sort_col1, sort_col2 = st.columns([3, 1])
        with sort_col1:
            sort_by = st.selectbox("Sort by", sort_options, key="sort_by_select")
        with sort_col2:
            sort_dir = st.selectbox("↕️", ["Desc", "Asc"], key="sort_dir_select")

    # Apply filters
    filtered_df = display_df.copy()
    if signal_filter and 'Signal' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Signal'].isin(signal_filter)]
    if min_total > 0 and 'Total' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Total'] >= min_total]
    if min_sentiment > 0 and 'Sentiment' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Sentiment'] >= min_sentiment]

    # Apply AI Strategy filter
    if 'AI Prob' in filtered_df.columns and ai_filter != "All Signals":
        if "≥55%" in ai_filter:
            filtered_df = filtered_df[filtered_df['AI Prob'] >= 0.55]
        elif "≥60%" in ai_filter:
            filtered_df = filtered_df[filtered_df['AI Prob'] >= 0.60]
        elif "≥65%" in ai_filter:
            filtered_df = filtered_df[filtered_df['AI Prob'] >= 0.65]

    # Sort - use dropdown sorting (column header sorting in Streamlit is unreliable)
    if sort_by in filtered_df.columns:
        ascending = (sort_dir == "Asc")
        filtered_df = filtered_df.sort_values(sort_by, ascending=ascending, na_position='last')
        # Reset index for clean display
        filtered_df = filtered_df.reset_index(drop=True)

    # =========================================================================
    # AI BATCH ANALYSIS SECTION
    # =========================================================================
    with st.expander("🤖 AI Batch Analysis - Deep Analysis on Top Signals", expanded=False):
        _render_ai_batch_analysis(filtered_df)

    # =========================================================================
    # PREDICTION TRACKER SECTION
    # =========================================================================
    with st.expander("📊 ML Learning Progress - Prediction Tracker", expanded=False):
        _render_prediction_tracker()

    # =========================================================================
    # SMART REFRESH STATUS
    # =========================================================================
    with st.expander("🔄 Smart Data Refresh - Avoid Duplicate Work", expanded=False):
        _render_smart_refresh_status()

    # =========================================================================
    # STOCK SELECTOR FOR DEEP DIVE
    # =========================================================================
    st.markdown("---")

    ticker_list = filtered_df['Ticker'].tolist() if 'Ticker' in filtered_df.columns else []

    # Check if there's a pre-selected ticker from Quick Add
    preselected = st.session_state.get('selected_ticker')

    # If preselected ticker is not in the filtered list, add it temporarily
    # (This handles newly added tickers that might not be in the current filter)
    if preselected and preselected not in ticker_list:
        ticker_list = [preselected] + ticker_list

    # Calculate default index
    default_index = 0
    if preselected and preselected in ticker_list:
        default_index = ticker_list.index(preselected) + 1  # +1 because we prepend ""

    col_select, col_refresh, col_info = st.columns([2, 1, 2])

    with col_select:
        selected = st.selectbox(
            "🔍 Select Stock for Deep Dive",
            options=[""] + ticker_list,
            index=default_index,
            key="ticker_deep_dive_select",
            help="Select a stock to see detailed analysis below the table"
        )
        if selected:
            st.session_state.selected_ticker = selected

    with col_refresh:
        if selected and st.button(f"🔄 Refresh {selected}", key="refresh_selected_btn"):
            _run_single_analysis(selected)

    with col_info:
        if selected:
            # Show quick summary
            row = filtered_df[filtered_df['Ticker'] == selected].iloc[0] if not filtered_df[filtered_df['Ticker'] == selected].empty else None
            if row is not None:
                signal = row.get('Signal', 'N/A')
                total = row.get('Total', 0)
                signal_emoji = "🟢" if "BUY" in str(signal) else "🔴" if "SELL" in str(signal) else "🟡"
                st.markdown(f"**{selected}**: {signal_emoji} {signal} | Score: {total}")

    # =========================================================================
    # STYLED TABLE
    # =========================================================================
    def color_signal(val):
        if pd.isna(val):
            return ''
        val = str(val).upper()
        if 'STRONG BUY' in val:
            return 'background-color: #00C853; color: white'
        elif 'BUY' in val:
            return 'background-color: #69F0AE'
        elif 'STRONG SELL' in val:
            return 'background-color: #FF1744; color: white'
        elif 'SELL' in val:
            return 'background-color: #FF8A80'
        return ''

    def color_score(val):
        if pd.isna(val):
            return ''
        try:
            v = float(val)
            if v >= 70:
                return 'background-color: #00C853; color: white'
            elif v >= 60:
                return 'background-color: #69F0AE'
            elif v <= 30:
                return 'background-color: #FF1744; color: white'
            elif v <= 40:
                return 'background-color: #FF8A80'
        except:
            pass
        return ''

    def color_squeeze(val):
        if pd.isna(val):
            return ''
        try:
            v = float(val)
            if v >= 70:
                return 'background-color: #FFD700; color: black'
            elif v >= 50:
                return 'background-color: #FFA500'
        except:
            pass
        return ''

    def color_ai_prob(val):
        """Color AI probability: green >= 55%, orange 45-55%, red < 45%"""
        if pd.isna(val):
            return ''
        try:
            v = float(val)
            if v >= 0.55:
                return 'background-color: #00C853; color: white'
            elif v >= 0.45:
                return 'background-color: #FFA500'
            else:
                return 'background-color: #FF1744; color: white'
        except:
            pass
        return ''

    # Apply styling

    # Apply styling (use map for pandas 2.1+ compatibility)

    # First, convert numeric columns to proper numeric types (handles "None" strings, etc.)
    numeric_cols = ['Price', 'TargetPrice', 'Upside%', 'AnalystPos%', 'DivYield%',
                    'PE', 'FwdPE', 'PEG', 'ROE', 'AI Prob',
                    'Total', 'Technical', 'Fundamental', 'Sentiment', 'OptFlow',
                    'Squeeze', 'Growth', 'Dividend', 'Gap', 'Likelihood',
                    'Articles', 'BuyRatings', 'TotalRatings']

    for col in numeric_cols:
        if col in filtered_df.columns:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')

    styled_df = filtered_df.style

    # Format numeric columns - CLEAN DECIMALS
    format_dict = {}

    # Price columns - 2 decimal places
    price_cols = ['Price', 'TargetPrice']
    for col in price_cols:
        if col in filtered_df.columns:
            format_dict[col] = '{:.2f}'

    # Percentage columns - 2 decimal places
    pct_cols = ['Upside%', 'AnalystPos%', 'DivYield%']
    for col in pct_cols:
        if col in filtered_df.columns:
            format_dict[col] = '{:.2f}'

    # Ratio columns - 2 decimal places
    ratio_cols = ['PE', 'FwdPE', 'PEG', 'ROE', 'AI Prob']
    for col in ratio_cols:
        if col in filtered_df.columns:
            format_dict[col] = '{:.2f}'

    # Score columns - integer (no decimals)
    score_cols = ['Total', 'Technical', 'Fundamental', 'Sentiment', 'OptFlow',
                  'Squeeze', 'Growth', 'Dividend', 'Gap', 'Likelihood']
    for col in score_cols:
        if col in filtered_df.columns:
            format_dict[col] = '{:.0f}'

    # Count columns - integer (no decimals)
    count_cols = ['Articles', 'BuyRatings', 'TotalRatings']
    for col in count_cols:
        if col in filtered_df.columns:
            format_dict[col] = '{:.0f}'

    # Apply format
    if format_dict:
        styled_df = styled_df.format(format_dict, na_rep='N/A')

    try:
        # Try newer pandas API first
        if 'Signal' in filtered_df.columns:
            styled_df = styled_df.map(color_signal, subset=['Signal'])
        for col in ['OptFlow', 'Sentiment', 'Fundamental', 'Technical']:
            if col in filtered_df.columns:
                styled_df = styled_df.map(color_score, subset=[col])
        if 'Squeeze' in filtered_df.columns:
            styled_df = styled_df.applymap(color_squeeze, subset=['Squeeze'])
        if 'AI Prob' in filtered_df.columns:
            styled_df = styled_df.applymap(color_ai_prob, subset=['AI Prob'])
    except AttributeError:
        # Fall back to older pandas API
        if 'Signal' in filtered_df.columns:
            styled_df = styled_df.applymap(color_signal, subset=['Signal'])
        for col in ['OptFlow', 'Sentiment', 'Fundamental', 'Technical']:
            if col in filtered_df.columns:
                styled_df = styled_df.applymap(color_score, subset=[col])
        if 'Squeeze' in filtered_df.columns:
            styled_df = styled_df.applymap(color_squeeze, subset=['Squeeze'])
        if 'AI Prob' in filtered_df.columns:
            styled_df = styled_df.applymap(color_ai_prob, subset=['AI Prob'])

        # Display table with row selection
    # Display table with row selection
    st.dataframe(styled_df, width='stretch', height=400)

    # Table stats and legend
    ai_filter_text = ""
    if 'AI Prob' in display_df.columns and ai_filter != "All Signals":
        ai_filter_text = f" | 🤖 {ai_filter}"
    st.caption(f"📊 Showing {len(filtered_df)} of {len(display_df)} tickers{ai_filter_text} | Earnings: ✓=reported | OptFlow: 50=neutral")

    # Show AI strategy counts
    if 'AI Prob' in display_df.columns:
        prob_55 = len(display_df[display_df['AI Prob'] >= 0.55])
        prob_60 = len(display_df[display_df['AI Prob'] >= 0.60])
        prob_65 = len(display_df[display_df['AI Prob'] >= 0.65])
        st.caption(f"🤖 AI Signals Today: Probability(≥55%): {prob_55} | Conservative(≥60%): {prob_60} | High Conviction(≥65%): {prob_65}")

    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "signals.csv", "text/csv", key="download_signals_csv")

    # =========================================================================
    # DEEP DIVE SECTION (if stock selected)
    # =========================================================================
    if selected:
        st.markdown("---")
        st.markdown(f"## 🔍 Deep Dive: {selected}")
        _render_stock_deep_dive(selected)


def _render_stock_deep_dive(ticker: str):
    """Render detailed analysis for a selected stock - uses full deep dive."""

    try:
        from src.core import get_signal_engine
        engine = get_signal_engine()
        signal = engine.generate_signal(ticker)
    except Exception as e:
        st.error(f"Error loading signal: {e}")
        return

    if not signal:
        st.warning(f"No signal data for {ticker}")
        return

    # Use the full deep dive renderer
    _render_deep_dive(signal)


def _render_ticker_news(ticker: str):
    """Render recent news for a ticker."""
    try:
        engine = get_engine()

        news_df = pd.read_sql(f"""
            SELECT headline, source, url, ai_sentiment_fast, published_at, created_at
            FROM news_articles
            WHERE ticker = '{ticker}'
            ORDER BY COALESCE(published_at, created_at) DESC
            LIMIT 8
        """, engine)

        if news_df.empty:
            st.caption("No recent news")
            return

        _display_news_items(news_df, max_items=8)

    except Exception as e:
        st.caption(f"Could not load news: {e}")
# ANALYSIS PANEL
# =============================================================================



# =============================================================================
# SIGNALS VIEW HELPERS
# =============================================================================

def _get_tickers() -> List[str]:
    """Get tickers from universe."""
    try:
        engine = get_engine()
        df = pd.read_sql("SELECT DISTINCT ticker FROM screener_scores ORDER BY ticker", engine)
        return df['ticker'].tolist() if not df.empty else []
    except:
        return []


def _render_signals_view(filter_type: str, sort_by: str):
    """Render signals table."""

    if 'signals_data' not in st.session_state or st.session_state.get('force_refresh'):
        with st.spinner("Loading..."):
            try:
                tickers = _get_tickers()
                engine = get_signal_engine()
                st.session_state.signals_data = engine.generate_signals_batch(tickers, max_workers=5)
                st.session_state.market_overview = get_market_overview()
                st.session_state.force_refresh = False
            except Exception as e:
                st.error(f"Error: {e}")
                return

    signals = st.session_state.signals_data
    overview = st.session_state.get('market_overview')

    # Market overview
    if overview:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            c = "🟢" if overview.spy_change >= 0 else "🔴"
            st.metric("SPY", f"{c} {overview.spy_change:+.2f}%")
        with col2:
            c = "🟢" if overview.qqq_change >= 0 else "🔴"
            st.metric("QQQ", f"{c} {overview.qqq_change:+.2f}%")
        with col3:
            st.metric("VIX", f"{overview.vix:.1f}")
        with col4:
            st.metric("Regime", overview.regime)

    st.markdown("---")

    # Table
    _render_signals_table(signals, filter_type, sort_by)


def _render_signals_table(signals: Dict[str, UnifiedSignal], filter_type: str, sort_by: str):
    """Render signals table with deep dive."""

    if not signals:
        st.warning("No signals")
        return

    signal_list = list(signals.values())

    # Filter
    if filter_type == "Stocks":
        signal_list = [s for s in signal_list if s.asset_type.value == 'stock']
    elif filter_type == "Bonds":
        signal_list = [s for s in signal_list if s.asset_type.value == 'bond']

    # Sort
    if "Today ↓" in sort_by:
        signal_list.sort(key=lambda x: x.today_score, reverse=True)
    elif "Today ↑" in sort_by:
        signal_list.sort(key=lambda x: x.today_score)
    elif "Long-term" in sort_by:
        signal_list.sort(key=lambda x: x.longterm_score, reverse=True)
    elif "Risk" in sort_by:
        signal_list.sort(key=lambda x: x.risk_score, reverse=True)

    # Table data
    table_data = []
    for s in signal_list:
        e = "🟢" if s.today_signal.value == 'bullish' else "🔴" if s.today_signal.value == 'bearish' else "🟡"
        table_data.append({
            'Ticker': s.ticker,
            'Today': f"{e} {s.today_score}%",
            'Long': f"{s.longterm_score}/100",
            'Risk': s.risk_level.value,
            'Reason': (s.signal_reason or '')[:40]
        })

    st.markdown(f"**{len(signal_list)} signals**")

    # Selector - restore ticker if we just refreshed
    tickers = [s.ticker for s in signal_list]

    # Check if we need to restore a ticker after refresh
    default_index = 0
    if '_refresh_ticker' in st.session_state:
        refresh_ticker = st.session_state.pop('_refresh_ticker')
        if refresh_ticker in tickers:
            default_index = tickers.index(refresh_ticker) + 1  # +1 because of empty string at start

    selected = st.selectbox("Details:", [""] + tickers, index=default_index, key="deep_dive")

    st.dataframe(pd.DataFrame(table_data), width='stretch', hide_index=True, height=350)

    # Deep dive
    if selected and selected in signals:
        st.markdown("---")
        _render_deep_dive(signals[selected])


