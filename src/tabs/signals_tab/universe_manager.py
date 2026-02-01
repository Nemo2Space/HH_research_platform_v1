"""
Signals Tab - Universe Manager

Ticker and universe management: add, remove, quick analyze.
"""
from .deep_dive import _render_deep_dive
from .shared import (
    st, pd, np, logger, datetime, date, timedelta, time, json,
    Dict, List, Optional, _to_native, text,
    get_engine, get_connection,
    SIGNAL_HUB_AVAILABLE, DB_AVAILABLE,
    UNIVERSE_SCORER_AVAILABLE, UniverseScorer,
    NEWS_COLLECTOR_AVAILABLE, NewsCollector,
    SENTIMENT_ANALYZER_AVAILABLE, SentimentAnalyzer,
    TECHNICAL_ANALYZER_AVAILABLE, TechnicalAnalyzer,
    FUNDAMENTAL_ANALYZER_AVAILABLE, FundamentalAnalyzer,
    OPTIONS_FLOW_AVAILABLE, OptionsFlowAnalyzer,
    ENHANCED_SCORING_AVAILABLE, ENHANCED_SCORES_DB_AVAILABLE,
)

if SIGNAL_HUB_AVAILABLE:
    from .shared import get_signal_engine, SignalEngine

if ENHANCED_SCORING_AVAILABLE:
    from .shared import get_enhanced_total_score

if ENHANCED_SCORES_DB_AVAILABLE:
    from src.analytics.enhanced_scores_db import compute_and_save_enhanced_scores

def _render_quick_add_ticker():
    """
    Render the Quick Add Ticker UI.
    Allows users to analyze any ticker on-demand, even if not in the universe.
    Uses the same pipeline as the full scanner for consistency.

    Features:
    - Add new tickers to watchlist
    - Force refresh existing tickers (re-fetch everything: news, sentiment, technical, etc.)
    """
    import os

    # Initialize session state for this feature
    if 'quick_add_result' not in st.session_state:
        st.session_state.quick_add_result = None
    if 'quick_add_error' not in st.session_state:
        st.session_state.quick_add_error = None

    st.markdown("**Analyze any ticker instantly** - generates full signals using the same pipeline as the scanner.")

    # Input row
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        ticker_input = st.text_input(
            "Ticker Symbol",
            placeholder="e.g., NVAX, CERT, ADPT",
            key="quick_add_ticker_input",
            label_visibility="collapsed"
        ).upper().strip()

    with col2:
        add_permanent = st.checkbox(
            "Save to watchlist",
            value=False,
            help="If checked, adds the ticker permanently to your universe.csv and database"
        )

    with col3:
        force_refresh = st.checkbox(
            "🔄 Force Refresh",
            value=False,
            help="Re-fetch everything: news, sentiment, technical, fundamentals. Use for existing tickers that need updated data."
        )

    with col4:
        analyze_clicked = st.button(
            "🚀 Analyze",
            type="primary",
            use_container_width=True,
            disabled=not ticker_input
        )

    # Process the analysis
    if analyze_clicked and ticker_input:
        st.session_state.quick_add_result = None
        st.session_state.quick_add_error = None

        # Parse multiple tickers (comma, space, or semicolon separated)
        import re
        tickers = [t.strip().upper() for t in re.split(r'[,;\s]+', ticker_input) if t.strip()]

        if not tickers:
            st.session_state.quick_add_error = "No valid tickers entered"
        elif len(tickers) == 1:
            # Single ticker - original behavior
            action_text = "🔄 Force refreshing" if force_refresh else "🔍 Analyzing"
            with st.spinner(f"{action_text} {tickers[0]}..."):
                try:
                    result = _quick_analyze_ticker(tickers[0], add_permanent, force_refresh=force_refresh)
                    if result['success']:
                        st.session_state.quick_add_result = result
                        st.session_state.quick_add_error = None
                        # Force refresh signals table to show new/updated ticker
                        st.session_state.signals_loaded = False
                        st.rerun()
                    else:
                        st.session_state.quick_add_error = result['error']
                        st.session_state.quick_add_result = None
                except Exception as e:
                    st.session_state.quick_add_error = str(e)
                    st.session_state.quick_add_result = None
                    logger.error(f"Quick add error: {e}")
        else:
            # Multiple tickers - process each one
            results = []
            errors = []
            action_text = "Force refreshing" if force_refresh else "Analyzing"
            progress_bar = st.progress(0, text=f"{action_text} {len(tickers)} tickers...")

            for i, ticker in enumerate(tickers):
                progress_bar.progress((i + 1) / len(tickers), text=f"🔍 {action_text} {ticker} ({i+1}/{len(tickers)})...")
                try:
                    result = _quick_analyze_ticker(ticker, add_permanent, force_refresh=force_refresh)
                    if result['success']:
                        results.append(result)
                    else:
                        errors.append(f"{ticker}: {result['error']}")
                except Exception as e:
                    errors.append(f"{ticker}: {str(e)}")
                    logger.error(f"Quick add error for {ticker}: {e}")

            progress_bar.empty()

            # Store results
            if results:
                # Store the last successful result for display, but mark as multi
                st.session_state.quick_add_result = {
                    'success': True,
                    'multi': True,
                    'count': len(results),
                    'tickers': [r['ticker'] for r in results],
                    'results': results,
                    'force_refreshed': force_refresh
                }
                st.session_state.signals_loaded = False

            if errors:
                st.session_state.quick_add_error = f"Failed: {'; '.join(errors)}"
            else:
                st.session_state.quick_add_error = None

            if results:
                st.rerun()


    # Show error if any
    if st.session_state.quick_add_error:
        st.error(f"❌ {st.session_state.quick_add_error}")

    # Show result if available
    if st.session_state.quick_add_result:
        result = st.session_state.quick_add_result

        # Handle multi-ticker results
        if result.get('multi'):
            action = "🔄 Force refreshed" if result.get('force_refreshed') else "✅ Analyzed"
            st.success(f"{action} {result['count']} tickers: {', '.join(result['tickers'])}")

            # Show summary for each ticker
            for single_result in result.get('results', []):
                ticker = single_result['ticker']
                signal = single_result.get('signal')

                with st.expander(f"📊 {ticker}", expanded=True):
                    if signal:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            score = signal.today_score if signal.today_score is not None else 50
                            try:
                                score = float(score)
                                color = "🟢" if score >= 65 else "🔴" if score <= 35 else "🟡"
                                signal_val = signal.today_signal.value if signal.today_signal else "HOLD"
                                st.metric("Signal", f"{color} {signal_val}", f"{score:.0f}%")
                            except (TypeError, ValueError):
                                st.metric("Signal", "N/A")
                        with col2:
                            try:
                                if signal.current_price:
                                    st.metric("Price", f"${float(signal.current_price):.2f}")
                                else:
                                    st.metric("Price", "N/A")
                            except (TypeError, ValueError):
                                st.metric("Price", "N/A")
                        with col3:
                            st.metric("Sector", signal.sector or "Unknown")
                        with col4:
                            try:
                                if signal.technical_score is not None:
                                    st.metric("Technical", f"{float(signal.technical_score):.0f}")
                                else:
                                    st.metric("Technical", "N/A")
                            except (TypeError, ValueError):
                                st.metric("Technical", "N/A")
                    else:
                        st.info("Signal data not available")
        else:
            # Single ticker result - original behavior
            ticker = result['ticker']

            # Build status message
            status_parts = []
            if result.get('force_refreshed'):
                status_parts.append("🔄 Force refreshed")
            if result.get('added_to_watchlist'):
                status_parts.append("Added to watchlist")
            if result.get('saved_to_db'):
                status_parts.append("Saved to database")
            else:
                status_parts.append("⚠️ Not saved to DB - may not appear in signals table")

            status_msg = " | ".join(status_parts) if status_parts else "Temporary"
            st.success(f"✅ {ticker} analyzed successfully! ({status_msg})")

            # Show signal summary
            if result.get('signal'):
                signal = result['signal']

                # Signal summary cards
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    score = signal.today_score if signal.today_score is not None else 50
                    try:
                        score = float(score)
                        color = "🟢" if score >= 65 else "🔴" if score <= 35 else "🟡"
                        signal_val = signal.today_signal.value if signal.today_signal else "HOLD"
                        st.metric("Signal", f"{color} {signal_val}", f"{score:.0f}%")
                    except (TypeError, ValueError):
                        st.metric("Signal", "N/A")

                with col2:
                    try:
                        if signal.current_price:
                            st.metric("Price", f"${float(signal.current_price):.2f}")
                        else:
                            st.metric("Price", "N/A")
                    except (TypeError, ValueError):
                        st.metric("Price", "N/A")

                with col3:
                    st.metric("Sector", signal.sector or "Unknown")

                with col4:
                    try:
                        if signal.technical_score is not None:
                            st.metric("Technical", f"{float(signal.technical_score):.0f}")
                        else:
                            st.metric("Technical", "N/A")
                    except (TypeError, ValueError):
                        st.metric("Technical", "N/A")

            # Show scores breakdown
            if result.get('scores'):
                scores = result['scores']
                st.markdown("**Score Breakdown:**")
                score_cols = st.columns(5)
                score_names = ['Technical', 'Fundamental', 'Sentiment', 'Options', 'Total']
                score_keys = ['technical_score', 'fundamental_score', 'sentiment_score', 'options_flow_score', 'total_score']

                for i, (name, key) in enumerate(zip(score_names, score_keys)):
                    with score_cols[i]:
                        val = scores.get(key)
                        if val is None:
                            val = 50  # Default to neutral
                        try:
                            val = float(val)
                            color = "🟢" if val >= 65 else "🔴" if val <= 35 else "🟡"
                            st.metric(name, f"{color} {val:.0f}")
                        except (TypeError, ValueError):
                            st.metric(name, "N/A")

            # Show AI Analysis Results (if available)
            if result.get('ai_result'):
                ai_res = result['ai_result']
                st.markdown("---")
                st.markdown("### 🤖 AI Analysis")

                # AI Decision row
                ai_cols = st.columns(4)
                with ai_cols[0]:
                    action_color = "🟢" if ai_res.ai_action == "BUY" else "🔴" if ai_res.ai_action == "SELL" else "🟡"
                    st.metric("AI Action", f"{action_color} {ai_res.ai_action}")
                with ai_cols[1]:
                    conf_color = "🟢" if ai_res.ai_confidence == "HIGH" else "🟡" if ai_res.ai_confidence == "MEDIUM" else "🔴"
                    st.metric("Confidence", f"{conf_color} {ai_res.ai_confidence}")
                with ai_cols[2]:
                    trade_icon = "✅" if ai_res.trade_allowed else "❌"
                    st.metric("Trade Allowed", f"{trade_icon} {'Yes' if ai_res.trade_allowed else 'No'}")
                with ai_cols[3]:
                    if ai_res.target_price and ai_res.entry_price:
                        upside = ((ai_res.target_price - ai_res.entry_price) / ai_res.entry_price) * 100
                        st.metric("Upside", f"{upside:.1f}%")
                    else:
                        st.metric("Upside", "N/A")

                # Entry/Exit plan
                plan_cols = st.columns(4)
                with plan_cols[0]:
                    st.metric("Entry", f"${ai_res.entry_price:.2f}" if ai_res.entry_price else "N/A")
                with plan_cols[1]:
                    st.metric("Stop Loss", f"${ai_res.stop_loss:.2f}" if ai_res.stop_loss else "N/A")
                with plan_cols[2]:
                    st.metric("Target", f"${ai_res.target_price:.2f}" if ai_res.target_price else "N/A")
                with plan_cols[3]:
                    st.metric("Position", ai_res.position_size or "N/A")

                # Bull/Bear case
                case_cols = st.columns(2)
                with case_cols[0]:
                    st.markdown("**🐂 Bull Case:**")
                    for point in ai_res.bull_case[:3]:
                        st.markdown(f"• {point}")
                with case_cols[1]:
                    st.markdown("**🐻 Bear Case:**")
                    for point in ai_res.bear_case[:3]:
                        st.markdown(f"• {point}")

                # Key risks
                if ai_res.key_risks:
                    st.markdown("**⚠️ Key Risks:**")
                    for risk in ai_res.key_risks[:3]:
                        st.markdown(f"• {risk}")

                # Blocking factors
                if ai_res.blocking_factors:
                    st.warning(f"**Blocking Factors:** {', '.join(ai_res.blocking_factors)}")

            # Buttons row
            col_btn1, col_btn2, col_btn3 = st.columns(3)

            with col_btn1:
                # Button to view in signals table
                if st.button(f"📋 View in Signals Table", key=f"view_signals_{ticker}", use_container_width=True):
                    st.session_state.selected_ticker = ticker
                    st.session_state.signals_loaded = True
                    st.session_state.force_refresh = True
                    if 'signals_data' in st.session_state:
                        del st.session_state['signals_data']
                    st.rerun()

            with col_btn2:
                # Direct deep dive toggle
                deep_dive_key = f"show_deep_dive_{ticker}"
                if deep_dive_key not in st.session_state:
                    st.session_state[deep_dive_key] = False

                if st.button(f"🔍 Deep Dive", key=f"deep_dive_btn_{ticker}", use_container_width=True):
                    st.session_state[deep_dive_key] = not st.session_state[deep_dive_key]
                    st.rerun()

            with col_btn3:
                # Clear button
                if st.button("🗑️ Clear", key=f"clear_result_{ticker}", use_container_width=True):
                    st.session_state.quick_add_result = None
                    st.session_state.selected_ticker = None
                    # Clear deep dive state
                    if f"show_deep_dive_{ticker}" in st.session_state:
                        del st.session_state[f"show_deep_dive_{ticker}"]
                    st.rerun()

            # Show deep dive inline if toggled
            if st.session_state.get(f"show_deep_dive_{ticker}", False):
                st.markdown("---")
                st.markdown(f"### 🔍 Deep Dive: {ticker}")

                # Use the signal from the result if available, otherwise fetch fresh
                if result.get('signal'):
                    _render_deep_dive(result['signal'])
                else:
                    # Fetch fresh signal for deep dive
                    try:
                        from src.core import get_signal_engine
                        engine = get_signal_engine()
                        signal = engine.generate_signal(ticker, force_refresh=True)
                        if signal:
                            _render_deep_dive(signal)
                        else:
                            st.warning(f"Could not generate signal for {ticker}")
                    except Exception as e:
                        st.error(f"Error loading deep dive: {e}")


def _quick_analyze_ticker(ticker: str, add_permanent: bool = False, force_refresh: bool = False) -> dict:
    """
    Analyze a ticker using the FULL signal pipeline - same as "Run Analysis".

    This runs:
    1. Fresh news collection
    2. Sentiment analysis
    3. Options flow & squeeze scores
    4. Technical analysis
    5. Fundamental data fetch (yfinance + Finviz + IBKR)
    6. Create/Update screener_scores with ALL scores
    7. Generate signal with committee
    8. Run AI Analysis (BatchAIAnalyzer) for deep insights

    Args:
        ticker: Stock ticker symbol
        add_permanent: If True, add to universe.csv
        force_refresh: If True, skip cache and re-fetch ALL data (news, sentiment, technical, etc.)

    Returns:
        dict with 'success', 'ticker', 'signal', 'scores', 'error', 'added_to_watchlist',
        'saved_to_db', 'force_refreshed', 'ai_result'
        dict with 'success', 'ticker', 'signal', 'scores', 'error', 'added_to_watchlist', 'saved_to_db', 'force_refreshed'
    """
    import os
    import yfinance as yf
    from datetime import date as dt_date

    ticker = ticker.upper().strip()

    if force_refresh:
        logger.info(f"{ticker}: FORCE REFRESH mode - re-fetching all data")

    # Validate ticker format
    if not ticker or len(ticker) > 10 or not ticker.replace('.', '').replace('-', '').isalnum():
        return {'success': False, 'error': f"Invalid ticker format: {ticker}"}

    # Quick validation - check if ticker exists via yfinance
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        if not info or info.get('regularMarketPrice') is None:
            try:
                fast = yf_ticker.fast_info
                if not hasattr(fast, 'last_price') or fast.last_price is None:
                    return {'success': False, 'error': f"Ticker '{ticker}' not found or has no price data"}
            except:
                return {'success': False, 'error': f"Ticker '{ticker}' not found or has no price data"}
    except Exception as e:
        return {'success': False, 'error': f"Could not validate ticker '{ticker}': {str(e)}"}

    # =========================================================================
    # FETCH DATA FROM MULTIPLE SOURCES (yfinance already fetched above)
    # =========================================================================

    # Source 1: yfinance data (already in 'info' dict)
    logger.info(f"{ticker}: Fetching from multiple data sources...")

    # Source 2: Finviz data
    finviz_data = {}
    try:
        from src.data.finviz import FinvizDataFetcher
        finviz_fetcher = FinvizDataFetcher()
        finviz_data = finviz_fetcher.get_fundamentals(ticker) or {}
        if finviz_data:
            logger.info(f"{ticker}: ✓ Finviz data fetched (Inst Own: {finviz_data.get('inst_own')}%)")

        # Also get analyst ratings from Finviz
        finviz_ratings = finviz_fetcher.get_analyst_ratings(ticker) or {}
        if finviz_ratings:
            finviz_data['finviz_buy_pct'] = finviz_ratings.get('buy_pct')
            finviz_data['finviz_total_ratings'] = finviz_ratings.get('total_ratings')
            logger.info(f"{ticker}: ✓ Finviz ratings: {finviz_ratings.get('total_positive')}/{finviz_ratings.get('total_ratings')} positive")
    except ImportError:
        logger.debug(f"{ticker}: Finviz module not available")
    except Exception as e:
        logger.debug(f"{ticker}: Finviz fetch failed: {e}")

    # Source 3: IBKR data (if available)
    ibkr_data = {}
    try:
        from src.data.ibkr_client import IBKRClient
        ibkr = IBKRClient()
        if ibkr.is_connected():
            ibkr_data = ibkr.get_fundamental_data(ticker) or {}
            if ibkr_data:
                logger.info(f"{ticker}: ✓ IBKR data fetched")
    except ImportError:
        logger.debug(f"{ticker}: IBKR module not available")
    except Exception as e:
        logger.debug(f"{ticker}: IBKR fetch failed: {e}")

    # =========================================================================
    # MERGE DATA FROM ALL SOURCES (prefer: IBKR > Finviz > yfinance)
    # =========================================================================
    def get_best_value(*values):
        """Return first non-None value from multiple sources."""
        for v in values:
            if v is not None and v != '' and not (isinstance(v, float) and pd.isna(v)):
                return v
        return None

    # Check if already in universe
    already_exists = _ticker_in_universe(ticker)

    # Add to universe.csv if requested and not already there
    if add_permanent and not already_exists:
        try:
            _add_ticker_to_universe(ticker, info)
            logger.info(f"Added {ticker} to universe.csv")
        except Exception as e:
            logger.warning(f"Could not add {ticker} to universe.csv: {e}")

    # =========================================================================
    # RUN FULL ANALYSIS PIPELINE (same as "Run Analysis" button)
    # =========================================================================
    saved_to_db = False
    scores = {}
    today = dt_date.today()

    try:

        # -----------------------------------------------------------------
        # STEP 1: Collect Fresh News
        # -----------------------------------------------------------------
        articles = []
        try:
            nc = NewsCollector()
            result = nc.collect_and_save(ticker, days_back=7, force_refresh=True)
            articles = result.get('articles', [])
            logger.info(f"{ticker}: Collected {len(articles)} news articles")
        except Exception as e:
            logger.warning(f"{ticker}: News collection failed: {e}")

        # -----------------------------------------------------------------
        # STEP 2: Analyze Sentiment
        # -----------------------------------------------------------------
        sentiment_score = None
        sentiment_data = {}
        if articles:
            try:
                sa = SentimentAnalyzer()
                sentiment_data = sa.analyze_ticker_sentiment(ticker, articles)
                sentiment_score = sentiment_data.get('sentiment_score')
                logger.info(f"{ticker}: Sentiment score = {sentiment_score}")
            except Exception as e:
                logger.warning(f"{ticker}: Sentiment analysis failed: {e}")

        # -----------------------------------------------------------------
        # STEP 3: Get Options Flow & Squeeze Scores
        # -----------------------------------------------------------------
        options_score = None
        squeeze_score = None
        try:
            scorer = UniverseScorer()
            scores_list, _ = scorer.score_and_save_universe(tickers=[ticker], max_workers=1)
            if scores_list:
                for score_obj in scores_list:
                    if score_obj.ticker == ticker:
                        options_score = score_obj.options_flow_score
                        squeeze_score = score_obj.short_squeeze_score
                        logger.info(f"{ticker}: Options={options_score}, Squeeze={squeeze_score}")
                        break
        except Exception as e:
            logger.warning(f"{ticker}: UniverseScorer failed: {e}")

        # -----------------------------------------------------------------
        # STEP 4: Get Technical Score
        # -----------------------------------------------------------------
        technical_score = None
        try:
            ta = TechnicalAnalyzer()
            tech_result = ta.analyze_ticker(ticker)
            if tech_result:
                technical_score = tech_result.get('technical_score', tech_result.get('score'))
                logger.info(f"{ticker}: Technical score = {technical_score}")
        except Exception as e:
            logger.debug(f"{ticker}: Technical analysis skipped: {e}")

        # -----------------------------------------------------------------
        # STEP 5: Get Fundamental Data (MERGED FROM ALL SOURCES)
        # -----------------------------------------------------------------
        fundamental_score = None
        growth_score = None
        dividend_score = None
        gap_score = 50  # Neutral default

        try:
            # Merge fundamental data from all sources (prefer: IBKR > Finviz > yfinance)
            # Valuation metrics
            pe_ratio = get_best_value(
                ibkr_data.get('pe_ratio'),
                finviz_data.get('pe'),
                info.get('trailingPE')
            )
            forward_pe = get_best_value(
                ibkr_data.get('forward_pe'),
                finviz_data.get('forward_pe'),
                info.get('forwardPE')
            )
            peg_ratio = get_best_value(
                ibkr_data.get('peg_ratio'),
                finviz_data.get('peg'),
                info.get('pegRatio')
            )

            # Profitability metrics
            roe = get_best_value(
                ibkr_data.get('roe'),
                finviz_data.get('roe'),
                info.get('returnOnEquity')
            )
            # Convert Finviz ROE from percentage to decimal if needed
            if roe and finviz_data.get('roe') and roe > 1:
                roe = roe / 100

            roa = get_best_value(
                ibkr_data.get('roa'),
                finviz_data.get('roa'),
                info.get('returnOnAssets')
            )
            if roa and finviz_data.get('roa') and roa > 1:
                roa = roa / 100

            profit_margin = get_best_value(
                ibkr_data.get('profit_margin'),
                finviz_data.get('profit_margin'),
                info.get('profitMargins')
            )
            if profit_margin and finviz_data.get('profit_margin') and profit_margin > 1:
                profit_margin = profit_margin / 100

            # Growth metrics
            revenue_growth = get_best_value(
                ibkr_data.get('revenue_growth'),
                finviz_data.get('sales_growth_yoy'),
                info.get('revenueGrowth')
            )
            if revenue_growth and finviz_data.get('sales_growth_yoy') and abs(revenue_growth) > 1:
                revenue_growth = revenue_growth / 100

            earnings_growth = get_best_value(
                ibkr_data.get('earnings_growth'),
                finviz_data.get('eps_growth_yoy'),
                info.get('earningsGrowth')
            )
            if earnings_growth and finviz_data.get('eps_growth_yoy') and abs(earnings_growth) > 1:
                earnings_growth = earnings_growth / 100

            # Dividend
            dividend_yield = get_best_value(
                ibkr_data.get('dividend_yield'),
                finviz_data.get('dividend_yield'),
                info.get('dividendYield')
            )
            if dividend_yield and finviz_data.get('dividend_yield') and dividend_yield > 1:
                dividend_yield = dividend_yield / 100

            # Financial health
            debt_equity = get_best_value(
                ibkr_data.get('debt_equity'),
                finviz_data.get('debt_eq'),
                info.get('debtToEquity')
            )
            current_ratio = get_best_value(
                ibkr_data.get('current_ratio'),
                finviz_data.get('current_ratio'),
                info.get('currentRatio')
            )

            # Company info
            sector = get_best_value(
                info.get('sector'),
                'Unknown'
            )
            industry = get_best_value(
                info.get('industry'),
                'Unknown'
            )

            # Finviz-specific data
            inst_own = finviz_data.get('inst_own')
            insider_own = finviz_data.get('insider_own')
            short_float = finviz_data.get('short_float')
            target_price_finviz = finviz_data.get('target_price')
            beta = get_best_value(finviz_data.get('beta'), info.get('beta'))
            rsi = finviz_data.get('rsi')

            # Log data sources used
            sources_used = []
            if any(ibkr_data.values()):
                sources_used.append("IBKR")
            if any(finviz_data.values()):
                sources_used.append("Finviz")
            sources_used.append("yfinance")
            logger.info(f"{ticker}: Data sources: {', '.join(sources_used)}")

            # Enhanced fundamental score calculation using all sources
            fund_points = 50  # Start neutral

            # PE analysis
            if pe_ratio and pe_ratio > 0:
                if pe_ratio < 15:
                    fund_points += 10
                elif pe_ratio < 25:
                    fund_points += 5
                elif pe_ratio > 50:
                    fund_points -= 10

            # PEG analysis
            if peg_ratio and peg_ratio > 0:
                if peg_ratio < 1:
                    fund_points += 10
                elif peg_ratio < 2:
                    fund_points += 5
                elif peg_ratio > 3:
                    fund_points -= 5

            # Profitability analysis
            if roe and roe > 0.15:
                fund_points += 5
            if roe and roe > 0.25:
                fund_points += 5
            if profit_margin and profit_margin > 0.1:
                fund_points += 5

            # Financial health
            if current_ratio and current_ratio > 1.5:
                fund_points += 5
            if debt_equity and debt_equity < 0.5:
                fund_points += 5
            elif debt_equity and debt_equity > 2:
                fund_points -= 5

            # Institutional ownership (from Finviz)
            if inst_own is not None:
                if inst_own >= 70:
                    fund_points += 5
                elif inst_own < 20:
                    fund_points -= 5

            # Short interest (from Finviz) - bearish indicator
            if short_float is not None:
                if short_float >= 20:
                    fund_points -= 10
                elif short_float >= 10:
                    fund_points -= 5

            fundamental_score = max(0, min(100, fund_points))

            # Growth score (enhanced)
            growth_points = 50
            if revenue_growth:
                if revenue_growth > 0.2:
                    growth_points += 20
                elif revenue_growth > 0.1:
                    growth_points += 15
                elif revenue_growth > 0:
                    growth_points += 5
                elif revenue_growth < -0.1:
                    growth_points -= 15
            if earnings_growth:
                if earnings_growth > 0.2:
                    growth_points += 20
                elif earnings_growth > 0.1:
                    growth_points += 15
                elif earnings_growth > 0:
                    growth_points += 5
                elif earnings_growth < -0.1:
                    growth_points -= 15
            growth_score = max(0, min(100, growth_points))

            # Dividend score
            dividend_points = 50
            if dividend_yield:
                if dividend_yield > 0.04:
                    dividend_points += 30
                elif dividend_yield > 0.02:
                    dividend_points += 20
                elif dividend_yield > 0.01:
                    dividend_points += 10
            dividend_score = max(0, min(100, dividend_points))

            logger.info(f"{ticker}: Fundamental={fundamental_score}, Growth={growth_score}, Dividend={dividend_score}")
            if inst_own is not None:
                logger.info(f"{ticker}: Finviz - Inst Own: {inst_own}%, Short Float: {short_float}%")

            # Save fundamentals to database (matching actual schema)
            with get_connection() as conn:
                with conn.cursor() as cur:
                    try:
                        # Check if record exists for today
                        cur.execute("SELECT 1 FROM fundamentals WHERE ticker = %s AND date = %s", (ticker, today))
                        exists = cur.fetchone() is not None

                        if exists:
                            cur.execute("""
                                UPDATE fundamentals SET
                                    pe_ratio = COALESCE(%s, pe_ratio),
                                    forward_pe = COALESCE(%s, forward_pe),
                                    peg_ratio = COALESCE(%s, peg_ratio),
                                    dividend_yield = COALESCE(%s, dividend_yield),
                                    revenue_growth = COALESCE(%s, revenue_growth),
                                    earnings_growth = COALESCE(%s, earnings_growth),
                                    roe = COALESCE(%s, roe),
                                    debt_to_equity = COALESCE(%s, debt_to_equity),
                                    current_ratio = COALESCE(%s, current_ratio),
                                    sector = COALESCE(%s, sector)
                                WHERE ticker = %s AND date = %s
                            """, (
                                _to_native(pe_ratio), _to_native(forward_pe), _to_native(peg_ratio),
                                _to_native(dividend_yield), _to_native(revenue_growth), _to_native(earnings_growth),
                                _to_native(roe), _to_native(debt_equity), _to_native(current_ratio),
                                sector,
                                ticker, today
                            ))
                        else:
                            cur.execute("""
                                INSERT INTO fundamentals (ticker, date, pe_ratio, forward_pe, peg_ratio,
                                                          dividend_yield, revenue_growth, earnings_growth, roe,
                                                          debt_to_equity, current_ratio, sector)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                ticker, today,
                                _to_native(pe_ratio), _to_native(forward_pe), _to_native(peg_ratio),
                                _to_native(dividend_yield), _to_native(revenue_growth), _to_native(earnings_growth),
                                _to_native(roe), _to_native(debt_equity), _to_native(current_ratio),
                                sector
                            ))
                        conn.commit()
                        logger.info(f"{ticker}: Saved fundamentals with sector='{sector}'")
                    except Exception as db_err:
                        logger.debug(f"{ticker}: Fundamentals save failed: {db_err}")
                        conn.rollback()

        except Exception as e:
            logger.debug(f"{ticker}: Fundamental data fetch failed: {e}")

        # -----------------------------------------------------------------
        # STEP 5b: Save Price to prices table
        # -----------------------------------------------------------------
        current_price = None
        try:
            current_price = info.get('regularMarketPrice') or info.get('currentPrice')
            if current_price:
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO prices (ticker, date, close, open, high, low, volume)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (ticker, date) DO UPDATE SET
                                close = EXCLUDED.close,
                                open = EXCLUDED.open,
                                high = EXCLUDED.high,
                                low = EXCLUDED.low,
                                volume = EXCLUDED.volume
                        """, (
                            ticker, today,
                            _to_native(current_price),
                            _to_native(info.get('regularMarketOpen')),
                            _to_native(info.get('regularMarketDayHigh')),
                            _to_native(info.get('regularMarketDayLow')),
                            _to_native(info.get('regularMarketVolume'))
                        ))
                    conn.commit()
                logger.info(f"{ticker}: Saved price ${current_price:.2f} to prices table")
        except Exception as e:
            logger.debug(f"{ticker}: Price save failed: {e}")

        # -----------------------------------------------------------------
        # STEP 5c: Save Analyst Ratings (merged from yfinance + Finviz)
        # -----------------------------------------------------------------
        try:
            # Get analyst recommendations from yfinance
            recommendations = info.get('recommendationKey', '')
            target_mean = info.get('targetMeanPrice')
            target_high = info.get('targetHighPrice')
            target_low = info.get('targetLowPrice')
            num_analysts = info.get('numberOfAnalystOpinions', 0)

            buy_count = 0
            hold_count = 0
            sell_count = 0

            # Method 1: Try recommendationTrend from yfinance info
            rec_summary = info.get('recommendationTrend', {}).get('trend', [])
            if rec_summary and len(rec_summary) > 0:
                latest = rec_summary[0]
                buy_count = (latest.get('strongBuy', 0) or 0) + (latest.get('buy', 0) or 0)
                hold_count = latest.get('hold', 0) or 0
                sell_count = (latest.get('sell', 0) or 0) + (latest.get('strongSell', 0) or 0)
                logger.info(f"{ticker}: Got analyst data from yfinance recommendationTrend")

            # Method 2: If no data, try yfinance recommendations DataFrame
            if buy_count == 0 and hold_count == 0 and sell_count == 0:
                try:
                    rec_df = yf_ticker.recommendations
                    if rec_df is not None and not rec_df.empty:
                        latest = rec_df.iloc[-1]
                        buy_count = (latest.get('strongBuy', 0) or 0) + (latest.get('buy', 0) or 0)
                        hold_count = latest.get('hold', 0) or 0
                        sell_count = (latest.get('sell', 0) or 0) + (latest.get('strongSell', 0) or 0)
                        logger.info(f"{ticker}: Got analyst data from yfinance recommendations DataFrame")
                except Exception as e:
                    logger.debug(f"{ticker}: Could not get yfinance recommendations DataFrame: {e}")

            # Method 3: Use Finviz data if still no yfinance data
            total_ratings = buy_count + hold_count + sell_count
            if total_ratings == 0 and finviz_data:
                finviz_total = finviz_data.get('finviz_total_ratings', 0) or 0
                finviz_buy_pct = finviz_data.get('finviz_buy_pct', 0) or 0
                if finviz_total > 0:
                    # Estimate buy/sell counts from Finviz buy percentage
                    buy_count = int(finviz_total * finviz_buy_pct / 100)
                    sell_count = int(finviz_total * 0.1)  # Estimate 10% sell
                    hold_count = finviz_total - buy_count - sell_count
                    total_ratings = finviz_total
                    logger.info(f"{ticker}: Using Finviz analyst data - {finviz_buy_pct}% positive ({finviz_total} ratings)")

            if total_ratings == 0:
                total_ratings = num_analysts or 0

            # Calculate analyst positivity (% buy)
            analyst_positivity = (buy_count / total_ratings * 100) if total_ratings > 0 else None

            # Merge with Finviz target price if yfinance is missing
            if not target_mean and finviz_data.get('target_price'):
                target_mean = finviz_data.get('target_price')
                logger.info(f"{ticker}: Using Finviz target price: ${target_mean}")

            if total_ratings > 0 or target_mean:
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        # Check if record exists for today
                        cur.execute("SELECT 1 FROM analyst_ratings WHERE ticker = %s AND date = %s", (ticker, today))
                        exists = cur.fetchone() is not None

                        if exists:
                            cur.execute("""
                                UPDATE analyst_ratings SET
                                    analyst_buy = COALESCE(%s, analyst_buy),
                                    analyst_hold = COALESCE(%s, analyst_hold),
                                    analyst_sell = COALESCE(%s, analyst_sell),
                                    analyst_total = COALESCE(%s, analyst_total),
                                    analyst_positivity = COALESCE(%s, analyst_positivity)
                                WHERE ticker = %s AND date = %s
                            """, (
                                _to_native(buy_count) if buy_count > 0 else None,
                                _to_native(hold_count) if hold_count > 0 else None,
                                _to_native(sell_count) if sell_count > 0 else None,
                                _to_native(total_ratings) if total_ratings > 0 else None,
                                _to_native(analyst_positivity),
                                ticker, today
                            ))
                        else:
                            cur.execute("""
                                INSERT INTO analyst_ratings (ticker, date, analyst_buy, analyst_hold, analyst_sell,
                                                             analyst_total, analyst_positivity)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, (
                                ticker, today,
                                _to_native(buy_count) if buy_count > 0 else None,
                                _to_native(hold_count) if hold_count > 0 else None,
                                _to_native(sell_count) if sell_count > 0 else None,
                                _to_native(total_ratings) if total_ratings > 0 else None,
                                _to_native(analyst_positivity)
                            ))
                    conn.commit()
                logger.info(f"{ticker}: Saved analyst ratings - Buy:{buy_count} Hold:{hold_count} Sell:{sell_count} Total:{total_ratings}" + (f" Positivity:{analyst_positivity:.1f}%" if analyst_positivity else ""))
        except Exception as e:
            logger.debug(f"{ticker}: Analyst ratings save failed: {e}")

        # -----------------------------------------------------------------
        # STEP 5d: Save Price Targets (yfinance + Finviz fallback)
        # -----------------------------------------------------------------
        try:
            # Get from yfinance first
            target_mean = info.get('targetMeanPrice')
            target_high = info.get('targetHighPrice')
            target_low = info.get('targetLowPrice')
            target_median = info.get('targetMedianPrice')

            # Use Finviz target as fallback
            if not target_mean and finviz_data.get('target_price'):
                target_mean = finviz_data.get('target_price')
                logger.info(f"{ticker}: Using Finviz target price: ${target_mean}")

            if target_mean:
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        # Check if record exists for today
                        cur.execute("SELECT 1 FROM price_targets WHERE ticker = %s AND date = %s", (ticker, today))
                        exists = cur.fetchone() is not None

                        if exists:
                            cur.execute("""
                                UPDATE price_targets SET
                                    target_mean = COALESCE(%s, target_mean),
                                    target_high = COALESCE(%s, target_high),
                                    target_low = COALESCE(%s, target_low)
                                WHERE ticker = %s AND date = %s
                            """, (
                                _to_native(target_mean),
                                _to_native(target_high),
                                _to_native(target_low),
                                ticker, today
                            ))
                        else:
                            cur.execute("""
                                INSERT INTO price_targets (ticker, date, target_mean, target_high, target_low)
                                VALUES (%s, %s, %s, %s, %s)
                            """, (
                                ticker, today,
                                _to_native(target_mean),
                                _to_native(target_high),
                                _to_native(target_low)
                            ))
                    conn.commit()
                logger.info(f"{ticker}: Saved price target ${target_mean:.2f}")
        except Exception as e:
            logger.debug(f"{ticker}: Price target save failed: {e}")

        # -----------------------------------------------------------------
        # STEP 5e: Save Earnings Calendar
        # -----------------------------------------------------------------
        try:
            earnings_dates = yf_ticker.earnings_dates
            if earnings_dates is not None and not earnings_dates.empty:
                # Get next upcoming earnings date
                for idx in earnings_dates.index:
                    earnings_dt = idx.date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
                    if earnings_dt >= today:
                        with get_connection() as conn:
                            with conn.cursor() as cur:
                                try:
                                    cur.execute("""
                                        INSERT INTO earnings_calendar (ticker, earnings_date)
                                        VALUES (%s, %s)
                                        ON CONFLICT (ticker) DO UPDATE SET
                                            earnings_date = EXCLUDED.earnings_date
                                    """, (ticker, earnings_dt))
                                except:
                                    pass
                            conn.commit()
                        logger.info(f"{ticker}: Saved earnings date {earnings_dt}")
                        break
        except Exception as e:
            logger.debug(f"{ticker}: Earnings calendar save failed: {e}")

        # -----------------------------------------------------------------
        # STEP 5f: Update fundamentals with ex_dividend_date
        # -----------------------------------------------------------------
        try:
            ex_div_date = info.get('exDividendDate')
            if ex_div_date:
                # Convert timestamp to date
                if isinstance(ex_div_date, (int, float)):
                    from datetime import datetime
                    ex_div_date = datetime.fromtimestamp(ex_div_date).date()

                with get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            UPDATE fundamentals SET ex_dividend_date = %s WHERE ticker = %s
                        """, (ex_div_date, ticker))
                    conn.commit()
                logger.info(f"{ticker}: Saved ex-dividend date {ex_div_date}")
        except Exception as e:
            logger.debug(f"{ticker}: Ex-dividend date save failed: {e}")

        # -----------------------------------------------------------------
        # STEP 6: Calculate Total Score
        # -----------------------------------------------------------------
        available_scores = []
        available_weights = []

        if sentiment_score is not None:
            available_scores.append(sentiment_score * 0.25)
            available_weights.append(0.25)
        if fundamental_score is not None:
            available_scores.append(fundamental_score * 0.25)
            available_weights.append(0.25)
        if technical_score is not None:
            available_scores.append(technical_score * 0.25)
            available_weights.append(0.25)
        if options_score is not None:
            available_scores.append(options_score * 0.15)
            available_weights.append(0.15)
        if squeeze_score is not None:
            available_scores.append(squeeze_score * 0.10)
            available_weights.append(0.10)

        if available_weights:
            total_weight = sum(available_weights)
            total_score = round(sum(available_scores) / total_weight) if total_weight > 0 else 50
            total_score = max(0, min(100, total_score))
        else:
            total_score = 50

        logger.info(f"{ticker}: Total score = {total_score}")

        # -----------------------------------------------------------------
        # STEP 7: Save to screener_scores (UPSERT)
        # -----------------------------------------------------------------
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO screener_scores (
                        ticker, date, 
                        sentiment_score, sentiment_weighted, article_count,
                        fundamental_score, technical_score, growth_score, dividend_score,
                        gap_score, total_score,
                        options_flow_score, short_squeeze_score,
                        created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                    )
                    ON CONFLICT (ticker, date) DO UPDATE SET
                        sentiment_score = EXCLUDED.sentiment_score,
                        sentiment_weighted = EXCLUDED.sentiment_weighted,
                        article_count = EXCLUDED.article_count,
                        fundamental_score = EXCLUDED.fundamental_score,
                        technical_score = EXCLUDED.technical_score,
                        growth_score = EXCLUDED.growth_score,
                        dividend_score = EXCLUDED.dividend_score,
                        gap_score = EXCLUDED.gap_score,
                        total_score = EXCLUDED.total_score,
                        options_flow_score = EXCLUDED.options_flow_score,
                        short_squeeze_score = EXCLUDED.short_squeeze_score
                """, (
                    ticker, today,
                    _to_native(sentiment_score),
                    _to_native(sentiment_data.get('sentiment_weighted', sentiment_score)),
                    _to_native(len(articles)),
                    _to_native(fundamental_score),
                    _to_native(technical_score),
                    _to_native(growth_score),
                    _to_native(dividend_score),
                    _to_native(gap_score),
                    _to_native(total_score),
                    _to_native(options_score),
                    _to_native(squeeze_score),
                ))
            conn.commit()
            saved_to_db = True
            logger.info(f"{ticker}: Saved to screener_scores for {today}")

        # -----------------------------------------------------------------
        # STEP 8: Save sentiment_scores
        # -----------------------------------------------------------------
        if sentiment_score is not None:
            sentiment_class = (
                'Very Bullish' if sentiment_score >= 70 else
                'Bullish' if sentiment_score >= 55 else
                'Neutral' if sentiment_score >= 45 else
                'Bearish' if sentiment_score >= 30 else
                'Very Bearish'
            )
            try:
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO sentiment_scores
                            (ticker, date, sentiment_raw, sentiment_weighted, ai_sentiment_fast,
                             article_count, relevant_article_count, sentiment_class)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s) 
                            ON CONFLICT (ticker, date) DO UPDATE SET
                                sentiment_raw = EXCLUDED.sentiment_raw,
                                sentiment_weighted = EXCLUDED.sentiment_weighted,
                                ai_sentiment_fast = EXCLUDED.ai_sentiment_fast,
                                article_count = EXCLUDED.article_count,
                                sentiment_class = EXCLUDED.sentiment_class
                        """, (
                            ticker, today,
                            _to_native(sentiment_score),
                            _to_native(sentiment_data.get('sentiment_weighted', sentiment_score)),
                            _to_native(sentiment_score),
                            _to_native(len(articles)),
                            _to_native(sentiment_data.get('relevant_count', len(articles))),
                            sentiment_class
                        ))
                    conn.commit()
            except Exception as e:
                logger.warning(f"{ticker}: sentiment_scores save failed: {e}")

        # Build scores dict for return
        scores = {
            'sentiment_score': sentiment_score,
            'fundamental_score': fundamental_score,
            'technical_score': technical_score,
            'options_flow_score': options_score,
            'short_squeeze_score': squeeze_score,
            'growth_score': growth_score,
            'dividend_score': dividend_score,
            'total_score': total_score,
            'article_count': len(articles),
        }

    except Exception as e:
        logger.error(f"{ticker}: Full analysis pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # -----------------------------------------------------------------
    # STEP 9: Generate UnifiedSignal with committee
    # -----------------------------------------------------------------
    signal = None
    try:
        from src.core import get_signal_engine
        engine = get_signal_engine()

        # Clear cache for this ticker
        if hasattr(engine, '_cache') and ticker in engine._cache:
            del engine._cache[ticker]

        signal = engine.generate_signal(ticker, force_refresh=True)
        logger.info(f"{ticker}: Generated signal - {signal.today_signal if signal else 'None'}")
    except Exception as e:
        logger.error(f"SignalEngine failed for {ticker}: {e}")

    # -----------------------------------------------------------------
    # STEP 10: Run AI Analysis (BatchAIAnalyzer) for deep insights
    # -----------------------------------------------------------------
    ai_result = None
    try:
        from src.ai.batch_ai_analysis import BatchAIAnalyzer

        analyzer = BatchAIAnalyzer()

        # Build signal data for AI context
        signal_data = {
            'total_score': scores.get('total_score', 50) if scores else 50,
            'signal_type': signal.today_signal.value if signal and signal.today_signal else 'HOLD',
            'technical_score': technical_score,
            'fundamental_score': fundamental_score,
            'sentiment_score': sentiment_score,
        }

        # Run AI analysis (fast_mode=False for full context with all collected data)
        logger.info(f"{ticker}: Running AI analysis...")
        ai_result = analyzer.analyze_ticker(ticker, signal_data, fast_mode=False)

        if ai_result:
            # Save to database
            save_ok = analyzer.save_result(ai_result)
            if save_ok:
                logger.info(f"{ticker}: AI Analysis complete - {ai_result.ai_action} ({ai_result.ai_confidence})")
            else:
                logger.warning(f"{ticker}: AI Analysis done but save failed")
        else:
            logger.warning(f"{ticker}: AI Analysis returned no result")

    except ImportError:
        logger.debug(f"{ticker}: BatchAIAnalyzer not available - skipping AI analysis")
    except Exception as e:
        logger.warning(f"{ticker}: AI Analysis failed: {e}")

    return {
        'success': True,
        'ticker': ticker,
        'signal': signal,
        'scores': scores,
        'added_to_watchlist': add_permanent and not already_exists,
        'saved_to_db': saved_to_db,
        'already_existed': already_exists,
        'force_refreshed': force_refresh,
        'ai_result': ai_result
    }


def _ticker_in_universe(ticker: str) -> bool:
    """Check if ticker already exists in universe.csv."""
    import os

    try:
        # Find universe.csv path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        universe_file = os.path.join(project_root, 'config', 'universe.csv')

        if not os.path.exists(universe_file):
            return False

        df = pd.read_csv(universe_file)
        return ticker.upper() in df['ticker'].str.upper().values
    except Exception as e:
        logger.debug(f"Error checking universe: {e}")
        return False


def _add_ticker_to_universe(ticker: str, yf_info: dict) -> bool:
    """
    Add a ticker to universe.csv with metadata from yfinance.

    Args:
        ticker: Stock ticker symbol
        yf_info: yfinance info dict

    Returns:
        True if added successfully
    """
    import os

    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        universe_file = os.path.join(project_root, 'config', 'universe.csv')

        # Read existing universe
        if os.path.exists(universe_file):
            df = pd.read_csv(universe_file)
        else:
            df = pd.DataFrame(columns=['ticker', 'name', 'sector', 'industry'])

        # Check if already exists
        if ticker.upper() in df['ticker'].str.upper().values:
            logger.info(f"{ticker} already in universe.csv")
            return True

        # Extract info from yfinance
        name = yf_info.get('shortName') or yf_info.get('longName') or ticker
        sector = yf_info.get('sector') or 'Unknown'
        industry = yf_info.get('industry') or 'Unknown'

        # Add new row
        new_row = pd.DataFrame([{
            'ticker': ticker.upper(),
            'name': name,
            'sector': sector,
            'industry': industry
        }])

        df = pd.concat([df, new_row], ignore_index=True)

        # Save
        df.to_csv(universe_file, index=False)
        logger.info(f"Added {ticker} to universe.csv: {name} ({sector}/{industry})")

        return True

    except Exception as e:
        logger.error(f"Error adding {ticker} to universe: {e}")
        raise


def _remove_ticker_from_universe(ticker: str) -> dict:
    """
    Remove a ticker from universe.csv and from all database tables.

    Args:
        ticker: Stock ticker symbol to remove

    Returns:
        dict with 'success', 'message', 'removed_from' list
    """
    import os

    ticker = ticker.upper().strip()
    removed_from = []
    errors = []

    try:
        # 1. Remove from universe.csv
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        universe_file = os.path.join(project_root, 'config', 'universe.csv')

        if os.path.exists(universe_file):
            df = pd.read_csv(universe_file)
            original_count = len(df)
            df = df[df['ticker'].str.upper() != ticker]

            if len(df) < original_count:
                df.to_csv(universe_file, index=False)
                removed_from.append('universe.csv')
                logger.info(f"Removed {ticker} from universe.csv")
            else:
                logger.info(f"{ticker} was not in universe.csv")

        # 2. Remove from database tables (ALWAYS try, regardless of universe.csv)
        try:

            tables_to_clean = [
                'screener_scores',
                'news_articles',
                'options_flow',
                'options_flow_scores',
                'trading_signals',
                'signals',
                'earnings_calendar',
                'historical_scores',
                'fundamentals',
                'analyst_ratings',
                'price_targets',
                'prices',
                'sentiment_scores',
                'committee_decisions',
                'agent_votes',
                'insider_transactions',
            ]

            with get_connection() as conn:
                with conn.cursor() as cur:
                    for table in tables_to_clean:
                        try:
                            # Check if table exists first
                            cur.execute("""
                                SELECT EXISTS (
                                    SELECT FROM information_schema.tables 
                                    WHERE table_name = %s
                                )
                            """, (table,))
                            if cur.fetchone()[0]:
                                cur.execute(f"DELETE FROM {table} WHERE ticker = %s", (ticker,))
                                if cur.rowcount > 0:
                                    removed_from.append(f'{table} ({cur.rowcount} rows)')
                                    logger.info(f"Removed {cur.rowcount} rows for {ticker} from {table}")
                        except Exception as e:
                            logger.debug(f"Could not clean {table}: {e}")

                conn.commit()

        except ImportError as e:
            errors.append(f"Database import error: {e}")
            logger.error(f"Database connection import failed: {e}")
        except Exception as e:
            errors.append(f"Database cleanup: {str(e)}")
            logger.warning(f"Database cleanup error: {e}")

        if removed_from:
            return {
                'success': True,
                'message': f"Successfully removed {ticker}",
                'removed_from': removed_from,
                'errors': errors
            }
        else:
            return {
                'success': False,
                'message': f"{ticker} was not found in watchlist or database",
                'removed_from': [],
                'errors': errors
            }

    except Exception as e:
        logger.error(f"Error removing {ticker}: {e}")
        return {
            'success': False,
            'message': f"Error removing {ticker}: {str(e)}",
            'removed_from': removed_from,
            'errors': [str(e)]
        }

def _get_universe_tickers() -> List[str]:
    """Get list of all tickers in universe.csv."""
    import os

    try:
        # From src/tabs/signals_tab/ go 4 levels up to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        universe_file = os.path.join(project_root, 'config', 'universe.csv')

        if os.path.exists(universe_file):
            df = pd.read_csv(universe_file)
            return sorted(df['ticker'].str.upper().tolist())
        
        # Fallback: try relative to current working directory
        if os.path.exists('config/universe.csv'):
            df = pd.read_csv('config/universe.csv')
            return sorted(df['ticker'].str.upper().tolist())
            
        logger.warning(f"Universe file not found at {universe_file}")
        return []
    except Exception as e:
        logger.error(f"Error reading universe: {e}")
        return []


def _render_remove_ticker():
    """
    Render the Remove Ticker UI.
    Allows users to remove tickers from their watchlist/universe.
    """
    import re

    # Initialize session state
    if 'remove_result' not in st.session_state:
        st.session_state.remove_result = None
    if 'remove_error' not in st.session_state:
        st.session_state.remove_error = None

    st.markdown("**Remove tickers from your watchlist** - removes from universe.csv and cleans up database entries.")

    # Get current universe for dropdown
    universe_tickers = _get_universe_tickers()

    # Two input methods: dropdown or text input
    col1, col2 = st.columns([2, 1])

    with col1:
        # Text input for manual entry (supports multiple)
        ticker_input = st.text_input(
            "Tickers to remove",
            placeholder="e.g., ADPT, DYN or select from dropdown",
            key="remove_ticker_input",
            label_visibility="collapsed"
        ).upper().strip()

    with col2:
        # Dropdown for existing tickers
        selected_from_list = st.selectbox(
            "Or select from watchlist",
            options=[""] + universe_tickers,
            key="remove_ticker_select",
            label_visibility="collapsed"
        )

    # Combine inputs
    tickers_to_remove = []
    if ticker_input:
        tickers_to_remove = [t.strip().upper() for t in re.split(r'[,;\s]+', ticker_input) if t.strip()]
    elif selected_from_list:
        tickers_to_remove = [selected_from_list]

    col1, col2 = st.columns([1, 1])

    with col1:
        also_clean_db = st.checkbox(
            "Also clean database entries",
            value=True,
            help="Remove associated data from screener_scores, news_articles, options_flow, etc."
        )

    with col2:
        remove_clicked = st.button(
            "🗑️ Remove",
            type="secondary",
            disabled=not tickers_to_remove,
            use_container_width=True
        )

    # Process removal
    if remove_clicked and tickers_to_remove:
        st.session_state.remove_result = None
        st.session_state.remove_error = None

        results = []
        errors = []

        with st.spinner(f"Removing {len(tickers_to_remove)} ticker(s)..."):
            for ticker in tickers_to_remove:
                result = _remove_ticker_from_universe(ticker)
                if result['success']:
                    results.append(result)
                else:
                    errors.append(f"{ticker}: {result['message']}")

        if results:
            st.session_state.remove_result = {
                'count': len(results),
                'tickers': [r['message'] for r in results],
                'details': results
            }
            # Force refresh of signals table
            st.session_state.signals_loaded = False
            if 'signals_data' in st.session_state:
                del st.session_state['signals_data']

        if errors:
            st.session_state.remove_error = "; ".join(errors)

        st.rerun()

    # Show results
    if st.session_state.remove_error:
        st.error(f"❌ {st.session_state.remove_error}")

    if st.session_state.remove_result:
        result = st.session_state.remove_result
        st.success(f"✅ Removed {result['count']} ticker(s)")

        # Show details
        for detail in result.get('details', []):
            if detail.get('removed_from'):
                with st.expander(f"📋 {detail['message']}", expanded=False):
                    st.write("Removed from:")
                    for loc in detail['removed_from']:
                        st.write(f"  • {loc}")

        if st.button("🗑️ Clear", key="clear_remove_result"):
            st.session_state.remove_result = None
            st.session_state.remove_error = None
            st.rerun()


