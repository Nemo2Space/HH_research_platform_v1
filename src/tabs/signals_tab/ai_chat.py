"""
Signals Tab - AI Chat

AI chat functionality, context builders, AI response generation.
"""
from .shared import (
    st, pd, np, logger, datetime, date, timedelta, timezone, time, json,
    Dict, List, Optional, _to_native, text,
    get_engine, get_connection,
    SIGNAL_HUB_AVAILABLE, DB_AVAILABLE, AI_SYSTEM_AVAILABLE,
    INSTITUTIONAL_CONTEXT_AVAILABLE,
    get_institutional_signals_context, get_trading_implications,
    AITradingSystem, UnifiedSignal, MarketSnapshot, compute_trade_levels, resolve_signal_conflict,
    apply_uncertainty_shrinkage, _get_extended_hours_price,
)

def _get_recent_news_headlines(ticker: str) -> list:
    """Get recent news headlines for a ticker from database."""
    try:
        import pandas as pd
        engine = get_engine()
        query = """
            SELECT headline FROM news_articles
            WHERE ticker = %s
            AND published_at >= NOW() - INTERVAL '7 days'
            ORDER BY published_at DESC
            LIMIT 15
        """
        df = pd.read_sql(query, engine, params=(ticker,))
        return df['headline'].tolist() if not df.empty else []
    except Exception:
        return []

def _render_ai_chat(signal: UnifiedSignal, additional_data: dict):
    """Render AI chat section with pre-computed metrics (LLM should not calculate)."""

    st.markdown("### 🤖 Ask AI About This Stock")

    # Get extended hours price for context
    extended_price = _get_extended_hours_price(signal.ticker)

    # ==========================================================================
    # BUILD MARKET SNAPSHOT - single source of truth for all metrics
    # ==========================================================================
    snapshot = MarketSnapshot(
        ticker=signal.ticker,
        snapshot_ts_utc=datetime.now(timezone.utc)
    )

    # Price data
    if extended_price.get('has_extended') and extended_price.get('extended_price', 0) > 0:
        snapshot.price = extended_price['extended_price']
        snapshot.price_source = extended_price.get('session', 'unknown').lower()
    else:
        snapshot.price = signal.current_price
        snapshot.price_source = 'regular'

    snapshot.previous_close = extended_price.get('regular_close', signal.current_price)
    if snapshot.previous_close > 0:
        snapshot.price_change_pct = ((snapshot.price - snapshot.previous_close) / snapshot.previous_close) * 100

    # 52W data - calculated from snapshot price
    snapshot.week_52_high = signal.week_52_high
    snapshot.week_52_low = signal.week_52_low
    if snapshot.week_52_high > 0 and snapshot.price > 0:
        snapshot.pct_from_52w_high = ((snapshot.price - snapshot.week_52_high) / snapshot.week_52_high) * 100
    if snapshot.week_52_low > 0 and snapshot.price > 0:
        snapshot.pct_from_52w_low = ((snapshot.price - snapshot.week_52_low) / snapshot.week_52_low) * 100

    # Scores
    snapshot.sentiment_score = signal.sentiment_score if signal.sentiment_score is not None else None
    snapshot.technical_score = signal.technical_score if signal.technical_score is not None else None
    snapshot.fundamental_score = signal.fundamental_score if signal.fundamental_score is not None else None
    snapshot.options_score = signal.options_score if signal.options_score is not None else None
    snapshot.earnings_score = signal.earnings_score if signal.earnings_score is not None else None
    snapshot.total_score = signal.today_score

    # Count available components
    components = [snapshot.sentiment_score, snapshot.technical_score,
                  snapshot.fundamental_score, snapshot.options_score, snapshot.earnings_score]
    snapshot.components_available = sum(1 for c in components if c is not None)
    snapshot.components_total = 5

    # Target
    target_mean = additional_data.get('target_mean', 0)
    if target_mean and target_mean > 0:
        snapshot.analyst_target = target_mean
        if snapshot.price > 0:
            snapshot.target_upside_pct = ((target_mean / snapshot.price) - 1) * 100

    # ==========================================================================
    # COMPUTE TRADE LEVELS (deterministic - LLM should NOT recalculate)
    # ==========================================================================
    risk_level = signal.risk_level.value if hasattr(signal.risk_level, 'value') else str(signal.risk_level)
    trade_levels = compute_trade_levels(
        current_price=snapshot.price,
        signal_score=signal.today_score or 50,
        risk_level=risk_level,
        analyst_target=snapshot.analyst_target,
        max_pain=None,  # Will be filled from options context
        week_52_high=snapshot.week_52_high,
        week_52_low=snapshot.week_52_low
    )

    # ==========================================================================
    # RESOLVE SIGNAL CONFLICT (deterministic policy)
    # ==========================================================================
    platform_direction = signal.today_signal.value if hasattr(signal.today_signal, 'value') else str(signal.today_signal)
    committee_direction = signal.committee_verdict or "HOLD"

    # Get ML status from additional_data
    ml_status = additional_data.get('ml_status', 'UNKNOWN')
    if 'alpha_context' in additional_data:
        alpha_ctx = additional_data.get('alpha_context', {})
        if isinstance(alpha_ctx, dict):
            ml_status = alpha_ctx.get('ml_status', ml_status)

    policy_decision = resolve_signal_conflict(
        platform_signal=platform_direction,
        platform_score=signal.today_score or 50,
        committee_signal=committee_direction,
        committee_confidence=signal.committee_confidence or 0.5,
        components_fresh=snapshot.components_available,
        ml_status=ml_status
    )

    # Update trade levels with policy decision
    trade_levels.position_size_multiplier = policy_decision.max_size

    # ==========================================================================
    # APPLY UNCERTAINTY SHRINKAGE
    # ==========================================================================
    if snapshot.total_score is not None:
        shrunk_score, data_confidence = apply_uncertainty_shrinkage(
            raw_score=snapshot.total_score,
            components_available=snapshot.components_available,
            components_total=snapshot.components_total,
            staleness_penalty=snapshot.staleness_penalty
        )
    else:
        shrunk_score, data_confidence = 50, "NONE"

    # ==========================================================================
    # BUILD CONTEXT WITH PRE-COMPUTED METRICS
    # ==========================================================================
    price_timestamp = snapshot.snapshot_ts_utc.strftime("%Y-%m-%d %H:%M UTC")

    # Conflict description
    if policy_decision.conflict_flag:
        conflict_desc = f"⚠️ CONFLICT: {' | '.join(policy_decision.reasons)}"
    else:
        conflict_desc = "✅ ALIGNED: Platform and Committee signals agree"

    # Pre-compute conditional strings (can't nest f-strings)
    analyst_target_str = f"${snapshot.analyst_target:.2f}" if snapshot.analyst_target else "UNKNOWN"
    target_upside_str = f"{snapshot.target_upside_pct:+.1f}%" if snapshot.target_upside_pct else "UNKNOWN"
    technical_str = snapshot.technical_score if snapshot.technical_score is not None else 'UNKNOWN'
    fundamental_str = snapshot.fundamental_score if snapshot.fundamental_score is not None else 'UNKNOWN'
    sentiment_str = snapshot.sentiment_score if snapshot.sentiment_score is not None else 'UNKNOWN'
    options_str = snapshot.options_score if snapshot.options_score is not None else 'UNKNOWN'
    earnings_str = snapshot.earnings_score if snapshot.earnings_score is not None else 'UNKNOWN'

    context = f"""
=== {signal.ticker} - {signal.company_name} ===
Sector: {signal.sector}
Data as of: {price_timestamp}

============================================================
PRE-COMPUTED METRICS (use these exactly - do NOT recalculate)
============================================================
PRICE DATA:
- Current Price: ${snapshot.price:.2f} ({snapshot.price_source})
- Previous Close: ${snapshot.previous_close:.2f}
- Change: {snapshot.price_change_pct:+.2f}%

52-WEEK RANGE:
- 52W High: ${snapshot.week_52_high:.2f}
- 52W Low: ${snapshot.week_52_low:.2f}
- Distance from 52W High: {snapshot.pct_from_52w_high:+.1f}%
- Distance from 52W Low: {snapshot.pct_from_52w_low:+.1f}%

TARGET:
- Analyst Target: {analyst_target_str}
- Target Upside: {target_upside_str}

TRADE LEVELS (pre-computed):
- Entry Price: ${trade_levels.entry_price:.2f} ({trade_levels.entry_method})
- Stop Loss: ${trade_levels.stop_loss:.2f} ({trade_levels.stop_distance_pct:+.1f}%, {trade_levels.stop_method})
- Target Price: ${trade_levels.target_price:.2f} ({trade_levels.target_upside_pct:+.1f}%)
- Risk/Reward: {trade_levels.risk_reward_ratio:.2f}:1
- Position Size: {trade_levels.position_size_multiplier}x

DATA QUALITY:
- Components Available: {snapshot.components_available}/{snapshot.components_total}
- Data Completeness: {data_confidence}
- Raw Score: {snapshot.total_score}
- Uncertainty-Adjusted Score: {shrunk_score:.1f}
============================================================

SIGNALS:
- Platform Signal: {platform_direction} ({signal.today_score}%)
- Long-term: {signal.longterm_score}/100
- Risk Level: {risk_level} ({signal.risk_score})
- Reason: {signal.signal_reason}

COMPONENT SCORES:
- Technical: {technical_str} ({signal.technical_signal})
- Fundamental: {fundamental_str} ({signal.fundamental_signal})
- Sentiment: {sentiment_str} ({signal.sentiment_signal})
- Options: {options_str} ({signal.options_signal})
- Earnings: {earnings_str} ({signal.earnings_signal})

COMMITTEE: {committee_direction} ({signal.committee_confidence:.0%})
SIGNAL CONSENSUS: {conflict_desc}

============================================================
POLICY DECISION (BINDING - LLM cannot override)
============================================================
Action: {policy_decision.action}
Max Position Size: {policy_decision.max_size}x
Confidence: {policy_decision.confidence}
Reasons: {'; '.join(policy_decision.reasons) if policy_decision.reasons else 'None'}
============================================================
"""

    if signal.in_portfolio:
        context += f"""
PORTFOLIO POSITION:
- Weight: {signal.portfolio_weight:.2%}
- P&L: {signal.portfolio_pnl_pct:+.1f}%
- Days Held: {signal.days_held}
"""

    # Add Earnings Intelligence if available
    ei = additional_data.get('earnings_intelligence')
    if ei and ei.in_compute_window:
        context += f"""
EARNINGS INTELLIGENCE:
- Days to Earnings: {ei.days_to_earnings}
- IES (Implied Expectations): {ei.ies:.0f}/100
- Regime: {ei.regime.value if hasattr(ei.regime, 'value') else ei.regime}
- Position Scale: {ei.position_scale:.0%}
- In Action Window: {ei.in_action_window}
"""
        if ei.risk_flags:
            context += f"- Risk Flags: {', '.join(ei.risk_flags)}\n"

    # Add Recent News Headlines
    try:
        news_headlines = _get_recent_news_headlines(signal.ticker)
        if news_headlines:
            context += f"\nRECENT NEWS ({len(news_headlines)} articles):\n"
            for i, headline in enumerate(news_headlines[:5], 1):
                context += f"  {i}. {headline[:100]}\n"
    except Exception as e:
        logger.debug(f"News context error: {e}")

    # Add Options Flow Details
    try:
        options_context = _get_options_flow_context(signal.ticker, snapshot.price)
        if options_context:
            context += options_context
    except Exception as e:
        logger.debug(f"Options flow context error: {e}")

    # Add Insider Trading Activity
    try:
        insider_context = _get_insider_context(signal.ticker)
        if insider_context:
            context += insider_context
    except Exception as e:
        logger.debug(f"Insider context error: {e}")

    # Add Short Squeeze Analysis
    try:
        squeeze_context = _get_squeeze_context(signal.ticker)
        if squeeze_context:
            context += squeeze_context
    except Exception as e:
        logger.debug(f"Squeeze context error: {e}")

    # Add Fundamentals Data
    try:
        fundamentals_context = _get_fundamentals_context(signal.ticker)
        if fundamentals_context:
            context += fundamentals_context
    except Exception as e:
        logger.debug(f"Fundamentals context error: {e}")

    # Add Analyst Ratings
    try:
        analyst_context = _get_analyst_context(signal.ticker)
        if analyst_context:
            context += analyst_context
    except Exception as e:
        logger.debug(f"Analyst context error: {e}")

    # Add Alpha Model ML Predictions
    try:
        # Pass platform data directly from signal object to avoid DB query issues
        alpha_context = _get_alpha_model_context(
            ticker=signal.ticker,
            platform_score=signal.today_score,
            platform_signal=signal.today_signal.value,
            technical_score=signal.technical_score
        )
        if alpha_context:
            context += alpha_context
    except Exception as e:
        context += f"\n🔍 ALPHA ERROR: {e}\n"

    # Add AI Rules section
    context += """
===================================================================
🚫 AI RESPONSE RULES - MUST FOLLOW
===================================================================
1. DO NOT claim "institutional" buying/selling based on P/C ratios alone
2. DO NOT invent specific dates for OPEC meetings, Fed decisions, or other events unless explicitly provided in the data
3. DO NOT use options data that is more than 3 days old for current recommendations
4. DO NOT use max pain values that are >20% away from current stock price (data error)
5. DO report the ACTUAL conflict between Platform and Committee signals
6. DO note when data is stale and recommend verification with real-time sources
7. Dividend yield may vary ±0.1% depending on methodology - acknowledge this uncertainty
===================================================================
"""

    # Initialize session state for this ticker
    chat_key = f'ai_chat_{signal.ticker}'
    if chat_key not in st.session_state:
        st.session_state[chat_key] = {'response': '', 'last_question': ''}

    # Quick buttons - these directly trigger AI call
    st.caption("Quick questions:")
    col1, col2, col3, col4 = st.columns(4)

    quick_question = None

    with col1:
        if st.button("📊 Full Analysis", key=f"ai_full_{signal.ticker}", width='stretch'):
            quick_question = f"Give me a complete analysis of {signal.ticker} including current signals, key metrics, and your buy/hold/sell recommendation with reasoning."
    with col2:
        if st.button("🎯 Trade Idea", key=f"ai_trade_{signal.ticker}", width='stretch'):
            quick_question = f"Give me a specific trade idea for {signal.ticker} with: entry price, target price, stop loss, position size suggestion, and time horizon."
    with col3:
        if st.button("⚠️ Risks", key=f"ai_risk_{signal.ticker}", width='stretch'):
            quick_question = f"What are the main risks for {signal.ticker}? Include market risks, company-specific risks, sector risks, and any upcoming catalysts to watch."
    with col4:
        if st.button("📰 News", key=f"ai_news_{signal.ticker}", width='stretch'):
            quick_question = f"Summarize the recent news and sentiment for {signal.ticker}. What are the key headlines and how might they affect the stock price?"

    # If quick button was pressed, run AI immediately
    if quick_question:
        with st.spinner("🤔 Analyzing..."):
            response, model_used = _get_ai_response(quick_question, context, signal.ticker)
            st.session_state[chat_key]['response'] = response
            st.session_state[chat_key]['last_question'] = quick_question
            st.session_state[chat_key]['model_used'] = model_used

    st.markdown("---")

    # Custom question input (text_area for multi-line)
    st.caption("Or ask your own question:")
    question = st.text_area(
        "Your question:",
        height=80,
        key=f"ai_input_{signal.ticker}",
        placeholder=f"e.g., Should I buy {signal.ticker} at current levels?\nWhat's the best entry point?\nCompare to competitors...",
        label_visibility="collapsed"
    )

    col_submit, col_clear = st.columns([1, 1])
    with col_submit:
        if st.button("🚀 Ask AI", key=f"ai_submit_{signal.ticker}", type="primary", width='stretch'):
            if question.strip():
                with st.spinner("🤔 Thinking..."):
                    response, model_used = _get_ai_response(question, context, signal.ticker)
                    st.session_state[chat_key]['response'] = response
                    st.session_state[chat_key]['last_question'] = question
                    st.session_state[chat_key]['model_used'] = model_used
                    st.rerun()
            else:
                st.warning("Please enter a question")

    with col_clear:
        if st.button("🗑️ Clear", key=f"ai_clear_{signal.ticker}", width='stretch'):
            st.session_state[chat_key] = {'response': '', 'last_question': '', 'model_used': ''}
            st.rerun()

    # Show response if exists
    if st.session_state[chat_key]['response']:
        st.markdown("---")

        # Show which AI model answered
        model_used = st.session_state[chat_key].get('model_used', 'Unknown')
        st.markdown(f"#### 💬 AI Response")
        st.caption(f"🤖 **Model:** {model_used}")

        if st.session_state[chat_key]['last_question']:
            st.caption(f"Q: {st.session_state[chat_key]['last_question'][:100]}...")
        st.markdown(st.session_state[chat_key]['response'])

    # Show context in expander
    with st.expander("📋 Data Context (what AI sees)"):
        st.code(context, language=None)


def _get_alpha_model_context(ticker: str, platform_score: float = None,
                              platform_signal: str = None, technical_score: float = None) -> str:
    """
    Get Alpha Model ML predictions with all enhancements applied:
    - Forecast shrinkage (context-aware)
    - ML reliability gate (vol-scaled bias, EWMA accuracy)
    - Calibrated probabilities (smoothed by sample size)
    - Decision policy (binding rules for LLM)
    - Sector neutralization
    - Sentiment velocity

    Args:
        ticker: Stock ticker
        platform_score: Platform score (e.g., 71 for BUY 71%) - passed directly to avoid DB query
        platform_signal: Platform signal (e.g., "BUY") - passed directly to avoid DB query
        technical_score: Technical score - passed directly to avoid DB query
    """
    import os
    import pandas as pd

    prediction = None
    alpha_model_available = False

    try:
        # From src/tabs/signals_tab/ go 4 levels up to project root
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "models", "multi_factor_alpha.pkl"
        )

        if os.path.exists(model_path):
            from src.ml.multi_factor_alpha import MultiFactorAlphaModel

            model = MultiFactorAlphaModel()
            model.load(model_path)

            # Use predict_live with single ticker
            result_df = model.predict_live(tickers=[ticker])

            if not result_df.empty:
                # Get the row for this ticker and convert to dict
                row = result_df.iloc[0]
                prediction = {
                    'signal': row.get('signal', 'HOLD'),
                    'conviction': row.get('conviction', 0.5),
                    'expected_return_5d': row.get('expected_return_5d', 0),
                    'expected_return_10d': row.get('expected_return_10d', 0),
                    'expected_return_20d': row.get('expected_return_20d', 0),
                    'prob_positive_5d': row.get('prob_positive_5d', 0.5),
                    'prob_beat_market_5d': row.get('prob_beat_market_5d', 0.5),
                    'regime': row.get('regime', 'UNKNOWN'),
                    'top_bullish_factors': row.get('top_bullish_factors', []),
                    'top_bearish_factors': row.get('top_bearish_factors', []),
                }
                alpha_model_available = True
    except Exception as e:
        logger.debug(f"Alpha model prediction error for {ticker}: {e}")
        prediction = None
        alpha_model_available = False

    # If no Alpha Model prediction, create a default one so enhancements still work
    if prediction is None:
        prediction = {
            'signal': 'HOLD',
            'conviction': 0.0,
            'expected_return_5d': 0,
            'expected_return_10d': 0,
            'expected_return_20d': 0,
            'prob_positive_5d': 0.5,
            'prob_beat_market_5d': 0.5,
            'regime': 'UNKNOWN',
            'top_bullish_factors': [],
            'top_bearish_factors': [],
        }

    # Use passed-in platform data if available, otherwise query DB
    platform_data_available = platform_score is not None and platform_signal is not None

    if not platform_data_available:
        # Fallback to database query
        try:
            query = """
                SELECT total_score, signal_type, technical_score
                FROM latest_scores
                WHERE ticker = %s
            """
            df = pd.read_sql(query, get_engine(), params=(ticker,))
            if not df.empty:
                prow = df.iloc[0]
                if pd.notna(prow['total_score']) and prow['total_score'] != 0:
                    platform_score = float(prow['total_score'])
                if pd.notna(prow['signal_type']) and prow['signal_type']:
                    platform_signal = str(prow['signal_type'])
                if pd.notna(prow['technical_score']) and prow['technical_score'] != 0:
                    technical_score = float(prow['technical_score'])
                platform_data_available = platform_score is not None and platform_signal is not None
        except Exception as e:
            logger.debug(f"Could not get platform scores: {e}")

    # Apply defaults if still missing
    if platform_score is None:
        platform_score = 50
    if platform_signal is None:
        platform_signal = "HOLD"
    if technical_score is None:
        technical_score = 50

    # Try to use enhanced context
    try:
        from src.ml.alpha_enhancements import build_enhanced_alpha_context

        context = build_enhanced_alpha_context(
            ticker=ticker,
            alpha_prediction=prediction,
            platform_score=platform_score,
            platform_signal=platform_signal,
            technical_score=technical_score
        )

        # Add note if Alpha Model wasn't available
        if not alpha_model_available:
            context = f"⚠️ NOTE: Alpha Model prediction unavailable for {ticker} (new ticker or missing data)\n\n" + context

        if not platform_data_available:
            context = f"⚠️ WARNING: Platform scores unavailable for {ticker} - using conservative defaults\n\n" + context

        return context

    except ImportError:
        # Fallback if enhancements module not available
        logger.warning("alpha_enhancements module not available, using basic format")
    except Exception as e:
        logger.warning(f"Enhancement error, using fallback: {e}")

    # Fallback: basic format without enhancements
    if alpha_model_available:
        context = f"""
🧠 ALPHA MODEL (ML PREDICTION):
- Signal: {prediction.get('signal', 'N/A')} (Conviction: {prediction.get('conviction', 0):.0%})
- Expected 5d Return: {prediction.get('expected_return_5d', 0):+.2%}
- Expected 10d Return: {prediction.get('expected_return_10d', 0):+.2%}
- P(Positive 5d): {prediction.get('prob_positive_5d', 0):.1%}
- P(Beat Market): {prediction.get('prob_beat_market_5d', 0):.1%}
- Market Regime: {prediction.get('regime', 'UNKNOWN')}

⚠️ NOTE: Enhanced context unavailable - showing raw predictions
"""
    else:
        context = f"""
🧠 ALPHA MODEL: No prediction available for {ticker}
   (New ticker or not in training data)

📊 PLATFORM SIGNAL: {platform_signal} ({platform_score:.0f}%)
   Using platform signal only for analysis.
"""
    return context


def _get_options_flow_context(ticker: str, display_price: float = 0) -> str:
    """Get options flow context for AI - fetches LIVE data if DB is stale.

    Args:
        ticker: Stock symbol
        display_price: The price being displayed to user (pre-market/after-hours if available)
                      Used for consistent percentage calculations
    """
    try:
        import pandas as pd
        from datetime import datetime, date
        import yfinance as yf

        engine = get_engine()

        # First check DB for recent data (within 1 day)
        query = """
            SELECT total_call_volume, total_put_volume, put_call_volume_ratio,
                   put_call_oi_ratio, overall_sentiment, sentiment_score, 
                   max_pain_price, scan_date, stock_price
            FROM options_flow_daily
            WHERE ticker = %s
            ORDER BY scan_date DESC
            LIMIT 1
        """
        df = pd.read_sql(query, engine, params=(ticker,))

        # Check if data is fresh (within 1 day)
        use_live = True
        db_data = None

        if not df.empty:
            row = df.iloc[0]
            scan_date = row.get('scan_date')
            if scan_date:
                if isinstance(scan_date, str):
                    scan_dt = datetime.strptime(scan_date[:10], "%Y-%m-%d").date()
                elif hasattr(scan_date, 'date'):
                    scan_dt = scan_date.date()
                else:
                    scan_dt = scan_date

                data_age = (date.today() - scan_dt).days
                if data_age <= 1:
                    use_live = False
                    db_data = row

        # Fetch LIVE data if DB is stale or empty
        if use_live:
            try:
                from src.analytics.yf_subprocess import get_stock_info, get_options_chain
                info = get_stock_info(ticker) or {}
                stock_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

                # Get options via subprocess (safe from Streamlit freeze)
                chain_result = get_options_chain(ticker, max_expiries=4)
                if chain_result is None:
                    return _format_stale_options_context(df, "No options data available")

                calls_df, puts_df, chain_price = chain_result
                if calls_df is None or puts_df is None or calls_df.empty or puts_df.empty:
                    return _format_stale_options_context(df, "Could not fetch options data")

                if not stock_price and chain_price:
                    stock_price = chain_price

                expiries_used = sorted(calls_df['expiry'].unique().tolist()) if 'expiry' in calls_df.columns else ["aggregated"]

                # Calculate metrics
                call_vol = calls_df['volume'].sum() if 'volume' in calls_df else 0
                put_vol = puts_df['volume'].sum() if 'volume' in puts_df else 0
                call_oi = calls_df['openInterest'].sum() if 'openInterest' in calls_df else 0
                put_oi = puts_df['openInterest'].sum() if 'openInterest' in puts_df else 0

                pc_volume = put_vol / call_vol if call_vol > 0 else 1.0
                pc_oi = put_oi / call_oi if call_oi > 0 else 1.0

                # Calculate max pain for NEAREST EXPIRY ONLY (verifiable against public sources)
                nearest_expiry = expiries_used[0] if expiries_used else None
                max_pain = 0
                max_pain_expiry = "N/A"

                if nearest_expiry:
                    try:
                        if 'expiry' in calls_df.columns:
                            nearest_calls = calls_df[calls_df['expiry'] == nearest_expiry]
                            nearest_puts = puts_df[puts_df['expiry'] == nearest_expiry]
                            if not nearest_calls.empty and not nearest_puts.empty:
                                max_pain = _calculate_max_pain_live(nearest_calls, nearest_puts)
                                max_pain_expiry = nearest_expiry
                            else:
                                max_pain = _calculate_max_pain_live(calls_df, puts_df)
                                max_pain_expiry = f"aggregated ({len(expiries_used)} expiries)"
                        else:
                            max_pain = _calculate_max_pain_live(calls_df, puts_df)
                            max_pain_expiry = f"aggregated ({len(expiries_used)} expiries)"
                    except:
                        max_pain = _calculate_max_pain_live(calls_df, puts_df)
                        max_pain_expiry = f"aggregated ({len(expiries_used)} expiries)"

                # Validate max pain
                # NOTE: AI rules say >20% = data error, so we use 15% as warning, 20% as error
                max_pain_note = ""
                if stock_price > 0 and max_pain > 0:
                    pct_diff = abs(max_pain - stock_price) / stock_price * 100
                    if pct_diff > 20:
                        max_pain_note = f" (⚠️ DATA ERROR: {pct_diff:.0f}% from price - DO NOT USE)"
                    elif pct_diff > 15:
                        max_pain_note = f" (⚠️ {pct_diff:.0f}% from price - verify)"

                # Determine sentiment
                if pc_volume < 0.5:
                    sentiment = "Bullish bias (high call volume vs puts)"
                elif pc_volume < 0.7:
                    sentiment = "Moderately bullish"
                elif pc_volume < 1.0:
                    sentiment = "Neutral to slightly bullish"
                elif pc_volume < 1.3:
                    sentiment = "Neutral to slightly bearish"
                else:
                    sentiment = "Bearish bias (high put volume vs calls)"

                # Calculate max pain distance from DISPLAY price (pre-market if available)
                price_for_calc = display_price if display_price > 0 else stock_price
                max_pain_distance = ((max_pain - price_for_calc) / price_for_calc * 100) if price_for_calc > 0 and max_pain > 0 else 0

                # Get current timestamp (timezone-aware UTC)
                from datetime import datetime, timezone
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

                context = f"""
OPTIONS FLOW (LIVE data, {timestamp}):
- Stock Price: ${stock_price:.2f} (regular close)
- Call Volume: {call_vol:,.0f}
- Put Volume: {put_vol:,.0f}
- P/C Ratio (volume): {pc_volume:.2f} - {sentiment}
- P/C Ratio (OI): {pc_oi:.2f}
- Max Pain: ${max_pain:.2f} for {max_pain_expiry} ({max_pain_distance:+.1f}% from ${price_for_calc:.2f}){max_pain_note}

⚠️ NOTES:
- P/C ratio is VOLUME-based (OI-based may differ)
- Max pain shown for nearest expiry ({max_pain_expiry}) - verify with public sources
- Low P/C does NOT prove institutional buying
"""
                return context

            except Exception as e:
                logger.debug(f"Live options fetch error: {e}")
                # Fall back to DB data with warning
                return _format_stale_options_context(df, f"Live fetch failed: {e}")

        else:
            # Use fresh DB data
            row = db_data
            # BUG FIX: Use "is None" check - P/C ratio of 0.0 is valid (extremely bearish)
            # "or 1.0" would incorrectly convert 0.0 to 1.0 (neutral)
            pc_val = row.get('put_call_volume_ratio')
            put_call = pc_val if pc_val is not None else 1.0
            pc_oi_val = row.get('put_call_oi_ratio')
            put_call_oi = pc_oi_val if pc_oi_val is not None else 1.0
            call_vol = row.get('total_call_volume', 0) or 0
            put_vol = row.get('total_put_volume', 0) or 0
            max_pain = row.get('max_pain_price', 0) or 0
            stock_price = row.get('stock_price', 0) or 0
            scan_date = row.get('scan_date', '')

            # Format date
            date_str = scan_date.strftime("%Y-%m-%d") if hasattr(scan_date, 'strftime') else str(scan_date)[:10]

            # Sentiment interpretation
            if put_call < 0.5:
                sentiment = "Bullish bias (high call volume vs puts)"
            elif put_call < 0.7:
                sentiment = "Moderately bullish"
            elif put_call < 1.0:
                sentiment = "Neutral to slightly bullish"
            elif put_call < 1.3:
                sentiment = "Neutral to slightly bearish"
            else:
                sentiment = "Bearish bias (high put volume vs calls)"

            # Calculate max pain distance from DISPLAY price (pre-market if available)
            price_for_calc = display_price if display_price > 0 else stock_price
            max_pain_distance = ((max_pain - price_for_calc) / price_for_calc * 100) if price_for_calc > 0 and max_pain > 0 else 0

            # Validate max pain
            # NOTE: AI rules say >20% = data error, so we use 15% as warning, 20% as error
            max_pain_str = f"${max_pain:.2f} ({max_pain_distance:+.1f}% from current ${price_for_calc:.2f})"
            if price_for_calc > 0 and max_pain > 0:
                pct_diff = abs(max_pain - price_for_calc) / price_for_calc * 100
                if pct_diff > 20:
                    max_pain_str = f"${max_pain:.2f} ({max_pain_distance:+.1f}% - ⚠️ DATA ERROR >20% from ${price_for_calc:.2f})"
                elif pct_diff > 15:
                    max_pain_str = f"${max_pain:.2f} ({max_pain_distance:+.1f}% - ⚠️ verify, >15% from ${price_for_calc:.2f})"

            context = f"""
OPTIONS FLOW (platform data, {date_str}):
- Call Volume: {call_vol:,.0f}
- Put Volume: {put_vol:,.0f}
- P/C Ratio (volume): {put_call:.2f} - {sentiment}
- P/C Ratio (OI): {put_call_oi:.2f}
- Max Pain: {max_pain_str}

⚠️ NOTES:
- P/C ratio is VOLUME-based (OI-based may differ)
- Low P/C does NOT prove institutional buying
"""
            return context

    except Exception as e:
        logger.debug(f"Options flow context error: {e}")
        return ""


def _calculate_max_pain_live(calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> float:
    """
    Calculate max pain from live options data.

    Max pain = strike where total payout to option holders is MINIMUM
    (i.e., where most options expire worthless, causing max pain to holders)

    At strike X:
    - Calls with strike < X are ITM: payout = (X - strike) * OI
    - Calls with strike >= X are OTM: payout = 0
    - Puts with strike > X are ITM: payout = (strike - X) * OI
    - Puts with strike <= X are OTM: payout = 0

    We want the strike X that MINIMIZES total payout.
    """
    if calls_df.empty or puts_df.empty:
        return 0

    all_strikes = sorted(set(
        list(calls_df['strike'].unique()) +
        list(puts_df['strike'].unique())
    ))

    if not all_strikes:
        return 0

    min_payout = float('inf')
    max_pain_strike = all_strikes[0]

    for test_strike in all_strikes:
        total_payout = 0

        # ITM Calls: strike < test_strike, payout = (test_strike - strike) * OI
        for _, row in calls_df.iterrows():
            strike = row['strike']
            oi = row.get('openInterest', 0) or 0
            if strike < test_strike:
                total_payout += oi * (test_strike - strike)

        # ITM Puts: strike > test_strike, payout = (strike - test_strike) * OI
        for _, row in puts_df.iterrows():
            strike = row['strike']
            oi = row.get('openInterest', 0) or 0
            if strike > test_strike:
                total_payout += oi * (strike - test_strike)

        if total_payout < min_payout:
            min_payout = total_payout
            max_pain_strike = test_strike

    return max_pain_strike


def _format_stale_options_context(df: pd.DataFrame, error_msg: str) -> str:
    """Format stale DB data with warning."""
    if df.empty:
        return f"""
OPTIONS FLOW: ⚠️ NO DATA AVAILABLE
Error: {error_msg}
"""

    row = df.iloc[0]
    scan_date = row.get('scan_date', 'Unknown')
    date_str = scan_date.strftime("%Y-%m-%d") if hasattr(scan_date, 'strftime') else str(scan_date)[:10]

    return f"""
OPTIONS FLOW (⚠️ STALE DATA from {date_str}):
- P/C Ratio (volume): {row.get('put_call_volume_ratio', 'N/A')}
- Max Pain: ${row.get('max_pain_price', 0):.2f}
⛔ DATA IS STALE - verify with real-time sources
Error: {error_msg}
"""


def _get_insider_context(ticker: str) -> str:
    """Get insider trading context for AI."""
    try:
        import pandas as pd

        engine = get_engine()
        query = """
            SELECT insider_name, title, transaction_type, shares, 
                   price_per_share, total_value, transaction_date
            FROM insider_transactions
            WHERE ticker = %s
            AND transaction_date >= NOW() - INTERVAL '90 days'
            ORDER BY transaction_date DESC
            LIMIT 5
        """
        df = pd.read_sql(query, engine, params=(ticker,))

        if df.empty:
            return ""

        # Calculate summary
        buys = df[df['transaction_type'] == 'P']
        sells = df[df['transaction_type'] == 'S']

        buy_value = buys['total_value'].sum() if not buys.empty else 0
        sell_value = sells['total_value'].sum() if not sells.empty else 0

        context = f"""
INSIDER ACTIVITY (90 days):
- Total Buys: {len(buys)} transactions (${buy_value:,.0f})
- Total Sells: {len(sells)} transactions (${sell_value:,.0f})
- Net: {'BUYING' if buy_value > sell_value else 'SELLING' if sell_value > buy_value else 'NEUTRAL'}
Recent Transactions:
"""
        for _, row in df.head(3).iterrows():
            tx_type = "BUY" if row['transaction_type'] == 'P' else "SELL"
            context += f"  - {tx_type}: {row['insider_name'][:20]} - ${row.get('total_value', 0):,.0f}\n"

        return context
    except Exception as e:
        logger.debug(f"Insider context error: {e}")
        return ""


def _get_squeeze_context(ticker: str) -> str:
    """Get short squeeze context for AI."""
    try:
        from src.analytics.short_squeeze import get_squeeze_report
        report = get_squeeze_report(ticker)
        if report:
            return f"\n{report}\n"
        return ""
    except Exception as e:
        logger.debug(f"Squeeze context error: {e}")
        return ""


def _get_fundamentals_context(ticker: str) -> str:
    """Get fundamentals context for AI."""
    try:
        import pandas as pd

        engine = get_engine()
        query = """
            SELECT market_cap, pe_ratio, forward_pe, pb_ratio, ps_ratio,
                   profit_margin, roe, roa, revenue_growth, earnings_growth,
                   dividend_yield, debt_to_equity, current_ratio, free_cash_flow
            FROM fundamentals
            WHERE ticker = %s
            ORDER BY date DESC
            LIMIT 1
        """
        df = pd.read_sql(query, engine, params=(ticker,))

        if df.empty:
            return ""

        row = df.iloc[0]

        # Helper to normalize percentage values from DB
        def normalize_pct(val):
            if val is None:
                return 0
            val = float(val)
            # If > 0.50 (50%), likely stored as percentage not decimal
            if val > 0.50:
                val = val / 100
            # If still > 0.50, divide again (handles 348% -> 3.48%)
            if val > 0.50:
                val = val / 100
            return val

        # Format market cap
        market_cap = row.get('market_cap', 0) or 0
        if market_cap >= 1e12:
            mc_str = f"${market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            mc_str = f"${market_cap/1e9:.2f}B"
        else:
            mc_str = f"${market_cap/1e6:.2f}M"

        # Normalize percentages
        profit_margin = normalize_pct(row.get('profit_margin', 0))
        roe = normalize_pct(row.get('roe', 0))
        roa = normalize_pct(row.get('roa', 0))
        revenue_growth = normalize_pct(row.get('revenue_growth', 0))
        earnings_growth = normalize_pct(row.get('earnings_growth', 0))
        dividend_yield = normalize_pct(row.get('dividend_yield', 0))

        # FIX: Debt/Equity - Yahoo/providers return as percentage, convert properly
        debt_equity_raw = row.get('debt_to_equity', 0) or 0
        # If value > 1, it's stored as percentage (e.g., 15.67 = 15.67%)
        if debt_equity_raw > 1:
            debt_equity_ratio = debt_equity_raw / 100  # Convert to ratio (0.1567)
            debt_equity_pct = debt_equity_raw  # Keep percentage (15.67%)
        else:
            debt_equity_ratio = debt_equity_raw  # Already a ratio
            debt_equity_pct = debt_equity_raw * 100  # Convert to percentage

        # Interpret leverage level for context
        if debt_equity_ratio < 0.3:
            leverage_desc = "LOW leverage - strong balance sheet"
        elif debt_equity_ratio < 0.6:
            leverage_desc = "MODERATE leverage"
        elif debt_equity_ratio < 1.0:
            leverage_desc = "ELEVATED leverage"
        else:
            leverage_desc = "HIGH leverage - monitor debt burden"

        context = f"""
FUNDAMENTALS (from platform database - may differ slightly from real-time sources):
- Market Cap: {mc_str}
- P/E Ratio: {row.get('pe_ratio', 'N/A')}
- Forward P/E: {row.get('forward_pe', 'N/A')}
- P/B Ratio: {row.get('pb_ratio', 'N/A')}
- P/S Ratio: {row.get('ps_ratio', 'N/A')}
- Profit Margin: {profit_margin*100:.1f}%
- ROE: {roe*100:.1f}%
- ROA: {roa*100:.1f}%
- Revenue Growth: {revenue_growth*100:.1f}%
- Earnings Growth: {earnings_growth*100:.1f}%
- Dividend Yield: {dividend_yield*100:.2f}%
- Debt/Equity: {debt_equity_ratio:.2f}x ({debt_equity_pct:.1f}%) - {leverage_desc}
- Current Ratio: {row.get('current_ratio', 'N/A')}
"""
        return context
    except Exception as e:
        logger.debug(f"Fundamentals context error: {e}")
        return ""


def _get_analyst_context(ticker: str) -> str:
    """Get analyst ratings context for AI."""
    try:
        import pandas as pd

        engine = get_engine()

        # Get analyst ratings
        query = """
            SELECT buy_count, hold_count, sell_count, 
                   strong_buy_count, strong_sell_count
            FROM analyst_ratings
            WHERE ticker = %s
            ORDER BY date DESC
            LIMIT 1
        """
        df = pd.read_sql(query, engine, params=(ticker,))

        # Get price targets
        pt_query = """
            SELECT target_mean, target_high, target_low, current_price,
                   upside_pct, analyst_count
            FROM price_targets
            WHERE ticker = %s
            ORDER BY date DESC
            LIMIT 1
        """
        pt_df = pd.read_sql(pt_query, engine, params=(ticker,))

        context = ""

        if not df.empty:
            row = df.iloc[0]
            total = (row.get('strong_buy_count', 0) or 0) + (row.get('buy_count', 0) or 0) + \
                    (row.get('hold_count', 0) or 0) + (row.get('sell_count', 0) or 0) + \
                    (row.get('strong_sell_count', 0) or 0)

            context += f"""
ANALYST RATINGS:
- Strong Buy: {row.get('strong_buy_count', 0)}
- Buy: {row.get('buy_count', 0)}
- Hold: {row.get('hold_count', 0)}
- Sell: {row.get('sell_count', 0)}
- Strong Sell: {row.get('strong_sell_count', 0)}
- Total Analysts: {total}
"""

        if not pt_df.empty:
            pt = pt_df.iloc[0]
            context += f"""
PRICE TARGETS:
- Mean Target: ${pt.get('target_mean', 0):.2f}
- High Target: ${pt.get('target_high', 0):.2f}
- Low Target: ${pt.get('target_low', 0):.2f}
- Upside: {pt.get('upside_pct', 0):.1f}%
- # Analysts: {pt.get('analyst_count', 0)}
"""

        return context
    except Exception as e:
        logger.debug(f"Analyst context error: {e}")
        return ""


def _get_ai_response(question: str, context: str, ticker: str) -> tuple:
    """
    Get AI response using configured AI model with STRICT prompting to prevent hallucinations.

    Returns:
        tuple: (response_text, model_name)
    """

    # STRICT PROMPT - prevents LLM from inventing data or doing its own math
    prompt = f"""You are a financial analyst assistant. Your role is to EXPLAIN pre-computed data, NOT to calculate or invent.

=== STRICT RULES (MUST FOLLOW) ===
1. Use ONLY numbers that appear in the DATA CONTEXT below
2. If a metric shows "UNKNOWN" or "N/A", write "UNKNOWN" - do NOT guess or estimate
3. NEVER invent dates for events (OPEC, Fed, earnings) unless explicitly listed
4. NEVER claim "institutional buying/selling" from P/C ratios alone
5. NEVER recalculate percentages - use the pre-computed values exactly as shown
6. NEVER add events, catalysts, or news not mentioned in the context
7. If asked about something not in the data, say "This information is not available in the current data"
8. For entry/stop/target, use the Decision Policy values - do not invent your own levels

=== PROHIBITED PHRASES ===
- "institutional buying/selling" (unless attributed to specific trade data)
- "smart money"
- Made-up dates like "OPEC meeting Jan X" (unless in CATALYSTS)
- Made-up indicators like "RSI shows..." (unless RSI is in data)

=== DATA CONTEXT ===
{context}

=== QUESTION ===
{question}

=== RESPONSE FORMAT ===
Provide a clear, structured response using ONLY the data above. Include:
1. Direct answer to the question
2. Supporting evidence (cite specific numbers from the data)
3. Risks and caveats
4. If any information is missing, explicitly state "Data not available" for that item

ANSWER:"""

    model_name = "Unknown"
    selected_model = st.session_state.get('ai_model', 'qwen_local')
    attempted_model = selected_model  # Track what we tried

    logger.info(f"AI Request - Selected model: {selected_model}")

    try:
        # Try to use AI settings component first (if available)
        try:
            from src.components.ai_settings import get_current_model_response, get_current_model

            model = get_current_model()
            if model:
                model_name = f"{model.icon} {model.name}"
                logger.info(f"Using ai_settings component: {model_name}")
                response = ""
                result = get_current_model_response(prompt, stream=False)
                if isinstance(result, str):
                    response = result
                else:
                    for chunk in result:
                        response += chunk

                # Clean thinking tags
                if '</think>' in response:
                    response = response.split('</think>')[-1].strip()

                return response, model_name
        except ImportError:
            logger.debug("ai_settings not installed - using direct API calls")
        except Exception as e:
            logger.warning(f"ai_settings error: {e}, falling back to direct API")

        # Direct API calls based on selected model
        import os

        # Try to get model config from ai_models_config
        try:
            from src.components.ai_models_config import get_all_models, get_model_api_id
            all_models = get_all_models()

            if selected_model in all_models:
                model_config = all_models[selected_model]
                provider = model_config.get("provider", "qwen")
                api_id = model_config.get("api_id", selected_model)
                model_name = f"{model_config.get('icon', '🤖')} {model_config.get('name', selected_model)}"
                api_key_env = model_config.get("api_key_env")

                logger.info(f"Using model config: {selected_model} -> {api_id} ({provider})")

                # OpenAI provider
                if provider == "openai":
                    api_key = os.getenv("OPENAI_API_KEY", "")
                    if not api_key:
                        return f"❌ OpenAI API key not set. Add OPENAI_API_KEY to your .env file.", f"❌ {model_name} (No API Key)"

                    try:
                        from openai import OpenAI
                        logger.info(f"Calling OpenAI API: {api_id}")
                        client = OpenAI(api_key=api_key, timeout=120)

                        # GPT-5.x and o-series use max_completion_tokens, older models use max_tokens
                        is_new_model = api_id.startswith(('gpt-5', 'o1', 'o3', 'o4'))

                        if is_new_model:
                            response = client.chat.completions.create(
                                model=api_id,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.15,
                                max_completion_tokens=3000
                            )
                        else:
                            response = client.chat.completions.create(
                                model=api_id,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.15,
                                max_tokens=3000
                            )

                        result = response.choices[0].message.content

                        if '</think>' in result:
                            result = result.split('</think>')[-1].strip()

                        logger.info(f"OpenAI response received from {api_id}")
                        return result, model_name

                    except Exception as e:
                        error_msg = str(e)[:100]
                        logger.error(f"OpenAI API error: {error_msg}")
                        return f"❌ OpenAI API Error: {error_msg}", f"❌ {model_name} (Error)"

                # Anthropic provider
                elif provider == "anthropic":
                    api_key = os.getenv("ANTHROPIC_API_KEY", "")
                    if not api_key:
                        return f"❌ Anthropic API key not set. Add ANTHROPIC_API_KEY to your .env file.", f"❌ {model_name} (No API Key)"

                    try:
                        import anthropic
                        logger.info(f"Calling Anthropic API: {api_id}")
                        client = anthropic.Anthropic(api_key=api_key)

                        response = client.messages.create(
                            model=api_id,
                            max_tokens=3000,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        result = response.content[0].text

                        logger.info(f"Anthropic response received from {api_id}")
                        return result, model_name

                    except ImportError:
                        return "❌ Anthropic package not installed. Run: pip install anthropic", f"❌ {model_name} (Package Missing)"
                    except Exception as e:
                        error_msg = str(e)[:100]
                        logger.error(f"Anthropic API error: {error_msg}")
                        return f"❌ Anthropic API Error: {error_msg}", f"❌ {model_name} (Error)"

                # Ollama provider
                elif provider == "ollama":
                    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

                    try:
                        from openai import OpenAI
                        logger.info(f"Calling Ollama API: {api_id}")
                        client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama", timeout=120)

                        response = client.chat.completions.create(
                            model=api_id,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.15
                        )
                        result = response.choices[0].message.content

                        if '</think>' in result:
                            result = result.split('</think>')[-1].strip()

                        logger.info(f"Ollama response received from {api_id}")
                        return result, model_name

                    except Exception as e:
                        error_msg = str(e)[:100]
                        logger.error(f"Ollama API error: {error_msg}")
                        return f"❌ Ollama Error: {error_msg}\n\nMake sure Ollama is running at {base_url}", f"❌ {model_name} (Error)"

                # Local Qwen provider - handle here instead of falling through
                elif provider == "qwen":
                    logger.info(f"Using local Qwen via config (selected: {selected_model})")
                    from src.ai.chat import AlphaChat

                    if 'alpha_chat' not in st.session_state:
                        st.session_state.alpha_chat = AlphaChat()

                    chat = st.session_state.alpha_chat

                    # Get model name from chat config
                    if hasattr(chat, 'config') and hasattr(chat.config, 'model'):
                        model_name = f"🏠 {chat.config.model}"
                    else:
                        model_name = "🏠 Qwen (Local)"

                    if chat.available:
                        response = ""
                        for chunk in chat.chat_stream(prompt, ticker=ticker):
                            response += chunk

                        # Clean thinking tags
                        if '</think>' in response:
                            response = response.split('</think>')[-1].strip()

                        logger.info(f"Qwen response received")
                        return response, model_name
                    else:
                        logger.error("Qwen server not available")
                        return _fallback_response(context, ticker), "❌ Qwen Unavailable"

        except ImportError:
            logger.debug("ai_models_config not available, using fallback")

        # Default fallback: Local Qwen via AlphaChat (when config not available)
        logger.info(f"Using local Qwen fallback (selected: {selected_model})")
        from src.ai.chat import AlphaChat

        if 'alpha_chat' not in st.session_state:
            st.session_state.alpha_chat = AlphaChat()

        chat = st.session_state.alpha_chat

        # Get model name from chat config
        if hasattr(chat, 'config') and hasattr(chat.config, 'model'):
            model_name = f"🏠 {chat.config.model}"
        else:
            model_name = "🏠 Qwen (Local)"

        if chat.available:
            response = ""
            for chunk in chat.chat_stream(prompt, ticker=ticker):
                response += chunk

            # Clean thinking tags
            if '</think>' in response:
                response = response.split('</think>')[-1].strip()

            logger.info(f"Qwen response received")
            return response, model_name
        else:
            logger.error("Qwen server not available")
            return _fallback_response(context, ticker), "❌ Qwen Unavailable"

    except Exception as e:
        logger.error(f"AI response error (attempted: {attempted_model}): {e}")
        return f"❌ Error with {attempted_model}: {str(e)[:100]}", f"❌ {attempted_model} (Failed)"


def _fallback_response(context: str, ticker: str) -> str:
    """Fallback when AI not available."""
    return f"""**{ticker} Summary** (AI unavailable)

Based on the data:
{context[:500]}...

*Start Qwen server for full AI analysis:*
```
llama-server.exe -m qwen3-32b-q6_k.gguf -c 32768 -ngl 99 --port 8080
```
"""


