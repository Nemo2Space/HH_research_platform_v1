"""
Signals Tab - Deep Dive Panel

Stock deep dive: price banner, chart, key stats, component scores, news.
"""
from .ai_chat import _render_ai_chat
from .analysis import _run_single_analysis
from .earnings_views import _render_news, _render_earnings_intelligence, _render_full_reaction_analysis_standalone
from .shared import (
    st, pd, np, logger, datetime, date, timedelta, time, json,
    Dict, List, Optional, _to_native, text,
    get_engine, get_connection,
    SIGNAL_HUB_AVAILABLE, DB_AVAILABLE,
    ENHANCED_SCORING_AVAILABLE, INSTITUTIONAL_SIGNALS_AVAILABLE,
    SEC_INSIGHTS_AVAILABLE, get_filing_insights,
    DATEUTIL_AVAILABLE, date_parser,
    UnifiedSignal, compute_trade_levels,
    render_institutional_signals, REACTION_ANALYZER_AVAILABLE, DUAL_ANALYST_AVAILABLE,
    DualAnalystService, _get_extended_hours_price,
)

if SIGNAL_HUB_AVAILABLE:
    from .shared import SignalStrength, RiskLevel

if ENHANCED_SCORING_AVAILABLE:
    from .shared import render_enhancement_breakdown, get_single_ticker_enhancement

def _render_deep_dive(signal: UnifiedSignal):
    """Render full deep dive with all features."""

    # Load additional data from DB
    additional_data = _load_additional_data(signal.ticker)

    # Get extended hours / current price
    extended_price_data = _get_extended_hours_price(signal.ticker)

    # Header
    c = "green" if signal.today_signal.value == 'bullish' else "red" if signal.today_signal.value == 'bearish' else "orange"
    st.markdown(f"## {signal.ticker} - {signal.company_name}")

    # PRICE BANNER - Show extended hours if available
    if extended_price_data.get('has_extended'):
        _render_price_banner(signal, extended_price_data)

    # Check for recent earnings - show alert banner
    ei = additional_data.get('earnings_intelligence')
    if ei and ei.is_post_earnings:
        _render_earnings_alert_banner(ei)

    # Signal summary row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### :{c}[{signal.today_signal.value.upper()}]")
        st.caption(f"Today: {signal.today_score}%")
    with col2:
        st.metric("Long-term", f"{signal.longterm_score}/100")
    with col3:
        risk_c = "🟢" if signal.risk_level.value == 'LOW' else "🔴" if signal.risk_level.value in ('HIGH',
                                                                                                 'EXTREME') else "🟡"
        st.metric("Risk", f"{risk_c} {signal.risk_level.value}")

    st.info(f"💡 {signal.signal_reason}")

    # =========================================================================
    # TRADE RECOMMENDATION (for BUY signals)
    # =========================================================================
    if 'BUY' in signal.today_signal.value.upper() or 'bullish' in signal.today_signal.value.lower() or signal.today_score >= 60:
        st.markdown("### 💰 Trade Recommendation")
        try:
            _render_trade_recommendation(signal)
        except Exception as e:
            st.caption(f"Trade recommendation unavailable: {e}")

    # =========================================================================
    # AI SUMMARY - Quick overview of the stock
    # =========================================================================
    st.markdown("### 📝 Quick Summary")
    try:
        _render_ai_summary(signal, additional_data)
    except Exception as e:
        st.error(f"Summary error: {e}")
        # Fallback simple summary
        st.markdown(f"""
**Overall Score: {signal.today_score}/100**

- 📰 Sentiment: {signal.sentiment_score or 'N/A'}
- 📊 Options Flow: {signal.options_score or 'N/A'}  
- 📈 Fundamental: {signal.fundamental_score or 'N/A'}
- 📉 Technical: {signal.technical_score or 'N/A'}
""")

    # Two columns layout
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Price Targets & Ratings
        st.markdown("#### 🎯 Price Targets & Ratings")
        _render_price_targets(signal, additional_data)

        # Price Chart
        st.markdown("#### 📈 Price Chart (6 Months)")
        _render_chart(signal.ticker, additional_data)

        # Component Scores
        st.markdown("#### 📊 Component Scores")
        _render_component_scores(signal)

        # Enhanced Scoring Breakdown (in expander to avoid slow page load)
        if ENHANCED_SCORING_AVAILABLE:
            with st.expander("📊 Enhanced Score Breakdown (click to load)", expanded=False):
                try:
                    render_enhancement_breakdown(signal.ticker)
                except Exception as e:
                    st.warning(f"Could not load enhancement breakdown: {e}")

        # News - with earnings filter if post-earnings
        if ei and ei.is_post_earnings:
            st.markdown("#### 📰 Earnings News & Recent Headlines")
            _render_news(signal.ticker, earnings_focus=True)
        else:
            st.markdown("#### 📰 Recent News")
            _render_news(signal.ticker, earnings_focus=False)

    with col_right:
        # Committee Decision
        st.markdown("#### 🎯 Committee Decision")
        verdict_color = "green" if 'BUY' in signal.committee_verdict else "red" if 'SELL' in signal.committee_verdict else "orange"
        st.markdown(f"**:{verdict_color}[{signal.committee_verdict}]** ({signal.committee_confidence:.0%} confidence)")
        if signal.committee_votes:
            votes_str = ", ".join([f"{k}: {v}" for k, v in signal.committee_votes.items()])
            st.caption(votes_str)

        # AI Analysis
        try:
            from src.ml.signals_tab_ai import _render_ai_analysis
            _render_ai_analysis(signal)
        except Exception as e:
            pass

        # Earnings Intelligence (IES/ECS) - shows for both pre and post earnings
        ei = additional_data.get('earnings_intelligence')
        if ei and ei.in_compute_window:
            if ei.is_post_earnings:
                st.markdown("#### 📊 Post-Earnings Analysis")
            else:
                st.markdown("#### 📊 Earnings Intelligence")
            _render_earnings_intelligence(ei)
        elif ei and ei.earnings_date:
            st.markdown("#### 📅 Earnings")
            st.caption(f"Next: {ei.earnings_date} ({ei.days_to_earnings} days)")
        elif signal.earnings_date or signal.days_to_earnings:
            st.markdown("#### 📅 Earnings")
            if signal.earnings_date:
                st.caption(f"Next: {signal.earnings_date}")
            if signal.days_to_earnings and signal.days_to_earnings < 999:
                st.caption(f"Days: {signal.days_to_earnings}")

        # Risk Assessment
        st.markdown("#### ⚠️ Risk Assessment")
        risk_color = "green" if signal.risk_level.value == 'LOW' else "red" if signal.risk_level.value in ('HIGH',
                                                                                                           'EXTREME') else "orange"
        st.markdown(f"**:{risk_color}[{signal.risk_level.value}]** (Score: {signal.risk_score})")
        if signal.risk_factors:
            for f in signal.risk_factors[:4]:
                st.caption(f"• {f}")

        # 52-Week Position
        st.markdown("#### 📍 52-Week Position")
        st.caption(f"From High: {signal.pct_from_high:+.1f}%")
        st.caption(f"From Low: {signal.pct_from_low:+.1f}%")

        # Insider Activity
        st.markdown("#### 🏦 Insider Activity")
        _render_insider_activity(additional_data)

        # Portfolio Position
        if signal.in_portfolio:
            st.markdown("#### 💼 Your Position")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Weight", f"{signal.portfolio_weight:.2%}")
                st.metric("Target", f"{signal.target_weight:.2%}")
            with col2:
                pnl_color = "green" if signal.portfolio_pnl_pct >= 0 else "red"
                st.metric("P&L", f":{pnl_color}[{signal.portfolio_pnl_pct:+.1f}%]")
                st.metric("Days Held", signal.days_held)

        # Flags
        if signal.flags:
            st.markdown("#### 🏷️ Flags")
            st.markdown(" ".join(signal.flags))

        # =====================================================================
        # PHASE 2/3: INSTITUTIONAL SIGNALS
        # =====================================================================
        if INSTITUTIONAL_SIGNALS_AVAILABLE:
            st.markdown("---")
            try:
                render_institutional_signals(
                    ticker=signal.ticker,
                    current_price=signal.current_price,
                    sector=getattr(signal, 'sector', None),
                    days_to_earnings=getattr(signal, 'days_to_earnings', 999) or 999,
                )
            except Exception as e:
                st.caption(f"Institutional signals error: {e}")

        # Run Analysis Button
        st.markdown("---")
        if st.button(f"🔄 Refresh {signal.ticker} Data", width='stretch'):
            _run_single_analysis(signal.ticker)

    # =========================================================================
    # FULL-WIDTH POST-EARNINGS REACTION ANALYSIS (Always available)
    # =========================================================================
    st.markdown("---")

    # Header with refresh button
    col_header, col_refresh = st.columns([4, 1])
    with col_header:
        st.markdown("## 📊 Post-Earnings Reaction Analysis")
    with col_refresh:
        if st.button("🔄 Refresh", key=f"refresh_reaction_{signal.ticker}", help="Force fresh analysis"):
            # Clear cache for this ticker
            cache_key = f"reaction_cache_{signal.ticker}"
            if cache_key in st.session_state:
                del st.session_state[cache_key]
            st.rerun()

    # Check if analysis is available
    if REACTION_ANALYZER_AVAILABLE:
        # Check for recent earnings using yfinance directly
        has_recent_earnings = False
        days_since_earnings = None

        try:
            import yfinance as yf
            from datetime import date
            stock = yf.Ticker(signal.ticker)
            ed = stock.earnings_dates

            if ed is not None and not ed.empty:
                today = date.today()
                for idx in ed.index:
                    try:
                        earnings_date = idx.date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
                        days_diff = (today - earnings_date).days
                        if 0 <= days_diff <= 10:  # Within 10 days
                            has_recent_earnings = True
                            days_since_earnings = days_diff
                            break
                    except:
                        continue
        except Exception as e:
            logger.debug(f"Earnings date check error: {e}")

        if has_recent_earnings:
            st.caption(f"📅 Earnings reported {days_since_earnings} day(s) ago")
            _render_full_reaction_analysis_standalone(signal.ticker)
        else:
            st.info("No recent earnings detected (within 10 days)")
            if st.button("🔍 Run Analysis Anyway", key=f"force_reaction_{signal.ticker}"):
                _render_full_reaction_analysis_standalone(signal.ticker)
    else:
        st.warning(
            "Reaction analyzer not available. Install reaction_analyzer.py in src/analytics/earnings_intelligence/")

    # =========================================================================
    # SEC FILING INSIGHTS SECTION
    # =========================================================================
    st.markdown("---")
    if SEC_INSIGHTS_AVAILABLE:
        with st.expander("📄 SEC Filing Insights", expanded=True):
            try:
                insights = get_filing_insights(signal.ticker)
                if insights.get('available'):
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        st.metric("Filing Score", f"{insights.get('score', 0):.0f}/100")
                    with col2:
                        st.metric("Rating", insights.get('score_label', 'Unknown'))
                    with col3:
                        quality = insights.get('data_quality', {})
                        st.caption(f"📅 {quality.get('freshness_days', 0)} days ago | {quality.get('filings_analyzed', 0)} filings")

                    # Factors
                    factors = insights.get('factors', {})
                    cols = st.columns(5)
                    for col, (key, label) in zip(cols, [('guidance', 'Guidance'), ('risk', 'Risk'), ('litigation', 'Litigation'), ('china', 'China'), ('ai_demand', 'AI')]):
                        with col:
                            st.metric(label, f"{factors.get(key, {}).get('score', 50):.0f}")

                    # Signals
                    col_b, col_r = st.columns(2)
                    with col_b:
                        for s in insights.get('bullish_signals', [])[:3]:
                            st.markdown(f"✅ {s}")
                    with col_r:
                        for s in insights.get('bearish_signals', [])[:3]:
                            st.markdown(f"⚠️ {s}")
                else:
                    st.info(f"No SEC filing data for {signal.ticker}. Run `python -m src.rag.sec_ingestion`")
            except Exception as e:
                st.error(f"SEC insights error: {e}")
    else:
        with st.expander("📄 SEC Filing Insights", expanded=False):
            st.info("SEC Filing Insights module not available")

    # =========================================================================
    # DUAL ANALYST SECTION
    # =========================================================================
    st.markdown("---")
    if DUAL_ANALYST_AVAILABLE:
        with st.expander("🔬 AI Dual Analysis (SQL + RAG)", expanded=False):
            cache_key = f"dual_signals_{signal.ticker}"

            if st.button("🚀 Run Dual Analysis", key=f"dual_btn_{signal.ticker}"):
                with st.spinner("Running dual analysis (15-30s)..."):
                    try:
                        service = DualAnalystService()
                        result = service.analyze_for_display(signal.ticker)
                        st.session_state[cache_key] = result
                    except Exception as e:
                        st.error(f"Failed: {e}")

            if cache_key in st.session_state:
                r = st.session_state[cache_key]

                # Agreement
                agreement = r.get('evaluation', {}).get('agreement_score', 0)
                st.progress(agreement, text=f"Agreement: {agreement:.0%}")

                # Two analysts
                col1, col2 = st.columns(2)
                ICONS = {"very_bullish": "🚀", "bullish": "📈", "neutral": "➡️", "bearish": "📉", "very_bearish": "🔻", "unknown": "❓"}

                with col1:
                    sql = r.get('sql_analyst', {})
                    icon = ICONS.get(sql.get('sentiment', 'unknown'), "❓")
                    st.markdown(f"**📊 Quant:** {icon} {sql.get('sentiment', 'unknown').replace('_', ' ').title()}")
                    st.caption(sql.get('summary', '')[:120])

                with col2:
                    rag = r.get('rag_analyst', {})
                    icon = ICONS.get(rag.get('sentiment', 'unknown'), "❓")
                    st.markdown(f"**📄 Qual:** {icon} {rag.get('sentiment', 'unknown').replace('_', ' ').title()}")
                    st.caption(rag.get('summary', '')[:120])

                # Synthesis
                syn = r.get('synthesis', {})
                icon = ICONS.get(syn.get('sentiment', 'unknown'), "❓")
                st.markdown(f"**🎯 Verdict:** {icon} {syn.get('sentiment', 'unknown').replace('_', ' ').title()} ({syn.get('confidence', 0):.0%})")
    else:
        with st.expander("🔬 AI Dual Analysis (SQL + RAG)", expanded=False):
            st.info("Dual Analyst Service not available")

    # AI Chat Section (full width)
    st.markdown("---")
    _render_ai_chat(signal, additional_data)


def _render_price_banner(signal: UnifiedSignal, extended_data: dict):
    """Render prominent price banner with extended hours info."""

    session = extended_data.get('session', '')
    ext_price = extended_data.get('extended_price', 0)
    ext_change = extended_data.get('extended_change', 0)
    ext_change_pct = extended_data.get('extended_change_pct', 0)
    prev_close = extended_data.get('regular_close', 0)

    # Determine color based on change
    if ext_change_pct <= -5:
        color = "red"
        emoji = "🔴📉"
        severity = "MAJOR LOSS"
    elif ext_change_pct <= -2:
        color = "red"
        emoji = "🔴"
        severity = "Down"
    elif ext_change_pct >= 5:
        color = "green"
        emoji = "🟢📈"
        severity = "MAJOR GAIN"
    elif ext_change_pct >= 2:
        color = "green"
        emoji = "🟢"
        severity = "Up"
    else:
        color = "orange"
        emoji = "🟡"
        severity = "Flat"

    # Create banner
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        st.markdown(f"**{session}:** ${ext_price:.2f}")

    with col2:
        st.markdown(f"**{emoji} {ext_change_pct:+.2f}%** ({severity})")

    with col3:
        st.markdown(f"Prev Close: ${prev_close:.2f}")

    # Big alert for major moves
    if abs(ext_change_pct) >= 5:
        if ext_change_pct < 0:
            st.error(f"🚨 **{session.upper()}: {signal.ticker} DOWN {abs(ext_change_pct):.1f}%** (${ext_change:+.2f})")
        else:
            st.success(f"🚀 **{session.upper()}: {signal.ticker} UP {ext_change_pct:.1f}%** (${ext_change:+.2f})")


def _render_earnings_alert_banner(ei):
    """Render prominent earnings alert banner."""

    days_since = abs(ei.days_to_earnings)

    # ECS color
    ecs_val = ei.ecs_category.value if hasattr(ei.ecs_category, 'value') else str(ei.ecs_category)

    if ecs_val in ('STRONG_MISS', 'MISS'):
        st.error(f"""
        🚨 **EARNINGS ALERT** - Reported {days_since} day(s) ago

        **Result:** {ecs_val.replace('_', ' ')} | **Reaction:** {ei.total_reaction_pct:+.1f}%
        """)
    elif ecs_val in ('STRONG_BEAT', 'BEAT'):
        st.success(f"""
        📊 **EARNINGS** - Reported {days_since} day(s) ago

        **Result:** {ecs_val.replace('_', ' ')} | **Reaction:** {ei.total_reaction_pct:+.1f}%
        """)
    else:
        st.info(f"""
        📊 **EARNINGS** - Reported {days_since} day(s) ago

        **Result:** {ecs_val.replace('_', ' ')} | **Reaction:** {ei.total_reaction_pct:+.1f}%
        """)


def _load_additional_data(ticker: str) -> dict:
    """Load additional data from database including earnings intelligence and current price."""
    data = {}
    try:
        engine = get_engine()

        # Get current/extended price FIRST - this is critical
        try:
            ext_price = _get_extended_hours_price(ticker)
            data['extended_price_data'] = ext_price
            data['current_price'] = ext_price.get('extended_price') or ext_price.get('regular_close', 0)
        except:
            pass

        # Price Targets
        try:
            df = pd.read_sql(f"""
                SELECT target_mean, target_high, target_low, target_upside_pct, analyst_count
                FROM price_targets WHERE ticker = '{ticker}' ORDER BY date DESC LIMIT 1
            """, engine)
            if not df.empty:
                row = df.iloc[0]
                data['target_mean'] = row.get('target_mean')
                data['target_high'] = row.get('target_high')
                data['target_low'] = row.get('target_low')
                data['target_upside'] = row.get('target_upside_pct')
                data['analyst_count'] = row.get('analyst_count')
        except:
            pass

        # Analyst Ratings
        try:
            df = pd.read_sql(f"""
                SELECT strong_buy, buy, hold, sell, strong_sell, consensus
                FROM analyst_ratings WHERE ticker = '{ticker}' ORDER BY date DESC LIMIT 1
            """, engine)
            if not df.empty:
                row = df.iloc[0]
                data['strong_buy'] = row.get('strong_buy', 0)
                data['buy'] = row.get('buy', 0)
                data['hold'] = row.get('hold', 0)
                data['sell'] = row.get('sell', 0)
                data['strong_sell'] = row.get('strong_sell', 0)
                data['consensus'] = row.get('consensus', '')
        except:
            pass

        # Insider Transactions
        try:
            df = pd.read_sql(f"""
                SELECT insider_name, transaction_type, shares_transacted, total_value, transaction_date
                FROM insider_transactions WHERE ticker = '{ticker}' 
                ORDER BY transaction_date DESC LIMIT 10
            """, engine)
            data['insider_transactions'] = df.to_dict('records') if not df.empty else []
        except:
            data['insider_transactions'] = []

        # Earnings Intelligence (IES/ECS) - with proper post-earnings handling
        try:
            from src.analytics.earnings_intelligence import enrich_screener_with_earnings
            ei = enrich_screener_with_earnings(ticker)
            data['earnings_intelligence'] = ei

            # If post-earnings, get actual results
            if ei and ei.is_post_earnings:
                import yfinance as yf
                stock = yf.Ticker(ticker)

                try:
                    hist = stock.earnings_history
                    if hist is not None and not hist.empty:
                        latest = hist.iloc[0]
                        data['earnings_actual'] = {
                            'eps_actual': latest.get('epsActual'),
                            'eps_estimate': latest.get('epsEstimate'),
                            'surprise_pct': None,
                        }
                        if data['earnings_actual']['eps_actual'] and data['earnings_actual']['eps_estimate']:
                            est = data['earnings_actual']['eps_estimate']
                            if est != 0:
                                data['earnings_actual']['surprise_pct'] = (
                                                                                  (data['earnings_actual'][
                                                                                       'eps_actual'] - est) / abs(est)
                                                                          ) * 100
                except:
                    pass

        except ImportError:
            # Try alternate path
            try:
                import sys
                sys.path.insert(0, '/mnt/user-data/outputs/signal_hub')
                from src.analytics.earnings_intelligence import enrich_screener_with_earnings
                ei = enrich_screener_with_earnings(ticker)
                data['earnings_intelligence'] = ei
            except:
                data['earnings_intelligence'] = None
        except Exception as e:
            logger.debug(f"EI error for {ticker}: {e}")
            data['earnings_intelligence'] = None

    except Exception as e:
        logger.error(f"Error loading additional data: {e}")

    return data


def _render_trade_recommendation(signal: UnifiedSignal):
    """Render trade recommendation with entry, stop loss, and target prices."""

    current_price = getattr(signal, 'current_price', 0) or 0
    if current_price <= 0:
        st.caption("Price not available for trade calculation")
        return

    # Get risk level safely
    risk_level = 'MEDIUM'
    if hasattr(signal, 'risk_level'):
        risk_level = signal.risk_level.value if hasattr(signal.risk_level, 'value') else str(signal.risk_level)

    # Get today score safely - FIX: use is None check (0 is a valid score)
    _today_score = getattr(signal, 'today_score', None)
    today_score = _today_score if _today_score is not None else 50

    # Calculate trade levels based on signal strength
    # Entry price - current price or slight pullback for strong signals
    if today_score >= 80:
        entry_price = current_price
        entry_note = "Enter at market"
    elif today_score >= 65:
        entry_price = current_price * 0.98  # 2% pullback
        entry_note = "Enter on 2% pullback"
    else:
        entry_price = current_price * 0.95  # 5% pullback
        entry_note = "Enter on 5% pullback"

    # Stop loss - based on risk level
    risk_multiplier = {
        'LOW': 5.0,
        'MEDIUM': 7.0,
        'HIGH': 10.0,
        'EXTREME': 12.0
    }.get(risk_level, 7.0)

    stop_loss = entry_price * (1 - risk_multiplier / 100)
    stop_pct = ((stop_loss - entry_price) / entry_price) * 100

    # Target price - use analyst target if available, otherwise calculate
    analyst_target = getattr(signal, 'target_price', None)
    if analyst_target and analyst_target > current_price:
        target_price = analyst_target
        target_note = "Analyst consensus"
    else:
        # Calculate based on 2.5:1 R/R
        risk_amount = entry_price - stop_loss
        target_price = entry_price + (risk_amount * 2.5)
        target_note = "2.5:1 R/R target"

    target_pct = ((target_price - entry_price) / entry_price) * 100

    # Display trade levels
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📍 Entry", f"${entry_price:.2f}")
        st.caption(entry_note)

    with col2:
        st.metric("🛑 Stop Loss", f"${stop_loss:.2f}", f"{stop_pct:.1f}%")

    with col3:
        st.metric("🎯 Target", f"${target_price:.2f}", f"+{target_pct:.1f}%")
        st.caption(target_note)

    with col4:
        rr_ratio = abs(target_pct / stop_pct) if stop_pct != 0 else 0
        rr_color = "🟢" if rr_ratio >= 2 else "🟡" if rr_ratio >= 1.5 else "🔴"
        st.metric("R/R Ratio", f"{rr_color} {rr_ratio:.1f}:1")


def _render_ai_summary(signal: UnifiedSignal, additional_data: dict):
    """Render a quick AI-generated summary of the stock."""

    # Build summary based on component scores
    summaries = []

    # Get scores safely - FIX: use is None check (0 is a valid score)
    _today = getattr(signal, 'today_score', None)
    _sent = getattr(signal, 'sentiment_score', None)
    _opts = getattr(signal, 'options_score', None)
    _fund = getattr(signal, 'fundamental_score', None)
    _tech = getattr(signal, 'technical_score', None)
    _squeeze = getattr(signal, 'squeeze_score', None)

    today_score = _today if _today is not None else 50
    sent = _sent if _sent is not None else 50
    opts = _opts if _opts is not None else 50
    fund = _fund if _fund is not None else 50
    tech = _tech if _tech is not None else 50
    squeeze = _squeeze if _squeeze is not None else 50

    # Overall signal
    if today_score >= 70:
        overall = "🟢 **Strong opportunity** - Multiple factors align positively."
    elif today_score >= 55:
        overall = "🟡 **Moderate opportunity** - Some positive factors, but mixed signals."
    elif today_score >= 45:
        overall = "⚪ **Neutral** - No clear direction, wait for better setup."
    elif today_score >= 30:
        overall = "🟠 **Caution advised** - More negative factors than positive."
    else:
        overall = "🔴 **Avoid** - Multiple warning signs present."

    summaries.append(overall)

    # Sentiment analysis
    if sent >= 70:
        summaries.append(f"📰 **Sentiment: Very Bullish ({sent})** - News and social media are overwhelmingly positive.")
    elif sent >= 55:
        summaries.append(f"📰 **Sentiment: Bullish ({sent})** - More positive than negative news coverage.")
    elif sent >= 45:
        summaries.append(f"📰 **Sentiment: Neutral ({sent})** - Mixed news coverage with no clear bias.")
    elif sent >= 30:
        summaries.append(f"📰 **Sentiment: Bearish ({sent})** - Negative news outweighs positive.")
    else:
        summaries.append(f"📰 **Sentiment: Very Bearish ({sent})** - Overwhelmingly negative press.")

    # Options flow
    if opts >= 70:
        summaries.append(f"📊 **Options Flow: Bullish ({opts})** - Smart money positioning for upside. Unusual call activity.")
    elif opts >= 55:
        summaries.append(f"📊 **Options Flow: Mildly Bullish ({opts})** - Slight bullish bias in options market.")
    elif opts >= 45:
        summaries.append(f"📊 **Options Flow: Neutral ({opts})** - No significant directional bias.")
    elif opts >= 30:
        summaries.append(f"📊 **Options Flow: Bearish ({opts})** - Put buying exceeds calls.")
    else:
        summaries.append(f"📊 **Options Flow: Very Bearish ({opts})** - Heavy put buying detected.")

    # Squeeze potential
    if squeeze >= 70:
        summaries.append(f"🚀 **Short Squeeze: High Potential ({squeeze})** - High short interest + bullish catalysts!")
    elif squeeze >= 50:
        summaries.append(f"🚀 **Short Squeeze: Moderate ({squeeze})** - Some short interest present.")
    else:
        summaries.append(f"🚀 **Short Squeeze: Low ({squeeze})** - Minimal short interest.")

    # Fundamentals
    if fund >= 70:
        summaries.append(f"📈 **Fundamentals: Strong ({fund})** - Solid financials, good growth, attractive valuation.")
    elif fund >= 55:
        summaries.append(f"📈 **Fundamentals: Good ({fund})** - Decent financial health.")
    elif fund >= 45:
        summaries.append(f"📈 **Fundamentals: Fair ({fund})** - Mixed financial picture.")
    elif fund >= 30:
        summaries.append(f"📈 **Fundamentals: Weak ({fund})** - Some concerning metrics.")
    else:
        summaries.append(f"📈 **Fundamentals: Poor ({fund})** - Significant concerns.")

    # Technical
    if tech >= 70:
        summaries.append(f"📉 **Technical: Bullish ({tech})** - Strong chart, above key MAs, momentum positive.")
    elif tech >= 55:
        summaries.append(f"📉 **Technical: Mildly Bullish ({tech})** - Some positive chart setups.")
    elif tech >= 45:
        summaries.append(f"📉 **Technical: Neutral ({tech})** - Consolidating, no clear direction.")
    elif tech >= 30:
        summaries.append(f"📉 **Technical: Bearish ({tech})** - Below key support levels.")
    else:
        summaries.append(f"📉 **Technical: Very Bearish ({tech})** - Strong downtrend.")

    # Earnings context
    ei = additional_data.get('earnings_intelligence') if additional_data else None
    if ei:
        try:
            if getattr(ei, 'is_post_earnings', False):
                reaction = getattr(ei, 'reaction_category', 'Unknown')
                summaries.append(f"📅 **Recent Earnings: {reaction}** - Stock reacted to latest report.")
            elif getattr(ei, 'days_to_earnings', 999) <= 14:
                days = ei.days_to_earnings
                summaries.append(f"📅 **Earnings in {days} days** - Increased volatility expected.")
        except:
            pass

    # Display summary in a nice box
    for summary in summaries:
        st.markdown(summary)


def _render_price_targets(signal: UnifiedSignal, additional_data: dict):
    """Render price targets section."""
    target = additional_data.get('target_mean')

    if target:
        upside = ((float(target) - signal.current_price) / signal.current_price * 100) if signal.current_price else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current", f"${signal.current_price:.2f}")
        with col2:
            color = "green" if upside > 0 else "red"
            st.metric("Target", f"${float(target):.2f}", f"{upside:+.1f}%")
        with col3:
            st.metric("Low", f"${float(additional_data.get('target_low', 0)):.2f}")
        with col4:
            st.metric("High", f"${float(additional_data.get('target_high', 0)):.2f}")

        # Analyst breakdown
        if additional_data.get('consensus'):
            cols = st.columns(5)
            labels = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']
            keys = ['strong_buy', 'buy', 'hold', 'sell', 'strong_sell']
            for i, (label, key) in enumerate(zip(labels, keys)):
                with cols[i]:
                    val = additional_data.get(key, 0) or 0
                    st.caption(f"{label}: {val}")
    else:
        st.caption("No analyst targets available")


def _render_chart(ticker: str, additional_data: dict):
    """Render Google Finance-style price chart with period selectors and key stats."""
    try:
        import yfinance as yf
        import plotly.graph_objects as go
        from datetime import datetime, timedelta

        stock = yf.Ticker(ticker)
        info = stock.info

        # Period selector
        period_options = {
            "1D": ("1d", "5m"),
            "5D": ("5d", "15m"),
            "1M": ("1mo", "1h"),
            "6M": ("6mo", "1d"),
            "YTD": ("ytd", "1d"),
            "1Y": ("1y", "1d"),
            "5Y": ("5y", "1wk"),
            "Max": ("max", "1mo"),
        }

        cols = st.columns(8)
        selected_period = st.session_state.get(f'chart_period_{ticker}', 'YTD')

        for i, period in enumerate(period_options.keys()):
            with cols[i]:
                if st.button(period, key=f"period_{ticker}_{period}",
                           width='stretch',
                           type="primary" if period == selected_period else "secondary"):
                    st.session_state[f'chart_period_{ticker}'] = period
                    selected_period = period
                    st.rerun()

        # Get data for selected period
        period_code, interval = period_options[selected_period]
        data = stock.history(period=period_code, interval=interval)

        if data.empty:
            st.caption("Chart unavailable")
            return

        # Calculate price change
        current_price = data['Close'].iloc[-1]
        start_price = data['Close'].iloc[0]
        price_change = current_price - start_price
        price_change_pct = (price_change / start_price) * 100
        is_positive = price_change >= 0

        # Color scheme
        line_color = "#00C805" if is_positive else "#FF5252"
        fill_color = "rgba(0, 200, 5, 0.1)" if is_positive else "rgba(255, 82, 82, 0.1)"

        # Price header
        st.markdown(f"""
        <div style="margin-bottom: 10px;">
            <span style="font-size: 2.2em; font-weight: bold;">{current_price:.2f}</span>
            <span style="font-size: 1em; color: #888;"> USD</span>
            <br>
            <span style="color: {line_color}; font-size: 1.1em;">
                {'+' if is_positive else ''}{price_change:.2f} ({'+' if is_positive else ''}{price_change_pct:.2f}%)
            </span>
            <span style="color: #888; font-size: 0.9em;"> {selected_period.lower()}</span>
        </div>
        """, unsafe_allow_html=True)

        # Create area chart
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Price',
            line=dict(color=line_color, width=2),
            fill='tozeroy',
            fillcolor=fill_color,
            hovertemplate='%{x}<br>$%{y:.2f}<extra></extra>'
        ))

        # Optional: Add target price line
        target = additional_data.get('target_mean')
        if target and selected_period in ['6M', 'YTD', '1Y']:
            fig.add_hline(y=float(target), line_dash="dash", line_color="#FFD700",
                          annotation_text=f"Target ${float(target):.2f}",
                          annotation_position="right")

        fig.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(
                showgrid=False,
                showticklabels=True,
                tickfont=dict(size=10, color='#888'),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.1)',
                tickfont=dict(size=10, color='#888'),
                tickprefix='$',
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            hovermode='x unified',
        )

        st.plotly_chart(fig, width='stretch')

        # Key stats grid (Google Finance style)
        _render_key_stats(stock, info, current_price)

    except Exception as e:
        st.caption(f"Chart error: {e}")


def _render_key_stats(stock, info: dict, current_price: float):
    """Render key stats in Google Finance style grid."""
    try:
        # Get values safely
        open_price = info.get('open') or info.get('regularMarketOpen', '-')
        high_price = info.get('dayHigh') or info.get('regularMarketDayHigh', '-')
        low_price = info.get('dayLow') or info.get('regularMarketDayLow', '-')
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE') or info.get('forwardPE', '-')
        high_52wk = info.get('fiftyTwoWeekHigh', '-')
        low_52wk = info.get('fiftyTwoWeekLow', '-')
        dividend_yield = info.get('dividendYield', 0)
        dividend_rate = info.get('dividendRate', 0)

        # Format market cap
        if market_cap and market_cap > 0:
            if market_cap >= 1e12:
                mkt_cap_str = f"{market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                mkt_cap_str = f"{market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                mkt_cap_str = f"{market_cap/1e6:.2f}M"
            else:
                mkt_cap_str = f"{market_cap:,.0f}"
        else:
            mkt_cap_str = "-"

        # Format dividend yield
        div_yield_str = f"{dividend_yield*100:.2f}%" if dividend_yield else "-"
        div_rate_str = f"{dividend_rate:.2f}" if dividend_rate else "-"

        # Format PE
        pe_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else str(pe_ratio)

        # Create stats grid
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div style="font-size: 0.85em;">
                <div style="color: #888;">Open</div>
                <div style="font-weight: 500;">{open_price if isinstance(open_price, str) else f'{open_price:.2f}'}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size: 0.85em; margin-top: 8px;">
                <div style="color: #888;">High</div>
                <div style="font-weight: 500;">{high_price if isinstance(high_price, str) else f'{high_price:.2f}'}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size: 0.85em; margin-top: 8px;">
                <div style="color: #888;">Low</div>
                <div style="font-weight: 500;">{low_price if isinstance(low_price, str) else f'{low_price:.2f}'}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="font-size: 0.85em;">
                <div style="color: #888;">Mkt cap</div>
                <div style="font-weight: 500;">{mkt_cap_str}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size: 0.85em; margin-top: 8px;">
                <div style="color: #888;">P/E ratio</div>
                <div style="font-weight: 500;">{pe_str}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size: 0.85em; margin-top: 8px;">
                <div style="color: #888;">52-wk high</div>
                <div style="font-weight: 500;">{high_52wk if isinstance(high_52wk, str) else f'{high_52wk:.2f}'}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style="font-size: 0.85em;">
                <div style="color: #888;">Dividend</div>
                <div style="font-weight: 500;">{div_yield_str}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size: 0.85em; margin-top: 8px;">
                <div style="color: #888;">Qtrly Div Amt</div>
                <div style="font-weight: 500;">{div_rate_str}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size: 0.85em; margin-top: 8px;">
                <div style="color: #888;">52-wk low</div>
                <div style="font-weight: 500;">{low_52wk if isinstance(low_52wk, str) else f'{low_52wk:.2f}'}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            # Price position in 52-week range
            if isinstance(high_52wk, (int, float)) and isinstance(low_52wk, (int, float)) and high_52wk > low_52wk:
                range_pct = (current_price - low_52wk) / (high_52wk - low_52wk) * 100
                range_color = "#00C805" if range_pct > 50 else "#FF5252"
                st.markdown(f"""
                <div style="font-size: 0.85em;">
                    <div style="color: #888;">52-wk range</div>
                    <div style="font-weight: 500; color: {range_color};">{range_pct:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Volume
            volume = info.get('volume', 0)
            avg_volume = info.get('averageVolume', 0)
            if volume:
                vol_str = f"{volume/1e6:.1f}M" if volume >= 1e6 else f"{volume/1e3:.0f}K"
                st.markdown(f"""
                <div style="font-size: 0.85em; margin-top: 8px;">
                    <div style="color: #888;">Volume</div>
                    <div style="font-weight: 500;">{vol_str}</div>
                </div>
                """, unsafe_allow_html=True)

            if avg_volume:
                avg_vol_str = f"{avg_volume/1e6:.1f}M" if avg_volume >= 1e6 else f"{avg_volume/1e3:.0f}K"
                st.markdown(f"""
                <div style="font-size: 0.85em; margin-top: 8px;">
                    <div style="color: #888;">Avg volume</div>
                    <div style="font-weight: 500;">{avg_vol_str}</div>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.caption(f"Stats error: {e}")


def _render_component_scores(signal: UnifiedSignal):
    """Render component scores with progress bars and detailed explanations."""

    # Score interpretation helper
    def get_score_interpretation(name: str, score: int) -> str:
        """Get detailed interpretation for each score type."""
        if name == "Technical":
            if score >= 70:
                return "Strong uptrend, above key MAs, momentum positive"
            elif score >= 55:
                return "Mild bullish setup, testing resistance"
            elif score >= 45:
                return "Consolidating, no clear direction"
            elif score >= 30:
                return "Weak, below moving averages"
            else:
                return "Strong downtrend, avoid"
        elif name == "Fundamental":
            if score >= 70:
                return "Strong financials, good growth, attractive valuation"
            elif score >= 55:
                return "Decent fundamentals, some positive metrics"
            elif score >= 45:
                return "Mixed fundamentals, fair valuation"
            elif score >= 30:
                return "Weak financials, concerning metrics"
            else:
                return "Poor fundamentals, high risk"
        elif name == "Sentiment":
            if score >= 70:
                return "Very positive news/social media sentiment"
            elif score >= 55:
                return "Bullish sentiment, positive coverage"
            elif score >= 45:
                return "Neutral sentiment, mixed coverage"
            elif score >= 30:
                return "Bearish sentiment, negative coverage"
            else:
                return "Very negative sentiment, strong concerns"
        elif name == "Options Flow":
            if score >= 70:
                return "Heavy call buying, smart money bullish"
            elif score >= 55:
                return "Slight bullish bias in options"
            elif score >= 45:
                return "Neutral options flow, no clear direction"
            elif score >= 30:
                return "Put buying exceeds calls, bearish"
            else:
                return "Heavy put buying, institutions hedging"
        elif name == "Squeeze":
            if score >= 70:
                return "HIGH squeeze potential - high SI + bullish setup"
            elif score >= 50:
                return "Moderate squeeze potential"
            else:
                return "Low squeeze risk, minimal short interest"
        elif name == "Earnings":
            if score >= 70:
                return "Strong earnings outlook, beat expected"
            elif score >= 55:
                return "Positive earnings expectations"
            elif score >= 45:
                return "Mixed earnings outlook"
            elif score >= 30:
                return "Weak earnings expected"
            else:
                return "Poor earnings outlook, miss likely"
        return ""

    components = [
        ("Technical", signal.technical_score, signal.technical_signal, signal.technical_reason),
        ("Fundamental", signal.fundamental_score, signal.fundamental_signal, signal.fundamental_reason),
        ("Sentiment", signal.sentiment_score, signal.sentiment_signal, signal.sentiment_reason),
        ("Options Flow", signal.options_score, signal.options_signal, signal.options_reason),
        ("Earnings", signal.earnings_score, signal.earnings_signal, signal.earnings_reason),
    ]

    # Add squeeze if available - FIX: use is not None check
    squeeze_score = getattr(signal, 'squeeze_score', None)
    if squeeze_score is not None:
        components.append(("Squeeze", squeeze_score, None, None))

    for name, score, sig, reason in components:
        # FIX: use is None check (0 is a valid score)
        score = score if score is not None else 50
        emoji = "🟢" if score >= 60 else "🔴" if score <= 40 else "🟡"

        col1, col2, col3 = st.columns([1.5, 2, 2])
        with col1:
            st.markdown(f"**{name}** {emoji}")
            st.caption(f"Score: {score}")
        with col2:
            st.progress(score / 100)
        with col3:
            # Show reason if available, otherwise show interpretation
            if reason:
                st.caption(reason[:80])
            else:
                interpretation = get_score_interpretation(name, score)
                st.caption(interpretation)




# =============================================================================
# INSIDER ACTIVITY
# =============================================================================

def _render_insider_activity(additional_data: dict):
    """Render insider activity."""
    transactions = additional_data.get('insider_transactions', [])

    # Filter valid
    transactions = [t for t in transactions
                    if t.get('shares_transacted') and float(t.get('shares_transacted', 0)) > 0
                    and t.get('transaction_type') in ('P', 'S')]

    if not transactions:
        st.caption("No recent insider activity")
        return

    # Calculate net
    buys = [t for t in transactions if t.get('transaction_type') == 'P']
    sells = [t for t in transactions if t.get('transaction_type') == 'S']

    buy_val = sum(float(t.get('total_value', 0) or 0) for t in buys)
    sell_val = sum(float(t.get('total_value', 0) or 0) for t in sells)

    if buy_val > sell_val * 1.5:
        st.success(f"🟢 Net Buying ({len(buys)} buys, {len(sells)} sells)")
    elif sell_val > buy_val * 1.5:
        st.error(f"🔴 Net Selling ({len(buys)} buys, {len(sells)} sells)")
    else:
        st.warning(f"🟡 Mixed ({len(buys)} buys, {len(sells)} sells)")

    # Show transactions
    for t in transactions[:4]:
        tx_type = "🟢 Buy" if t.get('transaction_type') == 'P' else "🔴 Sell"
        name = str(t.get('insider_name', ''))[:15]
        val = float(t.get('total_value', 0) or 0)
        val_str = f"${val / 1000:.0f}K" if val >= 1000 else f"${val:.0f}"
        st.caption(f"{tx_type}: {name} ({val_str})")




