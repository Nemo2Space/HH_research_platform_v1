"""
Deep Dive Tab - Comprehensive Single Ticker Analysis

Shows everything about one ticker:
- Price chart with key levels
- Component scores breakdown
- Committee decision
- News & sentiment
- Earnings intelligence
- Options flow
- Portfolio position
- AI Chat integration
- SEC Filing Insights (NEW)
- Dual Analyst Analysis (NEW)

Author: Alpha Research Platform
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from src.tabs.deep_dive_earnings import render_post_earnings_section
    EARNINGS_REACTION_AVAILABLE = True
except ImportError:
    EARNINGS_REACTION_AVAILABLE = False

# Import core components
try:
    from src.core import (
        UnifiedSignal,
        SignalStrength,
        RiskLevel,
        AssetType,
        generate_signal,
        get_signal_engine,
    )

    SIGNAL_HUB_AVAILABLE = True
except ImportError as e:
    SIGNAL_HUB_AVAILABLE = False
    logger.error(f"Signal Hub not available: {e}")

# =============================================================================
# NEW: SEC INSIGHTS INTEGRATION
# =============================================================================
try:
    from src.signals.filing_signal import get_filing_insights
    SEC_INSIGHTS_AVAILABLE = True
except ImportError:
    SEC_INSIGHTS_AVAILABLE = False
    logger.warning("SEC Filing Insights not available")

# =============================================================================
# NEW: DUAL ANALYST INTEGRATION
# =============================================================================
try:
    from src.ai.dual_analyst import DualAnalystService
    DUAL_ANALYST_AVAILABLE = True
except ImportError:
    DUAL_ANALYST_AVAILABLE = False
    logger.warning("Dual Analyst Service not available")


def render_deep_dive_tab(ticker: str = None):
    """
    Render the deep dive page for a single ticker.

    Args:
        ticker: Optional ticker to display. If None, shows selector.
    """
    st.markdown("## 🔍 Deep Dive Analysis")

    if not SIGNAL_HUB_AVAILABLE:
        st.error("Signal Hub not available")
        return

    # Ticker selector
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        # Get available tickers
        tickers = _get_available_tickers()

        # Use provided ticker or session state
        default_ticker = ticker or st.session_state.get('deep_dive_ticker', tickers[0] if tickers else 'AAPL')

        if default_ticker in tickers:
            default_idx = tickers.index(default_ticker)
        else:
            default_idx = 0

        selected_ticker = st.selectbox(
            "Select Ticker",
            options=tickers,
            index=default_idx,
            key="deep_dive_selector"
        )
        st.session_state.deep_dive_ticker = selected_ticker

    with col2:
        if st.button("🔄 Refresh Analysis", key="refresh_deep_dive"):
            st.session_state.pop(f'deep_dive_signal_{selected_ticker}', None)
            st.session_state.pop(f'dual_analysis_{selected_ticker}', None)

    # Generate or get cached signal
    cache_key = f'deep_dive_signal_{selected_ticker}'

    if cache_key not in st.session_state:
        with st.spinner(f"Analyzing {selected_ticker}..."):
            signal = generate_signal(selected_ticker)
            st.session_state[cache_key] = signal
    else:
        signal = st.session_state[cache_key]

    # Render the deep dive
    _render_deep_dive(signal)

    # Post-earnings reaction section (after main deep dive)
    # FIX: Use selected_ticker, not ticker parameter
    if EARNINGS_REACTION_AVAILABLE:
        render_post_earnings_section(selected_ticker)


def _get_available_tickers() -> List[str]:
    """Get list of tickers for selection."""
    try:
        from src.db.connection import get_engine
        import pandas as pd

        engine = get_engine()
        query = "SELECT DISTINCT ticker FROM screener_scores ORDER BY ticker"
        df = pd.read_sql(query, engine)

        if not df.empty:
            return df['ticker'].tolist()
    except:
        pass

    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']


def _render_deep_dive(signal: UnifiedSignal):
    """Render the full deep dive analysis."""

    # ================================================================
    # HEADER - Ticker info and key metrics
    # ================================================================
    _render_header(signal)

    st.markdown("---")

    # ================================================================
    # MAIN CONTENT - Two columns
    # ================================================================
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Price Chart
        _render_price_chart(signal)

        # Component Breakdown
        _render_component_breakdown(signal)

        # News & Sentiment (if available)
        _render_news_section(signal)

    with col_right:
        # Signal Summary Box
        _render_signal_summary(signal)

        # Committee Decision
        _render_committee_decision(signal)

        # Risk Assessment
        _render_risk_assessment(signal)

        # Earnings Intelligence
        _render_earnings_section(signal)

        # Portfolio Position (if held)
        if signal.in_portfolio:
            _render_portfolio_position(signal)

    st.markdown("---")

    # ================================================================
    # NEW: SEC FILING INSIGHTS SECTION
    # ================================================================
    if SEC_INSIGHTS_AVAILABLE:
        _render_sec_insights_section(signal.ticker)

    st.markdown("---")

    # ================================================================
    # NEW: DUAL ANALYST SECTION
    # ================================================================
    if DUAL_ANALYST_AVAILABLE:
        _render_dual_analyst_section(signal.ticker)

    st.markdown("---")

    # ================================================================
    # BOTTOM SECTION - Additional Analysis
    # ================================================================

    # Options Flow Details
    _render_options_flow(signal)

    # AI Chat Section
    _render_ai_chat_section(signal)


# =============================================================================
# NEW: SEC FILING INSIGHTS COMPONENT
# =============================================================================
def _render_sec_insights_section(ticker: str):
    """Render SEC Filing Insights section."""

    with st.expander("📄 SEC Filing Insights", expanded=True):
        if not SEC_INSIGHTS_AVAILABLE:
            st.info("SEC Filing Insights module not available")
            return

        try:
            insights = get_filing_insights(ticker)
        except Exception as e:
            st.error(f"Error loading SEC insights: {e}")
            return

        if not insights.get('available'):
            st.info(f"No SEC filing data available for {ticker}. Run `python -m src.rag.sec_ingestion` to ingest filings.")
            return

        # Score and rating
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            score = insights.get('score', 0)
            st.metric("Filing Score", f"{score:.0f}/100")

        with col2:
            label = insights.get('score_label', 'Unknown')
            st.metric("Rating", label)

        with col3:
            quality = insights.get('data_quality', {})
            st.caption(f"📅 Data from {quality.get('freshness_days', 0)} days ago")
            st.caption(f"📊 {quality.get('filings_analyzed', 0)} filings analyzed")

        # Factor breakdown
        st.markdown("**Factor Breakdown:**")
        factors = insights.get('factors', {})

        cols = st.columns(5)
        factor_items = [
            ('guidance', 'Guidance'),
            ('risk', 'Risk'),
            ('litigation', 'Litigation'),
            ('china', 'China'),
            ('ai_demand', 'AI Demand'),
        ]

        for col, (key, label) in zip(cols, factor_items):
            with col:
                factor = factors.get(key, {})
                factor_score = factor.get('score', 50)
                st.metric(label, f"{factor_score:.0f}")

        # Signals
        col_bull, col_bear = st.columns(2)

        with col_bull:
            bullish = insights.get('bullish_signals', [])
            if bullish:
                st.markdown("**✅ Bullish:**")
                for sig in bullish[:3]:
                    st.markdown(f"• {sig}")

        with col_bear:
            bearish = insights.get('bearish_signals', [])
            if bearish:
                st.markdown("**⚠️ Bearish:**")
                for sig in bearish[:3]:
                    st.markdown(f"• {sig}")


# =============================================================================
# NEW: DUAL ANALYST COMPONENT
# =============================================================================
def _render_dual_analyst_section(ticker: str):
    """Render Dual Analyst Analysis section."""

    with st.expander("🔬 AI Dual Analysis (SQL + RAG)", expanded=False):
        if not DUAL_ANALYST_AVAILABLE:
            st.info("Dual Analyst Service not available. Install with: `pip install -e .`")
            return

        # Check for cached result
        cache_key = f"dual_analysis_{ticker}"

        # Question input
        default_question = f"What is the investment outlook for {ticker}?"
        question = st.text_input(
            "Analysis Question",
            value=default_question,
            key=f"dual_question_{ticker}"
        )

        col_run, col_status = st.columns([1, 3])

        with col_run:
            run_analysis = st.button("🚀 Run Dual Analysis", key=f"dual_run_{ticker}")

        with col_status:
            if cache_key in st.session_state:
                result = st.session_state[cache_key]
                if 'timestamp' in result:
                    st.caption(f"Last run: {result['timestamp']}")

        # Run analysis
        if run_analysis:
            with st.spinner("Running dual analysis (15-30 seconds)..."):
                try:
                    service = DualAnalystService()
                    result = service.analyze_for_display(ticker, question)
                    result['timestamp'] = datetime.now().strftime('%H:%M:%S')
                    st.session_state[cache_key] = result
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    logger.exception(f"Dual analyst failed for {ticker}")
                    return

        # Display cached result
        if cache_key in st.session_state:
            result = st.session_state[cache_key]
            _render_dual_analyst_result(result)


def _render_dual_analyst_result(result: Dict[str, Any]):
    """Render dual analysis result."""

    # Sentiment colors and icons
    ICONS = {
        "very_bullish": "🚀",
        "bullish": "📈",
        "neutral": "➡️",
        "bearish": "📉",
        "very_bearish": "🔻",
        "unknown": "❓",
    }

    # Agreement score
    agreement = result.get('evaluation', {}).get('agreement_score', 0)
    st.progress(agreement, text=f"Analyst Agreement: {agreement:.0%}")

    # Two columns for analysts
    col_sql, col_rag = st.columns(2)

    with col_sql:
        sql = result.get('sql_analyst', {})
        sentiment = sql.get('sentiment', 'unknown')
        icon = ICONS.get(sentiment, "❓")
        confidence = sql.get('confidence', 0)

        st.markdown("### 📊 Quantitative Analyst")
        st.markdown(f"**{icon} {sentiment.replace('_', ' ').title()}** ({confidence:.0%})")
        st.markdown(f"_{sql.get('summary', 'No summary')}_")

        bullish = sql.get('bullish', [])
        if bullish:
            st.markdown("✅ " + ", ".join(bullish[:2]))

        bearish = sql.get('bearish', [])
        if bearish:
            st.markdown("⚠️ " + ", ".join(bearish[:2]))

    with col_rag:
        rag = result.get('rag_analyst', {})
        sentiment = rag.get('sentiment', 'unknown')
        icon = ICONS.get(sentiment, "❓")
        confidence = rag.get('confidence', 0)

        st.markdown("### 📄 Qualitative Analyst")
        st.markdown(f"**{icon} {sentiment.replace('_', ' ').title()}** ({confidence:.0%})")
        st.markdown(f"_{rag.get('summary', 'No summary')}_")

        bullish = rag.get('bullish', [])
        if bullish:
            st.markdown("✅ " + ", ".join(bullish[:2]))

        bearish = rag.get('bearish', [])
        if bearish:
            st.markdown("⚠️ " + ", ".join(bearish[:2]))

    # Synthesis
    st.divider()
    synthesis = result.get('synthesis', {})
    sentiment = synthesis.get('sentiment', 'unknown')
    icon = ICONS.get(sentiment, "❓")
    confidence = synthesis.get('confidence', 0)

    st.markdown(f"### 🎯 Final Verdict: {icon} {sentiment.replace('_', ' ').title()}")
    st.markdown(f"**Confidence:** {confidence:.0%}")
    st.markdown(synthesis.get('summary', ''))

    # Risk flags
    risk_flags = synthesis.get('risk_flags', [])
    if risk_flags:
        st.warning("**🚨 Risk Flags:** " + ", ".join(risk_flags))

    # Conflicts
    conflicts = result.get('evaluation', {}).get('conflicts', [])
    if conflicts:
        st.info("**⚠️ Analyst Conflicts:** " + ", ".join(conflicts))


def _render_header(signal: UnifiedSignal):
    """Render the header with ticker info and key metrics."""

    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

    with col1:
        # Ticker and name
        asset_icon = "🏦" if signal.asset_type == AssetType.BOND_ETF else "📈"
        st.markdown(f"### {asset_icon} {signal.ticker} - {signal.company_name}")

        # Sector and price
        price_str = f"${signal.current_price:.2f}" if signal.current_price else "N/A"
        st.caption(f"**{signal.sector}** | Price: {price_str}")

        # Flags
        if signal.flags:
            st.markdown(" ".join(signal.flags))

    with col2:
        emoji = signal.get_signal_emoji()
        st.metric(
            "Today Signal",
            f"{emoji} {signal.today_score}%",
            signal.today_signal.value
        )

    with col3:
        stars = signal.get_stars()
        st.metric(
            "Long-term",
            f"{stars}",
            f"{signal.longterm_score}/100"
        )

    with col4:
        risk_emoji = signal.get_risk_emoji()
        st.metric(
            "Risk",
            f"{risk_emoji} {signal.risk_level.value}",
            f"Score: {signal.risk_score}"
        )


def _render_signal_summary(signal: UnifiedSignal):
    """Render the signal summary box."""

    st.markdown("#### 📊 Signal Summary")

    # Color-coded box based on signal
    if signal.today_score >= 65:
        st.success(f"**{signal.signal_reason}**")
    elif signal.today_score <= 35:
        st.error(f"**{signal.signal_reason}**")
    else:
        st.warning(f"**{signal.signal_reason}**")

    # Key levels
    if signal.target_price or signal.stop_loss:
        st.markdown("**Key Levels:**")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Target", f"${signal.target_price:.2f}" if signal.target_price else "-")
        with cols[1]:
            st.metric("Stop", f"${signal.stop_loss:.2f}" if signal.stop_loss else "-")
        with cols[2]:
            if signal.target_price and signal.current_price and signal.current_price > 0:
                upside = ((signal.target_price - signal.current_price) / signal.current_price) * 100
                st.metric("Upside", f"{upside:.1f}%")
            else:
                st.metric("Upside", "-")


def _render_component_breakdown(signal: UnifiedSignal):
    """Render the component scores breakdown."""

    st.markdown("#### 📈 Component Scores")

    # Create component data
    components = [
        ("Technical", signal.technical_score, signal.technical_signal),
        ("Fundamental", signal.fundamental_score, signal.fundamental_signal),
        ("Sentiment", signal.sentiment_score, signal.sentiment_signal),
        ("Options", signal.options_score, signal.options_signal),
    ]

    # Create bar chart
    fig = go.Figure()

    colors = []
    for name, score, sig in components:
        if score >= 65:
            colors.append('#00C853')
        elif score <= 35:
            colors.append('#FF5252')
        else:
            colors.append('#FFC107')

    fig.add_trace(go.Bar(
        x=[c[0] for c in components],
        y=[c[1] for c in components],
        marker_color=colors,
        text=[f"{c[1]:.0f}" for c in components],
        textposition='outside'
    ))

    fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis=dict(range=[0, 100]),
        showlegend=False
    )

    st.plotly_chart(fig, width='stretch')

    # Show signal text for each component
    cols = st.columns(4)
    for i, (name, score, sig) in enumerate(components):
        with cols[i]:
            emoji = "🟢" if score >= 65 else "🔴" if score <= 35 else "🟡"
            st.caption(f"{emoji} {sig}")


def _render_committee_decision(signal: UnifiedSignal):
    """Render the committee decision section."""

    st.markdown("#### 🏛️ Committee Decision")

    if signal.committee_verdict:
        # Verdict with color
        verdict = signal.committee_verdict
        if verdict in ['STRONG_BUY', 'BUY']:
            st.success(f"**{verdict}**")
        elif verdict in ['STRONG_SELL', 'SELL']:
            st.error(f"**{verdict}**")
        else:
            st.warning(f"**{verdict}**")

        # Details - use committee_confidence (correct attribute)
        confidence = getattr(signal, 'committee_confidence', None)
        if confidence:
            try:
                conf_val = float(confidence) if not isinstance(confidence, (int, float)) else confidence
                st.metric("Confidence", f"{conf_val:.0%}" if conf_val <= 1 else f"{conf_val:.0f}%")
            except (ValueError, TypeError):
                st.metric("Confidence", str(confidence))

        # Expected alpha - use correct attribute name
        expected_alpha = getattr(signal, 'expected_alpha_bps', None) or getattr(signal, 'expected_alpha', None)
        if expected_alpha:
            st.metric("Expected Alpha", f"{expected_alpha} bps")

        # Horizon
        horizon = getattr(signal, 'horizon_days', None)
        if horizon:
            st.caption(f"Horizon: {horizon} days")

        # Agreement if available
        agreement = getattr(signal, 'committee_agreement', None)
        if agreement:
            try:
                agree_val = float(agreement) if not isinstance(agreement, (int, float)) else agreement
                st.caption(f"Agreement: {agree_val:.0%}" if agree_val <= 1 else f"Agreement: {agree_val:.0f}%")
            except (ValueError, TypeError):
                pass
    else:
        st.caption("No committee decision yet")
        st.button("🚀 Run Committee", key="run_committee_from_deep_dive")


def _render_risk_assessment(signal: UnifiedSignal):
    """Render the risk assessment section."""

    st.markdown("#### ⚠️ Risk Assessment")

    # Risk score gauge
    risk_score = getattr(signal, 'risk_score', None) or 50

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred" if risk_score >= 70 else "orange" if risk_score >= 40 else "green"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
        }
    ))

    fig.update_layout(height=150, margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig, width='stretch')

    # Risk flags - try multiple possible attribute names
    risk_flags = getattr(signal, 'risk_flags', None) or getattr(signal, 'risks', None) or []
    if risk_flags:
        for risk in risk_flags[:3]:
            st.caption(f"⚠️ {risk}")


def _render_earnings_section(signal: UnifiedSignal):
    """Render earnings intelligence section."""

    st.markdown("#### 📅 Earnings")

    if signal.earnings_date:
        days = signal.days_to_earnings
        if days and days > 0:
            if days <= 7:
                st.warning(f"⏰ **{days} days** until earnings")
            else:
                st.info(f"📅 {days} days until {signal.earnings_date}")
        elif days == 0:
            st.error("🔔 **Earnings TODAY**")
        else:
            st.caption(f"Last earnings: {signal.earnings_date}")
    else:
        st.caption("No earnings date available")


def _render_portfolio_position(signal: UnifiedSignal):
    """Render portfolio position info."""

    st.markdown("#### 💼 Your Position")

    if signal.position_size:
        st.metric("Shares", f"{signal.position_size:,.0f}")

    if signal.position_value:
        st.metric("Value", f"${signal.position_value:,.0f}")

    if signal.position_pnl_pct:
        color = "green" if signal.position_pnl_pct >= 0 else "red"
        st.markdown(f"P&L: <span style='color:{color}'>{signal.position_pnl_pct:+.1f}%</span>",
                    unsafe_allow_html=True)


def _render_price_chart(signal: UnifiedSignal):
    """Render price chart with key levels."""

    st.markdown("#### 📈 Price Chart")

    try:
        import yfinance as yf

        # Fetch price data
        stock = yf.Ticker(signal.ticker)
        hist = stock.history(period="6mo")

        if hist.empty:
            st.warning("Price data not available")
            return

        # Create candlestick chart
        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='Price'
        ))

        # Add moving averages
        if len(hist) >= 20:
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['MA20'],
                mode='lines',
                name='MA20',
                line=dict(color='orange', width=1)
            ))

        if len(hist) >= 50:
            hist['MA50'] = hist['Close'].rolling(window=50).mean()
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['MA50'],
                mode='lines',
                name='MA50',
                line=dict(color='blue', width=1)
            ))

        # Add target/stop lines if available
        if signal.target_price:
            fig.add_hline(y=signal.target_price, line_dash="dash", line_color="green",
                          annotation_text=f"Target: ${signal.target_price:.2f}")

        if signal.stop_loss:
            fig.add_hline(y=signal.stop_loss, line_dash="dash", line_color="red",
                          annotation_text=f"Stop: ${signal.stop_loss:.2f}")

        # Mark earnings date
        if signal.earnings_date and signal.days_to_earnings and 0 < signal.days_to_earnings <= 30:
            fig.add_vline(x=signal.earnings_date, line_dash="dot", line_color="purple",
                          annotation_text="ER")

        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, width='stretch')

    except Exception as e:
        st.warning(f"Could not load price chart: {e}")


def _render_news_section(signal: UnifiedSignal):
    """Render recent news section."""

    st.markdown("#### 📰 Recent News")

    try:
        from src.db.connection import get_engine
        import pandas as pd

        engine = get_engine()

        query = """
                SELECT headline, source, published_at, url, ai_sentiment_fast as sentiment_score
                FROM news_articles
                WHERE ticker = %s
                  AND headline IS NOT NULL AND headline != ''
                ORDER BY COALESCE(published_at, created_at) DESC LIMIT 5
                """
        df = pd.read_sql(query, engine, params=(signal.ticker,))

        if df.empty:
            st.caption("No recent news articles")
            return

        for _, row in df.iterrows():
            sent = row.get('sentiment_score', 50)
            emoji = "🟢" if sent and sent >= 60 else "🔴" if sent and sent <= 40 else "🟡"

            headline = row.get('headline', 'Untitled')[:80]
            source = row.get('source', 'Unknown')
            url = row.get('url', '')

            if url:
                st.markdown(f"{emoji} [{headline}]({url}) - *{source}*")
            else:
                st.markdown(f"{emoji} {headline} - *{source}*")

    except Exception as e:
        st.caption(f"News not available")


def _render_options_flow(signal: UnifiedSignal):
    """Render options flow analysis."""

    with st.expander("🔮 Options Flow Analysis", expanded=False):

        if signal.options_score == 50 and signal.options_reason == "Score 50":
            st.caption("No options flow data available")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Options Sentiment", signal.options_score)
            st.caption(signal.options_reason)

        with col2:
            emoji = "🟢" if signal.options_signal == "BUY" else "🔴" if signal.options_signal == "SELL" else "🟡"
            st.markdown(f"**Signal:** {emoji} {signal.options_signal}")

        # Try to load detailed options data
        try:
            from src.db.connection import get_engine
            import pandas as pd

            engine = get_engine()
            query = """
                    SELECT *
                    FROM options_flow_scores
                    WHERE ticker = %s
                    ORDER BY analysis_date DESC LIMIT 1
                    """
            df = pd.read_sql(query, engine, params=(signal.ticker,))

            if not df.empty:
                row = df.iloc[0]

                st.markdown("**Details:**")
                cols = st.columns(4)

                with cols[0]:
                    st.metric("Call Volume", f"{row.get('call_volume', 0):,.0f}")
                with cols[1]:
                    st.metric("Put Volume", f"{row.get('put_volume', 0):,.0f}")
                with cols[2]:
                    pc_ratio = row.get('put_call_ratio', 0)
                    st.metric("P/C Ratio", f"{pc_ratio:.2f}" if pc_ratio else "N/A")
                with cols[3]:
                    squeeze = row.get('short_squeeze_score', 0)
                    st.metric("Squeeze Score", f"{squeeze:.0f}" if squeeze else "N/A")

        except:
            pass


def _render_ai_chat_section(signal: UnifiedSignal):
    """Render AI chat integration section."""

    with st.expander("🤖 Ask AI About This Stock", expanded=False):

        st.markdown(f"Ask questions about **{signal.ticker}**")

        # Quick question buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Full Analysis", key="ai_full"):
                st.session_state.ai_question = f"Give me a complete analysis of {signal.ticker}"

        with col2:
            if st.button("🎯 Trade Idea", key="ai_trade"):
                st.session_state.ai_question = f"Should I buy or sell {signal.ticker}? Give me a trade idea with entry, target, and stop."

        with col3:
            if st.button("⚠️ Risks", key="ai_risks"):
                st.session_state.ai_question = f"What are the main risks for {signal.ticker} right now?"

        # Custom question
        question = st.text_input(
            "Or ask your own question:",
            value=st.session_state.get('ai_question', ''),
            key="ai_custom_question"
        )

        if st.button("🚀 Ask AI", key="ai_submit") and question:
            with st.spinner("Thinking..."):
                try:
                    # Try to use chat integration
                    from src.analytics.chat import process_chat_message

                    response = process_chat_message(question, ticker_context=signal.ticker)
                    st.markdown(response)

                except ImportError:
                    # Fallback - just show signal summary
                    st.info(f"""
                    **{signal.ticker} Summary:**

                    - Today Signal: {signal.today_signal.value} ({signal.today_score}%)
                    - Long-term: {signal.longterm_score}/100
                    - Risk: {signal.risk_level.value}

                    **Why:** {signal.signal_reason}

                    **Components:**
                    - Technical: {signal.technical_score} ({signal.technical_signal})
                    - Fundamental: {signal.fundamental_score} ({signal.fundamental_signal})
                    - Sentiment: {signal.sentiment_score} ({signal.sentiment_signal})
                    - Options: {signal.options_score} ({signal.options_signal})

                    *Full AI chat requires chat module integration*
                    """)
                except Exception as e:
                    st.error(f"AI error: {e}")


# ================================================================
# STANDALONE PAGE RUNNER
# ================================================================

def render_deep_dive_standalone():
    """Render deep dive as standalone page (for tab integration)."""
    render_deep_dive_tab()


if __name__ == "__main__":
    st.set_page_config(page_title="Deep Dive", layout="wide")
    render_deep_dive_tab()