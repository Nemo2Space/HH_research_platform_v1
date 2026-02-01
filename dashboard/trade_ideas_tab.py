"""
AI Trade Ideas Tab - Dashboard Component (ENHANCED VERSION)

Displays AI-generated trade recommendations with:
- Market context summary
- Top picks table (clickable)
- Detailed analysis panel
- Quick actions
- SEC Filing Insights integration (NEW)
- Dual Analyst quick analysis (NEW)

Author: Alpha Research Platform
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Any, Optional, Dict
import plotly.express as px
import plotly.graph_objects as go

from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# SEC INSIGHTS INTEGRATION
# =============================================================================
try:
    from src.signals.filing_signal import get_filing_insights
    SEC_INSIGHTS_AVAILABLE = True
except ImportError:
    SEC_INSIGHTS_AVAILABLE = False
    logger.warning("SEC Filing Insights not available for trade ideas")

# =============================================================================
# DUAL ANALYST INTEGRATION
# =============================================================================
try:
    from src.ai.dual_analyst import DualAnalystService
    DUAL_ANALYST_AVAILABLE = True
except ImportError:
    DUAL_ANALYST_AVAILABLE = False
    logger.warning("Dual Analyst Service not available for trade ideas")


# =============================================================================
# DISPLAY HELPER FUNCTIONS
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


def fmt_pct_change(val: float, base: float, spec: str = ".1f") -> str:
    """Calculate and format percentage change safely."""
    if val is None or base is None or base == 0:
        return "N/A"
    try:
        pct = ((val - base) / base) * 100
        return f"{pct:+{spec}}%"
    except (ValueError, TypeError, ZeroDivisionError):
        return "N/A"


def fmt_bool(val: Optional[bool], true_text: str = "Yes", false_text: str = "No", na_text: str = "N/A") -> str:
    """Format boolean value safely."""
    if val is None:
        return na_text
    return true_text if val else false_text


def safe_val(val: Any, default: float = 0) -> float:
    """Return value or default if None (for calculations)."""
    return val if val is not None else default


def _get_sec_score_badge(ticker: str) -> str:
    """Get SEC score badge for a ticker."""
    if not SEC_INSIGHTS_AVAILABLE:
        return ""

    try:
        insights = get_filing_insights(ticker)
        if not insights.get('available'):
            return "⚪"

        score = insights.get('score', 50)
        if score >= 70:
            return "🟢"
        elif score >= 50:
            return "🟡"
        else:
            return "🔴"
    except:
        return "⚪"


def _get_sec_score(ticker: str) -> Optional[int]:
    """Get SEC filing score for a ticker."""
    if not SEC_INSIGHTS_AVAILABLE:
        return None

    try:
        insights = get_filing_insights(ticker)
        if insights.get('available'):
            return insights.get('score')
    except:
        pass
    return None


# =============================================================================
# MAIN TAB RENDER FUNCTION
# =============================================================================

def render_trade_ideas_tab(positions: list = None, account_summary: dict = None):
    """
    Render the AI Trade Ideas tab.

    Args:
        positions: List of portfolio positions from IBKR
        account_summary: Account summary dict from IBKR
    """

    st.subheader("🤖 AI Trade Ideas")
    st.caption("AI-powered trade recommendations based on all platform data")

    # ============================================================
    # FILTER CONTROLS
    # ============================================================
    with st.expander("⚙️ Filter Settings", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            min_score = st.slider(
                "Min Total Score",
                0, 100, 40, 5,
                key="ideas_min_score",
                help="Minimum screener score to consider"
            )

        with col2:
            signal_filter = st.multiselect(
                "Signal Types",
                ["STRONG_BUY", "BUY", "HOLD", "SELL"],
                default=["STRONG_BUY", "BUY", "HOLD"],
                key="ideas_signal_filter",
                help="Include these signal types"
            )

        with col3:
            max_picks = st.slider(
                "Max Picks",
                5, 20, 10, 1,
                key="ideas_max_picks"
            )

        with col4:
            include_no_signal = st.checkbox(
                "Include stocks without signals",
                value=True,
                key="ideas_include_no_signal",
                help="Include stocks that have scores but no recent signal"
            )

        col5, col6, col7, col8 = st.columns(4)

        with col5:
            min_rs = st.slider(
                "Min Relative Strength",
                0, 100, 0, 10,
                key="ideas_min_rs",
                help="Minimum RS rating vs SPY"
            )

        with col6:
            require_bullish_options = st.checkbox(
                "Require Bullish Options",
                value=False,
                key="ideas_require_options"
            )

        with col7:
            skip_earnings = st.checkbox(
                "Skip if earnings < 7 days",
                value=True,
                key="ideas_skip_earnings"
            )

        with col8:
            sectors_filter = st.multiselect(
                "Sectors (empty = all)",
                ["Technology", "Healthcare", "Financials", "Consumer Discretionary",
                 "Consumer Staples", "Energy", "Industrials", "Materials",
                 "Utilities", "Real Estate", "Communication Services"],
                default=[],
                key="ideas_sectors"
            )

        # NEW: SEC Filter
        col9, col10, col11, col12 = st.columns(4)

        with col9:
            if SEC_INSIGHTS_AVAILABLE:
                min_sec_score = st.slider(
                    "Min SEC Filing Score",
                    0, 100, 0, 10,
                    key="ideas_min_sec_score",
                    help="Minimum SEC filing analysis score"
                )
            else:
                min_sec_score = 0
                st.caption("SEC insights N/A")

    # Generate button
    col_btn, col_status = st.columns([2, 3])

    with col_btn:
        generate_btn = st.button("🚀 Generate Trade Ideas", type="primary", key="generate_ideas")

    with col_status:
        if st.session_state.get('trade_ideas_result'):
            result = st.session_state.trade_ideas_result
            st.caption(f"Last generated: {result.generated_at.strftime('%H:%M:%S')}")

    if generate_btn:
        try:
            from src.analytics.trade_ideas import TradeIdeasGenerator

            # Build filter config from UI
            filter_config = {
                'min_score': min_score,
                'signal_types': signal_filter,
                'include_no_signal': include_no_signal,
                'min_rs': min_rs,
                'require_bullish_options': require_bullish_options,
                'skip_earnings_within_days': 7 if skip_earnings else 0,
                'sectors': sectors_filter if sectors_filter else None,
            }

            with st.spinner("Generating trade ideas..."):
                generator = TradeIdeasGenerator()
                result = generator.generate_ideas(
                    max_picks=max_picks,
                    filter_config=filter_config,
                    positions=positions or [],
                    account_summary=account_summary or {},
                )

            # Store result and generator
            st.session_state.trade_ideas_result = result
            st.session_state.trade_ideas_generator = generator

            st.rerun()

        except ImportError as e:
            st.error(f"Trade Ideas module not found: {e}")
            st.info("Make sure src/analytics/trade_ideas.py exists")
            return
        except Exception as e:
            st.error(f"Error generating trade ideas: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    # ============================================================
    # DISPLAY RESULTS
    # ============================================================
    if st.session_state.get('trade_ideas_result'):
        result = st.session_state.trade_ideas_result
        generator = st.session_state.get('trade_ideas_generator')

        # Market context
        if result.market_context:
            with st.expander("📊 Market Context", expanded=False):
                ctx = result.market_context
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Market Regime", ctx.get('regime', 'Unknown'))
                with cols[1]:
                    st.metric("VIX", fmt(ctx.get('vix'), ".1f"))
                with cols[2]:
                    st.metric("SPY Trend", ctx.get('spy_trend', 'Unknown'))
                with cols[3]:
                    st.metric("Sector Rotation", ctx.get('sector_rotation', 'N/A'))

        # Top picks
        if result.top_picks:
            st.markdown("### 🏆 Top Picks")

            # Build dataframe with SEC scores
            picks_data = []
            for pick in result.top_picks:
                # Get SEC score for this ticker
                sec_badge = _get_sec_score_badge(pick.ticker)
                sec_score = _get_sec_score(pick.ticker)

                picks_data.append({
                    'Ticker': pick.ticker,
                    'Score': fmt(pick.ai_score),
                    'Entry': fmt(pick.ai_entry, ".2f", "$"),
                    'Stop': fmt(pick.ai_stop_loss, ".2f", "$"),
                    'Target': fmt(pick.ai_target_1, ".2f", "$"),
                    'R:R': f"{pick.risk_reward_ratio:.1f}:1" if pick.risk_reward_ratio and pick.risk_reward_ratio > 0 else "N/A",
                    'Catalyst': (pick.ai_catalyst[:40] + "...") if pick.ai_catalyst and len(pick.ai_catalyst) > 40 else (pick.ai_catalyst or ""),
                    'Signal': pick.signal_type if pick.signal_type and pick.signal_type != "NOT_ANALYZED" else "N/A",
                    'Options': pick.options_sentiment if pick.options_sentiment and pick.options_sentiment != "NOT_ANALYZED" else "N/A",
                    'RS': fmt(pick.rs_rating),
                    'SEC': f"{sec_badge} {sec_score}" if sec_score else "N/A",
                    'Data': f"{pick.data_completeness:.0%}" if hasattr(pick, 'data_completeness') and pick.data_completeness else "N/A",
                })

            picks_df = pd.DataFrame(picks_data)

            # Apply SEC filter if set
            if SEC_INSIGHTS_AVAILABLE and min_sec_score > 0:
                # Filter based on SEC score
                filtered_picks = []
                for pick in result.top_picks:
                    sec_score = _get_sec_score(pick.ticker)
                    if sec_score is None or sec_score >= min_sec_score:
                        filtered_picks.append(pick)

                if len(filtered_picks) < len(result.top_picks):
                    st.info(f"Filtered to {len(filtered_picks)} picks with SEC score >= {min_sec_score}")

            # Display table
            st.dataframe(
                picks_df,
                width='stretch',
                hide_index=True,
                height=400
            )

            # Select ticker for detail view
            st.markdown("---")
            st.markdown("### 🔍 Detailed Analysis")

            ticker_options = [p.ticker for p in result.top_picks]

            col_select, col_btn, col_dual = st.columns([2, 1, 1])

            with col_select:
                selected_ticker = st.selectbox(
                    "Select a stock for detailed analysis",
                    options=ticker_options,
                    key="ideas_detail_ticker"
                )

            with col_btn:
                show_detail = st.button("📋 Show Details", key="show_detail_btn")

            with col_dual:
                if DUAL_ANALYST_AVAILABLE:
                    run_dual = st.button("🔬 Run Dual Analysis", key="run_dual_btn")
                else:
                    run_dual = False

            # Show detail for selected ticker
            if show_detail or st.session_state.get('ideas_selected_ticker'):
                st.session_state.ideas_selected_ticker = selected_ticker

                # Find the candidate
                selected_pick = next((p for p in result.top_picks if p.ticker == selected_ticker), None)

                if selected_pick and generator:
                    # Get detailed analysis
                    detail = generator.get_detailed_analysis(selected_pick)

                    # Display in columns
                    col_left, col_right = st.columns([3, 2])

                    with col_left:
                        st.code(detail, language=None)

                        # NEW: Show SEC insights for selected stock
                        if SEC_INSIGHTS_AVAILABLE:
                            _render_sec_mini_panel(selected_ticker)

                    with col_right:
                        # Quick stats cards
                        st.markdown("#### Quick Stats")

                        # Score gauge - FIXED: Handle None
                        score_value = safe_val(selected_pick.ai_score, 0)
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score_value,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "AI Score"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkgreen" if score_value >= 60 else "orange"},
                                'steps': [
                                    {'range': [0, 40], 'color': "lightcoral"},
                                    {'range': [40, 60], 'color': "lightyellow"},
                                    {'range': [60, 100], 'color': "lightgreen"}
                                ],
                            }
                        ))
                        fig.update_layout(height=200, margin=dict(t=50, b=0, l=0, r=0))
                        st.plotly_chart(fig, width='stretch')

                        # Key metrics - FIXED: Handle None values
                        st.metric("Entry Price", fmt(selected_pick.ai_entry, ".2f", "$"))

                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            # FIXED: Check for None before calculations
                            if selected_pick.ai_stop_loss and selected_pick.current_price and selected_pick.current_price > 0:
                                stop_pct = ((selected_pick.ai_stop_loss - selected_pick.current_price) / selected_pick.current_price * 100)
                                st.metric("Stop Loss", fmt(selected_pick.ai_stop_loss, ".2f", "$"), f"{stop_pct:.1f}%")
                            else:
                                st.metric("Stop Loss", fmt(selected_pick.ai_stop_loss, ".2f", "$"))

                        with col_m2:
                            # FIXED: Check for None before calculations
                            if selected_pick.ai_target_1 and selected_pick.current_price and selected_pick.current_price > 0:
                                target_pct = ((selected_pick.ai_target_1 - selected_pick.current_price) / selected_pick.current_price * 100)
                                st.metric("Target", fmt(selected_pick.ai_target_1, ".2f", "$"), f"+{target_pct:.1f}%")
                            else:
                                st.metric("Target", fmt(selected_pick.ai_target_1, ".2f", "$"))

                        # Position sizing helper
                        st.markdown("#### Position Sizing")
                        portfolio_value = account_summary.get('net_liquidation', 1000000) if account_summary else 1000000
                        risk_pct = st.slider("Risk % of Portfolio", 0.5, 3.0, 1.0, 0.5, key="risk_slider")

                        risk_amount = portfolio_value * (risk_pct / 100)

                        # FIXED: Calculate risk per share safely
                        current_price = safe_val(selected_pick.current_price, 0)
                        stop_loss = safe_val(selected_pick.ai_stop_loss, 0)
                        risk_per_share = current_price - stop_loss if current_price > 0 and stop_loss > 0 else 0

                        if risk_per_share > 0 and current_price > 0:
                            suggested_shares = int(risk_amount / risk_per_share)
                            position_value = suggested_shares * current_price
                            position_pct = (position_value / portfolio_value) * 100

                            st.write(f"**Suggested Position:**")
                            st.write(f"  Shares: {suggested_shares:,}")
                            st.write(f"  Value: ${position_value:,.0f}")
                            st.write(f"  Weight: {position_pct:.1f}%")
                            st.write(f"  Max Loss: ${risk_amount:,.0f}")
                        else:
                            st.warning("⚠️ Cannot calculate position size - missing price or stop loss data")

            # NEW: Run dual analysis
            if run_dual and DUAL_ANALYST_AVAILABLE and selected_ticker:
                _render_dual_analysis_for_trade(selected_ticker)

        else:
            st.info("No top picks found. Try adjusting your screener criteria or running a new scan.")

        # ============================================================
        # HONORABLE MENTIONS - FIXED: Handle None values
        # ============================================================
        if result.honorable_mentions:
            st.markdown("---")
            with st.expander("🏅 Honorable Mentions (HOLD)"):
                mentions_data = []
                for pick in result.honorable_mentions:
                    mentions_data.append({
                        'Ticker': pick.ticker,
                        'Score': fmt(pick.ai_score),
                        'Signal': pick.signal_type if pick.signal_type and pick.signal_type != "NOT_ANALYZED" else "N/A",
                        'Options': pick.options_sentiment if pick.options_sentiment and pick.options_sentiment != "NOT_ANALYZED" else "N/A",
                        'Reason': (pick.ai_catalyst[:50] if pick.ai_catalyst and len(pick.ai_catalyst) > 50 else pick.ai_catalyst) or "Meets criteria"
                    })
                st.dataframe(pd.DataFrame(mentions_data), width='stretch', hide_index=True)

        # ============================================================
        # AVOID LIST - FIXED: Handle None values
        # ============================================================
        if result.avoid_list:
            with st.expander("🚫 Avoid List"):
                avoid_data = []
                for pick in result.avoid_list:
                    risks = ", ".join(pick.ai_risks[:2]) if pick.ai_risks else "Multiple red flags"
                    avoid_data.append({
                        'Ticker': pick.ticker,
                        'Score': fmt(pick.ai_score),
                        'Risks': risks
                    })
                st.dataframe(pd.DataFrame(avoid_data), width='stretch', hide_index=True)

    else:
        # No ideas generated yet
        st.info("👆 Click 'Generate Trade Ideas' to get AI recommendations based on all platform data.")

        st.markdown("""
        **What the AI analyzes:**
        - ✅ Screener scores (sentiment, fundamental, technical)
        - ✅ Trading signals & committee decisions
        - ✅ Options flow & unusual activity
        - ✅ Short squeeze potential
        - ✅ Technical levels (support/resistance)
        - ✅ Relative strength vs SPY & sector
        - ✅ Liquidity & volume
        - ✅ Earnings calendar & events
        - ✅ Your current positions
        - ✅ **SEC Filing Insights** (NEW)

        **Output:**
        - 🏆 Top 10 picks with entry/stop/target
        - 📊 AI score and reasoning for each
        - ⚠️ Risk factors to watch
        - 💰 Position sizing guidance
        - 📄 SEC filing analysis scores
        """)


# =============================================================================
# NEW: SEC MINI PANEL FOR TRADE IDEAS
# =============================================================================
def _render_sec_mini_panel(ticker: str):
    """Render a compact SEC insights panel."""
    if not SEC_INSIGHTS_AVAILABLE:
        return

    try:
        insights = get_filing_insights(ticker)

        if not insights.get('available'):
            return

        st.markdown("#### 📄 SEC Filing Insights")

        col1, col2 = st.columns(2)

        with col1:
            score = insights.get('score', 0)
            label = insights.get('score_label', 'Unknown')
            st.metric("Filing Score", f"{score:.0f}/100", label)

        with col2:
            quality = insights.get('data_quality', {})
            st.caption(f"📅 {quality.get('freshness_days', 0)} days ago")
            st.caption(f"📊 {quality.get('filings_analyzed', 0)} filings")

        # Signals
        bullish = insights.get('bullish_signals', [])
        bearish = insights.get('bearish_signals', [])

        if bullish:
            st.markdown("✅ " + " • ".join(bullish[:2]))

        if bearish:
            st.markdown("⚠️ " + " • ".join(bearish[:2]))

    except Exception as e:
        logger.warning(f"Error rendering SEC mini panel for {ticker}: {e}")


# =============================================================================
# NEW: DUAL ANALYSIS FOR TRADE IDEAS
# =============================================================================
def _render_dual_analysis_for_trade(ticker: str):
    """Run and display dual analysis for a trade idea."""
    if not DUAL_ANALYST_AVAILABLE:
        return

    cache_key = f"dual_trade_{ticker}"

    if cache_key not in st.session_state:
        with st.spinner(f"Running dual analysis for {ticker}..."):
            try:
                service = DualAnalystService()
                question = f"Is {ticker} a good buy right now? Consider entry, exit, and risk."
                result = service.analyze_for_display(ticker, question)
                result['timestamp'] = datetime.now().strftime('%H:%M:%S')
                st.session_state[cache_key] = result
            except Exception as e:
                st.error(f"Dual analysis failed: {e}")
                return

    result = st.session_state[cache_key]

    st.markdown("---")
    st.markdown("### 🔬 Dual Analyst View")

    # Icons
    ICONS = {
        "very_bullish": "🚀", "bullish": "📈", "neutral": "➡️",
        "bearish": "📉", "very_bearish": "🔻", "unknown": "❓",
    }

    # Agreement
    agreement = result.get('evaluation', {}).get('agreement_score', 0)
    st.progress(agreement, text=f"Analyst Agreement: {agreement:.0%}")

    col_sql, col_rag = st.columns(2)

    with col_sql:
        sql = result.get('sql_analyst', {})
        sentiment = sql.get('sentiment', 'unknown')
        icon = ICONS.get(sentiment, "❓")
        st.markdown(f"**📊 Quant:** {icon} {sentiment.replace('_', ' ').title()}")
        st.caption(sql.get('summary', '')[:100])

    with col_rag:
        rag = result.get('rag_analyst', {})
        sentiment = rag.get('sentiment', 'unknown')
        icon = ICONS.get(sentiment, "❓")
        st.markdown(f"**📄 Qual:** {icon} {sentiment.replace('_', ' ').title()}")
        st.caption(rag.get('summary', '')[:100])

    # Synthesis
    synthesis = result.get('synthesis', {})
    sentiment = synthesis.get('sentiment', 'unknown')
    icon = ICONS.get(sentiment, "❓")
    st.markdown(f"**🎯 Verdict:** {icon} {sentiment.replace('_', ' ').title()} ({synthesis.get('confidence', 0):.0%})")


# Standalone test
if __name__ == "__main__":
    st.set_page_config(page_title="AI Trade Ideas Test", layout="wide")
    render_trade_ideas_tab()