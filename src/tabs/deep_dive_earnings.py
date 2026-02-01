"""
Deep Dive - Post-Earnings Reaction Section

Adds a post-earnings reaction analysis section to the Deep Dive tab.
Shows:
- Why the stock moved (from news)
- Quantitative assessment (oversold/justified)
- Specific recommendation with price levels

Usage:
    # In deep_dive_tab.py
    from src.tabs.deep_dive_earnings import render_post_earnings_section

    render_post_earnings_section(ticker)

Author: Alpha Research Platform
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any


def render_post_earnings_section(ticker: str):
    """
    Render the post-earnings reaction analysis section in Deep Dive.

    Args:
        ticker: Stock symbol
    """

    st.markdown("---")
    st.subheader("📊 Post-Earnings Reaction Analysis")

    # Check if we should show this section
    is_post, days_since = _is_post_earnings(ticker)

    if not is_post:
        st.info("This section shows when a stock has reported earnings within the last 5 days.")

        # Still allow manual analysis
        if st.button("🔍 Run Analysis Anyway", key=f"force_reaction_{ticker}"):
            _render_analysis(ticker)
        return

    st.caption(f"Earnings reported {days_since} day(s) ago")
    _render_analysis(ticker)


def _is_post_earnings(ticker: str) -> tuple:
    """
    Check if ticker is in post-earnings window (within 5 days).

    Returns:
        (is_post_earnings: bool, days_since: int)
    """
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        ed = stock.earnings_dates

        if ed is not None and not ed.empty:
            today = date.today()
            for idx in ed.index:
                earnings_date = idx.date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
                days_since = (today - earnings_date).days

                if 0 <= days_since <= 5:
                    return True, days_since

        return False, 0

    except Exception:
        return False, 0


def _render_analysis(ticker: str):
    """Render the full analysis."""

    try:
        from src.analytics.earnings_intelligence.reaction_analyzer import analyze_post_earnings

        with st.spinner("Analyzing post-earnings reaction..."):
            result = analyze_post_earnings(ticker)

        # Main metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Recommendation badge
            rec = result.recommendation.value
            rec_colors = {
                'STRONG_BUY': '🟢',
                'BUY_DIP': '🟢',
                'NIBBLE': '🟡',
                'WAIT': '🟡',
                'AVOID': '🔴',
                'SELL': '🔴',
                'TAKE_PROFITS': '🟠',
            }
            st.metric(
                "Recommendation",
                f"{rec_colors.get(rec, '⚪')} {rec}",
                delta=f"{result.confidence:.0f}% confidence"
            )

        with col2:
            # Reaction
            reaction = result.reaction_pct or 0
            st.metric(
                "Price Reaction",
                f"{reaction:+.1f}%",
                delta=f"${result.price_before:.2f} → ${result.price_current:.2f}" if result.price_before else None
            )

        with col3:
            # Oversold score
            score = result.oversold_score
            if score >= 65:
                status = "Oversold"
                color = "🟢"
            elif score <= 40:
                status = "Fair"
                color = "🟡"
            else:
                status = "Neutral"
                color = "⚪"
            st.metric(
                "Oversold Score",
                f"{color} {score:.0f}/100",
                delta=status
            )

        with col4:
            # Assessment
            assessment = result.reaction_assessment.value
            st.metric(
                "Assessment",
                assessment.replace("_", " ").title()
            )

        # Expandable sections
        with st.expander("🔍 Why Did It Move?", expanded=True):
            st.markdown(f"**Primary Reason:** {result.primary_reason}")

            if result.drop_reasons:
                st.markdown("**All Factors:**")
                for reason in result.drop_reasons[:5]:
                    st.markdown(f"• {reason}")

        with st.expander("📐 Quantitative Analysis", expanded=False):
            qm = result.quant_metrics

            quant_data = []

            if qm.implied_move_pct and qm.actual_move_pct:
                quant_data.append({
                    'Metric': 'Options Implied Move',
                    'Value': f"±{abs(qm.implied_move_pct):.1f}%",
                    'Comparison': f"Actual: {qm.actual_move_pct:+.1f}%"
                })
                if qm.move_vs_implied_ratio:
                    status = "⚠️ More than expected" if qm.move_vs_implied_ratio > 1.2 else "✅ Less than expected" if qm.move_vs_implied_ratio < 0.8 else "≈ As expected"
                    quant_data.append({
                        'Metric': 'Move vs Implied',
                        'Value': f"{qm.move_vs_implied_ratio:.2f}x",
                        'Comparison': status
                    })

            if qm.rsi_14:
                rsi_status = "🔴 Oversold" if qm.rsi_14 < 30 else "🟢 Overbought" if qm.rsi_14 > 70 else "Neutral"
                quant_data.append({
                    'Metric': 'RSI (14)',
                    'Value': f"{qm.rsi_14:.0f}",
                    'Comparison': rsi_status
                })

            if qm.reaction_percentile:
                quant_data.append({
                    'Metric': 'Reaction Percentile',
                    'Value': f"{qm.reaction_percentile:.0f}th",
                    'Comparison': 'vs historical moves'
                })

            if qm.sector_move_pct is not None:
                quant_data.append({
                    'Metric': 'Sector Move',
                    'Value': f"{qm.sector_move_pct:+.1f}%",
                    'Comparison': f"Stock vs sector: {qm.relative_to_sector:+.1f}%" if qm.relative_to_sector else ""
                })

            if qm.distance_to_52w_low_pct:
                quant_data.append({
                    'Metric': 'Distance to 52w Low',
                    'Value': f"{qm.distance_to_52w_low_pct:.1f}%",
                    'Comparison': "Near low" if qm.distance_to_52w_low_pct < 10 else ""
                })

            if quant_data:
                df = pd.DataFrame(quant_data)
                st.dataframe(df, hide_index=True, use_container_width=True)
            else:
                st.info("Limited quantitative data available")

        with st.expander("🎯 Trading Levels", expanded=False):
            if result.recommendation in ['BUY_DIP', 'STRONG_BUY', 'NIBBLE']:
                level_col1, level_col2, level_col3 = st.columns(3)

                with level_col1:
                    if result.entry_price:
                        st.metric("Entry", f"${result.entry_price:.2f}")

                with level_col2:
                    if result.stop_loss:
                        st.metric("Stop Loss", f"${result.stop_loss:.2f}")

                with level_col3:
                    if result.target_price:
                        st.metric("Target", f"${result.target_price:.2f}")

                if result.risk_reward_ratio:
                    st.markdown(f"**Risk/Reward:** {result.risk_reward_ratio:.1f}:1")

                if result.suggested_position_pct > 0:
                    st.markdown(f"**Suggested Position:** {result.suggested_position_pct:.0f}% of normal size")
            else:
                st.info("No entry recommended at this time")

        with st.expander("📰 Key Headlines", expanded=False):
            if result.key_headlines:
                for headline in result.key_headlines[:5]:
                    st.markdown(f"• {headline}")
            else:
                st.info("No headlines available")

        # Recommendation reason
        st.markdown("---")
        st.markdown(f"**💡 {result.recommendation_reason}**")

    except ImportError as e:
        st.error(f"Reaction analyzer not available: {e}")
        st.info("Install the reaction_analyzer module first")
    except Exception as e:
        st.error(f"Error running analysis: {e}")
        import traceback
        st.code(traceback.format_exc())


def get_earnings_reaction_card(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get a summary card for earnings reaction (for use in other views).

    Args:
        ticker: Stock symbol

    Returns:
        Dict with summary data or None
    """
    try:
        from src.analytics.earnings_intelligence.reaction_analyzer import analyze_post_earnings

        result = analyze_post_earnings(ticker)

        return {
            'ticker': ticker,
            'recommendation': result.recommendation.value,
            'assessment': result.reaction_assessment.value,
            'reaction_pct': result.reaction_pct,
            'oversold_score': result.oversold_score,
            'primary_reason': result.primary_reason,
            'confidence': result.confidence,
            'entry_price': result.entry_price,
            'target_price': result.target_price,
        }

    except Exception:
        return None


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import sys

    st.set_page_config(page_title="Post-Earnings Analysis", layout="wide")

    ticker = st.text_input("Ticker", value="NKE")

    if ticker:
        render_post_earnings_section(ticker.upper())