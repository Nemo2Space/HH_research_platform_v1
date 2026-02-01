"""
Macro Regime Tab - Dashboard Component

Shows current market regime (Risk-On vs Risk-Off) with all indicators.

Author: Alpha Research Platform
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from src.utils.logging import get_logger

logger = get_logger(__name__)


def render_macro_regime_tab():
    """Render the Macro Regime Detection tab."""

    st.markdown("### 🌍 Macro Regime Detection")
    st.caption("Detect Risk-On vs Risk-Off market environment to adjust your trading")

    # Check if module is available
    try:
        from src.analytics.macro_regime import (
            MacroRegimeDetector,
            get_current_regime,
            load_regime_from_cache,
            save_regime_to_cache,
            MarketRegime
        )
        REGIME_AVAILABLE = True
    except ImportError as e:
        st.error(f"Macro regime module not found: {e}")
        st.info("Make sure src/analytics/macro_regime.py exists")
        REGIME_AVAILABLE = False
        return

    # Load/Refresh buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        load_btn = st.button("📥 Load Regime", key="load_regime", type="primary")

    with col2:
        refresh_btn = st.button("🔄 Refresh (Live)", key="refresh_regime")

    # Check cache first
    cached_result = load_regime_from_cache(max_age_hours=4)

    if cached_result and not refresh_btn and not load_btn:
        result = cached_result
        cache_age = (datetime.now() - result.analysis_date).total_seconds() / 3600
        st.caption(f"📦 Cached {cache_age:.1f} hours ago")
    elif load_btn or refresh_btn:
        with st.spinner("Analyzing macro regime..." if refresh_btn else "Loading..."):
            try:
                detector = MacroRegimeDetector()
                result = detector.detect_regime(use_cache=not refresh_btn)
                save_regime_to_cache(result)
                st.session_state.macro_regime_result = result
            except Exception as e:
                st.error(f"Error detecting regime: {e}")
                return
    elif 'macro_regime_result' in st.session_state:
        result = st.session_state.macro_regime_result
    else:
        st.info("👆 Click 'Load Regime' to analyze current market conditions")
        return

    # Display Results
    st.markdown("---")

    # Main Regime Display
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Regime badge
        regime_colors = {
            'STRONG_RISK_ON': ('🟢', '#28a745', 'Strong Risk-On'),
            'RISK_ON': ('🟢', '#5cb85c', 'Risk-On'),
            'NEUTRAL': ('🟡', '#ffc107', 'Neutral'),
            'RISK_OFF': ('🔴', '#dc3545', 'Risk-Off'),
            'STRONG_RISK_OFF': ('🔴', '#c82333', 'Strong Risk-Off'),
        }

        emoji, color, label = regime_colors.get(result.regime.value, ('⚪', '#6c757d', 'Unknown'))

        st.markdown(f"""
        <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: white; margin: 0;">{emoji} {label}</h2>
            <p style="color: white; margin: 5px 0 0 0; font-size: 1.2em;">Score: {result.regime_score}/100</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("Confidence", f"{result.confidence:.0%}")

    with col3:
        st.metric("Growth Adj", f"{result.growth_adjustment:+d}")
        st.metric("Defensive Adj", f"{result.defensive_adjustment:+d}")

    st.markdown("---")

    # Gauge Chart
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result.regime_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Regime Score"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': '#ffcccc'},
                    {'range': [30, 45], 'color': '#ffe0cc'},
                    {'range': [45, 55], 'color': '#ffffcc'},
                    {'range': [55, 70], 'color': '#e0ffcc'},
                    {'range': [70, 100], 'color': '#ccffcc'},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': result.regime_score
                }
            }
        ))

        fig.update_layout(height=300, margin=dict(t=50, b=0, l=30, r=30))
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("#### 📖 Interpretation")
        st.markdown(f"""
        **{label}** means:

        {"🚀 **Favor growth stocks** (Tech, Consumer Cyclical)" if result.regime_score >= 55 else ""}
        {"🛡️ **Favor defensive stocks** (Utilities, Healthcare)" if result.regime_score < 45 else ""}
        {"⚖️ **Mixed environment** - be selective" if 45 <= result.regime_score < 55 else ""}

        **Signal Adjustments:**
        - Growth stocks: **{result.growth_adjustment:+d}** points
        - Defensive stocks: **{result.defensive_adjustment:+d}** points
        """)

    st.markdown("---")

    # Factors
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ✅ Risk-On Factors")
        if result.risk_on_factors:
            for factor in result.risk_on_factors:
                st.markdown(f"• {factor}")
        else:
            st.markdown("*No significant risk-on factors*")

    with col2:
        st.markdown("#### ⚠️ Risk-Off Factors")
        if result.risk_off_factors:
            for factor in result.risk_off_factors:
                st.markdown(f"• {factor}")
        else:
            st.markdown("*No significant risk-off factors*")

    st.markdown("---")

    # Indicator Details
    st.markdown("#### 📊 Indicator Details")

    indicators = []
    for ind_name in ['vix', 'yield_spread', 'stock_vs_bond', 'dollar_index',
                     'sector_leadership', 'market_breadth']:
        ind = getattr(result, ind_name, None)
        if ind:
            signal_color = "🟢" if ind.signal == 'RISK_ON' else "🔴" if ind.signal == 'RISK_OFF' else "🟡"
            indicators.append({
                'Indicator': ind.name,
                'Signal': f"{signal_color} {ind.signal}",
                'Score': ind.score,
                'Weight': ind.weight,
                'Description': ind.description
            })

    if indicators:
        df = pd.DataFrame(indicators)
        st.dataframe(df, width='stretch', hide_index=True)

    # Trading Implications
    st.markdown("---")
    st.markdown("#### 💡 Trading Implications")

    if result.regime_score >= 70:
        st.success("""
        **Strong Risk-On Environment**
        - Favor high-beta, growth stocks (NVDA, TSLA, AMD)
        - Consider reducing defensive positions
        - Good time for momentum strategies
        - Watch for complacency (VIX too low)
        """)
    elif result.regime_score >= 55:
        st.info("""
        **Risk-On Environment**
        - Slight bias toward growth stocks
        - Normal position sizing
        - Stay diversified
        """)
    elif result.regime_score >= 45:
        st.warning("""
        **Neutral Environment**
        - Be selective, no strong bias
        - Focus on stock-specific catalysts
        - Maintain balanced exposure
        """)
    elif result.regime_score >= 30:
        st.warning("""
        **Risk-Off Environment**
        - Favor defensive stocks (JNJ, PG, utilities)
        - Consider reducing growth exposure
        - Tighter stop losses recommended
        """)
    else:
        st.error("""
        **Strong Risk-Off Environment**
        - Heavy defensive bias
        - Consider raising cash
        - Avoid aggressive positions
        - Wait for regime to stabilize
        """)


# Standalone test
if __name__ == "__main__":
    st.set_page_config(page_title="Macro Regime", layout="wide")
    render_macro_regime_tab()