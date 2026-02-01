"""
Macro Event Integration for Trade Ideas

Provides macro headwinds/tailwinds block for trade recommendations.
Integrates with the MacroEventEngine.

Location: dashboard/macro_trade_integration.py
Author: HH Research Platform
"""

import streamlit as st
from typing import List, Dict, Optional

# Import macro engine
try:
    from src.analytics.macro_event_engine import (
        get_macro_engine,
        get_macro_factors,
        get_macro_context,
        refresh_macro_data,
        MacroFactorScores,
    )
    MACRO_ENGINE_AVAILABLE = True
except ImportError:
    MACRO_ENGINE_AVAILABLE = False


def render_macro_headwinds_tailwinds(tickers: List[str], expanded: bool = True):
    """
    Render macro headwinds/tailwinds block for Trade Ideas.

    Args:
        tickers: List of ticker symbols to analyze
        expanded: Whether to start expanded
    """

    if not MACRO_ENGINE_AVAILABLE:
        return

    if not tickers:
        return

    with st.expander("🌍 Macro Headwinds/Tailwinds", expanded=expanded):
        try:
            engine = get_macro_engine()
            exposure = engine.compute_portfolio_exposure(tickers)
            factors = engine.factors

            # Net impact header
            net = exposure.net_portfolio_impact
            if net > 3:
                st.success(f"**Net Macro Impact: +{net:.1f}** - Favorable macro environment")
            elif net < -3:
                st.error(f"**Net Macro Impact: {net:.1f}** - Challenging macro environment")
            else:
                st.info(f"**Net Macro Impact: {net:+.1f}** - Neutral macro environment")

            # Tailwinds and Headwinds columns
            col_tail, col_head = st.columns(2)

            with col_tail:
                st.markdown("**📈 Macro Tailwinds**")
                if exposure.top_tailwinds:
                    for ticker, impact in exposure.top_tailwinds[:5]:
                        # Get sector for context
                        from src.analytics.macro_event_engine import TICKER_SECTOR_MAP
                        sector = TICKER_SECTOR_MAP.get(ticker, 'Unknown')
                        st.success(f"**{ticker}** ({sector}): +{impact:.1f}")
                else:
                    st.caption("No significant tailwinds")

            with col_head:
                st.markdown("**📉 Macro Headwinds**")
                if exposure.top_headwinds:
                    for ticker, impact in exposure.top_headwinds[:5]:
                        from src.analytics.macro_event_engine import TICKER_SECTOR_MAP
                        sector = TICKER_SECTOR_MAP.get(ticker, 'Unknown')
                        st.error(f"**{ticker}** ({sector}): {impact:.1f}")
                else:
                    st.caption("No significant headwinds")

            # Active macro factors
            st.markdown("---")
            st.markdown("**Active Macro Factors:**")

            elevated = factors.get_elevated_factors(threshold=60)
            if elevated:
                factor_cols = st.columns(min(len(elevated), 4))
                for i, (factor, value) in enumerate(elevated[:4]):
                    with factor_cols[i]:
                        factor_display = factor.replace('_', ' ').title()
                        emoji = "🔴" if value >= 70 else "🟡"
                        st.metric(factor_display, f"{emoji} {value}")
            else:
                st.caption("No significantly elevated factors (all < 60)")

            # Refresh
            if st.button("🔄 Refresh Macro", key="refresh_macro_trade"):
                with st.spinner("Fetching..."):
                    try:
                        new_events, _ = refresh_macro_data()
                        st.success(f"Found {new_events} new events")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        except Exception as e:
            st.warning(f"Macro analysis unavailable: {e}")


def get_macro_adjustment_for_ticker(ticker: str) -> Dict:
    """
    Get macro-based score adjustment for a single ticker.
    READ-ONLY: Returns suggestion, doesn't modify scores.

    Returns:
        Dict with:
        - adjustment: suggested score delta (-10 to +10)
        - reason: explanation
        - factors: list of relevant factors
    """

    if not MACRO_ENGINE_AVAILABLE:
        return {'adjustment': 0, 'reason': 'Macro engine unavailable', 'factors': []}

    try:
        engine = get_macro_engine()
        exposure = engine.compute_portfolio_exposure([ticker])

        impact = exposure.ticker_impacts.get(ticker, 0)

        # Cap adjustment at ±10
        adjustment = max(-10, min(10, int(impact)))

        # Get relevant factors
        factors = engine.factors
        elevated = factors.get_elevated_factors(threshold=60)
        factor_list = [f for f, v in elevated]

        # Generate reason
        if adjustment > 3:
            reason = f"Macro tailwind (+{adjustment}): benefits from {', '.join(factor_list[:2]) if factor_list else 'current environment'}"
        elif adjustment < -3:
            reason = f"Macro headwind ({adjustment}): pressured by {', '.join(factor_list[:2]) if factor_list else 'current environment'}"
        else:
            reason = "Neutral macro exposure"

        return {
            'adjustment': adjustment,
            'reason': reason,
            'factors': factor_list,
            'raw_impact': impact,
        }

    except Exception as e:
        return {'adjustment': 0, 'reason': f'Error: {e}', 'factors': []}


def render_ticker_macro_badge(ticker: str):
    """
    Render a compact macro badge for a ticker.
    For use in ticker lists/tables.
    """

    if not MACRO_ENGINE_AVAILABLE:
        return

    try:
        result = get_macro_adjustment_for_ticker(ticker)
        adj = result['adjustment']

        if adj > 3:
            st.caption(f"🌍 +{adj} tailwind")
        elif adj < -3:
            st.caption(f"🌍 {adj} headwind")
        # Don't show if neutral

    except Exception:
        pass


def enhance_trade_pick_with_macro(pick_dict: Dict) -> Dict:
    """
    Enhance a trade pick dictionary with macro data.

    Args:
        pick_dict: Dict with ticker information

    Returns:
        Enhanced dict with macro fields
    """

    ticker = pick_dict.get('ticker') or pick_dict.get('Ticker', '')

    if not ticker or not MACRO_ENGINE_AVAILABLE:
        pick_dict['macro_adjustment'] = 0
        pick_dict['macro_reason'] = ''
        return pick_dict

    try:
        macro = get_macro_adjustment_for_ticker(ticker)
        pick_dict['macro_adjustment'] = macro['adjustment']
        pick_dict['macro_reason'] = macro['reason']
        pick_dict['macro_factors'] = macro['factors']
    except Exception:
        pick_dict['macro_adjustment'] = 0
        pick_dict['macro_reason'] = ''

    return pick_dict


# =============================================================================
# AI CHAT INTEGRATION
# =============================================================================

def get_macro_prompt_context(portfolio_tickers: Optional[List[str]] = None) -> str:
    """
    Get macro context for AI prompt injection.

    This should be added to the AI chat context to give the AI
    awareness of macro conditions.
    """

    if not MACRO_ENGINE_AVAILABLE:
        return ""

    try:
        return get_macro_context(portfolio_tickers)
    except Exception as e:
        return f"[Macro context error: {e}]"


def inject_macro_into_ai_context(existing_context: str, portfolio_tickers: Optional[List[str]] = None) -> str:
    """
    Inject macro context into existing AI context.
    """

    macro_context = get_macro_prompt_context(portfolio_tickers)

    if macro_context:
        return existing_context + "\n\n" + macro_context

    return existing_context


# =============================================================================
# STANDALONE WIDGET FOR MAIN DASHBOARD
# =============================================================================

def render_macro_dashboard_section(portfolio_tickers: Optional[List[str]] = None):
    """
    Full macro dashboard section for main dashboard.
    """

    if not MACRO_ENGINE_AVAILABLE:
        st.info("🌍 Macro Event Engine not available")
        st.caption("Copy `macro_event_engine.py` to `src/analytics/`")
        return

    st.markdown("## 🌍 Macro & Geopolitical")

    # Refresh row
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Refresh", key="macro_main_refresh", type="primary"):
            with st.spinner("Fetching macro news..."):
                try:
                    new_events, factors = refresh_macro_data()
                    st.success(f"Found {new_events} new events")
                except Exception as e:
                    st.error(f"Error: {e}")

    try:
        engine = get_macro_engine()
        factors = engine.compute_factor_scores()

        # =================================================================
        # ALERTS
        # =================================================================
        elevated = factors.get_elevated_factors(threshold=65)
        if elevated:
            st.markdown("### ⚠️ Active Alerts")
            alert_cols = st.columns(min(len(elevated), 4))
            for i, (factor, value) in enumerate(elevated[:4]):
                with alert_cols[i]:
                    factor_display = factor.replace('_', ' ').title()
                    if value >= 75:
                        st.error(f"**{factor_display}**\n\n🔴 {value}")
                    else:
                        st.warning(f"**{factor_display}**\n\n🟡 {value}")

        # =================================================================
        # FACTOR GAUGES
        # =================================================================
        st.markdown("### 📊 Macro Factors")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Geopolitical**")
            st.metric("Oil Supply Shock", factors.oil_supply_shock)
            st.metric("Conflict Risk", factors.conflict_risk)
            st.metric("Sanctions Risk", factors.sanctions_risk)

        with col2:
            st.markdown("**Policy**")
            st.metric("Trade War Risk", factors.trade_war_risk)
            st.metric("Regulation Risk", factors.regulation_risk)
            st.metric("Political Uncertainty", factors.political_uncertainty)

        with col3:
            st.markdown("**Macro Regime**")
            st.metric("Inflation Pressure", factors.inflation_pressure)
            st.metric("Recession Risk", factors.recession_risk)
            st.metric("Risk-Off", factors.risk_off_sentiment)

        # =================================================================
        # PORTFOLIO EXPOSURE (if tickers provided)
        # =================================================================
        if portfolio_tickers:
            st.markdown("### 💼 Portfolio Exposure")

            exposure = engine.compute_portfolio_exposure(portfolio_tickers)

            # Net impact
            net = exposure.net_portfolio_impact
            if net > 2:
                st.success(f"**Net Impact: +{net:.1f}** - Macro tailwind for portfolio")
            elif net < -2:
                st.error(f"**Net Impact: {net:.1f}** - Macro headwind for portfolio")
            else:
                st.info(f"**Net Impact: {net:+.1f}** - Neutral")

            col_tail, col_head = st.columns(2)

            with col_tail:
                st.markdown("**📈 Tailwinds**")
                for ticker, impact in exposure.top_tailwinds[:5]:
                    st.success(f"{ticker}: +{impact:.1f}")

            with col_head:
                st.markdown("**📉 Headwinds**")
                for ticker, impact in exposure.top_headwinds[:5]:
                    st.error(f"{ticker}: {impact:.1f}")

        # =================================================================
        # ACTIVE EVENTS
        # =================================================================
        events = engine.get_active_events()
        if events:
            with st.expander(f"📰 Active Events ({len(events)})", expanded=False):
                for event in sorted(events, key=lambda e: e.confidence, reverse=True)[:8]:
                    st.markdown(f"**{event.title}** [{event.event_type.value}]")
                    st.caption(f"Confidence: {event.confidence}% | Sources: {event.source_count}")
                    st.markdown("---")

    except Exception as e:
        st.error(f"Error loading macro data: {e}")
        import traceback
        st.code(traceback.format_exc())