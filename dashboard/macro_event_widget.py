"""
Macro Event Dashboard Widget

Displays macro/geopolitical events and their impact on the portfolio.
Integrates with the MacroEventEngine for deterministic factor scoring.

Location: dashboard/macro_event_widget.py
Author: HH Research Platform
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict

# Import macro engine
try:
    from src.analytics.macro_event_engine import (
        MacroEventEngine,
        MacroFactorScores,
        MacroEvent,
        PortfolioMacroExposure,
        get_macro_engine,
        get_macro_factors,
        get_macro_context,
        refresh_macro_data,
        EventType,
        EventSeverity,
    )
    MACRO_ENGINE_AVAILABLE = True
except ImportError:
    MACRO_ENGINE_AVAILABLE = False


def render_macro_event_widget(
    portfolio_tickers: Optional[List[str]] = None,
    expanded: bool = True,
    show_factors: bool = True,
    show_exposure: bool = True,
    show_events: bool = True,
):
    """
    Render the macro event widget.

    Args:
        portfolio_tickers: List of tickers to analyze exposure
        expanded: Whether expanders start expanded
        show_factors: Show factor gauges
        show_exposure: Show portfolio exposure
        show_events: Show active events list
    """

    if not MACRO_ENGINE_AVAILABLE:
        st.warning("🌍 Macro Event Engine not available")
        st.caption("Make sure `src/analytics/macro_event_engine.py` exists")
        return

    st.markdown("### 🌍 Macro & Geopolitical Intelligence")
    st.caption("Political events • Commodity shocks • Trade tensions • Portfolio impact")

    # Refresh button
    col_refresh, col_status = st.columns([1, 3])
    with col_refresh:
        if st.button("🔄 Refresh", key="macro_refresh", type="primary"):
            with st.spinner("Fetching macro news..."):
                try:
                    new_events, factors = refresh_macro_data()
                    st.session_state.macro_last_refresh = datetime.now()
                    st.success(f"Found {new_events} new events")
                except Exception as e:
                    st.error(f"Refresh failed: {e}")

    with col_status:
        last_refresh = st.session_state.get('macro_last_refresh')
        if last_refresh:
            age = (datetime.now() - last_refresh).seconds // 60
            st.caption(f"Last refresh: {age} min ago")
        else:
            st.caption("Click Refresh to fetch latest macro news")

    # Get engine and data
    try:
        engine = get_macro_engine()
        factors = engine.compute_factor_scores()

        # Get portfolio exposure if tickers provided
        exposure = None
        if portfolio_tickers:
            exposure = engine.compute_portfolio_exposure(portfolio_tickers)

    except Exception as e:
        st.error(f"Error loading macro data: {e}")
        return

    # ==========================================================================
    # ALERTS (Always visible)
    # ==========================================================================
    elevated = factors.get_elevated_factors(threshold=65)
    if elevated:
        st.markdown("#### ⚠️ Active Alerts")
        alert_cols = st.columns(min(len(elevated), 4))
        for i, (factor, value) in enumerate(elevated[:4]):
            with alert_cols[i]:
                factor_display = factor.replace('_', ' ').title()
                if value >= 75:
                    st.error(f"**{factor_display}**\n\n🔴 {value}")
                else:
                    st.warning(f"**{factor_display}**\n\n🟡 {value}")

    # ==========================================================================
    # FACTOR GAUGES
    # ==========================================================================
    if show_factors:
        with st.expander("📊 Macro Factor Scores", expanded=expanded):
            _render_factor_gauges(factors)

    # ==========================================================================
    # PORTFOLIO EXPOSURE
    # ==========================================================================
    if show_exposure and exposure:
        with st.expander("💼 Portfolio Macro Exposure", expanded=expanded):
            _render_portfolio_exposure(exposure)

    # ==========================================================================
    # ACTIVE EVENTS
    # ==========================================================================
    if show_events:
        with st.expander("📰 Active Macro Events", expanded=False):
            _render_active_events(engine)


def _render_factor_gauges(factors: MacroFactorScores):
    """Render factor score gauges."""

    # Group factors by category
    geopolitical = [
        ('Oil Supply Shock', factors.oil_supply_shock),
        ('Conflict Risk', factors.conflict_risk),
        ('Sanctions Risk', factors.sanctions_risk),
        ('Geopolitical Tension', factors.geopolitical_tension),
    ]

    policy = [
        ('Trade War Risk', factors.trade_war_risk),
        ('Regulation Risk', factors.regulation_risk),
        ('Political Uncertainty', factors.political_uncertainty),
    ]

    macro = [
        ('Inflation Pressure', factors.inflation_pressure),
        ('Recession Risk', factors.recession_risk),
        ('Risk-Off Sentiment', factors.risk_off_sentiment),
        ('FX Stress', factors.fx_stress),
    ]

    commodity = [
        ('Energy Disruption', factors.energy_disruption),
        ('Supply Chain Stress', factors.supply_chain_stress),
    ]

    # Render in columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🌍 Geopolitical**")
        for name, value in geopolitical:
            _render_factor_bar(name, value)

        st.markdown("**🏛️ Policy**")
        for name, value in policy:
            _render_factor_bar(name, value)

    with col2:
        st.markdown("**📈 Macro Regime**")
        for name, value in macro:
            _render_factor_bar(name, value)

        st.markdown("**🛢️ Commodity**")
        for name, value in commodity:
            _render_factor_bar(name, value)

    # Legend
    st.caption("Scale: 0-100 (50 = neutral) | 🔴 >70 elevated | 🟡 55-70 watch | 🟢 <55 normal")


def _render_factor_bar(name: str, value: int):
    """Render a single factor as a progress bar with color."""

    # Determine color
    if value >= 70:
        emoji = "🔴"
    elif value >= 55:
        emoji = "🟡"
    else:
        emoji = "🟢"

    col_name, col_bar, col_val = st.columns([2, 3, 1])
    with col_name:
        st.caption(name)
    with col_bar:
        # Progress bar (Streamlit progress is 0-1)
        st.progress(value / 100)
    with col_val:
        st.caption(f"{emoji} {value}")


def _render_portfolio_exposure(exposure: PortfolioMacroExposure):
    """Render portfolio macro exposure."""

    # Net impact
    net = exposure.net_portfolio_impact
    if net > 2:
        st.success(f"**Net Portfolio Impact:** +{net:.1f} (Tailwind)")
    elif net < -2:
        st.error(f"**Net Portfolio Impact:** {net:.1f} (Headwind)")
    else:
        st.info(f"**Net Portfolio Impact:** {net:+.1f} (Neutral)")

    # Tailwinds and Headwinds
    col_tail, col_head = st.columns(2)

    with col_tail:
        st.markdown("**📈 Tailwinds** (benefiting)")
        if exposure.top_tailwinds:
            for ticker, impact in exposure.top_tailwinds[:5]:
                st.success(f"{ticker}: +{impact:.1f}")
        else:
            st.caption("No significant tailwinds")

    with col_head:
        st.markdown("**📉 Headwinds** (pressured)")
        if exposure.top_headwinds:
            for ticker, impact in exposure.top_headwinds[:5]:
                st.error(f"{ticker}: {impact:.1f}")
        else:
            st.caption("No significant headwinds")

    # Sector breakdown
    if exposure.sector_exposures:
        st.markdown("**Sector Impacts:**")

        sector_data = []
        for sector, exp in exposure.sector_exposures.items():
            if sector != 'Unknown':
                sector_data.append({
                    'Sector': sector,
                    'Impact': exp.current_impact,
                    'Tailwinds': len(exp.tailwinds),
                    'Headwinds': len(exp.headwinds),
                })

        if sector_data:
            df = pd.DataFrame(sector_data)
            df = df.sort_values('Impact', ascending=False)

            # Style the dataframe
            st.dataframe(
                df,
                width='stretch',
                hide_index=True,
                column_config={
                    'Impact': st.column_config.NumberColumn(
                        'Impact',
                        format="%.1f",
                    ),
                }
            )


def _render_active_events(engine: MacroEventEngine):
    """Render list of active macro events."""

    events = engine.get_active_events()

    if not events:
        st.info("No active macro events detected. Click Refresh to fetch latest news.")
        return

    st.markdown(f"**{len(events)} Active Events:**")

    # Sort by confidence
    events = sorted(events, key=lambda e: e.confidence, reverse=True)

    for event in events[:10]:
        # Event type icon
        type_icons = {
            'CONFLICT': '⚔️',
            'SANCTIONS': '🚫',
            'COUP_POLITICAL_CRISIS': '🏛️',
            'TARIFF': '📦',
            'OPEC_DECISION': '🛢️',
            'ENERGY_DISRUPTION': '⚡',
            'CENTRAL_BANK': '🏦',
            'INFLATION_DATA': '📊',
            'RECESSION_SIGNAL': '📉',
            'FX_CRISIS': '💱',
            'ELECTION': '🗳️',
            'REGULATION': '📜',
        }
        icon = type_icons.get(event.event_type.value, '📰')

        # Severity color
        if event.severity.value >= 4:
            severity_badge = "🔴"
        elif event.severity.value >= 3:
            severity_badge = "🟡"
        else:
            severity_badge = "🟢"

        # Age
        age_hours = (datetime.now() - event.first_seen).total_seconds() / 3600
        if age_hours < 1:
            age_str = f"{int(age_hours * 60)}m ago"
        elif age_hours < 24:
            age_str = f"{int(age_hours)}h ago"
        else:
            age_str = f"{int(age_hours / 24)}d ago"

        # Render event
        with st.container():
            st.markdown(f"""
            {icon} **{event.title}** {severity_badge}
            
            {event.event_type.value} • {event.source_count} sources • {age_str} • Confidence: {event.confidence}%
            """)

            # Entities
            entities_str = []
            if event.entities.get('countries'):
                entities_str.append(f"🌍 {', '.join(event.entities['countries'][:3])}")
            if event.entities.get('commodities'):
                entities_str.append(f"🛢️ {', '.join(event.entities['commodities'][:2])}")

            if entities_str:
                st.caption(' | '.join(entities_str))

            st.markdown("---")


# =============================================================================
# COMPACT SIDEBAR WIDGET
# =============================================================================

def render_macro_sidebar_widget():
    """Compact macro widget for sidebar."""

    if not MACRO_ENGINE_AVAILABLE:
        st.sidebar.info("Macro engine not available")
        return

    st.sidebar.markdown("### 🌍 Macro Risks")

    try:
        factors = get_macro_factors()

        # Show only elevated factors
        elevated = factors.get_elevated_factors(threshold=60)

        if elevated:
            for factor, value in elevated[:3]:
                factor_display = factor.replace('_', ' ').title()
                emoji = "🔴" if value >= 70 else "🟡"
                st.sidebar.metric(factor_display, f"{emoji} {value}")
        else:
            st.sidebar.success("No elevated macro risks")

        # Event count
        engine = get_macro_engine()
        event_count = len(engine.get_active_events())
        st.sidebar.caption(f"📰 {event_count} active events")

    except Exception as e:
        st.sidebar.warning(f"Macro data unavailable")


# =============================================================================
# TRADE IDEAS INTEGRATION
# =============================================================================

def get_macro_headwinds_tailwinds(tickers: List[str]) -> Dict:
    """
    Get macro headwinds/tailwinds for a list of tickers.
    For integration into Trade Ideas.

    Returns:
        Dict with 'tailwinds', 'headwinds', 'alerts', 'context'
    """

    if not MACRO_ENGINE_AVAILABLE:
        return {
            'tailwinds': [],
            'headwinds': [],
            'alerts': [],
            'context': "Macro engine not available",
        }

    try:
        engine = get_macro_engine()
        exposure = engine.compute_portfolio_exposure(tickers)

        return {
            'tailwinds': [
                {'ticker': t, 'impact': i, 'reason': 'Macro tailwind'}
                for t, i in exposure.top_tailwinds
            ],
            'headwinds': [
                {'ticker': t, 'impact': i, 'reason': 'Macro headwind'}
                for t, i in exposure.top_headwinds
            ],
            'alerts': exposure.alerts,
            'context': engine.get_macro_context_for_ai(tickers),
            'net_impact': exposure.net_portfolio_impact,
        }
    except Exception as e:
        return {
            'tailwinds': [],
            'headwinds': [],
            'alerts': [f"Error: {e}"],
            'context': f"Error loading macro data: {e}",
        }


def render_macro_block_for_ticker(ticker: str):
    """
    Render a compact macro impact block for a single ticker.
    For use in ticker detail views.
    """

    if not MACRO_ENGINE_AVAILABLE:
        return

    try:
        engine = get_macro_engine()
        exposure = engine.compute_portfolio_exposure([ticker])

        impact = exposure.ticker_impacts.get(ticker, 0)

        if abs(impact) < 1:
            st.caption(f"🌍 Macro impact: Neutral ({impact:+.1f})")
        elif impact > 0:
            st.success(f"🌍 Macro tailwind: +{impact:.1f}")
        else:
            st.error(f"🌍 Macro headwind: {impact:.1f}")

        # Show relevant factors
        factors = engine.factors
        elevated = factors.get_elevated_factors(threshold=60)
        if elevated:
            factor_strs = [f.replace('_', ' ') for f, v in elevated[:2]]
            st.caption(f"Active factors: {', '.join(factor_strs)}")

    except Exception:
        pass


# =============================================================================
# AI CHAT INTEGRATION
# =============================================================================

def get_macro_context_for_chat(portfolio_tickers: Optional[List[str]] = None) -> str:
    """
    Get macro context block for AI chat.
    Designed to be appended to the AI prompt context.
    """

    if not MACRO_ENGINE_AVAILABLE:
        return ""

    try:
        return get_macro_context(portfolio_tickers)
    except Exception as e:
        return f"[Macro context unavailable: {e}]"


# =============================================================================
# MAIN - STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    # This would run in Streamlit
    print("Macro Event Widget - Run via Streamlit")
    print("Import and call render_macro_event_widget() in your app")