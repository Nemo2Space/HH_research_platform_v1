"""
Economic Calendar Streamlit Component
======================================
UI component for displaying economic calendar with:
- HIGH impact events only
- Zurich (local) time
- Refresh button
- AI Analysis button

Usage in your dashboard:
    from src.components.economic_calendar_ui import render_economic_calendar
    render_economic_calendar()

Author: Alpha Research Platform
"""

import streamlit as st
from datetime import datetime, date
from typing import List, Dict, Optional
import time

# Import our modules
try:
    from src.analytics.economic_calendar import (
        EconomicCalendarFetcher,
        get_economic_calendar,
        refresh_calendar,
        EconomicEvent
    )
    CALENDAR_AVAILABLE = True
except ImportError:
    CALENDAR_AVAILABLE = False

try:
    from src.analytics.economic_news_analyzer import (
        get_news_analyzer,
        analyze_all_economic_events,
        get_market_quick_summary,
        EventAnalysis,
        MarketAssessment
    )
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False


def render_economic_calendar():
    """
    Render the economic calendar component in Streamlit.
    Call this from your main dashboard.
    """
    st.markdown("### 📅 Economic Calendar")

    if not CALENDAR_AVAILABLE:
        st.error("Economic calendar module not available. Check imports.")
        return

    # Initialize session state
    if 'calendar_data' not in st.session_state:
        st.session_state.calendar_data = None
    if 'calendar_last_refresh' not in st.session_state:
        st.session_state.calendar_last_refresh = None
    if 'market_analysis' not in st.session_state:
        st.session_state.market_analysis = None
    if 'analyzing' not in st.session_state:
        st.session_state.analyzing = False

    # Buttons row
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("🔄 Refresh", key="refresh_calendar", use_container_width=True):
            with st.spinner("Fetching latest data..."):
                st.session_state.calendar_data = refresh_calendar()
                st.session_state.calendar_last_refresh = datetime.now()
                st.session_state.market_analysis = None  # Clear old analysis
            st.rerun()

    with col2:
        analyze_disabled = not ANALYZER_AVAILABLE
        if st.button("🤖 Analyze", key="analyze_news", use_container_width=True, disabled=analyze_disabled):
            st.session_state.analyzing = True
            st.rerun()

    with col3:
        if st.session_state.calendar_last_refresh:
            st.caption(f"Last updated: {st.session_state.calendar_last_refresh.strftime('%H:%M:%S')}")

    # Load calendar if not loaded
    if st.session_state.calendar_data is None:
        with st.spinner("Loading economic calendar..."):
            st.session_state.calendar_data = get_economic_calendar()
            st.session_state.calendar_last_refresh = datetime.now()

    calendar = st.session_state.calendar_data
    today = date.today()

    # Quick summary line
    if calendar.today_events:
        events_for_analysis = [
            {
                'name': e.event_name,
                'time': e.event_time,
                'actual': e.actual,
                'forecast': e.forecast,
                'previous': e.previous
            }
            for e in calendar.today_events
        ]

        if ANALYZER_AVAILABLE:
            quick_summary = get_market_quick_summary(events_for_analysis)
            st.markdown(f"**Today's Signal:** {quick_summary}")

    # Display today's HIGH impact events
    st.markdown(f"**📌 Today's High Impact Events** ({today.strftime('%A, %b %d')})")
    st.caption("Times shown in Zurich (local)")

    if calendar.today_events:
        # Group by time
        events_by_time = {}
        for event in calendar.today_events:
            time_key = event.event_time or "All Day"
            if time_key not in events_by_time:
                events_by_time[time_key] = []
            events_by_time[time_key].append(event)

        # Display events
        for time_slot, events in sorted(events_by_time.items()):
            st.markdown(f"**⏰ {time_slot}**")

            for event in events:
                # Determine status
                if event.actual:
                    # Data released - show with values
                    surprise = _calculate_simple_surprise(event)
                    emoji = "🟢" if surprise == "BEAT" else "🔴" if surprise == "MISS" else "🟡"

                    values = []
                    if event.actual:
                        values.append(f"**Actual: {event.actual}**")
                    if event.forecast:
                        values.append(f"Exp: {event.forecast}")
                    if event.previous:
                        values.append(f"Prev: {event.previous}")

                    value_str = " | ".join(values)
                    st.markdown(f"{emoji} {event.event_name}")
                    st.caption(f"   {value_str}")
                else:
                    # Pending
                    values = []
                    if event.forecast:
                        values.append(f"Exp: {event.forecast}")
                    if event.previous:
                        values.append(f"Prev: {event.previous}")

                    value_str = " | ".join(values) if values else ""
                    st.markdown(f"⏳ {event.event_name}")
                    if value_str:
                        st.caption(f"   {value_str}")
    else:
        st.info("No high-impact US events scheduled today")

    # Key dates
    st.markdown("---")
    st.markdown("**📆 Key Upcoming Dates**")

    dates_col1, dates_col2 = st.columns(2)

    with dates_col1:
        if calendar.last_fed_meeting:
            days_ago = (today - calendar.last_fed_meeting).days
            st.caption(f"Last Fed: {calendar.last_fed_meeting.strftime('%b %d')} ({days_ago}d ago)")
        if calendar.next_fed_meeting:
            st.caption(f"Next Fed: {calendar.next_fed_meeting.strftime('%b %d')} ({calendar.days_to_fed}d)")

    with dates_col2:
        if calendar.next_cpi:
            st.caption(f"Next CPI: {calendar.next_cpi.strftime('%b %d')} ({calendar.days_to_cpi}d)")
        if calendar.next_jobs:
            st.caption(f"Next NFP: {calendar.next_jobs.strftime('%b %d')} ({calendar.days_to_jobs}d)")

    # AI Analysis section (if triggered)
    if st.session_state.analyzing:
        _render_ai_analysis(calendar)
        st.session_state.analyzing = False

    # Show existing analysis if available
    elif st.session_state.market_analysis:
        _render_market_assessment(st.session_state.market_analysis)


def _calculate_simple_surprise(event: EconomicEvent) -> str:
    """Calculate simple surprise indicator."""
    if not event.actual or not event.forecast:
        return "PENDING"

    try:
        actual = event.actual.replace('%', '').replace('K', '000').replace('M', '000000')
        forecast = event.forecast.replace('%', '').replace('K', '000').replace('M', '000000')

        actual_val = float(actual)
        forecast_val = float(forecast)

        if forecast_val == 0:
            return "N/A"

        pct_diff = ((actual_val - forecast_val) / abs(forecast_val)) * 100

        if abs(pct_diff) < 1:
            return "IN-LINE"
        elif pct_diff > 0:
            return "BEAT"
        else:
            return "MISS"
    except:
        return "N/A"


def _render_ai_analysis(calendar):
    """Render AI analysis with loading indicator."""
    if not ANALYZER_AVAILABLE:
        st.error("AI Analyzer not available")
        return

    # Convert events to format for analyzer
    events_for_analysis = [
        {
            'name': e.event_name,
            'time': e.event_time,
            'actual': e.actual,
            'forecast': e.forecast,
            'previous': e.previous
        }
        for e in calendar.today_events
    ]

    # Check if any data released
    released = [e for e in events_for_analysis if e.get('actual')]

    if not released:
        st.warning("⏳ No data released yet. Analysis will be available after releases.")
        return

    st.markdown("---")
    st.markdown("### 🤖 AI Market Analysis")

    with st.spinner("🔍 Analyzing economic data with AI... (this may take 30-60 seconds)"):
        start_time = time.time()

        try:
            analysis = analyze_all_economic_events(events_for_analysis)
            elapsed = time.time() - start_time

            st.session_state.market_analysis = analysis
            st.caption(f"Analysis completed in {elapsed:.1f}s")

            _render_market_assessment(analysis)

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")


def _render_market_assessment(analysis: MarketAssessment):
    """Render the market assessment results."""
    st.markdown("---")
    st.markdown("### 🤖 AI Market Analysis")

    # Overall signal with color
    signal_colors = {
        "BULLISH": "🟢",
        "BEARISH": "🔴",
        "MIXED": "🟡",
        "PENDING": "⏳"
    }
    signal_emoji = signal_colors.get(analysis.overall_signal, "⚪")

    st.markdown(f"## {signal_emoji} Overall: **{analysis.overall_signal}**")

    # Summary
    if analysis.summary:
        st.markdown("**📈 Summary:**")
        st.markdown(analysis.summary)

    # Two column layout for takeaways and strategy
    col1, col2 = st.columns(2)

    with col1:
        if analysis.key_takeaways:
            st.markdown("**💡 Key Takeaways:**")
            for takeaway in analysis.key_takeaways:
                st.markdown(f"• {takeaway}")

    with col2:
        if analysis.trading_strategy:
            st.markdown("**🎯 Trading Strategy:**")
            for strategy in analysis.trading_strategy:
                st.markdown(f"• {strategy}")

    # Risks
    if analysis.risks:
        st.markdown("**⚠️ Risks to Watch:**")
        for risk in analysis.risks:
            st.markdown(f"• {risk}")

    # Sector impact
    if analysis.sector_impact:
        st.markdown("**🏭 Sector Impact:**")
        sectors_positive = [s for s, impact in analysis.sector_impact.items() if impact == "+"]
        sectors_negative = [s for s, impact in analysis.sector_impact.items() if impact == "-"]

        sec_col1, sec_col2 = st.columns(2)
        with sec_col1:
            if sectors_positive:
                st.markdown("🟢 **Positive:**")
                st.markdown(", ".join(sectors_positive))
        with sec_col2:
            if sectors_negative:
                st.markdown("🔴 **Negative:**")
                st.markdown(", ".join(sectors_negative))

    # Expandable full analysis
    with st.expander("📝 Full AI Analysis"):
        st.text(analysis.full_analysis)

    st.caption(f"Analysis timestamp: {analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")


def render_economic_calendar_compact():
    """
    Render a compact version of the economic calendar.
    Good for sidebars or smaller spaces.
    """
    if not CALENDAR_AVAILABLE:
        st.caption("Calendar unavailable")
        return

    calendar = get_economic_calendar()
    today = date.today()

    st.markdown("**📅 Economic Calendar**")

    # Quick signal
    if calendar.today_events and ANALYZER_AVAILABLE:
        events_data = [
            {'name': e.event_name, 'actual': e.actual, 'forecast': e.forecast}
            for e in calendar.today_events
        ]
        quick = get_market_quick_summary(events_data)
        st.caption(quick)

    # Count events
    released = len([e for e in calendar.today_events if e.actual])
    pending = len([e for e in calendar.today_events if not e.actual])

    if calendar.today_events:
        st.caption(f"Today: {released} released, {pending} pending")
    else:
        st.caption("No high-impact events today")

    # Key dates
    if calendar.next_fed_meeting:
        st.caption(f"Next Fed: {calendar.days_to_fed}d")


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    st.set_page_config(page_title="Economic Calendar Test", layout="wide")
    st.title("Economic Calendar Test")
    render_economic_calendar()