"""
Earnings Analysis Tab - Dashboard Component

Provides UI for:
- Viewing upcoming earnings calendar
- Analyzing earnings transcripts
- Viewing historical earnings analysis
- Integrating earnings into signals

Author: Alpha Research Platform
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


def render_earnings_tab(positions: list = None, account_summary: dict = None):
    """
    Render the Earnings Analysis tab.

    Args:
        positions: Current portfolio positions
        account_summary: Account summary data
    """
    st.subheader("📊 Earnings Analysis")
    st.caption("Auto-fetch and analyze earnings transcripts with AI")

    # Check if module is available
    try:
        from src.analytics.earnings_analyzer import (
            EarningsTranscriptAnalyzer,
            EarningsCalendarItem,
            get_upcoming_earnings,
            analyze_ticker_earnings,
            run_migration
        )
        EARNINGS_AVAILABLE = True
    except ImportError as e:
        st.error(f"Earnings analyzer module not found: {e}")
        st.info("Make sure src/analytics/earnings_analyzer.py exists")
        EARNINGS_AVAILABLE = False
        return

    # Initialize analyzer
    if 'earnings_analyzer' not in st.session_state:
        st.session_state.earnings_analyzer = EarningsTranscriptAnalyzer()

    analyzer = st.session_state.earnings_analyzer

    # Sub-tabs
    tab1, tab2, tab3 = st.tabs(["📅 Upcoming Earnings", "🔍 Analyze Earnings", "📈 Historical Analysis"])

    # ================================================================
    # TAB 1: Upcoming Earnings Calendar
    # ================================================================
    with tab1:
        st.markdown("### 📅 Upcoming Earnings Calendar")

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            scope = st.radio(
                "Scope",
                options=["Portfolio Only", "Full Universe"],
                key="earnings_scope",
                horizontal=True
            )
            portfolio_only = (scope == "Portfolio Only")

        with col2:
            days_ahead = st.slider("Days Ahead", 7, 60, 14, key="earnings_days")

        with col3:
            if st.button("🔄 Refresh Calendar", key="refresh_earnings_cal"):
                # Clear file cache
                import os
                cache_file = "data/earnings_calendar_cache.json"
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                st.session_state.pop('earnings_calendar', None)

        # Get tickers based on scope
        if portfolio_only and positions:
            tickers = [p.get('symbol', p.get('ticker')) for p in positions if p.get('symbol') or p.get('ticker')]
        else:
            # Get from session state or database
            tickers = st.session_state.get('universe_tickers', None)

        # Check FILE-BASED cache (persists across page refreshes)
        import os
        import json
        from datetime import datetime, timedelta

        cache_file = "data/earnings_calendar_cache.json"
        cache_valid = False
        upcoming = []

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)

                cache_time = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01'))
                cache_scope = cache_data.get('scope')
                cache_days = cache_data.get('days_ahead')

                # Cache valid for 24 hours with same parameters
                if (datetime.now() - cache_time) < timedelta(hours=24):
                    if cache_scope == scope and cache_days == days_ahead:
                        cache_valid = True
                        # Reconstruct EarningsCalendarItem objects
                        upcoming = []
                        for item in cache_data.get('items', []):
                            upcoming.append(EarningsCalendarItem(
                                ticker=item['ticker'],
                                company_name=item['company_name'],
                                earnings_date=date.fromisoformat(item['earnings_date']),
                                time_of_day=item.get('time_of_day', 'TBD'),
                                eps_estimate=item.get('eps_estimate'),
                                days_until=item.get('days_until', 0)
                            ))

                        # Recalculate days_until (may have changed)
                        today = date.today()
                        for item in upcoming:
                            item.days_until = (item.earnings_date - today).days

                        # Remove past earnings
                        upcoming = [e for e in upcoming if e.days_until >= 0]

                        hours_ago = (datetime.now() - cache_time).total_seconds() / 3600
                        st.caption(f"📦 Cached {hours_ago:.1f} hours ago. Click 'Refresh' to update.")
            except Exception as e:
                st.warning(f"Cache read error: {e}")
                cache_valid = False

        # Show Load button if no cache
        if not cache_valid and not upcoming:
            st.info("📅 Earnings calendar not loaded. Click below to fetch (takes ~2 min for full universe).")
            if st.button("📥 Load Earnings Calendar", key="load_earnings_cal", type="primary"):
                with st.spinner(f"Loading earnings calendar for {len(tickers) if tickers else 'all'} tickers..."):
                    try:
                        upcoming = analyzer.get_upcoming_earnings(tickers, days_ahead)

                        # Save to file cache
                        os.makedirs("data", exist_ok=True)
                        cache_data = {
                            'timestamp': datetime.now().isoformat(),
                            'scope': scope,
                            'days_ahead': days_ahead,
                            'items': [
                                {
                                    'ticker': e.ticker,
                                    'company_name': e.company_name,
                                    'earnings_date': e.earnings_date.isoformat(),
                                    'time_of_day': e.time_of_day,
                                    'eps_estimate': e.eps_estimate,
                                    'days_until': e.days_until
                                }
                                for e in upcoming
                            ]
                        }
                        with open(cache_file, 'w') as f:
                            json.dump(cache_data, f)

                        st.success(f"✅ Loaded {len(upcoming)} upcoming earnings")
                    except Exception as e:
                        st.error(f"Error loading calendar: {e}")

        if upcoming:
            # Convert to DataFrame for display
            cal_data = []
            for item in upcoming:
                cal_data.append({
                    'Ticker': item.ticker,
                    'Company': item.company_name[:30] + '...' if len(item.company_name) > 30 else item.company_name,
                    'Date': item.earnings_date,
                    'Days Until': item.days_until,
                    'Time': item.time_of_day
                })

            cal_df = pd.DataFrame(cal_data)

            # Color code by urgency
            def color_days(val):
                if val <= 3:
                    return 'background-color: #ffcccc'  # Red - imminent
                elif val <= 7:
                    return 'background-color: #fff3cd'  # Yellow - soon
                return ''

            styled_df = cal_df.style.applymap(color_days, subset=['Days Until'])
            st.dataframe(styled_df, width='stretch', hide_index=True)

            st.info(f"📊 {len(upcoming)} earnings in the next {days_ahead} days")

            # Quick stats
            col1, col2, col3 = st.columns(3)
            imminent = len([e for e in upcoming if e.days_until <= 3])
            this_week = len([e for e in upcoming if e.days_until <= 7])

            with col1:
                st.metric("⚠️ Next 3 Days", imminent)
            with col2:
                st.metric("📅 This Week", this_week)
            with col3:
                st.metric("📊 Total Upcoming", len(upcoming))
        else:
            st.info("No upcoming earnings found for the selected scope.")

    # ================================================================
    # TAB 2: Analyze Earnings
    # ================================================================
    with tab2:
        st.markdown("### 🔍 Analyze Earnings")

        col1, col2 = st.columns([2, 1])

        with col1:
            ticker_input = st.text_input(
                "Enter Ticker(s)",
                placeholder="AAPL, MSFT, GOOGL",
                key="earnings_ticker_input"
            )

        with col2:
            analyze_scope = st.radio(
                "Or analyze:",
                options=["Custom", "Portfolio", "Universe"],
                key="analyze_scope",
                horizontal=True
            )

        # Options
        col1, col2, col3 = st.columns(3)

        with col1:
            update_signals = st.checkbox(
                "Update signals with earnings",
                value=True,
                key="update_signals_earnings",
                help="Adjust screener scores based on earnings sentiment"
            )

        with col2:
            save_to_db = st.checkbox(
                "Save analysis to database",
                value=True,
                key="save_earnings_db"
            )

        with col3:
            use_ai = st.checkbox(
                "Use AI analysis",
                value=True,
                key="use_ai_earnings",
                help="Use LLM for deeper transcript analysis"
            )

        # Analyze button
        if st.button("🚀 Analyze Earnings", type="primary", key="run_earnings_analysis"):
            # Determine tickers to analyze
            if analyze_scope == "Custom" and ticker_input:
                tickers_to_analyze = [t.strip().upper() for t in ticker_input.split(',')]
            elif analyze_scope == "Portfolio" and positions:
                tickers_to_analyze = [p.get('symbol', p.get('ticker')) for p in positions]
            else:
                # Full universe
                tickers_to_analyze = st.session_state.get('universe_tickers', [])[:50]  # Limit

            if not tickers_to_analyze:
                st.warning("No tickers to analyze")
            else:
                progress = st.progress(0, text="Analyzing earnings...")
                results_container = st.container()

                results = []
                total = len(tickers_to_analyze)

                for i, ticker in enumerate(tickers_to_analyze):
                    try:
                        progress.progress((i + 1) / total, text=f"Analyzing {ticker}...")

                        # Analyze
                        result = analyzer.analyze_earnings(ticker)
                        results.append(result)

                        # Save if requested
                        if save_to_db:
                            analyzer.save_earnings_analysis(result)

                        # Update signals if requested
                        if update_signals and result.score_adjustment != 0:
                            analyzer.update_signal_with_earnings(ticker, result)

                    except Exception as e:
                        logger.error(f"Error analyzing {ticker}: {e}")

                progress.progress(1.0, text="Complete!")

                # Store results
                st.session_state.earnings_results = results

                # Summary
                with results_container:
                    st.success(f"✅ Analyzed {len(results)} tickers")

                    # Show results table
                    if results:
                        results_data = []
                        for r in results:
                            results_data.append({
                                'Ticker': r.ticker,
                                'EPS Surprise': f"{r.eps_surprise_pct:+.1f}%" if r.eps_surprise_pct else "N/A",
                                'Sentiment': r.overall_sentiment,
                                'Score': r.sentiment_score,
                                'Adjustment': f"{r.score_adjustment:+d}",
                                'Guidance': r.guidance_direction,
                                'Reason': r.adjustment_reason[:50] + '...' if len(r.adjustment_reason) > 50 else r.adjustment_reason
                            })

                        results_df = pd.DataFrame(results_data)

                        # Color code sentiment
                        def color_sentiment(val):
                            if 'BULLISH' in val:
                                return 'background-color: #d4edda; color: #155724'
                            elif 'BEARISH' in val:
                                return 'background-color: #f8d7da; color: #721c24'
                            return ''

                        def color_adjustment(val):
                            try:
                                num = int(val.replace('+', ''))
                                if num > 0:
                                    return 'color: green; font-weight: bold'
                                elif num < 0:
                                    return 'color: red; font-weight: bold'
                            except:
                                pass
                            return ''

                        styled = results_df.style.applymap(color_sentiment, subset=['Sentiment'])
                        styled = styled.applymap(color_adjustment, subset=['Adjustment'])

                        st.dataframe(styled, width='stretch', hide_index=True)

        # Show previous results
        if 'earnings_results' in st.session_state and st.session_state.earnings_results:
            st.markdown("---")
            st.markdown("#### Recent Analysis Results")

            for result in st.session_state.earnings_results[:5]:
                with st.expander(f"📊 {result.ticker} - {result.overall_sentiment}", expanded=False):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"""
                        **EPS Analysis:**
                        - Actual: ${result.eps_actual or 'N/A'}
                        - Estimate: ${result.eps_estimate or 'N/A'}
                        - Surprise: {result.eps_surprise_pct or 0:+.1f}%
                        
                        **Guidance:** {result.guidance_direction}
                        
                        {result.guidance_summary}
                        """)

                    with col2:
                        st.markdown(f"""
                        **AI Assessment:**
                        - Sentiment: {result.overall_sentiment}
                        - Score: {result.sentiment_score}/100
                        - Tone: {result.management_tone}
                        - Signal Impact: {result.score_adjustment:+d} points
                        """)

                    if result.key_highlights:
                        st.markdown("**Key Highlights:**")
                        for h in result.key_highlights[:3]:
                            st.markdown(f"- {h}")

                    if result.concerns:
                        st.markdown("**Concerns:**")
                        for c in result.concerns[:3]:
                            st.markdown(f"- ⚠️ {c}")

    # ================================================================
    # TAB 3: Historical Analysis
    # ================================================================
    with tab3:
        st.markdown("### 📈 Historical Earnings Analysis")

        try:
            from src.db.connection import get_engine

            query = """
                SELECT ticker, filing_date, fiscal_period,
                       eps_actual, eps_surprise_pct,
                       overall_sentiment, sentiment_score,
                       guidance_direction, score_adjustment,
                       adjustment_reason
                FROM earnings_analysis
                ORDER BY filing_date DESC
                LIMIT 50
            """

            df = pd.read_sql(query, get_engine())

            if not df.empty:
                # Filters
                col1, col2 = st.columns(2)

                with col1:
                    ticker_filter = st.multiselect(
                        "Filter by Ticker",
                        options=df['ticker'].unique().tolist(),
                        key="hist_ticker_filter"
                    )

                with col2:
                    sentiment_filter = st.multiselect(
                        "Filter by Sentiment",
                        options=['VERY_BULLISH', 'BULLISH', 'NEUTRAL', 'BEARISH', 'VERY_BEARISH'],
                        key="hist_sentiment_filter"
                    )

                # Apply filters
                if ticker_filter:
                    df = df[df['ticker'].isin(ticker_filter)]
                if sentiment_filter:
                    df = df[df['overall_sentiment'].isin(sentiment_filter)]

                # Display
                st.dataframe(df, width='stretch', hide_index=True)

                # Stats
                st.markdown("---")
                st.markdown("#### Summary Statistics")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    avg_surprise = df['eps_surprise_pct'].mean()
                    st.metric("Avg EPS Surprise", f"{avg_surprise:.1f}%" if pd.notna(avg_surprise) else "N/A")

                with col2:
                    bullish = len(df[df['overall_sentiment'].str.contains('BULLISH', na=False)])
                    st.metric("Bullish Reports", bullish)

                with col3:
                    bearish = len(df[df['overall_sentiment'].str.contains('BEARISH', na=False)])
                    st.metric("Bearish Reports", bearish)

                with col4:
                    avg_adj = df['score_adjustment'].mean()
                    st.metric("Avg Score Adjustment", f"{avg_adj:+.1f}" if pd.notna(avg_adj) else "N/A")

            else:
                st.info("No historical earnings analysis found. Run some analyses first!")

        except Exception as e:
            st.error(f"Error loading historical data: {e}")
            st.info("Make sure to run the database migration first.")

            if st.button("🔧 Run Migration", key="run_earnings_migration"):
                try:
                    run_migration()
                    st.success("✅ Migration complete! Refresh the page.")
                except Exception as me:
                    st.error(f"Migration error: {me}")


# Standalone test
if __name__ == "__main__":
    st.set_page_config(page_title="Earnings Analysis", layout="wide")
    render_earnings_tab()