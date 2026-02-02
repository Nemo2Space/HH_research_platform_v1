# CRITICAL: Apply nest_asyncio FIRST to allow nested event loops (Streamlit + ib_insync)
import nest_asyncio
nest_asyncio.apply()

# CRITICAL: Start ib_insync's asyncio event loop in background thread
# This allows qualifyContracts, placeOrder etc. to work in threaded environments
from ib_insync import util
util.startLoop()


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Fix path FIRST before importing src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables FIRST
from dotenv import load_dotenv

# AI Settings component (optional - will work without it)
try:
    from src.components.ai_settings import render_ai_sidebar
    AI_SETTINGS_AVAILABLE = True
except ImportError:
    AI_SETTINGS_AVAILABLE = False
    render_ai_sidebar = None
import time

try:
    from src.tabs.ai_assistant_tab_v4 import render_ai_assistant_tab
    AI_ASSISTANT_TAB_AVAILABLE = True
except ImportError:
    AI_ASSISTANT_TAB_AVAILABLE = False

# Performance & Backtest Tab
try:
    from src.tabs.performance_backtest_tab import render_performance_backtest_tab
    PERFORMANCE_TAB_AVAILABLE = True
except ImportError as e:
    print(f"Performance tab not available: {e}")
    PERFORMANCE_TAB_AVAILABLE = False

try:
    from src.tabs.signals_tab import render_signals_tab, render_signals_tab
    SIGNALS_TAB_AVAILABLE = True
except ImportError as e:
    SIGNALS_TAB_AVAILABLE = False

# Deep Dive Tab
try:
    from src.tabs.deep_dive_tab import render_deep_dive_tab
    DEEP_DIVE_TAB_AVAILABLE = True
except ImportError:
    DEEP_DIVE_TAB_AVAILABLE = False

# Portfolio Tab (must import the function, not just set the flag)
import asyncio
import traceback

# Ensure Streamlit thread has an event loop (required by eventkit/ib_insync on Py3.12)
try:
    asyncio.get_running_loop()
except RuntimeError:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

PORTFOLIO_IMPORT_ERROR = None

try:
    # Most correct when you run: streamlit run dashboard/app.py (from project root)
    from dashboard.portfolio_tab import render_portfolio_tab
    PORTFOLIO_AVAILABLE = True
except Exception as e1:
    try:
        # Fallback if your working dir is dashboard/ (less common)
        from portfolio_tab import render_portfolio_tab
        PORTFOLIO_AVAILABLE = True
    except Exception as e2:
        PORTFOLIO_AVAILABLE = False
        render_portfolio_tab = None
        PORTFOLIO_IMPORT_ERROR = (
            "Failed importing Portfolio tab.\n\n"
            f"dashboard.portfolio_tab error:\n{repr(e1)}\n\n"
            f"portfolio_tab error:\n{repr(e2)}\n\n"
            f"Traceback:\n{traceback.format_exc()}"
        )

# Alpha Model Tab
try:
    from src.tabs.alpha_model_tab import render_alpha_model_tab
    ALPHA_MODEL_TAB_AVAILABLE = True
except ImportError as e:
    print(f"Alpha Model tab not available: {e}")
    ALPHA_MODEL_TAB_AVAILABLE = False

from src.db.connection import get_connection

st.set_page_config(
    page_title="HH Research Platform",
    page_icon="🔬",
    layout="wide"
)

# Clear stale ticker selection on fresh app load (prevents auto-running AAPL analysis)
if '_app_initialized' not in st.session_state:
    st.session_state._app_initialized = True
    for key in ['selected_ticker', '_ticker_selected_this_session', 'deep_dive_ticker']:
        if key in st.session_state:
            del st.session_state[key]

# Start alert scheduler

# Start alert scheduler
try:
    from src.alerts.scheduler import start_scheduler
    start_scheduler()
except Exception as e:
    print(f"Scheduler not started: {e}")

try:
    from src.ai.learning import SignalLearner
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False

load_dotenv()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'run_committee_for' not in st.session_state:
    st.session_state.run_committee_for = None
if 'refresh_news_for' not in st.session_state:
    st.session_state.refresh_news_for = None
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = 0
if 'deep_dive_ticker' not in st.session_state:
    st.session_state.deep_dive_ticker = None
if 'current_portfolio_positions' not in st.session_state:
    st.session_state.current_portfolio_positions = []
if 'current_portfolio_summary' not in st.session_state:
    st.session_state.current_portfolio_summary = {}
if 'portfolio_last_updated' not in st.session_state:
    st.session_state.portfolio_last_updated = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 0

# Check URL params for tab
params = st.query_params
if 'tab' in params:
    st.session_state.selected_tab = int(params['tab'])

query_params = st.query_params
default_tab = int(query_params.get("tab", 0))

# ============================================================================
# STREAMLINED NAVIGATION - 6 MAIN PAGES
# ============================================================================
MAIN_PAGES = ["🔬 Research", "💼 Portfolio", "📊 Analytics", "🏦 Bonds", "📈 Performance", "🧠 Alpha Model", "🤖 AI Assistant", "⚙️ System"]

# ============================================================================
# SIDEBAR - Control Panel
# ============================================================================
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    selected_nav = st.radio(
        "Go to:",
        MAIN_PAGES,
        index=st.session_state.get('current_tab', 0),
        key="nav_radio",
        label_visibility="collapsed"
    )
    st.session_state.current_tab = MAIN_PAGES.index(selected_nav)

    st.markdown("---")

    # AI Model Selection (if available)
    if AI_SETTINGS_AVAILABLE and render_ai_sidebar:
        try:
            render_ai_sidebar()
            st.markdown("---")
        except Exception as e:
            st.caption(f"AI settings unavailable: {e}")

    st.title("🎛️ Control Panel")

    # System Status
    st.markdown("### System Status")
    status_placeholder = st.empty()

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT status, last_run, progress FROM system_status ORDER BY id DESC LIMIT 1")
                row = cur.fetchone()
                if row:
                    status, last_run, progress = row
                    if status == 'running':
                        status_placeholder.success(f"🟢 RUNNING - {progress}%")
                    elif status == 'paused':
                        status_placeholder.warning("🟡 PAUSED")
                    else:
                        status_placeholder.info("🔵 IDLE")
                else:
                    status_placeholder.info("🔵 IDLE")
    except:
        status_placeholder.info("🔵 IDLE")

    st.markdown("---")

    # Control Buttons
    st.markdown("### Quick Actions")

    force_fresh_news = st.checkbox(
        "🔄 Force Fresh News",
        value=False,
        help="Bypass 6-hour cache and fetch fresh news from all sources"
    )

    if st.button("📊 Run Full Screener", width='stretch'):
        with st.spinner("Running screener... Please wait"):
            import subprocess
            script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "run_full_screener.py")
            cmd = [sys.executable, script_path]
            if force_fresh_news:
                cmd.append("--fresh-news")
            subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
        st.success("✅ Screener complete!")
        st.cache_data.clear()
        time.sleep(1)
        st.rerun()

    if st.button("📅 Update Dates", width='stretch'):
        with st.spinner("Updating earnings dates..."):
            import subprocess
            script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "populate_dates.py")
            subprocess.run([sys.executable, script_path], cwd=os.path.dirname(os.path.dirname(__file__)))
        st.success("✅ Dates updated!")
        st.cache_data.clear()
        time.sleep(1)
        st.rerun()

    if st.button("🔄 Refresh Data", width='stretch'):
        st.cache_data.clear()
        st.rerun()


# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data(ttl=60)
def load_dashboard_data():
    """Load all data for dashboard."""
    data = {}

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Screener scores - latest per ticker (matching original schema)
            cur.execute("""
                SELECT DISTINCT ON (ticker)
                    ticker, date, sentiment_score, sentiment_weighted,
                    fundamental_score, growth_score, dividend_score,
                    technical_score, gap_score, gap_type,
                    likelihood_score, analyst_positivity, target_upside_pct,
                    insider_signal, institutional_signal,
                    composite_score, total_score, article_count,
                    options_flow_score, short_squeeze_score,
                    options_sentiment, squeeze_risk,
                    created_at
                FROM screener_scores
                ORDER BY ticker, date DESC, created_at DESC
            """)
            columns = ['ticker', 'date', 'sentiment_score', 'sentiment_weighted',
                      'fundamental_score', 'growth_score', 'dividend_score',
                      'technical_score', 'gap_score', 'gap_type',
                      'likelihood_score', 'analyst_positivity', 'target_upside_pct',
                      'insider_signal', 'institutional_signal',
                      'composite_score', 'total_score', 'article_count',
                      'options_flow_score', 'short_squeeze_score',
                      'options_sentiment', 'squeeze_risk',
                      'created_at']
            data['scores'] = pd.DataFrame(cur.fetchall(), columns=columns)

            # Fundamentals (matching original schema)
            cur.execute("""
                SELECT DISTINCT ON (ticker) ticker, sector, market_cap, pe_ratio, forward_pe,
                    pb_ratio, roe, profit_margin, revenue_growth, dividend_yield,
                    earnings_date, ex_dividend_date
                FROM fundamentals
                ORDER BY ticker, date DESC
            """)
            columns = ['ticker', 'sector', 'market_cap', 'pe_ratio', 'forward_pe',
                       'pb_ratio', 'roe', 'profit_margin', 'revenue_growth',
                       'dividend_yield', 'earnings_date', 'ex_dividend_date']
            data['fundamentals'] = pd.DataFrame(cur.fetchall(), columns=columns)

            # Analyst ratings (matching original schema)
            cur.execute("""
                SELECT DISTINCT ON (ticker) ticker, consensus_rating, analyst_buy, 
                       analyst_hold, analyst_sell, analyst_total, analyst_positivity
                FROM analyst_ratings
                ORDER BY ticker, date DESC
            """)
            columns = ['ticker', 'consensus', 'buy_count', 'hold_count', 'sell_count',
                      'total_ratings', 'positive_pct']
            data['analyst'] = pd.DataFrame(cur.fetchall(), columns=columns)

            # Price targets
            cur.execute("""
                SELECT DISTINCT ON (ticker) ticker, current_price, target_high, target_low,
                       target_mean, target_upside_pct, analyst_count
                FROM price_targets
                ORDER BY ticker, date DESC
            """)
            columns = ['ticker', 'current_price', 'target_high', 'target_low', 'target_mean',
                      'target_upside_pct', 'num_analysts']
            data['targets'] = pd.DataFrame(cur.fetchall(), columns=columns)

            # Latest prices
            cur.execute("""
                SELECT DISTINCT ON (ticker) ticker, close, volume, date
                FROM prices
                ORDER BY ticker, date DESC
            """)
            columns = ['ticker', 'price', 'volume', 'price_date']
            data['prices'] = pd.DataFrame(cur.fetchall(), columns=columns)

            # Trading signals
            cur.execute("""
                SELECT DISTINCT ON (ticker) ticker, signal_type, signal_strength, 
                       signal_color, signal_reason, sentiment_score,
                       fundamental_score, gap_score, likelihood_score, created_at
                FROM trading_signals
                ORDER BY ticker, created_at DESC
            """)
            columns = ['ticker', 'signal_type', 'signal_strength', 'signal_color',
                      'signal_reason', 'sentiment_score', 'fundamental_score',
                      'gap_score', 'likelihood_score', 'created_at']
            data['signals'] = pd.DataFrame(cur.fetchall(), columns=columns)

            # Committee decisions
            cur.execute("""
                SELECT ticker, verdict, conviction, expected_alpha_bps,
                       horizon_days, risks, created_at
                FROM committee_decisions
                ORDER BY created_at DESC
            """)
            columns = ['ticker', 'verdict', 'conviction', 'expected_alpha',
                      'horizon_days', 'risks', 'created_at']
            data['committee'] = pd.DataFrame(cur.fetchall(), columns=columns)

            # Agent votes
            cur.execute("""
                SELECT ticker, agent_role, buy_prob, confidence, rationale, created_at
                FROM agent_votes
                ORDER BY ticker, created_at DESC
            """)
            columns = ['ticker', 'agent_name', 'buy_prob', 'confidence', 'reasoning', 'created_at']
            data['agent_votes'] = pd.DataFrame(cur.fetchall(), columns=columns)

            # Insider transactions
            cur.execute("""
                SELECT ticker, insider_name, insider_title, transaction_type,
                       shares, price, value, transaction_date
                FROM insider_transactions
                ORDER BY transaction_date DESC LIMIT 100
            """)
            columns = ['ticker', 'insider_name', 'title', 'transaction_type',
                       'shares', 'price', 'value', 'transaction_date']
            data['insider'] = pd.DataFrame(cur.fetchall(), columns=columns)

            # Market sentiment
            cur.execute("""
                SELECT 
                    AVG(sentiment_score) as avg_sentiment,
                    COUNT(*) as ticker_count,
                    SUM(CASE WHEN sentiment_score >= 60 THEN 1 ELSE 0 END) as bullish_count,
                    SUM(CASE WHEN sentiment_score <= 40 THEN 1 ELSE 0 END) as bearish_count
                FROM screener_scores
                WHERE date = (SELECT MAX(date) FROM screener_scores)
                  AND sentiment_score IS NOT NULL
            """)
            row = cur.fetchone()
            if row and row[0]:
                avg_sent = float(row[0])
                ticker_count = int(row[1] or 0)
                bullish = int(row[2] or 0)
                bearish = int(row[3] or 0)

                if avg_sent >= 65 or (bullish > bearish * 2):
                    sent_class = "Bullish"
                elif avg_sent >= 55 or (bullish > bearish):
                    sent_class = "Slightly Bullish"
                elif avg_sent <= 35 or (bearish > bullish * 2):
                    sent_class = "Bearish"
                elif avg_sent <= 45 or (bearish > bullish):
                    sent_class = "Slightly Bearish"
                else:
                    sent_class = "Neutral"

                data['market_sentiment'] = {
                    'score': int(avg_sent),
                    'class': sent_class,
                    'bullish_count': bullish,
                    'bearish_count': bearish,
                    'ticker_count': ticker_count
                }
            else:
                data['market_sentiment'] = {'score': 50, 'class': 'Neutral'}

            # News counts
            cur.execute("""
                SELECT ticker, COUNT(*) as cnt
                FROM news_articles
                WHERE published_at > NOW() - INTERVAL '7 days'
                GROUP BY ticker
            """)
            data['news_counts'] = pd.DataFrame(cur.fetchall(), columns=['ticker', 'news_count'])

    return data


# Load data
try:
    data = load_dashboard_data()
except Exception as e:
    st.error(f"Database connection error: {e}")
    st.stop()


# ============================================================================
# MAIN CONTENT
# ============================================================================
st.title("🔬 HH Research Platform")

# Merge scores with other data
df = data['scores'].copy()

if not df.empty:
    if not data['fundamentals'].empty:
        df = df.merge(data['fundamentals'], on='ticker', how='left')
    if not data['analyst'].empty:
        df = df.merge(data['analyst'], on='ticker', how='left')
    if not data['targets'].empty:
        df = df.merge(data['targets'], on='ticker', how='left', suffixes=('', '_target'))
    if not data['prices'].empty:
        df = df.merge(data['prices'], on='ticker', how='left')
    if not data['signals'].empty:
        df = df.merge(data['signals'][['ticker', 'signal_type', 'signal_color']], on='ticker', how='left')
    if not data['news_counts'].empty:
        df = df.merge(data['news_counts'], on='ticker', how='left')


# ============================================================================
# TOP METRICS BAR
# ============================================================================
st.markdown("---")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Total Stocks", len(df) if not df.empty else 0)
with col2:
    mkt = data['market_sentiment']
    st.metric("Market Sentiment", f"{mkt['score']} ({mkt['class']})")
with col3:
    avg_sent = df['sentiment_score'].mean() if not df.empty and 'sentiment_score' in df.columns else 0
    st.metric("Avg Sentiment", f"{avg_sent:.1f}" if pd.notna(avg_sent) else "N/A")
with col4:
    avg_total = df['total_score'].mean() if not df.empty and 'total_score' in df.columns else 0
    st.metric("Avg Total Score", f"{avg_total:.1f}" if pd.notna(avg_total) else "N/A")
with col5:
    buy_count = len(df[df['signal_type'].isin(['STRONG BUY', 'BUY', 'WEAK BUY'])]) if not df.empty and 'signal_type' in df.columns else 0
    st.metric("🟢 Buy Signals", buy_count)
with col6:
    sell_count = len(df[df['signal_type'].isin(['STRONG SELL', 'SELL', 'WEAK SELL'])]) if not df.empty and 'signal_type' in df.columns else 0
    st.metric("🔴 Sell Signals", sell_count)

st.markdown("---")

selected_tab_index = st.session_state.current_tab

# Show current page title
st.markdown(f"### {MAIN_PAGES[selected_tab_index]}")

# ============================================================================
# PAGE 1: RESEARCH (Signals + Trade Ideas + Deep Dive + AI Chat)
# ============================================================================
if selected_tab_index == 0:

    # Create sub-tabs within Research
    research_tabs = st.tabs(["📈 Signals Hub", "💡 Trade Ideas", "🔍 Deep Dive", "🤖 AI Chat"])

    # =========================================================================
    # SUB-TAB 1: SIGNALS HUB
    # =========================================================================
    with research_tabs[0]:
        if SIGNALS_TAB_AVAILABLE:
            render_signals_tab()
        else:
            st.info("Falling back to basic signals view...")

            st.subheader("Trading Signals Summary")
            if not data['signals'].empty:
                signals_df = data['signals'].copy()
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### 🟢 BUY Signals")
                    buy_signals = signals_df[signals_df['signal_type'].str.contains('BUY', na=False)]
                    if not buy_signals.empty:
                        for _, row in buy_signals.head(15).iterrows():
                            strength = row.get('signal_strength', 0) or 0
                            st.markdown(f"**{row['ticker']}** - {row['signal_type']} (Strength: {strength})")
                    else:
                        st.info("No buy signals")
                with c2:
                    st.markdown("### 🔴 SELL Signals")
                    sell_signals = signals_df[signals_df['signal_type'].str.contains('SELL', na=False)]
                    if not sell_signals.empty:
                        for _, row in sell_signals.head(15).iterrows():
                            strength = row.get('signal_strength', 0) or 0
                            st.markdown(f"**{row['ticker']}** - {row['signal_type']} (Strength: {strength})")
                    else:
                        st.info("No sell signals")
            else:
                st.info("No trading signals available")

    # =========================================================================
    # SUB-TAB 2: TRADE IDEAS
    # =========================================================================
    with research_tabs[1]:
        st.subheader("💡 AI Trade Ideas")
        st.caption("AI-generated trade recommendations based on all platform data including Options Flow & Squeeze potential")

        try:
            from dashboard.trade_ideas_tab import render_trade_ideas_tab
            positions = st.session_state.get('current_portfolio_positions', [])
            summary = st.session_state.get('current_account_summary', {})
            render_trade_ideas_tab(positions, summary)
        except ImportError as e:
            st.error(f"Trade Ideas module not found: {e}")
            st.info("Make sure dashboard/trade_ideas_tab.py exists")
        except Exception as e:
            st.error(f"Error loading Trade Ideas: {e}")
            import traceback
            st.code(traceback.format_exc())

    # =========================================================================
    # SUB-TAB 3: DEEP DIVE
    # =========================================================================
    with research_tabs[2]:
        if DEEP_DIVE_TAB_AVAILABLE:
            render_deep_dive_tab()
        else:
            st.subheader("🔍 Deep Dive - Committee Analysis")

            # Ticker selector
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                available_tickers = df['ticker'].tolist() if not df.empty else []
                default_idx = 0
                if st.session_state.deep_dive_ticker in available_tickers:
                    default_idx = available_tickers.index(st.session_state.deep_dive_ticker)
                selected_ticker = st.selectbox("Select Ticker", options=available_tickers, index=default_idx, key="dd_ticker")
                st.session_state.deep_dive_ticker = selected_ticker
            with col2:
                if st.button("🚀 Run Committee Analysis", key="run_committee_btn"):
                    st.session_state.run_committee_for = selected_ticker
            with col3:
                if st.button("🔄 Refresh News", key="refresh_news_btn", help="Fetch fresh news & re-analyze sentiment"):
                    st.session_state.refresh_news_for = selected_ticker

            # Handle refresh news
            if 'refresh_news_for' in st.session_state and st.session_state.refresh_news_for:
                ticker_to_refresh = st.session_state.refresh_news_for
                st.session_state.refresh_news_for = None

                with st.spinner(f"Fetching fresh news for {ticker_to_refresh}..."):
                    try:
                        from src.data.news import NewsCollector
                        from src.screener.sentiment import SentimentAnalyzer
                        from datetime import date

                        news_collector = NewsCollector()
                        result = news_collector.collect_and_save(ticker_to_refresh, days_back=7, force_refresh=True)
                        articles = result.get('articles', [])
                        saved_count = result.get('saved', 0)

                        if articles:
                            st.info(f"📰 Collected {len(articles)} articles, saved {saved_count} new")
                            sentiment_analyzer = SentimentAnalyzer()
                            sentiment_result = sentiment_analyzer.analyze_ticker_sentiment(ticker_to_refresh, articles)

                            if sentiment_result:
                                sentiment_score = sentiment_result.get('sentiment_score', 50)
                                with get_connection() as conn:
                                    with conn.cursor() as cur:
                                        cur.execute("""
                                            UPDATE screener_scores 
                                            SET sentiment_score = %s, article_count = %s
                                            WHERE ticker = %s AND date = %s
                                        """, (sentiment_score, len(articles), ticker_to_refresh, date.today()))
                                    conn.commit()
                                st.success(f"✅ {ticker_to_refresh}: Sentiment = {sentiment_score}")
                                st.rerun()
                        else:
                            st.warning(f"⚠️ No articles found for {ticker_to_refresh}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            # Run committee
            if 'run_committee_for' in st.session_state and st.session_state.run_committee_for:
                ticker_to_analyze = st.session_state.run_committee_for
                st.session_state.run_committee_for = None

                with st.spinner(f"Running committee analysis for {ticker_to_analyze}..."):
                    import subprocess
                    script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "run_committee.py")
                    result = subprocess.run(
                        [sys.executable, script_path, "--ticker", ticker_to_analyze],
                        cwd=os.path.dirname(os.path.dirname(__file__)),
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        st.success("✅ Done! Refresh to see results.")
                    else:
                        st.error(f"❌ Error: {result.stderr}")

            if selected_ticker:
                # Show committee decision
                committee_df = data['committee']
                ticker_decision = committee_df[committee_df['ticker'] == selected_ticker]

                if not ticker_decision.empty:
                    decision = ticker_decision.iloc[0]
                    verdict = decision['verdict']
                    conviction = decision['conviction'] or 0
                    alpha = decision['expected_alpha'] or 0
                    horizon = decision['horizon_days'] or 0
                    verdict_color = '#00C853' if verdict == 'BUY' else '#FF1744' if verdict == 'SELL' else '#FFC107'

                    st.markdown(f"""
                    <div style="background-color: {verdict_color}; padding: 20px; border-radius: 10px; color: white; text-align: center;">
                        <h2>COMMITTEE VERDICT: {verdict}</h2>
                        <p>Conviction: {conviction}/100 | Expected Alpha: {alpha} bps | Horizon: {horizon} days</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown("### Agent Votes")
                    agent_votes = data['agent_votes']
                    ticker_votes = agent_votes[agent_votes['ticker'] == selected_ticker]

                    if not ticker_votes.empty:
                        for _, vote in ticker_votes.iterrows():
                            buy_prob = float(vote['buy_prob']) if pd.notna(vote['buy_prob']) else 0.5
                            vote_icon = "🟢" if buy_prob > 0.6 else "🔴" if buy_prob < 0.4 else "🟡"
                            conf = float(vote['confidence']) if pd.notna(vote['confidence']) else 0
                            st.markdown(f"**{vote['agent_name']}** {vote_icon} (Buy: {buy_prob:.0%}, Conf: {conf:.0%})")
                            st.caption(vote['reasoning'][:200] if vote['reasoning'] else '')
                    else:
                        st.info("No agent votes recorded")

                    if decision['risks']:
                        st.markdown("### ⚠️ Risks")
                        st.warning(decision['risks'])
                else:
                    st.info(f"No committee analysis for {selected_ticker}. Click 'Run Committee Analysis' to generate.")

                # Ticker details
                st.markdown("---")
                st.markdown(f"### {selected_ticker} Details")
                ticker_data = df[df['ticker'] == selected_ticker]
                if not ticker_data.empty:
                    row = ticker_data.iloc[0]
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Total Score", row.get('total_score', 'N/A'))
                        st.metric("Sentiment", row.get('sentiment_score', 'N/A'))
                    with c2:
                        st.metric("Fundamental", row.get('fundamental_score', 'N/A'))
                        st.metric("Technical", row.get('technical_score', 'N/A'))
                    with c3:
                        st.metric("Growth", row.get('growth_score', 'N/A'))
                        st.metric("Dividend", row.get('dividend_score', 'N/A'))
                    with c4:
                        st.metric("Price", f"${row.get('price', 0):.2f}" if pd.notna(row.get('price')) else 'N/A')
                        st.metric("Target", f"${row.get('target_mean', 0):.2f}" if pd.notna(row.get('target_mean')) else 'N/A')

                # News section
                st.markdown("---")
                st.markdown(f"### 📰 Recent News for {selected_ticker}")
                try:
                    with get_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute("""
                                SELECT DISTINCT ON (LEFT(headline, 60))
                                       headline, source, published_at, relevance_score, url
                                FROM news_articles
                                WHERE ticker = %s AND published_at > NOW() - INTERVAL '7 days'
                                ORDER BY LEFT(headline, 60), published_at DESC
                                LIMIT 20
                            """, (selected_ticker,))
                            news_rows = cur.fetchall()

                    if news_rows:
                        news_df = pd.DataFrame(news_rows, columns=['Headline', 'Source', 'Published', 'Relevance', 'URL'])
                        news_df['Published'] = pd.to_datetime(news_df['Published']).dt.strftime('%m-%d %H:%M')
                        st.dataframe(news_df[['Headline', 'Source', 'Published', 'Relevance']], width='stretch')
                    else:
                        st.info(f"No recent news for {selected_ticker}")
                except Exception as e:
                    st.error(f"Error loading news: {e}")

    # =========================================================================
    # SUB-TAB 4: AI CHAT
    # =========================================================================
    with research_tabs[3]:
        st.subheader("🤖 AI Trading Assistant")

        try:
            from src.ai.chat import AlphaChat
            from src.broker.ibkr_utils import load_ibkr_data_cached

            # Initialize chat
            if 'alpha_chat' not in st.session_state:
                st.session_state.alpha_chat = AlphaChat()
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            if 'daily_briefing' not in st.session_state:
                st.session_state.daily_briefing = None

            chat = st.session_state.alpha_chat

            # Portfolio loading
            if 'ibkr_data_loaded' not in st.session_state or not st.session_state.ibkr_data_loaded:
                with st.spinner("Loading portfolio data..."):
                    try:
                        positions = st.session_state.get('current_portfolio_positions', [])
                        summary = st.session_state.get('current_portfolio_summary', {})

                        if positions:
                            chat.set_ibkr_data(positions, summary)
                            st.session_state.ibkr_data_loaded = True
                        else:
                            ibkr_data = load_ibkr_data_cached(
                                account_id="", host="127.0.0.1", port=7496, fetch_live_prices=False
                            )
                            if ibkr_data and not ibkr_data.get('error'):
                                positions = ibkr_data.get('positions', [])
                                summary = ibkr_data.get('summary', {})
                                if positions:
                                    from src.broker.yahoo_prices import update_positions_with_yahoo_prices
                                    positions = update_positions_with_yahoo_prices(positions)
                                    total_value = sum(p.get('marketValue', 0) for p in positions)
                                    summary['net_liquidation'] = total_value
                                st.session_state.current_portfolio_positions = positions
                                st.session_state.current_portfolio_summary = summary
                                st.session_state.portfolio_last_updated = datetime.now()
                                chat.set_ibkr_data(positions, summary)
                            st.session_state.ibkr_data_loaded = True
                    except Exception as e:
                        chat.set_ibkr_data([], {})
                        st.session_state.ibkr_data_loaded = True

            if not chat.available:
                st.error("AI Chat not available. Check if LLM server is running.")
            else:
                # Daily Briefing Section
                st.markdown("---")
                col_brief1, col_brief2, col_brief3 = st.columns([2, 2, 1])
                with col_brief1:
                    if st.button("📊 Load Daily Briefing", width='stretch'):
                        with st.spinner("Analyzing portfolio..."):
                            try:
                                ibkr_data = load_ibkr_data_cached(
                                    account_id="", host="127.0.0.1", port=7496, fetch_live_prices=True
                                )
                                if ibkr_data.get('error'):
                                    st.error(f"IBKR Error: {ibkr_data['error']}")
                                else:
                                    briefing = chat.generate_daily_briefing(
                                        ibkr_data.get('positions', []),
                                        ibkr_data.get('summary')
                                    )
                                    st.session_state.daily_briefing = briefing
                            except Exception as e:
                                st.error(f"Error: {e}")
                with col_brief3:
                    if st.button("🗑️ Clear Briefing", key="clear_briefing"):
                        st.session_state.daily_briefing = None
                        st.session_state.chat_history = []
                        chat.clear_history()

                # Display briefing
                if st.session_state.daily_briefing:
                    briefing = st.session_state.daily_briefing
                    metrics = briefing['metrics']

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        score_color = "🟢" if metrics.risk_score in ['A', 'B+'] else "🟡" if metrics.risk_score in ['B', 'C+'] else "🔴"
                        st.metric("Risk Score", f"{score_color} {metrics.risk_score}", metrics.risk_level)
                    with c2:
                        st.metric("Portfolio Value", f"${briefing['portfolio_value']:,.0f}")
                    with c3:
                        st.metric("Positions", briefing['position_count'])
                    with c4:
                        st.metric("Alerts", metrics.total_alerts)

                    if briefing['alerts']:
                        with st.expander(f"🚨 ALERTS ({len(briefing['alerts'])} items)", expanded=True):
                            for alert in briefing['alerts'][:10]:
                                severity_color = "🔴" if alert['severity'] == 'high' else "🟡" if alert['severity'] == 'medium' else "🟢"
                                st.markdown(f"{alert['icon']} {severity_color} **{alert['symbol']}**: {alert['message']}")

                    if briefing['questions']:
                        with st.expander("💭 DAILY QUESTIONS", expanded=True):
                            for i, q in enumerate(briefing['questions'], 1):
                                st.markdown(f"**{i}.** {q}")

                    with st.expander("📊 Risk Details"):
                        cr1, cr2 = st.columns(2)
                        with cr1:
                            st.markdown("**Concentration**")
                            st.markdown(f"- Top 5: {metrics.top_5_concentration}%")
                            st.markdown(f"- Largest: {metrics.largest_position[0]} ({metrics.largest_position[1]}%)")
                            st.markdown(f"- Beta: {metrics.portfolio_beta}")
                        with cr2:
                            st.markdown("**Sector Exposure**")
                            for sector, weight in list(metrics.sector_breakdown.items())[:5]:
                                flag = "⚠️" if weight > 30 else ""
                                st.markdown(f"- {flag}{sector}: {weight}%")

                st.markdown("---")

                # Trade Journal
                with st.expander("📓 Trade Journal", expanded=False):
                    journal_tab1, journal_tab2, journal_tab3 = st.tabs(["📋 Open", "➕ Add", "📊 History"])

                    with journal_tab1:
                        try:
                            from src.db.repository import Repository
                            repo = Repository()
                            open_entries = repo.get_open_journal_entries()
                            if open_entries.empty:
                                st.info("No open journal entries.")
                            else:
                                for _, entry in open_entries.iterrows():
                                    c1, c2, c3 = st.columns([2, 3, 1])
                                    with c1:
                                        st.markdown(f"**{entry['ticker']}** - {entry['action']}")
                                        st.caption(f"Entry: ${entry['entry_price']:.2f}")
                                    with c2:
                                        if entry['thesis']:
                                            st.caption(entry['thesis'][:100])
                                    with c3:
                                        if st.button("🗑️", key=f"del_{entry['id']}"):
                                            repo.delete_journal_entry(entry['id'])
                                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

                    with journal_tab2:
                        st.markdown("##### Add New Entry")
                        jc1, jc2 = st.columns(2)
                        with jc1:
                            j_ticker = st.text_input("Ticker*", key="j_ticker", placeholder="AAPL").upper()
                            j_action = st.selectbox("Action*", ["BUY", "SELL", "ADD", "TRIM"], key="j_action")
                            j_entry_price = st.number_input("Entry Price*", min_value=0.0, step=0.01, key="j_entry_price")
                        with jc2:
                            j_target = st.number_input("Target Price", min_value=0.0, step=0.01, key="j_target")
                            j_stop = st.number_input("Stop Loss", min_value=0.0, step=0.01, key="j_stop")
                            j_conviction = st.slider("Conviction", 1, 10, 5, key="j_conviction")
                        j_thesis = st.text_area("Thesis*", key="j_thesis", placeholder="Why this trade?")

                        if st.button("💾 Save Entry", key="save_journal", width='stretch'):
                            if j_ticker and j_thesis and j_entry_price > 0:
                                try:
                                    from src.db.repository import Repository
                                    repo = Repository()
                                    from datetime import date
                                    entry_id = repo.save_journal_entry({
                                        'ticker': j_ticker, 'action': j_action,
                                        'entry_date': date.today(), 'entry_price': j_entry_price,
                                        'thesis': j_thesis, 'target_price': j_target if j_target > 0 else None,
                                        'stop_loss': j_stop if j_stop > 0 else None,
                                        'conviction': j_conviction
                                    })
                                    st.success(f"✅ Saved! (ID: {entry_id})")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                            else:
                                st.error("Fill required fields")

                    with journal_tab3:
                        try:
                            from src.db.repository import Repository
                            repo = Repository()
                            summary = repo.get_journal_summary()
                            sc1, sc2, sc3, sc4 = st.columns(4)
                            with sc1:
                                st.metric("Open", summary['open_positions'])
                            with sc2:
                                st.metric("Closed", summary['closed_positions'])
                            with sc3:
                                st.metric("Win Rate", f"{summary['win_rate']:.1f}%")
                            with sc4:
                                st.metric("Total P&L", f"${summary['total_pnl']:,.2f}")
                        except Exception as e:
                            st.error(f"Error: {e}")

                st.markdown("---")

                # Performance Tracking
                with st.expander("📈 Performance Tracking", expanded=False):
                    try:
                        from src.ai.performance_tracker import PerformanceTracker

                        pc1, pc2 = st.columns([1, 1])
                        with pc1:
                            if st.button("📸 Capture Snapshot", width='stretch'):
                                with st.spinner("Capturing..."):
                                    ibkr_data = load_ibkr_data_cached(
                                        account_id="", host="127.0.0.1", port=7496, fetch_live_prices=True
                                    )
                                    if not ibkr_data.get('error'):
                                        tracker = PerformanceTracker()
                                        snapshot_id = tracker.capture_snapshot(
                                            ibkr_data.get('positions', []),
                                            ibkr_data.get('summary', {})
                                        )
                                        st.success(f"✅ Snapshot captured! (ID: {snapshot_id})")
                        with pc2:
                            days_back = st.selectbox("Period", [7, 14, 30, 60, 90], index=2, key="perf_days")

                        tracker = PerformanceTracker()
                        chart_data = tracker.get_performance_chart_data(days=days_back)
                        summary = tracker.get_summary()

                        if summary['total_snapshots'] > 0:
                            mc1, mc2, mc3, mc4 = st.columns(4)
                            with mc1:
                                st.metric("Total Return", f"{summary['total_return_pct']:+.2f}%")
                            with mc2:
                                st.metric("vs SPY (Alpha)", f"{summary['alpha']:+.2f}%")
                            with mc3:
                                st.metric("Best Day", f"{summary['best_day']:+.2f}%")
                            with mc4:
                                st.metric("Worst Day", f"{summary['worst_day']:+.2f}%")

                            if not chart_data.empty and len(chart_data) > 1:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=chart_data['snapshot_date'], y=chart_data['portfolio_return'],
                                    mode='lines+markers', name='Portfolio', line=dict(color='#00cc00', width=2)
                                ))
                                fig.add_trace(go.Scatter(
                                    x=chart_data['snapshot_date'], y=chart_data['benchmark_return'],
                                    mode='lines', name='SPY', line=dict(color='#888888', width=1, dash='dash')
                                ))
                                fig.update_layout(title="Portfolio vs SPY", height=300)
                                st.plotly_chart(fig, width='stretch')
                        else:
                            st.info("No snapshots yet. Click 'Capture Snapshot' to start!")
                    except Exception as e:
                        st.error(f"Error: {e}")

                st.markdown("---")

                # Stock Analysis Input
                col_stock1, col_stock2 = st.columns([3, 1])
                with col_stock1:
                    ticker_input = st.text_input("📈 Analyze Stock", key="ticker_ctx", placeholder="e.g., AAPL")
                with col_stock2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    analyze_btn = st.button("🔍 Analyze", width='stretch', key="analyze_btn")

                # Chat controls
                col_title, col_sync, col_clear = st.columns([3, 1, 1])
                with col_title:
                    st.markdown("#### 💬 Chat")
                with col_sync:
                    if st.button("🔄 Sync", key="sync_portfolio", help="Sync portfolio data"):
                        positions = st.session_state.get('current_portfolio_positions', [])
                        summary = st.session_state.get('current_portfolio_summary', {})
                        if positions:
                            chat.set_ibkr_data(positions, summary)
                            st.success("✅ Synced!")
                            st.rerun()
                        else:
                            st.warning("No portfolio loaded")
                with col_clear:
                    if st.button("🗑️ Clear Chat", key="clear_chat"):
                        st.session_state.chat_history = []
                        if 'uploaded_files_chat' in st.session_state:
                            st.session_state.uploaded_files_chat = []
                        chat.clear_history()
                        st.rerun()

                # Portfolio sync status
                last_updated = st.session_state.get('portfolio_last_updated')
                if last_updated:
                    age = (datetime.now() - last_updated).seconds // 60
                    pos_count = len(st.session_state.get('current_portfolio_positions', []))
                    st.caption(f"📊 Portfolio: {pos_count} positions | {age}m ago")

                # Chat history
                for msg in st.session_state.chat_history:
                    role = "user" if msg["role"] == "user" else "assistant"
                    with st.chat_message(role):
                        st.write(msg["content"])
                        if msg.get("files"):
                            for f in msg["files"]:
                                st.caption(f"📎 {f['name']}")

                # File upload
                if 'uploaded_files_chat' not in st.session_state:
                    st.session_state.uploaded_files_chat = []
                if 'show_file_upload' not in st.session_state:
                    st.session_state.show_file_upload = False

                if st.session_state.uploaded_files_chat:
                    chips = " ".join([f"📎 {f['name'][:20]}" for f in st.session_state.uploaded_files_chat])
                    fc1, fc2 = st.columns([6, 1])
                    with fc1:
                        st.caption(chips)
                    with fc2:
                        if st.button("❌", key="clear_files"):
                            st.session_state.uploaded_files_chat = []
                            st.rerun()

                if not st.session_state.show_file_upload:
                    _, col_plus = st.columns([20, 1])
                    with col_plus:
                        if st.button("📎", key="toggle_upload", help="Attach file"):
                            st.session_state.show_file_upload = True
                            st.rerun()
                else:
                    st.markdown("##### 📎 Attach File")
                    uploaded_file = st.file_uploader(
                        "Select file", type=['csv', 'txt', 'json', 'md', 'py', 'pdf', 'xlsx'],
                        key="inline_file_upload", label_visibility="collapsed"
                    )
                    if uploaded_file:
                        try:
                            file_content = ""
                            if uploaded_file.name.endswith('.pdf'):
                                import PyPDF2, io
                                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                                for page in pdf_reader.pages[:10]:
                                    file_content += page.extract_text() + "\n"
                                file_content = file_content[:15000]
                            elif uploaded_file.name.endswith('.xlsx'):
                                df_file = pd.read_excel(uploaded_file)
                                file_content = df_file.to_string()[:15000]
                            else:
                                file_content = uploaded_file.read().decode('utf-8', errors='ignore')[:20000]

                            if file_content:
                                st.session_state.uploaded_files_chat.append({
                                    'name': uploaded_file.name, 'content': file_content
                                })
                                st.session_state.show_file_upload = False
                                st.success(f"✅ Attached: {uploaded_file.name}")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

                    if st.button("Cancel", key="cancel_upload"):
                        st.session_state.show_file_upload = False
                        st.rerun()

                # Chat input
                prompt = st.chat_input("Ask about stocks, portfolio, or analyze files...")

                if analyze_btn and ticker_input:
                    prompt = f"Give me a complete analysis of {ticker_input.upper()}. Include signal, scores, institutional data, and recommendation."

                if prompt:
                    user_display = prompt
                    user_files = []

                    if st.session_state.uploaded_files_chat:
                        file_context = "\n\n[ATTACHED FILES]\n"
                        for f in st.session_state.uploaded_files_chat:
                            file_context += f"\n--- {f['name']} ---\n```\n{f['content']}\n```\n"
                            user_files.append({'name': f['name']})
                        prompt = file_context + "\n[USER QUESTION]\n" + prompt

                    st.chat_message("user").write(user_display)
                    if user_files:
                        for f in user_files:
                            st.caption(f"📎 {f['name']}")

                    st.session_state.chat_history.append({
                        "role": "user", "content": user_display, "files": user_files
                    })

                    ticker = ticker_input.strip().upper() if ticker_input else None

                    if st.session_state.daily_briefing and "portfolio" in prompt.lower():
                        prompt = f"[PORTFOLIO CONTEXT]\n{st.session_state.daily_briefing['summary_text']}\n\n{prompt}"

                    with st.chat_message("assistant"):
                        response_placeholder = st.empty()
                        full_response = ""

                        try:
                            for chunk in chat.chat_stream(prompt, ticker=ticker):
                                full_response += chunk
                                display_text = full_response
                                if '<think>' in display_text and '</think>' not in display_text:
                                    display_text = "🤔 Thinking..."
                                elif '</think>' in display_text:
                                    display_text = display_text.split('</think>')[-1].strip()
                                response_placeholder.markdown(display_text + "▌")
                        except Exception as e:
                            full_response = f"Error: {str(e)}"

                        final_response = full_response
                        if '</think>' in final_response:
                            final_response = final_response.split('</think>')[-1].strip()
                        response_placeholder.markdown(final_response)

                    st.session_state.chat_history.append({"role": "assistant", "content": final_response})
                    st.session_state.uploaded_files_chat = []

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())


# ============================================================================
# PAGE 2: PORTFOLIO
# ============================================================================
if selected_tab_index == 1:
    if PORTFOLIO_AVAILABLE:
        render_portfolio_tab()
    else:
        st.subheader("💼 Portfolio Management")
        st.error("Portfolio module not available.")
        if PORTFOLIO_IMPORT_ERROR:
            st.code(PORTFOLIO_IMPORT_ERROR)

# ============================================================================
# PAGE 3: ANALYTICS
# ============================================================================
if selected_tab_index == 2:
    try:
        from dashboard.analytics_tab import render_analytics_tab
        positions = st.session_state.get('current_portfolio_positions', [])
        summary = st.session_state.get('current_portfolio_summary', {})
        render_analytics_tab(positions, summary)
    except ImportError as e:
        st.subheader("📊 Analytics & Optimization")
        st.error(f"Analytics module not found: {e}")
        st.info("Make sure dashboard/analytics_tab.py and src/analytics/ folder exist.")
    except Exception as e:
        st.error(f"Error loading analytics: {e}")
        import traceback
        st.code(traceback.format_exc())


# ============================================================================
# PAGE 4: BONDS
# ============================================================================
if selected_tab_index == 3:
    try:
        from dashboard.bond_signals_dashboard import render_bond_trading_tab
        render_bond_trading_tab()
    except ImportError as e:
        st.error(f"Bond Trading module not found: {e}")
        st.info("Make sure dashboard/bond_signals_dashboard.py exists")
    except Exception as e:
        st.error(f"Error loading Bond Trading: {e}")
        import traceback
        st.code(traceback.format_exc())


# ============================================================================
# PAGE 5: PERFORMANCE
# ============================================================================
if selected_tab_index == 4:
    if PERFORMANCE_TAB_AVAILABLE:
        render_performance_backtest_tab()
    else:
        st.error("Performance & Backtest tab not available.")
        st.info("Make sure src/tabs/performance_backtest_tab.py exists.")


# ============================================================================
# PAGE 6: ALPHA MODEL
# ============================================================================
if selected_tab_index == 5:
    if ALPHA_MODEL_TAB_AVAILABLE:
        render_alpha_model_tab()
    else:
        st.subheader("🧠 Multi-Factor Alpha Model")
        st.error("Alpha Model tab module not loaded.")
        st.info("Make sure `src/tabs/alpha_model_tab.py` and `src/ml/multi_factor_alpha.py` exist.")

        st.markdown("### 📋 Quick Setup")
        st.markdown("""
        1. Copy `multi_factor_alpha.py` to `src/ml/multi_factor_alpha.py`
        2. Copy `alpha_model_tab.py` to `src/tabs/alpha_model_tab.py`
        3. Train the model using the CLI
        """)

        st.code("""
# Train the model:
python scripts/alpha_model_cli.py train --min-date 2025-01-01 --folds 3

# Generate predictions:
python scripts/alpha_model_cli.py predict --top 15

# Single stock analysis:
python scripts/alpha_model_cli.py predict --ticker NVDA
        """, language="bash")

        st.markdown("### 📊 What is the Alpha Model?")
        st.markdown("""
        The Multi-Factor Alpha Model learns optimal factor weights from your historical data:
        
        - **Learns from data** - Weights are optimized based on what actually predicted returns
        - **Regime-aware** - Different weights for bull/bear/volatile markets
        - **Confidence intervals** - Not just "BUY" but "expected +2.3% with 65% confidence"
        - **Factor importance** - See which factors actually predict returns
        """)


# ============================================================================
# PAGE 7: AI ASSISTANT
# ============================================================================
if selected_tab_index == 6:
    if AI_ASSISTANT_TAB_AVAILABLE:
        render_ai_assistant_tab()
    else:
        st.subheader("🤖 AI Research Assistant")
        st.error("AI Assistant tab not available.")
        st.info("Make sure `src/tabs/ai_assistant_tab.py` and `src/ai/ai_research_assistant.py` exist.")


# ============================================================================
# PAGE 8: SYSTEM
# ============================================================================
if selected_tab_index == 7:
    st.subheader("⚙️ System Status")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Database Tables")
        with get_connection() as conn:
            with conn.cursor() as cur:
                tables = [
                    'screener_scores', 'trading_signals', 'fundamentals',
                    'analyst_ratings', 'price_targets', 'prices',
                    'news_articles', 'committee_decisions', 'agent_votes',
                    'insider_transactions', 'sec_filings', 'sec_chunks',
                    'signal_snapshots', 'options_flow'
                ]
                for table in tables:
                    try:
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cur.fetchone()[0]
                        st.metric(table, f"{count:,} rows")
                    except Exception as e:
                        st.metric(table, "N/A")

    with col2:
        st.markdown("### Quick Commands")
        st.code("# Run full screener with LLM sentiment\npython scripts/run_full_screener.py", language="bash")
        st.code("# Run committee analysis\npython scripts/run_committee.py --ticker AAPL", language="bash")
        st.code("# Generate signals\npython -c \"from src.core.signal_engine import SignalEngine; e = SignalEngine(); print(e.generate_signal('AAPL'))\"", language="bash")
        st.code("# Start dashboard\nstreamlit run dashboard/app.py", language="bash")

        st.markdown("### System Info")
        st.info("📡 IBKR: Port 7496 (TWS) / 4001 (Gateway)")
        st.info("🤖 Local LLM: Qwen3-32B @ localhost:8080")

        st.markdown("### Data Sources")
        st.markdown("""
        - **Options**: IBKR (real-time) → Yahoo (fallback)
        - **Prices**: Yahoo Finance
        - **Insider**: SEC EDGAR Form 4
        - **13F Holdings**: SEC EDGAR 13F
        - **News**: Multiple APIs
        """)

        st.markdown("### Last Data Update")
        if not df.empty and 'created_at' in df.columns:
            last_update = df['created_at'].max()
            if pd.notna(last_update):
                st.success(f"📅 {last_update}")

        st.markdown("### Institutional Signal Leaderboard")
        if st.button("🏆 Show Leaderboard", key="show_leaderboard"):
            try:
                from src.analytics.signal_performance import InstitutionalSignalTracker
                tracker = InstitutionalSignalTracker()
                best = tracker.get_best_institutional_signals(days_back=90, top_n=10)
                if not best.empty:
                    st.dataframe(best, width='stretch')
                else:
                    st.info("No institutional signal data yet. Run signal generation to populate.")
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
# AI Settings sidebar is now integrated in the sidebar section above
# This line was moved there
st.markdown("---")