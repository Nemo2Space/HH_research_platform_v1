"""
Institutional Signals Display for Signals Tab

Renders all institutional signals in the deep dive view:
- GEX/Gamma Analysis
- Dark Pool Flow
- Cross-Asset Signals
- Sentiment NLP
- Earnings Whisper
- Insider Transactions (Form 4)
- 13F Institutional Holdings

Location: src/ml/institutional_signals_display.py
"""

import streamlit as st
from datetime import datetime
from typing import Optional, Dict, Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ============================================================================
# IMPORTS - Phase 2 & 3 modules
# ============================================================================

# Phase 2: GEX/Gamma
try:
    from src.analytics.gex_analysis import analyze_gex, GEXRegime
    GEX_AVAILABLE = True
except ImportError:
    GEX_AVAILABLE = False

# Phase 2: Dark Pool
try:
    from src.analytics.dark_pool import analyze_dark_pool, DarkPoolSentiment
    DARK_POOL_AVAILABLE = True
except ImportError:
    DARK_POOL_AVAILABLE = False

# Phase 2: Cross-Asset
try:
    from src.analytics.cross_asset import get_cross_asset_signals, CrossAssetSignal
    CROSS_ASSET_AVAILABLE = True
except ImportError:
    CROSS_ASSET_AVAILABLE = False

# Phase 3: Sentiment NLP
try:
    from src.analytics.sentiment_nlp import (
        analyze_news_sentiment,
        is_llm_available,
        SentimentLevel
    )
    SENTIMENT_NLP_AVAILABLE = True
except ImportError:
    SENTIMENT_NLP_AVAILABLE = False

# Phase 3: Earnings Whisper
try:
    from src.analytics.earnings_whisper import (
        get_earnings_whisper,
        EarningsPrediction,
        WhisperSignal
    )
    EARNINGS_WHISPER_AVAILABLE = True
except ImportError:
    EARNINGS_WHISPER_AVAILABLE = False

# Form 4 Insider Tracker
try:
    from src.analytics.insider_tracker import get_insider_signal, InsiderSignal
    INSIDER_TRACKER_AVAILABLE = True
except ImportError:
    INSIDER_TRACKER_AVAILABLE = False

# 13F Institutional Tracker
try:
    from src.analytics.institutional_13f_tracker import (
        get_institutional_ownership,
        TickerInstitutionalOwnership
    )
    INSTITUTIONAL_13F_AVAILABLE = True
except ImportError:
    INSTITUTIONAL_13F_AVAILABLE = False


def get_modules_status() -> Dict[str, bool]:
    """Return availability status of all institutional modules."""
    return {
        'GEX/Gamma': GEX_AVAILABLE,
        'Dark Pool': DARK_POOL_AVAILABLE,
        'Cross-Asset': CROSS_ASSET_AVAILABLE,
        'Sentiment NLP': SENTIMENT_NLP_AVAILABLE,
        'Earnings Whisper': EARNINGS_WHISPER_AVAILABLE,
        'Insider Tracker': INSIDER_TRACKER_AVAILABLE,
        '13F Institutional': INSTITUTIONAL_13F_AVAILABLE,
    }


def render_institutional_signals(ticker: str, current_price: float = None,
                                  sector: str = None, days_to_earnings: int = 999):
    """
    Render all Phase 2 & Phase 3 institutional signals for a ticker.

    Args:
        ticker: Stock symbol
        current_price: Current stock price (optional, will fetch if not provided)
        sector: Stock sector (optional)
        days_to_earnings: Days until earnings (for whisper relevance)
    """
    st.markdown("#### 🏛️ Institutional Signals")

    # Check if any modules available
    modules = get_modules_status()
    if not any(modules.values()):
        st.caption("No institutional signal modules available")
        return

    # Get current price if not provided
    if current_price is None or current_price <= 0:
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            current_price = stock.history(period='1d')['Close'].iloc[-1]
        except Exception:
            current_price = 100  # Fallback

    # Create tabs for different signal types
    tabs = st.tabs(["📊 GEX/Gamma", "🏦 Dark Pool", "🌐 Cross-Asset", "🧠 Sentiment", "🎯 Whisper", "👔 Insider", "🏛️ 13F"])

    # Tab 1: GEX/Gamma
    with tabs[0]:
        _render_gex_signals(ticker, current_price)

    # Tab 2: Dark Pool
    with tabs[1]:
        _render_dark_pool_signals(ticker)

    # Tab 3: Cross-Asset
    with tabs[2]:
        _render_cross_asset_signals(sector)

    # Tab 4: Sentiment NLP
    with tabs[3]:
        _render_sentiment_signals(ticker)

    # Tab 5: Earnings Whisper
    with tabs[4]:
        _render_whisper_signals(ticker, days_to_earnings)

    # Tab 6: Insider Transactions
    with tabs[5]:
        _render_insider_signals(ticker)

    # Tab 7: 13F Institutional
    with tabs[6]:
        _render_13f_signals(ticker)


def _render_gex_signals(ticker: str, current_price: float):
    """Render GEX/Gamma analysis."""
    if not GEX_AVAILABLE:
        st.caption("GEX module not installed")
        return

    try:
        with st.spinner("Analyzing gamma exposure..."):
            gex = analyze_gex(ticker, current_price)

        # Main metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            regime_color = "green" if gex.gex_regime in [GEXRegime.VERY_POSITIVE, GEXRegime.POSITIVE] else \
                          "red" if gex.gex_regime in [GEXRegime.VERY_NEGATIVE, GEXRegime.NEGATIVE] else "orange"
            st.metric("GEX Regime", gex.gex_regime.value)

        with col2:
            signal_color = "green" if gex.signal == "BULLISH" else "red" if gex.signal == "BEARISH" else "orange"
            st.metric("Signal", f":{signal_color}[{gex.signal}]")

        with col3:
            st.metric("Strength", f"{gex.signal_strength}/100")

        # Key levels
        if gex.max_gamma_strike > 0:
            st.markdown("**Key Levels:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"📌 Max Gamma: ${gex.max_gamma_strike:.0f}")
            with col2:
                if gex.call_wall > 0:
                    st.caption(f"🟢 Call Wall: ${gex.call_wall:.0f}")
            with col3:
                if gex.put_wall > 0:
                    st.caption(f"🔴 Put Wall: ${gex.put_wall:.0f}")

            # Pin proximity
            dist_to_pin = abs(current_price - gex.max_gamma_strike) / current_price * 100
            if dist_to_pin < 2:
                st.warning(f"⚠️ Price is {dist_to_pin:.1f}% from max gamma - likely pinned")

        # Warnings
        if gex.warnings:
            for w in gex.warnings:
                st.caption(f"⚠️ {w}")

    except Exception as e:
        st.caption(f"GEX analysis error: {e}")


def _render_dark_pool_signals(ticker: str):
    """Render Dark Pool flow analysis."""
    if not DARK_POOL_AVAILABLE:
        st.caption("Dark Pool module not installed")
        return

    try:
        with st.spinner("Analyzing dark pool flow..."):
            dp = analyze_dark_pool(ticker)

        # Main metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            sent_color = "green" if "ACCUMULATION" in dp.sentiment.value else \
                        "red" if "DISTRIBUTION" in dp.sentiment.value else "orange"
            st.metric("Sentiment", dp.sentiment.value)

        with col2:
            st.metric("DP Score", f"{dp.sentiment_score}/100")

        with col3:
            bias_emoji = "🟢" if dp.institutional_bias == "BUYING" else \
                        "🔴" if dp.institutional_bias == "SELLING" else "⚪"
            st.metric("Inst. Bias", f"{bias_emoji} {dp.institutional_bias}")

        # Block trades
        if dp.block_buy_volume > 0 or dp.block_sell_volume > 0:
            st.markdown("**Block Activity:**")
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"🟢 Block Buys: {dp.block_buy_volume:,}")
            with col2:
                st.caption(f"🔴 Block Sells: {dp.block_sell_volume:,}")

        # Warnings
        if dp.warnings:
            for w in dp.warnings:
                st.caption(f"⚠️ {w}")

    except Exception as e:
        st.caption(f"Dark pool analysis error: {e}")


def _render_cross_asset_signals(sector: str = None):
    """Render Cross-Asset signals."""
    if not CROSS_ASSET_AVAILABLE:
        st.caption("Cross-Asset module not installed")
        return

    try:
        # Cache cross-asset (same for all tickers)
        cache_key = 'cross_asset_cache'

        if cache_key not in st.session_state:
            with st.spinner("Analyzing cross-asset signals..."):
                st.session_state[cache_key] = get_cross_asset_signals()

        xa = st.session_state[cache_key]

        # Main metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            signal_color = "green" if xa.primary_signal == CrossAssetSignal.RISK_ON else \
                          "red" if xa.primary_signal == CrossAssetSignal.RISK_OFF else "orange"
            st.metric("Signal", xa.primary_signal.value)

        with col2:
            risk_emoji = "🟢" if xa.risk_appetite == "RISK_ON" else "🔴"
            st.metric("Risk Appetite", f"{risk_emoji} {xa.risk_appetite}")

        with col3:
            st.metric("Cycle Phase", xa.cycle_phase.value)

        # Favored sectors
        if xa.favored_sectors:
            st.markdown("**Favored Sectors:**")
            sectors_str = ", ".join(xa.favored_sectors[:5])
            st.caption(sectors_str)

            # Highlight if current stock's sector is favored
            if sector and sector in xa.favored_sectors:
                st.success(f"✅ {sector} is currently favored by cross-asset signals")

        # Avoid sectors
        if xa.avoid_sectors:
            st.markdown("**Sectors to Avoid:**")
            avoid_str = ", ".join(xa.avoid_sectors[:5])
            st.caption(avoid_str)

            if sector and sector in xa.avoid_sectors:
                st.warning(f"⚠️ {sector} is currently unfavored by cross-asset signals")

    except Exception as e:
        st.caption(f"Cross-asset analysis error: {e}")


def _render_sentiment_signals(ticker: str):
    """Render Sentiment NLP analysis."""
    if not SENTIMENT_NLP_AVAILABLE:
        st.caption("Sentiment NLP module not installed")
        return

    # Check LLM availability
    llm_status = "🟢 LLM Active" if is_llm_available() else "🟡 Fallback Mode"
    st.caption(llm_status)

    try:
        # Get recent news from database
        news_headlines = _get_recent_news(ticker)

        if not news_headlines:
            st.caption("No recent news to analyze")
            return

        with st.spinner(f"Analyzing {len(news_headlines)} headlines..."):
            sentiment = analyze_news_sentiment(ticker, news_headlines)

        # Main metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            sent_color = "green" if sentiment.sentiment_score >= 60 else \
                        "red" if sentiment.sentiment_score <= 40 else "orange"
            st.metric("Sentiment", sentiment.sentiment_level.value)

        with col2:
            st.metric("Score", f"{sentiment.sentiment_score}/100")

        with col3:
            st.metric("Tone", sentiment.management_tone.value)

        # Key points
        if sentiment.key_positives:
            st.markdown("**Positives:**")
            for p in sentiment.key_positives[:3]:
                st.caption(f"✅ {p}")

        if sentiment.key_negatives:
            st.markdown("**Negatives:**")
            for n in sentiment.key_negatives[:3]:
                st.caption(f"❌ {n}")

        # Summary
        if sentiment.summary:
            st.markdown("**Summary:**")
            st.caption(sentiment.summary)

    except Exception as e:
        st.caption(f"Sentiment analysis error: {e}")


def _render_whisper_signals(ticker: str, days_to_earnings: int):
    """Render Earnings Whisper analysis."""
    if not EARNINGS_WHISPER_AVAILABLE:
        st.caption("Earnings Whisper module not installed")
        return

    try:
        with st.spinner("Analyzing earnings whisper..."):
            whisper = get_earnings_whisper(ticker)

        # Check if relevant
        if whisper.days_to_earnings > 60:
            st.caption(f"No earnings within 60 days")
            return

        # Earnings date info
        st.caption(f"📅 Earnings: {whisper.earnings_date} ({whisper.days_to_earnings} days)")

        # Main metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            pred_color = "green" if "BEAT" in whisper.prediction.value else \
                        "red" if "MISS" in whisper.prediction.value else "orange"
            st.metric("Prediction", whisper.prediction.value)

        with col2:
            prob_color = "green" if whisper.beat_probability >= 60 else \
                        "red" if whisper.beat_probability <= 40 else "orange"
            st.metric("Beat Prob", f"{whisper.beat_probability:.0f}%")

        with col3:
            st.metric("Exp. Surprise", f"{whisper.expected_surprise_pct:+.1f}%")

        # Signal
        signal_color = "green" if "BULLISH" in whisper.signal.value else \
                      "red" if "BEARISH" in whisper.signal.value else "orange"
        st.markdown(f"**Signal:** :{signal_color}[{whisper.signal.value}] (strength: {whisper.signal_strength})")

        # Component scores
        st.markdown("**Component Scores:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            rev_emoji = "🟢" if whisper.revision_score >= 60 else "🔴" if whisper.revision_score <= 40 else "🟡"
            st.caption(f"{rev_emoji} Revisions: {whisper.revision_score}")
        with col2:
            opt_emoji = "🟢" if whisper.options_score >= 60 else "🔴" if whisper.options_score <= 40 else "🟡"
            st.caption(f"{opt_emoji} Options: {whisper.options_score}")
        with col3:
            hist_emoji = "🟢" if whisper.historical_score >= 60 else "🔴" if whisper.historical_score <= 40 else "🟡"
            st.caption(f"{hist_emoji} History: {whisper.historical_score}")

        # Warnings
        if whisper.warnings:
            for w in whisper.warnings:
                st.warning(f"⚠️ {w}")

    except Exception as e:
        st.caption(f"Earnings whisper error: {e}")


def _render_insider_signals(ticker: str):
    """Render Form 4 Insider Transaction analysis."""
    if not INSIDER_TRACKER_AVAILABLE:
        st.caption("Insider Tracker module not installed")
        return

    try:
        with st.spinner("Analyzing insider transactions..."):
            signal = get_insider_signal(ticker)

        # Main metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            signal_color = "green" if "BUY" in signal.signal else \
                          "red" if "SELL" in signal.signal else "orange"
            st.metric("Signal", signal.signal)

        with col2:
            st.metric("Strength", f"{signal.signal_strength}/100")

        with col3:
            net_emoji = "🟢" if signal.net_value > 0 else "🔴" if signal.net_value < 0 else "⚪"
            st.metric("Net Value", f"{net_emoji} ${signal.net_value:,.0f}")

        # Key insider activity
        st.markdown("**Key Activity (90 days):**")
        col1, col2 = st.columns(2)

        with col1:
            st.caption(f"🟢 Buys: {signal.total_buys} (${signal.buy_value:,.0f})")
            if signal.ceo_bought:
                st.success("✅ CEO Bought")
            if signal.cfo_bought:
                st.success("✅ CFO Bought")

        with col2:
            st.caption(f"🔴 Sells: {signal.total_sells} (${signal.sell_value:,.0f})")
            if signal.ceo_sold:
                st.warning("⚠️ CEO Sold")
            if signal.cfo_sold:
                st.warning("⚠️ CFO Sold")

        # Cluster detection
        if signal.cluster_buying:
            st.success("🔥 Cluster Buying: 3+ insiders bought in 30 days")
        if signal.cluster_selling:
            st.error("🚨 Cluster Selling: 3+ insiders sold in 30 days")

        # Unique buyers/sellers
        st.caption(f"Unique Buyers: {signal.unique_buyers} | Unique Sellers: {signal.unique_sellers}")

        # Recent transactions
        if signal.recent_transactions:
            with st.expander("Recent Transactions", expanded=False):
                for t in signal.recent_transactions[:5]:
                    emoji = "🟢" if t.transaction_type.value == "PURCHASE" else "🔴"
                    st.caption(
                        f"{emoji} {t.transaction_date}: {t.insider_name} ({t.insider_role.value}) "
                        f"- {t.shares:,} shares @ ${t.price_per_share:.2f}"
                    )

    except Exception as e:
        st.caption(f"Insider analysis error: {e}")


def _render_13f_signals(ticker: str):
    """Render 13F Institutional Holdings analysis."""
    if not INSTITUTIONAL_13F_AVAILABLE:
        st.caption("13F Institutional module not installed")
        return

    st.caption("⚠️ Note: 13F data has 45+ day lag")

    try:
        with st.spinner("Checking institutional ownership..."):
            ownership = get_institutional_ownership(ticker)

        if ownership.num_institutions == 0:
            st.caption("No notable institutional holders found")
            return

        # Main metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            signal_color = "green" if "BUY" in ownership.signal else \
                          "red" if "SELL" in ownership.signal else "orange"
            st.metric("Signal", ownership.signal)

        with col2:
            st.metric("Notable Holders", ownership.num_institutions)

        with col3:
            st.metric("Total Shares", f"{ownership.total_shares:,}")

        # Buffett indicator
        if ownership.buffett_owns:
            if ownership.buffett_added:
                st.success("🎯 Warren Buffett OWNS & ADDED this quarter!")
            else:
                st.info("🎯 Warren Buffett owns this stock")

        # Activist involvement
        if ownership.activist_involved:
            st.warning("📢 Activist investor involved")

        # Recent changes
        if ownership.new_buyers:
            st.markdown("**New Buyers This Quarter:**")
            st.caption(", ".join(ownership.new_buyers))

        if ownership.added_by:
            st.markdown("**Added Position:**")
            st.caption(", ".join(ownership.added_by))

        if ownership.reduced_by:
            st.markdown("**Reduced Position:**")
            st.caption(", ".join(ownership.reduced_by))

        if ownership.sold_by:
            st.markdown("**Sold Entirely:**")
            st.caption(", ".join(ownership.sold_by))

        # Notable holders details
        if ownership.notable_holders:
            with st.expander("Notable Holder Details", expanded=False):
                for holder in ownership.notable_holders[:5]:
                    change_emoji = "🟢" if holder['change'] in ['NEW', 'ADDED'] else \
                                  "🔴" if holder['change'] in ['REDUCED', 'SOLD'] else "⚪"
                    st.caption(
                        f"{change_emoji} **{holder['name']}** ({holder['manager']}) - "
                        f"{holder['shares']:,} shares | {holder['change']} ({holder['change_pct']:+.1f}%)"
                    )

    except Exception as e:
        st.caption(f"13F analysis error: {e}")


def _get_recent_news(ticker: str) -> list:
    """Get recent news headlines from database."""
    try:
        from src.db.connection import get_engine
        import pandas as pd

        engine = get_engine()

        # Try news_articles table first
        try:
            query = """
                SELECT headline FROM news_articles
                WHERE ticker = %s
                AND published_at >= NOW() - INTERVAL '7 days'
                ORDER BY published_at DESC
                LIMIT 10
            """
            df = pd.read_sql(query, engine, params=(ticker,))
            if not df.empty:
                return df['headline'].tolist()
        except Exception:
            pass

        # Try news table as fallback
        try:
            query = """
                SELECT title as headline FROM news
                WHERE ticker = %s
                AND date >= CURRENT_DATE - INTERVAL '7 days'
                ORDER BY date DESC
                LIMIT 10
            """
            df = pd.read_sql(query, engine, params=(ticker,))
            if not df.empty:
                return df['headline'].tolist()
        except Exception:
            pass

        return []

    except Exception:
        return []


# ============================================================================
# COMPACT VERSION FOR TABLE VIEW
# ============================================================================

def get_institutional_summary(ticker: str, current_price: float = None) -> Dict[str, Any]:
    """
    Get a compact summary of institutional signals for table display.

    Returns dict with key metrics only.
    """
    result = {
        'gex_signal': None,
        'dark_pool_bias': None,
        'cross_asset_risk': None,
        'nlp_score': None,
        'whisper_prob': None,
    }

    try:
        if GEX_AVAILABLE and current_price:
            gex = analyze_gex(ticker, current_price)
            result['gex_signal'] = gex.signal
    except Exception:
        pass

    try:
        if DARK_POOL_AVAILABLE:
            dp = analyze_dark_pool(ticker)
            result['dark_pool_bias'] = dp.institutional_bias
    except Exception:
        pass

    try:
        if CROSS_ASSET_AVAILABLE:
            if 'cross_asset_cache' in st.session_state:
                xa = st.session_state['cross_asset_cache']
            else:
                xa = get_cross_asset_signals()
            result['cross_asset_risk'] = xa.risk_appetite
    except Exception:
        pass

    try:
        if EARNINGS_WHISPER_AVAILABLE:
            whisper = get_earnings_whisper(ticker)
            if whisper.days_to_earnings <= 30:
                result['whisper_prob'] = whisper.beat_probability
    except Exception:
        pass

    return result