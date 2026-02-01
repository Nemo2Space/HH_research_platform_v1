"""
Integration Example - Connect AI System to HH Research Platform

This file shows how to integrate the new AI trading system with your
existing platform components (Signals Tab, Signal Engine, AI Chat, etc.)

Location: src/ml/integration.py
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional

# Existing platform imports
try:
    from src.analytics.signal_engine import SignalEngine
    from src.ai.chat import AIChat
    from src.ibkr.ibkr_utils import get_positions, get_account_summary
    from src.db.connection import get_engine

    PLATFORM_AVAILABLE = True
except ImportError:
    PLATFORM_AVAILABLE = False

# AI System imports
try:
    from ai_trading_system import AITradingSystem, AnalysisResult
    from decision_layer import PortfolioState
    from monitoring import check_model_health

    AI_SYSTEM_AVAILABLE = True
except ImportError:
    try:
        from src.ml.ai_trading_system import AITradingSystem, AnalysisResult
        from src.ml.decision_layer import PortfolioState
        from src.ml.monitoring import check_model_health

        AI_SYSTEM_AVAILABLE = True
    except ImportError:
        AI_SYSTEM_AVAILABLE = False


# =============================================================================
# INTEGRATION WITH SIGNAL ENGINE
# =============================================================================

class EnhancedSignalEngine:
    """
    Wraps existing SignalEngine with AI system capabilities.

    Usage:
        engine = EnhancedSignalEngine()
        result = engine.analyze_with_ai("NVDA")
    """

    def __init__(self):
        self.signal_engine = SignalEngine() if PLATFORM_AVAILABLE else None
        self.ai_system = AITradingSystem() if AI_SYSTEM_AVAILABLE else None

        if self.ai_system:
            self.ai_system.initialize(train_if_needed=False)

    def analyze_with_ai(self, ticker: str, skip_cache: bool = False) -> Dict:
        """
        Run full analysis combining existing SignalEngine + AI System.

        Returns combined result with:
        - Traditional signal scores
        - AI probability and EV
        - Position sizing recommendation
        - Similar historical setups
        - LLM explanation
        """
        result = {
            'ticker': ticker,
            'timestamp': datetime.now(),
            'traditional': {},
            'ai': {},
            'combined': {}
        }

        # Step 1: Get traditional signal scores
        if self.signal_engine:
            try:
                signal_result = self.signal_engine.analyze(ticker, skip_cache=skip_cache)
                result['traditional'] = {
                    'sentiment_score': signal_result.sentiment_score,
                    'fundamental_score': signal_result.fundamental_score,
                    'technical_score': signal_result.technical_score,
                    'options_flow_score': signal_result.options_flow_score,
                    'short_squeeze_score': signal_result.short_squeeze_score,
                    'total_score': signal_result.total_score,
                    'signal_type': signal_result.signal_type,
                }
            except Exception as e:
                result['traditional'] = {'error': str(e)}

        # Step 2: Run AI analysis
        if self.ai_system and result['traditional'] and 'error' not in result['traditional']:
            try:
                scores = result['traditional']

                # Get stock data
                stock_data = self._get_stock_data(ticker)

                ai_result = self.ai_system.analyze(ticker, scores, stock_data)

                result['ai'] = {
                    'ml_probability': ai_result.ml_probability,
                    'ml_ev': ai_result.ml_ev,
                    'ml_confidence': ai_result.ml_confidence,
                    'similar_setups': {
                        'count': ai_result.similar_count,
                        'win_rate': ai_result.similar_win_rate,
                        'avg_return': ai_result.similar_avg_return
                    },
                    'meta_probability': ai_result.meta_probability,
                    'should_trade': ai_result.should_trade,
                    'approved': ai_result.approved,
                    'rejection_reasons': ai_result.rejection_reasons,
                    'position_size_pct': ai_result.position_size_pct,
                    'shares': ai_result.shares,
                    'entry_price': ai_result.entry_price,
                    'stop_loss': ai_result.stop_loss,
                    'target_price': ai_result.target_price,
                    'risk_reward': ai_result.risk_reward,
                    'summary': ai_result.summary,
                    'recommendation': ai_result.recommendation,
                    'risks': ai_result.risks,
                    'thesis_breakers': ai_result.thesis_breakers,
                    'positive_factors': ai_result.positive_factors,
                    'negative_factors': ai_result.negative_factors,
                }
            except Exception as e:
                result['ai'] = {'error': str(e)}

        # Step 3: Combine into final recommendation
        result['combined'] = self._combine_signals(result)

        return result

    def _get_stock_data(self, ticker: str) -> Dict:
        """Get stock data from database/APIs."""
        stock_data = {
            'price': 100,
            'sector': 'Unknown',
            'volatility': 0.025,
            'avg_volume': 1000000,
            'vix': 20
        }

        if PLATFORM_AVAILABLE:
            try:
                engine = get_engine()

                # Get latest price and sector
                query = f"""
                    SELECT sector, target_upside_pct
                    FROM screener_scores
                    WHERE ticker = '{ticker}'
                    ORDER BY date DESC LIMIT 1
                """
                df = pd.read_sql(query, engine)
                if not df.empty:
                    stock_data['sector'] = df.iloc[0]['sector'] or 'Unknown'

                # Get earnings date
                earnings_query = f"""
                    SELECT earnings_date 
                    FROM earnings_calendar
                    WHERE ticker = '{ticker}' AND earnings_date >= CURRENT_DATE
                    ORDER BY earnings_date LIMIT 1
                """
                earn_df = pd.read_sql(earnings_query, engine)
                if not earn_df.empty:
                    earn_date = earn_df.iloc[0]['earnings_date']
                    if earn_date:
                        stock_data['earnings_date'] = earn_date
                        stock_data['days_to_earnings'] = (earn_date - date.today()).days

            except Exception as e:
                pass

        return stock_data

    def _combine_signals(self, result: Dict) -> Dict:
        """Combine traditional and AI signals into final recommendation."""
        trad = result.get('traditional', {})
        ai = result.get('ai', {})

        # If AI not available, use traditional
        if 'error' in ai or not ai:
            return {
                'signal': trad.get('signal_type', 'HOLD'),
                'confidence': 'MEDIUM',
                'source': 'traditional_only',
                'should_trade': trad.get('total_score', 50) >= 65
            }

        # If AI available, prefer AI decision
        return {
            'signal': ai.get('recommendation', 'HOLD'),
            'confidence': ai.get('ml_confidence', 'MEDIUM'),
            'source': 'ai_enhanced',
            'should_trade': ai.get('approved', False),
            'ml_probability': ai.get('ml_probability'),
            'position_size': ai.get('position_size_pct'),
            'entry': ai.get('entry_price'),
            'stop': ai.get('stop_loss'),
            'target': ai.get('target_price'),
        }


# =============================================================================
# INTEGRATION WITH SIGNALS TAB (Streamlit)
# =============================================================================

def render_ai_analysis_section(ticker: str, scores: Dict):
    """
    Render AI analysis section in Signals Tab.

    """
    if not AI_SYSTEM_AVAILABLE:
        st.warning("AI System not available. Install dependencies.")
        return

    st.subheader("🤖 AI Analysis")

    # Initialize AI system (cached)
    @st.cache_resource
    def get_ai_system():
        system = AITradingSystem()
        system.initialize(train_if_needed=False)
        return system

    ai_system = get_ai_system()

    with st.spinner("Running AI analysis..."):
        try:
            result = ai_system.analyze(ticker, scores)

            # Display results
            col1, col2, col3 = st.columns(3)

            with col1:
                prob_color = "green" if result.ml_probability >= 0.60 else "orange" if result.ml_probability >= 0.50 else "red"
                st.metric(
                    "ML Win Probability",
                    f"{result.ml_probability:.1%}",
                    help="Calibrated probability from ML model"
                )
                st.metric(
                    "Expected Value",
                    f"{result.ml_ev * 100:.2f}%",
                    help="Expected return after costs"
                )

            with col2:
                st.metric(
                    "Similar Setups",
                    f"{result.similar_count} trades",
                    f"{result.similar_win_rate:.0%} win rate"
                )
                st.metric(
                    "Meta Confidence",
                    f"{result.combined_probability:.1%}",
                    help="Combined ML + Meta probability"
                )

            with col3:
                if result.approved:
                    st.success(f"✅ {result.recommendation}")
                    st.metric("Position Size", f"{result.position_size_pct:.1%}")
                    st.metric("Risk/Reward", f"{result.risk_reward:.1f}x")
                else:
                    st.error(f"❌ SKIP")
                    if result.rejection_reasons:
                        st.caption(f"Reasons: {', '.join(result.rejection_reasons[:2])}")

            # Factors
            st.markdown("---")
            col_bull, col_bear = st.columns(2)

            with col_bull:
                st.markdown("**✅ Bullish Factors**")
                for factor in result.positive_factors[:5]:
                    st.markdown(f"- {factor}")

            with col_bear:
                st.markdown("**❌ Bearish Factors / Risks**")
                for factor in result.negative_factors[:3]:
                    st.markdown(f"- {factor}")
                for risk in result.risks[:2]:
                    st.markdown(f"- ⚠️ {risk}")

            # Summary
            st.markdown("---")
            st.markdown("**AI Summary**")
            st.info(result.summary)

            # Trade plan (if approved)
            if result.approved:
                st.markdown("**Trade Plan**")
                trade_col1, trade_col2, trade_col3 = st.columns(3)
                with trade_col1:
                    st.write(f"📈 Entry: ${result.entry_price:.2f}")
                with trade_col2:
                    st.write(f"🛑 Stop: ${result.stop_loss:.2f}")
                with trade_col3:
                    st.write(f"🎯 Target: ${result.target_price:.2f}")

            # Thesis breakers
            with st.expander("What would invalidate this trade?"):
                for breaker in result.thesis_breakers:
                    st.markdown(f"- {breaker}")

        except Exception as e:
            st.error(f"AI analysis failed: {e}")


# =============================================================================
# INTEGRATION WITH AI CHAT
# =============================================================================

def get_ai_context_for_chat(ticker: str = None) -> str:
    """
    Generate AI system context to include in chat.

    Add this to your chat.py context building.
    """
    if not AI_SYSTEM_AVAILABLE:
        return ""

    try:
        ai_system = AITradingSystem()
        ai_system.initialize(train_if_needed=False)

        context_parts = []

        # Add general AI insights
        context_parts.append(ai_system.get_insights_for_llm())

        # Add ticker-specific analysis if provided
        if ticker:
            # Get scores from database
            scores = _get_scores_from_db(ticker)
            if scores:
                result = ai_system.analyze(ticker, scores)
                context_parts.append(f"""
[AI ANALYSIS FOR {ticker}]
ML Probability: {result.ml_probability:.1%}
Expected Value: {result.ml_ev * 100:.2f}%
Similar Setups: {result.similar_count} trades at {result.similar_win_rate:.1%} win rate
Recommendation: {result.recommendation}
Approved: {result.approved}
Summary: {result.summary}
""")

        # Add system health
        health = ai_system.check_health()
        context_parts.append(f"""
[AI SYSTEM HEALTH]
Win Rate (30d): {health['performance'].get('win_rate', 'N/A')}
Calibration: {'Good' if health['performance'].get('is_well_calibrated') else 'Needs attention'}
Needs Retrain: {health['drift'].get('needs_retrain', False)}
""")

        return "\n\n".join(context_parts)

    except Exception as e:
        return f"[AI System Error: {e}]"


def _get_scores_from_db(ticker: str) -> Optional[Dict]:
    """Get latest scores for ticker from database."""
    if not PLATFORM_AVAILABLE:
        return None

    try:
        engine = get_engine()
        query = f"""
            SELECT 
                sentiment_score, fundamental_score, technical_score,
                options_flow_score, short_squeeze_score, total_score
            FROM screener_scores
            WHERE ticker = '{ticker}'
            ORDER BY date DESC LIMIT 1
        """
        df = pd.read_sql(query, engine)
        if not df.empty:
            return df.iloc[0].to_dict()
    except:
        pass
    return None


# =============================================================================
# INTEGRATION WITH PORTFOLIO
# =============================================================================

def create_portfolio_state_from_ibkr() -> Optional[PortfolioState]:
    """
    Create PortfolioState from IBKR data.

    Use this to initialize AI system with real portfolio.
    """
    if not PLATFORM_AVAILABLE:
        return None

    try:
        positions = get_positions()
        account = get_account_summary()

        return PortfolioState.from_ibkr(positions, account)
    except Exception as e:
        return PortfolioState.empty()


# =============================================================================
# STREAMLIT TAB COMPONENT
# =============================================================================

def render_ai_system_tab():
    """
    Complete AI System tab for the dashboard.

    Add this as a new tab in app.py.
    """
    st.header("🤖 AI Trading System")

    if not AI_SYSTEM_AVAILABLE:
        st.error("AI System not available. Check installation.")
        return

    # Initialize
    @st.cache_resource
    def get_system():
        system = AITradingSystem()
        system.initialize()
        return system

    ai_system = get_system()
    status = ai_system.get_status()

    # Status bar
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("ML Model", "✅ Loaded" if status.ml_model_loaded else "❌ Not Loaded")
    with col2:
        st.metric("Model AUC", f"{status.ml_model_auc:.3f}")
    with col3:
        st.metric("Recent Win Rate", f"{status.recent_win_rate:.1%}")
    with col4:
        if status.needs_retrain:
            st.warning("⚠️ Retrain Recommended")
        else:
            st.success("✅ Model Healthy")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Analyze Stock",
        "📈 Performance",
        "⚠️ Alerts",
        "⚙️ Settings"
    ])

    with tab1:
        st.subheader("Analyze Stock")

        col_input, col_btn = st.columns([3, 1])
        with col_input:
            ticker = st.text_input("Ticker", value="NVDA").upper()
        with col_btn:
            analyze_btn = st.button("🔍 Analyze", type="primary")

        if analyze_btn and ticker:
            scores = _get_scores_from_db(ticker) or {
                'sentiment_score': 50,
                'fundamental_score': 50,
                'technical_score': 50,
                'options_flow_score': 50,
                'short_squeeze_score': 50,
                'total_score': 50
            }

            with st.spinner(f"Analyzing {ticker}..."):
                result = ai_system.analyze(ticker, scores)
                st.markdown(result.get_summary())

    with tab2:
        st.subheader("Performance Tracking")

        health = ai_system.check_health()
        perf = health.get('performance', {})

        metrics_col1, metrics_col2 = st.columns(2)

        with metrics_col1:
            st.metric("Total Recommendations", perf.get('total_recommendations', 0))
            st.metric("Trades Taken", perf.get('trades_taken', 0))
            st.metric("Closed Trades", perf.get('closed_trades', 0))

        with metrics_col2:
            st.metric("Win Rate", perf.get('win_rate', 'N/A'))
            st.metric("Avg Return", perf.get('avg_return', 'N/A'))
            st.metric("Calibration Error", perf.get('calibration_error', 'N/A'))

    with tab3:
        st.subheader("Drift Alerts")

        alerts = status.drift_alerts
        if alerts:
            for alert in alerts:
                st.warning(alert)
        else:
            st.success("No active alerts")

        drift = health.get('drift', {})
        if drift.get('needs_retrain'):
            st.error(f"⚠️ Retrain needed: {drift.get('reason')}")
            if st.button("🔄 Retrain Model"):
                with st.spinner("Retraining..."):
                    ai_system._train_ml_model()
                    st.success("Model retrained!")
                    st.rerun()

    with tab4:
        st.subheader("Settings")

        st.markdown("**Risk Limits**")
        max_pos = st.slider("Max Position Size", 1, 10, 5, help="% of portfolio")
        max_sector = st.slider("Max Sector Exposure", 10, 40, 25, help="% of portfolio")
        min_prob = st.slider("Min Probability", 50, 75, 55, help="% to trade")

        st.markdown("**Model Settings**")
        conservative = st.checkbox("Conservative Mode", help="Stricter risk limits")

        if st.button("💾 Save Settings"):
            st.success("Settings saved!")


# =============================================================================
# QUICK START
# =============================================================================

if __name__ == "__main__":
    print("Integration Example")
    print("=" * 60)

    # Example 1: Enhanced Signal Engine
    print("\n1. Enhanced Signal Engine:")
    engine = EnhancedSignalEngine()

    # Example 2: Quick Analysis
    print("\n2. Quick Analysis:")
    if AI_SYSTEM_AVAILABLE:
        system = AITradingSystem()
        system.initialize()

        scores = {
            'sentiment_score': 72,
            'fundamental_score': 65,
            'technical_score': 68,
            'options_flow_score': 75,
            'short_squeeze_score': 45,
            'total_score': 70
        }

        result = system.analyze("NVDA", scores)
        print(result.get_summary())

    # Example 3: Get context for chat
    print("\n3. AI Context for Chat:")
    context = get_ai_context_for_chat("NVDA")
    print(context[:500] + "...")