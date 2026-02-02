"""
AI Analysis Integration for Signals Tab

Renders AI analysis section in the signals deep dive.
Auto-populates RAG memory from historical data on first use.

Location: src/ml/signals_tab_ai.py
"""

import streamlit as st

# AI Trading System import
try:
    from src.ml.ai_trading_system import AITradingSystem
    from src.ml.rag_memory import RAGMemoryStore
    AI_SYSTEM_AVAILABLE = True
except ImportError:
    AI_SYSTEM_AVAILABLE = False


def _render_ai_analysis(signal):
    """Render AI analysis section for a signal."""

    if not AI_SYSTEM_AVAILABLE:
        return

    st.markdown("#### 🤖 AI Analysis")

    # Get or create AI system (cached in session state)
    if 'ai_system' not in st.session_state:
        try:
            system = AITradingSystem()
            system.initialize(train_if_needed=False)

            # Auto-populate RAG memory from historical data
            if system.rag_memory and len(system.rag_memory._cards) == 0:
                try:
                    count = system.rag_memory.load_from_db()
                    if count > 0:
                        st.session_state.rag_populated = True
                except Exception as e:
                    pass  # RAG will work without historical data

            st.session_state.ai_system = system

            # AUTO-UPDATE: Update AI performance outcomes once per session
            if 'ai_perf_updated' not in st.session_state:
                try:
                    from src.ml.ai_performance_tracker import AIPerformanceTracker
                    tracker = AIPerformanceTracker()
                    updated = tracker.update_outcomes(days_back=60)
                    st.session_state.ai_perf_updated = True
                except Exception:
                    pass  # Performance tracking is optional

        except Exception as e:
            st.caption(f"AI not available: {e}")
            return

    ai_system = st.session_state.ai_system

    # Check if model loaded
    if not ai_system.ml_predictor or not ai_system.ml_predictor.models:
        st.caption("⚠️ Model not trained")
        if st.button("Train Model", key=f"train_ai_{signal.ticker}"):
            with st.spinner("Training..."):
                ai_system._train_ml_model()
                # Also populate RAG
                if ai_system.rag_memory:
                    ai_system.rag_memory.load_from_db()
            st.rerun()
        return

    # Build scores from signal - pass ALL 9 features with domain-correct defaults
    scores = {
        'ticker': signal.ticker,
        'sentiment_score': getattr(signal, 'sentiment_score', None) or 50,
        'fundamental_score': getattr(signal, 'fundamental_score', None) or 50,
        'technical_score': getattr(signal, 'technical_score', None) or 50,
        'options_flow_score': getattr(signal, 'options_score', None) or 50,
        'short_squeeze_score': getattr(signal, 'short_squeeze_score', None) or 50,
        'total_score': signal.today_score or 50,
        'gap_score': getattr(signal, 'gap_score', None) or 50,
        'article_count': getattr(signal, 'article_count', None) or 0,
        'target_upside_pct': getattr(signal, 'target_upside_pct', None) or 0,
    }

    # Get days_to_earnings from signal (only if upcoming, not past)
    days_to_earnings = None
    if hasattr(signal, 'days_to_earnings') and signal.days_to_earnings:
        dte = signal.days_to_earnings
        if isinstance(dte, (int, float)) and dte > 0 and dte < 999:
            days_to_earnings = int(dte)

    stock_data = {
        'price': getattr(signal, 'current_price', 100) or 100,
        'sector': getattr(signal, 'sector', 'Unknown') or 'Unknown',
        'vix': 20,  # Could get from market overview if available
        'days_to_earnings': days_to_earnings,  # Only upcoming, not past
    }

    try:
        result = ai_system.analyze(signal.ticker, scores, stock_data)

        # Probability & EV
        prob_color = "green" if result.ml_probability >= 0.60 else "orange" if result.ml_probability >= 0.50 else "red"
        st.markdown(f"**Win Prob:** :{prob_color}[{result.ml_probability:.0%}] | **EV:** {result.ml_ev*100:.2f}%")

        # Similar setups (if we have data)
        if result.similar_count > 0:
            st.caption(f"📊 Similar setups: {result.similar_count} trades, {result.similar_win_rate:.0%} win rate")

        # Recommendation
        if result.approved:
            st.success(f"✅ {result.recommendation}")
            st.caption(f"Size: {result.position_size_pct:.1%} | R:R {result.risk_reward:.1f}x")
        else:
            st.warning(f"❌ SKIP")
            if result.rejection_reasons:
                st.caption(f"• {', '.join(result.rejection_reasons[:2])}")

        # Auto-log this recommendation for performance tracking
        try:
            from src.ml.ai_performance_tracker import AIPerformanceTracker
            tracker = AIPerformanceTracker()
            tracker.log_recommendation(
                ticker=signal.ticker,
                ai_probability=result.ml_probability,
                ai_ev=result.ml_ev,
                recommendation='BUY' if result.approved else 'SKIP',
                signal_scores=scores,
                entry_price=getattr(signal, 'current_price', None)
            )
        except Exception:
            pass  # Logging is optional, don't break UI

        # Factors (collapsed)
        with st.expander("Details"):
            if result.positive_factors:
                st.markdown("**Bullish:**")
                for f in result.positive_factors[:3]:
                    st.caption(f"✅ {f}")

            if result.negative_factors:
                st.markdown("**Bearish:**")
                for f in result.negative_factors[:3]:
                    st.caption(f"❌ {f}")

            if result.risks:
                st.markdown("**Risks:**")
                for r in result.risks[:3]:
                    st.caption(f"⚠️ {r}")

            # Show RAG memory stats
            if ai_system.rag_memory:
                stats = ai_system.rag_memory.get_stats()
                if stats['total_cards'] > 0:
                    st.markdown("---")
                    st.caption(f"📚 Historical Data: {stats['total_cards']} setups, {stats['win_rate']:.0%} win rate")

    except Exception as e:
        st.caption(f"Analysis error: {e}")


def populate_rag_memory_if_needed():
    """Helper to populate RAG memory."""
    if 'ai_system' in st.session_state:
        ai_system = st.session_state.ai_system
        if ai_system.rag_memory and len(ai_system.rag_memory._cards) == 0:
            count = ai_system.rag_memory.load_from_db()
            return count
    return 0


def get_ai_probabilities_batch(signal_list) -> dict:
    """
    Get AI win probabilities for all signals in batch.
    Returns dict: {ticker: {'prob': float, 'ev': float, 'approved': bool}}

    Uses caching to avoid recomputation on each rerun.
    """
    if not AI_SYSTEM_AVAILABLE:
        return {}

    # Check cache - cache for 5 minutes
    cache_key = 'ai_probs_cache'
    cache_time_key = 'ai_probs_cache_time'

    import time
    current_time = time.time()

    if cache_key in st.session_state and cache_time_key in st.session_state:
        # Use cache if less than 5 minutes old
        if current_time - st.session_state[cache_time_key] < 300:
            return st.session_state[cache_key]

    # Initialize AI system if needed
    if 'ai_system' not in st.session_state:
        try:
            system = AITradingSystem()
            system.initialize(train_if_needed=False)

            # Auto-populate RAG memory
            if system.rag_memory and len(system.rag_memory._cards) == 0:
                system.rag_memory.load_from_db()

            st.session_state.ai_system = system
        except Exception:
            return {}

    ai_system = st.session_state.ai_system

    # Check if model is loaded
    if not ai_system.ml_predictor or not ai_system.ml_predictor.models:
        return {}

    # Compute probabilities for all signals
    results = {}

    for signal in signal_list:
        try:
            scores = {
                'ticker': signal.ticker,
                'sentiment_score': getattr(signal, 'sentiment_score', None) or 50,
                'fundamental_score': getattr(signal, 'fundamental_score', None) or 50,
                'technical_score': getattr(signal, 'technical_score', None) or 50,
                'options_flow_score': getattr(signal, 'options_score', None) or 50,
                'short_squeeze_score': getattr(signal, 'short_squeeze_score', None) or 50,
                'total_score': signal.today_score or 50,
                'gap_score': getattr(signal, 'gap_score', None) or 50,
                'article_count': getattr(signal, 'article_count', None) or 0,
                'target_upside_pct': getattr(signal, 'target_upside_pct', None) or 0,
            }

            # Get ML prediction only (fast)
            ml_pred = ai_system.ml_predictor.predict(scores)

            results[signal.ticker] = {
                'prob': ml_pred.prob_win_5d,
                'ev': ml_pred.ev_5d,
                'approved': ml_pred.prob_win_5d >= 0.55 and ml_pred.ev_5d > 0
            }
        except Exception:
            continue

    # Cache results
    st.session_state[cache_key] = results
    st.session_state[cache_time_key] = current_time

    return results