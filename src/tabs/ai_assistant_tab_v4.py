"""
AI Research Tab v4 - Institutional Async Interface with RAG

Features:
- Per-user session isolation (no shared state)
- Structured responses with clickable citations
- Guardrail alerts prominently displayed
- Staleness warnings
- Proper disclaimers
- SEC Filing RAG integration (NEW)
- Dual Analyst quick analysis (NEW)

Author: HH Research Platform
Version: 4.1
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid
import re

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Import v4 Research Agent
try:
    from src.ai.ai_research_agent_v4 import (
        create_research_agent,
        research_sync,
        ResearchResponse,
        AsyncResearchAgent
    )

    AGENT_AVAILABLE = True
except ImportError as e:
    logger.error(f"AI Research Agent v4 import failed: {e}")
    AGENT_AVAILABLE = False

# =============================================================================
# NEW: RAG INTEGRATION
# =============================================================================
try:
    from src.rag.retriever import RAGRetriever

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.warning("RAG Retriever not available")

# =============================================================================
# NEW: SEC INSIGHTS INTEGRATION
# =============================================================================
try:
    from src.signals.filing_signal import get_filing_insights

    SEC_INSIGHTS_AVAILABLE = True
except ImportError:
    SEC_INSIGHTS_AVAILABLE = False
    logger.warning("SEC Filing Insights not available")

# =============================================================================
# NEW: DUAL ANALYST INTEGRATION
# =============================================================================
try:
    from src.ai.dual_analyst import DualAnalystService

    DUAL_ANALYST_AVAILABLE = True
except ImportError:
    DUAL_ANALYST_AVAILABLE = False
    logger.warning("Dual Analyst Service not available")


def _get_user_id() -> str:
    """Get or create a unique user ID for this session."""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())[:8]
    return st.session_state.user_id


def _get_history() -> List[Dict]:
    """Get conversation history for this session."""
    if 'research_history' not in st.session_state:
        st.session_state.research_history = []
    return st.session_state.research_history


def _add_to_history(role: str, content: str):
    """Add message to history."""
    history = _get_history()
    history.append({"role": role, "content": content})
    # Keep last 10 messages
    if len(history) > 10:
        st.session_state.research_history = history[-10:]


def _detect_ticker(text: str) -> Optional[str]:
    """Detect ticker symbol from user text."""
    # Common patterns: "about AAPL", "NVDA outlook", "analyze MSFT"
    patterns = [
        r'\b([A-Z]{1,5})\b(?:\s+stock|\s+outlook|\s+analysis)?',
        r'(?:about|analyze|for|on)\s+([A-Z]{1,5})\b',
    ]

    # Known tickers to validate against
    known_tickers = _get_known_tickers()

    for pattern in patterns:
        matches = re.findall(pattern, text.upper())
        for match in matches:
            if match in known_tickers:
                return match

    return None


def _get_known_tickers() -> set:
    """Get set of known tickers from database."""
    try:
        from src.db.connection import get_engine
        engine = get_engine()
        df = pd.read_sql("SELECT DISTINCT ticker FROM screener_scores", engine)
        return set(df['ticker'].tolist())
    except:
        # Fallback to common tickers
        return {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'MU',
                'INTC', 'CRM', 'ORCL', 'ADBE', 'NFLX', 'PYPL', 'SQ', 'SHOP', 'UBER'}


def _is_analysis_request(text: str) -> bool:
    """Detect if user is asking for stock analysis."""
    analysis_keywords = [
        'analyze', 'analysis', 'outlook', 'should i buy', 'should i sell',
        'what do you think', 'investment', 'recommendation', 'target',
        'risk', 'earnings', 'sec filing', '10-k', '10-q', 'guidance',
        'is it a good', 'is it worth', 'price target', 'bull case', 'bear case'
    ]

    text_lower = text.lower()
    return any(kw in text_lower for kw in analysis_keywords)


def _inject_rag_context(question: str, ticker: str) -> str:
    """Inject RAG context into the question if available."""
    if not RAG_AVAILABLE or not ticker:
        return question

    try:
        retriever = RAGRetriever()
        results = retriever.retrieve(query=question, ticker=ticker)

        if not results or not results.chunks:
            return question

        # Build context from chunks
        context_parts = []
        for chunk in results.chunks[:5]:  # Top 5 chunks
            context_parts.append(f"[{chunk.doc_type}] {chunk.content[:500]}")

        rag_context = "\n\n".join(context_parts)

        enhanced_question = f"""[SEC FILING CONTEXT]
{rag_context}

[USER QUESTION]
{question}

Please incorporate the SEC filing information in your analysis. Cite specific filings when relevant."""

        return enhanced_question

    except Exception as e:
        logger.warning(f"RAG injection failed: {e}")
        return question


def render_ai_assistant_tab():
    """Render the AI Research Assistant tab."""

    st.markdown("## 🔬 AI Research Assistant v4")
    st.caption("Institutional-grade research with async I/O, structured citations, and session isolation")

    if not AGENT_AVAILABLE:
        st.error("AI Research Agent v4 not available.")
        st.code("Copy ai_research_agent_v4.py to src/ai/ai_research_agent_v4.py")
        return

    # Session info
    user_id = _get_user_id()

    # Status bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.caption(f"Session: `{user_id}`")
    with col2:
        st.caption("🔒 Isolated State")
    with col3:
        st.caption("⚡ Async I/O")
    with col4:
        # NEW: Show RAG status
        if RAG_AVAILABLE:
            st.caption("📄 RAG Active")
        else:
            st.caption("📄 RAG N/A")

    st.markdown("---")

    # ==========================================================================
    # QUICK RESEARCH BUTTONS
    # ==========================================================================
    st.markdown("### 💡 Quick Research")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🚀 Short-term Plays", width='stretch'):
            _execute_research("What are the best stocks for short-term trading this week?")

    with col2:
        if st.button("📈 2025 Outlook", width='stretch'):
            _execute_research("What stocks have the best outlook for 2025?")

    with col3:
        if st.button("💎 Long-term Holds", width='stretch'):
            _execute_research("What are the best stocks to hold for 10+ years?")

    with col4:
        if st.button("⚠️ Stocks to Avoid", width='stretch'):
            _execute_research("Which stocks should I avoid right now and why?")

    # ==========================================================================
    # NEW: DUAL ANALYST QUICK BUTTONS
    # ==========================================================================
    if DUAL_ANALYST_AVAILABLE:
        st.markdown("### 🔬 Dual Analysis (SQL + RAG)")

        col1, col2, col3, col4 = st.columns(4)

        # Get top tickers for quick analysis
        top_tickers = ['NVDA', 'AAPL', 'MSFT', 'GOOGL']

        for i, ticker in enumerate(top_tickers):
            with [col1, col2, col3, col4][i]:
                if st.button(f"🔬 {ticker}", width='stretch', key=f"dual_{ticker}"):
                    _execute_dual_analysis(ticker)

    st.markdown("---")

    # ==========================================================================
    # CUSTOM RESEARCH
    # ==========================================================================
    st.markdown("### 🔍 Custom Research")

    user_question = st.text_area(
        "Ask your research question",
        placeholder="""Examples:
• Why is NKE falling today?
• Should I buy NVDA at current prices?
• Compare AAPL vs MSFT for 2025
• What's the outlook for AI stocks?
• What does MU's latest 10-K say about AI demand?""",
        height=100,
        key="research_input"
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
    with col_btn1:
        if st.button("🔬 Research", type="primary", width='stretch'):
            if user_question.strip():
                _execute_research(user_question.strip())
    with col_btn2:
        if st.button("🗑️ Clear History"):
            st.session_state.research_history = []
            st.session_state.research_responses = []
            st.rerun()
    with col_btn3:
        # NEW: Toggle RAG injection
        if RAG_AVAILABLE:
            use_rag = st.checkbox("📄 Include SEC filings", value=True, key="use_rag",
                                  help="Automatically inject relevant SEC filing context")
        else:
            use_rag = False

    st.markdown("---")

    # ==========================================================================
    # RESULTS DISPLAY
    # ==========================================================================
    st.markdown("### 📊 Research Results")

    if 'research_responses' not in st.session_state:
        st.session_state.research_responses = []

    if not st.session_state.research_responses:
        st.info("👋 Ask a question above to get started!")

        # NEW: Show available features
        with st.expander("📚 Available Features", expanded=False):
            st.markdown("""
            **Standard Research:**
            - Market analysis and stock recommendations
            - Technical and fundamental analysis
            - News sentiment analysis
            - Options flow analysis

            **SEC Filing Integration (RAG):**
            - 10-K and 10-Q analysis
            - Management guidance extraction
            - Risk factor identification
            - Earnings call insights

            **Dual Analyst System:**
            - Quantitative analyst (SQL data)
            - Qualitative analyst (SEC filings)
            - Agreement scoring
            - Conflict detection
            """)
        return

    # Display responses (newest first)
    for i, item in enumerate(reversed(st.session_state.research_responses)):
        question = item.get("question", "")
        response = item.get("response")
        response_type = item.get("type", "research")

        if not response:
            continue

        with st.container():
            # Question
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); 
                        padding: 12px 16px; border-radius: 8px; margin: 8px 0;
                        border-left: 4px solid #4da6ff;">
                <strong>🧑 Question:</strong> {question}
            </div>
            """, unsafe_allow_html=True)

            # Handle different response types
            if response_type == "dual_analysis":
                _render_dual_response(response)
            elif response_type == "research":
                _render_research_response(response)
            else:
                st.markdown(str(response))

            st.markdown("---")


def _render_research_response(response: 'ResearchResponse'):
    """Render a standard research response."""

    # Guardrail alerts
    if response.guardrails_triggered:
        for alert in response.guardrails_triggered:
            st.warning(f"**{alert.severity}** {alert.message}")

    # Staleness warning
    if response.data_staleness and response.data_staleness.get("is_stale"):
        st.warning(response.data_staleness["warning"])

    # Main response
    col_main, col_meta = st.columns([3, 1])

    with col_main:
        st.markdown(f"**Outlook:** `{response.outlook}` | **Confidence:** `{response.confidence}`")
        st.markdown(response.analysis)

    with col_meta:
        st.markdown("**Metadata:**")
        st.json({
            "request_id": response.metadata.get("request_id"),
            "horizon": response.metadata.get("horizon"),
            "latency_ms": response.metadata.get("latency_ms"),
            "sources": response.metadata.get("data_sources", {})
        })

    # Citations
    if response.citations:
        with st.expander(f"📚 Citations ({len(response.citations)})", expanded=False):
            for cit in response.citations:
                tier_color = {
                    1: "🟢", 2: "🟢", 3: "🟡", 4: "🟠", 5: "🔴"
                }.get(cit.tier, "⚪")

                st.markdown(f"""
                **[{cit.id}]** {tier_color} `{cit.tier_label}` - {cit.source}  
                {cit.claim[:100]}...  
                {"[Link](" + cit.url + ")" if cit.url else "No URL"}
                """)

    # Disclaimer
    st.caption(f"*{response.disclaimer}*")


def _render_dual_response(result: Dict[str, Any]):
    """Render a dual analyst response."""

    # Icons
    ICONS = {
        "very_bullish": "🚀", "bullish": "📈", "neutral": "➡️",
        "bearish": "📉", "very_bearish": "🔻", "unknown": "❓",
    }

    # Agreement score
    agreement = result.get('evaluation', {}).get('agreement_score', 0)
    st.progress(agreement, text=f"Analyst Agreement: {agreement:.0%}")

    # Two columns for analysts
    col_sql, col_rag = st.columns(2)

    with col_sql:
        sql = result.get('sql_analyst', {})
        sentiment = sql.get('sentiment', 'unknown')
        icon = ICONS.get(sentiment, "❓")
        confidence = sql.get('confidence', 0)

        st.markdown("### 📊 Quantitative Analyst")
        st.markdown(f"**{icon} {sentiment.replace('_', ' ').title()}** ({confidence:.0%})")
        st.markdown(f"_{sql.get('summary', 'No summary')}_")

        bullish = sql.get('bullish', [])
        if bullish:
            st.markdown("✅ " + ", ".join(bullish[:3]))

        bearish = sql.get('bearish', [])
        if bearish:
            st.markdown("⚠️ " + ", ".join(bearish[:3]))

    with col_rag:
        rag = result.get('rag_analyst', {})
        sentiment = rag.get('sentiment', 'unknown')
        icon = ICONS.get(sentiment, "❓")
        confidence = rag.get('confidence', 0)

        st.markdown("### 📄 Qualitative Analyst")
        st.markdown(f"**{icon} {sentiment.replace('_', ' ').title()}** ({confidence:.0%})")
        st.markdown(f"_{rag.get('summary', 'No summary')}_")

        bullish = rag.get('bullish', [])
        if bullish:
            st.markdown("✅ " + ", ".join(bullish[:3]))

        bearish = rag.get('bearish', [])
        if bearish:
            st.markdown("⚠️ " + ", ".join(bearish[:3]))

    # Synthesis
    st.divider()
    synthesis = result.get('synthesis', {})
    sentiment = synthesis.get('sentiment', 'unknown')
    icon = ICONS.get(sentiment, "❓")
    confidence = synthesis.get('confidence', 0)

    st.markdown(f"### 🎯 Final Verdict: {icon} {sentiment.replace('_', ' ').title()}")
    st.markdown(f"**Confidence:** {confidence:.0%}")
    st.markdown(synthesis.get('summary', ''))

    # Risk flags
    risk_flags = synthesis.get('risk_flags', [])
    if risk_flags:
        st.warning("**🚨 Risk Flags:** " + ", ".join(risk_flags))

    # Conflicts
    conflicts = result.get('evaluation', {}).get('conflicts', [])
    if conflicts:
        st.info("**⚠️ Analyst Conflicts:** " + ", ".join(conflicts))

    # Disclaimer
    st.caption(
        "*This analysis combines quantitative metrics with qualitative SEC filing analysis. Not financial advice.*")


def _execute_research(question: str):
    """Execute research with proper session isolation and RAG injection."""
    user_id = _get_user_id()
    history = _get_history()

    # Add question to history
    _add_to_history("user", question)

    # Detect ticker for RAG injection
    ticker = _detect_ticker(question)

    # Inject RAG context if enabled and ticker detected
    use_rag = st.session_state.get('use_rag', True)
    enhanced_question = question

    if use_rag and ticker and RAG_AVAILABLE and _is_analysis_request(question):
        enhanced_question = _inject_rag_context(question, ticker)

    with st.spinner("🔬 Researching (parallel queries)..."):
        # Call the sync wrapper which creates a FRESH agent instance
        response = research_sync(
            question=enhanced_question,
            user_id=user_id,
            history=history
        )

    # Add response to history
    if response.success:
        _add_to_history("assistant", response.analysis[:500])

    # Store response
    if 'research_responses' not in st.session_state:
        st.session_state.research_responses = []

    st.session_state.research_responses.append({
        "question": question,
        "response": response,
        "type": "research",
        "timestamp": datetime.now().isoformat()
    })

    st.rerun()


def _execute_dual_analysis(ticker: str):
    """Execute dual analyst analysis."""
    if not DUAL_ANALYST_AVAILABLE:
        st.error("Dual Analyst Service not available")
        return

    question = f"What is the investment outlook for {ticker}?"

    with st.spinner(f"🔬 Running dual analysis for {ticker}..."):
        try:
            service = DualAnalystService()
            result = service.analyze_for_display(ticker, question)
            result['timestamp'] = datetime.now().isoformat()
        except Exception as e:
            st.error(f"Dual analysis failed: {e}")
            return

    # Store response
    if 'research_responses' not in st.session_state:
        st.session_state.research_responses = []

    st.session_state.research_responses.append({
        "question": f"🔬 Dual Analysis: {ticker}",
        "response": result,
        "type": "dual_analysis",
        "timestamp": datetime.now().isoformat()
    })

    st.rerun()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    st.set_page_config(page_title="AI Research v4", layout="wide")
    render_ai_assistant_tab()