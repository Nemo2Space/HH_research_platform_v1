"""
Dual Analyst UI Components
==========================

Streamlit components for displaying dual analyst results.

Components:
- render_dual_analysis() - Full dual analyst panel
- render_sec_insights() - SEC insights summary
- render_opinion_card() - Single analyst opinion
- render_synthesis_card() - Synthesis result

Usage:
    import streamlit as st
    from src.components.dual_analyst_components import render_dual_analysis

    render_dual_analysis(ticker="MU")

Author: HH Research Platform
"""

import streamlit as st
from typing import Dict, Any, Optional
from datetime import datetime


# ============================================================
# SENTIMENT STYLING
# ============================================================

SENTIMENT_COLORS = {
    "very_bullish": "#00C853",
    "bullish": "#4CAF50",
    "neutral": "#9E9E9E",
    "bearish": "#FF5722",
    "very_bearish": "#D32F2F",
    "unknown": "#757575",
}

SENTIMENT_ICONS = {
    "very_bullish": "🚀",
    "bullish": "📈",
    "neutral": "➡️",
    "bearish": "📉",
    "very_bearish": "🔻",
    "unknown": "❓",
}

SENTIMENT_LABELS = {
    "very_bullish": "Very Bullish",
    "bullish": "Bullish",
    "neutral": "Neutral",
    "bearish": "Bearish",
    "very_bearish": "Very Bearish",
    "unknown": "Unknown",
}


def _get_agreement_color(score: float) -> str:
    """Get color based on agreement score."""
    if score >= 0.8:
        return "#4CAF50"  # Green
    elif score >= 0.6:
        return "#8BC34A"  # Light green
    elif score >= 0.4:
        return "#FFC107"  # Yellow
    else:
        return "#FF5722"  # Orange/red


def _format_confidence(conf: float) -> str:
    """Format confidence as percentage."""
    return f"{conf:.0%}"


# ============================================================
# MAIN DUAL ANALYSIS COMPONENT
# ============================================================

def render_dual_analysis(
    ticker: str,
    question: str = None,
    show_details: bool = True,
    key_prefix: str = "dual"
) -> Optional[Dict[str, Any]]:
    """
    Render full dual analyst panel.

    Args:
        ticker: Stock ticker to analyze
        question: Optional specific question
        show_details: Show detailed breakdowns
        key_prefix: Unique key prefix for Streamlit

    Returns:
        Analysis result dict or None if cancelled
    """
    from src.ai.dual_analyst import DualAnalystService

    st.markdown("### 🔍 Dual Analyst Analysis")

    # Question input
    if question is None:
        question = st.text_input(
            "Analysis Question",
            value=f"Provide a comprehensive investment analysis of {ticker}",
            key=f"{key_prefix}_question"
        )

    # Run analysis button
    if st.button("🚀 Run Dual Analysis", key=f"{key_prefix}_run"):
        with st.spinner("Running dual analysis... (this may take 15-30 seconds)"):
            try:
                service = DualAnalystService()
                result = service.analyze_for_display(ticker, question)

                # Store in session state
                st.session_state[f"{key_prefix}_result"] = result

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return None

    # Display results if available
    result = st.session_state.get(f"{key_prefix}_result")

    if result:
        _render_analysis_result(result, show_details, key_prefix)
        return result

    return None


def _render_analysis_result(
    result: Dict[str, Any],
    show_details: bool,
    key_prefix: str
):
    """Render the analysis result."""

    # Header with agreement score
    agreement = result['evaluation']['agreement_score']
    agreement_color = _get_agreement_color(agreement)

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(f"**Analysis for {result['ticker']}**")

    with col2:
        st.markdown(
            f"<span style='color:{agreement_color}; font-weight:bold;'>"
            f"Agreement: {agreement:.0%}</span>",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(f"⏱️ {result['total_latency_ms']}ms")

    st.divider()

    # Two-column layout for analysts
    col_sql, col_rag = st.columns(2)

    with col_sql:
        render_opinion_card(
            result['sql_analyst'],
            title="📊 Quantitative Analyst",
            subtitle="(Prices, Signals, Fundamentals)",
            key_prefix=f"{key_prefix}_sql"
        )

    with col_rag:
        render_opinion_card(
            result['rag_analyst'],
            title="📄 Qualitative Analyst",
            subtitle="(SEC Filings, Extracted Facts)",
            key_prefix=f"{key_prefix}_rag",
            show_citations=True
        )

    # Synthesis section
    st.divider()
    render_synthesis_card(result, key_prefix)

    # Conflicts warning
    if result['evaluation']['conflicts']:
        st.warning("⚠️ **Analyst Conflicts Detected:**")
        for conflict in result['evaluation']['conflicts']:
            st.markdown(f"- {conflict}")

    # Detailed breakdown
    if show_details:
        with st.expander("📋 Detailed Factor Breakdown"):
            _render_detailed_breakdown(result)


def render_opinion_card(
    opinion: Dict[str, Any],
    title: str,
    subtitle: str = "",
    key_prefix: str = "opinion",
    show_citations: bool = False
):
    """Render a single analyst opinion card."""

    sentiment = opinion.get('sentiment', 'unknown')
    color = SENTIMENT_COLORS.get(sentiment, "#757575")
    icon = SENTIMENT_ICONS.get(sentiment, "❓")
    label = SENTIMENT_LABELS.get(sentiment, "Unknown")
    confidence = opinion.get('confidence', 0)

    # Card container
    st.markdown(f"**{title}**")
    if subtitle:
        st.caption(subtitle)

    # Sentiment badge
    st.markdown(
        f"<div style='background-color:{color}20; border-left:4px solid {color}; "
        f"padding:10px; border-radius:4px; margin:10px 0;'>"
        f"<span style='font-size:1.5em;'>{icon}</span> "
        f"<strong style='color:{color};'>{label}</strong> "
        f"<span style='color:#666;'>({_format_confidence(confidence)} confidence)</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Summary
    summary = opinion.get('summary', 'No summary available')
    st.markdown(f"_{summary}_")

    # Key points
    key_points = opinion.get('key_points', [])
    if key_points:
        st.markdown("**Key Points:**")
        for point in key_points[:5]:
            st.markdown(f"• {point}")

    # Bullish/Bearish factors in columns
    col_bull, col_bear = st.columns(2)

    with col_bull:
        bullish = opinion.get('bullish', [])
        if bullish:
            st.markdown("✅ **Bullish:**")
            for factor in bullish[:3]:
                st.markdown(f"<small>• {factor}</small>", unsafe_allow_html=True)

    with col_bear:
        bearish = opinion.get('bearish', [])
        if bearish:
            st.markdown("⚠️ **Bearish:**")
            for factor in bearish[:3]:
                st.markdown(f"<small>• {factor}</small>", unsafe_allow_html=True)

    # Citations for RAG analyst
    if show_citations:
        citations = opinion.get('citations', [])
        if citations:
            st.caption(f"📎 {len(citations)} citations from SEC filings")


def render_synthesis_card(result: Dict[str, Any], key_prefix: str = "synth"):
    """Render the synthesis/final verdict card."""

    synthesis = result.get('synthesis', {})
    sentiment = synthesis.get('sentiment', 'unknown')
    color = SENTIMENT_COLORS.get(sentiment, "#757575")
    icon = SENTIMENT_ICONS.get(sentiment, "❓")
    label = SENTIMENT_LABELS.get(sentiment, "Unknown")
    confidence = synthesis.get('confidence', 0)

    st.markdown("### 🎯 Synthesis")

    # Main verdict box
    st.markdown(
        f"<div style='background-color:{color}15; border:2px solid {color}; "
        f"padding:20px; border-radius:8px; text-align:center;'>"
        f"<div style='font-size:2em;'>{icon}</div>"
        f"<div style='font-size:1.5em; font-weight:bold; color:{color};'>{label}</div>"
        f"<div style='color:#666;'>Confidence: {_format_confidence(confidence)}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Summary
    summary = synthesis.get('summary', '')
    if summary:
        st.markdown(f"\n{summary}")

    # Combined factors
    col1, col2, col3 = st.columns(3)

    with col1:
        bullish = synthesis.get('bullish_factors', [])
        if bullish:
            st.markdown("**✅ Bullish Factors:**")
            for f in bullish[:5]:
                st.markdown(f"• {f}")

    with col2:
        bearish = synthesis.get('bearish_factors', [])
        if bearish:
            st.markdown("**⚠️ Bearish Factors:**")
            for f in bearish[:5]:
                st.markdown(f"• {f}")

    with col3:
        risks = synthesis.get('risk_flags', [])
        if risks:
            st.markdown("**🚨 Risk Flags:**")
            for r in risks[:5]:
                st.markdown(f"• {r}")


def _render_detailed_breakdown(result: Dict[str, Any]):
    """Render detailed factor breakdown."""

    st.markdown("#### SQL Analyst Details")
    sql = result.get('sql_analyst', {})
    st.json({
        "data_sources": sql.get('data_sources', []),
        "risks": sql.get('risks', []),
        "latency_ms": sql.get('latency_ms', 0),
    })

    st.markdown("#### RAG Analyst Details")
    rag = result.get('rag_analyst', {})
    st.json({
        "citations": rag.get('citations', []),
        "risks": rag.get('risks', []),
        "latency_ms": rag.get('latency_ms', 0),
    })


# ============================================================
# SEC INSIGHTS COMPONENT
# ============================================================

def render_sec_insights(
    ticker: str,
    compact: bool = False,
    key_prefix: str = "sec"
):
    """
    Render SEC filing insights panel.

    Args:
        ticker: Stock ticker
        compact: Use compact display mode
        key_prefix: Unique key prefix
    """
    from src.signals.filing_signal import get_filing_insights

    insights = get_filing_insights(ticker)

    if not insights.get('available'):
        st.info(f"No SEC filing data available for {ticker}")
        return

    if compact:
        _render_sec_insights_compact(insights)
    else:
        _render_sec_insights_full(insights, key_prefix)


def _render_sec_insights_compact(insights: Dict[str, Any]):
    """Render compact SEC insights."""

    score = insights.get('score', 0)
    label = insights.get('score_label', 'Unknown')

    # Determine color
    if score >= 65:
        color = "#4CAF50"
    elif score >= 50:
        color = "#FFC107"
    else:
        color = "#FF5722"

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("SEC Filing Score", f"{score:.0f}", label)

    with col2:
        bullish = insights.get('bullish_signals', [])
        bearish = insights.get('bearish_signals', [])

        if bullish:
            st.markdown(f"✅ {bullish[0]}")
        if bearish:
            st.markdown(f"⚠️ {bearish[0]}")


def _render_sec_insights_full(insights: Dict[str, Any], key_prefix: str):
    """Render full SEC insights panel."""

    st.markdown("### 📄 SEC Filing Insights")

    score = insights.get('score', 0)
    label = insights.get('score_label', 'Unknown')
    factors = insights.get('factors', {})

    # Score display
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.metric("Filing Score", f"{score:.0f}/100")

    with col2:
        st.metric("Rating", label)

    with col3:
        quality = insights.get('data_quality', {})
        freshness = quality.get('freshness_days', 0)
        st.caption(f"Data from {freshness} days ago • {quality.get('filings_analyzed', 0)} filings analyzed")

    # Factor breakdown
    st.markdown("**Factor Breakdown:**")

    cols = st.columns(5)
    factor_names = ['guidance', 'risk', 'litigation', 'china', 'ai_demand']
    factor_labels = ['Guidance', 'Risk Profile', 'Litigation', 'China Exposure', 'AI Demand']

    for col, name, label in zip(cols, factor_names, factor_labels):
        with col:
            factor = factors.get(name, {})
            factor_score = factor.get('score', 50)
            weight = factor.get('weight', 0) * 100

            # Color based on score
            if factor_score >= 65:
                delta_color = "normal"
            elif factor_score >= 50:
                delta_color = "off"
            else:
                delta_color = "inverse"

            st.metric(
                label,
                f"{factor_score:.0f}",
                f"{weight:.0f}% weight",
            )

    # Additional details
    col_bull, col_bear = st.columns(2)

    with col_bull:
        bullish = insights.get('bullish_signals', [])
        if bullish:
            st.markdown("**✅ Bullish Signals:**")
            for signal in bullish:
                st.markdown(f"• {signal}")

    with col_bear:
        bearish = insights.get('bearish_signals', [])
        if bearish:
            st.markdown("**⚠️ Bearish Signals:**")
            for signal in bearish:
                st.markdown(f"• {signal}")

    # Guidance details
    guidance = factors.get('guidance', {})
    if guidance.get('direction'):
        st.info(f"📢 **Guidance Direction:** {guidance['direction'].upper()}")

    # China exposure details
    china = factors.get('china', {})
    if china.get('exposure'):
        with st.expander("🌏 China Exposure Details"):
            st.markdown(china['exposure'])


# ============================================================
# QUICK ANALYSIS COMPONENT (for Trade Ideas)
# ============================================================

def render_quick_dual_analysis(
    ticker: str,
    key_prefix: str = "quick"
) -> Optional[Dict[str, Any]]:
    """
    Render a compact quick analysis for trade ideas.
    Returns result if available in session state.
    """

    result_key = f"{key_prefix}_quick_result_{ticker}"

    # Check session state
    if result_key not in st.session_state:
        if st.button(f"🔍 Analyze {ticker}", key=f"{key_prefix}_btn_{ticker}"):
            with st.spinner("Analyzing..."):
                try:
                    from src.ai.dual_analyst import DualAnalystService
                    service = DualAnalystService()
                    result = service.analyze_for_display(ticker)
                    st.session_state[result_key] = result
                except Exception as e:
                    st.error(f"Failed: {e}")
                    return None

    result = st.session_state.get(result_key)

    if result:
        synthesis = result.get('synthesis', {})
        sentiment = synthesis.get('sentiment', 'unknown')
        color = SENTIMENT_COLORS.get(sentiment, "#757575")
        icon = SENTIMENT_ICONS.get(sentiment, "❓")
        label = SENTIMENT_LABELS.get(sentiment, "Unknown")

        # Compact display
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:10px;'>"
            f"<span style='font-size:1.2em;'>{icon}</span>"
            f"<strong style='color:{color};'>{label}</strong>"
            f"<span style='color:#666;'>({synthesis.get('confidence', 0):.0%})</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Show one bullish and one bearish
        bullish = synthesis.get('bullish_factors', [])
        bearish = synthesis.get('bearish_factors', [])

        if bullish:
            st.caption(f"✅ {bullish[0]}")
        if bearish:
            st.caption(f"⚠️ {bearish[0]}")

        # Expand for more
        with st.expander("View Full Analysis"):
            render_synthesis_card(result, f"{key_prefix}_{ticker}")

        return result

    return None


# ============================================================
# CHAT INTEGRATION HELPER
# ============================================================

def format_dual_analysis_for_chat(result: Dict[str, Any]) -> str:
    """
    Format dual analysis result as markdown for chat display.
    """
    synthesis = result.get('synthesis', {})
    sql = result.get('sql_analyst', {})
    rag = result.get('rag_analyst', {})
    evaluation = result.get('evaluation', {})

    sentiment = synthesis.get('sentiment', 'unknown')
    icon = SENTIMENT_ICONS.get(sentiment, "❓")
    label = SENTIMENT_LABELS.get(sentiment, "Unknown")

    md = f"""
## 🔍 Dual Analyst Analysis: {result.get('ticker', 'N/A')}

### 🎯 Final Verdict: {icon} {label}
**Confidence:** {synthesis.get('confidence', 0):.0%} | **Agreement:** {evaluation.get('agreement_score', 0):.0%}

{synthesis.get('summary', '')}

---

### 📊 Quantitative Analyst
**Sentiment:** {SENTIMENT_ICONS.get(sql.get('sentiment', 'unknown'), '❓')} {SENTIMENT_LABELS.get(sql.get('sentiment', 'unknown'), 'Unknown')}

{sql.get('summary', 'No summary')}

### 📄 Qualitative Analyst (SEC Filings)
**Sentiment:** {SENTIMENT_ICONS.get(rag.get('sentiment', 'unknown'), '❓')} {SENTIMENT_LABELS.get(rag.get('sentiment', 'unknown'), 'Unknown')}

{rag.get('summary', 'No summary')}

---

### Key Factors
"""

    # Bullish factors
    bullish = synthesis.get('bullish_factors', [])
    if bullish:
        md += "\n**✅ Bullish:**\n"
        for f in bullish[:5]:
            md += f"- {f}\n"

    # Bearish factors
    bearish = synthesis.get('bearish_factors', [])
    if bearish:
        md += "\n**⚠️ Bearish:**\n"
        for f in bearish[:5]:
            md += f"- {f}\n"

    # Risk flags
    risks = synthesis.get('risk_flags', [])
    if risks:
        md += "\n**🚨 Risk Flags:**\n"
        for r in risks[:5]:
            md += f"- {r}\n"

    # Conflicts
    conflicts = evaluation.get('conflicts', [])
    if conflicts:
        md += "\n**⚠️ Analyst Conflicts:**\n"
        for c in conflicts:
            md += f"- {c}\n"

    return md