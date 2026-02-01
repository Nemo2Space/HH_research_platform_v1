"""
Enhanced Portfolio Display - Complete Database Integration
"""

import streamlit as st
import pandas as pd

def render_comprehensive_stock_table(result) -> None:
    """Render ONE comprehensive table with ALL database fields."""

    if not result.success or not result.holdings:
        st.warning("No holdings to display")
        return

    # Build comprehensive data
    data = []
    for h in result.holdings:

        # Build "Why Chosen"
        why_parts = []

        # AI signals
        if h.ai_decision:
            if h.ai_decision.ai_action:
                why_parts.append(f"AI: {h.ai_decision.ai_action}")
            if h.ai_decision.ai_probability:
                why_parts.append(f"Prob: {h.ai_decision.ai_probability:.0f}%")
            if h.ai_decision.committee_verdict:
                why_parts.append(f"Committee: {h.ai_decision.committee_verdict}")

        # Catalysts
        if h.catalyst_info:
            if h.catalyst_info.days_to_fda:
                why_parts.append(f"{h.catalyst_info.catalyst_label} in {h.catalyst_info.days_to_fda}d")
            elif h.catalyst_info.days_to_earnings:
                why_parts.append(f"Earnings in {h.catalyst_info.days_to_earnings}d")

        # Score
        if h.composite_score >= 70:
            why_parts.append(f"High score ({h.composite_score:.0f})")

        why_chosen = " • ".join(why_parts) if why_parts else "Diversification"

        # Build row with ALL fields
        row = {
            'Ticker': h.ticker,
            'Company': h.company_name[:25] if h.company_name else h.ticker,
            'Weight': f"{h.weight_pct:.1f}%",
            'Value': f"${h.value:,.0f}",
            'Score': int(h.composite_score),
            'Conv': h.conviction or 'MED',
            'Why Chosen': why_chosen,
            'Sector': h.sector or '-',
            'Cap': f"${h.market_cap/1e9:.1f}B" if h.market_cap and h.market_cap > 1e9 else f"${h.market_cap/1e6:.0f}M" if h.market_cap else '-',
            'P/E': f"{h.pe_ratio:.1f}" if h.pe_ratio and h.pe_ratio > 0 else '-',
            'Rev Growth': f"{h.revenue_growth:.0f}%" if h.revenue_growth else '-',
            'AI Action': h.ai_decision.ai_action if h.ai_decision and h.ai_decision.ai_action else '-',
            'AI Prob': f"{h.ai_decision.ai_probability:.0f}%" if h.ai_decision and h.ai_decision.ai_probability else '-',
            'Committee': h.ai_decision.committee_verdict if h.ai_decision and h.ai_decision.committee_verdict else '-',
            'Catalyst': h.catalyst_info.catalyst_label if h.catalyst_info and h.catalyst_info.catalyst_label else '-',
            'Days': f"{h.catalyst_info.days_to_fda or h.catalyst_info.days_to_earnings}d" if h.catalyst_info and (h.catalyst_info.days_to_fda or h.catalyst_info.days_to_earnings) else '-',
            'Options': h.options_sentiment or '-',
            'Squeeze': h.squeeze_risk or '-',
        }

        data.append(row)

    df = pd.DataFrame(data)

    # Display
    st.markdown("### 📊 Portfolio Holdings")
    st.caption(f"{len(data)} positions • ${result.invested_value:,.0f} invested • Avg Score: {result.avg_score:.0f}")

    st.dataframe(
        df,
        width="stretch",
        height=600,
        hide_index=True,
    )

    # Metrics
    st.markdown("---")
    st.markdown("### 📈 Portfolio Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Holdings", result.num_holdings)
    with col2:
        st.metric("Invested", f"${result.invested_value:,.0f}")
    with col3:
        st.metric("Avg Score", f"{result.avg_score:.0f}")
    with col4:
        high_conv = sum(1 for h in result.holdings if h.conviction == 'HIGH')
        st.metric("High Conv", f"{high_conv}/{result.num_holdings}")
    with col5:
        with_cat = sum(1 for h in result.holdings if h.catalyst_info and (h.catalyst_info.days_to_fda or h.catalyst_info.days_to_earnings))
        st.metric("Catalysts", f"{with_cat}/{result.num_holdings}")

    # Expandable details
    st.markdown("---")
    st.markdown("### 🔍 Detailed Analysis")

    for h in result.holdings:
        with st.expander(f"**{h.ticker}** - {h.company_name} ({h.weight_pct:.2f}%)"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📊 Position**")
                st.write(f"Weight: {h.weight_pct:.2f}% | Value: ${h.value:,.0f}")
                st.write(f"Score: {h.composite_score:.0f} | Conv: {h.conviction}")

                st.markdown("**💰 Fundamentals**")
                if h.market_cap:
                    st.write(f"Cap: ${h.market_cap/1e9:.1f}B" if h.market_cap > 1e9 else f"${h.market_cap/1e6:.0f}M")
                if h.pe_ratio and h.pe_ratio > 0:
                    st.write(f"P/E: {h.pe_ratio:.1f}")
                if h.revenue_growth:
                    st.write(f"Revenue Growth: {h.revenue_growth:.1f}%")

            with col2:
                st.markdown("**🤖 AI Analysis**")
                if h.ai_decision:
                    if h.ai_decision.ai_action:
                        st.write(f"Action: {h.ai_decision.ai_action}")
                    if h.ai_decision.ai_probability:
                        st.write(f"Probability: {h.ai_decision.ai_probability:.0f}%")
                    if h.ai_decision.committee_verdict:
                        st.write(f"Committee: {h.ai_decision.committee_verdict}")

                st.markdown("**🎯 Catalysts**")
                if h.catalyst_info:
                    if h.catalyst_info.days_to_fda:
                        st.write(f"{h.catalyst_info.catalyst_label}: {h.catalyst_info.days_to_fda}d")
                    if h.catalyst_info.days_to_earnings:
                        st.write(f"Earnings: {h.catalyst_info.days_to_earnings}d")


def render_save_portfolio_section(result, engine_available: bool = True):
    """Save portfolio section."""
    if not result or not result.success:
        return

    st.markdown("---")
    st.markdown("### 💾 Save Portfolio")

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        name = st.text_input("Name", placeholder="Biotech Growth Q1 2026")
        desc = st.text_area("Description", height=80)

        if st.button("💾 Save", type="primary", width='stretch'):
            if not name:
                st.error("Enter a name")
            else:
                try:
                    from dashboard.portfolio_builder import save_portfolio
                    pid = save_portfolio(name, result, desc)
                    if pid:
                        st.success(f"✅ Saved! ID: {pid}")
                        st.balloons()
                except Exception as e:
                    st.error(f"Error: {e}")