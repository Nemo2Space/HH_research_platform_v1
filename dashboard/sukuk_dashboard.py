"""
HH Research Platform - Sukuk Dashboard Integration

Streamlit UI components for displaying sukuk signals and data.
Integrates with the bond trading dashboard.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

from src.utils.logging import get_logger

# Import sukuk modules
try:
    from src.analytics.sukuk_models import (
        SukukInstrument, SukukUniverse, SukukLiveData, SukukSignal,
        DataQuality, SukukAction, MaturityBucket, RiskLimits
    )
    from src.analytics.sukuk_signals import (
        SukukSignalGenerator, RatesRegime, PortfolioContext,
        build_sukuk_numeric_block, generate_sukuk_analysis
    )
    from src.analytics.sukuk_data import (
        load_sukuk_universe, fetch_sukuk_market_data, SukukIBKRClient
    )
except ImportError:
    # Fallback for testing
    from sukuk_models import (
        SukukInstrument, SukukUniverse, SukukLiveData, SukukSignal,
        DataQuality, SukukAction, MaturityBucket, RiskLimits
    )
    from sukuk_signals import (
        SukukSignalGenerator, RatesRegime, PortfolioContext,
        build_sukuk_numeric_block, generate_sukuk_analysis
    )
    from sukuk_data import (
        load_sukuk_universe, fetch_sukuk_market_data, SukukIBKRClient
    )

logger = get_logger(__name__)

# Default universe path
DEFAULT_UNIVERSE_PATH = "config/sukuk_universe_usd.json"


def _get_action_color(action: SukukAction) -> str:
    """Get color for action badge."""
    colors = {
        SukukAction.BUY: "green",
        SukukAction.HOLD: "orange",
        SukukAction.WATCH: "gray",
        SukukAction.AVOID: "red",
    }
    return colors.get(action, "gray")


def _get_quality_emoji(quality: DataQuality) -> str:
    """Get emoji for data quality."""
    emojis = {
        DataQuality.OK: "✅",
        DataQuality.DEGRADED: "⚠️",
        DataQuality.STALE: "🕐",
        DataQuality.MISSING: "❌",
        DataQuality.SUSPICIOUS: "🔍",
    }
    return emojis.get(quality, "❓")


def render_sukuk_section(
    rates_regime: Optional[RatesRegime] = None,
    portfolio_context: Optional[PortfolioContext] = None,
    universe_path: str = DEFAULT_UNIVERSE_PATH,
    ibkr_host: str = "127.0.0.1",
    ibkr_port: int = 7496
):
    """
    Render the sukuk section in the bond dashboard.

    Args:
        rates_regime: Current rates environment from bond analytics
        portfolio_context: Current portfolio holdings
        universe_path: Path to sukuk universe JSON
        ibkr_host: IBKR TWS host
        ibkr_port: IBKR TWS port
    """
    st.markdown("---")
    st.markdown("### 🕌 USD Sukuk Analysis")

    # Check if universe file exists
    universe_file = Path(universe_path)
    if not universe_file.exists():
        st.warning(f"Sukuk universe file not found: {universe_path}")
        st.info("Place your `sukuk_universe_usd.json` in the config folder.")
        return

    # Initialize session state
    if 'sukuk_signals' not in st.session_state:
        st.session_state.sukuk_signals = None
    if 'sukuk_data' not in st.session_state:
        st.session_state.sukuk_data = None
    if 'sukuk_load_time' not in st.session_state:
        st.session_state.sukuk_load_time = None

    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        refresh_clicked = st.button("🔄 Refresh Sukuk Data", key="refresh_sukuk")

    with col2:
        if st.session_state.sukuk_load_time:
            st.caption(f"Last update: {st.session_state.sukuk_load_time.strftime('%H:%M:%S')}")

    # Load data if refresh clicked or first load
    if refresh_clicked or st.session_state.sukuk_signals is None:
        with st.spinner("Fetching sukuk data from IBKR..."):
            try:
                result = generate_sukuk_analysis(
                    universe_path=str(universe_file),
                    rates_regime=rates_regime,
                    portfolio=portfolio_context,
                    ibkr_host=ibkr_host,
                    ibkr_port=ibkr_port
                )

                st.session_state.sukuk_signals = result['signals']
                st.session_state.sukuk_data = result
                st.session_state.sukuk_load_time = datetime.now()

                if result['warnings']:
                    for w in result['warnings']:
                        st.warning(w)

            except Exception as e:
                st.error(f"Error loading sukuk data: {e}")
                logger.error(f"Sukuk load error: {e}", exc_info=True)
                return

    # Display signals
    signals = st.session_state.sukuk_signals
    if not signals:
        st.info("No sukuk signals available. Click 'Refresh Sukuk Data' to load.")
        return

    # Summary metrics
    _render_sukuk_summary(signals)

    # Signal cards
    _render_sukuk_signals_table(signals)

    # Detailed analysis in expander
    with st.expander("📊 Detailed Sukuk Analysis", expanded=False):
        _render_sukuk_details(signals)

    # Store numeric block for AI chat
    if st.session_state.sukuk_data:
        st.session_state.sukuk_numeric_block = st.session_state.sukuk_data.get('numeric_block', '')


def _render_sukuk_summary(signals: List[SukukSignal]):
    """Render summary metrics for sukuk signals."""
    # Count by action
    buy_signals = [s for s in signals if s.action == SukukAction.BUY]
    hold_signals = [s for s in signals if s.action == SukukAction.HOLD]
    watch_signals = [s for s in signals if s.action == SukukAction.WATCH]

    # Data quality summary
    ok_count = sum(1 for s in signals if s.data_quality == DataQuality.OK)
    degraded_count = sum(1 for s in signals if s.data_quality != DataQuality.OK)

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🟢 BUY", len(buy_signals))
    with col2:
        st.metric("🟡 HOLD", len(hold_signals))
    with col3:
        st.metric("⚪ WATCH", len(watch_signals))
    with col4:
        quality_label = f"{ok_count} OK" if degraded_count == 0 else f"{ok_count} OK / {degraded_count} ⚠️"
        st.metric("📊 Data Quality", quality_label)

    # Best opportunities
    if buy_signals:
        best = max(buy_signals, key=lambda s: s.conviction)
        st.success(f"**Top Opportunity:** {best.instrument.name} - {best.conviction}% conviction, "
                  f"{best.carry_proxy:.2f}% coupon, {best.ttm_years:.1f}y maturity")


def _render_sukuk_signals_table(signals: List[SukukSignal]):
    """Render sukuk signals as a table."""
    if not signals:
        return

    # Build dataframe
    rows = []
    for sig in signals:
        inst = sig.instrument
        rows.append({
            'Name': inst.name,
            'Issuer': inst.issuer_bucket.replace('_', ' '),
            'Maturity': inst.maturity.strftime('%Y-%m-%d'),
            'TTM': f"{sig.ttm_years:.1f}y",
            'Coupon': f"{inst.coupon_rate_pct:.2f}%",
            'Price': f"${sig.price:.2f}" if sig.price else "N/A",
            'Spread': f"{sig.bid_ask_bps:.0f}bp" if sig.bid_ask_bps else "N/A",
            'Signal': sig.action.value,
            'Conv.': f"{sig.conviction}%",
            'Quality': _get_quality_emoji(sig.data_quality),
        })

    df = pd.DataFrame(rows)

    # Style the dataframe
    def highlight_signal(val):
        if val == 'BUY':
            return 'background-color: #d4edda; color: #155724;'
        elif val == 'HOLD':
            return 'background-color: #fff3cd; color: #856404;'
        elif val == 'WATCH':
            return 'background-color: #f8f9fa; color: #6c757d;'
        elif val == 'AVOID':
            return 'background-color: #f8d7da; color: #721c24;'
        return ''

    styled_df = df.style.applymap(highlight_signal, subset=['Signal'])

    st.dataframe(styled_df, width='stretch', hide_index=True)


def _render_sukuk_details(signals: List[SukukSignal]):
    """Render detailed sukuk analysis."""
    # Group by issuer bucket
    by_issuer = {}
    for sig in signals:
        bucket = sig.instrument.issuer_bucket
        if bucket not in by_issuer:
            by_issuer[bucket] = []
        by_issuer[bucket].append(sig)

    # Display by issuer group
    for bucket, bucket_signals in sorted(by_issuer.items()):
        with st.expander(f"📁 {bucket.replace('_', ' ')}", expanded=False):
            for sig in bucket_signals:
                inst = sig.instrument

                # Signal badge color
                badge_color = _get_action_color(sig.action)

                st.markdown(f"""
                **{inst.name}** :{badge_color}[{sig.action.value}]
                
                | Field | Value |
                |-------|-------|
                | ISIN | `{inst.isin}` |
                | Issuer | {inst.issuer} |
                | Maturity | {inst.maturity} ({sig.ttm_years:.1f} years) |
                | Coupon | {inst.coupon_rate_pct:.2f}% |
                | Price | {"${:.2f}".format(sig.price) if sig.price else "N/A"} |
                | Bid/Ask Spread | {"{:.0f} bps".format(sig.bid_ask_bps) if sig.bid_ask_bps else "N/A"} |
                | Conviction | {sig.conviction}% |
                | Size Cap | {sig.size_cap_pct:.1f}% |
                | Data Quality | {sig.data_quality.value} |
                
                **Reason:** {sig.reason}
                """)

                if sig.current_weight_pct > 0:
                    st.caption(f"Current position: {sig.current_weight_pct:.1f}% | "
                              f"Issuer exposure: {sig.issuer_exposure_pct:.1f}%")

                st.markdown("---")


def get_sukuk_context_for_ai() -> str:
    """
    Get sukuk context for AI chat integration.

    Returns the numeric block if sukuk data is loaded.
    """
    if 'sukuk_numeric_block' in st.session_state and st.session_state.sukuk_numeric_block:
        return st.session_state.sukuk_numeric_block
    return ""


def render_sukuk_mini_widget():
    """
    Render a compact sukuk summary widget for sidebar or header.
    """
    signals = st.session_state.get('sukuk_signals', [])

    if not signals:
        st.caption("🕌 Sukuk: Not loaded")
        return

    buy_count = sum(1 for s in signals if s.action == SukukAction.BUY)
    total = len(signals)

    if buy_count > 0:
        st.success(f"🕌 Sukuk: {buy_count}/{total} BUY signals")
    else:
        st.info(f"🕌 Sukuk: {total} instruments tracked")


# =============================================================================
# CONID LOOKUP UTILITY (for populating universe JSON)
# =============================================================================

def render_sukuk_conid_lookup():
    """
    Render utility to look up sukuk conids by ISIN.
    Useful for populating the universe JSON file.
    """
    st.markdown("### 🔍 Sukuk CONID Lookup")
    st.caption("Use this to find IBKR conids for sukuk by ISIN.")

    isin = st.text_input("Enter ISIN:", placeholder="XS2322423539")

    if st.button("Search IBKR") and isin:
        with st.spinner("Searching IBKR..."):
            try:
                from src.analytics.sukuk_data import search_sukuk_by_isin
                result = search_sukuk_by_isin(isin)

                if result:
                    st.success("Found!")
                    st.json(result)
                else:
                    st.warning(f"No bond found for ISIN: {isin}")

            except Exception as e:
                st.error(f"Search error: {e}")


# =============================================================================
# INTEGRATION WITH BOND DASHBOARD
# =============================================================================

def integrate_sukuk_with_bond_context(
    bond_context: str,
    sukuk_context: str
) -> str:
    """
    Combine bond ETF context with sukuk context for AI.

    Args:
        bond_context: Existing bond ETF numeric block
        sukuk_context: Sukuk numeric block

    Returns:
        Combined context string
    """
    if not sukuk_context:
        return bond_context

    combined = bond_context + "\n\n" + sukuk_context

    # Add integration note
    combined += """

⚠️ SUKUK ANALYSIS NOTES:
- Sukuk use COUPON RATE as carry proxy (NOT YTM - that requires more data)
- Bid/ask spreads indicate liquidity; >200 bps = illiquid
- Phase 1 signals - duration/YTM calculations coming in Phase 2
- All prices from IBKR real-time feed
"""

    return combined