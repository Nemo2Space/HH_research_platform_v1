"""
Signals Tab - Main Entry Point

Main render_signals_tab function that ties all modules together.
"""

from .shared import st, logger, time, SIGNAL_HUB_AVAILABLE

from .job_manager import (
    _get_job_status, _ensure_job_tables, _start_job, _stop_job,
    _complete_job, _reset_job, _get_job_log, _get_job_stats, _get_processed_tickers,
)

from .universe_manager import _render_quick_add_ticker, _render_remove_ticker, _get_universe_tickers

from .table_view import _render_signals_table_view

from .analysis import _process_next


def _render_analysis_panel():
    """Render the analysis control panel."""
    st.markdown("### 🔬 Analysis Panel")

    job = _get_job_status()
    tickers = _get_universe_tickers()
    processed = _get_processed_tickers()

    remaining = [t for t in tickers if t not in processed]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Tickers", len(tickers))
    with col2:
        st.metric("Processed", len(processed))
    with col3:
        st.metric("Remaining", len(remaining))
    with col4:
        st.metric("Status", job['status'].upper())

    # Warning if job says completed but there are remaining tickers
    if job['status'] == 'completed' and remaining:
        st.warning(f"⚠️ Job marked complete but {len(remaining)} tickers still need processing. Press Start to continue.")

    # Control buttons
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

    with btn_col1:
        if job['status'] != 'running':
            btn_label = "▶️ Continue" if remaining and job['status'] in ('stopped', 'completed') else "▶️ Start"
            if st.button(btn_label, use_container_width=True):
                _start_job(len(tickers))
                st.rerun()
        else:
            if st.button("⏸️ Pause", use_container_width=True):
                _stop_job()
                st.rerun()

    with btn_col2:
        if st.button("🔄 Reset", use_container_width=True):
            _reset_job()
            st.rerun()

    with btn_col3:
        skip_analyzed = st.checkbox("Skip if analyzed today", value=True)

    with btn_col4:
        force_news = st.checkbox("Force fresh news", value=False)

    # Show progress bar
    if tickers:
        progress = len(processed) / len(tickers)
        st.progress(progress, text=f"{len(processed)}/{len(tickers)} ({progress*100:.0f}%)")

    # Auto-process if running AND there are remaining tickers
    if job['status'] == 'running':
        if remaining:
            st.info(f"🔄 Processing... Stay on this page. {len(remaining)} remaining.")
            _process_next(remaining, skip_if_analyzed_today=skip_analyzed, force_fresh_news=force_news)
        else:
            _complete_job()
            st.success("✅ Analysis complete!")
            st.balloons()
            time.sleep(2)
            st.rerun()

    # Show log
    st.markdown("#### Recent Activity")
    log_df = _get_job_log()
    if not log_df.empty:
        st.dataframe(log_df, use_container_width=True, hide_index=True)

    # Stats
    stats = _get_job_stats()
    if stats['total'] > 0:
        st.markdown(f"**Stats:** {stats['analyzed']} analyzed, {stats['skipped']} skipped, {stats['failed']} failed, {stats['news_total']} news articles")

def render_signals_tab():
    """Render the Signals tab with Universe-style table and stock deep dive."""

    st.markdown("### 📊 Signal Hub")

    if not SIGNAL_HUB_AVAILABLE:
        st.error("Signal Hub not available.")
        return

    # Session state initialization
    if 'signals_loaded' not in st.session_state:
        st.session_state.signals_loaded = False
    if 'show_analysis' not in st.session_state:
        st.session_state.show_analysis = False
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = None

    # Check job from DB
    _ensure_job_tables()
    job = _get_job_status()

    # Auto-show analysis if job running
    if job['status'] == 'running':
        st.session_state.show_analysis = True

    # Top action buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("📥 Load Signals", type="primary", use_container_width=True):
            # Clear any cached data to force fresh database read
            if 'signals_data' in st.session_state:
                del st.session_state['signals_data']
            st.session_state.signals_loaded = True
            st.session_state.show_analysis = False
            st.session_state.selected_ticker = None
            st.session_state.force_refresh = True  # Force fresh load
            st.rerun()

    with col2:
        if st.button("🚀 Run Analysis", type="secondary", use_container_width=True):
            st.session_state.show_analysis = True
            st.rerun()

    with col3:
        # Status indicator
        if job['status'] == 'running':
            st.info(f"🔄 Running: {job['processed_count']}/{job['total_count']}")
        elif job['status'] == 'completed':
            st.success(f"✅ Completed: {job['processed_count']} tickers")

    # =========================================================================
    # QUICK ADD TICKER - Analyze any stock on-demand
    # =========================================================================
    with st.expander("➕ Quick Add & Analyze Ticker", expanded=False):
        _render_quick_add_ticker()

    # =========================================================================
    # REMOVE TICKER - Remove stocks from watchlist
    # =========================================================================
    with st.expander("🗑️ Remove Ticker from Watchlist", expanded=False):
        _render_remove_ticker()

    # Show analysis panel or signals table
    if st.session_state.show_analysis:
        _render_analysis_panel()
    elif st.session_state.signals_loaded:
        _render_signals_table_view()
    else:
        st.info("👆 Click **Load Signals** to view the signals table, or **Run Analysis** to refresh all data")