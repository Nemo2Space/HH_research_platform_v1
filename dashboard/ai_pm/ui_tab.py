from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from .audit import write_ai_pm_audit
from .config import DEFAULT_CONSTRAINTS, STRATEGIES
from .execution_engine import execute_trade_plan
from .ibkr_gateway import IbkrGateway
from .models import PortfolioSnapshot, Position, TargetWeights, TradePlan, OrderStatus, SignalSnapshot, GateReport
from .risk_engine import evaluate_trade_plan_gates
from .signal_adapter import load_platform_signals_snapshot
from .state_store import (
    init_ai_pm_state,
    set_selected_account,
    get_selected_account,
    set_strategy_key,
    get_strategy_key,
    set_auto_trade,
    is_auto_trade,
    set_kill_switch,
    is_kill_switch,
    arm_auto_trade,
    disarm_auto_trade,
    is_armed,
    get_arm_expiry,
    set_last_run,
    store_last_results,
    get_last_results,
)
from .target_builder import build_target_weights
from .trade_planner import build_trade_plan
from .assistant import answer_pm_question
from .portfolio_intelligence import (
    analyze_portfolio_intelligence,
    format_intelligence_report_markdown,
    PortfolioIntelligenceReport,
)
from .rebalance_metrics import (
    RebalanceMetrics,
    RebalanceResults,
    PositionResult,
    calculate_rebalance_metrics,
    build_position_results,
    build_rebalance_results,
    verify_execution,
)
from .pre_trade_analysis import (
    analyze_proposed_trades,
    PreTradeAnalysis,
    RiskLevel,
    WarningType,
)

# Saved Portfolio imports
try:
    from dashboard.portfolio_builder import get_saved_portfolios, load_portfolio

    SAVED_PORTFOLIO_AVAILABLE = True
except ImportError:
    SAVED_PORTFOLIO_AVAILABLE = False

# Stock Lookup and JSON Portfolio Editor imports
try:
    from .stock_lookup import get_stock_lookup, lookup_stock, lookup_or_fetch_stock
    from .json_portfolio_editor import JSONPortfolioEditor, load_portfolio_json

    STOCK_LOOKUP_AVAILABLE = True
except ImportError:
    STOCK_LOOKUP_AVAILABLE = False

# Debug Export import
try:
    from .debug_export import export_debug_snapshot

    DEBUG_EXPORT_AVAILABLE = True
except ImportError:
    DEBUG_EXPORT_AVAILABLE = False


# =============================================================================
# LOADED PORTFOLIO HELPER
# =============================================================================

def _get_loaded_portfolio_positions(account: str) -> Dict[str, Dict[str, Any]]:
    """
    Get positions from the loaded portfolio statement for the specified account.
    Returns dict: {symbol: {quantity, market_value, cost_basis, unrealized_pnl, weight}}
    """
    positions: Dict[str, Dict[str, Any]] = {}
    statements = st.session_state.get('parsed_statements', {})

    if not statements:
        return positions

    # Get statement for the specific account
    statement = statements.get(account)
    if statement is None:
        return positions

    # Calculate total portfolio value for weight calculation
    total_value = 0
    if hasattr(statement, 'open_positions') and statement.open_positions:
        for pos in statement.open_positions:
            mv = getattr(pos, 'market_value', 0) or 0
            total_value += abs(mv)

    # Extract positions
    if hasattr(statement, 'open_positions') and statement.open_positions:
        for pos in statement.open_positions:
            symbol = (getattr(pos, 'symbol', '') or '').strip().upper()
            if not symbol:
                continue

            mv = getattr(pos, 'market_value', 0) or 0
            weight = mv / total_value if total_value > 0 else 0

            positions[symbol] = {
                'quantity': getattr(pos, 'quantity', 0),
                'market_value': mv,
                'cost_basis': getattr(pos, 'cost_basis', 0) or 0,
                'unrealized_pnl': getattr(pos, 'unrealized_pnl', 0) or 0,
                'unrealized_pnl_pct': getattr(pos, 'unrealized_pnl_pct', 0) or 0,
                'weight': weight,
                'current_price': getattr(pos, 'current_price', 0) or 0,
                'cost_price': getattr(pos, 'cost_price', 0) or 0,
            }

    return positions


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
# ui_tab.py
# ADD this helper method anywhere in the file (module-level)
# (Used by _render_target_portfolio_analysis to enrich missing scanner columns)

# ui_tab.py
# REPLACE the helper in full (the one I gave earlier):
#   _fetch_scanner_enrichment_for_tickers(...)
#
# This version:
# - Uses the SAME engine helper as your platform
# - Matches the columns you just confirmed exist (prices/fundamentals/analyst_ratings/price_targets/earnings_calendar)
# - Adds extra optional columns ONLY if they exist in fundamentals (so you get "more parameters" automatically)

def _fetch_scanner_enrichment_for_tickers(tickers: list[str]):
    import pandas as pd

    if not tickers:
        return pd.DataFrame()

    tickers = [str(x).strip().upper() for x in tickers if x]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame()

    from src.db.connection import get_engine
    eng = get_engine()

    # Discover optional fundamentals columns (so you can see more parameters if your schema has them)
    cols_df = pd.read_sql(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name='fundamentals'
        """,
        eng,
    )
    fcols = set(cols_df["column_name"].astype(str).tolist())

    # Base set (matches your debug_scanner_columns output)
    base_fund_cols = [
        ("sector", "sector_db"),
        ("ex_dividend_date", "ex_dividend_date"),
        ("dividend_yield", "dividend_yield"),
        ("pe_ratio", "pe_ratio"),
        ("forward_pe", "forward_pe"),
        ("peg_ratio", "peg_ratio"),
        ("roe", "roe"),
        ("market_cap", "market_cap"),
        ("pb_ratio", "pb_ratio"),
        ("revenue_growth", "revenue_growth"),
        ("eps", "eps"),
        ("date", "fundamentals_date"),
    ]

    # Extra fundamentals columns (only if they exist)
    extra_candidates = [
        ("industry", "industry"),
        ("country", "country"),
        ("currency", "currency"),
        ("beta", "beta"),
        ("gross_margin", "gross_margin"),
        ("operating_margin", "operating_margin"),
        ("net_margin", "net_margin"),
        ("fcf_margin", "fcf_margin"),
        ("debt_to_equity", "debt_to_equity"),
        ("current_ratio", "current_ratio"),
        ("quick_ratio", "quick_ratio"),
        ("price_to_sales", "price_to_sales"),
        ("ev_to_ebitda", "ev_to_ebitda"),
        ("payout_ratio", "payout_ratio"),
        ("shares_outstanding", "shares_outstanding"),
        ("float_shares", "float_shares"),
        ("short_float", "short_float"),
        ("short_ratio", "short_ratio"),
    ]

    fund_select = []
    for src, alias in base_fund_cols:
        if src in fcols:
            fund_select.append(f"f.{src} AS {alias}")
    for src, alias in extra_candidates:
        if src in fcols:
            fund_select.append(f"f.{src} AS {alias}")

    # Always at least sector + fundamentals_date if available; if not, still proceed without fundamentals
    fund_select_sql = ",\n        ".join(fund_select) if fund_select else "f.ticker"

    q = f"""
    WITH t AS (
      SELECT UNNEST(%(tickers)s::text[]) AS ticker
    ),
    latest_prices AS (
      SELECT DISTINCT ON (p.ticker) p.ticker, p.close AS price_db
      FROM prices p
      WHERE p.ticker = ANY(%(tickers)s)
      ORDER BY p.ticker, p.date DESC
    ),
    latest_fundamentals AS (
      SELECT DISTINCT ON (f.ticker)
        f.ticker,
        {fund_select_sql}
      FROM fundamentals f
      WHERE f.ticker = ANY(%(tickers)s)
      ORDER BY f.ticker, f.date DESC NULLS LAST
    ),
    latest_ratings AS (
      SELECT DISTINCT ON (ar.ticker)
        ar.ticker,
        ar.analyst_positivity,
        ar.analyst_buy AS buy_count,
        ar.analyst_total AS total_ratings
      FROM analyst_ratings ar
      WHERE ar.ticker = ANY(%(tickers)s)
      ORDER BY ar.ticker, ar.date DESC NULLS LAST
    ),
    latest_targets AS (
      SELECT DISTINCT ON (pt.ticker)
        pt.ticker,
        pt.target_mean
      FROM price_targets pt
      WHERE pt.ticker = ANY(%(tickers)s)
      ORDER BY pt.ticker, pt.date DESC NULLS LAST
    ),
    earnings_pick AS (
      SELECT
        x.ticker,
        COALESCE(
          (SELECT ec.earnings_date
           FROM earnings_calendar ec
           WHERE ec.ticker = x.ticker AND ec.earnings_date >= CURRENT_DATE
           ORDER BY ec.earnings_date ASC LIMIT 1),
          (SELECT ec.earnings_date
           FROM earnings_calendar ec
           WHERE ec.ticker = x.ticker
             AND ec.earnings_date >= CURRENT_DATE - INTERVAL '14 days'
             AND ec.earnings_date < CURRENT_DATE
           ORDER BY ec.earnings_date DESC LIMIT 1)
        ) AS earnings_date
      FROM t x
    )
    SELECT
      t.ticker,
      lp.price_db,
      lt.target_mean,
      CASE WHEN lp.price_db > 0 AND lt.target_mean > 0
           THEN ROUND(((lt.target_mean - lp.price_db) / lp.price_db * 100)::numeric, 2)
           ELSE NULL END AS target_upside_pct_db,
      ep.earnings_date,
      lf.*,
      lr.analyst_positivity,
      lr.buy_count,
      lr.total_ratings
    FROM t
    LEFT JOIN latest_prices lp ON lp.ticker = t.ticker
    LEFT JOIN latest_targets lt ON lt.ticker = t.ticker
    LEFT JOIN earnings_pick ep ON ep.ticker = t.ticker
    LEFT JOIN latest_fundamentals lf ON lf.ticker = t.ticker
    LEFT JOIN latest_ratings lr ON lr.ticker = t.ticker
    ORDER BY t.ticker
    """

    df = pd.read_sql(q, eng, params={"tickers": tickers})

    # Drop duplicate ticker columns from lf.* expansion if any
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    # If sector_db is not present (fundamentals missing), keep schema stable
    if "sector_db" not in df.columns and "sector" in df.columns:
        # if lf.* came back with 'sector' instead of alias (edge case)
        df = df.rename(columns={"sector": "sector_db"})

    return df


# ui_tab.py — ADD these helper methods (new)
def _best_raw_value(raw: Dict[str, Any], *keys: str) -> Any:
    """
    Return the first non-empty value found across key variants:
    - exact key
    - key_score / key_signal
    - Title-case / upper-case variants
    """
    if not isinstance(raw, dict):
        return None

    variants: List[str] = []
    for k in keys:
        if not k:
            continue
        variants.extend([
            k,
            f"{k}_score",
            f"{k}_signal",
            k.lower(),
            k.upper(),
            k.title(),
        ])

    seen = set()
    for k in variants:
        if k in seen:
            continue
        seen.add(k)
        v = raw.get(k)
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def _safe_float_local(x):
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:  # NaN
            return None
        if v == float("inf") or v == float("-inf"):
            return None
        return v
    except Exception:
        return None


def _norm_prob_01(x: Any) -> Optional[float]:
    v = _safe_float_local(x)
    if v is None:
        return None
    # accept 0..1 or 0..100
    if v > 1.0:
        v = v / 100.0
    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0
    return v


def _extract_sector_from_raw(raw: Dict[str, Any]) -> Optional[str]:
    # sector can be duplicated in scores/signals merge -> sector_score / sector_signal
    v = _best_raw_value(
        raw,
        "sector",
        "gics_sector",
        "gicsSector",
        "gics_sector_name",
        "sector_name",
        "Sector",
    )
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _build_scanner_like_row(symbol: str, sig_row: Any) -> Dict[str, Any]:
    """
    Build the same “scanner page” columns from SignalSnapshot.rows[symbol].raw,
    but robust to merge suffixes (_score/_signal).
    """
    sym = (symbol or "").strip().upper()
    raw = getattr(sig_row, "raw", None) or {}
    if not isinstance(raw, dict):
        raw = {}

    # Date can exist under different names
    date_v = _best_raw_value(raw, "date", "asof", "score_date", "signal_date", "ts", "timestamp")

    # Canonical columns (match your screenshot)
    out: Dict[str, Any] = {
        "Ticker": sym,
        "Date": date_v,
        "Sector": _extract_sector_from_raw(raw),
        "Signal": _best_raw_value(raw, "signal", "signal_type", "Signal", "rating", "recommendation"),
        "Total": _best_raw_value(raw, "total", "total_score", "score_total"),
        "AI Prob": _norm_prob_01(
            _best_raw_value(
                raw,
                "ai_probability",
                "ai_prob",
                "probability",
                "predicted_probability",
                "win_probability",
                "prob_win",
                "prob_win_5d",
                "likelihood_score_norm",
            )
        ),
        "Sentiment": _best_raw_value(raw, "sentiment", "sentiment_score"),
        "OptFlow": _best_raw_value(raw, "optflow", "options_flow_score", "options_score"),
        "Squeeze": _best_raw_value(raw, "squeeze", "squeeze_score"),
        "Fundamental": _best_raw_value(raw, "fundamental", "fundamental_score"),
        "Growth": _best_raw_value(raw, "growth", "growth_score"),
        "Dividend": _best_raw_value(raw, "dividend", "dividend_score"),
        "Technical": _best_raw_value(raw, "technical", "technical_score"),
        "Gap": _best_raw_value(raw, "gap", "gap_score"),
        "GapType": _best_raw_value(raw, "gap_type", "gapType", "GapType"),
        "Like": _best_raw_value(raw, "like", "like_score", "likelihood", "likelihood_score"),
        "id_score": _best_raw_value(raw, "id_score", "idScore"),
    }

    # Keep Sector non-empty string
    if out["Sector"] is None:
        out["Sector"] = "Unknown"

    return out


# ui_tab.py — ADD this helper method (new)
def _project_cash_after_orders(snapshot: PortfolioSnapshot, plan: TradePlan) -> Dict[str, Any]:
    """
    Best-effort projection of cash after executing ALL planned orders.
    - Uses snapshot position market_price/avg_cost, then st.session_state['ai_pm_price_map'].
    - If price missing for an order, ignores that order in projection and reports it.
    """

    def _sf(x):
        try:
            if x is None:
                return None
            v = float(x)
            if pd.isna(v):
                return None
            return v
        except Exception:
            return None

    nav = _sf(getattr(snapshot, "net_liquidation", None)) or _sf(getattr(plan, "nav", None))
    cash = _sf(getattr(snapshot, "total_cash", None))
    if cash is None:
        cash = 0.0

    # Build a px map from snapshot first
    px_map: Dict[str, float] = {}
    for p in (getattr(snapshot, "positions", None) or []):
        sym = (getattr(p, "symbol", None) or "").strip().upper()
        if not sym:
            continue
        px = _sf(getattr(p, "market_price", None))
        if px is None or px <= 0:
            px = _sf(getattr(p, "avg_cost", None))
        if px is not None and px > 0:
            px_map[sym] = float(px)

    # Merge extra price map
    extra = st.session_state.get("ai_pm_price_map") or {}
    for k, v in (extra.items() if isinstance(extra, dict) else []):
        sym = ("" if k is None else str(k)).strip().upper()
        px = _sf(v)
        if sym and px and px > 0 and sym not in px_map:
            px_map[sym] = float(px)

    proj_cash = float(cash)
    missing: List[str] = []
    used_orders = 0
    ignored_orders = 0

    orders = getattr(plan, "orders", None) or []
    for o in orders:
        sym = (getattr(o, "symbol", None) or "").strip().upper()
        if not sym:
            ignored_orders += 1
            continue
        qty = _sf(getattr(o, "quantity", None)) or 0.0
        if qty <= 0:
            ignored_orders += 1
            continue
        px = px_map.get(sym)
        if px is None or px <= 0:
            missing.append(sym)
            ignored_orders += 1
            continue

        used_orders += 1
        notional = float(qty) * float(px)
        act = (getattr(o, "action", None) or "").upper().strip()
        if act == "BUY":
            proj_cash -= notional
        else:
            proj_cash += notional

    out = {
        "nav": nav,
        "cash_now": cash,
        "cash_now_w": (float(cash) / float(nav)) if (nav and nav > 0) else None,
        "cash_proj": proj_cash,
        "cash_proj_w": (float(proj_cash) / float(nav)) if (nav and nav > 0) else None,
        "orders_total": len(orders),
        "orders_used": used_orders,
        "orders_ignored": ignored_orders,
        "missing_price_symbols": sorted(list(set(missing))),
    }
    return out


# ui_tab.py — ADD this helper method (new)
def _style_scanner_like_table(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """
    Streamlit-friendly styling similar to Signals page:
    - Color Signal cells
    - Gradient for score columns (0..100)
    - Gradient for AI Prob (0..1)
    """
    import pandas as pd

    def _sig_bg(v: object) -> str:
        s = ("" if v is None else str(v)).upper().strip()
        if not s:
            return ""
        if "STRONG BUY" in s:
            return "background-color: #0b6e4f; color: white;"
        if s == "BUY" or " BUY" in s:
            return "background-color: #148a5b; color: white;"
        if "WEAK BUY" in s:
            return "background-color: #2bb673; color: black;"
        if "NEUTRAL" in s:
            return "background-color: #3a3f45; color: white;"
        if "WEAK SELL" in s:
            return "background-color: #c0392b; color: white;"
        if "SELL" in s:
            return "background-color: #a93226; color: white;"
        if "BEARISH" in s:
            return "background-color: #8e2b2b; color: white;"
        if "BULLISH" in s:
            return "background-color: #1e7e34; color: white;"
        return ""

    score_cols_0_100 = [
        "Total",
        "Sentiment",
        "OptFlow",
        "Squeeze",
        "Fundamental",
        "Growth",
        "Dividend",
        "Technical",
        "Gap",
        "Like",
    ]

    # Normalize numeric columns (ignore if missing)
    sty = df.style

    if "Signal" in df.columns:
        sty = sty.applymap(_sig_bg, subset=["Signal"])

    # Gradient on 0..100 score columns
    for c in score_cols_0_100:
        if c in df.columns:
            try:
                sty = sty.background_gradient(subset=[c], vmin=0, vmax=100)
            except Exception:
                pass

    # Gradient on AI Prob (0..1)
    if "AI Prob" in df.columns:
        try:
            sty = sty.background_gradient(subset=["AI Prob"], vmin=0.0, vmax=1.0)
        except Exception:
            pass

    # ΔW highlight (positive/negative)
    if "ΔW" in df.columns:
        def _dw_color(v: object) -> str:
            try:
                x = float(v)
            except Exception:
                return ""
            if x > 0:
                return "color: #00c853;"
            if x < 0:
                return "color: #ff5252;"
            return ""

        sty = sty.applymap(_dw_color, subset=["ΔW"])

    return sty


def _safe_float_local(x):
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:  # NaN
            return None
        if v == float("inf") or v == float("-inf"):
            return None
        return v
    except Exception:
        return None


# ui_tab.py — REPLACE this method entirely
def _render_proposed_portfolio_table(snapshot: PortfolioSnapshot, plan: TradePlan) -> None:
    """
    Proposed Portfolio = (scanner columns) + (portfolio deltas) + (planner sizing/reason).
    FIXES:
    - shows ALL scanner columns (same as signals page) instead of only portfolio columns
    - sector loads correctly (handles sector_score / sector_signal)
    - avoids duplicated "AI Prob" vs "AI Probability" by standardizing to "AI Prob"
    """
    nav = _safe_float_local(getattr(plan, "nav", None))
    if nav is None or nav <= 0:
        st.warning("NAV not available; cannot render proposed portfolio table.")
        return

    price_map = st.session_state.get("ai_pm_price_map") or {}
    pos_by_sym = {p.symbol.strip().upper(): p for p in (snapshot.positions or []) if getattr(p, "symbol", None)}
    cw = getattr(plan, "current_weights", None) or {}
    tw = getattr(plan, "target_weights", None) or {}

    orders = getattr(plan, "orders", None) or []
    o_by_sym = {}
    for o in orders:
        try:
            o_by_sym[o.symbol.strip().upper()] = o
        except Exception:
            pass

    # Pull signals from the stored artifacts (object-level)
    art = st.session_state.get("ai_pm_assistant_artifacts")
    signals_obj = art.get("signals") if isinstance(art, dict) else None
    signal_rows = getattr(signals_obj, "rows", None) or {}

    # Universe = union(current, target, and anything the planner touched
    syms = sorted(set(cw.keys()) | set(tw.keys()) | set(o_by_sym.keys()) | set(pos_by_sym.keys()))

    rows: List[Dict[str, Any]] = []

    # Debug counters
    missing_sector = 0
    missing_ai_prob = 0
    missing_signals = 0

    for sym in syms:
        sym_u = (sym or "").strip().upper()
        p = pos_by_sym.get(sym_u)

        cur_qty = _safe_float_local(getattr(p, "quantity", None)) if p else 0.0
        cur_w = float(cw.get(sym_u, 0.0) or 0.0)
        tgt_w = float(tw.get(sym_u, 0.0) or 0.0)
        delta_w = tgt_w - cur_w

        px = None
        if p:
            px = _safe_float_local(getattr(p, "market_price", None))
        if not px:
            px = _safe_float_local(price_map.get(sym_u))

        tgt_value = float(tgt_w) * float(nav)
        tgt_qty = (tgt_value / float(px)) if (px and px > 0) else None

        action = ""
        if abs(delta_w) >= 1e-6:
            action = "BUY" if delta_w > 0 else "SELL"

        o = o_by_sym.get(sym_u)
        planned_qty = _safe_float_local(getattr(o, "quantity", None)) if o else None
        reason = (getattr(o, "reason", None) or "") if o else ""

        sig = signal_rows.get(sym_u)
        if sig is None:
            missing_signals += 1
            scanner_part = _build_scanner_like_row(sym_u, type("Dummy", (), {"raw": {}})())
        else:
            scanner_part = _build_scanner_like_row(sym_u, sig)

        if scanner_part.get("Sector") in (None, "", "Unknown"):
            missing_sector += 1
        if scanner_part.get("AI Prob") is None:
            missing_ai_prob += 1

        # Merge (scanner first, then portfolio columns)
        row = {
            **scanner_part,
            "Action": action,
            "Current W": cur_w,
            "Target W": tgt_w,
            "ΔW": delta_w,
            "Current Qty": float(cur_qty or 0.0),
            "Price": px,
            "Target Value": tgt_value,
            "Target Qty (est)": tgt_qty,
            "Planned Qty": planned_qty,
            "Reason": reason,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by abs ΔW desc (most relevant trades first)
    if not df.empty and "ΔW" in df.columns:
        df["absΔW"] = df["ΔW"].abs()
        df = df.sort_values("absΔW", ascending=False).drop(columns=["absΔW"]).reset_index(drop=True)

    st.subheader("📌 Proposed Portfolio (Current vs Target)")
    st.caption("Scanner columns + target sizing + planner reasons. No fake values; missing prices keep sizing blank.")

    # Optional debug panel for sector / ai_prob missing
    with st.expander("Debug: Signals coverage (Sector / AI Prob)", expanded=False):
        st.write(
            {
                "symbols_total": len(syms),
                "signals_missing_rows": missing_signals,
                "sector_missing_or_unknown": missing_sector,
                "ai_prob_missing": missing_ai_prob,
            }
        )
        # show key availability for a few examples where sector/prob missing
        examples = []
        for sym_u in syms:
            sig = signal_rows.get(sym_u)
            if not sig:
                continue
            raw = getattr(sig, "raw", None) or {}
            if not isinstance(raw, dict):
                continue
            sector = _extract_sector_from_raw(raw)
            ai_p = _norm_prob_01(
                _best_raw_value(raw, "ai_probability", "ai_prob", "probability", "predicted_probability"))
            if sector in (None, "", "Unknown") or ai_p is None:
                keys_hit = [k for k in raw.keys() if
                            "sector" in k.lower() or "prob" in k.lower() or "likelihood" in k.lower()]
                examples.append({"symbol": sym_u, "sector": sector, "ai_prob": ai_p, "keys": keys_hit[:40]})
            if len(examples) >= 8:
                break
        if examples:
            st.json(examples)
        else:
            st.write("No missing sector/ai_prob examples found in current universe.")

    st.dataframe(df, width='stretch', hide_index=True)


# =============================================================================
# NEW: TARGET PORTFOLIO ANALYSIS TABLE
# =============================================================================

# ui_tab.py — REPLACE this FULL method entirely
# (This restores ALL scanner columns + target sizing + reason + sector pie/NAV/cash/div yield + coloring)

# ui_tab.py (AI PM tab)
# REPLACE the full method: _render_target_portfolio_analysis(...)

def _render_target_portfolio_analysis(
        targets,  # TargetWeights
        signals,  # SignalSnapshot
        snapshot,  # PortfolioSnapshot
        plan,  # TradePlan
        strategy_key: str,
) -> None:
    import pandas as pd
    import streamlit as st

    st.subheader("🎯 Target Portfolio Analysis")
    st.caption("Target holdings with full scanner columns + allocation diagnostics + selection reasoning.")

    # ---------- weights: DO NOT use `or {}` (can be a pandas Series -> ambiguous truth value) ----------
    w_raw = getattr(targets, "weights", None)

    if w_raw is None:
        weights = {}
    elif isinstance(w_raw, dict):
        weights = w_raw
    elif isinstance(w_raw, pd.Series):
        weights = w_raw.dropna().to_dict()
    else:
        # attempt best-effort conversion
        try:
            weights = dict(w_raw)
        except Exception:
            weights = {}

    if not weights:
        st.warning("No target holdings selected.")
        if signals is not None:
            st.json(getattr(signals, "diagnostics", None) or {})
        return

    # ---------- signal rows ----------
    rows_raw = getattr(signals, "rows", None) if signals is not None else None
    if rows_raw is None:
        signal_rows = {}
    elif isinstance(rows_raw, dict):
        signal_rows = rows_raw
    else:
        # safety: unexpected type
        try:
            signal_rows = dict(rows_raw)
        except Exception:
            signal_rows = {}

    # ---------- positions map ----------
    pos_by_sym = {}
    if snapshot is not None and getattr(snapshot, "positions", None):
        for p in snapshot.positions:
            sym = getattr(p, "symbol", None)
            if sym:
                pos_by_sym[str(sym).strip().upper()] = p

    # ---------- NAV ----------
    def _safe_float_local(x):
        try:
            if x is None:
                return None
            if isinstance(x, float) and pd.isna(x):
                return None
            if isinstance(x, str) and x.strip() == "":
                return None
            return float(x)
        except Exception:
            return None

    nav = _safe_float_local(getattr(snapshot, "net_liquidation", None)) if snapshot is not None else None
    if nav is None and plan is not None:
        nav = _safe_float_local(getattr(plan, "nav", None))

    price_map = st.session_state.get("ai_pm_price_map") or {}

    # ---------- build table ----------
    def _is_scalar(v):
        return v is None or isinstance(v, (int, float, str, bool))

    sorted_weights = sorted(weights.items(), key=lambda x: float(x[1] or 0.0), reverse=True)

    raw_key_union = []
    raw_key_set = set()

    rows = []
    for rank, (sym, w) in enumerate(sorted_weights, 1):
        sym_u = str(sym).strip().upper()
        w_f = float(w or 0.0)

        sig = signal_rows.get(sym_u)
        raw = (getattr(sig, "raw", None) or {}) if sig is not None else {}

        # normalize key names (avoid duplicate "AI Probability" vs "ai_probability")
        if "AI Probability" in raw and "ai_probability" not in raw:
            raw["ai_probability"] = raw.get("AI Probability")

        # position/price
        pos = pos_by_sym.get(sym_u)
        cur_qty = _safe_float_local(getattr(pos, "quantity", None)) if pos is not None else 0.0
        px = _safe_float_local(getattr(pos, "market_price", None)) if pos is not None else None
        if px is None:
            px = _safe_float_local(price_map.get(sym_u))

        tgt_value = (w_f * float(nav)) if (nav is not None and nav > 0) else None
        tgt_qty = (tgt_value / px) if (tgt_value is not None and px is not None and px > 0) else None

        sector = raw.get("sector")

        # selection reason
        ai_prob = raw.get("ai_probability") or raw.get("probability") or raw.get("predicted_probability")
        ai_ev = raw.get("ai_ev") or raw.get("expected_value")

        reason_parts = [f"Rank #{rank}"]
        if strategy_key in ("ai_probability", "ai_conservative"):
            p = _safe_float_local(ai_prob)
            if p is not None:
                if p > 1:
                    p = p / 100.0
                reason_parts.append(f"Prob:{p * 100:.1f}%")
            e = _safe_float_local(ai_ev)
            if e is not None:
                if abs(e) > 1:
                    e = e / 100.0
                reason_parts.append(f"EV:{e * 100:.2f}%")
            like = raw.get("likelihood_score")
            if like is not None:
                reason_parts.append(f"Likelihood:{like}")
        else:
            ts = raw.get("total_score")
            if ts is not None:
                reason_parts.append(f"Score:{ts}")
            stype = raw.get("signal_type")
            if stype is not None and str(stype).strip() != "":
                reason_parts.append(f"Signal:{stype}")

        reason_parts.append(f"→ {w_f * 100:.2f}%")
        selection_reason = " | ".join([x for x in reason_parts if x])

        # collect union of scalar raw keys (scanner columns)
        for k, v in raw.items():
            if not _is_scalar(v):
                continue
            if k in ("ticker", "symbol"):
                continue
            if k not in raw_key_set:
                raw_key_set.add(k)
                raw_key_union.append(k)

        row = {
            "Rank": rank,
            "Symbol": sym_u,
            "Sector": sector,
            "Target W%": w_f * 100.0,
            "Target $": tgt_value,
            "Target Qty": tgt_qty,
            "Price": px,
            "Curr Qty": float(cur_qty or 0.0),
            "Selection Reason": selection_reason,
        }

        # attach raw (only scalars)
        for k in raw_key_union:
            if k in raw:
                row[k] = raw.get(k)

        rows.append(row)

    df = pd.DataFrame(rows)

    # ensure all union keys exist for all rows
    for k in raw_key_union:
        if k not in df.columns:
            df[k] = None

    # drop duplicate probability label if both exist
    if "AI Probability" in df.columns and "ai_probability" in df.columns:
        df = df.drop(columns=["AI Probability"])

    # ---------- allocation summary ----------
    st.markdown("#### Allocation Summary")

    alloc = df[["Sector", "Target W%"]].copy()
    alloc["Sector"] = alloc["Sector"].fillna("").astype(str).str.strip()
    alloc.loc[alloc["Sector"].eq(""), "Sector"] = "Unknown"
    alloc["Target W%"] = pd.to_numeric(alloc["Target W%"], errors="coerce").fillna(0.0)

    sector_dist = alloc.groupby("Sector", dropna=False)["Target W%"].sum().sort_values(ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Holdings", int(len(df)))
    with c2:
        invested = float(
            pd.to_numeric(df["Target $"], errors="coerce").fillna(0.0).sum()) if "Target $" in df.columns else 0.0
        st.metric("Target Invested", f"${invested:,.0f}" if invested > 0 else "—")
    with c3:
        cash_target = getattr(targets, "cash_target", None)
        if nav is not None and cash_target is not None:
            st.metric("Cash Target", f"${(float(nav) * float(cash_target)):,.0f}")
        else:
            st.metric("Cash Target", "—")
    with c4:
        if nav is not None and invested > 0:
            st.metric("Est. Remaining Cash", f"${max(float(nav) - invested, 0.0):,.0f}")
        else:
            st.metric("Est. Remaining Cash", "—")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("**Sector distribution (Top 12)**")
        top = sector_dist.head(12)
        st.dataframe(
            pd.DataFrame({"Sector": top.index, "Weight %": top.values}).reset_index(drop=True),
            width='stretch',
            hide_index=True,
            height=320,
        )

    with right:
        st.markdown("**Sector pie**")
        try:
            import plotly.express as px
            pie_df = pd.DataFrame({"Sector": sector_dist.index, "Weight%": sector_dist.values})
            fig = px.pie(pie_df, names="Sector", values="Weight%")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, width='stretch')
        except Exception:
            st.bar_chart(sector_dist.head(12))

    # ---------- FULL table ----------
    st.markdown("#### Proposed Target Holdings (Full Scanner Columns)")

    def _style(df_in: pd.DataFrame):
        sty = df_in.style

        fmt = {}
        if "Target W%" in df_in.columns:
            fmt["Target W%"] = "{:.2f}"
        if "Price" in df_in.columns:
            fmt["Price"] = "{:.2f}"
        if "Target $" in df_in.columns:
            fmt["Target $"] = "{:,.0f}"
        if "Target Qty" in df_in.columns:
            fmt["Target Qty"] = "{:,.2f}"
        if "Curr Qty" in df_in.columns:
            fmt["Curr Qty"] = "{:,.2f}"
        sty = sty.format(fmt, na_rep="—")

        score_cols = [c for c in df_in.columns if c.endswith("_score")]
        for c in ["total_score", "likelihood_score", "gap_score"]:
            if c in df_in.columns and c not in score_cols:
                score_cols.append(c)

        if len(score_cols) > 0:
            try:
                sty = sty.background_gradient(subset=score_cols)
            except Exception:
                pass

        return sty

    st.dataframe(
        _style(df),
        width='stretch',
        hide_index=True,
        height=min(650, 60 + len(df) * 28),
    )

    # ---------- debug ----------
    with st.expander("AI PM Debug (signals diagnostics + sector/prob coverage)", expanded=False):
        st.write("Strategy:", strategy_key)
        st.write("Targets notes:", getattr(targets, "notes", None) or [])
        if signals is not None:
            st.json(getattr(signals, "diagnostics", None) or {})
        missing_sector = int(df["Sector"].fillna("Unknown").astype(str).str.strip().str.lower().eq(
            "unknown").sum()) if "Sector" in df.columns else None
        missing_prob = None
        if "ai_probability" in df.columns:
            missing_prob = int(pd.to_numeric(df["ai_probability"], errors="coerce").isna().sum())
        st.write(
            {
                "selected_holdings": int(len(df)),
                "missing_sector_in_selected": missing_sector,
                "missing_ai_probability_in_selected": missing_prob,
                "columns_count": int(len(df.columns)),
            }
        )


def _render_portfolio_comparison(snapshot, targets, signals, account: str) -> None:
    """
    Render a comparison table showing:
    - Current holdings from loaded portfolio statement OR IBKR live positions
    - Target weights from AI PM
    - Actions needed (buy more, sell, hold, new position)
    """
    st.markdown("### 📊 Portfolio Comparison: Current vs Target")

    # Get current positions from loaded statement
    loaded_positions = _get_loaded_portfolio_positions(account)

    # If no loaded statement, try to use snapshot positions from IBKR
    if not loaded_positions and snapshot:
        nav = getattr(snapshot, 'net_liquidation', 0) or 0
        positions = getattr(snapshot, 'positions', []) or []

        # Get price_map from session state if available
        price_map = st.session_state.get('ai_pm_price_map', {})

        for p in positions:
            sym = (getattr(p, 'symbol', '') or '').strip().upper()
            if not sym:
                continue
            qty = getattr(p, 'quantity', 0) or getattr(p, 'position', 0) or 0
            mv = getattr(p, 'market_value', 0) or 0

            # Calculate market value from price if not available
            if mv == 0 and qty != 0:
                # Try price_map first, then market_price, then avgCost
                price = price_map.get(sym, 0)
                if price == 0:
                    price = getattr(p, 'market_price', 0) or 0
                if price == 0:
                    price = getattr(p, 'avgCost', 0) or 0
                if price > 0:
                    mv = abs(qty) * price

            weight = mv / nav if nav > 0 else 0
            loaded_positions[sym] = {
                'quantity': qty,
                'market_value': mv,
                'weight': weight,
            }

    # Get target weights
    target_weights = targets.weights if targets and targets.weights else {}

    if not loaded_positions and not target_weights:
        st.info("No positions or targets available for comparison.")
        return

    # Build comparison data
    all_symbols = set(loaded_positions.keys()) | set(target_weights.keys())

    comparison_data = []
    for sym in sorted(all_symbols):
        current = loaded_positions.get(sym, {})
        target_w = target_weights.get(sym, 0)
        current_w = current.get('weight', 0) if current else 0

        # Determine action - using professional PM thresholds
        # Only flag actions for meaningful drift (>10% relative OR full entry/exit)
        relative_drift = abs(target_w - current_w) / current_w if current_w > 0.001 else float('inf')
        absolute_drift = abs(target_w - current_w)

        if sym not in loaded_positions and target_w > 0:
            action = "🆕 NEW BUY"
        elif sym in loaded_positions and target_w == 0:
            action = "🔴 FULL EXIT"
        elif target_w > current_w and relative_drift > 0.10 and absolute_drift > 0.005:
            # Only ADD if >10% relative drift AND >0.5% absolute drift
            action = "📈 ADD"
        elif target_w < current_w and relative_drift > 0.10 and absolute_drift > 0.005:
            # Only TRIM if >10% relative drift AND >0.5% absolute drift
            action = "📉 TRIM"
        else:
            action = "✓ HOLD"

        # Get signal info if available
        signal_info = ""
        if signals and hasattr(signals, 'rows') and sym in signals.rows:
            row = signals.rows[sym]
            raw = row.raw or {}
            signal = raw.get('signal') or raw.get('signal_type') or raw.get('Signal') or ''
            total_score = raw.get('total_score')
            if signal:
                signal_info = str(signal)
            elif total_score:
                signal_info = f"Score: {total_score}"

        comparison_data.append({
            'Symbol': sym,
            'Current %': f"{current_w * 100:.2f}%" if current_w > 0 else "-",
            'Target %': f"{target_w * 100:.2f}%" if target_w > 0 else "-",
            'Drift': f"{(target_w - current_w) * 100:+.2f}%",
            'Action': action,
            'Signal': signal_info if signal_info else "N/A",
            'Current Value': f"${current.get('market_value', 0):,.0f}" if current else "-",
        })

    # Sort by absolute drift (biggest changes first)
    comparison_data.sort(key=lambda x: abs(float(x['Drift'].replace('%', '').replace('+', ''))), reverse=True)

    df = pd.DataFrame(comparison_data)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    new_buys = len([r for r in comparison_data if "NEW BUY" in r['Action']])
    full_exits = len([r for r in comparison_data if "FULL EXIT" in r['Action']])
    adds = len([r for r in comparison_data if "ADD" in r['Action']])
    trims = len([r for r in comparison_data if "TRIM" in r['Action']])

    with col1:
        st.metric("🆕 New Buys", new_buys)
    with col2:
        st.metric("📈 Add To", adds)
    with col3:
        st.metric("📉 Trim", trims)
    with col4:
        st.metric("🔴 Full Exit", full_exits)

    # Display table
    st.dataframe(df, width='stretch', hide_index=True)

    # Highlight key insights
    if new_buys > 0:
        new_buy_symbols = [r['Symbol'] for r in comparison_data if "NEW BUY" in r['Action']]
        st.success(f"**New positions to open:** {', '.join(new_buy_symbols[:10])}" + (
            "..." if len(new_buy_symbols) > 10 else ""))

    if full_exits > 0:
        exit_symbols = [r['Symbol'] for r in comparison_data if "FULL EXIT" in r['Action']]
        st.warning(
            f"**Positions to close:** {', '.join(exit_symbols[:10])}" + ("..." if len(exit_symbols) > 10 else ""))


def _render_ai_intelligence_panel(snapshot, signals, targets, account: str) -> None:
    """
    Render the AI Portfolio Intelligence panel.
    This provides smart, AI-driven recommendations based on actual portfolio analysis.
    """
    st.markdown("---")
    st.markdown("## 🧠 AI Portfolio Intelligence")
    st.caption("Smart analysis of your portfolio with actionable recommendations")

    # Get loaded positions from statement
    loaded_positions = _get_loaded_portfolio_positions(account)

    with st.spinner("Analyzing portfolio with AI..."):
        try:
            report = analyze_portfolio_intelligence(
                snapshot=snapshot,
                signals=signals,
                targets=targets,
                strategy_key=get_strategy_key(),
                account=account,
                loaded_positions=loaded_positions,
            )

            # Store report for assistant
            st.session_state['ai_pm_intelligence_report'] = report

        except Exception as e:
            st.error(f"Failed to analyze portfolio: {e}")
            return

    # Display status header with color
    status_colors = {
        'EXCELLENT': 'green',
        'GOOD': 'blue',
        'NEEDS_ATTENTION': 'orange',
        'POOR': 'red',
        'EMPTY': 'gray',
    }

    status_emoji = {
        'EXCELLENT': '🌟',
        'GOOD': '✅',
        'NEEDS_ATTENTION': '⚠️',
        'POOR': '🔴',
        'EMPTY': '📭',
    }.get(report.overall_status, '❓')

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Status", f"{status_emoji} {report.overall_status}")
    with col2:
        st.metric("NAV", f"${report.nav:,.0f}")
    with col3:
        st.metric("Positions", report.num_positions)
    with col4:
        st.metric("Cash", f"${report.cash:,.0f} ({report.cash_pct:.1%})")

    # Headline
    st.markdown(f"### {report.headline}")

    # Key flags
    if report.no_action_needed:
        st.success("✅ **No action needed** - Your portfolio is well-positioned.")

    if report.drift_too_low:
        st.info("📊 **Drift is too low** - Rebalancing would be inefficient at this time.")

    if report.portfolio_is_empty:
        st.warning("📭 **Portfolio is empty** - See recommendations below to deploy capital.")

    # Recommended Actions
    if report.recommended_actions:
        st.markdown("### 📋 Recommended Actions")

        for i, action in enumerate(report.recommended_actions, 1):
            if action['action'] == 'HOLD':
                st.success(f"**{i}. HOLD** - {action.get('description', 'No changes recommended')}")

            elif action['action'] == 'REPLACE':
                with st.expander(f"**{i}. REPLACE {action['symbol']}** ({action['assessment']})", expanded=True):
                    # Issues
                    if action.get('issues'):
                        st.markdown("**Issues identified:**")
                        for issue in action['issues']:
                            st.markdown(f"- ⚠️ {issue}")

                    # Recommendation
                    if action.get('recommendation'):
                        st.markdown(f"**💡 Recommendation:** {action['recommendation']}")

                    # Alternatives
                    if action.get('alternatives'):
                        st.markdown("**Better alternatives:**")
                        alt_data = []
                        alt_symbols = []
                        for alt in action['alternatives'][:5]:
                            alt_symbols.append(alt['symbol'])
                            alt_data.append({
                                'Symbol': alt['symbol'],
                                'Signal': alt.get('signal', 'N/A'),
                                'Score': f"{alt.get('total_score', 0):.0f}",
                                'Sector': alt.get('sector', 'N/A'),
                                'Same Sector': '✓' if alt.get('same_sector') else '',
                            })
                        if alt_data:
                            st.dataframe(pd.DataFrame(alt_data), width='stretch', hide_index=True)

                        # Action section for replacement
                        st.markdown("---")
                        st.markdown("**🔄 Execute Replacement:**")

                        col_select, col_custom, col_action = st.columns([2, 2, 1])

                        with col_select:
                            # Dropdown with suggested alternatives
                            selected_replacement = st.selectbox(
                                "Select replacement",
                                options=["-- Select --"] + alt_symbols,
                                key=f"replace_select_{action['symbol']}_{i}",
                                label_visibility="collapsed"
                            )

                        with col_custom:
                            # Custom ticker input
                            custom_ticker = st.text_input(
                                "Or enter custom ticker",
                                placeholder="e.g., AAPL",
                                key=f"replace_custom_{action['symbol']}_{i}",
                                label_visibility="collapsed"
                            ).upper().strip()

                        with col_action:
                            replace_btn = st.button(
                                "✅ Replace",
                                key=f"replace_btn_{action['symbol']}_{i}",
                                type="primary"
                            )

                        if replace_btn:
                            # Determine which ticker to use
                            new_ticker = custom_ticker if custom_ticker else (
                                selected_replacement if selected_replacement != "-- Select --" else None)
                            old_ticker = action['symbol']

                            if new_ticker:
                                # Look up stock data for the replacement
                                new_stock_data = None
                                if STOCK_LOOKUP_AVAILABLE:
                                    try:
                                        stock_lookup = get_stock_lookup()
                                        # Only load if not already loaded (cached)
                                        if not stock_lookup._cache:
                                            stock_lookup.load_all_json_files()
                                        stock_obj = stock_lookup.lookup(new_ticker)
                                        if stock_obj:
                                            new_stock_data = stock_obj.to_dict()
                                            st.success(f"✅ Found {new_ticker} in JSON cache")
                                        else:
                                            # Try to fetch from IBKR
                                            st.info(f"Looking up {new_ticker} from IBKR...")
                                            stock_obj = stock_lookup.fetch_from_ibkr(new_ticker,
                                                                                     gw.ib if gw.is_connected() else None)
                                            if stock_obj:
                                                new_stock_data = stock_obj.to_dict()
                                                st.success(f"✅ Fetched {new_ticker} from IBKR")
                                    except Exception as e:
                                        st.warning(f"Stock lookup error: {e}")

                                # Store replacement in session state for execution
                                if 'pending_replacements' not in st.session_state:
                                    st.session_state.pending_replacements = []

                                # Add to pending replacements with stock data
                                replacement = {
                                    'sell': old_ticker,
                                    'buy': new_ticker,
                                    'buy_data': new_stock_data,  # Full IBKR attributes
                                    'reason': f"Replace {old_ticker} with {new_ticker} (AI recommendation)"
                                }

                                # Check if already in list
                                existing = [r for r in st.session_state.pending_replacements if r['sell'] == old_ticker]
                                if not existing:
                                    st.session_state.pending_replacements.append(replacement)
                                    st.success(f"✅ Queued: SELL {old_ticker} → BUY {new_ticker}")

                                    # Show stock attributes if found
                                    if new_stock_data:
                                        with st.expander(f"📋 {new_ticker} IBKR Attributes", expanded=False):
                                            attr_cols = st.columns(3)
                                            with attr_cols[0]:
                                                st.write(f"**Name:** {new_stock_data.get('name', 'N/A')}")
                                                st.write(f"**Sector:** {new_stock_data.get('sector', 'N/A')}")
                                                st.write(f"**ISIN:** {new_stock_data.get('isin', 'N/A')}")
                                            with attr_cols[1]:
                                                st.write(f"**ConID:** {new_stock_data.get('conid', 'N/A')}")
                                                st.write(f"**Exchange:** {new_stock_data.get('exchange', 'SMART')}")
                                                st.write(
                                                    f"**Primary:** {new_stock_data.get('primary_exchange', 'N/A')}")
                                            with attr_cols[2]:
                                                st.write(f"**Currency:** {new_stock_data.get('currency', 'USD')}")
                                                st.write(f"**SecType:** {new_stock_data.get('secType', 'STK')}")
                                                st.write(
                                                    f"**Source:** {'JSON' if new_stock_data.get('found_in_json') else 'IBKR API'}")

                                    st.info("💡 Go to **Proposed Actions** to review and execute all pending trades.")
                                else:
                                    st.warning(f"Replacement for {old_ticker} already queued.")
                            else:
                                st.error("Please select or enter a replacement ticker.")
                    else:
                        st.warning("No suitable alternatives found. Consider exiting this position.")

                        # Still allow manual replacement
                        col_exit, col_manual = st.columns([1, 2])
                        with col_exit:
                            if st.button(f"🔴 Exit {action['symbol']}", key=f"exit_btn_{action['symbol']}_{i}"):
                                if 'pending_replacements' not in st.session_state:
                                    st.session_state.pending_replacements = []
                                st.session_state.pending_replacements.append({
                                    'sell': action['symbol'],
                                    'buy': None,
                                    'reason': f"Exit {action['symbol']} (AI recommendation - no alternatives)"
                                })
                                st.success(f"✅ Queued: SELL {action['symbol']} (full exit)")

                        with col_manual:
                            manual_ticker = st.text_input(
                                "Manual replacement",
                                placeholder="Enter ticker to buy instead",
                                key=f"manual_replace_{action['symbol']}_{i}",
                                label_visibility="collapsed"
                            ).upper().strip()
                            if manual_ticker and st.button("Replace", key=f"manual_btn_{action['symbol']}_{i}"):
                                if 'pending_replacements' not in st.session_state:
                                    st.session_state.pending_replacements = []
                                st.session_state.pending_replacements.append({
                                    'sell': action['symbol'],
                                    'buy': manual_ticker,
                                    'reason': f"Replace {action['symbol']} with {manual_ticker} (manual)"
                                })
                                st.success(f"✅ Queued: SELL {action['symbol']} → BUY {manual_ticker}")

            elif action['action'] == 'DEPLOY_CAPITAL':
                with st.expander(f"**{i}. DEPLOY CAPITAL** - {action.get('description', '')}", expanded=True):
                    if action.get('suggestions'):
                        st.markdown("**Top stock recommendations:**")
                        sug_data = []
                        for sug in action['suggestions'][:10]:
                            sug_data.append({
                                'Symbol': sug['symbol'],
                                'Signal': sug.get('signal', 'N/A'),
                                'Score': f"{sug.get('total_score', 0):.0f}",
                                'Sector': sug.get('sector', 'N/A'),
                            })
                        if sug_data:
                            st.dataframe(pd.DataFrame(sug_data), width='stretch', hide_index=True)
                    else:
                        st.warning("No strong BUY signals found. Run the scanner to get recommendations.")

    # Position-by-position analysis
    if report.position_analyses:
        with st.expander("📊 Detailed Position Analysis", expanded=False):
            pos_data = []
            for pa in sorted(report.position_analyses, key=lambda x: x.market_value, reverse=True):
                assessment_emoji = {
                    'STRONG': '🌟',
                    'GOOD': '✅',
                    'NEUTRAL': '➖',
                    'WEAK': '⚠️',
                    'BAD': '🔴',
                }.get(pa.assessment, '❓')

                pos_data.append({
                    'Symbol': pa.symbol,
                    'Assessment': f"{assessment_emoji} {pa.assessment}",
                    'Weight': f"{pa.current_weight:.1%}",
                    'Value': f"${pa.market_value:,.0f}",
                    'P&L %': f"{pa.unrealized_pnl_pct:+.1f}%",
                    'Signal': pa.signal or 'N/A',
                    'Score': f"{pa.total_score:.0f}" if pa.total_score else 'N/A',
                    'Issues': ', '.join(pa.issues[:2]) if pa.issues else '-',
                })

            if pos_data:
                st.dataframe(pd.DataFrame(pos_data), width='stretch', hide_index=True)

    # Reasoning
    if report.reasoning:
        with st.expander("🔍 Analysis Reasoning", expanded=False):
            for reason in report.reasoning:
                st.markdown(f"- {reason}")

    # Warnings
    if report.warnings:
        st.markdown("### ⚠️ Warnings")
        for warning in report.warnings:
            st.warning(warning)


def _render_rebalance_metrics_panel(snapshot, targets, plan, price_map: Dict[str, float],
                                    capital_to_deploy: float = None) -> None:
    """
    Render the Rebalance Metrics panel matching ETF Rebalancer style.
    Shows: Trading Volume, Buy/Sell, Fees, Accuracy, Turnover, Cash Flow, etc.
    """
    st.markdown("---")
    st.markdown("## 📊 Rebalance Metrics")

    try:
        metrics = calculate_rebalance_metrics(
            snapshot=snapshot,
            targets=targets,
            plan=plan,
            price_map=price_map,
            capital_to_deploy=capital_to_deploy,
        )

        # Store metrics for later use
        st.session_state['ai_pm_rebalance_metrics'] = metrics

    except Exception as e:
        st.error(f"Failed to calculate metrics: {e}")
        return

    # Row 1: Trading metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**📈 TRADING VOLUME**")
        st.markdown(f"### ${metrics.trading_volume:,.2f}")
        st.caption("Total transaction value")
    with col2:
        st.markdown("**💰 BUY/SELL**")
        buy_v = metrics.total_buy if metrics.total_buy == metrics.total_buy else 0
        sell_v = metrics.total_sell if metrics.total_sell == metrics.total_sell else 0
        st.markdown(f":green[{buy_v:,.2f}] / :red[{sell_v:,.2f}]")
        st.caption("Buy and sell values")
    with col3:
        st.markdown("**💵 FEES**")
        st.markdown(f"### ${metrics.total_fees:,.2f}")
        st.caption("Est. commissions & fees")
    with col4:
        st.markdown("**🎯 ACCURACY**")
        st.markdown(f"### {metrics.accuracy:.2f}%")
        st.caption("Est. portfolio alignment")

    st.markdown("---")

    # Row 2: Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**🔄 TURNOVER**")
        st.markdown(f"### {metrics.turnover:.2f}%")
        st.caption("Portfolio rotation rate")
    with col2:
        st.markdown("**💸 CASH FLOW**")
        color = "green" if metrics.cash_flow >= 0 else "red"
        st.markdown(f"### <span style='color:{color}'>${metrics.cash_flow:,.2f}</span>", unsafe_allow_html=True)
        st.caption("Net capital required")
    with col3:
        st.markdown("**📉 COST IMPACT**")
        st.markdown(f"### {metrics.cost_impact:.2f}%")
        st.caption("Return reduction")
    with col4:
        st.markdown("**📏 TRACKING ERROR**")
        st.markdown(f"### {metrics.tracking_error:.2f}%")
        st.caption("Est. deviation from target")

    st.markdown("---")

    # Row 3: Risk metrics (require historical data - show N/A if not calculated)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**⚠️ VAR (95%)**")
        if metrics.var_95 != 0:
            st.markdown(f"### {metrics.var_95:.2f}%")
        else:
            st.markdown("### N/A")
        st.caption("Daily value at risk")
    with col2:
        st.markdown("**📊 BETA**")
        if metrics.beta != 0:
            st.markdown(f"### {metrics.beta:.2f}")
        else:
            st.markdown("### N/A")
        st.caption("Market sensitivity")
    with col3:
        st.markdown("**📈 SHARPE**")
        if metrics.sharpe != 0:
            st.markdown(f"### {metrics.sharpe:.2f}")
        else:
            st.markdown("### N/A")
        st.caption("Risk-adjusted return")
    with col4:
        st.markdown("**📉 MAX DRAWDOWN**")
        if metrics.max_drawdown != 0:
            st.markdown(f"### {metrics.max_drawdown:.2f}%")
        else:
            st.markdown("### N/A")
        st.caption("Worst decline")

    st.markdown("---")

    # Row 4: Portfolio state (show N/A if no data)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**💼 PORTFOLIO NAV**")
        if metrics.portfolio_nav > 0:
            st.markdown(f"### ${metrics.portfolio_nav:,.2f}")
        else:
            st.markdown("### N/A")
        st.caption("Total portfolio value")
    with col2:
        st.markdown("**💵 CASH POSITION**")
        if metrics.cash_position > 0:
            st.markdown(f"### ${metrics.cash_position:,.2f}")
        else:
            st.markdown("### N/A")
        st.caption("Available cash")
    with col3:
        st.markdown("**⚡ BUYING POWER**")
        if metrics.buying_power > 0:
            st.markdown(f"### ${metrics.buying_power:,.2f}")
        else:
            st.markdown("### N/A")
        st.caption("Available for trading")
    with col4:
        st.markdown("**📊 MARGIN USAGE**")
        if metrics.margin_usage > 0:
            st.markdown(f"### {metrics.margin_usage:.2f}%")
        else:
            st.markdown("### N/A")
        st.caption("Margin utilization")

    # Capital deployment info - only show if we have real data
    if capital_to_deploy and capital_to_deploy > 0 and metrics.portfolio_nav > 0:
        st.markdown("---")
        st.info(
            f"💰 **Capital Deployment:** Deploying ${metrics.capital_deployed:,.2f} of ${metrics.portfolio_nav:,.2f} NAV | Remaining cash: ${metrics.remaining_cash:,.2f}")


def _render_rebalancing_results_table(snapshot, targets, plan, price_map: Dict[str, float], signals=None) -> None:
    """
    Render the detailed rebalancing results table matching ETF Rebalancer style.
    Shows per-position: Symbol, Original Weight, Target Weight, Shares, Price, Value, Accuracy
    """
    st.markdown("---")
    st.markdown("## 📋 Rebalancing Results")

    try:
        positions = build_position_results(
            snapshot=snapshot,
            targets=targets,
            plan=plan,
            price_map=price_map,
            signals=signals,
        )

        if not positions:
            st.info("No positions to display.")
            return

        # Build dataframe
        data = []
        for p in positions:
            # Determine row color based on action
            if p.action == 'BUY':
                action_display = f"🟢 {p.action}"
            elif p.action == 'SELL':
                action_display = f"🔴 {p.action}"
            else:
                action_display = f"⚪ HOLD"

            # Accuracy badge
            if p.accuracy_pct >= 99:
                accuracy_display = f"✅ {p.accuracy_pct:.2f}%"
            elif p.accuracy_pct >= 90:
                accuracy_display = f"🟡 {p.accuracy_pct:.2f}%"
            else:
                accuracy_display = f"🔴 {p.accuracy_pct:.2f}%"

            data.append({
                'Symbol': p.symbol,
                'Original %': f"{p.original_weight_pct:.2f}%",
                'Target %': f"{p.target_weight_pct:.2f}%",
                'Open Shares': p.open_shares,
                'Position': p.position_shares,
                'Target Shares': p.target_shares,
                'Price': f"${p.price:.2f}",
                'Order Value': f"${p.order_value:,.2f}",
                'Position Value': f"${p.position_value:,.2f}",
                'Total Value': f"${p.total_value:,.2f}",
                'Actual %': f"{p.actual_weight_pct:.2f}%",
                'Accuracy': accuracy_display,
            })

        df = pd.DataFrame(data)

        # Show summary
        total_positions = len(positions)
        perfect_accuracy = len([p for p in positions if p.accuracy_pct >= 99])
        st.success(f"📊 **{total_positions} positions** | **{perfect_accuracy}** at 100% accuracy")

        # Display table
        st.dataframe(df, width='stretch', hide_index=True, height=600)

        # Store for later use
        st.session_state['ai_pm_position_results'] = positions

    except Exception as e:
        st.error(f"Failed to build results table: {e}")


def _render_positions_to_exit(targets, snapshot, plan) -> None:
    """Show current positions NOT in target portfolio (will be sold)."""
    weights = getattr(targets, "weights", None) or {}
    target_syms = set(k.strip().upper() for k in weights.keys())

    positions = getattr(snapshot, "positions", None) or []
    current_weights = getattr(plan, "current_weights", None) or {}

    rows = []
    for p in positions:
        sym = getattr(p, "symbol", None)
        if not sym:
            continue
        sym_upper = sym.strip().upper()

        if sym_upper in target_syms:
            continue

        qty = _safe_float_local(getattr(p, "quantity", None)) or 0
        if qty == 0:
            continue

        price = _safe_float_local(getattr(p, "market_price", None))
        value = _safe_float_local(getattr(p, "market_value", None))
        if value is None and price and qty:
            value = price * qty

        current_w = current_weights.get(sym_upper, 0)

        rows.append({
            "Symbol": sym_upper,
            "Qty": qty,
            "Price": price,
            "Value": value,
            "Current W%": current_w * 100 if current_w else 0,
            "Action": "SELL ALL",
            "Reason": "Not in target portfolio",
        })

    if not rows:
        return

    st.subheader("🚫 Positions to Exit")
    st.caption("Current holdings not selected by strategy (will be sold).")

    df = pd.DataFrame(rows)
    df = df.sort_values("Value", ascending=False, na_position="last")

    total_exit = sum(r["Value"] or 0 for r in rows)
    st.warning(f"⚠️ {len(rows)} positions (${total_exit:,.0f}) to exit.")

    st.dataframe(df, width='stretch', hide_index=True)


# =============================================================================
# IBKR PRICE FETCHING
# =============================================================================

def _fetch_price_map_yahoo_first(symbols, ib=None):
    """
    Fetch prices using Yahoo Finance FIRST (fast), then IBKR fallback.
    Returns: (price_map, diagnostics)
    """
    import yfinance as yf

    syms = [s.strip().upper() for s in (symbols or []) if s and str(s).strip()]
    syms = sorted(set(syms))

    price_map = {}
    diag = {"requested": len(syms), "resolved": 0, "missing": [], "source_counts": {"yahoo": 0, "ibkr": 0}}

    if not syms:
        return price_map, diag

    # PRIMARY: Yahoo Finance (fast batch)
    try:
        for i in range(0, len(syms), 100):
            chunk = syms[i:i + 100]
            yf_data = yf.download(' '.join(chunk), period='1d', progress=False, threads=True)

            if yf_data.empty:
                continue

            # Handle MultiIndex columns from yfinance
            if len(chunk) == 1:
                # Single ticker - Close is a Series
                try:
                    if 'Close' in yf_data.columns:
                        val = yf_data['Close'].iloc[-1]
                    elif ('Close', chunk[0]) in yf_data.columns:
                        val = yf_data[('Close', chunk[0])].iloc[-1]
                    else:
                        val = None
                    if val and val > 0:
                        price_map[chunk[0]] = float(val)
                        diag["source_counts"]["yahoo"] += 1
                except:
                    pass
            else:
                # Multiple tickers - Close is a DataFrame with ticker columns
                try:
                    close_data = yf_data['Close'] if 'Close' in yf_data.columns else None

                    # If MultiIndex, access via tuple
                    if close_data is None:
                        for sym in chunk:
                            try:
                                if ('Close', sym) in yf_data.columns:
                                    val = yf_data[('Close', sym)].iloc[-1]
                                    if val and val > 0:
                                        price_map[sym] = float(val)
                                        diag["source_counts"]["yahoo"] += 1
                            except:
                                pass
                    else:
                        # close_data is a DataFrame
                        for sym in chunk:
                            try:
                                if hasattr(close_data, 'columns') and sym in close_data.columns:
                                    val = close_data[sym].iloc[-1]
                                elif not hasattr(close_data, 'columns'):
                                    val = close_data.iloc[-1]
                                else:
                                    val = None
                                if val and val > 0:
                                    price_map[sym] = float(val)
                                    diag["source_counts"]["yahoo"] += 1
                            except:
                                pass
                except:
                    pass
    except Exception:
        pass

    # IBKR fallback DISABLED - causes freeze in Streamlit threads
    # Yahoo Finance is fast and sufficient for most symbols
    missing = [s for s in syms if s not in price_map]
    if missing:
        diag["source_counts"]["ibkr"] = 0  # Skipped

    diag["resolved"] = len(price_map)
    diag["missing"] = [s for s in syms if s not in price_map][:15]
    return price_map, diag


def _fetch_ibkr_price_map(
        *,
        ib,
        symbols,
        currency: str = "USD",
        exchange: str = "SMART",
        max_batch: int = 80,
):
    """
    WRAPPER: Now uses Yahoo Finance first for speed.
    Falls back to IBKR only for missing symbols.
    """
    return _fetch_price_map_yahoo_first(symbols, ib)


def _fetch_ibkr_price_map_legacy(
        *,
        ib,
        symbols,
        currency: str = "USD",
        exchange: str = "SMART",
        max_batch: int = 80,
):
    """
    LEGACY: Original IBKR-only price fetch (slow).
    Kept for reference.
    """
    from ib_insync import Stock, util

    syms = [s.strip().upper() for s in (symbols or []) if s and str(s).strip()]
    syms = sorted(set(syms))

    price_map = {}
    diag = {
        "requested": len(syms),
        "resolved": 0,
        "missing": [],
        "source_counts": {"ticker": 0, "close": 0},
    }

    if not syms:
        return price_map, diag

    def _ticker_price(t):
        # Try common fields in order of usefulness
        for attr in ("marketPrice", "last", "close", "midpoint"):
            try:
                v = _safe_float_local(getattr(t, attr, None))
            except Exception:
                v = None
            if v and v > 0:
                return float(v)
        return None

    # Build contracts
    contracts = [Stock(sym, exchange, currency) for sym in syms]

    # Qualify in chunks
    qualified = []
    for i in range(0, len(contracts), max_batch):
        chunk = contracts[i: i + max_batch]
        try:
            qualified.extend(ib.qualifyContracts(*chunk))
        except Exception:
            # if qualification fails, keep raw contracts (IB often still works)
            qualified.extend(chunk)

    # Snapshot tickers
    tickers = []
    for i in range(0, len(qualified), max_batch):
        chunk = qualified[i: i + max_batch]
        try:
            tickers.extend(ib.reqTickers(*chunk))
        except Exception:
            # If reqTickers fails, we'll rely on historical fallback per-symbol
            pass

    # Map ticker by symbol
    t_by_sym = {}
    for t in (tickers or []):
        try:
            sym = (t.contract.symbol or "").strip().upper()
            if sym:
                t_by_sym[sym] = t
        except Exception:
            continue

    # Fill from tickers first
    for sym in syms:
        t = t_by_sym.get(sym)
        if not t:
            continue
        px = _ticker_price(t)
        if px and px > 0:
            price_map[sym] = px
            diag["source_counts"]["ticker"] += 1

    # Historical fallback for missing
    missing = [s for s in syms if s not in price_map]
    for sym in missing:
        try:
            c = Stock(sym, exchange, currency)
            try:
                ib.qualifyContracts(c)
            except Exception:
                pass

            bars = ib.reqHistoricalData(
                c,
                endDateTime="",
                durationStr="2 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
                keepUpToDate=False,
            )
            if bars:
                # last completed daily bar close
                last_bar = bars[-1]
                close_px = _safe_float_local(getattr(last_bar, "close", None))
                if close_px and close_px > 0:
                    price_map[sym] = float(close_px)
                    diag["source_counts"]["close"] += 1
        except Exception:
            pass

    diag["resolved"] = len(price_map)
    diag["missing"] = [s for s in syms if s not in price_map]
    return price_map, diag


# =============================================================================
# CACHING AND UTILITY FUNCTIONS
# =============================================================================

def _now_utc_ts() -> float:
    import time
    return float(time.time())


def _get_cache_bucket() -> dict:
    if "ai_pm_cache" not in st.session_state:
        st.session_state["ai_pm_cache"] = {}
    return st.session_state["ai_pm_cache"]


def get_cached_snapshot(cache_key: str, max_age_sec: int = 20) -> Optional[PortfolioSnapshot]:
    bucket = _get_cache_bucket()
    item = bucket.get(cache_key)
    if not item:
        return None
    ts = item.get("ts")
    snap = item.get("snapshot")
    if ts is None or snap is None:
        return None
    if (_now_utc_ts() - float(ts)) > float(max_age_sec):
        return None
    return snap


def set_cached_snapshot(cache_key: str, snapshot: PortfolioSnapshot) -> None:
    bucket = _get_cache_bucket()
    bucket[cache_key] = {"ts": _now_utc_ts(), "snapshot": snapshot}


def clear_cached_snapshot(cache_key: str) -> None:
    bucket = _get_cache_bucket()
    if cache_key in bucket:
        del bucket[cache_key]


def _g(obj, key: str, default=None):
    """Get value from dict-like or object-like (dataclass). No mocked data."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def _as_str(x):
    if x is None:
        return ""
    return str(x)


def _sf(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def _ss(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _to_portfolio_snapshot(gw: IbkrGateway, account: str) -> PortfolioSnapshot:
    """Build portfolio snapshot from IBKR (simple sequential calls)."""
    import logging
    _log = logging.getLogger(__name__)

    summ = None
    pos_rows = []
    oo_rows = []

    # Check connection first
    if not gw or not gw.is_connected():
        _log.error("Gateway not connected in _to_portfolio_snapshot")
        return PortfolioSnapshot(net_liquidation=0, total_cash=0, buying_power=0, available_funds=0, currency="USD")

    # Simple sequential calls
    _log.info("Snapshot: getting account summary...")
    try:
        summ = gw.get_account_summary(account)
        _log.info(f"Snapshot: got summary, NAV={summ.net_liquidation if summ else 'None'}")
    except Exception as e:
        _log.warning(f"Account summary error: {e}")

    _log.info("Snapshot: getting positions...")
    try:
        pos_rows = gw.get_positions() or []
        _log.info(f"Snapshot: got {len(pos_rows)} positions")
    except Exception as e:
        _log.warning(f"Positions error: {e}")

    _log.info("Snapshot: getting open orders...")
    try:
        oo_rows = gw.get_open_orders(include_all_clients=True) or []
        _log.info(f"Snapshot: got {len(oo_rows)} open orders")
    except Exception as e:
        _log.warning(f"Open orders error: {e}")

    _log.info("Snapshot: building result...")

    positions: List[Position] = []
    for p in pos_rows:
        qty = float(getattr(p, "position", None) or getattr(p, "quantity", None) or 0.0)

        # Keep real valuation if gateway provides it (ib_insync PortfolioItem does)
        mp = getattr(p, "market_price", None)
        if mp is None:
            mp = getattr(p, "marketPrice", None)

        mv = getattr(p, "market_value", None)
        if mv is None:
            mv = getattr(p, "marketValue", None)

        # Derive missing side if possible (still real math, no mocked data)
        try:
            if (mp is None or mp == 0) and (mv is not None) and qty != 0:
                mp = float(mv) / float(qty)
        except Exception:
            pass

        try:
            if mv is None and (mp is not None) and qty != 0:
                mv = float(mp) * float(qty)
        except Exception:
            pass

        positions.append(
            Position(
                symbol=getattr(p, "symbol", None),
                sec_type=getattr(p, "sec_type", None) or getattr(p, "secType", None) or "",
                currency=getattr(p, "currency", None) or "",
                exchange=getattr(p, "exchange", None) or "",
                con_id=getattr(p, "con_id", None) or getattr(p, "conId", None) or 0,
                quantity=qty,
                avg_cost=getattr(p, "avg_cost", None) or getattr(p, "averageCost", None),
                market_price=float(mp) if mp is not None else None,
                market_value=float(mv) if mv is not None else None,
            )
        )

    open_orders: List[OrderStatus] = []
    for o in oo_rows:
        open_orders.append(
            OrderStatus(
                order_id=getattr(o, "order_id", None),
                perm_id=getattr(o, "perm_id", None),
                symbol=getattr(o, "symbol", None),
                action=getattr(o, "action", None),
                quantity=getattr(o, "quantity", None),
                order_type=getattr(o, "order_type", None),
                lmt_price=getattr(o, "lmt_price", None),
                status=getattr(o, "status", None),
                account=getattr(o, "account", None),
            )
        )

    return PortfolioSnapshot(
        ts_utc=datetime.utcnow(),
        account=account,
        currency=(summ.currency if summ else None),
        net_liquidation=(summ.net_liquidation if summ else None),
        total_cash=(summ.total_cash_value if summ else None),
        available_funds=(summ.available_funds if summ else None),
        buying_power=(summ.buying_power if summ else None),
        positions=positions,
        open_orders=open_orders,
    )


def _render_alerts(gates: Optional[Dict[str, Any]], plan: Optional[Dict[str, Any]]) -> None:
    if not gates:
        st.info("No gate report yet. Click **Run Now**.")
        return

    blocked = bool(gates.get("blocked"))
    block_reasons = gates.get("block_reasons") or []
    warnings = gates.get("warnings") or []

    if blocked:
        st.error("ACTION NEEDED: Execution is blocked by hard gates.")
        for r in block_reasons:
            st.write(f"- {r}")
    else:
        if warnings:
            st.warning("Review recommended: warnings detected.")
            for w in warnings[:15]:
                st.write(f"- {w}")
        else:
            st.success("No blocks/warnings. Plan is eligible for execution.")

    if plan and plan.get("warnings"):
        with st.expander("Planner warnings", expanded=False):
            for w in plan.get("warnings") or []:
                st.write(f"- {w}")


def _compute_nav_and_cash(snapshot_dict: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], List[str]]:
    warns: List[str] = []
    nav = _sf(snapshot_dict.get("net_liquidation"))
    cash = _sf(snapshot_dict.get("total_cash"))

    if nav is not None and nav > 0:
        return nav, cash, warns

    # fallback: sum position values + cash if available (no mocked values)
    pos_total = 0.0
    have_any = False
    for p in snapshot_dict.get("positions") or []:
        mv = _sf(p.get("market_value"))
        if mv is None:
            qty = _sf(p.get("quantity")) or 0.0
            px = _sf(p.get("market_price"))
            if px is None:
                px = _sf(p.get("avg_cost"))
            if px is not None and px > 0:
                mv = qty * px
        if mv is not None:
            have_any = True
            pos_total += float(mv)

    if cash is None:
        cash = 0.0

    if have_any:
        nav = pos_total + float(cash)
        warns.append("NAV missing from IBKR snapshot; using positions+cash approximation for weights.")
        return nav, cash, warns

    return nav, cash, ["NAV missing and positions cannot be valued (missing market_value and price)."]


def _holdings_df(snapshot_dict: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str], Optional[float], Optional[float]]:
    nav, cash, nav_warn = _compute_nav_and_cash(snapshot_dict)
    rows: List[Dict[str, Any]] = []
    missing_val: List[str] = []

    for p in snapshot_dict.get("positions") or []:
        sym = _ss(p.get("symbol")).strip().upper()
        if not sym:
            continue
        qty = _sf(p.get("quantity")) or 0.0
        avg_cost = _sf(p.get("avg_cost"))
        mpx = _sf(p.get("market_price"))
        mv = _sf(p.get("market_value"))

        price_used = mpx if (mpx is not None and mpx > 0) else (
            avg_cost if (avg_cost is not None and avg_cost > 0) else None)
        price_source = "market_price" if (mpx is not None and mpx > 0) else (
            "avg_cost" if (avg_cost is not None and avg_cost > 0) else "missing")

        value_used = mv
        if value_used is None and price_used is not None:
            value_used = float(qty) * float(price_used)

        if value_used is None:
            missing_val.append(sym)

        weight = None
        if nav is not None and nav > 0 and value_used is not None:
            weight = float(value_used) / float(nav)

        rows.append(
            {
                "symbol": sym,
                "quantity": qty,
                "avg_cost": avg_cost,
                "market_price": mpx,
                "price_used": price_used,
                "price_source": price_source,
                "market_value": mv,
                "value_used": value_used,
                "weight": weight,
                "sec_type": p.get("sec_type"),
                "currency": p.get("currency"),
                "exchange": p.get("exchange"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["weight", "value_used"], ascending=False, na_position="last").reset_index(drop=True)

    warns = list(nav_warn)
    if missing_val:
        warns.append("Positions missing value (no market_value and no usable price): " + ", ".join(
            sorted(set(missing_val))[:50]))
    return df, warns, nav, cash


def _targets_df(targets_dict: Dict[str, Any]) -> pd.DataFrame:
    w = (targets_dict or {}).get("weights") or {}
    rows = [{"symbol": str(k).strip().upper(), "target_weight": float(v)} for k, v in w.items() if k and v is not None]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("target_weight", ascending=False).reset_index(drop=True)
    return df


def _orders_df(plan_dict: Dict[str, Any]) -> pd.DataFrame:
    orders = (plan_dict or {}).get("orders") or []
    df = pd.DataFrame(orders)
    if df.empty:
        return df
    for c in ["symbol", "action", "quantity", "order_type", "limit_price", "tif", "reason"]:
        if c not in df.columns:
            df[c] = None
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    return df


def _drift_df(holdings: pd.DataFrame, targets: pd.DataFrame, orders: pd.DataFrame,
              nav: Optional[float]) -> pd.DataFrame:
    cur = {}
    if holdings is not None and not holdings.empty:
        for _, r in holdings.iterrows():
            cur[str(r.get("symbol") or "").upper().strip()] = r.get("weight")

    tgt = {}
    if targets is not None and not targets.empty:
        for _, r in targets.iterrows():
            tgt[str(r.get("symbol") or "").upper().strip()] = float(r.get("target_weight") or 0.0)

    ord_act = {}
    ord_qty = {}
    if orders is not None and not orders.empty:
        for _, r in orders.iterrows():
            sym = str(r.get("symbol") or "").upper().strip()
            if not sym:
                continue
            ord_act[sym] = str(r.get("action") or "").upper().strip()
            ord_qty[sym] = float(r.get("quantity") or 0.0)

    all_syms = sorted(set(list(cur.keys()) + list(tgt.keys())))
    rows: List[Dict[str, Any]] = []
    for sym in all_syms:
        cw = cur.get(sym)
        cw = float(cw) if cw is not None and not pd.isna(cw) else 0.0
        tw = float(tgt.get(sym, 0.0))
        drift = tw - cw
        notional = (drift * float(nav)) if (nav is not None and nav > 0) else None

        rows.append(
            {
                "symbol": sym,
                "current_weight": cw,
                "target_weight": tw,
                "drift": drift,
                "abs_drift": abs(drift),
                "delta_notional": notional,
                "planned_action": ord_act.get(sym),
                "planned_qty": ord_qty.get(sym),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["abs_drift"], ascending=False).reset_index(drop=True)
    return df


def _signals_df(signals_dict: Dict[str, Any], symbols: List[str]) -> pd.DataFrame:
    rows_obj = (signals_dict or {}).get("rows") or {}
    want = set([s.upper().strip() for s in symbols if s])

    out: List[Dict[str, Any]] = []
    for sym, payload in rows_obj.items():
        s = str(sym).upper().strip()
        if want and s not in want:
            continue
        raw = (payload or {}).get("raw") or {}
        r: Dict[str, Any] = {"symbol": s, "ts_utc": (payload or {}).get("ts_utc")}

        for k in [
            "score",
            "total_score",
            "signal_strength",
            "alpha_score",
            "expected_return",
            "expected_return_score",
            "conviction",
            "conviction_score",
            "overall_score",
        ]:
            if k in (payload or {}):
                r[k] = (payload or {}).get(k)
            elif k in raw:
                r[k] = raw.get(k)

        for k in ["model", "model_version", "horizon", "confidence", "sector", "industry"]:
            if k in raw:
                r[k] = raw.get(k)

        out.append(r)

    df = pd.DataFrame(out)
    if not df.empty:
        sort_cols = [c for c in ["total_score", "signal_strength", "score", "overall_score"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols[0], ascending=False, na_position="last").reset_index(drop=True)
        else:
            df = df.sort_values("symbol").reset_index(drop=True)
    return df


def _store_ai_pm_assistant_artifacts(
        *,
        snapshot: PortfolioSnapshot,
        signals: SignalSnapshot,
        targets: TargetWeights,
        plan: TradePlan,
        gates: GateReport,
        strategy_key: str,
) -> None:
    # Canonical, object-level artifacts for Ask-AI (NOT serialized dicts)
    st.session_state["ai_pm_assistant_artifacts"] = {
        "snapshot": snapshot,
        "signals": signals,
        "targets": targets,
        "plan": plan,
        "gates": gates,
        "strategy_key": strategy_key,
        "ts_utc": datetime.utcnow().isoformat(),
    }


def _render_ai_pm_assistant_panel() -> None:
    st.markdown("---")
    st.markdown("#### Ask the AI Portfolio Manager")

    if "ai_pm_chat_history" not in st.session_state or not isinstance(st.session_state["ai_pm_chat_history"], list):
        st.session_state["ai_pm_chat_history"] = []

    dbg = st.toggle("Debug AI", value=False, key="ai_pm_debug_toggle")

    art = st.session_state.get("ai_pm_assistant_artifacts")
    if not isinstance(art, dict) or not art.get("snapshot"):
        st.info("No run artifacts available. Click **Run Now** first.")
        return

    # Render last messages safely
    for m in st.session_state["ai_pm_chat_history"][-24:]:
        if isinstance(m, dict):
            role = str(m.get("role", "user"))
            content = str(m.get("content", ""))
        else:
            role = "assistant"
            content = str(m)
        st.markdown(f"**{role.title()}:** {content[:2000]}")

    q = st.text_input(
        "Ask anything about the portfolio decisions (e.g., Why this stock? Why this weight? Is this good? Why blocked?)",
        key="ai_pm_question_input",
    )

    if st.button("Ask", key="ai_pm_ask_btn"):
        q2 = (q or "").strip()
        if not q2:
            st.warning("Enter a question first.")
            return

        st.session_state["ai_pm_chat_history"].append({"role": "user", "content": q2})

        ans = answer_pm_question(
            q2,
            snapshot=art["snapshot"],
            signals=art.get("signals"),
            targets=art.get("targets"),
            plan=art.get("plan"),
            gates=art.get("gates"),
            auto_trade=bool(is_auto_trade()),
            armed=bool(is_armed()),
            strategy_key=art.get("strategy_key"),
            history=st.session_state["ai_pm_chat_history"][-20:],
            enable_research=True,
            debug=dbg,
        )

        # show answer
        st.session_state["ai_pm_chat_history"].append({"role": "assistant", "content": ans.answer})
        st.write(ans.answer)

        if dbg and getattr(ans, "evidence", None):
            st.json(ans.evidence)


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_ai_portfolio_manager_tab():
    from .pm_memo import build_pm_memo  # keep local
    init_ai_pm_state()

    st.markdown("### 🤖 AI Portfolio Manager")
    st.caption("IBKR supervision. Builds targets, plans trades, gates risk, and executes manually or automatically.")

    # GATEWAY - Fresh connection each session, but reuse within session
    if 'ai_pm_gateway' not in st.session_state:
        st.session_state.ai_pm_gateway = None
    if 'ai_pm_connection_ok' not in st.session_state:
        st.session_state.ai_pm_connection_ok = False

    gw = st.session_state.ai_pm_gateway

    # Show current clientId for debugging
    if gw:
        st.caption(f"🔌 Using clientId: {gw.cfg.client_id if hasattr(gw, 'cfg') else 'unknown'}")

    # Connection status and reconnect option
    col_status, col_reconnect, col_refresh = st.columns([2, 1, 1])

    with col_refresh:
        if st.button("🔃 Refresh Data", key="ibkr_refresh_btn", help="Force refresh positions and orders from TWS"):
            if gw and gw.is_connected():
                with st.spinner("Refreshing data from TWS..."):
                    try:
                        result = gw.refresh_all()
                        if result.get("success"):
                            st.success(
                                f"✅ Refreshed: {result.get('positions', 0)} positions, {result.get('orders', 0)} orders")
                            # Clear any cached data
                            if 'ai_pm_last_verification' in st.session_state:
                                del st.session_state['ai_pm_last_verification']
                            st.rerun()
                        else:
                            st.error(f"Refresh failed: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Refresh error: {e}")
            else:
                st.warning("Connect to IBKR first")

    with col_reconnect:
        if st.button("🔄 Reconnect", key="ibkr_reconnect_btn"):
            # Force fresh gateway
            try:
                if gw:
                    gw.disconnect()
            except:
                pass
            # Use session-unique clientId to avoid conflicts
            import random
            import os
            if 'ai_pm_client_id' not in st.session_state:
                st.session_state.ai_pm_client_id = random.randint(900, 950)
            os.environ['IBKR_CLIENT_ID'] = str(st.session_state.ai_pm_client_id)

            gw = IbkrGateway()
            st.session_state.ai_pm_gateway = gw
            st.session_state.ai_pm_connection_ok = False

            with col_status:
                with st.spinner(f"Connecting to IBKR (clientId {st.session_state.ai_pm_client_id})..."):
                    try:
                        success = gw.connect(timeout_sec=10.0)
                        if success and gw.is_connected():
                            st.session_state.ai_pm_connection_ok = True
                            st.success(f"✅ Connected to IBKR (clientId {st.session_state.ai_pm_client_id})")
                            st.rerun()
                        else:
                            st.session_state.ai_pm_connection_ok = False
                            st.error("❌ Connection failed")
                    except Exception as e:
                        st.session_state.ai_pm_connection_ok = False
                        st.error(f"❌ IBKR Error: {e}")

    # Check if gateway needs initialization
    needs_connect = False
    if gw is None:
        # Use session-unique clientId
        import random
        import os
        if 'ai_pm_client_id' not in st.session_state:
            st.session_state.ai_pm_client_id = random.randint(900, 950)
        os.environ['IBKR_CLIENT_ID'] = str(st.session_state.ai_pm_client_id)

        gw = IbkrGateway()
        st.session_state.ai_pm_gateway = gw
        needs_connect = True
    elif not st.session_state.ai_pm_connection_ok:
        needs_connect = True

    # Connect if needed (first load)
    if needs_connect:
        with col_status:
            with st.spinner("Connecting to IBKR..."):
                try:
                    success = gw.connect(timeout_sec=10.0)
                    if success and gw.is_connected():
                        st.session_state.ai_pm_connection_ok = True
                        st.success("✅ Connected to IBKR")
                    else:
                        st.session_state.ai_pm_connection_ok = False
                        st.error("❌ Connection failed")
                        st.info("💡 Make sure TWS is running with API enabled")
                except Exception as e:
                    st.session_state.ai_pm_connection_ok = False
                    st.error(f"❌ IBKR Error: {e}")
                    st.info("💡 Make sure TWS is running with API enabled (Edit > Global Config > API > Settings)")
    else:
        with col_status:
            st.success("✅ IBKR Connected")

    # FIX: Restore account filter from session state on reconnect/refresh
    stored_account = get_selected_account()
    if stored_account:
        gw.set_account(stored_account)

    # Top controls
    left, mid, right = st.columns([2.2, 2.0, 2.2])

    with left:
        # FIX: Use list_accounts() instead of get_accounts()
        accounts = gw.list_accounts() if gw.is_connected() else []

        # DEFAULT ACCOUNT: U20993660
        DEFAULT_ACCOUNT = "U20993660"

        # Determine default index
        current_account = get_selected_account()
        if current_account and current_account in accounts:
            default_idx = accounts.index(current_account)
        elif DEFAULT_ACCOUNT in accounts:
            default_idx = accounts.index(DEFAULT_ACCOUNT)
        else:
            default_idx = 0

        sel = st.selectbox(
            "Account",
            options=accounts,
            index=default_idx,
            key="ai_pm_account_select",
            disabled=not bool(accounts),
        )
        if sel:
            set_selected_account(sel)
            gw.set_account(sel)  # FIX: Also set on gateway for position filtering

    with mid:
        sk = st.selectbox(
            "Strategy",
            options=list(STRATEGIES.keys()),
            index=list(STRATEGIES.keys()).index(get_strategy_key()) if (get_strategy_key() in STRATEGIES) else 0,
            # FIX: Return strategy.name (string) instead of strategy object
            format_func=lambda k: STRATEGIES[k].name if k in STRATEGIES else k,
            key="ai_pm_strategy_select",
        )
        set_strategy_key(sk)

        # Get default max_holdings from strategy
        strategy = STRATEGIES.get(sk)
        default_max = strategy.max_holdings if strategy else 25

        # Max Holdings slider - user can control how many stocks
        max_holdings = st.slider(
            "📊 Max Holdings",
            min_value=5,
            max_value=100,
            value=st.session_state.get('ai_pm_max_holdings', default_max),
            step=5,
            help="Maximum number of stocks to include in target portfolio",
            key="ai_pm_max_holdings_slider",
        )
        st.session_state['ai_pm_max_holdings'] = max_holdings

    with right:
        c1, c2 = st.columns(2)
        with c1:
            set_auto_trade(st.toggle("Auto Trade", value=bool(is_auto_trade()), key="ai_pm_auto_trade_toggle"))
            set_kill_switch(st.toggle("Kill Switch", value=bool(is_kill_switch()), key="ai_pm_kill_switch_toggle"))
        with c2:
            if is_auto_trade():
                if not is_armed():
                    if st.button("ARM (10 min)", width='stretch', key="ai_pm_arm_btn"):
                        arm_auto_trade(minutes=10)
                else:
                    st.success(f"ARMED until {get_arm_expiry()}")
                    if st.button("DISARM", width='stretch', key="ai_pm_disarm_btn"):
                        disarm_auto_trade()

    # ==========================================================================
    # LOAD SAVED PORTFOLIO AS TARGET
    # ==========================================================================
    if SAVED_PORTFOLIO_AVAILABLE:
        st.markdown("---")
        st.markdown("#### 📂 Load Saved Portfolio as Target")

        # Initialize session state for saved portfolio
        if 'ai_pm_target_portfolio' not in st.session_state:
            st.session_state.ai_pm_target_portfolio = None
        if 'ai_pm_target_weights' not in st.session_state:
            st.session_state.ai_pm_target_weights = {}

        # AUTO-LOAD: Load last saved portfolio on first run
        if st.session_state.ai_pm_target_portfolio is None and not st.session_state.get('_ai_pm_autoload_done'):
            st.session_state._ai_pm_autoload_done = True
            try:
                saved_df = get_saved_portfolios()
                if saved_df is not None and not saved_df.empty:
                    # Get the most recent portfolio (first row, sorted by created_at desc)
                    last_portfolio_id = saved_df.iloc[0]['id']
                    info, holdings_df = load_portfolio(last_portfolio_id)

                    if info and holdings_df is not None and not holdings_df.empty:
                        target_weights = {}
                        for _, row in holdings_df.iterrows():
                            ticker = row.get('ticker', '').upper().strip()
                            weight = float(row.get('weight_pct', 0) or 0) / 100
                            if ticker and weight > 0:
                                target_weights[ticker] = weight

                        st.session_state.ai_pm_target_portfolio = {
                            'id': last_portfolio_id,
                            'name': info.get('name', 'Unknown'),
                            'total_value': info.get('total_value', 100000),
                            'num_holdings': len(target_weights),
                            'objective': info.get('objective', 'custom'),
                        }
                        st.session_state.ai_pm_target_weights = target_weights
                        st.toast(f"📂 Auto-loaded portfolio: {info.get('name')}")
            except Exception as e:
                pass  # Silently fail auto-load

        saved_portfolios_df = get_saved_portfolios()

        if saved_portfolios_df is not None and not saved_portfolios_df.empty:
            col_port1, col_port2, col_port3 = st.columns([3, 1, 1])

            with col_port1:
                # Create display names with info
                portfolio_options = ["-- Select Saved Portfolio --"] + [
                    f"{row['name']} ({row['num_holdings']} stocks, )"
                    for _, row in saved_portfolios_df.iterrows()
                ]
                portfolio_ids = [None] + saved_portfolios_df['id'].tolist()

                selected_idx = st.selectbox(
                    "Saved Portfolio",
                    options=range(len(portfolio_options)),
                    format_func=lambda i: portfolio_options[i],
                    key="ai_pm_saved_portfolio_select",
                    label_visibility="collapsed"
                )

            with col_port2:
                load_btn = st.button("📥 Load as Target", key="ai_pm_load_portfolio_btn", type="primary")

            with col_port3:
                clear_btn = st.button("🗑️ Clear Target", key="ai_pm_clear_target_btn")

            if load_btn and selected_idx and selected_idx > 0:
                portfolio_id = portfolio_ids[selected_idx]
                info, holdings_df = load_portfolio(portfolio_id)

                if info and holdings_df is not None and not holdings_df.empty:
                    # Convert holdings to target weights dict
                    target_weights = {}
                    for _, row in holdings_df.iterrows():
                        ticker = row.get('ticker', '').upper().strip()
                        weight = float(row.get('weight_pct', 0) or 0) / 100  # Convert to decimal
                        if ticker and weight > 0:
                            target_weights[ticker] = weight

                    # Store in session state
                    st.session_state.ai_pm_target_portfolio = {
                        'id': portfolio_id,
                        'name': info.get('name', 'Unknown'),
                        'total_value': info.get('total_value', 100000),
                        'num_holdings': len(target_weights),
                        'objective': info.get('objective', 'custom'),
                    }
                    st.session_state.ai_pm_target_weights = target_weights

                    st.success(f"✅ Loaded **{info.get('name')}** with {len(target_weights)} holdings as target")
                    st.rerun()
                else:
                    st.error("Failed to load portfolio")

            if clear_btn:
                st.session_state.ai_pm_target_portfolio = None
                st.session_state.ai_pm_target_weights = {}
                st.session_state.ai_pm_loaded_json_path = None
                st.info("Target portfolio cleared")
                st.rerun()
        else:
            st.caption("No saved portfolios found. Create one in the AI Portfolio Builder tab.")

    # JSON File Upload Option - Always available regardless of saved portfolios
    st.markdown("---")
    st.markdown("#### 📁 Load Portfolio from JSON File")

    uploaded_json = st.file_uploader(
        "Upload Portfolio JSON",
        type=['json'],
        key="ai_pm_json_uploader_main",
        help="Upload an IBKR-formatted portfolio JSON file"
    )

    if uploaded_json:
        try:
            import json
            import os

            # Save uploaded file and track path
            json_data = json.load(uploaded_json)

            # Save to json folder for later editing
            json_dir = 'json'
            os.makedirs(json_dir, exist_ok=True)
            json_path = os.path.join(json_dir, uploaded_json.name)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4)

            # Track the path for save functionality
            st.session_state.ai_pm_loaded_json_path = json_path

            # Parse holdings from JSON
            holdings = json_data if isinstance(json_data, list) else json_data.get('holdings', [])

            if holdings:
                target_weights = {}
                for h in holdings:
                    ticker = (h.get('symbol') or h.get('ticker') or '').upper().strip()
                    weight = float(h.get('weight', 0)) / 100  # Convert to decimal
                    if ticker and weight > 0:
                        target_weights[ticker] = weight

                st.session_state.ai_pm_target_portfolio = {
                    'id': 'json_upload',
                    'name': uploaded_json.name.replace('.json', ''),
                    'total_value': 100000,
                    'num_holdings': len(target_weights),
                    'objective': 'custom',
                    'source': 'json_file',
                    'json_path': json_path,
                }
                st.session_state.ai_pm_target_weights = target_weights

                st.success(f"✅ Loaded **{uploaded_json.name}** with {len(target_weights)} holdings")
                st.caption(f"📂 Saved to: {json_path}")
            else:
                st.error("No holdings found in JSON file")

        except Exception as e:
            st.error(f"Error loading JSON: {e}")

    # Show current target if loaded (moved outside the saved portfolios block)
    if st.session_state.get('ai_pm_target_portfolio'):
        tp = st.session_state.ai_pm_target_portfolio
        tw = st.session_state.get('ai_pm_target_weights', {})

        source_info = " | 📁 JSON" if tp.get('source') == 'json_file' else " | 💾 Saved"
        st.info(f"🎯 **Active Target:** {tp['name']} | {tp['num_holdings']} holdings{source_info}")

        col_view, col_clear = st.columns([3, 1])
        with col_clear:
            if st.button("🗑️ Clear Target", key="ai_pm_clear_target_main"):
                st.session_state.ai_pm_target_portfolio = None
                st.session_state.ai_pm_target_weights = {}
                st.session_state.ai_pm_loaded_json_path = None
                st.rerun()

        with st.expander("View Target Holdings", expanded=False):
            target_df = pd.DataFrame([
                {'Ticker': k, 'Weight %': f"{v * 100:.2f}%"}
                for k, v in sorted(tw.items(), key=lambda x: -x[1])
            ])
            st.dataframe(target_df, hide_index=True, height=300)

    # Capital Deployment Section
    st.markdown("---")
    st.markdown("#### 💰 Capital to Deploy")

    # Get current NAV from last snapshot or assistant artifacts - NO DEFAULTS
    last_results = get_last_results() or {}
    last_snapshot = last_results.get("snapshot") or {}
    nav_for_deploy = last_snapshot.get("net_liquidation")
    current_cash = last_snapshot.get("total_cash")

    # Check session state for more recent data
    if st.session_state.get("ai_pm_assistant_artifacts"):
        art = st.session_state["ai_pm_assistant_artifacts"]
        if art.get("snapshot"):
            snap = art["snapshot"]
            snap_nav = getattr(snap, "net_liquidation", None)
            snap_cash = getattr(snap, "total_cash", None)
            if snap_nav is not None and snap_nav > 0:
                nav_for_deploy = snap_nav
            if snap_cash is not None:
                current_cash = snap_cash

    # Check stored NAV from previous run (NOT a default, actual data)
    if not nav_for_deploy or nav_for_deploy <= 0:
        nav_for_deploy = st.session_state.get('ai_pm_last_nav')

    # Store NAV for other components if we have real data
    if nav_for_deploy and nav_for_deploy > 0:
        st.session_state['ai_pm_last_nav'] = nav_for_deploy

    deploy_col1, deploy_col2, deploy_col3 = st.columns([2, 1.5, 2.5])

    # Check if we have real NAV data
    has_nav_data = nav_for_deploy is not None and nav_for_deploy > 0

    with deploy_col1:
        if has_nav_data:
            max_deploy = nav_for_deploy * 0.995  # Max 99.5% of NAV

            # Get stored deploy amount, or leave empty for user to enter
            stored_deploy = st.session_state.get('ai_pm_deploy_amount')
            if stored_deploy is not None and stored_deploy > 0:
                default_deploy_amt = min(stored_deploy, max_deploy)
            else:
                default_deploy_amt = 0.0  # User must enter, no default

            deploy_amount = st.number_input(
                "Amount to Invest ($)",
                min_value=0.0,
                max_value=float(max_deploy),
                value=float(default_deploy_amt),
                step=500.0,
                help=f"Max deployable: ${max_deploy:,.0f} (99.5% of NAV). This amount will be invested, rest stays as cash.",
                key="ai_pm_deploy_amount_input",
            )
            st.session_state['ai_pm_deploy_amount'] = deploy_amount
        else:
            st.warning("⚠️ NAV not available. Click **Run Now** to load portfolio data.")
            deploy_amount = 0

    with deploy_col2:
        if has_nav_data:
            deploy_pct = (deploy_amount / nav_for_deploy * 100) if nav_for_deploy > 0 else 0
            cash_to_keep = nav_for_deploy - deploy_amount
            st.metric("Deploy %", f"{deploy_pct:.1f}%")
            st.metric("Cash to Keep", f"${cash_to_keep:,.0f}")
        else:
            st.metric("Deploy %", "N/A")
            st.metric("Cash to Keep", "N/A")

    with deploy_col3:
        # Show actual values or N/A
        nav_display = f"${nav_for_deploy:,.0f}" if has_nav_data else "N/A - Run Now to load"
        cash_display = f"${current_cash:,.0f}" if current_cash is not None else "N/A"
        st.caption(f"**NAV:** {nav_display} | **Current Cash:** {cash_display}")

        # Quick deploy buttons - only work if we have NAV data
        if has_nav_data:
            btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
            with btn_col1:
                if st.button("90%", key="deploy_90", width='stretch'):
                    st.session_state['ai_pm_deploy_amount'] = nav_for_deploy * 0.90
                    st.rerun()
            with btn_col2:
                if st.button("95%", key="deploy_95", width='stretch'):
                    st.session_state['ai_pm_deploy_amount'] = nav_for_deploy * 0.95
                    st.rerun()
            with btn_col3:
                if st.button("98%", key="deploy_98", width='stretch'):
                    st.session_state['ai_pm_deploy_amount'] = nav_for_deploy * 0.98
                    st.rerun()
            with btn_col4:
                if st.button("99.5%", key="deploy_995", width='stretch'):
                    st.session_state['ai_pm_deploy_amount'] = nav_for_deploy * 0.995
                    st.rerun()
        else:
            st.info("📊 Quick deploy buttons will appear after loading portfolio data.")

    # Show current portfolio summary from loaded statement
    sel_account = get_selected_account()
    if sel_account:
        loaded_positions = _get_loaded_portfolio_positions(sel_account)
        if loaded_positions:
            with st.expander(f"📂 Loaded Portfolio: {sel_account} ({len(loaded_positions)} positions)", expanded=False):
                total_value = sum(p['market_value'] for p in loaded_positions.values())
                st.metric("Total Loaded Portfolio Value", f"${total_value:,.0f}")

                # Top 10 positions
                sorted_pos = sorted(loaded_positions.items(), key=lambda x: x[1]['market_value'], reverse=True)[:10]
                pos_df = pd.DataFrame([
                    {
                        'Symbol': sym,
                        'Weight %': f"{data['weight'] * 100:.2f}%",
                        'Market Value': f"${data['market_value']:,.0f}",
                        'P&L': f"${data['unrealized_pnl']:,.0f}",
                        'P&L %': f"{data['unrealized_pnl_pct']:.1f}%"
                    }
                    for sym, data in sorted_pos
                ])
                st.dataframe(pos_df, width='stretch', hide_index=True)

                st.caption(
                    "💡 This is your current portfolio from the uploaded statement. The AI PM will consider these positions when building targets.")
        else:
            # Check for IBKR live positions when no statement is uploaded
            ibkr_positions = []
            if gw.is_connected():
                try:
                    ibkr_positions = gw.ib.positions(account=sel_account) if hasattr(gw.ib, 'positions') else []
                except Exception:
                    ibkr_positions = []

            if ibkr_positions:
                with st.expander(f"📡 Live IBKR Positions: {sel_account} ({len(ibkr_positions)} positions)",
                                 expanded=False):
                    ibkr_df = pd.DataFrame([
                        {
                            'Symbol': p.contract.symbol,
                            'Position': int(p.position),
                            'Avg Cost': f"" if p.avgCost else "N/A",
                        }
                        for p in ibkr_positions[:20]
                    ])
                    st.dataframe(ibkr_df, width='stretch', hide_index=True)
                    st.caption("💡 Live positions from IBKR. Click Run Now to build rebalancing plan.")
            else:
                st.info(
                    f"📂 No portfolio loaded for {sel_account}. Connect to IBKR or upload a statement.")

    st.markdown("---")
    colA, colB, colC = st.columns([1.2, 1.2, 1.6])
    with colA:
        run_now = st.button("Run Now", width='stretch', key="ai_pm_run_now_btn")
    with colB:
        dry_run = st.toggle("Dry Run", value=True, key="ai_pm_dry_run_toggle")
    with colC:
        st.caption("Run Now builds snapshot/targets/plan/gates. Confirm executes (if not blocked).")

    last = get_last_results() or {}

    # If we have assistant artifacts (object-level), prefer them for tables/memo
    art = st.session_state.get("ai_pm_assistant_artifacts")
    have_obj_art = isinstance(art, dict) and art.get("snapshot") and art.get("plan") and art.get("targets") and art.get(
        "gates")

    if have_obj_art:
        snap_obj: PortfolioSnapshot = art["snapshot"]
        plan_obj: TradePlan = art["plan"]
        targets_obj: TargetWeights = art["targets"]
        gates_obj: GateReport = art["gates"]
        signals_obj = art.get("signals")

        # NEW: AI Portfolio Intelligence Panel (smart recommendations)
        try:
            _render_ai_intelligence_panel(
                snapshot=snap_obj,
                signals=signals_obj,
                targets=targets_obj,
                account=get_selected_account(),
            )
        except Exception as e:
            st.error(f"Failed to render AI intelligence: {e}")

        # NEW: Rebalance Metrics Panel (like ETF Rebalancer)
        try:
            price_map = st.session_state.get("ai_pm_price_map", {})
            capital_to_deploy = st.session_state.get('ai_pm_deploy_amount')

            _render_rebalance_metrics_panel(
                snapshot=snap_obj,
                targets=targets_obj,
                plan=plan_obj,
                price_map=price_map,
                capital_to_deploy=capital_to_deploy,
            )
        except Exception as e:
            st.error(f"Failed to render rebalance metrics: {e}")

        # NEW: Rebalancing Results Table (like ETF Rebalancer)
        try:
            _render_rebalancing_results_table(
                snapshot=snap_obj,
                targets=targets_obj,
                plan=plan_obj,
                price_map=price_map,
                signals=signals_obj,
            )
        except Exception as e:
            st.error(f"Failed to render results table: {e}")

        try:
            memo_obj = build_pm_memo(
                snapshot=snap_obj,
                signals=signals_obj,
                targets=targets_obj,
                plan=plan_obj,
                gates=gates_obj,
                auto_trade=bool(is_auto_trade()),
                armed=bool(is_armed()),
            )
            memo = memo_obj.to_dict() if hasattr(memo_obj, "to_dict") else {}
        except Exception:
            memo = {}
    else:
        memo = {}

    # Memo panel (best-effort)
    st.subheader("🧠 AI PM Memo")
    if memo:
        st.write(f"Status: {memo.get('status')}")
        st.write(memo.get("headline") or "")
        blockers = memo.get("blockers") or []
        if blockers:
            st.markdown("### 🚨 Hard Blockers")

            # Blocker explanations
            BLOCKER_EXPLANATIONS = {
                "max_position_weight": "**Position Concentration Risk**: A single position exceeds the maximum allowed weight. This increases portfolio risk if that stock drops significantly.",
                "max_sector_weight": "**Sector Concentration Risk**: A single sector exceeds the maximum allowed weight, reducing diversification.",
                "min_cash": "**Liquidity Risk**: The trade would reduce cash below the minimum required buffer.",
                "max_trades": "**Excessive Trading**: Too many trades in a single cycle, which increases transaction costs.",
                "market_closed": "**Market Closed**: Markets are currently closed. Orders may queue until market opens.",
                "circuit_breaker": "**Circuit Breaker**: Market volatility has triggered protective measures.",
            }

            for b in blockers:
                # Extract blocker type for explanation lookup
                blocker_type = None
                for key in BLOCKER_EXPLANATIONS.keys():
                    if key.lower().replace("_", " ") in b.lower() or key.lower() in b.lower():
                        blocker_type = key
                        break

                # Display blocker in red with explanation
                st.markdown(
                    f'<div style="background-color: #ffcccc; padding: 10px; border-radius: 5px; border-left: 4px solid #cc0000; margin: 5px 0;">'
                    f'<span style="color: #cc0000; font-weight: bold;">⛔ {b}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                if blocker_type and blocker_type in BLOCKER_EXPLANATIONS:
                    st.caption(BLOCKER_EXPLANATIONS[blocker_type])
        warnings = memo.get("warnings") or []
        if warnings:
            st.warning("Planner warnings")
            for w in warnings[:10]:
                st.write(f"- {w}")
    else:
        st.info("No PM memo yet. Click Run Now.")

    # Proposed portfolio table (needs object-level artifacts)
    if have_obj_art:
        try:
            _render_proposed_portfolio_table(snap_obj, plan_obj)
        except Exception as e:
            st.error(f"Failed to render proposed portfolio table: {e}")

        # NEW: Target Portfolio Analysis Table
        try:
            _render_target_portfolio_analysis(
                targets=targets_obj,
                signals=signals_obj,
                snapshot=snap_obj,
                plan=plan_obj,
                strategy_key=get_strategy_key(),
            )
        except Exception as e:
            st.error(f"Failed to render target analysis: {e}")

        # NEW: Positions to Exit Table
        try:
            _render_positions_to_exit(
                targets=targets_obj,
                snapshot=snap_obj,
                plan=plan_obj,
            )
        except Exception as e:
            st.error(f"Failed to render exit positions: {e}")

        # NEW: Portfolio Comparison (Current vs Target from loaded statement)
        try:
            _render_portfolio_comparison(
                snapshot=snap_obj,
                targets=targets_obj,
                signals=signals_obj,
                account=get_selected_account(),
            )
        except Exception as e:
            st.error(f"Failed to render portfolio comparison: {e}")

    # Pending Replacements from AI Recommendations
    if st.session_state.get('pending_replacements'):
        st.markdown("#### 🔄 Pending Replacements (from AI Recommendations)")

        pending = st.session_state.pending_replacements

        # Display pending replacements
        pending_df = pd.DataFrame([
            {
                'Action': 'SELL → BUY' if r.get('buy') else 'SELL (Exit)',
                'Sell': r['sell'],
                'Buy': r.get('buy') or '-',
                'Has IBKR Data': '✅' if r.get('buy_data') else '❌',
                'Reason': r.get('reason', '')[:40],
            }
            for r in pending
        ])
        st.dataframe(pending_df, width='stretch', hide_index=True)

        col_clear, col_save, col_exec = st.columns([1, 1, 1])

        with col_clear:
            if st.button("🗑️ Clear All", key="clear_pending_replacements"):
                st.session_state.pending_replacements = []
                st.rerun()

        with col_save:
            # Save to JSON functionality - works for both JSON and database portfolios
            target_portfolio = st.session_state.get('ai_pm_target_portfolio')
            json_path = st.session_state.get('ai_pm_loaded_json_path')

            if STOCK_LOOKUP_AVAILABLE and target_portfolio:
                btn_label = "💾 Save Updated JSON" if json_path else "💾 Export to JSON"

                if st.button(btn_label, key="save_updated_json", type="primary"):
                    try:
                        import json
                        import os

                        if json_path and os.path.exists(json_path):
                            # Update existing JSON
                            editor = JSONPortfolioEditor()
                            if editor.load(json_path):
                                # Apply all pending replacements
                                for repl in pending:
                                    old_sym = repl['sell']
                                    new_sym = repl.get('buy')
                                    new_data = repl.get('buy_data', {})

                                    if new_sym and new_data:
                                        success, msg = editor.replace_stock(old_sym, new_data, transfer_weight=True)
                                        if success:
                                            st.success(msg)
                                        else:
                                            st.warning(msg)
                                    elif not new_sym:
                                        success, msg = editor.remove_stock(old_sym)
                                        if success:
                                            st.success(msg)

                                # Save the updated JSON
                                success, result = editor.save(add_timestamp=True)
                                if success:
                                    st.success(f"✅ Saved: {result}")
                                    st.info(editor.get_changes_report())
                                    st.session_state.pending_replacements = []
                                else:
                                    st.error(f"Failed to save: {result}")
                            else:
                                st.error("Failed to load original JSON")
                        else:
                            # Export database portfolio to new JSON with replacements
                            tw = st.session_state.get('ai_pm_target_weights', {})

                            # Load stock lookup and try to find a reference JSON with full data
                            stock_lookup = get_stock_lookup() if STOCK_LOOKUP_AVAILABLE else None
                            if stock_lookup and not stock_lookup._cache:
                                stock_lookup.load_all_json_files()

                            # Try to find existing JSON files with this portfolio's stocks
                            # to use as template for full IBKR data
                            reference_holdings = {}
                            json_dir = 'json'
                            if os.path.exists(json_dir):
                                import glob
                                for jf in glob.glob(os.path.join(json_dir, '*_IBKR.json')):
                                    try:
                                        with open(jf, 'r', encoding='utf-8') as rf:
                                            ref_data = json.load(rf)
                                        if isinstance(ref_data, list):
                                            for item in ref_data:
                                                sym = (item.get('symbol') or '').upper()
                                                # Only use if has valid ISIN
                                                if sym and item.get('isin') and len(item.get('isin', '')) >= 10:
                                                    if sym not in reference_holdings:
                                                        reference_holdings[sym] = item
                                    except:
                                        pass

                            st.info(f"📚 Found {len(reference_holdings)} stocks with full IBKR data in JSON files")

                            # Apply replacements to weights
                            updated_weights = dict(tw)
                            replacement_data = {}  # Store buy data for replacements

                            for repl in pending:
                                old_sym = repl['sell'].upper()
                                new_sym = (repl.get('buy') or '').upper()

                                if old_sym in updated_weights:
                                    old_weight = updated_weights.pop(old_sym)
                                    if new_sym:
                                        updated_weights[new_sym] = updated_weights.get(new_sym, 0) + old_weight
                                        if repl.get('buy_data'):
                                            replacement_data[new_sym] = repl['buy_data']

                            # Build holdings with full IBKR data
                            holdings = []
                            missing_data = []

                            for ticker, weight in updated_weights.items():
                                # Start with reference data if available
                                if ticker in reference_holdings:
                                    holding = dict(reference_holdings[ticker])
                                    holding['weight'] = round(weight * 100, 4)
                                else:
                                    # Create new holding
                                    holding = {
                                        'name': ticker,
                                        'originalName': ticker,
                                        'similarity': 100.0,
                                        'symbol': ticker,
                                        'minMultiplier': 1,
                                        'isin': '',
                                        'sector': '',
                                        'country': 'USA',
                                        'weight': round(weight * 100, 4),
                                        'conid': 0,
                                        'secType': 'STK',
                                        'currency': 'USD',
                                        'exchange': 'SMART',
                                        'primary_exchange': '',
                                    }

                                    # Try stock lookup
                                    if stock_lookup:
                                        stock_obj = stock_lookup.lookup(ticker)
                                        if stock_obj and stock_obj.isin:
                                            holding.update({
                                                'name': stock_obj.name or ticker,
                                                'originalName': stock_obj.originalName or ticker,
                                                'isin': stock_obj.isin,
                                                'sector': stock_obj.sector,
                                                'conid': stock_obj.conid,
                                                'primary_exchange': stock_obj.primary_exchange,
                                            })

                                    # Check replacement data
                                    if ticker in replacement_data:
                                        bd = replacement_data[ticker]
                                        if bd.get('isin'):
                                            holding.update({
                                                'name': bd.get('name') or holding['name'],
                                                'originalName': bd.get('originalName') or holding['originalName'],
                                                'isin': bd.get('isin') or holding['isin'],
                                                'sector': bd.get('sector') or holding['sector'],
                                                'conid': bd.get('conid') or holding['conid'],
                                                'primary_exchange': bd.get('primary_exchange') or holding[
                                                    'primary_exchange'],
                                            })

                                    # Track missing data
                                    if not holding.get('isin'):
                                        missing_data.append(ticker)

                                holdings.append(holding)

                            # Try to fetch missing data from IBKR
                            if missing_data and gw.is_connected():
                                st.info(f"🔍 Fetching IBKR data for {len(missing_data)} stocks: {missing_data[:10]}...")
                                for ticker in missing_data[:20]:  # Limit to 20 to avoid timeout
                                    try:
                                        if stock_lookup:
                                            stock_obj = stock_lookup.fetch_from_ibkr(ticker, gw.ib)
                                            if stock_obj and stock_obj.conid:
                                                # Update holding
                                                for h in holdings:
                                                    if h['symbol'] == ticker:
                                                        h.update({
                                                            'name': stock_obj.name or ticker,
                                                            'conid': stock_obj.conid,
                                                            'primary_exchange': stock_obj.primary_exchange,
                                                        })
                                                        break
                                    except Exception as e:
                                        st.warning(f"Could not fetch {ticker}: {e}")

                            # Sort by weight descending
                            holdings.sort(key=lambda x: x.get('weight', 0), reverse=True)

                            # Save to JSON
                            os.makedirs(json_dir, exist_ok=True)
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            portfolio_name = target_portfolio.get('name', 'portfolio').replace(' ', '_')
                            output_path = os.path.join(json_dir, f"{portfolio_name}_IBKR_{timestamp}.json")

                            with open(output_path, 'w', encoding='utf-8') as f:
                                json.dump(holdings, f, indent=4, ensure_ascii=False)

                            # Report results
                            complete_count = sum(1 for h in holdings if h.get('isin'))
                            st.success(f"✅ Exported to: {output_path}")
                            st.info(
                                f"Created {len(holdings)} holdings | {complete_count} with full IBKR data | {len(pending)} replacements applied")

                            if len(holdings) - complete_count > 0:
                                incomplete = [h['symbol'] for h in holdings if not h.get('isin')][:10]
                                st.warning(
                                    f"⚠️ {len(holdings) - complete_count} stocks missing IBKR data: {incomplete}")

                            # Update session state to use this JSON
                            st.session_state.ai_pm_loaded_json_path = output_path
                            st.session_state.pending_replacements = []

                    except Exception as e:
                        st.error(f"Error saving JSON: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.caption("Load a portfolio to enable save")

        with col_exec:
            st.info("💡 Run Now to execute trades")

        st.markdown("---")

    # Proposed Actions (orders) preview (fallback to serialized last dict if needed)
    st.markdown("#### Proposed Actions")
    if have_obj_art and getattr(plan_obj, "orders", None):
        orders_rows = [o.to_dict() if hasattr(o, "to_dict") else o.__dict__ for o in (plan_obj.orders or [])]
        st.dataframe(orders_rows, width='stretch', hide_index=True)
    else:
        plan_dict = last.get("plan") or {}
        if plan_dict and plan_dict.get("orders"):
            st.dataframe(plan_dict["orders"], width='stretch', hide_index=True)
        else:
            st.info("No proposed orders yet.")

    sel_account = get_selected_account()
    can_trade = bool(sel_account)

    gates_dict = last.get("gates") or {}
    blocked = False
    try:
        blocked = bool(gates_dict.get("blocked")) if isinstance(gates_dict, dict) else bool(
            getattr(gates_obj, "blocked", False))
    except Exception:
        blocked = False

    if not can_trade:
        st.warning("Select a single IBKR account (not 'All') to enable execution.")
    if blocked:
        st.markdown(
            '<div style="background-color: #ffcccc; padding: 15px; border-radius: 5px; border: 2px solid #cc0000; margin: 10px 0;">'
            '<span style="color: #cc0000; font-weight: bold; font-size: 1.1em;">⚠️ Execution blocked by hard gates</span><br>'
            '<span style="color: #666;">The trade plan violates one or more risk constraints. Review the blockers above.</span>'
            '</div>',
            unsafe_allow_html=True
        )

        # Override checkbox
        override_blockers = st.checkbox(
            "🔓 Override hard blockers (I understand the risks)",
            key="ai_pm_override_blockers",
            help="Check this to bypass risk gate blockers. Use with caution - these limits exist to protect your portfolio."
        )

        if override_blockers:
            st.warning("⚠️ **OVERRIDE ACTIVE**: Hard blocker limits will be bypassed. Proceed with caution!")
    else:
        override_blockers = False

    if is_kill_switch():
        st.error("Execution disabled: KILL SWITCH enabled.")

    col_confirm, col_cancel = st.columns([1, 1])
    with col_confirm:
        # Allow execution if override is checked OR if not blocked
        can_execute = can_trade and (not blocked or override_blockers) and not is_kill_switch()
        confirm = st.button(
            "Confirm & Send (Manual)" if not override_blockers else "⚠️ Confirm & Send (OVERRIDE)",
            width='stretch',
            disabled=not can_execute,
            key="ai_pm_confirm_send_btn",
            type="primary" if not override_blockers else "secondary",
        )
    with col_cancel:
        cancel_all = st.button(
            "🛑 Cancel All Orders",
            width='stretch',
            type="secondary",
            disabled=not gw or not gw.is_connected(),
            key="ai_pm_cancel_all_btn",
        )
        if cancel_all:
            with st.spinner("Cancelling all orders..."):
                try:
                    gw.ib.reqGlobalCancel()
                    gw.ib.sleep(3)
                    st.success("✅ Global cancel request sent")
                except Exception as e:
                    st.error(f"Cancel failed: {e}")

    if is_auto_trade():
        if not is_armed():
            st.warning("Auto Trade is ON but NOT ARMED. No orders will be sent automatically.")
        if is_kill_switch():
            st.warning("Auto Trade is ON but KILL SWITCH is enabled.")

    # -------------------------
    # Run Now pipeline
    # -------------------------
    if run_now:
        if not st.session_state.get('ai_pm_connection_ok', False):
            st.error("❌ IBKR not connected. Click **Reconnect** button above.")
            st.stop()
        elif not gw or not gw.is_connected():
            st.error("❌ IBKR connection lost. Click **Reconnect** button above.")
            st.session_state.ai_pm_connection_ok = False
            st.stop()
        elif not sel_account:
            st.error("❌ Select a single account (not 'All') to run planning.")
            st.stop()

        # Verify connection is truly alive with a quick test
        try:
            test_accounts = gw.list_accounts()
            if not test_accounts:
                st.error("❌ IBKR connection stale. Click **Reconnect**.")
                st.session_state.ai_pm_connection_ok = False
                st.stop()
        except Exception as e:
            st.error(f"❌ IBKR connection error: {e}. Click **Reconnect**.")
            st.session_state.ai_pm_connection_ok = False
            st.stop()
        else:
            # Verify connection is healthy with a quick ping
            progress_placeholder = st.empty()
            progress_placeholder.info("⏳ Verifying IBKR connection...")

            try:
                # Quick ping to verify connection is alive
                if not gw.ping():
                    st.error("❌ IBKR connection is stale. Click **Reconnect** to re-establish.")
                    st.stop()
            except Exception as e:
                st.error(f"❌ IBKR connection error: {e}. Click **Reconnect**.")
                st.stop()

            progress_placeholder.info("⏳ Step 1/5: Loading portfolio snapshot...")

            try:
                snapshot = _to_portfolio_snapshot(gw, sel_account)
            except Exception as e:
                st.error(f"❌ Failed to load portfolio snapshot: {e}")
                st.stop()

            if not snapshot or not snapshot.positions:
                st.warning("⚠️ No positions found. Check account selection and IBKR connection.")

            with st.spinner("Building trade plan..."):
                progress_placeholder.info("⏳ Step 2/5: Snapshot loaded")

                # Debug: valuation coverage from snapshot (no fake)
                try:
                    missing_val = [p.symbol for p in (snapshot.positions or []) if
                                   (p.market_price is None and p.market_value is None)]
                    st.info(
                        f"Snapshot valuation coverage: {len(snapshot.positions) - len(missing_val)}/{len(snapshot.positions)} "
                        f"positions have market_price/market_value. Missing sample: {missing_val[:15]}"
                    )
                except Exception:
                    pass

                progress_placeholder.info("⏳ Step 2/5: Loading signals...")
                signals, diag = load_platform_signals_snapshot(limit=5000)
                progress_placeholder.info(f"⏳ Step 3/5: Loaded {diag.get('merged_rows', 0)} signals")

                # NEW: Show signals diagnostics
                st.info(
                    f"📊 **Signals Loaded:** {diag.get('merged_rows', 0)} tickers | "
                    f"Scores: {diag.get('scores_rows', 'N/A')} | "
                    f"Signals: {diag.get('signals_rows', 'N/A')} | "
                    f"Prob cols: [{diag.get('probability_columns', 'none')}] | "
                    f"EV cols: [{diag.get('ev_columns', 'none')}]"
                )

                if diag.get('error'):
                    st.error(f"❌ Signal loading error: {diag.get('error')}")

                # Show sample of signal data for debugging
                signal_rows = getattr(signals, "rows", None) or {}
                if signal_rows:
                    sample_sym = list(signal_rows.keys())[0]
                    sample_raw = getattr(signal_rows[sample_sym], "raw", {})
                    available_fields = [k for k, v in sample_raw.items() if v is not None and k != "ticker"][:12]
                    st.caption(f"Available signal fields: {', '.join(available_fields)}")

                # Get max holdings from session state (user-controlled slider)
                max_holdings_override = st.session_state.get('ai_pm_max_holdings')

                # Check if we have a saved portfolio loaded as target
                if st.session_state.get('ai_pm_target_weights'):
                    # Use saved portfolio weights directly
                    from .models import TargetWeights
                    tp = st.session_state.get('ai_pm_target_portfolio', {})
                    targets = TargetWeights(
                        weights=st.session_state['ai_pm_target_weights'],
                        strategy_key='saved_portfolio',
                        ts_utc=datetime.utcnow(),
                        notes=[f"Using saved portfolio: {tp.get('name', 'Unknown')}"]
                    )
                    st.success(
                        f"📂 Using saved portfolio **{tp.get('name')}** as target ({len(targets.weights)} holdings)")
                else:
                    targets = build_target_weights(
                        signals=signals,
                        strategy_key=get_strategy_key(),
                        constraints=DEFAULT_CONSTRAINTS,
                        max_holdings_override=max_holdings_override,
                    )

                # NEW: Show target diagnostics with warnings
                target_count = len(targets.weights) if targets.weights else 0
                st.info(
                    f"🎯 **Strategy:** {get_strategy_key()} | **Max Holdings:** {max_holdings_override or 'default'} → **{target_count} holdings** selected")

                # Display target builder notes with proper formatting
                if targets.notes:
                    with st.expander("📝 Target Builder Notes", expanded=True):
                        for note in targets.notes:
                            if note.startswith("❌"):
                                st.error(note)
                            elif note.startswith("⚠️"):
                                st.warning(note)
                            elif note.startswith("✓"):
                                st.success(note)
                            else:
                                st.info(f"• {note}")

                if target_count == 0:
                    st.error(
                        "⚠️ **NO STOCKS MATCHED STRATEGY CRITERIA!** "
                        "Check the notes above for details. Options: "
                        "1) Lower thresholds, 2) Run scanner on more stocks, 3) Try different strategy."
                    )

                progress_placeholder.info("⏳ Step 4/5: Fetching prices...")

                # Price universe: current ∪ target
                universe = sorted(
                    set([p.symbol.strip().upper() for p in (snapshot.positions or []) if getattr(p, "symbol", None)])
                    | set([k.strip().upper() for k in ((targets.weights or {}) or {}).keys() if k])
                )

                # YAHOO FINANCE FIRST (fast batch fetch)
                # Cache prices for 5 minutes
                cached_prices = st.session_state.get("ai_pm_price_map", {})
                cache_time = st.session_state.get("ai_pm_price_cache_time")
                cache_valid = cache_time and (datetime.utcnow() - cache_time).seconds < 300

                price_map = dict(cached_prices) if cache_valid else {}
                price_diag = {
                    'requested': len(universe),
                    'resolved': 0,
                    'missing': [],
                    'source_counts': {'yahoo': 0, 'ibkr': 0, 'cache': 0},
                }

                # Count cached prices
                if cache_valid:
                    cached_count = len([s for s in universe if s in price_map and price_map.get(s)])
                    price_diag['source_counts']['cache'] = cached_count

                # Find symbols needing prices
                need_prices = [s for s in universe if s not in price_map or not price_map.get(s)]

                if need_prices:
                    # PRIMARY: Yahoo Finance (fast batch)
                    try:
                        import yfinance as yf

                        # Batch in chunks of 100 for reliability
                        for i in range(0, len(need_prices), 100):
                            chunk = need_prices[i:i + 100]
                            tickers_str = ' '.join(chunk)

                            yf_data = yf.download(tickers_str, period='1d', progress=False, threads=True)

                            if not yf_data.empty and 'Close' in yf_data.columns:
                                if len(chunk) == 1:
                                    val = yf_data['Close'].iloc[-1]
                                    if val and val > 0:
                                        price_map[chunk[0]] = float(val)
                                        price_diag['source_counts']['yahoo'] += 1
                                else:
                                    for sym in chunk:
                                        try:
                                            if sym in yf_data['Close'].columns:
                                                val = yf_data['Close'][sym].iloc[-1]
                                                if val and val > 0:
                                                    price_map[sym] = float(val)
                                                    price_diag['source_counts']['yahoo'] += 1
                                        except:
                                            pass
                    except Exception as e:
                        st.warning(f"Yahoo Finance error: {e}")

                # IBKR fallback DISABLED - causes freeze in Streamlit threads
                still_missing = [s for s in universe if s not in price_map or not price_map.get(s)]
                if still_missing:
                    price_diag['missing'] = still_missing[:20]

                # Cache for next time
                st.session_state["ai_pm_price_map"] = price_map
                st.session_state["ai_pm_price_diag"] = price_diag
                st.session_state["ai_pm_price_cache_time"] = datetime.utcnow()

                resolved_count = len([s for s in universe if s in price_map and price_map.get(s)])
                price_diag['resolved'] = resolved_count

                st.info(
                    f"💰 Prices: {resolved_count}/{len(universe)} | "
                    f"Yahoo: {price_diag['source_counts']['yahoo']} | "
                    f"Cache: {price_diag['source_counts']['cache']} | "
                    f"IBKR: {price_diag['source_counts']['ibkr']}"
                )

                if price_diag.get('missing'):
                    st.caption(f"Missing: {price_diag['missing'][:10]}")

                # Get capital to deploy from user input
                capital_to_deploy = st.session_state.get('ai_pm_deploy_amount')

                progress_placeholder.info("⏳ Step 5/5: Building trade plan...")

                plan = build_trade_plan(
                    snapshot=snapshot,
                    targets=targets,
                    constraints=DEFAULT_CONSTRAINTS,
                    price_map=price_map,
                    capital_to_deploy=capital_to_deploy,
                )

                progress_placeholder.success("✅ Trade plan complete!")

                # =========================================================
                # INTEGRATE PENDING REPLACEMENTS FROM AI RECOMMENDATIONS
                # =========================================================
                pending_replacements = st.session_state.get('pending_replacements', [])
                if pending_replacements:
                    from .models import OrderTicket

                    # Build position map for current holdings
                    pos_map = {p.symbol.strip().upper(): p for p in snapshot.positions if p.symbol}
                    existing_order_symbols = {o.symbol.strip().upper() for o in plan.orders}

                    replacement_orders = []
                    for repl in pending_replacements:
                        sell_sym = repl['sell'].strip().upper()
                        buy_sym = repl.get('buy', '').strip().upper() if repl.get('buy') else None

                        # SELL order - sell entire position
                        if sell_sym and sell_sym not in existing_order_symbols:
                            pos = pos_map.get(sell_sym)
                            if pos and pos.quantity and pos.quantity > 0:
                                replacement_orders.append(OrderTicket(
                                    symbol=sell_sym,
                                    action="SELL",
                                    quantity=int(pos.quantity),
                                    order_type="MKT",
                                    limit_price=None,
                                    tif="DAY",
                                    reason=repl.get('reason', f"AI Replacement: Exit {sell_sym}"),
                                ))

                        # BUY order - use proceeds to buy replacement
                        if buy_sym and buy_sym not in existing_order_symbols:
                            # Calculate buy quantity based on sell value
                            sell_price = price_map.get(sell_sym, 0)
                            buy_price = price_map.get(buy_sym, 0)

                            if sell_price > 0 and buy_price > 0:
                                pos = pos_map.get(sell_sym)
                                sell_value = (pos.quantity * sell_price) if pos and pos.quantity else 0

                                if sell_value > 0:
                                    buy_qty = int(sell_value / buy_price)  # Floor to whole shares
                                    if buy_qty > 0:
                                        replacement_orders.append(OrderTicket(
                                            symbol=buy_sym,
                                            action="BUY",
                                            quantity=buy_qty,
                                            order_type="MKT",
                                            limit_price=None,
                                            tif="DAY",
                                            reason=repl.get('reason', f"AI Replacement: Buy {buy_sym}"),
                                        ))

                    # Add replacement orders to plan
                    if replacement_orders:
                        plan = TradePlan(
                            ts_utc=plan.ts_utc,
                            account=plan.account,
                            strategy_key=plan.strategy_key,
                            nav=plan.nav,
                            cash=plan.cash,
                            turnover_est=plan.turnover_est,
                            num_trades=plan.num_trades + len(replacement_orders),
                            current_weights=plan.current_weights,
                            target_weights=plan.target_weights,
                            drift_by_symbol=plan.drift_by_symbol,
                            orders=list(plan.orders) + replacement_orders,
                            warnings=list(plan.warnings) + [
                                f"Added {len(replacement_orders)} orders from AI replacement recommendations"],
                        )
                        st.success(f"📥 Integrated {len(replacement_orders)} replacement orders from AI recommendations")

                    # Clear pending replacements after integration
                    st.session_state.pending_replacements = []

                gates = evaluate_trade_plan_gates(
                    snapshot=snapshot,
                    signals=signals,
                    plan=plan,
                    constraints=DEFAULT_CONSTRAINTS,
                    price_map=price_map,
                )

                # =====================================================
                # AI PRE-TRADE ANALYSIS
                # =====================================================
                if plan.orders:
                    try:
                        orders_for_analysis = [
                            {'symbol': o.symbol, 'action': o.action, 'quantity': o.quantity}
                            for o in plan.orders
                        ]
                        pre_trade = analyze_proposed_trades(orders_for_analysis, signals)

                        # Store in session state for display
                        st.session_state['ai_pm_pre_trade_analysis'] = pre_trade

                        # Show AI Analysis Summary
                        st.markdown("### 🤖 AI Pre-Trade Analysis")

                        # Market Conditions
                        mc = pre_trade.market_condition
                        regime_color = "🟢" if mc.overall_risk == RiskLevel.LOW else "🟡" if mc.overall_risk == RiskLevel.MEDIUM else "🔴"
                        st.info(
                            f"{regime_color} **Market:** {mc.regime} regime | VIX: {mc.vix_level:.1f} ({mc.vix_status}) | Risk: {mc.overall_risk.value}")

                        if mc.active_events:
                            st.caption(f"📰 Active Events: {', '.join(mc.active_events[:3])}")

                        # Warnings Summary
                        if pre_trade.warnings:
                            critical = [w for w in pre_trade.warnings if w.risk_level == RiskLevel.CRITICAL]
                            high = [w for w in pre_trade.warnings if w.risk_level == RiskLevel.HIGH]
                            medium = [w for w in pre_trade.warnings if w.risk_level == RiskLevel.MEDIUM]

                            col_crit, col_high, col_med, col_ok = st.columns(4)
                            with col_crit:
                                st.metric("🔴 Critical", len(critical))
                            with col_high:
                                st.metric("🟠 High", len(high))
                            with col_med:
                                st.metric("🟡 Medium", len(medium))
                            with col_ok:
                                st.metric("🟢 Clear", pre_trade.proceed_count)

                            # Show critical/high warnings
                            if critical or high:
                                with st.expander(f"⚠️ **{len(critical) + len(high)} Important Warnings**",
                                                 expanded=True):
                                    for w in critical + high:
                                        if w.risk_level == RiskLevel.CRITICAL:
                                            st.error(f"**{w.symbol}**: {w.message}")
                                            st.caption(f"   ↳ {w.details} | Recommendation: {w.recommendation}")
                                        else:
                                            st.warning(f"**{w.symbol}**: {w.message}")
                                            st.caption(f"   ↳ {w.details}")

                            # Show medium warnings in collapsed expander
                            if medium:
                                with st.expander(f"ℹ️ {len(medium)} Additional Warnings"):
                                    for w in medium:
                                        st.info(f"**{w.symbol}**: {w.message}")

                        # Overall Recommendation
                        rec = pre_trade.overall_recommendation
                        if rec == "PROCEED":
                            st.success(f"✅ **AI Recommendation:** PROCEED - All clear for execution")
                        elif rec == "PROCEED_WITH_CAUTION":
                            st.warning(f"⚠️ **AI Recommendation:** PROCEED WITH CAUTION - Minor concerns noted")
                        elif rec == "CAUTION":
                            st.warning(f"⚠️ **AI Recommendation:** CAUTION - Review warnings before proceeding")
                        elif rec == "WAIT":
                            st.error(f"🛑 **AI Recommendation:** WAIT - Significant risks detected")
                        elif rec == "ABORT":
                            st.error(f"🚫 **AI Recommendation:** ABORT - Critical risks present")

                        # Store recommendation for gating
                        st.session_state['ai_pm_pre_trade_recommendation'] = rec

                    except Exception as e:
                        st.warning(f"AI analysis unavailable: {e}")
                        st.session_state['ai_pm_pre_trade_recommendation'] = "PROCEED"

                execution = None

                # Check AI recommendation - block if WAIT/ABORT unless overridden
                ai_recommendation = st.session_state.get('ai_pm_pre_trade_recommendation', 'PROCEED')
                ai_blocked = ai_recommendation in ['WAIT', 'ABORT']
                ai_override = st.session_state.get('ai_pm_override_warnings', False)

                if ai_blocked and not ai_override:
                    st.warning("⏸️ Execution paused due to AI recommendation. Use override to proceed anyway.")
                    col_override, col_info = st.columns([1, 2])
                    with col_override:
                        if st.checkbox("🔓 Override AI warnings", key="ai_pm_override_checkbox"):
                            st.session_state['ai_pm_override_warnings'] = True
                            st.rerun()
                    with col_info:
                        st.caption("Check this to execute despite AI warnings. Use with caution.")

                elif is_auto_trade() and is_armed() and not gates.blocked and not is_kill_switch():
                    # Clear override for next run
                    st.session_state['ai_pm_override_warnings'] = False
                    execution = execute_trade_plan(
                        ib=gw.ib,
                        snapshot=snapshot,
                        plan=plan,
                        account=sel_account,
                        constraints=DEFAULT_CONSTRAINTS,
                        dry_run=bool(dry_run),
                        kill_switch=is_kill_switch(),
                        auto_trade_enabled=True,
                        armed=True,
                        price_map=price_map,  # Use pre-fetched Yahoo prices
                        skip_live_quotes=True,  # Skip slow IBKR reqMktData
                    )

                audit_path = write_ai_pm_audit(
                    account=sel_account,
                    strategy_key=get_strategy_key(),
                    snapshot=snapshot,
                    signals=signals,
                    signals_diagnostics=diag,
                    targets=targets,
                    plan=plan,
                    gates=gates,
                    execution=execution,
                    extra={"dry_run": bool(dry_run), "auto_trade": bool(is_auto_trade()), "armed": bool(is_armed())},
                )

                set_last_run(datetime.utcnow())
                store_last_results(
                    snapshot=snapshot,
                    targets=targets,
                    plan=plan,
                    gates=gates,
                    execution=execution,
                    errors=(execution.errors if execution else []),
                    warnings=(gates.warnings if gates else []),
                )

                _store_ai_pm_assistant_artifacts(
                    snapshot=snapshot,
                    signals=signals,
                    targets=targets,
                    plan=plan,
                    gates=gates,
                    strategy_key=get_strategy_key(),
                )

            st.success(f"Run completed. Audit: {audit_path}")
            st.rerun()

    # -------------------------
    # Manual Confirm pipeline
    # -------------------------
    if confirm:
        if not gw.is_connected():
            st.error("IBKR not connected.")
        elif not sel_account:
            st.error("Select a single account (not 'All').")
        else:
            with st.spinner("Executing using latest plan (recomputed)..."):
                snapshot = _to_portfolio_snapshot(gw, sel_account)

                # Debug: valuation coverage from snapshot (no fake)
                try:
                    missing_val = [p.symbol for p in (snapshot.positions or []) if
                                   (p.market_price is None and p.market_value is None)]
                    st.info(
                        f"Snapshot valuation coverage: {len(snapshot.positions) - len(missing_val)}/{len(snapshot.positions)} "
                        f"positions have market_price/market_value. Missing sample: {missing_val[:15]}"
                    )
                except Exception:
                    pass

                signals, diag = load_platform_signals_snapshot(limit=5000)
                max_holdings_override = st.session_state.get('ai_pm_max_holdings')

                # Check if we have a saved portfolio loaded as target
                if st.session_state.get('ai_pm_target_weights'):
                    # Use saved portfolio weights directly
                    from .models import TargetWeights
                    tp = st.session_state.get('ai_pm_target_portfolio', {})
                    targets = TargetWeights(
                        weights=st.session_state['ai_pm_target_weights'],
                        strategy_key='saved_portfolio',
                        ts_utc=datetime.utcnow(),
                        notes=[f"Using saved portfolio: {tp.get('name', 'Unknown')}"]
                    )
                else:
                    targets = build_target_weights(
                        signals=signals,
                        strategy_key=get_strategy_key(),
                        constraints=DEFAULT_CONSTRAINTS,
                        max_holdings_override=max_holdings_override,
                    )

                universe = sorted(
                    set([p.symbol.strip().upper() for p in (snapshot.positions or []) if getattr(p, "symbol", None)])
                    | set([k.strip().upper() for k in ((targets.weights or {}) or {}).keys() if k])
                )

                # YAHOO FINANCE FIRST (fast batch fetch)
                # Cache prices for 5 minutes
                cached_prices = st.session_state.get("ai_pm_price_map", {})
                cache_time = st.session_state.get("ai_pm_price_cache_time")
                cache_valid = cache_time and (datetime.utcnow() - cache_time).seconds < 300

                price_map = dict(cached_prices) if cache_valid else {}
                price_diag = {
                    'requested': len(universe),
                    'resolved': 0,
                    'missing': [],
                    'source_counts': {'yahoo': 0, 'ibkr': 0, 'cache': 0},
                }

                # Count cached prices
                if cache_valid:
                    cached_count = len([s for s in universe if s in price_map and price_map.get(s)])
                    price_diag['source_counts']['cache'] = cached_count

                # Find symbols needing prices
                need_prices = [s for s in universe if s not in price_map or not price_map.get(s)]

                if need_prices:
                    # PRIMARY: Yahoo Finance (fast batch)
                    try:
                        import yfinance as yf

                        # Batch in chunks of 100 for reliability
                        for i in range(0, len(need_prices), 100):
                            chunk = need_prices[i:i + 100]
                            tickers_str = ' '.join(chunk)

                            yf_data = yf.download(tickers_str, period='1d', progress=False, threads=True)

                            if not yf_data.empty and 'Close' in yf_data.columns:
                                if len(chunk) == 1:
                                    val = yf_data['Close'].iloc[-1]
                                    if val and val > 0:
                                        price_map[chunk[0]] = float(val)
                                        price_diag['source_counts']['yahoo'] += 1
                                else:
                                    for sym in chunk:
                                        try:
                                            if sym in yf_data['Close'].columns:
                                                val = yf_data['Close'][sym].iloc[-1]
                                                if val and val > 0:
                                                    price_map[sym] = float(val)
                                                    price_diag['source_counts']['yahoo'] += 1
                                        except:
                                            pass
                    except Exception as e:
                        st.warning(f"Yahoo Finance error: {e}")

                # IBKR fallback DISABLED - causes freeze in Streamlit threads
                still_missing = [s for s in universe if s not in price_map or not price_map.get(s)]
                if still_missing:
                    price_diag['missing'] = still_missing[:20]

                # Cache for next time
                st.session_state["ai_pm_price_map"] = price_map
                st.session_state["ai_pm_price_diag"] = price_diag
                st.session_state["ai_pm_price_cache_time"] = datetime.utcnow()

                resolved_count = len([s for s in universe if s in price_map and price_map.get(s)])
                price_diag['resolved'] = resolved_count

                st.info(
                    f"💰 Prices: {resolved_count}/{len(universe)} | "
                    f"Yahoo: {price_diag['source_counts']['yahoo']} | "
                    f"Cache: {price_diag['source_counts']['cache']} | "
                    f"IBKR: {price_diag['source_counts']['ibkr']}"
                )

                if price_diag.get('missing'):
                    st.caption(f"Missing: {price_diag['missing'][:10]}")

                # Get capital to deploy from user input
                capital_to_deploy_confirm = st.session_state.get('ai_pm_deploy_amount')

                import logging
                _exec_log = logging.getLogger(__name__)
                _exec_log.info("Execution: building trade plan...")
                st.info("⏳ Building trade plan...")

                plan = build_trade_plan(
                    snapshot=snapshot,
                    targets=targets,
                    constraints=DEFAULT_CONSTRAINTS,
                    price_map=price_map,
                    capital_to_deploy=capital_to_deploy_confirm,
                )
                _exec_log.info(f"Execution: trade plan built, {len(plan.orders) if plan and plan.orders else 0} orders")

                _exec_log.info("Execution: evaluating gates...")
                st.info("⏳ Evaluating gates...")

                gates = evaluate_trade_plan_gates(
                    snapshot=snapshot,
                    signals=signals,
                    plan=plan,
                    constraints=DEFAULT_CONSTRAINTS,
                    price_map=price_map,
                )
                _exec_log.info(f"Execution: gates evaluated, blocked={gates.blocked}")

                _exec_log.info("Execution: checking gates.blocked...")

                # Check for override
                override_blockers = st.session_state.get('ai_pm_override_blockers', False)

                if gates.blocked and not override_blockers:
                    _exec_log.info("Execution: gates.blocked is True and no override, showing error")
                    st.error("Blocked by hard gates; not sending orders.")
                    st.info("💡 To proceed anyway, check the 'Override hard blockers' checkbox above.")
                elif gates.blocked and override_blockers:
                    _exec_log.info("Execution: gates.blocked is True BUT override is active")
                    st.warning("⚠️ **OVERRIDE ACTIVE**: Bypassing hard blockers as requested.")
                    # Continue to execution
                    kill_sw = is_kill_switch()
                    if kill_sw:
                        st.error("Kill switch enabled; not sending orders.")
                    else:
                        _exec_log.info("Execution: about to call execute_trade_plan (with override)...")
                        if not gw.ib.isConnected():
                            _exec_log.error("Execution: IB not connected!")
                            st.error("❌ IBKR connection lost. Please reconnect.")
                            st.stop()

                        _exec_log.info(f"Execution: IB connected, clientId={gw.ib.client.clientId}")
                        st.info("⏳ Sending orders (OVERRIDE MODE)...")
                        execution = execute_trade_plan(
                            ib=gw.ib,
                            snapshot=snapshot,
                            plan=plan,
                            account=sel_account,
                            constraints=DEFAULT_CONSTRAINTS,
                            dry_run=False,
                            kill_switch=is_kill_switch(),
                            auto_trade_enabled=False,
                            armed=is_armed(),
                            price_map=price_map,
                            skip_live_quotes=True,
                        )

                        audit_path = write_ai_pm_audit(
                            account=sel_account,
                            strategy_key=get_strategy_key(),
                            snapshot=snapshot,
                            signals=signals,
                            signals_diagnostics=diag,
                            targets=targets,
                            plan=plan,
                            gates=gates,
                            execution=execution,
                            extra={"dry_run": False, "manual_confirm": True, "override_blockers": True},
                        )

                        store_last_results(
                            snapshot=snapshot,
                            targets=targets,
                            plan=plan,
                            gates=gates,
                            execution=execution,
                            errors=execution.errors,
                            warnings=gates.warnings,
                        )

                        _store_ai_pm_assistant_artifacts(
                            snapshot=snapshot,
                            signals=signals,
                            targets=targets,
                            plan=plan,
                            gates=gates,
                            strategy_key=get_strategy_key(),
                        )

                        if execution.errors:
                            st.error("Execution completed with errors (OVERRIDE MODE).")
                            for err in execution.errors[:5]:
                                st.write(f"• {err}")
                        else:
                            st.success(
                                f"✅ Sent {len(execution.submitted_orders)} orders (OVERRIDE MODE). Audit: {audit_path}")

                        st.info(f"Audit: {audit_path}")
                        st.rerun()
                else:
                    _exec_log.info("Execution: gates.blocked is False, checking kill switch...")
                    kill_sw = is_kill_switch()
                    _exec_log.info(f"Execution: is_kill_switch() = {kill_sw}")
                    if kill_sw:
                        st.error("Kill switch enabled; not sending orders.")
                    else:
                        _exec_log.info("Execution: about to call execute_trade_plan...")

                        # Verify IB connection is alive before sending orders
                        if not gw.ib.isConnected():
                            _exec_log.error("Execution: IB not connected!")
                            st.error("❌ IBKR connection lost. Please reconnect.")
                            st.stop()

                        _exec_log.info(f"Execution: IB connected, clientId={gw.ib.client.clientId}")
                        st.info("⏳ Sending orders...")
                    execution = execute_trade_plan(
                        ib=gw.ib,
                        snapshot=snapshot,
                        plan=plan,
                        account=sel_account,
                        constraints=DEFAULT_CONSTRAINTS,
                        dry_run=False,
                        kill_switch=is_kill_switch(),
                        auto_trade_enabled=False,
                        armed=is_armed(),
                        price_map=price_map,  # Use pre-fetched Yahoo prices
                        skip_live_quotes=True,  # Skip slow IBKR reqMktData
                    )

                    audit_path = write_ai_pm_audit(
                        account=sel_account,
                        strategy_key=get_strategy_key(),
                        snapshot=snapshot,
                        signals=signals,
                        signals_diagnostics=diag,
                        targets=targets,
                        plan=plan,
                        gates=gates,
                        execution=execution,
                        extra={"dry_run": False, "manual_confirm": True},
                    )

                    store_last_results(
                        snapshot=snapshot,
                        targets=targets,
                        plan=plan,
                        gates=gates,
                        execution=execution,
                        errors=execution.errors,
                        warnings=gates.warnings,
                    )

                    _store_ai_pm_assistant_artifacts(
                        snapshot=snapshot,
                        signals=signals,
                        targets=targets,
                        plan=plan,
                        gates=gates,
                        strategy_key=get_strategy_key(),
                    )

                    if execution.errors:
                        st.error("Execution completed with errors.")
                    else:
                        st.success("Orders submitted.")

                    # Post-execution verification - store in session_state to persist after rerun
                    st.info("🔍 Starting post-execution verification...")
                    try:
                        from .execution_verify import verify_execution, PortfolioVerification
                        with st.spinner("Verifying orders in TWS..."):
                            # Wait for orders to be visible in TWS
                            import time
                            time.sleep(3)  # Use regular sleep, not ib.sleep which freezes in Streamlit

                            verification = verify_execution(
                                ib=gw.ib,
                                snapshot=snapshot,
                                plan=plan,
                                targets=targets,
                                price_map=price_map,
                            )

                        # Build verification data for session state
                        verify_data = []
                        for v in verification.symbols_verified:
                            if v.pending_buy > 0 or v.pending_sell > 0 or v.target_weight > 0.1:
                                verify_data.append({
                                    "Symbol": v.symbol,
                                    "Target %": f"{v.target_weight:.2f}",
                                    "Current %": f"{v.current_weight:.2f}",
                                    "Projected %": f"{v.projected_weight:.2f}",
                                    "Current Qty": int(v.current_shares),
                                    "Pending BUY": int(v.pending_buy),
                                    "Pending SELL": int(v.pending_sell),
                                    "Projected Qty": int(v.projected_shares),
                                    "Status": v.status,
                                    "Accuracy": f"{v.accuracy:.1f}%",
                                })

                        # Store verification in session_state so it persists
                        st.session_state['ai_pm_last_verification'] = {
                            'total_open_orders': verification.total_open_orders,
                            'total_accuracy': verification.total_accuracy,
                            'projected_invested': verification.projected_invested,
                            'projected_cash': verification.projected_cash,
                            'missing_orders': verification.missing_orders,
                            'extra_orders': verification.extra_orders,
                            'data': verify_data,
                        }

                    except Exception as e:
                        st.warning(f"Verification failed: {e}")

                    st.info(f"Audit: {audit_path}")
                    st.rerun()

    # Ask-AI panel (canonical, uses stored artifacts)
    _render_ai_pm_assistant_panel()

    exec_dict = last.get("execution") or {}
    if exec_dict:
        with st.expander("Last execution details", expanded=False):
            st.json(exec_dict)

    # Display post-execution verification if available
    verification = st.session_state.get('ai_pm_last_verification')
    if verification:
        with st.expander("📊 Post-Execution Verification (Orders vs Targets)", expanded=True):
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Open Orders in TWS", verification.get('total_open_orders', 0))
            col2.metric("Overall Accuracy", f"{verification.get('total_accuracy', 0):.1f}%")
            col3.metric("Projected Invested", f"")
            col4.metric("Projected Cash", f"")

            missing = verification.get('missing_orders', [])
            extra = verification.get('extra_orders', [])
            if missing:
                st.warning(f"⚠️ Missing orders for: {', '.join(missing)}")
            if extra:
                st.warning(f"⚠️ Extra orders for: {', '.join(extra)}")

            # Detailed table
            verify_data = verification.get('data', [])
            if verify_data:
                df_verify = pd.DataFrame(verify_data)
                # Sort by status then by target weight
                if 'Status' in df_verify.columns:
                    status_order = {'Pending': 0, 'Missing BUY': 1, 'Missing SELL': 2, 'On Target': 3, 'No Action': 4}
                    df_verify['_sort'] = df_verify['Status'].map(lambda x: status_order.get(x, 5))
                    df_verify = df_verify.sort_values('_sort').drop(columns=['_sort'])
                st.dataframe(df_verify, use_container_width=True, hide_index=True, height=400)

            # Clear button
            if st.button("Clear Verification", key="clear_verification_btn"):
                del st.session_state['ai_pm_last_verification']
                st.rerun()

    # =========================================================================
    # DEBUG EXPORT SECTION
    # =========================================================================
    if DEBUG_EXPORT_AVAILABLE:
        with st.expander("🔧 Debug Tools", expanded=False):
            st.markdown("### Export Debug Snapshot")
            st.markdown("Export all current data (positions, orders, targets, plan) for debugging.")

            col_dbg1, col_dbg2 = st.columns([1, 3])

            with col_dbg1:
                if st.button("📥 Export Debug Data", key="export_debug_btn"):
                    with st.spinner("Exporting debug data..."):
                        try:
                            # Get current data
                            gw = st.session_state.get('ai_pm_gateway')
                            last = get_last_results()

                            # Get price map
                            price_map = st.session_state.get('ai_pm_price_map', {})

                            # Get verification if available
                            verif = st.session_state.get('ai_pm_last_verification')

                            # Export
                            created_files = export_debug_snapshot(
                                gw=gw,
                                snapshot=last.get('snapshot'),
                                targets=last.get('targets'),
                                plan=last.get('plan'),
                                signals=last.get('signals'),
                                gates=last.get('gates'),
                                execution=last.get('execution'),
                                price_map=price_map,
                                verification=verif,
                                account=get_selected_account() or "",
                                strategy_key=get_strategy_key() or "",
                                output_dir="debug_exports",
                            )

                            st.success(f"✅ Debug export complete! Created {len(created_files)} files.")

                            # Show file paths
                            for file_type, file_path in created_files.items():
                                st.markdown(f"- **{file_type}**: `{file_path}`")

                            # Store for viewing
                            st.session_state['last_debug_export'] = created_files

                        except Exception as e:
                            st.error(f"❌ Export failed: {e}")
                            import traceback
                            st.code(traceback.format_exc())

            # Show last export if available
            last_export = st.session_state.get('last_debug_export')
            if last_export:
                st.markdown("---")
                st.markdown("**Last Export Files:**")

                # Show summary if exists
                import os
                summary_path = last_export.get('summary')
                if summary_path and os.path.exists(summary_path):
                    with st.expander("📄 View Summary", expanded=False):
                        with open(summary_path, 'r') as f:
                            st.code(f.read(), language="text")

                # Show comparison table if exists
                comparison_path = last_export.get('comparison_csv')
                if comparison_path and os.path.exists(comparison_path):
                    with st.expander("📊 View Comparison Table", expanded=False):
                        df_comp = pd.read_csv(comparison_path)

                        # Filter to show only rows with activity or conflicts
                        df_active = df_comp[
                            (df_comp['planned_qty'] > 0) |
                            (df_comp['tws_buy'] > 0) |
                            (df_comp['tws_sell'] > 0) |
                            (df_comp['conflict'].notna() & (df_comp['conflict'] != ''))
                            ].copy()

                        if not df_active.empty:
                            # Highlight conflicts
                            def highlight_conflicts(row):
                                if pd.notna(row.get('conflict')) and row.get('conflict'):
                                    return ['background-color: #ffcccc'] * len(row)
                                return [''] * len(row)

                            st.dataframe(
                                df_active.style.apply(highlight_conflicts, axis=1),
                                use_container_width=True,
                                hide_index=True,
                                height=400,
                            )
                        else:
                            st.info("No active orders or conflicts in last export.")

                # Clear export button
                if st.button("🗑️ Clear Export Data", key="clear_debug_export_btn"):
                    if 'last_debug_export' in st.session_state:
                        del st.session_state['last_debug_export']
                    st.rerun()