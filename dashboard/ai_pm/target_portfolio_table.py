# target_portfolio_table.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


# -------------------------------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------------------------------

# Raw keys you do NOT want to show even if present under target["raw"].
# Leave empty to show everything; add keys here to hide them.
EXCLUDED_RAW_KEYS: Set[str] = {
    # examples:
    # "sentiment_score",
    # "fundamental_score",
    # "gap_score",
    # "peg_ratio",
    # "ex_dividend_date",
    # "analyst_positivity",
    # "likelihood_score",
}

# Base (computed) columns you do NOT want to show even if computed into the base row.
EXCLUDED_BASE_COLUMNS: Set[str] = {
    # examples:
    # "Sentiment",
    # "Fundamental",
    # "Likelihood",
}


# If your DB enrichment columns are ratios (0..1) but should be displayed as %,
# list them here by *final* dataframe column name.
# NOTE: This file also supports auto-detection if a column name ends with "_pct" or contains "pct".
PCT_RATIO_COLUMNS: Set[str] = {
    "dividend_yield",  # often 0.03 -> 3.00
}

# If a column is already in percentage points (e.g. 66 means 66%), list it here.
PCT_ALREADY_COLUMNS: Set[str] = {
    "target_upside_pct",
    "Target W%",
}


# -------------------------------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------------------------------


def _normalize_symbol(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper()
    return s if s else None


def _is_empty_cell(v: Any) -> bool:
    """True if v should be considered empty for 'drop all-empty columns'."""
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    if isinstance(v, (list, tuple, set, dict)) and len(v) == 0:
        return True
    return False


def _drop_all_empty_columns(df: pd.DataFrame, protect: Optional[Set[str]] = None) -> pd.DataFrame:
    protect = protect or set()
    keep_cols: List[str] = []
    for c in df.columns:
        if c in protect:
            keep_cols.append(c)
            continue
        col = df[c]
        # If there is ANY non-empty cell, keep the column
        if any(not _is_empty_cell(v) for v in col.tolist()):
            keep_cols.append(c)
    return df[keep_cols]


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Best-effort conversion of columns to numeric dtype so downstream rounding works.
    Keeps columns as-is if conversion yields no non-null numeric values.
    """
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind in ("i", "u", "f"):
            continue  # already numeric
        # Try convert; if we get any numeric values, adopt it
        s = pd.to_numeric(out[c], errors="coerce")
        if s.notna().any():
            out[c] = s
    return out


def _maybe_scale_pct_ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        cn = str(c)
        if cn in PCT_ALREADY_COLUMNS:
            continue
        if cn in PCT_RATIO_COLUMNS or cn.endswith("_pct_ratio") or ("pct" in cn and cn not in PCT_ALREADY_COLUMNS):
            s = pd.to_numeric(out[c], errors="coerce")
            # Heuristic: if values look like 0..1 ratios, multiply by 100
            if s.notna().any():
                mx = s.max(skipna=True)
                if mx is not None and mx <= 1.5:
                    out[c] = s * 100.0
    return out


def _apply_numeric_rounding(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    out = df.copy()
    num_cols = [c for c in out.columns if out[c].dtype.kind in ("i", "u", "f")]
    if num_cols:
        out[num_cols] = out[num_cols].round(decimals)
    return out


def _humanize_big_number(x: Any) -> Optional[str]:
    v = _safe_float(x)
    if v is None:
        return None
    a = abs(v)
    sign = "-" if v < 0 else ""

    # T, B, M, K
    if a >= 1e12:
        return f"{sign}{a/1e12:.1f}T"
    if a >= 1e11:
        return f"{sign}{a/1e9:.0f}B"
    if a >= 1e9:
        return f"{sign}{a/1e9:.1f}B"
    if a >= 1e8:
        return f"{sign}{a/1e6:.0f}M"
    if a >= 1e6:
        return f"{sign}{a/1e6:.1f}M"
    if a >= 1e3:
        return f"{sign}{a/1e3:.1f}K"
    return f"{sign}{a:.0f}"


def _apply_humanized_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Market cap: prefer compact representation
    for cap_col in ("market_cap", "Market Cap", "mkt_cap"):
        if cap_col in out.columns:
            out[cap_col] = out[cap_col].map(_humanize_big_number)
    return out


def _format_all_numeric_to_2dp(df: pd.DataFrame, protect: Optional[Set[str]] = None) -> pd.DataFrame:
    """
    Force *display* formatting of all numeric columns to exactly 2 decimals (XX.xx).
    This avoids Streamlit/pandas showing 6+ decimals when column_config formatting is ignored.

    Returns a dataframe suitable for display (numbers become strings).
    """
    protect = protect or set()
    out = df.copy()

    for c in out.columns:
        if c in protect:
            continue

        # If pandas thinks numeric, format directly
        if out[c].dtype.kind in ("i", "u", "f"):
            out[c] = out[c].map(lambda v: None if _is_empty_cell(v) else f"{float(v):.2f}")
            continue

        # If object column but numeric-like, format values that can be parsed
        # (leave non-numeric values unchanged)
        def _fmt(v: Any):
            if _is_empty_cell(v):
                return None
            fv = _safe_float(v)
            if fv is None:
                return v
            return f"{fv:.2f}"

        out[c] = out[c].map(_fmt)

    return out


# -------------------------------------------------------------------------------------------------
# STREAMLIT COLUMN CONFIG
# -------------------------------------------------------------------------------------------------


def _column_config_for_streamlit(st, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build Streamlit column_config. Works even if some columns are strings.
    """
    col_cfg: Dict[str, Any] = {}

    # Basic width hints
    text_small = dict(width="small")
    text_medium = dict(width="medium")

    for c in df.columns:
        name = str(c)

        # common columns
        if name in ("Rank",):
            col_cfg[name] = st.column_config.TextColumn(label=name, **text_small)
            continue

        if name in ("Symbol", "Ticker", "ticker"):
            col_cfg[name] = st.column_config.TextColumn(label=name, **text_small)
            continue

        if name.lower().endswith("date") or "date" in name.lower():
            col_cfg[name] = st.column_config.TextColumn(label=name, **text_small)
            continue

        if name in ("sector", "Sector"):
            col_cfg[name] = st.column_config.TextColumn(label=name, **text_medium)
            continue

        if name in ("Action", "Reason"):
            col_cfg[name] = st.column_config.TextColumn(label=name, **text_medium)
            continue

        # Market cap (humanized string)
        if name in ("market_cap", "Market Cap", "mkt_cap"):
            col_cfg[name] = st.column_config.TextColumn(label=name, **text_small)
            continue

        # Everything else: show as text (since we force 2-decimal strings for numeric display)
        col_cfg[name] = st.column_config.TextColumn(label=name, **text_small)

    return col_cfg


# -------------------------------------------------------------------------------------------------
# MAIN RENDER FUNCTION
# -------------------------------------------------------------------------------------------------


def render_proposed_target_holdings_full_scanner_columns(
    *,
    st,
    ai_targets: Sequence[Dict[str, Any]],
    title: str = "Proposed Target Holdings (Full Scanner Columns)",
    hide_empty_cols: bool = True,
) -> None:
    """
    Renders the AI PM target list as a table, including ALL raw scanner/enrichment columns.

    ai_targets expected shape (typical):
      [
        {
          "symbol": "AAPL",
          "target_weight": 0.032,
          "action": "BUY"/"SELL"/"HOLD"/None,
          "reason": "...",
          "raw": { ... scanner columns ... }
        },
        ...
      ]

    Rules:
    - Include all raw keys unless excluded by EXCLUDED_RAW_KEYS.
    - Drop any columns that are entirely empty across all rows (if hide_empty_cols=True).
    - Display numbers with exactly 2 decimals.
    - Humanize market cap like 120B / 4.1T.
    """
    if not ai_targets:
        st.info("No proposed target holdings to display.")
        return

    # 1) Discover all raw keys across targets
    raw_keys: Set[str] = set()
    for t in ai_targets:
        raw = t.get("raw")
        if isinstance(raw, dict):
            raw_keys |= set(raw.keys())

    # Keep stable order: base cols first, then sorted raw keys
    ordered_raw_cols = sorted(raw_keys)

    # 2) Build rows
    rows: List[Dict[str, Any]] = []
    for i, t in enumerate(ai_targets, start=1):
        sym = _normalize_symbol(t.get("symbol") or t.get("ticker"))
        if not sym:
            continue

        tw = t.get("target_weight", t.get("weight"))
        twf = _safe_float(tw)
        tw_pct = None if twf is None else twf * 100.0

        row: Dict[str, Any] = {
            "Rank": i,
            "Symbol": sym,
            "Target W%": tw_pct,
            "Action": t.get("action"),
            "Reason": t.get("reason"),
        }

        # Raw/scanner keys
        raw = t.get("raw") if isinstance(t.get("raw"), dict) else {}
        for k in ordered_raw_cols:
            if k in EXCLUDED_RAW_KEYS:
                continue
            row[k] = raw.get(k)

        rows.append(row)

    if not rows:
        st.info("No rows to display.")
        return

    df = pd.DataFrame(rows)

    # Base columns
    base_cols = ["Rank", "Symbol", "Target W%", "Action", "Reason"]
    base_cols = [c for c in base_cols if c not in EXCLUDED_BASE_COLUMNS]

    # Ensure base cols exist
    for c in base_cols:
        if c not in df.columns:
            df[c] = None

    # Reorder
    cols = base_cols + [c for c in ordered_raw_cols if c in df.columns]
    cols += [c for c in df.columns if c not in set(cols)]
    df = df[cols]

    # Coerce numeric columns so rounding/scaling works
    df = _coerce_numeric_columns(df)

    # Scale ratio columns -> percent (where appropriate)
    df = _maybe_scale_pct_ratios(df)

    # Round ALL numeric columns to 2 decimals (internal)
    df = _apply_numeric_rounding(df)

    # Humanize market cap
    df = _apply_humanized_columns(df)

    # Drop all-empty columns
    if hide_empty_cols:
        df = _drop_all_empty_columns(df, protect={"Rank", "Symbol"})

    # Force display formatting to exactly 2 decimals for any numeric values
    df_display = _format_all_numeric_to_2dp(df, protect={"Rank"})

    st.subheader(title)

    try:
        col_cfg = _column_config_for_streamlit(st, df_display)
    except Exception:
        col_cfg = {}

    st.dataframe(
        df_display,
        width='stretch',
        hide_index=True,
        column_config=col_cfg,
    )

    with st.expander("Debug: raw keys per target", expanded=False):
        for t in ai_targets[:50]:
            sym = _normalize_symbol(t.get("symbol") or t.get("ticker"))
            raw = t.get("raw") if isinstance(t.get("raw"), dict) else {}
            st.write(sym, sorted(raw.keys()))
