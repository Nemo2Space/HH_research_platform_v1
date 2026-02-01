# dashboard/ai_pm/signal_adapter.py
"""
FIXED VERSION v2 - Full DB Enrichment for AI Portfolio Manager

This version enriches signals with ALL scanner columns from:
- screener_scores (total_score, likelihood_score)
- trading_signals (sentiment_score, fundamental_score, gap_score, etc.)
- fundamentals (sector, pe_ratio, market_cap, dividend_yield, roe, etc.)
- analyst_ratings (analyst_positivity, buy_count, total_ratings)
- price_targets (target_mean, target_upside_pct)
- earnings_calendar (earnings_date)

This ensures the Target Portfolio Analysis table shows the same rich data
as your main Signals/Scanner page.

Key fixes:
1. Added _fetch_full_enrichment_via_sql() to pull from fundamentals/ratings/targets/earnings
2. Properly merges enrichment data with base scores/signals
3. Preserves ALL columns in SignalRow.raw dict for UI display
4. Adds comprehensive diagnostics for debugging
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .models import SignalRow, SignalSnapshot

# Robust import for Repository (your project layout may vary)
try:
    from src.db.repository import Repository  # type: ignore
except Exception:  # pragma: no cover
    try:
        from repository import Repository  # type: ignore
    except Exception:  # pragma: no cover
        Repository = None  # type: ignore


def _to_dt(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    try:
        if isinstance(x, datetime):
            return x
        if isinstance(x, float) and pd.isna(x):
            return None
        return pd.to_datetime(x, utc=True, errors="coerce").to_pydatetime()
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
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


def _clamp01(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _norm0100_to_01(x: Any) -> Optional[float]:
    v = _safe_float(x)
    if v is None:
        return None
    # treat >1 as 0..100 score
    if v > 1.0:
        return _clamp01(v / 100.0)
    return _clamp01(v)


def _pick_ticker_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("ticker", "symbol", "Ticker", "Symbol"):
        if c in df.columns:
            return c
    return None


def _standardize_tickers(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    tcol = _pick_ticker_col(out)
    if tcol and tcol != "ticker":
        out = out.rename(columns={tcol: "ticker"})
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    return out


def _merge_latest_scores_and_signals(scores_df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
    s = _standardize_tickers(scores_df)
    g = _standardize_tickers(signals_df)

    if not s.empty and "date" in s.columns:
        s = s.sort_values(["ticker", "date"], ascending=[True, False]).drop_duplicates("ticker", keep="first")
    if not g.empty and "date" in g.columns:
        g = g.sort_values(["ticker", "date"], ascending=[True, False]).drop_duplicates("ticker", keep="first")

    if s.empty and g.empty:
        return pd.DataFrame()
    if s.empty:
        return g
    if g.empty:
        return s

    return pd.merge(s, g, on="ticker", how="outer", suffixes=("_score", "_signal"))


def _coalesce_into(df: pd.DataFrame, target_col: str, sources: List[str]) -> None:
    """
    Create/overwrite target_col by coalescing sources (first non-null wins).
    """
    if df is None or df.empty:
        return
    if target_col not in df.columns:
        df[target_col] = pd.NA

    for c in sources:
        if c in df.columns:
            df[target_col] = df[target_col].combine_first(df[c])


def _canonicalize_ai_probability(df: pd.DataFrame) -> List[str]:
    """
    Creates canonical df['ai_probability'] in 0..1 range.
    Returns list of probability source columns used/available.
    """
    if df is None or df.empty:
        if df is not None and "ai_probability" not in df.columns:
            df["ai_probability"] = pd.NA
        return []

    cols = list(df.columns)
    lc = {c: c.lower() for c in cols}

    def _match(names: List[str]) -> List[str]:
        want = set(names)
        return [c for c in cols if lc.get(c, "") in want]

    # Priority: explicit ai_probability -> probability-ish -> likelihood_score_norm -> likelihood_score
    sources: List[str] = []
    sources += _match(["ai_probability", "ai_prob", "win_probability", "probability", "prob_win", "prob_win_5d"])
    sources += _match(["likelihood_score_norm"])
    sources += _match(["likelihood_score"])

    # Also consider suffixed versions (common after merges)
    for base in ["ai_probability", "probability", "likelihood_score_norm", "likelihood_score"]:
        for sfx in ["_score", "_signal"]:
            c = f"{base}{sfx}"
            if c in df.columns and c not in sources:
                sources.append(c)

    _coalesce_into(df, "ai_probability", sources)

    # Normalize to numeric 0..1
    v = pd.to_numeric(df["ai_probability"], errors="coerce")
    v = v.where(v.isna() | (v <= 1.0), v / 100.0)
    v = v.clip(lower=0.0, upper=1.0)
    df["ai_probability"] = v

    return sources


def _canonicalize_ev(df: pd.DataFrame) -> List[str]:
    """
    Creates canonical df['ai_ev'] in decimal units (0.01 == 1%).
    Returns list of EV source columns used/available.
    """
    if df is None or df.empty:
        if df is not None and "ai_ev" not in df.columns:
            df["ai_ev"] = pd.NA
        return []

    cols = list(df.columns)
    lc = {c: c.lower() for c in cols}

    def _match(names: List[str]) -> List[str]:
        want = set(names)
        return [c for c in cols if lc.get(c, "") in want]

    sources: List[str] = []
    sources += _match(["ai_ev", "expected_value", "ev", "expected_value_5d"])
    for base in ["ai_ev", "expected_value", "ev", "expected_value_5d"]:
        for sfx in ["_score", "_signal"]:
            c = f"{base}{sfx}"
            if c in df.columns and c not in sources:
                sources.append(c)

    _coalesce_into(df, "ai_ev", sources)

    v = pd.to_numeric(df["ai_ev"], errors="coerce")
    # If looks percent-like (abs > 1), normalize to decimal
    v = v.where(v.isna() | (v.abs() <= 1.0), v / 100.0)
    df["ai_ev"] = v

    return sources


def _get_table_columns(engine: Any, table_name: str, schema: str = "public") -> List[str]:
    """Get list of columns for a table."""
    try:
        q = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %(schema)s AND table_name = %(table)s
        ORDER BY ordinal_position
        """
        df = pd.read_sql(q, engine, params={"schema": schema, "table": table_name})
        return df["column_name"].astype(str).tolist() if not df.empty else []
    except Exception:
        return []


def _fetch_full_enrichment_via_sql(engine: Any, tickers: List[str], diag: Dict[str, Any]) -> pd.DataFrame:
    """
    Fetches enrichment data from multiple tables:
    - prices (latest close)
    - fundamentals (sector, pe_ratio, market_cap, etc.)
    - analyst_ratings
    - price_targets
    - earnings_calendar

    This matches the same query pattern as debug_ai_pm_columns_v1.py
    """
    if not tickers or engine is None:
        return pd.DataFrame()

    # Discover available columns in each table
    fund_cols = set(_get_table_columns(engine, "fundamentals"))
    ratings_cols = set(_get_table_columns(engine, "analyst_ratings"))
    targets_cols = set(_get_table_columns(engine, "price_targets"))

    diag["available_fundamentals_cols"] = sorted(list(fund_cols))[:30]

    # Build fundamentals select clause - include all available useful columns
    fund_cols_wanted = [
        "sector", "ex_dividend_date", "dividend_yield", "market_cap",
        "pe_ratio", "forward_pe", "peg_ratio", "roe", "pb_ratio",
        "revenue_growth", "eps", "current_ratio", "debt_to_equity",
        "dividend_payout_ratio", "earnings_growth", "eps_growth",
        "gross_margin", "operating_margin", "profit_margin", "ps_ratio",
        "industry", "country", "currency", "beta", "net_margin",
        "fcf_margin", "quick_ratio", "price_to_sales", "ev_to_ebitda",
        "payout_ratio", "shares_outstanding", "float_shares",
        "short_float", "short_ratio"
    ]

    fund_select_parts = []
    for col in fund_cols_wanted:
        if col in fund_cols:
            fund_select_parts.append(f"f.{col}")

    # Always include date as fundamentals_date
    if "date" in fund_cols:
        fund_select_parts.append("f.date AS fundamentals_date")

    fund_select_sql = ", ".join(fund_select_parts) if fund_select_parts else "NULL AS no_fundamentals"

    # Build ratings select clause
    ratings_parts = []
    if "date" in ratings_cols:
        ratings_parts.append("ar.date AS ratings_date")
    if "analyst_positivity" in ratings_cols:
        ratings_parts.append("ar.analyst_positivity")
    if "analyst_buy" in ratings_cols:
        ratings_parts.append("ar.analyst_buy AS buy_count")
    if "analyst_total" in ratings_cols:
        ratings_parts.append("ar.analyst_total AS total_ratings")

    ratings_select_sql = ", ".join(ratings_parts) if ratings_parts else "NULL AS no_ratings"

    # Build targets select clause
    targets_parts = []
    if "date" in targets_cols:
        targets_parts.append("pt.date AS targets_date")
    if "target_mean" in targets_cols:
        targets_parts.append("pt.target_mean")

    targets_select_sql = ", ".join(targets_parts) if targets_parts else "NULL AS no_targets"

    q = f"""
    WITH t AS (
      SELECT UNNEST(%(tickers)s::text[]) AS ticker
    ),
    latest_prices AS (
      SELECT DISTINCT ON (p.ticker)
        p.ticker,
        p.close AS db_price,
        p.date AS price_date
      FROM prices p
      WHERE p.ticker = ANY(%(tickers)s)
      ORDER BY p.ticker, p.date DESC NULLS LAST
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
        {ratings_select_sql}
      FROM analyst_ratings ar
      WHERE ar.ticker = ANY(%(tickers)s)
      ORDER BY ar.ticker, ar.date DESC NULLS LAST
    ),
    latest_targets AS (
      SELECT DISTINCT ON (pt.ticker)
        pt.ticker,
        {targets_select_sql}
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
      lp.db_price,
      lp.price_date,
      lf.*,
      lr.{ratings_select_sql.split(',')[0].split(' AS ')[0].replace('ar.', '') if ratings_parts else 'ticker'},
      {"lr." + ", lr.".join([p.split(" AS ")[0].replace("ar.", "") for p in ratings_parts[1:]]) + "," if len(ratings_parts) > 1 else ""}
      lt.{targets_select_sql.split(',')[0].split(' AS ')[0].replace('pt.', '') if targets_parts else 'ticker'},
      {"lt." + ", lt.".join([p.split(" AS ")[0].replace("pt.", "") for p in targets_parts[1:]]) + "," if len(targets_parts) > 1 else ""}
      CASE WHEN lp.db_price > 0 AND lt.target_mean > 0
           THEN ROUND(((lt.target_mean - lp.db_price) / lp.db_price * 100)::numeric, 2)
           ELSE NULL END AS target_upside_pct,
      ep.earnings_date
    FROM t
    LEFT JOIN latest_prices lp ON lp.ticker = t.ticker
    LEFT JOIN latest_fundamentals lf ON lf.ticker = t.ticker
    LEFT JOIN latest_ratings lr ON lr.ticker = t.ticker
    LEFT JOIN latest_targets lt ON lt.ticker = t.ticker
    LEFT JOIN earnings_pick ep ON ep.ticker = t.ticker
    ORDER BY t.ticker
    """

    # Actually, let me build this more carefully to avoid SQL syntax issues
    q = f"""
    WITH t AS (
      SELECT UNNEST(%(tickers)s::text[]) AS ticker
    ),
    latest_prices AS (
      SELECT DISTINCT ON (p.ticker)
        p.ticker,
        p.close AS db_price,
        p.date AS price_date
      FROM prices p
      WHERE p.ticker = ANY(%(tickers)s)
      ORDER BY p.ticker, p.date DESC NULLS LAST
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
        ar.date AS ratings_date,
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
        pt.date AS targets_date,
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
      lp.db_price,
      lp.price_date,
      lf.*,
      lr.ratings_date,
      lr.analyst_positivity,
      lr.buy_count,
      lr.total_ratings,
      lt.targets_date,
      lt.target_mean,
      CASE WHEN lp.db_price > 0 AND lt.target_mean > 0
           THEN ROUND(((lt.target_mean - lp.db_price) / lp.db_price * 100)::numeric, 2)
           ELSE NULL END AS target_upside_pct,
      ep.earnings_date
    FROM t
    LEFT JOIN latest_prices lp ON lp.ticker = t.ticker
    LEFT JOIN latest_fundamentals lf ON lf.ticker = t.ticker
    LEFT JOIN latest_ratings lr ON lr.ticker = t.ticker
    LEFT JOIN latest_targets lt ON lt.ticker = t.ticker
    LEFT JOIN earnings_pick ep ON ep.ticker = t.ticker
    ORDER BY t.ticker
    """

    try:
        df = pd.read_sql(q, engine, params={"tickers": tickers})

        # Remove duplicate ticker columns from lf.* expansion
        ticker_cols = [c for c in df.columns if c == "ticker"]
        if len(ticker_cols) > 1:
            # Keep only first ticker column
            df = df.loc[:, ~df.columns.duplicated()]

        diag["enrichment_rows"] = len(df)
        diag["enrichment_columns"] = list(df.columns)
        diag["enrichment_success"] = True
        return df
    except Exception as e:
        diag["enrichment_error"] = str(e)
        diag["enrichment_success"] = False
        return pd.DataFrame()


def load_platform_signals_snapshot(limit: int = 5000) -> Tuple[SignalSnapshot, Dict[str, Any]]:
    """
    Loads latest screener_scores + trading_signals, merges them,
    THEN enriches with fundamentals/ratings/targets/earnings from DB,
    canonicalizes ai_probability, and builds SignalSnapshot.

    Returns:
      (SignalSnapshot, diagnostics_dict)
    """
    if Repository is None:
        snap = SignalSnapshot(ts_utc=datetime.utcnow(), rows={})
        return snap, {"error": "Repository import failed in signal_adapter.py"}

    repo = Repository()

    diag: Dict[str, Any] = {
        "scores_rows": 0,
        "signals_rows": 0,
        "merged_rows": 0,
        "probability_columns": [],
        "ev_columns": [],
        "symbols_total": 0,
        "signals_missing_rows": 0,
        "sector_unknown_final": 0,
        "ai_prob_missing": 0,
        "samples": [],
        "enrichment_rows": 0,
        "enrichment_columns": [],
        "enrichment_success": False,
    }

    # 1) Pull base data from repo
    try:
        scores_df = repo.get_latest_scores(limit=limit)
        signals_df = repo.get_latest_signals()
    except Exception as e:
        snap = SignalSnapshot(ts_utc=datetime.utcnow(), rows={})
        diag["error"] = f"repo read failed: {e}"
        return snap, diag

    diag["scores_rows"] = int(0 if scores_df is None else len(scores_df))
    diag["signals_rows"] = int(0 if signals_df is None else len(signals_df))

    if scores_df is None or scores_df.empty:
        snap = SignalSnapshot(ts_utc=datetime.utcnow(), rows={})
        diag["error"] = "scores_df empty"
        return snap, diag

    # signals_df can be empty; still proceed using scores_df
    if signals_df is None:
        signals_df = pd.DataFrame()

    # 2) Merge scores + signals (outer)
    merged = _merge_latest_scores_and_signals(scores_df, signals_df)
    if merged.empty:
        snap = SignalSnapshot(ts_utc=datetime.utcnow(), rows={})
        diag["error"] = "merged empty"
        return snap, diag

    diag["merged_rows"] = int(len(merged))

    # Ensure ticker normalized
    merged = _standardize_tickers(merged)

    # 3) Get all tickers for enrichment
    tickers = merged["ticker"].dropna().astype(str).str.strip().str.upper().unique().tolist()
    diag["symbols_total"] = len(tickers)

    # 4) Fetch full enrichment from DB (fundamentals, ratings, targets, earnings)
    engine = getattr(repo, "engine", None)
    if engine is not None and tickers:
        enrich_df = _fetch_full_enrichment_via_sql(engine, tickers, diag)

        if not enrich_df.empty:
            # Standardize ticker column
            enrich_df = _standardize_tickers(enrich_df)

            # Identify columns that exist in both dataframes
            existing_cols = set(merged.columns)
            new_cols = set(enrich_df.columns) - {"ticker"}
            overlap_cols = existing_cols & new_cols

            # For overlapping columns, we'll use suffixes and then coalesce
            if overlap_cols:
                merged = pd.merge(
                    merged,
                    enrich_df,
                    on="ticker",
                    how="left",
                    suffixes=("", "_enrich")
                )
                # Coalesce overlapping columns (prefer enrichment if original is null)
                for col in overlap_cols:
                    enrich_col = f"{col}_enrich"
                    if enrich_col in merged.columns:
                        merged[col] = merged[col].combine_first(merged[enrich_col])
                        merged = merged.drop(columns=[enrich_col])
            else:
                merged = pd.merge(merged, enrich_df, on="ticker", how="left")

            diag["enrichment_merged"] = True
    else:
        diag["enrichment_merged"] = False
        diag["enrichment_reason"] = "no engine or no tickers"

    # 5) Canonicalize probability + EV
    diag["probability_columns"] = _canonicalize_ai_probability(merged)
    diag["ev_columns"] = _canonicalize_ev(merged)

    # 6) Ensure sector column exists and handle missing
    if "sector" not in merged.columns:
        merged["sector"] = "Unknown"
    else:
        # Fill missing/blank sectors
        s = merged["sector"].fillna("").astype(str).str.strip()
        missing_mask = s.eq("") | s.str.lower().isin(["unknown", "none", "nan"])
        merged.loc[missing_mask, "sector"] = "Unknown"

    diag["sector_unknown_final"] = int((merged["sector"].astype(str).str.strip().str.lower() == "unknown").sum())

    # 7) Diagnostics
    diag["signals_missing_rows"] = int(merged["ticker"].isna().sum()) if "ticker" in merged.columns else int(
        len(merged))
    diag["ai_prob_missing"] = int(pd.to_numeric(merged["ai_probability"], errors="coerce").isna().sum())
    diag["final_columns"] = list(merged.columns)

    # 8) Build SignalSnapshot rows (KEEP ALL COLUMNS IN raw)
    rows: Dict[str, SignalRow] = {}
    now = datetime.utcnow()

    # choose a reasonable asof column
    asof_col = None
    for c in ["created_at", "date", "score_date", "signal_date", "asof", "timestamp", "ts_utc"]:
        if c in merged.columns:
            asof_col = c
            break

    for _, r in merged.iterrows():
        sym = str(r.get("ticker") or "").strip().upper()
        if not sym:
            continue

        raw: Dict[str, Any] = r.to_dict()

        # guarantee canonical keys in raw
        raw["ticker"] = sym
        raw["ai_probability"] = _safe_float(raw.get("ai_probability"))
        raw["ai_ev"] = _safe_float(raw.get("ai_ev"))
        raw["sector"] = raw.get("sector") if raw.get("sector") is not None else "Unknown"

        ers = _norm0100_to_01(raw.get("ai_probability"))
        if ers is None:
            # fallback to total_score
            ers = _norm0100_to_01(raw.get("total_score"))

        conv = _norm0100_to_01(raw.get("likelihood_score"))

        rf = _norm0100_to_01(raw.get("risk_flag_score"))

        asof_val = _to_dt(raw.get(asof_col)) if asof_col else None

        rows[sym] = SignalRow(
            symbol=sym,
            expected_return_score=ers,
            risk_flag_score=rf,
            conviction_score=conv,
            raw=raw,
            asof_utc=asof_val,
            source="db",
        )

    # 9) Samples (first 10 tickers with full detail)
    sample_syms = list(rows.keys())[:10]
    samples: List[Dict[str, Any]] = []
    for s in sample_syms:
        rr = rows[s].raw or {}
        samples.append(
            {
                "symbol": s,
                "sector": rr.get("sector"),
                "ai_probability": rr.get("ai_probability"),
                "ai_ev": rr.get("ai_ev"),
                "total_score": rr.get("total_score"),
                "likelihood_score": rr.get("likelihood_score"),
                "pe_ratio": rr.get("pe_ratio"),
                "market_cap": rr.get("market_cap"),
                "analyst_positivity": rr.get("analyst_positivity"),
                "earnings_date": str(rr.get("earnings_date")) if rr.get("earnings_date") else None,
                "target_mean": rr.get("target_mean"),
                "target_upside_pct": rr.get("target_upside_pct"),
                "keys_count": len(rr.keys()),
                "keys_sample": sorted(list(rr.keys()))[:40],
            }
        )
    diag["samples"] = samples

    snap = SignalSnapshot(ts_utc=now, rows=rows)
    return snap, diag