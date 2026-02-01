#!/usr/bin/env python3
r"""
debug_ai_pm_columns_v1.py

Usage:
  python .\debug_ai_pm_columns_v1.py
  python .\debug_ai_pm_columns_v1.py --csv .\dashboard\exports\2026-01-18T02-37_export.csv
  python .\debug_ai_pm_columns_v1.py --limit 25
  python .\debug_ai_pm_columns_v1.py --dump-sql
"""

from __future__ import annotations

import argparse
import csv as csvlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

try:
    from dotenv import load_dotenv, find_dotenv
except Exception:
    load_dotenv = None
    find_dotenv = None


SEP = "=" * 120


def _hdr(title: str) -> None:
    print("\n" + SEP)
    print(title)
    print(SEP)


def _load_env() -> Optional[str]:
    if load_dotenv is None:
        return None

    cwd = Path.cwd()
    candidate = cwd / ".env"
    loaded_path = None

    if candidate.exists():
        load_dotenv(candidate, override=False)
        loaded_path = str(candidate)
    else:
        if find_dotenv is not None:
            p = find_dotenv(usecwd=True)
            if p:
                load_dotenv(p, override=False)
                loaded_path = p
    return loaded_path


def _make_engine() -> Engine:
    for k in ("DATABASE_URL", "POSTGRES_URL", "SQLALCHEMY_DATABASE_URL"):
        v = os.getenv(k)
        if v:
            return create_engine(v, pool_pre_ping=True)

    pg_host = os.getenv("POSTGRES_HOST")
    pg_port = os.getenv("POSTGRES_PORT") or "5432"
    pg_db = os.getenv("POSTGRES_DB")
    pg_user = os.getenv("POSTGRES_USER")
    pg_pass = os.getenv("POSTGRES_PASSWORD") or ""

    if pg_host and pg_db and pg_user:
        url = f"postgresql+psycopg2://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
        return create_engine(url, pool_pre_ping=True)

    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT") or "5432"
    db = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    pw = os.getenv("PGPASSWORD") or ""

    if host and db and user:
        url = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"
        return create_engine(url, pool_pre_ping=True)

    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT") or "5432"
    db = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    pw = os.getenv("DB_PASSWORD") or ""

    if host and db and user:
        url = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"
        return create_engine(url, pool_pre_ping=True)

    loaded_path = _load_env()
    detected = [k for k in os.environ.keys() if k.upper() in {
        "DATABASE_URL", "POSTGRES_URL", "SQLALCHEMY_DATABASE_URL",
        "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD",
        "PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD",
        "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"
    }]

    raise RuntimeError(
        "DB connection is missing credentials.\n"
        + (f".env loaded from: {loaded_path}\n" if loaded_path else "")
        + f"Detected relevant env keys: {detected}\n"
        + "Provide one of:\n"
        + "  - DATABASE_URL / POSTGRES_URL / SQLALCHEMY_DATABASE_URL\n"
        + "  - POSTGRES_HOST + POSTGRES_DB + POSTGRES_USER (+ optional POSTGRES_PORT/POSTGRES_PASSWORD)\n"
        + "  - PGUSER + PGDATABASE (+ optional PGHOST/PGPORT/PGPASSWORD)\n"
        + "  - DB_USER + DB_NAME (+ optional DB_HOST/DB_PORT/DB_PASSWORD)\n"
    )


def _sql(engine: Engine, q: str, params: Optional[Dict] = None) -> pd.DataFrame:
    return pd.read_sql(text(q), engine, params=params or {})


def _connect_ok(engine: Engine) -> None:
    df = _sql(engine, "SELECT 1 AS ok")
    if df.empty or int(df.iloc[0]["ok"]) != 1:
        raise RuntimeError("DB connectivity check failed")


def _sniff_delimiter(path: Path) -> str:
    # Try sniffing from a small sample; fall back to comma.
    try:
        sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
        dialect = csvlib.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return ","


def _read_csv_tickers(csv_path: str) -> Tuple[List[str], str]:
    p = Path(csv_path)

    if not p.exists():
        return [], f"CSV not found: {csv_path}"

    delim = _sniff_delimiter(p)

    # Try a couple of encodings + detected delimiter
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            df = pd.read_csv(p, encoding=enc, sep=delim)
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        return [], f"CSV unreadable: {csv_path} | {last_err}"

    # Normalize column names for robust matching
    norm = {c: str(c).strip() for c in df.columns}
    df.rename(columns=norm, inplace=True)

    # Common column candidates
    candidates = ["Symbol", "symbol", "Ticker", "ticker", "TICKER", "SYMBOL"]
    col = next((c for c in candidates if c in df.columns), None)

    # If delimiter was wrong, we often get 1 giant column; try alternate separators
    if col is None and len(df.columns) == 1:
        # retry with common alternates
        for alt in [",", ";", "\t", "|"]:
            if alt == delim:
                continue
            try:
                df2 = pd.read_csv(p, encoding="utf-8-sig", sep=alt)
                norm2 = {c: str(c).strip() for c in df2.columns}
                df2.rename(columns=norm2, inplace=True)
                col2 = next((c for c in candidates if c in df2.columns), None)
                if col2 is not None:
                    df = df2
                    col = col2
                    delim = alt
                    break
            except Exception:
                pass

    if col is None:
        return [], (
            f"CSV loaded but no Symbol/ticker column found. "
            f"Detected delimiter='{delim}'. Columns={df.columns.tolist()[:60]}"
        )

    tickers = (
        df[col]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"": None, "NAN": None})
        .dropna()
        .unique()
        .tolist()
    )
    tickers = [t for t in tickers if t and t != "NONE"]
    return tickers, f"source: CSV [{col}] delimiter='{delim}' rows={len(df)}"


def _table_exists(engine: Engine, table: str, schema: str = "public") -> bool:
    q = """
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = :schema
      AND table_name = :table
    LIMIT 1
    """
    df = _sql(engine, q, {"schema": schema, "table": table})
    return not df.empty


def _table_columns(engine: Engine, table: str, schema: str = "public") -> List[str]:
    q = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = :schema
      AND table_name = :table
    ORDER BY ordinal_position
    """
    df = _sql(engine, q, {"schema": schema, "table": table})
    return df["column_name"].astype(str).tolist() if not df.empty else []


def _pick_existing(cols: List[str], wanted: List[str]) -> List[str]:
    s = set(cols)
    return [c for c in wanted if c in s]


def _pull_enrichment(engine: Engine, tickers: List[str], dump_sql: bool = False) -> pd.DataFrame:
    prices_cols = _table_columns(engine, "prices")
    fundamentals_cols = _table_columns(engine, "fundamentals")
    scores_cols = _table_columns(engine, "screener_scores")
    signals_cols = _table_columns(engine, "trading_signals")
    ratings_cols = _table_columns(engine, "analyst_ratings")
    targets_cols = _table_columns(engine, "price_targets")

    price_value_col = "close" if "close" in prices_cols else ("price" if "price" in prices_cols else None)
    if price_value_col is None:
        raise RuntimeError("prices table has no close/price column")

    fundamentals_wanted = [
        "date", "sector", "ex_dividend_date", "dividend_yield", "market_cap",
        "pe_ratio", "forward_pe", "peg_ratio", "roe", "pb_ratio", "revenue_growth",
        "eps", "current_ratio", "debt_to_equity", "dividend_payout_ratio",
        "earnings_growth", "eps_growth", "gross_margin", "operating_margin",
        "profit_margin", "ps_ratio",
    ]
    fundamentals_pick = _pick_existing(fundamentals_cols, fundamentals_wanted)

    scores_wanted = ["date", "total_score", "likelihood_score"]
    scores_pick = _pick_existing(scores_cols, scores_wanted)

    signals_wanted = ["created_at", "date", "likelihood_score", "sentiment_score", "analyst_positivity", "fundamental_score", "gap_score"]
    signals_pick = _pick_existing(signals_cols, signals_wanted)

    ratings_wanted = ["date", "analyst_positivity", "analyst_buy", "analyst_total"]
    ratings_pick = _pick_existing(ratings_cols, ratings_wanted)

    targets_wanted = ["date", "target_mean"]
    targets_pick = _pick_existing(targets_cols, targets_wanted)

    lf_parts = []
    for c in fundamentals_pick:
        lf_parts.append("f.date AS fundamentals_date" if c == "date" else f"f.{c}")
    lf_select = ",\n        ".join(lf_parts) if lf_parts else "NULL::date AS fundamentals_date"

    sc_parts = []
    for c in scores_pick:
        sc_parts.append("s.date AS scores_date" if c == "date" else f"s.{c}")
    sc_select = ",\n        ".join(sc_parts) if sc_parts else "NULL::date AS scores_date"

    ls_parts = []
    for c in signals_pick:
        if c == "created_at":
            ls_parts.append("x.created_at AS signals_created_at")
        elif c == "date":
            ls_parts.append("x.date AS signals_date")
        elif c == "likelihood_score":
            ls_parts.append("x.likelihood_score AS ts_likelihood_score")
        elif c == "analyst_positivity":
            ls_parts.append("x.analyst_positivity AS ts_analyst_positivity")
        else:
            ls_parts.append(f"x.{c}")
    ls_select = ",\n        ".join(ls_parts) if ls_parts else "NULL::timestamptz AS signals_created_at"

    lr_parts = []
    for c in ratings_pick:
        if c == "date":
            lr_parts.append("ar.date AS ratings_date")
        elif c == "analyst_buy":
            lr_parts.append("ar.analyst_buy AS buy_count")
        elif c == "analyst_total":
            lr_parts.append("ar.analyst_total AS total_ratings")
        else:
            lr_parts.append(f"ar.{c}")
    lr_select = ",\n        ".join(lr_parts) if lr_parts else "NULL::date AS ratings_date"

    lt_parts = []
    for c in targets_pick:
        lt_parts.append("pt.date AS targets_date" if c == "date" else f"pt.{c}")
    lt_select = ",\n        ".join(lt_parts) if lt_parts else "NULL::date AS targets_date"

    q = f"""
    WITH t AS (
      SELECT DISTINCT UNNEST(:tickers) AS ticker
    ),
    latest_prices AS (
      SELECT DISTINCT ON (p.ticker)
        p.ticker,
        p.{price_value_col} AS price,
        p.date AS price_date
      FROM prices p
      WHERE p.ticker = ANY(:tickers)
      ORDER BY p.ticker, p.date DESC NULLS LAST
    ),
    latest_fundamentals AS (
      SELECT DISTINCT ON (f.ticker)
        f.ticker,
        {lf_select}
      FROM fundamentals f
      WHERE f.ticker = ANY(:tickers)
      ORDER BY f.ticker, f.date DESC NULLS LAST
    ),
    latest_scores AS (
      SELECT DISTINCT ON (s.ticker)
        s.ticker,
        {sc_select}
      FROM screener_scores s
      WHERE s.ticker = ANY(:tickers)
      ORDER BY s.ticker, s.date DESC NULLS LAST
    ),
    latest_signals AS (
      SELECT DISTINCT ON (x.ticker)
        x.ticker,
        {ls_select}
      FROM trading_signals x
      WHERE x.ticker = ANY(:tickers)
      ORDER BY x.ticker, x.created_at DESC NULLS LAST
    ),
    latest_ratings AS (
      SELECT DISTINCT ON (ar.ticker)
        ar.ticker,
        {lr_select}
      FROM analyst_ratings ar
      WHERE ar.ticker = ANY(:tickers)
      ORDER BY ar.ticker, ar.date DESC NULLS LAST
    ),
    latest_targets AS (
      SELECT DISTINCT ON (pt.ticker)
        pt.ticker,
        {lt_select}
      FROM price_targets pt
      WHERE pt.ticker = ANY(:tickers)
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
      p.price,
      p.price_date,
      sc.scores_date,
      sc.total_score,
      sc.likelihood_score,
      ls.signals_created_at,
      ls.signals_date,
      ls.ts_likelihood_score,
      ls.sentiment_score,
      ls.ts_analyst_positivity,
      ls.fundamental_score,
      ls.gap_score,
      lr.ratings_date,
      lr.analyst_positivity,
      lr.buy_count,
      lr.total_ratings,
      lt.targets_date,
      lt.target_mean,
      CASE WHEN p.price > 0 AND lt.target_mean > 0
           THEN ROUND(((lt.target_mean - p.price) / p.price * 100)::numeric, 2)
           ELSE NULL END AS target_upside_pct,
      ep.earnings_date,
      lf.fundamentals_date,
      lf.sector,
      lf.ex_dividend_date,
      lf.dividend_yield,
      lf.market_cap,
      lf.pe_ratio,
      lf.forward_pe,
      lf.peg_ratio,
      lf.roe,
      lf.pb_ratio,
      lf.revenue_growth,
      lf.eps,
      lf.current_ratio,
      lf.debt_to_equity,
      lf.dividend_payout_ratio,
      lf.earnings_growth,
      lf.eps_growth,
      lf.gross_margin,
      lf.operating_margin,
      lf.profit_margin,
      lf.ps_ratio
    FROM t
    LEFT JOIN latest_prices p ON p.ticker = t.ticker
    LEFT JOIN latest_scores sc ON sc.ticker = t.ticker
    LEFT JOIN latest_signals ls ON ls.ticker = t.ticker
    LEFT JOIN latest_ratings lr ON lr.ticker = t.ticker
    LEFT JOIN latest_targets lt ON lt.ticker = t.ticker
    LEFT JOIN earnings_pick ep ON ep.ticker = t.ticker
    LEFT JOIN latest_fundamentals lf ON lf.ticker = t.ticker
    ORDER BY t.ticker
    """

    if dump_sql:
        _hdr("SQL (debug)")
        print(q.strip())

    return _sql(engine, q, {"tickers": tickers})


def _coverage(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rows = []
    for c in df.columns:
        nulls = int(df[c].isna().sum())
        rows.append({"column": c, "nulls": nulls, "nonnull": int(n - nulls)})
    return pd.DataFrame(rows).sort_values(["nulls", "column"], ascending=[True, True])


def _fallback_tickers_from_scores(engine: Engine, limit: int) -> Tuple[List[str], str]:
    q = """
    SELECT DISTINCT s.ticker
    FROM screener_scores s
    ORDER BY s.ticker
    LIMIT :limit
    """
    df = _sql(engine, q, {"limit": int(limit)})
    tickers = df["ticker"].astype(str).tolist() if not df.empty else []
    return tickers, "fallback: screener_scores sample"


def main(argv: Optional[List[str]] = None) -> int:
    _load_env()

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Path to CSV export with tickers/Symbol column")
    ap.add_argument("--limit", type=int, default=10, help="Fallback ticker limit from screener_scores")
    ap.add_argument("--dump-sql", action="store_true", help="Print the final SQL used for enrichment")
    args = ap.parse_args(argv)

    eng = _make_engine()
    _connect_ok(eng)

    host = os.getenv("POSTGRES_HOST") or os.getenv("PGHOST") or os.getenv("DB_HOST") or "localhost"
    port = os.getenv("POSTGRES_PORT") or os.getenv("PGPORT") or os.getenv("DB_PORT") or "5432"
    db = os.getenv("POSTGRES_DB") or os.getenv("PGDATABASE") or os.getenv("DB_NAME") or "alpha_platform"
    print(f"Connected OK: {host}:{port}/{db}")

    tickers: List[str] = []
    source_msg = ""

    if args.csv:
        tickers, source_msg = _read_csv_tickers(args.csv)

    if not tickers:
        fb_tickers, fb_msg = _fallback_tickers_from_scores(eng, args.limit)
        tickers = fb_tickers
        if args.csv:
            source_msg = f"{source_msg} | {fb_msg}"
        else:
            source_msg = fb_msg

    _hdr("Tickers source")
    print(source_msg)
    print(f"tickers_count: {len(tickers)}")
    print(f"tickers_sample: {tickers[:10]}")

    _hdr("DB table presence")
    for tname in ["fundamentals", "prices", "screener_scores", "trading_signals", "analyst_ratings", "price_targets", "earnings_calendar"]:
        print(f"{tname}: {_table_exists(eng, tname)}")

    df_enrich = _pull_enrichment(eng, tickers, dump_sql=args.dump_sql)

    _hdr("DB enrichment columns")
    print(f"columns_count: {len(df_enrich.columns)}")
    print(list(df_enrich.columns))

    _hdr("DB enrichment sample (first 10)")
    if df_enrich.empty:
        print("(empty result)")
    else:
        with pd.option_context("display.max_columns", 200, "display.width", 220):
            print(df_enrich.head(10).to_string(index=False))

    _hdr("DB enrichment coverage (null/non-null per column)")
    cov = _coverage(df_enrich)
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(cov.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
