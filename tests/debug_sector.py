# debug_sector.py
# Run (dry):
#   python debug_sector.py AAPL ABBV JNJ NVDA TSM
# Run (APPLY backfill of LATEST rows only):
#   python debug_sector.py --apply AAPL ABBV JNJ NVDA TSM
# Run (APPLY for ALL tickers):
#   python debug_sector.py --apply
#
# Uses your project DB engine (src.db.connection.get_engine) so it won't fail with "no password supplied".

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import pandas as pd


def _sql(engine, q: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    return pd.read_sql(q, engine, params=params or {})


def _p(title: str) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def _norm_tickers(args: List[str]) -> List[str]:
    out = []
    for a in args:
        s = str(a).strip().upper()
        if s and not s.startswith("--"):
            out.append(s)
    return list(dict.fromkeys(out))


def _sector_state_df(engine, tickers: Optional[List[str]]) -> pd.DataFrame:
    params: Dict[str, Any] = {}
    filt = ""
    if tickers:
        filt = "WHERE ticker = ANY(%(tickers)s)"
        params["tickers"] = tickers

    q = f"""
    WITH f_latest AS (
        SELECT DISTINCT ON (ticker)
            ticker,
            date AS latest_f_date,
            sector AS latest_sector
        FROM fundamentals
        {filt}
        ORDER BY ticker, date DESC
    ),
    f_last_nonnull AS (
        SELECT DISTINCT ON (ticker)
            ticker,
            date AS last_nonnull_date,
            sector AS last_nonnull_sector
        FROM fundamentals
        {filt}
        {"AND" if filt else "WHERE"} sector IS NOT NULL
        ORDER BY ticker, date DESC
    )
    SELECT
        l.ticker,
        l.latest_f_date,
        l.latest_sector,
        n.last_nonnull_date,
        n.last_nonnull_sector,
        CASE
          WHEN l.latest_sector IS NULL AND n.last_nonnull_sector IS NOT NULL THEN 'NULL-OVERWRITE'
          WHEN l.latest_sector IS NULL AND n.last_nonnull_sector IS NULL THEN 'MISSING'
          ELSE 'OK'
        END AS sector_state
    FROM f_latest l
    LEFT JOIN f_last_nonnull n ON n.ticker = l.ticker
    ORDER BY sector_state DESC, l.ticker
    """
    return _sql(engine, q, params)


def _apply_backfill_latest_rows(engine, tickers: Optional[List[str]]) -> int:
    """
    Fix ONLY the LATEST fundamentals row per ticker when that latest row has sector NULL,
    using the last known non-null sector for that ticker.
    Returns number of updated rows.
    """
    params: Dict[str, Any] = {}
    tick_filter = ""
    if tickers:
        tick_filter = "AND f.ticker = ANY(%(tickers)s)"
        params["tickers"] = tickers

    q = f"""
    WITH latest AS (
        SELECT DISTINCT ON (f.ticker)
            f.ticker,
            f.date AS latest_date,
            f.sector AS latest_sector
        FROM fundamentals f
        WHERE 1=1
          {tick_filter}
        ORDER BY f.ticker, f.date DESC
    ),
    nonnull AS (
        SELECT DISTINCT ON (f.ticker)
            f.ticker,
            f.date AS nn_date,
            f.sector AS nn_sector
        FROM fundamentals f
        WHERE f.sector IS NOT NULL
          {tick_filter}
        ORDER BY f.ticker, f.date DESC
    ),
    fix AS (
        SELECT l.ticker, l.latest_date, n.nn_sector
        FROM latest l
        JOIN nonnull n USING (ticker)
        WHERE l.latest_sector IS NULL
    )
    UPDATE fundamentals f
    SET sector = fix.nn_sector
    FROM fix
    WHERE f.ticker = fix.ticker
      AND f.date = fix.latest_date
    """
    with engine.begin() as conn:
        res = conn.exec_driver_sql(q, params)
        # SQLAlchemy rowcount is reliable for UPDATE here
        return int(res.rowcount or 0)


def _aapl_latest(engine) -> pd.DataFrame:
    return _sql(
        engine,
        """
        SELECT ticker, date, sector
        FROM fundamentals
        WHERE ticker='AAPL'
        ORDER BY date DESC
        LIMIT 10
        """,
    )


def _universe_missing_latest_sector(engine) -> pd.DataFrame:
    return _sql(
        engine,
        """
        WITH latest_scores AS (
          SELECT DISTINCT ON (s.ticker) s.ticker
          FROM screener_scores s
          ORDER BY s.ticker, s.date DESC
        ),
        f_latest AS (
          SELECT DISTINCT ON (f.ticker) f.ticker, f.sector
          FROM fundamentals f
          ORDER BY f.ticker, f.date DESC
        )
        SELECT
          COUNT(*) AS latest_scores_tickers,
          SUM(CASE WHEN fl.sector IS NULL THEN 1 ELSE 0 END) AS missing_latest_sector
        FROM latest_scores ls
        LEFT JOIN f_latest fl ON fl.ticker = ls.ticker
        """,
    )


def main(argv: List[str]) -> int:
    apply = "--apply" in argv
    tickers = _norm_tickers(argv)

    from src.db.connection import get_engine
    engine = get_engine()

    # quick connection check
    _sql(engine, "SELECT 1 AS ok")
    print("Connected OK:", str(engine.url).split("@")[-1])

    if apply:
        _p("APPLY: backfilling latest-row NULL sectors")
        updated = _apply_backfill_latest_rows(engine, tickers if tickers else None)
        print("rows_updated:", updated)

    _p(f"Tickers under test: {tickers if tickers else '(ALL)'}")
    df_state = _sector_state_df(engine, tickers if tickers else None)
    with pd.option_context("display.max_columns", 200, "display.width", 220):
        print(df_state.head(200).to_string(index=False))

    _p("Counts")
    print("tickers:", len(df_state))
    print("NULL-OVERWRITE:", int((df_state["sector_state"] == "NULL-OVERWRITE").sum()))
    print("MISSING:", int((df_state["sector_state"] == "MISSING").sum()))
    print("OK:", int((df_state["sector_state"] == "OK").sum()))

    _p("AAPL fundamentals (latest 10)")
    with pd.option_context("display.max_columns", 50, "display.width", 220):
        print(_aapl_latest(engine).to_string(index=False))

    _p("Universe counts (latest_scores tickers vs missing latest sector)")
    with pd.option_context("display.max_columns", 50, "display.width", 220):
        print(_universe_missing_latest_sector(engine).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
