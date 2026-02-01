# debug_scanner_columns.py
# Run:
#   python .\debug_scanner_columns.py MU AAPL NVDA
#
# Purpose:
# - Pull the SAME enrichment fields your scanner page uses (fundamentals + analyst_ratings + price_targets + earnings + prices)
# - Show which columns exist + sample rows
# - This confirms DB can provide the “missing parameters” before touching AI PM UI.

from __future__ import annotations
import sys
import pandas as pd

def _sql(engine, q: str, params=None) -> pd.DataFrame:
    return pd.read_sql(q, engine, params=params or {})

def main(argv: list[str]) -> int:
    # uses your project engine helper (same as signals_tab)
    from src.db.connection import get_engine
    eng = get_engine()

    tickers = [a.strip().upper() for a in argv if a.strip()] or ["AAPL", "NVDA", "MU"]
    print("Tickers:", tickers)

    q = """
    WITH t AS (
      SELECT UNNEST(%(tickers)s::text[]) AS ticker
    ),
    latest_prices AS (
      SELECT DISTINCT ON (p.ticker) p.ticker, p.close AS price
      FROM prices p
      WHERE p.ticker = ANY(%(tickers)s)
      ORDER BY p.ticker, p.date DESC
    ),
    latest_fundamentals AS (
      SELECT DISTINCT ON (f.ticker)
        f.ticker,
        f.date AS fundamentals_date,
        f.sector,
        f.ex_dividend_date,
        f.dividend_yield,
        f.pe_ratio,
        f.forward_pe,
        f.peg_ratio,
        f.roe,
        f.market_cap,
        f.pb_ratio,
        f.revenue_growth,
        f.eps
      FROM fundamentals f
      WHERE f.ticker = ANY(%(tickers)s)
      ORDER BY f.ticker, f.date DESC
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
      lp.price,
      lt.target_mean,
      CASE WHEN lp.price > 0 AND lt.target_mean > 0
           THEN ROUND(((lt.target_mean - lp.price) / lp.price * 100)::numeric, 2)
           ELSE NULL END AS target_upside_pct,
      ep.earnings_date,
      lf.sector,
      lf.ex_dividend_date,
      lf.dividend_yield,
      lf.pe_ratio,
      lf.forward_pe,
      lf.peg_ratio,
      lf.roe,
      lf.market_cap,
      lf.pb_ratio,
      lf.revenue_growth,
      lf.eps,
      lr.analyst_positivity,
      lr.buy_count,
      lr.total_ratings,
      lf.fundamentals_date
    FROM t
    LEFT JOIN latest_prices lp ON lp.ticker = t.ticker
    LEFT JOIN latest_targets lt ON lt.ticker = t.ticker
    LEFT JOIN earnings_pick ep ON ep.ticker = t.ticker
    LEFT JOIN latest_fundamentals lf ON lf.ticker = t.ticker
    LEFT JOIN latest_ratings lr ON lr.ticker = t.ticker
    ORDER BY t.ticker
    """

    df = _sql(eng, q, {"tickers": tickers})

    print("\nColumns:", len(df.columns))
    print(df.columns.tolist())
    print("\nSample:\n", df.head(20).to_string(index=False))

    # quick coverage stats
    cov = {}
    for c in ["sector", "price", "target_mean", "earnings_date", "dividend_yield", "pe_ratio", "market_cap", "analyst_positivity"]:
        if c in df.columns:
            cov[c] = int(pd.to_numeric(df[c], errors="ignore").isna().sum()) if c != "sector" else int(df[c].isna().sum())
    print("\nNull counts:", cov)

    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
