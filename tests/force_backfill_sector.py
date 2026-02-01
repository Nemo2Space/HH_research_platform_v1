# force_backfill_sector.py
# Run:
#   python force_backfill_sector.py
#
# This will:
# 1) show BEFORE counts
# 2) UPDATE fundamentals.sector where NULL using last non-null sector per ticker
# 3) print how many rows were updated (via RETURNING)
# 4) show AFTER counts + AAPL latest rows

from __future__ import annotations
import pandas as pd

def main():
    from src.db.connection import get_engine
    engine = get_engine()
    print("Connected OK:", str(engine.url).split("@")[-1])

    before = pd.read_sql(
        """
        SELECT
          COUNT(*) AS rows_total,
          COUNT(sector) AS sector_nonnull,
          COUNT(*) - COUNT(sector) AS sector_null
        FROM fundamentals
        """,
        engine,
    )
    print("\nBEFORE:\n", before.to_string(index=False))

    with engine.begin() as conn:
        res = conn.exec_driver_sql(
            """
            WITH last_nonnull AS (
              SELECT DISTINCT ON (ticker)
                ticker,
                sector
              FROM fundamentals
              WHERE sector IS NOT NULL
              ORDER BY ticker, date DESC
            )
            UPDATE fundamentals f
            SET sector = ln.sector
            FROM last_nonnull ln
            WHERE f.ticker = ln.ticker
              AND f.sector IS NULL
            RETURNING f.ticker, f.date, f.sector
            """
        )
        rows = res.fetchall()
        print(f"\nUPDATED rows: {len(rows)}")
        if rows:
            print("Sample updated rows:", rows[:10])

    after = pd.read_sql(
        """
        SELECT
          COUNT(*) AS rows_total,
          COUNT(sector) AS sector_nonnull,
          COUNT(*) - COUNT(sector) AS sector_null
        FROM fundamentals
        """,
        engine,
    )
    print("\nAFTER:\n", after.to_string(index=False))

    aapl = pd.read_sql(
        """
        SELECT ticker, date, sector
        FROM fundamentals
        WHERE ticker='AAPL'
        ORDER BY date DESC
        LIMIT 5
        """,
        engine,
    )
    print("\nAAPL latest 5:\n", aapl.to_string(index=False))

    universe = pd.read_sql(
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
        engine,
    )
    print("\nUniverse missing latest sector:\n", universe.to_string(index=False))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
