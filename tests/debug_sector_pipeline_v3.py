import argparse
import sys

import pandas as pd


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write updates into fundamentals")
    ap.add_argument("--dry-run", action="store_true", help="force dry run")
    args = ap.parse_args(argv)

    # Import your repository so it uses your existing connection config
    try:
        from dashboard.ai_pm.repository import Repository  # if you placed this under dashboard/ai_pm
    except Exception:
        try:
            from repository import Repository  # if you run it from same folder
        except Exception as e:
            print("ERROR: cannot import Repository. Put this script next to repository.py or fix import path.")
            raise

    repo = Repository()
    eng = repo.engine

    def _sql(q: str, params=None) -> pd.DataFrame:
        return pd.read_sql(q, eng, params=params or {})

    print("Connected OK")

    # Find tickers where latest row sector is NULL but older row has a non-null sector
    df = _sql("""
        WITH latest AS (
            SELECT DISTINCT ON (ticker)
                ticker, date AS latest_date, sector AS latest_sector
            FROM fundamentals
            ORDER BY ticker, date DESC NULLS LAST
        ),
        last_nonnull AS (
            SELECT DISTINCT ON (ticker)
                ticker, date AS last_nonnull_date, sector AS last_nonnull_sector
            FROM fundamentals
            WHERE sector IS NOT NULL
            ORDER BY ticker, date DESC NULLS LAST
        )
        SELECT
            l.ticker,
            l.latest_date,
            l.latest_sector,
            n.last_nonnull_date,
            n.last_nonnull_sector
        FROM latest l
        JOIN last_nonnull n USING (ticker)
        WHERE l.latest_sector IS NULL
    """)

    print("\nBEFORE:")
    stats = _sql("SELECT COUNT(*) AS rows_total, SUM(CASE WHEN sector IS NOT NULL THEN 1 ELSE 0 END) AS sector_nonnull, SUM(CASE WHEN sector IS NULL THEN 1 ELSE 0 END) AS sector_null FROM fundamentals")
    print(stats.to_string(index=False))

    print("\nLatest NULL but older non-null (tickers):")
    print(pd.DataFrame({"null_overwrite_tickers": [len(df)]}).to_string(index=False))

    do_apply = args.apply and (not args.dry_run)
    if not do_apply:
        print("\nDRY-RUN ONLY. Re-run with --apply to write.")
        return 0

    # Apply: set latest-row sector = last_nonnull_sector for those tickers
    # Safe because it only fills NULLs on the LATEST row.
    with eng.begin() as conn:
        res = conn.execute(
            """
            WITH last_nonnull AS (
                SELECT DISTINCT ON (ticker)
                    ticker, sector
                FROM fundamentals
                WHERE sector IS NOT NULL
                ORDER BY ticker, date DESC NULLS LAST
            ),
            latest AS (
                SELECT DISTINCT ON (ticker)
                    ticker, date
                FROM fundamentals
                ORDER BY ticker, date DESC NULLS LAST
            )
            UPDATE fundamentals f
            SET sector = n.sector
            FROM latest l
            JOIN last_nonnull n USING (ticker)
            WHERE f.ticker = l.ticker
              AND f.date = l.date
              AND f.sector IS NULL
            """
        )
        # sqlalchemy may not give rowcount reliably for this statement; re-check:
    after = _sql("SELECT COUNT(*) AS rows_total, SUM(CASE WHEN sector IS NOT NULL THEN 1 ELSE 0 END) AS sector_nonnull, SUM(CASE WHEN sector IS NULL THEN 1 ELSE 0 END) AS sector_null FROM fundamentals")
    print("\nAFTER:")
    print(after.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
