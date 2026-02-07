"""
Backfill missing return_1d, return_5d, return_10d, return_20d in historical_scores.

Fetches daily close prices from yfinance, maps to trading days after score_date,
and updates the database in bulk.

Usage: python backfill_returns.py [--dry-run] [--min-date 2026-01-01]
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import sys
import time

# Add project root
sys.path.insert(0, '.')
from src.ml.db_helper import get_engine

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

# Parse args
DRY_RUN = '--dry-run' in sys.argv
MIN_DATE = None
for i, arg in enumerate(sys.argv):
    if arg == '--min-date' and i + 1 < len(sys.argv):
        MIN_DATE = sys.argv[i + 1]

HORIZONS = {
    'return_1d': ('price_1d', 1),
    'return_5d': ('price_5d', 5),
    'return_10d': ('price_10d', 10),
    'return_20d': ('price_20d', 20),
}


def get_rows_needing_backfill(engine, min_date=None):
    """Get all rows where returns are NULL but we have an entry price."""
    query = """
        SELECT id, score_date, ticker, op_price,
               return_1d, return_5d, return_10d, return_20d,
               price_1d, price_5d, price_10d, price_20d
        FROM historical_scores
        WHERE op_price IS NOT NULL AND op_price > 0
          AND (return_1d IS NULL OR return_5d IS NULL)
    """
    if min_date:
        query += f" AND score_date >= '{min_date}'"
    query += " ORDER BY score_date, ticker"

    df = pd.read_sql(query, engine)
    logger.info(f"Found {len(df)} rows needing backfill across {df['ticker'].nunique()} tickers")
    return df


def fetch_prices_for_ticker(ticker, start_date, end_date):
    """Download daily close prices for a ticker."""
    try:
        # Add buffer: 30 trading days after end_date for 20d returns
        buffer_end = end_date + timedelta(days=45)
        data = yf.Ticker(ticker).history(
            start=start_date.strftime('%Y-%m-%d'),
            end=buffer_end.strftime('%Y-%m-%d'),
            auto_adjust=True
        )
        if data is not None and not data.empty:
            # Return a series of close prices indexed by date (date only, no tz)
            data.index = data.index.tz_localize(None).date
            return data['Close']
    except Exception as e:
        logger.warning(f"  {ticker}: yfinance error - {e}")
    return None


def find_trading_day_price(closes, score_date, n_days):
    """Find close price n trading days after score_date."""
    if closes is None:
        return None
    # Get all dates after score_date
    future_dates = sorted([d for d in closes.index if d > score_date])
    if len(future_dates) >= n_days:
        target_date = future_dates[n_days - 1]
        return float(closes[target_date])
    return None


def backfill():
    engine = get_engine()
    rows = get_rows_needing_backfill(engine, MIN_DATE)

    if rows.empty:
        logger.info("Nothing to backfill!")
        return

    # Group by ticker
    tickers = rows.groupby('ticker').agg(
        min_date=('score_date', 'min'),
        max_date=('score_date', 'max'),
        count=('id', 'count')
    ).reset_index()

    logger.info(f"Need to fetch prices for {len(tickers)} tickers")
    logger.info(f"Date range: {tickers['min_date'].min()} to {tickers['max_date'].max()}")

    if DRY_RUN:
        logger.info("DRY RUN - no database updates will be made")

    # Process each ticker
    total_updated = 0
    total_skipped = 0
    failed_tickers = []

    for idx, ticker_row in tickers.iterrows():
        ticker = ticker_row['ticker']
        min_dt = pd.Timestamp(ticker_row['min_date'])
        max_dt = pd.Timestamp(ticker_row['max_date'])
        count = ticker_row['count']

        # Fetch prices
        closes = fetch_prices_for_ticker(
            ticker,
            min_dt - timedelta(days=5),  # small buffer before
            max_dt + timedelta(days=1)
        )

        if closes is None or len(closes) == 0:
            logger.warning(f"  {ticker}: No price data - skipping {count} rows")
            failed_tickers.append(ticker)
            continue

        # Process each row for this ticker
        ticker_rows = rows[rows['ticker'] == ticker]
        updates = []

        for _, row in ticker_rows.iterrows():
            score_date = row['score_date']
            if hasattr(score_date, 'date'):
                score_date = score_date.date()
            op_price = float(row['op_price'])

            update = {'id': row['id']}
            has_update = False

            for ret_col, (price_col, n_days) in HORIZONS.items():
                if pd.notna(row[ret_col]):
                    continue  # Already has this return

                price = find_trading_day_price(closes, score_date, n_days)
                if price is not None and price > 0:
                    ret_val = round((price / op_price - 1) * 100, 4)
                    update[price_col] = round(price, 2)
                    update[ret_col] = ret_val
                    has_update = True

            if has_update:
                updates.append(update)

        if not updates:
            logger.info(f"  [{idx+1}/{len(tickers)}] {ticker}: {count} rows - no future prices available yet")
            total_skipped += count
            continue

        # Bulk update database
        if not DRY_RUN:
            conn = engine.raw_connection()
            cur = conn.cursor()
            try:
                for upd in updates:
                    set_parts = []
                    values = []
                    for key in ['price_1d', 'return_1d', 'price_5d', 'return_5d',
                                'price_10d', 'return_10d', 'price_20d', 'return_20d']:
                        if key in upd:
                            set_parts.append(f"{key} = %s")
                            values.append(upd[key])
                    if set_parts:
                        values.append(upd['id'])
                        sql = f"UPDATE historical_scores SET {', '.join(set_parts)} WHERE id = %s"
                        cur.execute(sql, values)

                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"  {ticker}: DB error - {e}")
                failed_tickers.append(ticker)
                continue
            finally:
                cur.close()
                conn.close()

        total_updated += len(updates)
        filled_horizons = sum(1 for u in updates for k in u if k.startswith('return_'))

        if (idx + 1) % 25 == 0 or idx < 5:
            logger.info(f"  [{idx+1}/{len(tickers)}] {ticker}: {len(updates)}/{count} rows updated "
                       f"({filled_horizons} return values)")

        # Rate limit yfinance
        if (idx + 1) % 50 == 0:
            logger.info(f"  --- Progress: {total_updated} updated, {total_skipped} skipped ---")
            time.sleep(2)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"BACKFILL COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Rows updated: {total_updated}")
    logger.info(f"  Rows skipped (no future price): {total_skipped}")
    logger.info(f"  Failed tickers: {len(failed_tickers)}")
    if failed_tickers:
        logger.info(f"  Failed: {', '.join(failed_tickers[:20])}")

    # Verify
    if not DRY_RUN:
        q = """
        SELECT
            COUNT(*) as total,
            COUNT(return_1d) as has_1d,
            COUNT(return_5d) as has_5d,
            COUNT(return_10d) as has_10d,
            COUNT(return_20d) as has_20d
        FROM historical_scores
        WHERE op_price IS NOT NULL AND op_price > 0
        """
        verify = pd.read_sql(q, engine)
        print(f"\n  DB STATUS:")
        print(f"  Total rows:  {int(verify['total'].iloc[0])}")
        print(f"  Has 1d ret:  {int(verify['has_1d'].iloc[0])}")
        print(f"  Has 5d ret:  {int(verify['has_5d'].iloc[0])}")
        print(f"  Has 10d ret: {int(verify['has_10d'].iloc[0])}")
        print(f"  Has 20d ret: {int(verify['has_20d'].iloc[0])}")


if __name__ == '__main__':
    backfill()