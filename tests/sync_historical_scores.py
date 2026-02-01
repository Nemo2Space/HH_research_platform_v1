"""
Sync screener_scores to historical_scores for backtesting.

The backtest engine reads from historical_scores, but the daily screener
saves to screener_scores. This script syncs them.

Usage:
    python sync_historical_scores.py           # Sync missing data
    python sync_historical_scores.py --all     # Rebuild all historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import sys
import os

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.connection import get_connection, get_engine

def get_historical_schema():
    """Get the historical_scores table schema."""
    return """
    CREATE TABLE IF NOT EXISTS historical_scores (
        id SERIAL PRIMARY KEY,
        score_date DATE NOT NULL,
        ticker VARCHAR(20) NOT NULL,
        sector VARCHAR(100),
        sentiment NUMERIC(8,2),
        fundamental_score NUMERIC(8,2),
        growth_score NUMERIC(8,2),
        dividend_score NUMERIC(8,2),
        total_score NUMERIC(8,2),
        gap_score NUMERIC(8,2),
        mkt_score NUMERIC(8,2),
        signal_type VARCHAR(20),
        signal_correct BOOLEAN,
        op_price NUMERIC(15,4),
        return_1d NUMERIC(10,4),
        return_5d NUMERIC(10,4),
        return_10d NUMERIC(10,4),
        return_20d NUMERIC(10,4),
        price_1d NUMERIC(15,4),
        price_5d NUMERIC(15,4),
        price_10d NUMERIC(15,4),
        price_20d NUMERIC(15,4),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(score_date, ticker)
    )
    """

def ensure_table_exists():
    """Create historical_scores table if it doesn't exist."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(get_historical_schema())
        conn.commit()
    print("✅ historical_scores table ready")

def get_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Get price data for calculating returns."""
    engine = get_engine()

    query = """
        SELECT ticker, date, close, adj_close
        FROM prices
        WHERE ticker = ANY(%(tickers)s)
          AND date >= %(start_date)s
          AND date <= %(end_date)s
        ORDER BY ticker, date
    """

    df = pd.read_sql(query, engine, params={
        'tickers': tickers,
        'start_date': start_date,
        'end_date': end_date
    })

    return df

def calculate_forward_returns(scores_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate forward returns for each score date."""

    if prices_df.empty:
        print("⚠️ No price data available")
        scores_df['return_1d'] = None
        scores_df['return_5d'] = None
        scores_df['return_10d'] = None
        scores_df['return_20d'] = None
        scores_df['price_1d'] = None
        scores_df['price_5d'] = None
        scores_df['price_10d'] = None
        scores_df['price_20d'] = None
        return scores_df

    # Use adj_close if available, else close
    prices_df['price'] = prices_df['adj_close'].fillna(prices_df['close'])

    # Create price lookup by ticker and date
    price_dict = {}
    for _, row in prices_df.iterrows():
        key = (row['ticker'], row['date'])
        price_dict[key] = row['price']

    # Get sorted dates for each ticker
    ticker_dates = {}
    for ticker in prices_df['ticker'].unique():
        ticker_df = prices_df[prices_df['ticker'] == ticker].sort_values('date')
        ticker_dates[ticker] = ticker_df['date'].tolist()

    results = []
    for _, row in scores_df.iterrows():
        ticker = row['ticker']
        score_date = pd.to_datetime(row['date']).date()

        # Get entry price (price on score date or next available)
        entry_price = None
        dates = ticker_dates.get(ticker, [])

        for d in dates:
            if d >= score_date:
                entry_price = price_dict.get((ticker, d))
                break

        if entry_price is None:
            # Use the score date price if available
            entry_price = price_dict.get((ticker, score_date))

        # Calculate forward returns
        return_1d = None
        return_5d = None
        return_10d = None
        return_20d = None
        price_1d = None
        price_5d = None
        price_10d = None
        price_20d = None

        if entry_price and dates:
            # Find prices at future dates
            for d in dates:
                days_diff = (d - score_date).days
                future_price = price_dict.get((ticker, d))

                if future_price and entry_price > 0:
                    ret = (future_price - entry_price) / entry_price * 100

                    if days_diff == 1 or (days_diff > 0 and return_1d is None and days_diff <= 3):
                        return_1d = ret
                        price_1d = future_price
                    if days_diff == 5 or (days_diff >= 4 and days_diff <= 6 and return_5d is None):
                        return_5d = ret
                        price_5d = future_price
                    if days_diff == 10 or (days_diff >= 8 and days_diff <= 12 and return_10d is None):
                        return_10d = ret
                        price_10d = future_price
                    if days_diff == 20 or (days_diff >= 18 and days_diff <= 22 and return_20d is None):
                        return_20d = ret
                        price_20d = future_price

        result = row.to_dict()
        result['op_price'] = entry_price
        result['return_1d'] = return_1d
        result['return_5d'] = return_5d
        result['return_10d'] = return_10d
        result['return_20d'] = return_20d
        result['price_1d'] = price_1d
        result['price_5d'] = price_5d
        result['price_10d'] = price_10d
        result['price_20d'] = price_20d
        results.append(result)

    return pd.DataFrame(results)

def sync_historical_scores(rebuild_all: bool = False):
    """Sync screener_scores to historical_scores."""

    engine = get_engine()

    # Ensure table exists
    ensure_table_exists()

    # Get date ranges
    if rebuild_all:
        # Get all dates from screener_scores
        start_date = None
        print("🔄 Rebuilding ALL historical data...")
    else:
        # Get max date in historical_scores
        max_date_df = pd.read_sql(
            "SELECT MAX(score_date) as max_date FROM historical_scores",
            engine
        )
        max_date = max_date_df['max_date'].iloc[0]

        if max_date:
            start_date = max_date + timedelta(days=1)
            print(f"📅 Syncing from {start_date} onwards...")
        else:
            start_date = None
            print("📅 No existing data, syncing all...")

    # Load screener_scores that need to be synced
    query = """
        SELECT 
            date,
            ticker,
            sentiment_score as sentiment,
            fundamental_score,
            growth_score,
            dividend_score,
            total_score,
            gap_score,
            composite_score as mkt_score,
            signal_type
        FROM screener_scores
    """

    if start_date:
        query += f" WHERE date >= '{start_date}'"

    query += " ORDER BY date, ticker"

    scores_df = pd.read_sql(query, engine)

    if scores_df.empty:
        print("✅ No new data to sync")
        return

    print(f"📊 Found {len(scores_df)} records to sync")
    print(f"   Date range: {scores_df['date'].min()} to {scores_df['date'].max()}")
    print(f"   Tickers: {scores_df['ticker'].nunique()}")

    # Get sector data from fundamentals
    sectors_df = pd.read_sql("""
        SELECT DISTINCT ON (ticker) ticker, sector
        FROM fundamentals
        ORDER BY ticker, date DESC
    """, engine)
    sectors_map = dict(zip(sectors_df['ticker'], sectors_df['sector']))
    scores_df['sector'] = scores_df['ticker'].map(sectors_map)

    # Get price data for return calculations
    tickers = scores_df['ticker'].unique().tolist()
    min_date = scores_df['date'].min()
    max_date = scores_df['date'].max() + timedelta(days=25)  # Extra days for forward returns

    print(f"📈 Loading price data...")
    prices_df = get_price_data(tickers, str(min_date), str(max_date))
    print(f"   Loaded {len(prices_df)} price records")

    # Calculate forward returns
    print(f"🧮 Calculating forward returns...")
    enriched_df = calculate_forward_returns(scores_df, prices_df)

    # Determine signal correctness (if we have return data)
    enriched_df['signal_correct'] = enriched_df.apply(
        lambda r: (r['return_5d'] > 0 if r['signal_type'] in ['BUY', 'STRONG_BUY', 'WEAK_BUY']
                   else r['return_5d'] < 0 if r['signal_type'] in ['SELL', 'STRONG_SELL', 'WEAK_SELL']
                   else None) if pd.notna(r.get('return_5d')) else None,
        axis=1
    )

    # Rename date column
    enriched_df = enriched_df.rename(columns={'date': 'score_date'})

    # Insert into historical_scores
    print(f"💾 Inserting into historical_scores...")

    inserted = 0
    skipped = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            for _, row in enriched_df.iterrows():
                try:
                    cur.execute("""
                        INSERT INTO historical_scores (
                            score_date, ticker, sector, sentiment, fundamental_score,
                            growth_score, dividend_score, total_score, gap_score, mkt_score,
                            signal_type, signal_correct, op_price,
                            return_1d, return_5d, return_10d, return_20d,
                            price_1d, price_5d, price_10d, price_20d
                        ) VALUES (
                            %(score_date)s, %(ticker)s, %(sector)s, %(sentiment)s, %(fundamental_score)s,
                            %(growth_score)s, %(dividend_score)s, %(total_score)s, %(gap_score)s, %(mkt_score)s,
                            %(signal_type)s, %(signal_correct)s, %(op_price)s,
                            %(return_1d)s, %(return_5d)s, %(return_10d)s, %(return_20d)s,
                            %(price_1d)s, %(price_5d)s, %(price_10d)s, %(price_20d)s
                        )
                        ON CONFLICT (score_date, ticker) DO UPDATE SET
                            sector = COALESCE(EXCLUDED.sector, fundamentals.sector),
                            sentiment = EXCLUDED.sentiment,
                            fundamental_score = EXCLUDED.fundamental_score,
                            growth_score = EXCLUDED.growth_score,
                            dividend_score = EXCLUDED.dividend_score,
                            total_score = EXCLUDED.total_score,
                            gap_score = EXCLUDED.gap_score,
                            mkt_score = EXCLUDED.mkt_score,
                            signal_type = EXCLUDED.signal_type,
                            signal_correct = EXCLUDED.signal_correct,
                            op_price = EXCLUDED.op_price,
                            return_1d = EXCLUDED.return_1d,
                            return_5d = EXCLUDED.return_5d,
                            return_10d = EXCLUDED.return_10d,
                            return_20d = EXCLUDED.return_20d,
                            price_1d = EXCLUDED.price_1d,
                            price_5d = EXCLUDED.price_5d,
                            price_10d = EXCLUDED.price_10d,
                            price_20d = EXCLUDED.price_20d
                    """, {
                        'score_date': row['score_date'],
                        'ticker': row['ticker'],
                        'sector': row.get('sector'),
                        'sentiment': row.get('sentiment'),
                        'fundamental_score': row.get('fundamental_score'),
                        'growth_score': row.get('growth_score'),
                        'dividend_score': row.get('dividend_score'),
                        'total_score': row.get('total_score'),
                        'gap_score': row.get('gap_score'),
                        'mkt_score': row.get('mkt_score'),
                        'signal_type': row.get('signal_type'),
                        'signal_correct': row.get('signal_correct'),
                        'op_price': row.get('op_price'),
                        'return_1d': row.get('return_1d'),
                        'return_5d': row.get('return_5d'),
                        'return_10d': row.get('return_10d'),
                        'return_20d': row.get('return_20d'),
                        'price_1d': row.get('price_1d'),
                        'price_5d': row.get('price_5d'),
                        'price_10d': row.get('price_10d'),
                        'price_20d': row.get('price_20d'),
                    })
                    inserted += 1
                except Exception as e:
                    skipped += 1
                    if skipped <= 5:
                        print(f"   ⚠️ Error inserting {row['ticker']} {row['score_date']}: {e}")

        conn.commit()

    print(f"\n✅ Sync complete!")
    print(f"   Inserted/Updated: {inserted}")
    print(f"   Skipped (errors): {skipped}")

    # Verify
    verify_df = pd.read_sql("""
        SELECT MIN(score_date) as min_date, 
               MAX(score_date) as max_date, 
               COUNT(*) as total_rows
        FROM historical_scores
    """, engine)
    print(f"\n📊 Historical scores now has:")
    print(f"   Date range: {verify_df['min_date'].iloc[0]} to {verify_df['max_date'].iloc[0]}")
    print(f"   Total rows: {verify_df['total_rows'].iloc[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sync screener_scores to historical_scores')
    parser.add_argument('--all', action='store_true', help='Rebuild all historical data')
    args = parser.parse_args()

    sync_historical_scores(rebuild_all=args.all)