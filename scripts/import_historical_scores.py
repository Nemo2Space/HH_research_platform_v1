"""
Alpha Platform - Historical Scores Import & Backtesting

This script:
1. Creates the historical_scores table
2. Imports CSV files with past scores
3. Fetches actual price changes after each score date
4. Enables backtesting and ML training

Usage:
    python scripts/import_historical_scores.py --file path/to/scores.csv
    python scripts/import_historical_scores.py --dir path/to/csv/folder
    python scripts/import_historical_scores.py --update-returns  # Fill in actual returns
"""

import os
import sys
import argparse
import glob
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv

load_dotenv()

from src.db.connection import get_connection
from src.utils.logging import get_logger

logger = get_logger(__name__)


def create_table():
    """Create the historical_scores table."""
    sql = """
          CREATE TABLE IF NOT EXISTS historical_scores \
          ( \
              id \
              SERIAL \
              PRIMARY \
              KEY, \
              score_date \
              DATE \
              NOT \
              NULL, \
              ticker \
              VARCHAR \
          ( \
              10 \
          ) NOT NULL,
              sector VARCHAR \
          ( \
              50 \
          ),
              volume BIGINT,
              sentiment INT,
              fundamental_score INT,
              growth_score INT,
              dividend_score INT,
              counter INT,
              total_score NUMERIC \
          ( \
              6, \
              2 \
          ),
              op_price NUMERIC \
          ( \
              12, \
              4 \
          ),
              target_avg_price NUMERIC \
          ( \
              8, \
              2 \
          ),
              mkt_score NUMERIC \
          ( \
              6, \
              2 \
          ),
              gap_score NUMERIC \
          ( \
              6, \
              2 \
          ),
              earn_date DATE,
              positive_ratings INT,
              total_ratings INT,
              exit_date DATE,
              dividend_yield NUMERIC \
          ( \
              8, \
              2 \
          ),

              -- Performance tracking (filled later via update_returns)
              price_1d NUMERIC \
          ( \
              12, \
              4 \
          ),
              price_5d NUMERIC \
          ( \
              12, \
              4 \
          ),
              price_10d NUMERIC \
          ( \
              12, \
              4 \
          ),
              price_20d NUMERIC \
          ( \
              12, \
              4 \
          ),
              return_1d NUMERIC \
          ( \
              8, \
              4 \
          ),
              return_5d NUMERIC \
          ( \
              8, \
              4 \
          ),
              return_10d NUMERIC \
          ( \
              8, \
              4 \
          ),
              return_20d NUMERIC \
          ( \
              8, \
              4 \
          ),

              -- Signal classification
              signal_type VARCHAR \
          ( \
              20 \
          ),
              signal_correct BOOLEAN, -- Was the signal direction correct?

              created_at TIMESTAMP DEFAULT NOW \
          ( \
          ),
              UNIQUE \
          ( \
              score_date, \
              ticker \
          )
              );

          CREATE INDEX IF NOT EXISTS idx_hist_scores_date ON historical_scores(score_date);
          CREATE INDEX IF NOT EXISTS idx_hist_scores_ticker ON historical_scores(ticker);
          CREATE INDEX IF NOT EXISTS idx_hist_scores_sentiment ON historical_scores(sentiment);
          CREATE INDEX IF NOT EXISTS idx_hist_scores_total ON historical_scores(total_score); \
          """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)

    logger.info("Created historical_scores table")


def parse_date_from_filename(filename: str) -> datetime:
    """Extract date from filename like '2025-12-08_sentiment_likelyhood_sorted.csv'"""
    basename = os.path.basename(filename)
    # Try to extract YYYY-MM-DD from start of filename
    try:
        date_str = basename[:10]  # First 10 chars
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        # If no date in filename, use file modification time
        return datetime.fromtimestamp(os.path.getmtime(filename))


def import_csv(filepath: str, score_date: datetime = None):
    """
    Import a single CSV file into historical_scores table.

    Args:
        filepath: Path to CSV file
        score_date: Date of the scores (extracted from filename if not provided)
    """
    if score_date is None:
        score_date = parse_date_from_filename(filepath)

    logger.info(f"Importing {filepath} with date {score_date.strftime('%Y-%m-%d')}")

    # Read CSV
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            logger.warning(f"Skipping empty file: {filepath}")
            return 0
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return 0
    # Map column names to database columns
    column_mapping = {
        'Ticker': 'ticker',
        'Sector': 'sector',
        'Volume': 'volume',
        'Sentiment': 'sentiment',
        'FundamentalScore': 'fundamental_score',
        'GrowthScore': 'growth_score',
        'DividendScore': 'dividend_score',
        'Counter': 'counter',
        'Total': 'total_score',
        'OpPrice': 'op_price',
        'TargetAvgPrice': 'target_avg_price',
        'MktScore': 'mkt_score',
        'GapType': 'gap_score',  # In old system this was a score
        'EarnDate': 'earn_date',
        'PositiveRatings': 'positive_ratings',
        'TotalRatings': 'total_ratings',
        'ExitDate': 'exit_date',
        'DividendYield': 'dividend_yield',
    }

    df = df.rename(columns=column_mapping)

    # Add score_date
    df['score_date'] = score_date.strftime('%Y-%m-%d')

    # Clean data
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
    df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce').fillna(50).astype(int)
    df['fundamental_score'] = pd.to_numeric(df['fundamental_score'], errors='coerce').fillna(50).astype(int)
    df['growth_score'] = pd.to_numeric(df['growth_score'], errors='coerce').fillna(50).astype(int)
    df['dividend_score'] = pd.to_numeric(df['dividend_score'], errors='coerce').fillna(50).astype(int)
    df['counter'] = pd.to_numeric(df['counter'], errors='coerce').fillna(0).astype(int)
    df['positive_ratings'] = pd.to_numeric(df['positive_ratings'], errors='coerce').fillna(0).astype(int)
    df['total_ratings'] = pd.to_numeric(df['total_ratings'], errors='coerce').fillna(0).astype(int)

    # Handle dates
    def parse_date(val):
        if pd.isna(val) or val == '0.0' or val == 0 or val == '0':
            return None
        try:
            return pd.to_datetime(val).strftime('%Y-%m-%d')
        except:
            return None

    df['earn_date'] = df['earn_date'].apply(parse_date)
    df['exit_date'] = df['exit_date'].apply(parse_date)

    # Generate signal type based on total score
    def get_signal_type(total):
        if total >= 70:
            return 'STRONG_BUY'
        elif total >= 60:
            return 'BUY'
        elif total >= 55:
            return 'WEAK_BUY'
        elif total >= 45:
            return 'NEUTRAL'
        elif total >= 40:
            return 'WEAK_SELL'
        else:
            return 'SELL'

    df['signal_type'] = df['total_score'].apply(get_signal_type)

    # Insert into database
    inserted = 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                try:
                    cur.execute("""
                                INSERT INTO historical_scores (score_date, ticker, sector, volume, sentiment,
                                                               fundamental_score, growth_score, dividend_score, counter,
                                                               total_score, op_price, target_avg_price, mkt_score,
                                                               gap_score, earn_date, positive_ratings, total_ratings,
                                                               exit_date, dividend_yield, signal_type)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (score_date, ticker) DO
                                UPDATE SET
                                    sentiment = EXCLUDED.sentiment,
                                    fundamental_score = EXCLUDED.fundamental_score,
                                    total_score = EXCLUDED.total_score,
                                    signal_type = EXCLUDED.signal_type
                                """, (
                                    row['score_date'],
                                    row['ticker'],
                                    row.get('sector'),
                                    row['volume'],
                                    row['sentiment'],
                                    row['fundamental_score'],
                                    row['growth_score'],
                                    row['dividend_score'],
                                    row['counter'],
                                    row.get('total_score'),
                                    row.get('op_price'),
                                    row.get('target_avg_price'),
                                    row.get('mkt_score'),
                                    row.get('gap_score'),
                                    row.get('earn_date'),
                                    row['positive_ratings'],
                                    row['total_ratings'],
                                    row.get('exit_date'),
                                    row.get('dividend_yield'),
                                    row['signal_type'],
                                ))
                    inserted += 1
                except Exception as e:
                    logger.error(f"Error inserting {row['ticker']}: {e}")

    logger.info(f"Imported {inserted} scores from {filepath}")
    return inserted


def update_returns():
    """
    Fetch actual price changes after each score date and calculate returns.
    This enables backtesting to see if signals were correct.
    """
    logger.info("Updating returns for historical scores...")

    # Get all scores that don't have returns yet
    query = """
            SELECT DISTINCT score_date, ticker, op_price
            FROM historical_scores
            WHERE return_5d IS NULL
            ORDER BY score_date DESC \
            """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    if not rows:
        logger.info("All returns already calculated")
        return

    logger.info(f"Fetching returns for {len(rows)} score/ticker combinations")

    # Group by ticker for efficient fetching
    ticker_dates = {}
    for score_date, ticker, op_price in rows:
        if ticker not in ticker_dates:
            ticker_dates[ticker] = []
        ticker_dates[ticker].append((score_date, op_price))

    updates = []

    for ticker, dates in ticker_dates.items():
        try:
            # Get price history
            min_date = min(d[0] for d in dates)
            max_date = max(d[0] for d in dates) + timedelta(days=30)

            stock = yf.Ticker(ticker)
            hist = stock.history(start=min_date, end=max_date)

            if hist.empty:
                logger.warning(f"{ticker}: No price history available")
                continue

            for score_date, op_price in dates:
                try:
                    # Find prices at different intervals
                    date_1d = score_date + timedelta(days=1)
                    date_5d = score_date + timedelta(days=5)
                    date_10d = score_date + timedelta(days=10)
                    date_20d = score_date + timedelta(days=20)

                    def get_price(target_date, max_days=5):
                        """Get price on or near target date."""
                        for offset in range(max_days):
                            d = target_date + timedelta(days=offset)
                            d_str = d.strftime('%Y-%m-%d')
                            if d_str in hist.index.strftime('%Y-%m-%d').tolist():
                                idx = hist.index.strftime('%Y-%m-%d').tolist().index(d_str)
                                return float(hist['Close'].iloc[idx])
                        return None

                    price_1d = get_price(date_1d)
                    price_5d = get_price(date_5d)
                    price_10d = get_price(date_10d)
                    price_20d = get_price(date_20d)

                    # Calculate returns
                    base_price = float(op_price) if op_price else None

                    if base_price and base_price > 0:
                        return_1d = ((price_1d - base_price) / base_price * 100) if price_1d else None
                        return_5d = ((price_5d - base_price) / base_price * 100) if price_5d else None
                        return_10d = ((price_10d - base_price) / base_price * 100) if price_10d else None
                        return_20d = ((price_20d - base_price) / base_price * 100) if price_20d else None
                    else:
                        return_1d = return_5d = return_10d = return_20d = None

                    updates.append((
                        price_1d, price_5d, price_10d, price_20d,
                        return_1d, return_5d, return_10d, return_20d,
                        score_date, ticker
                    ))

                except Exception as e:
                    logger.error(f"{ticker} on {score_date}: Error calculating returns - {e}")

        except Exception as e:
            logger.error(f"{ticker}: Error fetching history - {e}")

    # Batch update
    if updates:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for update in updates:
                    cur.execute("""
                                UPDATE historical_scores
                                SET price_1d       = %s,
                                    price_5d       = %s,
                                    price_10d      = %s,
                                    price_20d      = %s,
                                    return_1d      = %s,
                                    return_5d      = %s,
                                    return_10d     = %s,
                                    return_20d     = %s,
                                    signal_correct = CASE
                                                         WHEN signal_type IN ('STRONG_BUY', 'BUY', 'WEAK_BUY') AND %s > 0
                                                             THEN TRUE
                                                         WHEN signal_type IN ('SELL', 'WEAK_SELL') AND %s < 0 THEN TRUE
                                                         WHEN signal_type = 'NEUTRAL' THEN NULL
                                                         ELSE FALSE
                                        END
                                WHERE score_date = %s
                                  AND ticker = %s
                                """,
                                update[:8] + (update[6], update[6]) + update[8:])  # return_10d used for signal_correct

        logger.info(f"Updated returns for {len(updates)} scores")


def analyze_performance():
    """Analyze historical signal performance."""
    query = """
            SELECT signal_type, \
                   COUNT(*) as count,
            AVG(return_1d) as avg_return_1d,
            AVG(return_5d) as avg_return_5d,
            AVG(return_10d) as avg_return_10d,
            AVG(return_20d) as avg_return_20d,
            SUM(CASE WHEN signal_correct = TRUE THEN 1 ELSE 0 END)::float / 
                NULLIF(SUM(CASE WHEN signal_correct IS NOT NULL THEN 1 ELSE 0 END), 0) * 100 as accuracy_pct
            FROM historical_scores
            WHERE return_10d IS NOT NULL
            GROUP BY signal_type
            ORDER BY avg_return_10d DESC \
            """

    with get_connection() as conn:
        df = pd.read_sql(query, conn)

    print("\n" + "=" * 70)
    print("HISTORICAL SIGNAL PERFORMANCE ANALYSIS")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    return df


def main():
    parser = argparse.ArgumentParser(description='Import historical scores for backtesting')
    parser.add_argument('--file', type=str, help='Single CSV file to import')
    parser.add_argument('--dir', type=str, help='Directory with CSV files to import')
    parser.add_argument('--update-returns', action='store_true', help='Fetch actual price changes')
    parser.add_argument('--analyze', action='store_true', help='Analyze signal performance')
    parser.add_argument('--create-table', action='store_true', help='Create the table only')

    args = parser.parse_args()

    # Always ensure table exists
    create_table()

    if args.create_table:
        print("Table created successfully")
        return

    if args.file:
        import_csv(args.file)

    if args.dir:
        csv_files = glob.glob(os.path.join(args.dir, '*.csv'))
        for f in sorted(csv_files):
            import_csv(f)

    if args.update_returns:
        update_returns()

    if args.analyze:
        analyze_performance()

    # If no specific action, show help
    if not any([args.file, args.dir, args.update_returns, args.analyze, args.create_table]):
        parser.print_help()


if __name__ == "__main__":
    main()