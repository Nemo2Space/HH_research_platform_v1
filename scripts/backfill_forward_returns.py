"""
Backfill Forward Returns for Alpha Model Training
===================================================

This script calculates forward returns (1d, 5d, 10d, 20d) for all historical_scores
rows that are missing this data.

Usage:
    python scripts/backfill_forward_returns.py

Author: HH Research Platform
Date: January 2026
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.db.connection import get_engine
from sqlalchemy import text

def load_prices():
    """Load all price data."""
    print("📊 Loading price data...")
    engine = get_engine()

    with engine.connect() as conn:
        prices = pd.read_sql(text("""
            SELECT ticker, date, close as price
            FROM prices
            WHERE close IS NOT NULL AND close > 0
            ORDER BY ticker, date
        """), conn)

    print(f"   Loaded {len(prices):,} price records for {prices['ticker'].nunique()} tickers")
    print(f"   Date range: {prices['date'].min()} to {prices['date'].max()}")

    return prices


def load_historical_scores_missing_returns():
    """Load historical scores that are missing forward returns."""
    print("\n📊 Loading historical scores missing returns...")
    engine = get_engine()

    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT id, ticker, score_date, op_price
            FROM historical_scores
            WHERE op_price IS NOT NULL 
              AND op_price > 0
              AND (return_5d IS NULL OR return_10d IS NULL OR return_20d IS NULL)
            ORDER BY ticker, score_date
        """), conn)

    print(f"   Found {len(df):,} rows missing forward returns")
    print(f"   Tickers: {df['ticker'].nunique()}")

    return df


def calculate_forward_returns(scores_df, prices_df):
    """
    Calculate forward returns for each score row.

    For each score_date, we look up the price on that date and the price
    N days later, then calculate the return.
    """
    print("\n🔧 Calculating forward returns...")

    # Create a price lookup dict: {(ticker, date): price}
    prices_df['date'] = pd.to_datetime(prices_df['date']).dt.date
    price_lookup = {}
    for _, row in prices_df.iterrows():
        key = (row['ticker'], row['date'])
        price_lookup[key] = row['price']

    print(f"   Built price lookup with {len(price_lookup):,} entries")

    # Convert score_date to date
    scores_df['score_date'] = pd.to_datetime(scores_df['score_date']).dt.date

    # Calculate returns
    results = []
    missing_prices = 0
    calculated = 0

    horizons = [1, 5, 10, 20]

    for idx, row in scores_df.iterrows():
        ticker = row['ticker']
        score_date = row['score_date']
        base_price = row['op_price']

        # Get base date price from lookup (or use op_price as fallback)
        base_key = (ticker, score_date)
        if base_key in price_lookup:
            base_price = price_lookup[base_key]

        if base_price is None or base_price <= 0:
            missing_prices += 1
            continue

        returns = {'id': row['id']}
        has_any_return = False

        for horizon in horizons:
            # Find trading days forward (approximate by adding calendar days)
            # In reality, you'd want to find exact trading days
            future_date = score_date + timedelta(days=horizon)

            # Look for price within a few days (in case of weekends/holidays)
            future_price = None
            for offset in range(0, 5):  # Try up to 5 days forward
                check_date = future_date + timedelta(days=offset)
                check_key = (ticker, check_date)
                if check_key in price_lookup:
                    future_price = price_lookup[check_key]
                    break

            if future_price and future_price > 0:
                ret = ((future_price - base_price) / base_price) * 100  # Percentage return
                returns[f'return_{horizon}d'] = round(ret, 4)
                has_any_return = True
            else:
                returns[f'return_{horizon}d'] = None

        if has_any_return:
            results.append(returns)
            calculated += 1

        # Progress update
        if (idx + 1) % 5000 == 0:
            print(f"   Processed {idx + 1:,} / {len(scores_df):,} rows...")

    print(f"\n   ✅ Calculated returns for {calculated:,} rows")
    print(f"   ⚠️  Missing base prices: {missing_prices:,}")

    return pd.DataFrame(results)


def update_database(returns_df):
    """Update the database with calculated returns."""
    if returns_df.empty:
        print("\n⚠️  No returns to update")
        return

    print(f"\n📝 Updating database with {len(returns_df):,} rows...")

    engine = get_engine()

    # Update in batches
    batch_size = 1000
    updated = 0

    with engine.connect() as conn:
        for i in range(0, len(returns_df), batch_size):
            batch = returns_df.iloc[i:i+batch_size]

            for _, row in batch.iterrows():
                update_parts = []
                params = {'id': int(row['id'])}

                if pd.notna(row.get('return_1d')):
                    update_parts.append("return_1d = :return_1d")
                    params['return_1d'] = float(row['return_1d'])  # Convert to Python float

                if pd.notna(row.get('return_5d')):
                    update_parts.append("return_5d = :return_5d")
                    params['return_5d'] = float(row['return_5d'])  # Convert to Python float

                if pd.notna(row.get('return_10d')):
                    update_parts.append("return_10d = :return_10d")
                    params['return_10d'] = float(row['return_10d'])  # Convert to Python float

                if pd.notna(row.get('return_20d')):
                    update_parts.append("return_20d = :return_20d")
                    params['return_20d'] = float(row['return_20d'])  # Convert to Python float

                if update_parts:
                    sql = f"UPDATE historical_scores SET {', '.join(update_parts)} WHERE id = :id"
                    conn.execute(text(sql), params)
                    updated += 1

            conn.commit()
            print(f"   Updated {min(i + batch_size, len(returns_df)):,} / {len(returns_df):,} rows...")

    print(f"\n   ✅ Successfully updated {updated:,} rows in database")


def verify_results():
    """Verify the backfill results."""
    print("\n📊 Verifying results...")

    engine = get_engine()

    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN return_5d IS NOT NULL THEN 1 ELSE 0 END) as has_return_5d,
                SUM(CASE WHEN return_10d IS NOT NULL THEN 1 ELSE 0 END) as has_return_10d,
                SUM(CASE WHEN return_20d IS NOT NULL THEN 1 ELSE 0 END) as has_return_20d
            FROM historical_scores
            WHERE op_price IS NOT NULL AND op_price > 0
        """), conn)

    print(f"\n   Results:")
    print(f"   Total rows with price: {df['total'].iloc[0]:,}")
    print(f"   Has return_5d:  {df['has_return_5d'].iloc[0]:,} ({df['has_return_5d'].iloc[0]/df['total'].iloc[0]*100:.1f}%)")
    print(f"   Has return_10d: {df['has_return_10d'].iloc[0]:,} ({df['has_return_10d'].iloc[0]/df['total'].iloc[0]*100:.1f}%)")
    print(f"   Has return_20d: {df['has_return_20d'].iloc[0]:,} ({df['has_return_20d'].iloc[0]/df['total'].iloc[0]*100:.1f}%)")

    # Check date distribution
    df2 = pd.read_sql(text("""
        SELECT 
            DATE_TRUNC('month', score_date) as month,
            COUNT(*) as usable_rows
        FROM historical_scores
        WHERE op_price IS NOT NULL AND op_price > 0 AND return_5d IS NOT NULL
        GROUP BY DATE_TRUNC('month', score_date)
        ORDER BY month
    """), conn)

    print(f"\n   Usable data by month:")
    for _, row in df2.iterrows():
        print(f"      {row['month'].strftime('%Y-%m')}: {row['usable_rows']:,} rows")


def main():
    print("=" * 70)
    print("  BACKFILL FORWARD RETURNS FOR ALPHA MODEL")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    # Load data
    prices = load_prices()
    scores = load_historical_scores_missing_returns()

    if scores.empty:
        print("\n✅ All rows already have forward returns!")
        verify_results()
        return

    # Calculate returns
    returns_df = calculate_forward_returns(scores, prices)

    if returns_df.empty:
        print("\n⚠️  Could not calculate any returns - check price data coverage")
        return

    # Preview
    print(f"\n📋 Preview of calculated returns:")
    print(returns_df.head(10))

    # Confirm update
    confirm = input(f"\n⚠️  Update {len(returns_df):,} rows in database? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Cancelled.")
        return

    # Update database
    update_database(returns_df)

    # Verify
    verify_results()

    print("\n" + "=" * 70)
    print("  DONE! You can now retrain the alpha model with more data.")
    print("  Run: python -c \"from src.ml.multi_factor_alpha import train_alpha_model; train_alpha_model()\"")
    print("=" * 70)


if __name__ == "__main__":
    main()