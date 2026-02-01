"""
Fetch and Insert VIX Historical Data
=====================================

Downloads VIX historical data from Yahoo Finance and inserts it into the prices table.
This enables proper market regime detection for the alpha model.

Usage:
    python scripts/fetch_vix_data.py

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

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance not installed. Run: pip install yfinance")


def fetch_vix_from_yahoo(start_date='2020-01-01'):
    """Fetch VIX data from Yahoo Finance."""
    print(f"📊 Fetching VIX data from Yahoo Finance...")
    print(f"   Start date: {start_date}")

    vix = yf.Ticker("^VIX")
    df = vix.history(start=start_date)

    if df.empty:
        print("   ❌ No data returned from Yahoo Finance")
        return None

    # Reset index to get date as column
    df = df.reset_index()

    # Rename columns to match prices table schema
    df = df.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    # Add required columns
    df['ticker'] = '^VIX'
    df['adj_close'] = df['close']  # VIX doesn't have adjustments

    # Select only needed columns
    df = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]

    # Convert date to date only (no time)
    df['date'] = pd.to_datetime(df['date']).dt.date

    print(f"   ✅ Fetched {len(df)} days of VIX data")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   VIX range: {df['close'].min():.2f} to {df['close'].max():.2f}")

    return df


def analyze_vix_regimes(df):
    """Analyze what regimes the VIX data would produce."""
    print(f"\n📈 VIX Regime Analysis:")

    # Classify regimes
    def classify(vix):
        if vix < 15:
            return 'BULL'
        elif vix < 20:
            return 'NEUTRAL'
        elif vix < 30:
            return 'BEAR'
        else:
            return 'HIGH_VOL'

    df['regime'] = df['close'].apply(classify)

    # Count by regime
    regime_counts = df['regime'].value_counts()
    total = len(df)

    print(f"\n   Regime distribution:")
    for regime, count in regime_counts.items():
        pct = count / total * 100
        print(f"      {regime:10}: {count:5} days ({pct:5.1f}%)")

    # Show by year
    df['year'] = pd.to_datetime(df['date']).dt.year
    print(f"\n   By year:")
    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]
        avg_vix = year_df['close'].mean()
        dominant_regime = year_df['regime'].mode().iloc[0] if len(year_df) > 0 else 'N/A'
        print(f"      {year}: Avg VIX={avg_vix:.1f}, Dominant regime={dominant_regime}")

    return df


def insert_vix_data(df):
    """Insert VIX data into prices table."""
    print(f"\n📝 Inserting VIX data into database...")

    engine = get_engine()

    # First, delete any existing VIX data
    with engine.connect() as conn:
        result = conn.execute(text("DELETE FROM prices WHERE ticker = '^VIX'"))
        deleted = result.rowcount
        conn.commit()
        if deleted > 0:
            print(f"   Deleted {deleted} existing VIX rows")

    # Insert new data
    inserted = 0
    batch_size = 500

    with engine.connect() as conn:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]

            for _, row in batch.iterrows():
                try:
                    conn.execute(text("""
                        INSERT INTO prices (ticker, date, open, high, low, close, adj_close, volume)
                        VALUES (:ticker, :date, :open, :high, :low, :close, :adj_close, :volume)
                        ON CONFLICT (ticker, date) DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            adj_close = EXCLUDED.adj_close,
                            volume = EXCLUDED.volume
                    """), {
                        'ticker': row['ticker'],
                        'date': row['date'],
                        'open': float(row['open']) if pd.notna(row['open']) else None,
                        'high': float(row['high']) if pd.notna(row['high']) else None,
                        'low': float(row['low']) if pd.notna(row['low']) else None,
                        'close': float(row['close']) if pd.notna(row['close']) else None,
                        'adj_close': float(row['adj_close']) if pd.notna(row['adj_close']) else None,
                        'volume': int(row['volume']) if pd.notna(row['volume']) else 0
                    })
                    inserted += 1
                except Exception as e:
                    print(f"   ⚠️ Error inserting {row['date']}: {e}")

            conn.commit()
            print(f"   Inserted {min(i + batch_size, len(df))} / {len(df)} rows...")

    print(f"\n   ✅ Successfully inserted {inserted} VIX records")


def verify_insertion():
    """Verify VIX data was inserted correctly."""
    print(f"\n📊 Verifying insertion...")

    engine = get_engine()

    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT 
                COUNT(*) as total_rows,
                MIN(date) as min_date,
                MAX(date) as max_date,
                AVG(close) as avg_vix,
                MIN(close) as min_vix,
                MAX(close) as max_vix
            FROM prices
            WHERE ticker = '^VIX'
        """), conn)

    print(f"   VIX data in database:")
    print(f"      Total rows: {df['total_rows'].iloc[0]}")
    print(f"      Date range: {df['min_date'].iloc[0]} to {df['max_date'].iloc[0]}")
    print(f"      VIX range: {df['min_vix'].iloc[0]:.2f} to {df['max_vix'].iloc[0]:.2f}")
    print(f"      Avg VIX: {df['avg_vix'].iloc[0]:.2f}")


def main():
    print("=" * 70)
    print("  FETCH AND INSERT VIX DATA")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    if not YFINANCE_AVAILABLE:
        print("\n❌ yfinance is required. Install with: pip install yfinance")
        return

    # Fetch VIX data
    df = fetch_vix_from_yahoo(start_date='2020-01-01')

    if df is None or df.empty:
        print("\n❌ Failed to fetch VIX data")
        return

    # Analyze regimes
    df = analyze_vix_regimes(df)

    # Confirm insertion
    confirm = input(f"\n⚠️  Insert {len(df)} VIX records into database? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Cancelled.")
        return

    # Insert data
    insert_vix_data(df)

    # Verify
    verify_insertion()

    print("\n" + "=" * 70)
    print("  DONE! Now retrain the alpha model to see different regimes:")
    print('  python -c "from src.ml.multi_factor_alpha import train_alpha_model; train_alpha_model()"')
    print("=" * 70)


if __name__ == "__main__":
    main()