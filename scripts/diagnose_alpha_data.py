"""
Alpha Model Data Diagnostic Script v2
======================================

First discovers what tables exist, then checks data availability.

Usage:
    python scripts/diagnose_alpha_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Database connection
try:
    from src.db.connection import get_engine
    from sqlalchemy import text
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    print("❌ Cannot import database connection")


def run_query(sql):
    """Run a SQL query and return DataFrame."""
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def discover_tables():
    """Discover all tables in the database."""
    print("\n" + "=" * 70)
    print("  1. DISCOVERING DATABASE TABLES")
    print("=" * 70)

    try:
        tables = run_query("""
            SELECT table_name, 
                   pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) as size
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)

        print(f"\n  📊 Found {len(tables)} tables:")

        # Categorize tables
        score_tables = []
        price_tables = []
        other_tables = []

        for _, row in tables.iterrows():
            name = row['table_name']
            size = row['size']

            if 'score' in name.lower() or 'signal' in name.lower() or 'alpha' in name.lower() or 'analysis' in name.lower():
                score_tables.append((name, size))
            elif 'price' in name.lower() or 'daily' in name.lower() or 'ohlc' in name.lower():
                price_tables.append((name, size))
            else:
                other_tables.append((name, size))

        print(f"\n  📈 SCORE/SIGNAL/ANALYSIS TABLES ({len(score_tables)}):")
        for name, size in score_tables:
            print(f"       - {name:40} ({size})")

        print(f"\n  💰 PRICE/DAILY TABLES ({len(price_tables)}):")
        for name, size in price_tables:
            print(f"       - {name:40} ({size})")

        print(f"\n  📋 OTHER TABLES ({len(other_tables)}):")
        for name, size in other_tables[:20]:
            print(f"       - {name:40} ({size})")
        if len(other_tables) > 20:
            print(f"       ... and {len(other_tables) - 20} more")

        return {
            'score_tables': [t[0] for t in score_tables],
            'price_tables': [t[0] for t in price_tables],
            'all_tables': tables['table_name'].tolist()
        }

    except Exception as e:
        print(f"  ❌ Error discovering tables: {e}")
        return None


def analyze_table(table_name):
    """Analyze a specific table."""
    print(f"\n  {'─' * 60}")
    print(f"  Analyzing: {table_name}")
    print(f"  {'─' * 60}")

    try:
        # Get row count and date range
        info = run_query(f"""
            SELECT 
                COUNT(*) as total_rows
            FROM {table_name}
        """)

        print(f"     Total rows: {info['total_rows'].iloc[0]:,}")

        # Get columns
        cols = run_query(f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """)

        col_list = cols['column_name'].tolist()
        print(f"     Columns ({len(col_list)}): {', '.join(col_list[:10])}")
        if len(col_list) > 10:
            print(f"                    ... and {len(col_list) - 10} more")

        # Check for date column
        date_cols = [c for c in col_list if 'date' in c.lower() or 'time' in c.lower()]
        if date_cols:
            date_col = date_cols[0]
            date_range = run_query(f"""
                SELECT MIN({date_col}) as min_date, MAX({date_col}) as max_date
                FROM {table_name}
            """)
            print(f"     Date range ({date_col}): {date_range['min_date'].iloc[0]} to {date_range['max_date'].iloc[0]}")

        # Check for ticker column
        ticker_cols = [c for c in col_list if 'ticker' in c.lower() or 'symbol' in c.lower()]
        if ticker_cols:
            ticker_col = ticker_cols[0]
            ticker_count = run_query(f"""
                SELECT COUNT(DISTINCT {ticker_col}) as unique_tickers
                FROM {table_name}
            """)
            print(f"     Unique tickers: {ticker_count['unique_tickers'].iloc[0]:,}")

        # Check for score columns
        score_cols = [c for c in col_list if 'score' in c.lower()]
        if score_cols:
            print(f"     Score columns: {', '.join(score_cols)}")

        return True

    except Exception as e:
        print(f"     ❌ Error: {e}")
        return False


def check_alpha_data_loader():
    """Check what the AlphaDataLoader actually does."""
    print("\n" + "=" * 70)
    print("  2. ALPHA DATA LOADER ANALYSIS")
    print("=" * 70)

    try:
        from src.ml.multi_factor_alpha import MultiFactorAlphaModel, AlphaDataLoader

        # Check AlphaDataLoader methods
        loader = AlphaDataLoader()

        print(f"\n  📋 AlphaDataLoader methods:")
        methods = [m for m in dir(loader) if not m.startswith('_') and callable(getattr(loader, m))]
        for m in methods:
            print(f"       - {m}")

        # Try to load data using the correct method
        print(f"\n  🔧 Trying to load historical data...")

        if hasattr(loader, 'load_historical_data'):
            df = loader.load_historical_data(min_date='2023-01-01')
            if df is not None and not df.empty:
                print(f"     ✅ load_historical_data() returned {len(df):,} rows")
                print(f"     Columns: {list(df.columns)[:10]}")
                if 'date' in df.columns:
                    print(f"     Date range: {df['date'].min()} to {df['date'].max()}")
                if 'ticker' in df.columns:
                    print(f"     Unique tickers: {df['ticker'].nunique()}")

                # Check for return columns
                return_cols = [c for c in df.columns if 'return' in c.lower() or 'fwd' in c.lower()]
                if return_cols:
                    print(f"     Return columns: {return_cols}")
                else:
                    print(f"     ⚠️  No return columns found!")
            else:
                print(f"     ⚠️  load_historical_data() returned empty/None")

        if hasattr(loader, 'get_live_factors'):
            print(f"\n  🔧 Trying to get live factors...")
            live = loader.get_live_factors()
            if live is not None and not live.empty:
                print(f"     ✅ get_live_factors() returned {len(live):,} rows")
            else:
                print(f"     ⚠️  get_live_factors() returned empty/None")

        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model_training():
    """Try to understand what data the model needs for training."""
    print("\n" + "=" * 70)
    print("  3. MODEL TRAINING ATTEMPT")
    print("=" * 70)

    try:
        from src.ml.multi_factor_alpha import MultiFactorAlphaModel

        model = MultiFactorAlphaModel()

        print(f"\n  🔧 Model initialized")
        print(f"     Target horizons: {model.target_horizons}")
        print(f"     Use regime models: {model.use_regime_models}")
        print(f"     Use sector models: {model.use_sector_models}")

        # Try a dry run of training
        print(f"\n  🔧 Attempting to load training data...")

        # Load data through the model's data loader
        df = model.data_loader.load_historical_data(min_date='2020-01-01')

        if df is None or df.empty:
            print(f"     ❌ No data loaded!")
            return False

        print(f"\n  📊 Training Data Available:")
        print(f"     Total samples: {len(df):,}")
        print(f"     Unique tickers: {df['ticker'].nunique() if 'ticker' in df.columns else 'N/A'}")

        if 'date' in df.columns:
            print(f"     Date range: {df['date'].min()} to {df['date'].max()}")

            # Show data by year
            df['year'] = pd.to_datetime(df['date']).dt.year
            print(f"\n  📅 Data by Year:")
            for year, count in df.groupby('year').size().items():
                print(f"       {year}: {count:,} rows")

        # Check for required columns
        print(f"\n  📋 Column Analysis:")

        score_cols = [c for c in df.columns if 'score' in c.lower()]
        print(f"     Score columns ({len(score_cols)}): {score_cols[:8]}")

        return_cols = [c for c in df.columns if 'return' in c.lower() or 'fwd' in c.lower()]
        print(f"     Return columns ({len(return_cols)}): {return_cols}")

        if not return_cols:
            print(f"\n  ⚠️  WARNING: No return columns found!")
            print(f"     The model needs forward return columns to train")
            print(f"     Expected columns like: fwd_return_5d, fwd_return_10d, return_5d, etc.")

        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("  ALPHA MODEL DATA DIAGNOSTIC v2")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    if not ENGINE_AVAILABLE:
        print("\n❌ Cannot connect to database. Check your connection settings.")
        return

    # Step 1: Discover all tables
    tables_info = discover_tables()

    if tables_info:
        # Analyze score tables
        if tables_info['score_tables']:
            print("\n" + "=" * 70)
            print("  ANALYZING SCORE/SIGNAL TABLES")
            print("=" * 70)

            for table in tables_info['score_tables'][:5]:
                analyze_table(table)

        # Analyze price tables
        if tables_info['price_tables']:
            print("\n" + "=" * 70)
            print("  ANALYZING PRICE/DAILY TABLES")
            print("=" * 70)

            for table in tables_info['price_tables'][:5]:
                analyze_table(table)

    # Step 2: Check AlphaDataLoader
    check_alpha_data_loader()

    # Step 3: Check model training
    check_model_training()

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    if tables_info:
        if tables_info['score_tables']:
            print(f"\n  ✅ Score/analysis tables exist: {', '.join(tables_info['score_tables'][:3])}")
        else:
            print(f"\n  ❌ No score tables found - run signal generation first")

        if tables_info['price_tables']:
            print(f"  ✅ Price tables exist: {', '.join(tables_info['price_tables'][:3])}")
        else:
            print(f"  ❌ No price tables found - need price data for returns")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()