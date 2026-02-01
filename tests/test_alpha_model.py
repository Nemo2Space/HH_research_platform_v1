"""
Test script for Multi-Factor Alpha Model

Run: python test_alpha_model.py
"""

from dotenv import load_dotenv

load_dotenv()

from src.ml.multi_factor_alpha import MultiFactorAlphaModel

print("=" * 60)
print("MULTI-FACTOR ALPHA MODEL - TEST")
print("=" * 60)

print("\n1. Creating model...")
model = MultiFactorAlphaModel()
print("   ✅ Model created")

print("\n2. Testing data load...")
try:
    df = model.data_loader.load_historical_data(min_date='2023-01-01')
    print(f"   ✅ Loaded {len(df)} rows")

    if len(df) > 0:
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Tickers: {df['ticker'].nunique()}")
        print(f"\n   Columns: {list(df.columns)}")
        print(f"\n   Sample data:")
        print(df.head(3).to_string())

        # Check for required columns
        print("\n3. Checking required columns...")
        required = ['return_5d', 'return_10d', 'sentiment_score', 'fundamental_score', 'technical_score']
        for col in required:
            if col in df.columns:
                non_null = df[col].notna().sum()
                print(f"   ✅ {col}: {non_null} non-null values")
            else:
                print(f"   ❌ {col}: MISSING")

        print("\n4. Ready to train!")
        print("   Run: python scripts/alpha_model_cli.py train --min-date 2023-01-01")
    else:
        print("   ❌ No data loaded - check your historical_scores table")

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback

    traceback.print_exc()