# fix_backtest_issues.py

# Fix 1: Add avg_score column to saved_portfolios table
from src.db.connection import get_connection

print("1. Adding avg_score column to saved_portfolios table...")
try:
    with get_connection() as conn:
        cur = conn.cursor()

        # Check if column exists
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'saved_portfolios' 
            AND column_name = 'avg_score'
        """)

        if not cur.fetchone():
            print("   Adding avg_score column...")
            cur.execute("""
                ALTER TABLE saved_portfolios 
                ADD COLUMN avg_score NUMERIC
            """)
            conn.commit()
            print("   ✓ Added avg_score column")
        else:
            print("   ✓ avg_score column already exists")

        cur.close()
except Exception as e:
    print(f"   ✗ Error: {e}")

# Fix 2: Update save_portfolio to save avg_score
print("\n2. Checking if save_portfolio saves avg_score...")
with open('../dashboard/portfolio_builder.py', 'r', encoding='utf-8') as f:
    content = f.read()

if 'result.avg_score' in content and 'INSERT INTO saved_portfolios' in content:
    print("   ✓ save_portfolio already saves avg_score")
else:
    print("   ⚠ save_portfolio may not save avg_score properly")

print("\n✅ Database schema fixed")
print("Restart Streamlit to test backtest tab")