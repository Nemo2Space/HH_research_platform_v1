"""
Diagnostic and Fix Script for Portfolio Holdings Conviction Column Issue
=========================================================================

This script will:
1. Check if the conviction column exists in saved_portfolio_holdings
2. Add the column if it's missing
3. Verify the fix worked

Run this script to fix the "column conviction does not exist" error.

Usage:
    python fix_conviction_column.py
"""

import sys

# Try to import the database connection
try:
    from src.db.connection import get_connection

    DB_IMPORT = "src.db.connection"
except ImportError:
    try:
        from src.utils.db import get_db_connection as get_connection

        DB_IMPORT = "src.utils.db"
    except ImportError:
        import psycopg2


        def get_connection():
            return psycopg2.connect(
                host="localhost",
                port=5432,
                dbname="alpha_platform",
                user="alpha",
                password="alpha_secure_2024"
            )


        DB_IMPORT = "direct psycopg2"


def check_column_exists(conn, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    query = """
        SELECT EXISTS (
            SELECT 1 
            FROM information_schema.columns 
            WHERE table_name = %s 
            AND column_name = %s
        )
    """
    cur = conn.cursor()
    cur.execute(query, (table_name, column_name))
    exists = cur.fetchone()[0]
    cur.close()
    return exists


def get_table_columns(conn, table_name: str) -> list:
    """Get all columns for a table."""
    query = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns 
        WHERE table_name = %s
        ORDER BY ordinal_position
    """
    cur = conn.cursor()
    cur.execute(query, (table_name,))
    columns = cur.fetchall()
    cur.close()
    return columns


def add_conviction_column(conn) -> bool:
    """Add the conviction column to saved_portfolio_holdings."""
    try:
        cur = conn.cursor()
        cur.execute("""
            ALTER TABLE saved_portfolio_holdings 
            ADD COLUMN IF NOT EXISTS conviction VARCHAR(20) DEFAULT 'MEDIUM'
        """)
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        print(f"❌ Error adding column: {e}")
        conn.rollback()
        return False


def check_table_exists(conn, table_name: str) -> bool:
    """Check if a table exists."""
    query = """
        SELECT EXISTS (
            SELECT 1 
            FROM information_schema.tables 
            WHERE table_name = %s
        )
    """
    cur = conn.cursor()
    cur.execute(query, (table_name,))
    exists = cur.fetchone()[0]
    cur.close()
    return exists


def test_query(conn) -> bool:
    """Test the exact query that was failing."""
    try:
        cur = conn.cursor()
        # This is the exact query from portfolio_backtester.py
        cur.execute("""
            SELECT 
                ticker,
                weight_pct,
                value,
                shares,
                score,
                conviction
            FROM saved_portfolio_holdings
            LIMIT 1
        """)
        cur.fetchall()
        cur.close()
        return True
    except Exception as e:
        print(f"❌ Query test failed: {e}")
        return False


def get_portfolio_count(conn) -> int:
    """Get count of saved portfolios."""
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM saved_portfolios")
        count = cur.fetchone()[0]
        cur.close()
        return count
    except:
        return 0


def get_holdings_count(conn) -> int:
    """Get count of saved holdings."""
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM saved_portfolio_holdings")
        count = cur.fetchone()[0]
        cur.close()
        return count
    except:
        return 0


def main():
    print("=" * 70)
    print("Portfolio Holdings Conviction Column - Diagnostic & Fix Script")
    print("=" * 70)
    print()

    print(f"📦 Database import method: {DB_IMPORT}")
    print()

    # Connect to database
    print("🔌 Connecting to database...")
    try:
        conn = get_connection()
        if hasattr(conn, '__enter__'):
            # Context manager style
            conn = conn.__enter__()
        print("✅ Connected successfully")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)

    print()
    print("-" * 70)
    print("STEP 1: Check if tables exist")
    print("-" * 70)

    # Check tables exist
    portfolios_exists = check_table_exists(conn, 'saved_portfolios')
    holdings_exists = check_table_exists(conn, 'saved_portfolio_holdings')

    print(f"  saved_portfolios table exists: {'✅ Yes' if portfolios_exists else '❌ No'}")
    print(f"  saved_portfolio_holdings table exists: {'✅ Yes' if holdings_exists else '❌ No'}")

    if not holdings_exists:
        print()
        print("❌ The saved_portfolio_holdings table doesn't exist!")
        print("   You need to save a portfolio first to create the tables.")
        conn.close()
        sys.exit(1)

    print()
    print("-" * 70)
    print("STEP 2: Check current table schema")
    print("-" * 70)

    columns = get_table_columns(conn, 'saved_portfolio_holdings')
    print(f"  Current columns in saved_portfolio_holdings:")
    for col_name, data_type, nullable in columns:
        print(f"    - {col_name}: {data_type} (nullable: {nullable})")

    print()
    print("-" * 70)
    print("STEP 3: Check for conviction column")
    print("-" * 70)

    conviction_exists = check_column_exists(conn, 'saved_portfolio_holdings', 'conviction')
    print(f"  conviction column exists: {'✅ Yes' if conviction_exists else '❌ No (THIS IS THE PROBLEM!)'}")

    if not conviction_exists:
        print()
        print("-" * 70)
        print("STEP 4: Adding conviction column")
        print("-" * 70)

        print("  Adding 'conviction VARCHAR(20) DEFAULT MEDIUM' column...")
        success = add_conviction_column(conn)

        if success:
            print("  ✅ Column added successfully!")

            # Verify it was added
            conviction_exists = check_column_exists(conn, 'saved_portfolio_holdings', 'conviction')
            print(f"  Verification - conviction column now exists: {'✅ Yes' if conviction_exists else '❌ No'}")
        else:
            print("  ❌ Failed to add column")
            conn.close()
            sys.exit(1)
    else:
        print()
        print("  ℹ️  Column already exists, no fix needed for schema.")

    print()
    print("-" * 70)
    print("STEP 5: Test the exact failing query")
    print("-" * 70)

    query_works = test_query(conn)
    print(f"  Query test: {'✅ PASSED' if query_works else '❌ FAILED'}")

    print()
    print("-" * 70)
    print("STEP 6: Data summary")
    print("-" * 70)

    portfolio_count = get_portfolio_count(conn)
    holdings_count = get_holdings_count(conn)
    print(f"  Saved portfolios: {portfolio_count}")
    print(f"  Saved holdings: {holdings_count}")

    print()
    print("=" * 70)
    if query_works:
        print("✅ SUCCESS! The fix has been applied.")
        print()
        print("You can now:")
        print("  1. View Portfolio Holdings - should work")
        print("  2. Run Backtest - should work")
        print()
        print("No need to restart - just refresh the page and try again!")
    else:
        print("❌ There may still be an issue. Check the errors above.")
    print("=" * 70)

    conn.close()


if __name__ == "__main__":
    main()