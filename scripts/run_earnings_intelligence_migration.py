"""
Earnings Intelligence - Migration Runner

Run this script to create the earnings_intelligence tables.

Usage:
    python -m scripts.run_earnings_intelligence_migration

Or from Python:
    from scripts.run_earnings_intelligence_migration import run_migration, verify_migration
    run_migration()
    verify_migration()

Author: Alpha Research Platform
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables BEFORE importing db modules
from dotenv import load_dotenv
load_dotenv(project_root / '.env')

from src.db.connection import get_connection, get_engine
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_migration(sql_file: str = None) -> bool:
    """
    Run the earnings intelligence migration.

    Args:
        sql_file: Path to SQL file. Defaults to sql/create_earnings_intelligence.sql

    Returns:
        True if successful
    """
    if sql_file is None:
        sql_file = os.path.join(project_root, "sql", "create_earnings_intelligence.sql")

    if not os.path.exists(sql_file):
        logger.error(f"Migration file not found: {sql_file}")
        return False

    logger.info(f"Running migration from: {sql_file}")

    try:
        # Read SQL file with explicit UTF-8 encoding (fixes Windows issue)
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_content = f.read()

        # Execute migration
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_content)
            conn.commit()

        logger.info("Migration completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def verify_migration() -> dict:
    """
    Verify the migration was successful.

    Returns:
        Dict with verification results
    """
    results = {
        'tables_exist': False,
        'earnings_intelligence_columns': 0,
        'earnings_history_columns': 0,
        'ies_cache_columns': 0,
        'indexes_count': 0,
        'all_good': False
    }

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check tables exist
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('earnings_intelligence', 'earnings_history', 'ies_cache')
                """)
                tables = [row[0] for row in cur.fetchall()]
                results['tables_exist'] = len(tables) == 3

                # Count columns in earnings_intelligence
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM information_schema.columns 
                    WHERE table_name = 'earnings_intelligence'
                """)
                results['earnings_intelligence_columns'] = cur.fetchone()[0]

                # Count columns in earnings_history
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM information_schema.columns 
                    WHERE table_name = 'earnings_history'
                """)
                results['earnings_history_columns'] = cur.fetchone()[0]

                # Count columns in ies_cache
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM information_schema.columns 
                    WHERE table_name = 'ies_cache'
                """)
                results['ies_cache_columns'] = cur.fetchone()[0]

                # Count indexes
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM pg_indexes 
                    WHERE tablename IN ('earnings_intelligence', 'earnings_history', 'ies_cache')
                """)
                results['indexes_count'] = cur.fetchone()[0]

        # Check if all good
        results['all_good'] = (
            results['tables_exist'] and
            results['earnings_intelligence_columns'] >= 50 and  # We have ~55 columns
            results['earnings_history_columns'] >= 6 and
            results['ies_cache_columns'] >= 10 and
            results['indexes_count'] >= 10
        )

        return results

    except Exception as e:
        logger.error(f"Verification error: {e}")
        results['error'] = str(e)
        return results


def print_verification_report(results: dict):
    """Print a formatted verification report."""
    print("\n" + "=" * 60)
    print("EARNINGS INTELLIGENCE MIGRATION VERIFICATION")
    print("=" * 60)

    status = "[OK]" if results.get('all_good') else "[FAIL]"
    print(f"\n{status} Overall Status: {'PASSED' if results.get('all_good') else 'FAILED'}")

    tables_ok = "[OK]" if results.get('tables_exist') else "[FAIL]"
    print(f"\n{tables_ok} Tables Created: {'Yes (3/3)' if results.get('tables_exist') else 'Missing'}")
    print(f"   - earnings_intelligence: {results.get('earnings_intelligence_columns', 0)} columns")
    print(f"   - earnings_history: {results.get('earnings_history_columns', 0)} columns")
    print(f"   - ies_cache: {results.get('ies_cache_columns', 0)} columns")

    print(f"\n[INFO] Indexes Created: {results.get('indexes_count', 0)}")

    if results.get('error'):
        print(f"\n[WARN] Error: {results.get('error')}")

    print("\n" + "=" * 60)


def show_table_schema():
    """Display the earnings_intelligence table schema."""
    try:
        import pandas as pd

        query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'earnings_intelligence'
            ORDER BY ordinal_position
        """

        df = pd.read_sql(query, get_engine())

        print("\nearnings_intelligence Table Schema:")
        print("-" * 60)
        for _, row in df.iterrows():
            nullable = "NULL" if row['is_nullable'] == 'YES' else "NOT NULL"
            print(f"  {row['column_name']:<30} {row['data_type']:<15} {nullable}")

    except Exception as e:
        logger.error(f"Error showing schema: {e}")


def test_insert_and_query():
    """Test basic insert and query operations."""
    print("\n[TEST] Testing Insert & Query...")

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Insert test record
                cur.execute("""
                    INSERT INTO earnings_intelligence (
                        ticker, earnings_date, sector, ies, regime, data_quality
                    ) VALUES (
                        'TEST', CURRENT_DATE + INTERVAL '7 days', 'Technology', 
                        65.5, 'NORMAL', 'HIGH'
                    )
                    ON CONFLICT (ticker, earnings_date) DO UPDATE SET
                        ies = EXCLUDED.ies,
                        updated_at = NOW()
                    RETURNING id
                """)
                test_id = cur.fetchone()[0]

                # Query it back
                cur.execute("""
                    SELECT ticker, ies, regime, data_quality
                    FROM earnings_intelligence
                    WHERE id = %s
                """, (test_id,))
                row = cur.fetchone()

                # Clean up
                cur.execute("DELETE FROM earnings_intelligence WHERE ticker = 'TEST'")

            conn.commit()

        print(f"   [OK] Insert: Created record ID {test_id}")
        print(f"   [OK] Query: Retrieved {row}")
        print(f"   [OK] Delete: Cleaned up test record")
        print("   [OK] All CRUD operations working!")
        return True

    except Exception as e:
        print(f"   [FAIL] Test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Earnings Intelligence Migration")
    parser.add_argument('--verify-only', action='store_true', help="Only verify, don't run migration")
    parser.add_argument('--show-schema', action='store_true', help="Show table schema")
    parser.add_argument('--test', action='store_true', help="Run insert/query test")

    args = parser.parse_args()

    if args.verify_only:
        results = verify_migration()
        print_verification_report(results)
    elif args.show_schema:
        show_table_schema()
    elif args.test:
        test_insert_and_query()
    else:
        # Run migration
        success = run_migration()

        if success:
            # Verify
            results = verify_migration()
            print_verification_report(results)

            # Run test
            test_insert_and_query()