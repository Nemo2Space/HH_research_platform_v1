"""
Database Migration Runner

Run this to create missing tables and columns for Signal Hub.

Usage:
    python -m src.analytics.run_migration
"""

from src.db.connection import get_connection
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_migration():
    """Run all migrations."""

    print("\n" + "=" * 60)
    print("SIGNAL HUB DATABASE MIGRATION")
    print("=" * 60 + "\n")

    migrations = [
        ("Create earnings_calendar table", CREATE_EARNINGS_CALENDAR),
        ("Add sentiment_signal to screener_scores", ADD_SENTIMENT_SIGNAL),
        ("Add article_count to screener_scores", ADD_ARTICLE_COUNT),
        ("Add earnings columns to screener_scores", ADD_EARNINGS_COLUMNS),
        ("Add relevance_score to news_articles", ADD_RELEVANCE_SCORE),
        ("Insert NKE earnings data", INSERT_NKE_EARNINGS),
    ]

    success = 0
    failed = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            for name, sql in migrations:
                try:
                    print(f"Running: {name}...")
                    cur.execute(sql)
                    conn.commit()
                    print(f"   ✅ {name}")
                    success += 1
                except Exception as e:
                    conn.rollback()
                    error_msg = str(e)
                    if "already exists" in error_msg or "duplicate" in error_msg.lower():
                        print(f"   ⏭️ {name} (already done)")
                        success += 1
                    else:
                        print(f"   ❌ {name}: {error_msg[:80]}")
                        failed += 1

    print("\n" + "-" * 60)
    print(f"DONE: {success} successful, {failed} failed")
    print("-" * 60)

    # Verify
    print("\nVerifying...")
    verify_migration()


def verify_migration():
    """Verify migration was successful."""

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Check earnings_calendar
            try:
                cur.execute("SELECT COUNT(*) FROM earnings_calendar")
                count = cur.fetchone()[0]
                print(f"   ✅ earnings_calendar: {count} rows")
            except Exception as e:
                print(f"   ❌ earnings_calendar: {e}")

            # Check NKE earnings
            try:
                cur.execute("""
                            SELECT earnings_date, eps_actual, eps_surprise_pct, reaction_pct
                            FROM earnings_calendar
                            WHERE ticker = 'NKE'
                            ORDER BY earnings_date DESC LIMIT 3
                            """)
                rows = cur.fetchall()
                if rows:
                    print(f"   ✅ NKE earnings: {len(rows)} records")
                    for row in rows:
                        print(f"      {row[0]}: EPS ${row[1]}, surprise {row[2]}%, reaction {row[3]}%")
                else:
                    print("   ⚠️ NKE earnings: no data")
            except Exception as e:
                print(f"   ❌ NKE earnings: {e}")

            # Check screener_scores columns
            try:
                cur.execute("""
                            SELECT column_name
                            FROM information_schema.columns
                            WHERE table_name = 'screener_scores'
                              AND column_name IN ('sentiment_signal', 'article_count', 'earnings_score')
                            """)
                cols = [r[0] for r in cur.fetchall()]
                print(f"   ✅ screener_scores new columns: {cols}")
            except Exception as e:
                print(f"   ❌ screener_scores columns: {e}")


# SQL Statements
CREATE_EARNINGS_CALENDAR = """
                           CREATE TABLE IF NOT EXISTS earnings_calendar \
                           ( \
                               id \
                               SERIAL \
                               PRIMARY \
                               KEY, \
                               ticker \
                               VARCHAR \
                           ( \
                               10 \
                           ) NOT NULL,
                               earnings_date DATE NOT NULL,
                               earnings_time VARCHAR \
                           ( \
                               10 \
                           ),
                               eps_estimate DECIMAL \
                           ( \
                               10, \
                               4 \
                           ),
                               eps_actual DECIMAL \
                           ( \
                               10, \
                               4 \
                           ),
                               eps_surprise_pct DECIMAL \
                           ( \
                               10, \
                               4 \
                           ),
                               revenue_estimate BIGINT,
                               revenue_actual BIGINT,
                               revenue_surprise_pct DECIMAL \
                           ( \
                               10, \
                               4 \
                           ),
                               guidance_direction VARCHAR \
                           ( \
                               20 \
                           ),
                               price_before DECIMAL \
                           ( \
                               10, \
                               2 \
                           ),
                               price_after DECIMAL \
                           ( \
                               10, \
                               2 \
                           ),
                               reaction_pct DECIMAL \
                           ( \
                               10, \
                               4 \
                           ),
                               created_at TIMESTAMP DEFAULT NOW \
                           ( \
                           ),
                               updated_at TIMESTAMP DEFAULT NOW \
                           ( \
                           ),
                               UNIQUE \
                           ( \
                               ticker, \
                               earnings_date \
                           )
                               ) \
                           """

ADD_SENTIMENT_SIGNAL = """
                       ALTER TABLE screener_scores
                           ADD COLUMN IF NOT EXISTS sentiment_signal VARCHAR (20) \
                       """

ADD_ARTICLE_COUNT = """
                    ALTER TABLE screener_scores
                        ADD COLUMN IF NOT EXISTS article_count INTEGER DEFAULT 0 \
                    """

ADD_EARNINGS_COLUMNS = """
DO $$ 
BEGIN
    BEGIN
        ALTER TABLE screener_scores ADD COLUMN earnings_score INTEGER;
    EXCEPTION WHEN duplicate_column THEN NULL;
    END;
    BEGIN
        ALTER TABLE screener_scores ADD COLUMN earnings_signal VARCHAR(20);
    EXCEPTION WHEN duplicate_column THEN NULL;
    END;
    BEGIN
        ALTER TABLE screener_scores ADD COLUMN days_to_earnings INTEGER;
    EXCEPTION WHEN duplicate_column THEN NULL;
    END;
END $$
"""

ADD_RELEVANCE_SCORE = """
                      ALTER TABLE news_articles
                          ADD COLUMN IF NOT EXISTS relevance_score INTEGER \
                      """

INSERT_NKE_EARNINGS = """
                      INSERT INTO earnings_calendar (ticker, earnings_date, earnings_time, \
                                                     eps_estimate, eps_actual, eps_surprise_pct, \
                                                     price_before, reaction_pct, updated_at) \
                      VALUES ('NKE', '2025-12-18', 'AMC', 0.38, 0.53, 41.33, 65.69, -11.04, NOW()), \
                             ('NKE', '2025-09-30', 'AMC', 0.27, 0.49, 81.32, NULL, NULL, NOW()), \
                             ('NKE', '2025-06-26', 'AMC', 0.12, 0.14, 13.48, NULL, NULL, NOW()), \
                             ('NKE', '2025-03-20', 'AMC', 0.29, 0.54, 85.52, NULL, NULL, NOW()), \
                             ('NKE', '2026-03-19', 'AMC', 0.48, NULL, NULL, NULL, NULL, \
                              NOW()) ON CONFLICT (ticker, earnings_date) DO \
                      UPDATE SET
                          eps_estimate = EXCLUDED.eps_estimate, \
                          eps_actual = EXCLUDED.eps_actual, \
                          eps_surprise_pct = EXCLUDED.eps_surprise_pct, \
                          price_before = EXCLUDED.price_before, \
                          reaction_pct = EXCLUDED.reaction_pct, \
                          updated_at = NOW() \
                      """

if __name__ == "__main__":
    run_migration()