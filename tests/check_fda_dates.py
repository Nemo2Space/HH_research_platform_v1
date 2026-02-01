"""
Clean FDA Calendar - Remove Expired Events
==========================================
CRITICAL: This script REMOVES expired events instead of fabricating future dates.

DO NOT shift dates forward - this creates fake data.
Instead, delete expired events and refresh from real sources.

Run: python check_fda_dates_fixed.py
"""

import sys
sys.path.insert(0, '..')

def get_db_connection():
    try:
        from src.db.connection import get_connection
        cm = get_connection()
        return cm.__enter__()
    except:
        import psycopg2
        return psycopg2.connect(
            host="localhost", port=5432, dbname="alpha_platform",
            user="alpha", password="alpha_secure_2024"
        )

print("=" * 80)
print("CLEANING FDA CALENDAR - REMOVING EXPIRED EVENTS")
print("=" * 80)

conn = get_db_connection()

# Check before
print("\n1. Current FDA calendar status:")
with conn.cursor() as cur:
    cur.execute("""
        SELECT COUNT(*), MIN(expected_date), MAX(expected_date)
        FROM fda_calendar
    """)
    row = cur.fetchone()
    print(f"   Total events: {row[0]}")
    print(f"   Date range: {row[1]} to {row[2]}")

    cur.execute("SELECT COUNT(*) FROM fda_calendar WHERE expected_date >= CURRENT_DATE")
    future_count = cur.fetchone()[0]
    print(f"   Future events (>= today): {future_count}")

    cur.execute("SELECT COUNT(*) FROM fda_calendar WHERE expected_date < CURRENT_DATE")
    expired_count = cur.fetchone()[0]
    print(f"   Expired events (< today): {expired_count}")

# Delete expired events - DO NOT FABRICATE
print("\n2. Removing expired events (no fabrication)...")
with conn.cursor() as cur:
    cur.execute("""
        DELETE FROM fda_calendar
        WHERE expected_date < CURRENT_DATE
    """)
    deleted = cur.rowcount
    conn.commit()
    print(f"   Deleted {deleted} expired events")

# Check after
print("\n3. FDA calendar after cleanup:")
with conn.cursor() as cur:
    cur.execute("""
        SELECT COUNT(*), MIN(expected_date), MAX(expected_date)
        FROM fda_calendar
    """)
    row = cur.fetchone()
    print(f"   Total events: {row[0]}")
    if row[0] > 0:
        print(f"   Date range: {row[1]} to {row[2]}")

    cur.execute("SELECT COUNT(*) FROM fda_calendar WHERE expected_date >= CURRENT_DATE")
    future_count = cur.fetchone()[0]
    print(f"   Future events (>= today): {future_count}")

# Show upcoming catalysts
if future_count > 0:
    print("\n4. Upcoming FDA Catalysts (next 15):")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT ticker, drug_name, expected_date, catalyst_type, priority
            FROM fda_calendar
            WHERE expected_date >= CURRENT_DATE
            ORDER BY expected_date
            LIMIT 15
        """)
        rows = cur.fetchall()
        for row in rows:
            print(f"   {row[0]}: {row[1]} - {row[2]} ({row[3]}) [{row[4]}]")
else:
    print("\n⚠️  WARNING: No future FDA events in database!")
    print("   You need to refresh from a real data source.")
    print("   DO NOT fabricate dates by adding +1 year to old data.")

conn.close()

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print("\nNEXT STEPS:")
print("1. If you have zero future events, refresh from a legitimate data source")
print("2. Never shift dates forward - that's data fabrication")
print("3. Update your data pipeline to fetch fresh FDA calendar data")
print("=" * 80)