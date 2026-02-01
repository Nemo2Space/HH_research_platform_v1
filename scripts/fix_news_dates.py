"""
Fix News Dates Migration Script

This script updates existing news articles in the database that have
unparsed date strings (like "Tue, 17 Dec 2024 09:05:23 GMT") and converts
them to proper ISO format timestamps.

Run this once after deploying the updated news.py to fix historical data.

Usage:
    python scripts/fix_news_dates.py
"""

import os
import sys
from datetime import datetime
from typing import Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load .env file BEFORE any other imports
from dotenv import load_dotenv
env_path = os.path.join(project_root, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"Loaded .env from: {env_path}")
else:
    print(f"WARNING: .env file not found at {env_path}")

# For robust date parsing
try:
    from dateutil import parser as date_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    date_parser = None
    DATEUTIL_AVAILABLE = False
    print("WARNING: python-dateutil not installed. Some dates may not parse.")
    print("Install with: pip install python-dateutil")

import pandas as pd


def parse_news_date(date_value) -> Optional[str]:
    """
    Parse various date formats from news sources and return ISO format string.
    """
    if date_value is None or date_value == '' or pd.isna(date_value):
        return None

    try:
        dt = None

        # Already a datetime
        if isinstance(date_value, datetime):
            dt = date_value
        elif isinstance(date_value, pd.Timestamp):
            dt = date_value.to_pydatetime()
        else:
            date_str = str(date_value).strip()

            # Check if already ISO format
            if 'T' in date_str or (len(date_str) == 10 and date_str[4] == '-'):
                try:
                    dt = pd.to_datetime(date_str)
                    if isinstance(dt, pd.Timestamp):
                        dt = dt.to_pydatetime()
                except:
                    pass

            # Try dateutil (handles "Tue, 17 Dec 2024" etc)
            if dt is None and DATEUTIL_AVAILABLE and date_parser:
                try:
                    dt = date_parser.parse(date_str, fuzzy=True)
                except:
                    pass

            # Manual parsing for common formats
            if dt is None:
                formats = [
                    "%a, %d %b %Y %H:%M:%S %Z",
                    "%a, %d %b %Y %H:%M:%S %z",
                    "%a, %d %b %Y %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S%z",
                    "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d",
                    "%d %b %Y",
                    "%B %d, %Y",
                ]
                for fmt in formats:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        break
                    except:
                        continue

        if dt is not None:
            # Remove timezone and return ISO format
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            return dt.isoformat()

    except Exception as e:
        pass

    return None


def fix_news_dates():
    """
    Find and fix news articles with unparsed date strings.
    """
    try:
        from src.db.connection import get_engine, get_connection
    except ImportError as e:
        print(f"ERROR: Could not import database connection: {e}")
        print("Make sure you're running from the project root directory.")
        return

    # Verify DB connection
    print("\nConnecting to database...")
    try:
        from sqlalchemy import text
        engine = get_engine()
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("Database connection successful!")
    except Exception as e:
        print(f"ERROR: Database connection failed: {e}")
        print("\nCheck your .env file has these variables:")
        print("  POSTGRES_HOST=localhost")
        print("  POSTGRES_PORT=5432")
        print("  POSTGRES_DB=alpha_platform")
        print("  POSTGRES_USER=alpha")
        print("  POSTGRES_PASSWORD=your_password")
        return

    # Get articles where published_at looks like an unparsed string
    # (contains day name like "Mon", "Tue", etc)
    print("\nSearching for articles with unparsed dates...")

    query = """
        SELECT id, published_at, headline 
        FROM news_articles 
        WHERE published_at IS NOT NULL 
        AND (
            published_at::text LIKE '%Mon,%'
            OR published_at::text LIKE '%Tue,%'
            OR published_at::text LIKE '%Wed,%'
            OR published_at::text LIKE '%Thu,%'
            OR published_at::text LIKE '%Fri,%'
            OR published_at::text LIKE '%Sat,%'
            OR published_at::text LIKE '%Sun,%'
        )
        LIMIT 1000
    """

    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        # If the LIKE query fails (published_at is timestamp type), try a different approach
        print(f"Note: No text-format dates found (query returned: {e})")
        print("Checking for any articles that might need date parsing...")

        # Get recent articles and check if any dates seem weird
        query = """
            SELECT id, published_at, headline 
            FROM news_articles 
            WHERE published_at IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 100
        """
        df = pd.read_sql(query, engine)

        if not df.empty:
            # Check a sample
            sample = df['published_at'].iloc[0]
            print(f"Sample published_at value: {sample} (type: {type(sample).__name__})")

            if isinstance(sample, (datetime, pd.Timestamp)):
                print("Dates are already in proper timestamp format. No migration needed!")
                return

    if df.empty:
        print("No articles found that need date fixing.")
        return

    print(f"Found {len(df)} articles to check...")

    fixed_count = 0
    skipped_count = 0
    error_count = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                article_id = row['id']
                old_date = row['published_at']

                # Check if it needs parsing (contains weekday name)
                old_date_str = str(old_date) if old_date else ''
                weekdays = ['Mon,', 'Tue,', 'Wed,', 'Thu,', 'Fri,', 'Sat,', 'Sun,']
                needs_fix = any(day in old_date_str for day in weekdays)

                if not needs_fix:
                    skipped_count += 1
                    continue

                # Parse the date
                new_date = parse_news_date(old_date)

                if new_date:
                    try:
                        cur.execute("""
                            UPDATE news_articles 
                            SET published_at = %s 
                            WHERE id = %s
                        """, (new_date, article_id))
                        fixed_count += 1

                        if fixed_count % 100 == 0:
                            print(f"  Fixed {fixed_count} articles...")
                            conn.commit()
                    except Exception as e:
                        error_count += 1
                        print(f"  Error updating article {article_id}: {e}")
                else:
                    error_count += 1
                    print(f"  Could not parse date: '{old_date}' for article {article_id}")

            conn.commit()

    print(f"\n{'='*60}")
    print("Migration Complete!")
    print(f"{'='*60}")
    print(f"  Fixed:   {fixed_count} articles")
    print(f"  Skipped: {skipped_count} articles (already valid)")
    print(f"  Errors:  {error_count} articles")


if __name__ == "__main__":
    print("=" * 60)
    print("News Date Fix Migration")
    print("=" * 60)
    print()

    fix_news_dates()