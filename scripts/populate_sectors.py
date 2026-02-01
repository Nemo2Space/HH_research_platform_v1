"""
Alpha Platform - Populate Sector Data

Fetches sector information from yfinance and updates fundamentals table.

Usage:
    python scripts/populate_sectors.py
    python scripts/populate_sectors.py --ticker AAPL
    python scripts/populate_sectors.py --missing-only
"""

import os
import sys
import argparse
import time

# Setup path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv

load_dotenv()

import yfinance as yf
from src.db.connection import get_connection
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Sector mapping to normalize names
SECTOR_MAPPING = {
    'technology': 'Technology',
    'financial services': 'Financial',
    'financials': 'Financial',
    'healthcare': 'Healthcare',
    'health care': 'Healthcare',
    'consumer cyclical': 'Consumer Cyclical',
    'consumer discretionary': 'Consumer Cyclical',
    'consumer defensive': 'Consumer Defensive',
    'consumer staples': 'Consumer Defensive',
    'communication services': 'Communication Services',
    'industrials': 'Industrials',
    'energy': 'Energy',
    'utilities': 'Utilities',
    'real estate': 'Real Estate',
    'basic materials': 'Basic Materials',
    'materials': 'Basic Materials',
}


def normalize_sector(sector: str) -> str:
    """Normalize sector name to standard format."""
    if not sector:
        return None
    sector_lower = sector.lower().strip()
    return SECTOR_MAPPING.get(sector_lower, sector.title())


def get_sector_from_yfinance(ticker: str) -> dict:
    """
    Fetch sector and industry from yfinance.

    Returns:
        dict with 'sector' and 'industry', or empty dict on failure
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        sector = info.get('sector')
        industry = info.get('industry')

        if sector:
            return {
                'sector': normalize_sector(sector),
                'industry': industry
            }
        return {}

    except Exception as e:
        logger.debug(f"Error fetching {ticker}: {e}")
        return {}


def update_ticker_sector(ticker: str) -> bool:
    """
    Update sector for a single ticker in fundamentals table.

    Returns:
        True if updated, False otherwise
    """
    data = get_sector_from_yfinance(ticker)

    if not data.get('sector'):
        logger.debug(f"No sector found for {ticker}")
        return False

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Update all rows for this ticker
                cur.execute("""
                            UPDATE fundamentals
                            SET sector = %s
                            WHERE ticker = %s
                            """, (data['sector'], ticker))

                rows_updated = cur.rowcount
                conn.commit()

                if rows_updated > 0:
                    logger.info(f"{ticker}: {data['sector']}")
                    return True
                else:
                    # No existing row, try to insert
                    logger.debug(f"No fundamentals row for {ticker}")
                    return False

    except Exception as e:
        logger.error(f"Database error for {ticker}: {e}")
        return False


def populate_all_sectors(missing_only: bool = True):
    """
    Populate sectors for all tickers in fundamentals table.

    Args:
        missing_only: If True, only update tickers with NULL sector
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                if missing_only:
                    cur.execute("""
                                SELECT DISTINCT ticker
                                FROM fundamentals
                                WHERE sector IS NULL
                                   OR sector = ''
                                ORDER BY ticker
                                """)
                else:
                    cur.execute("""
                                SELECT DISTINCT ticker
                                FROM fundamentals
                                ORDER BY ticker
                                """)

                tickers = [row[0] for row in cur.fetchall()]

        if not tickers:
            print("No tickers to update")
            return

        print(f"Updating sectors for {len(tickers)} tickers...")

        updated = 0
        failed = 0

        for i, ticker in enumerate(tickers):
            # Progress
            if (i + 1) % 20 == 0:
                print(f"Progress: {i + 1}/{len(tickers)} ({updated} updated)")

            if update_ticker_sector(ticker):
                updated += 1
            else:
                failed += 1

            # Rate limiting - yfinance can be throttled
            time.sleep(0.2)

        print(f"\nComplete: {updated} updated, {failed} failed/skipped")

    except Exception as e:
        logger.error(f"Error: {e}")


def show_sector_stats():
    """Show current sector distribution."""
    print("\n" + "=" * 60)
    print("SECTOR DISTRIBUTION IN FUNDAMENTALS")
    print("=" * 60)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                            SELECT COALESCE(sector, 'NULL/Missing') as sector,
                                   COUNT(DISTINCT ticker)           as ticker_count
                            FROM fundamentals
                            GROUP BY sector
                            ORDER BY ticker_count DESC
                            """)

                print(f"{'Sector':<30} {'Tickers':>10}")
                print("-" * 42)
                for row in cur.fetchall():
                    print(f"{row[0]:<30} {row[1]:>10}")

                # Total
                cur.execute("SELECT COUNT(DISTINCT ticker) FROM fundamentals")
                total = cur.fetchone()[0]
                print("-" * 42)
                print(f"{'TOTAL':<30} {total:>10}")

    except Exception as e:
        print(f"Error: {e}")


def populate_historical_scores_sectors():
    """
    Also update sector in historical_scores table from fundamentals.
    """
    print("\nUpdating historical_scores sectors from fundamentals...")

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                            UPDATE historical_scores h
                            SET sector = f.sector FROM (
                        SELECT DISTINCT ON (ticker) ticker, sector
                        FROM fundamentals
                        WHERE sector IS NOT NULL AND sector != ''
                        ORDER BY ticker, date DESC
                    ) f
                            WHERE h.ticker = f.ticker
                              AND (h.sector IS NULL
                               OR h.sector = ''
                               OR h.sector = '0.0')
                            """)

                rows_updated = cur.rowcount
                conn.commit()

                print(f"Updated {rows_updated} rows in historical_scores")

    except Exception as e:
        logger.error(f"Error updating historical_scores: {e}")


def main():
    parser = argparse.ArgumentParser(description="Populate sector data from yfinance")
    parser.add_argument("--ticker", help="Update single ticker")
    parser.add_argument("--missing-only", action="store_true", default=True,
                        help="Only update tickers with missing sector (default)")
    parser.add_argument("--all", action="store_true",
                        help="Update all tickers, even those with existing sector")
    parser.add_argument("--stats", action="store_true", help="Show sector statistics")
    parser.add_argument("--sync-historical", action="store_true",
                        help="Sync sectors to historical_scores table")

    args = parser.parse_args()

    # First, ensure sector column exists
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                            ALTER TABLE fundamentals
                                ADD COLUMN IF NOT EXISTS sector VARCHAR (50)
                            """)
                conn.commit()
                logger.info("Ensured sector column exists")
    except Exception as e:
        logger.error(f"Could not add sector column: {e}")

    if args.stats:
        show_sector_stats()
    elif args.ticker:
        if update_ticker_sector(args.ticker):
            print(f"Updated {args.ticker}")
        else:
            print(f"Failed to update {args.ticker}")
    elif args.sync_historical:
        populate_historical_scores_sectors()
    else:
        missing_only = not args.all
        populate_all_sectors(missing_only=missing_only)

        # Also sync to historical_scores
        populate_historical_scores_sectors()

        # Show final stats
        show_sector_stats()


if __name__ == "__main__":
    main()
