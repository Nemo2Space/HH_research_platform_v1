"""
Alpha Platform - Populate Earnings & Ex-Dividend Dates

Fetches earnings_date and ex_dividend_date from Yahoo Finance.

Usage:
    python scripts/populate_dates.py
    python scripts/populate_dates.py --ticker AAPL
    python scripts/populate_dates.py --stats
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Setup path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv

load_dotenv()

import yfinance as yf
from src.db.connection import get_connection
from src.utils.logging import get_logger

logger = get_logger(__name__)


def ensure_columns_exist():
    """Add earnings_date and ex_dividend_date columns if they don't exist."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Add earnings_date column
                cur.execute("""
                            ALTER TABLE fundamentals
                                ADD COLUMN IF NOT EXISTS earnings_date DATE
                            """)

                # Add ex_dividend_date column
                cur.execute("""
                            ALTER TABLE fundamentals
                                ADD COLUMN IF NOT EXISTS ex_dividend_date DATE
                            """)

                conn.commit()
                logger.info("Ensured earnings_date and ex_dividend_date columns exist")
                return True
    except Exception as e:
        logger.error(f"Error adding columns: {e}")
        return False


def get_dates_from_yfinance(ticker: str) -> dict:
    """
    Fetch earnings_date and ex_dividend_date from Yahoo Finance.

    Returns:
        dict with 'earnings_date' and 'ex_dividend_date' (as date strings or None)
    """
    result = {
        'earnings_date': None,
        'ex_dividend_date': None,
        'dividend_yield': None
    }

    try:
        stock = yf.Ticker(ticker)

        # Method 1: Try calendar data
        if hasattr(stock, 'calendar') and stock.calendar is not None:
            calendar = stock.calendar

            # Get earnings date
            if 'Earnings Date' in calendar and calendar['Earnings Date'] is not None:
                earnings_dates = calendar['Earnings Date']
                if isinstance(earnings_dates, list) and len(earnings_dates) > 0:
                    result['earnings_date'] = earnings_dates[0].strftime('%Y-%m-%d')
                elif hasattr(earnings_dates, 'strftime'):
                    result['earnings_date'] = earnings_dates.strftime('%Y-%m-%d')

            # Get ex-dividend date from calendar
            if 'Ex-Dividend Date' in calendar and calendar['Ex-Dividend Date'] is not None:
                ex_div = calendar['Ex-Dividend Date']
                if hasattr(ex_div, 'strftime'):
                    result['ex_dividend_date'] = ex_div.strftime('%Y-%m-%d')

        # Method 2: Try info dict for additional data
        if hasattr(stock, 'info') and stock.info:
            info = stock.info

            # Get ex-dividend date if not found in calendar
            if not result['ex_dividend_date'] and 'exDividendDate' in info:
                ex_div_ts = info['exDividendDate']
                if ex_div_ts:
                    try:
                        result['ex_dividend_date'] = datetime.fromtimestamp(ex_div_ts).strftime('%Y-%m-%d')
                    except:
                        pass

            # Get dividend yield
            if 'dividendYield' in info and info['dividendYield']:
                result['dividend_yield'] = info['dividendYield'] * 100  # Convert to percentage

        # Method 3: Try dividends history for ex-dividend date
        if not result['ex_dividend_date']:
            try:
                if hasattr(stock, 'dividends') and not stock.dividends.empty:
                    # Get most recent dividend date
                    result['ex_dividend_date'] = stock.dividends.index[-1].strftime('%Y-%m-%d')
            except:
                pass

        return result

    except Exception as e:
        logger.debug(f"Error fetching dates for {ticker}: {e}")
        return result


def update_ticker_dates(ticker: str) -> bool:
    """
    Update earnings_date and ex_dividend_date for a single ticker.

    Returns:
        True if updated, False otherwise
    """
    data = get_dates_from_yfinance(ticker)

    if not data['earnings_date'] and not data['ex_dividend_date']:
        logger.debug(f"No dates found for {ticker}")
        return False

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Build dynamic update query
                updates = []
                params = []

                if data['earnings_date']:
                    updates.append("earnings_date = %s")
                    params.append(data['earnings_date'])

                if data['ex_dividend_date']:
                    updates.append("ex_dividend_date = %s")
                    params.append(data['ex_dividend_date'])

                if not updates:
                    return False

                params.append(ticker)

                query = f"""
                    UPDATE fundamentals 
                    SET {', '.join(updates)}
                    WHERE ticker = %s
                """

                cur.execute(query, params)
                rows_updated = cur.rowcount
                conn.commit()

                if rows_updated > 0:
                    earnings = data['earnings_date'] or 'N/A'
                    ex_div = data['ex_dividend_date'] or 'N/A'
                    logger.info(f"{ticker}: Earnings={earnings}, Ex-Div={ex_div}")
                    return True
                else:
                    logger.debug(f"No fundamentals row for {ticker}")
                    return False

    except Exception as e:
        logger.error(f"Database error for {ticker}: {e}")
        return False


def populate_all_dates(missing_only: bool = True):
    """
    Populate dates for all tickers in fundamentals table.

    Args:
        missing_only: If True, only update tickers with NULL dates
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                if missing_only:
                    cur.execute("""
                                SELECT DISTINCT ticker
                                FROM fundamentals
                                WHERE earnings_date IS NULL
                                   OR ex_dividend_date IS NULL
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

        print(f"Updating dates for {len(tickers)} tickers...")

        updated = 0
        failed = 0

        for i, ticker in enumerate(tickers):
            # Progress
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(tickers)} ({updated} updated)")

            if update_ticker_dates(ticker):
                updated += 1
            else:
                failed += 1

            # Rate limiting - yfinance can be throttled
            time.sleep(0.3)

        print(f"\nComplete: {updated} updated, {failed} failed/skipped")

    except Exception as e:
        logger.error(f"Error: {e}")


def show_stats():
    """Show current dates statistics."""
    print("\n" + "=" * 70)
    print("EARNINGS & EX-DIVIDEND DATE STATISTICS")
    print("=" * 70)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Count stats
                cur.execute("""
                            SELECT COUNT(DISTINCT ticker)                                                 as total,
                                   COUNT(DISTINCT CASE WHEN earnings_date IS NOT NULL THEN ticker END)    as has_earnings,
                                   COUNT(DISTINCT CASE WHEN ex_dividend_date IS NOT NULL THEN ticker END) as has_ex_div
                            FROM fundamentals
                            """)
                row = cur.fetchone()
                print(f"Total Tickers: {row[0]}")
                print(f"With Earnings Date: {row[1]}")
                print(f"With Ex-Dividend Date: {row[2]}")

                # Upcoming earnings
                print("\n" + "-" * 40)
                print("UPCOMING EARNINGS (next 30 days):")
                print("-" * 40)
                cur.execute("""
                            SELECT DISTINCT
                            ON (ticker) ticker, earnings_date
                            FROM fundamentals
                            WHERE earnings_date >= CURRENT_DATE
                              AND earnings_date <= CURRENT_DATE + INTERVAL '30 days'
                            ORDER BY ticker, date DESC
                            """)
                rows = cur.fetchall()
                if rows:
                    # Sort by date
                    rows = sorted(rows, key=lambda x: x[1])
                    for ticker, date in rows:
                        print(f"  {ticker:<8} {date}")
                else:
                    print("  No upcoming earnings in next 30 days")

                # Upcoming ex-dividend
                print("\n" + "-" * 40)
                print("UPCOMING EX-DIVIDEND (next 30 days):")
                print("-" * 40)
                cur.execute("""
                            SELECT DISTINCT
                            ON (ticker) ticker, ex_dividend_date
                            FROM fundamentals
                            WHERE ex_dividend_date >= CURRENT_DATE
                              AND ex_dividend_date <= CURRENT_DATE + INTERVAL '30 days'
                            ORDER BY ticker, date DESC
                            """)
                rows = cur.fetchall()
                if rows:
                    rows = sorted(rows, key=lambda x: x[1])
                    for ticker, date in rows:
                        print(f"  {ticker:<8} {date}")
                else:
                    print("  No upcoming ex-dividend dates in next 30 days")

    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Populate earnings and ex-dividend dates")
    parser.add_argument("--ticker", help="Update single ticker")
    parser.add_argument("--missing-only", action="store_true", default=True,
                        help="Only update tickers with missing dates (default)")
    parser.add_argument("--all", action="store_true",
                        help="Update all tickers")
    parser.add_argument("--stats", action="store_true", help="Show statistics")

    args = parser.parse_args()

    # Ensure columns exist
    if not ensure_columns_exist():
        print("Failed to ensure columns exist")
        return

    if args.stats:
        show_stats()
    elif args.ticker:
        if update_ticker_dates(args.ticker):
            print(f"Updated {args.ticker}")
        else:
            print(f"No dates found for {args.ticker}")
    else:
        missing_only = not args.all
        populate_all_dates(missing_only=missing_only)
        show_stats()


if __name__ == "__main__":
    main()