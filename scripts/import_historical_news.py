"""
Alpha Platform - Historical News Import

Imports news CSV files with pattern: _YYYY__MM__DD__TICKER_news_data.csv

CSV Structure:
- Date: Publication date (various formats)
- Title: News headline
- Source: News source

Usage:
    # Import single file
    python scripts/import_historical_news.py --file "path/to/_2025__12__10__CRWD_news_data.csv"

    # Import all files from folder
    python scripts/import_historical_news.py --dir "path/to/news_folder"

    # Analyze news sentiment accuracy
    python scripts/import_historical_news.py --analyze
"""

import os
import sys
import re
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Setup path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.db.connection import get_connection
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_filename(filepath: str) -> dict:
    """
    Parse filename to extract date and ticker.

    Supported patterns (in order of priority):
    - _YYYY__MM__DD__TICKER_news_data.csv  (e.g., _2025__12__10__CRWD_news_data.csv)
    - (YYYY, M, D)_TICKER_news_data.csv    (e.g., (2025, 9, 9)_XOM_news_data.csv)
    - TICKER_news_YYYY-MM-DD.csv
    - YYYY-MM-DD_TICKER_news.csv

    Fallback: Try to extract ticker from any filename containing stock symbols

    Returns:
        dict with 'date' (may be None) and 'ticker', or None if completely unparseable
    """
    filename = os.path.basename(filepath)

    # Pattern 1: _YYYY__MM__DD__TICKER_news_data.csv
    pattern1 = r'_(\d{4})__(\d{2})__(\d{2})__([A-Z]+)_news_data\.csv'
    match1 = re.match(pattern1, filename)
    if match1:
        year, month, day, ticker = match1.groups()
        return {
            'date': f"{year}-{month.zfill(2)}-{day.zfill(2)}",
            'ticker': ticker
        }

    # Pattern 2: (YYYY, M, D)_TICKER_news_data.csv  (your format)
    pattern2 = r'\((\d{4}),\s*(\d{1,2}),\s*(\d{1,2})\)_([A-Z]+)_news_data\.csv'
    match2 = re.match(pattern2, filename)
    if match2:
        year, month, day, ticker = match2.groups()
        return {
            'date': f"{year}-{month.zfill(2)}-{day.zfill(2)}",
            'ticker': ticker
        }

    # Pattern 3: TICKER_news_YYYY-MM-DD.csv
    pattern3 = r'([A-Z]+)_news_(\d{4}-\d{2}-\d{2})\.csv'
    match3 = re.match(pattern3, filename)
    if match3:
        ticker, date = match3.groups()
        return {'date': date, 'ticker': ticker}

    # Pattern 4: YYYY-MM-DD_TICKER_news.csv
    pattern4 = r'(\d{4}-\d{2}-\d{2})_([A-Z]+)_news.*\.csv'
    match4 = re.match(pattern4, filename)
    if match4:
        date, ticker = match4.groups()
        return {'date': date, 'ticker': ticker}

    # Pattern 5: (YYYY, MM, DD)_TICKER_news_data.csv (with leading zeros)
    pattern5 = r'\((\d{4}),\s*(\d{1,2}),\s*(\d{1,2})\)_([A-Za-z]+)_news'
    match5 = re.search(pattern5, filename)
    if match5:
        year, month, day, ticker = match5.groups()
        return {
            'date': f"{year}-{month.zfill(2)}-{day.zfill(2)}",
            'ticker': ticker.upper()
        }

    # =========================================================================
    # FALLBACK: Try to extract ticker and date from any format
    # =========================================================================

    extracted_date = None
    extracted_ticker = None

    # Try to find a date anywhere in filename
    # Format: YYYY-MM-DD
    date_match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', filename)
    if date_match:
        y, m, d = date_match.groups()
        extracted_date = f"{y}-{m.zfill(2)}-{d.zfill(2)}"

    # Format: YYYY_MM_DD or YYYY__MM__DD
    if not extracted_date:
        date_match2 = re.search(r'(\d{4})_+(\d{1,2})_+(\d{1,2})', filename)
        if date_match2:
            y, m, d = date_match2.groups()
            extracted_date = f"{y}-{m.zfill(2)}-{d.zfill(2)}"

    # Format: (YYYY, M, D) - tuple style
    if not extracted_date:
        date_match3 = re.search(r'\((\d{4}),\s*(\d{1,2}),\s*(\d{1,2})\)', filename)
        if date_match3:
            y, m, d = date_match3.groups()
            extracted_date = f"{y}-{m.zfill(2)}-{d.zfill(2)}"

    # Try to find ticker (1-5 uppercase letters, common stock symbol pattern)
    # Look for pattern like _AAPL_ or _MSFT_ or )_AAPL_ etc
    ticker_match = re.search(r'[_\)\s]([A-Z]{1,5})[_\s\.]', filename.upper())
    if ticker_match:
        extracted_ticker = ticker_match.group(1)

    # Alternative: find standalone uppercase word that looks like ticker
    if not extracted_ticker:
        ticker_match2 = re.search(r'[^A-Z]([A-Z]{2,5})[^A-Z]', filename.upper())
        if ticker_match2:
            potential_ticker = ticker_match2.group(1)
            # Filter out common non-ticker words
            non_tickers = {'NEWS', 'DATA', 'CSV', 'FILE', 'STOCK', 'PRICE'}
            if potential_ticker not in non_tickers:
                extracted_ticker = potential_ticker

    # If we found at least a ticker, return it
    if extracted_ticker:
        if not extracted_date:
            logger.debug(f"Extracted ticker {extracted_ticker} from {filename} (no date found)")
        return {
            'date': extracted_date,  # May be None
            'ticker': extracted_ticker
        }

    # Completely failed
    logger.warning(f"Could not parse filename: {filename}")
    return None


def parse_date(date_str: str) -> datetime:
    """
    Parse various date formats from news CSV.

    Handles:
    - "Sun, 07 Dec 2025 14:02:00 GMT"
    - "2025-12-10T16:12:00"
    - "2025-12-01T07:03:44+00:00"
    - "2025-06-04T00:00:00+00:00"
    """
    if pd.isna(date_str) or not date_str:
        return None

    date_str = str(date_str).strip()

    # Try different formats
    formats = [
        "%a, %d %b %Y %H:%M:%S %Z",  # Sun, 07 Dec 2025 14:02:00 GMT
        "%Y-%m-%dT%H:%M:%S",          # 2025-12-10T16:12:00
        "%Y-%m-%dT%H:%M:%S%z",        # 2025-12-01T07:03:44+00:00
        "%Y-%m-%d %H:%M:%S",          # 2025-12-10 16:12:00
        "%Y-%m-%d",                    # 2025-12-10
    ]

    for fmt in formats:
        try:
            # Handle timezone offset manually for some formats
            clean_date = date_str.replace('+00:00', '').replace('Z', '')
            dt = datetime.strptime(clean_date[:19], fmt[:len(fmt.split('%')[0]) + 19] if 'T' in fmt else fmt)
            return dt
        except:
            continue

    # Last resort: try pandas
    try:
        return pd.to_datetime(date_str).to_pydatetime()
    except:
        logger.warning(f"Could not parse date: {date_str}")
        return None


def get_source_credibility(source: str) -> int:
    """Get credibility score for a news source."""
    if not source:
        return 5

    source_lower = source.lower()

    credibility_map = {
        'bloomberg': 10, 'wsj': 10, 'wall street journal': 10,
        'financial times': 10, 'ft.com': 10,
        'reuters': 10, 'associated press': 10, 'ap news': 10,
        'cnbc': 9, 'barrons': 9, "barron's": 9, 'investor business daily': 9,
        'forbes': 8, 'marketwatch': 8, 'morningstar': 8, 'the economist': 8,
        'seeking alpha': 7, 'seekingalpha': 7, 'yahoo finance': 7, 'yahoo': 7,
        'zacks': 7, 'investopedia': 7, 'tipranks': 7,
        'thestreet': 6, 'the street': 6, 'motley fool': 6, 'benzinga': 6,
        'business insider': 5, 'insider': 5, 'insider monkey': 5,
        'gurufocus': 5, 'simplywall': 5, '24/7 wall st': 5,
        'defense world': 4, 'quiver quantitative': 4,
        'reddit': 3, 'stocktwits': 3,
        'nasdaq': 6, 'money morning': 5,
    }

    for key, score in credibility_map.items():
        if key in source_lower:
            return score

    return 5  # Default


def import_news_csv(filepath: str) -> int:
    """
    Import a single news CSV file.

    Robust handling:
    - Skips files that can't be parsed
    - Tries to read CSV with different encodings
    - Skips empty/corrupt files
    - Continues on individual row errors

    Returns:
        Number of articles imported
    """
    # Parse filename for ticker and date
    file_info = parse_filename(filepath)
    if not file_info:
        logger.debug(f"Skipping unparseable filename: {filepath}")
        return 0

    ticker = file_info['ticker']
    score_date = file_info.get('date')  # May be None

    logger.info(f"Importing {os.path.basename(filepath)} - Ticker: {ticker}, Date: {score_date or 'from content'}")

    # Try to read CSV with different approaches
    df = None
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            if not df.empty:
                break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            logger.debug(f"Empty file: {filepath}")
            return 0
        except Exception as e:
            logger.debug(f"Read error with {encoding}: {e}")
            continue

    if df is None or df.empty:
        logger.debug(f"Could not read or empty: {filepath}")
        return 0

    # Normalize column names (handle variations)
    df.columns = [c.strip() for c in df.columns]

    # Map possible column names
    column_mappings = {
        'title': ['Title', 'title', 'Headline', 'headline', 'TITLE', 'Subject', 'subject'],
        'source': ['Source', 'source', 'SOURCE', 'Publisher', 'publisher', 'Provider'],
        'date': ['Date', 'date', 'DATE', 'Published', 'published', 'PublishedAt', 'Time', 'Timestamp']
    }

    # Find actual column names
    actual_cols = {}
    for key, possible_names in column_mappings.items():
        for name in possible_names:
            if name in df.columns:
                actual_cols[key] = name
                break

    # Must have at least title
    if 'title' not in actual_cols:
        logger.debug(f"No title column found in {filepath}. Columns: {df.columns.tolist()}")
        return 0

    # Process and insert
    imported = 0
    skipped = 0

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for idx, row in df.iterrows():
                    try:
                        # Get title
                        title = str(row[actual_cols['title']])[:500] if pd.notna(row[actual_cols['title']]) else None
                        if not title or len(title.strip()) < 5:
                            skipped += 1
                            continue

                        # Get source (optional)
                        source = 'Unknown'
                        if 'source' in actual_cols and pd.notna(row.get(actual_cols['source'])):
                            source = str(row[actual_cols['source']])[:100]

                        # Get date (optional - use from filename if not in row)
                        published_at = None
                        if 'date' in actual_cols and pd.notna(row.get(actual_cols['date'])):
                            published_at = parse_date(row[actual_cols['date']])

                        # Fallback to filename date
                        if not published_at and score_date:
                            try:
                                published_at = datetime.strptime(score_date, '%Y-%m-%d')
                            except:
                                pass

                        credibility = get_source_credibility(source)

                        # Insert into news_articles table
                        cur.execute("""
                            INSERT INTO news_articles (
                                ticker, headline, source, published_at, 
                                credibility_score, url, created_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
                            ON CONFLICT (ticker, headline) DO UPDATE SET
                                source = EXCLUDED.source,
                                published_at = COALESCE(EXCLUDED.published_at, news_articles.published_at),
                                credibility_score = EXCLUDED.credibility_score
                        """, (
                            ticker,
                            title.strip(),
                            source,
                            published_at,
                            credibility,
                            ''  # No URL in this CSV format
                                    ))
                        imported += 1

                    except Exception as e:
                        # Skip individual row errors
                        skipped += 1
                        logger.debug(f"Row error: {e}")
                        continue

                conn.commit()

        if imported > 0:
            logger.info(f"Imported {imported} articles for {ticker} (skipped {skipped})")
        return imported

    except Exception as e:
        logger.error(f"Database error for {filepath}: {e}")
        return 0


def import_news_directory(directory: str) -> dict:
    """
    Import all news CSV files from a directory.

    Robust handling:
    - Skips bad files
    - Shows progress
    - Continues on errors

    Returns:
        Dict with results per file
    """
    results = {}

    # Find all CSV files (various patterns)
    path = Path(directory)
    csv_files = list(path.glob("*.csv"))

    if not csv_files:
        logger.warning(f"No CSV files found in {directory}")
        return results

    logger.info(f"Found {len(csv_files)} CSV files to process")

    total_imported = 0
    successful_files = 0
    failed_files = 0

    for i, csv_file in enumerate(sorted(csv_files)):
        # Progress indicator
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Processing {i + 1}/{len(csv_files)}...")

        try:
            count = import_news_csv(str(csv_file))
            results[str(csv_file)] = count
            total_imported += count
            if count > 0:
                successful_files += 1
            else:
                failed_files += 1
        except Exception as e:
            logger.debug(f"Error processing {csv_file}: {e}")
            results[str(csv_file)] = 0
            failed_files += 1
            continue

    logger.info(f"Complete: {total_imported} articles from {successful_files} files ({failed_files} skipped)")
    return results


def analyze_news_sentiment_accuracy():
    """
    Analyze how news sentiment correlates with actual returns.

    Joins historical_scores (which has returns) with news_articles
    to see if positive news led to positive returns.
    """
    query = """
        WITH ticker_news AS (
            SELECT 
                ticker,
                DATE(published_at) as news_date,
                COUNT(*) as article_count,
                AVG(credibility_score) as avg_credibility
            FROM news_articles
            WHERE published_at IS NOT NULL
            GROUP BY ticker, DATE(published_at)
        ),
        scores_with_news AS (
            SELECT 
                h.ticker,
                h.score_date,
                h.sentiment,
                h.signal_type,
                h.return_5d,
                h.return_10d,
                h.signal_correct,
                tn.article_count,
                tn.avg_credibility
            FROM historical_scores h
            LEFT JOIN ticker_news tn 
                ON h.ticker = tn.ticker 
                AND tn.news_date BETWEEN h.score_date - INTERVAL '3 days' AND h.score_date
            WHERE h.return_10d IS NOT NULL
        )
        SELECT 
            CASE 
                WHEN article_count >= 10 THEN 'HIGH (10+)'
                WHEN article_count >= 5 THEN 'MEDIUM (5-9)'
                WHEN article_count >= 1 THEN 'LOW (1-4)'
                ELSE 'NO NEWS'
            END as news_coverage,
            COUNT(*) as count,
            ROUND(AVG(sentiment)::numeric, 1) as avg_sentiment,
            ROUND(AVG(return_5d)::numeric, 2) as avg_return_5d,
            ROUND(AVG(return_10d)::numeric, 2) as avg_return_10d,
            ROUND(100.0 * SUM(CASE WHEN signal_correct THEN 1 ELSE 0 END) / 
                  NULLIF(SUM(CASE WHEN signal_correct IS NOT NULL THEN 1 ELSE 0 END), 0), 1) as accuracy_pct
        FROM scores_with_news
        GROUP BY 
            CASE 
                WHEN article_count >= 10 THEN 'HIGH (10+)'
                WHEN article_count >= 5 THEN 'MEDIUM (5-9)'
                WHEN article_count >= 1 THEN 'LOW (1-4)'
                ELSE 'NO NEWS'
            END
        ORDER BY count DESC
    """

    print("\n" + "=" * 70)
    print("NEWS COVERAGE VS SIGNAL ACCURACY")
    print("=" * 70)

    try:
        with get_connection() as conn:
            df = pd.read_sql(query, conn)

        if df.empty:
            print("No data available for analysis")
            return

        print(df.to_string(index=False))

        # Additional analysis: Sentiment vs Returns
        print("\n" + "=" * 70)
        print("SENTIMENT SCORE VS ACTUAL RETURNS")
        print("=" * 70)

        query2 = """
            SELECT 
                CASE 
                    WHEN sentiment >= 70 THEN 'HIGH (70+)'
                    WHEN sentiment >= 50 THEN 'MEDIUM (50-69)'
                    WHEN sentiment >= 30 THEN 'LOW (30-49)'
                    ELSE 'VERY LOW (<30)'
                END as sentiment_bucket,
                COUNT(*) as count,
                ROUND(AVG(return_5d)::numeric, 2) as avg_return_5d,
                ROUND(AVG(return_10d)::numeric, 2) as avg_return_10d,
                ROUND(100.0 * SUM(CASE WHEN return_10d > 0 THEN 1 ELSE 0 END) / 
                      NULLIF(COUNT(*), 0), 1) as positive_pct
            FROM historical_scores
            WHERE return_10d IS NOT NULL
            AND sentiment IS NOT NULL
            GROUP BY 
                CASE 
                    WHEN sentiment >= 70 THEN 'HIGH (70+)'
                    WHEN sentiment >= 50 THEN 'MEDIUM (50-69)'
                    WHEN sentiment >= 30 THEN 'LOW (30-49)'
                    ELSE 'VERY LOW (<30)'
                END
            ORDER BY avg_return_10d DESC
        """

        with get_connection() as conn:
            df2 = pd.read_sql(query2, conn)

        if not df2.empty:
            print(df2.to_string(index=False))

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        print(f"Error: {e}")


def get_news_stats():
    """Show news database statistics."""
    print("\n" + "=" * 70)
    print("NEWS DATABASE STATISTICS")
    print("=" * 70)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Total articles
                cur.execute("SELECT COUNT(*) FROM news_articles")
                total = cur.fetchone()[0]
                print(f"Total Articles: {total:,}")

                # By source
                cur.execute("""
                    SELECT source, COUNT(*) as cnt 
                    FROM news_articles 
                    GROUP BY source 
                    ORDER BY cnt DESC 
                    LIMIT 10
                """)
                print("\nTop Sources:")
                for row in cur.fetchall():
                    print(f"  {row[0][:30]:<30} {row[1]:>6}")

                # By ticker
                cur.execute("""
                    SELECT ticker, COUNT(*) as cnt 
                    FROM news_articles 
                    GROUP BY ticker 
                    ORDER BY cnt DESC 
                    LIMIT 10
                """)
                print("\nTop Tickers:")
                for row in cur.fetchall():
                    print(f"  {row[0]:<10} {row[1]:>6}")

                # Date range
                cur.execute("""
                    SELECT MIN(published_at), MAX(published_at) 
                    FROM news_articles 
                    WHERE published_at IS NOT NULL
                """)
                row = cur.fetchone()
                if row[0] and row[1]:
                    print(f"\nDate Range: {row[0].strftime('%Y-%m-%d')} to {row[1].strftime('%Y-%m-%d')}")

    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Import historical news CSV files")
    parser.add_argument("--file", help="Import single CSV file")
    parser.add_argument("--dir", help="Import all CSV files from directory")
    parser.add_argument("--analyze", action="store_true", help="Analyze news sentiment accuracy")
    parser.add_argument("--stats", action="store_true", help="Show news database statistics")

    args = parser.parse_args()

    if args.file:
        count = import_news_csv(args.file)
        print(f"Imported {count} articles")

    elif args.dir:
        results = import_news_directory(args.dir)
        total = sum(results.values())
        print(f"\nTotal imported: {total} articles from {len(results)} files")

    elif args.analyze:
        analyze_news_sentiment_accuracy()

    elif args.stats:
        get_news_stats()

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/import_historical_news.py --file ./news/_2025__12__10__CRWD_news_data.csv")
        print("  python scripts/import_historical_news.py --dir ./news/")
        print("  python scripts/import_historical_news.py --stats")
        print("  python scripts/import_historical_news.py --analyze")


if __name__ == "__main__":
    main()
