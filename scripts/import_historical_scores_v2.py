#!/usr/bin/env python3
"""
Alpha Platform - Historical Scores Import V3 (Ultra Robust)

Features:
1. Auto-detects and CREATES new columns in database when found in CSV
2. Per-field error handling - bad values become NULL, record still imports
3. Uses SAVEPOINT to avoid transaction abort on row errors
4. Dynamic schema sync with type inference
5. Handles all edge cases (overflow, type mismatch, etc.)

Usage:
    python import_historical_scores_v3.py --file path/to/scores.csv
    python import_historical_scores_v3.py --dir path/to/csv/folder
    python import_historical_scores_v3.py --stats

Location: scripts/import_historical_scores_v3.py
Author: Alpha Research Platform
"""

import os
import sys
import argparse
import glob
import re
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DB connection
import psycopg2
from contextlib import contextmanager


def get_engine():
    from sqlalchemy import create_engine
    return create_engine(
        f"postgresql://{os.getenv('POSTGRES_USER', 'alpha')}:{os.getenv('POSTGRES_PASSWORD', '')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'alpha_platform')}"
    )


@contextmanager
def get_connection():
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database=os.getenv('POSTGRES_DB', 'alpha_platform'),
        user=os.getenv('POSTGRES_USER', 'alpha'),
        password=os.getenv('POSTGRES_PASSWORD', ''),
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# =============================================================================
# COLUMN MAPPING - CSV column names -> DB column names
# =============================================================================

COLUMN_ALIASES = {
    # DB Column -> Possible CSV column names (case-insensitive)
    'ticker': ['ticker', 'symbol', 'stock', 'tickers', 'symbols'],
    'sentiment': ['sentiment', 'sentiments', 'sentiment_score', 'sent', 'news_sentiment'],
    'sector': ['sector', 'sectors', 'gics_sector', 'industry_sector'],
    'volume': ['volume', 'vol', 'avg_volume', 'trading_volume'],
    'fundamental_score': ['fundamentalscore', 'fundamental_score', 'fundamental', 'fund_score'],
    'growth_score': ['growthscore', 'growth_score', 'growth'],
    'dividend_score': ['dividendscore', 'dividend_score', 'dividend', 'div_score'],
    'counter': ['counter', 'count', 'signal_count'],
    'total_score': ['total', 'totalscore', 'total_score', 'composite_score', 'score'],
    'op_price': ['opprice', 'op_price', 'open_price', 'entry_price', 'curprice', 'cur_price', 'current_price',
                 'clprice', 'cl_price', 'close_price'],
    'target_avg_price': ['targetavgprice', 'target_avg_price', 'target_price', 'price_target'],
    'mkt_score': ['mktscore', 'mkt_score', 'market_score'],
    'gap_score': ['gaptype', 'gap_type', 'gap_score', 'gap'],
    'earn_date': ['earndate', 'earn_date', 'earnings_date', 'next_earnings'],
    'positive_ratings': ['positiveratings', 'positive_ratings', 'pos_ratings', 'buy_ratings'],
    'total_ratings': ['totalratings', 'total_ratings', 'analyst_count', 'num_analysts'],
    'exit_date': ['exitdate', 'exit_date', 'target_date'],
    'dividend_yield': ['dividendyield', 'dividend_yield', 'div_yield', 'yield'],
    'likelihood_score': ['likelihood', 'likelihoodactual', 'likelihoodpredicted', 'likelihood_score',
                         'likelihood_actual', 'likelihood_predicted', 'prob', 'probability'],
    'technical_score': ['technical_score', 'technicalscore', 'technical', 'tech_score'],
    'options_flow_score': ['options_flow_score', 'optionsflow', 'options_score', 'options'],
    'short_squeeze_score': ['short_squeeze_score', 'squeeze_score', 'short_interest'],
    'insider_signal': ['insider_signal', 'insider', 'insider_score'],
    'institutional_signal': ['institutional_signal', 'institutional', 'inst_signal'],
}

# DB column types - used for new column creation
COLUMN_TYPES = {
    'ticker': 'VARCHAR(20)',
    'sentiment': 'INT',
    'sector': 'VARCHAR(100)',
    'volume': 'BIGINT',
    'fundamental_score': 'INT',
    'growth_score': 'INT',
    'dividend_score': 'INT',
    'counter': 'INT',
    'total_score': 'NUMERIC(12,4)',
    'op_price': 'NUMERIC(12,4)',
    'target_avg_price': 'NUMERIC(12,4)',
    'mkt_score': 'NUMERIC(12,4)',
    'gap_score': 'VARCHAR(50)',
    'earn_date': 'DATE',
    'positive_ratings': 'INT',
    'total_ratings': 'INT',
    'exit_date': 'DATE',
    'dividend_yield': 'NUMERIC(12,4)',
    'likelihood_score': 'NUMERIC(20,4)',
    'technical_score': 'INT',
    'options_flow_score': 'INT',
    'short_squeeze_score': 'INT',
    'insider_signal': 'INT',
    'institutional_signal': 'INT',
    'signal_type': 'VARCHAR(20)',
    'source_file': 'VARCHAR(255)',
    'score_date': 'DATE',
}

# Default type for unknown columns
DEFAULT_COLUMN_TYPE = 'VARCHAR(255)'


# =============================================================================
# RESULT TRACKING
# =============================================================================

@dataclass
class ImportResult:
    filepath: str
    success: bool
    records_imported: int = 0
    records_updated: int = 0
    records_skipped: int = 0
    fields_nullified: int = 0
    new_columns_created: List[str] = field(default_factory=list)
    error_message: str = ""
    score_date: Optional[datetime] = None


@dataclass
class BatchImportResult:
    total_files: int = 0
    successful_files: int = 0
    skipped_files: int = 0
    total_records: int = 0
    total_fields_nullified: int = 0
    new_columns: set = field(default_factory=set)
    file_results: List[ImportResult] = field(default_factory=list)

    def add_result(self, result: ImportResult):
        self.file_results.append(result)
        self.total_files += 1
        if result.success:
            self.successful_files += 1
            self.total_records += result.records_imported
            self.total_fields_nullified += result.fields_nullified
            self.new_columns.update(result.new_columns_created)
        else:
            self.skipped_files += 1

    def print_summary(self):
        print("\n" + "=" * 70)
        print("IMPORT SUMMARY")
        print("=" * 70)
        print(f"Total files processed:    {self.total_files}")
        print(f"Successful imports:       {self.successful_files}")
        print(f"Skipped files:            {self.skipped_files}")
        print(f"Total records imported:   {self.total_records}")
        print(f"Fields nullified (bad):   {self.total_fields_nullified}")
        if self.new_columns:
            print(f"New columns created:      {', '.join(sorted(self.new_columns))}")


# =============================================================================
# SCHEMA MANAGEMENT
# =============================================================================

def get_existing_columns(conn) -> Dict[str, str]:
    """Get dict of column_name -> data_type for historical_scores."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'historical_scores'
        """)
        return {row[0]: row[1] for row in cur.fetchall()}


def ensure_table_exists():
    """Create base table if not exists."""
    sql = """
    CREATE TABLE IF NOT EXISTS historical_scores (
        id SERIAL PRIMARY KEY,
        score_date DATE NOT NULL,
        ticker VARCHAR(20) NOT NULL,
        sentiment INT,
        signal_type VARCHAR(20),
        source_file VARCHAR(255),
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(score_date, ticker)
    );
    CREATE INDEX IF NOT EXISTS idx_hist_scores_date ON historical_scores(score_date);
    CREATE INDEX IF NOT EXISTS idx_hist_scores_ticker ON historical_scores(ticker);
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    logger.info("Ensured historical_scores table exists")


def add_column_if_missing(conn, col_name: str, col_type: str = None) -> bool:
    """Add a column to the table if it doesn't exist. Returns True if added."""
    if col_type is None:
        col_type = COLUMN_TYPES.get(col_name, DEFAULT_COLUMN_TYPE)

    try:
        with conn.cursor() as cur:
            cur.execute(f"ALTER TABLE historical_scores ADD COLUMN IF NOT EXISTS {col_name} {col_type}")
        conn.commit()
        logger.info(f"  Added new column: {col_name} ({col_type})")
        return True
    except Exception as e:
        conn.rollback()
        logger.warning(f"  Could not add column {col_name}: {e}")
        return False


def infer_column_type(values: pd.Series) -> str:
    """Infer PostgreSQL type from pandas Series."""
    # Drop nulls for analysis
    non_null = values.dropna()
    if len(non_null) == 0:
        return DEFAULT_COLUMN_TYPE

    # Check if all numeric
    try:
        numeric_vals = pd.to_numeric(non_null, errors='coerce')
        if numeric_vals.notna().all():
            # Check if integers
            if (numeric_vals == numeric_vals.astype(int)).all():
                max_val = abs(numeric_vals).max()
                if max_val < 2147483647:
                    return 'INT'
                return 'BIGINT'
            return 'NUMERIC(20,4)'
    except:
        pass

    # Check if dates
    try:
        pd.to_datetime(non_null, errors='raise')
        return 'DATE'
    except:
        pass

    # Default to varchar
    max_len = non_null.astype(str).str.len().max()
    if max_len < 50:
        return 'VARCHAR(50)'
    elif max_len < 255:
        return 'VARCHAR(255)'
    return 'TEXT'


# =============================================================================
# VALUE CLEANING
# =============================================================================

def clean_value(value: Any, target_type: str = 'str') -> Any:
    """Clean and convert a value. Returns None on any error."""
    if pd.isna(value) or value is None:
        return None

    try:
        if target_type == 'int':
            return int(float(value))
        elif target_type == 'float':
            return float(value)
        elif target_type == 'date':
            if isinstance(value, datetime):
                return value.date()
            if isinstance(value, str):
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%b %d, %Y']:
                    try:
                        return datetime.strptime(value.strip(), fmt).date()
                    except ValueError:
                        continue
            return None
        else:  # str
            s = str(value).strip()
            return s if s and s.lower() != 'nan' else None
    except (ValueError, TypeError):
        return None


def clean_mkt_score(value) -> Optional[float]:
    """Parse mkt_score: 50, [50], [50,60] -> mean value."""
    if pd.isna(value) or value is None:
        return None
    s = str(value).strip().replace('[', '').replace(']', '').strip()
    if not s:
        return None
    try:
        values = [float(x.strip()) for x in s.split(',') if x.strip()]
        return sum(values) / len(values) if values else None
    except (ValueError, TypeError):
        return None


def classify_signal(sentiment: int, total_score: float = None) -> str:
    """Classify signal based on scores."""
    score = total_score if total_score is not None else sentiment
    if score is None:
        return 'UNKNOWN'
    if score >= 80:
        return 'STRONG_BUY'
    elif score >= 65:
        return 'BUY'
    elif score >= 50:
        return 'WEAK_BUY'
    elif score >= 35:
        return 'NEUTRAL'
    elif score >= 20:
        return 'WEAK_SELL'
    return 'SELL'


# =============================================================================
# COLUMN DETECTION
# =============================================================================

def normalize_col(col: str) -> str:
    """Normalize column name."""
    return col.lower().strip().replace(' ', '_').replace('-', '_')


def map_csv_to_db_columns(csv_columns: List[str]) -> Dict[str, str]:
    """
    Map CSV columns to DB columns.
    Returns dict: db_column -> csv_column
    """
    mapping = {}
    csv_normalized = {normalize_col(c): c for c in csv_columns}

    for db_col, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if normalize_col(alias) in csv_normalized:
                mapping[db_col] = csv_normalized[normalize_col(alias)]
                break

    # Also map any CSV columns that directly match DB columns
    for csv_col in csv_columns:
        norm = normalize_col(csv_col)
        if norm not in [normalize_col(m) for m in mapping.values()]:
            # This CSV column isn't mapped yet - use as-is for new columns
            mapping[norm] = csv_col

    return mapping


# =============================================================================
# DATE EXTRACTION
# =============================================================================

def parse_date_from_filename(filename: str) -> Optional[datetime]:
    """Extract date from filename."""
    basename = os.path.basename(filename)

    patterns = [
        r'^(\d{4})-(\d{1,2})-(\d{1,2})',  # 2024-09-17
        r'_(\d{4})__(\d{1,2})__(\d{1,2})__',  # _2025__12__10__
        r'\((\d{4}),\s*(\d{1,2}),\s*(\d{1,2})\)',  # (2025, 9, 9)
        r'(\d{4})_(\d{1,2})_(\d{1,2})',  # 2024_09_17
    ]

    for pattern in patterns:
        match = re.search(pattern, basename)
        if match:
            y, m, d = match.groups()
            try:
                return datetime(int(y), int(m), int(d))
            except ValueError:
                continue

    # Fallback to file mtime
    try:
        return datetime.fromtimestamp(os.path.getmtime(filename))
    except:
        return datetime.now()


# =============================================================================
# MAIN IMPORT FUNCTION
# =============================================================================

def import_csv(filepath: str, score_date: datetime = None) -> ImportResult:
    """Import a CSV with full error handling per field."""
    result = ImportResult(filepath=filepath, success=False)

    # Parse date
    if score_date is None:
        score_date = parse_date_from_filename(filepath)
    result.score_date = score_date

    # Read CSV
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            result.error_message = "Empty file"
            return result
    except Exception as e:
        result.error_message = f"Read error: {str(e)[:50]}"
        return result

    # Map columns
    col_mapping = map_csv_to_db_columns(list(df.columns))

    # Check mandatory
    if 'ticker' not in col_mapping:
        result.error_message = "Missing ticker column"
        return result
    if 'sentiment' not in col_mapping:
        result.error_message = "Missing sentiment column"
        return result

    source_file = os.path.basename(filepath)
    imported = 0
    updated = 0
    skipped = 0
    fields_nullified = 0
    new_columns = []

    with get_connection() as conn:
        # Get existing DB columns
        existing_cols = get_existing_columns(conn)

        # Check if we need to add any columns
        for db_col in col_mapping.keys():
            if db_col not in existing_cols and db_col not in ['ticker', 'sentiment']:
                # Infer type from data
                csv_col = col_mapping[db_col]
                if csv_col in df.columns:
                    col_type = COLUMN_TYPES.get(db_col, infer_column_type(df[csv_col]))
                    if add_column_if_missing(conn, db_col, col_type):
                        new_columns.append(db_col)
                        existing_cols[db_col] = col_type

        # Build dynamic INSERT
        db_columns = ['score_date', 'ticker', 'sentiment', 'signal_type', 'source_file']
        for db_col in col_mapping.keys():
            if db_col not in db_columns and db_col in existing_cols:
                db_columns.append(db_col)

        with conn.cursor() as cur:
            for idx, row in df.iterrows():
                # Extract ticker and sentiment (mandatory)
                ticker = clean_value(row[col_mapping['ticker']], 'str')
                sentiment = clean_value(row[col_mapping['sentiment']], 'int')

                if not ticker:
                    skipped += 1
                    continue

                ticker = ticker.upper().strip()

                # Build values dict with error handling per field
                values = {
                    'score_date': score_date.date() if score_date else None,
                    'ticker': ticker,
                    'sentiment': sentiment,
                    'signal_type': classify_signal(sentiment),
                    'source_file': source_file,
                }

                # Process each mapped column
                for db_col, csv_col in col_mapping.items():
                    if db_col in ['ticker', 'sentiment']:
                        continue
                    if db_col not in existing_cols:
                        continue
                    if csv_col not in df.columns:
                        continue

                    raw_val = row[csv_col]

                    # Special handling for mkt_score
                    if db_col == 'mkt_score':
                        values[db_col] = clean_mkt_score(raw_val)
                    # Determine target type from COLUMN_TYPES or existing schema
                    elif db_col in COLUMN_TYPES:
                        type_str = COLUMN_TYPES[db_col]
                        if 'INT' in type_str or 'BIGINT' in type_str:
                            values[db_col] = clean_value(raw_val, 'int')
                        elif 'NUMERIC' in type_str or 'FLOAT' in type_str or 'DOUBLE' in type_str:
                            values[db_col] = clean_value(raw_val, 'float')
                        elif 'DATE' in type_str:
                            values[db_col] = clean_value(raw_val, 'date')
                        else:
                            values[db_col] = clean_value(raw_val, 'str')
                    else:
                        # Default to string
                        values[db_col] = clean_value(raw_val, 'str')

                    # Count nullified fields
                    if values[db_col] is None and pd.notna(raw_val):
                        fields_nullified += 1

                # Build INSERT with only columns we have values for
                insert_cols = [c for c in db_columns if c in values]
                placeholders = ', '.join(['%s'] * len(insert_cols))
                col_list = ', '.join(insert_cols)

                # Build UPDATE clause
                update_parts = []
                for c in insert_cols:
                    if c not in ['score_date', 'ticker']:
                        update_parts.append(f"{c} = COALESCE(EXCLUDED.{c}, historical_scores.{c})")
                update_clause = ', '.join(update_parts)

                sql = f"""
                    INSERT INTO historical_scores ({col_list})
                    VALUES ({placeholders})
                    ON CONFLICT (score_date, ticker) DO UPDATE SET {update_clause}
                """

                insert_values = [values.get(c) for c in insert_cols]

                # Use SAVEPOINT for per-row error handling
                try:
                    cur.execute("SAVEPOINT row_save")
                    cur.execute(sql, insert_values)
                    cur.execute("RELEASE SAVEPOINT row_save")
                    imported += 1
                except Exception as e:
                    cur.execute("ROLLBACK TO SAVEPOINT row_save")
                    # Try again with problematic fields set to NULL
                    retry_values = values.copy()
                    error_str = str(e).lower()

                    # Identify and nullify problematic numeric fields
                    for db_col in insert_cols:
                        if db_col in ['score_date', 'ticker', 'source_file', 'signal_type']:
                            continue
                        col_type = COLUMN_TYPES.get(db_col, '')
                        if 'NUMERIC' in col_type or 'INT' in col_type:
                            retry_values[db_col] = None
                            fields_nullified += 1

                    retry_vals = [retry_values.get(c) for c in insert_cols]
                    try:
                        cur.execute("SAVEPOINT row_save2")
                        cur.execute(sql, retry_vals)
                        cur.execute("RELEASE SAVEPOINT row_save2")
                        imported += 1
                    except Exception as e2:
                        cur.execute("ROLLBACK TO SAVEPOINT row_save2")
                        print(f"  SKIP {ticker}: {str(e2)[:60]}")
                        skipped += 1

        conn.commit()

    result.success = True
    result.records_imported = imported
    result.records_skipped = skipped
    result.fields_nullified = fields_nullified
    result.new_columns_created = new_columns

    logger.info(f"Imported {filepath}: {imported} records ({skipped} skipped, {fields_nullified} fields nullified)")
    if new_columns:
        logger.info(f"  New columns created: {', '.join(new_columns)}")

    return result


def import_directory(dirpath: str) -> BatchImportResult:
    """Import all CSV files from directory."""
    batch = BatchImportResult()

    csv_files = sorted(glob.glob(os.path.join(dirpath, '*.csv')))
    logger.info(f"Found {len(csv_files)} CSV files in {dirpath}")

    for i, filepath in enumerate(csv_files):
        if (i + 1) % 50 == 0:
            print(f"Processing {i + 1}/{len(csv_files)}...")

        result = import_csv(filepath)
        batch.add_result(result)

    return batch


def show_stats():
    """Show database statistics."""
    print("\n" + "=" * 70)
    print("HISTORICAL SCORES DATABASE STATISTICS")
    print("=" * 70)

    try:
        engine = get_engine()
        stats = pd.read_sql("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT ticker) as unique_tickers,
                COUNT(DISTINCT score_date) as unique_dates,
                MIN(score_date) as earliest_date,
                MAX(score_date) as latest_date
            FROM historical_scores
        """, engine)

        s = stats.iloc[0]
        print(f"Total Records:   {s['total_records']:,}")
        print(f"Unique Tickers:  {s['unique_tickers']:,}")
        print(f"Unique Dates:    {s['unique_dates']:,}")
        print(f"Date Range:      {s['earliest_date']} to {s['latest_date']}")

    except Exception as e:
        print(f"Error: {e}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Import historical scores (V3 - Ultra Robust)')
    parser.add_argument('--file', type=str, help='Single CSV file')
    parser.add_argument('--dir', type=str, help='Directory with CSVs')
    parser.add_argument('--stats', action='store_true', help='Show stats')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    ensure_table_exists()

    if args.file:
        result = import_csv(args.file)
        print(f"Imported: {result.records_imported}, Skipped: {result.records_skipped}")
    elif args.dir:
        batch = import_directory(args.dir)
        batch.print_summary()
    elif args.stats:
        show_stats()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()