"""
Historical Scores Sync Utility

Automatically syncs screener_scores to historical_scores for backtesting.
Call sync_to_historical() after saving to screener_scores.

Usage:
    from src.utils.historical_sync import sync_to_historical

    # After saving a score to screener_scores:
    sync_to_historical(ticker, score_date, scores_dict)

    # Or sync all recent:
    sync_recent_to_historical(days=1)
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
_engine = None
_get_connection = None

def _init_db():
    """Initialize database connections lazily."""
    global _engine, _get_connection
    if _engine is None:
        from src.db.connection import get_engine, get_connection
        _engine = get_engine()
        _get_connection = get_connection


def sync_to_historical(
    ticker: str,
    score_date: date,
    scores: Dict[str, Any],
    sector: str = None
) -> bool:
    """
    Sync a single score record to historical_scores.
    Call this after saving to screener_scores.

    Args:
        ticker: Stock ticker
        score_date: Date of the score
        scores: Dict with score values (sentiment, fundamental_score, total_score, etc.)
        sector: Optional sector name

    Returns:
        True if successful
    """
    _init_db()

    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO historical_scores (
                        score_date, ticker, sector, sentiment, 
                        fundamental_score, growth_score, dividend_score, 
                        total_score, gap_score, mkt_score
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (score_date, ticker) DO UPDATE SET
                        sentiment = EXCLUDED.sentiment,
                        fundamental_score = EXCLUDED.fundamental_score,
                        growth_score = EXCLUDED.growth_score,
                        dividend_score = EXCLUDED.dividend_score,
                        total_score = EXCLUDED.total_score,
                        gap_score = EXCLUDED.gap_score,
                        mkt_score = EXCLUDED.mkt_score
                """, (
                    score_date,
                    ticker,
                    sector,
                    scores.get('sentiment_score') or scores.get('sentiment'),
                    scores.get('fundamental_score'),
                    scores.get('growth_score'),
                    scores.get('dividend_score'),
                    scores.get('total_score'),
                    scores.get('gap_score'),
                    scores.get('composite_score') or scores.get('mkt_score'),
                ))
            conn.commit()
        return True

    except Exception as e:
        logger.error(f"Failed to sync {ticker} to historical_scores: {e}")
        return False


def sync_recent_to_historical(days: int = 1) -> int:
    """
    Sync recent screener_scores to historical_scores.
    Call this at the end of a screener run.

    Args:
        days: How many days back to sync (default 1 = today only)

    Returns:
        Number of records synced
    """
    _init_db()

    try:
        cutoff_date = date.today() - timedelta(days=days)

        # Load recent screener_scores
        query = """
            SELECT 
                date as score_date, ticker, 
                sentiment_score as sentiment,
                fundamental_score, growth_score, dividend_score, 
                total_score, gap_score, composite_score as mkt_score
            FROM screener_scores
            WHERE date >= %s
        """

        df = pd.read_sql(query, _engine, params=(cutoff_date,))

        if df.empty:
            logger.info("No recent scores to sync")
            return 0

        # Replace NaN with None for PostgreSQL
        df = df.replace({np.nan: None})

        # Get sectors
        try:
            sectors = pd.read_sql(
                "SELECT DISTINCT ON (ticker) ticker, sector FROM fundamentals ORDER BY ticker, date DESC",
                _engine
            )
            df = df.merge(sectors, on='ticker', how='left')
        except Exception:
            df['sector'] = None

        # Insert into historical_scores
        count = 0
        with _get_connection() as conn:
            with conn.cursor() as cur:
                for _, row in df.iterrows():
                    try:
                        cur.execute("""
                            INSERT INTO historical_scores (
                                score_date, ticker, sector, sentiment, 
                                fundamental_score, growth_score, dividend_score, 
                                total_score, gap_score, mkt_score
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (score_date, ticker) DO UPDATE SET
                                sentiment = EXCLUDED.sentiment,
                                fundamental_score = EXCLUDED.fundamental_score,
                                growth_score = EXCLUDED.growth_score,
                                dividend_score = EXCLUDED.dividend_score,
                                total_score = EXCLUDED.total_score,
                                gap_score = EXCLUDED.gap_score,
                                mkt_score = EXCLUDED.mkt_score
                        """, (
                            row['score_date'],
                            row['ticker'],
                            row.get('sector'),
                            row.get('sentiment'),
                            row.get('fundamental_score'),
                            row.get('growth_score'),
                            row.get('dividend_score'),
                            row.get('total_score'),
                            row.get('gap_score'),
                            row.get('mkt_score'),
                        ))
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to sync {row['ticker']}: {e}")
            conn.commit()

        logger.info(f"Synced {count} records to historical_scores")
        return count

    except Exception as e:
        logger.error(f"Failed to sync recent scores: {e}")
        return 0


def ensure_historical_table():
    """Create historical_scores table if it doesn't exist."""
    _init_db()

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS historical_scores (
                    id SERIAL PRIMARY KEY,
                    score_date DATE NOT NULL,
                    ticker VARCHAR(20) NOT NULL,
                    sector VARCHAR(100),
                    sentiment NUMERIC(8,2),
                    fundamental_score NUMERIC(8,2),
                    growth_score NUMERIC(8,2),
                    dividend_score NUMERIC(8,2),
                    total_score NUMERIC(8,2),
                    gap_score NUMERIC(8,2),
                    mkt_score NUMERIC(8,2),
                    signal_type VARCHAR(20),
                    signal_correct BOOLEAN,
                    op_price NUMERIC(15,4),
                    return_1d NUMERIC(10,4),
                    return_5d NUMERIC(10,4),
                    return_10d NUMERIC(10,4),
                    return_20d NUMERIC(10,4),
                    price_1d NUMERIC(15,4),
                    price_5d NUMERIC(15,4),
                    price_10d NUMERIC(15,4),
                    price_20d NUMERIC(15,4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(score_date, ticker)
                )
            """)
        conn.commit()
    logger.info("historical_scores table ready")