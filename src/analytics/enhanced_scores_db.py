"""
Enhanced Scores Database Storage

Stores enhanced scoring results in database so they load instantly in deep dive.
Scores are computed during batch analysis and stored, not calculated on page load.

Usage:
    # During batch analysis:
    from src.analytics.enhanced_scores_db import save_enhanced_scores, compute_and_save_enhanced_scores

    # Save pre-computed scores
    save_enhanced_scores(ticker, scores_dict)

    # Or compute and save in one step
    compute_and_save_enhanced_scores(ticker, row_data)

    # During deep dive (fast load):
    from src.analytics.enhanced_scores_db import load_enhanced_scores
    scores = load_enhanced_scores(ticker)  # Returns cached scores or None

Author: Alpha Research Platform
Version: 2024-12-23
"""

import json
from datetime import datetime, date
from typing import Dict, Any, Optional, Tuple
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Database connection
try:
    from src.db.connection import get_engine, get_connection
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("Database connection not available for enhanced scores storage")


# =============================================================================
# DATABASE SCHEMA
# =============================================================================

MIGRATION_SQL = """
-- Enhanced scores table to store pre-computed enhancement adjustments
CREATE TABLE IF NOT EXISTS enhanced_scores (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    
    -- Individual component scores
    pe_relative_score INTEGER DEFAULT 0,
    peg_score INTEGER DEFAULT 0,
    price_target_score INTEGER DEFAULT 0,
    macd_score INTEGER DEFAULT 0,
    volume_score INTEGER DEFAULT 0,
    insider_score INTEGER DEFAULT 0,
    revision_score INTEGER DEFAULT 0,
    earnings_surprise_score INTEGER DEFAULT 0,
    
    -- Total adjustment
    total_adjustment INTEGER DEFAULT 0,
    
    -- Reasons/details as JSON
    details JSONB,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Unique constraint
    UNIQUE(ticker, date)
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_enhanced_scores_ticker_date 
ON enhanced_scores(ticker, date DESC);

-- Index for finding stale scores
CREATE INDEX IF NOT EXISTS idx_enhanced_scores_updated 
ON enhanced_scores(updated_at);
"""


def run_migration():
    """Create the enhanced_scores table if it doesn't exist."""
    if not DB_AVAILABLE:
        logger.error("Database not available for migration")
        return False

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(MIGRATION_SQL)
            conn.commit()
        logger.info("Enhanced scores table migration completed")
        return True
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


# =============================================================================
# SAVE ENHANCED SCORES
# =============================================================================

def _to_native(val):
    """Convert numpy/pandas types to native Python types for SQL compatibility."""
    if val is None:
        return None
    if hasattr(val, 'item'):  # numpy scalar types have .item() method
        return val.item()
    # Check for numpy types by module name
    type_name = type(val).__module__
    if type_name == 'numpy':
        if hasattr(val, 'item'):
            return val.item()
        return float(val) if 'float' in str(type(val)) else int(val) if 'int' in str(type(val)) else val
    return val


def save_enhanced_scores(
    ticker: str,
    pe_relative_score: int = 0,
    peg_score: int = 0,
    price_target_score: int = 0,
    macd_score: int = 0,
    volume_score: int = 0,
    insider_score: int = 0,
    revision_score: int = 0,
    earnings_surprise_score: int = 0,
    total_adjustment: int = 0,
    details: Dict[str, str] = None,
    score_date: date = None,
) -> bool:
    """
    Save enhanced scores to database.

    Args:
        ticker: Stock ticker
        *_score: Individual component scores
        total_adjustment: Sum of all adjustments
        details: Dict of reason strings for each component
        score_date: Date for the scores (defaults to today)

    Returns:
        True if saved successfully
    """
    if not DB_AVAILABLE:
        return False

    if score_date is None:
        score_date = date.today()

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO enhanced_scores (
                        ticker, date,
                        pe_relative_score, peg_score, price_target_score,
                        macd_score, volume_score, insider_score,
                        revision_score, earnings_surprise_score,
                        total_adjustment, details,
                        created_at, updated_at
                    ) VALUES (
                        %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s, %s,
                        NOW(), NOW()
                    )
                    ON CONFLICT (ticker, date) DO UPDATE SET
                        pe_relative_score = EXCLUDED.pe_relative_score,
                        peg_score = EXCLUDED.peg_score,
                        price_target_score = EXCLUDED.price_target_score,
                        macd_score = EXCLUDED.macd_score,
                        volume_score = EXCLUDED.volume_score,
                        insider_score = EXCLUDED.insider_score,
                        revision_score = EXCLUDED.revision_score,
                        earnings_surprise_score = EXCLUDED.earnings_surprise_score,
                        total_adjustment = EXCLUDED.total_adjustment,
                        details = EXCLUDED.details,
                        updated_at = NOW()
                """, (
                    ticker, score_date,
                    _to_native(pe_relative_score), _to_native(peg_score), _to_native(price_target_score),
                    _to_native(macd_score), _to_native(volume_score), _to_native(insider_score),
                    _to_native(revision_score), _to_native(earnings_surprise_score),
                    _to_native(total_adjustment), json.dumps(details) if details else None,
                ))
            conn.commit()
        return True
    except Exception as e:
        logger.warning(f"Failed to save enhanced scores for {ticker}: {e}")
        return False


def save_enhanced_scores_from_dict(ticker: str, scores_dict: Dict[str, Any], score_date: date = None) -> bool:
    """
    Save enhanced scores from a dictionary (as returned by get_single_ticker_enhancement).
    """
    return save_enhanced_scores(
        ticker=ticker,
        pe_relative_score=scores_dict.get('pe_relative', {}).get('score', 0),
        peg_score=scores_dict.get('peg', {}).get('score', 0),
        price_target_score=scores_dict.get('price_target', {}).get('score', 0),
        macd_score=scores_dict.get('macd', {}).get('score', 0),
        volume_score=scores_dict.get('volume', {}).get('score', 0),
        insider_score=scores_dict.get('insider', {}).get('score', 0),
        revision_score=scores_dict.get('revisions', {}).get('score', 0),
        earnings_surprise_score=scores_dict.get('earnings_surprise', {}).get('score', 0),
        total_adjustment=scores_dict.get('total_adjustment', 0),
        details={
            'pe_relative': scores_dict.get('pe_relative', {}).get('reason', ''),
            'peg': scores_dict.get('peg', {}).get('reason', ''),
            'price_target': scores_dict.get('price_target', {}).get('reason', ''),
            'macd': scores_dict.get('macd', {}).get('reason', ''),
            'volume': scores_dict.get('volume', {}).get('reason', ''),
            'insider': scores_dict.get('insider', {}).get('reason', ''),
            'revisions': scores_dict.get('revisions', {}).get('reason', ''),
            'earnings_surprise': scores_dict.get('earnings_surprise', {}).get('reason', ''),
        },
        score_date=score_date,
    )


# =============================================================================
# LOAD ENHANCED SCORES
# =============================================================================

def load_enhanced_scores(ticker: str, max_age_days: int = 1) -> Optional[Dict[str, Any]]:
    """
    Load enhanced scores from database.

    Args:
        ticker: Stock ticker
        max_age_days: Maximum age of scores to consider valid (default 1 day)

    Returns:
        Dict with scores and details, or None if not found/stale
    """
    if not DB_AVAILABLE:
        return None

    try:
        query = """
            SELECT 
                pe_relative_score, peg_score, price_target_score,
                macd_score, volume_score, insider_score,
                revision_score, earnings_surprise_score,
                total_adjustment, details, date, updated_at
            FROM enhanced_scores
            WHERE ticker = %s
              AND date >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY date DESC
            LIMIT 1
        """

        df = pd.read_sql(query, get_engine(), params=(ticker, max_age_days))

        if df.empty:
            return None

        row = df.iloc[0]
        details = row.get('details') or {}
        if isinstance(details, str):
            details = json.loads(details)

        return {
            'ticker': ticker,
            'pe_relative': {'score': row.get('pe_relative_score', 0), 'reason': details.get('pe_relative', '')},
            'peg': {'score': row.get('peg_score', 0), 'reason': details.get('peg', '')},
            'price_target': {'score': row.get('price_target_score', 0), 'reason': details.get('price_target', '')},
            'macd': {'score': row.get('macd_score', 0), 'reason': details.get('macd', '')},
            'volume': {'score': row.get('volume_score', 0), 'reason': details.get('volume', '')},
            'insider': {'score': row.get('insider_score', 0), 'reason': details.get('insider', '')},
            'revisions': {'score': row.get('revision_score', 0), 'reason': details.get('revisions', '')},
            'earnings_surprise': {'score': row.get('earnings_surprise_score', 0), 'reason': details.get('earnings_surprise', '')},
            'total_adjustment': row.get('total_adjustment', 0),
            'date': row.get('date'),
            'updated_at': row.get('updated_at'),
            'from_cache': True,
        }
    except Exception as e:
        logger.debug(f"Could not load enhanced scores for {ticker}: {e}")
        return None


def load_enhanced_scores_batch(tickers: list, max_age_days: int = 1) -> Dict[str, Dict[str, Any]]:
    """
    Load enhanced scores for multiple tickers at once.

    Returns:
        Dict mapping ticker -> scores dict
    """
    if not DB_AVAILABLE or not tickers:
        return {}

    try:
        # Build placeholders for IN clause
        placeholders = ','.join(['%s'] * len(tickers))

        query = f"""
            SELECT DISTINCT ON (ticker)
                ticker,
                pe_relative_score, peg_score, price_target_score,
                macd_score, volume_score, insider_score,
                revision_score, earnings_surprise_score,
                total_adjustment, details, date
            FROM enhanced_scores
            WHERE ticker IN ({placeholders})
              AND date >= CURRENT_DATE - INTERVAL '{max_age_days} days'
            ORDER BY ticker, date DESC
        """

        df = pd.read_sql(query, get_engine(), params=tuple(tickers))

        results = {}
        for _, row in df.iterrows():
            ticker = row['ticker']
            details = row.get('details') or {}
            if isinstance(details, str):
                details = json.loads(details)

            results[ticker] = {
                'ticker': ticker,
                'pe_relative': {'score': row.get('pe_relative_score', 0), 'reason': details.get('pe_relative', '')},
                'peg': {'score': row.get('peg_score', 0), 'reason': details.get('peg', '')},
                'price_target': {'score': row.get('price_target_score', 0), 'reason': details.get('price_target', '')},
                'macd': {'score': row.get('macd_score', 0), 'reason': details.get('macd', '')},
                'volume': {'score': row.get('volume_score', 0), 'reason': details.get('volume', '')},
                'insider': {'score': row.get('insider_score', 0), 'reason': details.get('insider', '')},
                'revisions': {'score': row.get('revisions_score', 0), 'reason': details.get('revisions', '')},
                'earnings_surprise': {'score': row.get('earnings_surprise_score', 0), 'reason': details.get('earnings_surprise', '')},
                'total_adjustment': row.get('total_adjustment', 0),
                'from_cache': True,
            }

        return results
    except Exception as e:
        logger.warning(f"Could not load batch enhanced scores: {e}")
        return {}


# =============================================================================
# COMPUTE AND SAVE (for batch analysis)
# =============================================================================

def compute_and_save_enhanced_scores(
    ticker: str,
    row_data: Dict[str, Any] = None,
    price_history: pd.Series = None,
    volume_history: pd.Series = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Compute enhanced scores and save to database.
    Call this during batch analysis.

    Args:
        ticker: Stock ticker
        row_data: Dict with PE, sector, price, target, etc.
        price_history: Price series for MACD
        volume_history: Volume series for accumulation/distribution

    Returns:
        Tuple of (total_adjustment, scores_dict)
    """
    try:
        from src.analytics.enhanced_scoring import get_enhanced_scores
        from src.analytics.enhanced_scoring_integration import get_price_volume_history

        # Fetch price/volume if not provided
        if price_history is None:
            price_history, volume_history = get_price_volume_history(ticker, days=60)

        # Normalize row data
        if row_data is None:
            row_data = {}

        # Compute enhanced scores
        scores = get_enhanced_scores(
            ticker=ticker,
            current_price=row_data.get('price') or row_data.get('Price'),
            target_price=row_data.get('target_mean') or row_data.get('TargetPrice'),
            pe_ratio=row_data.get('pe_ratio') or row_data.get('PE'),
            forward_pe=row_data.get('forward_pe'),
            peg_ratio=row_data.get('peg_ratio'),
            sector=row_data.get('sector') or row_data.get('Sector'),
            buy_count=int(row_data.get('buy_count') or row_data.get('analyst_buy') or 0),
            total_ratings=int(row_data.get('total_ratings') or row_data.get('analyst_total') or 0),
            price_history=price_history,
            volume_history=volume_history,
        )

        # Save to database
        save_enhanced_scores(
            ticker=ticker,
            pe_relative_score=scores.pe_relative_score,
            peg_score=scores.peg_score,
            price_target_score=scores.price_target_score,
            macd_score=scores.macd_score,
            volume_score=scores.volume_score,
            insider_score=scores.insider_score,
            revision_score=scores.revision_score,
            earnings_surprise_score=scores.earnings_surprise_score,
            total_adjustment=scores.enhancement_total,
            details=scores.details,
        )

        # Return in standard format
        result = {
            'ticker': ticker,
            'pe_relative': {'score': scores.pe_relative_score, 'reason': scores.details.get('pe_relative', '')},
            'peg': {'score': scores.peg_score, 'reason': scores.details.get('peg', '')},
            'price_target': {'score': scores.price_target_score, 'reason': scores.details.get('price_target', '')},
            'macd': {'score': scores.macd_score, 'reason': scores.details.get('macd', '')},
            'volume': {'score': scores.volume_score, 'reason': scores.details.get('volume', '')},
            'insider': {'score': scores.insider_score, 'reason': scores.details.get('insider', '')},
            'revisions': {'score': scores.revision_score, 'reason': scores.details.get('revisions', '')},
            'earnings_surprise': {'score': scores.earnings_surprise_score, 'reason': scores.details.get('earnings_surprise', '')},
            'total_adjustment': scores.enhancement_total,
        }

        return scores.enhancement_total, result

    except Exception as e:
        logger.warning(f"Failed to compute/save enhanced scores for {ticker}: {e}")
        return 0, {}


# =============================================================================
# CLI / MIGRATION RUNNER
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "migrate":
        print("Running enhanced_scores table migration...")
        if run_migration():
            print("✅ Migration completed successfully")
        else:
            print("❌ Migration failed")
            sys.exit(1)

    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        ticker = sys.argv[2] if len(sys.argv) > 2 else "AAPL"
        print(f"\nTesting enhanced scores for {ticker}...")

        # Run migration first
        run_migration()

        # Compute and save
        print("Computing scores...")
        adjustment, scores = compute_and_save_enhanced_scores(ticker)
        print(f"Total adjustment: {adjustment:+d}")

        # Load back
        print("\nLoading from database...")
        loaded = load_enhanced_scores(ticker)
        if loaded:
            print(f"✅ Loaded successfully: adjustment={loaded['total_adjustment']:+d}")
            for key in ['pe_relative', 'peg', 'price_target', 'macd', 'volume', 'insider', 'revisions', 'earnings_surprise']:
                score = loaded[key]['score']
                reason = loaded[key]['reason'][:50] if loaded[key]['reason'] else ''
                print(f"   {key}: {score:+3d}  {reason}")
        else:
            print("❌ Failed to load")

    else:
        print("Usage:")
        print("  python enhanced_scores_db.py migrate  - Create database table")
        print("  python enhanced_scores_db.py test [TICKER]  - Test compute/save/load")