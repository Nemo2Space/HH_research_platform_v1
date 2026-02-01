"""
Prediction Tracker Module

Tracks Alpha Model predictions and outcomes to monitor ML learning.
Provides dashboard for visualizing accuracy, calibration, and progress.

Author: Alpha Research Platform
Version: 2024-12-28
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from src.db.connection import get_engine, get_connection
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# DATABASE SETUP
# ============================================================================

def ensure_predictions_table():
    """Create alpha_predictions table if it doesn't exist, or add missing columns."""

    # First, check if table exists and what columns it has
    check_sql = """
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name = 'alpha_predictions'
    """

    required_columns = {
        'ticker': 'VARCHAR(20) NOT NULL',
        'prediction_date': 'DATE NOT NULL',
        'predicted_return_5d': 'DECIMAL(8,4)',
        'predicted_return_10d': 'DECIMAL(8,4)',
        'predicted_return_20d': 'DECIMAL(8,4)',
        'predicted_direction': 'VARCHAR(10)',
        'predicted_probability': 'DECIMAL(5,4)',
        'alpha_signal': 'VARCHAR(20)',
        'alpha_conviction': 'DECIMAL(5,4)',
        'platform_signal': 'VARCHAR(20)',
        'platform_score': 'DECIMAL(5,2)',
        'price_at_prediction': 'DECIMAL(12,4)',
        'actual_return_5d': 'DECIMAL(8,4)',
        'actual_return_10d': 'DECIMAL(8,4)',
        'actual_return_20d': 'DECIMAL(8,4)',
        'price_at_5d': 'DECIMAL(12,4)',
        'price_at_10d': 'DECIMAL(12,4)',
        'price_at_20d': 'DECIMAL(12,4)',
        'outcome_updated_at': 'TIMESTAMP',
        'prediction_correct_5d': 'BOOLEAN',
        'absolute_error_5d': 'DECIMAL(8,4)',
        'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
    }

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if table exists
                cur.execute(check_sql)
                existing_columns = {row[0] for row in cur.fetchall()}

                if not existing_columns:
                    # Table doesn't exist - create it
                    create_sql = """
                    CREATE TABLE IF NOT EXISTS alpha_predictions (
                        id SERIAL PRIMARY KEY,
                        ticker VARCHAR(20) NOT NULL,
                        prediction_date DATE NOT NULL,
                        predicted_return_5d DECIMAL(8,4),
                        predicted_return_10d DECIMAL(8,4),
                        predicted_return_20d DECIMAL(8,4),
                        predicted_direction VARCHAR(10),
                        predicted_probability DECIMAL(5,4),
                        alpha_signal VARCHAR(20),
                        alpha_conviction DECIMAL(5,4),
                        platform_signal VARCHAR(20),
                        platform_score DECIMAL(5,2),
                        price_at_prediction DECIMAL(12,4),
                        actual_return_5d DECIMAL(8,4),
                        actual_return_10d DECIMAL(8,4),
                        actual_return_20d DECIMAL(8,4),
                        price_at_5d DECIMAL(12,4),
                        price_at_10d DECIMAL(12,4),
                        price_at_20d DECIMAL(12,4),
                        outcome_updated_at TIMESTAMP,
                        prediction_correct_5d BOOLEAN,
                        absolute_error_5d DECIMAL(8,4),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(ticker, prediction_date)
                    );
                    CREATE INDEX IF NOT EXISTS idx_alpha_pred_ticker ON alpha_predictions(ticker);
                    CREATE INDEX IF NOT EXISTS idx_alpha_pred_date ON alpha_predictions(prediction_date DESC);
                    """
                    cur.execute(create_sql)
                    logger.info("Created alpha_predictions table")
                else:
                    # Table exists - add missing columns
                    for col_name, col_type in required_columns.items():
                        if col_name not in existing_columns:
                            # Remove DEFAULT clause for ALTER TABLE
                            col_type_clean = col_type.replace(' DEFAULT CURRENT_TIMESTAMP', '')
                            alter_sql = f"ALTER TABLE alpha_predictions ADD COLUMN IF NOT EXISTS {col_name} {col_type_clean}"
                            try:
                                cur.execute(alter_sql)
                                logger.info(f"Added column {col_name} to alpha_predictions")
                            except Exception as e:
                                logger.debug(f"Column {col_name} might already exist: {e}")

                    # Try to add unique constraint if missing
                    try:
                        cur.execute("""
                            ALTER TABLE alpha_predictions 
                            ADD CONSTRAINT alpha_predictions_ticker_date_key 
                            UNIQUE (ticker, prediction_date)
                        """)
                    except:
                        pass  # Constraint might already exist

            conn.commit()
        logger.info("alpha_predictions table ensured")
        return True
    except Exception as e:
        logger.error(f"Error creating/updating alpha_predictions table: {e}")
        return False


def reset_predictions_table():
    """Drop and recreate the alpha_predictions table (use with caution!)."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS alpha_predictions CASCADE")
            conn.commit()
        logger.info("Dropped alpha_predictions table")
        return ensure_predictions_table()
    except Exception as e:
        logger.error(f"Error resetting alpha_predictions table: {e}")
        return False


# ============================================================================
# SAVE PREDICTIONS
# ============================================================================

def save_prediction(
    ticker: str,
    predicted_return_5d: float,
    predicted_probability: float,
    alpha_signal: str,
    alpha_conviction: float,
    platform_signal: str = None,
    platform_score: float = None,
    price_at_prediction: float = None,
    predicted_return_10d: float = None,
    predicted_return_20d: float = None
) -> bool:
    """
    Save a new prediction to the database.
    """
    ensure_predictions_table()

    prediction_date = datetime.now().date()
    predicted_direction = 'UP' if predicted_return_5d > 0.005 else ('DOWN' if predicted_return_5d < -0.005 else 'NEUTRAL')

    sql = """
    INSERT INTO alpha_predictions (
        ticker, prediction_date, 
        predicted_return_5d, predicted_return_10d, predicted_return_20d,
        predicted_direction, predicted_probability,
        alpha_signal, alpha_conviction,
        platform_signal, platform_score, price_at_prediction
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    ON CONFLICT (ticker, prediction_date) 
    DO UPDATE SET
        predicted_return_5d = EXCLUDED.predicted_return_5d,
        predicted_return_10d = EXCLUDED.predicted_return_10d,
        predicted_return_20d = EXCLUDED.predicted_return_20d,
        predicted_direction = EXCLUDED.predicted_direction,
        predicted_probability = EXCLUDED.predicted_probability,
        alpha_signal = EXCLUDED.alpha_signal,
        alpha_conviction = EXCLUDED.alpha_conviction,
        platform_signal = EXCLUDED.platform_signal,
        platform_score = EXCLUDED.platform_score,
        price_at_prediction = EXCLUDED.price_at_prediction
    """

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    ticker, prediction_date,
                    predicted_return_5d, predicted_return_10d, predicted_return_20d,
                    predicted_direction, predicted_probability,
                    alpha_signal, alpha_conviction,
                    platform_signal, platform_score, price_at_prediction
                ))
            conn.commit()
        logger.debug(f"Saved prediction for {ticker}: {alpha_signal} ({predicted_return_5d:+.2%})")
        return True
    except Exception as e:
        logger.error(f"Error saving prediction for {ticker}: {e}")
        return False


# ============================================================================
# UPDATE OUTCOMES
# ============================================================================

def update_outcomes(days_back: int = 30) -> int:
    """
    Update actual outcomes for predictions that are now past their horizon.
    """
    ensure_predictions_table()

    cutoff_5d = datetime.now().date() - timedelta(days=5)
    lookback = datetime.now().date() - timedelta(days=days_back)

    query = """
    SELECT id, ticker, prediction_date, price_at_prediction, predicted_return_5d
    FROM alpha_predictions
    WHERE prediction_date >= %s
      AND prediction_date <= %s
      AND actual_return_5d IS NULL
      AND price_at_prediction IS NOT NULL
    """

    try:
        df = pd.read_sql(query, get_engine(), params=(lookback, cutoff_5d))

        if df.empty:
            logger.info("No predictions to update")
            return 0

        updated = 0

        for _, row in df.iterrows():
            ticker = row['ticker']
            pred_date = row['prediction_date']
            pred_price = float(row['price_at_prediction'])
            pred_return = float(row['predicted_return_5d'] or 0)

            try:
                import yfinance as yf

                stock = yf.Ticker(ticker)
                hist = stock.history(start=pred_date, end=datetime.now().date() + timedelta(days=1))

                if hist.empty or len(hist) < 5:
                    continue

                # Get prices at each horizon
                price_5d = hist.iloc[min(5, len(hist)-1)]['Close'] if len(hist) > 5 else hist.iloc[-1]['Close']
                price_10d = hist.iloc[min(10, len(hist)-1)]['Close'] if len(hist) > 10 else None
                price_20d = hist.iloc[min(20, len(hist)-1)]['Close'] if len(hist) > 20 else None

                return_5d = (price_5d / pred_price - 1)
                return_10d = (price_10d / pred_price - 1) if price_10d else None
                return_20d = (price_20d / pred_price - 1) if price_20d else None

                # Was prediction correct (direction)?
                pred_up = pred_return > 0
                actual_up = return_5d > 0
                correct = pred_up == actual_up
                abs_error = abs(pred_return - return_5d)

                update_sql = """
                UPDATE alpha_predictions
                SET actual_return_5d = %s,
                    actual_return_10d = %s,
                    actual_return_20d = %s,
                    price_at_5d = %s,
                    price_at_10d = %s,
                    price_at_20d = %s,
                    outcome_updated_at = %s,
                    prediction_correct_5d = %s,
                    absolute_error_5d = %s
                WHERE id = %s
                """

                with get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(update_sql, (
                            return_5d, return_10d, return_20d,
                            price_5d, price_10d, price_20d,
                            datetime.now(), correct, abs_error,
                            row['id']
                        ))
                    conn.commit()

                updated += 1
                logger.debug(f"Updated {ticker}: actual 5d return = {return_5d:+.2%}")

            except Exception as e:
                logger.debug(f"Could not update outcome for {ticker}: {e}")
                continue

        logger.info(f"Updated {updated} prediction outcomes")
        return updated

    except Exception as e:
        logger.error(f"Error updating outcomes: {e}")
        return 0


# ============================================================================
# STATISTICS & METRICS
# ============================================================================

@dataclass
class PredictionStats:
    """Statistics about prediction performance."""
    total_predictions: int = 0
    predictions_with_outcome: int = 0
    pending_outcomes: int = 0

    direction_accuracy: float = 0.0
    mean_absolute_error: float = 0.0
    win_rate_overall: float = 0.0
    win_rate_high_conf: float = 0.0
    win_rate_low_conf: float = 0.0
    buy_signal_win_rate: float = 0.0
    sell_signal_win_rate: float = 0.0

    accuracy_last_7d: float = 0.0
    accuracy_last_30d: float = 0.0

    samples_for_ml: int = 0
    ml_gate_status: str = "BLOCKED"
    samples_to_degraded: int = 40
    samples_to_tradable: int = 80


def get_prediction_stats() -> PredictionStats:
    """Calculate comprehensive prediction statistics."""
    ensure_predictions_table()

    stats = PredictionStats()

    try:
        # Basic counts
        count_sql = """
        SELECT 
            COUNT(*) as total,
            COUNT(actual_return_5d) as with_outcome,
            COUNT(*) FILTER (WHERE actual_return_5d IS NULL) as pending
        FROM alpha_predictions
        """

        df = pd.read_sql(count_sql, get_engine())
        if not df.empty:
            stats.total_predictions = int(df.iloc[0]['total'])
            stats.predictions_with_outcome = int(df.iloc[0]['with_outcome'])
            stats.pending_outcomes = int(df.iloc[0]['pending'])

        if stats.predictions_with_outcome > 0:
            accuracy_sql = """
            SELECT 
                AVG(CASE WHEN prediction_correct_5d THEN 1.0 ELSE 0.0 END) as direction_accuracy,
                AVG(absolute_error_5d) as mae,
                AVG(CASE WHEN predicted_direction = 'UP' AND actual_return_5d > 0 THEN 1.0
                         WHEN predicted_direction = 'DOWN' AND actual_return_5d < 0 THEN 1.0
                         ELSE 0.0 END) as win_rate
            FROM alpha_predictions
            WHERE actual_return_5d IS NOT NULL
            """

            df = pd.read_sql(accuracy_sql, get_engine())
            if not df.empty:
                stats.direction_accuracy = float(df.iloc[0]['direction_accuracy'] or 0)
                stats.mean_absolute_error = float(df.iloc[0]['mae'] or 0)
                stats.win_rate_overall = float(df.iloc[0]['win_rate'] or 0)

        # ML gate status
        stats.samples_for_ml = stats.predictions_with_outcome
        stats.samples_to_degraded = max(0, 40 - stats.samples_for_ml)
        stats.samples_to_tradable = max(0, 80 - stats.samples_for_ml)

        if stats.samples_for_ml >= 80 and stats.direction_accuracy >= 0.55:
            stats.ml_gate_status = "TRADABLE"
        elif stats.samples_for_ml >= 40:
            stats.ml_gate_status = "DEGRADED"
        else:
            stats.ml_gate_status = "BLOCKED"

        return stats

    except Exception as e:
        logger.error(f"Error calculating prediction stats: {e}")
        return stats


def get_recent_predictions(limit: int = 20) -> pd.DataFrame:
    """Get recent predictions with their outcomes."""
    ensure_predictions_table()

    query = """
    SELECT 
        ticker,
        prediction_date,
        alpha_signal,
        predicted_return_5d,
        predicted_probability,
        platform_signal,
        price_at_prediction,
        actual_return_5d,
        prediction_correct_5d,
        CASE 
            WHEN actual_return_5d IS NULL THEN 'PENDING'
            WHEN prediction_correct_5d THEN 'WIN'
            ELSE 'LOSS'
        END as result
    FROM alpha_predictions
    ORDER BY prediction_date DESC, ticker
    LIMIT %s
    """

    try:
        return pd.read_sql(query, get_engine(), params=(limit,))
    except Exception as e:
        logger.error(f"Error getting recent predictions: {e}")
        return pd.DataFrame()


def get_calibration_data() -> pd.DataFrame:
    """Get data for calibration curve."""
    ensure_predictions_table()

    query = """
    SELECT 
        ROUND(predicted_probability * 10) / 10 as prob_bucket,
        COUNT(*) as count,
        AVG(CASE WHEN actual_return_5d > 0 THEN 1.0 ELSE 0.0 END) as actual_win_rate
    FROM alpha_predictions
    WHERE actual_return_5d IS NOT NULL
      AND predicted_probability IS NOT NULL
    GROUP BY ROUND(predicted_probability * 10) / 10
    ORDER BY prob_bucket
    """

    try:
        return pd.read_sql(query, get_engine())
    except Exception as e:
        logger.error(f"Error getting calibration data: {e}")
        return pd.DataFrame()


def get_accuracy_over_time() -> pd.DataFrame:
    """Get rolling accuracy over time."""
    ensure_predictions_table()

    query = """
    SELECT 
        prediction_date,
        COUNT(*) as predictions,
        AVG(CASE WHEN prediction_correct_5d THEN 1.0 ELSE 0.0 END) as accuracy
    FROM alpha_predictions
    WHERE actual_return_5d IS NOT NULL
    GROUP BY prediction_date
    ORDER BY prediction_date
    """

    try:
        df = pd.read_sql(query, get_engine())
        if not df.empty and len(df) > 0:
            df['rolling_accuracy'] = df['accuracy'].rolling(7, min_periods=1).mean()
        return df
    except Exception as e:
        logger.error(f"Error getting accuracy over time: {e}")
        return pd.DataFrame()


# ============================================================================
# AUTO-SAVE HELPER
# ============================================================================

def auto_save_from_signal(
    ticker: str,
    signal_data: dict,
    alpha_prediction: dict,
    current_price: float = None
) -> bool:
    """Auto-save prediction when a signal is generated."""
    return save_prediction(
        ticker=ticker,
        predicted_return_5d=alpha_prediction.get('expected_return_5d', 0),
        predicted_return_10d=alpha_prediction.get('expected_return_10d'),
        predicted_return_20d=alpha_prediction.get('expected_return_20d'),
        predicted_probability=alpha_prediction.get('prob_positive_5d', 0.5),
        alpha_signal=alpha_prediction.get('signal', 'HOLD'),
        alpha_conviction=alpha_prediction.get('conviction', 0.5),
        platform_signal=signal_data.get('signal_type'),
        platform_score=signal_data.get('total_score'),
        price_at_prediction=current_price
    )