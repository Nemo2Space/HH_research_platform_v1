"""
ML Auto-Maintenance

Automated jobs for return backfill and model retraining.
Hooks into the existing APScheduler in src/alerts/scheduler.py.

Jobs:
- Daily return backfill (22:00 Zurich, after US market close)
- Weekly model retrain (Sunday 03:00 Zurich)

Usage:
    from src.ml.auto_maintenance import register_ml_jobs
    register_ml_jobs(scheduler)

Author: Auto-generated
"""

import os
import sys
import time
import traceback
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml.db_helper import get_engine

import logging
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = 'models/signal_predictor.pkl'
BACKUP_DIR = 'models/backups'
MIN_SAMPLES_FOR_RETRAIN = 100  # Minimum new samples since last train
RETRAIN_COOLDOWN_DAYS = 5      # Don't retrain more than once per 5 days
MAX_TICKERS_PER_RUN = 200      # Limit tickers per backfill to avoid rate limits
BACKFILL_LOOKBACK_DAYS = 45    # Only backfill scores from last 45 days

HORIZONS = {
    'return_1d': ('price_1d', 1),
    'return_5d': ('price_5d', 5),
    'return_10d': ('price_10d', 10),
    'return_20d': ('price_20d', 20),
}


# =============================================================================
# RETURN BACKFILL
# =============================================================================

def backfill_returns_job():
    """
    Daily job: fill missing return_1d/5d/10d/20d in historical_scores.
    Runs in background thread via APScheduler.
    """
    logger.info("=" * 50)
    logger.info("AUTO-MAINTENANCE: Starting return backfill")
    logger.info("=" * 50)

    try:
        engine = get_engine()
        cutoff_date = (date.today() - timedelta(days=BACKFILL_LOOKBACK_DAYS)).isoformat()

        # Find rows needing backfill (recent only)
        query = f"""
            SELECT id, score_date, ticker, op_price,
                   return_1d, return_5d, return_10d, return_20d,
                   price_1d, price_5d, price_10d, price_20d
            FROM historical_scores
            WHERE op_price IS NOT NULL AND op_price > 0
              AND (return_1d IS NULL OR return_5d IS NULL)
              AND score_date >= '{cutoff_date}'
            ORDER BY score_date DESC, ticker
        """
        rows = pd.read_sql(query, engine)

        if rows.empty:
            logger.info("AUTO-MAINTENANCE: No rows need backfill")
            return

        tickers = rows['ticker'].unique()
        logger.info(f"AUTO-MAINTENANCE: {len(rows)} rows across {len(tickers)} tickers need backfill")

        # Limit tickers per run to avoid rate limits
        if len(tickers) > MAX_TICKERS_PER_RUN:
            tickers = tickers[:MAX_TICKERS_PER_RUN]
            rows = rows[rows['ticker'].isin(tickers)]
            logger.info(f"AUTO-MAINTENANCE: Limited to {MAX_TICKERS_PER_RUN} tickers this run")

        # Import yfinance subprocess wrapper
        from src.analytics.yf_subprocess import get_stock_history

        total_updated = 0
        failed = []

        for ticker in tickers:
            ticker_rows = rows[rows['ticker'] == ticker]
            min_date = ticker_rows['score_date'].min()
            max_date = ticker_rows['score_date'].max()

            # Fetch price history via subprocess (safe from hangs)
            try:
                # Need enough buffer for 20d returns
                start_date = (pd.Timestamp(min_date) - timedelta(days=5)).strftime('%Y-%m-%d')
                end_date = (pd.Timestamp(max_date) + timedelta(days=35)).strftime('%Y-%m-%d')

                hist = get_stock_history(ticker, period=f"60d", timeout=12)
                if hist is None or hist.empty:
                    failed.append(ticker)
                    continue

                # Build date -> close price map
                closes = {}
                for _, row in hist.iterrows():
                    d = row.get('date', row.name)
                    if isinstance(d, str):
                        d = d[:10]
                    try:
                        d = pd.Timestamp(d).date()
                    except:
                        continue
                    closes[d] = float(row['Close'])

                if not closes:
                    failed.append(ticker)
                    continue

                sorted_dates = sorted(closes.keys())

            except Exception as e:
                logger.debug(f"AUTO-MAINTENANCE: {ticker} fetch error: {e}")
                failed.append(ticker)
                continue

            # Process each row
            updates = []
            for _, row in ticker_rows.iterrows():
                score_date = row['score_date']
                if hasattr(score_date, 'date'):
                    score_date = score_date.date()
                op_price = float(row['op_price'])

                update = {'id': row['id']}
                has_update = False

                for ret_col, (price_col, n_days) in HORIZONS.items():
                    if pd.notna(row[ret_col]):
                        continue

                    # Find n-th trading day after score_date
                    future_dates = [d for d in sorted_dates if d > score_date]
                    if len(future_dates) >= n_days:
                        target_date = future_dates[n_days - 1]
                        price = closes[target_date]
                        if price > 0:
                            ret_val = round((price / op_price - 1) * 100, 4)
                            update[price_col] = round(price, 2)
                            update[ret_col] = ret_val
                            has_update = True

                if has_update:
                    updates.append(update)

            # Bulk update DB
            if updates:
                try:
                    conn = engine.raw_connection()
                    cur = conn.cursor()
                    for upd in updates:
                        set_parts = []
                        values = []
                        for key in ['price_1d', 'return_1d', 'price_5d', 'return_5d',
                                    'price_10d', 'return_10d', 'price_20d', 'return_20d']:
                            if key in upd:
                                set_parts.append(f"{key} = %s")
                                values.append(upd[key])
                        if set_parts:
                            values.append(upd['id'])
                            cur.execute(
                                f"UPDATE historical_scores SET {', '.join(set_parts)} WHERE id = %s",
                                values
                            )
                    conn.commit()
                    cur.close()
                    conn.close()
                    total_updated += len(updates)
                except Exception as e:
                    logger.error(f"AUTO-MAINTENANCE: {ticker} DB error: {e}")
                    try:
                        conn.rollback()
                        cur.close()
                        conn.close()
                    except:
                        pass

            # Small delay to avoid rate limits
            time.sleep(0.3)

        logger.info(f"AUTO-MAINTENANCE: Backfill complete - {total_updated} rows updated, "
                   f"{len(failed)} tickers failed")

    except Exception as e:
        logger.error(f"AUTO-MAINTENANCE: Backfill error: {e}")
        logger.debug(traceback.format_exc())


# =============================================================================
# MODEL RETRAIN
# =============================================================================

def retrain_model_job():
    """
    Weekly job: retrain ML model with latest data.
    Only retrains if enough new data has accumulated.
    """
    logger.info("=" * 50)
    logger.info("AUTO-MAINTENANCE: Starting model retrain check")
    logger.info("=" * 50)

    try:
        # Check cooldown
        cooldown_file = os.path.join(BACKUP_DIR, '.last_retrain')
        if os.path.exists(cooldown_file):
            last_train = datetime.fromtimestamp(os.path.getmtime(cooldown_file))
            days_since = (datetime.now() - last_train).days
            if days_since < RETRAIN_COOLDOWN_DAYS:
                logger.info(f"AUTO-MAINTENANCE: Retrain skipped - last trained {days_since}d ago "
                           f"(cooldown: {RETRAIN_COOLDOWN_DAYS}d)")
                return

        # Check if enough new data
        engine = get_engine()
        count_query = """
            SELECT COUNT(*) as n
            FROM historical_scores
            WHERE return_5d IS NOT NULL
              AND op_price IS NOT NULL AND op_price > 0
        """
        total_samples = pd.read_sql(count_query, engine).iloc[0]['n']

        # Check model's sample count
        current_samples = 0
        if os.path.exists(MODEL_PATH):
            try:
                import pickle
                with open(MODEL_PATH, 'rb') as f:
                    state = pickle.load(f)
                report = state.get('validation_report')
                if report and hasattr(report, 'total_samples'):
                    current_samples = report.total_samples
            except:
                pass

        new_samples = total_samples - current_samples
        logger.info(f"AUTO-MAINTENANCE: DB has {total_samples} samples, "
                   f"model trained on {current_samples}, delta={new_samples}")

        if new_samples < MIN_SAMPLES_FOR_RETRAIN:
            logger.info(f"AUTO-MAINTENANCE: Not enough new data ({new_samples} < {MIN_SAMPLES_FOR_RETRAIN})")
            return

        # Backup current model
        os.makedirs(BACKUP_DIR, exist_ok=True)
        if os.path.exists(MODEL_PATH):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(BACKUP_DIR, f'signal_predictor_{timestamp}.pkl')
            import shutil
            shutil.copy2(MODEL_PATH, backup_path)
            logger.info(f"AUTO-MAINTENANCE: Backed up model to {backup_path}")

            # Keep only last 5 backups
            backups = sorted(Path(BACKUP_DIR).glob('signal_predictor_*.pkl'))
            for old_backup in backups[:-5]:
                old_backup.unlink()
                logger.info(f"AUTO-MAINTENANCE: Cleaned up old backup {old_backup.name}")

        # Retrain using lite model approach (3 features on full dataset)
        logger.info("AUTO-MAINTENANCE: Retraining lite model...")
        _do_retrain(engine, total_samples)

        # Update cooldown
        os.makedirs(BACKUP_DIR, exist_ok=True)
        Path(cooldown_file).touch()

        logger.info("AUTO-MAINTENANCE: Retrain complete!")

    except Exception as e:
        logger.error(f"AUTO-MAINTENANCE: Retrain error: {e}")
        logger.debug(traceback.format_exc())


def _do_retrain(engine, expected_samples):
    """Execute the actual model retraining."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    import xgboost as xgb

    LITE_FEATURES = [
        'sentiment_score', 'fundamental_score', 'gap_score', 'fundamental_missing'
    ]
    FEATURE_DEFAULTS = {
        'sentiment_score': 50, 'fundamental_score': 50,
        'gap_score': 50, 'fundamental_missing': 0,
    }
    FEATURE_CLIPS = {
        'sentiment_score': (0, 100), 'fundamental_score': (0, 100),
        'gap_score': (0, 100), 'fundamental_missing': (0, 1),
    }
    COST_PCT = 0.15

    # Load data
    query = """
        SELECT
            h.score_date as date, h.ticker,
            COALESCE(h.sentiment, 50) as sentiment_score,
            COALESCE(h.fundamental_score, 50) as fundamental_score,
            COALESCE(CASE WHEN h.gap_score ~ '^[0-9.\\-]+$'
                     THEN h.gap_score::numeric ELSE NULL END, 50) as gap_score,
            CASE WHEN h.fundamental_score IS NULL THEN 1 ELSE 0 END as fundamental_missing,
            h.return_1d, h.return_5d, h.return_10d
        FROM historical_scores h
        WHERE h.return_5d IS NOT NULL AND h.op_price IS NOT NULL AND h.op_price > 0
        ORDER BY h.score_date, h.ticker
    """
    df = pd.read_sql(query, engine)
    logger.info(f"AUTO-MAINTENANCE: Loaded {len(df)} training samples")

    if len(df) < 500:
        logger.warning("AUTO-MAINTENANCE: Not enough data for retraining")
        return

    # Prepare returns and labels
    df['return_2d'] = df['return_1d'] + (df['return_5d'] - df['return_1d']) * 0.25
    for h in [1, 2, 5, 10]:
        col = f'return_{h}d'
        if col in df.columns:
            df[f'{col}_net'] = df[col] - COST_PCT
            df[f'win_{h}d'] = (df[f'{col}_net'] > 0).astype(int)

    # Fill and clip
    for col in LITE_FEATURES:
        df[col] = df[col].fillna(FEATURE_DEFAULTS.get(col, 50))
        clip = FEATURE_CLIPS.get(col)
        if clip:
            df[col] = df[col].clip(clip[0], clip[1])

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(df[LITE_FEATURES].values)

    # Return stats for EV
    avg_win_return = {}
    avg_loss_return = {}
    for h in [1, 2, 5, 10]:
        col = f'return_{h}d_net'
        if col in df.columns:
            returns = df[col].dropna()
            avg_win_return[h] = returns[returns > 0].mean() if (returns > 0).any() else 2.0
            avg_loss_return[h] = abs(returns[returns <= 0].mean()) if (returns <= 0).any() else 2.0

    # Train for each horizon with 3-way split
    dates = np.sort(df['date'].unique())
    n_dates = len(dates)
    cal_size = max(int(n_dates * 0.15), 30)
    purge = 5

    cal_dates = dates[n_dates - cal_size:]
    train_dates = dates[:n_dates - cal_size - purge]

    train_mask = df['date'].isin(train_dates)
    cal_mask = df['date'].isin(cal_dates)

    X_train, X_cal = X[train_mask], X[cal_mask]

    models = {}
    calibrators = {}
    feature_importance = {}

    for horizon in [5, 10]:
        target_col = f'win_{horizon}d'
        if target_col not in df.columns:
            continue

        y = df[target_col].values
        y_train, y_cal = y[train_mask], y[cal_mask]

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, eval_metric='logloss')
        model.fit(X_train, y_train)

        feature_importance[horizon] = dict(zip(LITE_FEATURES, model.feature_importances_))

        cal_method = 'sigmoid' if len(X_cal) < 100 else 'isotonic'
        try:
            calibrator = CalibratedClassifierCV(model, cv='prefit', method=cal_method)
            calibrator.fit(X_cal, y_cal)
            calibrators[horizon] = calibrator
        except:
            calibrators[horizon] = model

        models[horizon] = model
        logger.info(f"AUTO-MAINTENANCE: Trained {horizon}d model "
                   f"(train={len(X_train)}, cal={len(X_cal)})")

    # Save using MLSignalPredictor format
    from src.ml.signal_predictor import MLSignalPredictor, SignalDataLoader
    predictor = MLSignalPredictor()
    predictor.models = models
    predictor.calibrators = calibrators
    predictor.feature_names = LITE_FEATURES
    predictor.feature_importance = feature_importance
    predictor.avg_win_return = avg_win_return
    predictor.avg_loss_return = avg_loss_return
    predictor.data_loader.scaler = scaler
    predictor.data_loader.FEATURE_DEFAULTS = FEATURE_DEFAULTS
    predictor.data_loader.FEATURE_CLIPS = FEATURE_CLIPS
    predictor.data_loader.MISSING_INDICATOR_MAP = {
        'fundamental_score': 'fundamental_missing',
    }

    # Generate a minimal report
    from src.ml.signal_predictor import ModelReport
    predictor.validation_report = ModelReport(
        model_name="XGBoost-Lite-Auto",
        trained_at=datetime.now(),
        total_samples=len(df),
        feature_count=len(LITE_FEATURES),
        target_horizon=5,
        folds=[],
        mean_accuracy=0.5,
        mean_auc=0.5,
        mean_brier=0.25,
        mean_win_rate=0.5,
        mean_return=0,
        feature_importance=feature_importance.get(5, {}),
        is_well_calibrated=False,
        calibration_error=0.25,
        beats_baseline=False,
        baseline_auc=0.5,
        improvement_vs_baseline=0
    )

    predictor.save(MODEL_PATH)
    logger.info(f"AUTO-MAINTENANCE: Model saved ({len(df)} samples, {len(LITE_FEATURES)} features)")


# =============================================================================
# SCHEDULER REGISTRATION
# =============================================================================

def register_ml_jobs(scheduler):
    """
    Register ML maintenance jobs with the alert scheduler.

    Call this after scheduler is created but before start().

    Args:
        scheduler: AlertScheduler instance
    """
    if not scheduler.scheduler:
        logger.warning("AUTO-MAINTENANCE: Scheduler not available, skipping ML job registration")
        return

    from apscheduler.triggers.cron import CronTrigger

    # Daily return backfill at 22:30 Zurich (after US market close ~22:00 CET)
    scheduler.scheduler.add_job(
        backfill_returns_job,
        CronTrigger(hour=22, minute=30),
        id='ml_return_backfill',
        name='ML Return Backfill',
        replace_existing=True,
        misfire_grace_time=3600,  # Allow 1h delay if missed
    )
    logger.info("AUTO-MAINTENANCE: Scheduled return backfill at 22:30 Zurich")

    # Weekly model retrain on Sunday at 03:00 Zurich
    scheduler.scheduler.add_job(
        retrain_model_job,
        CronTrigger(day_of_week='sun', hour=3, minute=0),
        id='ml_model_retrain',
        name='ML Model Retrain',
        replace_existing=True,
        misfire_grace_time=7200,  # Allow 2h delay if missed
    )
    logger.info("AUTO-MAINTENANCE: Scheduled model retrain at Sunday 03:00 Zurich")


# =============================================================================
# MANUAL RUN
# =============================================================================

if __name__ == '__main__':
    """Run maintenance jobs manually."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    import argparse
    parser = argparse.ArgumentParser(description='ML Auto-Maintenance')
    parser.add_argument('--backfill', action='store_true', help='Run return backfill')
    parser.add_argument('--retrain', action='store_true', help='Run model retrain')
    parser.add_argument('--both', action='store_true', help='Run both')
    args = parser.parse_args()

    if args.backfill or args.both:
        backfill_returns_job()
    if args.retrain or args.both:
        retrain_model_job()
    if not (args.backfill or args.retrain or args.both):
        print("Usage:")
        print("  python -m src.ml.auto_maintenance --backfill")
        print("  python -m src.ml.auto_maintenance --retrain")
        print("  python -m src.ml.auto_maintenance --both")