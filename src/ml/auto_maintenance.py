"""
ML Auto-Maintenance - Phase 3

Automated jobs for return backfill and model retraining.
Hooks into the existing APScheduler in src/alerts/scheduler.py.

Jobs:
- Daily return backfill (22:30 Zurich, after US market close)
- Weekly model retrain (Sunday 03:00 Zurich)

Phase 3 changes:
- File lock (_FileLock) prevents duplicate retrain runs from Streamlit reloads
- Deploy gate: only replace model if new AUC/Brier improves over current
- 3-way time split in _do_retrain (train | purge | cal | purge | test)
- Removed global StandardScaler from retrain path (XGBoost is scale-invariant)
- EV stats from GROSS returns conditioned on NET-win label (no double-count)
- Real ModelReport with actual test-set AUC/Brier (no more placeholder 0.5/0.25)
- executemany for backfill DB writes (faster + fewer round trips)
- 180d fetch window for reliable 20-trading-day forward returns
- Pinned Europe/Zurich timezone in scheduler registration

Usage:
    from src.ml.auto_maintenance import register_ml_jobs
    register_ml_jobs(scheduler)

Location: src/ml/auto_maintenance.py
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
# FILE LOCK (prevents duplicate retrain from Streamlit reloads)
# =============================================================================

class _FileLock:
    """
    Simple exclusive file lock using OS-level O_CREAT | O_EXCL.
    Raises FileExistsError if lock is already held (unless stale).
    """
    def __init__(self, lock_path: str, stale_seconds: int = 6 * 3600):
        self.lock_path = lock_path
        self.stale_seconds = stale_seconds
        self.fd = None

    def __enter__(self):
        # Clear stale lock (e.g. left by crashed process)
        if os.path.exists(self.lock_path):
            age = time.time() - os.path.getmtime(self.lock_path)
            if age > self.stale_seconds:
                try:
                    os.remove(self.lock_path)
                except Exception:
                    pass

        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
        self.fd = os.open(self.lock_path, flags)
        os.write(self.fd, str(os.getpid()).encode("utf-8"))
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.fd is not None:
                os.close(self.fd)
        finally:
            try:
                if os.path.exists(self.lock_path):
                    os.remove(self.lock_path)
            except Exception:
                pass


# =============================================================================
# RETURN BACKFILL
# =============================================================================

def backfill_returns_job():
    """
    Daily job: fill missing return_1d/5d/10d/20d in historical_scores.
    Uses executemany for speed + fewer DB round trips.
    Fetches 180d history per ticker to reliably compute 20-trading-day returns.
    """
    logger.info("=" * 50)
    logger.info("AUTO-MAINTENANCE: Starting return backfill")
    logger.info("=" * 50)

    try:
        engine = get_engine()
        cutoff_date = (date.today() - timedelta(days=BACKFILL_LOOKBACK_DAYS)).isoformat()

        # Find rows needing backfill (check ALL return columns, not just 1d/5d)
        query = f"""
            SELECT id, score_date, ticker, op_price,
                   return_1d, return_5d, return_10d, return_20d
            FROM historical_scores
            WHERE op_price IS NOT NULL AND op_price > 0
              AND (return_1d IS NULL OR return_5d IS NULL
                   OR return_10d IS NULL OR return_20d IS NULL)
              AND score_date >= '{cutoff_date}'
            ORDER BY score_date DESC, ticker
        """
        rows = pd.read_sql(query, engine)

        if rows.empty:
            logger.info("AUTO-MAINTENANCE: No rows need backfill")
            return

        tickers = rows['ticker'].dropna().unique().tolist()
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

        # Prepare parameterized update statement (COALESCE preserves existing values)
        sql = """
            UPDATE historical_scores
            SET
              price_1d   = COALESCE(%s, price_1d),
              return_1d  = COALESCE(%s, return_1d),
              price_5d   = COALESCE(%s, price_5d),
              return_5d  = COALESCE(%s, return_5d),
              price_10d  = COALESCE(%s, price_10d),
              return_10d = COALESCE(%s, return_10d),
              price_20d  = COALESCE(%s, price_20d),
              return_20d = COALESCE(%s, return_20d)
            WHERE id = %s
        """

        conn = engine.raw_connection()
        cur = conn.cursor()

        try:
            for ticker in tickers:
                trows = rows[rows['ticker'] == ticker]
                if trows.empty:
                    continue

                # Fetch enough history (20 trading days forward ≈ 30 calendar days + buffer)
                hist = get_stock_history(ticker, period="180d", timeout=12)
                if hist is None or hist.empty:
                    failed.append(ticker)
                    continue

                # Build date -> close price map
                closes = {}
                for _, r in hist.iterrows():
                    d = r.get('date', r.name)
                    if isinstance(d, str):
                        d = d[:10]
                    try:
                        d = pd.Timestamp(d).date()
                    except Exception:
                        continue
                    try:
                        closes[d] = float(r['Close'])
                    except Exception:
                        continue

                if not closes:
                    failed.append(ticker)
                    continue

                sorted_dates = sorted(closes.keys())

                # Build batch of (p1, r1, p5, r5, p10, r10, p20, r20, id) tuples
                batch_params = []
                for _, row in trows.iterrows():
                    score_date = row['score_date']
                    if hasattr(score_date, 'date'):
                        score_date = score_date.date()
                    op_price = float(row['op_price'])

                    future_dates = [d for d in sorted_dates if d > score_date]
                    if not future_dates:
                        continue

                    # For each horizon, compute price + return if currently NULL
                    p1 = r1 = p5 = r5 = p10 = r10 = p20 = r20 = None

                    def _calc(n_days):
                        if len(future_dates) >= n_days:
                            td = future_dates[n_days - 1]
                            px = closes.get(td)
                            if px and px > 0:
                                ret = (px / op_price - 1.0) * 100.0
                                return round(px, 2), round(ret, 4)
                        return None, None

                    if pd.isna(row['return_1d']):
                        p1, r1 = _calc(1)
                    if pd.isna(row['return_5d']):
                        p5, r5 = _calc(5)
                    if pd.isna(row['return_10d']):
                        p10, r10 = _calc(10)
                    if pd.isna(row['return_20d']):
                        p20, r20 = _calc(20)

                    if any(v is not None for v in [p1, r1, p5, r5, p10, r10, p20, r20]):
                        batch_params.append((p1, r1, p5, r5, p10, r10, p20, r20, int(row['id'])))

                if batch_params:
                    cur.executemany(sql, batch_params)
                    conn.commit()
                    total_updated += len(batch_params)

                # Small delay to avoid rate limits
                time.sleep(0.3)

        finally:
            try:
                cur.close()
                conn.close()
            except Exception:
                pass

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
    Adds:
      - file lock (avoid overlapping retrains from Streamlit reloads)
      - atomic save (done by MLSignalPredictor.save)
      - deploy gate: only replace model if metrics improve vs current
    """
    logger.info("=" * 50)
    logger.info("AUTO-MAINTENANCE: Starting model retrain check")
    logger.info("=" * 50)

    lock_path = os.path.join(BACKUP_DIR, '.retrain.lock')
    os.makedirs(BACKUP_DIR, exist_ok=True)

    try:
        with _FileLock(lock_path, stale_seconds=6 * 3600):
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
            total_samples = int(pd.read_sql(count_query, engine).iloc[0]['n'])

            # Extract current model's metrics for deploy gate comparison
            current_auc = None
            current_brier = None
            current_samples = 0

            if os.path.exists(MODEL_PATH):
                try:
                    import pickle
                    with open(MODEL_PATH, 'rb') as f:
                        state = pickle.load(f)
                    report = state.get('validation_report')
                    if report:
                        current_samples = int(getattr(report, 'total_samples', 0) or 0)
                        ca = getattr(report, 'mean_auc', None)
                        cb = getattr(report, 'mean_brier', None)
                        current_auc = float(ca) if ca is not None else None
                        current_brier = float(cb) if cb is not None else None
                except Exception:
                    pass

            new_samples = total_samples - current_samples
            logger.info(f"AUTO-MAINTENANCE: DB has {total_samples} samples, "
                       f"model trained on {current_samples}, delta={new_samples}")

            if new_samples < MIN_SAMPLES_FOR_RETRAIN:
                logger.info(f"AUTO-MAINTENANCE: Not enough new data ({new_samples} < {MIN_SAMPLES_FOR_RETRAIN})")
                return

            # Backup current model
            if os.path.exists(MODEL_PATH):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = os.path.join(BACKUP_DIR, f'signal_predictor_{timestamp}.pkl')
                import shutil
                shutil.copy2(MODEL_PATH, backup_path)
                logger.info(f"AUTO-MAINTENANCE: Backed up model to {backup_path}")

                # Keep only last 5 backups
                backups = sorted(Path(BACKUP_DIR).glob('signal_predictor_*.pkl'))
                for old_backup in backups[:-5]:
                    old_backup.unlink(missing_ok=True)

            # Retrain with deploy gate
            logger.info("AUTO-MAINTENANCE: Retraining model...")
            improved = _do_retrain(
                engine,
                expected_samples=total_samples,
                current_auc=current_auc,
                current_brier=current_brier,
            )

            if not improved:
                logger.info("AUTO-MAINTENANCE: New model not better — keeping current model")
                return

            # Update cooldown only if model was actually deployed
            Path(cooldown_file).touch()
            logger.info("AUTO-MAINTENANCE: Retrain complete (model updated)")

    except FileExistsError:
        logger.info("AUTO-MAINTENANCE: Retrain already running (lock exists), skipping")
    except Exception as e:
        logger.error(f"AUTO-MAINTENANCE: Retrain error: {e}")
        logger.debug(traceback.format_exc())


def _do_retrain(engine, expected_samples, current_auc=None, current_brier=None) -> bool:
    """
    Execute model retraining with:
    - 3-way chronological split (train | purge | cal | purge | test)
    - No global scaler (XGBoost is scale-invariant)
    - EV from gross returns conditioned on net-win label
    - Deploy gate: only return True if new metrics improve over current

    Returns True if the newly trained model was deployed, False otherwise.
    """
    import xgboost as xgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score, brier_score_loss

    # Keep aligned with production feature coverage
    LITE_FEATURES = ['sentiment_score', 'fundamental_score', 'gap_score', 'fundamental_missing']

    FEATURE_DEFAULTS = {
        'sentiment_score': 50, 'fundamental_score': 50,
        'gap_score': 50, 'fundamental_missing': 0,
    }
    FEATURE_CLIPS = {
        'sentiment_score': (0, 100), 'fundamental_score': (0, 100),
        'gap_score': (0, 100), 'fundamental_missing': (0, 1),
    }
    COST_PCT = 0.15
    purge = 5

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
        return False

    # Prepare returns and labels
    df['return_2d'] = df['return_1d'] + (df['return_5d'] - df['return_1d']) * 0.25
    for h in [1, 2, 5, 10]:
        col = f'return_{h}d'
        if col in df.columns:
            df[f'{col}_net'] = df[col] - COST_PCT
            df[f'win_{h}d'] = (df[f'{col}_net'] > 0).astype(int)

    # Fill and clip features
    for col in LITE_FEATURES:
        df[col] = df[col].fillna(FEATURE_DEFAULTS.get(col, 50))
        lo, hi = FEATURE_CLIPS.get(col, (None, None))
        if lo is not None:
            df[col] = df[col].clip(lo, hi)

    # Phase 3: NO global scaler — raw features go straight to XGBoost
    X = df[LITE_FEATURES].values.astype(float, copy=False)

    # EV stats: GROSS returns conditioned on NET-win label
    avg_win_return = {}
    avg_loss_return = {}
    for h in [1, 2, 5, 10]:
        win_col = f'win_{h}d'
        ret_col = f'return_{h}d'
        if win_col in df.columns and ret_col in df.columns:
            r = df[ret_col].astype(float)
            w = df[win_col].astype(int)
            wins = r[w == 1].dropna()
            losses = r[w == 0].dropna()
            avg_win_return[h] = float(wins.mean()) if len(wins) else 2.0
            avg_loss_return[h] = float(abs(losses.mean())) if len(losses) else 2.0

    # --- Chronological 3-way split: TRAIN | purge | CAL | purge | TEST ---
    dates = np.sort(df['date'].unique())
    n_dates = len(dates)
    if n_dates < 120:
        logger.warning("AUTO-MAINTENANCE: Not enough unique dates for safe 3-way split")
        return False

    test_size = max(int(n_dates * 0.15), 30)
    cal_size = max(int(n_dates * 0.15), 30)

    test_dates = dates[-test_size:]
    cal_end_idx = n_dates - test_size - purge
    cal_start_idx = max(cal_end_idx - cal_size, 0)
    cal_dates = dates[cal_start_idx:cal_end_idx]

    train_end_idx = cal_start_idx - purge
    if train_end_idx <= 0:
        logger.warning("AUTO-MAINTENANCE: Split invalid (train too small)")
        return False
    train_dates = dates[:train_end_idx]

    train_mask = df['date'].isin(train_dates)
    cal_mask = df['date'].isin(cal_dates)
    test_mask = df['date'].isin(test_dates)

    X_train, X_cal, X_test = X[train_mask], X[cal_mask], X[test_mask]

    if len(X_train) < 200 or len(X_cal) < 50 or len(X_test) < 50:
        logger.warning("AUTO-MAINTENANCE: Split sizes too small for stable retrain")
        return False

    logger.info(f"AUTO-MAINTENANCE: Split — train={len(X_train)}, cal={len(X_cal)}, test={len(X_test)}")

    # Train 5d + 10d models
    models = {}
    calibrators = {}
    feature_importance = {}

    for horizon in [5, 10]:
        target_col = f'win_{horizon}d'
        if target_col not in df.columns:
            continue

        y_train = df.loc[train_mask, target_col].values.astype(int)
        y_cal = df.loc[cal_mask, target_col].values.astype(int)
        y_test = df.loc[test_mask, target_col].values.astype(int)

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, eval_metric='logloss',
        )
        model.fit(X_train, y_train)

        feature_importance[horizon] = dict(zip(LITE_FEATURES, model.feature_importances_))

        cal_method = 'sigmoid' if len(X_cal) < 100 else 'isotonic'
        try:
            calibrator = CalibratedClassifierCV(model, cv='prefit', method=cal_method)
            calibrator.fit(X_cal, y_cal)
            calibrators[horizon] = calibrator
        except Exception:
            calibrators[horizon] = model

        models[horizon] = model
        logger.info(f"AUTO-MAINTENANCE: Trained {horizon}d model "
                   f"(train={len(X_train)}, cal={len(X_cal)}, test={len(X_test)})")

    # --- Evaluate candidate on held-out TEST set ---
    y_test_5 = df.loc[test_mask, 'win_5d'].values.astype(int)
    try:
        p = calibrators[5].predict_proba(X_test)[:, 1]
        new_auc = roc_auc_score(y_test_5, p) if len(np.unique(y_test_5)) > 1 else 0.5
        new_brier = brier_score_loss(y_test_5, p)
    except Exception:
        new_auc = 0.5
        new_brier = 1.0

    logger.info(f"AUTO-MAINTENANCE: Candidate TEST metrics: AUC={new_auc:.4f}, Brier={new_brier:.4f}")
    if current_auc is not None:
        logger.info(f"AUTO-MAINTENANCE: Current  model metrics: AUC={current_auc:.4f}, Brier={current_brier:.4f}")

    # Compute real accuracy, win_rate, avg_return from test set
    new_acc = float(((p >= 0.5).astype(int) == y_test_5).mean())
    min_prob = 0.55
    trade_mask = p >= min_prob
    new_trades = int(trade_mask.sum())
    new_win_rate = float(y_test_5[trade_mask].mean()) if new_trades > 0 else float('nan')
    new_avg_return = float('nan')
    ret_net_col = 'return_5d_net'
    if new_trades > 0 and ret_net_col in df.columns:
        test_rets = df.loc[test_mask, ret_net_col].values.astype(float)
        traded_rets = test_rets[trade_mask]
        new_avg_return = float(np.nanmean(traded_rets)) if len(traded_rets) > 0 else float('nan')

    # --- Deploy gate ---
    # If current metrics exist: require a small improvement in AUC (+0.002) OR Brier (-0.002)
    # If no current model: require "not worse than baseline" + reasonable Brier
    if current_auc is not None and current_brier is not None:
        improved = (new_auc >= current_auc + 0.002) or (new_brier <= current_brier - 0.002)
        logger.info(f"AUTO-MAINTENANCE: Deploy gate (vs current): improved={improved}")
    else:
        improved = (new_auc >= 0.50) and (new_brier <= 0.27)
        logger.info(f"AUTO-MAINTENANCE: Deploy gate (vs baseline): improved={improved}")

    if not improved:
        return False

    # --- Save using MLSignalPredictor format (inherits atomic save) ---
    from src.ml.signal_predictor import MLSignalPredictor, ModelReport
    predictor = MLSignalPredictor()
    predictor.models = models
    predictor.calibrators = calibrators
    predictor.feature_names = LITE_FEATURES
    predictor.feature_importance = feature_importance
    predictor.avg_win_return = avg_win_return
    predictor.avg_loss_return = avg_loss_return

    # Phase 3: Real ModelReport with actual test-set metrics
    predictor.validation_report = ModelReport(
        model_name="XGBoost-Lite-Auto",
        trained_at=datetime.now(),
        total_samples=len(df),
        feature_count=len(LITE_FEATURES),
        target_horizon=5,
        folds=[],
        mean_accuracy=float(new_acc),
        mean_auc=float(new_auc),
        mean_brier=float(new_brier),
        mean_win_rate=float(new_win_rate) if not np.isnan(new_win_rate) else 0.0,
        mean_return=float(new_avg_return) if not np.isnan(new_avg_return) else 0.0,
        feature_importance=feature_importance.get(5, {}),
        is_well_calibrated=float(new_brier) < 0.25,
        calibration_error=float(new_brier),
        beats_baseline=float(new_auc) > 0.52,
        baseline_auc=0.5,
        improvement_vs_baseline=float(new_auc) - 0.5,
    )

    predictor.save(MODEL_PATH)
    logger.info(f"AUTO-MAINTENANCE: Model deployed ({len(df)} samples, {len(LITE_FEATURES)} features, "
               f"AUC={new_auc:.4f}, Brier={new_brier:.4f})")
    return True


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

    tz = "Europe/Zurich"

    # Daily return backfill at 22:30 Zurich (after US market close ~22:00 CET)
    scheduler.scheduler.add_job(
        backfill_returns_job,
        CronTrigger(hour=22, minute=30, timezone=tz),
        id='ml_return_backfill',
        name='ML Return Backfill',
        replace_existing=True,
        misfire_grace_time=3600,  # Allow 1h delay if missed
    )
    logger.info("AUTO-MAINTENANCE: Scheduled return backfill at 22:30 Zurich")

    # Weekly model retrain on Sunday at 03:00 Zurich
    scheduler.scheduler.add_job(
        retrain_model_job,
        CronTrigger(day_of_week='sun', hour=3, minute=0, timezone=tz),
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