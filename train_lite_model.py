"""
Train "Lite" model using only features with full historical coverage.

Uses: sentiment_score, fundamental_score, gap_score + fundamental_missing
On: Full 25K+ dataset (Feb 2024 - Jan 2026)
With: 3-way time-split calibration (Phase 2)

This produces a working model NOW. When enough full-feature data accumulates
(~6000+ samples, est. March/April 2026), retrain with all 13 features.

Usage: python train_lite_model.py
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, date
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
import xgboost as xgb
import pickle
import os

from src.ml.db_helper import get_engine
from src.ml.signal_predictor import MLSignalPredictor, SignalDataLoader

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

LITE_FEATURES = [
    'sentiment_score',
    'fundamental_score',
    'gap_score',
    'fundamental_missing',
]

FEATURE_DEFAULTS = {
    'sentiment_score': 50,
    'fundamental_score': 50,
    'gap_score': 50,
    'fundamental_missing': 0,
}

FEATURE_CLIPS = {
    'sentiment_score': (0, 100),
    'fundamental_score': (0, 100),
    'gap_score': (0, 100),
    'fundamental_missing': (0, 1),
}

MODEL_PATH = 'models/signal_predictor.pkl'
BACKUP_PATH = 'models/signal_predictor_phase2_full.pkl'
COST_PCT = 0.15
MIN_PROBABILITY = 0.55


def load_data():
    """Load full dataset with returns."""
    engine = get_engine()
    query = """
        SELECT
            h.score_date as date,
            h.ticker,
            h.sector,
            COALESCE(h.sentiment, 50) as sentiment_score,
            COALESCE(h.fundamental_score, 50) as fundamental_score,
            COALESCE(CASE WHEN h.gap_score ~ '^[0-9.\\-]+$'
                     THEN h.gap_score::numeric ELSE NULL END, 50) as gap_score,
            CASE WHEN h.fundamental_score IS NULL THEN 1 ELSE 0 END as fundamental_missing,
            h.op_price as entry_price,
            h.return_1d,
            h.return_5d,
            h.return_10d,
            h.return_20d,
            h.signal_type
        FROM historical_scores h
        WHERE h.return_5d IS NOT NULL
          AND h.op_price IS NOT NULL AND h.op_price > 0
        ORDER BY h.score_date, h.ticker
    """
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} samples from {df['date'].min()} to {df['date'].max()}")
    logger.info(f"  Tickers: {df['ticker'].nunique()}")
    logger.info(f"  fundamental_missing: {df['fundamental_missing'].mean()*100:.1f}%")

    # Approximate 2d return
    df['return_2d'] = df['return_1d'] + (df['return_5d'] - df['return_1d']) * (2-1)/(5-1)

    # Cost-adjusted returns and binary labels
    for h in [1, 2, 5, 10]:
        col = f'return_{h}d'
        if col in df.columns:
            df[f'{col}_net'] = df[col] - COST_PCT
            df[f'win_{h}d'] = (df[f'{col}_net'] > 0).astype(int)

    # Fill and clip
    for col in LITE_FEATURES:
        default = FEATURE_DEFAULTS.get(col, 50)
        df[col] = df[col].fillna(default)
        clip_range = FEATURE_CLIPS.get(col)
        if clip_range:
            df[col] = df[col].clip(clip_range[0], clip_range[1])

    return df


def walk_forward_evaluate(df, X, y, horizon, n_splits=4):
    """3-way walk-forward: train -> calibrate -> test."""
    dates = np.sort(df['date'].unique())
    total = len(dates)
    test_size = total // 6  # Smaller test windows = more folds
    cal_size = max(test_size // 2, 20)
    purge = 5

    results = []
    for fold in range(n_splits):
        test_end = total - fold * test_size
        test_start = test_end - test_size
        cal_end = test_start - purge
        cal_start = cal_end - cal_size
        train_end = cal_start - purge

        if train_end <= 50:
            continue

        train_dates = dates[:train_end]
        cal_dates = dates[cal_start:cal_end]
        test_dates = dates[test_start:test_end]

        train_mask = df['date'].isin(train_dates)
        cal_mask = df['date'].isin(cal_dates)
        test_mask = df['date'].isin(test_dates)

        X_train, y_train = X[train_mask], y[train_mask]
        X_cal, y_cal = X[cal_mask], y[cal_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(X_train) < 100 or len(X_cal) < 20 or len(X_test) < 20:
            continue

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            min_child_weight=20,  # Prevent overfitting on 4 features
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, eval_metric='logloss')
        model.fit(X_train, y_train)

        # Calibrate on separate set
        cal_method = 'sigmoid' if len(X_cal) < 100 else 'isotonic'
        try:
            calibrator = CalibratedClassifierCV(model, cv='prefit', method=cal_method)
            calibrator.fit(X_cal, y_cal)
            probs = calibrator.predict_proba(X_test)[:, 1]
        except:
            probs = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
        brier = brier_score_loss(y_test, probs)

        trade_mask = probs >= MIN_PROBABILITY
        trades = int(trade_mask.sum())
        win_rate = float(y_test[trade_mask].mean()) if trades > 0 else 0.0

        # Also check calibration: predicted vs actual in buckets
        buckets = [(0.4, 0.5), (0.5, 0.55), (0.55, 0.65), (0.65, 0.8)]
        cal_info = []
        for lo, hi in buckets:
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() >= 5:
                actual = y_test[mask].mean()
                predicted = probs[mask].mean()
                cal_info.append(f"[{lo:.0%}-{hi:.0%}] pred={predicted:.1%} act={actual:.1%} n={mask.sum()}")

        results.append({
            'fold': fold,
            'train_n': len(X_train), 'cal_n': len(X_cal), 'test_n': len(X_test),
            'auc': auc, 'brier': brier,
            'trades': trades, 'win_rate': win_rate,
            'cal_method': cal_method,
            'period': f"{train_dates[-1]} → {test_dates[0]}..{test_dates[-1]}"
        })

        logger.info(f"  Fold {fold}: AUC={auc:.4f}  Brier={brier:.4f}  "
                    f"Trades={trades}  WR={win_rate:.1%}  ({len(X_train)}/{len(X_cal)}/{len(X_test)})")
        for ci in cal_info:
            logger.info(f"    {ci}")

    return results


def train_final_model(df, X, y, horizon):
    """Train final model with calibration holdout."""
    dates = np.sort(df['date'].unique())
    n_dates = len(dates)

    cal_size = max(int(n_dates * 0.15), 30)
    purge = 5

    cal_dates = dates[n_dates - cal_size:]
    train_dates = dates[:n_dates - cal_size - purge]

    train_mask = df['date'].isin(train_dates)
    cal_mask = df['date'].isin(cal_dates)

    X_train, y_train = X[train_mask], y[train_mask]
    X_cal, y_cal = X[cal_mask], y[cal_mask]

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)

    importance = dict(zip(LITE_FEATURES, model.feature_importances_))

    cal_method = 'sigmoid' if len(X_cal) < 100 else 'isotonic'
    try:
        calibrator = CalibratedClassifierCV(model, cv='prefit', method=cal_method)
        calibrator.fit(X_cal, y_cal)
        logger.info(f"  Final {horizon}d: trained={len(X_train)} calibrated={len(X_cal)} ({cal_method})")
    except:
        calibrator = model
        logger.warning(f"  Final {horizon}d: calibration failed, using raw model")

    return model, calibrator, importance


def main():
    print("=" * 60)
    print("LITE MODEL TRAINING")
    print(f"Features: {', '.join(LITE_FEATURES)}")
    print("=" * 60)

    # Load data
    df = load_data()

    # Quick signal check
    print(f"\n--- Signal Check ---")
    for feat in ['sentiment_score', 'fundamental_score', 'gap_score']:
        q25, q75 = df[feat].quantile(0.25), df[feat].quantile(0.75)
        low_wr = (df[df[feat] <= q25]['return_5d_net'] > 0).mean()
        high_wr = (df[df[feat] >= q75]['return_5d_net'] > 0).mean()
        direction = "CONTRARIAN" if low_wr > high_wr else "MOMENTUM"
        print(f"  {feat:25s}  low_quartile_WR={low_wr:.1%}  high_quartile_WR={high_wr:.1%}  ({direction})")

    missing_wr = (df[df['fundamental_missing']==1]['return_5d_net'] > 0).mean()
    present_wr = (df[df['fundamental_missing']==0]['return_5d_net'] > 0).mean()
    print(f"  {'fundamental_missing':25s}  missing_WR={missing_wr:.1%}  present_WR={present_wr:.1%}")

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[LITE_FEATURES].values)

    # Calculate return stats for EV
    avg_win_return = {}
    avg_loss_return = {}
    for h in [1, 2, 5, 10]:
        col = f'return_{h}d_net'
        if col in df.columns:
            returns = df[col].dropna()
            avg_win_return[h] = returns[returns > 0].mean() if (returns > 0).any() else 2.0
            avg_loss_return[h] = abs(returns[returns <= 0].mean()) if (returns <= 0).any() else 2.0

    print(f"\n--- Return Stats ---")
    for h in [5, 10]:
        print(f"  {h}d: avg_win={avg_win_return.get(h,0):.3f}%  avg_loss={avg_loss_return.get(h,0):.3f}%")

    # Walk-forward evaluation
    all_folds = {}
    models = {}
    calibrators = {}
    feature_importance = {}

    for horizon in [5, 10]:
        target_col = f'win_{horizon}d'
        if target_col not in df.columns:
            continue

        y = df[target_col].values
        base_rate = y.mean()
        print(f"\n{'='*60}")
        print(f"HORIZON: {horizon}d  (base rate: {base_rate:.1%})")
        print(f"{'='*60}")

        folds = walk_forward_evaluate(df, X, y, horizon, n_splits=4)
        all_folds[horizon] = folds

        if folds:
            mean_auc = np.mean([f['auc'] for f in folds])
            mean_brier = np.mean([f['brier'] for f in folds])
            total_trades = sum(f['trades'] for f in folds)
            mean_wr = np.mean([f['win_rate'] for f in folds if f['trades'] > 0]) if any(f['trades'] > 0 for f in folds) else 0

            print(f"\n  Summary: AUC={mean_auc:.4f}  Brier={mean_brier:.4f}  "
                  f"Total Trades={total_trades}  Mean WR={mean_wr:.1%}")

        # Train final model
        model, calibrator, importance = train_final_model(df, X, y, horizon)
        models[horizon] = model
        calibrators[horizon] = calibrator
        feature_importance[horizon] = importance

    # Build and save via MLSignalPredictor
    predictor = MLSignalPredictor()
    predictor.models = models
    predictor.calibrators = calibrators
    predictor.feature_names = LITE_FEATURES
    predictor.feature_importance = feature_importance
    predictor.avg_win_return = avg_win_return
    predictor.avg_loss_return = avg_loss_return
    predictor.data_loader.scaler = scaler

    # Update the data_loader defaults/clips to match lite features
    predictor.data_loader.FEATURE_DEFAULTS = FEATURE_DEFAULTS
    predictor.data_loader.FEATURE_CLIPS = FEATURE_CLIPS
    predictor.data_loader.MISSING_INDICATOR_MAP = {
        'fundamental_score': 'fundamental_missing',
    }

    # Backup current model
    if os.path.exists(MODEL_PATH):
        import shutil
        shutil.copy2(MODEL_PATH, BACKUP_PATH)
        logger.info(f"Backed up current model to {BACKUP_PATH}")

    predictor.save(MODEL_PATH)

    # Print final summary
    print(f"\n{'='*60}")
    print(f"LITE MODEL SAVED")
    print(f"{'='*60}")
    print(f"Samples: {len(df)}")
    print(f"Features: {len(LITE_FEATURES)} ({', '.join(LITE_FEATURES)})")

    combined_imp = {}
    for h, imp in feature_importance.items():
        for feat, val in imp.items():
            combined_imp[feat] = combined_imp.get(feat, 0) + val
    total = sum(combined_imp.values()) or 1
    combined_imp = {k: v/total for k, v in combined_imp.items()}

    print(f"\nFeature Importance (combined):")
    for feat, imp in sorted(combined_imp.items(), key=lambda x: -x[1]):
        print(f"  {feat:25s} {imp:.4f}")

    # Quick test prediction
    print(f"\n--- Test Prediction ---")
    test_scores = {
        'ticker': 'TEST',
        'sentiment_score': 70,
        'fundamental_score': 65,
        'gap_score': 55,
    }
    result = predictor.predict(test_scores)
    print(f"  Bullish test:  prob_5d={result.prob_win_5d:.3f}  ev_5d={result.ev_5d:.4f}  should_trade={result.should_trade}")

    test_scores2 = {
        'ticker': 'TEST2',
        'sentiment_score': 35,
        'fundamental_score': 40,
        'gap_score': 70,
    }
    result2 = predictor.predict(test_scores2)
    print(f"  Bearish test:  prob_5d={result2.prob_win_5d:.3f}  ev_5d={result2.ev_5d:.4f}  should_trade={result2.should_trade}")

    test_scores3 = {
        'ticker': 'TEST3',
        'sentiment_score': 50,
        'fundamental_score': None,
        'gap_score': 50,
    }
    result3 = predictor.predict(test_scores3)
    print(f"  Neutral+miss:  prob_5d={result3.prob_win_5d:.3f}  ev_5d={result3.ev_5d:.4f}  should_trade={result3.should_trade}")

    print(f"\nModel ready. Restart Streamlit to use new model.")


if __name__ == '__main__':
    main()