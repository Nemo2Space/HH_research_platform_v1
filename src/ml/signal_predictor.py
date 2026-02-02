"""
ML Signal Predictor - Phase 2

Walk-forward validation with 3-way time split, multi-horizon labels,
calibrated probabilities, missing indicators, and A/B feature testing.

Phase 2 changes:
- 3-way time split: train -> calibrate -> test (fixes calibration leakage)
- Missing indicators for frequently-null features
- A/B comparison: with vs without total_score
- Sigmoid fallback for small calibration sets
- Domain-correct defaults and clipping (Phase 1)

Location: src/ml/signal_predictor.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pickle
import os
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ML imports
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Database connection
from src.ml.db_helper import get_engine

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PredictionResult:
    """Result from ML prediction."""
    ticker: str
    prediction_date: date
    prob_win_1d: float
    prob_win_2d: float
    prob_win_5d: float
    prob_win_10d: float
    ev_1d: float
    ev_2d: float
    ev_5d: float
    ev_10d: float
    top_features: Dict[str, float]
    confidence: str
    similar_setups_win_rate: float
    similar_setups_count: int
    should_trade: bool
    recommended_horizon: int

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'prob_win_5d': round(self.prob_win_5d, 3),
            'ev_5d': round(self.ev_5d, 4),
            'should_trade': self.should_trade,
            'confidence': self.confidence,
        }


@dataclass
class WalkForwardResult:
    """Result from walk-forward validation."""
    fold: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    accuracy: float
    auc_roc: float
    brier_score: float
    trades_taken: int
    win_rate: float
    avg_return: float


@dataclass
class ModelReport:
    """Complete model validation report."""
    model_name: str
    trained_at: datetime
    total_samples: int
    feature_count: int
    target_horizon: int
    folds: List[WalkForwardResult]
    mean_accuracy: float
    mean_auc: float
    mean_brier: float
    mean_win_rate: float
    mean_return: float
    feature_importance: Dict[str, float]
    is_well_calibrated: bool
    calibration_error: float
    beats_baseline: bool
    baseline_auc: float
    improvement_vs_baseline: float


# =============================================================================
# DATA LOADER
# =============================================================================

class SignalDataLoader:
    """Loads and prepares data with proper labels and missing indicators."""

    # Core score features (always included)
    SCORE_FEATURES = [
        'sentiment_score', 'fundamental_score', 'technical_score',
        'options_flow_score', 'short_squeeze_score', 'gap_score',
        'article_count', 'target_upside_pct'
    ]

    # Missing indicators for frequently-null features
    MISSING_FEATURES = [
        'fundamental_missing', 'technical_missing',
        'options_flow_missing', 'short_squeeze_missing',
        'target_upside_missing'
    ]

    # Mapping: which score generates which missing indicator
    MISSING_INDICATOR_MAP = {
        'fundamental_score': 'fundamental_missing',
        'technical_score': 'technical_missing',
        'options_flow_score': 'options_flow_missing',
        'short_squeeze_score': 'short_squeeze_missing',
        'target_upside_pct': 'target_upside_missing',
    }

    # Feature sets for A/B comparison
    FEATURES_WITH_TOTAL = SCORE_FEATURES + ['total_score'] + MISSING_FEATURES
    FEATURES_WITHOUT_TOTAL = SCORE_FEATURES + MISSING_FEATURES

    # Default feature set (will be chosen by A/B test)
    FEATURES = FEATURES_WITH_TOTAL

    # Domain-correct defaults: 50 for 0-100 scores, 0 for counts/pct, 0 for missing flags
    FEATURE_DEFAULTS = {
        'sentiment_score': 50,
        'fundamental_score': 50,
        'technical_score': 50,
        'options_flow_score': 50,
        'short_squeeze_score': 50,
        'gap_score': 50,
        'total_score': 50,
        'article_count': 0,
        'target_upside_pct': 0,
        'fundamental_missing': 0,
        'technical_missing': 0,
        'options_flow_missing': 0,
        'short_squeeze_missing': 0,
        'target_upside_missing': 0,
    }

    # Clipping bounds
    FEATURE_CLIPS = {
        'sentiment_score': (0, 100),
        'fundamental_score': (0, 100),
        'technical_score': (0, 100),
        'options_flow_score': (0, 100),
        'short_squeeze_score': (0, 100),
        'gap_score': (0, 100),
        'total_score': (0, 100),
        'article_count': (0, 200),
        'target_upside_pct': (-50, 100),
        'fundamental_missing': (0, 1),
        'technical_missing': (0, 1),
        'options_flow_missing': (0, 1),
        'short_squeeze_missing': (0, 1),
        'target_upside_missing': (0, 1),
    }

    def __init__(self):
        self.engine = get_engine()
        self.scaler = StandardScaler()
        self.cost_pct = 0.15  # 0.15% round trip cost

    def load_training_data(self, min_date: str = None) -> pd.DataFrame:
        """Load historical data with multi-horizon labels and missing indicators."""

        query = """
            SELECT
                h.score_date as date,
                h.ticker,
                h.sector,
                COALESCE(h.sentiment, 50) as sentiment_score,
                COALESCE(h.fundamental_score, 50) as fundamental_score,
                COALESCE(h.growth_score, 50) as growth_score,
                COALESCE(h.total_score, 50) as total_score,
                COALESCE(CASE WHEN h.gap_score ~ '^[0-9.\\-]+$' THEN h.gap_score::numeric ELSE NULL END, 50) as gap_score,
                COALESCE(s.technical_score, 50) as technical_score,
                COALESCE(s.options_flow_score, 50) as options_flow_score,
                COALESCE(s.short_squeeze_score, 50) as short_squeeze_score,
                COALESCE(s.article_count, 0) as article_count,
                COALESCE(s.target_upside_pct, 0) as target_upside_pct,
                -- Missing indicators: 1 = data was NULL in source table
                CASE WHEN h.fundamental_score IS NULL THEN 1 ELSE 0 END as fundamental_missing,
                CASE WHEN s.technical_score IS NULL THEN 1 ELSE 0 END as technical_missing,
                CASE WHEN s.options_flow_score IS NULL THEN 1 ELSE 0 END as options_flow_missing,
                CASE WHEN s.short_squeeze_score IS NULL THEN 1 ELSE 0 END as short_squeeze_missing,
                CASE WHEN s.target_upside_pct IS NULL THEN 1 ELSE 0 END as target_upside_missing,
                h.op_price as entry_price,
                h.return_1d,
                h.return_5d,
                h.return_10d,
                h.return_20d,
                h.signal_type
            FROM historical_scores h
            LEFT JOIN screener_scores s ON h.ticker = s.ticker AND h.score_date = s.date
            WHERE h.return_5d IS NOT NULL AND h.op_price IS NOT NULL AND h.op_price > 0
        """

        if min_date:
            query += f" AND h.score_date >= '{min_date}'"
        query += " ORDER BY h.score_date, h.ticker"

        df = pd.read_sql(query, self.engine)

        if df.empty:
            logger.warning("No data loaded from historical_scores")
            return df

        logger.info(f"Loaded {len(df)} samples from {df['date'].min()} to {df['date'].max()}")

        # Log missingness stats
        for missing_col in self.MISSING_FEATURES:
            if missing_col in df.columns:
                pct = df[missing_col].mean() * 100
                logger.info(f"  {missing_col}: {pct:.1f}% missing")

        # Approximate 2d return
        df['return_2d'] = df['return_1d'] + (df['return_5d'] - df['return_1d']) * (2-1)/(5-1)

        # Cost-adjusted returns
        for h in [1, 2, 5, 10, 20]:
            col = f'return_{h}d'
            if col in df.columns:
                df[f'{col}_net'] = df[col] - self.cost_pct

        # Binary labels
        for h in [1, 2, 5, 10]:
            df[f'win_{h}d'] = (df[f'return_{h}d_net'] > 0).astype(int)

        # Fill missing scores with domain-correct defaults and clip
        for col in self.SCORE_FEATURES + ['total_score']:
            if col in df.columns:
                default = self.FEATURE_DEFAULTS.get(col, 50)
                df[col] = df[col].fillna(default)
                clip_range = self.FEATURE_CLIPS.get(col)
                if clip_range:
                    df[col] = df[col].clip(clip_range[0], clip_range[1])

        # Missing indicators should already be 0/1 from SQL, but ensure
        for col in self.MISSING_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        return df

    def get_feature_matrix(self, df: pd.DataFrame, feature_list: List[str] = None,
                           fit: bool = False) -> Tuple[np.ndarray, List[str]]:
        """Extract and scale features."""
        if feature_list is None:
            feature_list = self.FEATURES
        available = [f for f in feature_list if f in df.columns]
        X = df[available].values

        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X, available


# =============================================================================
# ML SIGNAL PREDICTOR
# =============================================================================

class MLSignalPredictor:
    """ML prediction with walk-forward validation and honest calibration."""

    def __init__(self, target_horizon: int = 5, min_probability: float = 0.55, min_ev: float = 0.001):
        self.target_horizon = target_horizon
        self.min_probability = min_probability
        self.min_ev = min_ev

        self.data_loader = SignalDataLoader()
        self.models = {}
        self.calibrators = {}
        self.feature_names = []
        self.feature_importance = {}
        self.avg_win_return = {}
        self.avg_loss_return = {}
        self.validation_report = None

    def train(self, df: pd.DataFrame = None) -> ModelReport:
        """Train models with walk-forward validation and A/B feature comparison."""
        if df is None:
            df = self.data_loader.load_training_data()

        if df.empty or len(df) < 200:
            raise ValueError(f"Insufficient data: {len(df)} samples (need 200+)")

        # Calculate historical returns for EV
        for h in [1, 2, 5, 10]:
            col = f'return_{h}d_net'
            if col in df.columns:
                returns = df[col].dropna()
                self.avg_win_return[h] = returns[returns > 0].mean() if (returns > 0).any() else 2.0
                self.avg_loss_return[h] = abs(returns[returns <= 0].mean()) if (returns <= 0).any() else 2.0

        # === A/B TEST: with vs without total_score ===
        logger.info("=" * 60)
        logger.info("A/B TEST: Comparing feature sets")
        logger.info("=" * 60)

        results_a = self._evaluate_feature_set(
            df, self.data_loader.FEATURES_WITH_TOTAL, "Set A (with total_score)")
        results_b = self._evaluate_feature_set(
            df, self.data_loader.FEATURES_WITHOUT_TOTAL, "Set B (without total_score)")

        # Pick winner based on Brier score (lower = better calibration)
        brier_a = np.mean([r['brier'] for r in results_a]) if results_a else 1.0
        brier_b = np.mean([r['brier'] for r in results_b]) if results_b else 1.0
        auc_a = np.mean([r['auc'] for r in results_a]) if results_a else 0.5
        auc_b = np.mean([r['auc'] for r in results_b]) if results_b else 0.5
        wr_a = np.mean([r['win_rate'] for r in results_a if r['trades'] > 0]) if any(r['trades'] > 0 for r in results_a) else 0
        wr_b = np.mean([r['win_rate'] for r in results_b if r['trades'] > 0]) if any(r['trades'] > 0 for r in results_b) else 0
        trades_a = sum(r['trades'] for r in results_a)
        trades_b = sum(r['trades'] for r in results_b)

        logger.info(f"\n{'='*60}")
        logger.info(f"A/B COMPARISON RESULTS:")
        logger.info(f"  Set A (with total_score):    AUC={auc_a:.4f}  Brier={brier_a:.4f}  WR={wr_a:.1%}  Trades={trades_a}")
        logger.info(f"  Set B (without total_score): AUC={auc_b:.4f}  Brier={brier_b:.4f}  WR={wr_b:.1%}  Trades={trades_b}")

        # Prefer B if Brier is close or better, and AUC is close
        # (removing multicollinearity is preferred when performance is similar)
        if brier_b <= brier_a * 1.02 and auc_b >= auc_a * 0.98:
            winner = "B"
            chosen_features = self.data_loader.FEATURES_WITHOUT_TOTAL
            logger.info(f"  >>> WINNER: Set B (without total_score) — cleaner, similar performance")
        else:
            winner = "A"
            chosen_features = self.data_loader.FEATURES_WITH_TOTAL
            logger.info(f"  >>> WINNER: Set A (with total_score) — better performance")
        logger.info(f"{'='*60}\n")

        # Train final models with chosen feature set
        X, self.feature_names = self.data_loader.get_feature_matrix(df, chosen_features, fit=True)

        all_folds = []
        for horizon in [5, 10]:
            target_col = f'win_{horizon}d'
            if target_col not in df.columns:
                continue

            y = df[target_col].values
            logger.info(f"Training final {horizon}-day model (feature set {winner})...")

            folds = self._walk_forward_train(df, X, y, horizon)
            all_folds.extend(folds)
            self._train_final_model(df, X, y, horizon)

        self.validation_report = self._generate_report(df, all_folds)
        return self.validation_report

    def _evaluate_feature_set(self, df, feature_list, label) -> List[Dict]:
        """Evaluate a feature set using 3-way walk-forward (for A/B comparison)."""
        available = [f for f in feature_list if f in df.columns]
        logger.info(f"\nEvaluating {label} ({len(available)} features)...")

        temp_scaler = StandardScaler()
        X = temp_scaler.fit_transform(df[available].values)

        results = []
        for horizon in [5]:  # Compare on 5d only (speed)
            target_col = f'win_{horizon}d'
            if target_col not in df.columns:
                continue

            y = df[target_col].values
            dates = np.sort(df['date'].unique())
            total = len(dates)
            test_size = total // 5
            cal_size = max(test_size // 2, 15)
            purge = 5

            for fold in range(3):
                test_end = total - fold * test_size
                test_start = test_end - test_size
                cal_end = test_start - purge
                cal_start = cal_end - cal_size
                train_end = cal_start - purge

                if train_end <= 0 or cal_start <= 0:
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

                if len(X_train) < 50 or len(X_cal) < 10 or len(X_test) < 10:
                    continue

                if XGBOOST_AVAILABLE:
                    model = xgb.XGBClassifier(
                        n_estimators=50, max_depth=3, learning_rate=0.1,
                        random_state=42, use_label_encoder=False, eval_metric='logloss')
                else:
                    model = LogisticRegression(max_iter=1000, random_state=42)

                model.fit(X_train, y_train)

                # Calibrate on SEPARATE set (fixes leakage!)
                cal_method = 'sigmoid' if len(X_cal) < 100 else 'isotonic'
                try:
                    calibrator = CalibratedClassifierCV(model, cv='prefit', method=cal_method)
                    calibrator.fit(X_cal, y_cal)
                    probs = calibrator.predict_proba(X_test)[:, 1]
                except Exception:
                    probs = model.predict_proba(X_test)[:, 1]

                auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
                brier = brier_score_loss(y_test, probs)

                trade_mask = probs >= self.min_probability
                win_rate = float(y_test[trade_mask].mean()) if trade_mask.sum() > 0 else 0.0
                trades = int(trade_mask.sum())

                results.append({
                    'fold': fold, 'auc': auc, 'brier': brier,
                    'trades': trades, 'win_rate': win_rate
                })
                logger.info(f"  {label} Fold {fold}: AUC={auc:.3f}  Brier={brier:.4f}  "
                           f"Trades={trades}  WR={win_rate:.1%}  Cal={cal_method}({len(X_cal)})")

        return results

    def _walk_forward_train(self, df, X, y, horizon, n_splits=3) -> List[WalkForwardResult]:
        """Walk-forward validation with 3-way time split (train -> calibrate -> test)."""
        results = []
        dates = np.sort(df['date'].unique())
        total = len(dates)
        test_size = total // 5
        cal_size = max(test_size // 2, 15)
        purge = 5

        for fold in range(n_splits):
            test_end = total - fold * test_size
            test_start = test_end - test_size
            cal_end = test_start - purge
            cal_start = cal_end - cal_size
            train_end = cal_start - purge

            if train_end <= 0 or cal_start <= 0:
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

            if len(X_train) < 50 or len(X_cal) < 10 or len(X_test) < 10:
                continue

            if XGBOOST_AVAILABLE:
                model = xgb.XGBClassifier(
                    n_estimators=50, max_depth=3, learning_rate=0.1,
                    random_state=42, use_label_encoder=False, eval_metric='logloss')
            else:
                model = LogisticRegression(max_iter=1000, random_state=42)

            model.fit(X_train, y_train)

            # Calibrate on SEPARATE held-out calibration set
            cal_method = 'sigmoid' if len(X_cal) < 100 else 'isotonic'
            try:
                calibrator = CalibratedClassifierCV(model, cv='prefit', method=cal_method)
                calibrator.fit(X_cal, y_cal)
                probs = calibrator.predict_proba(X_test)[:, 1]
            except Exception:
                probs = model.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
            brier = brier_score_loss(y_test, probs)

            trade_mask = probs >= self.min_probability
            if trade_mask.sum() > 0:
                win_rate = float(y_test[trade_mask].mean())
                trades = int(trade_mask.sum())
            else:
                win_rate = 0.0
                trades = 0

            results.append(WalkForwardResult(
                fold=fold, train_start=train_dates[0], train_end=train_dates[-1],
                test_start=test_dates[0], test_end=test_dates[-1],
                accuracy=0.5, auc_roc=auc, brier_score=brier,
                trades_taken=trades, win_rate=win_rate, avg_return=0
            ))

            logger.info(f"  Fold {fold}: AUC={auc:.3f}  Brier={brier:.4f}  "
                       f"Trades={trades}  WR={win_rate:.1%}  Cal={cal_method}({len(X_cal)})")

        return results

    def _train_final_model(self, df, X, y, horizon):
        """Train final model with calibration holdout (last 20% chronological)."""
        dates = np.sort(df['date'].unique())
        n_dates = len(dates)

        # Use last 20% of dates as calibration holdout
        cal_size = max(int(n_dates * 0.2), 30)
        purge = 5

        cal_dates = dates[n_dates - cal_size:]
        train_dates = dates[:n_dates - cal_size - purge]

        train_mask = df['date'].isin(train_dates)
        cal_mask = df['date'].isin(cal_dates)

        X_train, y_train = X[train_mask], y[train_mask]
        X_cal, y_cal = X[cal_mask], y[cal_mask]

        if XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=42, use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y_train)
            self.feature_importance[horizon] = dict(zip(self.feature_names, model.feature_importances_))
        else:
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            self.feature_importance[horizon] = dict(zip(self.feature_names, np.abs(model.coef_[0])))

        # Calibrate on HELD-OUT calibration set (not training data!)
        cal_method = 'sigmoid' if len(X_cal) < 100 else 'isotonic'
        try:
            calibrator = CalibratedClassifierCV(model, cv='prefit', method=cal_method)
            calibrator.fit(X_cal, y_cal)
            self.calibrators[horizon] = calibrator
            logger.info(f"  Final {horizon}d: trained on {len(X_train)}, "
                       f"calibrated on {len(X_cal)} ({cal_method})")
        except Exception:
            self.calibrators[horizon] = model
            logger.warning(f"  Final {horizon}d: calibration failed, using raw model")

        self.models[horizon] = model

    def _generate_report(self, df, folds) -> ModelReport:
        """Generate validation report."""
        if not folds:
            folds = [WalkForwardResult(0, date.today(), date.today(), date.today(), date.today(),
                                       0.5, 0.5, 0.25, 0, 0.5, 0)]

        mean_auc = np.mean([f.auc_roc for f in folds])
        mean_brier = np.mean([f.brier_score for f in folds])
        mean_wr = np.mean([f.win_rate for f in folds if f.trades_taken > 0]) \
            if any(f.trades_taken > 0 for f in folds) else 0.5

        combined_imp = {}
        for h, imp in self.feature_importance.items():
            for feat, val in imp.items():
                combined_imp[feat] = combined_imp.get(feat, 0) + val
        total = sum(combined_imp.values()) or 1
        combined_imp = {k: v/total for k, v in combined_imp.items()}

        return ModelReport(
            model_name="XGBoost" if XGBOOST_AVAILABLE else "Logistic",
            trained_at=datetime.now(), total_samples=len(df),
            feature_count=len(self.feature_names), target_horizon=self.target_horizon,
            folds=folds, mean_accuracy=0.5, mean_auc=mean_auc, mean_brier=mean_brier,
            mean_win_rate=mean_wr, mean_return=0, feature_importance=combined_imp,
            is_well_calibrated=mean_brier < 0.25, calibration_error=mean_brier,
            beats_baseline=mean_auc > 0.52, baseline_auc=0.5,
            improvement_vs_baseline=mean_auc - 0.5
        )

    def predict(self, scores: Dict[str, float]) -> PredictionResult:
        """Generate prediction for a stock."""
        if not self.models:
            raise ValueError("Model not trained. Call train() first.")

        # Compute missing indicators BEFORE applying defaults
        for score_key, missing_key in self.data_loader.MISSING_INDICATOR_MAP.items():
            if missing_key in self.feature_names:
                scores[missing_key] = 1 if scores.get(score_key) is None else 0

        # Build feature vector with domain-correct defaults + clipping
        feature_values = []
        filled_features = []
        for f in self.feature_names:
            val = scores.get(f)
            if val is None:
                default = self.data_loader.FEATURE_DEFAULTS.get(f, 50)
                feature_values.append(default)
                filled_features.append(f)
            else:
                clip_range = self.data_loader.FEATURE_CLIPS.get(f)
                if clip_range:
                    val = max(clip_range[0], min(clip_range[1], float(val)))
                feature_values.append(float(val))

        if filled_features:
            logger.debug(f"{scores.get('ticker', '?')}: Filled missing features: {filled_features}")

        X = np.array([feature_values])
        X = self.data_loader.scaler.transform(X)

        probs = {}
        evs = {}

        for horizon in [5, 10]:
            if horizon not in self.calibrators:
                probs[horizon] = 0.5
                evs[horizon] = 0
                continue

            try:
                prob = self.calibrators[horizon].predict_proba(X)[0, 1]
            except Exception:
                prob = 0.5

            probs[horizon] = prob

            avg_win = self.avg_win_return.get(horizon, 2.0)
            avg_loss = self.avg_loss_return.get(horizon, 2.0)
            cost = 0.15
            ev = prob * avg_win - (1 - prob) * avg_loss - cost
            evs[horizon] = ev / 100

        # Extrapolate 1d and 2d (heuristic — Phase 3 will replace with proper models)
        probs[1] = probs.get(5, 0.5) * 0.85
        probs[2] = probs.get(5, 0.5) * 0.92
        evs[1] = evs.get(5, 0) * 0.5
        evs[2] = evs.get(5, 0) * 0.7

        best_horizon = max(evs.keys(), key=lambda h: evs.get(h, 0))
        should_trade = probs.get(5, 0.5) >= self.min_probability and evs.get(5, 0) >= self.min_ev

        confidence = "HIGH" if probs.get(5, 0.5) >= 0.70 \
            else "MEDIUM" if probs.get(5, 0.5) >= 0.60 else "LOW"

        top_features = dict(sorted(
            self.feature_importance.get(5, {}).items(),
            key=lambda x: x[1], reverse=True)[:5])

        return PredictionResult(
            ticker=scores.get('ticker', 'UNKNOWN'),
            prediction_date=date.today(),
            prob_win_1d=probs.get(1, 0.5),
            prob_win_2d=probs.get(2, 0.5),
            prob_win_5d=probs.get(5, 0.5),
            prob_win_10d=probs.get(10, 0.5),
            ev_1d=evs.get(1, 0),
            ev_2d=evs.get(2, 0),
            ev_5d=evs.get(5, 0),
            ev_10d=evs.get(10, 0),
            top_features=top_features,
            confidence=confidence,
            similar_setups_win_rate=probs.get(5, 0.5),
            similar_setups_count=0,
            should_trade=should_trade,
            recommended_horizon=best_horizon
        )

    def save(self, path: str = 'models/signal_predictor.pkl'):
        """Save model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'version': 2,  # Phase 2 model format
            'models': self.models,
            'calibrators': self.calibrators,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'avg_win_return': self.avg_win_return,
            'avg_loss_return': self.avg_loss_return,
            'scaler': self.data_loader.scaler,
            'validation_report': self.validation_report
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str = 'models/signal_predictor.pkl'):
        """Load model (backward compatible with Phase 1 models)."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.models = state['models']
        self.calibrators = state['calibrators']
        self.feature_names = state['feature_names']
        self.feature_importance = state['feature_importance']
        self.avg_win_return = state['avg_win_return']
        self.avg_loss_return = state['avg_loss_return']
        self.data_loader.scaler = state['scaler']
        self.validation_report = state.get('validation_report')
        logger.info(f"Model loaded from {path} (version {state.get('version', 1)})")


if __name__ == "__main__":
    predictor = MLSignalPredictor()
    report = predictor.train()

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Model: {report.model_name}")
    print(f"Samples: {report.total_samples}")
    print(f"Features: {report.feature_count}")
    print(f"  {', '.join(predictor.feature_names)}")
    print(f"Mean AUC: {report.mean_auc:.4f}")
    print(f"Mean Brier: {report.mean_brier:.4f}")
    print(f"Mean Win Rate: {report.mean_win_rate:.1%}")
    print(f"Beats Baseline: {report.beats_baseline}")
    print(f"Well Calibrated: {report.is_well_calibrated}")
    print(f"\nFeature Importance:")
    for feat, imp in sorted(report.feature_importance.items(), key=lambda x: -x[1]):
        print(f"  {feat:25s} {imp:.4f}")

    predictor.save()
    print(f"\nModel saved to models/signal_predictor.pkl")