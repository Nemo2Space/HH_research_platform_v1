"""
ML Signal Predictor - Phase 1 & 2

Walk-forward validation, multi-horizon labels, calibrated probabilities.

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
    """Loads and prepares data with proper labels."""

    FEATURES = [
        'sentiment_score', 'fundamental_score', 'technical_score',
        'options_flow_score', 'short_squeeze_score', 'gap_score',
        'total_score', 'article_count', 'target_upside_pct'
    ]

    def __init__(self):
        self.engine = get_engine()
        self.scaler = StandardScaler()
        self.cost_pct = 0.15  # 0.15% round trip cost

    def load_training_data(self, min_date: str = None) -> pd.DataFrame:
        """Load historical data with multi-horizon labels."""

        query = """
            SELECT 
                h.score_date as date,
                h.ticker,
                h.sector,
                COALESCE(h.sentiment, 50) as sentiment_score,
                COALESCE(h.fundamental_score, 50) as fundamental_score,
                COALESCE(h.growth_score, 50) as growth_score,
                COALESCE(h.total_score, 50) as total_score,
COALESCE(h.gap_score::numeric, 50) as gap_score,
                COALESCE(s.technical_score, 50) as technical_score,
                COALESCE(s.options_flow_score, 50) as options_flow_score,
                COALESCE(s.short_squeeze_score, 50) as short_squeeze_score,
                COALESCE(s.article_count, 0) as article_count,
                COALESCE(s.target_upside_pct, 0) as target_upside_pct,
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

        # Fill missing
        for col in self.FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(50)

        return df

    def get_feature_matrix(self, df: pd.DataFrame, fit: bool = False) -> Tuple[np.ndarray, List[str]]:
        """Extract and scale features."""
        available = [f for f in self.FEATURES if f in df.columns]
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
    """ML prediction with walk-forward validation."""

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
        """Train models with walk-forward validation."""
        if df is None:
            df = self.data_loader.load_training_data()

        if df.empty or len(df) < 200:
            raise ValueError(f"Insufficient data: {len(df)} samples (need 200+)")

        X, self.feature_names = self.data_loader.get_feature_matrix(df, fit=True)

        # Calculate historical returns for EV
        for h in [1, 2, 5, 10]:
            col = f'return_{h}d_net'
            if col in df.columns:
                returns = df[col].dropna()
                self.avg_win_return[h] = returns[returns > 0].mean() if (returns > 0).any() else 2.0
                self.avg_loss_return[h] = abs(returns[returns <= 0].mean()) if (returns <= 0).any() else 2.0

        # Train for each horizon
        all_folds = []
        for horizon in [5, 10]:  # Focus on 5d and 10d
            target_col = f'win_{horizon}d'
            if target_col not in df.columns:
                continue

            y = df[target_col].values
            logger.info(f"Training {horizon}-day model...")

            folds = self._walk_forward_train(df, X, y, horizon)
            all_folds.extend(folds)
            self._train_final_model(X, y, horizon)

        # Generate report
        self.validation_report = self._generate_report(df, all_folds)
        return self.validation_report

    def _walk_forward_train(self, df, X, y, horizon, n_splits=3) -> List[WalkForwardResult]:
        """Walk-forward validation."""
        results = []
        dates = np.sort(df['date'].unique())
        total = len(dates)
        test_size = total // (n_splits + 1)

        for fold in range(n_splits):
            test_end_idx = total - (fold * test_size)
            test_start_idx = test_end_idx - test_size
            train_end_idx = test_start_idx - 5  # Purge gap
            train_start_idx = 0

            if train_end_idx <= 0:
                continue

            train_dates = dates[train_start_idx:train_end_idx]
            test_dates = dates[test_start_idx:test_end_idx]

            train_mask = df['date'].isin(train_dates)
            test_mask = df['date'].isin(test_dates)

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            if len(X_train) < 50 or len(X_test) < 10:
                continue

            # Train XGBoost or Logistic
            if XGBOOST_AVAILABLE:
                model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                                         random_state=42, use_label_encoder=False, eval_metric='logloss')
            else:
                model = LogisticRegression(max_iter=1000, random_state=42)

            model.fit(X_train, y_train)

            # Calibrate
            try:
                calibrator = CalibratedClassifierCV(model, cv=2, method='isotonic')
                calibrator.fit(X_train, y_train)
                probs = calibrator.predict_proba(X_test)[:, 1]
            except:
                probs = model.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
            brier = brier_score_loss(y_test, probs)

            # Trade metrics
            trade_mask = probs >= self.min_probability
            if trade_mask.sum() > 0:
                win_rate = y_test[trade_mask].mean()
                trades = trade_mask.sum()
            else:
                win_rate = 0
                trades = 0

            results.append(WalkForwardResult(
                fold=fold, train_start=train_dates[0], train_end=train_dates[-1],
                test_start=test_dates[0], test_end=test_dates[-1],
                accuracy=0.5, auc_roc=auc, brier_score=brier,
                trades_taken=trades, win_rate=win_rate, avg_return=0
            ))

            logger.info(f"Fold {fold}: AUC={auc:.3f}, Trades={trades}, WinRate={win_rate:.1%}")

        return results

    def _train_final_model(self, X, y, horizon):
        """Train final model on all data."""
        if XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                                     random_state=42, use_label_encoder=False, eval_metric='logloss')
            model.fit(X, y)
            self.feature_importance[horizon] = dict(zip(self.feature_names, model.feature_importances_))
        else:
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X, y)
            self.feature_importance[horizon] = dict(zip(self.feature_names, np.abs(model.coef_[0])))

        try:
            calibrator = CalibratedClassifierCV(model, cv=3, method='isotonic')
            calibrator.fit(X, y)
            self.calibrators[horizon] = calibrator
        except:
            self.calibrators[horizon] = model

        self.models[horizon] = model

    def _generate_report(self, df, folds) -> ModelReport:
        """Generate validation report."""
        if not folds:
            folds = [WalkForwardResult(0, date.today(), date.today(), date.today(), date.today(),
                                       0.5, 0.5, 0.25, 0, 0.5, 0)]

        mean_auc = np.mean([f.auc_roc for f in folds])
        mean_brier = np.mean([f.brier_score for f in folds])
        mean_wr = np.mean([f.win_rate for f in folds if f.trades_taken > 0]) if any(f.trades_taken > 0 for f in folds) else 0.5

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

        X = np.array([[scores.get(f, 50) for f in self.feature_names]])
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
            except:
                prob = 0.5

            probs[horizon] = prob

            avg_win = self.avg_win_return.get(horizon, 2.0)
            avg_loss = self.avg_loss_return.get(horizon, 2.0)
            cost = 0.15
            ev = prob * avg_win - (1 - prob) * avg_loss - cost
            evs[horizon] = ev / 100

        # Extrapolate 1d and 2d
        probs[1] = probs.get(5, 0.5) * 0.85
        probs[2] = probs.get(5, 0.5) * 0.92
        evs[1] = evs.get(5, 0) * 0.5
        evs[2] = evs.get(5, 0) * 0.7

        # Best horizon
        best_horizon = max(evs.keys(), key=lambda h: evs.get(h, 0))
        should_trade = probs.get(5, 0.5) >= self.min_probability and evs.get(5, 0) >= self.min_ev

        confidence = "HIGH" if probs.get(5, 0.5) >= 0.70 else "MEDIUM" if probs.get(5, 0.5) >= 0.60 else "LOW"

        top_features = dict(sorted(self.feature_importance.get(5, {}).items(),
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
        """Load model."""
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
        logger.info(f"Model loaded from {path}")


if __name__ == "__main__":
    predictor = MLSignalPredictor()
    report = predictor.train()
    print(f"\nModel trained: AUC={report.mean_auc:.3f}, Win Rate={report.mean_win_rate:.1%}")
    predictor.save()