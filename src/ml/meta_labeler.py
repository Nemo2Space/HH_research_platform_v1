"""
Meta-Labeling Model - Phase 5

Two-stage prediction system:
- Model A: Predicts direction/return (ML model)
- Model B: Predicts whether to TAKE the trade (this file)

Key principle: Historical similar setup performance OVERRIDES ML optimism.
If similar setups lost money historically, don't trust ML's positive prediction.

Location: src/ml/meta_labeler.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pickle
import logging

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_score, recall_score, f1_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetaLabelResult:
    """Result from meta-labeling model."""
    should_trade: bool
    confidence: float

    primary_signal_prob: float  # From ML Model
    meta_prob: float  # From this model
    combined_prob: float  # Final combined

    positive_factors: List[str]
    negative_factors: List[str]

    size_multiplier: float

    def get_summary(self) -> str:
        action = "✅ TAKE" if self.should_trade else "❌ SKIP"
        return (f"{action} trade: Primary={self.primary_signal_prob:.1%}, "
                f"Meta={self.meta_prob:.1%}, Combined={self.combined_prob:.1%}")


class MetaLabeler:
    """
    Second-stage model that decides whether to take a trade.

    CRITICAL: Historical similar setup performance can VETO ML predictions.
    If similar setups historically lost money, we don't trade regardless
    of how optimistic the ML model is.
    """

    # Thresholds for similar setup validation
    SIMILAR_SETUP_BLOCK_THRESHOLD = 0.35  # Block trade if similar win rate < 35%
    SIMILAR_SETUP_WARN_THRESHOLD = 0.45   # Warning if < 45%
    SIMILAR_SETUP_MIN_COUNT = 5           # Need at least 5 similar setups for this to apply

    def __init__(self,
                 min_trade_prob: float = 0.55,
                 min_primary_prob: float = 0.50):
        self.min_trade_prob = min_trade_prob
        self.min_primary_prob = min_primary_prob

        self.model = None
        self.calibrator = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.feature_names = []
        self.feature_importance = {}

    def predict(self,
                primary_prediction: Dict,
                scores: Dict,
                market_context: Dict = None,
                similar_setups: Dict = None) -> MetaLabelResult:
        """
        Predict whether to take this trade.

        Historical similar setup performance can BLOCK trades even when
        ML model is optimistic.
        """
        market_context = market_context or {}
        similar_setups = similar_setups or {}

        primary_prob = primary_prediction.get('prob_win_5d', 0.5)

        # Use rule-based approach (most common case)
        return self._rule_based_decision(primary_prediction, scores,
                                         market_context, similar_setups)

    def _rule_based_decision(self, primary_prediction, scores,
                             market_context, similar_setups) -> MetaLabelResult:
        """Rule-based decision with strong similar setup validation."""
        primary_prob = primary_prediction.get('prob_win_5d', 0.5)
        positive = []
        negative = []

        # Start with ML probability
        meta_prob = primary_prob

        # =====================================================================
        # SIMILAR SETUPS - CRITICAL CHECK (can block trade)
        # =====================================================================
        similar_wr = similar_setups.get('win_rate', 0.5)
        similar_count = similar_setups.get('count', 0)
        blocked_by_history = False

        if similar_count >= self.SIMILAR_SETUP_MIN_COUNT:
            if similar_wr < self.SIMILAR_SETUP_BLOCK_THRESHOLD:
                # BLOCK: Historical data strongly negative
                negative.append(f"⛔ Similar setups: {similar_wr:.0%} win rate ({similar_count} trades) - BLOCKING")
                meta_prob *= 0.5  # Severe penalty
                blocked_by_history = True
            elif similar_wr < self.SIMILAR_SETUP_WARN_THRESHOLD:
                # WARNING: Historical data negative
                negative.append(f"Similar setups: {similar_wr:.0%} win rate ({similar_count} trades)")
                meta_prob *= 0.7  # Strong penalty
            elif similar_wr >= 0.60:
                # POSITIVE: Historical data supports the trade
                positive.append(f"Similar setups: {similar_wr:.0%} win rate ({similar_count} trades)")
                meta_prob *= 1.1
        elif similar_count > 0:
            # Limited data - note it but don't penalize heavily
            if similar_wr < 0.40:
                negative.append(f"Similar setups: {similar_wr:.0%} (only {similar_count} trades)")
                meta_prob *= 0.85

        # =====================================================================
        # SCORE AGREEMENT CHECK
        # =====================================================================
        components = [scores.get(f, 50) for f in
                     ['sentiment_score', 'fundamental_score', 'technical_score', 'options_flow_score']]
        std_dev = np.std(components)

        if std_dev < 10:
            positive.append("Strong score agreement")
            meta_prob *= 1.05
        elif std_dev > 20:
            negative.append("Scores disagree")
            meta_prob *= 0.90

        # =====================================================================
        # VIX CHECK
        # =====================================================================
        vix = market_context.get('vix', 20)
        if vix > 30:
            negative.append(f"High VIX ({vix:.0f})")
            meta_prob *= 0.85
        elif vix < 15:
            positive.append(f"Low VIX ({vix:.0f})")
            meta_prob *= 1.05

        # =====================================================================
        # EARNINGS CHECK (only upcoming earnings)
        # =====================================================================
        days_to_earn = market_context.get('days_to_earnings')
        if days_to_earn is not None and isinstance(days_to_earn, (int, float)):
            if 0 < days_to_earn <= 3:
                negative.append(f"Earnings in {int(days_to_earn)} days - binary risk")
                meta_prob *= 0.7
            elif 0 < days_to_earn <= 7:
                negative.append(f"Earnings in {int(days_to_earn)} days")
                meta_prob *= 0.85

        # =====================================================================
        # FINAL CALCULATION
        # =====================================================================
        meta_prob = max(0.1, min(0.95, meta_prob))
        combined = np.sqrt(primary_prob * meta_prob)

        # Decision logic - historical data can veto
        if blocked_by_history:
            should_trade = False  # Hard block
        else:
            should_trade = (
                combined >= self.min_trade_prob and
                primary_prob >= self.min_primary_prob
            )

        # Size multiplier based on confidence
        if combined >= 0.70 and not blocked_by_history:
            size_mult = 1.0
        elif combined >= 0.60:
            size_mult = 0.75
        elif combined >= 0.55:
            size_mult = 0.5
        else:
            size_mult = 0.25

        return MetaLabelResult(
            should_trade=should_trade,
            confidence=combined,
            primary_signal_prob=primary_prob,
            meta_prob=meta_prob,
            combined_prob=combined,
            positive_factors=positive,
            negative_factors=negative,
            size_multiplier=size_mult
        )

    def get_meta_features(self,
                          primary_prediction: Dict,
                          scores: Dict,
                          market_context: Dict,
                          similar_setups: Dict) -> np.ndarray:
        """Build meta-features for trained model (if available)."""
        features = []
        self.feature_names = []

        # Primary model outputs
        features.append(primary_prediction.get('prob_win_5d', 0.5))
        self.feature_names.append('primary_prob')

        features.append(primary_prediction.get('ev_5d', 0) * 100)
        self.feature_names.append('primary_ev')

        # Score features
        features.append(scores.get('sentiment_score', 50))
        self.feature_names.append('sentiment')

        features.append(scores.get('options_flow_score', 50))
        self.feature_names.append('options_flow')

        features.append(scores.get('total_score', 50))
        self.feature_names.append('total_score')

        # Market context
        features.append(market_context.get('vix', 20))
        self.feature_names.append('vix')

        # Similar setups
        features.append(similar_setups.get('win_rate', 0.5) * 100)
        self.feature_names.append('similar_win_rate')

        features.append(min(similar_setups.get('count', 0), 50))
        self.feature_names.append('similar_count')

        return np.array(features)

    def train(self, training_data: pd.DataFrame) -> Dict:
        """Train the meta-labeling model."""
        if not ML_AVAILABLE:
            logger.warning("sklearn not available, using rule-based only")
            return {'trained': False}

        if training_data.empty or len(training_data) < 100:
            logger.warning("Insufficient training data for meta-labeler")
            return {'trained': False}

        # Build features
        X = []
        y = []

        for _, row in training_data.iterrows():
            features = self.get_meta_features(
                primary_prediction={
                    'prob_win_5d': row.get('primary_prob', 0.5),
                    'ev_5d': row.get('primary_ev', 0)
                },
                scores={
                    'sentiment_score': row.get('sentiment_score', 50),
                    'options_flow_score': row.get('options_flow_score', 50),
                    'total_score': row.get('total_score', 50)
                },
                market_context={'vix': row.get('vix', 20)},
                similar_setups={
                    'win_rate': row.get('similar_win_rate', 0.5),
                    'count': row.get('similar_count', 0)
                }
            )
            X.append(features)
            y.append(1 if row.get('was_profitable', False) else 0)

        X = np.array(X)
        y = np.array(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=42
        )
        self.model.fit(X_scaled, y)

        # Calibrate
        try:
            self.calibrator = CalibratedClassifierCV(self.model, cv=3, method='isotonic')
            self.calibrator.fit(X_scaled, y)
        except:
            self.calibrator = self.model

        # Feature importance
        self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))

        logger.info("Meta-labeler trained successfully")
        return {
            'trained': True,
            'samples': len(X),
            'feature_importance': self.feature_importance
        }

    def save(self, path: str = 'models/meta_labeler.pkl'):
        """Save trained model."""
        if self.model is None:
            logger.warning("No model to save")
            return

        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            'model': self.model,
            'calibrator': self.calibrator,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Meta-labeler saved to {path}")

    def load(self, path: str = 'models/meta_labeler.pkl'):
        """Load trained model."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.model = state['model']
        self.calibrator = state['calibrator']
        self.scaler = state['scaler']
        self.feature_names = state['feature_names']
        self.feature_importance = state['feature_importance']

        logger.info(f"Meta-labeler loaded from {path}")


# Convenience function
def should_take_trade(primary_prob: float, scores: Dict,
                      vix: float = 20, similar_win_rate: float = 0.5,
                      similar_count: int = 0) -> MetaLabelResult:
    """Quick check if a trade should be taken."""
    labeler = MetaLabeler()

    return labeler.predict(
        primary_prediction={'prob_win_5d': primary_prob},
        scores=scores,
        market_context={'vix': vix},
        similar_setups={'win_rate': similar_win_rate, 'count': similar_count}
    )