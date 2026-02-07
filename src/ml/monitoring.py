"""
Monitoring & Feedback Loop - Phase 7

Continuous monitoring system that:
1. Tracks model performance over time
2. Detects drift (calibration, feature distribution, regime)
3. Triggers retraining only when needed
4. Logs all recommendations and outcomes

Key insight: Don't retrain on a fixed schedule. Retrain when metrics degrade.

Location: src/ml/monitoring.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class DriftType(Enum):
    """Types of drift that can occur."""
    CALIBRATION = "Probability calibration drift"
    PERFORMANCE = "Win rate / return degradation"
    FEATURE = "Feature distribution shift"
    REGIME = "Market regime change"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class DriftAlert:
    """Alert when drift is detected."""
    alert_time: datetime
    drift_type: DriftType
    level: AlertLevel
    metric_name: str
    current_value: float
    expected_value: float
    threshold: float
    message: str
    recommended_action: str


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot."""
    snapshot_date: date
    period_days: int

    # Overall metrics
    total_recommendations: int
    trades_taken: int
    trades_skipped: int

    # Outcome metrics (for closed trades)
    closed_trades: int
    wins: int
    losses: int
    win_rate: float

    # Return metrics
    avg_return: float
    total_return: float
    best_trade: float
    worst_trade: float

    # Calibration metrics
    avg_predicted_prob: float
    actual_hit_rate: float
    calibration_error: float  # Predicted - Actual

    # By regime
    metrics_by_regime: Dict[str, Dict] = field(default_factory=dict)

    # By signal type
    metrics_by_signal: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class RetrainTrigger:
    """Trigger for model retraining."""
    triggered: bool
    trigger_reason: str
    severity: AlertLevel
    metrics: Dict[str, float]
    recommended_actions: List[str]


# =============================================================================
# RECOMMENDATION LOGGER
# =============================================================================

class RecommendationLogger:
    """
    Logs all AI recommendations and tracks outcomes.

    This creates the feedback loop:
    Recommendation → Outcome → Learning → Better Recommendations
    """

    CREATE_TABLE_SQL = """
                       CREATE TABLE IF NOT EXISTS ai_recommendations \
                       ( \
                           id \
                           SERIAL \
                           PRIMARY \
                           KEY,

                           -- Recommendation details \
                           ticker \
                           VARCHAR \
                       ( \
                           10 \
                       ) NOT NULL,
                           recommendation_date DATE NOT NULL,
                           recommendation_type VARCHAR \
                       ( \
                           20 \
                       ), -- BUY, SELL, SKIP

                       -- ML model outputs
                           ml_probability DECIMAL \
                       ( \
                           5, \
                           4 \
                       ),
                           ml_ev DECIMAL \
                       ( \
                           8, \
                           6 \
                       ),
                           ml_confidence VARCHAR \
                       ( \
                           20 \
                       ),

                           -- Meta-labeler outputs
                           meta_probability DECIMAL \
                       ( \
                           5, \
                           4 \
                       ),
                           combined_probability DECIMAL \
                       ( \
                           5, \
                           4 \
                       ),

                           -- Decision layer outputs
                           approved BOOLEAN,
                           rejection_reasons TEXT[],

                           -- Trade plan
                           entry_price DECIMAL \
                       ( \
                           12, \
                           4 \
                       ),
                           stop_loss DECIMAL \
                       ( \
                           12, \
                           4 \
                       ),
                           target_price DECIMAL \
                       ( \
                           12, \
                           4 \
                       ),
                           position_size_pct DECIMAL \
                       ( \
                           5, \
                           4 \
                       ),
                           horizon_days INTEGER,

                           -- Context at time
                           vix_level DECIMAL \
                       ( \
                           5, \
                           2 \
                       ),
                           regime_score DECIMAL \
                       ( \
                           5, \
                           2 \
                       ),
                           similar_setups_win_rate DECIMAL \
                       ( \
                           5, \
                           4 \
                       ),

                           -- Outcome (filled later)
                           outcome VARCHAR \
                       ( \
                           10 \
                       ), -- WIN, LOSS, SCRATCH, OPEN
                           actual_return DECIMAL \
                       ( \
                           8, \
                           4 \
                       ),
                           actual_holding_days INTEGER,
                           exit_reason VARCHAR \
                       ( \
                           20 \
                       ),
                           exit_date DATE,

                           -- Metadata
                           created_at TIMESTAMP DEFAULT NOW \
                       ( \
                       ),
                           updated_at TIMESTAMP DEFAULT NOW \
                       ( \
                       ), \
                           UNIQUE \
                       ( \
                           ticker, \
                           recommendation_date \
                       )
                           );

                       CREATE INDEX IF NOT EXISTS idx_recs_date ON ai_recommendations(recommendation_date);
                       CREATE INDEX IF NOT EXISTS idx_recs_outcome ON ai_recommendations(outcome);
                       CREATE INDEX IF NOT EXISTS idx_recs_type ON ai_recommendations(recommendation_type); \
                       """

    def __init__(self, engine=None):
        self.engine = engine
        self._memory_log: List[Dict] = []

        if engine:
            self._ensure_table()

    def _ensure_table(self):
        """Create table if not exists."""
        try:
            with self.engine.connect() as conn:
                conn.execute(self.CREATE_TABLE_SQL)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to create recommendations table: {e}")

    def log_recommendation(self,
                           ticker: str,
                           recommendation_type: str,
                           ml_prediction: Dict,
                           meta_result: Dict,
                           decision_result: Dict,
                           market_context: Dict) -> bool:
        """Log a new recommendation."""

        record = {
            'ticker': ticker,
            'recommendation_date': date.today(),
            'recommendation_type': recommendation_type,

            # ML outputs
            'ml_probability': ml_prediction.get('prob_win_5d'),
            'ml_ev': ml_prediction.get('ev_5d'),
            'ml_confidence': ml_prediction.get('confidence'),

            # Meta-labeler
            'meta_probability': meta_result.get('meta_prob'),
            'combined_probability': meta_result.get('combined_prob'),

            # Decision
            'approved': decision_result.get('approved'),
            'rejection_reasons': decision_result.get('rejection_reasons', []),

            # Trade plan
            'entry_price': decision_result.get('entry_price'),
            'stop_loss': decision_result.get('stop_loss'),
            'target_price': decision_result.get('target_price'),
            'position_size_pct': decision_result.get('position_pct'),
            'horizon_days': decision_result.get('recommended_horizon', 5),

            # Context
            'vix_level': market_context.get('vix'),
            'regime_score': market_context.get('regime_score'),
            'similar_setups_win_rate': market_context.get('similar_win_rate'),

            # Outcome (to be filled later)
            'outcome': 'OPEN',
            'actual_return': None,
            'created_at': datetime.now()
        }

        self._memory_log.append(record)

        if self.engine:
            return self._save_to_db(record)
        return True

    def _save_to_db(self, record: Dict) -> bool:
        """Save record to database."""
        try:
            df = pd.DataFrame([record])
            df.to_sql('ai_recommendations', self.engine, if_exists='append', index=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save recommendation: {e}")
            return False

    def update_outcome(self, ticker: str, recommendation_date: date,
                       outcome: str, actual_return: float,
                       holding_days: int, exit_reason: str) -> bool:
        """Update outcome for a recommendation."""

        # Update memory log
        for record in self._memory_log:
            if record['ticker'] == ticker and record['recommendation_date'] == recommendation_date:
                record['outcome'] = outcome
                record['actual_return'] = actual_return
                record['actual_holding_days'] = holding_days
                record['exit_reason'] = exit_reason
                record['exit_date'] = date.today()
                break

        if self.engine:
            try:
                with self.engine.connect() as conn:
                    conn.execute("""
                                 UPDATE ai_recommendations
                                 SET outcome             = %s,
                                     actual_return       = %s,
                                     actual_holding_days = %s,
                                     exit_reason         = %s,
                                     exit_date           = %s,
                                     updated_at          = NOW()
                                 WHERE ticker = %s
                                   AND recommendation_date = %s
                                 """, (outcome, actual_return, holding_days, exit_reason,
                                       date.today(), ticker, recommendation_date))
                    conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to update outcome: {e}")
                return False
        return True

    def get_recent_recommendations(self, days: int = 30) -> pd.DataFrame:
        """Get recent recommendations with outcomes."""
        if self.engine:
            query = f"""
                SELECT * FROM ai_recommendations
                WHERE recommendation_date >= CURRENT_DATE - INTERVAL '{days} days'
                ORDER BY recommendation_date DESC
            """
            return pd.read_sql(query, self.engine)
        else:
            cutoff = date.today() - timedelta(days=days)
            recent = [r for r in self._memory_log
                      if r['recommendation_date'] >= cutoff]
            return pd.DataFrame(recent)


# =============================================================================
# DRIFT DETECTOR
# =============================================================================

class DriftDetector:
    """
    Detects various types of drift that indicate model degradation.

    Types of drift:
    1. Calibration drift: Predicted probabilities don't match actual outcomes
    2. Performance drift: Win rates or returns declining
    3. Feature drift: Input feature distributions changing
    4. Regime drift: Market regime has changed
    """

    def __init__(self,
                 calibration_threshold: float = 0.10,  # 10% miscalibration
                 win_rate_threshold: float = 0.10,  # 10% drop in win rate
                 return_threshold: float = 0.50,  # 50% drop in returns
                 drift_window_days: int = 14):
        """
        Args:
            calibration_threshold: Max allowed calibration error
            win_rate_threshold: Max allowed win rate drop vs baseline
            return_threshold: Max allowed return drop vs baseline
            drift_window_days: Window for detecting drift
        """
        self.calibration_threshold = calibration_threshold
        self.win_rate_threshold = win_rate_threshold
        self.return_threshold = return_threshold
        self.drift_window_days = drift_window_days

        # Baseline metrics (set during training)
        self.baseline_win_rate: float = 0.60
        self.baseline_avg_return: float = 2.0
        self.baseline_calibration: float = 0.0

        # Alert history
        self.alerts: List[DriftAlert] = []

    def set_baseline(self, win_rate: float, avg_return: float, calibration: float = 0):
        """Set baseline metrics from training."""
        self.baseline_win_rate = win_rate
        self.baseline_avg_return = avg_return
        self.baseline_calibration = calibration
        logger.info(f"Baseline set: WR={win_rate:.1%}, AvgRet={avg_return:.2f}%")

    def check_all(self, recommendations: pd.DataFrame,
                  current_regime: Dict = None) -> List[DriftAlert]:
        """Run all drift checks."""
        alerts = []

        if len(recommendations) < 10:
            return alerts

        # Filter to closed trades only
        closed = recommendations[recommendations['outcome'].isin(['WIN', 'LOSS', 'SCRATCH'])]

        if len(closed) >= 10:
            # Calibration drift
            cal_alert = self._check_calibration_drift(closed)
            if cal_alert:
                alerts.append(cal_alert)

            # Performance drift
            perf_alert = self._check_performance_drift(closed)
            if perf_alert:
                alerts.append(perf_alert)

        # Regime drift
        if current_regime:
            regime_alert = self._check_regime_drift(current_regime)
            if regime_alert:
                alerts.append(regime_alert)

        self.alerts.extend(alerts)
        return alerts

    def _check_calibration_drift(self, closed: pd.DataFrame) -> Optional[DriftAlert]:
        """Check if predicted probabilities match actual outcomes."""

        if 'ml_probability' not in closed.columns:
            return None

        # Get predictions and actuals
        probs = closed['ml_probability'].dropna()
        actuals = (closed['outcome'] == 'WIN').astype(int)

        if len(probs) < 10:
            return None

        # Calculate calibration error
        avg_predicted = probs.mean()
        actual_rate = actuals.mean()
        calibration_error = avg_predicted - actual_rate

        if abs(calibration_error) > self.calibration_threshold:
            level = AlertLevel.CRITICAL if abs(calibration_error) > 0.15 else AlertLevel.WARNING

            if calibration_error > 0:
                message = f"Model is overconfident: predicted {avg_predicted:.1%} but actual {actual_rate:.1%}"
                action = "Consider recalibrating probabilities or retraining"
            else:
                message = f"Model is underconfident: predicted {avg_predicted:.1%} but actual {actual_rate:.1%}"
                action = "Model may be too conservative - review thresholds"

            return DriftAlert(
                alert_time=datetime.now(),
                drift_type=DriftType.CALIBRATION,
                level=level,
                metric_name="calibration_error",
                current_value=calibration_error,
                expected_value=0,
                threshold=self.calibration_threshold,
                message=message,
                recommended_action=action
            )

        return None

    def _check_performance_drift(self, closed: pd.DataFrame) -> Optional[DriftAlert]:
        """Check if win rate or returns have degraded."""

        # Calculate current metrics
        wins = (closed['outcome'] == 'WIN').sum()
        total = len(closed)
        current_win_rate = wins / total if total > 0 else 0

        returns = closed['actual_return'].dropna()
        current_avg_return = returns.mean() if len(returns) > 0 else 0

        # Check win rate
        wr_drop = self.baseline_win_rate - current_win_rate
        if wr_drop > self.win_rate_threshold:
            return DriftAlert(
                alert_time=datetime.now(),
                drift_type=DriftType.PERFORMANCE,
                level=AlertLevel.CRITICAL if wr_drop > 0.15 else AlertLevel.WARNING,
                metric_name="win_rate",
                current_value=current_win_rate,
                expected_value=self.baseline_win_rate,
                threshold=self.win_rate_threshold,
                message=f"Win rate dropped from {self.baseline_win_rate:.1%} to {current_win_rate:.1%}",
                recommended_action="Review recent trades for pattern. Consider retraining."
            )

        # Check returns
        if self.baseline_avg_return > 0:
            ret_drop_pct = (self.baseline_avg_return - current_avg_return) / self.baseline_avg_return
            if ret_drop_pct > self.return_threshold:
                return DriftAlert(
                    alert_time=datetime.now(),
                    drift_type=DriftType.PERFORMANCE,
                    level=AlertLevel.WARNING,
                    metric_name="avg_return",
                    current_value=current_avg_return,
                    expected_value=self.baseline_avg_return,
                    threshold=self.return_threshold,
                    message=f"Avg return dropped from {self.baseline_avg_return:.2f}% to {current_avg_return:.2f}%",
                    recommended_action="Check if market regime has changed"
                )

        return None

    def _check_regime_drift(self, current_regime: Dict) -> Optional[DriftAlert]:
        """Check if market regime has changed significantly."""

        # This would compare current regime to regime during training
        # For now, just check for extreme regimes
        vix = current_regime.get('vix')
        regime_score = current_regime.get('score')

        if vix is not None and vix > 35:
            return DriftAlert(
                alert_time=datetime.now(),
                drift_type=DriftType.REGIME,
                level=AlertLevel.WARNING,
                metric_name="vix",
                current_value=vix,
                expected_value=20,
                threshold=35,
                message=f"VIX at {vix:.1f} - extreme volatility regime",
                recommended_action="Reduce position sizes. Model may not be calibrated for this regime."
            )

        if regime_score is not None and (regime_score < 25 or regime_score > 75):
            direction = "Risk-Off" if regime_score < 25 else "Strong Risk-On"
            return DriftAlert(
                alert_time=datetime.now(),
                drift_type=DriftType.REGIME,
                level=AlertLevel.INFO,
                metric_name="regime_score",
                current_value=regime_score,
                expected_value=50,
                threshold=25,
                message=f"Market in {direction} regime (score: {regime_score:.0f})",
                recommended_action="Verify model performs well in this regime historically"
            )

        return None

    def should_retrain(self, recommendations: pd.DataFrame) -> RetrainTrigger:
        """Determine if model should be retrained."""

        alerts = self.check_all(recommendations)

        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        warning_alerts = [a for a in alerts if a.level == AlertLevel.WARNING]

        if critical_alerts:
            return RetrainTrigger(
                triggered=True,
                trigger_reason=critical_alerts[0].message,
                severity=AlertLevel.CRITICAL,
                metrics={a.metric_name: a.current_value for a in critical_alerts},
                recommended_actions=[a.recommended_action for a in critical_alerts]
            )

        if len(warning_alerts) >= 2:
            return RetrainTrigger(
                triggered=True,
                trigger_reason="Multiple warning alerts detected",
                severity=AlertLevel.WARNING,
                metrics={a.metric_name: a.current_value for a in warning_alerts},
                recommended_actions=[a.recommended_action for a in warning_alerts]
            )

        return RetrainTrigger(
            triggered=False,
            trigger_reason="No significant drift detected",
            severity=AlertLevel.INFO,
            metrics={},
            recommended_actions=[]
        )


# =============================================================================
# PERFORMANCE TRACKER
# =============================================================================

class PerformanceTracker:
    """Tracks model performance over time."""

    def __init__(self, logger: RecommendationLogger):
        self.logger = logger
        self.snapshots: List[PerformanceSnapshot] = []

    def generate_snapshot(self, period_days: int = 30) -> PerformanceSnapshot:
        """Generate performance snapshot for the period."""

        recs = self.logger.get_recent_recommendations(period_days)

        if recs.empty:
            return self._empty_snapshot(period_days)

        # Filter trades that were taken (approved)
        taken = recs[recs['approved'] == True]
        skipped = recs[recs['approved'] == False]

        # Closed trades
        closed = taken[taken['outcome'].isin(['WIN', 'LOSS', 'SCRATCH'])]

        if closed.empty:
            return PerformanceSnapshot(
                snapshot_date=date.today(),
                period_days=period_days,
                total_recommendations=len(recs),
                trades_taken=len(taken),
                trades_skipped=len(skipped),
                closed_trades=0,
                wins=0, losses=0, win_rate=0,
                avg_return=0, total_return=0, best_trade=0, worst_trade=0,
                avg_predicted_prob=0, actual_hit_rate=0, calibration_error=0
            )

        wins = (closed['outcome'] == 'WIN').sum()
        losses = (closed['outcome'] == 'LOSS').sum()
        win_rate = wins / len(closed) if len(closed) > 0 else 0

        returns = closed['actual_return'].dropna()
        avg_return = returns.mean() if len(returns) > 0 else 0
        total_return = returns.sum() if len(returns) > 0 else 0
        best = returns.max() if len(returns) > 0 else 0
        worst = returns.min() if len(returns) > 0 else 0

        # Calibration
        probs = closed['ml_probability'].dropna()
        avg_prob = probs.mean() if len(probs) > 0 else 0
        actual_rate = win_rate
        cal_error = avg_prob - actual_rate

        snapshot = PerformanceSnapshot(
            snapshot_date=date.today(),
            period_days=period_days,
            total_recommendations=len(recs),
            trades_taken=len(taken),
            trades_skipped=len(skipped),
            closed_trades=len(closed),
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            avg_return=avg_return,
            total_return=total_return,
            best_trade=best,
            worst_trade=worst,
            avg_predicted_prob=avg_prob,
            actual_hit_rate=actual_rate,
            calibration_error=cal_error
        )

        self.snapshots.append(snapshot)
        return snapshot

    def _empty_snapshot(self, period_days: int) -> PerformanceSnapshot:
        return PerformanceSnapshot(
            snapshot_date=date.today(),
            period_days=period_days,
            total_recommendations=0,
            trades_taken=0,
            trades_skipped=0,
            closed_trades=0,
            wins=0, losses=0, win_rate=0,
            avg_return=0, total_return=0, best_trade=0, worst_trade=0,
            avg_predicted_prob=0, actual_hit_rate=0, calibration_error=0
        )

    def get_performance_summary(self) -> Dict:
        """Get summary of recent performance."""
        if not self.snapshots:
            self.generate_snapshot(30)

        latest = self.snapshots[-1] if self.snapshots else self._empty_snapshot(30)

        return {
            'period_days': latest.period_days,
            'total_recommendations': latest.total_recommendations,
            'trades_taken': latest.trades_taken,
            'closed_trades': latest.closed_trades,
            'win_rate': f"{latest.win_rate:.1%}",
            'avg_return': f"{latest.avg_return:+.2f}%",
            'calibration_error': f"{latest.calibration_error:+.2f}",
            'is_well_calibrated': abs(latest.calibration_error) < 0.10
        }


# =============================================================================
# CONVENIENCE
# =============================================================================

def check_model_health() -> Dict:
    """Quick check of model health."""
    logger_instance = RecommendationLogger()
    tracker = PerformanceTracker(logger_instance)
    detector = DriftDetector()

    snapshot = tracker.generate_snapshot(30)

    detector.set_baseline(0.60, 2.0)
    recs = logger_instance.get_recent_recommendations(30)
    trigger = detector.should_retrain(recs)

    return {
        'snapshot': tracker.get_performance_summary(),
        'needs_retrain': trigger.triggered,
        'retrain_reason': trigger.trigger_reason,
        'alerts': [a.message for a in detector.alerts]
    }