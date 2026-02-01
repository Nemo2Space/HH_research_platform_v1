"""
AI Performance Tracker

Tracks AI trading recommendations and their actual outcomes.
Logs predictions to database and calculates accuracy metrics.

Location: src/ml/ai_performance_tracker.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

from src.utils.logging import get_logger

# Use db_helper which has credentials from .env
try:
    from src.ml.db_helper import get_engine, get_connection
except ImportError:
    from src.db.connection import get_connection, get_engine

logger = get_logger(__name__)


@dataclass
class AIRecommendation:
    """Single AI recommendation record."""
    ticker: str
    recommendation_date: date
    ai_probability: float
    ai_ev: float
    recommendation: str  # BUY, SKIP
    signal_scores: Dict[str, float]
    entry_price: Optional[float] = None

    # Outcomes (filled in later)
    return_5d: Optional[float] = None
    return_10d: Optional[float] = None
    was_correct: Optional[bool] = None


@dataclass
class AIPerformanceMetrics:
    """AI performance summary."""
    total_recommendations: int
    buy_recommendations: int
    skip_recommendations: int

    # Accuracy
    buy_accuracy: float  # % of BUY recommendations that were profitable
    skip_accuracy: float  # % of SKIP recommendations that would have lost money
    overall_accuracy: float

    # Calibration
    calibration_error: float  # How well probabilities match actual win rates

    # Returns
    avg_return_on_buys: float
    avg_return_on_skips: float  # What we avoided

    # By probability bucket
    accuracy_by_probability: Dict[str, float]


class AIPerformanceTracker:
    """
    Track and analyze AI recommendation performance.

    Features:
    - Log AI recommendations to database
    - Calculate actual returns after N days
    - Compare AI accuracy vs signal-based approach
    - Calibration analysis (predicted prob vs actual win rate)
    """

    def __init__(self):
        self.engine = get_engine()
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Create ai_recommendations table if not exists."""
        create_sql = """
        CREATE TABLE IF NOT EXISTS ai_recommendations (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10) NOT NULL,
            recommendation_date DATE NOT NULL,
            ai_probability DECIMAL(5,4),
            ai_ev DECIMAL(8,5),
            recommendation VARCHAR(10),
            entry_price DECIMAL(12,4),
            signal_scores JSONB,
            return_5d DECIMAL(8,4),
            return_10d DECIMAL(8,4),
            was_correct BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, recommendation_date)
        );
        
        CREATE INDEX IF NOT EXISTS idx_ai_rec_date ON ai_recommendations(recommendation_date);
        CREATE INDEX IF NOT EXISTS idx_ai_rec_ticker ON ai_recommendations(ticker);
        """

        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_sql)
                conn.commit()
            logger.info("AI recommendations table ready")
        except Exception as e:
            logger.warning(f"Could not create table: {e}")

    def log_recommendation(self,
                           ticker: str,
                           ai_probability: float,
                           ai_ev: float,
                           recommendation: str,
                           signal_scores: Dict[str, float],
                           entry_price: float = None) -> bool:
        """
        Log an AI recommendation to the database.

        Called when AI system analyzes a stock and makes a decision.
        """
        today = date.today()

        # Convert numpy types to Python native types
        ai_probability = float(ai_probability) if ai_probability is not None else None
        ai_ev = float(ai_ev) if ai_ev is not None else None
        entry_price = float(entry_price) if entry_price is not None else None

        # Clean signal_scores - convert any numpy types
        clean_scores = {}
        if signal_scores:
            for k, v in signal_scores.items():
                if v is not None:
                    try:
                        clean_scores[k] = float(v) if hasattr(v, '__float__') else v
                    except (TypeError, ValueError):
                        clean_scores[k] = str(v)

        insert_sql = """
        INSERT INTO ai_recommendations 
            (ticker, recommendation_date, ai_probability, ai_ev, 
             recommendation, entry_price, signal_scores)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, recommendation_date) 
        DO UPDATE SET
            ai_probability = EXCLUDED.ai_probability,
            ai_ev = EXCLUDED.ai_ev,
            recommendation = EXCLUDED.recommendation,
            entry_price = EXCLUDED.entry_price,
            signal_scores = EXCLUDED.signal_scores,
            updated_at = CURRENT_TIMESTAMP
        """

        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_sql, (
                        ticker,
                        today,
                        ai_probability,
                        ai_ev,
                        recommendation,
                        entry_price,
                        json.dumps(clean_scores)
                    ))
                conn.commit()
            logger.debug(f"Logged AI recommendation: {ticker} -> {recommendation} ({ai_probability:.1%})")
            return True
        except Exception as e:
            logger.error(f"Failed to log recommendation: {e}")
            return False

    def update_outcomes(self, days_back: int = 30) -> int:
        """
        Update outcomes for past recommendations.

        Fetches actual returns from historical_scores and updates
        the ai_recommendations table.
        """
        # Get recommendations that need outcome updates
        query = """
        SELECT r.id, r.ticker, r.recommendation_date, r.recommendation
        FROM ai_recommendations r
        WHERE r.return_5d IS NULL
          AND r.recommendation_date <= CURRENT_DATE - INTERVAL '5 days'
          AND r.recommendation_date >= CURRENT_DATE - INTERVAL '%s days'
        """

        try:
            pending = pd.read_sql(query % days_back, self.engine)
        except Exception as e:
            logger.error(f"Failed to get pending recommendations: {e}")
            return 0

        if pending.empty:
            return 0

        updated = 0

        for _, row in pending.iterrows():
            ticker = row['ticker']
            rec_date = row['recommendation_date']
            recommendation = row['recommendation']

            # Get actual return from historical_scores
            return_query = """
            SELECT return_5d, return_10d
            FROM historical_scores
            WHERE ticker = %s AND score_date = %s
            """

            try:
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(return_query, (ticker, rec_date))
                        result = cur.fetchone()

                if result:
                    return_5d, return_10d = result

                    # Determine if recommendation was correct
                    if recommendation == 'BUY':
                        was_correct = return_5d > 0 if return_5d else None
                    else:  # SKIP
                        was_correct = return_5d <= 0 if return_5d else None

                    # Update the record
                    update_sql = """
                    UPDATE ai_recommendations
                    SET return_5d = %s, return_10d = %s, was_correct = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """

                    with get_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute(update_sql, (return_5d, return_10d, was_correct, row['id']))
                        conn.commit()

                    updated += 1
            except Exception as e:
                logger.warning(f"Failed to update {ticker}: {e}")
                continue

        logger.info(f"Updated outcomes for {updated} recommendations")
        return updated

    def get_performance_metrics(self, days_back: int = 90) -> AIPerformanceMetrics:
        """Calculate comprehensive AI performance metrics."""

        query = """
        SELECT 
            ticker, recommendation_date, ai_probability, ai_ev,
            recommendation, return_5d, return_10d, was_correct
        FROM ai_recommendations
        WHERE recommendation_date >= CURRENT_DATE - INTERVAL '%s days'
          AND return_5d IS NOT NULL
        """

        try:
            df = pd.read_sql(query % days_back, self.engine)
        except Exception as e:
            logger.error(f"Failed to get performance data: {e}")
            return self._empty_metrics()

        if df.empty:
            return self._empty_metrics()

        total = len(df)
        buys = df[df['recommendation'] == 'BUY']
        skips = df[df['recommendation'] == 'SKIP']

        # Accuracy
        buy_correct = buys['was_correct'].sum() if len(buys) > 0 else 0
        buy_accuracy = buy_correct / len(buys) if len(buys) > 0 else 0

        skip_correct = skips['was_correct'].sum() if len(skips) > 0 else 0
        skip_accuracy = skip_correct / len(skips) if len(skips) > 0 else 0

        overall_correct = df['was_correct'].sum()
        overall_accuracy = overall_correct / total if total > 0 else 0

        # Calibration (how well probabilities match actual win rates)
        calibration_error = self._calculate_calibration_error(df)

        # Returns
        avg_return_buys = buys['return_5d'].mean() if len(buys) > 0 else 0
        avg_return_skips = skips['return_5d'].mean() if len(skips) > 0 else 0

        # By probability bucket
        accuracy_by_prob = self._calculate_accuracy_by_probability(df)

        return AIPerformanceMetrics(
            total_recommendations=total,
            buy_recommendations=len(buys),
            skip_recommendations=len(skips),
            buy_accuracy=buy_accuracy,
            skip_accuracy=skip_accuracy,
            overall_accuracy=overall_accuracy,
            calibration_error=calibration_error,
            avg_return_on_buys=float(avg_return_buys) if avg_return_buys else 0,
            avg_return_on_skips=float(avg_return_skips) if avg_return_skips else 0,
            accuracy_by_probability=accuracy_by_prob
        )

    def _calculate_calibration_error(self, df: pd.DataFrame) -> float:
        """Calculate expected calibration error."""
        if df.empty:
            return 0

        # Bucket by probability
        df['prob_bucket'] = pd.cut(df['ai_probability'],
                                   bins=[0, 0.4, 0.5, 0.6, 0.7, 1.0],
                                   labels=['<40%', '40-50%', '50-60%', '60-70%', '>70%'])

        calibration_errors = []
        for bucket, group in df.groupby('prob_bucket', observed=True):
            if len(group) < 3:
                continue
            predicted_prob = group['ai_probability'].mean()
            actual_win_rate = (group['return_5d'] > 0).mean()
            calibration_errors.append(abs(predicted_prob - actual_win_rate))

        return np.mean(calibration_errors) if calibration_errors else 0

    def _calculate_accuracy_by_probability(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate accuracy by probability bucket."""
        result = {}

        buckets = [
            ('<50%', 0, 0.50),
            ('50-55%', 0.50, 0.55),
            ('55-60%', 0.55, 0.60),
            ('60-65%', 0.60, 0.65),
            ('>65%', 0.65, 1.0),
        ]

        for label, low, high in buckets:
            mask = (df['ai_probability'] >= low) & (df['ai_probability'] < high)
            bucket_df = df[mask]

            if len(bucket_df) > 0:
                win_rate = (bucket_df['return_5d'] > 0).mean()
                result[label] = float(win_rate)

        return result

    def _empty_metrics(self) -> AIPerformanceMetrics:
        return AIPerformanceMetrics(
            total_recommendations=0,
            buy_recommendations=0,
            skip_recommendations=0,
            buy_accuracy=0,
            skip_accuracy=0,
            overall_accuracy=0,
            calibration_error=0,
            avg_return_on_buys=0,
            avg_return_on_skips=0,
            accuracy_by_probability={}
        )

    def get_recent_recommendations(self, days_back: int = 7) -> pd.DataFrame:
        """Get recent AI recommendations with outcomes."""
        query = """
        SELECT 
            ticker, recommendation_date, ai_probability, ai_ev,
            recommendation, entry_price, return_5d, was_correct
        FROM ai_recommendations
        WHERE recommendation_date >= CURRENT_DATE - INTERVAL '%s days'
        ORDER BY recommendation_date DESC, ai_probability DESC
        """

        try:
            return pd.read_sql(query % days_back, self.engine)
        except Exception as e:
            logger.error(f"Failed to get recent recommendations: {e}")
            return pd.DataFrame()

    def compare_ai_vs_signals(self, days_back: int = 90) -> Dict[str, any]:
        """
        Compare AI strategy vs signal-based strategy.

        Returns comparison metrics showing which approach performs better.
        """
        # AI recommendations
        ai_query = """
        SELECT recommendation, return_5d
        FROM ai_recommendations
        WHERE recommendation_date >= CURRENT_DATE - INTERVAL '%s days'
          AND return_5d IS NOT NULL
        """

        # Signal-based (BUY/STRONG_BUY signals)
        signal_query = """
        SELECT signal_type, return_5d
        FROM historical_scores
        WHERE score_date >= CURRENT_DATE - INTERVAL '%s days'
          AND return_5d IS NOT NULL
          AND signal_type IN ('BUY', 'STRONG_BUY')
        """

        try:
            ai_df = pd.read_sql(ai_query % days_back, self.engine)
            signal_df = pd.read_sql(signal_query % days_back, self.engine)
        except Exception as e:
            logger.error(f"Failed to compare strategies: {e}")
            return {}

        # AI BUYs only
        ai_buys = ai_df[ai_df['recommendation'] == 'BUY']

        comparison = {
            'ai_strategy': {
                'total_trades': len(ai_buys),
                'win_rate': (ai_buys['return_5d'] > 0).mean() if len(ai_buys) > 0 else 0,
                'avg_return': ai_buys['return_5d'].mean() if len(ai_buys) > 0 else 0,
            },
            'signal_strategy': {
                'total_trades': len(signal_df),
                'win_rate': (signal_df['return_5d'] > 0).mean() if len(signal_df) > 0 else 0,
                'avg_return': signal_df['return_5d'].mean() if len(signal_df) > 0 else 0,
            }
        }

        # Calculate improvement
        if comparison['signal_strategy']['win_rate'] > 0:
            comparison['improvement'] = {
                'win_rate_delta': comparison['ai_strategy']['win_rate'] - comparison['signal_strategy']['win_rate'],
                'return_delta': comparison['ai_strategy']['avg_return'] - comparison['signal_strategy']['avg_return'],
            }

        return comparison


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def log_ai_analysis(analysis_result) -> bool:
    """
    Helper to log AI analysis result to tracker.

    Call this from ai_trading_system.py after each analysis.

    Usage in ai_trading_system.py:
        from src.ml.ai_performance_tracker import log_ai_analysis

        result = self.analyze(ticker, scores, stock_data)
        log_ai_analysis(result)  # Add this line
    """
    try:
        tracker = AIPerformanceTracker()

        recommendation = 'BUY' if analysis_result.approved else 'SKIP'

        return tracker.log_recommendation(
            ticker=analysis_result.ticker,
            ai_probability=analysis_result.ml_probability,
            ai_ev=analysis_result.ml_ev,
            recommendation=recommendation,
            signal_scores={
                'total_score': getattr(analysis_result, 'total_score', 0),
                'combined_prob': analysis_result.combined_probability,
            },
            entry_price=analysis_result.entry_price
        )
    except Exception as e:
        logger.warning(f"Failed to log AI analysis: {e}")
        return False


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    tracker = AIPerformanceTracker()

    if len(sys.argv) > 1:
        if sys.argv[1] == '--update':
            updated = tracker.update_outcomes()
            print(f"Updated {updated} recommendation outcomes")

        elif sys.argv[1] == '--metrics':
            metrics = tracker.get_performance_metrics()
            print("\n📊 AI Performance Metrics")
            print("=" * 50)
            print(f"Total Recommendations: {metrics.total_recommendations}")
            print(f"BUY Recommendations: {metrics.buy_recommendations}")
            print(f"SKIP Recommendations: {metrics.skip_recommendations}")
            print(f"\nBUY Accuracy: {metrics.buy_accuracy:.1%}")
            print(f"SKIP Accuracy: {metrics.skip_accuracy:.1%}")
            print(f"Overall Accuracy: {metrics.overall_accuracy:.1%}")
            print(f"\nAvg Return on BUYs: {metrics.avg_return_on_buys:+.2f}%")
            print(f"Avg Return on SKIPs (avoided): {metrics.avg_return_on_skips:+.2f}%")
            print(f"\nCalibration Error: {metrics.calibration_error:.3f}")

            if metrics.accuracy_by_probability:
                print("\nAccuracy by Probability:")
                for bucket, accuracy in metrics.accuracy_by_probability.items():
                    print(f"  {bucket}: {accuracy:.1%}")

        elif sys.argv[1] == '--compare':
            comparison = tracker.compare_ai_vs_signals()
            print("\n📊 AI vs Signal Strategy Comparison")
            print("=" * 50)

            if 'ai_strategy' in comparison:
                ai = comparison['ai_strategy']
                sig = comparison['signal_strategy']

                print(f"\n{'Metric':<20} {'AI Strategy':<15} {'Signals':<15}")
                print("-" * 50)
                print(f"{'Total Trades':<20} {ai['total_trades']:<15} {sig['total_trades']:<15}")
                print(f"{'Win Rate':<20} {ai['win_rate']:.1%:<15} {sig['win_rate']:.1%:<15}")
                print(f"{'Avg Return':<20} {ai['avg_return']:+.2f}%{'':<10} {sig['avg_return']:+.2f}%")

                if 'improvement' in comparison:
                    imp = comparison['improvement']
                    print(f"\n{'Improvement':<20}")
                    print(f"{'Win Rate Delta':<20} {imp['win_rate_delta']:+.1%}")
                    print(f"{'Return Delta':<20} {imp['return_delta']:+.2f}%")

        elif sys.argv[1] == '--recent':
            df = tracker.get_recent_recommendations()
            if not df.empty:
                print("\n📋 Recent AI Recommendations")
                print(df.to_string(index=False))
            else:
                print("No recent recommendations found")

    else:
        print("Usage:")
        print("  python -m src.ml.ai_performance_tracker --update   # Update outcomes")
        print("  python -m src.ml.ai_performance_tracker --metrics  # Show metrics")
        print("  python -m src.ml.ai_performance_tracker --compare  # Compare AI vs Signals")
        print("  python -m src.ml.ai_performance_tracker --recent   # Show recent")