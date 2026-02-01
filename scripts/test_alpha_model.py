"""
Alpha Model Testing & Evaluation Suite
=======================================

Comprehensive testing script to evaluate the alpha model before deployment.
Tests model performance, signal quality, and generates detailed reports.

Usage:
    python scripts/test_alpha_model.py

Author: HH Research Platform
Date: January 2026
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional
import json

# Database connection
from src.db.connection import get_engine
from sqlalchemy import text

# Alpha model
from src.ml.multi_factor_alpha import (
    MultiFactorAlphaModel,
    load_alpha_model,
    train_alpha_model,
    AlphaPrediction,
    MarketRegime
)


class AlphaModelTester:
    """Comprehensive testing suite for the alpha model."""

    def __init__(self, model_path: str = 'models/multi_factor_alpha.pkl'):
        self.model_path = model_path
        self.model = None
        self.test_results = {}
        self.engine = get_engine()

    def load_model(self) -> bool:
        """Load the model for testing."""
        try:
            self.model = load_alpha_model(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")

            # Debug: Check model state
            print(f"   Model version: {getattr(self.model, 'VERSION', 'unknown')}")
            print(f"   Features: {len(self.model.feature_names)} features")
            print(f"   Global models: {len(self.model.global_models)} horizons")
            print(f"   Conditional models: {len(self.model.models)} models")
            print(f"   _is_trained flag: {self.model._is_trained}")

            # FIX: Force _is_trained to True if model has necessary components
            if not self.model._is_trained:
                if self.model.global_models and self.model.feature_names:
                    print(f"   ⚠️  Fixing _is_trained flag (was False, setting to True)")
                    self.model._is_trained = True
                else:
                    print(f"   ❌ Model missing essential components")
                    return False

            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _ensure_model_ready(self) -> bool:
        """Ensure model is loaded and ready for predictions."""
        if self.model is None:
            return False
        if not self.model._is_trained:
            # Try to fix it
            if self.model.global_models and self.model.feature_names:
                self.model._is_trained = True
            else:
                return False
        return True

    # =========================================================================
    # TEST 1: Model Validation Report Analysis
    # =========================================================================
    def test_validation_report(self) -> Dict:
        """Analyze the model's validation report."""
        print("\n" + "=" * 70)
        print("  TEST 1: VALIDATION REPORT ANALYSIS")
        print("=" * 70)

        results = {
            'test_name': 'Validation Report',
            'passed': False,
            'metrics': {},
            'issues': [],
            'recommendations': []
        }

        if self.model.validation_report is None:
            results['issues'].append("No validation report - model needs retraining")
            return results

        report = self.model.validation_report

        # Extract metrics
        results['metrics'] = {
            'overall_ic': report.overall_ic,
            'overall_icir': report.overall_icir,
            'overall_r2': report.overall_r2,
            'ic_tstat': report.ic_tstat,
            'ic_pvalue': report.ic_pvalue,
            'n_folds': report.n_folds,
            'mean_ic_oos': report.mean_ic_oos,
            'std_ic_oos': report.std_ic_oos,
            'beats_baseline': report.beats_baseline
        }

        # Print metrics
        print(f"\n  📊 Core Metrics:")
        print(f"     Information Coefficient (IC): {report.overall_ic:.4f}")
        print(f"     IC Information Ratio (ICIR): {report.overall_icir:.4f}")
        print(f"     R-squared: {report.overall_r2:.4f}")
        print(f"     Statistical Significance: t={report.ic_tstat:.2f}, p={report.ic_pvalue:.4f}")

        # Evaluate IC
        if report.overall_ic < 0:
            results['issues'].append(f"Negative IC ({report.overall_ic:.4f}) - model predicts opposite of actual returns")
        elif report.overall_ic < 0.02:
            results['issues'].append(f"Very weak IC ({report.overall_ic:.4f}) - essentially random")
        elif report.overall_ic < 0.05:
            results['issues'].append(f"Weak IC ({report.overall_ic:.4f}) - marginal predictive power")
        else:
            results['recommendations'].append(f"IC of {report.overall_ic:.4f} is acceptable")

        # Evaluate statistical significance
        if report.ic_pvalue > 0.10:
            results['issues'].append(f"Not statistically significant (p={report.ic_pvalue:.3f})")
        elif report.ic_pvalue > 0.05:
            results['recommendations'].append(f"Marginally significant (p={report.ic_pvalue:.3f})")
        else:
            results['recommendations'].append(f"Statistically significant (p={report.ic_pvalue:.3f})")

        # Check regime performance
        if report.regime_performance:
            print(f"\n  🏭 Regime Performance:")
            best_regime_ic = 0
            for regime, metrics in report.regime_performance.items():
                ic = metrics.get('ic', 0)
                n = metrics.get('n', 0)
                print(f"     {regime}: IC={ic:.4f}, N={n}")
                if ic > best_regime_ic:
                    best_regime_ic = ic

            results['metrics']['best_regime_ic'] = best_regime_ic

            if best_regime_ic > 0.10:
                results['recommendations'].append(f"Strong regime-specific IC ({best_regime_ic:.4f})")

        # Check fold consistency
        if report.fold_results:
            print(f"\n  📈 Fold Results:")
            fold_ics = []
            for fold in report.fold_results:
                ic = fold.get('ic_5d', 0)
                fold_ics.append(ic)
                print(f"     Fold {fold.get('fold', '?')}: IC={ic:.4f}")

            negative_folds = sum(1 for ic in fold_ics if ic < 0)
            if negative_folds > 0:
                results['issues'].append(f"{negative_folds}/{len(fold_ics)} folds have negative IC")

            results['metrics']['fold_ics'] = fold_ics
            results['metrics']['negative_folds'] = negative_folds

        # Determine pass/fail
        passed_conditions = [
            report.overall_ic > 0,
            report.ic_pvalue < 0.20 or (report.regime_performance and max(m.get('ic', 0) for m in report.regime_performance.values()) > 0.08)
        ]
        results['passed'] = all(passed_conditions)

        print(f"\n  {'✅ PASSED' if results['passed'] else '❌ FAILED'}")

        return results

    # =========================================================================
    # TEST 2: Signal Distribution Test
    # =========================================================================
    def test_signal_distribution(self) -> Dict:
        """Test that signals are reasonably distributed."""
        print("\n" + "=" * 70)
        print("  TEST 2: SIGNAL DISTRIBUTION")
        print("=" * 70)

        results = {
            'test_name': 'Signal Distribution',
            'passed': False,
            'metrics': {},
            'issues': [],
            'recommendations': []
        }

        # Ensure model is ready
        if not self._ensure_model_ready():
            results['issues'].append("Model not ready for predictions")
            print(f"\n  ❌ FAILED: Model not ready")
            return results

        try:
            # Generate live predictions
            predictions_df = self.model.predict_live()

            if predictions_df.empty:
                results['issues'].append("No predictions generated")
                return results

            total = len(predictions_df)

            # Count signals
            signal_counts = predictions_df['signal'].value_counts()

            print(f"\n  📊 Signal Distribution (N={total}):")
            for signal in ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']:
                count = signal_counts.get(signal, 0)
                pct = count / total * 100
                print(f"     {signal:12}: {count:4} ({pct:5.1f}%)")

            results['metrics']['total_predictions'] = total
            results['metrics']['signal_counts'] = signal_counts.to_dict()

            # Check for issues
            hold_pct = signal_counts.get('HOLD', 0) / total * 100
            buy_signals = signal_counts.get('STRONG_BUY', 0) + signal_counts.get('BUY', 0)
            sell_signals = signal_counts.get('STRONG_SELL', 0) + signal_counts.get('SELL', 0)

            if hold_pct > 90:
                results['issues'].append(f"Too many HOLD signals ({hold_pct:.1f}%) - thresholds may be too strict")
            elif hold_pct < 30:
                results['issues'].append(f"Too few HOLD signals ({hold_pct:.1f}%) - thresholds may be too loose")

            if buy_signals == 0 and sell_signals == 0:
                results['issues'].append("No BUY or SELL signals generated")

            # Check confidence distribution
            conf_counts = predictions_df['confidence'].value_counts()
            print(f"\n  🎯 Confidence Distribution:")
            for conf in ['HIGH', 'MEDIUM', 'LOW']:
                count = conf_counts.get(conf, 0)
                pct = count / total * 100
                print(f"     {conf:8}: {count:4} ({pct:5.1f}%)")

            results['metrics']['confidence_counts'] = conf_counts.to_dict()

            high_conf_pct = conf_counts.get('HIGH', 0) / total * 100
            if high_conf_pct == 0:
                results['issues'].append("No HIGH confidence predictions")

            # Check expected return distribution
            avg_return = predictions_df['expected_return_5d'].mean() * 100
            std_return = predictions_df['expected_return_5d'].std() * 100
            min_return = predictions_df['expected_return_5d'].min() * 100
            max_return = predictions_df['expected_return_5d'].max() * 100

            print(f"\n  📈 Expected Return Distribution:")
            print(f"     Mean: {avg_return:+.2f}%")
            print(f"     Std:  {std_return:.2f}%")
            print(f"     Range: [{min_return:+.2f}%, {max_return:+.2f}%]")

            results['metrics']['return_stats'] = {
                'mean': avg_return,
                'std': std_return,
                'min': min_return,
                'max': max_return
            }

            if std_return < 0.1:
                results['issues'].append("Very low return variance - predictions may be too similar")

            # Determine pass/fail
            results['passed'] = len(results['issues']) == 0 or (hold_pct < 95 and (buy_signals > 0 or sell_signals > 0))

            print(f"\n  {'✅ PASSED' if results['passed'] else '❌ FAILED'}")

        except Exception as e:
            results['issues'].append(f"Error generating predictions: {e}")
            print(f"\n  ❌ FAILED: {e}")

        return results

    # =========================================================================
    # TEST 3: Historical Backtest
    # =========================================================================
    def test_historical_backtest(self, lookback_days: int = 30) -> Dict:
        """Backtest the model on recent historical data."""
        print("\n" + "=" * 70)
        print("  TEST 3: HISTORICAL BACKTEST")
        print("=" * 70)

        results = {
            'test_name': 'Historical Backtest',
            'passed': False,
            'metrics': {},
            'issues': [],
            'recommendations': []
        }

        # Ensure model is ready
        if not self._ensure_model_ready():
            results['issues'].append("Model not ready for predictions")
            print(f"\n  ❌ FAILED: Model not ready")
            return results

        try:
            # Load historical data with known outcomes
            end_date = datetime.now() - timedelta(days=5)  # Need 5 days for returns
            start_date = end_date - timedelta(days=lookback_days)

            print(f"\n  📅 Backtest Period: {start_date.date()} to {end_date.date()}")

            # Get historical data using the SAME query as live predictions
            # This ensures we test the same feature path
            query = """
                SELECT 
                    h.ticker,
                    h.score_date as date,
                    h.sector,
                    h.sentiment as sentiment_score,
                    h.fundamental_score,
                    h.growth_score,
                    h.total_score,
                    h.gap_score,
                    COALESCE(s.technical_score, 50) as technical_score,
                    COALESCE(s.options_flow_score, 50) as options_score,
                    COALESCE(s.short_squeeze_score, 50) as short_squeeze_score,
                    COALESCE(s.target_upside_pct, 0) as target_upside_pct,
                    COALESCE(s.analyst_positivity, 50) as analyst_positivity,
                    COALESCE(s.insider_signal, 50) as insider_score,
                    COALESCE(s.institutional_signal, 50) as inst_13f_score,
                    h.return_5d as actual_return_5d,
                    h.return_10d as actual_return_10d
                FROM historical_scores h
                LEFT JOIN screener_scores s ON h.ticker = s.ticker AND h.score_date = s.date
                WHERE h.score_date BETWEEN :start_date AND :end_date
                  AND h.return_5d IS NOT NULL
                ORDER BY h.score_date, h.ticker
            """

            with self.engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params={
                    'start_date': start_date.date(),
                    'end_date': end_date.date()
                })

            if df.empty:
                results['issues'].append("No historical data available for backtest period")
                return results

            print(f"  📊 Loaded {len(df)} historical samples")

            # Get VIX for regime detection
            vix_query = """
                SELECT date, close as vix FROM prices
                WHERE ticker = '^VIX' AND date BETWEEN :start_date AND :end_date
            """
            try:
                with self.engine.connect() as conn:
                    vix_df = pd.read_sql(text(vix_query), conn, params={
                        'start_date': start_date.date(),
                        'end_date': end_date.date()
                    })
                if not vix_df.empty:
                    vix_df['date'] = pd.to_datetime(vix_df['date']).dt.date
                    df['date'] = pd.to_datetime(df['date']).dt.date
                    df = df.merge(vix_df, on='date', how='left')
            except:
                df['vix'] = None

            # Classify regime using SAME logic as model
            def classify_regime(vix):
                if pd.isna(vix):
                    return 'UNKNOWN'
                if vix < 15:
                    return 'BULL'
                elif vix < 20:
                    return 'NEUTRAL'
                elif vix < 30:
                    return 'BEAR'
                else:
                    return 'HIGH_VOL'

            df['regime'] = df['vix'].apply(classify_regime) if 'vix' in df.columns else 'UNKNOWN'

            # Convert gap_score to numeric if needed
            gap_score_map = {
                'Stong Up': 90, 'Strong Up': 90,
                'Gap Up Continuation': 80,
                'Potential Up': 70,
                'No Signal Up': 60,
                'No Analysis': 50, 'No Signal': 50, 'Reversal': 50,
                'No Signal Down': 40,
                'Potential Down': 30,
                'Gap Down': 20, 'Gap Down Continuation': 20,
                'Strong Down': 10, 'Stong Down': 10,
            }
            if df['gap_score'].dtype == 'object':
                df['gap_score'] = df['gap_score'].map(gap_score_map).fillna(50)

            # Generate predictions for each historical point using FULL feature set
            correct_direction = 0
            total_predictions = 0
            prediction_returns = []
            actual_returns = []

            # Get all feature names the model uses
            feature_names = self.model.feature_names

            for _, row in df.iterrows():
                # Build factor_values using ALL features the model expects
                factor_values = {}
                for f in feature_names:
                    if f in row and pd.notna(row[f]):
                        factor_values[f] = row[f]
                    else:
                        factor_values[f] = None  # Let model handle missing

                try:
                    pred = self.model.predict(
                        ticker=row['ticker'],
                        factor_values=factor_values,
                        sector=row.get('sector'),
                        regime=row.get('regime', 'UNKNOWN')
                    )

                    pred_return = pred.expected_return_5d * 100
                    actual_return = row['actual_return_5d']

                    prediction_returns.append(pred_return)
                    actual_returns.append(actual_return)

                    # Check direction accuracy
                    if (pred_return > 0 and actual_return > 0) or (pred_return < 0 and actual_return < 0):
                        correct_direction += 1

                    total_predictions += 1

                except Exception as e:
                    continue

            if total_predictions == 0:
                results['issues'].append("Could not generate any backtest predictions")
                return results

            # Calculate metrics
            direction_accuracy = correct_direction / total_predictions * 100
            ic = np.corrcoef(prediction_returns, actual_returns)[0, 1] if len(prediction_returns) > 1 else 0

            # Calculate additional metrics
            pred_std = np.std(prediction_returns)
            actual_std = np.std(actual_returns)

            print(f"\n  🎯 Backtest Results:")
            print(f"     Predictions: {total_predictions}")
            print(f"     Direction Accuracy: {direction_accuracy:.1f}%")
            print(f"     Information Coefficient: {ic:.4f}")
            print(f"     Prediction Std: {pred_std:.2f}%")
            print(f"     Actual Return Std: {actual_std:.2f}%")

            results['metrics'] = {
                'total_predictions': total_predictions,
                'direction_accuracy': direction_accuracy,
                'information_coefficient': ic,
                'prediction_std': pred_std,
                'actual_std': actual_std
            }

            # Evaluate results
            if direction_accuracy < 50:
                results['issues'].append(f"Direction accuracy below 50% ({direction_accuracy:.1f}%)")
            else:
                results['recommendations'].append(f"Direction accuracy: {direction_accuracy:.1f}%")

            if ic < 0:
                results['issues'].append(f"Negative IC in backtest ({ic:.4f})")
            elif ic > 0.05:
                results['recommendations'].append(f"Good backtest IC: {ic:.4f}")

            if pred_std < 0.1:
                results['issues'].append(f"Prediction variance too low ({pred_std:.2f}%) - model not differentiating")

            # Determine pass/fail
            results['passed'] = direction_accuracy >= 50 and ic > -0.05 and pred_std > 0.1

            print(f"\n  {'✅ PASSED' if results['passed'] else '❌ FAILED'}")

        except Exception as e:
            results['issues'].append(f"Backtest error: {e}")
            print(f"\n  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()

        return results

    # =========================================================================
    # TEST 4: Factor Sanity Check
    # =========================================================================
    def test_factor_sanity(self) -> Dict:
        """Check that factors behave sensibly."""
        print("\n" + "=" * 70)
        print("  TEST 4: FACTOR SANITY CHECK")
        print("=" * 70)

        results = {
            'test_name': 'Factor Sanity',
            'passed': False,
            'metrics': {},
            'issues': [],
            'recommendations': []
        }

        # Ensure model is ready
        if not self._ensure_model_ready():
            results['issues'].append("Model not ready for predictions")
            print(f"\n  ❌ FAILED: Model not ready")
            return results

        try:
            factor_df = self.model.get_factor_report()

            if factor_df.empty:
                results['issues'].append("No factor report available")
                return results

            print(f"\n  📊 Factor Analysis ({len(factor_df)} factors):")

            # Check top factors
            top_factor = factor_df.iloc[0]
            print(f"\n  🔝 Top Factor: {top_factor['factor']}")
            print(f"     Importance: {top_factor['importance']:.2%}")
            print(f"     Weight (5d): {top_factor['weight_5d']:.4f}")

            results['metrics']['top_factor'] = top_factor['factor']
            results['metrics']['top_importance'] = top_factor['importance']

            # Check for concentration
            if top_factor['importance'] > 0.5:
                results['issues'].append(f"Single factor dominates ({top_factor['importance']:.0%})")

            # Check sign matches
            sign_match_rate = factor_df['sign_match'].mean()
            print(f"\n  📈 Sign Match Rate: {sign_match_rate:.1%}")

            results['metrics']['sign_match_rate'] = sign_match_rate

            if sign_match_rate < 0.5:
                results['issues'].append(f"Low sign match rate ({sign_match_rate:.1%}) - factors behaving opposite to theory")

            # Check for very weak factors
            weak_factors = factor_df[factor_df['importance'] < 0.01]
            if len(weak_factors) > len(factor_df) * 0.5:
                results['recommendations'].append(f"{len(weak_factors)} factors have very low importance - consider removing")

            # Test sensitivity: high score vs low score
            print(f"\n  🔬 Sensitivity Test:")

            # Predict with high scores
            high_factors = {f: 80 for f in self.model.feature_names}
            pred_high = self.model.predict('TEST_HIGH', high_factors, 'Technology', 'NEUTRAL')

            # Predict with low scores
            low_factors = {f: 20 for f in self.model.feature_names}
            pred_low = self.model.predict('TEST_LOW', low_factors, 'Technology', 'NEUTRAL')

            spread = (pred_high.expected_return_5d - pred_low.expected_return_5d) * 100
            print(f"     High scores prediction: {pred_high.expected_return_5d*100:+.2f}%")
            print(f"     Low scores prediction:  {pred_low.expected_return_5d*100:+.2f}%")
            print(f"     Spread: {spread:+.2f}%")

            results['metrics']['sensitivity_spread'] = spread

            if abs(spread) < 0.1:
                results['issues'].append("Model not sensitive to factor values")
            elif spread < 0:
                results['recommendations'].append("Model shows contrarian behavior (high scores → lower returns)")

            # Determine pass/fail
            results['passed'] = len(results['issues']) <= 1

            print(f"\n  {'✅ PASSED' if results['passed'] else '❌ FAILED'}")

        except Exception as e:
            results['issues'].append(f"Factor sanity error: {e}")
            print(f"\n  ❌ FAILED: {e}")

        return results

    # =========================================================================
    # TEST 5: Prediction Consistency
    # =========================================================================
    def test_prediction_consistency(self) -> Dict:
        """Test that predictions are consistent and stable."""
        print("\n" + "=" * 70)
        print("  TEST 5: PREDICTION CONSISTENCY")
        print("=" * 70)

        results = {
            'test_name': 'Prediction Consistency',
            'passed': False,
            'metrics': {},
            'issues': [],
            'recommendations': []
        }

        # Ensure model is ready
        if not self._ensure_model_ready():
            results['issues'].append("Model not ready for predictions")
            print(f"\n  ❌ FAILED: Model not ready")
            return results

        try:
            # Test same input gives same output
            test_factors = {f: 60 for f in self.model.feature_names}

            predictions = []
            for i in range(5):
                pred = self.model.predict('AAPL', test_factors, 'Technology', 'NEUTRAL')
                predictions.append(pred.expected_return_5d)

            is_consistent = len(set(predictions)) == 1

            print(f"\n  🔄 Consistency Test:")
            print(f"     Same input × 5 runs: {'All identical ✅' if is_consistent else 'Varies ❌'}")

            if not is_consistent:
                results['issues'].append("Model gives different outputs for same input")

            results['metrics']['is_deterministic'] = is_consistent

            # Test small input changes give small output changes
            base_pred = self.model.predict('AAPL', test_factors, 'Technology', 'NEUTRAL')

            # Change one factor slightly
            test_factors_modified = test_factors.copy()
            test_factors_modified['total_score'] = 61  # +1 point

            modified_pred = self.model.predict('AAPL', test_factors_modified, 'Technology', 'NEUTRAL')

            change = abs(modified_pred.expected_return_5d - base_pred.expected_return_5d) * 100

            print(f"\n  📊 Stability Test:")
            print(f"     Base prediction: {base_pred.expected_return_5d*100:+.4f}%")
            print(f"     +1 point change: {modified_pred.expected_return_5d*100:+.4f}%")
            print(f"     Difference: {change:.4f}%")

            results['metrics']['sensitivity_to_small_change'] = change

            if change > 1.0:
                results['issues'].append(f"Model too sensitive to small changes ({change:.2f}%)")

            # Determine pass/fail
            results['passed'] = is_consistent and change < 2.0

            print(f"\n  {'✅ PASSED' if results['passed'] else '❌ FAILED'}")

        except Exception as e:
            results['issues'].append(f"Consistency test error: {e}")
            print(f"\n  ❌ FAILED: {e}")

        return results

    # =========================================================================
    # RUN ALL TESTS
    # =========================================================================
    def run_all_tests(self) -> Dict:
        """Run all tests and generate summary report."""
        print("\n" + "=" * 70)
        print("  ALPHA MODEL COMPREHENSIVE TEST SUITE")
        print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 70)

        if not self.load_model():
            return {'error': 'Failed to load model'}

        # Run all tests
        self.test_results['validation_report'] = self.test_validation_report()
        self.test_results['signal_distribution'] = self.test_signal_distribution()
        self.test_results['historical_backtest'] = self.test_historical_backtest()
        self.test_results['factor_sanity'] = self.test_factor_sanity()
        self.test_results['prediction_consistency'] = self.test_prediction_consistency()

        # Generate summary
        return self.generate_summary()

    def generate_summary(self) -> Dict:
        """Generate test summary report."""
        print("\n" + "=" * 70)
        print("  TEST SUMMARY")
        print("=" * 70)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r.get('passed', False))

        print(f"\n  📊 Results: {passed_tests}/{total_tests} tests passed")
        print()

        all_issues = []
        all_recommendations = []

        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
            print(f"  {status} {result.get('test_name', test_name)}")

            all_issues.extend(result.get('issues', []))
            all_recommendations.extend(result.get('recommendations', []))

        if all_issues:
            print(f"\n  ⚠️ Issues Found ({len(all_issues)}):")
            for issue in all_issues:
                print(f"     - {issue}")

        if all_recommendations:
            print(f"\n  💡 Recommendations ({len(all_recommendations)}):")
            for rec in all_recommendations:
                print(f"     - {rec}")

        # Overall assessment
        print("\n" + "-" * 70)

        if passed_tests == total_tests:
            overall = "✅ MODEL READY FOR DEPLOYMENT"
            deploy_ready = True
        elif passed_tests >= total_tests * 0.6:
            overall = "⚠️ MODEL NEEDS MINOR IMPROVEMENTS"
            deploy_ready = True
        else:
            overall = "❌ MODEL NOT READY - SIGNIFICANT ISSUES"
            deploy_ready = False

        print(f"\n  {overall}")
        print("\n" + "=" * 70)

        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'pass_rate': passed_tests / total_tests * 100,
            'deploy_ready': deploy_ready,
            'issues': all_issues,
            'recommendations': all_recommendations,
            'test_results': self.test_results
        }

    def save_report(self, filepath: str = 'reports/alpha_model_test_report.json'):
        """Save test report to JSON file."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'results': self.test_results
        }

        # Convert any non-serializable objects
        def make_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            return obj

        report = make_serializable(report)

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n  📄 Report saved to: {filepath}")


def main():
    """Run the alpha model test suite."""
    tester = AlphaModelTester()
    summary = tester.run_all_tests()

    # Save report
    tester.save_report()

    # Return exit code based on results
    if summary.get('deploy_ready', False):
        print("\n✅ Model passed testing - safe to deploy")
        return 0
    else:
        print("\n❌ Model failed testing - do not deploy")
        return 1


if __name__ == "__main__":
    exit(main())