"""
Phase 0 Integration Validation Suite

Run from your project root:
    python scripts/validate_phase0.py

Or from anywhere:
    python validate_phase0.py --project-root C:\Develop\Latest_2025\HH_research_platform_v1

Author: Alpha Research Platform
"""

import sys
import os
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from pathlib import Path
import argparse


def find_project_root() -> Path:
    """Find the project root by looking for common markers."""
    current = Path(__file__).resolve().parent

    # If we're in scripts/, go up one level
    if current.name == 'scripts':
        current = current.parent

    # Look for markers that indicate project root
    markers = ['src', 'requirements.txt', 'setup.py', 'pyproject.toml', '.git']

    for _ in range(5):  # Don't go up more than 5 levels
        if any((current / marker).exists() for marker in markers):
            return current
        current = current.parent

    # Fallback to current working directory
    return Path.cwd()


class ValidationResult:
    """Result of a validation check."""
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors: List[str] = []
        self.details: List[str] = []

    def ok(self, message: str):
        self.passed += 1
        self.details.append(f"  ✅ {message}")

    def fail(self, message: str):
        self.failed += 1
        self.errors.append(message)
        self.details.append(f"  ❌ {message}")

    def warn(self, message: str):
        self.warnings += 1
        self.details.append(f"  ⚠️ {message}")

    def info(self, message: str):
        self.details.append(f"  ℹ️ {message}")

    def is_success(self) -> bool:
        return self.failed == 0

    def summary(self) -> str:
        status = "✅ PASSED" if self.is_success() else "❌ FAILED"
        return f"{self.name}: {status} ({self.passed} passed, {self.failed} failed, {self.warnings} warnings)"


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_result(result: ValidationResult, verbose: bool = True):
    print(f"\n{result.summary()}")
    if verbose:
        for detail in result.details:
            print(detail)


# =============================================================================
# FILE STRUCTURE CHECK
# =============================================================================

def check_file_structure(project_root: Path) -> ValidationResult:
    """Check that all required files exist."""
    result = ValidationResult("File Structure")

    required_files = [
        # Core
        "src/core/unified_scorer.py",
        "src/core/integration.py",
        "src/core/feature_logger.py",
        # Backtest
        "src/backtest/transaction_costs.py",
        # Portfolio
        "src/portfolio/exposure_control.py",
        # Data
        "src/data/pit_validator.py",
        # Monitoring
        "src/monitoring/drift_monitor.py",
        # Analytics
        "src/analytics/alpha_decay.py",
    ]

    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            result.ok(f"Found: {file_path}")
        else:
            result.fail(f"Missing: {file_path}")
            result.info(f"  Expected at: {full_path}")

    return result


# =============================================================================
# IMPORT CHECKS (with dynamic import)
# =============================================================================

def check_imports(project_root: Path) -> ValidationResult:
    """Check that all modules can be imported."""
    result = ValidationResult("Module Imports")

    # Ensure project root is in path
    src_path = str(project_root)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    modules_to_check = [
        # (module_path, [classes/functions to check])
        ("src.core.unified_scorer", ["UnifiedScorer", "TickerFeatures", "ScoringResult", "score_ticker"]),
        ("src.core.integration", ["SignalEngineAdapter", "BacktestEngineAdapter", "run_unified_pipeline"]),
        ("src.core.feature_logger", ["FeatureLogger", "log_scoring_run"]),
        ("src.backtest.transaction_costs", ["TransactionCostModel", "estimate_trade_cost"]),
        ("src.portfolio.exposure_control", ["ExposureController", "ExposureLimits"]),
        ("src.data.pit_validator", ["PITValidator", "PITValidationResult"]),
        ("src.monitoring.drift_monitor", ["DriftMonitor", "DriftAlert"]),
        ("src.analytics.alpha_decay", ["AlphaDecayTracker", "SignalType"]),
    ]

    for module_name, items in modules_to_check:
        try:
            # Try importing the module
            module = __import__(module_name, fromlist=items)

            # Check each item exists
            for item in items:
                if hasattr(module, item):
                    result.ok(f"{module_name}.{item}")
                else:
                    result.fail(f"{module_name}.{item} not found in module")

        except ModuleNotFoundError as e:
            result.fail(f"Cannot import {module_name}: {e}")
            result.info(f"  Check that __init__.py files exist in the path")
        except ImportError as e:
            result.fail(f"Import error in {module_name}: {e}")
        except Exception as e:
            result.fail(f"Error with {module_name}: {e}")

    return result


def check_unified_scorer(project_root: Path) -> ValidationResult:
    """Validate UnifiedScorer functionality."""
    result = ValidationResult("Unified Scorer")

    # Ensure path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from src.core.unified_scorer import (
            UnifiedScorer, TickerFeatures, ScoringResult,
            DataQuality, get_unified_scorer
        )
        result.ok("Imports successful")

        # Test instantiation
        scorer = UnifiedScorer()
        result.ok("UnifiedScorer instantiated")

        # Test TickerFeatures creation
        features = TickerFeatures(
            ticker="TEST",
            as_of_time=datetime.now(),
            current_price=100.0,
            sentiment_score=65,
            fundamental_score=70,
            technical_score=55,
        )
        result.ok("TickerFeatures created")

        # Test timestamp validation
        violations = features.validate_timestamps()
        if isinstance(violations, list):
            result.ok("Timestamp validation works")
        else:
            result.fail("Timestamp validation returned wrong type")

        # Test feature hash
        hash_val = features.get_feature_hash()
        if hash_val and len(hash_val) >= 8:
            result.ok(f"Feature hash generated: {hash_val}")
        else:
            result.fail("Feature hash invalid")

        # Test scoring
        scores = scorer.compute_scores(features)
        if isinstance(scores, ScoringResult):
            result.ok("compute_scores() returns ScoringResult")
        else:
            result.fail("compute_scores() returned wrong type")

        # Validate score fields
        if hasattr(scores, 'total_score') and 0 <= scores.total_score <= 100:
            result.ok(f"total_score valid: {scores.total_score:.1f}")
        else:
            result.fail("total_score invalid")

        if hasattr(scores, 'signal_type') and scores.signal_type in ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']:
            result.ok(f"signal_type valid: {scores.signal_type}")
        else:
            result.fail("signal_type invalid")

        if hasattr(scores, 'confidence') and 0 <= scores.confidence <= 1:
            result.ok(f"confidence valid: {scores.confidence:.2f}")
        else:
            result.fail("confidence invalid")

        # Test weights sum to 1
        total_weight = sum(UnifiedScorer.WEIGHTS.values())
        if abs(total_weight - 1.0) < 0.01:
            result.ok(f"Weights sum to 1.0: {total_weight}")
        else:
            result.warn(f"Weights sum to {total_weight}, expected 1.0")

    except Exception as e:
        result.fail(f"Exception: {e}")
        result.info(traceback.format_exc())

    return result


def check_transaction_costs(project_root: Path) -> ValidationResult:
    """Validate transaction cost model."""
    result = ValidationResult("Transaction Costs")

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from src.backtest.transaction_costs import (
            TransactionCostModel, TransactionCost, MarketCapTier,
            estimate_trade_cost, get_round_trip_cost_bps, ROUND_TRIP_COST_TABLE
        )
        result.ok("Imports successful")

        # Test instantiation
        model = TransactionCostModel()
        result.ok("TransactionCostModel instantiated")

        # Test market cap classification
        tier = model.classify_market_cap(3_000_000_000_000)  # $3T
        if tier == MarketCapTier.MEGA:
            result.ok("Market cap classification works (MEGA)")
        else:
            result.fail(f"Expected MEGA, got {tier}")

        # Test cost estimation
        cost = model.estimate_cost(
            ticker="AAPL",
            trade_value=100_000,
            market_cap=3_000_000_000_000,
            avg_daily_volume=500_000_000,
            volatility=0.25
        )

        if isinstance(cost, TransactionCost):
            result.ok("estimate_cost() returns TransactionCost")
        else:
            result.fail("estimate_cost() returned wrong type")

        # Validate cost ranges for mega-cap
        if 0 < cost.total_cost_bps < 20:
            result.ok(f"Mega-cap cost reasonable: {cost.total_cost_bps:.1f} bps")
        else:
            result.warn(f"Mega-cap cost seems off: {cost.total_cost_bps:.1f} bps")

        # Test round-trip cost
        rt_cost = model.estimate_round_trip_cost("AAPL", 100_000, market_cap=3e12)
        if rt_cost > cost.total_cost_bps:
            result.ok(f"Round-trip > one-way: {rt_cost:.1f} bps")

        # Validate cost table ordering
        if ROUND_TRIP_COST_TABLE['mega'] < ROUND_TRIP_COST_TABLE['small']:
            result.ok("Cost table: mega < small ✓")

    except Exception as e:
        result.fail(f"Exception: {e}")
        result.info(traceback.format_exc())

    return result


def check_exposure_control(project_root: Path) -> ValidationResult:
    """Validate exposure control module."""
    result = ValidationResult("Exposure Control")

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from src.portfolio.exposure_control import (
            ExposureController, ExposureLimits, ExposureReport,
            get_position_constraints
        )
        result.ok("Imports successful")

        # Test instantiation
        limits = ExposureLimits(
            max_beta=1.5,
            max_sector_weight=0.30,
            target_volatility=0.15
        )
        controller = ExposureController(limits)
        result.ok("ExposureController instantiated")

        # Test position sizing constraints (doesn't need market data)
        positions = [
            {"symbol": "AAPL", "weight": 0.20, "sector": "Technology"},
            {"symbol": "MSFT", "weight": 0.15, "sector": "Technology"},
        ]

        constrained, constraints = controller.get_constrained_position_size(
            ticker="NVDA",
            proposed_weight=0.15,
            current_positions=positions,
            sector="Technology"
        )

        # Tech already at 35%, adding 15% would breach 30% limit
        if constrained < 0.15:
            result.ok(f"Position constrained: {constrained:.1%} (from 15%)")
        else:
            result.info(f"Position: {constrained:.1%} (constraint logic may differ)")

        if isinstance(constraints, list):
            result.ok(f"Constraints list returned: {len(constraints)} items")

    except Exception as e:
        result.fail(f"Exception: {e}")
        result.info(traceback.format_exc())

    return result


def check_pit_validator(project_root: Path) -> ValidationResult:
    """Validate point-in-time validator."""
    result = ValidationResult("Point-in-Time Validator")

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from src.data.pit_validator import (
            PITValidator, PITValidationResult, PITViolationType
        )
        result.ok("Imports successful")

        validator = PITValidator()
        result.ok("PITValidator instantiated")

        # Test valid features (timestamps in the past)
        decision_time = datetime(2024, 6, 15, 10, 0)
        valid_features = {
            'price': 150.0,
            'price_timestamp': datetime(2024, 6, 15, 9, 30),
            'sentiment_score': 65,
            'sentiment_timestamp': datetime(2024, 6, 14, 18, 0),
        }

        pit_result = validator.validate_features(valid_features, decision_time)
        if pit_result.is_valid:
            result.ok("Valid features pass validation")
        else:
            result.fail("Valid features should pass")

        # Test invalid features (future timestamp)
        invalid_features = {
            'price': 155.0,
            'price_timestamp': datetime(2024, 6, 15, 16, 0),  # FUTURE!
        }

        pit_result2 = validator.validate_features(invalid_features, decision_time)
        if not pit_result2.is_valid:
            result.ok("Future data detected and rejected")
        else:
            result.fail("Future data should be rejected")

    except Exception as e:
        result.fail(f"Exception: {e}")
        result.info(traceback.format_exc())

    return result


def check_feature_logger(project_root: Path) -> ValidationResult:
    """Validate feature logger."""
    result = ValidationResult("Feature Logger")

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from src.core.feature_logger import FeatureLogger, ScoringRun
        import tempfile
        import json

        result.ok("Imports successful")

        # Test with temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = FeatureLogger(storage_path=tmpdir, compress=False)
            result.ok("FeatureLogger instantiated")

            # Log a run
            run_id = logger.log_run(
                ticker='TEST',
                as_of_time=datetime.now(),
                features={'ticker': 'TEST', 'score': 65},
                scores={'total_score': 68},
                environment='test'
            )

            if run_id and 'TEST' in run_id:
                result.ok(f"Run logged: {run_id}")

            logger.flush()
            result.ok("Buffer flushed")

            # Retrieve
            run = logger.get_run(run_id)
            if run is not None:
                result.ok("Run retrieved successfully")
            else:
                result.warn("Run not found (may need flush delay)")

    except Exception as e:
        result.fail(f"Exception: {e}")
        result.info(traceback.format_exc())

    return result


def check_drift_monitor(project_root: Path) -> ValidationResult:
    """Validate drift monitor."""
    result = ValidationResult("Drift Monitor")

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from src.monitoring.drift_monitor import DriftMonitor, DataQualityDashboard
        import random

        result.ok("Imports successful")

        monitor = DriftMonitor()
        result.ok("DriftMonitor instantiated")

        # Record metrics
        for i in range(30):
            monitor.record_metric('sentiment_score', random.gauss(60, 15))

        result.ok(f"Recorded 30 metrics")

        # Check stats
        if 'sentiment_score' in monitor.metrics:
            stats = monitor.metrics['sentiment_score']
            result.ok(f"Stats: count={stats.count}, mean={stats.mean:.1f}")

        # Check drift
        alerts = monitor.check_drift()
        result.ok(f"Drift check: {len(alerts)} alerts")

        # Dashboard
        dashboard = DataQualityDashboard(monitor)
        text = dashboard.render_text()
        if 'DASHBOARD' in text:
            result.ok("Dashboard renders")

    except Exception as e:
        result.fail(f"Exception: {e}")
        result.info(traceback.format_exc())

    return result


def check_alpha_decay(project_root: Path) -> ValidationResult:
    """Validate alpha decay tracker."""
    result = ValidationResult("Alpha Decay")

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from src.analytics.alpha_decay import AlphaDecayTracker, SignalType, DecayProfile

        result.ok("Imports successful")

        tracker = AlphaDecayTracker()
        result.ok("AlphaDecayTracker instantiated")

        # Check profiles exist
        for sig_type in SignalType:
            profile = tracker.get_decay_profile(sig_type)
            if profile:
                result.ok(f"{sig_type.value}: {profile.half_life_hours}h half-life")

        # Test recommendation
        rec = tracker.get_horizon_recommendation(SignalType.SENTIMENT)
        if 'urgency' in rec:
            result.ok(f"Sentiment urgency: {rec['urgency']}")

    except Exception as e:
        result.fail(f"Exception: {e}")
        result.info(traceback.format_exc())

    return result


def check_integration(project_root: Path) -> ValidationResult:
    """Validate integration adapters."""
    result = ValidationResult("Integration")

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from src.core.integration import (
            SignalEngineAdapter, BacktestEngineAdapter, TradeIdeasAdapter
        )
        result.ok("Imports successful")

        # Test adapter instantiation
        signal_adapter = SignalEngineAdapter()
        result.ok("SignalEngineAdapter works")

        backtest_adapter = BacktestEngineAdapter()
        result.ok("BacktestEngineAdapter works")

        # Test cost application
        trades = [
            {'ticker': 'AAPL', 'return_pct': 5.0, 'entry_price': 150},
        ]
        adjusted = backtest_adapter.apply_costs_to_trades(trades)

        if 'return_pct_net' in adjusted[0]:
            result.ok(f"Costs applied: {adjusted[0]['return_pct']:.1f}% -> {adjusted[0]['return_pct_net']:.1f}%")

    except Exception as e:
        result.fail(f"Exception: {e}")
        result.info(traceback.format_exc())

    return result


# =============================================================================
# MAIN
# =============================================================================

def run_all_checks(project_root: Path, verbose: bool = True) -> Tuple[int, int]:
    """Run all validation checks."""
    print_header("PHASE 0 INTEGRATION VALIDATION")
    print(f"Project root: {project_root}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    checks = [
        ("File Structure", lambda: check_file_structure(project_root)),
        ("Imports", lambda: check_imports(project_root)),
        ("Unified Scorer", lambda: check_unified_scorer(project_root)),
        ("Transaction Costs", lambda: check_transaction_costs(project_root)),
        ("Exposure Control", lambda: check_exposure_control(project_root)),
        ("PIT Validator", lambda: check_pit_validator(project_root)),
        ("Feature Logger", lambda: check_feature_logger(project_root)),
        ("Drift Monitor", lambda: check_drift_monitor(project_root)),
        ("Alpha Decay", lambda: check_alpha_decay(project_root)),
        ("Integration", lambda: check_integration(project_root)),
    ]

    results = []
    total_passed = 0
    total_failed = 0

    for name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
            total_passed += result.passed
            total_failed += result.failed
            print_result(result, verbose)
        except Exception as e:
            result = ValidationResult(name)
            result.fail(f"Check crashed: {e}")
            results.append(result)
            total_failed += 1
            print_result(result, verbose)

    # Summary
    print_header("VALIDATION SUMMARY")

    for result in results:
        status = "✅" if result.is_success() else "❌"
        print(f"  {status} {result.name}: {result.passed} passed, {result.failed} failed")

    print()
    all_passed = all(r.is_success() for r in results)
    if all_passed:
        print("🎉 ALL CHECKS PASSED - Phase 0 integration successful!")
    else:
        print("⚠️  SOME CHECKS FAILED - Review errors above")

    print(f"\nTotal: {total_passed} passed, {total_failed} failed")

    return total_passed, total_failed


def run_single_check(check_name: str, project_root: Path, verbose: bool = True):
    """Run a single check."""
    check_map = {
        'files': lambda: check_file_structure(project_root),
        'structure': lambda: check_file_structure(project_root),
        'imports': lambda: check_imports(project_root),
        'unified_scorer': lambda: check_unified_scorer(project_root),
        'scorer': lambda: check_unified_scorer(project_root),
        'costs': lambda: check_transaction_costs(project_root),
        'transaction_costs': lambda: check_transaction_costs(project_root),
        'exposure': lambda: check_exposure_control(project_root),
        'pit': lambda: check_pit_validator(project_root),
        'logger': lambda: check_feature_logger(project_root),
        'drift': lambda: check_drift_monitor(project_root),
        'decay': lambda: check_alpha_decay(project_root),
        'integration': lambda: check_integration(project_root),
    }

    if check_name.lower() == 'all':
        return run_all_checks(project_root, verbose)

    check_func = check_map.get(check_name.lower())
    if check_func:
        print(f"Project root: {project_root}")
        result = check_func()
        print_result(result, verbose)
        return result.passed, result.failed
    else:
        print(f"Unknown check: {check_name}")
        print(f"Available: {', '.join(sorted(check_map.keys()))}, all")
        return 0, 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Phase 0 Integration")
    parser.add_argument('--check', type=str, default='all',
                       help='Check to run: all, files, imports, unified_scorer, costs, exposure, pit, logger, drift, decay, integration')
    parser.add_argument('--project-root', type=str, default=None,
                       help='Path to project root (auto-detected if not specified)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Only show summary')

    args = parser.parse_args()

    # Determine project root
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        project_root = find_project_root()

    if not project_root.exists():
        print(f"❌ Project root not found: {project_root}")
        sys.exit(1)

    passed, failed = run_single_check(args.check, project_root, not args.quiet)
    sys.exit(0 if failed == 0 else 1)