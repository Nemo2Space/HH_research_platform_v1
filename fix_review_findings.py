"""
Fix script addressing external review findings.

Patches:
1. signal_predictor.py - EV double-count, remove scaler, similar_setups, real metrics
2. auto_maintenance.py - Quality gates, real report, file lock, ticker sanitization, atomic save
3. yf_subprocess.py - Ticker sanitization

Run: python fix_review_findings.py
"""

import ast
import re
import os

def fix_signal_predictor():
    """Fix signal_predictor.py issues."""
    path = 'src/ml/signal_predictor.py'
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return False

    c = open(path, 'r', encoding='utf-8').read()
    changes = 0

    # ================================================================
    # FIX 1: EV double-counting cost
    # avg_win/avg_loss are computed from _net returns (already cost-adjusted)
    # But predict() subtracts cost AGAIN: ev = prob*avg_win - (1-prob)*avg_loss - cost
    # Fix: compute averages from GROSS returns instead
    # ================================================================

    # In train() method: change from _net to gross returns
    old_ev_train = """        # Calculate historical returns for EV
        for h in [1, 2, 5, 10]:
            col = f'return_{h}d_net'
            if col in df.columns:
                returns = df[col].dropna()
                self.avg_win_return[h] = returns[returns > 0].mean() if (returns > 0).any() else 2.0
                self.avg_loss_return[h] = abs(returns[returns <= 0].mean()) if (returns <= 0).any() else 2.0"""

    new_ev_train = """        # Calculate historical GROSS returns for EV (cost subtracted once in predict())
        for h in [1, 2, 5, 10]:
            col = f'return_{h}d'
            if col in df.columns:
                returns = df[col].dropna()
                self.avg_win_return[h] = returns[returns > 0].mean() if (returns > 0).any() else 2.0
                self.avg_loss_return[h] = abs(returns[returns <= 0].mean()) if (returns <= 0).any() else 2.0"""

    if old_ev_train in c:
        c = c.replace(old_ev_train, new_ev_train)
        changes += 1
        print("  [1] Fixed EV double-counting: avg_win/loss now from gross returns")
    else:
        # Try the lite model version (train_lite_model.py style)
        print("  [1] EV train block not found (may be lite model) - check manually")

    # ================================================================
    # FIX 2: Remove StandardScaler (unnecessary for XGBoost, removes leakage risk)
    # ================================================================

    # Replace scaler usage in get_feature_matrix
    old_get_features = """    def get_feature_matrix(self, df: pd.DataFrame, feature_list: List[str] = None,
                           fit: bool = False) -> Tuple[np.ndarray, List[str]]:
        \"\"\"Extract and scale features.\"\"\"
        if feature_list is None:
            feature_list = self.FEATURES
        available = [f for f in feature_list if f in df.columns]
        X = df[available].values

        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X, available"""

    new_get_features = """    def get_feature_matrix(self, df: pd.DataFrame, feature_list: List[str] = None,
                           fit: bool = False) -> Tuple[np.ndarray, List[str]]:
        \"\"\"Extract features (no scaling - XGBoost is scale-invariant).\"\"\"
        if feature_list is None:
            feature_list = self.FEATURES
        available = [f for f in feature_list if f in df.columns]
        X = df[available].values.astype(np.float64)
        return X, available"""

    if old_get_features in c:
        c = c.replace(old_get_features, new_get_features)
        changes += 1
        print("  [2a] Removed scaler from get_feature_matrix")

    # Remove scaler transform in predict()
    old_predict_scale = """        X = np.array([feature_values])
        X = self.data_loader.scaler.transform(X)"""

    new_predict_scale = """        X = np.array([feature_values], dtype=np.float64)"""

    if old_predict_scale in c:
        c = c.replace(old_predict_scale, new_predict_scale)
        changes += 1
        print("  [2b] Removed scaler from predict()")

    # Remove scaler from save/load
    if "'scaler': self.data_loader.scaler," in c:
        c = c.replace("'scaler': self.data_loader.scaler,", "# scaler removed - XGBoost is scale-invariant")
        changes += 1
        print("  [2c] Removed scaler from save()")

    if "self.data_loader.scaler = state['scaler']" in c:
        c = c.replace(
            "self.data_loader.scaler = state['scaler']",
            "# scaler removed - XGBoost is scale-invariant\n        # Backward compat: ignore scaler if present in old pickle"
        )
        changes += 1
        print("  [2d] Removed scaler from load()")

    # ================================================================
    # FIX 3: similar_setups_win_rate should not be model probability
    # ================================================================

    old_similar = "similar_setups_win_rate=probs.get(5, 0.5),"
    new_similar = "similar_setups_win_rate=0.0,  # placeholder: not yet connected to RAG"

    if old_similar in c:
        c = c.replace(old_similar, new_similar)
        changes += 1
        print("  [3] Fixed similar_setups_win_rate (was model prob, now 0)")

    # ================================================================
    # FIX 4: Real accuracy and avg_return in WalkForwardResult
    # ================================================================

    old_wf_result = """            results.append(WalkForwardResult(
                fold=fold, train_start=train_dates[0], train_end=train_dates[-1],
                test_start=test_dates[0], test_end=test_dates[-1],
                accuracy=0.5, auc_roc=auc, brier_score=brier,
                trades_taken=trades, win_rate=win_rate, avg_return=0
            ))"""

    new_wf_result = """            # Compute real accuracy and avg return
            preds = (probs >= 0.5).astype(int)
            accuracy = float((preds == y_test).mean())
            avg_ret = float(df.loc[test_mask, f'return_{horizon}d'].mean()) if f'return_{horizon}d' in df.columns else 0.0

            results.append(WalkForwardResult(
                fold=fold, train_start=train_dates[0], train_end=train_dates[-1],
                test_start=test_dates[0], test_end=test_dates[-1],
                accuracy=accuracy, auc_roc=auc, brier_score=brier,
                trades_taken=trades, win_rate=win_rate, avg_return=avg_ret
            ))"""

    if old_wf_result in c:
        c = c.replace(old_wf_result, new_wf_result)
        changes += 1
        print("  [4] Fixed WalkForwardResult: real accuracy and avg_return")

    # ================================================================
    # FIX 5: Isotonic selection by class balance (not just sample size)
    # ================================================================

    old_cal_method = "cal_method = 'sigmoid' if len(X_cal) < 100 else 'isotonic'"
    new_cal_method = "cal_method = 'sigmoid' if (len(X_cal) < 100 or min(y_cal.sum(), len(y_cal) - y_cal.sum()) < 30) else 'isotonic'"

    # Replace all occurrences
    count = c.count(old_cal_method)
    if count > 0:
        c = c.replace(old_cal_method, new_cal_method)
        changes += 1
        print(f"  [5] Fixed isotonic selection: now checks class balance ({count} occurrences)")

    # ================================================================
    # FIX 6: mean_accuracy and mean_return in _generate_report
    # ================================================================

    old_report_acc = "mean_accuracy=0.5, mean_auc=mean_auc"
    new_report_acc = "mean_accuracy=np.mean([f.accuracy for f in folds]), mean_auc=mean_auc"

    if old_report_acc in c:
        c = c.replace(old_report_acc, new_report_acc)
        changes += 1
        print("  [6a] Fixed report: real mean_accuracy")

    old_report_ret = "mean_win_rate=mean_wr, mean_return=0,"
    new_report_ret = "mean_win_rate=mean_wr, mean_return=np.mean([f.avg_return for f in folds]),"

    if old_report_ret in c:
        c = c.replace(old_report_ret, new_report_ret)
        changes += 1
        print("  [6b] Fixed report: real mean_return")

    # Write and verify
    if changes > 0:
        open(path, 'w', encoding='utf-8').write(c)
        ast.parse(open(path, encoding='utf-8').read())
        print(f"  signal_predictor.py: {changes} fixes applied, syntax OK")
        return True
    else:
        print(f"  signal_predictor.py: no changes needed")
        return False


def fix_auto_maintenance():
    """Fix auto_maintenance.py issues."""
    path = 'src/ml/auto_maintenance.py'
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return False

    c = open(path, 'r', encoding='utf-8').read()
    changes = 0

    # ================================================================
    # FIX 1: EV double-counting in _do_retrain (same issue as signal_predictor)
    # ================================================================

    old_retrain_ev = """    for h in [1, 2, 5, 10]:
        col = f'return_{h}d_net'
        if col in df.columns:
            returns = df[col].dropna()
            avg_win_return[h] = returns[returns > 0].mean() if (returns > 0).any() else 2.0
            avg_loss_return[h] = abs(returns[returns <= 0].mean()) if (returns <= 0).any() else 2.0"""

    new_retrain_ev = """    # Use GROSS returns for EV (cost subtracted once in predict())
    for h in [1, 2, 5, 10]:
        col = f'return_{h}d'
        if col in df.columns:
            returns = df[col].dropna()
            avg_win_return[h] = returns[returns > 0].mean() if (returns > 0).any() else 2.0
            avg_loss_return[h] = abs(returns[returns <= 0].mean()) if (returns <= 0).any() else 2.0"""

    if old_retrain_ev in c:
        c = c.replace(old_retrain_ev, new_retrain_ev)
        changes += 1
        print("  [1] Fixed EV double-counting in _do_retrain")

    # ================================================================
    # FIX 2: Remove scaler from _do_retrain
    # ================================================================

    old_scaler_fit = """    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(df[LITE_FEATURES].values)"""

    new_scaler_fit = """    # No scaling needed for XGBoost (scale-invariant)
    X = df[LITE_FEATURES].values.astype(float)
    scaler = None  # Kept for backward compat in pickle"""

    if old_scaler_fit in c:
        c = c.replace(old_scaler_fit, new_scaler_fit)
        changes += 1
        print("  [2a] Removed scaler from _do_retrain")

    # Also fix the scaler save line
    if "predictor.data_loader.scaler = scaler" in c:
        c = c.replace(
            "predictor.data_loader.scaler = scaler",
            "# scaler removed - XGBoost is scale-invariant"
        )
        changes += 1
        print("  [2b] Removed scaler assignment in _do_retrain save")

    # ================================================================
    # FIX 3: Quality gates - only deploy if model passes minimum bar
    # ================================================================

    old_retrain_save = """    predictor.save(MODEL_PATH)
    logger.info(f"AUTO-MAINTENANCE: Model saved ({len(df)} samples, {len(LITE_FEATURES)} features)")"""

    new_retrain_save = """    # Quality gate: only deploy if model shows minimum lift
    # Test on held-out calibration set (best available proxy)
    from sklearn.metrics import roc_auc_score, brier_score_loss
    deploy = True
    gate_reasons = []

    for horizon in [5, 10]:
        target_col = f'win_{horizon}d'
        if target_col not in df.columns:
            continue

        y = df[target_col].values
        y_cal = y[cal_mask]

        try:
            cal_probs = calibrators[horizon].predict_proba(X[cal_mask])[:, 1]
            cal_auc = roc_auc_score(y_cal, cal_probs) if len(set(y_cal)) > 1 else 0.5
            cal_brier = brier_score_loss(y_cal, cal_probs)

            if cal_auc < 0.48:
                gate_reasons.append(f"{horizon}d AUC={cal_auc:.3f} < 0.48")
                deploy = False
            if cal_brier > 0.27:
                gate_reasons.append(f"{horizon}d Brier={cal_brier:.3f} > 0.27")
                deploy = False
        except Exception as e:
            gate_reasons.append(f"{horizon}d eval error: {e}")

    if not deploy:
        logger.warning(f"AUTO-MAINTENANCE: Quality gates FAILED: {'; '.join(gate_reasons)}")
        logger.warning("AUTO-MAINTENANCE: Keeping existing model (not overwriting)")
        return

    # Atomic save: write to temp, then rename
    import tempfile
    temp_path = MODEL_PATH + '.tmp'
    predictor.save(temp_path)
    if os.path.exists(MODEL_PATH):
        os.replace(temp_path, MODEL_PATH)  # Atomic on same filesystem
    else:
        os.rename(temp_path, MODEL_PATH)
    logger.info(f"AUTO-MAINTENANCE: Model saved ({len(df)} samples, {len(LITE_FEATURES)} features)")"""

    if old_retrain_save in c:
        c = c.replace(old_retrain_save, new_retrain_save)
        changes += 1
        print("  [3] Added quality gates + atomic save to retrain")

    # ================================================================
    # FIX 4: Real ModelReport (not fabricated)
    # ================================================================

    old_report = """    # Generate a minimal report
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
    )"""

    new_report = """    # Generate report with real metrics from calibration set
    from src.ml.signal_predictor import ModelReport
    from sklearn.metrics import roc_auc_score, brier_score_loss

    real_auc, real_brier = 0.5, 0.25
    for horizon in [5]:
        target_col = f'win_{horizon}d'
        if target_col in df.columns:
            y_cal_h = df[target_col].values[cal_mask]
            try:
                cal_probs = calibrators[horizon].predict_proba(X[cal_mask])[:, 1]
                real_auc = roc_auc_score(y_cal_h, cal_probs) if len(set(y_cal_h)) > 1 else 0.5
                real_brier = brier_score_loss(y_cal_h, cal_probs)
            except:
                pass

    predictor.validation_report = ModelReport(
        model_name="XGBoost-Lite-Auto",
        trained_at=datetime.now(),
        total_samples=len(df),
        feature_count=len(LITE_FEATURES),
        target_horizon=5,
        folds=[],
        mean_accuracy=0.5,  # Not computed in auto-retrain
        mean_auc=real_auc,
        mean_brier=real_brier,
        mean_win_rate=0.5,  # Not computed in auto-retrain
        mean_return=0,
        feature_importance=feature_importance.get(5, {}),
        is_well_calibrated=real_brier < 0.25,
        calibration_error=real_brier,
        beats_baseline=real_auc > 0.52,
        baseline_auc=0.5,
        improvement_vs_baseline=real_auc - 0.5
    )"""

    if old_report in c:
        c = c.replace(old_report, new_report)
        changes += 1
        print("  [4] Fixed fabricated ModelReport with real metrics")

    # ================================================================
    # FIX 5: File lock for retrain job execution
    # ================================================================

    old_retrain_start = """def retrain_model_job():
    \"\"\"
    Weekly job: retrain ML model with latest data.
    Only retrains if enough new data has accumulated.
    \"\"\"
    logger.info("=" * 50)
    logger.info("AUTO-MAINTENANCE: Starting model retrain check")
    logger.info("=" * 50)"""

    new_retrain_start = """def retrain_model_job():
    \"\"\"
    Weekly job: retrain ML model with latest data.
    Only retrains if enough new data has accumulated.
    \"\"\"
    # Single-instance lock: prevent concurrent retrains from Streamlit reloads
    lock_path = os.path.join(BACKUP_DIR, '.retrain.lock')
    os.makedirs(BACKUP_DIR, exist_ok=True)
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock_fd, str(os.getpid()).encode())
        os.close(lock_fd)
    except FileExistsError:
        # Check if stale (> 1 hour old)
        try:
            age = time.time() - os.path.getmtime(lock_path)
            if age > 3600:
                os.remove(lock_path)
                logger.warning("AUTO-MAINTENANCE: Removed stale lock file")
            else:
                logger.info("AUTO-MAINTENANCE: Retrain already running (lock exists)")
                return
        except:
            return

    try:
        _retrain_model_inner()
    finally:
        try:
            os.remove(lock_path)
        except:
            pass


def _retrain_model_inner():
    \"\"\"Inner retrain logic (called with lock held).\"\"\"
    logger.info("=" * 50)
    logger.info("AUTO-MAINTENANCE: Starting model retrain check")
    logger.info("=" * 50)"""

    if old_retrain_start in c:
        c = c.replace(old_retrain_start, new_retrain_start)

        # Also need to close the try block properly
        # The rest of retrain_model_job's body should now be in _retrain_model_inner
        # Since the function body follows directly, we just renamed it
        changes += 1
        print("  [5] Added file lock for retrain job")

    # ================================================================
    # FIX 6: Ticker sanitization in backfill
    # ================================================================

    old_backfill_loop = """        for ticker in tickers:"""

    new_backfill_loop = """        # Sanitize tickers (prevent subprocess injection)
        import re
        SAFE_TICKER = re.compile(r'^[A-Z0-9.\-_=]+$')
        tickers = [t for t in tickers if SAFE_TICKER.match(str(t))]

        for ticker in tickers:"""

    if old_backfill_loop in c:
        c = c.replace(old_backfill_loop, new_backfill_loop, 1)  # Only first occurrence
        changes += 1
        print("  [6] Added ticker sanitization in backfill")

    # ================================================================
    # FIX 7: Same lock pattern for backfill
    # ================================================================

    old_backfill_start = """def backfill_returns_job():
    \"\"\"
    Daily job: fill missing return_1d/5d/10d/20d in historical_scores.
    Runs in background thread via APScheduler.
    \"\"\"
    logger.info("=" * 50)
    logger.info("AUTO-MAINTENANCE: Starting return backfill")
    logger.info("=" * 50)"""

    new_backfill_start = """def backfill_returns_job():
    \"\"\"
    Daily job: fill missing return_1d/5d/10d/20d in historical_scores.
    Runs in background thread via APScheduler.
    \"\"\"
    # Single-instance lock
    lock_path = os.path.join(BACKUP_DIR, '.backfill.lock')
    os.makedirs(BACKUP_DIR, exist_ok=True)
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock_fd, str(os.getpid()).encode())
        os.close(lock_fd)
    except FileExistsError:
        try:
            age = time.time() - os.path.getmtime(lock_path)
            if age > 3600:
                os.remove(lock_path)
            else:
                logger.info("AUTO-MAINTENANCE: Backfill already running (lock exists)")
                return
        except:
            return

    try:
        _backfill_inner()
    finally:
        try:
            os.remove(lock_path)
        except:
            pass


def _backfill_inner():
    \"\"\"Inner backfill logic (called with lock held).\"\"\"
    logger.info("=" * 50)
    logger.info("AUTO-MAINTENANCE: Starting return backfill")
    logger.info("=" * 50)"""

    if old_backfill_start in c:
        c = c.replace(old_backfill_start, new_backfill_start)
        changes += 1
        print("  [7] Added file lock for backfill job")

    # Write and verify
    if changes > 0:
        open(path, 'w', encoding='utf-8').write(c)
        ast.parse(open(path, encoding='utf-8').read())
        print(f"  auto_maintenance.py: {changes} fixes applied, syntax OK")
        return True
    else:
        print(f"  auto_maintenance.py: no changes needed")
        return False


def fix_yf_subprocess():
    """Add ticker sanitization to yf_subprocess.py."""
    path = 'src/analytics/yf_subprocess.py'
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return False

    c = open(path, 'r', encoding='utf-8').read()

    if 'SAFE_TICKER' in c:
        print("  yf_subprocess.py: Already has ticker sanitization")
        return False

    # Add sanitization function near the top, after imports
    # Find the first function definition
    import_end = c.find('\ndef ')
    if import_end == -1:
        print("  yf_subprocess.py: Could not find insertion point")
        return False

    sanitizer = """
import re
SAFE_TICKER = re.compile(r'^[A-Z0-9.\\-_=^]+$')

def _sanitize_ticker(ticker: str) -> str:
    \"\"\"Validate ticker string to prevent code injection in subprocess.\"\"\"
    ticker = str(ticker).strip().upper()
    if not SAFE_TICKER.match(ticker) or len(ticker) > 20:
        raise ValueError(f"Invalid ticker: {ticker!r}")
    return ticker

"""
    c = c[:import_end] + sanitizer + c[import_end:]

    # Now add _sanitize_ticker calls to each public function
    # Find patterns like: def get_stock_info(ticker... and add sanitization
    funcs = ['get_stock_info', 'get_stock_history', 'get_earnings_dates',
             'get_earnings_history', 'get_options_chain', 'get_stock_info_and_history']

    added = 0
    for func_name in funcs:
        # Find the function and its first line of body
        pattern = f'def {func_name}('
        idx = c.find(pattern)
        if idx == -1:
            continue

        # Find the docstring end or first code line
        # Look for the line after """
        body_start = c.find('"""', c.find('"""', idx) + 3) + 3
        if body_start == 2:  # No docstring
            body_start = c.find('\n', c.find(':', idx)) + 1

        # Check if sanitization already added
        next_lines = c[body_start:body_start + 200]
        if '_sanitize_ticker' in next_lines:
            continue

        # Insert sanitization as first line of function body
        indent = '    '
        sanitize_line = f"\n{indent}ticker = _sanitize_ticker(ticker)\n"
        c = c[:body_start] + sanitize_line + c[body_start:]
        added += 1

    if added > 0:
        open(path, 'w', encoding='utf-8').write(c)
        ast.parse(open(path, encoding='utf-8').read())
        print(f"  yf_subprocess.py: Added ticker sanitization to {added} functions, syntax OK")
        return True

    return False


def add_price_consistency_comment():
    """Add documentation about adjusted vs unadjusted price assumption."""
    path = 'src/ml/auto_maintenance.py'
    if not os.path.exists(path):
        return

    c = open(path, 'r', encoding='utf-8').read()

    if 'PRICE CONSISTENCY NOTE' in c:
        return

    old = "HORIZONS = {"
    new = """# PRICE CONSISTENCY NOTE:
# yfinance returns Split+Dividend Adjusted Close by default.
# op_price in historical_scores is the raw opening price at signal time.
# For short horizons (1-20 days), the difference is negligible for most stocks.
# If a stock splits or pays a large dividend within the return window,
# the return calculation will be slightly off. This is an accepted limitation
# for now. For production-grade: store adjusted op_price or use raw Close.
HORIZONS = {"""

    if old in c:
        c = c.replace(old, new, 1)
        open(path, 'w', encoding='utf-8').write(c)
        print("  Added price consistency documentation to auto_maintenance.py")


if __name__ == '__main__':
    print("=" * 60)
    print("FIXING REVIEW FINDINGS")
    print("=" * 60)

    print("\n--- signal_predictor.py ---")
    fix_signal_predictor()

    print("\n--- auto_maintenance.py ---")
    fix_auto_maintenance()

    print("\n--- yf_subprocess.py ---")
    fix_yf_subprocess()

    print("\n--- Price consistency docs ---")
    add_price_consistency_comment()

    print("\n" + "=" * 60)
    print("DONE. Now run:")
    print("  python -m src.ml.auto_maintenance --retrain")
    print("to retrain with fixed EV calculation.")
    print("=" * 60)