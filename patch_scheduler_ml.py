"""Patch scheduler.py to include ML maintenance jobs."""
import ast

# ============================================================
# 1. Patch scheduler.py - add ML jobs to default schedule
# ============================================================
content = open('src/alerts/scheduler.py', 'r', encoding='utf-8').read()

# Add register_ml_jobs call to setup_default_schedule
old = """        self.add_regime_check(interval_minutes=120)

        logger.info("Default schedule configured")"""

new = """        self.add_regime_check(interval_minutes=120)

        # ML auto-maintenance: daily return backfill + weekly model retrain
        try:
            from src.ml.auto_maintenance import register_ml_jobs
            register_ml_jobs(self)
        except Exception as e:
            logger.warning(f"Could not register ML maintenance jobs: {e}")

        logger.info("Default schedule configured")"""

if 'register_ml_jobs' not in content:
    if old in content:
        content = content.replace(old, new)
        open('src/alerts/scheduler.py', 'w', encoding='utf-8').write(content)
        print("1. scheduler.py: Added ML maintenance jobs to default schedule")
    else:
        print("1. scheduler.py: WARNING - could not find insertion point")
else:
    print("1. scheduler.py: Already patched")

# Verify syntax
ast.parse(open('src/alerts/scheduler.py', encoding='utf-8').read())
print("   scheduler.py: Syntax OK")

# ============================================================
# 2. Verify auto_maintenance.py is in place
# ============================================================
import os
target = 'src/ml/auto_maintenance.py'
if os.path.exists(target):
    ast.parse(open(target, encoding='utf-8').read())
    print(f"2. {target}: Exists and syntax OK")
else:
    print(f"2. {target}: NOT FOUND - copy auto_maintenance.py to {target}")

print("\nDone! Restart Streamlit to activate automated ML maintenance.")
print("\nSchedule:")
print("  - Return backfill: Daily 22:30 Zurich (after US market close)")
print("  - Model retrain:   Sunday 03:00 Zurich (weekly)")
print("\nManual run:")
print("  python -m src.ml.auto_maintenance --backfill")
print("  python -m src.ml.auto_maintenance --retrain")
print("  python -m src.ml.auto_maintenance --both")