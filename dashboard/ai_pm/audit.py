from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, Optional


def _jsonable(obj: Any) -> Any:
    # datetime first (important because dataclass->asdict can contain nested datetimes)
    if isinstance(obj, datetime):
        return obj.isoformat()

    if obj is None:
        return None

    # primitive types
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # dataclasses -> dict, then recurse to convert nested datetimes
    if is_dataclass(obj):
        try:
            return _jsonable(asdict(obj))
        except Exception:
            return {"_repr": repr(obj)}

    # containers
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}

    return {"_repr": repr(obj)}


def write_ai_pm_audit(
    *,
    base_dir: str = "logs/ai_pm",
    account: str,
    strategy_key: str,
    snapshot: Any = None,
    signals: Any = None,
    signals_diagnostics: Optional[Dict[str, Any]] = None,
    targets: Any = None,
    plan: Any = None,
    gates: Any = None,
    execution: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Writes a single JSON audit file and returns its filepath.
    """
    os.makedirs(base_dir, exist_ok=True)

    ts = datetime.utcnow()
    safe_account = (account or "UNKNOWN").replace("/", "_").replace("\\", "_").replace(" ", "_")
    safe_strategy = (strategy_key or "UNKNOWN").replace("/", "_").replace("\\", "_").replace(" ", "_")

    fname = f"{ts.strftime('%Y%m%d_%H%M%S')}_acct-{safe_account}_strat-{safe_strategy}.json"
    path = os.path.join(base_dir, fname)

    payload = {
        "ts_utc": ts.isoformat(),
        "account": account,
        "strategy_key": strategy_key,
        "snapshot": _jsonable(snapshot),
        "signals": _jsonable(signals),
        "signals_diagnostics": _jsonable(signals_diagnostics or {}),
        "targets": _jsonable(targets),
        "plan": _jsonable(plan),
        "gates": _jsonable(gates),
        "execution": _jsonable(execution),
        "extra": _jsonable(extra or {}),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path
