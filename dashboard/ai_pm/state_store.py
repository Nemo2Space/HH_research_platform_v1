from __future__ import annotations

from dataclasses import is_dataclass, asdict
from datetime import datetime, date
from typing import Any, Dict, Optional

import streamlit as st


_STATE_KEY = "ai_pm_state_v1"
_LAST_KEY = "ai_pm_last_v1"


def _to_jsonable(x: Any) -> Any:
    # Never fabricate data. Only convert types to JSON-safe shapes.
    if x is None:
        return None

    if isinstance(x, (str, int, float, bool)):
        return x

    if isinstance(x, (datetime, date)):
        # ISO format for audit + session store
        return x.isoformat()

    # pandas Timestamp / numpy scalars
    try:
        import pandas as pd  # optional

        if isinstance(x, pd.Timestamp):
            return x.to_pydatetime().isoformat()
    except Exception:
        pass

    try:
        import numpy as np  # optional

        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
    except Exception:
        pass

    if is_dataclass(x):
        return _to_jsonable(asdict(x))

    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]

    # generic objects (best effort)
    if hasattr(x, "__dict__"):
        try:
            return _to_jsonable(vars(x))
        except Exception:
            pass

    # fallback: string repr (still not fabricated; just for visibility/debug)
    return str(x)


def init_ai_pm_state() -> None:
    if _STATE_KEY not in st.session_state:
        st.session_state[_STATE_KEY] = {
            "selected_account": None,
            "strategy_key": None,
            "auto_trade": False,
            "kill_switch": False,
            "armed_until_utc": None,
            "last_run_utc": None,
        }
    if _LAST_KEY not in st.session_state:
        st.session_state[_LAST_KEY] = {
            "snapshot": None,
            "signals": None,
            "signals_diagnostics": None,
            "targets": None,
            "plan": None,
            "gates": None,
            "execution": None,
            "errors": [],
            "warnings": [],
        }


def _s() -> Dict[str, Any]:
    init_ai_pm_state()
    return st.session_state[_STATE_KEY]


def _l() -> Dict[str, Any]:
    init_ai_pm_state()
    return st.session_state[_LAST_KEY]


def set_selected_account(account: Optional[str]) -> None:
    _s()["selected_account"] = account


def get_selected_account() -> Optional[str]:
    return _s().get("selected_account")


def set_strategy_key(strategy_key: Optional[str]) -> None:
    _s()["strategy_key"] = strategy_key


def get_strategy_key() -> Optional[str]:
    return _s().get("strategy_key")


def set_auto_trade(v: bool) -> None:
    _s()["auto_trade"] = bool(v)


def is_auto_trade() -> bool:
    return bool(_s().get("auto_trade"))


def set_kill_switch(v: bool) -> None:
    _s()["kill_switch"] = bool(v)


def is_kill_switch() -> bool:
    return bool(_s().get("kill_switch"))


def arm_auto_trade(minutes: int = 240) -> None:
    _s()["armed_until_utc"] = datetime.utcnow().timestamp() + (int(minutes) * 60)


def disarm_auto_trade() -> None:
    _s()["armed_until_utc"] = None


def is_armed() -> bool:
    ts = _s().get("armed_until_utc")
    if ts is None:
        return False
    try:
        return float(ts) > datetime.utcnow().timestamp()
    except Exception:
        return False


def get_arm_expiry() -> Optional[datetime]:
    ts = _s().get("armed_until_utc")
    if ts is None:
        return None
    try:
        return datetime.utcfromtimestamp(float(ts))
    except Exception:
        return None


def set_last_run(ts_utc: datetime) -> None:
    _s()["last_run_utc"] = _to_jsonable(ts_utc)


def store_last_results(
    *,
    snapshot: Any = None,
    signals: Any = None,
    signals_diagnostics: Any = None,
    targets: Any = None,
    plan: Any = None,
    gates: Any = None,
    execution: Any = None,
    errors: Any = None,
    warnings: Any = None,
) -> None:
    last = _l()
    last["snapshot"] = _to_jsonable(snapshot)
    last["signals"] = _to_jsonable(signals)
    last["signals_diagnostics"] = _to_jsonable(signals_diagnostics)
    last["targets"] = _to_jsonable(targets)
    last["plan"] = _to_jsonable(plan)
    last["gates"] = _to_jsonable(gates)
    last["execution"] = _to_jsonable(execution)
    last["errors"] = _to_jsonable(errors or [])
    last["warnings"] = _to_jsonable(warnings or [])


def get_last_results() -> Dict[str, Any]:
    return _l()
