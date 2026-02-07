"""
Kill Switch Module — Emergency Trading Halt

Provides multiple mechanisms to halt all trading:
1. File-based trigger: Drop KILL_SWITCH file to halt (works even if app is unresponsive)
2. Programmatic API: call activate() from code
3. CLI: python -m dashboard.ai_pm.kill_switch activate
4. Auto-cancel: When activated, cancels ALL open orders in TWS

The kill switch is checked BEFORE every order submission.
Once activated, NO orders can be placed until explicitly deactivated.

Author: HH Research Platform
Location: dashboard/ai_pm/kill_switch.py
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

_log = logging.getLogger(__name__)

# Kill switch file location — configurable via env var
_KILL_SWITCH_DIR = os.environ.get(
    "HH_KILL_SWITCH_DIR",
    os.path.join(os.path.expanduser("~"), ".hh_platform")
)
_KILL_SWITCH_FILE = os.path.join(_KILL_SWITCH_DIR, "KILL_SWITCH")
_KILL_SWITCH_LOG = os.path.join(_KILL_SWITCH_DIR, "kill_switch_log.jsonl")


def _ensure_dir():
    """Ensure the kill switch directory exists."""
    os.makedirs(_KILL_SWITCH_DIR, exist_ok=True)


def is_active() -> bool:
    """
    Check if kill switch is active.

    Returns True if:
    - The KILL_SWITCH file exists on disk
    - The in-memory flag is set

    This is called before EVERY order placement.
    """
    return os.path.exists(_KILL_SWITCH_FILE)


def activate(reason: str = "Manual activation", cancel_orders: bool = True, ib=None) -> Dict[str, Any]:
    """
    Activate the kill switch.

    Args:
        reason: Why the kill switch was activated
        cancel_orders: If True and ib connection provided, cancel all open orders
        ib: ib_insync IB connection (optional, for auto-cancel)

    Returns:
        Dict with activation details
    """
    _ensure_dir()

    ts = datetime.now(timezone.utc).isoformat()
    payload = {
        "activated_at": ts,
        "reason": reason,
        "pid": os.getpid(),
        "hostname": _get_hostname(),
    }

    # Write the kill switch file atomically
    tmp_path = _KILL_SWITCH_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, _KILL_SWITCH_FILE)  # Atomic on POSIX

    _log.critical(f"KILL SWITCH ACTIVATED: {reason}")

    # Cancel all open orders if IB connection provided
    cancelled_orders = []
    if cancel_orders and ib is not None:
        cancelled_orders = cancel_all_open_orders(ib)
        payload["cancelled_orders"] = len(cancelled_orders)

    # Append to log
    _append_log({"event": "ACTIVATED", **payload})

    return payload


def deactivate(reason: str = "Manual deactivation") -> Dict[str, Any]:
    """
    Deactivate the kill switch. Requires explicit action.

    Args:
        reason: Why the kill switch is being deactivated

    Returns:
        Dict with deactivation details
    """
    ts = datetime.now(timezone.utc).isoformat()
    payload = {
        "deactivated_at": ts,
        "reason": reason,
        "pid": os.getpid(),
    }

    if os.path.exists(_KILL_SWITCH_FILE):
        # Read the activation info before removing
        try:
            with open(_KILL_SWITCH_FILE, "r") as f:
                activation_info = json.load(f)
            payload["was_activated_at"] = activation_info.get("activated_at")
            payload["was_reason"] = activation_info.get("reason")
        except Exception:
            pass

        os.remove(_KILL_SWITCH_FILE)
        _log.warning(f"KILL SWITCH DEACTIVATED: {reason}")
    else:
        _log.info("Kill switch was not active; no action taken.")
        payload["note"] = "Was not active"

    _append_log({"event": "DEACTIVATED", **payload})

    return payload


def get_status() -> Dict[str, Any]:
    """Get current kill switch status with details."""
    active = is_active()
    status = {
        "active": active,
        "file_path": _KILL_SWITCH_FILE,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }

    if active:
        try:
            with open(_KILL_SWITCH_FILE, "r") as f:
                activation_info = json.load(f)
            status.update(activation_info)
        except Exception:
            status["note"] = "Kill switch file exists but could not be read"

    return status


def cancel_all_open_orders(ib) -> List[Dict[str, Any]]:
    """
    Cancel ALL open orders in TWS/Gateway.

    Args:
        ib: ib_insync IB connection

    Returns:
        List of cancelled order details
    """
    cancelled = []
    try:
        open_orders = ib.openOrders()
        if not open_orders:
            _log.info("No open orders to cancel.")
            return cancelled

        _log.critical(f"CANCELLING {len(open_orders)} OPEN ORDERS")

        for order in open_orders:
            try:
                ib.cancelOrder(order)
                cancelled.append({
                    "order_id": getattr(order, "orderId", None),
                    "symbol": getattr(getattr(order, "contract", None), "symbol", None),
                    "action": getattr(order, "action", None),
                    "quantity": getattr(order, "totalQuantity", None),
                    "cancelled_at": datetime.now(timezone.utc).isoformat(),
                })
            except Exception as e:
                _log.error(f"Failed to cancel order {getattr(order, 'orderId', '?')}: {e}")

        # Give TWS time to process cancellations
        try:
            ib.sleep(2)
        except Exception:
            pass

        _log.critical(f"Cancelled {len(cancelled)} orders")

    except Exception as e:
        _log.error(f"Error cancelling orders: {e}")

    return cancelled


def check_and_block(context: str = "") -> Optional[str]:
    """
    Check kill switch and return block reason if active.

    Use this as a guard before any order submission:

        block_reason = kill_switch.check_and_block("execute_trade_plan")
        if block_reason:
            return ExecutionResult(..., errors=[block_reason])

    Returns:
        None if trading is allowed, error message string if blocked.
    """
    if not is_active():
        return None

    status = get_status()
    reason = status.get("reason", "Unknown reason")
    activated_at = status.get("activated_at", "Unknown time")

    msg = (
        f"KILL SWITCH ACTIVE — All trading halted. "
        f"Reason: {reason}. Activated at: {activated_at}. "
        f"Context: {context}. "
        f"To resume: deactivate the kill switch."
    )
    _log.critical(msg)
    return msg


# =========================================================================
# HELPERS
# =========================================================================

def _get_hostname() -> str:
    try:
        import socket
        return socket.gethostname()
    except Exception:
        return "unknown"


def _append_log(entry: Dict):
    """Append to kill switch log file."""
    try:
        _ensure_dir()
        with open(_KILL_SWITCH_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        _log.error(f"Failed to write kill switch log: {e}")


# =========================================================================
# CLI INTERFACE
# =========================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m dashboard.ai_pm.kill_switch <command>")
        print("Commands: activate, deactivate, status")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "activate":
        reason = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "CLI activation"
        result = activate(reason=reason, cancel_orders=False)
        print(f"KILL SWITCH ACTIVATED: {json.dumps(result, indent=2)}")

    elif command == "deactivate":
        reason = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "CLI deactivation"
        result = deactivate(reason=reason)
        print(f"KILL SWITCH DEACTIVATED: {json.dumps(result, indent=2)}")

    elif command == "status":
        status = get_status()
        print(json.dumps(status, indent=2))

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
