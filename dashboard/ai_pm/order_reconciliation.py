"""
Order Reconciliation Module — Session Tracking & Orphan Detection

Prevents the 83-order desync scenario by:
1. Tagging every system order with a unique session prefix
2. Reconciling open orders at startup against the current session
3. Identifying and flagging orphan orders (from previous sessions or manual)
4. Providing a reconciliation report before any new orders are placed

Author: HH Research Platform
Location: dashboard/ai_pm/order_reconciliation.py
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

# Session tracking directory
_SESSION_DIR = os.environ.get(
    "HH_SESSION_DIR",
    os.path.join(os.path.expanduser("~"), ".hh_platform", "sessions")
)

# Prefix for all system-generated order references
_ORDER_REF_PREFIX = "HH_"


def _ensure_dir():
    os.makedirs(_SESSION_DIR, exist_ok=True)


class OrderSession:
    """
    Tracks a trading session and all orders placed within it.

    Each session gets a unique ID. Every order placed through the system
    gets tagged with this session ID in the order reference field.
    This allows us to distinguish system vs. manual orders and
    current-session vs. stale orders.
    """

    def __init__(self, account: str):
        self.account = account
        self.session_id = f"{_ORDER_REF_PREFIX}{uuid.uuid4().hex[:8]}"
        self.started_at = datetime.now(timezone.utc)
        self.orders_placed: List[Dict[str, Any]] = []
        self._save_session()

    def _save_session(self):
        """Persist session info to disk."""
        _ensure_dir()
        path = os.path.join(_SESSION_DIR, f"session_{self.session_id}.json")
        data = {
            "session_id": self.session_id,
            "account": self.account,
            "started_at": self.started_at.isoformat(),
            "orders_placed": len(self.orders_placed),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def get_order_ref(self, symbol: str, action: str) -> str:
        """
        Generate a unique order reference for tagging.

        Format: HH_<session_8chars>_<symbol>_<action>_<seq>
        This goes into the order's ocaRef field in IBKR.
        """
        seq = len(self.orders_placed) + 1
        return f"{self.session_id}_{symbol}_{action}_{seq}"

    def record_order(self, order_info: Dict[str, Any]):
        """Record an order placed in this session."""
        order_info["session_id"] = self.session_id
        order_info["recorded_at"] = datetime.now(timezone.utc).isoformat()
        self.orders_placed.append(order_info)
        self._save_session()

    @property
    def order_count(self) -> int:
        return len(self.orders_placed)


def reconcile_open_orders(
    ib,
    session: Optional[OrderSession] = None,
) -> Dict[str, Any]:
    """
    Reconcile open orders in TWS against the current session.

    This should be called BEFORE placing any new orders to detect:
    - Orphan orders from previous sessions
    - Manual orders placed outside the system
    - Stale orders that should have been cancelled

    Args:
        ib: ib_insync IB connection
        session: Current OrderSession (if None, all orders are considered orphans)

    Returns:
        Reconciliation report dict
    """
    report = {
        "reconciled_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session.session_id if session else None,
        "total_open_orders": 0,
        "system_orders_current_session": 0,
        "system_orders_previous_sessions": 0,
        "manual_orders": 0,
        "orphan_orders": [],
        "manual_order_details": [],
        "warnings": [],
        "is_clean": True,
    }

    try:
        open_trades = ib.openTrades()
        report["total_open_orders"] = len(open_trades)

        if not open_trades:
            _log.info("Reconciliation: No open orders in TWS.")
            return report

        for trade in open_trades:
            order = getattr(trade, "order", None)
            contract = getattr(trade, "contract", None)
            if not order:
                continue

            order_ref = getattr(order, "ocaGroup", "") or getattr(order, "orderRef", "") or ""
            order_id = getattr(order, "orderId", None)
            perm_id = getattr(order, "permId", None)
            symbol = getattr(contract, "symbol", "?") if contract else "?"
            action = getattr(order, "action", "?")
            qty = getattr(order, "totalQuantity", 0)

            order_detail = {
                "order_id": order_id,
                "perm_id": perm_id,
                "symbol": symbol,
                "action": action,
                "quantity": qty,
                "order_ref": order_ref,
            }

            if order_ref.startswith(_ORDER_REF_PREFIX):
                # System order — check if current session
                if session and order_ref.startswith(session.session_id):
                    report["system_orders_current_session"] += 1
                else:
                    report["system_orders_previous_sessions"] += 1
                    report["orphan_orders"].append(order_detail)
            else:
                # Manual order (placed via TWS UI or other tool)
                report["manual_orders"] += 1
                report["manual_order_details"].append(order_detail)

        # Determine if reconciliation is clean
        total_orphans = report["system_orders_previous_sessions"] + report["manual_orders"]
        report["is_clean"] = total_orphans == 0

        if not report["is_clean"]:
            report["warnings"].append(
                f"Found {total_orphans} non-current-session orders in TWS: "
                f"{report['system_orders_previous_sessions']} from previous sessions, "
                f"{report['manual_orders']} manual orders."
            )
            _log.warning(
                f"ORDER RECONCILIATION WARNING: {total_orphans} orphan/manual orders detected. "
                f"Review before placing new orders."
            )

    except Exception as e:
        report["warnings"].append(f"Reconciliation error: {e}")
        report["is_clean"] = False
        _log.error(f"Reconciliation failed: {e}")

    return report


def tag_ib_order(ib_order, order_ref: str):
    """
    Tag an IB order with the session reference.

    Args:
        ib_order: ib_insync order object (LimitOrder, MarketOrder, etc.)
        order_ref: Reference string from OrderSession.get_order_ref()
    """
    try:
        ib_order.orderRef = order_ref
    except Exception:
        pass  # Some order types may not support orderRef


def get_previous_sessions() -> List[Dict[str, Any]]:
    """List all previous sessions from disk."""
    _ensure_dir()
    sessions = []
    for fname in os.listdir(_SESSION_DIR):
        if fname.startswith("session_") and fname.endswith(".json"):
            try:
                with open(os.path.join(_SESSION_DIR, fname)) as f:
                    sessions.append(json.load(f))
            except Exception:
                pass
    sessions.sort(key=lambda s: s.get("started_at", ""), reverse=True)
    return sessions


def cleanup_old_sessions(keep_last_n: int = 10):
    """Remove old session files, keeping the most recent N."""
    sessions = get_previous_sessions()
    if len(sessions) <= keep_last_n:
        return

    for session in sessions[keep_last_n:]:
        sid = session.get("session_id", "")
        path = os.path.join(_SESSION_DIR, f"session_{sid}.json")
        try:
            os.remove(path)
        except Exception:
            pass
