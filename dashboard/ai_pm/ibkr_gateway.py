from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ib_insync import IB, util

# Allow nested asyncio loops (required for Streamlit + ib_insync)
try:
    import nest_asyncio

    nest_asyncio.apply()
except ImportError:
    pass

# Allow nested asyncio loops (required for Streamlit + ib_insync)
try:
    import nest_asyncio

    nest_asyncio.apply()
except ImportError:
    pass

# Ensure ib_insync has an event loop (safe to call multiple times)
try:
    util.startLoop()
except Exception:
    pass


@dataclass(frozen=True)
class IbkrConnectionConfig:
    host: str = "127.0.0.1"
    port: int = 7496
    client_id: int = 998  # AI PM uses 998 to avoid collision with main app (999)
    readonly: bool = False


@dataclass(frozen=True)
class AccountSummary:
    account: str
    net_liquidation: Optional[float]
    total_cash_value: Optional[float]
    available_funds: Optional[float]
    buying_power: Optional[float]
    currency: Optional[str]


@dataclass(frozen=True)
class PositionRow:
    account: str
    symbol: str
    sec_type: str
    exchange: str
    currency: str
    con_id: int
    position: float
    avg_cost: Optional[float]


@dataclass(frozen=True)
class OpenOrderRow:
    account: Optional[str]
    order_id: Optional[int]
    perm_id: Optional[int]
    symbol: Optional[str]
    action: Optional[str]
    quantity: Optional[float]
    order_type: Optional[str]
    lmt_price: Optional[float]
    status: Optional[str]


class IbkrGateway:
    """
    Single source of truth for IBKR connectivity + account selection + portfolio reads.
    (v1) Uses ib_insync only, compatible with your existing working rebalancer patterns.
    """

    def __init__(self, cfg: Optional[IbkrConnectionConfig] = None):
        self.cfg = cfg or self._cfg_from_env()
        self.ib = IB()
        self._selected_account: Optional[str] = None

    @staticmethod
    def _cfg_from_env() -> IbkrConnectionConfig:
        host = os.getenv("IBKR_HOST", "127.0.0.1").strip()
        port = int(os.getenv("IBKR_PORT", "7496").strip())
        client_id = int(os.getenv("IBKR_CLIENT_ID", "998").strip())
        readonly = os.getenv("IBKR_READONLY", "false").strip().lower() in ("1", "true", "yes", "y")
        return IbkrConnectionConfig(host=host, port=port, client_id=client_id, readonly=readonly)

    # -----------------------------
    # Connection lifecycle
    # -----------------------------
    def connect(self, timeout_sec: float = 6.0) -> bool:
        if self.ib.isConnected():
            return True

        try:
            self.ib.connect(self.cfg.host, self.cfg.port, clientId=self.cfg.client_id, timeout=timeout_sec)
            return self.ib.isConnected()
        except Exception:
            return False

    def disconnect(self) -> None:
        try:
            if self.ib.isConnected():
                self.ib.disconnect()
        except Exception:
            pass

    def is_connected(self) -> bool:
        return bool(self.ib.isConnected())

    def ping(self) -> bool:
        """Cheap sanity check that does not require market data."""
        if not self.is_connected():
            return False
        try:
            _ = self.ib.client.getReqId()
            return True
        except Exception:
            return False

    # -----------------------------
    # Account selection
    # -----------------------------
    def list_accounts(self) -> List[str]:
        if not self.is_connected():
            return []
        try:
            accts = list(self.ib.managedAccounts() or [])
            return [a for a in accts if a]
        except Exception:
            return []

    def set_account(self, account: Optional[str]) -> None:
        """
        Set the active account used for filtering.
        - account=None means "All" (no filtering).
        """
        self._selected_account = account.strip() if isinstance(account, str) and account.strip() else None

    def get_account(self) -> Optional[str]:
        return self._selected_account

    # -----------------------------
    # Portfolio reads
    # -----------------------------
    def get_account_summary(self, account: Optional[str] = None, timeout_sec: float = 4.0) -> Optional[AccountSummary]:
        """
        Returns key account summary fields (single account only).
        Adds a REAL timeout so Streamlit doesn't freeze.
        """
        if not self.is_connected():
            return None

        acct = account or self._selected_account
        if not acct:
            accts = self.list_accounts()
            acct = accts[0] if accts else None
        if not acct:
            return None

        tags = ["NetLiquidation", "TotalCashValue", "AvailableFunds", "BuyingPower"]

        def _filter_rows(rows):
            # rows are typically AccountValue objects with fields: account, tag, currency, value
            out = []
            for r in rows or []:
                try:
                    if getattr(r, "account", None) == acct and getattr(r, "tag", None) in tags:
                        out.append(r)
                except Exception:
                    continue
            return out

        try:
            rows = None

            # Use accountValues() instead of accountSummary() - it has NetLiquidation
            try:
                rows = self.ib.accountValues(acct)
            except Exception:
                rows = None

            rows = _filter_rows(rows)

            d = {(r.tag, r.currency): r.value for r in rows if getattr(r, "tag", None) in tags}

            def _get(tag: str) -> Tuple[Optional[float], Optional[str]]:
                for (t, c), v in d.items():
                    if t == tag:
                        try:
                            return float(v), c
                        except Exception:
                            return None, c
                return None, None

            net_liq, cur = _get("NetLiquidation")
            cash, _ = _get("TotalCashValue")
            avail, _ = _get("AvailableFunds")
            bp, _ = _get("BuyingPower")

            return AccountSummary(
                account=acct,
                net_liquidation=net_liq,
                total_cash_value=cash,
                available_funds=avail,
                buying_power=bp,
                currency=cur,
            )

        except Exception:
            return None

    def get_positions(self) -> List[PositionRow]:
        """
        Returns positions for selected account; if selected_account is None, returns all.
        """
        if not self.is_connected():
            return []

        acct = self._selected_account
        out: List[PositionRow] = []
        try:
            pos = self.ib.positions()
            for p in pos:
                if acct and getattr(p, "account", None) != acct:
                    continue

                c = p.contract
                out.append(
                    PositionRow(
                        account=getattr(p, "account", "") or "",
                        symbol=getattr(c, "symbol", None) or "",
                        sec_type=getattr(c, "secType", None) or "",
                        exchange=getattr(c, "exchange", None) or "",
                        currency=getattr(c, "currency", None) or "",
                        con_id=int(getattr(c, "conId", 0) or 0),
                        position=float(getattr(p, "position", 0.0) or 0.0),
                        avg_cost=(
                            float(getattr(p, "avgCost", 0.0)) if getattr(p, "avgCost", None) is not None else None),
                    )
                )
            return out
        except Exception:
            return []

    def get_open_orders(self, include_all_clients: bool = True) -> List[OpenOrderRow]:
        """
        Returns open orders.
        If include_all_clients=True, creates a FRESH connection as master client (clientId=0)
        to see ALL orders - matching your working Flask app pattern.
        Uses threading with timeout to prevent Streamlit freeze.
        """
        if not self.is_connected():
            return []

        acct = self._selected_account

        if include_all_clients:
            # Use a separate thread with timeout to prevent Streamlit freeze
            # This matches the working pattern from your Flask app.py
            import threading

            result = {"orders": [], "error": None}

            def fetch_all_orders():
                from ib_insync import IB
                ib_master = None
                try:
                    ib_master = IB()
                    # Connect as master client (clientId=0) like your Flask app
                    ib_master.connect(self.cfg.host, self.cfg.port, clientId=0, timeout=5)
                    ib_master.reqAutoOpenOrders(True)
                    ib_master.reqAllOpenOrders()
                    ib_master.sleep(2)  # ib.sleep works in this thread context

                    trades = list(ib_master.openTrades() or [])

                    for tr in trades:
                        order = tr.order
                        contract = tr.contract
                        status = tr.orderStatus

                        o_acct = getattr(order, "account", None)
                        if acct and o_acct != acct:
                            continue

                        result["orders"].append(
                            OpenOrderRow(
                                account=o_acct,
                                order_id=getattr(order, "orderId", None),
                                perm_id=getattr(order, "permId", None),
                                symbol=getattr(contract, "symbol", None),
                                action=getattr(order, "action", None),
                                quantity=(float(getattr(order, "totalQuantity", 0.0)) if getattr(order, "totalQuantity",
                                                                                                 None) is not None else None),
                                order_type=getattr(order, "orderType", None),
                                lmt_price=(
                                    float(getattr(order, "lmtPrice", 0.0)) if getattr(order, "lmtPrice", None) not in (
                                        None, 0, 0.0) else None),
                                status=getattr(status, "status", None),
                            )
                        )
                except Exception as e:
                    result["error"] = str(e)
                finally:
                    if ib_master:
                        try:
                            ib_master.disconnect()
                        except:
                            pass

            thread = threading.Thread(target=fetch_all_orders, daemon=True)
            thread.start()
            thread.join(timeout=10)  # 10 second timeout

            if thread.is_alive():
                # Timeout - thread is still running (frozen)
                import logging
                logging.getLogger(__name__).warning("get_open_orders timed out - using cached data")
                # Fall back to cached data from main connection
                return self._get_cached_orders()

            if result["error"]:
                import logging
                logging.getLogger(__name__).warning(f"get_open_orders error: {result['error']} - using cached data")
                return self._get_cached_orders()

            return result["orders"]
        else:
            # Just use cached data from main connection
            return self._get_cached_orders()

    def _get_cached_orders(self) -> List[OpenOrderRow]:
        """Get orders from the cached openTrades() - no blocking calls."""
        try:
            trades = list(self.ib.openTrades() or [])
            out: List[OpenOrderRow] = []
            acct = self._selected_account

            for tr in trades:
                order = tr.order
                contract = tr.contract
                status = tr.orderStatus

                o_acct = getattr(order, "account", None)
                if acct and o_acct != acct:
                    continue

                out.append(
                    OpenOrderRow(
                        account=o_acct,
                        order_id=getattr(order, "orderId", None),
                        perm_id=getattr(order, "permId", None),
                        symbol=getattr(contract, "symbol", None),
                        action=getattr(order, "action", None),
                        quantity=(float(getattr(order, "totalQuantity", 0.0)) if getattr(order, "totalQuantity",
                                                                                         None) is not None else None),
                        order_type=getattr(order, "orderType", None),
                        lmt_price=(
                            float(getattr(order, "lmtPrice", 0.0)) if getattr(order, "lmtPrice", None) not in (None, 0,
                                                                                                               0.0) else None),
                        status=getattr(status, "status", None),
                    )
                )
            return out
        except Exception:
            return []

    # -----------------------------
    # Order placement (stub in v1 step 2)
    # Execution logic will be implemented in Step 8 (execution_engine.py).
    # -----------------------------
    def place_orders(self, order_tickets: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Placeholder: will be replaced by execution_engine integration later.
        For now: validates connection + selected account.
        """
        if not self.is_connected():
            return False, ["IBKR not connected"]

        if self.cfg.readonly:
            return False, ["IBKR gateway is in READONLY mode (IBKR_READONLY=true)"]

        acct = self._selected_account
        if not acct:
            return False, ["No account selected (set_account)"]

        return False, ["place_orders not implemented yet (Step 8)"]

    def refresh_orders_cache(self) -> int:
        """
        Force refresh the orders cache by reconnecting.
        Returns the number of open orders after refresh.
        """
        result = self.refresh_all()
        return result.get("orders", 0)

    def refresh_positions_cache(self) -> int:
        """
        Force refresh the positions cache by reconnecting.
        Returns the number of positions after refresh.
        """
        result = self.refresh_all()
        return result.get("positions", 0)

    def refresh_all(self) -> dict:
        """
        Force refresh all cached data from TWS by reconnecting.
        This is the safest way to get fresh data in Streamlit.
        Returns counts for each type.
        """
        if not self.is_connected():
            return {"positions": 0, "orders": 0, "error": "Not connected"}

        result = {}

        try:
            # Save connection params
            host = self.cfg.host
            port = self.cfg.port
            client_id = self.cfg.client_id

            # Disconnect
            self.ib.disconnect()
            time.sleep(0.5)

            # Reconnect
            self.ib.connect(host, port, clientId=client_id, timeout=10)
            time.sleep(1)

            # Now read fresh data
            result["positions"] = len(self.ib.positions() or [])
            result["orders"] = len(self.ib.openTrades() or [])
            result["success"] = True
            return result

        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
            return result