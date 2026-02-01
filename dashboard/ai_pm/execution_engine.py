from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import math
import time

from ib_insync import IB, Stock, Contract, MarketOrder, LimitOrder, Trade

from .config import DEFAULT_CONSTRAINTS, RiskConstraints
from .models import ExecutionResult, OrderTicket, PortfolioSnapshot, TradePlan


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _round_price(px: float) -> float:
    # v1: simple rounding; IBKR will often accept raw decimals for US stocks/ETFs.
    try:
        return float(round(float(px), 4))
    except Exception:
        return px


def _round_qty(q: float) -> int:
    """
    Round quantity to whole number.
    IBKR API does not support fractional shares via API (Error 10243).
    Use math.floor to be conservative (don't overbuy).
    """
    try:
        return int(math.floor(float(q)))
    except Exception:
        return int(q) if q else 0


def _is_etf_like(snapshot: PortfolioSnapshot, symbol: str) -> bool:
    # Best-effort: if current holdings show ETF secType or you later enrich, this can improve.
    for p in snapshot.positions:
        if (p.symbol or "").strip().upper() == symbol.strip().upper():
            if (p.sec_type or "").upper() == "STK":
                return False
            if (p.sec_type or "").upper() == "ETF":
                return True
    return False


def _make_contract(symbol: str, snapshot: PortfolioSnapshot) -> Contract:
    """
    ib_insync does not always provide an ETF() helper depending on version.
    For IBKR, ETFs trade as STK anyway, so Stock(...) is correct for both stocks and ETFs.
    """
    sym = symbol.strip().upper()
    return Stock(sym, "SMART", "USD")


def _fetch_quote(ib: IB, contract: Contract, timeout_sec: float = 2.0) -> Dict[str, Optional[float]]:
    """
    Best-effort quote using ib_insync reqMktData.
    Returns bid/ask/last/mid if available.
    """
    q = {"bid": None, "ask": None, "last": None, "mid": None}
    try:
        ticker = ib.reqMktData(contract, "", False, False)
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            bid = _safe_float(getattr(ticker, "bid", None))
            ask = _safe_float(getattr(ticker, "ask", None))
            last = _safe_float(getattr(ticker, "last", None))

            if bid is not None:
                q["bid"] = bid
            if ask is not None:
                q["ask"] = ask
            if last is not None:
                q["last"] = last

            if q["bid"] is not None and q["ask"] is not None:
                q["mid"] = (q["bid"] + q["ask"]) / 2.0
                break

            ib.sleep(0.05)

        # final mid fallback
        if q["mid"] is None and q["bid"] is not None and q["ask"] is not None:
            q["mid"] = (q["bid"] + q["ask"]) / 2.0
        return q
    except Exception:
        return q
    finally:
        try:
            ib.cancelMktData(contract)
        except Exception:
            pass


def _choose_order(
        *,
        action: str,
        quantity: float,
        quote: Dict[str, Optional[float]],
        use_limit_preferred: bool = True,
        limit_buffer_bps: float = 10.0,
) -> Tuple[Any, Optional[float], str]:
    """
    v1 order selection:
    - prefer LIMIT at mid +/- buffer if bid/ask available
    - else MARKET
    Returns: (ib_order, chosen_limit_price, order_type_str)
    """
    qty = _round_qty(quantity)

    bid = quote.get("bid")
    ask = quote.get("ask")
    mid = quote.get("mid")
    last = quote.get("last")

    if use_limit_preferred and mid is not None and bid is not None and ask is not None and bid > 0 and ask > 0:
        px = float(mid)
        buf = float(limit_buffer_bps) / 10000.0  # bps -> fraction
        if action.upper() == "BUY":
            px = px * (1.0 + buf)
        else:
            px = px * (1.0 - buf)
        px = _round_price(px)
        return LimitOrder(action.upper(), qty, px, tif="DAY"), px, "LMT"

    # Fallback: if only last exists, still try a limit at last +/- buffer
    if use_limit_preferred and last is not None and last > 0:
        px = float(last)
        buf = float(limit_buffer_bps) / 10000.0
        if action.upper() == "BUY":
            px = px * (1.0 + buf)
        else:
            px = px * (1.0 - buf)
        px = _round_price(px)
        return LimitOrder(action.upper(), qty, px, tif="DAY"), px, "LMT"

    return MarketOrder(action.upper(), qty, tif="DAY"), None, "MKT"


def _slice_quantity(
        *,
        total_qty: float,
        max_slice_notional: float,
        price_hint: Optional[float],
) -> List[int]:
    """
    Simple slicing by max slice notional.
    If no price hint, returns single slice.
    Returns list of integer quantities.
    """
    total_qty_int = _round_qty(total_qty)
    if total_qty_int <= 0:
        return []
    if price_hint is None or price_hint <= 0:
        return [total_qty_int]

    max_slice_qty = int(math.floor(max_slice_notional / float(price_hint)))
    if max_slice_qty <= 0:
        max_slice_qty = 1  # At least 1 share per slice

    if total_qty_int <= max_slice_qty:
        return [total_qty_int]

    n = int(math.ceil(total_qty_int / max_slice_qty))
    slices = []
    remaining = total_qty_int
    for _ in range(n):
        q = min(remaining, max_slice_qty)
        if q > 0:
            slices.append(q)
        remaining -= q
        if remaining <= 0:
            break
    return slices


def execute_trade_plan(
        *,
        ib: IB,
        snapshot: PortfolioSnapshot,
        plan: TradePlan,
        account: str,
        constraints: RiskConstraints = DEFAULT_CONSTRAINTS,
        dry_run: bool = True,
        kill_switch: bool = False,
        auto_trade_enabled: bool = False,
        armed: bool = False,
        limit_buffer_bps: float = 10.0,
        max_slice_nav_pct: float = 1.00,  # disabled - send full orders (no slicing)
        price_map: Optional[Dict[str, float]] = None,  # Pre-fetched prices (Yahoo) to avoid slow IBKR quotes
        skip_live_quotes: bool = True,  # Skip slow IBKR reqMktData calls
) -> ExecutionResult:
    """
    Execute orders for a plan.

    Policy implemented:
      - Missing live quote => skip that symbol (do not place blind orders)
      - Cash below cash_min:
          - manual mode: return without executing (caller should show recommendation)
          - auto+armed: execute SELL-FIRST (SELL orders only), defer buys
      - Turnover cap is enforced upstream by gates; this function does not override that.
    """
    import logging
    import sys
    import nest_asyncio
    nest_asyncio.apply()
    _log = logging.getLogger(__name__)
    _log.info(f"execute_trade_plan: START dry_run={dry_run}, orders={len(plan.orders) if plan and plan.orders else 0}")
    print(f"DEBUG execute_trade_plan: START dry_run={dry_run}", flush=True)
    sys.stdout.flush()
    
    ts = datetime.utcnow()
    _log.info("execute_trade_plan: initialized timestamp")
    
    errors: List[str] = []
    notes: List[str] = []
    submitted_orders: List[Dict[str, Any]] = []
    _log.info("execute_trade_plan: initialized lists")

    _log.info("execute_trade_plan: checking ib.isConnected()...")
    if not ib.isConnected():
        _log.error("execute_trade_plan: IB not connected!")
        return ExecutionResult(ts_utc=ts, account=account, strategy_key=plan.strategy_key,
                               submitted=False, submitted_orders=[],
                               errors=["IBKR not connected"], notes=[])
    _log.info("execute_trade_plan: IB is connected")

    if kill_switch:
        return ExecutionResult(ts_utc=ts, account=account, strategy_key=plan.strategy_key,
                               submitted=False, submitted_orders=[],
                               errors=["Kill switch is enabled; execution blocked."], notes=[])

    if auto_trade_enabled and not armed:
        return ExecutionResult(ts_utc=ts, account=account, strategy_key=plan.strategy_key,
                               submitted=False, submitted_orders=[],
                               errors=["Auto-trade enabled but system is not ARMED; execution blocked."], notes=[])

    if not account:
        return ExecutionResult(ts_utc=ts, account=account, strategy_key=plan.strategy_key,
                               submitted=False, submitted_orders=[],
                               errors=["No account provided for execution."], notes=[])

    _log.info("execute_trade_plan: passed all checks, getting NAV...")
    nav = _safe_float(plan.nav) or _safe_float(snapshot.net_liquidation) or 0.0
    _log.info(f"execute_trade_plan: NAV={nav}")
    
    _log.info("execute_trade_plan: getting cash...")
    cash = _safe_float(snapshot.total_cash)
    _log.info(f"execute_trade_plan: cash={cash}")
    if cash is None:
        cash = 0.0
        notes.append("Cash missing from snapshot; assuming 0 for execution policy decisions (may be inaccurate).")

    _log.info("execute_trade_plan: calculating max_slice_notional...")
    max_slice_notional = float(nav) * float(max_slice_nav_pct) if nav > 0 else float("inf")
    _log.info(f"execute_trade_plan: max_slice_notional={max_slice_notional}")

    _log.info("execute_trade_plan: getting orders_all...")
    orders_all = plan.orders or []
    _log.info(f"execute_trade_plan: orders_all count={len(orders_all)}")
    
    if not orders_all:
        return ExecutionResult(ts_utc=ts, account=account, strategy_key=plan.strategy_key,
                               submitted=False, submitted_orders=[],
                               errors=[], notes=["No orders to execute."])

    _log.info("execute_trade_plan: building px_map...")
    # Use passed price_map if available, otherwise build from snapshot
    if price_map:
        _log.info(f"execute_trade_plan: using passed price_map with {len(price_map)} prices")
        px_map: Dict[str, float] = dict(price_map)
    else:
        _log.info(f"execute_trade_plan: building px_map from snapshot ({len(snapshot.positions)} positions)...")
        px_map: Dict[str, float] = {}
        for p in snapshot.positions:
            sym = (p.symbol or "").strip().upper()
            if not sym:
                continue
            px = _safe_float(p.market_price)
            if px is None or px <= 0:
                px = _safe_float(p.avg_cost)
            if px is not None and px > 0:
                px_map[sym] = float(px)
        _log.info(f"execute_trade_plan: px_map built from snapshot with {len(px_map)} prices")
    _log.info("execute_trade_plan: projecting cash...")
    
    # Project cash if executing full plan using snapshot prices (best-effort).
    # If price missing for an order symbol, ignore its impact in projection (conservative).
    proj_cash = float(cash)
    _log.info(f"execute_trade_plan: proj_cash init={proj_cash}, nav={nav}")
    if nav > 0:
        _log.info(f"execute_trade_plan: looping {len(orders_all)} orders for cash projection...")
        sys.stdout.flush()
        for i, o in enumerate(orders_all):
            if i % 10 == 0:
                _log.info(f"execute_trade_plan: cash projection order {i}/{len(orders_all)}")
                sys.stdout.flush()
            sym = (o.symbol or "").strip().upper()
            if not sym:
                continue
            qty = _safe_float(o.quantity) or 0.0
            if qty <= 0:
                continue
            px = px_map.get(sym)
            if px is None or px <= 0:
                continue
            notional = float(qty) * float(px)
            if (o.action or "").upper() == "BUY":
                proj_cash -= notional
            else:
                proj_cash += notional

        proj_cash_w = proj_cash / float(nav)
        _log.info(f"execute_trade_plan: proj_cash_w={proj_cash_w:.4f}")
        sys.stdout.flush()
    else:
        proj_cash_w = None
        _log.info("execute_trade_plan: proj_cash_w=None (nav<=0)")
        sys.stdout.flush()

    _log.info(f"execute_trade_plan: checking cash policy... cash_min={constraints.cash_min}")
    sys.stdout.flush()
    # Cash policy
    if proj_cash_w is not None and proj_cash_w < float(constraints.cash_min):
        if auto_trade_enabled and armed and not kill_switch:
            # SELL-FIRST mode
            sells = [o for o in orders_all if
                     (o.action or "").upper() == "SELL" and (_safe_float(o.quantity) or 0.0) > 0]
            if not sells:
                return ExecutionResult(ts_utc=ts, account=account, strategy_key=plan.strategy_key,
                                       submitted=False, submitted_orders=[],
                                       errors=[
                                           f"Cash would fall below cash_min ({constraints.cash_min:.2%}) but no SELL orders exist to restore cash."],
                                       notes=[])
            notes.append(
                f"Sell-first mode: projected cash {proj_cash_w * 100:.2f}% < cash_min {constraints.cash_min * 100:.2f}%. "
                f"Executing SELL orders only; buys deferred."
            )
            orders = sells
        else:
            return ExecutionResult(ts_utc=ts, account=account, strategy_key=plan.strategy_key,
                                   submitted=False, submitted_orders=[],
                                   errors=[
                                       f"Execution blocked: projected cash {proj_cash_w * 100:.2f}% < cash_min {constraints.cash_min * 100:.2f}%. Run SELL-FIRST first."],
                                   notes=[])
    else:
        orders = orders_all

    _log.info(f"execute_trade_plan: cash policy passed, orders count={len(orders)}")
    sys.stdout.flush()
    
    # Build contracts
    _log.info("execute_trade_plan: building contracts...")
    sys.stdout.flush()
    contract_by_symbol: Dict[str, Contract] = {}
    for o in orders:
        sym = (o.symbol or "").strip().upper()
        if not sym:
            continue
        if sym in contract_by_symbol:
            continue
        contract_by_symbol[sym] = _make_contract(sym, snapshot)

    # Skip qualifyContracts - it deadlocks in threaded environments (Streamlit)
    # IBKR accepts basic Stock() contracts for order placement without qualification
    _log.info(f"execute_trade_plan: built {len(contract_by_symbol)} contracts")
    sys.stdout.flush()
    
    try:
        _log.info("execute_trade_plan: skipping qualifyContracts (causes thread deadlock)")
        sys.stdout.flush()
        qualified = list(contract_by_symbol.values())
        _log.info(f"execute_trade_plan: {len(qualified)} contracts ready for order placement")
        sys.stdout.flush()
        for c in qualified:
            sym = (getattr(c, "symbol", "") or "").strip().upper()
            if sym:
                contract_by_symbol[sym] = c
    except Exception as e:
        return ExecutionResult(ts_utc=ts, account=account, strategy_key=plan.strategy_key,
                               submitted=False, submitted_orders=[],
                               errors=[f"Contract qualification failed: {e!r}"], notes=[])

    placed_any = False

    for o in orders:
        sym = (o.symbol or "").strip().upper()
        if not sym:
            continue

        contract = contract_by_symbol.get(sym)
        if contract is None:
            errors.append(f"{sym}: no qualified contract")
            continue

        raw_qty = _safe_float(o.quantity) or 0.0
        qty = _round_qty(raw_qty)
        if qty <= 0:
            if raw_qty > 0:
                notes.append(f"{sym}: quantity {raw_qty:.4f} rounds to 0 shares; skipped (too small to trade).")
            continue

        # Use pre-fetched price if available (much faster than IBKR reqMktData)
        if skip_live_quotes and price_map and sym in price_map and price_map[sym]:
            pre_price = float(price_map[sym])
            quote = {"bid": pre_price * 0.999, "ask": pre_price * 1.001, "last": pre_price, "mid": pre_price}
        else:
            quote = _fetch_quote(ib, contract, timeout_sec=2.0)

        # If quote is totally missing, SKIP (your policy: missing prices -> skip)
        if quote.get("bid") is None and quote.get("ask") is None and quote.get("last") is None and quote.get(
                "mid") is None:
            notes.append(f"{sym}: missing live quote (bid/ask/last unavailable); skipped execution.")
            continue

        prefer_limit = (o.order_type or "").upper() in ("", "LMT", "LIMIT", "MKT", "MARKET")
        force_market = (o.order_type or "").upper() in ("MKT", "MARKET")
        if force_market:
            prefer_limit = False

        price_hint = quote.get("mid") or quote.get("last") or quote.get("ask") or quote.get("bid")
        slices = _slice_quantity(total_qty=qty, max_slice_notional=max_slice_notional, price_hint=price_hint)

        if len(slices) > 1:
            notes.append(f"{sym}: order sliced into {len(slices)} parts (max_slice_nav_pct={max_slice_nav_pct:.2%}).")

        for i, slice_qty in enumerate(slices, start=1):
            ib_order, lmt_px, typ = _choose_order(
                action=o.action,
                quantity=slice_qty,
                quote=quote,
                use_limit_preferred=prefer_limit,
                limit_buffer_bps=limit_buffer_bps,
            )
            try:
                ib_order.account = account
            except Exception:
                pass

            ticket_repr = {
                "symbol": sym,
                "action": o.action,
                "quantity": float(slice_qty),
                "order_type": typ,
                "limit_price": float(lmt_px) if lmt_px is not None else None,
                "tif": getattr(ib_order, "tif", None),
                "reason": o.reason,
                "slice": f"{i}/{len(slices)}" if len(slices) > 1 else None,
            }

            if dry_run:
                submitted_orders.append({**ticket_repr, "submitted": False, "dry_run": True})
                continue

            try:
                _log.info(f"execute_trade_plan: placing order {sym} {o.action} {qty}")
                tr: Trade = ib.placeOrder(contract, ib_order)
                _log.info(f"execute_trade_plan: order placed for {sym}")
                placed_any = True

                t0 = time.time()
                while time.time() - t0 < 1.5:
                    st = getattr(tr, "orderStatus", None)
                    s = getattr(st, "status", None) if st is not None else None
                    if s:
                        break
                    ib.sleep(0.05)

                st = getattr(tr, "orderStatus", None)
                status = getattr(st, "status", None) if st is not None else None
                order_id = getattr(getattr(tr, "order", None), "orderId", None)
                perm_id = getattr(getattr(tr, "order", None), "permId", None)

                submitted_orders.append(
                    {
                        **ticket_repr,
                        "submitted": True,
                        "dry_run": False,
                        "order_id": order_id,
                        "perm_id": perm_id,
                        "status": status,
                    }
                )
            except Exception as e:
                errors.append(f"{sym}: order placement failed: {e!r}")
                submitted_orders.append({**ticket_repr, "submitted": False, "dry_run": False, "error": repr(e)})

    submitted = bool(placed_any) if not dry_run else False
    if dry_run:
        notes.append("Dry-run mode: no orders were sent.")
    
    # CRITICAL: Allow ib_insync event loop to transmit orders to IBKR
    if placed_any and not dry_run:
        _log.info("execute_trade_plan: waiting for orders to transmit to IBKR...")
        try:
            ib.sleep(3)  # Give IBKR time to process all orders
            _log.info("execute_trade_plan: orders transmitted")
        except Exception as e:
            _log.warning(f"execute_trade_plan: ib.sleep failed: {e}")

    return ExecutionResult(
        ts_utc=ts,
        account=account,
        strategy_key=plan.strategy_key,
        submitted=submitted,
        submitted_orders=submitted_orders,
        errors=errors,
        notes=notes,
    )