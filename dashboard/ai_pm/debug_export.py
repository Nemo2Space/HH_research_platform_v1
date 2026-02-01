# dashboard/ai_pm/debug_export.py
"""
Comprehensive Debug Export for AI Portfolio Manager
====================================================

Exports ALL relevant data for debugging:
- Current positions from IBKR
- Open orders from TWS
- Target weights and shares
- Trade plan (proposed orders)
- Price map
- Snapshot data (NAV, cash, etc.)
- Verification results
- Gate results
- Session state

Outputs:
- JSON file with all structured data
- CSV files for tabular data (positions, orders, targets)
- Summary text file
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, date
from typing import Any, Dict, List, Optional
import logging

_log = logging.getLogger(__name__)


def _to_serializable(obj: Any) -> Any:
    """Convert any object to JSON-serializable format."""
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if is_dataclass(obj):
        return _to_serializable(asdict(obj))

    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(v) for v in obj]

    # Handle pandas Timestamp
    try:
        import pandas as pd
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
    except ImportError:
        pass

    # Handle numpy types
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass

    # Generic object with __dict__
    if hasattr(obj, '__dict__'):
        try:
            return _to_serializable(vars(obj))
        except Exception:
            pass

    # Fallback to string
    return str(obj)


def _safe_float(x) -> Optional[float]:
    """Safely convert to float."""
    if x is None:
        return None
    try:
        v = float(x)
        import math
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except:
        return None


def export_debug_snapshot(
        *,
        ib=None,
        gw=None,  # IbkrGateway instance
        snapshot=None,
        targets=None,
        plan=None,
        signals=None,
        gates=None,
        execution=None,
        price_map: Optional[Dict[str, float]] = None,
        verification=None,
        account: str = "",
        strategy_key: str = "",
        extra_context: Optional[Dict[str, Any]] = None,
        output_dir: str = "debug_exports",
) -> Dict[str, str]:
    """
    Export comprehensive debug snapshot to files.

    Args:
        ib: IB connection (ib_insync IB instance)
        gw: IbkrGateway instance
        snapshot: PortfolioSnapshot
        targets: TargetWeights
        plan: TradePlan
        signals: SignalSnapshot
        gates: GateReport
        execution: ExecutionResult
        price_map: Dict of symbol -> price
        verification: PortfolioVerification
        account: Account ID
        strategy_key: Strategy name
        extra_context: Any additional context to include
        output_dir: Directory to save files

    Returns:
        Dict with paths to created files
    """
    ts = datetime.now()
    ts_str = ts.strftime("%Y-%m-%dT%H-%M-%S")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Base filename
    base_name = f"debug_{account}_{ts_str}" if account else f"debug_{ts_str}"

    created_files = {}

    # ==========================================================================
    # 1. FETCH LIVE DATA FROM IBKR (if connection available)
    # ==========================================================================
    live_positions = []
    live_orders = []
    live_account_values = {}

    # Get IB connection
    ib_conn = ib or (gw.ib if gw else None)

    if ib_conn and ib_conn.isConnected():
        _log.info("debug_export: Fetching live data from IBKR...")

        # Fetch positions
        try:
            positions_raw = ib_conn.positions()
            for p in positions_raw:
                live_positions.append({
                    "account": getattr(p, 'account', ''),
                    "symbol": getattr(p.contract, 'symbol', '') if p.contract else '',
                    "sec_type": getattr(p.contract, 'secType', '') if p.contract else '',
                    "exchange": getattr(p.contract, 'exchange', '') if p.contract else '',
                    "currency": getattr(p.contract, 'currency', '') if p.contract else '',
                    "con_id": getattr(p.contract, 'conId', 0) if p.contract else 0,
                    "quantity": float(getattr(p, 'position', 0) or 0),
                    "avg_cost": _safe_float(getattr(p, 'avgCost', None)),
                    "market_value": _safe_float(getattr(p, 'position', 0)) * _safe_float(
                        getattr(p, 'avgCost', 0)) if getattr(p, 'avgCost', None) else None,
                })
        except Exception as e:
            _log.warning(f"debug_export: Failed to fetch positions: {e}")

        # Fetch open orders/trades
        try:
            trades_raw = ib_conn.openTrades()
            for t in trades_raw:
                order = t.order
                contract = t.contract
                status = t.orderStatus

                live_orders.append({
                    "order_id": getattr(order, 'orderId', None),
                    "perm_id": getattr(order, 'permId', None),
                    "client_id": getattr(order, 'clientId', None),
                    "account": getattr(order, 'account', ''),
                    "symbol": getattr(contract, 'symbol', '') if contract else '',
                    "sec_type": getattr(contract, 'secType', '') if contract else '',
                    "exchange": getattr(contract, 'exchange', '') if contract else '',
                    "action": getattr(order, 'action', ''),
                    "quantity": float(getattr(order, 'totalQuantity', 0) or 0),
                    "order_type": getattr(order, 'orderType', ''),
                    "limit_price": _safe_float(getattr(order, 'lmtPrice', None)),
                    "aux_price": _safe_float(getattr(order, 'auxPrice', None)),
                    "tif": getattr(order, 'tif', ''),
                    "status": getattr(status, 'status', '') if status else '',
                    "filled": float(getattr(status, 'filled', 0) or 0) if status else 0,
                    "remaining": float(getattr(status, 'remaining', 0) or 0) if status else 0,
                    "avg_fill_price": _safe_float(getattr(status, 'avgFillPrice', None)) if status else None,
                })
        except Exception as e:
            _log.warning(f"debug_export: Failed to fetch open orders: {e}")

        # Fetch account values
        try:
            if account:
                acct_values = ib_conn.accountValues(account)
            else:
                acct_values = ib_conn.accountValues()

            for av in acct_values:
                key = f"{getattr(av, 'tag', '')}_{getattr(av, 'currency', '')}"
                live_account_values[key] = {
                    "tag": getattr(av, 'tag', ''),
                    "value": getattr(av, 'value', ''),
                    "currency": getattr(av, 'currency', ''),
                    "account": getattr(av, 'account', ''),
                }
        except Exception as e:
            _log.warning(f"debug_export: Failed to fetch account values: {e}")

    # ==========================================================================
    # 2. BUILD COMPREHENSIVE DEBUG OBJECT
    # ==========================================================================

    debug_data = {
        "export_timestamp": ts.isoformat(),
        "export_timestamp_utc": datetime.utcnow().isoformat(),
        "account": account,
        "strategy_key": strategy_key,

        # Live IBKR data
        "ibkr_connected": bool(ib_conn and ib_conn.isConnected()),
        "live_positions": live_positions,
        "live_positions_count": len(live_positions),
        "live_open_orders": live_orders,
        "live_open_orders_count": len(live_orders),
        "live_account_values": live_account_values,

        # Snapshot data
        "snapshot": _to_serializable(snapshot) if snapshot else None,

        # Target weights
        "targets": _to_serializable(targets) if targets else None,

        # Trade plan
        "plan": _to_serializable(plan) if plan else None,

        # Price map
        "price_map": price_map or {},
        "price_map_count": len(price_map) if price_map else 0,

        # Signals (summary only to keep file size manageable)
        "signals_summary": {
            "count": len(signals.rows) if signals and hasattr(signals, 'rows') else 0,
            "symbols": list(signals.rows.keys())[:100] if signals and hasattr(signals, 'rows') else [],
        } if signals else None,

        # Gates
        "gates": _to_serializable(gates) if gates else None,

        # Execution result
        "execution": _to_serializable(execution) if execution else None,

        # Verification
        "verification": _to_serializable(verification) if verification else None,

        # Extra context
        "extra_context": _to_serializable(extra_context) if extra_context else None,
    }

    # ==========================================================================
    # 3. CALCULATE DERIVED DATA FOR ANALYSIS
    # ==========================================================================

    # Build comparison table: Position vs Target vs Orders
    comparison_rows = []

    all_symbols = set()

    # Collect symbols from all sources
    if live_positions:
        all_symbols.update(p['symbol'] for p in live_positions if p.get('symbol'))
    if live_orders:
        all_symbols.update(o['symbol'] for o in live_orders if o.get('symbol'))
    if targets and hasattr(targets, 'weights'):
        all_symbols.update(targets.weights.keys())
    if plan and hasattr(plan, 'orders'):
        all_symbols.update(o.symbol for o in plan.orders if o.symbol)

    # NAV for weight calculations
    nav = 0
    if snapshot:
        nav = _safe_float(getattr(snapshot, 'net_liquidation', None)) or 0
    if nav == 0 and live_account_values:
        for k, v in live_account_values.items():
            if 'NetLiquidation' in k:
                nav = _safe_float(v.get('value')) or 0
                break

    for sym in sorted(all_symbols):
        sym = sym.upper().strip()
        if not sym:
            continue

        # Current position
        pos_qty = 0
        pos_avg_cost = None
        for p in live_positions:
            if p.get('symbol', '').upper() == sym:
                pos_qty = p.get('quantity', 0)
                pos_avg_cost = p.get('avg_cost')
                break

        # Price
        price = (price_map or {}).get(sym, 0)
        if price == 0 and pos_avg_cost:
            price = pos_avg_cost

        # Current value and weight
        current_value = pos_qty * price if price else 0
        current_weight = (current_value / nav * 100) if nav > 0 else 0

        # Target weight and shares
        target_weight = 0
        target_shares = 0
        if targets and hasattr(targets, 'weights'):
            target_weight = targets.weights.get(sym, 0) * 100
            if nav > 0 and price > 0:
                target_shares = int((target_weight / 100) * nav / price)

        # Planned orders
        planned_action = ""
        planned_qty = 0
        if plan and hasattr(plan, 'orders'):
            for o in plan.orders:
                if (o.symbol or '').upper() == sym:
                    planned_action = o.action
                    planned_qty = int(o.quantity or 0)
                    break

        # TWS orders
        tws_buy = 0
        tws_sell = 0
        tws_status = ""
        for o in live_orders:
            if o.get('symbol', '').upper() == sym:
                qty = int(o.get('quantity', 0))
                action = (o.get('action', '') or '').upper()
                if action == 'BUY':
                    tws_buy += qty
                elif action == 'SELL':
                    tws_sell += qty
                tws_status = o.get('status', '')

        # Projected
        projected_qty = pos_qty + tws_buy - tws_sell
        projected_value = projected_qty * price if price else 0
        projected_weight = (projected_value / nav * 100) if nav > 0 else 0

        # Detect conflicts
        conflict = ""
        if planned_action == "BUY" and tws_sell > 0:
            conflict = f"CONFLICT: Plan=BUY, TWS=SELL"
        elif planned_action == "SELL" and tws_buy > 0:
            conflict = f"CONFLICT: Plan=SELL, TWS=BUY"
        elif planned_qty > 0 and tws_buy == 0 and tws_sell == 0:
            conflict = f"MISSING: Plan={planned_action} {planned_qty}, TWS=none"
        elif tws_buy > 0 and planned_action != "BUY":
            conflict = f"EXTRA: TWS=BUY {tws_buy}, Plan={planned_action or 'none'}"
        elif tws_sell > 0 and planned_action != "SELL":
            conflict = f"EXTRA: TWS=SELL {tws_sell}, Plan={planned_action or 'none'}"

        comparison_rows.append({
            "symbol": sym,
            "price": price,
            "current_qty": pos_qty,
            "current_value": current_value,
            "current_weight_pct": round(current_weight, 4),
            "target_weight_pct": round(target_weight, 4),
            "target_shares": target_shares,
            "planned_action": planned_action,
            "planned_qty": planned_qty,
            "tws_buy": tws_buy,
            "tws_sell": tws_sell,
            "tws_status": tws_status,
            "projected_qty": projected_qty,
            "projected_value": projected_value,
            "projected_weight_pct": round(projected_weight, 4),
            "conflict": conflict,
        })

    debug_data["comparison_table"] = comparison_rows
    debug_data["comparison_table_count"] = len(comparison_rows)
    debug_data["nav"] = nav

    # Count conflicts
    conflicts = [r for r in comparison_rows if r.get('conflict')]
    debug_data["conflicts_count"] = len(conflicts)
    debug_data["conflicts"] = conflicts

    # ==========================================================================
    # 4. WRITE FILES
    # ==========================================================================

    # Main JSON file
    json_path = os.path.join(output_dir, f"{base_name}.json")
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2, default=str)
        created_files['json'] = json_path
        _log.info(f"debug_export: Created {json_path}")
    except Exception as e:
        _log.error(f"debug_export: Failed to write JSON: {e}")

    # CSV files
    try:
        import pandas as pd

        # Positions CSV
        if live_positions:
            pos_path = os.path.join(output_dir, f"{base_name}_positions.csv")
            pd.DataFrame(live_positions).to_csv(pos_path, index=False)
            created_files['positions_csv'] = pos_path

        # Open Orders CSV
        if live_orders:
            orders_path = os.path.join(output_dir, f"{base_name}_orders.csv")
            pd.DataFrame(live_orders).to_csv(orders_path, index=False)
            created_files['orders_csv'] = orders_path

        # Comparison table CSV
        if comparison_rows:
            comp_path = os.path.join(output_dir, f"{base_name}_comparison.csv")
            pd.DataFrame(comparison_rows).to_csv(comp_path, index=False)
            created_files['comparison_csv'] = comp_path

        # Planned orders CSV
        if plan and hasattr(plan, 'orders') and plan.orders:
            planned = []
            for o in plan.orders:
                planned.append({
                    "symbol": o.symbol,
                    "action": o.action,
                    "quantity": o.quantity,
                    "order_type": o.order_type,
                    "reason": o.reason,
                })
            plan_path = os.path.join(output_dir, f"{base_name}_planned.csv")
            pd.DataFrame(planned).to_csv(plan_path, index=False)
            created_files['planned_csv'] = plan_path

        # Target weights CSV
        if targets and hasattr(targets, 'weights') and targets.weights:
            targets_list = []
            for sym, w in targets.weights.items():
                price = (price_map or {}).get(sym, 0)
                target_value = w * nav if nav > 0 else 0
                target_shares = int(target_value / price) if price > 0 else 0
                targets_list.append({
                    "symbol": sym,
                    "target_weight_pct": round(w * 100, 4),
                    "target_value": round(target_value, 2),
                    "target_shares": target_shares,
                    "price": price,
                })
            targets_path = os.path.join(output_dir, f"{base_name}_targets.csv")
            pd.DataFrame(targets_list).to_csv(targets_path, index=False)
            created_files['targets_csv'] = targets_path

    except ImportError:
        _log.warning("debug_export: pandas not available, skipping CSV exports")
    except Exception as e:
        _log.error(f"debug_export: Failed to write CSV files: {e}")

    # Summary text file
    summary_path = os.path.join(output_dir, f"{base_name}_summary.txt")
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("AI PORTFOLIO MANAGER - DEBUG EXPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Export Time: {ts.isoformat()}\n")
            f.write(f"Account: {account}\n")
            f.write(f"Strategy: {strategy_key}\n")
            f.write(f"NAV: ${nav:,.2f}\n")
            f.write(f"IBKR Connected: {debug_data['ibkr_connected']}\n\n")

            f.write("-" * 80 + "\n")
            f.write("COUNTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Live Positions: {len(live_positions)}\n")
            f.write(f"Live Open Orders in TWS: {len(live_orders)}\n")
            f.write(f"Target Symbols: {len(targets.weights) if targets and hasattr(targets, 'weights') else 0}\n")
            f.write(f"Planned Orders: {len(plan.orders) if plan and hasattr(plan, 'orders') else 0}\n")
            f.write(f"Price Map Entries: {len(price_map) if price_map else 0}\n\n")

            if conflicts:
                f.write("-" * 80 + "\n")
                f.write(f"🚨 CONFLICTS DETECTED: {len(conflicts)}\n")
                f.write("-" * 80 + "\n")
                for c in conflicts[:20]:
                    f.write(f"  {c['symbol']}: {c['conflict']}\n")
                if len(conflicts) > 20:
                    f.write(f"  ... and {len(conflicts) - 20} more\n")
                f.write("\n")

            f.write("-" * 80 + "\n")
            f.write("COMPARISON TABLE (First 50 symbols with activity)\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Symbol':<8} {'CurrQty':>8} {'Curr%':>7} {'Tgt%':>7} {'Plan':>12} {'TWS':>12} {'Conflict'}\n")
            f.write("-" * 80 + "\n")

            # Show symbols with activity first
            active = [r for r in comparison_rows if r['planned_qty'] > 0 or r['tws_buy'] > 0 or r['tws_sell'] > 0]
            for r in active[:50]:
                plan_str = f"{r['planned_action']} {r['planned_qty']}" if r['planned_qty'] > 0 else "-"
                tws_str = ""
                if r['tws_buy'] > 0:
                    tws_str = f"BUY {r['tws_buy']}"
                if r['tws_sell'] > 0:
                    tws_str += f"SELL {r['tws_sell']}" if tws_str else f"SELL {r['tws_sell']}"
                if not tws_str:
                    tws_str = "-"

                f.write(
                    f"{r['symbol']:<8} {r['current_qty']:>8} {r['current_weight_pct']:>6.2f}% {r['target_weight_pct']:>6.2f}% {plan_str:>12} {tws_str:>12} {r['conflict']}\n")

            f.write("\n")
            f.write("-" * 80 + "\n")
            f.write("FILES CREATED\n")
            f.write("-" * 80 + "\n")
            for k, v in created_files.items():
                f.write(f"  {k}: {v}\n")

        created_files['summary'] = summary_path
        _log.info(f"debug_export: Created {summary_path}")

    except Exception as e:
        _log.error(f"debug_export: Failed to write summary: {e}")

    return created_files


def export_from_session_state(output_dir: str = "debug_exports") -> Dict[str, str]:
    """
    Export debug snapshot using data from Streamlit session state.
    Call this from ui_tab.py when user clicks "Export Debug".
    """
    import streamlit as st

    # Get gateway from session state
    gw = st.session_state.get('ai_pm_gateway')

    # Get last results
    last = st.session_state.get('ai_pm_last_v1', {})

    # Get other session state data
    snapshot = last.get('snapshot')
    targets = last.get('targets')
    plan = last.get('plan')
    signals = last.get('signals')
    gates = last.get('gates')
    execution = last.get('execution')

    # Get price map
    price_map = st.session_state.get('ai_pm_price_map', {})

    # Get verification
    verification = st.session_state.get('ai_pm_last_verification')

    # Get account and strategy
    account = st.session_state.get('ai_pm_state_v1', {}).get('selected_account', '')
    strategy_key = st.session_state.get('ai_pm_state_v1', {}).get('strategy_key', '')

    return export_debug_snapshot(
        gw=gw,
        snapshot=snapshot,
        targets=targets,
        plan=plan,
        signals=signals,
        gates=gates,
        execution=execution,
        price_map=price_map,
        verification=verification,
        account=account,
        strategy_key=strategy_key,
        extra_context={
            "session_state_keys": list(st.session_state.keys()),
        },
        output_dir=output_dir,
    )