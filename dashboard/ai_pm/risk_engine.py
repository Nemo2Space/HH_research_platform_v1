# dashboard/ai_pm/risk_engine.py
"""
Risk-engine helpers for AI Portfolio Manager
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import math

@dataclass
class GateReport:
    ts_utc: datetime
    hard_blocks: List[str] = field(default_factory=list)
    soft_warnings: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    passed: bool = True
    blocked: bool = False
    summary: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

def _safe_float(x) -> float:
    if x is None:
        return 0.0
    try:
        v = float(x)
        return 0.0 if (math.isnan(v) or math.isinf(v)) else v
    except:
        return 0.0

def _sector_map_from_signals(signals) -> Dict[str, str]:
    sector_map = {}
    if signals and hasattr(signals, 'rows'):
        for row in (signals.rows or []):
            sym = (getattr(row, 'symbol', '') or '').strip().upper()
            sector = getattr(row, 'sector', None) or 'Unknown'
            if sym:
                sector_map[sym] = sector
    return sector_map

def _project_post_trade_weights(snapshot, plan, price_map: Optional[Dict[str, float]] = None):
    warnings = []
    nav = _safe_float(getattr(snapshot, 'net_liquidation', None))
    cash = _safe_float(getattr(snapshot, 'total_cash', None))
    if nav <= 0:
        return {}, 1.0, ["NAV is zero; cannot project weights."]
    
    px_map: Dict[str, float] = {}
    for p in (snapshot.positions or []):
        sym = (getattr(p, 'symbol', '') or '').strip().upper()
        if not sym:
            continue
        px = _safe_float(getattr(p, 'market_price', None))
        if px <= 0:
            px = _safe_float(getattr(p, 'avgCost', None))
        if px > 0:
            px_map[sym] = px
    
    if price_map:
        for sym, px in price_map.items():
            sym_upper = (sym or "").strip().upper()
            px_val = _safe_float(px)
            if sym_upper and px_val > 0 and sym_upper not in px_map:
                px_map[sym_upper] = px_val
    
    pos_val = {}
    for p in (snapshot.positions or []):
        sym = (getattr(p, 'symbol', '') or '').strip().upper()
        qty = _safe_float(getattr(p, 'quantity', 0))
        if sym and qty != 0:
            px = px_map.get(sym, 0)
            pos_val[sym] = qty * px if px > 0 else 0
    
    delta_cash = 0.0
    for order in (plan.orders or []):
        sym = (getattr(order, 'symbol', '') or '').strip().upper()
        action = (getattr(order, 'action', '') or '').upper()
        qty = _safe_float(getattr(order, 'quantity', 0))
        px = px_map.get(sym, 0)
        if px <= 0:
            warnings.append(f"Missing price for {sym}; cannot project impact accurately.")
            continue
        order_val = qty * px
        if action == 'BUY':
            pos_val[sym] = pos_val.get(sym, 0) + order_val
            delta_cash -= order_val
        elif action == 'SELL':
            pos_val[sym] = pos_val.get(sym, 0) - order_val
            delta_cash += order_val
    
    proj_cash = cash + delta_cash
    proj_nav = sum(pos_val.values()) + proj_cash
    if proj_nav <= 0:
        proj_nav = nav
    
    proj_w = {sym: val / proj_nav for sym, val in pos_val.items() if val > 0}
    proj_cash_w = proj_cash / proj_nav if proj_nav > 0 else 0
    
    return proj_w, proj_cash_w, warnings

def evaluate_trade_plan_gates(
    *,
    snapshot,
    signals,
    plan,
    constraints,
    price_map: Optional[Dict[str, float]] = None,
) -> GateReport:
    ts = datetime.utcnow()
    hard_blocks = []
    soft_warnings = []
    metrics = {}
    
    nav = _safe_float(getattr(snapshot, 'net_liquidation', None))
    cash = _safe_float(getattr(snapshot, 'total_cash', None))
    
    turnover = _safe_float(getattr(plan, 'turnover_est', None))
    max_turnover = _safe_float(getattr(constraints, 'max_turnover_pct', 0.5))
    if turnover and turnover > max_turnover:
        hard_blocks.append(f"Turnover {turnover*100:.1f}% exceeds max {max_turnover*100:.1f}%")
    metrics['turnover_pct'] = turnover * 100 if turnover else 0
    
    proj_w, proj_cash_w, proj_warn = _project_post_trade_weights(snapshot, plan, price_map)
    soft_warnings.extend(proj_warn)
    
    cash_min = _safe_float(getattr(constraints, 'cash_min_pct', 0.02))
    cash_max = _safe_float(getattr(constraints, 'cash_max_pct', 0.15))
    
    if proj_cash_w < cash_min:
        hard_blocks.append(f"Projected cash {proj_cash_w*100:.1f}% < cash_min {cash_min*100:.1f}%")
    if proj_cash_w > cash_max:
        hard_blocks.append(f"Projected cash {proj_cash_w*100:.1f}% > cash_max {cash_max*100:.1f}%")
    metrics['projected_cash_pct'] = proj_cash_w * 100
    
    max_pos = _safe_float(getattr(constraints, 'max_position_pct', 0.10))
    for sym, w in proj_w.items():
        if w > max_pos:
            soft_warnings.append(f"{sym} projected at {w*100:.1f}% > max {max_pos*100:.1f}%")
    
    passed = len(hard_blocks) == 0
    blocked = len(hard_blocks) > 0
    summary = "PASSED" if passed else f"BLOCKED: {len(hard_blocks)} issue(s)"
    
    all_warnings = hard_blocks + soft_warnings
    
    return GateReport(
        ts_utc=ts,
        hard_blocks=hard_blocks,
        soft_warnings=soft_warnings,
        warnings=all_warnings,
        passed=passed,
        blocked=blocked,
        summary=summary,
        metrics=metrics,
    )
