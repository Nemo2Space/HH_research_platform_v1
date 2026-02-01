# Post-Execution Verification Module for AI PM
# Creates verification panel to compare orders sent vs TWS state

import sys
sys.path.insert(0, '..')

verification_code = '''
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from collections import defaultdict
import logging

_log = logging.getLogger(__name__)

@dataclass
class OrderVerification:
    """Verification result for a single symbol"""
    symbol: str
    target_weight: float  # Target weight %
    target_shares: int  # Target shares
    current_shares: int  # Current position shares
    pending_buy: int  # Pending BUY order shares
    pending_sell: int  # Pending SELL order shares
    projected_shares: int  # After orders fill
    current_value: float  # Current position value
    projected_value: float  # Projected value after fills
    current_weight: float  # Current weight %
    projected_weight: float  # Projected weight %
    accuracy: float  # How close to target (%)
    status: str  # "On Target", "Pending", "Missing Order", "Extra Order"
    price: float

@dataclass 
class PortfolioVerification:
    """Full portfolio verification result"""
    nav: float
    total_positions: int
    total_open_orders: int
    symbols_verified: List[OrderVerification]
    missing_orders: List[str]  # Symbols that should have orders but don't
    extra_orders: List[str]  # Symbols with unexpected orders
    total_accuracy: float  # Overall portfolio accuracy %
    projected_invested: float  # Total projected invested value
    projected_cash: float  # Projected cash after fills


def verify_execution(
    *,
    ib,
    snapshot,
    plan,
    targets,
    price_map: Dict[str, float],
) -> PortfolioVerification:
    """
    Verify execution by comparing:
    - Current positions from snapshot
    - Open orders from TWS
    - Planned orders
    - Target weights
    
    Returns detailed verification for each symbol.
    """
    _log.info("verify_execution: fetching open orders from TWS...")
    
    # Fetch fresh open orders from TWS
    open_trades = ib.openTrades()
    _log.info(f"verify_execution: found {len(open_trades)} open orders in TWS")
    
    nav = float(snapshot.net_liquidation or 0)
    
    # Build current positions map
    positions_map: Dict[str, int] = {}
    for p in snapshot.positions:
        sym = (p.symbol or "").strip().upper()
        if sym:
            positions_map[sym] = int(p.quantity or 0)
    
    # Build open orders map (net: BUY positive, SELL negative)
    orders_map: Dict[str, Dict[str, int]] = defaultdict(lambda: {"buy": 0, "sell": 0})
    for trade in open_trades:
        sym = trade.contract.symbol.upper()
        qty = int(trade.order.totalQuantity or 0)
        action = (trade.order.action or "").upper()
        if action == "BUY":
            orders_map[sym]["buy"] += qty
        elif action == "SELL":
            orders_map[sym]["sell"] += qty
    
    # Build planned orders map for comparison
    planned_map: Dict[str, Dict[str, int]] = defaultdict(lambda: {"buy": 0, "sell": 0})
    for o in plan.orders:
        sym = (o.symbol or "").strip().upper()
        qty = int(o.quantity or 0)
        action = (o.action or "").upper()
        if action == "BUY":
            planned_map[sym]["buy"] += qty
        elif action == "SELL":
            planned_map[sym]["sell"] += qty
    
    # Target shares map
    target_shares_map: Dict[str, int] = {}
    for sym, weight in targets.weights.items():
        sym = sym.strip().upper()
        price = price_map.get(sym, 0)
        if price > 0 and nav > 0:
            target_value = nav * weight
            target_shares_map[sym] = int(target_value / price)
    
    # Get all symbols involved
    all_symbols = set(positions_map.keys()) | set(orders_map.keys()) | set(target_shares_map.keys())
    
    verifications = []
    missing_orders = []
    extra_orders = []
    total_projected_value = 0
    
    for sym in sorted(all_symbols):
        price = price_map.get(sym, 0)
        target_weight = targets.weights.get(sym, 0) * 100  # Convert to %
        target_shares = target_shares_map.get(sym, 0)
        current_shares = positions_map.get(sym, 0)
        pending_buy = orders_map.get(sym, {}).get("buy", 0)
        pending_sell = orders_map.get(sym, {}).get("sell", 0)
        planned_buy = planned_map.get(sym, {}).get("buy", 0)
        planned_sell = planned_map.get(sym, {}).get("sell", 0)
        
        projected_shares = current_shares + pending_buy - pending_sell
        current_value = current_shares * price
        projected_value = projected_shares * price
        
        current_weight = (current_value / nav * 100) if nav > 0 else 0
        projected_weight = (projected_value / nav * 100) if nav > 0 else 0
        
        # Calculate accuracy (how close projected is to target)
        if target_weight > 0:
            accuracy = max(0, 100 - abs(projected_weight - target_weight) / target_weight * 100)
        elif projected_weight == 0:
            accuracy = 100  # Both zero = perfect
        else:
            accuracy = 0  # Target is 0 but we have position
        
        # Determine status
        if pending_buy > 0 or pending_sell > 0:
            status = "Pending"
        elif abs(projected_weight - target_weight) < 0.1:
            status = "On Target"
        elif planned_buy > 0 and pending_buy == 0:
            status = "Missing BUY"
            missing_orders.append(sym)
        elif planned_sell > 0 and pending_sell == 0:
            status = "Missing SELL"
            missing_orders.append(sym)
        elif pending_buy > planned_buy or pending_sell > planned_sell:
            status = "Extra Order"
            extra_orders.append(sym)
        else:
            status = "No Action"
        
        total_projected_value += projected_value
        
        verifications.append(OrderVerification(
            symbol=sym,
            target_weight=target_weight,
            target_shares=target_shares,
            current_shares=current_shares,
            pending_buy=pending_buy,
            pending_sell=pending_sell,
            projected_shares=projected_shares,
            current_value=current_value,
            projected_value=projected_value,
            current_weight=current_weight,
            projected_weight=projected_weight,
            accuracy=accuracy,
            status=status,
            price=price,
        ))
    
    # Calculate overall accuracy
    total_accuracy = sum(v.accuracy * v.target_weight for v in verifications) / 100 if verifications else 0
    projected_cash = nav - total_projected_value
    
    return PortfolioVerification(
        nav=nav,
        total_positions=len(positions_map),
        total_open_orders=len(open_trades),
        symbols_verified=verifications,
        missing_orders=missing_orders,
        extra_orders=extra_orders,
        total_accuracy=total_accuracy,
        projected_invested=total_projected_value,
        projected_cash=projected_cash,
    )


def format_verification_summary(v: PortfolioVerification) -> str:
    """Format verification as text summary"""
    lines = [
        f"{'='*60}",
        f"POST-EXECUTION VERIFICATION",
        f"{'='*60}",
        f"NAV: \",
        f"Positions: {v.total_positions}",
        f"Open Orders in TWS: {v.total_open_orders}",
        f"Overall Accuracy: {v.total_accuracy:.1f}%",
        f"Projected Invested: \",
        f"Projected Cash: \",
        "",
    ]
    
    if v.missing_orders:
        lines.append(f"⚠️ MISSING ORDERS: {', '.join(v.missing_orders)}")
    if v.extra_orders:
        lines.append(f"⚠️ EXTRA ORDERS: {', '.join(v.extra_orders)}")
    
    lines.append("")
    lines.append(f"{'Symbol':<8} {'Target%':>8} {'Current%':>9} {'Project%':>9} {'Status':<15} {'Accuracy':>8}")
    lines.append("-" * 70)
    
    # Sort by status (Pending first, then by target weight)
    sorted_v = sorted(v.symbols_verified, key=lambda x: (x.status != "Pending", -x.target_weight))
    
    for s in sorted_v[:30]:  # Show top 30
        lines.append(
            f"{s.symbol:<8} {s.target_weight:>7.2f}% {s.current_weight:>8.2f}% {s.projected_weight:>8.2f}% "
            f"{s.status:<15} {s.accuracy:>7.1f}%"
        )
    
    if len(v.symbols_verified) > 30:
        lines.append(f"... and {len(v.symbols_verified) - 30} more symbols")
    
    return "\\n".join(lines)
'''

# Write the verification module
with open('../dashboard/ai_pm/execution_verify.py', 'w', encoding='utf-8') as f:
    f.write(verification_code)

print("✅ Created dashboard/ai_pm/execution_verify.py")
