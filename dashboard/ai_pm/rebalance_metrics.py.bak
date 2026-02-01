# dashboard/ai_pm/rebalance_metrics.py
"""
Enhanced Rebalance Metrics Module for AI Portfolio Manager
==========================================================

Provides the same detailed metrics and results display as the ETF Rebalancer:
- Trading Volume, Buy/Sell breakdown
- Fees, Accuracy, Turnover
- Cash Flow, Cost Impact, Tracking Error
- VaR, Beta, Sharpe, Max Drawdown
- Portfolio NAV, Cash Position, Buying Power, Margin Usage
- Detailed per-position results table with accuracy

This ensures the AI PM has the same professional-grade output as your ETF rebalancer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import math

import pandas as pd


@dataclass
class RebalanceMetrics:
    """Comprehensive rebalancing metrics matching ETF Rebalancer display."""

    # Trading metrics
    trading_volume: float = 0.0  # Total transaction value (buys + sells)
    total_buy: float = 0.0  # Total buy order value
    total_sell: float = 0.0  # Total sell order value
    total_fees: float = 0.0  # Estimated fees and commissions

    # Performance metrics
    accuracy: float = 0.0  # Est. portfolio alignment (%)
    turnover: float = 0.0  # Portfolio rotation rate (%)
    cash_flow: float = 0.0  # Net capital required
    cost_impact: float = 0.0  # Return reduction (%)
    tracking_error: float = 0.0  # Est. deviation from target (%)

    # Risk metrics
    var_95: float = 0.0  # Daily value at risk (%)
    beta: float = 1.0  # Market sensitivity
    sharpe: float = 0.0  # Risk-adjusted return
    max_drawdown: float = 0.0  # Worst decline (%)

    # Portfolio state
    portfolio_nav: float = 0.0  # Total portfolio value
    cash_position: float = 0.0  # Available cash
    buying_power: float = 0.0  # Available for trading
    margin_usage: float = 0.0  # Margin utilization (%)

    # Deployment
    capital_to_deploy: float = 0.0  # Amount user wants to deploy
    capital_deployed: float = 0.0  # Actual amount being deployed
    remaining_cash: float = 0.0  # Cash after deployment

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trading_volume': self.trading_volume,
            'total_buy': self.total_buy,
            'total_sell': self.total_sell,
            'total_fees': self.total_fees,
            'accuracy': self.accuracy,
            'turnover': self.turnover,
            'cash_flow': self.cash_flow,
            'cost_impact': self.cost_impact,
            'tracking_error': self.tracking_error,
            'var_95': self.var_95,
            'beta': self.beta,
            'sharpe': self.sharpe,
            'max_drawdown': self.max_drawdown,
            'portfolio_nav': self.portfolio_nav,
            'cash_position': self.cash_position,
            'buying_power': self.buying_power,
            'margin_usage': self.margin_usage,
            'capital_to_deploy': self.capital_to_deploy,
            'capital_deployed': self.capital_deployed,
            'remaining_cash': self.remaining_cash,
        }


@dataclass
class PositionResult:
    """Detailed result for a single position after rebalancing."""
    symbol: str

    # Weights
    original_weight_pct: float = 0.0  # Current weight (%)
    target_weight_pct: float = 0.0  # Target weight (%)

    # Shares
    open_shares: int = 0  # Open order shares
    position_shares: int = 0  # Current position shares
    target_shares: int = 0  # Target shares

    # Values
    price: float = 0.0  # Current/last price
    order_value: float = 0.0  # Value of orders
    position_value: float = 0.0  # Value of current position
    total_value: float = 0.0  # Total value (position + orders)

    # Result
    actual_weight_pct: float = 0.0  # Achieved weight (%)
    accuracy_pct: float = 0.0  # How close to target (%)

    # Action
    action: str = ""  # BUY, SELL, HOLD
    order_quantity: int = 0  # Shares to trade

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'original_weight_pct': self.original_weight_pct,
            'target_weight_pct': self.target_weight_pct,
            'open_shares': self.open_shares,
            'position_shares': self.position_shares,
            'target_shares': self.target_shares,
            'price': self.price,
            'order_value': self.order_value,
            'position_value': self.position_value,
            'total_value': self.total_value,
            'actual_weight_pct': self.actual_weight_pct,
            'accuracy_pct': self.accuracy_pct,
            'action': self.action,
            'order_quantity': self.order_quantity,
        }


@dataclass
class RebalanceResults:
    """Complete rebalancing results with metrics and position details."""
    ts_utc: datetime
    account: str
    strategy_key: str

    # Summary
    status: str = "PENDING"  # PENDING, EXECUTING, COMPLETE, FAILED
    message: str = ""

    # Metrics
    metrics: RebalanceMetrics = field(default_factory=RebalanceMetrics)

    # Position results
    positions: List[PositionResult] = field(default_factory=list)

    # Orders sent
    orders_sent: int = 0
    orders_filled: int = 0
    orders_failed: int = 0

    # Verification
    verified: bool = False
    verification_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ts_utc': self.ts_utc.isoformat(),
            'account': self.account,
            'strategy_key': self.strategy_key,
            'status': self.status,
            'message': self.message,
            'metrics': self.metrics.to_dict(),
            'positions': [p.to_dict() for p in self.positions],
            'orders_sent': self.orders_sent,
            'orders_filled': self.orders_filled,
            'orders_failed': self.orders_failed,
            'verified': self.verified,
            'verification_message': self.verification_message,
        }


def _safe_float(x) -> float:
    if x is None:
        return 0.0
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v
    except:
        return 0.0


def _safe_int(x) -> int:
    try:
        return int(_safe_float(x))
    except:
        return 0


def calculate_rebalance_metrics(
        *,
        snapshot,
        targets,
        plan,
        price_map: Dict[str, float],
        capital_to_deploy: Optional[float] = None,
) -> RebalanceMetrics:
    """
    Calculate comprehensive rebalancing metrics matching ETF Rebalancer style.
    """
    metrics = RebalanceMetrics()

    # Get NAV and cash from snapshot
    nav = _safe_float(getattr(snapshot, 'net_liquidation', None))
    cash = _safe_float(getattr(snapshot, 'total_cash', None))
    buying_power = _safe_float(getattr(snapshot, 'buying_power', None))

    metrics.portfolio_nav = nav
    metrics.cash_position = cash
    metrics.buying_power = buying_power if buying_power > 0 else cash

    # Capital deployment
    if capital_to_deploy and capital_to_deploy > 0:
        metrics.capital_to_deploy = capital_to_deploy
        metrics.capital_deployed = min(capital_to_deploy, nav * 0.995)
        metrics.remaining_cash = nav - metrics.capital_deployed
    else:
        metrics.capital_to_deploy = nav
        metrics.capital_deployed = nav * 0.95  # Default 95%
        metrics.remaining_cash = nav * 0.05

    # Get current positions
    positions = getattr(snapshot, 'positions', []) or []
    position_map = {}
    for p in positions:
        sym = (getattr(p, 'symbol', '') or '').strip().upper()
        if sym:
            position_map[sym] = {
                'quantity': _safe_float(getattr(p, 'quantity', 0)),
                'market_value': _safe_float(getattr(p, 'market_value', 0)),
            }

    # Get target weights from plan (NORMALIZED by trade_planner)
    target_weights = getattr(plan, 'target_weights', None) or {}
    if not target_weights:
        target_weights = targets.weights if targets and targets.weights else {}

    # Get orders from plan
    orders = getattr(plan, 'orders', []) or []

    # Calculate trading metrics
    total_buy = 0.0
    total_sell = 0.0
    total_fees = 0.0

    for order in orders:
        sym = (getattr(order, 'symbol', '') or '').strip().upper()
        action = (getattr(order, 'action', '') or '').upper()
        qty = _safe_float(getattr(order, 'quantity', 0))

        price = price_map.get(sym, 0)
        if price <= 0:
            continue

        order_value = qty * price

        # Estimate commission (IBKR tiered: $0.0035/share, min $0.35, max 1% of trade value)
        commission = min(max(qty * 0.0035, 0.35), order_value * 0.01)

        if action == 'BUY':
            total_buy += order_value
        elif action == 'SELL':
            total_sell += order_value

        total_fees += commission

    metrics.trading_volume = total_buy + total_sell
    metrics.total_buy = total_buy
    metrics.total_sell = total_sell
    metrics.total_fees = total_fees
    metrics.cash_flow = total_sell - total_buy  # Negative means cash needed

    # Calculate turnover
    if nav > 0:
        metrics.turnover = (metrics.trading_volume / nav) * 100
        metrics.cost_impact = (total_fees / nav) * 100

    # Calculate margin usage
    if nav > 0:
        total_position_value = sum(p['market_value'] for p in position_map.values())
        metrics.margin_usage = (total_position_value / nav) * 100 if nav > 0 else 0

    # Calculate expected accuracy
    # Compare target weights to what we'll achieve
    total_target_value = 0
    total_achieved_value = 0

    for sym, target_w in target_weights.items():
        price = price_map.get(sym, 0)
        if price <= 0:
            continue

        target_value = target_w * nav
        total_target_value += target_value

        # Current position value
        current = position_map.get(sym, {})
        current_value = current.get('market_value', 0)

        # Find order for this symbol
        order_value = 0
        for order in orders:
            if (getattr(order, 'symbol', '') or '').strip().upper() == sym:
                action = (getattr(order, 'action', '') or '').upper()
                qty = _safe_float(getattr(order, 'quantity', 0))
                ov = qty * price
                if action == 'BUY':
                    order_value += ov
                elif action == 'SELL':
                    order_value -= ov
                break

        achieved_value = current_value + order_value
        total_achieved_value += achieved_value

    if total_target_value > 0:
        metrics.accuracy = (total_achieved_value / total_target_value) * 100
        metrics.tracking_error = abs(100 - metrics.accuracy)
    else:
        metrics.accuracy = 100
        metrics.tracking_error = 0

    # Risk metrics - NOT calculated without historical data
    # These remain at 0.0/None until we have real historical data to calculate them
    # DO NOT use placeholder/fake values
    metrics.var_95 = 0.0  # Requires historical data
    metrics.beta = 0.0  # Requires historical data
    metrics.sharpe = 0.0  # Requires historical data
    metrics.max_drawdown = 0.0  # Requires historical data

    return metrics


def build_position_results(
        *,
        snapshot,
        targets,
        plan,
        price_map: Dict[str, float],
        signals=None,
) -> List[PositionResult]:
    """
    Build detailed position-by-position results matching ETF Rebalancer table.
    """
    results = []

    nav = _safe_float(getattr(snapshot, 'net_liquidation', None))
    if nav <= 0:
        return results

    # Build position map
    positions = getattr(snapshot, 'positions', []) or []
    position_map = {}
    for p in positions:
        sym = (getattr(p, 'symbol', '') or '').strip().upper()
        if sym:
            qty = _safe_float(getattr(p, 'quantity', 0)) or _safe_float(getattr(p, 'position', 0))
            # Try to get market_value directly, or calculate from price
            mv = _safe_float(getattr(p, 'market_value', 0))
            if mv == 0 and qty != 0:
                # Calculate from price_map if available
                price = price_map.get(sym, 0)
                if price > 0:
                    mv = abs(qty) * price
                else:
                    # Fall back to avgCost * quantity
                    avg_cost = _safe_float(getattr(p, 'avgCost', 0)) or _safe_float(getattr(p, 'average_cost', 0))
                    if avg_cost > 0:
                        mv = abs(qty) * avg_cost
            position_map[sym] = {
                'quantity': qty,
                'market_value': mv,
            }

    # Get target weights from plan (NORMALIZED by trade_planner)
    target_weights = getattr(plan, 'target_weights', None) or {}
    if not target_weights:
        target_weights = targets.weights if targets and targets.weights else {}

    # Get current weights from plan
    current_weights = getattr(plan, 'current_weights', {}) or {}

    # Get orders from plan
    orders = getattr(plan, 'orders', []) or []
    order_map = {}
    for order in orders:
        sym = (getattr(order, 'symbol', '') or '').strip().upper()
        action = (getattr(order, 'action', '') or '').upper()
        qty = _safe_float(getattr(order, 'quantity', 0))
        order_map[sym] = {'action': action, 'quantity': qty}

    # Build results for all symbols (current + target)
    all_symbols = set(position_map.keys()) | set(target_weights.keys())

    for sym in sorted(all_symbols):
        price = price_map.get(sym, 0)

        # Current position
        current = position_map.get(sym, {})
        current_qty = _safe_int(current.get('quantity', 0))
        current_value = _safe_float(current.get('market_value', 0))
        current_weight = (current_value / nav) if nav > 0 else 0  # FIX: use actual position weight

        # Target
        target_weight = _safe_float(target_weights.get(sym, 0))
        target_value = target_weight * nav
        target_qty = _safe_int(target_value / price) if price > 0 else 0

        # Order
        order_info = order_map.get(sym, {})
        action = order_info.get('action', 'HOLD')
        order_qty = _safe_int(order_info.get('quantity', 0))
        order_value = order_qty * price if price > 0 else 0

        # Calculate achieved values
        if action == 'BUY':
            total_qty = current_qty + order_qty
            total_value = current_value + order_value
        elif action == 'SELL':
            total_qty = current_qty - order_qty
            total_value = current_value - order_value
        else:
            total_qty = current_qty
            total_value = current_value

        # Calculate actual weight and accuracy
        actual_weight = (total_value / nav) if nav > 0 else 0

        # FIXED: Calculate accuracy based on ACHIEVABLE target (whole shares)
        # The achievable target is what we CAN achieve with whole share constraints
        # Example: Target 0.50% with  stock = 2.5 shares needed, but we can only hold 2
        # So achievable_weight = 2 *  / NAV = the best we can do
        if target_weight > 0 and price > 0:
            # Calculate achievable target weight based on target_qty (rounded shares)
            achievable_value = target_qty * price
            achievable_weight = achievable_value / nav if nav > 0 else 0

            # Accuracy = how close actual is to achievable (not theoretical)
            if achievable_weight > 0:
                accuracy = min((actual_weight / achievable_weight) * 100, 100)
            else:
                accuracy = 0
        elif target_weight == 0 and actual_weight == 0:
            accuracy = 100  # Both zero = perfect
        elif target_weight == 0:
            accuracy = 0  # Target is 0 but we have position
        else:
            accuracy = 0

        results.append(PositionResult(
            symbol=sym,
            original_weight_pct=current_weight * 100,
            target_weight_pct=target_weight * 100,
            open_shares=order_qty if action in ('BUY', 'SELL') else 0,
            position_shares=current_qty,
            target_shares=target_qty,
            price=price,
            order_value=order_value,
            position_value=current_value,
            total_value=total_value,
            actual_weight_pct=actual_weight * 100,
            accuracy_pct=accuracy,
            action=action if action else 'HOLD',
            order_quantity=order_qty,
        ))

    # Sort by target weight descending
    results.sort(key=lambda x: x.target_weight_pct, reverse=True)

    return results


def build_rebalance_results(
        *,
        snapshot,
        targets,
        plan,
        signals,
        price_map: Dict[str, float],
        account: str,
        strategy_key: str,
        capital_to_deploy: Optional[float] = None,
) -> RebalanceResults:
    """
    Build complete rebalancing results with metrics and position details.
    """
    ts = datetime.utcnow()

    metrics = calculate_rebalance_metrics(
        snapshot=snapshot,
        targets=targets,
        plan=plan,
        price_map=price_map,
        capital_to_deploy=capital_to_deploy,
    )

    positions = build_position_results(
        snapshot=snapshot,
        targets=targets,
        plan=plan,
        price_map=price_map,
        signals=signals,
    )

    orders = getattr(plan, 'orders', []) or []

    return RebalanceResults(
        ts_utc=ts,
        account=account,
        strategy_key=strategy_key,
        status="PENDING",
        message=f"Ready to execute {len(orders)} orders",
        metrics=metrics,
        positions=positions,
        orders_sent=0,
        orders_filled=0,
        orders_failed=0,
    )


def verify_execution(
        *,
        ib,
        results: RebalanceResults,
        target_weights: Dict[str, float],
        price_map: Dict[str, float],
        account: str,
        tolerance_pct: float = 2.0,
) -> RebalanceResults:
    """
    Verify that orders were executed correctly by comparing actual positions
    to expected positions after execution.

    Returns updated RebalanceResults with verification status.
    """
    from datetime import datetime
    import time

    # Wait for orders to settle
    time.sleep(2)

    # Get current positions from IBKR
    try:
        ib.reqAllOpenOrders()
        time.sleep(1)
        current_positions = ib.positions()
        open_orders = ib.openTrades()
    except Exception as e:
        results.verified = False
        results.verification_message = f"Failed to fetch positions: {e}"
        return results

    # Build current position map
    position_map = {}
    total_value = 0
    for pos in current_positions:
        if hasattr(pos, 'contract') and hasattr(pos, 'position'):
            sym = getattr(pos.contract, 'symbol', '').strip().upper()
            qty = float(getattr(pos, 'position', 0))
            price = price_map.get(sym, 0)
            value = qty * price
            position_map[sym] = {'quantity': qty, 'value': value}
            total_value += value

    # Check each target position
    mismatches = []
    for sym, target_w in target_weights.items():
        target_value = target_w * total_value if total_value > 0 else 0
        actual = position_map.get(sym, {})
        actual_value = actual.get('value', 0)

        if target_value > 0:
            diff_pct = abs(actual_value - target_value) / target_value * 100
            if diff_pct > tolerance_pct:
                mismatches.append({
                    'symbol': sym,
                    'target_value': target_value,
                    'actual_value': actual_value,
                    'diff_pct': diff_pct,
                })

    # Check for pending orders
    pending_count = len([o for o in open_orders if o.orderStatus.status not in ('Filled', 'Cancelled')])

    # Update results
    if not mismatches and pending_count == 0:
        results.verified = True
        results.verification_message = "✅ All positions match targets within tolerance"
        results.status = "COMPLETE"
    elif pending_count > 0:
        results.verified = False
        results.verification_message = f"⏳ {pending_count} orders still pending"
        results.status = "EXECUTING"
    else:
        results.verified = False
        mismatch_syms = [m['symbol'] for m in mismatches[:5]]
        results.verification_message = f"⚠️ {len(mismatches)} positions differ from targets: {', '.join(mismatch_syms)}"
        results.status = "COMPLETE"  # Orders sent, but not perfectly matched

    # Update metrics with actual values
    results.metrics.portfolio_nav = total_value

    return results