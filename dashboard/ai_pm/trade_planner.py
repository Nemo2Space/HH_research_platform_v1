from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import math

from .config import DEFAULT_CONSTRAINTS, RiskConstraints
from .models import OrderTicket, PortfolioSnapshot, Position, TargetWeights, TradePlan


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


def _pos_market_value(p: Position) -> Optional[float]:
    mv = _safe_float(p.market_value)
    if mv is not None:
        return mv

    px = _safe_float(p.market_price)
    if px is not None:
        return px * float(p.quantity or 0.0)

    ac = _safe_float(p.avg_cost)
    if ac is not None:
        return ac * float(p.quantity or 0.0)

    return None


def _pos_price(p: Position) -> Optional[float]:
    px = _safe_float(p.market_price)
    if px is not None and px > 0:
        return px

    ac = _safe_float(p.avg_cost)
    if ac is not None and ac > 0:
        return ac

    return None


def _compute_current_weights(snapshot: PortfolioSnapshot, price_map: Optional[Dict[str, float]] = None) -> Tuple[
    Dict[str, float], Dict[str, float], List[str]]:
    """
    Compute current portfolio weights from snapshot positions.

    FIX: Uses price_map (fresh Yahoo prices) when available instead of
    potentially stale Position.market_value from IBKR.
    """
    warnings: List[str] = []
    values: Dict[str, float] = {}

    nav = _safe_float(snapshot.net_liquidation)
    cash = _safe_float(snapshot.total_cash)

    # Build price map lookup (normalize keys to uppercase)
    pm = {}
    if price_map:
        for k, v in price_map.items():
            if k and v:
                fv = _safe_float(v)
                if fv and fv > 0:
                    pm[k.strip().upper()] = fv

    pos_total = 0.0
    for p in snapshot.positions:
        sym = (p.symbol or "").strip().upper()
        if not sym:
            continue

        qty = _safe_float(p.quantity) or 0
        if qty == 0:
            continue

        # FIX: Use price_map price if available (fresh from Yahoo Finance)
        # Otherwise fall back to position's market_value (may be stale from IBKR)
        px = pm.get(sym)
        if px and px > 0:
            mv = qty * px  # Calculate from fresh price
        else:
            mv = _pos_market_value(p)
            if mv is None:
                continue

        values[sym] = float(mv)
        pos_total += float(mv)

    if nav is None or nav <= 0:
        if cash is None:
            cash = 0.0
        nav = pos_total + float(cash)
        warnings.append("NAV missing from IBKR snapshot; using positions+cash approximation.")

    if nav <= 0:
        return {}, values, ["NAV is zero; cannot compute weights."]

    weights: Dict[str, float] = {}
    for sym, mv in values.items():
        weights[sym] = float(mv) / float(nav)

    return weights, values, warnings


def _clamp_qty(q: float) -> float:
    if q is None:
        return 0.0
    try:
        q = float(q)
    except Exception:
        return 0.0
    if abs(q) < 1e-6:
        return 0.0
    return q


def build_trade_plan(
        *,
        snapshot: PortfolioSnapshot,
        targets: TargetWeights,
        constraints: RiskConstraints = DEFAULT_CONSTRAINTS,
        price_map: Optional[Dict[str, float]] = None,
        capital_to_deploy: Optional[float] = None,
) -> TradePlan:
    """
    Build a trade plan from current snapshot to target weights.

    Parameters:
    -----------
    snapshot : PortfolioSnapshot
        Current portfolio state from IBKR
    targets : TargetWeights
        Target portfolio weights
    constraints : RiskConstraints
        Trading constraints
    price_map : Dict[str, float], optional
        Price map for symbols not in snapshot
    capital_to_deploy : float, optional
        Specific amount to deploy. If provided, target weights are scaled
        to this amount instead of full NAV. This allows you to control
        exactly how much capital is invested vs kept as cash.

        Example: NAV=$20,000, capital_to_deploy=$19,000
        - Target weights will be scaled to $19,000
        - $1,000 will remain as cash
    """
    ts = datetime.utcnow()

    nav = _safe_float(snapshot.net_liquidation)
    cash = _safe_float(snapshot.total_cash)

    current_w, current_val, w_warnings = _compute_current_weights(snapshot, price_map)

    warnings: List[str] = []
    warnings.extend(w_warnings)

    tgt_w = {k.strip().upper(): float(v) for k, v in (targets.weights or {}).items() if
             k and (v is not None) and float(v) > 0.0}
    s = sum(tgt_w.values())

    # FIX: Only normalize if targets exceed 100% (over-allocated)
    # If targets sum to less than 100%, keep them as-is (remainder stays as cash)
    if s > 1.0:
        # Over-allocated: scale down proportionally
        tgt_w = {k: v / s for k, v in tgt_w.items()}
        warnings.append(f"Target weights summed to {s * 100:.1f}%; normalized to 100%.")
    elif s > 0 and s < 0.99:
        # Under-allocated: keep as-is, the remainder is implicitly cash
        warnings.append(f"Target weights sum to {s * 100:.1f}%; {(1 - s) * 100:.1f}% will remain as cash.")
    elif s <= 0:
        tgt_w = {}
    # else: targets sum to ~100%, use as-is

    if nav is None or nav <= 0:
        nav = sum(current_val.values()) + float(cash or 0.0)
    if nav <= 0:
        return TradePlan(
            ts_utc=ts,
            account=snapshot.account,
            strategy_key=targets.strategy_key,
            nav=None,
            cash=cash,
            turnover_est=None,
            num_trades=0,
            current_weights=current_w,
            target_weights=tgt_w,
            drift_by_symbol={},
            orders=[],
            warnings=["NAV is zero; cannot build trades."] + warnings,
        )

    # CAPITAL TO DEPLOY LOGIC
    # If capital_to_deploy is specified, we use that as the amount to invest
    # Otherwise, we use full NAV (minus cash_target from constraints)

    effective_nav = float(nav)
    deploy_ratio = 1.0

    if capital_to_deploy is not None and capital_to_deploy > 0:
        # Validate: can't deploy more than 99.5% of NAV
        max_deploy = float(nav) * 0.995
        if capital_to_deploy > max_deploy:
            capital_to_deploy = max_deploy
            warnings.append(f"Capital to deploy capped at 99.5% of NAV (${max_deploy:,.0f})")

        # Calculate the ratio of capital to deploy vs NAV
        deploy_ratio = capital_to_deploy / float(nav)
        effective_nav = capital_to_deploy

        # Adjust target weights to reflect the deploy ratio
        # Target weights sum to 1.0, but we only want to deploy deploy_ratio of NAV
        tgt_w = {k: v * deploy_ratio for k, v in tgt_w.items()}

        warnings.append(f"Capital to deploy: ${capital_to_deploy:,.0f} ({deploy_ratio * 100:.1f}% of NAV)")
        warnings.append(f"Cash to keep: ${(float(nav) - capital_to_deploy):,.0f}")

    pos_map: Dict[str, Position] = {p.symbol.strip().upper(): p for p in snapshot.positions if p.symbol}

    symbols = sorted(set(current_w.keys()) | set(tgt_w.keys()))

    drift: Dict[str, float] = {}
    orders: List[OrderTicket] = []
    traded_notional = 0.0

    pm: Dict[str, float] = {}
    if price_map:
        for k, v in price_map.items():
            if not k:
                continue
            fv = _safe_float(v)
            if fv is None or fv <= 0:
                continue
            pm[k.strip().upper()] = float(fv)

    for sym in symbols:
        cw = float(current_w.get(sym, 0.0) or 0.0)
        tw = float(tgt_w.get(sym, 0.0) or 0.0)
        d = abs(cw - tw)
        drift[sym] = d

        delta_w = tw - cw
        if abs(delta_w) < 1e-6:
            continue

        # PROFESSIONAL PM DRIFT THRESHOLD
        # Skip rebalancing if drift is not meaningful enough to justify transaction costs
        # Criteria: Must have BOTH:
        #   1. Absolute drift > 0.5% (0.005) - meaningful in portfolio terms
        #   2. Relative drift > 10% - meaningful relative to position size
        # Exception: Full exits (tw == 0) or new positions (cw == 0) always proceed

        is_full_exit = cw > 0 and tw == 0
        is_new_position = cw == 0 and tw > 0

        if not is_full_exit and not is_new_position:
            relative_drift = d / cw if cw > 0.001 else float('inf')
            absolute_drift = d

            # Skip if drift is too small to justify rebalancing
            if absolute_drift < 0.005 or relative_drift < 0.10:
                # Still record drift but don't create order
                continue

        # Use FULL NAV for delta calculation (not effective_nav)
        # because current_w is based on full NAV
        delta_value = float(delta_w) * float(nav)

        p = pos_map.get(sym)
        px = _pos_price(p) if p else None
        if px is None or px <= 0:
            px = pm.get(sym)

        if px is None or px <= 0:
            warnings.append(f"Missing price for {sym}; skipping order sizing.")
            continue

        qty = delta_value / float(px)
        qty = _clamp_qty(qty)
        if qty == 0.0:
            continue

        # Round to whole shares - use floor for sells (conservative), ceil for buys
        if qty > 0:
            qty_rounded = math.ceil(qty)  # BUY: round up
        else:
            qty_rounded = math.floor(qty)  # SELL: round down (more negative)

        # Skip if rounded qty is 0
        if qty_rounded == 0:
            continue

        action = "BUY" if qty_rounded > 0 else "SELL"
        qty_abs = abs(qty_rounded)

        # VALIDATION: For SELL orders, ensure qty doesn't exceed current position
        if action == "SELL":
            current_pos = pos_map.get(sym)
            current_qty = int(getattr(current_pos, 'quantity', 0) or 0) if current_pos else 0
            if qty_abs > current_qty:
                if current_qty > 0:
                    qty_abs = current_qty  # Cap at current position
                    warnings.append(f"{sym}: Sell qty capped to position size ({current_qty})")
                else:
                    warnings.append(f"{sym}: Cannot sell - no position found")
                    continue

        notional = qty_abs * float(px)
        if notional < float(constraints.min_order_notional_usd):
            continue

        reason = f"Rebalance: current={cw:.3f}, target={tw:.3f}, drift={d:.3f}"
        orders.append(
            OrderTicket(
                symbol=sym,
                action=action,
                quantity=qty_abs,  # Now always a whole number
                order_type="MKT",
                limit_price=None,
                tif="DAY",
                reason=reason,
            )
        )

        traded_notional += notional

    turnover_est = traded_notional / float(nav) if nav > 0 else None

    # =========================================================================
    # CAP BUY ORDERS TO AVAILABLE CASH (prevents orders exceeding capital)
    # =========================================================================
    buy_total = sum(o.quantity * pm.get(o.symbol.strip().upper(), 0) for o in orders if o.action == "BUY")
    max_buy = float(cash or 0) * 0.95  # Target 95% deployment
    
    if buy_total > max_buy and buy_total > 0:
        scale = max_buy / buy_total
        warnings.append(f"Scaling orders to fit 95% cash (${max_buy:,.0f})")
        new_orders = []
        running_total = 0.0
        # Sort by target weight descending to prioritize larger positions
        buy_orders = sorted([o for o in orders if o.action == "BUY"], 
                           key=lambda x: tgt_w.get(x.symbol.strip().upper(), 0), reverse=True)
        sell_orders = [o for o in orders if o.action == "SELL"]
        
        for o in buy_orders:
            sym = o.symbol.strip().upper()
            px = pm.get(sym, 0)
            if px <= 0:
                continue
            new_qty = max(1, int(o.quantity * scale))  # Use int (floor) not round
            order_cost = new_qty * px
            if running_total + order_cost <= max_buy:
                new_orders.append(type(o)(
                    symbol=o.symbol, action=o.action, quantity=new_qty,
                    order_type=o.order_type, limit_price=o.limit_price,
                    tif=o.tif, reason=o.reason
                ))
                running_total += order_cost
        
        new_orders.extend(sell_orders)
        orders = new_orders
        traded_notional = sum(o.quantity * pm.get(o.symbol.strip().upper(), 0) for o in orders)
        turnover_est = traded_notional / float(nav) if nav > 0 else None

    if len(orders) > int(constraints.max_trades_per_cycle):
        def _order_notional(o: OrderTicket) -> float:
            px2 = pm.get(o.symbol.strip().upper())
            if px2 is None:
                p2 = pos_map.get(o.symbol.strip().upper())
                px2 = _pos_price(p2) if p2 else None
            if px2 is None:
                return float("-inf")
            return float(o.quantity) * float(px2)

        orders_sorted = sorted(orders, key=_order_notional, reverse=True)
        orders = orders_sorted[: int(constraints.max_trades_per_cycle)]
        warnings.append(f"Trade count capped to {constraints.max_trades_per_cycle}; trimmed smaller orders.")

    # =========================================================================
    # BLACKROCK-STYLE CASH SWEEP (PASS 2)
    # =========================================================================
    # After primary rebalance, check if excess cash remains
    # Deploy to underweight positions if: cash > 0.5% of NAV AND trade is fee-efficient
    # Fee efficiency: order_value must be > 50x the commission (~) =  minimum
    # =========================================================================

    CASH_BUFFER_PCT = 0.005  # Keep 0.5% as cash buffer
    MIN_ORDER_FOR_SWEEP = 750.0  # Minimum order - institutional standard (fee < 1%)
    COMMISSION_PER_TRADE = 5.0  # Estimated commission per trade
    MAX_FEE_DRAG_PCT = 0.01  # Max 1% fee drag (stricter)  # Don't make trades where fee > 2% of order value

    # Calculate projected cash after current orders
    buy_total = sum(o.quantity * pm.get(o.symbol, 0) for o in orders if o.action == "BUY")
    sell_total = sum(o.quantity * pm.get(o.symbol, 0) for o in orders if o.action == "SELL")

    current_cash = float(cash or 0)
    projected_cash = current_cash + sell_total - buy_total

    cash_buffer = float(nav) * CASH_BUFFER_PCT
    excess_cash = projected_cash - cash_buffer

    if excess_cash > MIN_ORDER_FOR_SWEEP:
        # Find underweight positions (not already in orders, below target)
        existing_order_symbols = {o.symbol.strip().upper() for o in orders}

        # Calculate underweight scores: (target - actual) / target
        underweight_candidates = []
        for sym in tgt_w.keys():
            if sym in existing_order_symbols:
                continue  # Already have an order for this

            cw = float(current_w.get(sym, 0) or 0)
            tw = float(tgt_w.get(sym, 0) or 0)

            if tw <= 0 or cw >= tw:
                continue  # Not underweight

            # Relative underweight score
            underweight_pct = (tw - cw) / tw
            shortfall_value = (tw - cw) * float(nav)

            px = pm.get(sym)
            if px is None or px <= 0:
                continue

            underweight_candidates.append({
                'symbol': sym,
                'underweight_pct': underweight_pct,
                'shortfall_value': shortfall_value,
                'price': px,
                'current_weight': cw,
                'target_weight': tw,
            })

        # Sort by underweight percentage (most underweight first)
        underweight_candidates.sort(key=lambda x: x['underweight_pct'], reverse=True)

        # Deploy excess cash to underweight positions
        remaining_cash = excess_cash
        sweep_orders = []

        for candidate in underweight_candidates:
            if remaining_cash < MIN_ORDER_FOR_SWEEP:
                break

            sym = candidate['symbol']
            px = candidate['price']
            shortfall = candidate['shortfall_value']

            # Don't deploy more than the shortfall
            deploy_amount = min(remaining_cash, shortfall)

            # Must meet minimum order size
            if deploy_amount < MIN_ORDER_FOR_SWEEP:
                continue

            # Check fee efficiency: commission should be < 2% of order
            if COMMISSION_PER_TRADE / deploy_amount > MAX_FEE_DRAG_PCT:
                continue

            # Calculate shares (round down for buys in sweep - conservative)
            shares = math.floor(deploy_amount / px)
            if shares < 1:
                continue

            order_value = shares * px

            # FINAL CHECK: Verify order value meets minimum after share rounding
            if order_value < MIN_ORDER_FOR_SWEEP:
                continue

            # FINAL CHECK: Verify fee efficiency on actual order value
            if COMMISSION_PER_TRADE / order_value > MAX_FEE_DRAG_PCT:
                continue

            sweep_orders.append(OrderTicket(
                symbol=sym,
                action="BUY",
                quantity=shares,
                order_type="MKT",
                limit_price=None,
                tif="DAY",
                reason=f"Cash sweep: deploying excess cash to underweight position",
            ))

            remaining_cash -= order_value
            traded_notional += order_value

        if sweep_orders:
            orders.extend(sweep_orders)
            deployed = excess_cash - remaining_cash
            warnings.append(
                f"Cash sweep: Deployed  excess cash to {len(sweep_orders)} underweight positions. "
                f"Remaining cash:  ({(remaining_cash + cash_buffer) / nav * 100:.2f}% of NAV)"
            )

    # Update turnover estimate after sweep
    turnover_est = traded_notional / float(nav) if nav > 0 else None

    return TradePlan(
        ts_utc=ts,
        account=snapshot.account,
        strategy_key=targets.strategy_key,
        nav=float(nav),
        cash=cash,
        turnover_est=turnover_est,
        num_trades=len(orders),
        current_weights=current_w,
        target_weights=tgt_w,
        drift_by_symbol=drift,
        orders=orders,
        warnings=warnings,
    )