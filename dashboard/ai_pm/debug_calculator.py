"""
AI PM Debug Calculator
======================
Traces through EVERY calculation step to find discrepancies.

Run from your project root:
    python -m dashboard.ai_pm.debug_calculator

Or import and call in Streamlit:
    from dashboard.ai_pm.debug_calculator import run_full_debug
    run_full_debug(ib, snapshot, targets, plan, price_map)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import math


@dataclass
class SymbolDebug:
    """Debug trace for a single symbol"""
    symbol: str

    # From IBKR positions
    ibkr_qty: float = 0
    ibkr_market_value: float = 0
    ibkr_market_price: float = 0
    ibkr_avg_cost: float = 0

    # From price_map (Yahoo)
    yahoo_price: float = 0

    # Calculated values
    calc_market_value_ibkr: float = 0  # Using IBKR price
    calc_market_value_yahoo: float = 0  # Using Yahoo price

    # Weights
    weight_using_ibkr: float = 0
    weight_using_yahoo: float = 0
    weight_from_plan: float = 0  # What plan.current_weights says

    # Target
    target_weight: float = 0
    target_value: float = 0
    target_shares: int = 0

    # Delta calculation
    delta_weight: float = 0
    delta_value: float = 0
    delta_shares_exact: float = 0
    delta_shares_rounded: int = 0

    # From plan
    plan_action: str = ""
    plan_qty: int = 0
    plan_reason: str = ""

    # From TWS
    tws_pending_buy: int = 0
    tws_pending_sell: int = 0

    # Projected
    projected_qty: int = 0
    projected_value: float = 0
    projected_weight: float = 0

    # Discrepancies
    issues: List[str] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []


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


def debug_symbol(
        symbol: str,
        snapshot,
        targets,
        plan,
        price_map: Dict[str, float],
        tws_orders: List[Any] = None,
        nav_override: float = None,
) -> SymbolDebug:
    """
    Full debug trace for a single symbol.
    """
    debug = SymbolDebug(symbol=symbol)

    # === 1. GET NAV ===
    nav = nav_override or _safe_float(getattr(snapshot, 'net_liquidation', 0))
    if nav <= 0:
        debug.issues.append(f"NAV is 0 or invalid: {nav}")
        return debug

    # === 2. IBKR POSITION DATA ===
    positions = getattr(snapshot, 'positions', []) or []
    for p in positions:
        sym = (getattr(p, 'symbol', '') or '').strip().upper()
        if sym == symbol.upper():
            debug.ibkr_qty = _safe_float(getattr(p, 'quantity', 0))
            debug.ibkr_market_value = _safe_float(getattr(p, 'market_value', 0))
            debug.ibkr_market_price = _safe_float(getattr(p, 'market_price', 0))
            debug.ibkr_avg_cost = _safe_float(getattr(p, 'avg_cost', 0))
            break

    # === 3. YAHOO PRICE ===
    debug.yahoo_price = _safe_float(price_map.get(symbol.upper(), 0))

    # === 4. CALCULATE MARKET VALUES ===
    if debug.ibkr_market_price > 0:
        debug.calc_market_value_ibkr = debug.ibkr_qty * debug.ibkr_market_price
    elif debug.ibkr_avg_cost > 0:
        debug.calc_market_value_ibkr = debug.ibkr_qty * debug.ibkr_avg_cost
    else:
        debug.calc_market_value_ibkr = debug.ibkr_market_value

    if debug.yahoo_price > 0:
        debug.calc_market_value_yahoo = debug.ibkr_qty * debug.yahoo_price

    # === 5. CALCULATE WEIGHTS ===
    debug.weight_using_ibkr = (debug.calc_market_value_ibkr / nav * 100) if nav > 0 else 0
    debug.weight_using_yahoo = (debug.calc_market_value_yahoo / nav * 100) if nav > 0 else 0

    # === 6. WHAT DOES PLAN SAY? ===
    current_weights = getattr(plan, 'current_weights', {}) or {}
    debug.weight_from_plan = _safe_float(current_weights.get(symbol.upper(), 0)) * 100  # Convert to %

    # === 7. TARGET DATA ===
    target_weights = getattr(targets, 'weights', {}) or {}
    debug.target_weight = _safe_float(target_weights.get(symbol.upper(), 0)) * 100  # Convert to %

    debug.target_value = (debug.target_weight / 100) * nav

    price_for_calc = debug.yahoo_price if debug.yahoo_price > 0 else debug.ibkr_market_price
    if price_for_calc > 0:
        debug.target_shares = int(debug.target_value / price_for_calc)

    # === 8. DELTA CALCULATION ===
    debug.delta_weight = debug.target_weight - debug.weight_using_yahoo
    debug.delta_value = (debug.delta_weight / 100) * nav

    if price_for_calc > 0:
        debug.delta_shares_exact = debug.delta_value / price_for_calc
        if debug.delta_shares_exact > 0:
            debug.delta_shares_rounded = math.ceil(debug.delta_shares_exact)
        else:
            debug.delta_shares_rounded = math.floor(debug.delta_shares_exact)

    # === 9. WHAT DID PLAN GENERATE? ===
    orders = getattr(plan, 'orders', []) or []
    for o in orders:
        osym = (getattr(o, 'symbol', '') or '').strip().upper()
        if osym == symbol.upper():
            debug.plan_action = getattr(o, 'action', '')
            debug.plan_qty = int(_safe_float(getattr(o, 'quantity', 0)))
            debug.plan_reason = getattr(o, 'reason', '')
            break

    # === 10. TWS ORDERS ===
    if tws_orders:
        for trade in tws_orders:
            try:
                tsym = trade.contract.symbol.upper()
                if tsym == symbol.upper():
                    action = trade.order.action.upper()
                    qty = int(trade.order.totalQuantity)
                    if action == 'BUY':
                        debug.tws_pending_buy += qty
                    elif action == 'SELL':
                        debug.tws_pending_sell += qty
            except:
                pass

    # === 11. PROJECTED VALUES ===
    debug.projected_qty = int(debug.ibkr_qty + debug.tws_pending_buy - debug.tws_pending_sell)
    debug.projected_value = debug.projected_qty * price_for_calc if price_for_calc > 0 else 0
    debug.projected_weight = (debug.projected_value / nav * 100) if nav > 0 else 0

    # === 12. FIND ISSUES ===

    # Issue: Price mismatch
    if debug.ibkr_market_price > 0 and debug.yahoo_price > 0:
        price_diff_pct = abs(debug.ibkr_market_price - debug.yahoo_price) / debug.yahoo_price * 100
        if price_diff_pct > 2:
            debug.issues.append(
                f"Price mismatch: IBKR=${debug.ibkr_market_price:.2f} vs Yahoo=${debug.yahoo_price:.2f} "
                f"(diff={price_diff_pct:.1f}%)"
            )

    # Issue: Weight mismatch
    if abs(debug.weight_from_plan - debug.weight_using_yahoo) > 0.1:
        debug.issues.append(
            f"Weight mismatch: plan.current_weights={debug.weight_from_plan:.2f}% vs "
            f"calculated={debug.weight_using_yahoo:.2f}%"
        )

    # Issue: Order quantity mismatch
    expected_action = "BUY" if debug.delta_shares_rounded > 0 else "SELL" if debug.delta_shares_rounded < 0 else "HOLD"
    expected_qty = abs(debug.delta_shares_rounded)

    if debug.plan_action and debug.plan_action != expected_action:
        debug.issues.append(
            f"Action mismatch: plan={debug.plan_action} but should be {expected_action}"
        )

    if debug.plan_qty > 0 and abs(debug.plan_qty - expected_qty) > 5:
        debug.issues.append(
            f"Qty mismatch: plan={debug.plan_qty} but calculated={expected_qty} "
            f"(diff={debug.plan_qty - expected_qty})"
        )

    # Issue: Overshooting target
    if debug.target_weight > 0:
        overshoot = debug.projected_weight - debug.target_weight
        if abs(overshoot) > 0.3:  # More than 0.3% off
            debug.issues.append(
                f"Target overshoot: projected={debug.projected_weight:.2f}% vs "
                f"target={debug.target_weight:.2f}% (off by {overshoot:+.2f}%)"
            )

    return debug


def print_symbol_debug(debug: SymbolDebug, verbose: bool = True):
    """Print debug info for a symbol"""
    print(f"\n{'=' * 70}")
    print(f"DEBUG: {debug.symbol}")
    print(f"{'=' * 70}")

    print(f"\n📊 IBKR POSITION DATA:")
    print(f"   Quantity:      {debug.ibkr_qty}")
    print(f"   Market Value:  ${debug.ibkr_market_value:,.2f}")
    print(f"   Market Price:  ${debug.ibkr_market_price:.2f}")
    print(f"   Avg Cost:      ${debug.ibkr_avg_cost:.2f}")

    print(f"\n💰 PRICE COMPARISON:")
    print(f"   IBKR Price:    ${debug.ibkr_market_price:.2f}")
    print(f"   Yahoo Price:   ${debug.yahoo_price:.2f}")
    if debug.ibkr_market_price > 0 and debug.yahoo_price > 0:
        diff = (debug.yahoo_price - debug.ibkr_market_price) / debug.ibkr_market_price * 100
        print(f"   Difference:    {diff:+.2f}%")

    print(f"\n📐 MARKET VALUE CALCULATION:")
    print(f"   Using IBKR:    ${debug.calc_market_value_ibkr:,.2f}")
    print(f"   Using Yahoo:   ${debug.calc_market_value_yahoo:,.2f}")

    print(f"\n⚖️ WEIGHT COMPARISON:")
    print(f"   Using IBKR:    {debug.weight_using_ibkr:.2f}%")
    print(f"   Using Yahoo:   {debug.weight_using_yahoo:.2f}%")
    print(f"   From Plan:     {debug.weight_from_plan:.2f}%")
    print(f"   Target:        {debug.target_weight:.2f}%")

    print(f"\n🎯 TARGET CALCULATION:")
    print(f"   Target Value:  ${debug.target_value:,.2f}")
    print(f"   Target Shares: {debug.target_shares}")

    print(f"\n📈 DELTA CALCULATION:")
    print(f"   Delta Weight:  {debug.delta_weight:+.2f}%")
    print(f"   Delta Value:   ${debug.delta_value:+,.2f}")
    print(f"   Delta Shares (exact):   {debug.delta_shares_exact:+.2f}")
    print(f"   Delta Shares (rounded): {debug.delta_shares_rounded:+d}")

    print(f"\n📋 PLAN OUTPUT:")
    print(f"   Action:        {debug.plan_action or 'NONE'}")
    print(f"   Quantity:      {debug.plan_qty}")
    print(f"   Reason:        {debug.plan_reason}")

    print(f"\n📡 TWS ORDERS:")
    print(f"   Pending BUY:   {debug.tws_pending_buy}")
    print(f"   Pending SELL:  {debug.tws_pending_sell}")

    print(f"\n🔮 PROJECTED (After Orders):")
    print(f"   Quantity:      {debug.projected_qty}")
    print(f"   Value:         ${debug.projected_value:,.2f}")
    print(f"   Weight:        {debug.projected_weight:.2f}%")

    if debug.issues:
        print(f"\n🚨 ISSUES DETECTED:")
        for issue in debug.issues:
            print(f"   ❌ {issue}")
    else:
        print(f"\n✅ No issues detected")


def run_full_debug(
        ib,  # IB connection
        snapshot,
        targets,
        plan,
        price_map: Dict[str, float],
        symbols: List[str] = None,  # If None, debug all symbols with issues
        verbose: bool = True,
) -> Dict[str, SymbolDebug]:
    """
    Run full debug on all or specified symbols.

    Usage in Streamlit:
        from dashboard.ai_pm.debug_calculator import run_full_debug

        results = run_full_debug(
            ib=gw.ib,
            snapshot=snapshot,
            targets=targets,
            plan=plan,
            price_map=price_map,
            symbols=['AVGO', 'AMD', 'GE'],  # Or None for all
        )
    """
    # Get TWS orders
    tws_orders = []
    if ib:
        try:
            ib.reqAllOpenOrders()
            ib.sleep(2)
            tws_orders = ib.openTrades()
        except:
            pass

    # Get NAV
    nav = _safe_float(getattr(snapshot, 'net_liquidation', 0))

    print(f"\n{'#' * 70}")
    print(f"# AI PM DEBUG CALCULATOR")
    print(f"# NAV: ${nav:,.2f}")
    print(f"# Time: {datetime.now().isoformat()}")
    print(f"{'#' * 70}")

    # Determine which symbols to debug
    if symbols is None:
        # Get all symbols from positions + targets
        positions = getattr(snapshot, 'positions', []) or []
        target_weights = getattr(targets, 'weights', {}) or {}

        all_symbols = set()
        for p in positions:
            sym = (getattr(p, 'symbol', '') or '').strip().upper()
            if sym:
                all_symbols.add(sym)
        for sym in target_weights.keys():
            all_symbols.add(sym.upper())

        symbols = sorted(all_symbols)

    results = {}
    issues_found = []

    for sym in symbols:
        debug = debug_symbol(
            symbol=sym,
            snapshot=snapshot,
            targets=targets,
            plan=plan,
            price_map=price_map,
            tws_orders=tws_orders,
            nav_override=nav,
        )
        results[sym] = debug

        if debug.issues:
            issues_found.append(debug)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {len(results)} symbols analyzed, {len(issues_found)} with issues")
    print(f"{'=' * 70}")

    if issues_found:
        print(f"\n🚨 SYMBOLS WITH ISSUES:")
        for debug in issues_found:
            print(f"\n  {debug.symbol}:")
            for issue in debug.issues:
                print(f"    ❌ {issue}")

    # Print verbose details for symbols with issues
    if verbose:
        for debug in issues_found:
            print_symbol_debug(debug, verbose=True)

    return results


def debug_single_symbol_standalone(
        symbol: str,
        ib,  # IB connection (can be None)
        account: str = None,
):
    """
    Standalone debug for a single symbol - fetches everything fresh.

    Usage:
        from dashboard.ai_pm.debug_calculator import debug_single_symbol_standalone
        debug_single_symbol_standalone('AVGO', gw.ib, 'DUK415187')
    """
    import yfinance as yf

    symbol = symbol.upper()

    print(f"\n{'#' * 70}")
    print(f"# STANDALONE DEBUG: {symbol}")
    print(f"{'#' * 70}")

    # 1. Get fresh Yahoo price
    print(f"\n1️⃣ Fetching Yahoo price...")
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1d')
        yahoo_price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
        print(f"   Yahoo Price: ${yahoo_price:.2f}")
    except Exception as e:
        print(f"   ❌ Yahoo error: {e}")
        yahoo_price = 0

    # 2. Get IBKR data
    if ib and ib.isConnected():
        print(f"\n2️⃣ Fetching IBKR position...")
        positions = ib.positions()

        ibkr_qty = 0
        ibkr_price = 0
        ibkr_value = 0
        ibkr_cost = 0

        for pos in positions:
            if pos.contract.symbol.upper() == symbol:
                ibkr_qty = float(pos.position)
                ibkr_cost = float(pos.avgCost)
                print(f"   Quantity: {ibkr_qty}")
                print(f"   Avg Cost: ${ibkr_cost:.2f}")
                break
        else:
            print(f"   ⚠️ No position found for {symbol}")

        # Get live quote
        print(f"\n3️⃣ Fetching IBKR live quote...")
        try:
            from ib_insync import Stock
            contract = Stock(symbol, 'SMART', 'USD')
            ib.qualifyContracts(contract)
            ticker = ib.reqMktData(contract, '', False, False)
            ib.sleep(2)

            ibkr_price = ticker.marketPrice() if ticker.marketPrice() else 0
            if ibkr_price and ibkr_price > 0:
                print(f"   IBKR Live Price: ${ibkr_price:.2f}")
            else:
                ibkr_price = ticker.last if ticker.last else 0
                print(f"   IBKR Last Price: ${ibkr_price:.2f}")

            ib.cancelMktData(contract)
        except Exception as e:
            print(f"   ❌ IBKR quote error: {e}")

        # Get account summary
        print(f"\n4️⃣ Fetching account NAV...")
        try:
            account_values = ib.accountSummary(account)
            nav = 0
            for av in account_values:
                if av.tag == 'NetLiquidation':
                    nav = float(av.value)
                    break
            print(f"   NAV: ${nav:,.2f}")
        except Exception as e:
            print(f"   ❌ Account error: {e}")
            nav = 0

        # Get open orders
        print(f"\n5️⃣ Fetching TWS open orders...")
        try:
            ib.reqAllOpenOrders()
            ib.sleep(2)
            trades = ib.openTrades()

            pending_buy = 0
            pending_sell = 0
            for trade in trades:
                if trade.contract.symbol.upper() == symbol:
                    action = trade.order.action.upper()
                    qty = int(trade.order.totalQuantity)
                    status = trade.orderStatus.status
                    print(f"   Order: {action} {qty} @ {status}")
                    if action == 'BUY':
                        pending_buy += qty
                    elif action == 'SELL':
                        pending_sell += qty

            if pending_buy == 0 and pending_sell == 0:
                print(f"   No pending orders for {symbol}")
            else:
                print(f"   Total Pending BUY: {pending_buy}")
                print(f"   Total Pending SELL: {pending_sell}")
        except Exception as e:
            print(f"   ❌ Orders error: {e}")
            pending_buy = 0
            pending_sell = 0

        # Calculate
        print(f"\n{'=' * 70}")
        print(f"CALCULATIONS:")
        print(f"{'=' * 70}")

        price = yahoo_price if yahoo_price > 0 else ibkr_price

        current_value = ibkr_qty * price
        current_weight = (current_value / nav * 100) if nav > 0 else 0

        projected_qty = ibkr_qty + pending_buy - pending_sell
        projected_value = projected_qty * price
        projected_weight = (projected_value / nav * 100) if nav > 0 else 0

        print(f"\n   Using Price: ${price:.2f}")
        print(f"\n   CURRENT:")
        print(f"      Qty:    {ibkr_qty:.0f}")
        print(f"      Value:  ${current_value:,.2f}")
        print(f"      Weight: {current_weight:.2f}%")

        print(f"\n   PROJECTED (after orders):")
        print(f"      Qty:    {projected_qty:.0f}")
        print(f"      Value:  ${projected_value:,.2f}")
        print(f"      Weight: {projected_weight:.2f}%")

        print(f"\n   COMPARISON:")
        print(f"      Yahoo Price:  ${yahoo_price:.2f}")
        print(f"      IBKR Price:   ${ibkr_price:.2f}")
        if ibkr_price > 0:
            price_diff = (yahoo_price - ibkr_price) / ibkr_price * 100
            print(f"      Difference:   {price_diff:+.2f}%")
    else:
        print(f"\n❌ IBKR not connected - cannot fetch live data")


# Streamlit UI helper
def render_debug_ui(st, gw, snapshot, targets, plan, price_map):
    """
    Render debug UI in Streamlit.

    Add to ui_tab.py:
        from .debug_calculator import render_debug_ui
        render_debug_ui(st, gw, snapshot, targets, plan, price_map)
    """
    st.markdown("### 🔍 Debug Calculator")

    symbol_input = st.text_input("Symbol to debug:", value="AVGO", key="debug_symbol_input")

    if st.button("🔬 Run Debug", key="run_debug_btn"):
        if symbol_input:
            with st.spinner(f"Debugging {symbol_input}..."):
                tws_orders = []
                if gw and gw.is_connected():
                    try:
                        gw.ib.reqAllOpenOrders()
                        gw.ib.sleep(2)
                        tws_orders = gw.ib.openTrades()
                    except:
                        pass

                debug = debug_symbol(
                    symbol=symbol_input.upper(),
                    snapshot=snapshot,
                    targets=targets,
                    plan=plan,
                    price_map=price_map,
                    tws_orders=tws_orders,
                )

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**📊 Position Data**")
                    st.write(f"IBKR Qty: {debug.ibkr_qty}")
                    st.write(f"IBKR Price: ${debug.ibkr_market_price:.2f}")
                    st.write(f"Yahoo Price: ${debug.yahoo_price:.2f}")

                    st.markdown("**⚖️ Weights**")
                    st.write(f"Current (Yahoo): {debug.weight_using_yahoo:.2f}%")
                    st.write(f"Current (Plan): {debug.weight_from_plan:.2f}%")
                    st.write(f"Target: {debug.target_weight:.2f}%")

                with col2:
                    st.markdown("**📈 Delta Calculation**")
                    st.write(f"Delta Weight: {debug.delta_weight:+.2f}%")
                    st.write(f"Delta Shares (exact): {debug.delta_shares_exact:+.2f}")
                    st.write(f"Delta Shares (rounded): {debug.delta_shares_rounded:+d}")

                    st.markdown("**📋 Plan Generated**")
                    st.write(f"Action: {debug.plan_action or 'NONE'}")
                    st.write(f"Quantity: {debug.plan_qty}")

                st.markdown("**🔮 Projected**")
                st.write(f"Projected Qty: {debug.projected_qty}")
                st.write(f"Projected Weight: {debug.projected_weight:.2f}%")

                if debug.issues:
                    st.error("🚨 Issues Found:")
                    for issue in debug.issues:
                        st.write(f"❌ {issue}")
                else:
                    st.success("✅ No issues detected")


if __name__ == "__main__":
    print("Debug Calculator Module")
    print("=" * 50)
    print("\nUsage in Python/Streamlit:")
    print("""
    from dashboard.ai_pm.debug_calculator import run_full_debug, debug_single_symbol_standalone

    # Full debug with existing objects:
    results = run_full_debug(
        ib=gw.ib,
        snapshot=snapshot,
        targets=targets,
        plan=plan,
        price_map=price_map,
        symbols=['AVGO', 'AMD', 'GE'],
    )

    # Standalone debug (fetches everything fresh):
    debug_single_symbol_standalone('AVGO', gw.ib, 'DUK415187')
    """)