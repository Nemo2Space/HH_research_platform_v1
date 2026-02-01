with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_code = '''    _log.info(f"execute_trade_plan: cash={cash}")
    if cash is None:
        cash = 0.0
        notes.append("Cash missing from snapshot; assuming 0 for execution policy decisions (may be inaccurate).")

    max_slice_notional = float(nav) * float(max_slice_nav_pct) if nav > 0 else float("inf")

    orders_all = plan.orders or []
    if not orders_all:
        return ExecutionResult(ts_utc=ts, account=account, strategy_key=plan.strategy_key,
                               submitted=False, submitted_orders=[],
                               errors=[], notes=["No orders to execute."])

    # Build a price map from snapshot (used only to decide sell-first when auto+armed)
    px_map: Dict[str, float] = {}
    for p in snapshot.positions:'''

new_code = '''    _log.info(f"execute_trade_plan: cash={cash}")
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

    _log.info("execute_trade_plan: building px_map from snapshot...")
    # Build a price map from snapshot (used only to decide sell-first when auto+armed)
    px_map: Dict[str, float] = {}
    for p in snapshot.positions:'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/execution_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added logging after cash")
else:
    print("❌ Could not find code block")
