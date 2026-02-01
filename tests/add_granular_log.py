with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the start of execute_trade_plan and add detailed logging
old_start = '''    import logging
    import sys
    _log = logging.getLogger(__name__)
    _log.info(f"execute_trade_plan: START dry_run={dry_run}, orders={len(plan.orders) if plan and plan.orders else 0}")
    print(f"DEBUG execute_trade_plan: START dry_run={dry_run}", flush=True)
    sys.stdout.flush()
    
    ts = datetime.utcnow()
    errors: List[str] = []
    notes: List[str] = []
    submitted_orders: List[Dict[str, Any]] = []

    if not ib.isConnected():
        return ExecutionResult(ts_utc=ts, account=account, strategy_key=plan.strategy_key,
                               submitted=False, submitted_orders=[],
                               errors=["IBKR not connected"], notes=[])

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

    nav = _safe_float(plan.nav) or _safe_float(snapshot.net_liquidation) or 0.0
    cash = _safe_float(snapshot.total_cash)'''

new_start = '''    import logging
    import sys
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
    _log.info(f"execute_trade_plan: cash={cash}")'''

if old_start in content:
    content = content.replace(old_start, new_start)
    with open('../dashboard/ai_pm/execution_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added detailed logging inside execute_trade_plan")
else:
    print("❌ Could not find start block - checking what's there")
    idx = content.find('execute_trade_plan: START')
    if idx > 0:
        print(repr(content[idx:idx+800]))
