with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace using smaller chunks
old1 = '''    _log.info("execute_trade_plan: building px_map from snapshot...")
    # Build a price map from snapshot (used only to decide sell-first when auto+armed)
    px_map: Dict[str, float] = {}
    _log.info(f"execute_trade_plan: iterating {len(snapshot.positions)} positions...")'''

new1 = '''    _log.info("execute_trade_plan: building px_map...")
    # Use passed price_map if available, otherwise build from snapshot
    if price_map:
        _log.info(f"execute_trade_plan: using passed price_map with {len(price_map)} prices")
        px_map: Dict[str, float] = dict(price_map)
    else:
        _log.info(f"execute_trade_plan: building px_map from snapshot ({len(snapshot.positions)} positions)...")
        px_map: Dict[str, float] = {}'''

if old1 in content:
    content = content.replace(old1, new1)
    
    # Now fix the loop indentation - add extra indent for else block
    old_loop = '''    for p in snapshot.positions:
        sym = (p.symbol or "").strip().upper()
        if not sym:
            continue
        px = _safe_float(p.market_price)
        if px is None or px <= 0:
            px = _safe_float(p.avg_cost)
        if px is not None and px > 0:
            px_map[sym] = float(px)

    _log.info(f"execute_trade_plan: px_map built with {len(px_map)} prices")'''
    
    new_loop = '''        for p in snapshot.positions:
            sym = (p.symbol or "").strip().upper()
            if not sym:
                continue
            px = _safe_float(p.market_price)
            if px is None or px <= 0:
                px = _safe_float(p.avg_cost)
            if px is not None and px > 0:
                px_map[sym] = float(px)
        _log.info(f"execute_trade_plan: px_map built from snapshot with {len(px_map)} prices")'''
    
    if old_loop in content:
        content = content.replace(old_loop, new_loop)
        with open('../dashboard/ai_pm/execution_engine.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Fixed: Now uses passed price_map")
    else:
        print("❌ Could not find loop section")
else:
    print("❌ Could not find first section")
