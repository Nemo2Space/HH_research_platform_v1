with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix: Use passed price_map if available instead of building px_map from scratch
old_code = '''    _log.info("execute_trade_plan: building px_map from snapshot...")
    # Build a price map from snapshot (used only to decide sell-first when auto+armed)
    px_map: Dict[str, float] = {}
    _log.info(f"execute_trade_plan: iterating {len(snapshot.positions)} positions...")
    for p in snapshot.positions:
        sym = (p.symbol or "").strip().upper()
        if not sym:
            continue
        px = _safe_float(p.market_price)
        if px is None or px <= 0:
            px = _safe_float(p.avg_cost)
        if px is not None and px > 0:
            px_map[sym] = float(px)
    _log.info(f"execute_trade_plan: px_map built with {len(px_map)} prices")'''

new_code = '''    _log.info("execute_trade_plan: building px_map...")
    # Use passed price_map if available, otherwise build from snapshot
    if price_map:
        _log.info(f"execute_trade_plan: using passed price_map with {len(price_map)} prices")
        px_map: Dict[str, float] = dict(price_map)
    else:
        _log.info(f"execute_trade_plan: building px_map from snapshot ({len(snapshot.positions)} positions)...")
        px_map = {}
        for p in snapshot.positions:
            sym = (p.symbol or "").strip().upper()
            if not sym:
                continue
            px = _safe_float(p.market_price)
            if px is None or px <= 0:
                px = _safe_float(p.avg_cost)
            if px is not None and px > 0:
                px_map[sym] = float(px)
        _log.info(f"execute_trade_plan: px_map built with {len(px_map)} prices")'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/execution_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed: Now uses passed price_map instead of rebuilding")
else:
    print("❌ Could not find exact code block")
    # Show what's there
    idx = content.find('building px_map')
    if idx > 0:
        print(f"Found at {idx}:")
        print(repr(content[idx:idx+600]))
