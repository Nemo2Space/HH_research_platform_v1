with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add logging after the px_map loop and before cash projection
old_code = '''    _log.info(f"execute_trade_plan: iterating {len(snapshot.positions)} positions...")
    for p in snapshot.positions:
        sym = (p.symbol or "").strip().upper()
        if not sym:
            continue
        px = _safe_float(p.market_price)
        if px is None or px <= 0:
            px = _safe_float(p.avg_cost)
        if px is not None and px > 0:
            px_map[sym] = float(px)
    # Project cash if executing full plan using snapshot prices (best-effort).'''

new_code = '''    _log.info(f"execute_trade_plan: iterating {len(snapshot.positions)} positions...")
    for p in snapshot.positions:
        sym = (p.symbol or "").strip().upper()
        if not sym:
            continue
        px = _safe_float(p.market_price)
        if px is None or px <= 0:
            px = _safe_float(p.avg_cost)
        if px is not None and px > 0:
            px_map[sym] = float(px)
    _log.info(f"execute_trade_plan: px_map built with {len(px_map)} prices")
    _log.info(f"execute_trade_plan: projecting cash...")
    # Project cash if executing full plan using snapshot prices (best-effort).'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/execution_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added logging after px_map loop")
else:
    print("❌ Could not find code")
