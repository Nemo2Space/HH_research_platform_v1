with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the cash projection section
old_code = '''    _log.info("execute_trade_plan: projecting cash...")
    
    # Project cash if executing full plan using snapshot prices (best-effort).
    # If price missing for an order symbol, ignore its impact in projection (conservative).
    proj_cash = float(cash)
    if nav > 0:
        for o in orders_all:'''

new_code = '''    _log.info("execute_trade_plan: projecting cash...")
    
    # Project cash if executing full plan using snapshot prices (best-effort).
    # If price missing for an order symbol, ignore its impact in projection (conservative).
    proj_cash = float(cash)
    _log.info(f"execute_trade_plan: proj_cash init={proj_cash}, nav={nav}")
    if nav > 0:
        _log.info(f"execute_trade_plan: looping {len(orders_all)} orders for cash projection...")
        for o in orders_all:'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/execution_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added logging to cash projection")
else:
    print("❌ Could not find code")
    # Find what's there
    idx = content.find('projecting cash')
    if idx > 0:
        print(f"Found at {idx}:")
        print(repr(content[idx:idx+500]))
