with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add logging before qualifyContracts
old_code = '''    # qualify
    try:
        qualified = ib.qualifyContracts(*list(contract_by_symbol.values()))'''

new_code = '''    # qualify
    _log.info(f"execute_trade_plan: built {len(contract_by_symbol)} contracts, now qualifying...")
    sys.stdout.flush()
    try:
        _log.info("execute_trade_plan: calling ib.qualifyContracts()...")
        sys.stdout.flush()
        qualified = ib.qualifyContracts(*list(contract_by_symbol.values()))
        _log.info(f"execute_trade_plan: qualifyContracts returned {len(qualified)} contracts")
        sys.stdout.flush()'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/execution_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added logging around qualifyContracts")
else:
    print("❌ Could not find code")
    idx = content.find('qualifyContracts')
    if idx > 0:
        print(f"Found at {idx}:")
        print(repr(content[idx-50:idx+100]))
