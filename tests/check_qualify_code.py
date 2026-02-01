with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the qualifyContracts section completely
old_code = '''    # Skip qualifyContracts to avoid thread deadlock - IBKR accepts basic Stock() contracts
    _log.info(f"execute_trade_plan: built {len(contract_by_symbol)} contracts")
    sys.stdout.flush()
    _log.info("execute_trade_plan: skipping qualifyContracts (not needed for basic stocks)")
    sys.stdout.flush()
    
    try:
        # No qualification needed - proceed directly to order placement
        qualified = list(contract_by_symbol.values())
        _log.info(f"execute_trade_plan: {len(qualified)} contracts ready")
        sys.stdout.flush()'''

if old_code in content:
    print("Already has skip code, checking what comes next...")
    idx = content.find(old_code)
    print(content[idx:idx+len(old_code)+500])
else:
    # Find the current qualifyContracts code
    idx = content.find('calling ib.qualifyContracts()')
    if idx > 0:
        print(f"Found qualifyContracts at {idx}")
        print(content[idx-200:idx+600])
