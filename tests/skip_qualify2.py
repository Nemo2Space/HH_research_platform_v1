with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the entire qualifyContracts try block
old_code = '''    # qualify - use asyncio event loop to avoid thread deadlock
    _log.info(f"execute_trade_plan: built {len(contract_by_symbol)} contracts, now qualifying...")
    sys.stdout.flush()
    try:
        _log.info("execute_trade_plan: calling ib.qualifyContracts()...")
        sys.stdout.flush()
        
        # Handle threading: ensure event loop exists
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            _log.info("execute_trade_plan: creating new event loop for thread")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run qualifyContracts
        qualified = ib.qualifyContracts(*list(contract_by_symbol.values()))
        _log.info(f"execute_trade_plan: qualifyContracts returned {len(qualified)} contracts")
        sys.stdout.flush()'''

new_code = '''    # Skip qualifyContracts - it deadlocks in threaded environments (Streamlit)
    # IBKR accepts basic Stock() contracts for order placement without qualification
    _log.info(f"execute_trade_plan: built {len(contract_by_symbol)} contracts")
    sys.stdout.flush()
    
    try:
        _log.info("execute_trade_plan: skipping qualifyContracts (causes thread deadlock)")
        sys.stdout.flush()
        qualified = list(contract_by_symbol.values())
        _log.info(f"execute_trade_plan: {len(qualified)} contracts ready for order placement")
        sys.stdout.flush()'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/execution_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Replaced qualifyContracts with skip")
else:
    print("❌ Could not find exact code block")
    # Show what's there
    idx = content.find('qualify - use asyncio')
    if idx > 0:
        print(f"Found at {idx}:")
        print(repr(content[idx:idx+800]))
