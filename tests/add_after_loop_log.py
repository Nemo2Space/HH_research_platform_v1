with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add logging after proj_cash_w calculation
old_code = '''        proj_cash_w = proj_cash / float(nav)
    else:
        proj_cash_w = None
    # Cash policy'''

new_code = '''        proj_cash_w = proj_cash / float(nav)
        _log.info(f"execute_trade_plan: proj_cash_w={proj_cash_w:.4f}")
        sys.stdout.flush()
    else:
        proj_cash_w = None
        _log.info("execute_trade_plan: proj_cash_w=None (nav<=0)")
        sys.stdout.flush()
    
    _log.info(f"execute_trade_plan: checking cash policy... cash_min={constraints.cash_min}")
    sys.stdout.flush()
    # Cash policy'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/execution_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added logging after proj_cash_w")
else:
    print("❌ Could not find code")
    idx = content.find('proj_cash_w = proj_cash')
    if idx > 0:
        print(f"Found at {idx}:")
        print(repr(content[idx:idx+200]))
