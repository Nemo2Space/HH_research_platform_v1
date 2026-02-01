with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Use exact text from repr
old_code = 'orders = orders_all\n\n    # Build contracts'

new_code = '''orders = orders_all

    _log.info(f"execute_trade_plan: cash policy passed, orders count={len(orders)}")
    sys.stdout.flush()
    
    # Build contracts
    _log.info("execute_trade_plan: building contracts...")
    sys.stdout.flush()'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/execution_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added logging before contract building")
else:
    print("❌ Could not find code")
