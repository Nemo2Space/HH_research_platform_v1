with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add sys.stdout.flush after every log and more granular logging in the cash projection loop
old_code = '''    _log.info(f"execute_trade_plan: looping {len(orders_all)} orders for cash projection...")
        for o in orders_all:'''

new_code = '''    _log.info(f"execute_trade_plan: looping {len(orders_all)} orders for cash projection...")
        sys.stdout.flush()
        for i, o in enumerate(orders_all):
            if i % 10 == 0:
                _log.info(f"execute_trade_plan: cash projection order {i}/{len(orders_all)}")
                sys.stdout.flush()'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/execution_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added granular logging to cash projection loop")
else:
    print("❌ Could not find code")
    idx = content.find('looping')
    if idx > 0:
        print(f"Found at {idx}:")
        print(repr(content[idx:idx+300]))
