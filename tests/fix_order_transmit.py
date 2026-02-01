with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find where we return ExecutionResult and add ib.sleep before it
# We need to add ib.sleep() after all orders are placed

old_return = '''    submitted = bool(placed_any) if not dry_run else False
    if dry_run:
        notes.append("Dry-run mode: no orders were sent.")

    return ExecutionResult('''

new_return = '''    submitted = bool(placed_any) if not dry_run else False
    if dry_run:
        notes.append("Dry-run mode: no orders were sent.")
    
    # CRITICAL: Allow ib_insync event loop to transmit orders to IBKR
    if placed_any and not dry_run:
        _log.info("execute_trade_plan: waiting for orders to transmit to IBKR...")
        try:
            ib.sleep(3)  # Give IBKR time to process all orders
            _log.info("execute_trade_plan: orders transmitted")
        except Exception as e:
            _log.warning(f"execute_trade_plan: ib.sleep failed: {e}")

    return ExecutionResult('''

if old_return in content:
    content = content.replace(old_return, new_return)
    with open('../dashboard/ai_pm/execution_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added ib.sleep(3) after placing orders")
else:
    print("❌ Could not find return block")
    # Try to find what's there
    idx = content.find('submitted = bool(placed_any)')
    if idx > 0:
        print("Found at:", idx)
        print(repr(content[idx:idx+300]))
