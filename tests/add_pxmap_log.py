with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the px_map building section and add logging
old_code = '''    _log.info("execute_trade_plan: building px_map from snapshot...")
    # Build a price map from snapshot (used only to decide sell-first when auto+armed)
    px_map: Dict[str, float] = {}
    for p in snapshot.positions:'''

new_code = '''    _log.info("execute_trade_plan: building px_map from snapshot...")
    # Build a price map from snapshot (used only to decide sell-first when auto+armed)
    px_map: Dict[str, float] = {}
    _log.info(f"execute_trade_plan: iterating {len(snapshot.positions)} positions...")
    for p in snapshot.positions:'''

if old_code in content:
    content = content.replace(old_code, new_code)
    print("✅ Added logging after px_map init")
else:
    print("❌ Could not find px_map section")

# Find what comes after the px_map loop
idx = content.find('execute_trade_plan: building px_map from snapshot')
if idx > 0:
    print("\nCode after px_map logging:")
    print(content[idx:idx+1500])

with open('../dashboard/ai_pm/execution_engine.py', 'w', encoding='utf-8') as f:
    f.write(content)
