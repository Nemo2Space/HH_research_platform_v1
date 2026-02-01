with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the execute_trade_plan call in the confirm section and add price_map
old_execute = '''                    execution = execute_trade_plan(
                        ib=gw.ib,
                        snapshot=snapshot,
                        plan=plan,
                        account=sel_account,
                        constraints=DEFAULT_CONSTRAINTS,
                        dry_run=False,
                        kill_switch=is_kill_switch(),
                        auto_trade_enabled=False,
                        armed=is_armed(),
                    )'''

new_execute = '''                    execution = execute_trade_plan(
                        ib=gw.ib,
                        snapshot=snapshot,
                        plan=plan,
                        account=sel_account,
                        constraints=DEFAULT_CONSTRAINTS,
                        dry_run=False,
                        kill_switch=is_kill_switch(),
                        auto_trade_enabled=False,
                        armed=is_armed(),
                        price_map=price_map,  # Use pre-fetched Yahoo prices
                        skip_live_quotes=True,  # Skip slow IBKR reqMktData
                    )'''

if old_execute in content:
    content = content.replace(old_execute, new_execute)
    print("✅ Updated execute_trade_plan call in confirm section")
else:
    print("❌ Could not find execute_trade_plan in confirm section")

with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
    f.write(content)
