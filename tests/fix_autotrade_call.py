with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_call = '''execute_trade_plan(
                        ib=gw.ib,
                        snapshot=snapshot,
                        plan=plan,
                        account=sel_account,
                        constraints=DEFAULT_CONSTRAINTS,
                        dry_run=bool(dry_run),
                        kill_switch=is_kill_switch(),
                        auto_trade_enabled=True,
                        armed=True,
                    )'''

new_call = '''execute_trade_plan(
                        ib=gw.ib,
                        snapshot=snapshot,
                        plan=plan,
                        account=sel_account,
                        constraints=DEFAULT_CONSTRAINTS,
                        dry_run=bool(dry_run),
                        kill_switch=is_kill_switch(),
                        auto_trade_enabled=True,
                        armed=True,
                        price_map=price_map,  # Use pre-fetched Yahoo prices
                        skip_live_quotes=True,  # Skip slow IBKR reqMktData
                    )'''

if old_call in content:
    content = content.replace(old_call, new_call)
    with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed auto-trade execute_trade_plan call")
else:
    print("❌ Could not find auto-trade call")
