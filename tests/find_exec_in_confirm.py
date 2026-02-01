with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find execute_trade_plan in the confirm flow (second call is the manual one)
idx = content.find('if confirm:')
if idx > 0:
    # Get everything from confirm to find execute_trade_plan
    confirm_section = content[idx:idx+12000]
    exec_idx = confirm_section.find('execute_trade_plan(')
    if exec_idx > 0:
        print("execute_trade_plan call in confirm flow:")
        print("="*70)
        print(confirm_section[exec_idx-200:exec_idx+1000])
