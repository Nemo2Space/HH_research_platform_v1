with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the execute_trade_plan call
idx = content.find('execute_trade_plan(')
if idx > 0:
    print("execute_trade_plan call found at:", idx)
    print("\nContext:")
    print(content[idx-100:idx+800])
