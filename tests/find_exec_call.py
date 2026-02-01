with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the execute_trade_plan call
idx = content.find('about to call execute_trade_plan')
if idx > 0:
    print("Found at index", idx)
    print(repr(content[idx:idx+400]))
else:
    print("Not found")
