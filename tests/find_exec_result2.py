with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find where execution result is handled
idx = content.find('execute_trade_plan(')
if idx > 0:
    # Find the second occurrence (manual confirm)
    idx2 = content.find('execute_trade_plan(', idx + 100)
    if idx2 > 0:
        print(f"Found second execute_trade_plan at {idx2}")
        # Show what comes after - the result handling
        print(content[idx2:idx2+2500])
