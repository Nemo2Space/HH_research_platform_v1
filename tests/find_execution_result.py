with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find where execution result is shown after orders are sent
# Look for "st.success" after execute_trade_plan
idx = content.find('execution.submitted')
if idx > 0:
    print(f"Found execution.submitted at {idx}")
    # Show context around it
    print(content[idx:idx+1500])
