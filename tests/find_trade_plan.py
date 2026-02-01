with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the trade plan building section
idx = content.find('Building trade plan')
if idx > 0:
    print(f"Found at {idx}")
    print(content[idx:idx+1500])
