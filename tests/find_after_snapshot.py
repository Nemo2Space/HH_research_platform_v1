with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find what happens after "Snapshot: building result..."
idx = content.find('Snapshot: building result')
if idx > 0:
    print("Code after 'building result':")
    print(content[idx:idx+1500])
