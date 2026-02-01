with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find what happens after "Step 4/5: Fetching prices"
idx = content.find('Step 4/5: Fetching prices')
if idx > 0:
    print(f"Found at {idx}")
    print(content[idx:idx+3000])
