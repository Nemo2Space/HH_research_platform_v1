with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find _fetch_price_map_yahoo_first function
idx = content.find('def _fetch_price_map_yahoo_first')
if idx > 0:
    print(f"Found at {idx}")
    print(content[idx:idx+2000])
