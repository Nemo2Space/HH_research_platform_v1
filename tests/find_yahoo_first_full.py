with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the rest of _fetch_price_map_yahoo_first - look for IBKR fallback
idx = content.find('def _fetch_price_map_yahoo_first')
if idx > 0:
    # Get the full function
    func_end = content.find('\ndef ', idx + 100)
    if func_end > 0:
        print(content[idx:func_end])
