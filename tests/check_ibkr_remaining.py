with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Check if there's still an IBKR fallback in confirm section
idx = content.find('_fetch_ibkr_price_map')
if idx > 0:
    print(f"Still found _fetch_ibkr_price_map at {idx}")
    print(content[idx-200:idx+400])
else:
    print("✅ No more _fetch_ibkr_price_map calls")
