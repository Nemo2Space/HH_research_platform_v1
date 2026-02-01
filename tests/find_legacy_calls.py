with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find calls to _fetch_ibkr_price_map_legacy
import re
matches = list(re.finditer(r'_fetch_ibkr_price_map_legacy', content))
print(f"_fetch_ibkr_price_map_legacy: {len(matches)} occurrences")
for m in matches:
    print(f"  At {m.start()}: {content[m.start()-20:m.start()+80]}")
