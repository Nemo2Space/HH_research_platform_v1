with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find calls to _ticker_price
import re
matches = list(re.finditer(r'_ticker_price\(', content))
print(f"_ticker_price calls: {len(matches)}")
for m in matches:
    # Skip function definition
    if content[m.start()-4:m.start()] == 'def ':
        continue
    print(f"  At {m.start()}: {content[m.start()-30:m.start()+80]}")
