with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find IBKR fallback in Run Now section
idx = content.find('_fetch_ibkr_price_map')
count = 0
while idx > 0:
    count += 1
    print(f"\n=== Occurrence {count} at {idx} ===")
    print(content[idx-100:idx+300])
    idx = content.find('_fetch_ibkr_price_map', idx + 1)

print(f"\nTotal occurrences: {count}")
