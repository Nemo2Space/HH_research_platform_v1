with open('../dashboard/portfolio_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

print("Searching for ticker filtering logic...")
print("="*60)

for i, line in enumerate(lines):
    if 'tickers_include' in line or 'whitelist' in line or 'restrict_to_tickers' in line:
        # Show context (5 lines before and after)
        start = max(0, i-2)
        end = min(len(lines), i+3)
        print(f"\n--- Lines {start+1}-{end+1} ---")
        for j in range(start, end):
            marker = ">>>" if j == i else "   "
            print(f"{marker} {j+1}: {lines[j][:100]}")
