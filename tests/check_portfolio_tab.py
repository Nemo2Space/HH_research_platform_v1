with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Show the first 100 lines to understand the structure
lines = content.split('\n')
print(f"Total lines: {len(lines)}")
print("\n--- First 80 lines ---")
for i, line in enumerate(lines[:80]):
    print(f"{i+1}: {line[:100]}")
