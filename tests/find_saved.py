with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the Saved Portfolios section and add JSON import before it
# Looking for the saved portfolios rendering area

# First, let me find the exact location
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'Saved Portfolios' in line or 'saved_portfolios' in line.lower():
        print(f"Line {i+1}: {line[:100]}")
