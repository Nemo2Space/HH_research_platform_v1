with open('../dashboard/portfolio_builder.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Find save_portfolio and load_portfolio functions
print("Searching for save/load portfolio functions...")
for i, line in enumerate(lines):
    if 'def save_portfolio' in line or 'def load_portfolio' in line or 'def get_saved_portfolios' in line:
        print(f"\nLine {i+1}: {line}")
        for j in range(i, min(len(lines), i+30)):
            print(f"{j+1}: {lines[j][:100]}")
        print("...")
