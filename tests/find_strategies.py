with open('../dashboard/portfolio_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

print("Searching for strategies and templates...")
for i, line in enumerate(lines):
    if 'TEMPLATE' in line or 'STRATEGY' in line or 'strategy' in line.lower() and ('=' in line or 'class' in line or 'def' in line):
        print(f"Line {i+1}: {line[:100]}")
