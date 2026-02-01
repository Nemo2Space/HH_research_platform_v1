with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find where it fetches live quotes
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'quote' in line.lower() or 'reqMktData' in line or 'ticker' in line.lower() or 'bid' in line.lower() or 'ask' in line.lower():
        print(f"{i+1}: {line.strip()[:90]}")
