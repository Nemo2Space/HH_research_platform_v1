with open('../dashboard/ai_pm/trade_planner.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
print(f"Total lines: {len(lines)}")

# Find where orders are created
for i, line in enumerate(lines):
    if 'quantity' in line.lower() and ('order' in line.lower() or 'OrderTicket' in line):
        print(f"Line {i+1}: {line[:100]}")
