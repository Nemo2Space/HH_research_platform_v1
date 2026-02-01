with open('../dashboard/ai_pm/trade_planner.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Show lines 100-200
print("Lines 100-200:")
for i in range(99, min(200, len(lines))):
    print(f"{i+1}: {lines[i]}")
