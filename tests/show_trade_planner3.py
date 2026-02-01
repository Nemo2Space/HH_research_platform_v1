with open('../dashboard/ai_pm/trade_planner.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Show lines 200-286
print("Lines 200-286:")
for i in range(199, min(286, len(lines))):
    print(f"{i+1}: {lines[i]}")
