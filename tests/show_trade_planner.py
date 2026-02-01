with open('../dashboard/ai_pm/trade_planner.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Show the first 100 lines to understand the structure
print("First 100 lines:")
for i in range(min(100, len(lines))):
    print(f"{i+1}: {lines[i]}")
