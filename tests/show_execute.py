with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Show execute_trade_plan function
for i, line in enumerate(lines):
    if 'def execute_trade_plan' in line:
        print(f"Found at line {i+1}:")
        for j in range(i, min(i+80, len(lines))):
            print(f"{j+1}: {lines[j]}")
        break
