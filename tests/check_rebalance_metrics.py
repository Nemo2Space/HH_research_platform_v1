with open('../dashboard/ai_pm/rebalance_metrics.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
print(f"Total lines: {len(lines)}")

# Find build_position_results function
for i, line in enumerate(lines):
    if 'def build_position_results' in line:
        print(f"\n--- Line {i+1} context ---")
        for j in range(i, min(len(lines), i+60)):
            print(f"{j+1}: {lines[j]}")
        break
