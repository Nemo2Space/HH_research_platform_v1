with open('../dashboard/ai_pm/models.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

for i, line in enumerate(lines):
    if 'class TargetWeights' in line or 'TargetWeights' in line and 'dataclass' in lines[max(0,i-1)]:
        print(f"\n--- Line {i+1} context ---")
        for j in range(max(0, i-2), min(len(lines), i+20)):
            marker = ">>>" if j == i else "   "
            print(f"{marker} {j+1}: {lines[j]}")
        break
