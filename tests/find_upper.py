with open('../dashboard/ai_pm/portfolio_intelligence.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Find where .upper() is called
for i, line in enumerate(lines):
    if '.upper()' in line:
        print(f"\nLine {i+1}: {line[:100]}")
        for j in range(max(0, i-3), min(len(lines), i+3)):
            marker = ">>>" if j == i else "   "
            print(f"{marker} {j+1}: {lines[j][:100]}")
