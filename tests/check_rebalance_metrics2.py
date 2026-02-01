with open('../dashboard/ai_pm/rebalance_metrics.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Show lines 396-450
print("Lines 396-450:")
for i in range(395, 450):
    print(f"{i+1}: {lines[i]}")
