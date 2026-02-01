with open('../dashboard/ai_pm/ibkr_gateway.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Show lines 152-230
print("get_account_summary function:")
print("="*70)
for i in range(151, min(230, len(lines))):
    print(f"{i+1}: {lines[i]}")
