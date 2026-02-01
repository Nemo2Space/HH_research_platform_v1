print("="*70)
print("GATE CONSTRAINTS CONFIGURATION")
print("="*70)

with open('../dashboard/ai_pm/config.py', 'r', encoding='utf-8') as f:
    content = f.read()
print(content)
