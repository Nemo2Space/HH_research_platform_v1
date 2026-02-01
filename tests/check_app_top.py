with open('../dashboard/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
print("First 25 lines of dashboard/app.py:")
print("="*70)
for i, line in enumerate(lines[:25]):
    print(f"{i+1}: {line}")
