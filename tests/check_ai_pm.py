with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
print(f"Total lines: {len(lines)}")
print("\n--- First 100 lines ---")
for i, line in enumerate(lines[:100]):
    print(f"{i+1}: {line[:100]}")
