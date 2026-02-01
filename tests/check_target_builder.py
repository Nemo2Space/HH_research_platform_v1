with open('../dashboard/ai_pm/target_builder.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
print(f"Total lines: {len(lines)}")
print("\n--- First 80 lines ---")
for i, line in enumerate(lines[:80]):
    print(f"{i+1}: {line[:100]}")
