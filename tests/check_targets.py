with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Find where targets are built and used
print("Searching for build_target_weights and target usage...")
for i, line in enumerate(lines):
    if 'build_target_weights' in line or 'targets.weights' in line:
        print(f"\nLine {i+1}: {line[:120]}")
