with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find how Run Now is triggered and what it calls
lines = content.split('\n')

print("Searching for Run Now flow...")
print("="*60)

# Find run_now button and the subsequent code
in_run_now = False
for i, line in enumerate(lines):
    if 'run_now' in line.lower() and ('button' in line.lower() or 'if run_now' in line.lower()):
        in_run_now = True
    
    if in_run_now:
        print(f"{i+1}: {line[:100]}")
        if i > 3440:  # Stop after a while
            break
