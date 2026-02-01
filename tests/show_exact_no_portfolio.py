with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Show exact lines 2455-2480
print("Exact lines 2455-2480:")
for i in range(2454, 2480):
    # Print with repr to see exact characters
    print(f"{i+1}: {repr(lines[i][:80])}")
