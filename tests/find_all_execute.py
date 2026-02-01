with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all execute_trade_plan calls
import re
matches = list(re.finditer(r'execute_trade_plan\(', content))
print(f"Found {len(matches)} execute_trade_plan calls:\n")

for i, m in enumerate(matches):
    start = m.start()
    # Find the closing parenthesis
    depth = 0
    end = start
    for j, c in enumerate(content[start:start+1000]):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                end = start + j + 1
                break
    
    print(f"Call #{i+1} at position {start}:")
    print(content[start:end])
    print("\n" + "-"*50 + "\n")
