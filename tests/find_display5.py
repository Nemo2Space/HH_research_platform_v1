with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

print("Context around enhanced display (lines 2660-2695):")
for i in range(2659, 2695):
    print(f"{i+1}: {lines[i]}")
