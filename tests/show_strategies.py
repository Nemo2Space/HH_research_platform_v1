with open('../dashboard/portfolio_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

print("=== InvestmentStrategy Enum ===")
in_enum = False
for i, line in enumerate(lines):
    if 'class InvestmentStrategy' in line:
        in_enum = True
    if in_enum:
        print(f"{i+1}: {line}")
        if line.strip() == '' and i > 40:
            break

print("\n=== STRATEGY_SCORING_MODELS Keys ===")
for i, line in enumerate(lines):
    if 'STRATEGY_SCORING_MODELS = {' in line:
        for j in range(i, min(len(lines), i+250)):
            print(f"{j+1}: {lines[j][:100]}")
            if lines[j].strip() == '}' and j > i + 10:
                break
        break
