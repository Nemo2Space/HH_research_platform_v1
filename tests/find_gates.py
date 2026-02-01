import os

# Search for gate definitions
patterns = ['cash_min', 'max_sector', 'blocked', 'GateReport', 'hard gate']

for root, dirs, files in os.walk('..'):
    dirs[:] = [d for d in dirs if d not in ['__pycache__', '.venv', 'venv', '.git']]
    
    for f in files:
        if not f.endswith('.py'):
            continue
        filepath = os.path.join(root, f)
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            
            for pattern in patterns:
                if pattern in content.lower() or pattern in content:
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if pattern in line.lower() or pattern in line:
                            print(f"{filepath}:{i+1}: {line.strip()[:80]}")
        except:
            pass
