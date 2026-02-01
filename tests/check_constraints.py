with open('../dashboard/ai_pm/config.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Check current cash_min
import re
match = re.search(r'cash_min: float = ([-\d.]+)', content)
if match:
    print(f"Current cash_min: {match.group(1)}")
    
    if match.group(1) != '-1.00':
        content = content.replace(match.group(0), 'cash_min: float = -1.00')
        with open('../dashboard/ai_pm/config.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Updated to cash_min: float = -1.00")
    else:
        print("✅ Already set to -1.00")
