with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all remaining IBKR-related calls that could freeze
# Look for ib.qualify, ib.reqTickers, ib.reqMktData
import re

patterns = [
    r'ib\.qualifyContracts',
    r'ib\.reqTickers', 
    r'ib\.reqMktData',
    r'gw\.ib\.qualifyContracts',
    r'gw\.ib\.reqTickers',
    r'gw\.ib\.reqMktData',
]

for pat in patterns:
    matches = list(re.finditer(pat, content))
    if matches:
        print(f"\n{pat}: {len(matches)} occurrences")
        for m in matches:
            print(f"  At {m.start()}: {content[m.start()-20:m.start()+50]}")
