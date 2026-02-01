import sys
sys.path.insert(0, '..')

# Check TradePlan definition
with open('../dashboard/ai_pm/models.py', 'r') as f:
    content = f.read()

idx = content.find('class TradePlan')
if idx > 0:
    print("TradePlan definition:")
    print(content[idx:idx+800])
