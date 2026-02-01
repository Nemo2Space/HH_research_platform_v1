with open('../dashboard/ai_pm/ibkr_gateway.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Check if nest_asyncio is being used
if 'nest_asyncio' in content:
    print("✅ nest_asyncio is imported")
else:
    print("❌ nest_asyncio is NOT imported - THIS IS THE PROBLEM")

# Show the imports
lines = content.split('\n')
for i, line in enumerate(lines[:30]):
    print(f"{i+1}: {line}")
