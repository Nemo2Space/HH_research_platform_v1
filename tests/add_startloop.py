with open('../dashboard/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Check if util.startLoop() is already there
if 'util.startLoop()' in content:
    print("util.startLoop() already present")
else:
    # Find where nest_asyncio is applied
    idx = content.find('nest_asyncio.apply()')
    if idx > 0:
        print(f"Found nest_asyncio.apply() at {idx}")
        # Add util.startLoop() right after
        old_code = 'nest_asyncio.apply()'
        new_code = '''nest_asyncio.apply()

# CRITICAL: Start ib_insync's asyncio event loop in background thread
# This allows qualifyContracts, placeOrder etc. to work in threaded environments
from ib_insync import util
util.startLoop()'''
        
        content = content.replace(old_code, new_code)
        with open('../dashboard/app.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Added util.startLoop() to dashboard/app.py")
    else:
        print("❌ Could not find nest_asyncio.apply()")
