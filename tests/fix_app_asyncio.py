with open('../dashboard/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Check if nest_asyncio is already at the top
if 'nest_asyncio' in lines[0] or 'nest_asyncio' in lines[1]:
    print("✅ nest_asyncio already at top")
else:
    # Add nest_asyncio at the very beginning
    new_header = '''# CRITICAL: Apply nest_asyncio FIRST to allow nested event loops (Streamlit + ib_insync)
import nest_asyncio
nest_asyncio.apply()

'''
    content = new_header + content
    
    with open('../dashboard/app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added nest_asyncio at the TOP of app.py")
