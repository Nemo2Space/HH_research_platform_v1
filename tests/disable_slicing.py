with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find max_slice_nav_pct parameter and increase it
old_param = 'max_slice_nav_pct: float = 0.02,  # slice orders above 2% NAV'
new_param = 'max_slice_nav_pct: float = 1.00,  # disabled - send full orders (no slicing)'

if old_param in content:
    content = content.replace(old_param, new_param)
    with open('../dashboard/ai_pm/execution_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Disabled order slicing (max_slice_nav_pct = 100%)")
else:
    # Find what's there
    idx = content.find('max_slice_nav_pct')
    if idx > 0:
        print(f"Found at {idx}:")
        print(repr(content[idx:idx+100]))
