with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find where px_map is built and should use passed price_map
print("Searching for px_map usage...")
print("="*70)

# Find the px_map building section
idx = content.find('px_map: Dict[str, float] = {}')
if idx > 0:
    print(f"\nFound px_map initialization at {idx}:")
    print(content[idx-100:idx+800])
