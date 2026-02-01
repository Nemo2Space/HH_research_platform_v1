with open('../dashboard/portfolio_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Check if our fix is there
if 'Skipping theme filter - user specified' in content:
    print("✅ Theme filter fix IS in the code")
else:
    print("❌ Theme filter fix NOT found!")

# Find all filtering operations
lines = content.split('\n')
print("\n--- ALL FILTER OPERATIONS ---")
for i, line in enumerate(lines):
    if '= df[df[' in line or 'df = df[' in line:
        print(f"Line {i+1}: {line.strip()[:100]}")
