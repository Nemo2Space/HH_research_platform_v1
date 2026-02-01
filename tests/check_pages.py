import os

# Check if the file exists
if os.path.exists('../dashboard/pages/manual_portfolio.py'):
    print("✅ File exists at dashboard/pages/manual_portfolio.py")
else:
    print("❌ File NOT found at dashboard/pages/manual_portfolio.py")

# Check dashboard structure
print("\nDashboard structure:")
for root, dirs, files in os.walk('../dashboard'):
    level = root.replace('dashboard', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        if file.endswith('.py'):
            print(f"{subindent}{file}")
