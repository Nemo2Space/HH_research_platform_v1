"""
Diagnostic script - Check why src.ml import fails
Run from project root: python check_imports.py
"""

import os
import sys

print("=" * 60)
print("Python Import Diagnostic")
print("=" * 60)

# 1. Check current directory
print(f"\n1. Current directory: {os.getcwd()}")

# 2. Check Python path
print(f"\n2. Python path:")
for p in sys.path[:5]:
    print(f"   - {p}")

# 3. Check if src folder exists
print(f"\n3. Checking folders:")
folders_to_check = ['src', 'src/ml', 'src\\ml']
for folder in folders_to_check:
    exists = os.path.exists(folder)
    print(f"   {folder}: {'✅ exists' if exists else '❌ not found'}")

# 4. Check for __init__.py files
print(f"\n4. Checking __init__.py files:")
init_files = ['src/__init__.py', 'src\\__init__.py', 'src/ml/__init__.py', 'src\\ml\\__init__.py']
for f in init_files:
    exists = os.path.exists(f)
    print(f"   {f}: {'✅ exists' if exists else '❌ not found'}")

# 5. Check what's in src/ml
print(f"\n5. Contents of src/ml:")
ml_path = 'src/ml' if os.path.exists('../src/ml') else 'src\\ml'
if os.path.exists(ml_path):
    for f in os.listdir(ml_path):
        print(f"   - {f}")
else:
    print("   ❌ Directory not found")

# 6. Try importing
print(f"\n6. Attempting imports:")

try:
    import src
    print("   ✅ import src")
except ImportError as e:
    print(f"   ❌ import src: {e}")

try:
    import src.ml
    print("   ✅ import src.ml")
except ImportError as e:
    print(f"   ❌ import src.ml: {e}")

try:
    from src.ml import ai_trading_system
    print("   ✅ from src.ml import ai_trading_system")
except ImportError as e:
    print(f"   ❌ from src.ml import ai_trading_system: {e}")

# 7. Suggest fix
print("\n" + "=" * 60)
print("SOLUTION:")
print("=" * 60)

if not os.path.exists('../src/__init__.py') and not os.path.exists('../src/__init__.py'):
    print("\n❌ Missing src/__init__.py")
    print("   Fix: Create empty file at src/__init__.py")
    print("   Run: echo. > src\\__init__.py")

if not os.path.exists('../src/ml/__init__.py') and not os.path.exists('../src/ml/__init__.py'):
    print("\n❌ Missing src/ml/__init__.py")
    print("   Fix: Create empty file at src/ml/__init__.py")
    print("   Run: echo. > src\\ml\\__init__.py")

print("\nOr run these commands in PowerShell:")
print('   New-Item -ItemType File -Path "src\\__init__.py" -Force')
print('   New-Item -ItemType File -Path "src\\ml\\__init__.py" -Force')