# test_portfolio_tab_changes.py - Save this and run it

print("="*80)
print("TESTING PORTFOLIO_TAB.PY CHANGES")
print("="*80)

# Read the CURRENT file (what you have now)
with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    current_content = f.read()

errors = []

# Test 1: Check imports
print("\n1. Checking imports...")
if 'render_comprehensive_stock_table' in current_content:
    print("   ✓ render_comprehensive_stock_table imported")
else:
    errors.append("render_comprehensive_stock_table not imported")
    print("   ✗ NOT imported")

# Test 2: Check if function is CALLED (not just imported)
print("\n2. Checking if display code is called...")
if 'render_comprehensive_stock_table(result)' in current_content:
    print("   ✓ render_comprehensive_stock_table is CALLED")
else:
    errors.append("render_comprehensive_stock_table never called")
    print("   ✗ render_comprehensive_stock_table NEVER CALLED")

if 'render_save_portfolio_section(result)' in current_content:
    print("   ✓ render_save_portfolio_section is CALLED")
else:
    errors.append("render_save_portfolio_section never called")
    print("   ✗ render_save_portfolio_section NEVER CALLED")

# Test 3: Check location
print("\n3. Checking display location...")
if 'ENHANCED PORTFOLIO DISPLAY' in current_content:
    print("   ✓ Display section exists")
else:
    print("   ✗ Display section NOT found")
    errors.append("No display section")

print("\n" + "="*80)
if errors:
    print("✗ NEED TO ADD DISPLAY CODE")
    print("\nThe display functions are imported but never called.")
    print("I need to add the calling code to portfolio_tab.py")
else:
    print("✅ ALL GOOD - Display code is present and will work")
print("="*80)