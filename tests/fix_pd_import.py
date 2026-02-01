with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the verification section and fix the import
old_code = '''                    # Detailed table
                        import pandas as pd
                        verify_data = []'''

new_code = '''                    # Detailed table
                        verify_data = []'''

if old_code in content:
    content = content.replace(old_code, new_code)
    
    # Also need to make sure pd is imported at the top or before usage
    # Check if pandas is imported at top of file
    if 'import pandas as pd' not in content[:2000]:
        # Add it after other imports
        old_import = 'import streamlit as st'
        new_import = 'import streamlit as st\nimport pandas as pd'
        if old_import in content and 'import pandas as pd' not in content:
            content = content.replace(old_import, new_import, 1)
    
    with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed pandas import")
else:
    print("❌ Could not find code")
    # Check if pandas is imported
    if 'import pandas' in content:
        print("pandas is already imported somewhere")
        idx = content.find('import pandas')
        print(f"At position {idx}: {content[idx:idx+50]}")
