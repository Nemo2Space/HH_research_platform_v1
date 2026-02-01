with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the JSON import section to use session state
old_start = '''        # JSON Import Section
        st.markdown("##### 📁 Import Portfolio from JSON")
        uploaded_json = st.file_uploader("Upload JSON file", type=['json'], key="pb_json_upload")
        
        if uploaded_json is not None:
            try:
                import json as json_lib
                data = json_lib.load(uploaded_json)'''

new_start = '''        # JSON Import Section
        st.markdown("##### 📁 Import Portfolio from JSON")
        
        # Initialize session state for JSON data
        if 'json_portfolio_data' not in ss:
            ss.json_portfolio_data = None
        
        uploaded_json = st.file_uploader("Upload JSON file", type=['json'], key="pb_json_upload")
        
        # Parse and store in session state when file is uploaded
        if uploaded_json is not None:
            try:
                import json as json_lib
                uploaded_json.seek(0)  # Reset file pointer
                data = json_lib.load(uploaded_json)
                ss.json_portfolio_data = data  # Store in session state'''

if old_start in content:
    content = content.replace(old_start, new_start)
    
    # Also need to handle the case when data is loaded from session state
    # Find the build button section and ensure it uses session state
    
    with open('../dashboard/portfolio_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added session state for JSON data")
else:
    print("❌ Could not find the start block")
    if '# JSON Import Section' in content:
        print("Found marker")
