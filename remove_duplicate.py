with open('dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the duplicate JSON upload inside the saved portfolios block
old_duplicate = '''            if clear_btn:
                st.session_state.ai_pm_target_portfolio = None
                st.session_state.ai_pm_target_weights = {}
                st.info("Target portfolio cleared")
                st.rerun()
            
            # JSON File Upload Option
            st.markdown("---")
            st.markdown("**📁 Or Load from JSON File:**")
            
            uploaded_json = st.file_uploader(
                "Upload Portfolio JSON",
                type=['json'],
                key="ai_pm_json_uploader",
                help="Upload an IBKR-formatted portfolio JSON file"
            )
            
            if uploaded_json:
                try:
                    import json
                    import tempfile
                    import os
                    
                    # Save uploaded file temporarily and track path
                    json_data = json.load(uploaded_json)
                    
                    # Save to json folder for later editing
                    json_dir = 'json'
                    os.makedirs(json_dir, exist_ok=True)
                    json_path = os.path.join(json_dir, uploaded_json.name)
                    
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=4)
                    
                    # Track the path for save functionality
                    st.session_state.ai_pm_loaded_json_path = json_path
                    
                    # Parse holdings from JSON
                    holdings = json_data if isinstance(json_data, list) else json_data.get('holdings', [])
                    
                    if holdings:
                        target_weights = {}
                        for h in holdings:
                            ticker = (h.get('symbol') or h.get('ticker') or '').upper().strip()
                            weight = float(h.get('weight', 0)) / 100  # Convert to decimal
                            if ticker and weight > 0:
                                target_weights[ticker] = weight
                        
                        st.session_state.ai_pm_target_portfolio = {
                            'id': 'json_upload',
                            'name': uploaded_json.name.replace('.json', ''),
                            'total_value': 100000,
                            'num_holdings': len(target_weights),
                            'objective': 'custom',
                            'source': 'json_file',
                            'json_path': json_path,
                        }
                        st.session_state.ai_pm_target_weights = target_weights
                        
                        st.success(f"✅ Loaded **{uploaded_json.name}** with {len(target_weights)} holdings")
                        st.caption(f"📂 Saved to: {json_path}")
                    else:
                        st.error("No holdings found in JSON file")
                        
                except Exception as e:
                    st.error(f"Error loading JSON: {e}")
            
            # Show current target if loaded
            if st.session_state.ai_pm_target_portfolio:
                tp = st.session_state.ai_pm_target_portfolio
                tw = st.session_state.ai_pm_target_weights
                
                source_info = f" | 📁 JSON" if tp.get('source') == 'json_file' else ""
                st.info(f"🎯 **Active Target:** {tp['name']} | {tp['num_holdings']} holdings{source_info}")
                
                with st.expander("View Target Holdings", expanded=False):
                    target_df = pd.DataFrame([
                        {'Ticker': k, 'Weight %': f"{v*100:.2f}%"}
                        for k, v in sorted(tw.items(), key=lambda x: -x[1])
                    ])
                    st.dataframe(target_df, hide_index=True, height=300)
        else:'''

new_simplified = '''            if clear_btn:
                st.session_state.ai_pm_target_portfolio = None
                st.session_state.ai_pm_target_weights = {}
                st.session_state.ai_pm_loaded_json_path = None
                st.info("Target portfolio cleared")
                st.rerun()
        else:'''

if old_duplicate in content:
    content = content.replace(old_duplicate, new_simplified)
    with open('dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Removed duplicate JSON upload section")
else:
    print("❌ Could not find duplicate section - checking for variations...")
    
    # Try to find the section with truncated lines
    if "ai_pm_json_uploader" in content and "ai_pm_json_uploader_main" in content:
        print("Both uploaders found - need manual inspection")
