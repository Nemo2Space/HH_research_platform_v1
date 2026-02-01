with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add verification display after "Last execution details"
old_code = '''    exec_dict = last.get("execution") or {}
    if exec_dict:
        with st.expander("Last execution details", expanded=False):
            st.json(exec_dict)'''

new_code = '''    exec_dict = last.get("execution") or {}
    if exec_dict:
        with st.expander("Last execution details", expanded=False):
            st.json(exec_dict)
    
    # Display post-execution verification if available
    verification = st.session_state.get('ai_pm_last_verification')
    if verification:
        with st.expander("📊 Post-Execution Verification (Orders vs Targets)", expanded=True):
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Open Orders in TWS", verification.get('total_open_orders', 0))
            col2.metric("Overall Accuracy", f"{verification.get('total_accuracy', 0):.1f}%")
            col3.metric("Projected Invested", f"")
            col4.metric("Projected Cash", f"")
            
            missing = verification.get('missing_orders', [])
            extra = verification.get('extra_orders', [])
            if missing:
                st.warning(f"⚠️ Missing orders for: {', '.join(missing)}")
            if extra:
                st.warning(f"⚠️ Extra orders for: {', '.join(extra)}")
            
            # Detailed table
            verify_data = verification.get('data', [])
            if verify_data:
                df_verify = pd.DataFrame(verify_data)
                # Sort by status then by target weight
                if 'Status' in df_verify.columns:
                    status_order = {'Pending': 0, 'Missing BUY': 1, 'Missing SELL': 2, 'On Target': 3, 'No Action': 4}
                    df_verify['_sort'] = df_verify['Status'].map(lambda x: status_order.get(x, 5))
                    df_verify = df_verify.sort_values('_sort').drop(columns=['_sort'])
                st.dataframe(df_verify, use_container_width=True, hide_index=True, height=400)
            
            # Clear button
            if st.button("Clear Verification", key="clear_verification_btn"):
                del st.session_state['ai_pm_last_verification']
                st.rerun()'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added verification display panel")
else:
    print("❌ Could not find code block")
