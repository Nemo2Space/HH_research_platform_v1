with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all IbkrGateway() creations with unique clientId
old_gw1 = '''            gw = IbkrGateway()
            st.session_state.ai_pm_gateway = gw
            st.session_state.ai_pm_connection_ok = False
            
            with col_status:
                with st.spinner("Connecting to IBKR..."):
                    try:
                        success = gw.connect(timeout_sec=10.0)
                        if success and gw.is_connected():
                            st.session_state.ai_pm_connection_ok = True
                            st.success("✅ Connected to IBKR")
                            st.rerun()
                        else:
                            st.session_state.ai_pm_connection_ok = False
                            st.error("❌ Connection failed")
                    except Exception as e:
                        st.session_state.ai_pm_connection_ok = False
                        st.error(f"❌ IBKR Error: {e}")'''

new_gw1 = '''            # Use session-unique clientId to avoid conflicts
            import random
            import os
            if 'ai_pm_client_id' not in st.session_state:
                st.session_state.ai_pm_client_id = random.randint(900, 950)
            os.environ['IBKR_CLIENT_ID'] = str(st.session_state.ai_pm_client_id)
            
            gw = IbkrGateway()
            st.session_state.ai_pm_gateway = gw
            st.session_state.ai_pm_connection_ok = False
            
            with col_status:
                with st.spinner(f"Connecting to IBKR (clientId {st.session_state.ai_pm_client_id})..."):
                    try:
                        success = gw.connect(timeout_sec=10.0)
                        if success and gw.is_connected():
                            st.session_state.ai_pm_connection_ok = True
                            st.success(f"✅ Connected to IBKR (clientId {st.session_state.ai_pm_client_id})")
                            st.rerun()
                        else:
                            st.session_state.ai_pm_connection_ok = False
                            st.error("❌ Connection failed")
                    except Exception as e:
                        st.session_state.ai_pm_connection_ok = False
                        st.error(f"❌ IBKR Error: {e}")'''

if old_gw1 in content:
    content = content.replace(old_gw1, new_gw1)
    print("✅ Updated first gateway init")
else:
    print("❌ Could not find first gateway init")

# Also update the second gateway creation
old_gw2 = '''    if gw is None:
        gw = IbkrGateway()
        st.session_state.ai_pm_gateway = gw
        needs_connect = True
    elif not st.session_state.ai_pm_connection_ok:
        needs_connect = True'''

new_gw2 = '''    if gw is None:
        # Use session-unique clientId
        import random
        import os
        if 'ai_pm_client_id' not in st.session_state:
            st.session_state.ai_pm_client_id = random.randint(900, 950)
        os.environ['IBKR_CLIENT_ID'] = str(st.session_state.ai_pm_client_id)
        
        gw = IbkrGateway()
        st.session_state.ai_pm_gateway = gw
        needs_connect = True
    elif not st.session_state.ai_pm_connection_ok:
        needs_connect = True'''

if old_gw2 in content:
    content = content.replace(old_gw2, new_gw2)
    print("✅ Updated second gateway init")
else:
    print("❌ Could not find second gateway init")

with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
    f.write(content)
