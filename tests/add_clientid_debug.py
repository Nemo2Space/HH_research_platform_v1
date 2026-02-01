with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add clientId logging to the gateway creation
old_gateway_create = '''    gw = st.session_state.ai_pm_gateway
    
    # Connection status and reconnect option
    col_status, col_reconnect = st.columns([3, 1])'''

new_gateway_create = '''    gw = st.session_state.ai_pm_gateway
    
    # Show current clientId for debugging
    if gw:
        st.caption(f"🔌 Using clientId: {gw.cfg.client_id if hasattr(gw, 'cfg') else 'unknown'}")
    
    # Connection status and reconnect option
    col_status, col_reconnect = st.columns([3, 1])'''

if old_gateway_create in content:
    content = content.replace(old_gateway_create, new_gateway_create)
    print("✅ Added clientId display")
else:
    print("⚠️ Could not find gateway section")

# Also make sure we use a unique clientId (randomize it)
old_gateway_init = '''        gw = IbkrGateway()
        st.session_state.ai_pm_gateway = gw
        st.session_state.ai_pm_connection_ok = False'''

new_gateway_init = '''        # Use unique clientId to avoid conflicts
        import random
        import os
        unique_id = random.randint(900, 950)
        os.environ['IBKR_CLIENT_ID'] = str(unique_id)
        gw = IbkrGateway()
        st.session_state.ai_pm_gateway = gw
        st.session_state.ai_pm_connection_ok = False
        st.info(f"Creating new gateway with clientId {unique_id}")'''

if old_gateway_init in content:
    content = content.replace(old_gateway_init, new_gateway_init)
    print("✅ Added unique clientId generation")
else:
    print("⚠️ Could not find gateway init")

with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
    f.write(content)
