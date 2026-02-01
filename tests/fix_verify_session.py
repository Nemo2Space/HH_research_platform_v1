with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the verification block and fix it - store in session_state and fix types
old_code = '''                    # Post-execution verification
                    try:
                        from .execution_verify import verify_execution, PortfolioVerification
                        with st.spinner("Verifying orders in TWS..."):
                            # Wait for orders to be visible in TWS
                            import time
                            time.sleep(3)  # Use regular sleep, not ib.sleep which freezes in Streamlit
                            
                            verification = verify_execution(
                                ib=gw.ib,
                                snapshot=snapshot,
                                plan=plan,
                                targets=targets,
                                price_map=price_map,
                            )
                        
                        st.subheader("📊 Post-Execution Verification")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Open Orders in TWS", verification.total_open_orders)
                        col2.metric("Overall Accuracy", f"{verification.total_accuracy:.1f}%")
                        col3.metric("Projected Invested", f"")
                        col4.metric("Projected Cash", f"")
                        
                        if verification.missing_orders:
                            st.warning(f"⚠️ Missing orders for: {', '.join(verification.missing_orders)}")
                        if verification.extra_orders:
                            st.warning(f"⚠️ Extra orders for: {', '.join(verification.extra_orders)}")
                        
                        # Detailed table
                        verify_data = []
                        for v in verification.symbols_verified:
                            if v.pending_buy > 0 or v.pending_sell > 0 or v.target_weight > 0.1:
                                verify_data.append({
                                    "Symbol": v.symbol,
                                    "Target %": f"{v.target_weight:.2f}",
                                    "Current %": f"{v.current_weight:.2f}",
                                    "Projected %": f"{v.projected_weight:.2f}",
                                    "Current Qty": v.current_shares,
                                    "Pending BUY": v.pending_buy if v.pending_buy > 0 else "",
                                    "Pending SELL": v.pending_sell if v.pending_sell > 0 else "",
                                    "Projected Qty": v.projected_shares,
                                    "Status": v.status,
                                    "Accuracy": f"{v.accuracy:.1f}%",
                                })
                        
                        if verify_data:
                            df_verify = pd.DataFrame(verify_data)
                            st.dataframe(df_verify, use_container_width=True, hide_index=True)
                        
                    except Exception as e:
                        st.warning(f"Verification failed: {e}")'''

new_code = '''                    # Post-execution verification - store in session_state to persist after rerun
                    try:
                        from .execution_verify import verify_execution, PortfolioVerification
                        with st.spinner("Verifying orders in TWS..."):
                            # Wait for orders to be visible in TWS
                            import time
                            time.sleep(3)  # Use regular sleep, not ib.sleep which freezes in Streamlit
                            
                            verification = verify_execution(
                                ib=gw.ib,
                                snapshot=snapshot,
                                plan=plan,
                                targets=targets,
                                price_map=price_map,
                            )
                        
                        # Build verification data for session state
                        verify_data = []
                        for v in verification.symbols_verified:
                            if v.pending_buy > 0 or v.pending_sell > 0 or v.target_weight > 0.1:
                                verify_data.append({
                                    "Symbol": v.symbol,
                                    "Target %": f"{v.target_weight:.2f}",
                                    "Current %": f"{v.current_weight:.2f}",
                                    "Projected %": f"{v.projected_weight:.2f}",
                                    "Current Qty": int(v.current_shares),
                                    "Pending BUY": int(v.pending_buy),
                                    "Pending SELL": int(v.pending_sell),
                                    "Projected Qty": int(v.projected_shares),
                                    "Status": v.status,
                                    "Accuracy": f"{v.accuracy:.1f}%",
                                })
                        
                        # Store verification in session_state so it persists
                        st.session_state['ai_pm_last_verification'] = {
                            'total_open_orders': verification.total_open_orders,
                            'total_accuracy': verification.total_accuracy,
                            'projected_invested': verification.projected_invested,
                            'projected_cash': verification.projected_cash,
                            'missing_orders': verification.missing_orders,
                            'extra_orders': verification.extra_orders,
                            'data': verify_data,
                        }
                        
                    except Exception as e:
                        st.warning(f"Verification failed: {e}")'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed verification to store in session_state")
else:
    print("❌ Could not find code block")
