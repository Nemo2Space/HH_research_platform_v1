with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add debug logging around the execution steps
old_code = '''                # Get capital to deploy from user input
                capital_to_deploy_confirm = st.session_state.get('ai_pm_deploy_amount')

                plan = build_trade_plan(
                    snapshot=snapshot,
                    targets=targets,
                    constraints=DEFAULT_CONSTRAINTS,
                    price_map=price_map,
                    capital_to_deploy=capital_to_deploy_confirm,
                )
                gates = evaluate_trade_plan_gates(snapshot=snapshot, signals=signals, plan=plan,
                                                  constraints=DEFAULT_CONSTRAINTS)

                if gates.blocked:'''

new_code = '''                # Get capital to deploy from user input
                capital_to_deploy_confirm = st.session_state.get('ai_pm_deploy_amount')

                import logging
                _exec_log = logging.getLogger(__name__)
                _exec_log.info("Execution: building trade plan...")
                st.info("⏳ Building trade plan...")
                
                plan = build_trade_plan(
                    snapshot=snapshot,
                    targets=targets,
                    constraints=DEFAULT_CONSTRAINTS,
                    price_map=price_map,
                    capital_to_deploy=capital_to_deploy_confirm,
                )
                _exec_log.info(f"Execution: trade plan built, {len(plan.orders) if plan and plan.orders else 0} orders")
                
                _exec_log.info("Execution: evaluating gates...")
                st.info("⏳ Evaluating gates...")
                
                gates = evaluate_trade_plan_gates(snapshot=snapshot, signals=signals, plan=plan,
                                                  constraints=DEFAULT_CONSTRAINTS)
                _exec_log.info(f"Execution: gates evaluated, blocked={gates.blocked}")

                if gates.blocked:'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added debug logging to execution flow")
else:
    print("❌ Could not find execution code")
