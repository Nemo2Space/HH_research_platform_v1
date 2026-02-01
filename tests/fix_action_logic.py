with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the action classification logic in Portfolio Comparison
old_action_logic = '''        # Determine action
        if sym not in loaded_positions and target_w > 0:
            action = "🆕 NEW BUY"
        elif sym in loaded_positions and target_w == 0:
            action = "🔴 FULL EXIT"
        elif target_w > current_w * 1.1:  # Target is >10% higher
            action = "📈 ADD"
        elif target_w < current_w * 0.9:  # Target is >10% lower
            action = "📉 TRIM"
        else:
            action = "✓ HOLD"'''

new_action_logic = '''        # Determine action - using professional PM thresholds
        # Only flag actions for meaningful drift (>10% relative OR full entry/exit)
        relative_drift = abs(target_w - current_w) / current_w if current_w > 0.001 else float('inf')
        absolute_drift = abs(target_w - current_w)
        
        if sym not in loaded_positions and target_w > 0:
            action = "🆕 NEW BUY"
        elif sym in loaded_positions and target_w == 0:
            action = "🔴 FULL EXIT"
        elif target_w > current_w and relative_drift > 0.10 and absolute_drift > 0.005:
            # Only ADD if >10% relative drift AND >0.5% absolute drift
            action = "📈 ADD"
        elif target_w < current_w and relative_drift > 0.10 and absolute_drift > 0.005:
            # Only TRIM if >10% relative drift AND >0.5% absolute drift
            action = "📉 TRIM"
        else:
            action = "✓ HOLD"'''

if old_action_logic in content:
    content = content.replace(old_action_logic, new_action_logic)
    with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed Portfolio Comparison action logic")
else:
    print("❌ Could not find action logic section")
