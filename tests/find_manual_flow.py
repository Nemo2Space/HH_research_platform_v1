with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the manual execution flow - where "Confirm & Send (Manual)" leads
idx = content.find('ai_pm_confirm_send_btn')
if idx > 0:
    # Find the if block that handles this button
    start = max(0, idx - 500)
    end = min(len(content), idx + 3000)
    print("Manual execution flow code:")
    print("="*70)
    print(content[start:end])
