with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find "if confirm:" which handles the manual send button
idx = content.find('if confirm:')
if idx > 0:
    end = min(len(content), idx + 4000)
    print("Manual confirm execution code:")
    print("="*70)
    print(content[idx:end])
else:
    # Try alternate patterns
    idx = content.find('if confirm and')
    if idx > 0:
        end = min(len(content), idx + 4000)
        print("Found 'if confirm and':")
        print(content[idx:end])
    else:
        print("Could not find confirm handler")
