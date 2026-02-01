with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find where to add the verification display - after the Cancel All Orders button section
# Look for the execution details expander
idx = content.find('Last execution details')
if idx > 0:
    print(f"Found 'Last execution details' at {idx}")
    # Show context
    print(content[idx-200:idx+300])
