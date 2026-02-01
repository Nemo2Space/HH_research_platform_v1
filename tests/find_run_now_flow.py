with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all occurrences of ib. calls in the Run Now section
# Look for potential freezing points after "Building trade plan"
idx = content.find('Building trade plan')
if idx > 0:
    section = content[idx:idx+5000]
    print("Section after 'Building trade plan':")
    print(section)
