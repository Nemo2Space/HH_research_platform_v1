with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Search for st.rerun, st.experimental_rerun, or anything that could cause reruns
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'rerun' in line.lower() or 'st.stop' in line:
        print(f"{i+1}: {line.strip()[:90]}")
