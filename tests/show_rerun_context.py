with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Show context around line 2568
print("Around line 2568 (st.rerun):")
for i in range(2560, 2580):
    print(f"{i+1}: {lines[i]}")
