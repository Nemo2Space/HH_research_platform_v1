with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Search for the exact line
lines = content.split('\n')
for i, line in enumerate(lines):
    if "span style='color:green'" in line and 'total_buy' in line:
        print(f"Found at line {i+1}")
        print(f"Content: {repr(line)}")
        
        # Replace this line with fixed version
        old_line = lines[i]
        new_line = '        st.markdown(f":green[****] / :red[****]")'
        lines[i] = new_line
        print(f"Replaced with: {new_line}")
        
        # Also remove the unsafe_allow_html line if it's the next line
        if i+1 < len(lines) and 'unsafe_allow_html' in lines[i+1]:
            # Merge with previous line or remove
            pass

content = '\n'.join(lines)
with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done")
