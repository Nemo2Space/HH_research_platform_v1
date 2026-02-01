with open('../dashboard/ai_pm/config.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Change cash_min from -0.20 to -0.50 (allow up to -50%)
old_line = 'cash_min: float = -0.20               # -20% min cash (allows margin)'
new_line = 'cash_min: float = -0.50               # -50% min cash (allows significant margin)'

if old_line in content:
    content = content.replace(old_line, new_line)
    with open('../dashboard/ai_pm/config.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Updated cash_min from -20% to -50%")
else:
    print("❌ Could not find cash_min line")
