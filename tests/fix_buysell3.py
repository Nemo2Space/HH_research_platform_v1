with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the broken line and the whole section
old_broken = '''        st.markdown("**💰 BUY/SELL**")
        st.markdown(
        st.markdown(f":green[****] / :red[****]")
            unsafe_allow_html=True)
        st.caption("Buy and sell values")'''

new_fixed = '''        st.markdown("**💰 BUY/SELL**")
        buy_v = metrics.total_buy if metrics.total_buy == metrics.total_buy else 0
        sell_v = metrics.total_sell if metrics.total_sell == metrics.total_sell else 0
        st.markdown(f":green[{buy_v:,.2f}] / :red[{sell_v:,.2f}]")
        st.caption("Buy and sell values")'''

if old_broken in content:
    content = content.replace(old_broken, new_fixed)
    with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed broken BUY/SELL section")
else:
    print("❌ Could not find broken section, checking current state...")
    # Show current state
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'BUY/SELL' in line:
            print(f"\nLines {i+1}-{i+8}:")
            for j in range(i, min(len(lines), i+8)):
                print(f"{j+1}: {lines[j]}")
            break
