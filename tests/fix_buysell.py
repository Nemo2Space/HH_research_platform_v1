with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the BUY/SELL display to handle potential NaN values and use simpler display
old_code = '''    with col2:
        st.markdown("**💰 BUY/SELL**")
        st.markdown(
            f"### <span style='color:green'>\</span> / <span style='color:red'>\</span>",
            unsafe_allow_html=True)
        st.caption("Buy and sell values")'''

new_code = '''    with col2:
        st.markdown("**💰 BUY/SELL**")
        buy_val = metrics.total_buy if metrics.total_buy and not (isinstance(metrics.total_buy, float) and (metrics.total_buy != metrics.total_buy)) else 0
        sell_val = metrics.total_sell if metrics.total_sell and not (isinstance(metrics.total_sell, float) and (metrics.total_sell != metrics.total_sell)) else 0
        st.markdown(f":green[**\**] / :red[**\**]")
        st.caption("Buy and sell values")'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed BUY/SELL display")
else:
    print("❌ Could not find the BUY/SELL section")
    # Try alternate
    if "BUY/SELL" in content:
        print("Found BUY/SELL text")
