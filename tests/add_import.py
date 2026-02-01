with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find where to add strategy selector - after the JSON import section
# Look for the markdown("---") after JSON import

# First, let's find imports and add InvestmentStrategy
old_import = '''from dashboard.portfolio_builder import (
        get_latest_stock_universe,
        get_sector_summary,
        get_top_stocks_by_category,
        build_portfolio_ai_context,
        build_portfolio_instructions,
        get_ai_response,
        build_portfolio_from_intent,
        save_portfolio,
        get_saved_portfolios,
        load_portfolio,
        delete_portfolio,'''

new_import = '''from dashboard.portfolio_builder import (
        get_latest_stock_universe,
        get_sector_summary,
        get_top_stocks_by_category,
        build_portfolio_ai_context,
        build_portfolio_instructions,
        get_ai_response,
        build_portfolio_from_intent,
        save_portfolio,
        get_saved_portfolios,
        load_portfolio,
        delete_portfolio,
        PORTFOLIO_TEMPLATES,'''

if old_import in content:
    content = content.replace(old_import, new_import)
    print("✅ Updated imports")
else:
    print("⚠️ Import block not found exactly, checking if PORTFOLIO_TEMPLATES already imported")
    if 'PORTFOLIO_TEMPLATES' in content:
        print("   PORTFOLIO_TEMPLATES already imported")

with open('../dashboard/portfolio_tab.py', 'w', encoding='utf-8') as f:
    f.write(content)
