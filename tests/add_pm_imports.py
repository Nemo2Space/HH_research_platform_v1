with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add import for portfolio functions at the top
old_imports = '''from .rebalance_metrics import (
    RebalanceMetrics,
    RebalanceResults,
    PositionResult,
    calculate_rebalance_metrics,
    build_position_results,
    build_rebalance_results,
    verify_execution,
)'''

new_imports = '''from .rebalance_metrics import (
    RebalanceMetrics,
    RebalanceResults,
    PositionResult,
    calculate_rebalance_metrics,
    build_position_results,
    build_rebalance_results,
    verify_execution,
)

# Saved Portfolio imports
try:
    from dashboard.portfolio_builder import get_saved_portfolios, load_portfolio
    SAVED_PORTFOLIO_AVAILABLE = True
except ImportError:
    SAVED_PORTFOLIO_AVAILABLE = False'''

if old_imports in content:
    content = content.replace(old_imports, new_imports)
    print("✅ Added imports for saved portfolio")
else:
    print("❌ Could not find imports section")

with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
    f.write(content)
