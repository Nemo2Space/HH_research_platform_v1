"""
Signals Tab Package

Split into modules for maintainability:
- shared.py: Common imports, feature flags, utilities
- job_manager.py: Background job tracking
- universe_manager.py: Ticker/universe management
- table_view.py: Signals table rendering
- analysis.py: Analysis pipeline
- deep_dive.py: Deep dive panel
- earnings_views.py: Earnings views
- ai_chat.py: AI chat functionality
- main.py: Main entry point

Usage:
    from src.tabs.signals_tab import render_signals_tab
"""

from .main import render_signals_tab

__all__ = ['render_signals_tab']
