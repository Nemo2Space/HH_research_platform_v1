"""
Signal Hub Tabs Module

Contains all UI tabs for the Signal Hub:
- signals_tab: Main signals command center
- deep_dive_tab: Detailed single ticker analysis

Author: Alpha Research Platform
"""

from src.tabs.signals_tab import render_signals_tab, render_signals_tab
from src.tabs.deep_dive_tab import render_deep_dive_tab, render_deep_dive_standalone

__all__ = [
    'render_signals_tab',
    'render_deep_dive_tab',
    'render_deep_dive_standalone',
]