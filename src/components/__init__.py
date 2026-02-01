"""
UI Components Module
====================
Streamlit UI components for the Alpha Research Platform.
"""

try:
    from src.components.economic_calendar_ui import (
        render_economic_calendar,
        render_economic_calendar_compact,
    )
except ImportError as e:
    print(f"Warning: Could not import economic_calendar_ui: {e}")