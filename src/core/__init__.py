"""
Signal Hub Core Module

Contains the foundational components:
- UnifiedSignal: The single source of truth for any ticker
- SignalEngine: Combines all analyzers into unified signals
- MarketOverview: Market-wide summary
- SignalSnapshot: Historical signal tracking

Author: Alpha Research Platform
"""

from src.core.unified_signal import (
    UnifiedSignal,
    MarketOverview,
    SignalSnapshot,
    ComponentScore,
    SignalStrength,
    RiskLevel,
    AssetType,
    BOND_ETFS,
)

from src.core.signal_engine import (
    SignalEngine,
    get_signal_engine,
    generate_signal,
    generate_signals,
    get_market_overview,
)

__all__ = [
    # Models
    'UnifiedSignal',
    'MarketOverview',
    'SignalSnapshot',
    'ComponentScore',

    # Enums
    'SignalStrength',
    'RiskLevel',
    'AssetType',

    # Constants
    'BOND_ETFS',

    # Engine
    'SignalEngine',
    'get_signal_engine',
    'generate_signal',
    'generate_signals',
    'get_market_overview',
]