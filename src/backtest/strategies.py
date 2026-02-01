"""
Alpha Platform - Backtest Strategies

Predefined strategy configurations for backtesting.
Includes AI-based strategies using ML model predictions.
"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    name: str
    display_name: str
    description: str
    strategy_type: str
    default_params: Dict[str, Any]
    param_options: Dict[str, List]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'strategy_type': self.strategy_type,
            'default_params': self.default_params,
            'param_options': self.param_options
        }


# ============================================================================
# PREDEFINED STRATEGIES
# ============================================================================

STRATEGIES: Dict[str, StrategyConfig] = {

    'aggressive_buy': StrategyConfig(
        name='aggressive_buy',
        display_name='Aggressive Buy Signals',
        description='Buy on any buy signal (STRONG_BUY, BUY, WEAK_BUY)',
        strategy_type='signal_based',
        default_params={
            'buy_signals': ['STRONG_BUY', 'BUY', 'WEAK_BUY'],
            'sell_signals': []
        },
        param_options={
            'buy_signals': [
                ['STRONG_BUY'],
                ['STRONG_BUY', 'BUY'],
                ['STRONG_BUY', 'BUY', 'WEAK_BUY'],
            ]
        }
    ),

    'conservative_buy': StrategyConfig(
        name='conservative_buy',
        display_name='Conservative Buy Signals',
        description='Only buy on STRONG_BUY signals',
        strategy_type='signal_based',
        default_params={
            'buy_signals': ['STRONG_BUY'],
            'sell_signals': []
        },
        param_options={}
    ),

    'buy_signals': StrategyConfig(
        name='buy_signals',
        display_name='Buy Signals (BUY + STRONG_BUY)',
        description='Buy on BUY and STRONG_BUY signals',
        strategy_type='signal_based',
        default_params={
            'buy_signals': ['STRONG_BUY', 'BUY'],
            'sell_signals': []
        },
        param_options={}
    ),

    'long_short': StrategyConfig(
        name='long_short',
        display_name='Long/Short Signal Strategy',
        description='Long on buy signals, short on sell signals',
        strategy_type='signal_based',
        default_params={
            'buy_signals': ['STRONG_BUY', 'BUY'],
            'sell_signals': ['SELL', 'STRONG_SELL']
        },
        param_options={}
    ),

    'high_sentiment': StrategyConfig(
        name='high_sentiment',
        display_name='High Sentiment',
        description='Buy when sentiment score >= 65',
        strategy_type='sentiment_only',
        default_params={
            'buy_threshold': 65,
            'sell_threshold': 35
        },
        param_options={
            'buy_threshold': [60, 65, 70, 75, 80],
            'sell_threshold': [20, 25, 30, 35, 40]
        }
    ),

    'high_fundamental': StrategyConfig(
        name='high_fundamental',
        display_name='High Fundamental Score',
        description='Buy when fundamental score >= 65',
        strategy_type='fundamental_only',
        default_params={
            'buy_threshold': 65,
            'sell_threshold': 35
        },
        param_options={
            'buy_threshold': [60, 65, 70, 75],
            'sell_threshold': [25, 30, 35, 40]
        }
    ),

    'high_total_score': StrategyConfig(
        name='high_total_score',
        display_name='High Total Score',
        description='Buy when total score >= 55',
        strategy_type='score_threshold',
        default_params={
            'score_column': 'total_score',
            'buy_threshold': 55,
            'sell_threshold': 45
        },
        param_options={
            'buy_threshold': [50, 55, 60, 65, 70],
            'sell_threshold': [30, 35, 40, 45, 50]
        }
    ),

    'quality_momentum': StrategyConfig(
        name='quality_momentum',
        display_name='Quality + Momentum',
        description='Buy when 2+ of: sentiment>=55, fundamental>=55, total>=50',
        strategy_type='composite',
        default_params={
            'min_sentiment': 55,
            'min_fundamental': 55,
            'min_total': 50,
            'require_all': False
        },
        param_options={
            'min_sentiment': [50, 55, 60, 65, 70],
            'min_fundamental': [50, 55, 60, 65, 70],
            'require_all': [True, False]
        }
    ),

    'triple_screen': StrategyConfig(
        name='triple_screen',
        display_name='Triple Screen (All Conditions)',
        description='Only buy when ALL conditions met: sentiment>=60, fundamental>=60, total>=55',
        strategy_type='composite',
        default_params={
            'min_sentiment': 60,
            'min_fundamental': 60,
            'min_total': 55,
            'require_all': True
        },
        param_options={}
    ),

    # =========================================================================
    # AI-BASED STRATEGIES
    # =========================================================================

    'ai_probability': StrategyConfig(
        name='ai_probability',
        display_name='🤖 AI Probability',
        description='ML model predicts win probability. Trades when prob >= 55% and EV > 0.',
        strategy_type='ai_probability',
        default_params={
            'min_probability': 0.55,
            'min_ev': 0.001,
            'max_trades_per_day': 5
        },
        param_options={
            'min_probability': [0.50, 0.55, 0.60, 0.65, 0.70],
            'max_trades_per_day': [3, 5, 10, 20]
        }
    ),

    'ai_conservative': StrategyConfig(
        name='ai_conservative',
        display_name='🤖 AI Conservative',
        description='Conservative AI: probability >= 60%, strong positive EV, max 3 trades/day.',
        strategy_type='ai_probability',
        default_params={
            'min_probability': 0.60,
            'min_ev': 0.005,
            'max_trades_per_day': 3
        },
        param_options={}
    ),

    'ai_aggressive': StrategyConfig(
        name='ai_aggressive',
        display_name='🤖 AI Aggressive',
        description='Aggressive AI: probability >= 50%, allows more trades.',
        strategy_type='ai_probability',
        default_params={
            'min_probability': 0.50,
            'min_ev': -0.005,
            'max_trades_per_day': 10
        },
        param_options={}
    ),

    'ai_high_conviction': StrategyConfig(
        name='ai_high_conviction',
        display_name='🤖 AI High Conviction',
        description='Only trade when AI probability >= 65%. Fewer but higher quality trades.',
        strategy_type='ai_probability',
        default_params={
            'min_probability': 0.65,
            'min_ev': 0.01,
            'max_trades_per_day': 3
        },
        param_options={}
    ),
}


# ============================================================================
# HOLDING PERIOD OPTIONS
# ============================================================================

HOLDING_PERIODS = [
    {'days': 1, 'label': '1 Day', 'description': 'Day trading / overnight'},
    {'days': 5, 'label': '5 Days', 'description': 'Weekly swing trade'},
    {'days': 10, 'label': '10 Days', 'description': '2-week hold'},
    {'days': 20, 'label': '20 Days', 'description': 'Monthly position'},
]


# ============================================================================
# BENCHMARK OPTIONS
# ============================================================================

BENCHMARKS = [
    {'ticker': 'SPY', 'name': 'S&P 500 ETF'},
    {'ticker': 'QQQ', 'name': 'Nasdaq 100 ETF'},
    {'ticker': 'VOO', 'name': 'Vanguard S&P 500'},
    {'ticker': 'IWM', 'name': 'Russell 2000 ETF'},
    {'ticker': 'DIA', 'name': 'Dow Jones ETF'},
]

# ============================================================================
# PARAMETER GRIDS FOR OPTIMIZATION
# ============================================================================

OPTIMIZATION_GRIDS = {
    'signal_based': {
        'buy_signals': [
            ['STRONG_BUY'],
            ['STRONG_BUY', 'BUY'],
            ['STRONG_BUY', 'BUY', 'WEAK_BUY'],
        ],
    },

    'score_threshold': {
        'buy_threshold': [50, 55, 60, 65, 70, 75],
        'sell_threshold': [25, 30, 35, 40, 45, 50],
    },

    'sentiment_only': {
        'buy_threshold': [55, 60, 65, 70, 75, 80],
        'sell_threshold': [20, 25, 30, 35, 40],
    },

    'fundamental_only': {
        'buy_threshold': [55, 60, 65, 70, 75],
        'sell_threshold': [25, 30, 35, 40, 45],
    },

    'composite': {
        'min_sentiment': [50, 55, 60, 65, 70],
        'min_fundamental': [50, 55, 60, 65, 70],
        'min_total': [45, 50, 55, 60],
        'require_all': [True, False],
    },

    'ai_probability': {
        'min_probability': [0.50, 0.55, 0.60, 0.65, 0.70],
        'min_ev': [-0.005, 0, 0.005, 0.01],
        'max_trades_per_day': [3, 5, 10],
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_strategy(name: str) -> StrategyConfig:
    """Get strategy configuration by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]


def get_all_strategies() -> List[StrategyConfig]:
    """Get all available strategies."""
    return list(STRATEGIES.values())


def get_strategy_names() -> List[str]:
    """Get list of strategy names."""
    return list(STRATEGIES.keys())


def get_ai_strategies() -> List[StrategyConfig]:
    """Get only AI-based strategies."""
    return [s for s in STRATEGIES.values() if s.strategy_type == 'ai_probability']