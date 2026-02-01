"""
AI Strategy Integration for Backtest Engine

This file provides:
1. AI strategy that uses ML probability for trade decisions
2. Add to src/backtest/ai_strategy.py

Usage:
    Add 'ai_probability' to your strategies.py
    Engine will call _run_ai_strategy when strategy='ai_probability'
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, Any, List
from dataclasses import dataclass

# Import will be relative when placed in src/backtest/
# from .engine import Trade


@dataclass
class Trade:
    """Trade record - copy from engine.py or import."""
    ticker: str
    entry_date: any
    exit_date: any
    entry_price: float
    exit_price: float
    signal_type: str
    direction: str
    return_pct: float
    holding_days: int
    scores: Dict[str, Any] = None

    def __post_init__(self):
        self.scores = self.scores or {}

    @property
    def is_winner(self) -> bool:
        return self.return_pct > 0 if self.direction == 'LONG' else self.return_pct < 0


class AIBacktestStrategy:
    """
    AI-based trading strategy for backtesting.

    Uses trained ML model to compute win probability for each historical record,
    then simulates trades based on probability threshold.
    """

    def __init__(self):
        self.ml_predictor = None
        self._initialized = False

    def initialize(self):
        """Load the ML model."""
        if self._initialized:
            return True

        try:
            from src.ml.signal_predictor import MLSignalPredictor

            self.ml_predictor = MLSignalPredictor()

            # Try to load existing model
            import os
            model_path = 'models/signal_predictor.pkl'
            if os.path.exists(model_path):
                self.ml_predictor.load(model_path)
                self._initialized = True
                return True
            else:
                print("Warning: ML model not found. Train first with: python -m src.ml.ai_trading_system --train")
                return False
        except Exception as e:
            print(f"Failed to initialize AI strategy: {e}")
            return False

    def compute_probability(self, row: pd.Series) -> float:
        """Compute ML win probability for a single row."""
        if not self._initialized or not self.ml_predictor:
            return 0.5

        scores = {
            'ticker': row.get('ticker', 'UNKNOWN'),
            'sentiment_score': float(row.get('sentiment', 50) or 50),
            'fundamental_score': float(row.get('fundamental_score', 50) or 50),
            'technical_score': float(row.get('technical_score', 50) or 50),
            'options_flow_score': float(row.get('options_flow_score', 50) or 50),
            'short_squeeze_score': float(row.get('short_squeeze_score', 50) or 50),
            'total_score': float(row.get('total_score', 50) or 50),
        }

        try:
            prediction = self.ml_predictor.predict(scores)
            return prediction.prob_win_5d
        except Exception as e:
            return 0.5

    def run_strategy(self, df: pd.DataFrame, holding_period: int,
                     params: Dict[str, Any]) -> List[Trade]:
        """
        Run AI probability strategy on historical data.

        Args:
            df: Historical scores DataFrame
            holding_period: Days to hold (5, 10, 20)
            params: Strategy parameters:
                - min_probability: Minimum win probability to trade (default 0.55)
                - min_ev: Minimum expected value (default 0.001)
                - max_trades_per_day: Max trades per day (default 5)
        """
        if not self.initialize():
            return []

        min_prob = params.get('min_probability', 0.55)
        min_ev = params.get('min_ev', 0.001)
        max_per_day = params.get('max_trades_per_day', 5)
        log_to_tracker = params.get('log_to_tracker', True)  # Auto-log by default

        trades = []
        return_col = f'return_{holding_period}d' if holding_period in [1, 5, 10, 20] else 'return_5d'
        price_col = f'price_{holding_period}d' if holding_period in [1, 5, 10, 20] else 'price_5d'

        # Initialize performance tracker for auto-logging
        tracker = None
        if log_to_tracker:
            try:
                from src.ml.ai_performance_tracker import AIPerformanceTracker
                tracker = AIPerformanceTracker()
            except Exception:
                pass

        # Group by date to limit trades per day
        grouped = df.groupby('date')

        for date, day_df in grouped:
            day_trades = []

            for _, row in day_df.iterrows():
                ret = row.get(return_col)
                if pd.isna(ret) or ret is None:
                    continue

                # Compute AI probability
                prob = self.compute_probability(row)

                # Compute expected value
                avg_win = 2.5  # Typical win ~2.5%
                avg_loss = 2.0  # Typical loss ~2%
                ev = prob * avg_win - (1 - prob) * avg_loss - 0.15  # 0.15% cost
                ev = ev / 100  # Convert to decimal

                # Check thresholds
                if prob >= min_prob and ev >= min_ev:
                    entry_price = row.get('entry_price', 0)
                    exit_price = row.get(price_col, entry_price)

                    day_trades.append({
                        'trade': Trade(
                            ticker=row['ticker'],
                            entry_date=row['date'],
                            exit_date=row['date'] + timedelta(days=holding_period),
                            entry_price=float(entry_price) if entry_price else 0,
                            exit_price=float(exit_price) if exit_price else 0,
                            signal_type=f"AI_PROB_{prob:.0%}",
                            direction='LONG',
                            return_pct=float(ret),
                            holding_days=holding_period,
                            scores={
                                'ai_probability': prob,
                                'ai_ev': ev,
                                'sentiment': row.get('sentiment'),
                                'fundamental': row.get('fundamental_score'),
                                'total': row.get('total_score'),
                            }
                        ),
                        'prob': prob,
                        'ev': ev
                    })

            # Take top N trades by probability
            day_trades.sort(key=lambda x: x['prob'], reverse=True)
            for t in day_trades[:max_per_day]:
                trades.append(t['trade'])

                # Auto-log to performance tracker
                if tracker:
                    try:
                        tracker.log_recommendation(
                            ticker=t['trade'].ticker,
                            ai_probability=t['prob'],
                            ai_ev=t['ev'],
                            recommendation='BUY',
                            signal_scores=t['trade'].scores,
                            entry_price=t['trade'].entry_price
                        )
                    except Exception:
                        pass

        # Auto-update outcomes after backtest completes
        if tracker:
            try:
                tracker.update_outcomes(days_back=90)
            except Exception:
                pass

        return trades


# ===========================================================================
# FUNCTION TO ADD TO engine.py
# ===========================================================================
"""
Add this method to BacktestEngine class in engine.py:

    def _run_ai_strategy(self, df: pd.DataFrame, holding_period: int,
                         params: Dict[str, Any]) -> List[Trade]:
        '''AI probability-based strategy.'''
        from src.backtest.ai_strategy import AIBacktestStrategy
        
        strategy = AIBacktestStrategy()
        return strategy.run_strategy(df, holding_period, params)


And add this to the strategy dispatcher in run_backtest():

        elif strategy == 'ai_probability':
            trades = self._run_ai_strategy(df, holding_period, params)
"""


# ===========================================================================
# STRATEGY CONFIG TO ADD TO strategies.py
# ===========================================================================
"""
Add this to STRATEGIES dict in strategies.py:

    'ai_probability': StrategyConfig(
        name='ai_probability',
        display_name='🤖 AI Probability Strategy',
        description='Uses ML model to predict win probability. Only trades when probability >= threshold.',
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
        description='Conservative AI strategy: probability >= 60%, positive EV only.',
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
        description='Aggressive AI strategy: probability >= 50%, any EV.',
        strategy_type='ai_probability',
        default_params={
            'min_probability': 0.50,
            'min_ev': -0.01,
            'max_trades_per_day': 10
        },
        param_options={}
    ),
"""


# ===========================================================================
# TEST
# ===========================================================================
if __name__ == "__main__":
    # Test the AI strategy
    from src.ml.db_helper import get_engine

    engine = get_engine()
    query = """
        SELECT * FROM historical_scores 
        WHERE return_5d IS NOT NULL 
        ORDER BY score_date DESC 
        LIMIT 100
    """
    df = pd.read_sql(query, engine)

    strategy = AIBacktestStrategy()
    trades = strategy.run_strategy(df, holding_period=5, params={'min_probability': 0.55})

    print(f"\nAI Strategy Backtest Test")
    print(f"Total signals: {len(df)}")
    print(f"Trades taken: {len(trades)}")

    if trades:
        wins = sum(1 for t in trades if t.return_pct > 0)
        avg_ret = np.mean([t.return_pct for t in trades])
        print(f"Win rate: {wins/len(trades):.1%}")
        print(f"Avg return: {avg_ret:+.2f}%")