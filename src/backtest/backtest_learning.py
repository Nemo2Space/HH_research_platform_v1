"""
Backtest Learning Integration

Connects backtest results to AI learning:
1. Creates backtest_results table
2. Auto-saves after each backtest
3. Provides query interface for AI to learn from historical backtests

Location: src/backtest/backtest_learning.py
"""

import json
from datetime import datetime, date
from typing import Dict, List, Optional, Any
import pandas as pd
from sqlalchemy import text

try:
    from src.ml.db_helper import get_engine, get_connection
except ImportError:
    from src.db.connection import get_connection, get_engine

from src.utils.logging import get_logger

logger = get_logger(__name__)


class BacktestLearning:
    """
    Manages backtest results for AI learning.

    - Stores backtest performance in database
    - Provides query interface for AI to learn from past backtests
    - Tracks which strategies perform best under which conditions
    """

    def __init__(self):
        self.engine = get_engine()
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Create backtest_results table if not exists."""
        create_sql = """
        CREATE TABLE IF NOT EXISTS backtest_results (
            id SERIAL PRIMARY KEY,
            strategy_name VARCHAR(100) NOT NULL,
            run_date DATE NOT NULL,
            start_date DATE,
            end_date DATE,
            holding_period INT,
            benchmark VARCHAR(10),
            total_trades INT,
            winning_trades INT,
            losing_trades INT,
            win_rate DECIMAL(5,4),
            avg_return DECIMAL(8,4),
            total_return DECIMAL(10,4),
            sharpe_ratio DECIMAL(6,4),
            sortino_ratio DECIMAL(6,4),
            max_drawdown DECIMAL(8,4),
            alpha DECIMAL(8,4),
            benchmark_return DECIMAL(8,4),
            returns_by_signal JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON backtest_results(strategy_name);
        CREATE INDEX IF NOT EXISTS idx_backtest_date ON backtest_results(run_date);
        """

        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_sql)
                conn.commit()
            logger.debug("Backtest results table ready")
        except Exception as e:
            logger.warning(f"Could not create backtest_results table: {e}")

    def save_result(self, result) -> bool:
        """
        Save a backtest result to the database.

        Args:
            result: BacktestResult object from engine.py

        Returns:
            True if saved successfully
        """
        insert_sql = """
        INSERT INTO backtest_results (
            strategy_name, run_date, start_date, end_date,
            holding_period, benchmark, total_trades, winning_trades,
            losing_trades, win_rate, avg_return, total_return,
            sharpe_ratio, sortino_ratio, max_drawdown, alpha,
            benchmark_return, returns_by_signal
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """

        try:
            # Convert numpy/pandas types to Python native
            returns_by_signal = {}
            if hasattr(result, 'returns_by_signal') and result.returns_by_signal:
                for k, v in result.returns_by_signal.items():
                    if isinstance(v, dict):
                        returns_by_signal[k] = {
                            kk: float(vv) if hasattr(vv, '__float__') else vv
                            for kk, vv in v.items()
                        }
                    else:
                        returns_by_signal[k] = float(v) if hasattr(v, '__float__') else v

            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_sql, (
                        result.strategy_name,
                        date.today(),
                        result.start_date,
                        result.end_date,
                        result.holding_period,
                        result.benchmark,
                        int(result.total_trades),
                        int(getattr(result, 'winning_trades', 0)),
                        int(getattr(result, 'losing_trades', 0)),
                        float(result.win_rate),
                        float(result.avg_return),
                        float(getattr(result, 'total_return', 0)),
                        float(result.sharpe_ratio),
                        float(getattr(result, 'sortino_ratio', 0)),
                        float(getattr(result, 'max_drawdown', 0)),
                        float(result.alpha),
                        float(getattr(result, 'benchmark_return', 0)),
                        json.dumps(returns_by_signal)
                    ))
                conn.commit()

            logger.info(f"Saved backtest result: {result.strategy_name} "
                       f"({result.total_trades} trades, {result.win_rate:.1%} win rate)")
            return True

        except Exception as e:
            logger.error(f"Failed to save backtest result: {e}")
            return False

    def get_strategy_performance(self, strategy_name: str = None,
                                  days_back: int = 90) -> pd.DataFrame:
        """
        Get historical backtest performance for strategies.

        Args:
            strategy_name: Filter by strategy (None for all)
            days_back: How far back to look

        Returns:
            DataFrame with backtest results
        """
        base_query = """
        SELECT 
            strategy_name,
            run_date,
            holding_period,
            total_trades,
            win_rate,
            avg_return,
            sharpe_ratio,
            alpha
        FROM backtest_results
        WHERE run_date >= CURRENT_DATE - INTERVAL ':days days'
        """

        if strategy_name:
            base_query += " AND strategy_name = :strategy"
            base_query += " ORDER BY run_date DESC"
            query = text(base_query)
            params = {"days": days_back, "strategy": strategy_name}
        else:
            base_query += " ORDER BY run_date DESC"
            query = text(base_query.replace(":days", str(days_back)))
            params = {}

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, params)
                return pd.DataFrame(result.fetchall(), columns=result.keys())
        except Exception as e:
            logger.error(f"Failed to get strategy performance: {e}")
            return pd.DataFrame()

    def get_best_strategies(self, metric: str = 'sharpe_ratio',
                            top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the best performing strategies by a given metric.

        Args:
            metric: 'win_rate', 'avg_return', 'sharpe_ratio', 'alpha'
            top_n: Number of top strategies to return

        Returns:
            List of strategy performance dicts
        """
        valid_metrics = ['win_rate', 'avg_return', 'sharpe_ratio', 'alpha', 'total_return']
        if metric not in valid_metrics:
            metric = 'sharpe_ratio'

        query = text(f"""
        SELECT 
            strategy_name,
            COUNT(*) as run_count,
            AVG(win_rate) as avg_win_rate,
            AVG(avg_return) as avg_return,
            AVG(sharpe_ratio) as avg_sharpe,
            AVG(alpha) as avg_alpha,
            SUM(total_trades) as total_trades
        FROM backtest_results
        WHERE run_date >= CURRENT_DATE - INTERVAL '90 days'
        GROUP BY strategy_name
        HAVING COUNT(*) >= 1
        ORDER BY AVG({metric}) DESC
        LIMIT :top_n
        """)

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {"top_n": top_n})
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Failed to get best strategies: {e}")
            return []

    def get_ai_learning_context(self) -> Dict[str, Any]:
        """
        Get summarized backtest data for AI context.

        This is called by LLM integration to inform AI decisions.
        Returns a dict that can be added to AI prompts.
        """
        context = {
            'backtest_available': False,
            'best_strategies': [],
            'ai_vs_signal_comparison': {},
            'recommendations': []
        }

        try:
            # Get best strategies
            best = self.get_best_strategies('sharpe_ratio', 5)
            if best:
                context['backtest_available'] = True
                context['best_strategies'] = best

            # Compare AI vs signal-based strategies
            ai_query = text("""
            SELECT 
                'AI Strategies' as category,
                AVG(win_rate) as win_rate,
                AVG(avg_return) as avg_return,
                AVG(sharpe_ratio) as sharpe,
                SUM(total_trades) as trades
            FROM backtest_results
            WHERE strategy_name LIKE 'ai_%'
              AND run_date >= CURRENT_DATE - INTERVAL '90 days'
            """)

            signal_query = text("""
            SELECT 
                'Signal Strategies' as category,
                AVG(win_rate) as win_rate,
                AVG(avg_return) as avg_return,
                AVG(sharpe_ratio) as sharpe,
                SUM(total_trades) as trades
            FROM backtest_results
            WHERE strategy_name NOT LIKE 'ai_%'
              AND run_date >= CURRENT_DATE - INTERVAL '90 days'
            """)

            with self.engine.connect() as conn:
                ai_result = conn.execute(ai_query)
                ai_df = pd.DataFrame(ai_result.fetchall(), columns=ai_result.keys())

                signal_result = conn.execute(signal_query)
                signal_df = pd.DataFrame(signal_result.fetchall(), columns=signal_result.keys())

            if not ai_df.empty and ai_df['trades'].iloc[0]:
                context['ai_vs_signal_comparison']['ai'] = {
                    'win_rate': float(ai_df['win_rate'].iloc[0] or 0),
                    'avg_return': float(ai_df['avg_return'].iloc[0] or 0),
                    'sharpe': float(ai_df['sharpe'].iloc[0] or 0),
                    'trades': int(ai_df['trades'].iloc[0] or 0),
                }

            if not signal_df.empty and signal_df['trades'].iloc[0]:
                context['ai_vs_signal_comparison']['signal'] = {
                    'win_rate': float(signal_df['win_rate'].iloc[0] or 0),
                    'avg_return': float(signal_df['avg_return'].iloc[0] or 0),
                    'sharpe': float(signal_df['sharpe'].iloc[0] or 0),
                    'trades': int(signal_df['trades'].iloc[0] or 0),
                }

            # Generate recommendations
            if best:
                top_strategy = best[0]
                context['recommendations'].append(
                    f"Best strategy: {top_strategy['strategy_name']} "
                    f"(Sharpe: {top_strategy['avg_sharpe']:.2f}, "
                    f"Win Rate: {top_strategy['avg_win_rate']:.1%})"
                )

        except Exception as e:
            logger.warning(f"Failed to get AI learning context: {e}")

        return context

    def format_for_llm(self) -> str:
        """
        Format backtest learning data for LLM prompt.

        Returns a string that can be included in AI analysis.
        """
        context = self.get_ai_learning_context()

        if not context['backtest_available']:
            return "No backtest history available yet."

        lines = ["## Backtest Performance History"]

        # Best strategies
        if context['best_strategies']:
            lines.append("\n**Top Strategies (by Sharpe Ratio):**")
            for i, s in enumerate(context['best_strategies'][:3], 1):
                lines.append(
                    f"{i}. {s['strategy_name']}: "
                    f"Sharpe {s['avg_sharpe']:.2f}, "
                    f"Win Rate {s['avg_win_rate']:.1%}, "
                    f"Avg Return {s['avg_return']:.2f}%"
                )

        # AI vs Signal comparison
        comp = context.get('ai_vs_signal_comparison', {})
        if 'ai' in comp and 'signal' in comp:
            ai = comp['ai']
            sig = comp['signal']
            lines.append("\n**AI vs Signal-Based Strategies:**")
            lines.append(f"- AI: {ai['win_rate']:.1%} win rate, {ai['sharpe']:.2f} Sharpe ({ai['trades']} trades)")
            lines.append(f"- Signal: {sig['win_rate']:.1%} win rate, {sig['sharpe']:.2f} Sharpe ({sig['trades']} trades)")

            if ai['sharpe'] > sig['sharpe']:
                lines.append("- **AI strategies outperforming signal-based**")
            else:
                lines.append("- Signal strategies currently outperforming AI")

        return "\n".join(lines)


# =============================================================================
# AUTO-SAVE INTEGRATION
# =============================================================================

def auto_save_backtest_result(result) -> bool:
    """
    Helper function to auto-save backtest results.

    Call this after running a backtest:
        from src.backtest.backtest_learning import auto_save_backtest_result
        result = engine.run_backtest(...)
        auto_save_backtest_result(result)
    """
    try:
        learning = BacktestLearning()
        return learning.save_result(result)
    except Exception as e:
        logger.warning(f"Failed to auto-save backtest: {e}")
        return False


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    learning = BacktestLearning()

    if len(sys.argv) > 1:
        if sys.argv[1] == '--best':
            print("\n📊 Best Performing Strategies")
            print("=" * 60)
            best = learning.get_best_strategies('sharpe_ratio', 10)
            for i, s in enumerate(best, 1):
                print(f"{i}. {s['strategy_name']}")
                print(f"   Win Rate: {s['avg_win_rate']:.1%}")
                print(f"   Avg Return: {s['avg_return']:.2f}%")
                print(f"   Sharpe: {s['avg_sharpe']:.2f}")
                print(f"   Trades: {s['total_trades']}")
                print()

        elif sys.argv[1] == '--context':
            print("\n📊 AI Learning Context")
            print("=" * 60)
            print(learning.format_for_llm())

        elif sys.argv[1] == '--history':
            df = learning.get_strategy_performance(days_back=90)
            if not df.empty:
                print("\n📊 Recent Backtest History")
                print(df.to_string(index=False))
            else:
                print("No backtest history found")
    else:
        print("Usage:")
        print("  python -m src.backtest.backtest_learning --best     # Show best strategies")
        print("  python -m src.backtest.backtest_learning --context  # Show AI context")
        print("  python -m src.backtest.backtest_learning --history  # Show history")