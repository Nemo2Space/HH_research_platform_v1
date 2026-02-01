"""
Alpha Platform - AI Learning Module

This module makes the AI smarter by:
1. Learning from historical signal performance
2. Providing context about what worked in the past
3. Adjusting confidence based on sector/signal patterns
4. Building prompts with historical insights

Usage:
    from src.ai.learning import SignalLearner

    learner = SignalLearner()

    # Get historical context for a ticker
    context = learner.get_historical_context('AAPL', 'Technology')

    # Adjust signal confidence based on historical performance
    adjusted_signal = learner.adjust_signal_confidence(signal, 'Technology')

    # Build enhanced prompt with historical insights
    prompt = learner.build_enhanced_prompt(ticker, scores, sector)
"""

import os
import sys
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from src.db.connection import get_connection
from src.db.repository import Repository
from src.utils.logging import get_logger
from datetime import datetime, timedelta

logger = get_logger(__name__)


@dataclass
class HistoricalPattern:
    """Historical performance pattern for a sector/signal combination."""
    sector: str
    signal_type: str
    count: int
    avg_return_1d: float
    avg_return_5d: float
    avg_return_10d: float
    avg_return_20d: float
    accuracy: float
    confidence: float  # How confident we should be in this signal


@dataclass
class SignalAdjustment:
    """Adjustment to apply to a signal based on historical learning."""
    original_signal: str
    adjusted_signal: str
    confidence_multiplier: float
    reason: str


class SignalLearner:
    """
    AI Learning module that uses historical data to improve signal generation.
    """

    def __init__(self, repository: Optional[Repository] = None):
        self.repo = repository or Repository()
        self._cache = {}
        self._cache_time = None
        self._cache_ttl = 3600  # 1 hour cache

    def _refresh_cache_if_needed(self):
        """Refresh performance cache if stale."""
        now = datetime.now()
        if self._cache_time is None or (now - self._cache_time).seconds > self._cache_ttl:
            self._load_performance_data()
            self._cache_time = now

    def _load_performance_data(self):
        """Load historical performance data into cache."""
        query = """
                SELECT sector, \
                       signal_type, \
                       COUNT(*) as count,
                AVG(return_1d) as avg_return_1d,
                AVG(return_5d) as avg_return_5d,
                AVG(return_10d) as avg_return_10d,
                AVG(return_20d) as avg_return_20d,
                SUM(CASE WHEN signal_correct = TRUE THEN 1 ELSE 0 END)::float / 
                    NULLIF(SUM(CASE WHEN signal_correct IS NOT NULL THEN 1 ELSE 0 END), 0) as accuracy
                FROM historical_scores
                WHERE return_10d IS NOT NULL
                  AND sector IS NOT NULL
                  AND sector != '0.0'
                GROUP BY sector, signal_type
                HAVING COUNT (*) >= 3 \
                """

        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    rows = cur.fetchall()

            self._cache = {}
            for row in rows:
                sector, signal_type, count, r1d, r5d, r10d, r20d, accuracy = row

                # Calculate confidence based on sample size and accuracy
                sample_confidence = min(1.0, count / 50)  # Max confidence at 50+ samples
                accuracy_val = accuracy if accuracy else 0.5
                confidence = sample_confidence * (0.5 + accuracy_val * 0.5)

                key = (sector, signal_type)
                self._cache[key] = HistoricalPattern(
                    sector=sector,
                    signal_type=signal_type,
                    count=count,
                    avg_return_1d=r1d or 0,
                    avg_return_5d=r5d or 0,
                    avg_return_10d=r10d or 0,
                    avg_return_20d=r20d or 0,
                    accuracy=accuracy_val,
                    confidence=confidence
                )

            logger.info(f"Loaded {len(self._cache)} historical patterns")

        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
            self._cache = {}

    def get_backtest_insights(self) -> Dict[str, Any]:
        """
        Get insights from saved backtest results.

        Returns:
            Dict with best strategies, performance summary, recommendations
        """
        query = """
                SELECT strategy_name, \
                       holding_period, \
                       total_trades, \
                       win_rate, \
                       avg_return, \
                       sharpe_ratio, \
                       alpha, \
                       run_date
                FROM backtest_results
                WHERE total_trades >= 10
                ORDER BY sharpe_ratio DESC LIMIT 20 \
                """

        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    rows = cur.fetchall()

            if not rows:
                return {
                    'has_data': False,
                    'message': 'No backtest results available. Run backtests first.'
                }

            results = []
            for row in rows:
                results.append({
                    'strategy': row[0],
                    'holding_period': row[1],
                    'trades': row[2],
                    'win_rate': float(row[3]) if row[3] else 0,
                    'avg_return': float(row[4]) if row[4] else 0,
                    'sharpe': float(row[5]) if row[5] else 0,
                    'alpha': float(row[6]) if row[6] else 0,
                    'run_date': row[7].strftime('%Y-%m-%d') if row[7] else None
                })

            # Find best strategy
            best = results[0] if results else None

            # Calculate averages
            avg_win_rate = sum(r['win_rate'] for r in results) / len(results)
            avg_sharpe = sum(r['sharpe'] for r in results) / len(results)

            return {
                'has_data': True,
                'total_backtests': len(results),
                'best_strategy': best,
                'avg_win_rate': round(avg_win_rate, 4),
                'avg_sharpe': round(avg_sharpe, 4),
                'all_results': results,
                'recommendation': self._generate_recommendation(results)
            }

        except Exception as e:
            logger.error(f"Error getting backtest insights: {e}")
            return {'has_data': False, 'message': str(e)}

    def _generate_recommendation(self, results: List[Dict]) -> str:
        """Generate recommendation based on backtest results."""
        if not results:
            return "Run more backtests to generate recommendations."

        best = results[0]

        # Find best holding period
        holding_periods = {}
        for r in results:
            hp = r['holding_period']
            if hp not in holding_periods:
                holding_periods[hp] = []
            holding_periods[hp].append(r['sharpe'])

        best_hp = max(holding_periods.keys(),
                      key=lambda x: sum(holding_periods[x]) / len(holding_periods[x]))

        recommendation = f"Best strategy: {best['strategy']} "
        recommendation += f"(Sharpe: {best['sharpe']:.2f}, Win Rate: {best['win_rate']:.1%}). "
        recommendation += f"Optimal holding period: {best_hp} days."

        if best['sharpe'] > 1.0:
            recommendation += " Strong risk-adjusted returns."
        elif best['sharpe'] > 0.5:
            recommendation += " Moderate risk-adjusted returns."
        else:
            recommendation += " Consider refining strategy parameters."

        return recommendation

    def get_strategy_performance_by_signal(self, signal_type: str) -> Dict[str, Any]:
        """
        Get backtest performance for strategies that use a specific signal type.

        Args:
            signal_type: Signal type (e.g., 'BUY', 'STRONG_BUY')

        Returns:
            Dict with performance metrics for this signal
        """
        query = """
                SELECT strategy_name, \
                       holding_period, \
                       avg_return, \
                       win_rate, \
                       sharpe_ratio, \
                       returns_by_signal
                FROM backtest_results
                WHERE returns_by_signal LIKE %s
                ORDER BY sharpe_ratio DESC LIMIT 10 \
                """

        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (f'%{signal_type}%',))
                    rows = cur.fetchall()

            if not rows:
                return {
                    'signal_type': signal_type,
                    'has_data': False
                }

            results = []
            for row in rows:
                results.append({
                    'strategy': row[0],
                    'holding_period': row[1],
                    'avg_return': float(row[2]) if row[2] else 0,
                    'win_rate': float(row[3]) if row[3] else 0,
                    'sharpe': float(row[4]) if row[4] else 0,
                })

            return {
                'signal_type': signal_type,
                'has_data': True,
                'best_strategy': results[0] if results else None,
                'all_results': results
            }

        except Exception as e:
            logger.error(f"Error getting signal performance: {e}")
            return {'signal_type': signal_type, 'has_data': False}

    def get_pattern(self, sector: str, signal_type: str) -> Optional[HistoricalPattern]:
        """Get historical pattern for sector/signal combination."""
        self._refresh_cache_if_needed()
        return self._cache.get((sector, signal_type))

    def get_sector_insights(self, sector: str) -> Dict[str, Any]:
        """
        Get insights about how signals perform in a specific sector.

        Returns:
            Dict with best/worst signals, recommendations, etc.
        """
        self._refresh_cache_if_needed()

        sector_patterns = [p for (s, _), p in self._cache.items() if s == sector]

        if not sector_patterns:
            return {
                'sector': sector,
                'has_data': False,
                'message': f"No historical data available for {sector}"
            }

        # Sort by 10-day return
        sorted_by_return = sorted(sector_patterns, key=lambda p: p.avg_return_10d, reverse=True)

        # Best and worst signals
        best = sorted_by_return[0] if sorted_by_return else None
        worst = sorted_by_return[-1] if sorted_by_return else None

        # Calculate sector average
        total_samples = sum(p.count for p in sector_patterns)
        weighted_avg = sum(
            p.avg_return_10d * p.count for p in sector_patterns) / total_samples if total_samples > 0 else 0

        # Determine if sector is bullish/bearish
        if weighted_avg > 1.0:
            sector_bias = "bullish"
        elif weighted_avg < -1.0:
            sector_bias = "bearish"
        else:
            sector_bias = "neutral"

        return {
            'sector': sector,
            'has_data': True,
            'total_samples': total_samples,
            'sector_bias': sector_bias,
            'avg_10d_return': round(weighted_avg, 2),
            'best_signal': {
                'type': best.signal_type,
                'avg_return': round(best.avg_return_10d, 2),
                'accuracy': round(best.accuracy * 100, 1) if best.accuracy else None,
                'samples': best.count
            } if best else None,
            'worst_signal': {
                'type': worst.signal_type,
                'avg_return': round(worst.avg_return_10d, 2),
                'accuracy': round(worst.accuracy * 100, 1) if worst.accuracy else None,
                'samples': worst.count
            } if worst else None,
            'all_signals': [
                {
                    'type': p.signal_type,
                    'avg_return': round(p.avg_return_10d, 2),
                    'accuracy': round(p.accuracy * 100, 1) if p.accuracy else None,
                    'samples': p.count
                }
                for p in sorted_by_return
            ]
        }

    def adjust_signal_confidence(self, signal_type: str, sector: str,
                                 base_confidence: float = 0.5) -> SignalAdjustment:
        """
        Adjust signal confidence based on historical performance AND backtest results.

        Args:
            signal_type: The generated signal (BUY, SELL, etc.)
            sector: The stock's sector
            base_confidence: Base confidence level (0-1)

        Returns:
            SignalAdjustment with adjusted confidence and reasoning
        """
        pattern = self.get_pattern(sector, signal_type)

        if not pattern:
            return SignalAdjustment(
                original_signal=signal_type,
                adjusted_signal=signal_type,
                confidence_multiplier=1.0,
                reason=f"No historical data for {signal_type} in {sector}"
            )

        # Calculate confidence multiplier
        # Based on: historical accuracy, sample size, return direction

        is_bullish_signal = signal_type in ['STRONG_BUY', 'BUY', 'WEAK_BUY']
        historical_positive = pattern.avg_return_10d > 0

        # Signal alignment check
        if is_bullish_signal and historical_positive:
            alignment_bonus = 0.2
            alignment_reason = "historically profitable"
        elif not is_bullish_signal and not historical_positive:
            alignment_bonus = 0.2
            alignment_reason = "historically accurate"
        elif is_bullish_signal and not historical_positive:
            alignment_bonus = -0.3
            alignment_reason = "historically underperforms"
        else:
            alignment_bonus = -0.2
            alignment_reason = "historically contrarian"

        # Accuracy bonus
        accuracy_bonus = (pattern.accuracy - 0.5) * 0.4 if pattern.accuracy else 0

        # Sample size factor
        sample_factor = min(1.0, pattern.count / 30)

        # Calculate final multiplier
        multiplier = 1.0 + (alignment_bonus + accuracy_bonus) * sample_factor
        multiplier = max(0.3, min(1.5, multiplier))  # Clamp between 0.3 and 1.5

        # Determine if we should adjust the signal
        adjusted_signal = signal_type
        if multiplier < 0.6 and is_bullish_signal:
            # Downgrade bullish signal
            if signal_type == 'STRONG_BUY':
                adjusted_signal = 'BUY'
            elif signal_type == 'BUY':
                adjusted_signal = 'WEAK_BUY'
            elif signal_type == 'WEAK_BUY':
                adjusted_signal = 'NEUTRAL'

        reason = (
            f"{sector} {signal_type}: {alignment_reason} "
            f"(avg return: {pattern.avg_return_10d:.1f}%, "
            f"accuracy: {pattern.accuracy * 100:.0f}%, "
            f"samples: {pattern.count})"
        )

        # ----- NEW: Adjust based on backtest insights -----
        try:
            backtest_insights = self.get_backtest_insights()
            if backtest_insights.get('has_data') and backtest_insights.get('best_strategy'):
                best = backtest_insights['best_strategy']

                # Boost or reduce confidence based on backtest Sharpe
                if best['sharpe'] > 1.0:
                    multiplier = min(1.5, multiplier * 1.1)
                    reason += f" | Backtest confirms (Sharpe {best['sharpe']:.2f})"
                elif best['sharpe'] < 0.3 and best['trades'] >= 50:
                    multiplier = max(0.3, multiplier * 0.9)
                    reason += f" | Backtest caution (Sharpe {best['sharpe']:.2f})"
        except Exception as e:
            logger.debug(f"Backtest insights unavailable: {e}")
        # ----- END NEW -----

        return SignalAdjustment(
            original_signal=signal_type,
            adjusted_signal=adjusted_signal,
            confidence_multiplier=round(multiplier, 2),
            reason=reason
        )

    def get_similar_historical_signals(self, ticker: str, sector: str,
                                       sentiment: int, fundamental: int,
                                       limit: int = 5) -> List[Dict]:
        """
        Find similar historical signals and their outcomes.

        This is RAG-style retrieval for the AI to learn from similar past situations.
        """
        query = """
                SELECT ticker, \
                       score_date, \
                       sector, \
                       sentiment, \
                       fundamental_score, \
                       total_score, \
                       signal_type, \
                       return_5d, \
                       return_10d, \
                       return_20d, \
                       signal_correct
                FROM historical_scores
                WHERE sector = %(sector)s
                  AND return_10d IS NOT NULL
                  AND ABS(sentiment - %(sentiment)s) <= 15
                  AND ABS(fundamental_score - %(fundamental)s) <= 15
                ORDER BY ABS(sentiment - %(sentiment)s) + ABS(fundamental_score - %(fundamental)s), \
                         score_date DESC
                    LIMIT %(limit)s \
                """

        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, {
                        'sector': sector,
                        'sentiment': sentiment,
                        'fundamental': fundamental,
                        'limit': limit
                    })
                    rows = cur.fetchall()

            results = []
            for row in rows:
                results.append({
                    'ticker': row[0],
                    'date': row[1].strftime('%Y-%m-%d') if row[1] else None,
                    'sector': row[2],
                    'sentiment': row[3],
                    'fundamental': row[4],
                    'total_score': float(row[5]) if row[5] else None,
                    'signal': row[6],
                    'return_5d': round(float(row[7]), 2) if row[7] else None,
                    'return_10d': round(float(row[8]), 2) if row[8] else None,
                    'return_20d': round(float(row[9]), 2) if row[9] else None,
                    'was_correct': row[10]
                })

            return results

        except Exception as e:
            logger.error(f"Error finding similar signals: {e}")
            return []

    def build_enhanced_prompt(self, ticker: str, scores: Dict[str, Any],
                              sector: str) -> str:
        """
        Build an enhanced prompt for the LLM that includes historical insights.

        This makes the AI smarter by giving it context about what worked in the past.
        """
        # Get sector insights
        insights = self.get_sector_insights(sector)

        # Get similar historical signals
        similar = self.get_similar_historical_signals(
            ticker=ticker,
            sector=sector,
            sentiment=scores.get('sentiment_score', 50),
            fundamental=scores.get('fundamental_score', 50),
            limit=5
        )

        # Build the context section
        context_parts = []

        # Sector performance context
        if insights.get('has_data'):
            context_parts.append(f"""
HISTORICAL SECTOR PERFORMANCE ({sector}):
- Overall sector bias: {insights['sector_bias'].upper()}
- Average 10-day return: {insights['avg_10d_return']}%
- Best performing signal: {insights['best_signal']['type']} ({insights['best_signal']['avg_return']}% avg return, {insights['best_signal']['accuracy']}% accuracy)
- Worst performing signal: {insights['worst_signal']['type']} ({insights['worst_signal']['avg_return']}% avg return)
- Total historical samples: {insights['total_samples']}
""")

        # Similar historical cases
        if similar:
            context_parts.append("\nSIMILAR HISTORICAL CASES:")
            for s in similar[:3]:
                outcome = "✓" if s['was_correct'] else "✗" if s['was_correct'] is False else "?"
                context_parts.append(
                    f"- {s['ticker']} ({s['date']}): Sentiment={s['sentiment']}, "
                    f"Fundamental={s['fundamental']} → {s['signal']} → "
                    f"{s['return_10d']:+.1f}% in 10d {outcome}"
                )

        # Key learnings
        context_parts.append(f"""
KEY LEARNINGS FOR {sector}:
""")

        if insights.get('has_data'):
            for sig in insights.get('all_signals', [])[:4]:
                direction = "↑" if sig['avg_return'] > 0 else "↓"
                context_parts.append(
                    f"- {sig['type']}: {direction} {sig['avg_return']:+.1f}% "
                    f"(accuracy: {sig['accuracy']}%, n={sig['samples']})"
                )

        return "\n".join(context_parts)

    def get_signal_recommendation(self, ticker: str, scores: Dict[str, Any],
                                  sector: str, proposed_signal: str) -> Dict[str, Any]:
        """
        Get AI-enhanced signal recommendation with historical context.

        Returns:
            Dict with:
            - recommended_signal: The AI-adjusted signal
            - confidence: Confidence level (0-1)
            - reasoning: Explanation of the recommendation
            - historical_context: What we learned from history
        """
        # Get adjustment based on historical performance
        adjustment = self.adjust_signal_confidence(proposed_signal, sector)

        # Get sector insights
        insights = self.get_sector_insights(sector)

        # Get similar cases
        similar = self.get_similar_historical_signals(
            ticker=ticker,
            sector=sector,
            sentiment=scores.get('sentiment_score', 50),
            fundamental=scores.get('fundamental_score', 50),
            limit=5
        )

        # Calculate success rate of similar cases
        if similar:
            correct = sum(1 for s in similar if s['was_correct'] is True)
            total_with_outcome = sum(1 for s in similar if s['was_correct'] is not None)
            similar_success_rate = correct / total_with_outcome if total_with_outcome > 0 else 0.5
        else:
            similar_success_rate = 0.5

        # Build reasoning
        reasoning_parts = [adjustment.reason]

        if similar:
            avg_return = sum(s['return_10d'] for s in similar if s['return_10d']) / len(similar)
            reasoning_parts.append(
                f"Similar historical cases averaged {avg_return:+.1f}% return "
                f"with {similar_success_rate * 100:.0f}% success rate."
            )

        if insights.get('has_data'):
            if insights['sector_bias'] == 'bullish' and 'BUY' in proposed_signal:
                reasoning_parts.append(f"Sector is historically bullish, supporting the signal.")
            elif insights['sector_bias'] == 'bearish' and 'SELL' in proposed_signal:
                reasoning_parts.append(f"Sector is historically bearish, supporting the signal.")

        # Calculate final confidence
        base_confidence = 0.5
        confidence = base_confidence * adjustment.confidence_multiplier
        confidence = confidence * (0.7 + similar_success_rate * 0.3)
        confidence = min(0.95, max(0.1, confidence))

        return {
            'ticker': ticker,
            'sector': sector,
            'proposed_signal': proposed_signal,
            'recommended_signal': adjustment.adjusted_signal,
            'confidence': round(confidence, 2),
            'confidence_multiplier': adjustment.confidence_multiplier,
            'reasoning': " ".join(reasoning_parts),
            'sector_bias': insights.get('sector_bias', 'unknown'),
            'similar_cases_count': len(similar),
            'similar_success_rate': round(similar_success_rate, 2) if similar else None,
            'historical_context': self.build_enhanced_prompt(ticker, scores, sector)
        }


def test_learning():
    """Test the learning module."""
    learner = SignalLearner()

    # Test sector insights
    print("\n" + "=" * 60)
    print("TECHNOLOGY SECTOR INSIGHTS")
    print("=" * 60)
    insights = learner.get_sector_insights('Technology')
    print(f"Sector bias: {insights.get('sector_bias', 'unknown')}")
    print(f"Avg 10d return: {insights.get('avg_10d_return', 'N/A')}%")
    if insights.get('best_signal'):
        print(f"Best signal: {insights['best_signal']['type']} ({insights['best_signal']['avg_return']}%)")

    # Test signal adjustment
    print("\n" + "=" * 60)
    print("SIGNAL ADJUSTMENTS")
    print("=" * 60)
    for signal in ['STRONG_BUY', 'BUY', 'WEAK_BUY', 'NEUTRAL', 'WEAK_SELL', 'SELL']:
        adj = learner.adjust_signal_confidence(signal, 'Technology')
        print(f"{signal:12} → {adj.adjusted_signal:12} (x{adj.confidence_multiplier})")

    # Test recommendation
    print("\n" + "=" * 60)
    print("FULL RECOMMENDATION FOR AAPL")
    print("=" * 60)
    scores = {
        'sentiment_score': 70,
        'fundamental_score': 75,
        'total_score': 65
    }
    rec = learner.get_signal_recommendation('AAPL', scores, 'Technology', 'BUY')
    print(f"Proposed: {rec['proposed_signal']}")
    print(f"Recommended: {rec['recommended_signal']}")
    print(f"Confidence: {rec['confidence']}")
    print(f"Reasoning: {rec['reasoning']}")
    print(f"\nHistorical Context:")
    print(rec['historical_context'])


if __name__ == "__main__":
    test_learning()