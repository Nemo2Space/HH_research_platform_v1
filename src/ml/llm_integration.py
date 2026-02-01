"""
LLM Integration Layer - Phase 6

Qwen integration with tool-enforced computation.
CRITICAL: LLM never invents statistics - all numbers come from tools.

Location: src/ml/llm_integration.py
"""

import os
import json
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradeAnalysis:
    """Complete trade analysis from LLM."""
    ticker: str
    summary: str
    recommendation: str  # BUY, SELL, HOLD, SKIP
    confidence_qualifier: str  # HIGH, MEDIUM, LOW
    bullish_factors: List[str]
    bearish_factors: List[str]
    risks: List[str]
    thesis_breakers: List[str]
    entry_reasoning: str
    exit_reasoning: str
    position_size_reasoning: str


class LLMIntegration:
    """
    LLM integration for trade analysis.

    Key principle: LLM explains computed outputs, never computes itself.
    All statistics come from tools/data, not LLM invention.
    """

    SYSTEM_PROMPT = """You are a quantitative trading analyst assistant.
    
CRITICAL RULES:
1. NEVER invent or estimate statistics. All numbers must come from the provided data.
2. Only reference numbers that appear in the context.
3. If data is missing, say "data not available" rather than estimating.
4. Be concise and direct.

Your role:
1. Explain ML model outputs in plain language
2. Summarize the trading setup
3. Identify key risks from the data
4. Provide actionable insights
"""

    def __init__(self, llm_backend: str = None):
        self.llm_backend = llm_backend or os.environ.get('LLM_DEFAULT_BACKEND', 'qwen')
        self.llm = self._init_llm()
        self._backtest_context_cache = None
        self._backtest_context_time = None

    def _init_llm(self):
        """Initialize LLM backend."""
        try:
            if self.llm_backend == 'qwen':
                from src.ai.qwen_client import QwenClient
                return QwenClient()
        except ImportError:
            pass
        return None

    def _get_backtest_context(self) -> str:
        """Get backtest learning context for AI analysis."""
        from datetime import timedelta
        now = datetime.now()

        # Cache for 10 minutes
        if (self._backtest_context_cache and self._backtest_context_time and
            now - self._backtest_context_time < timedelta(minutes=10)):
            return self._backtest_context_cache

        try:
            from src.backtest.backtest_learning import BacktestLearning
            learning = BacktestLearning()
            self._backtest_context_cache = learning.format_for_llm()
            self._backtest_context_time = now
            return self._backtest_context_cache
        except Exception as e:
            logger.debug(f"Could not get backtest context: {e}")
            return ""

    def analyze_trade(self,
                      ticker: str,
                      ml_prediction: Dict,
                      decision_result: Dict,
                      similar_setups: Dict,
                      scores: Dict,
                      market_context: Dict = None) -> TradeAnalysis:
        """
        Generate comprehensive trade analysis.

        Args:
            ticker: Stock symbol
            ml_prediction: ML model outputs (prob_win_5d, ev_5d, etc.)
            decision_result: Decision layer output (approved, rejection_reasons, etc.)
            similar_setups: RAG memory results (win_rate, avg_return, etc.)
            scores: Signal scores (sentiment, fundamental, etc.)
            market_context: Optional market data (vix, days_to_earnings, etc.)
        """
        market_context = market_context or {}

        # Build analysis context
        context = self._build_analysis_context(
            ticker, ml_prediction, decision_result, similar_setups, scores, market_context
        )

        # Try LLM first, fallback to rule-based
        if self.llm:
            try:
                response = self._call_llm(context)
                return self._parse_analysis(ticker, response, ml_prediction, decision_result)
            except Exception as e:
                logger.warning(f"LLM call failed: {e}, using rule-based analysis")

        # Rule-based analysis (data-driven, no hardcoding)
        analysis_json = self._generate_data_driven_analysis(
            ticker, ml_prediction, decision_result, similar_setups, scores, market_context
        )

        return self._parse_analysis(ticker, analysis_json, ml_prediction, decision_result)

    def _build_analysis_context(self, ticker, ml_pred, decision, similar, scores, market_ctx) -> str:
        """Build context for LLM with all computed data."""

        # Get backtest learning context
        backtest_ctx = self._get_backtest_context()

        context = f"""
TRADE ANALYSIS REQUEST: {ticker}
{'='*50}

ML MODEL OUTPUT:
- Win Probability (5-day): {ml_pred.get('prob_win_5d', 0.5):.1%}
- Expected Value: {ml_pred.get('ev_5d', 0)*100:.2f}%
- Confidence: {ml_pred.get('confidence', 'MEDIUM')}

DECISION LAYER:
- Approved: {decision.get('approved', False)}
- Rejection Reasons: {', '.join(decision.get('rejection_reasons', [])) or 'None'}

SIMILAR HISTORICAL SETUPS:
- Count: {similar.get('closed_count', 0)}
- Win Rate: {similar.get('win_rate', 0):.1%}
- Avg Return: {similar.get('avg_return', 0):.2f}%

SIGNAL SCORES:
- Sentiment: {scores.get('sentiment_score', 50):.0f}
- Fundamental: {scores.get('fundamental_score', 50):.0f}
- Technical: {scores.get('technical_score', 50):.0f}
- Options Flow: {scores.get('options_flow_score', 50):.0f}
- Total: {scores.get('total_score', 50):.0f}

MARKET CONTEXT:
- VIX: {market_ctx.get('vix', 'N/A')}
- Days to Earnings: {market_ctx.get('days_to_earnings', 'N/A')}

{backtest_ctx}

Based on the above data, provide:
1. A brief summary (1-2 sentences)
2. Bullish factors from the data
3. Bearish factors from the data
4. Key risks (ONLY those supported by the data)
5. What would invalidate this thesis
"""
        return context

    def _call_llm(self, context: str) -> str:
        """Call actual LLM with context."""
        prompt = f"{self.SYSTEM_PROMPT}\n\n{context}"
        response = self.llm.generate(prompt, max_tokens=1000)
        return response

    def _generate_data_driven_analysis(self, ticker: str, ml_prediction: Dict,
                                       decision_result: Dict, similar_setups: Dict,
                                       scores: Dict, market_context: Dict) -> str:
        """
        Generate analysis based purely on provided data.
        NO hardcoded values - all risks/factors derived from data.
        """
        prob = ml_prediction.get('prob_win_5d', 0.5)
        ev = ml_prediction.get('ev_5d', 0)
        wr = similar_setups.get('win_rate', 0)
        similar_count = similar_setups.get('closed_count', 0)
        approved = decision_result.get('approved', False)
        rejection_reasons = decision_result.get('rejection_reasons', [])

        # Determine recommendation based on data
        if approved and prob >= 0.65 and wr >= 0.60:
            recommendation = "BUY"
            confidence = "HIGH"
            summary = f"{ticker} shows strong setup with {prob:.0%} win probability."
            if similar_count > 0:
                summary += f" Similar setups had {wr:.0%} win rate."
        elif approved and prob >= 0.55:
            recommendation = "BUY"
            confidence = "MEDIUM"
            summary = f"{ticker} shows moderate setup with {prob:.0%} win probability. Consider smaller position."
        else:
            recommendation = "SKIP"
            confidence = "LOW"
            if rejection_reasons:
                reasons_str = ', '.join(rejection_reasons[:2])
                summary = f"{ticker} does not meet criteria. Rejected: {reasons_str}"
            else:
                summary = f"{ticker} does not meet criteria with {prob:.0%} probability."

        # ===== BULLISH FACTORS (data-driven) =====
        bullish = []

        if scores.get('sentiment_score', 50) >= 65:
            bullish.append(f"Strong sentiment ({scores['sentiment_score']:.0f})")

        if scores.get('options_flow_score', 50) >= 60:
            bullish.append(f"Bullish options flow ({scores['options_flow_score']:.0f})")

        if scores.get('fundamental_score', 50) >= 70:
            bullish.append(f"Strong fundamentals ({scores['fundamental_score']:.0f})")

        if scores.get('technical_score', 50) >= 65:
            bullish.append(f"Positive technicals ({scores['technical_score']:.0f})")

        if similar_count > 0 and wr >= 0.60:
            bullish.append(f"Similar setups: {wr:.0%} win rate ({similar_count} trades)")

        if prob >= 0.65:
            bullish.append(f"High ML probability ({prob:.0%})")

        if ev > 0.005:
            bullish.append(f"Positive expected value (+{ev*100:.2f}%)")

        # ===== BEARISH FACTORS (data-driven) =====
        bearish = []

        if scores.get('sentiment_score', 50) < 45:
            bearish.append(f"Weak sentiment ({scores.get('sentiment_score', 50):.0f})")

        if scores.get('options_flow_score', 50) < 45:
            bearish.append(f"Bearish options flow ({scores.get('options_flow_score', 50):.0f})")

        if similar_count > 0 and wr < 0.45:
            bearish.append(f"Similar setups struggled ({wr:.0%} win rate)")

        if prob < 0.50:
            bearish.append(f"Low ML probability ({prob:.0%})")

        if ev < 0:
            bearish.append(f"Negative expected value ({ev*100:.2f}%)")

        # Score disagreement
        component_scores = [
            scores.get('sentiment_score', 50),
            scores.get('fundamental_score', 50),
            scores.get('technical_score', 50),
            scores.get('options_flow_score', 50)
        ]
        import numpy as np
        if np.std(component_scores) > 15:
            bearish.append("Scores disagree")

        # ===== RISKS (data-driven, NOT hardcoded) =====
        risks = []

        # Earnings risk - ONLY if earnings is upcoming (positive days)
        days_to_earnings = market_context.get('days_to_earnings')
        if days_to_earnings is not None:
            if isinstance(days_to_earnings, (int, float)) and 0 < days_to_earnings <= 7:
                risks.append(f"Earnings in {int(days_to_earnings)} days - binary event risk")

        # VIX risk - only if VIX data provided and elevated
        vix = market_context.get('vix')
        if vix and vix > 25:
            risks.append(f"Elevated volatility (VIX: {vix:.0f})")

        # Limited similar data
        if similar_count < 5:
            risks.append(f"Limited historical data ({similar_count} similar setups)")

        # High variance in similar setups
        best_ret = similar_setups.get('best_return', 0)
        worst_ret = similar_setups.get('worst_return', 0)
        if similar_count > 0 and (best_ret - worst_ret) > 10:
            risks.append(f"High variance in similar setups ({worst_ret:+.1f}% to {best_ret:+.1f}%)")

        # Low confidence from ML model
        if ml_prediction.get('confidence') == 'LOW':
            risks.append("Low model confidence")

        # ===== THESIS BREAKERS (data-driven) =====
        thesis_breakers = []

        # Always relevant
        thesis_breakers.append("Price breaks below stop loss level")

        # Based on what's driving the signal
        if scores.get('sentiment_score', 50) >= 60:
            thesis_breakers.append("Sentiment score drops below 50")

        if scores.get('options_flow_score', 50) >= 55:
            thesis_breakers.append("Options flow turns bearish")

        if prob >= 0.55:
            thesis_breakers.append("ML probability drops below 50%")

        return json.dumps({
            'summary': summary,
            'recommendation': recommendation,
            'confidence': confidence,
            'bullish_factors': bullish,
            'bearish_factors': bearish,
            'risks': risks,
            'thesis_breakers': thesis_breakers
        })

    def _parse_analysis(self, ticker: str, analysis_text: str,
                        ml_prediction: Dict, decision_result: Dict) -> TradeAnalysis:
        """Parse analysis output into structured TradeAnalysis."""

        try:
            data = json.loads(analysis_text)
        except json.JSONDecodeError:
            data = {
                'summary': analysis_text[:500] if analysis_text else 'Analysis unavailable',
                'recommendation': 'REVIEW',
                'confidence': 'MEDIUM',
                'bullish_factors': [],
                'bearish_factors': [],
                'risks': [],
                'thesis_breakers': []
            }

        # Determine recommendation if not set
        if data.get('recommendation') == 'REVIEW':
            if decision_result.get('approved'):
                prob = ml_prediction.get('prob_win_5d', 0.5)
                data['recommendation'] = 'BUY' if prob >= 0.55 else 'HOLD'
            else:
                data['recommendation'] = 'SKIP'

        return TradeAnalysis(
            ticker=ticker,
            summary=data.get('summary', ''),
            recommendation=data.get('recommendation', 'REVIEW'),
            confidence_qualifier=data.get('confidence', 'MEDIUM'),
            bullish_factors=data.get('bullish_factors', []),
            bearish_factors=data.get('bearish_factors', []),
            risks=data.get('risks', []),
            thesis_breakers=data.get('thesis_breakers', []),
            entry_reasoning=data.get('entry_reasoning', ''),
            exit_reasoning=data.get('exit_reasoning', ''),
            position_size_reasoning=data.get('position_size_reasoning', '')
        )

    def generate_insights_for_context(self,
                                      ml_report: Dict,
                                      performance_summary: Dict,
                                      regime_info: Dict) -> str:
        """Generate insights to include in LLM context for chat."""

        insights = []

        # ML model insights
        if ml_report:
            auc = ml_report.get('mean_auc', 0.5)
            wr = ml_report.get('mean_win_rate', 0.5)
            insights.append(f"ML Model: AUC={auc:.3f}, Historical Win Rate={wr:.1%}")

            if ml_report.get('beats_baseline'):
                insights.append("Model beats baseline - signals have edge")
            else:
                insights.append("Model at baseline - signals may need improvement")

            # Top features
            fi = ml_report.get('feature_importance', {})
            if fi:
                top = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:3]
                top_str = ", ".join([f"{k}:{v:.0%}" for k, v in top])
                insights.append(f"Top features: {top_str}")

        # Recent performance
        if performance_summary:
            recent_wr = performance_summary.get('win_rate', 0)
            recent_ret = performance_summary.get('avg_return', 0)
            insights.append(f"Recent: {recent_wr:.0%} win rate, {recent_ret:+.2f}% avg return")

        # Regime
        if regime_info:
            vix = regime_info.get('vix', 20)
            insights.append(f"Current VIX: {vix:.0f}")

        return "\n".join(insights)