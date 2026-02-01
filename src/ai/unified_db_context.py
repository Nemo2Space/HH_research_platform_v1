"""
Unified Database Context for AI Chat

Provides comprehensive database access for the AI to answer any question.
This module can be imported into chat.py to enhance its capabilities.

Author: HH Research Platform
Version: 2024-01-01
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import re

from src.db.connection import get_connection, get_engine
from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# GENERAL RESEARCH KEYWORDS (for triggering database research)
# =============================================================================

RESEARCH_KEYWORDS = [
    # Best/Top queries
    'best stock', 'top stock', 'what to buy', 'should i buy', 'recommend',
    'highest score', 'strong buy', 'best opportunity', 'best pick',

    # Worst/Sell queries
    'worst stock', 'avoid', 'should i sell', 'weak', 'bearish', 'sell signal',

    # Analysis queries
    'momentum', 'technical', 'breakout', 'trend',
    'value', 'cheap', 'undervalued', 'pe ratio', 'fundamental',
    'option', 'unusual activity', 'smart money',

    # Sector queries
    'sector', 'industry', 'which sector', 'sector analysis',

    # General market
    'market', 'overview', 'summary', 'how is the market',

    # Comparison
    'compare', 'versus', 'vs', 'better than',

    # Screener
    'screen', 'filter', 'find stocks', 'show me',

    # Today/Now
    'today', 'right now', 'currently', 'at this moment'
]


def needs_research_context(question: str) -> bool:
    """
    Determine if the question needs comprehensive database research.
    """
    question_lower = question.lower()

    # Check for research keywords
    for kw in RESEARCH_KEYWORDS:
        if kw in question_lower:
            return True

    # Check for specific patterns
    patterns = [
        r'what.*(stock|buy|sell|trade)',
        r'which.*(stock|sector|industry)',
        r'give me.*(stocks|picks|ideas)',
        r'show me.*(best|top|worst)',
        r'find.*(stocks|opportunities)',
        r'list.*(stocks|buys|sells)',
    ]

    for pattern in patterns:
        if re.search(pattern, question_lower):
            return True

    return False


def get_comprehensive_context(question: str, limit: int = 15) -> str:
    """
    Build comprehensive database context for AI based on the question.

    This gives the AI access to ALL relevant data to answer any question.
    """
    context_parts = []
    question_lower = question.lower()

    try:
        engine = get_engine()

        # =================================================================
        # 1. MARKET OVERVIEW (Always include)
        # =================================================================
        overview_df = pd.read_sql("""
            SELECT 
                COUNT(*) as total_stocks,
                COUNT(*) FILTER (WHERE total_score >= 80) as strong_buys,
                COUNT(*) FILTER (WHERE total_score >= 65 AND total_score < 80) as buys,
                COUNT(*) FILTER (WHERE total_score > 35 AND total_score < 65) as holds,
                COUNT(*) FILTER (WHERE total_score <= 35 AND total_score > 20) as sells,
                COUNT(*) FILTER (WHERE total_score <= 20) as strong_sells,
                ROUND(AVG(total_score)::numeric, 1) as avg_score,
                ROUND(AVG(sentiment_score)::numeric, 1) as avg_sentiment,
                ROUND(AVG(technical_score)::numeric, 1) as avg_technical,
                MAX(date) as analysis_date
            FROM screener_scores
            WHERE date = (SELECT MAX(date) FROM screener_scores)
        """, engine)

        if not overview_df.empty:
            o = overview_df.iloc[0]
            context_parts.append(f"""
## MARKET OVERVIEW (as of {o['analysis_date']})
- Total stocks analyzed: {o['total_stocks']}
- Average score: {o['avg_score']}/100
- Average sentiment: {o['avg_sentiment']}/100
- Average technical: {o['avg_technical']}/100

Signal Distribution:
- 🟢 STRONG BUY (80+): {o['strong_buys']} stocks
- 🟢 BUY (65-79): {o['buys']} stocks  
- 🟡 HOLD (36-64): {o['holds']} stocks
- 🔴 SELL (21-35): {o['sells']} stocks
- 🔴 STRONG SELL (0-20): {o['strong_sells']} stocks
""")

        # =================================================================
        # 2. TOP BUY SIGNALS (for buy-related questions)
        # =================================================================
        if any(kw in question_lower for kw in ['buy', 'best', 'top', 'recommend', 'opportunity', 'today', 'now']):
            buy_df = pd.read_sql(f"""
                SELECT s.ticker, s.total_score, s.sentiment_score, s.fundamental_score, 
                       s.technical_score, s.options_flow_score, CASE WHEN s.total_score >= 80 THEN 'STRONG_BUY' WHEN s.total_score >= 65 THEN 'BUY' WHEN s.total_score <= 35 THEN 'SELL' ELSE 'HOLD' END as signal,
                       f.sector, f.pe_ratio, f.revenue_growth,
                       pt.target_mean,
                       lp.price as current_price,
                       CASE WHEN lp.price > 0 AND pt.target_mean > 0 
                            THEN ROUND(((pt.target_mean - lp.price) / lp.price * 100)::numeric, 1)
                            ELSE NULL END as upside_pct
                FROM screener_scores s
                LEFT JOIN fundamentals f ON s.ticker = f.ticker
                LEFT JOIN price_targets pt ON s.ticker = pt.ticker
                LEFT JOIN (
                    SELECT DISTINCT ON (ticker) ticker, close as price
                    FROM prices ORDER BY ticker, date DESC
                ) lp ON s.ticker = lp.ticker
                WHERE s.date = (SELECT MAX(date) FROM screener_scores)
                  AND s.total_score >= 60
                ORDER BY s.total_score DESC
                LIMIT {limit}
            """, engine)

            if not buy_df.empty:
                context_parts.append(f"\n## TOP BUY SIGNALS ({len(buy_df)} stocks with score >= 60)")
                context_parts.append(buy_df.to_string(index=False))

        # =================================================================
        # 3. SELL/AVOID SIGNALS (for sell-related questions)
        # =================================================================
        if any(kw in question_lower for kw in ['sell', 'avoid', 'worst', 'weak', 'bearish']):
            sell_df = pd.read_sql(f"""
                SELECT s.ticker, s.total_score, s.sentiment_score, s.fundamental_score,
                       s.technical_score, s.options_flow_score, CASE WHEN s.total_score >= 80 THEN 'STRONG_BUY' WHEN s.total_score >= 65 THEN 'BUY' WHEN s.total_score <= 35 THEN 'SELL' ELSE 'HOLD' END as signal,
                       f.sector
                FROM screener_scores s
                LEFT JOIN fundamentals f ON s.ticker = f.ticker
                WHERE s.date = (SELECT MAX(date) FROM screener_scores)
                  AND s.total_score <= 40
                ORDER BY s.total_score ASC
                LIMIT {limit}
            """, engine)

            if not sell_df.empty:
                context_parts.append(f"\n## SELL/AVOID SIGNALS ({len(sell_df)} stocks with score <= 40)")
                context_parts.append(sell_df.to_string(index=False))

        # =================================================================
        # 4. SECTOR ANALYSIS (for sector questions)
        # =================================================================
        if any(kw in question_lower for kw in ['sector', 'industry', 'which sector']):
            sector_df = pd.read_sql("""
                SELECT f.sector, 
                       COUNT(*) as stock_count,
                       ROUND(AVG(s.total_score)::numeric, 1) as avg_score,
                       ROUND(AVG(s.sentiment_score)::numeric, 1) as avg_sentiment,
                       ROUND(AVG(s.technical_score)::numeric, 1) as avg_technical,
                       COUNT(*) FILTER (WHERE s.total_score >= 65) as buy_signals,
                       COUNT(*) FILTER (WHERE s.total_score <= 35) as sell_signals
                FROM screener_scores s
                JOIN fundamentals f ON s.ticker = f.ticker
                WHERE s.date = (SELECT MAX(date) FROM screener_scores)
                  AND f.sector IS NOT NULL
                GROUP BY f.sector
                ORDER BY avg_score DESC
            """, engine)

            if not sector_df.empty:
                context_parts.append("\n## SECTOR ANALYSIS")
                context_parts.append(sector_df.to_string(index=False))

        # =================================================================
        # 5. MOMENTUM/TECHNICAL PLAYS
        # =================================================================
        if any(kw in question_lower for kw in ['momentum', 'technical', 'breakout', 'trend']):
            momentum_df = pd.read_sql(f"""
                SELECT s.ticker, s.total_score, s.technical_score, s.options_flow_score,
                       s.sentiment_score, CASE WHEN s.total_score >= 80 THEN 'STRONG_BUY' WHEN s.total_score >= 65 THEN 'BUY' WHEN s.total_score <= 35 THEN 'SELL' ELSE 'HOLD' END as signal, f.sector
                FROM screener_scores s
                LEFT JOIN fundamentals f ON s.ticker = f.ticker
                WHERE s.date = (SELECT MAX(date) FROM screener_scores)
                  AND s.technical_score >= 65
                ORDER BY s.technical_score DESC
                LIMIT {limit}
            """, engine)

            if not momentum_df.empty:
                context_parts.append("\n## HIGH MOMENTUM/TECHNICAL STOCKS")
                context_parts.append(momentum_df.to_string(index=False))

        # =================================================================
        # 6. OPTIONS FLOW (smart money)
        # =================================================================
        if any(kw in question_lower for kw in ['option', 'smart money', 'flow', 'unusual']):
            options_df = pd.read_sql(f"""
                SELECT s.ticker, s.total_score, s.options_flow_score, 
                       s.sentiment_score, CASE WHEN s.total_score >= 80 THEN 'STRONG_BUY' WHEN s.total_score >= 65 THEN 'BUY' WHEN s.total_score <= 35 THEN 'SELL' ELSE 'HOLD' END as signal, f.sector
                FROM screener_scores s
                LEFT JOIN fundamentals f ON s.ticker = f.ticker
                WHERE s.date = (SELECT MAX(date) FROM screener_scores)
                  AND s.options_flow_score >= 65
                ORDER BY s.options_flow_score DESC
                LIMIT {limit}
            """, engine)

            if not options_df.empty:
                context_parts.append("\n## HIGH OPTIONS FLOW (Smart Money)")
                context_parts.append(options_df.to_string(index=False))

        # =================================================================
        # 7. VALUE PLAYS
        # =================================================================
        if any(kw in question_lower for kw in ['value', 'cheap', 'undervalued', 'pe', 'fundamental']):
            value_df = pd.read_sql(f"""
                SELECT s.ticker, s.total_score, s.fundamental_score, f.pe_ratio, 
                       f.peg_ratio, f.price_to_book, f.roe, f.sector,
                       pt.target_mean, pt.recommendation
                FROM screener_scores s
                JOIN fundamentals f ON s.ticker = f.ticker
                LEFT JOIN price_targets pt ON s.ticker = pt.ticker
                WHERE s.date = (SELECT MAX(date) FROM screener_scores)
                  AND s.fundamental_score >= 60
                  AND f.pe_ratio > 0 AND f.pe_ratio < 30
                ORDER BY s.fundamental_score DESC
                LIMIT {limit}
            """, engine)

            if not value_df.empty:
                context_parts.append("\n## VALUE STOCKS (Strong fundamentals, reasonable PE)")
                context_parts.append(value_df.to_string(index=False))

        # =================================================================
        # 8. SPECIFIC TICKER LOOKUP
        # =================================================================
        # Extract potential tickers from question
        ticker_matches = re.findall(r'\b([A-Z]{1,5})\b', question)
        for potential_ticker in ticker_matches:
            if len(potential_ticker) >= 2:
                ticker_df = pd.read_sql(f"""
                    SELECT 
                        s.ticker, s.date, s.total_score, 
                        s.sentiment_score, s.fundamental_score, s.technical_score,
                        s.options_flow_score, s.short_squeeze_score,
                        CASE WHEN s.total_score >= 80 THEN 'STRONG_BUY' WHEN s.total_score >= 65 THEN 'BUY' WHEN s.total_score <= 35 THEN 'SELL' ELSE 'HOLD' END as signal, 
                        f.sector, f.industry, f.market_cap, f.pe_ratio, f.forward_pe,
                        f.peg_ratio, f.roe, f.debt_to_equity, f.revenue_growth, f.profit_margin,
                        pt.target_low, pt.target_mean, pt.target_high, pt.recommendation, pt.number_of_analysts,
                        lp.price as current_price,
                        CASE WHEN lp.price > 0 AND pt.target_mean > 0 
                             THEN ROUND(((pt.target_mean - lp.price) / lp.price * 100)::numeric, 1)
                             ELSE NULL END as upside_pct
                    FROM screener_scores s
                    LEFT JOIN fundamentals f ON s.ticker = f.ticker
                    LEFT JOIN price_targets pt ON s.ticker = pt.ticker
                    LEFT JOIN (
                        SELECT DISTINCT ON (ticker) ticker, close as price
                        FROM prices ORDER BY ticker, date DESC
                    ) lp ON s.ticker = lp.ticker
                    WHERE s.ticker = '{potential_ticker}'
                    ORDER BY s.date DESC
                    LIMIT 1
                """, engine)

                if not ticker_df.empty:
                    context_parts.append(f"\n## DETAILED ANALYSIS: {potential_ticker}")
                    # Transpose for better readability
                    for col in ticker_df.columns:
                        val = ticker_df.iloc[0][col]
                        if pd.notna(val):
                            context_parts.append(f"- {col}: {val}")

        # =================================================================
        # 9. RECENT AI ANALYSIS
        # =================================================================
        if any(kw in question_lower for kw in ['ai analysis', 'ai recommendation', 'ai says', 'ai think']):
            ai_df = pd.read_sql(f"""
                SELECT ticker, analysis_date, ai_action, ai_confidence,
                       entry_price, target_price, stop_loss
                FROM ai_analysis
                WHERE analysis_date >= CURRENT_DATE - INTERVAL '7 days'
                ORDER BY analysis_date DESC
                LIMIT {limit}
            """, engine)

            if not ai_df.empty:
                context_parts.append("\n## RECENT AI ANALYSIS RESULTS")
                context_parts.append(ai_df.to_string(index=False))

        # =================================================================
        # 10. ALWAYS INCLUDE TOP 5 SIGNALS FOR CONTEXT
        # =================================================================
        if len(context_parts) == 1:  # Only overview was added
            quick_df = pd.read_sql("""
                SELECT s.ticker, s.total_score, CASE WHEN s.total_score >= 80 THEN 'STRONG_BUY' WHEN s.total_score >= 65 THEN 'BUY' WHEN s.total_score <= 35 THEN 'SELL' ELSE 'HOLD' END as signal, f.sector,
                       s.sentiment_score, s.technical_score, s.options_flow_score
                FROM screener_scores s
                LEFT JOIN fundamentals f ON s.ticker = f.ticker
                WHERE s.date = (SELECT MAX(date) FROM screener_scores)
                ORDER BY s.total_score DESC
                LIMIT 10
            """, engine)

            if not quick_df.empty:
                context_parts.append("\n## CURRENT TOP 10 STOCKS BY SCORE")
                context_parts.append(quick_df.to_string(index=False))

    except Exception as e:
        logger.error(f"Error building comprehensive context: {e}")
        context_parts.append(f"\n[Error accessing database: {e}]")

    return "\n".join(context_parts)


def get_research_system_prompt() -> str:
    """
    Get enhanced system prompt for research capabilities.
    """
    return """
## Additional Research Capabilities

You have access to the platform's complete database including:
- screener_scores: Main signals with total_score (0-100), sentiment, fundamental, technical, options_flow scores
- fundamentals: PE ratios, growth rates, margins, sector/industry
- price_targets: Analyst targets and recommendations
- ai_analysis: Recent AI analysis results with entry/target/stop prices

## Scoring System
- Total Score 0-100 (combined signal strength):
  - 80+ = STRONG BUY
  - 65-79 = BUY  
  - 36-64 = HOLD
  - 21-35 = SELL
  - 0-20 = STRONG SELL

## When answering general research questions:
1. Always cite specific data from the context
2. Mention actual ticker symbols and their scores
3. Explain WHY stocks are good/bad based on the data
4. Include risk considerations
5. Be specific and actionable

## Example Response Format:
"Based on current signals, the top opportunities are:
1. **TICKER** (Score: XX) - [reason based on scores]
2. **TICKER** (Score: XX) - [reason based on scores]
..."
"""


# =============================================================================
# INTEGRATION FUNCTION FOR CHAT.PY
# =============================================================================

def enhance_chat_context(question: str, existing_context: str = "") -> str:
    """
    Enhance existing chat context with research data if needed.

    Call this from chat.py to add research capabilities.
    """
    if needs_research_context(question):
        research_context = get_comprehensive_context(question)
        return f"{existing_context}\n\n{research_context}"
    return existing_context


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test questions
    test_questions = [
        "What are the best stocks to buy today?",
        "Which sector is strongest right now?",
        "Show me stocks with high options flow",
        "Analyze AAPL for me"
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"Needs research: {needs_research_context(q)}")
        if needs_research_context(q):
            context = get_comprehensive_context(q)
            print(f"Context length: {len(context)} chars")
            print(context[:500] + "...")