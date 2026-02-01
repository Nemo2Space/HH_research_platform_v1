"""
AI Research Assistant - Full Database Access & Web Search

A comprehensive AI assistant that can:
1. Query ALL database tables (screener_scores, fundamentals, options_flow, news, etc.)
2. Search the web for real-time market news
3. Analyze portfolio context
4. Provide actionable trading insights

Author: HH Research Platform
Version: 2024-01-01
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from openai import OpenAI
import os
import re
import requests

from src.db.connection import get_connection, get_engine
from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AIResearchConfig:
    """AI Research Assistant configuration."""
    base_url: str = os.getenv("LLM_QWEN_BASE_URL", "http://172.23.193.91:8090/v1")
    model: str = os.getenv("LLM_QWEN_MODEL", "Qwen3-32B-Q6_K.gguf")
    temperature: float = 0.2
    max_tokens: int = 4000
    top_p: float = 0.9


# =============================================================================
# DATABASE SCHEMA KNOWLEDGE
# =============================================================================

DATABASE_SCHEMA = """
## Available Database Tables

### screener_scores (Main signals table - most important!)
- ticker VARCHAR(10) - Stock symbol
- date DATE - Analysis date
- total_score INTEGER - Overall score 0-100 (80+ = STRONG BUY, 65+ = BUY, 35- = SELL, 20- = STRONG SELL)
- sentiment_score INTEGER - News sentiment score 0-100
- fundamental_score INTEGER - Fundamentals score 0-100
- technical_score INTEGER - Technical analysis score 0-100
- options_flow_score INTEGER - Options flow bullishness 0-100
- short_squeeze_score INTEGER - Short squeeze potential 0-100
- committee_signal VARCHAR(20) - Committee decision (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
- committee_reasoning TEXT - Why the committee made this decision

### fundamentals
- ticker VARCHAR(10)
- market_cap DECIMAL - Market capitalization
- pe_ratio DECIMAL - Price/Earnings ratio
- forward_pe DECIMAL - Forward P/E
- peg_ratio DECIMAL - PEG ratio
- price_to_book DECIMAL - P/B ratio
- roe DECIMAL - Return on Equity
- roa DECIMAL - Return on Assets
- debt_to_equity DECIMAL - D/E ratio
- revenue_growth DECIMAL - Revenue growth %
- earnings_growth DECIMAL - Earnings growth %
- profit_margin DECIMAL - Profit margin %
- sector VARCHAR(50) - Company sector
- industry VARCHAR(100) - Company industry

### price_targets
- ticker VARCHAR(10)
- target_low DECIMAL - Lowest analyst target
- target_mean DECIMAL - Mean analyst target
- target_high DECIMAL - Highest analyst target
- target_median DECIMAL - Median analyst target
- recommendation VARCHAR(20) - Analyst consensus (buy, hold, sell)
- number_of_analysts INTEGER - How many analysts

### prices (Historical price data)
- ticker VARCHAR(10)
- date DATE
- open, high, low, close DECIMAL
- volume BIGINT
- adj_close DECIMAL

### options_flow
- ticker VARCHAR(10)
- expiration DATE
- strike DECIMAL
- option_type VARCHAR(4) - CALL or PUT
- volume INTEGER
- open_interest INTEGER
- premium DECIMAL
- updated_at TIMESTAMP

### news_articles
- ticker VARCHAR(10)
- title TEXT
- summary TEXT
- source VARCHAR(100)
- published_at TIMESTAMP
- sentiment_label VARCHAR(20)
- sentiment_score DECIMAL

### sentiment_scores
- ticker VARCHAR(10)
- date DATE
- sentiment_score INTEGER - 0-100
- positive_count INTEGER
- negative_count INTEGER
- article_count INTEGER

### universe (Tracked stocks)
- ticker VARCHAR(10)
- name VARCHAR(100)
- sector VARCHAR(50)
- market_cap_category VARCHAR(20) - mega, large, mid, small

### ai_analysis (AI analysis results)
- ticker VARCHAR(10)
- analysis_date DATE
- ai_action VARCHAR(20) - BUY, SELL, HOLD
- ai_confidence VARCHAR(20) - HIGH, MEDIUM, LOW
- entry_price DECIMAL
- target_price DECIMAL
- stop_loss DECIMAL
- bull_case TEXT
- bear_case TEXT
- key_risks TEXT

### alpha_predictions (ML predictions tracking)
- ticker VARCHAR(10)
- prediction_date DATE
- predicted_return_5d DECIMAL
- predicted_direction VARCHAR(10)
- alpha_signal VARCHAR(20)
- actual_return_5d DECIMAL (filled after 5 days)
"""


# =============================================================================
# QUERY TEMPLATES FOR COMMON QUESTIONS
# =============================================================================

QUERY_TEMPLATES = {
    "top_buys": """
        SELECT s.ticker, s.total_score, s.sentiment_score, s.fundamental_score, 
               s.technical_score, s.options_flow_score,
               CASE 
                   WHEN s.total_score >= 80 THEN 'STRONG_BUY'
                   WHEN s.total_score >= 65 THEN 'BUY'
                   WHEN s.total_score >= 50 THEN 'HOLD'
                   WHEN s.total_score >= 35 THEN 'SELL'
                   ELSE 'STRONG_SELL'
               END as signal,
               f.sector, f.pe_ratio, f.revenue_growth,
               pt.target_mean,
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
          AND s.total_score >= 65
        ORDER BY s.total_score DESC
        LIMIT {limit}
    """,

    "top_sells": """
        SELECT s.ticker, s.total_score, s.sentiment_score, s.fundamental_score,
               s.technical_score, s.options_flow_score,
               CASE 
                   WHEN s.total_score >= 80 THEN 'STRONG_BUY'
                   WHEN s.total_score >= 65 THEN 'BUY'
                   WHEN s.total_score >= 50 THEN 'HOLD'
                   WHEN s.total_score >= 35 THEN 'SELL'
                   ELSE 'STRONG_SELL'
               END as signal,
               f.sector, f.pe_ratio
        FROM screener_scores s
        LEFT JOIN fundamentals f ON s.ticker = f.ticker
        WHERE s.date = (SELECT MAX(date) FROM screener_scores)
          AND s.total_score <= 35
        ORDER BY s.total_score ASC
        LIMIT {limit}
    """,

    "strong_momentum": """
        SELECT s.ticker, s.total_score, s.technical_score, s.options_flow_score,
               s.sentiment_score, 
               CASE 
                   WHEN s.total_score >= 80 THEN 'STRONG_BUY'
                   WHEN s.total_score >= 65 THEN 'BUY'
                   ELSE 'HOLD'
               END as signal,
               f.sector
        FROM screener_scores s
        LEFT JOIN fundamentals f ON s.ticker = f.ticker
        WHERE s.date = (SELECT MAX(date) FROM screener_scores)
          AND s.technical_score >= 70
          AND s.options_flow_score >= 60
        ORDER BY (s.technical_score + s.options_flow_score) DESC
        LIMIT {limit}
    """,

    "value_plays": """
        SELECT s.ticker, s.total_score, s.fundamental_score, f.pe_ratio, 
               f.peg_ratio, f.price_to_book, f.roe, f.sector,
               pt.target_mean, pt.recommendation
        FROM screener_scores s
        JOIN fundamentals f ON s.ticker = f.ticker
        LEFT JOIN price_targets pt ON s.ticker = pt.ticker
        WHERE s.date = (SELECT MAX(date) FROM screener_scores)
          AND s.fundamental_score >= 65
          AND f.pe_ratio < 25
          AND f.pe_ratio > 0
        ORDER BY s.fundamental_score DESC
        LIMIT {limit}
    """,

    "high_options_activity": """
        SELECT s.ticker, s.total_score, s.options_flow_score, 
               s.sentiment_score, 
               CASE 
                   WHEN s.total_score >= 80 THEN 'STRONG_BUY'
                   WHEN s.total_score >= 65 THEN 'BUY'
                   ELSE 'HOLD'
               END as signal,
               f.sector
        FROM screener_scores s
        LEFT JOIN fundamentals f ON s.ticker = f.ticker
        WHERE s.date = (SELECT MAX(date) FROM screener_scores)
          AND s.options_flow_score >= 70
        ORDER BY s.options_flow_score DESC
        LIMIT {limit}
    """,

    "recent_news_movers": """
        SELECT s.ticker, s.total_score, s.sentiment_score, 
               CASE 
                   WHEN s.total_score >= 80 THEN 'STRONG_BUY'
                   WHEN s.total_score >= 65 THEN 'BUY'
                   WHEN s.total_score <= 35 THEN 'SELL'
                   ELSE 'HOLD'
               END as signal,
               f.sector
        FROM screener_scores s
        LEFT JOIN fundamentals f ON s.ticker = f.ticker
        WHERE s.date = (SELECT MAX(date) FROM screener_scores)
        ORDER BY ABS(s.sentiment_score - 50) DESC
        LIMIT {limit}
    """,

    "sector_summary": """
        SELECT f.sector, 
               COUNT(*) as stock_count,
               ROUND(AVG(s.total_score), 1) as avg_score,
               ROUND(AVG(s.sentiment_score), 1) as avg_sentiment,
               ROUND(AVG(s.technical_score), 1) as avg_technical,
               COUNT(*) FILTER (WHERE s.total_score >= 65) as buy_signals,
               COUNT(*) FILTER (WHERE s.total_score <= 35) as sell_signals
        FROM screener_scores s
        JOIN fundamentals f ON s.ticker = f.ticker
        WHERE s.date = (SELECT MAX(date) FROM screener_scores)
        GROUP BY f.sector
        ORDER BY avg_score DESC
    """,

    "ticker_deep_dive": """
        SELECT 
            s.ticker, s.date, s.total_score, 
            s.sentiment_score, s.fundamental_score, s.technical_score,
            s.options_flow_score, s.short_squeeze_score,
            CASE 
                WHEN s.total_score >= 80 THEN 'STRONG_BUY'
                WHEN s.total_score >= 65 THEN 'BUY'
                WHEN s.total_score >= 50 THEN 'HOLD'
                WHEN s.total_score >= 35 THEN 'SELL'
                ELSE 'STRONG_SELL'
            END as signal,
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
        WHERE s.ticker = '{ticker}'
        ORDER BY s.date DESC
        LIMIT 1
    """,

    "recent_ai_analysis": """
        SELECT ticker, analysis_date, ai_action, ai_confidence,
               entry_price, target_price, stop_loss,
               bull_case, bear_case, key_risks
        FROM ai_analysis
        WHERE analysis_date >= CURRENT_DATE - INTERVAL '7 days'
        ORDER BY analysis_date DESC, ai_confidence DESC
        LIMIT {limit}
    """
}


# =============================================================================
# AI RESEARCH ASSISTANT CLASS
# =============================================================================

class AIResearchAssistant:
    """
    Comprehensive AI Research Assistant with full database access.
    """

    def __init__(self, config: Optional[AIResearchConfig] = None):
        self.config = config or AIResearchConfig()
        self.history: List[Dict[str, str]] = []

        # Initialize OpenAI client
        try:
            self.client = OpenAI(
                base_url=self.config.base_url,
                api_key="not-needed",
                timeout=300
            )
            self.available = True
            logger.info(f"AI Research Assistant connected to {self.config.base_url}")
        except Exception as e:
            logger.error(f"AI Research Assistant initialization failed: {e}")
            self.available = False
            self.client = None

    def _execute_sql(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        try:
            engine = get_engine()
            return pd.read_sql(query, engine)
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return pd.DataFrame()

    def _get_market_overview(self) -> Dict[str, Any]:
        """Get current market overview."""
        overview = {}

        # Score distribution
        try:
            df = self._execute_sql("""
                SELECT 
                    COUNT(*) as total_stocks,
                    COUNT(*) FILTER (WHERE total_score >= 80) as strong_buys,
                    COUNT(*) FILTER (WHERE total_score >= 65 AND total_score < 80) as buys,
                    COUNT(*) FILTER (WHERE total_score > 35 AND total_score < 65) as holds,
                    COUNT(*) FILTER (WHERE total_score <= 35 AND total_score > 20) as sells,
                    COUNT(*) FILTER (WHERE total_score <= 20) as strong_sells,
                    ROUND(AVG(total_score), 1) as avg_score,
                    ROUND(AVG(sentiment_score), 1) as avg_sentiment
                FROM screener_scores
                WHERE date = (SELECT MAX(date) FROM screener_scores)
            """)
            if not df.empty:
                overview['distribution'] = df.iloc[0].to_dict()
        except Exception as e:
            logger.error(f"Market overview error: {e}")

        # Latest analysis date
        try:
            df = self._execute_sql("SELECT MAX(date) as latest FROM screener_scores")
            if not df.empty:
                overview['latest_date'] = str(df.iloc[0]['latest'])
        except:
            pass

        return overview

    def _detect_query_intent(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """
        Detect the intent of the user's question and return appropriate query type.
        """
        question_lower = question.lower()
        params = {'limit': 10}

        # Check for specific ticker mention
        ticker_match = re.search(r'\b([A-Z]{1,5})\b', question)

        # Best stocks to buy
        if any(kw in question_lower for kw in ['best stock', 'what to buy', 'should i buy',
                                                'top buy', 'best buy', 'recommend', 'opportunity',
                                                'highest score', 'strong buy']):
            return 'top_buys', params

        # Stocks to sell/avoid
        if any(kw in question_lower for kw in ['sell', 'avoid', 'worst', 'weak', 'bearish']):
            return 'top_sells', params

        # Momentum/Technical
        if any(kw in question_lower for kw in ['momentum', 'technical', 'trend', 'breakout']):
            return 'strong_momentum', params

        # Value investing
        if any(kw in question_lower for kw in ['value', 'cheap', 'undervalued', 'pe ratio', 'fundamental']):
            return 'value_plays', params

        # Options activity
        if any(kw in question_lower for kw in ['option', 'call', 'put', 'unusual activity']):
            return 'high_options_activity', params

        # News/sentiment
        if any(kw in question_lower for kw in ['news', 'sentiment', 'headline']):
            return 'recent_news_movers', params

        # Sector analysis
        if any(kw in question_lower for kw in ['sector', 'industry', 'which sector']):
            return 'sector_summary', params

        # AI analysis
        if any(kw in question_lower for kw in ['ai analysis', 'ai recommendation', 'ai says']):
            return 'recent_ai_analysis', params

        # Specific ticker deep dive
        if ticker_match and len(ticker_match.group(1)) >= 2:
            potential_ticker = ticker_match.group(1)
            # Verify it's a real ticker
            check_df = self._execute_sql(f"SELECT ticker FROM universe WHERE ticker = '{potential_ticker}' LIMIT 1")
            if not check_df.empty:
                return 'ticker_deep_dive', {'ticker': potential_ticker}

        # Default: top buys
        return 'top_buys', params

    def _build_context(self, question: str) -> str:
        """Build comprehensive context for the AI based on the question."""
        context_parts = []
        data_found = False

        # 1. Market Overview
        overview = self._get_market_overview()
        if overview and overview.get('distribution'):
            data_found = True
            context_parts.append(f"""
## Current Market Overview (as of {overview.get('latest_date', 'today')})
- Total stocks tracked: {overview.get('distribution', {}).get('total_stocks', 'N/A')}
- Average score: {overview.get('distribution', {}).get('avg_score', 'N/A')}/100
- Average sentiment: {overview.get('distribution', {}).get('avg_sentiment', 'N/A')}/100
- Strong Buys (80+): {overview.get('distribution', {}).get('strong_buys', 0)}
- Buys (65-79): {overview.get('distribution', {}).get('buys', 0)}
- Holds (36-64): {overview.get('distribution', {}).get('holds', 0)}
- Sells (21-35): {overview.get('distribution', {}).get('sells', 0)}
- Strong Sells (0-20): {overview.get('distribution', {}).get('strong_sells', 0)}
""")

        # 2. Query based on intent
        intent, params = self._detect_query_intent(question)
        logger.info(f"Query intent: {intent}, params: {params}")

        if intent in QUERY_TEMPLATES:
            query = QUERY_TEMPLATES[intent].format(**params)
            logger.info(f"Executing query for intent: {intent}")
            df = self._execute_sql(query)

            if not df.empty:
                data_found = True
                context_parts.append(f"\n## ACTUAL DATABASE DATA ({intent.replace('_', ' ').title()})")
                context_parts.append("THE FOLLOWING IS REAL DATA FROM THE DATABASE - USE ONLY THESE VALUES:\n")
                context_parts.append(df.to_string(index=False))
                context_parts.append("\n")
            else:
                logger.warning(f"Query returned empty for intent: {intent}")
                context_parts.append(f"\n## No data returned for {intent}\n")

        # 3. If asking about best buys but query failed, try simpler query
        if not data_found:
            logger.warning("No data found, trying simple query")
            simple_df = self._execute_sql("""
                SELECT ticker, total_score, sentiment_score, fundamental_score, 
                       technical_score, options_flow_score
                FROM screener_scores
                WHERE date = (SELECT MAX(date) FROM screener_scores)
                ORDER BY total_score DESC
                LIMIT 15
            """)
            if not simple_df.empty:
                data_found = True
                context_parts.append("\n## ACTUAL DATABASE DATA (Top Stocks by Score)")
                context_parts.append("THE FOLLOWING IS REAL DATA - USE ONLY THESE VALUES:\n")
                context_parts.append(simple_df.to_string(index=False))

        # 4. Critical warning if no data
        if not data_found:
            context_parts.append("""
## ⚠️ NO DATA AVAILABLE
Could not retrieve data from the database. 
DO NOT make up or invent any stock recommendations.
Instead, tell the user that database data is not available and they should check the system.
""")
        else:
            context_parts.append("""
## ⚠️ CRITICAL INSTRUCTION
You MUST only use the EXACT numbers shown above from the database.
DO NOT invent, estimate, or hallucinate any scores or data.
If a stock is not in the data above, say you don't have data for it.
Only recommend stocks that appear in the data with scores >= 65 for buy signals.
""")

        return "\n".join(context_parts)

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the AI."""
        return f"""You are an expert AI trading assistant for the HH Research Platform.

## CRITICAL RULES - YOU MUST FOLLOW THESE:
1. ONLY use data explicitly provided in the context below
2. NEVER invent, estimate, or make up stock scores or data
3. If a stock is not in the provided data, say "I don't have current data for that stock"
4. Quote the EXACT scores from the database (e.g., "AAPL has a total score of 44")
5. DO NOT hallucinate or fabricate any numbers

## Scoring System
- Total Score (0-100): Combined signal strength
  - 80+ = STRONG BUY
  - 65-79 = BUY  
  - 36-64 = HOLD
  - 21-35 = SELL
  - 0-20 = STRONG SELL

- Component Scores (each 0-100):
  - Sentiment Score: News and social media sentiment
  - Fundamental Score: Financial health (PE, growth, margins)
  - Technical Score: Chart patterns, momentum, trends
  - Options Flow Score: Smart money positioning via options

## Your Role
1. Analyze ONLY the data provided in context - never make up data
2. Give recommendations based on the ACTUAL scores shown
3. Explain WHY stocks are good/bad using the REAL numbers
4. If asked about a stock not in the data, say you don't have data for it
5. Be specific - quote actual tickers and their EXACT scores from the data

## Response Format
- Reference the exact scores from the database
- Say "According to the database, TICKER has a score of X" 
- List stocks in order by their actual total_score
- If no stocks have score >= 65, say there are no strong buy signals currently

Current date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""

    def ask(self, question: str, include_history: bool = True) -> str:
        """
        Ask the AI assistant a question.

        Args:
            question: The user question
            include_history: Whether to include conversation history

        Returns:
            AI response
        """
        if not self.available:
            return "AI Assistant is not available. Please check the connection to the LLM server."

        try:
            # Build context
            context = self._build_context(question)

            # Build messages
            messages = [{"role": "system", "content": self._build_system_prompt()}]

            # Add history (last 4 exchanges)
            if include_history and self.history:
                messages.extend(self.history[-8:])

            # Add current question with context
            user_message = f"""## User Question
{question}

## Database Context
{context}

Please analyze this data and answer the question. Be specific with ticker symbols and scores."""

            messages.append({"role": "user", "content": user_message})

            # Call LLM
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p
            )

            answer = response.choices[0].message.content

            # Clean up /think tags if present
            if '</think>' in answer:
                answer = answer.split('</think>')[-1].strip()

            # Update history
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": answer})

            # Keep history manageable
            if len(self.history) > 20:
                self.history = self.history[-16:]

            return answer

        except Exception as e:
            logger.error(f"AI Research Assistant error: {e}")
            return f"Error processing your question: {str(e)}"

    def get_quick_recommendations(self, count: int = 5) -> Dict[str, Any]:
        """
        Get quick buy/sell recommendations without AI processing.
        Useful for fast dashboard display.
        """
        result = {'buys': [], 'sells': [], 'timestamp': datetime.now().isoformat()}

        try:
            # Top buys
            buy_df = self._execute_sql(QUERY_TEMPLATES['top_buys'].format(limit=count))
            if not buy_df.empty:
                result['buys'] = buy_df.to_dict('records')

            # Top sells
            sell_df = self._execute_sql(QUERY_TEMPLATES['top_sells'].format(limit=count))
            if not sell_df.empty:
                result['sells'] = sell_df.to_dict('records')

        except Exception as e:
            logger.error(f"Quick recommendations error: {e}")

        return result

    def clear_history(self):
        """Clear conversation history."""
        self.history = []
        logger.info("Conversation history cleared")


# =============================================================================
# STREAMLIT UI COMPONENT
# =============================================================================

def render_ai_assistant_chat():
    """Render the AI Assistant chat interface in Streamlit."""
    import streamlit as st

    st.markdown("### 🤖 AI Research Assistant")
    st.caption("Ask me anything about your stocks, market conditions, or trading opportunities.")

    # Initialize assistant in session state
    if 'ai_assistant' not in st.session_state:
        st.session_state.ai_assistant = AIResearchAssistant()

    if 'ai_chat_messages' not in st.session_state:
        st.session_state.ai_chat_messages = []

    assistant = st.session_state.ai_assistant

    # Quick action buttons
    st.markdown("#### 💡 Quick Questions")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔥 Best Buys Today", use_container_width=True):
            st.session_state.ai_pending_question = "What are the best stocks to buy today? Give me the top 5 with highest scores and explain why."

    with col2:
        if st.button("⚠️ Stocks to Avoid", use_container_width=True):
            st.session_state.ai_pending_question = "Which stocks should I avoid or sell right now? Show me the weakest signals."

    with col3:
        if st.button("📊 Sector Analysis", use_container_width=True):
            st.session_state.ai_pending_question = "Give me a sector-by-sector breakdown. Which sectors are strongest right now?"

    with col4:
        if st.button("🎯 Momentum Plays", use_container_width=True):
            st.session_state.ai_pending_question = "What are the best momentum stocks with strong technical and options flow scores?"

    st.markdown("---")

    # Chat container
    chat_container = st.container(height=400)

    with chat_container:
        for msg in st.session_state.ai_chat_messages:
            if msg['role'] == 'user':
                st.markdown(f"**🧑 You:** {msg['content']}")
            else:
                st.markdown(f"**🤖 Assistant:** {msg['content']}")
            st.markdown("---")

    # Check for pending question from quick buttons
    if 'ai_pending_question' in st.session_state and st.session_state.ai_pending_question:
        question = st.session_state.ai_pending_question
        st.session_state.ai_pending_question = None

        # Add to messages
        st.session_state.ai_chat_messages.append({'role': 'user', 'content': question})

        # Get response
        with st.spinner("🔍 Analyzing data..."):
            response = assistant.ask(question)

        st.session_state.ai_chat_messages.append({'role': 'assistant', 'content': response})
        st.rerun()

    # Chat input
    col_input, col_clear = st.columns([5, 1])

    with col_input:
        user_input = st.text_input(
            "Ask a question...",
            placeholder="e.g., What are the best stocks to buy today?",
            key="ai_chat_input",
            label_visibility="collapsed"
        )

    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.ai_chat_messages = []
            assistant.clear_history()
            st.rerun()

    # Process input
    if user_input:
        # Add to messages
        st.session_state.ai_chat_messages.append({'role': 'user', 'content': user_input})

        # Get response
        with st.spinner("🔍 Analyzing data..."):
            response = assistant.ask(user_input)

        st.session_state.ai_chat_messages.append({'role': 'assistant', 'content': response})
        st.rerun()


# =============================================================================
# STANDALONE FUNCTION FOR QUICK ACCESS
# =============================================================================

_assistant_instance: Optional[AIResearchAssistant] = None

def get_ai_assistant() -> AIResearchAssistant:
    """Get singleton AI Research Assistant instance."""
    global _assistant_instance
    if _assistant_instance is None:
        _assistant_instance = AIResearchAssistant()
    return _assistant_instance


def ask_ai(question: str) -> str:
    """Quick function to ask the AI assistant a question."""
    return get_ai_assistant().ask(question)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("Testing AI Research Assistant...")

    assistant = AIResearchAssistant()

    if assistant.available:
        # Test question
        response = assistant.ask("What are the best stocks to buy today?")
        print("\n" + "="*60)
        print("RESPONSE:")
        print("="*60)
        print(response)
    else:
        print("AI Assistant not available - check LLM server connection")