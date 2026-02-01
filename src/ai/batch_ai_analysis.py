"""
Batch AI Analysis Module

Processes top signals through AI to get structured analysis results.
Stores results in database for filtering and display.

Author: Alpha Research Platform
Version: 2024-12-28
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import re

from src.db.connection import get_engine, get_connection
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AIAnalysisResult:
    """Structured AI analysis result for a ticker."""
    ticker: str
    analysis_date: datetime

    # Core Decision
    ai_action: str  # BUY, SELL, HOLD, WAIT_FOR_PULLBACK
    ai_confidence: str  # HIGH, MEDIUM, LOW
    trade_allowed: bool
    blocking_factors: List[str]

    # Entry/Exit Plan
    entry_price: Optional[float]
    entry_type: str  # MARKET, LIMIT, PULLBACK, STAGED
    stop_loss: Optional[float]
    target_price: Optional[float]
    position_size: str  # "1.0x", "0.5x", "0.25x"
    time_horizon: str  # "1-2 weeks", "2-4 weeks", "1-3 months"

    # Analysis Summary
    bull_case: List[str]  # Top 3 bull points
    bear_case: List[str]  # Top 3 bear points
    key_risks: List[str]  # Top 3 risks

    # Platform Data
    platform_score: int
    platform_signal: str
    alpha_signal: str
    ml_reliability: str  # BLOCKED, DEGRADED, TRADABLE

    # Raw Response
    full_analysis: str

    def to_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        d = asdict(self)
        # Convert lists to JSON strings for DB storage
        d['blocking_factors'] = json.dumps(d['blocking_factors'])
        d['bull_case'] = json.dumps(d['bull_case'])
        d['bear_case'] = json.dumps(d['bear_case'])
        d['key_risks'] = json.dumps(d['key_risks'])
        return d

    @classmethod
    def from_db_row(cls, row: dict) -> 'AIAnalysisResult':
        """Create from database row."""
        # Parse JSON strings back to lists
        row['blocking_factors'] = json.loads(row.get('blocking_factors', '[]'))
        row['bull_case'] = json.loads(row.get('bull_case', '[]'))
        row['bear_case'] = json.loads(row.get('bear_case', '[]'))
        row['key_risks'] = json.loads(row.get('key_risks', '[]'))
        return cls(**row)


# ============================================================================
# STRUCTURED PROMPT FOR JSON OUTPUT
# ============================================================================

STRUCTURED_ANALYSIS_PROMPT = """You are a quantitative trading analyst. Analyze this stock and provide your assessment in STRICT JSON format.

IMPORTANT: Your response must be ONLY valid JSON, no other text before or after.

Based on the data context provided, fill in this JSON structure:

```json
{{
  "ai_action": "BUY|SELL|HOLD|WAIT_FOR_PULLBACK",
  "ai_confidence": "HIGH|MEDIUM|LOW",
  "trade_allowed": true|false,
  "blocking_factors": ["factor1", "factor2"],
  "entry_price": 123.45,
  "entry_type": "MARKET|LIMIT|PULLBACK|STAGED",
  "stop_loss": 115.00,
  "target_price": 140.00,
  "position_size": "1.0x|0.5x|0.25x|0x",
  "time_horizon": "1-2 weeks|2-4 weeks|1-3 months",
  "bull_case": ["point1", "point2", "point3"],
  "bear_case": ["point1", "point2", "point3"],
  "key_risks": ["risk1", "risk2", "risk3"],
  "one_line_summary": "Brief summary of the recommendation"
}}
```

RULES:
1. MUST follow Decision Policy if present - never upgrade action or exceed size cap
2. If trade_allowed is false in Decision Policy, set trade_allowed to false
3. entry_price should be null if action is HOLD or SELL
4. stop_loss should be ~5-10% below entry for BUY
5. target_price should reflect realistic upside (use analyst targets if available)
6. position_size must not exceed Decision Policy size cap
7. bull_case, bear_case, key_risks should each have exactly 3 items
8. Be concise - each point should be one short sentence

DATA CONTEXT:
{context}

Respond with ONLY the JSON object, no markdown, no explanation."""


# ============================================================================
# BATCH ANALYZER CLASS
# ============================================================================

class BatchAIAnalyzer:
    """Processes multiple tickers through AI for structured analysis."""

    def __init__(self):
        """Initialize the batch analyzer."""
        self._ensure_table_exists()
        self.chat_assistant = None

    def _ensure_table_exists(self):
        """Create the ai_analysis table if it doesn't exist."""
        create_sql = """
                     CREATE TABLE IF NOT EXISTS ai_analysis \
                     ( \
                         id \
                         SERIAL \
                         PRIMARY \
                         KEY, \
                         ticker \
                         VARCHAR \
                     ( \
                         20 \
                     ) NOT NULL,
                         analysis_date TIMESTAMP NOT NULL,

                         -- Core Decision
                         ai_action VARCHAR \
                     ( \
                         20 \
                     ),
                         ai_confidence VARCHAR \
                     ( \
                         10 \
                     ),
                         trade_allowed BOOLEAN,
                         blocking_factors TEXT,

                         -- Entry/Exit Plan
                         entry_price DECIMAL \
                     ( \
                         10, \
                         2 \
                     ),
                         entry_type VARCHAR \
                     ( \
                         20 \
                     ),
                         stop_loss DECIMAL \
                     ( \
                         10, \
                         2 \
                     ),
                         target_price DECIMAL \
                     ( \
                         10, \
                         2 \
                     ),
                         position_size VARCHAR \
                     ( \
                         10 \
                     ),
                         time_horizon VARCHAR \
                     ( \
                         30 \
                     ),

                         -- Analysis Summary
                         bull_case TEXT,
                         bear_case TEXT,
                         key_risks TEXT,
                         one_line_summary TEXT,

                         -- Platform Data
                         platform_score INTEGER,
                         platform_signal VARCHAR \
                     ( \
                         20 \
                     ),
                         alpha_signal VARCHAR \
                     ( \
                         20 \
                     ),
                         ml_reliability VARCHAR \
                     ( \
                         30 \
                     ),

                         -- Raw Response
                         full_analysis TEXT,

                         -- Metadata
                         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                         );

                     -- Index for fast lookups
                     CREATE INDEX IF NOT EXISTS idx_ai_analysis_ticker ON ai_analysis(ticker);
                     CREATE INDEX IF NOT EXISTS idx_ai_analysis_date ON ai_analysis(analysis_date DESC);
                     CREATE INDEX IF NOT EXISTS idx_ai_analysis_action ON ai_analysis(ai_action); \
                     """
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_sql)
                conn.commit()
            logger.info("ai_analysis table ensured")
        except Exception as e:
            logger.error(f"Error creating ai_analysis table: {e}")

    def _get_chat_assistant(self):
        """Get or create the chat assistant (AlphaChat)."""
        if self.chat_assistant is None:
            try:
                from src.ai.chat import AlphaChat
                self.chat_assistant = AlphaChat()
                logger.info(f"AlphaChat initialized, available={self.chat_assistant.available}")
            except Exception as e:
                logger.error(f"Could not initialize AlphaChat: {e}")
                return None
        return self.chat_assistant

    def _call_ai_direct(self, prompt: str) -> Optional[str]:
        """
        Call AI directly using OpenAI client (bypasses AlphaChat complexity).
        This is more reliable for structured JSON output.
        """
        try:
            from openai import OpenAI
            import os
            from dotenv import load_dotenv

            load_dotenv()

            base_url = os.getenv("LLM_QWEN_BASE_URL", "http://172.23.193.91:8090/v1")
            model = os.getenv("LLM_QWEN_MODEL", "Qwen3-32B-Q6_K.gguf")

            logger.info(f"Direct AI call to {base_url} with model {model}")

            client = OpenAI(
                base_url=base_url,
                api_key="not-needed"  # Local Qwen doesn't need API key
            )

            # Simple system prompt for JSON output
            system_prompt = """You are a quantitative trading analyst. 
You MUST respond with ONLY valid JSON, no other text, no markdown, no explanation.
Follow the exact JSON structure requested."""

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Very low for consistent JSON
                max_tokens=2000
            )

            result = response.choices[0].message.content

            # Clean up thinking tags if present
            if '</think>' in result:
                result = result.split('</think>')[-1].strip()

            logger.info(f"Direct AI call successful, response length: {len(result)}")
            return result

        except Exception as e:
            logger.error(f"Direct AI call failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _build_context_for_ticker(self, ticker: str, fast_mode: bool = True) -> str:
        """
        Build the full data context for a ticker.

        Args:
            ticker: Stock symbol
            fast_mode: If True, skip slow features (SEC calls) and use cached data only
        """
        context_parts = []

        try:
            # In fast mode, just load from database (no API calls)
            if fast_mode:
                context = self._build_fast_context(ticker)
                if context:
                    return context

            # Full mode: use signal engine (slower but more complete)
            from src.core import get_signal_engine
            engine = get_signal_engine()
            signal = engine.generate_signal(ticker, force_refresh=False)

            if signal:
                # Basic info
                context_parts.append(f"=== {ticker} - {signal.company_name} ===")
                context_parts.append(f"Sector: {signal.sector}")
                context_parts.append(f"Current Price: ${signal.current_price:.2f}" if signal.current_price else "")
                context_parts.append(f"")
                context_parts.append(f"SIGNALS:")
                context_parts.append(f"- Today: {signal.today_signal.value} ({signal.today_score}%)")
                context_parts.append(f"- Risk: {signal.risk_level.value} ({signal.risk_score})")
                context_parts.append(f"")
                context_parts.append(f"COMPONENTS:")
                context_parts.append(f"- Technical: {signal.technical_score}")
                context_parts.append(f"- Fundamental: {signal.fundamental_score}")
                context_parts.append(f"- Sentiment: {signal.sentiment_score}")
                context_parts.append(f"- Options: {signal.options_score}")
                context_parts.append(f"")

                # Get enhanced alpha context
                try:
                    from src.ml.alpha_enhancements import build_enhanced_alpha_context

                    # Create default prediction for new tickers
                    prediction = {
                        'signal': 'HOLD',
                        'conviction': 0.0,
                        'expected_return_5d': 0,
                        'prob_positive_5d': 0.5,
                        'regime': 'UNKNOWN',
                    }

                    # Try to get actual alpha prediction
                    try:
                        import os
                        model_path = os.path.join(
                            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            "models", "multi_factor_alpha.pkl"
                        )
                        if os.path.exists(model_path):
                            from src.ml.multi_factor_alpha import MultiFactorAlphaModel
                            model = MultiFactorAlphaModel()
                            model.load(model_path)
                            result_df = model.predict_live(tickers=[ticker])
                            if not result_df.empty:
                                row = result_df.iloc[0]
                                prediction = {
                                    'signal': row.get('signal', 'HOLD'),
                                    'conviction': row.get('conviction', 0.5),
                                    'expected_return_5d': row.get('expected_return_5d', 0),
                                    'prob_positive_5d': row.get('prob_positive_5d', 0.5),
                                    'regime': row.get('regime', 'UNKNOWN'),
                                }
                    except Exception as e:
                        logger.debug(f"Alpha model not available for {ticker}: {e}")

                    alpha_context = build_enhanced_alpha_context(
                        ticker=ticker,
                        alpha_prediction=prediction,
                        platform_score=signal.today_score,
                        platform_signal=signal.today_signal.value,
                        technical_score=signal.technical_score
                    )
                    context_parts.append(alpha_context)
                except Exception as e:
                    logger.debug(f"Could not get enhanced context: {e}")

        except Exception as e:
            logger.error(f"Error building context for {ticker}: {e}")
            context_parts.append(f"Error loading data for {ticker}: {e}")

        return "\n".join(context_parts)

    def _build_fast_context(self, ticker: str) -> Optional[str]:
        """
        Build context from database only - NO API calls.
        This is 10-100x faster than full context building.
        """
        try:
            # Load all data from database in one query
            # Note: screener_scores doesn't have signal_type column - we compute it from total_score
            # Also: target_mean and target_upside_pct are in price_targets table, not fundamentals
            query = """
                    SELECT s.ticker, \
                           s.total_score, \
                           s.technical_score, \
                           s.fundamental_score, \
                           s.sentiment_score, \
                           s.options_flow_score, \
                           s.short_squeeze_score, \
                           s.gap_score, \
                           s.likelihood_score, \
                           f.sector, \
                           f.industry, \
                           f.pe_ratio, \
                           f.forward_pe, \
                           f.roe, \
                           pt.target_mean, \
                           CASE \
                               WHEN lp.price > 0 AND pt.target_mean > 0 \
                                   THEN ROUND(((pt.target_mean - lp.price) / lp.price * 100):: numeric, 2) \
                               ELSE NULL END as target_upside_pct, \
                           lp.price, \
                           CASE \
                               WHEN s.total_score >= 80 THEN 'STRONG BUY' \
                               WHEN s.total_score >= 65 THEN 'BUY' \
                               WHEN s.total_score >= 55 THEN 'WEAK BUY' \
                               WHEN s.total_score <= 20 THEN 'STRONG SELL' \
                               WHEN s.total_score <= 35 THEN 'SELL' \
                               WHEN s.total_score <= 45 THEN 'WEAK SELL' \
                               ELSE 'HOLD' \
                               END           as signal_type
                    FROM screener_scores s
                             LEFT JOIN fundamentals f ON s.ticker = f.ticker
                             LEFT JOIN price_targets pt ON s.ticker = pt.ticker
                             LEFT JOIN (SELECT DISTINCT \
                                        ON (ticker) ticker, close as price \
                                        FROM prices \
                                        ORDER BY ticker, date DESC) lp ON s.ticker = lp.ticker
                    WHERE s.ticker = %s
                    ORDER BY s.date DESC LIMIT 1 \
                    """

            df = pd.read_sql(query, get_engine(), params=(ticker,))

            if df.empty:
                logger.warning(f"No data found for {ticker} in screener_scores")
                return None

            row = df.iloc[0]

            # Get company name from yfinance as fallback (fast call)
            company_name = ticker
            try:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                info = stock.info
                company_name = info.get('shortName') or info.get('longName') or ticker
            except:
                pass

            # Build context from DB data
            context_parts = []
            context_parts.append(f"=== {ticker} - {company_name} ===")
            context_parts.append(f"Sector: {row.get('sector', 'Unknown')}")

            price = row.get('price')
            if price and not pd.isna(price):
                context_parts.append(f"Current Price: ${float(price):.2f}")

            context_parts.append("")
            context_parts.append("SIGNALS:")
            context_parts.append(f"- Signal: {row.get('signal_type', 'HOLD')} ({row.get('total_score', 50)}%)")

            context_parts.append("")
            context_parts.append("COMPONENTS:")
            context_parts.append(f"- Technical: {row.get('technical_score', 50)}")
            context_parts.append(f"- Fundamental: {row.get('fundamental_score', 50)}")
            context_parts.append(f"- Sentiment: {row.get('sentiment_score', 50)}")
            context_parts.append(f"- Options Flow: {row.get('options_flow_score', 50)}")
            context_parts.append(f"- Short Squeeze: {row.get('short_squeeze_score', 0)}")

            # Fundamentals
            context_parts.append("")
            context_parts.append("FUNDAMENTALS:")
            pe = row.get('pe_ratio')
            if pe and not pd.isna(pe):
                context_parts.append(f"- P/E: {float(pe):.1f}")
            target = row.get('target_mean')
            if target and not pd.isna(target):
                context_parts.append(f"- Analyst Target: ${float(target):.2f}")
            upside = row.get('target_upside_pct')
            if upside and not pd.isna(upside):
                context_parts.append(f"- Upside: {float(upside):.1f}%")

            # Add decision policy context (fast version)
            context_parts.append("")
            context_parts.append("DECISION POLICY:")
            total_score = row.get('total_score', 50)
            if total_score >= 65:
                context_parts.append("- Recommendation: BUY (platform score >= 65)")
                context_parts.append("- Size Cap: 0.5x (ML not available)")
            elif total_score <= 35:
                context_parts.append("- Recommendation: SELL (platform score <= 35)")
                context_parts.append("- Size Cap: 0.5x")
            else:
                context_parts.append("- Recommendation: HOLD (neutral)")
                context_parts.append("- Size Cap: N/A")

            return "\n".join(context_parts)

        except Exception as e:
            logger.debug(f"Fast context failed for {ticker}: {e}")
            return None

    def _parse_json_response(self, response: str, ticker: str) -> Optional[dict]:
        """Parse the AI response to extract JSON."""
        try:
            # Try to find JSON in the response
            # First, try direct parse
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass

            # Try to find JSON block in markdown
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            # Try to find raw JSON object
            json_match = re.search(r'\{[^{}]*"ai_action"[^{}]*\}', response, re.DOTALL)
            if json_match:
                # Find the full JSON by matching braces
                start = response.find('{')
                if start >= 0:
                    brace_count = 0
                    end = start
                    for i, c in enumerate(response[start:], start):
                        if c == '{':
                            brace_count += 1
                        elif c == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end = i + 1
                                break
                    try:
                        return json.loads(response[start:end])
                    except:
                        pass

            logger.warning(f"Could not parse JSON from response for {ticker}")
            return None

        except Exception as e:
            logger.error(f"JSON parse error for {ticker}: {e}")
            return None

    def analyze_ticker(self, ticker: str, signal_data: dict = None, fast_mode: bool = True) -> Optional[
        AIAnalysisResult]:
        """
        Run AI analysis on a single ticker.

        Args:
            ticker: Stock ticker
            signal_data: Optional pre-loaded signal data
            fast_mode: Use fast context (DB only) vs full context (API calls)

        Returns:
            AIAnalysisResult or None if failed
        """
        logger.info(f"Analyzing {ticker}... (fast_mode={fast_mode})")

        try:
            # Build context (fast mode = DB only, no API calls)
            context = self._build_context_for_ticker(ticker, fast_mode=fast_mode)
            logger.info(f"{ticker}: Context built, length={len(context)}")

            if not context or len(context) < 50:
                logger.warning(f"{ticker}: Context too short or empty")
                return None

            # Build prompt
            prompt = STRUCTURED_ANALYSIS_PROMPT.format(context=context)
            logger.info(f"{ticker}: Prompt built, length={len(prompt)}")

            # Call AI directly (more reliable for JSON output)
            logger.info(f"{ticker}: Calling AI...")
            response = self._call_ai_direct(prompt)

            if not response:
                logger.warning(f"{ticker}: Empty response from AI")
                return None

            logger.info(f"{ticker}: Got response, length={len(response)}")

            # Parse JSON from response
            parsed = self._parse_json_response(response, ticker)

            if not parsed:
                logger.warning(f"{ticker}: Failed to parse AI response as JSON")
                logger.debug(f"{ticker}: Raw response: {response[:500]}...")
                return None

            logger.info(f"{ticker}: Parsed JSON successfully - action={parsed.get('ai_action')}")

            # Extract platform data
            platform_score = signal_data.get('total_score', 50) if signal_data else 50
            platform_signal = signal_data.get('signal_type', 'HOLD') if signal_data else 'HOLD'

            # Create result object
            result = AIAnalysisResult(
                ticker=ticker,
                analysis_date=datetime.now(),

                ai_action=parsed.get('ai_action', 'HOLD'),
                ai_confidence=parsed.get('ai_confidence', 'LOW'),
                trade_allowed=parsed.get('trade_allowed', False),
                blocking_factors=parsed.get('blocking_factors', []),

                entry_price=parsed.get('entry_price'),
                entry_type=parsed.get('entry_type', 'LIMIT'),
                stop_loss=parsed.get('stop_loss'),
                target_price=parsed.get('target_price'),
                position_size=parsed.get('position_size', '0x'),
                time_horizon=parsed.get('time_horizon', 'N/A'),

                bull_case=parsed.get('bull_case', [])[:3],
                bear_case=parsed.get('bear_case', [])[:3],
                key_risks=parsed.get('key_risks', [])[:3],

                platform_score=platform_score,
                platform_signal=platform_signal,
                alpha_signal=parsed.get('alpha_signal', 'N/A'),
                ml_reliability=parsed.get('ml_reliability', 'UNKNOWN'),

                full_analysis=response
            )

            logger.info(f"{ticker}: Created AIAnalysisResult - {result.ai_action} ({result.ai_confidence})")
            return result

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def save_result(self, result: AIAnalysisResult) -> bool:
        """Save analysis result to database and prediction tracker."""
        try:
            # Delete existing result for today (upsert)
            delete_sql = """
                         DELETE \
                         FROM ai_analysis
                         WHERE ticker = %s \
                           AND analysis_date::date = %s:: date \
                         """

            insert_sql = """
                         INSERT INTO ai_analysis (ticker, analysis_date, \
                                                  ai_action, ai_confidence, trade_allowed, blocking_factors, \
                                                  entry_price, entry_type, stop_loss, target_price, position_size, \
                                                  time_horizon, \
                                                  bull_case, bear_case, key_risks, \
                                                  platform_score, platform_signal, alpha_signal, ml_reliability, \
                                                  full_analysis) \
                         VALUES (%s, %s, \
                                 %s, %s, %s, %s, \
                                 %s, %s, %s, %s, %s, %s, \
                                 %s, %s, %s, \
                                 %s, %s, %s, %s, \
                                 %s) \
                         """

            with get_connection() as conn:
                with conn.cursor() as cur:
                    # Delete existing
                    cur.execute(delete_sql, (result.ticker, result.analysis_date))

                    # Insert new
                    cur.execute(insert_sql, (
                        result.ticker, result.analysis_date,
                        result.ai_action, result.ai_confidence, result.trade_allowed,
                        json.dumps(result.blocking_factors),
                        result.entry_price, result.entry_type, result.stop_loss,
                        result.target_price, result.position_size, result.time_horizon,
                        json.dumps(result.bull_case), json.dumps(result.bear_case),
                        json.dumps(result.key_risks),
                        result.platform_score, result.platform_signal,
                        result.alpha_signal, result.ml_reliability,
                        result.full_analysis
                    ))
                conn.commit()

            logger.info(f"Saved AI analysis for {result.ticker}")

            # =========================================================================
            # ALSO SAVE TO PREDICTION TRACKER for ML Learning Progress
            # =========================================================================
            try:
                from src.ml.prediction_tracker import save_prediction

                # Calculate predicted return based on AI action and entry/target
                predicted_return_5d = 0.0
                if result.entry_price and result.target_price:
                    predicted_return_5d = (result.target_price - result.entry_price) / result.entry_price
                    # Assume 5-day horizon gets ~40% of the full target move
                    predicted_return_5d = predicted_return_5d * 0.4
                elif result.ai_action == 'BUY':
                    predicted_return_5d = 0.02  # Default 2% expected
                elif result.ai_action == 'SELL':
                    predicted_return_5d = -0.02  # Default -2% expected

                # Convert confidence to probability
                confidence_map = {'HIGH': 0.75, 'MEDIUM': 0.60, 'LOW': 0.50}
                predicted_probability = confidence_map.get(result.ai_confidence, 0.50)

                # Get current price
                current_price = result.entry_price
                if not current_price:
                    try:
                        import yfinance as yf
                        stock = yf.Ticker(result.ticker)
                        current_price = stock.fast_info.get('lastPrice') or stock.info.get('currentPrice')
                    except:
                        current_price = None

                # Save the prediction
                save_prediction(
                    ticker=result.ticker,
                    predicted_return_5d=predicted_return_5d,
                    predicted_probability=predicted_probability,
                    alpha_signal=result.ai_action,  # Use AI action as the signal
                    alpha_conviction=predicted_probability,
                    platform_signal=result.platform_signal,
                    platform_score=result.platform_score,
                    price_at_prediction=current_price,
                    predicted_return_10d=predicted_return_5d * 1.5 if predicted_return_5d else None,
                    predicted_return_20d=predicted_return_5d * 2.5 if predicted_return_5d else None
                )
                logger.info(f"Saved prediction for {result.ticker} to ML tracker")

            except ImportError:
                logger.debug("Prediction tracker not available")
            except Exception as e:
                logger.warning(f"Could not save prediction for {result.ticker}: {e}")

            return True

        except Exception as e:
            logger.error(f"Error saving AI analysis for {result.ticker}: {e}")
            return False

    def analyze_batch(self, tickers: List[str], signal_data: Dict[str, dict] = None,
                      progress_callback=None, fast_mode: bool = True) -> List[AIAnalysisResult]:
        """
        Analyze multiple tickers.

        Args:
            tickers: List of tickers to analyze
            signal_data: Optional dict of ticker -> signal data
            progress_callback: Optional callback(current, total, ticker, elapsed, eta) for progress updates
            fast_mode: Use fast context (DB only) vs full context (API calls)

        Returns:
            List of AIAnalysisResult
        """
        import time

        results = []
        errors = []
        total = len(tickers)
        start_time = time.time()

        logger.info(f"Starting batch analysis of {total} tickers (fast_mode={fast_mode})")

        for i, ticker in enumerate(tickers):
            ticker_start = time.time()

            # Calculate ETA
            elapsed = time.time() - start_time
            if i > 0:
                avg_per_ticker = elapsed / i
                remaining = (total - i) * avg_per_ticker
                eta_str = f"{remaining:.0f}s" if remaining < 60 else f"{remaining / 60:.1f}m"
            else:
                eta_str = "calculating..."

            if progress_callback:
                progress_callback(i + 1, total, ticker, elapsed, eta_str)

            try:
                data = signal_data.get(ticker, {}) if signal_data else {}
                result = self.analyze_ticker(ticker, data, fast_mode=fast_mode)

                ticker_elapsed = time.time() - ticker_start

                if result:
                    self.save_result(result)
                    results.append(result)
                    logger.info(f"✅ {ticker}: {result.ai_action} ({result.ai_confidence}) - {ticker_elapsed:.1f}s")
                else:
                    errors.append(ticker)
                    logger.warning(f"❌ {ticker}: No result - {ticker_elapsed:.1f}s")

            except Exception as e:
                errors.append(ticker)
                logger.error(f"❌ {ticker}: Error - {e}")
                continue

        total_time = time.time() - start_time
        logger.info(f"Batch complete: {len(results)}/{total} successful in {total_time:.1f}s")
        if errors:
            logger.warning(f"Failed tickers: {errors}")

        return results

    def get_latest_analysis(self, ticker: str) -> Optional[dict]:
        """Get the most recent AI analysis for a ticker."""
        try:
            query = """
                    SELECT * \
                    FROM ai_analysis
                    WHERE ticker = %s
                    ORDER BY analysis_date DESC LIMIT 1 \
                    """
            df = pd.read_sql(query, get_engine(), params=(ticker,))

            if df.empty:
                return None

            row = df.iloc[0].to_dict()
            # Parse JSON fields
            for field in ['blocking_factors', 'bull_case', 'bear_case', 'key_risks']:
                if row.get(field):
                    try:
                        row[field] = json.loads(row[field])
                    except:
                        row[field] = []

            return row

        except Exception as e:
            logger.error(f"Error getting analysis for {ticker}: {e}")
            return None

    def get_all_latest_analyses(self) -> pd.DataFrame:
        """Get the latest AI analysis for all tickers."""
        try:
            query = """
                    SELECT DISTINCT \
                    ON (ticker)
                        ticker, analysis_date,
                        ai_action, ai_confidence, trade_allowed,
                        entry_price, stop_loss, target_price, position_size,
                        time_horizon, platform_score, platform_signal
                    FROM ai_analysis
                    ORDER BY ticker, analysis_date DESC \
                    """
            return pd.read_sql(query, get_engine())
        except Exception as e:
            logger.error(f"Error getting all analyses: {e}")
            return pd.DataFrame()

    def get_actionable_signals(self, min_confidence: str = 'LOW') -> pd.DataFrame:
        """Get signals where AI recommends action (BUY or SELL)."""
        confidence_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        min_level = confidence_order.get(min_confidence, 1)

        try:
            query = """
                    SELECT DISTINCT \
                    ON (ticker)
                        ticker, analysis_date,
                        ai_action, ai_confidence, trade_allowed,
                        entry_price, stop_loss, target_price, position_size,
                        time_horizon, platform_score, platform_signal,
                        bull_case, key_risks
                    FROM ai_analysis
                    WHERE ai_action IN ('BUY' \
                        , 'SELL')
                      AND trade_allowed = true
                    ORDER BY ticker, analysis_date DESC \
                    """
            df = pd.read_sql(query, get_engine())

            # Filter by confidence
            if not df.empty:
                df['confidence_level'] = df['ai_confidence'].map(confidence_order)
                df = df[df['confidence_level'] >= min_level]
                df = df.drop('confidence_level', axis=1)

            return df

        except Exception as e:
            logger.error(f"Error getting actionable signals: {e}")
            return pd.DataFrame()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_batch_analysis(top_n: int = 20, sort_by: str = 'total_score',
                       progress_callback=None) -> List[AIAnalysisResult]:
    """
    Run batch AI analysis on top N signals.

    Args:
        top_n: Number of top signals to analyze
        sort_by: Column to sort by (total_score, sentiment_score, etc.)
        progress_callback: Optional callback for progress updates

    Returns:
        List of AIAnalysisResult
    """
    analyzer = BatchAIAnalyzer()

    # Get top signals from database
    try:
        query = f"""
        SELECT DISTINCT ON (ticker)
            ticker, total_score, signal_type, 
            technical_score, fundamental_score, sentiment_score
        FROM screener_scores
        WHERE total_score IS NOT NULL
        ORDER BY ticker, date DESC
        """
        df = pd.read_sql(query, get_engine())

        if df.empty:
            logger.warning("No signals found in database")
            return []

        # Sort and get top N
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)

        top_tickers = df.head(top_n)['ticker'].tolist()

        # Build signal data dict
        signal_data = {}
        for _, row in df.iterrows():
            signal_data[row['ticker']] = row.to_dict()

        # Run analysis
        return analyzer.analyze_batch(top_tickers, signal_data, progress_callback)

    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        return []


def get_ai_analysis_for_display() -> pd.DataFrame:
    """Get AI analysis results formatted for table display."""
    analyzer = BatchAIAnalyzer()
    return analyzer.get_all_latest_analyses()