"""
AI Trade Ideas Generator - FIXED VERSION

The central feature that combines ALL platform data to generate
actionable trade recommendations.

FIXES APPLIED:
- All numeric fields are Optional[float/int] instead of defaulting to 0/50
- All status fields use "NOT_ANALYZED" instead of empty string or fake defaults
- options_score: was 50 (neutral) when no data → now None
- squeeze_risk: was "LOW" when no data → now "NOT_ANALYZED"
- rsi_14: was 50 (neutral) when no data → now None
- All database queries use safe_float/safe_int helpers
- Added data_completeness tracking

Author: Alpha Research Platform
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.logging import get_logger
from src.db.connection import get_engine

# Analytics modules
from src.analytics.options_flow import OptionsFlowAnalyzer
from src.analytics.short_squeeze import ShortSqueezeDetector
from src.analytics.market_context import MarketContextAnalyzer, get_market_context
from src.analytics.technical_analysis import TechnicalAnalyzer

# Phase 0: Exposure Control
try:
    from src.portfolio.exposure_control import ExposureController, ExposureLimits
    EXPOSURE_CONTROL_AVAILABLE = True
except ImportError:
    EXPOSURE_CONTROL_AVAILABLE = False

# Earnings Intelligence
try:
    from src.analytics.earnings_intelligence import (
        enrich_screener_with_earnings,
        EarningsEnrichment,
    )
    EARNINGS_INTELLIGENCE_AVAILABLE = True
except ImportError:
    EARNINGS_INTELLIGENCE_AVAILABLE = False

logger = get_logger(__name__)

# Try to import macro regime
try:
    from src.analytics.macro_regime import get_current_regime, get_regime_adjustment
    MACRO_REGIME_AVAILABLE = True
except ImportError:
    MACRO_REGIME_AVAILABLE = False
    logger.warning("Macro regime module not available")

# Phase 1: Crowding Score
try:
    from src.analytics.crowding_score import get_crowding_score, CrowdingLevel
    CROWDING_SCORE_AVAILABLE = True
except ImportError:
    CROWDING_SCORE_AVAILABLE = False
    logger.warning("Crowding Score module not available")

# Phase 1: Multi-dimensional Regime
try:
    from src.analytics.multi_regime import get_current_regime as get_multi_regime, is_risk_on
    MULTI_REGIME_AVAILABLE = True
except ImportError:
    MULTI_REGIME_AVAILABLE = False
    logger.warning("Multi-dimensional Regime module not available")

# Phase 2: GEX/Gamma Analysis
try:
    from src.analytics.gex_analysis import analyze_gex, GEXRegime
    GEX_AVAILABLE = True
except ImportError:
    GEX_AVAILABLE = False
    logger.warning("GEX Analysis module not available")

# Phase 2: Dark Pool Flow
try:
    from src.analytics.dark_pool import analyze_dark_pool, DarkPoolSentiment
    DARK_POOL_AVAILABLE = True
except ImportError:
    DARK_POOL_AVAILABLE = False
    logger.warning("Dark Pool module not available")

# Phase 2: Cross-Asset Signals
try:
    from src.analytics.cross_asset import get_cross_asset_signals, CrossAssetSignal
    CROSS_ASSET_AVAILABLE = True
except ImportError:
    CROSS_ASSET_AVAILABLE = False
    logger.warning("Cross-Asset module not available")

# Phase 3: Sentiment NLP
try:
    from src.analytics.sentiment_nlp import (
        get_ticker_sentiment,
        analyze_news_sentiment,
        is_llm_available,
        SentimentLevel
    )
    SENTIMENT_NLP_AVAILABLE = True
except ImportError:
    SENTIMENT_NLP_AVAILABLE = False
    logger.warning("Sentiment NLP module not available")

# Phase 3: Earnings Whisper
try:
    from src.analytics.earnings_whisper import (
        get_earnings_whisper,
        EarningsPrediction,
        WhisperSignal
    )
    EARNINGS_WHISPER_AVAILABLE = True
except ImportError:
    EARNINGS_WHISPER_AVAILABLE = False
    logger.warning("Earnings Whisper module not available")

# Insider & 13F Tracking
try:
    from src.analytics.insider_tracker import get_insider_signal
    INSIDER_TRACKER_AVAILABLE = True
except ImportError:
    INSIDER_TRACKER_AVAILABLE = False

try:
    from src.analytics.institutional_13f_tracker import get_institutional_ownership
    INSTITUTIONAL_13F_AVAILABLE = True
except ImportError:
    INSTITUTIONAL_13F_AVAILABLE = False


# =============================================================================
# HELPER FUNCTIONS - Safe type conversion
# =============================================================================

def safe_float(value) -> Optional[float]:
    """Convert to float, return None if not possible or NULL."""
    if value is None:
        return None
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def safe_int(value) -> Optional[int]:
    """Convert to int, return None if not possible or NULL."""
    if value is None:
        return None
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def safe_str(value, default: str = "") -> str:
    """Convert to string, return default if None."""
    if value is None:
        return default
    if pd.isna(value):
        return default
    return str(value)


# =============================================================================
# TRADE CANDIDATE - FIXED with Optional fields
# =============================================================================

@dataclass
class TradeCandidate:
    """
    Complete data package for AI to evaluate a trade opportunity.

    FIXED: All numeric fields are Optional to distinguish between
    "value is 0" and "value is unknown/not available".
    """
    # Basic Info
    ticker: str
    company_name: str = ""
    sector: str = ""
    current_price: Optional[float] = None  # FIXED: was 0

    # Earnings Intelligence (IES/EQS/ECS)
    ies: Optional[float] = None  # FIXED: was 0
    ies_regime: str = "UNKNOWN"  # FIXED: was ""
    ecs_category: str = "PENDING"  # FIXED: was ""
    ei_position_scale: Optional[float] = None  # FIXED: was 1.0
    ei_score_adjustment: Optional[int] = None  # FIXED: was 0
    ei_risk_flags: List[str] = field(default_factory=list)

    # ============================================================
    # FROM DATABASE - FIXED: All Optional
    # ============================================================

    # Screener Scores
    total_score: Optional[float] = None  # FIXED: was 0
    sentiment_score: Optional[float] = None  # FIXED: was 0
    fundamental_score: Optional[float] = None  # FIXED: was 0
    technical_score: Optional[float] = None  # FIXED: was 0
    growth_score: Optional[float] = None  # FIXED: was 0
    dividend_score: Optional[float] = None  # FIXED: was 0
    article_count: Optional[int] = None  # FIXED: was 0
    score_date: Optional[str] = None  # FIXED: was ""

    # Trading Signal
    signal_type: str = "NOT_ANALYZED"  # FIXED: was ""
    signal_strength: Optional[float] = None  # FIXED: was 0
    signal_date: Optional[str] = None  # FIXED: was ""

    # Committee Decision
    committee_verdict: str = "NOT_ANALYZED"  # FIXED: was ""
    committee_conviction: Optional[float] = None  # FIXED: was 0
    committee_reasoning: str = ""

    # Fundamentals - all Optional
    pe_ratio: Optional[float] = None  # FIXED: was 0
    forward_pe: Optional[float] = None  # FIXED: was 0
    peg_ratio: Optional[float] = None  # FIXED: was 0
    revenue_growth: Optional[float] = None  # FIXED: was 0
    earnings_growth: Optional[float] = None  # FIXED: was 0
    profit_margin: Optional[float] = None  # FIXED: was 0
    debt_to_equity: Optional[float] = None  # FIXED: was 0

    # Analyst Ratings
    analyst_rating: str = "NOT_RATED"  # FIXED: was ""
    price_target: Optional[float] = None  # FIXED: was 0
    price_target_low: Optional[float] = None  # FIXED: was 0
    price_target_high: Optional[float] = None  # FIXED: was 0
    upside_pct: Optional[float] = None  # FIXED: was 0
    analyst_count: Optional[int] = None  # FIXED: was 0

    # Earnings Calendar
    earnings_date: Optional[str] = None  # FIXED: was ""
    days_to_earnings: Optional[int] = None  # FIXED: was 999
    earnings_safe: Optional[bool] = None  # FIXED: was True

    # Earnings Analysis (from latest report)
    earnings_sentiment: str = "NOT_ANALYZED"  # FIXED: was ""
    earnings_sentiment_score: Optional[int] = None  # FIXED: was 50
    eps_surprise_pct: Optional[float] = None  # FIXED: was 0
    guidance_direction: str = "UNKNOWN"  # FIXED: was ""
    earnings_score_adjustment: Optional[int] = None  # FIXED: was 0

    # Insider Trading
    insider_buys_3m: Optional[int] = None  # FIXED: was 0
    insider_sells_3m: Optional[int] = None  # FIXED: was 0
    insider_net: str = "UNKNOWN"  # FIXED: was ""

    # Institutional Ownership
    institutional_pct: Optional[float] = None  # FIXED: was 0
    institutional_change: Optional[float] = None  # FIXED: was 0

    # ============================================================
    # FROM LIVE ANALYSIS - FIXED: All Optional or explicit status
    # ============================================================

    # Options Flow - CRITICAL FIXES
    options_sentiment: str = "NOT_ANALYZED"  # FIXED: was "UNKNOWN"
    options_score: Optional[float] = None  # FIXED: was 0 - THIS WAS CRITICAL!
    put_call_ratio: Optional[float] = None  # FIXED: was 0
    unusual_calls: Optional[bool] = None  # FIXED: was False
    max_pain: Optional[float] = None  # FIXED: was 0

    # Short Squeeze - CRITICAL FIXES
    squeeze_score: Optional[float] = None  # FIXED: was 0
    squeeze_risk: str = "NOT_ANALYZED"  # FIXED: was "LOW" - THIS WAS CRITICAL!
    short_pct_float: Optional[float] = None  # FIXED: was 0
    days_to_cover: Optional[float] = None  # FIXED: was 0

    # Technical Levels
    support_1: Optional[float] = None  # FIXED: was 0
    resistance_1: Optional[float] = None  # FIXED: was 0
    distance_to_support_pct: Optional[float] = None  # FIXED: was 0
    risk_reward_ratio: Optional[float] = None  # FIXED: was 0
    above_50ma: Optional[bool] = None  # FIXED: was False
    above_200ma: Optional[bool] = None  # FIXED: was False
    rsi_14: Optional[float] = None  # FIXED: was 50 - THIS WAS CRITICAL!
    trend_20d: str = "UNKNOWN"  # FIXED: was "NEUTRAL"

    # Relative Strength
    rs_rating: Optional[int] = None  # FIXED: was 50
    vs_spy_20d: Optional[float] = None  # FIXED: was 0
    vs_sector_20d: Optional[float] = None  # FIXED: was 0
    sector_momentum: str = "UNKNOWN"  # FIXED: was "NEUTRAL"

    # Liquidity
    avg_dollar_volume: Optional[float] = None  # FIXED: was 0
    liquidity_score: str = "UNKNOWN"  # FIXED: was "MEDIUM"
    relative_volume: Optional[float] = None  # FIXED: was 0

    # Macro Regime Adjustment
    regime_adjustment: Optional[int] = None  # FIXED: was 0
    is_growth_stock: Optional[bool] = None  # FIXED: was False
    is_defensive_stock: Optional[bool] = None  # FIXED: was False

    # ============================================================
    # YOUR POSITION (if owned)
    # ============================================================
    currently_owned: bool = False
    shares_owned: int = 0
    current_weight_pct: float = 0
    avg_cost: float = 0
    unrealized_pnl: float = 0
    unrealized_pnl_pct: float = 0

    # ============================================================
    # AI RECOMMENDATION (filled after analysis)
    # ============================================================
    ai_score: Optional[float] = None  # FIXED: was 0
    ai_action: str = "NOT_SCORED"  # FIXED: was ""
    ai_entry: Optional[float] = None  # FIXED: was 0
    ai_stop_loss: Optional[float] = None  # FIXED: was 0
    ai_target_1: Optional[float] = None  # FIXED: was 0
    ai_target_2: Optional[float] = None  # FIXED: was 0
    ai_catalyst: str = ""
    ai_risks: List[str] = field(default_factory=list)
    ai_reasoning: str = ""

    # Phase 0: Exposure Control
    max_position_weight: float = 0.10
    exposure_constraints: List[str] = field(default_factory=list)
    sector_current_weight: float = 0.0

    # Phase 1: Crowding Score
    crowding_score: Optional[float] = None  # FIXED: was 50.0
    crowding_level: str = "UNKNOWN"  # FIXED: was "MODERATE"
    short_squeeze_risk_crowding: Optional[bool] = None  # FIXED: was False
    crowding_warnings: List[str] = field(default_factory=list)

    # Phase 1: Multi-dimensional Regime
    regime_score: Optional[int] = None  # FIXED: was 50
    regime_state: str = "UNKNOWN"  # FIXED: was "NEUTRAL"
    regime_risk_on: Optional[bool] = None  # FIXED: was True
    regime_favored_strategies: List[str] = field(default_factory=list)

    # Phase 2: GEX/Gamma Analysis
    gex_regime: str = "NOT_ANALYZED"  # FIXED: was "NEUTRAL"
    gex_signal: str = "NOT_ANALYZED"  # FIXED: was "NEUTRAL"
    gex_signal_strength: Optional[int] = None  # FIXED: was 50
    max_gamma_strike: Optional[float] = None  # FIXED: was 0.0
    call_wall: Optional[float] = None  # FIXED: was 0.0
    put_wall: Optional[float] = None  # FIXED: was 0.0
    gex_warnings: List[str] = field(default_factory=list)

    # Phase 2: Dark Pool Flow
    dark_pool_sentiment: str = "NOT_ANALYZED"  # FIXED: was "NEUTRAL"
    dark_pool_score: Optional[int] = None  # FIXED: was 50
    institutional_bias: str = "UNKNOWN"  # FIXED: was "NEUTRAL"
    block_buy_volume: Optional[int] = None  # FIXED: was 0
    block_sell_volume: Optional[int] = None  # FIXED: was 0
    dark_pool_warnings: List[str] = field(default_factory=list)

    # Phase 2: Cross-Asset Context
    cross_asset_signal: str = "NOT_ANALYZED"  # FIXED: was "NEUTRAL"
    cross_asset_risk_on: Optional[bool] = None  # FIXED: was True
    cycle_phase: str = "UNKNOWN"  # FIXED: was "MID_CYCLE"
    cross_asset_favored_sectors: List[str] = field(default_factory=list)

    # Phase 3: Sentiment NLP
    nlp_sentiment: str = "NOT_ANALYZED"  # FIXED: was "NEUTRAL"
    nlp_sentiment_score: Optional[int] = None  # FIXED: was 50
    nlp_management_tone: str = "UNKNOWN"  # FIXED: was "NEUTRAL"
    nlp_key_positives: List[str] = field(default_factory=list)
    nlp_key_negatives: List[str] = field(default_factory=list)
    nlp_summary: str = ""

    # Phase 3: Earnings Whisper
    whisper_prediction: str = "NOT_ANALYZED"  # FIXED: was "INLINE"
    whisper_beat_probability: Optional[float] = None  # FIXED: was 50.0
    whisper_expected_surprise: Optional[float] = None  # FIXED: was 0.0
    whisper_signal: str = "NOT_ANALYZED"  # FIXED: was "NEUTRAL"
    whisper_revision_score: Optional[int] = None  # FIXED: was 50
    whisper_options_score: Optional[int] = None  # FIXED: was 50
    whisper_historical_score: Optional[int] = None  # FIXED: was 50
    whisper_warnings: List[str] = field(default_factory=list)

    # Phase 4: Insider Transactions
    insider_signal: str = "NOT_ANALYZED"  # FIXED: was "NEUTRAL"
    insider_signal_strength: Optional[int] = None  # FIXED: was 50
    insider_ceo_bought: Optional[bool] = None  # FIXED: was False
    insider_cfo_bought: Optional[bool] = None  # FIXED: was False
    insider_cluster_buying: Optional[bool] = None  # FIXED: was False
    insider_cluster_selling: Optional[bool] = None  # FIXED: was False
    insider_net_value: Optional[float] = None  # FIXED: was 0.0

    # Phase 4: 13F Institutional Holdings
    inst_13f_signal: str = "NOT_ANALYZED"  # FIXED: was "NEUTRAL"
    inst_13f_strength: Optional[int] = None  # FIXED: was 50
    inst_buffett_owns: Optional[bool] = None  # FIXED: was False
    inst_buffett_added: Optional[bool] = None  # FIXED: was False
    inst_activist_involved: Optional[bool] = None  # FIXED: was False
    inst_new_buyers: List[str] = field(default_factory=list)
    inst_num_notable: Optional[int] = None  # FIXED: was 0

    # Data quality tracking - NEW
    data_completeness: float = 0.0
    missing_critical_data: List[str] = field(default_factory=list)

    @property
    def is_analyzable(self) -> bool:
        """Returns True if we have minimum data for AI analysis."""
        return self.current_price is not None and self.total_score is not None

    @property
    def score_display(self) -> str:
        """Display score with confidence indicator."""
        if self.total_score is None:
            return "N/A"
        missing_key = self.options_score is None or self.squeeze_score is None
        return f"{self.total_score:.0f}{'*' if missing_key else ''}"

    def get_display_value(self, field_name: str, format_spec: str = ".0f") -> str:
        """Get field value formatted for display, N/A if None."""
        value = getattr(self, field_name, None)
        if value is None:
            return "N/A"
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, (int, float)):
            return f"{value:{format_spec}}"
        return str(value)


@dataclass
class TradeIdeasResult:
    """Result from the trade ideas generator."""
    generated_at: datetime
    market_context: str
    top_picks: List[TradeCandidate]
    honorable_mentions: List[TradeCandidate]
    avoid_list: List[TradeCandidate]
    summary: str


class TradeIdeasGenerator:
    """
    Generates AI-powered trade recommendations by combining
    all platform data.
    """

    def __init__(self):
        self.options_analyzer = OptionsFlowAnalyzer()
        self.squeeze_detector = ShortSqueezeDetector()
        self.market_analyzer = MarketContextAnalyzer()
        self.tech_analyzer = TechnicalAnalyzer()
        self.engine = get_engine()

        # Phase 0: Exposure Control
        if EXPOSURE_CONTROL_AVAILABLE:
            self.exposure_controller = ExposureController()
        else:
            self.exposure_controller = None

        # Phase 1: Cache for regime (fetch once per run)
        self._cached_regime = None

        # Phase 2: Cache for cross-asset (fetch once per run)
        self._cached_cross_asset = None

        # Phase 3: Check LLM availability once
        self._llm_available = None

    def generate_ideas(self,
                       portfolio_positions: List[Dict] = None,
                       max_picks: int = 10,
                       filter_config: Dict = None,
                       use_ai_ranking: bool = True) -> TradeIdeasResult:
        """
        Generate trade ideas for today.
        """
        logger.info("Starting Trade Ideas generation...")

        # Default filter config
        if filter_config is None:
            filter_config = {
                'min_score': 40,
                'signal_types': ['STRONG_BUY', 'BUY', 'HOLD'],
                'include_no_signal': True,
                'min_rs': 0,
                'require_bullish_options': False,
                'skip_earnings_within_days': 7,
                'sectors': None,
            }

        # Step 1: Get market context
        market_ctx = self.market_analyzer.get_market_context()
        market_summary = self.market_analyzer.get_context_for_ai()

        # Step 2: Get all candidates from database
        candidates = self._get_candidates_from_db()
        logger.info(f"Found {len(candidates)} candidates from database")

        # Step 3: Pre-filter to top candidates
        filtered = self._pre_filter_candidates(candidates, market_ctx, filter_config)
        logger.info(f"Pre-filtered to {len(filtered)} candidates")

        # Step 4: Enrich with live data (options, squeeze, technicals)
        enriched = self._enrich_candidates(filtered, portfolio_positions)
        logger.info(f"Enriched {len(enriched)} candidates with live data")

        # Step 5: Post-filter based on enriched data
        post_filtered = self._post_filter_candidates(enriched, filter_config)
        logger.info(f"Post-filtered to {len(post_filtered)} candidates")

        # Step 5.5: Apply Phase 2 enrichment (GEX, Dark Pool, Cross-Asset)
        post_filtered = self._apply_phase2_analysis(post_filtered)

        # Step 5.6: Apply Phase 3 enrichment (Sentiment NLP, Earnings Whisper)
        post_filtered = self._apply_phase3_analysis(post_filtered)

        # Step 6: Score and rank
        scored = self._score_candidates(post_filtered, market_ctx)

        # Step 6.5: Apply exposure constraints (Phase 0)
        scored = self._apply_exposure_constraints(scored, portfolio_positions)

        # Step 6.6: Apply crowding analysis (Phase 1)
        scored = self._apply_crowding_analysis(scored)

        # Step 7: Separate into categories
        top_picks = [c for c in scored if c.ai_action in ['STRONG_BUY', 'BUY']][:max_picks]
        honorable = [c for c in scored if c.ai_action == 'HOLD'][:5]
        avoid = [c for c in scored if c.ai_action == 'AVOID'][:5]

        # Step 8: Generate summary
        summary = self._generate_summary(top_picks, market_ctx)

        result = TradeIdeasResult(
            generated_at=datetime.now(),
            market_context=market_summary,
            top_picks=top_picks,
            honorable_mentions=honorable,
            avoid_list=avoid,
            summary=summary
        )

        logger.info(f"Generated {len(top_picks)} top picks")

        return result

    def _get_candidates_from_db(self) -> List[TradeCandidate]:
        """Get all candidates with scores from database - FIXED with proper None handling."""
        candidates = []

        # Query to get latest scores and signals for each ticker
        # FIXED: Removed COALESCE with fake defaults in the SQL
        query = """
                WITH latest_scores AS (SELECT DISTINCT ON (ticker)
                    ticker, date as score_date,
                    sentiment_score, fundamental_score, total_score,
                    article_count,
                    options_flow_score, short_squeeze_score,
                    options_sentiment, squeeze_risk
                FROM screener_scores
                WHERE date >= CURRENT_DATE - INTERVAL '7 days'
                ORDER BY ticker, date DESC
                    ),
                    latest_signals AS (
                SELECT DISTINCT ON (ticker)
                    ticker, date as signal_date,
                    signal_type, signal_strength
                FROM trading_signals
                WHERE date >= CURRENT_DATE - INTERVAL '7 days'
                ORDER BY ticker, date DESC
                    ),
                    latest_fundamentals AS (
                SELECT DISTINCT ON (ticker)
                    ticker, sector,
                    pe_ratio, forward_pe, peg_ratio,
                    revenue_growth, earnings_growth, profit_margin,
                    debt_to_equity
                FROM fundamentals
                ORDER BY ticker, date DESC
                    ),
                    latest_earnings AS (
                SELECT DISTINCT ON (ticker)
                    ticker,
                    overall_sentiment as earnings_sentiment,
                    sentiment_score as earnings_sentiment_score,
                    eps_surprise_pct,
                    guidance_direction,
                    score_adjustment as earnings_score_adjustment
                FROM earnings_analysis
                ORDER BY ticker, filing_date DESC
                    )
                SELECT s.ticker,
                       s.score_date,
                       s.sentiment_score,
                       s.fundamental_score,
                       s.total_score,
                       s.article_count,
                       s.options_flow_score,
                       s.short_squeeze_score,
                       s.options_sentiment,
                       s.squeeze_risk,
                       sig.signal_date,
                       sig.signal_type,
                       sig.signal_strength,
                       f.sector,
                       f.pe_ratio,
                       f.forward_pe,
                       f.peg_ratio,
                       f.revenue_growth,
                       f.earnings_growth,
                       f.profit_margin,
                       f.debt_to_equity,
                       e.earnings_sentiment,
                       e.earnings_sentiment_score,
                       e.eps_surprise_pct,
                       e.guidance_direction,
                       e.earnings_score_adjustment
                FROM latest_scores s
                         LEFT JOIN latest_signals sig ON s.ticker = sig.ticker
                         LEFT JOIN latest_fundamentals f ON s.ticker = f.ticker
                         LEFT JOIN latest_earnings e ON s.ticker = e.ticker
                ORDER BY s.total_score DESC NULLS LAST
                """

        try:
            df = pd.read_sql(query, self.engine)

            for _, row in df.iterrows():
                # FIXED: Use safe_float/safe_int instead of "or 0" pattern
                candidate = TradeCandidate(
                    ticker=row['ticker'],
                    sector=safe_str(row.get('sector')),
                    score_date=safe_str(row.get('score_date')),

                    # Screener scores - FIXED: None if not available
                    sentiment_score=safe_float(row.get('sentiment_score')),
                    fundamental_score=safe_float(row.get('fundamental_score')),
                    total_score=safe_float(row.get('total_score')),
                    article_count=safe_int(row.get('article_count')),

                    # Signal - FIXED: explicit NOT_ANALYZED
                    signal_date=safe_str(row.get('signal_date')),
                    signal_type=safe_str(row.get('signal_type'), 'NOT_ANALYZED'),
                    signal_strength=safe_float(row.get('signal_strength')),

                    # Fundamentals
                    pe_ratio=safe_float(row.get('pe_ratio')),
                    forward_pe=safe_float(row.get('forward_pe')),
                    peg_ratio=safe_float(row.get('peg_ratio')),
                    revenue_growth=safe_float(row.get('revenue_growth')),
                    earnings_growth=safe_float(row.get('earnings_growth')),
                    profit_margin=safe_float(row.get('profit_margin')),
                    debt_to_equity=safe_float(row.get('debt_to_equity')),

                    # Options flow - CRITICAL FIX: None instead of 50!
                    options_score=safe_float(row.get('options_flow_score')),
                    options_sentiment=safe_str(row.get('options_sentiment'), 'NOT_ANALYZED'),

                    # Short squeeze - CRITICAL FIX: None and NOT_ANALYZED!
                    squeeze_score=safe_float(row.get('short_squeeze_score')),
                    squeeze_risk=safe_str(row.get('squeeze_risk'), 'NOT_ANALYZED'),

                    # Earnings analysis
                    earnings_sentiment=safe_str(row.get('earnings_sentiment'), 'NOT_ANALYZED'),
                    earnings_sentiment_score=safe_int(row.get('earnings_sentiment_score')),
                    eps_surprise_pct=safe_float(row.get('eps_surprise_pct')),
                    guidance_direction=safe_str(row.get('guidance_direction'), 'UNKNOWN'),
                    earnings_score_adjustment=safe_int(row.get('earnings_score_adjustment')),
                )

                # Calculate data completeness
                candidate.data_completeness = self._calculate_data_completeness(candidate)

                candidates.append(candidate)

            logger.info(f"Loaded {len(candidates)} candidates from database")

        except Exception as e:
            logger.error(f"Error getting candidates from DB: {e}")

            # Fallback: try simpler query with just scores
            try:
                fallback_query = """
                                 SELECT DISTINCT ON (ticker)
                                     ticker, date as score_date,
                                     sentiment_score, fundamental_score,
                                     total_score, article_count
                                 FROM screener_scores
                                 WHERE date >= CURRENT_DATE - INTERVAL '7 days'
                                 ORDER BY ticker, date DESC, total_score DESC NULLS LAST
                                 """
                df = pd.read_sql(fallback_query, self.engine)

                for _, row in df.iterrows():
                    candidate = TradeCandidate(
                        ticker=row['ticker'],
                        score_date=safe_str(row.get('score_date')),
                        sentiment_score=safe_float(row.get('sentiment_score')),
                        fundamental_score=safe_float(row.get('fundamental_score')),
                        total_score=safe_float(row.get('total_score')),
                        article_count=safe_int(row.get('article_count')),
                    )
                    # Mark as incomplete
                    candidate.missing_critical_data = [
                        'options_flow', 'squeeze', 'technical_levels',
                        'committee', 'signal', 'fundamentals'
                    ]
                    candidate.data_completeness = 0.2
                    candidates.append(candidate)

                logger.info(f"Fallback: Loaded {len(candidates)} candidates from screener_scores only")

            except Exception as e2:
                logger.error(f"Fallback query also failed: {e2}")

        return candidates

    def _calculate_data_completeness(self, c: TradeCandidate) -> float:
        """Calculate what proportion of key fields have data."""
        key_fields = [
            c.total_score,
            c.sentiment_score,
            c.fundamental_score,
            c.options_score,
            c.squeeze_score,
            c.signal_strength,
        ]
        available = sum(1 for f in key_fields if f is not None)
        return available / len(key_fields)

    def _pre_filter_candidates(self, candidates: List[TradeCandidate],
                               market_ctx, filter_config: Dict) -> List[TradeCandidate]:
        """Pre-filter to reduce candidates before expensive API calls."""
        filtered = []

        min_score = filter_config.get('min_score', 40)
        signal_types = filter_config.get('signal_types', ['STRONG_BUY', 'BUY', 'HOLD'])
        include_no_signal = filter_config.get('include_no_signal', True)
        sectors = filter_config.get('sectors', None)

        for c in candidates:
            # Check score threshold - FIXED: handle None
            if c.total_score is None or c.total_score < min_score:
                continue

            # Check signal type
            has_matching_signal = c.signal_type in signal_types
            has_no_signal = c.signal_type in ['NOT_ANALYZED', 'HOLD', None, '']

            if not has_matching_signal:
                if not (include_no_signal and has_no_signal):
                    continue

            # Check sector filter
            if sectors and c.sector and c.sector not in sectors:
                continue

            filtered.append(c)

        # Limit to top 50 to avoid too many API calls
        filtered = filtered[:50]

        logger.info(f"Pre-filter: {len(candidates)} -> {len(filtered)} candidates")

        return filtered

    def _post_filter_candidates(self, candidates: List[TradeCandidate],
                                filter_config: Dict) -> List[TradeCandidate]:
        """Post-filter after enrichment with live data."""
        filtered = []

        min_rs = filter_config.get('min_rs', 0)
        require_bullish_options = filter_config.get('require_bullish_options', False)
        skip_earnings_days = filter_config.get('skip_earnings_within_days', 7)

        for c in candidates:
            # Check relative strength - FIXED: handle None
            if min_rs > 0 and c.rs_rating is not None and c.rs_rating < min_rs:
                continue

            # Check options sentiment
            if require_bullish_options and c.options_sentiment != 'BULLISH':
                continue

            # Check earnings proximity - FIXED: handle None
            if skip_earnings_days > 0 and c.days_to_earnings is not None:
                if c.days_to_earnings <= skip_earnings_days:
                    c.earnings_safe = False
                else:
                    c.earnings_safe = True

            filtered.append(c)

        return filtered

    def _enrich_candidates(self, candidates: List[TradeCandidate],
                           portfolio_positions: List[Dict] = None) -> List[TradeCandidate]:
        """Enrich candidates with live data from APIs."""
        # Build position lookup
        position_lookup = {}
        if portfolio_positions:
            for pos in portfolio_positions:
                symbol = pos.get('symbol', '')
                if symbol and symbol not in ['USD', 'CASH']:
                    position_lookup[symbol] = pos

        enriched = []

        def enrich_single(candidate: TradeCandidate) -> TradeCandidate:
            """Enrich a single candidate."""
            ticker = candidate.ticker

            try:
                # Get current price and technical analysis
                tech = self.tech_analyzer.analyze_ticker(ticker, candidate.sector)

                if tech.current_price and tech.current_price > 0:
                    candidate.current_price = tech.current_price

                    # Technical levels - FIXED: preserve None
                    if tech.levels:
                        candidate.support_1 = tech.levels.support_1 if tech.levels.support_1 else None
                        candidate.resistance_1 = tech.levels.resistance_1 if tech.levels.resistance_1 else None
                        candidate.distance_to_support_pct = tech.levels.distance_to_support_pct if hasattr(tech.levels, 'distance_to_support_pct') else None
                        candidate.risk_reward_ratio = tech.levels.risk_reward_ratio if hasattr(tech.levels, 'risk_reward_ratio') else None
                        candidate.above_50ma = tech.levels.above_50ma if hasattr(tech.levels, 'above_50ma') else None
                        candidate.above_200ma = tech.levels.above_200ma if hasattr(tech.levels, 'above_200ma') else None
                        candidate.rsi_14 = tech.levels.rsi_14 if hasattr(tech.levels, 'rsi_14') and tech.levels.rsi_14 else None
                        candidate.trend_20d = tech.levels.trend_20d if hasattr(tech.levels, 'trend_20d') and tech.levels.trend_20d else "UNKNOWN"

                    # Relative strength
                    if tech.relative_strength:
                        candidate.rs_rating = tech.relative_strength.rs_rating if hasattr(tech.relative_strength, 'rs_rating') else None
                        candidate.vs_spy_20d = tech.relative_strength.vs_spy_20d if hasattr(tech.relative_strength, 'vs_spy_20d') else None
                        candidate.vs_sector_20d = tech.relative_strength.vs_sector_20d if hasattr(tech.relative_strength, 'vs_sector_20d') else None
                        candidate.sector_momentum = tech.relative_strength.sector_momentum if hasattr(tech.relative_strength, 'sector_momentum') and tech.relative_strength.sector_momentum else "UNKNOWN"

                    # Liquidity
                    if tech.liquidity:
                        candidate.avg_dollar_volume = tech.liquidity.avg_dollar_volume if hasattr(tech.liquidity, 'avg_dollar_volume') else None
                        candidate.liquidity_score = tech.liquidity.liquidity_score if hasattr(tech.liquidity, 'liquidity_score') and tech.liquidity.liquidity_score else "UNKNOWN"
                        candidate.relative_volume = tech.liquidity.relative_volume if hasattr(tech.liquidity, 'relative_volume') else None

                    # Calculate upside to price target
                    if candidate.price_target and candidate.price_target > 0:
                        candidate.upside_pct = ((candidate.price_target - candidate.current_price) /
                                                candidate.current_price) * 100

                    # Earnings Intelligence enrichment
                    if EARNINGS_INTELLIGENCE_AVAILABLE:
                        try:
                            ei = enrich_screener_with_earnings(candidate.ticker)
                            candidate.ies = ei.ies  # Now properly None if not calculated
                            candidate.ies_regime = ei.regime.value if ei.regime else "UNKNOWN"
                            candidate.ecs_category = ei.ecs_category.value if ei.ecs_category else "PENDING"
                            candidate.ei_position_scale = ei.position_scale
                            candidate.ei_score_adjustment = ei.score_adjustment
                            candidate.ei_risk_flags = ei.risk_flags

                            # Apply score adjustment - FIXED: handle None
                            if candidate.total_score is not None and ei.score_adjustment:
                                candidate.total_score += ei.score_adjustment

                            if hasattr(ei, 'in_action_window') and ei.in_action_window:
                                candidate.earnings_safe = False
                        except Exception as e:
                            logger.debug(f"{candidate.ticker}: EI error: {e}")

                # Options flow
                try:
                    options = self.options_analyzer.analyze_ticker(ticker)
                    if options:
                        candidate.options_sentiment = options.overall_sentiment or "NOT_ANALYZED"
                        candidate.options_score = options.sentiment_score  # Could be None
                        candidate.put_call_ratio = options.put_call_volume_ratio
                        candidate.max_pain = options.max_pain_price
                        candidate.unusual_calls = any(
                            a.option_type == 'CALL' and a.severity == 'HIGH'
                            for a in options.alerts
                        ) if options.alerts else False
                except Exception as e:
                    logger.debug(f"Options error for {ticker}: {e}")

                # Short squeeze
                try:
                    squeeze = self.squeeze_detector.analyze_ticker(ticker)
                    if squeeze:
                        candidate.squeeze_score = squeeze.squeeze_score  # Now properly None if not calculated
                        candidate.squeeze_risk = squeeze.squeeze_risk or "NOT_ANALYZED"
                        candidate.short_pct_float = squeeze.short_percent_of_float
                        candidate.days_to_cover = squeeze.days_to_cover
                except Exception as e:
                    logger.debug(f"Squeeze error for {ticker}: {e}")

                # Your position
                if ticker in position_lookup:
                    pos = position_lookup[ticker]
                    candidate.currently_owned = True
                    candidate.shares_owned = int(pos.get('shares', 0))
                    candidate.current_weight_pct = float(pos.get('weight', 0)) * 100
                    candidate.avg_cost = float(pos.get('avg_cost', 0))
                    candidate.unrealized_pnl = float(pos.get('unrealized_pnl', 0))
                    if candidate.avg_cost > 0 and candidate.current_price:
                        candidate.unrealized_pnl_pct = ((candidate.current_price - candidate.avg_cost) /
                                                        candidate.avg_cost) * 100

            except Exception as e:
                logger.debug(f"Error enriching {ticker}: {e}")

            return candidate

        # Enrich in parallel for speed
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(enrich_single, c): c for c in candidates}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result.current_price and result.current_price > 0:
                        enriched.append(result)
                except Exception as e:
                    logger.debug(f"Enrichment error: {e}")

        return enriched

    def _score_candidates(self, candidates: List[TradeCandidate],
                          market_ctx) -> List[TradeCandidate]:
        """Score and rank candidates using a rules-based approach."""
        for c in candidates:
            score = 50  # Start neutral
            reasons = []
            risks = []

            # ============================================================
            # POSITIVE FACTORS - FIXED: Check for None before comparisons
            # ============================================================

            # Platform signal (0-15)
            if c.signal_type == 'STRONG_BUY':
                score += 15
                strength_str = f"{c.signal_strength:.0f}" if c.signal_strength else "N/A"
                reasons.append(f"Strong BUY signal (strength: {strength_str})")
            elif c.signal_type == 'BUY':
                score += 10
                strength_str = f"{c.signal_strength:.0f}" if c.signal_strength else "N/A"
                reasons.append(f"BUY signal (strength: {strength_str})")

            # Committee conviction (0-10)
            if c.committee_verdict in ['BUY', 'STRONG_BUY']:
                if c.committee_conviction is not None and c.committee_conviction >= 70:
                    score += 10
                    reasons.append(f"Committee: {c.committee_verdict} ({c.committee_conviction:.0f}% conviction)")
                else:
                    score += 5

            # Options flow (0-15) - FIXED: handle None
            if c.options_score is not None:
                if c.options_score >= 70:
                    score += 15
                    reasons.append(f"Strong bullish options flow (score: {c.options_score:.0f})")
                elif c.options_score >= 60:
                    score += 10
                    if c.unusual_calls:
                        score += 3
                        reasons.append(f"Bullish options flow with unusual calls (score: {c.options_score:.0f})")
                    else:
                        reasons.append(f"Bullish options flow (score: {c.options_score:.0f})")
            elif c.options_sentiment == 'BULLISH':
                score += 7
                if c.unusual_calls:
                    score += 3
                    reasons.append("Bullish options flow with unusual call buying")
                else:
                    reasons.append("Bullish options flow")

            # Short squeeze potential (0-12) - FIXED: handle None
            if c.squeeze_score is not None:
                if c.squeeze_score >= 70:
                    score += 12
                    reasons.append(f"🔥 High squeeze potential (score: {c.squeeze_score:.0f}/100)")
                elif c.squeeze_score >= 50:
                    score += 8
                    reasons.append(f"⚠️ Elevated squeeze potential (score: {c.squeeze_score:.0f}/100)")
            elif c.squeeze_risk == 'EXTREME':
                score += 8
                reasons.append("Extreme squeeze potential")
            elif c.squeeze_risk == 'HIGH':
                score += 5
                reasons.append("High squeeze potential")

            # Relative strength (0-8) - FIXED: handle None
            if c.rs_rating is not None:
                if c.rs_rating >= 70:
                    score += 8
                    reasons.append(f"Strong relative strength (RS: {c.rs_rating})")
                elif c.rs_rating >= 60:
                    score += 4

            # Earnings sentiment (0-15)
            if c.earnings_sentiment == 'VERY_BULLISH':
                score += 15
                eps_str = f"{c.eps_surprise_pct:+.1f}%" if c.eps_surprise_pct else "N/A"
                reasons.append(f"🎯 Excellent earnings ({eps_str} surprise, {c.guidance_direction} guidance)")
            elif c.earnings_sentiment == 'BULLISH':
                score += 10
                eps_str = f"{c.eps_surprise_pct:+.1f}%" if c.eps_surprise_pct else "N/A"
                reasons.append(f"Strong earnings ({eps_str} surprise)")
            elif c.earnings_sentiment == 'BEARISH':
                score -= 5
                eps_str = f"{c.eps_surprise_pct:+.1f}%" if c.eps_surprise_pct else "N/A"
                risks.append(f"Weak earnings ({eps_str} surprise)")
            elif c.earnings_sentiment == 'VERY_BEARISH':
                score -= 10
                eps_str = f"{c.eps_surprise_pct:+.1f}%" if c.eps_surprise_pct else "N/A"
                risks.append(f"Poor earnings ({eps_str} surprise, {c.guidance_direction} guidance)")

            # Guidance direction
            if c.guidance_direction == 'RAISED':
                score += 5
                if c.earnings_sentiment not in ['VERY_BULLISH', 'BULLISH']:
                    reasons.append("Raised guidance")
            elif c.guidance_direction == 'LOWERED':
                score -= 5
                risks.append("Lowered guidance")

            # Technicals (0-5) - FIXED: handle None
            if c.above_50ma and c.above_200ma:
                score += 5
                reasons.append("Above 50 & 200 MA")
            elif c.above_200ma:
                score += 2

            # ============================================================
            # NEGATIVE FACTORS
            # ============================================================

            # Platform sell signal
            if c.signal_type == 'STRONG_SELL':
                score -= 20
                risks.append("Strong SELL signal")
            elif c.signal_type == 'SELL':
                score -= 10
                risks.append("SELL signal")

            # Upcoming earnings risk
            if c.earnings_safe is False:
                score -= 5
                days_str = str(c.days_to_earnings) if c.days_to_earnings is not None else "?"
                risks.append(f"⚠️ Earnings in {days_str} days")

            # Bearish options - FIXED: handle None
            if c.options_sentiment == 'BEARISH':
                score -= 8
                risks.append("Bearish options flow")
            elif c.options_score is not None and c.options_score < 40:
                score -= 8
                risks.append("Bearish options flow")

            # High volatility market + high beta stock
            if hasattr(market_ctx, 'vix_level') and market_ctx.vix_level > 25:
                if c.sector in ['Technology', 'Consumer Cyclical']:
                    score -= 3
                    risks.append("High VIX + high-beta sector")

            # Crowding penalty (Phase 1) - FIXED: handle None
            if c.crowding_level == 'EXTREME':
                score -= 10
                crowding_str = f"{c.crowding_score:.0f}" if c.crowding_score else "N/A"
                risks.append(f"EXTREME crowding (score: {crowding_str})")
            elif c.crowding_level == 'HIGH':
                score -= 5
                crowding_str = f"{c.crowding_score:.0f}" if c.crowding_score else "N/A"
                risks.append(f"HIGH crowding (score: {crowding_str})")

            # Short interest without squeeze setup - FIXED: handle None
            if c.short_pct_float is not None and c.short_pct_float > 20:
                if c.squeeze_score is None or c.squeeze_score < 50:
                    score -= 3
                    risks.append(f"High short interest ({c.short_pct_float:.1f}%) without squeeze setup")

            # ============================================================
            # MACRO REGIME ADJUSTMENT (Phase 1)
            # ============================================================
            if MACRO_REGIME_AVAILABLE:
                try:
                    regime_adj = get_regime_adjustment(c.ticker, c.sector)
                    c.regime_adjustment = regime_adj

                    # Classify stock type
                    if c.sector in ['Technology', 'Consumer Cyclical', 'Communication Services']:
                        c.is_growth_stock = True
                    elif c.sector in ['Utilities', 'Consumer Defensive', 'Healthcare']:
                        c.is_defensive_stock = True

                    if regime_adj != 0:
                        score += regime_adj
                        if regime_adj > 0:
                            reasons.append(f"🌍 Macro regime favorable ({regime_adj:+d})")
                        else:
                            risks.append(f"🌍 Macro regime unfavorable ({regime_adj:+d})")
                except Exception as e:
                    logger.debug(f"Error getting regime adjustment: {e}")

            # ============================================================
            # MULTI-DIMENSIONAL REGIME ADJUSTMENT (Phase 1)
            # ============================================================
            if MULTI_REGIME_AVAILABLE:
                try:
                    if self._cached_regime is None:
                        self._cached_regime = get_multi_regime()

                    regime = self._cached_regime
                    c.regime_score = regime.regime_score
                    c.regime_state = regime.overall_regime.value
                    c.regime_risk_on = regime.risk_on
                    c.regime_favored_strategies = regime.favored_strategies

                    # Adjust score based on regime alignment
                    if not regime.risk_on:
                        if c.is_growth_stock:
                            score -= 5
                            risks.append(f"Growth stock in risk-off regime (score: {regime.regime_score})")
                        if c.is_defensive_stock:
                            score += 3
                            reasons.append("Defensive stock favored in current regime")

                    # Strategy alignment
                    if 'Momentum' in regime.favored_strategies:
                        if c.rs_rating is not None and c.rs_rating >= 70:
                            score += 3
                            reasons.append("Momentum favored in current regime")
                    if 'Value' in regime.favored_strategies:
                        if c.pe_ratio is not None and 0 < c.pe_ratio < 15:
                            score += 3
                            reasons.append("Value favored in current regime")

                except Exception as e:
                    logger.debug(f"Error getting multi-regime: {e}")

            # ============================================================
            # PHASE 2: GEX ADJUSTMENT
            # ============================================================
            if c.gex_signal == 'BULLISH':
                score += 5
                reasons.append(f"GEX bullish ({c.gex_regime})")
            elif c.gex_signal == 'BEARISH':
                score -= 5
                risks.append(f"GEX bearish ({c.gex_regime})")
            elif c.gex_signal == 'PINNED' and c.max_gamma_strike is not None and c.max_gamma_strike > 0:
                if c.current_price and c.current_price > 0:
                    dist_to_pin = abs(c.current_price - c.max_gamma_strike) / c.current_price * 100
                    if dist_to_pin < 2:
                        risks.append(f"Price pinned near ${c.max_gamma_strike:.0f}")

            # ============================================================
            # PHASE 2: DARK POOL ADJUSTMENT
            # ============================================================
            if c.institutional_bias == 'BUYING':
                score += 8
                dp_str = str(c.dark_pool_score) if c.dark_pool_score else "N/A"
                reasons.append(f"Institutional buying detected (DP score: {dp_str})")
            elif c.institutional_bias == 'SELLING':
                score -= 8
                dp_str = str(c.dark_pool_score) if c.dark_pool_score else "N/A"
                risks.append(f"Institutional selling detected (DP score: {dp_str})")

            # ============================================================
            # PHASE 2: CROSS-ASSET ADJUSTMENT
            # ============================================================
            if c.cross_asset_favored_sectors and c.sector in c.cross_asset_favored_sectors:
                score += 5
                reasons.append(f"Sector favored by cross-asset ({c.cycle_phase})")
            if c.cross_asset_risk_on is False and c.is_growth_stock:
                score -= 3
                risks.append("Cross-asset signals risk-off")

            # PHASE 4: INSIDER SCORING
            if c.insider_ceo_bought:
                score += 15
                reasons.append("🎯 CEO bought stock")
            if c.insider_cfo_bought:
                score += 12
                reasons.append("💰 CFO bought stock")
            if c.insider_cluster_buying:
                score += 10
                reasons.append("🔥 Cluster buying (3+ insiders)")
            if c.insider_cluster_selling:
                score -= 10
                risks.append("🚨 Cluster selling")
            if c.insider_signal == "STRONG_BUY":
                score += 8
            elif c.insider_signal == "STRONG_SELL":
                score -= 8

            # PHASE 4: 13F SCORING
            if c.inst_buffett_added:
                score += 12
                reasons.append("🎯 Buffett ADDED position")
            elif c.inst_buffett_owns:
                score += 6
                reasons.append("✓ Buffett holds")
            if c.inst_activist_involved:
                score += 3
                reasons.append("📢 Activist involved")
            if len(c.inst_new_buyers) >= 2:
                score += 5
                reasons.append(f"13F: {len(c.inst_new_buyers)} new buyers")

            # ============================================================
            # PHASE 3: SENTIMENT NLP ADJUSTMENT - FIXED: handle None
            # ============================================================
            if c.nlp_sentiment_score is not None:
                if c.nlp_sentiment_score >= 70:
                    score += 8
                    reasons.append(f"Strong NLP sentiment (score: {c.nlp_sentiment_score}, tone: {c.nlp_management_tone})")
                elif c.nlp_sentiment_score >= 60:
                    score += 4
                    reasons.append(f"Positive NLP sentiment (score: {c.nlp_sentiment_score})")
                elif c.nlp_sentiment_score <= 30:
                    score -= 8
                    risks.append(f"Negative NLP sentiment (score: {c.nlp_sentiment_score})")
                elif c.nlp_sentiment_score <= 40:
                    score -= 4
                    risks.append(f"Weak NLP sentiment (score: {c.nlp_sentiment_score})")

            # ============================================================
            # PHASE 3: EARNINGS WHISPER ADJUSTMENT - FIXED: handle None
            # ============================================================
            if c.days_to_earnings is not None and c.days_to_earnings <= 14:
                if c.whisper_beat_probability is not None:
                    if c.whisper_beat_probability >= 70:
                        score += 8
                        exp_str = f"{c.whisper_expected_surprise:+.1f}%" if c.whisper_expected_surprise else "N/A"
                        reasons.append(f"🎯 High beat probability ({c.whisper_beat_probability:.0f}%, exp: {exp_str})")
                    elif c.whisper_beat_probability >= 60:
                        score += 4
                        reasons.append(f"Likely beat ({c.whisper_beat_probability:.0f}%)")
                    elif c.whisper_beat_probability <= 30:
                        score -= 10
                        risks.append(f"⚠️ High miss probability ({100-c.whisper_beat_probability:.0f}%)")
                    elif c.whisper_beat_probability <= 40:
                        score -= 5
                        risks.append(f"Elevated miss risk ({100-c.whisper_beat_probability:.0f}%)")

            # ============================================================
            # FINAL SCORE AND ACTION
            # ============================================================
            c.ai_score = max(0, min(100, score))

            if c.ai_score >= 75:
                c.ai_action = 'STRONG_BUY'
            elif c.ai_score >= 60:
                c.ai_action = 'BUY'
            elif c.ai_score >= 40:
                c.ai_action = 'HOLD'
            else:
                c.ai_action = 'AVOID'

            # Set price targets - FIXED: handle None
            if c.support_1 and c.support_1 > 0:
                c.ai_stop_loss = c.support_1
            elif c.current_price:
                c.ai_stop_loss = c.current_price * 0.95

            if c.resistance_1 and c.resistance_1 > 0:
                c.ai_target_1 = c.resistance_1
            elif c.price_target and c.price_target > 0:
                c.ai_target_1 = c.price_target
            elif c.current_price:
                c.ai_target_1 = c.current_price * 1.10

            if c.price_target_high and c.price_target_high > 0:
                c.ai_target_2 = c.price_target_high
            elif c.ai_target_1:
                c.ai_target_2 = c.ai_target_1 * 1.1

            c.ai_entry = c.current_price

            # Best catalyst
            if reasons:
                c.ai_catalyst = reasons[0]

            c.ai_risks = risks
            c.ai_reasoning = "; ".join(reasons[:3]) if reasons else "Meets basic criteria"

        # Sort by AI score - FIXED: handle None
        candidates.sort(key=lambda x: x.ai_score if x.ai_score is not None else -1, reverse=True)

        return candidates

    def _apply_exposure_constraints(self, candidates: List[TradeCandidate],
                                    portfolio_positions: List[Dict] = None) -> List[TradeCandidate]:
        """Apply exposure constraints to limit position sizes."""
        if not EXPOSURE_CONTROL_AVAILABLE or not self.exposure_controller:
            for c in candidates:
                c.max_position_weight = 0.10
            return candidates

        current_positions = []
        sector_weights = {}

        if portfolio_positions:
            for pos in portfolio_positions:
                symbol = pos.get('symbol') or pos.get('ticker', '')
                sector = pos.get('sector', 'Unknown')
                weight = float(pos.get('weight', 0))

                if symbol and symbol not in ['USD', 'CASH']:
                    current_positions.append({
                        'ticker': symbol,
                        'sector': sector,
                        'weight': weight,
                    })
                    if sector not in sector_weights:
                        sector_weights[sector] = 0
                    sector_weights[sector] += weight

        for c in candidates:
            try:
                constraints = self.exposure_controller.get_position_constraints(
                    ticker=c.ticker,
                    sector=c.sector,
                    current_positions=current_positions,
                )
                c.max_position_weight = constraints.max_weight
                c.exposure_constraints = constraints.reasons
                c.sector_current_weight = sector_weights.get(c.sector, 0)
            except Exception as e:
                logger.debug(f"{c.ticker}: Exposure constraint error: {e}")
                c.max_position_weight = 0.10

        return candidates

    def _apply_crowding_analysis(self, candidates: List[TradeCandidate]) -> List[TradeCandidate]:
        """Apply crowding score analysis."""
        if not CROWDING_SCORE_AVAILABLE:
            return candidates

        for c in candidates:
            try:
                metrics = get_crowding_score(c.ticker)
                c.crowding_score = metrics.total_crowding_score
                c.crowding_level = metrics.crowding_level.value
                c.short_squeeze_risk_crowding = metrics.short_squeeze_risk
                c.crowding_warnings = metrics.warnings
            except Exception as e:
                logger.debug(f"{c.ticker}: Crowding analysis failed: {e}")

        return candidates

    def _apply_phase2_analysis(self, candidates: List[TradeCandidate]) -> List[TradeCandidate]:
        """Apply Phase 2 analysis: GEX, Dark Pool, Cross-Asset."""
        if CROSS_ASSET_AVAILABLE and self._cached_cross_asset is None:
            try:
                self._cached_cross_asset = get_cross_asset_signals()
            except Exception as e:
                logger.debug(f"Cross-asset analysis failed: {e}")

        for c in candidates:
            # GEX Analysis
            if GEX_AVAILABLE:
                try:
                    gex = analyze_gex(c.ticker, c.current_price)
                    c.gex_regime = gex.gex_regime.value
                    c.gex_signal = gex.signal
                    c.gex_signal_strength = gex.signal_strength
                    c.max_gamma_strike = gex.max_gamma_strike
                    c.call_wall = gex.call_wall
                    c.put_wall = gex.put_wall
                    c.gex_warnings = gex.warnings
                except Exception as e:
                    logger.debug(f"{c.ticker}: GEX analysis failed: {e}")

            # Dark Pool Analysis
            if DARK_POOL_AVAILABLE:
                try:
                    dp = analyze_dark_pool(c.ticker)
                    c.dark_pool_sentiment = dp.sentiment.value
                    c.dark_pool_score = dp.sentiment_score
                    c.institutional_bias = dp.institutional_bias
                    c.block_buy_volume = dp.block_buy_volume
                    c.block_sell_volume = dp.block_sell_volume
                    c.dark_pool_warnings = dp.warnings
                except Exception as e:
                    logger.debug(f"{c.ticker}: Dark pool analysis failed: {e}")

            # PHASE 4: INSIDER TRANSACTIONS
            if INSIDER_TRACKER_AVAILABLE:
                try:
                    insider = get_insider_signal(c.ticker)
                    c.insider_signal = insider.signal
                    c.insider_signal_strength = insider.signal_strength
                    c.insider_ceo_bought = insider.ceo_bought
                    c.insider_cfo_bought = insider.cfo_bought
                    c.insider_cluster_buying = insider.cluster_buying
                    c.insider_cluster_selling = insider.cluster_selling
                    c.insider_net_value = insider.net_value
                except Exception as e:
                    logger.debug(f"{c.ticker}: Insider error: {e}")

            # PHASE 4: 13F INSTITUTIONAL
            if INSTITUTIONAL_13F_AVAILABLE:
                try:
                    inst = get_institutional_ownership(c.ticker)
                    c.inst_13f_signal = inst.signal
                    c.inst_13f_strength = inst.signal_strength
                    c.inst_buffett_owns = inst.buffett_owns
                    c.inst_buffett_added = inst.buffett_added
                    c.inst_activist_involved = inst.activist_involved
                    c.inst_new_buyers = inst.new_buyers[:3]
                    c.inst_num_notable = inst.num_institutions
                except Exception as e:
                    logger.debug(f"{c.ticker}: 13F error: {e}")

            # Cross-Asset Context (same for all)
            if self._cached_cross_asset:
                try:
                    xa = self._cached_cross_asset
                    c.cross_asset_signal = xa.primary_signal.value
                    c.cross_asset_risk_on = xa.risk_appetite == "RISK_ON"
                    c.cycle_phase = xa.cycle_phase.value
                    c.cross_asset_favored_sectors = xa.favored_sectors
                except Exception as e:
                    logger.debug(f"{c.ticker}: Cross-asset assignment failed: {e}")

        return candidates

    def _apply_phase3_analysis(self, candidates: List[TradeCandidate]) -> List[TradeCandidate]:
        """Apply Phase 3 analysis: Sentiment NLP, Earnings Whisper."""
        if self._llm_available is None and SENTIMENT_NLP_AVAILABLE:
            try:
                self._llm_available = is_llm_available()
            except Exception:
                self._llm_available = False

        for c in candidates:
            # Earnings Whisper - FIXED: handle None days_to_earnings
            if EARNINGS_WHISPER_AVAILABLE:
                if c.days_to_earnings is not None and c.days_to_earnings <= 30:
                    try:
                        whisper = get_earnings_whisper(c.ticker)
                        c.whisper_prediction = whisper.prediction.value
                        c.whisper_beat_probability = whisper.beat_probability
                        c.whisper_expected_surprise = whisper.expected_surprise_pct
                        c.whisper_signal = whisper.signal.value
                        c.whisper_revision_score = whisper.revision_score
                        c.whisper_options_score = whisper.options_score
                        c.whisper_historical_score = whisper.historical_score
                        c.whisper_warnings = whisper.warnings

                        if whisper.earnings_date and whisper.earnings_date != "Unknown":
                            c.earnings_date = whisper.earnings_date
                            c.days_to_earnings = whisper.days_to_earnings
                    except Exception as e:
                        logger.debug(f"{c.ticker}: Earnings whisper failed: {e}")

            # Sentiment NLP
            if SENTIMENT_NLP_AVAILABLE:
                try:
                    news_headlines = self._get_recent_news(c.ticker)
                    if news_headlines:
                        sentiment = analyze_news_sentiment(c.ticker, news_headlines)
                        c.nlp_sentiment = sentiment.sentiment_level.value
                        c.nlp_sentiment_score = sentiment.sentiment_score
                        c.nlp_management_tone = sentiment.management_tone.value
                        c.nlp_key_positives = sentiment.key_positives[:3]
                        c.nlp_key_negatives = sentiment.key_negatives[:3]
                        c.nlp_summary = sentiment.summary
                except Exception as e:
                    logger.debug(f"{c.ticker}: Sentiment NLP failed: {e}")

        return candidates

    def _get_recent_news(self, ticker: str) -> List[str]:
        """Get recent news headlines for a ticker from database."""
        try:
            query = """
                SELECT headline FROM news_articles
                WHERE ticker = %s
                AND published_at >= NOW() - INTERVAL '7 days'
                ORDER BY published_at DESC
                LIMIT 10
            """
            df = pd.read_sql(query, self.engine, params=(ticker,))
            return df['headline'].tolist() if not df.empty else []
        except Exception:
            try:
                query = """
                    SELECT title as headline FROM news
                    WHERE ticker = %s
                    AND date >= CURRENT_DATE - INTERVAL '7 days'
                    ORDER BY date DESC
                    LIMIT 10
                """
                df = pd.read_sql(query, self.engine, params=(ticker,))
                return df['headline'].tolist() if not df.empty else []
            except Exception:
                return []

    def _generate_summary(self, top_picks: List[TradeCandidate], market_ctx) -> str:
        """Generate a summary of today's picks."""
        lines = [
            f"🤖 AI TRADE IDEAS - {date.today()}",
            f"Generated at {datetime.now().strftime('%H:%M')}",
            f"",
        ]

        lines.append(f"📊 MARKET: {market_ctx.market_regime} | VIX: {market_ctx.vix_level:.1f} ({market_ctx.vix_regime})")

        if market_ctx.high_impact_today:
            lines.append("⚠️ HIGH IMPACT EVENT TODAY - Reduce position sizes")
        elif market_ctx.fed_meeting_this_week:
            lines.append("⚠️ Fed meeting this week - Expect volatility")

        if market_ctx.hot_sectors:
            lines.append(f"🔥 Hot Sectors: {', '.join(market_ctx.hot_sectors)}")

        lines.append("")
        lines.append(f"🏆 TOP {len(top_picks)} PICKS:")

        for i, pick in enumerate(top_picks, 1):
            score_str = f"{pick.ai_score:.0f}" if pick.ai_score else "N/A"
            entry_str = f"${pick.ai_entry:.2f}" if pick.ai_entry else "N/A"
            stop_str = f"${pick.ai_stop_loss:.2f}" if pick.ai_stop_loss else "N/A"
            target_str = f"${pick.ai_target_1:.2f}" if pick.ai_target_1 else "N/A"

            lines.append(f"  {i}. {pick.ticker} - {pick.ai_action} (Score: {score_str})")
            lines.append(f"     Entry: {entry_str} | Stop: {stop_str} | Target: {target_str}")
            lines.append(f"     {pick.ai_catalyst}")

        return "\n".join(lines)

    def get_detailed_analysis(self, candidate: TradeCandidate) -> str:
        """Get detailed AI analysis for a single candidate."""
        # Helper for safe formatting
        def fmt(val, spec=".0f", prefix="", suffix=""):
            if val is None:
                return "N/A"
            return f"{prefix}{val:{spec}}{suffix}"

        def fmt_pct(val, current_price):
            if val is None or current_price is None or current_price == 0:
                return "N/A"
            return f"{((val - current_price) / current_price * 100):.1f}%"

        lines = [
            f"{'=' * 60}",
            f"🤖 AI DETAILED ANALYSIS: {candidate.ticker}",
            f"{'=' * 60}",
            f"",
            f"📊 AI SCORE: {fmt(candidate.ai_score)}/100 ({candidate.ai_action})",
            f"💰 Current Price: {fmt(candidate.current_price, '.2f', '$')}",
            f"📈 Data Completeness: {candidate.data_completeness:.0%}",
            f"",
            f"🎯 TRADE SETUP:",
            f"   Entry: {fmt(candidate.ai_entry, '.2f', '$')}",
            f"   Stop Loss: {fmt(candidate.ai_stop_loss, '.2f', '$')} ({fmt_pct(candidate.ai_stop_loss, candidate.current_price)})",
            f"   Target 1: {fmt(candidate.ai_target_1, '.2f', '$')} ({fmt_pct(candidate.ai_target_1, candidate.current_price)})",
            f"   Target 2: {fmt(candidate.ai_target_2, '.2f', '$')} ({fmt_pct(candidate.ai_target_2, candidate.current_price)})",
            f"   Risk/Reward: {fmt(candidate.risk_reward_ratio, '.1f')}:1",
            f"   Max Position: {candidate.max_position_weight:.1%}" + (
                f" ⚠️ ({candidate.exposure_constraints[0]})" if candidate.exposure_constraints else ""),
            f"",
            f"📈 PLATFORM SIGNALS:",
            f"   Signal: {candidate.signal_type} (strength: {fmt(candidate.signal_strength)})",
            f"   Committee: {candidate.committee_verdict} ({fmt(candidate.committee_conviction)}% conviction)",
            f"   Sentiment Score: {fmt(candidate.sentiment_score)}",
            f"   Total Score: {fmt(candidate.total_score)}",
            f"",
            f"🔮 OPTIONS FLOW:",
            f"   Sentiment: {candidate.options_sentiment} (score: {fmt(candidate.options_score)})",
            f"   Put/Call Ratio: {fmt(candidate.put_call_ratio, '.2f')}",
            f"   Unusual Calls: {'YES 🚀' if candidate.unusual_calls else 'No' if candidate.unusual_calls is False else 'N/A'}",
            f"   Max Pain: {fmt(candidate.max_pain, '.2f', '$')}",
            f"",
            f"🩳 SHORT SQUEEZE:",
            f"   Score: {fmt(candidate.squeeze_score)}/100 ({candidate.squeeze_risk})",
            f"   Short % Float: {fmt(candidate.short_pct_float, '.1f', suffix='%')}",
            f"   Days to Cover: {fmt(candidate.days_to_cover, '.1f')}",
            f"",
            f"📊 TECHNICALS:",
            f"   Relative Strength: {fmt(candidate.rs_rating)}/100",
            f"   vs SPY (20d): {fmt(candidate.vs_spy_20d, '+.1f', suffix='%')}",
            f"   vs Sector (20d): {fmt(candidate.vs_sector_20d, '+.1f', suffix='%')}",
            f"   Trend: {candidate.trend_20d}",
            f"   RSI: {fmt(candidate.rsi_14)}",
            f"   Above 50 MA: {'✅' if candidate.above_50ma else '❌' if candidate.above_50ma is False else 'N/A'}",
            f"   Above 200 MA: {'✅' if candidate.above_200ma else '❌' if candidate.above_200ma is False else 'N/A'}",
            f"",
            f"📍 KEY LEVELS:",
            f"   Support: {fmt(candidate.support_1, '.2f', '$')} ({fmt(candidate.distance_to_support_pct, '.1f', suffix='% below')})",
            f"   Resistance: {fmt(candidate.resistance_1, '.2f', '$')}",
            f"",
            f"💧 LIQUIDITY: {candidate.liquidity_score}",
            f"   Avg $ Volume: {fmt(candidate.avg_dollar_volume, '.1f') if candidate.avg_dollar_volume is None else f'${candidate.avg_dollar_volume / 1e6:.1f}M'}",
            f"   Relative Volume: {fmt(candidate.relative_volume, '.1f', suffix='x')}",
            f"",
            f"🏦 CROWDING: {candidate.crowding_level} (score: {fmt(candidate.crowding_score)}/100)",
            f"   Squeeze Risk: {'⚠️ YES' if candidate.short_squeeze_risk_crowding else 'No' if candidate.short_squeeze_risk_crowding is False else 'N/A'}",
            f"",
            f"🌍 REGIME: {candidate.regime_state} (score: {fmt(candidate.regime_score)}/100)",
            f"   Risk Appetite: {'🟢 ON' if candidate.regime_risk_on else '🔴 OFF' if candidate.regime_risk_on is False else 'N/A'}",
            f"   Favored: {', '.join(candidate.regime_favored_strategies[:3]) if candidate.regime_favored_strategies else 'N/A'}",
            f"",
            f"📊 GEX/GAMMA:",
            f"   Regime: {candidate.gex_regime} | Signal: {candidate.gex_signal}",
            f"   Max Gamma: {fmt(candidate.max_gamma_strike, '.0f', '$')}" if candidate.max_gamma_strike else f"   Max Gamma: N/A",
            f"   Call Wall: {fmt(candidate.call_wall, '.0f', '$')} | Put Wall: {fmt(candidate.put_wall, '.0f', '$')}" if candidate.call_wall else f"   Walls: N/A",
            f"",
            f"🏦 DARK POOL:",
            f"   Sentiment: {candidate.dark_pool_sentiment} (score: {fmt(candidate.dark_pool_score)})",
            f"   Institutional Bias: {candidate.institutional_bias}",
            f"   Block Buy: {fmt(candidate.block_buy_volume, ',')} | Sell: {fmt(candidate.block_sell_volume, ',')}",
            f"",
            f"🌐 CROSS-ASSET:",
            f"   Signal: {candidate.cross_asset_signal} | Cycle: {candidate.cycle_phase}",
            f"   Risk: {'🟢 ON' if candidate.cross_asset_risk_on else '🔴 OFF' if candidate.cross_asset_risk_on is False else 'N/A'}",
            f"   Favored Sectors: {', '.join(candidate.cross_asset_favored_sectors[:3]) if candidate.cross_asset_favored_sectors else 'N/A'}",
            f"",
            f"🧠 SENTIMENT NLP:",
            f"   Sentiment: {candidate.nlp_sentiment} (score: {fmt(candidate.nlp_sentiment_score)})",
            f"   Management Tone: {candidate.nlp_management_tone}",
        ]

        if candidate.nlp_key_positives:
            lines.append(f"   ✅ Positives: {'; '.join(candidate.nlp_key_positives[:2])}")
        if candidate.nlp_key_negatives:
            lines.append(f"   ❌ Negatives: {'; '.join(candidate.nlp_key_negatives[:2])}")

        lines.extend([
            f"",
            f"🎯 EARNINGS WHISPER:",
            f"   Prediction: {candidate.whisper_prediction}",
            f"   Beat Probability: {fmt(candidate.whisper_beat_probability, '.0f', suffix='%')}",
            f"   Expected Surprise: {fmt(candidate.whisper_expected_surprise, '+.1f', suffix='%')}",
            f"   Signal: {candidate.whisper_signal}",
            f"   Components: Rev={fmt(candidate.whisper_revision_score)} | Opt={fmt(candidate.whisper_options_score)} | Hist={fmt(candidate.whisper_historical_score)}",
        ])

        if candidate.whisper_warnings:
            lines.append(f"   ⚠️ {'; '.join(candidate.whisper_warnings[:2])}")

        lines.extend([
            f"",
            f"📅 EVENTS:",
            f"   Earnings: {candidate.earnings_date or 'N/A'} ({fmt(candidate.days_to_earnings)} days) {'✅ Safe' if candidate.earnings_safe else '⚠️ CAUTION' if candidate.earnings_safe is False else 'Unknown'}",
            f"",
        ])

        # Add earnings analysis if available
        if candidate.earnings_sentiment and candidate.earnings_sentiment != "NOT_ANALYZED":
            lines.extend([
                f"📊 LAST EARNINGS REPORT:",
                f"   Sentiment: {candidate.earnings_sentiment} (score: {fmt(candidate.earnings_sentiment_score)}/100)",
                f"   EPS Surprise: {fmt(candidate.eps_surprise_pct, '+.1f', suffix='%')}",
                f"   Guidance: {candidate.guidance_direction}",
                f"   Score Impact: {fmt(candidate.earnings_score_adjustment, '+d')} points",
                f"",
            ])

        # Add Earnings Intelligence (IES/ECS) if available
        if candidate.ies is not None or (candidate.ecs_category and candidate.ecs_category != "PENDING"):
            lines.extend([
                f"📊 EARNINGS INTELLIGENCE:",
                f"   IES: {fmt(candidate.ies)}/100 ({candidate.ies_regime})",
                f"   ECS: {candidate.ecs_category}",
                f"   Position Scale: {fmt(candidate.ei_position_scale, '.0%')}",
                f"   Score Adjustment: {fmt(candidate.ei_score_adjustment, '+d')}",
            ])
            if candidate.ei_risk_flags:
                flags_str = ", ".join(candidate.ei_risk_flags)
                lines.append(f"   ⚠️ Risk Flags: {flags_str}")
            lines.append("")

        if candidate.currently_owned:
            lines.extend([
                f"💼 YOUR POSITION:",
                f"   Shares: {candidate.shares_owned}",
                f"   Weight: {candidate.current_weight_pct:.1f}%",
                f"   Avg Cost: ${candidate.avg_cost:.2f}",
                f"   P&L: ${candidate.unrealized_pnl:+,.0f} ({candidate.unrealized_pnl_pct:+.1f}%)",
                f"",
            ])

        lines.extend([
            f"✅ BULLISH FACTORS:",
            f"   {candidate.ai_reasoning}",
            f"",
        ])

        if candidate.ai_risks:
            lines.append(f"⚠️ RISKS:")
            for risk in candidate.ai_risks:
                lines.append(f"   • {risk}")

        return "\n".join(lines)


# Convenience function
def generate_trade_ideas(portfolio_positions: List[Dict] = None,
                         max_picks: int = 10) -> TradeIdeasResult:
    """Generate trade ideas."""
    generator = TradeIdeasGenerator()
    return generator.generate_ideas(portfolio_positions, max_picks)


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    generator = TradeIdeasGenerator()
    result = generator.generate_ideas(max_picks=5)

    print(result.market_context)
    print()
    print(result.summary)

    if result.top_picks:
        print("\n" + "=" * 60)
        print(generator.get_detailed_analysis(result.top_picks[0]))