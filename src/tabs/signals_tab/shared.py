"""
Signals Tab - Shared Configuration

All imports, feature flags, data classes, and utility functions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import logging
import json
import os
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# =============================================================================
# LOGGING
# =============================================================================
from src.utils.logging import get_logger
logger = get_logger(__name__)

_self_check_logger = logging.getLogger("signals_tab.verify")
_self_check_logger.info("signals_tab package loaded (modular version)")

# =============================================================================
# DATE PARSING
# =============================================================================
try:
    from dateutil import parser as date_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    date_parser = None
    DATEUTIL_AVAILABLE = False

# =============================================================================
# SQLALCHEMY
# =============================================================================
try:
    from sqlalchemy import text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    text = None
    SQLALCHEMY_AVAILABLE = False

# =============================================================================
# CORE IMPORTS
# =============================================================================
try:
    from src.core import (
        UnifiedSignal, MarketOverview, SignalSnapshot, ComponentScore,
        SignalStrength, RiskLevel, AssetType, BOND_ETFS,
        SignalEngine, get_signal_engine, get_market_overview,
    )
    SIGNAL_HUB_AVAILABLE = True
except ImportError as e:
    SIGNAL_HUB_AVAILABLE = False
    logger.error(f"Signal Hub not available: {e}")
    class UnifiedSignal: pass
    class SignalStrength: pass
    class RiskLevel: pass

try:
    from src.db.connection import get_engine, get_connection
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    get_engine = None
    get_connection = None
    logger.error("Database connection not available")

# =============================================================================
# OPTIONAL FEATURES
# =============================================================================
try:
    from src.signals.filing_signal import get_filing_insights
    SEC_INSIGHTS_AVAILABLE = True
except ImportError:
    SEC_INSIGHTS_AVAILABLE = False
    get_filing_insights = None

try:
    from src.ai.dual_analyst import DualAnalystService
    DUAL_ANALYST_AVAILABLE = True
except ImportError:
    DUAL_ANALYST_AVAILABLE = False
    DualAnalystService = None

try:
    from src.ml.institutional_signals_display import render_institutional_signals
    INSTITUTIONAL_SIGNALS_AVAILABLE = True
except ImportError:
    INSTITUTIONAL_SIGNALS_AVAILABLE = False
    render_institutional_signals = None

try:
    from src.ml.institutional_signals_context import (
        get_institutional_signals_context, get_trading_implications,
    )
    INSTITUTIONAL_CONTEXT_AVAILABLE = True
except ImportError:
    INSTITUTIONAL_CONTEXT_AVAILABLE = False
    get_institutional_signals_context = None
    get_trading_implications = None

try:
    from src.analytics.enhanced_scoring_integration import (
        apply_enhanced_scores_to_dataframe, get_enhanced_total_score,
        compute_enhanced_total_score, render_enhancement_breakdown,
        get_single_ticker_enhancement,
    )
    ENHANCED_SCORING_AVAILABLE = True
except ImportError:
    ENHANCED_SCORING_AVAILABLE = False

try:
    from src.analytics.enhanced_scores_db import (
        compute_and_save_enhanced_scores, load_enhanced_scores,
        run_migration as run_enhanced_scores_migration,
    )
    ENHANCED_SCORES_DB_AVAILABLE = True
    try:
        run_enhanced_scores_migration()
    except:
        pass
except ImportError:
    ENHANCED_SCORES_DB_AVAILABLE = False

try:
    from src.ml.signals_tab_ai import get_ai_probabilities_batch
    AI_PROB_AVAILABLE = True
except ImportError:
    AI_PROB_AVAILABLE = False

try:
    from src.analytics.earnings_intelligence.reaction_analyzer import analyze_post_earnings
    REACTION_ANALYZER_AVAILABLE = True
except ImportError:
    REACTION_ANALYZER_AVAILABLE = False
    analyze_post_earnings = None

try:
    from src.ml.ai_trading_system import AITradingSystem
    AI_SYSTEM_AVAILABLE = True
except ImportError:
    AI_SYSTEM_AVAILABLE = False
    AITradingSystem = None

try:
    from src.analytics.universe_scorer import UniverseScorer
    UNIVERSE_SCORER_AVAILABLE = True
except ImportError:
    UNIVERSE_SCORER_AVAILABLE = False
    UniverseScorer = None

try:
    from src.data.news import NewsCollector
    NEWS_COLLECTOR_AVAILABLE = True
except ImportError:
    NEWS_COLLECTOR_AVAILABLE = False
    NewsCollector = None

try:
    from src.screener.sentiment import SentimentAnalyzer
    SENTIMENT_ANALYZER_AVAILABLE = True
except ImportError:
    SENTIMENT_ANALYZER_AVAILABLE = False
    SentimentAnalyzer = None

try:
    from src.analytics.technical_analysis import TechnicalAnalyzer
    TECHNICAL_ANALYZER_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYZER_AVAILABLE = False
    TechnicalAnalyzer = None

try:
    from src.analytics.fundamental_analysis import FundamentalAnalyzer
    FUNDAMENTAL_ANALYZER_AVAILABLE = True
except ImportError:
    FUNDAMENTAL_ANALYZER_AVAILABLE = False
    FundamentalAnalyzer = None

try:
    from src.analytics.options_flow import OptionsFlowAnalyzer
    OPTIONS_FLOW_AVAILABLE = True
except ImportError:
    OPTIONS_FLOW_AVAILABLE = False
    OptionsFlowAnalyzer = None

@dataclass
class MarketSnapshot:
    """
    Single-timestamp market data snapshot for consistent analysis.
    All derived metrics should be computed from this snapshot only.
    """
    ticker: str
    snapshot_ts_utc: datetime

    # Price data
    price: float = 0.0
    price_source: str = "unknown"  # 'regular', 'premarket', 'afterhours'
    previous_close: float = 0.0
    price_change_pct: float = 0.0

    # 52W data
    week_52_high: float = 0.0
    week_52_low: float = 0.0
    pct_from_52w_high: float = 0.0
    pct_from_52w_low: float = 0.0

    # Options data
    options_ts: Optional[datetime] = None
    max_pain: float = 0.0
    max_pain_expiry: str = "UNKNOWN"
    max_pain_distance_pct: float = 0.0
    pc_ratio_volume: float = 1.0
    pc_ratio_oi: float = 1.0
    call_volume: int = 0
    put_volume: int = 0
    options_data_quality: str = "UNKNOWN"  # 'FRESH', 'STALE', 'MISSING'

    # Fundamentals
    fundamentals_asof: Optional[date] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    dividend_yield: Optional[float] = None
    debt_equity: Optional[float] = None

    # Scores
    sentiment_score: Optional[int] = None
    technical_score: Optional[int] = None
    fundamental_score: Optional[int] = None
    options_score: Optional[int] = None
    earnings_score: Optional[int] = None
    total_score: Optional[int] = None

    # Data quality
    components_available: int = 0
    components_total: int = 5
    staleness_penalty: float = 0.0

    # Target
    analyst_target: Optional[float] = None
    target_upside_pct: Optional[float] = None


@dataclass
class TradeLevels:
    """
    Pre-computed trade levels - LLM should NOT calculate these.
    """
    entry_price: float = 0.0
    entry_method: str = "market"  # 'market', 'limit_2pct', 'pullback_5pct'

    stop_loss: float = 0.0
    stop_method: str = "pct_based"  # 'max_pain', 'atr_2x', 'support_level', 'pct_based'
    stop_distance_pct: float = 0.0

    target_price: float = 0.0
    target_method: str = "analyst"  # 'analyst_consensus', 'rr_2.5x', 'resistance'
    target_upside_pct: float = 0.0

    risk_reward_ratio: float = 0.0
    position_size_multiplier: float = 1.0  # 0.25x, 0.5x, 1.0x


@dataclass
class PolicyDecision:
    """
    Deterministic policy decision based on signals and data quality.
    """
    action: str = "HOLD"  # BUY, SELL, HOLD
    max_size: float = 1.0  # Position size cap
    confidence: str = "LOW"  # LOW, MEDIUM, HIGH
    conflict_flag: bool = False
    reasons: List[str] = field(default_factory=list)


def apply_uncertainty_shrinkage(
    raw_score: float,
    components_available: int,
    components_total: int = 5,
    staleness_penalty: float = 0.0
) -> Tuple[float, str]:
    """
    Shrink score toward 50 based on data quality.

    Missing/stale data should reduce confidence, not produce false neutral.

    Args:
        raw_score: The computed score (0-100)
        components_available: Number of components with fresh data
        components_total: Total components expected
        staleness_penalty: 0-1, penalty for stale data

    Returns:
        Tuple of (shrunk_score, confidence_level)
    """
    if components_available == 0:
        return 50.0, "NONE"

    # k decreases when components are missing or stale
    k = (components_available / components_total) * (1 - staleness_penalty)

    # Shrink toward neutral (50)
    shrunk_score = 50 + k * (raw_score - 50)

    # Determine confidence based on data quality
    if k >= 0.8:
        confidence = "HIGH"
    elif k >= 0.5:
        confidence = "MEDIUM"
    elif k >= 0.3:
        confidence = "LOW"
    else:
        confidence = "VERY_LOW"

    return round(shrunk_score, 1), confidence


def compute_trade_levels(
    current_price: float,
    signal_score: int,
    risk_level: str,
    analyst_target: Optional[float] = None,
    max_pain: Optional[float] = None,
    week_52_high: Optional[float] = None,
    week_52_low: Optional[float] = None
) -> TradeLevels:
    """
    Compute trade levels deterministically in code.
    LLM should NOT recalculate these.
    """
    levels = TradeLevels()
    levels.entry_price = current_price

    # Entry method based on signal strength
    if signal_score >= 80:
        levels.entry_method = "market"
    elif signal_score >= 65:
        levels.entry_price = current_price * 0.98
        levels.entry_method = "limit_2pct"
    else:
        levels.entry_price = current_price * 0.95
        levels.entry_method = "pullback_5pct"

    # Stop loss based on risk level
    risk_pct = {
        'LOW': 5.0,
        'MEDIUM': 7.0,
        'HIGH': 10.0,
        'EXTREME': 12.0
    }.get(risk_level, 7.0)

    # Check if max pain can be used as stop
    if max_pain and max_pain > 0:
        max_pain_pct_below = ((current_price - max_pain) / current_price) * 100
        if 3 <= max_pain_pct_below <= 15:  # Max pain within reasonable range
            levels.stop_loss = max_pain
            levels.stop_method = "max_pain"
            levels.stop_distance_pct = -max_pain_pct_below
        else:
            levels.stop_loss = levels.entry_price * (1 - risk_pct / 100)
            levels.stop_method = "pct_based"
            levels.stop_distance_pct = -risk_pct
    else:
        levels.stop_loss = levels.entry_price * (1 - risk_pct / 100)
        levels.stop_method = "pct_based"
        levels.stop_distance_pct = -risk_pct

    # Target price
    if analyst_target and analyst_target > current_price:
        levels.target_price = analyst_target
        levels.target_method = "analyst_consensus"
    else:
        # Use 2.5:1 R/R
        risk_amount = levels.entry_price - levels.stop_loss
        levels.target_price = levels.entry_price + (risk_amount * 2.5)
        levels.target_method = "rr_2.5x"

    levels.target_upside_pct = ((levels.target_price - levels.entry_price) / levels.entry_price) * 100

    # Risk/reward ratio
    if levels.stop_distance_pct != 0:
        levels.risk_reward_ratio = abs(levels.target_upside_pct / levels.stop_distance_pct)

    return levels


def resolve_signal_conflict(
    platform_signal: str,
    platform_score: int,
    committee_signal: str,
    committee_confidence: float,
    components_fresh: int,
    ml_status: str
) -> PolicyDecision:
    """
    Deterministic conflict resolution.

    This function mechanically determines action, size cap, and confidence
    based on signal conflicts and data quality.
    """
    decision = PolicyDecision()
    decision.action = platform_signal
    decision.max_size = 1.0
    decision.confidence = "HIGH"

    # Rule 1: ML blocked = cap at 0.25x, MEDIUM confidence max
    if ml_status in ['BLOCKED', 'BLOCKED_INSUFFICIENT_DATA', 'UNRELIABLE']:
        decision.max_size = min(decision.max_size, 0.25)
        decision.confidence = "MEDIUM"
        decision.reasons.append(f"ML blocked ({ml_status}) - size capped at 0.25x")

    # Rule 2: Committee disagrees = cap at 0.25x, confidence to LOW
    if platform_signal != committee_signal:
        decision.max_size = min(decision.max_size, 0.25)
        decision.confidence = "LOW"
        decision.conflict_flag = True
        decision.reasons.append(f"Signal conflict (Platform {platform_signal} vs Committee {committee_signal})")

    # Rule 3: < 3 fresh components = HOLD regardless of score
    if components_fresh < 3:
        decision.action = "HOLD"
        decision.confidence = "LOW"
        decision.reasons.append(f"Insufficient fresh data ({components_fresh}/5 components)")

    # Rule 4: Extreme conflict (BUY vs SELL) = force HOLD
    if (platform_signal == "BUY" and "SELL" in committee_signal) or \
       (platform_signal == "SELL" and "BUY" in committee_signal):
        decision.action = "HOLD"
        decision.confidence = "LOW"
        decision.reasons.append("Extreme signal conflict - forced HOLD")

    # Rule 5: Low score + conflict = HOLD
    if platform_score < 55 and decision.conflict_flag:
        decision.action = "HOLD"
        decision.reasons.append("Low score with conflict - forced HOLD")

    return decision


def compute_options_confidence(
    total_volume: int,
    market_is_open: bool,
    hours_since_open: float,
    oi_concentration_near_spot: float = 0.0
) -> Tuple[str, str]:
    """
    Compute confidence level for options metrics.

    Returns:
        Tuple of (confidence_level, reason)
    """
    # Volume thresholds
    MIN_VOLUME_FOR_HIGH = 10000
    MIN_VOLUME_FOR_MEDIUM = 1000

    if total_volume < MIN_VOLUME_FOR_MEDIUM:
        return "LOW", f"Low volume ({total_volume:,} contracts)"

    # Early in session = lower confidence
    if market_is_open and hours_since_open < 1.5:
        return "LOW", "Early session - volume incomplete"

    if total_volume >= MIN_VOLUME_FOR_HIGH:
        return "HIGH", f"High volume ({total_volume:,} contracts)"

    return "MEDIUM", f"Moderate volume ({total_volume:,} contracts)"


def validate_max_pain(
    max_pain: float,
    spot_price: float,
    days_to_expiry: int,
    oi_concentration: float = 0.0
) -> Tuple[bool, str, str]:
    """
    Validate if max pain can be used as support/resistance.

    Returns:
        Tuple of (is_valid, confidence, reason)
    """
    if max_pain <= 0 or spot_price <= 0:
        return False, "NONE", "Missing data"

    distance_pct = abs((max_pain - spot_price) / spot_price) * 100

    # Rule: >20% = data error
    if distance_pct > 20:
        return False, "NONE", f"DATA ERROR: {distance_pct:.1f}% from spot (>20%)"

    # Rule: >15% = questionable
    if distance_pct > 15:
        return False, "LOW", f"Far from spot ({distance_pct:.1f}%)"

    # Rule: Expiry too far = less relevant
    if days_to_expiry > 14:
        return True, "LOW", f"Expiry {days_to_expiry}d out - less pinning effect"

    # Rule: Within 10% and expiry < 7 days = high confidence
    if distance_pct <= 10 and days_to_expiry <= 7:
        return True, "HIGH", f"Near spot ({distance_pct:.1f}%), expiry in {days_to_expiry}d"

    return True, "MEDIUM", f"{distance_pct:.1f}% from spot, {days_to_expiry}d to expiry"



# =============================================================================
# NUMPY TYPE CONVERSION HELPER
# =============================================================================
def _to_native(val):
    """
    Convert numpy/pandas types to native Python types for SQL compatibility.
    Prevents 'np.float64(...)' from being passed as string to PostgreSQL.
    Also handles infinity values which can't be stored in numeric fields.
    """
    if val is None:
        return None
    # Check for numpy types
    if hasattr(val, 'item'):  # numpy scalar types have .item() method
        val = val.item()
    # Check for pandas NA types
    if pd.isna(val):
        return None
    # Handle infinity values
    if isinstance(val, float):
        if val == float('inf') or val == float('-inf'):
            return None
    # Handle string values that might be numeric
    if isinstance(val, str):
        try:
            val = float(val)
            if val == float('inf') or val == float('-inf'):
                return None
        except:
            return None
    # Check for numpy types by module name (handles case where numpy isn't imported)
    type_name = type(val).__module__
    if type_name == 'numpy':
        if hasattr(val, 'item'):
            return val.item()
        return float(val) if 'float' in str(type(val)) else int(val) if 'int' in str(type(val)) else val
    return val

# Enhanced Scoring Integration
try:
    from src.analytics.enhanced_scoring_integration import (
        apply_enhanced_scores_to_dataframe,
        get_enhanced_total_score,
        compute_enhanced_total_score,
        render_enhancement_breakdown,
        get_single_ticker_enhancement,
    )
    ENHANCED_SCORING_AVAILABLE = True
    logger.info("Enhanced scoring module loaded successfully")
except ImportError as e:
    ENHANCED_SCORING_AVAILABLE = False
    logger.warning(f"Enhanced scoring not available: {e}")

# Enhanced Scores Database Storage
try:
    from src.analytics.enhanced_scores_db import (
        compute_and_save_enhanced_scores,
        load_enhanced_scores,
        run_migration as run_enhanced_scores_migration,
    )
    ENHANCED_SCORES_DB_AVAILABLE = True
    # Run migration once on import (creates table if not exists)
    try:
        run_enhanced_scores_migration()
    except Exception as e:
        logger.debug(f"Enhanced scores migration skipped: {e}")
except ImportError:
    ENHANCED_SCORES_DB_AVAILABLE = False
    logger.debug("Enhanced scores DB not available")

# AI Win Probability for table


def _get_ai_probabilities_for_table(ticker_scores: dict) -> dict:
    """
    Get AI win probabilities for display in the signals table.
    Uses caching to avoid recomputation.
    """
    import time

    # Check cache (5 minute TTL)
    cache_key = 'ai_table_probs'
    cache_time_key = 'ai_table_probs_time'
    current_time = time.time()

    if cache_key in st.session_state and cache_time_key in st.session_state:
        if current_time - st.session_state[cache_time_key] < 300:
            return st.session_state[cache_key]

    if not AI_SYSTEM_AVAILABLE:
        return {}

    # Initialize AI system if needed
    if 'ai_system' not in st.session_state:
        try:
            system = AITradingSystem()
            system.initialize(train_if_needed=False)
            if system.rag_memory and len(system.rag_memory._cards) == 0:
                system.rag_memory.load_from_db()
            st.session_state.ai_system = system
        except Exception:
            return {}

    ai_system = st.session_state.ai_system

    if not ai_system.ml_predictor or not ai_system.ml_predictor.models:
        return {}

    # Compute probabilities
    results = {}
    for ticker, scores in ticker_scores.items():
        try:
            scores['ticker'] = ticker
            ml_pred = ai_system.ml_predictor.predict(scores)
            results[ticker] = {
                'prob': ml_pred.prob_win_5d,
                'ev': ml_pred.ev_5d
            }
        except Exception:
            continue

    # Cache results
    st.session_state[cache_key] = results
    st.session_state[cache_time_key] = current_time

    return results

def _get_cached_reaction_analysis(ticker: str, force_refresh: bool = False):
    """Get reaction analysis from cache or run fresh analysis.

    Caches results in session_state to avoid re-running expensive analysis
    every time the deep dive is opened.

    Cache key: reaction_cache_{ticker}
    Cache expires: After 1 hour or on force_refresh
    """
    if not REACTION_ANALYZER_AVAILABLE:
        return None

    cache_key = f"reaction_cache_{ticker}"
    cache_time_key = f"reaction_cache_time_{ticker}"

    # Check if we have a cached result that's less than 1 hour old
    if not force_refresh and cache_key in st.session_state:
        cached_time = st.session_state.get(cache_time_key)
        if cached_time:
            age_minutes = (datetime.now() - cached_time).total_seconds() / 60
            if age_minutes < 60:  # Cache valid for 1 hour
                logger.debug(f"{ticker}: Using cached reaction analysis ({age_minutes:.0f} min old)")
                return st.session_state[cache_key]

    # Run fresh analysis
    try:
        with st.spinner(f"Analyzing {ticker} earnings reaction..."):
            reaction = analyze_post_earnings(ticker)

            # Cache the result
            st.session_state[cache_key] = reaction
            st.session_state[cache_time_key] = datetime.now()

            logger.info(f"{ticker}: Cached new reaction analysis")
            return reaction
    except Exception as e:
        logger.error(f"{ticker}: Reaction analysis error: {e}")
        return None


# =============================================================================
# EXTENDED HOURS PRICE (moved here to avoid circular imports)
# =============================================================================
def _get_extended_hours_price(ticker: str) -> dict:
    """Get pre-market or after-hours price if available."""
    result = {
        'has_extended': False,
        'regular_close': 0,
        'extended_price': 0,
        'extended_change': 0,
        'extended_change_pct': 0,
        'session': '',  # 'pre-market' or 'after-hours'
    }

    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)

        # Get quote data
        info = stock.info

        # Get all available price fields
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        yf_prev_close = info.get('previousClose') or info.get('regularMarketPreviousClose', 0)
        pre_price = info.get('preMarketPrice', 0) or 0
        post_price = info.get('postMarketPrice', 0) or 0

        # Determine the best baseline (last regular market close)
        if pre_price > 0 or post_price > 0:
            baseline = current_price if current_price > 0 else yf_prev_close
        else:
            baseline = yf_prev_close if yf_prev_close > 0 else current_price

        result['regular_close'] = baseline

        # Check for pre-market
        if pre_price > 0:
            result['has_extended'] = True
            result['extended_price'] = pre_price
            result['session'] = 'Pre-Market'
            if baseline > 0:
                result['extended_change'] = pre_price - baseline
                result['extended_change_pct'] = (result['extended_change'] / baseline) * 100
            return result

        # Check for after-hours
        if post_price > 0:
            result['has_extended'] = True
            result['extended_price'] = post_price
            result['session'] = 'After-Hours'
            if baseline > 0:
                result['extended_change'] = post_price - baseline
                result['extended_change_pct'] = (result['extended_change'] / baseline) * 100
            return result

        # Current price during market hours (no extended session)
        if current_price and current_price > 0:
            result['extended_price'] = current_price
            result['session'] = 'Market'
            if yf_prev_close > 0:
                result['extended_change'] = current_price - yf_prev_close
                result['extended_change_pct'] = (result['extended_change'] / yf_prev_close) * 100
                if abs(result['extended_change_pct']) > 0.1:
                    result['has_extended'] = True

    except Exception as e:
        logger.debug(f"Extended hours error for {ticker}: {e}")

    return result