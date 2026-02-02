"""
Signals Tab - Clean, minimal implementation with persistent job tracking.

FIXES APPLIED:
1. Better date parsing for news articles (handles Google News format)
2. Refresh button now clears SignalEngine cache to show updated scores
3. Fixed UniverseScorer usage (list not dict) - Issue #1
4. Separate FundamentalAnalyzer/TechnicalAnalyzer calls - Issue #2
5. Recompute total_score fresh (not copy stale) - Issue #4
6. Data quality tracking - Issue #5
7. Per-ticker MAX(date) SQL query - prevents ticker dropout
8. Enhanced scoring: PE relative, price target, MACD, insider, analyst revisions
9. numpy type conversion (_to_native) - fixes np.float64 SQL errors
10. Pre-logging of tickers - prevents infinite loop on errors
11. Progress value clamping - fixes > 1.0 progress error on completion
12. Google Finance-style chart with period selectors and key stats

VERSION: 2024-12-23-CHART-UPGRADE
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional

# ============================================================================
# SELF-VERIFICATION: Log which file is loaded (remove after confirming)
# ============================================================================
import logging
_self_check_logger = logging.getLogger("signals_tab.verify")
_self_check_logger.info(f"✅ signals_tab.py loaded from: {__file__}")
_self_check_logger.info(f"   VERSION: 2024-12-23-CHART-UPGRADE (with Google Finance-style chart)")
# ============================================================================

# For robust date parsing (handles "Tue, 17 Dec 2024" formats from Google News)
try:
    from dateutil import parser as date_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    date_parser = None
    DATEUTIL_AVAILABLE = False

from src.utils.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# SEC FILING INSIGHTS INTEGRATION
# =============================================================================
try:
    from src.signals.filing_signal import get_filing_insights
    SEC_INSIGHTS_AVAILABLE = True
except ImportError:
    SEC_INSIGHTS_AVAILABLE = False
    logger.warning("SEC Filing Insights not available")

# =============================================================================
# DUAL ANALYST INTEGRATION
# =============================================================================
try:
    from src.ai.dual_analyst import DualAnalystService
    DUAL_ANALYST_AVAILABLE = True
except ImportError:
    DUAL_ANALYST_AVAILABLE = False
    logger.warning("Dual Analyst Service not available")

# =============================================================================
# ARCHITECTURE IMPROVEMENTS: MarketSnapshot, TradeLevels, PolicyResolver
# =============================================================================
from dataclasses import dataclass, field
from typing import Tuple

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
# END ARCHITECTURE IMPROVEMENTS
# =============================================================================

# Phase 2 & 3: Institutional Signals Display
try:
    from src.ml.institutional_signals_display import render_institutional_signals
    INSTITUTIONAL_SIGNALS_AVAILABLE = True
except ImportError:
    INSTITUTIONAL_SIGNALS_AVAILABLE = False

# Institutional Signals Context for AI
try:
    from src.ml.institutional_signals_context import (
        get_institutional_signals_context,
        get_trading_implications,
    )
    INSTITUTIONAL_CONTEXT_AVAILABLE = True
except ImportError:
    INSTITUTIONAL_CONTEXT_AVAILABLE = False
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
try:
    from src.ml.signals_tab_ai import get_ai_probabilities_batch
    AI_PROB_AVAILABLE = True
except ImportError:
    AI_PROB_AVAILABLE = False

# Post-earnings reaction analyzer
try:
    from src.analytics.earnings_intelligence.reaction_analyzer import analyze_post_earnings

    REACTION_ANALYZER_AVAILABLE = True
except ImportError:
    REACTION_ANALYZER_AVAILABLE = False

# AI Trading System import
try:
    from src.ml.ai_trading_system import AITradingSystem
    AI_SYSTEM_AVAILABLE = True
except ImportError:
    AI_SYSTEM_AVAILABLE = False



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

# Import core components
try:
    from src.core import (
        UnifiedSignal, MarketOverview, SignalStrength, RiskLevel,
        get_market_overview, get_signal_engine,
    )

    SIGNAL_HUB_AVAILABLE = True
except ImportError as e:
    SIGNAL_HUB_AVAILABLE = False
    logger.error(f"Signal Hub not available: {e}")


# =============================================================================
# DATABASE JOB TRACKING
# =============================================================================

def _get_job_status() -> dict:
    """Get job status from database."""
    try:
        from src.db.connection import get_engine
        engine = get_engine()
        df = pd.read_sql("SELECT * FROM analysis_job ORDER BY id DESC LIMIT 1", engine)
        if df.empty:
            return {'status': 'idle', 'processed_count': 0, 'total_count': 0}
        row = df.iloc[0]
        return {
            'status': row.get('status', 'idle'),
            'processed_count': int(row.get('processed_count', 0) or 0),
            'total_count': int(row.get('total_count', 0) or 0),
        }
    except:
        return {'status': 'idle', 'processed_count': 0, 'total_count': 0}


def _ensure_job_tables():
    """Create job tables if needed."""
    try:
        from src.db.connection import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            # Create main job tracking table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS analysis_job (
                    id SERIAL PRIMARY KEY,
                    status VARCHAR(20) DEFAULT 'idle',
                    total_count INTEGER DEFAULT 0,
                    processed_count INTEGER DEFAULT 0,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            # Create log table with UNIQUE constraint on ticker
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS analysis_job_log (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) UNIQUE,
                    status VARCHAR(10),
                    news_count INTEGER DEFAULT 0,
                    sentiment_score INTEGER,
                    options_score INTEGER,
                    processed_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            # Initialize job row if needed
            result = conn.execute(text("SELECT COUNT(*) FROM analysis_job"))
            if result.fetchone()[0] == 0:
                conn.execute(text("INSERT INTO analysis_job (status) VALUES ('idle')"))
            conn.commit()
    except Exception as e:
        logger.debug(f"Init job tables error: {e}")


def _start_job(total: int):
    """Start or resume job."""
    try:
        from src.db.connection import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            # Check if we have existing progress
            result = conn.execute(text("SELECT processed_count, total_count, status FROM analysis_job LIMIT 1"))
            row = result.fetchone()

            if row and row[0] > 0 and row[2] in ('stopped', 'idle'):
                # Resume - just change status, keep progress
                conn.execute(text("UPDATE analysis_job SET status='running', updated_at=NOW()"))
            else:
                # Fresh start - DROP and recreate log table to ensure UNIQUE constraint
                conn.execute(text("DROP TABLE IF EXISTS analysis_job_log"))
                conn.execute(text("""
                    CREATE TABLE analysis_job_log (
                        id SERIAL PRIMARY KEY,
                        ticker VARCHAR(10) UNIQUE,
                        status VARCHAR(10),
                        news_count INTEGER DEFAULT 0,
                        sentiment_score INTEGER,
                        options_score INTEGER,
                        processed_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """))
                conn.execute(text(
                    "UPDATE analysis_job SET status='running', total_count=:t, processed_count=0, updated_at=NOW()"),
                             {'t': total})
            conn.commit()
            conn.commit()
    except Exception as e:
        st.error(f"Start error: {e}")


def _stop_job():
    """Stop job."""
    try:
        from src.db.connection import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("UPDATE analysis_job SET status='stopped', updated_at=NOW()"))
            conn.commit()
    except:
        pass


def _complete_job():
    """Mark job complete and clear all caches for fresh data."""
    try:
        from src.db.connection import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("UPDATE analysis_job SET status='completed', updated_at=NOW()"))
            conn.commit()

        # Clear all caches so Load Signals shows fresh data
        keys_to_clear = ['signals_data', 'market_overview']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        # Clear SignalEngine cache entirely
        try:
            from src.core import get_signal_engine
            engine = get_signal_engine()
            if hasattr(engine, '_cache'):
                engine._cache.clear()
                logger.info("SignalEngine cache cleared")
        except:
            pass

        # Set force refresh flag and trigger signals load
        st.session_state.force_refresh = True
        st.session_state.signals_loaded = True
        st.session_state.show_analysis = False  # Switch to signals view
        logger.info("Analysis job completed - all caches cleared, switching to signals view")
    except Exception as e:
        logger.error(f"Complete job error: {e}")


def _reset_job():
    """Reset job - drop and recreate log table."""
    try:
        from src.db.connection import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS analysis_job_log"))
            conn.execute(text("""
                CREATE TABLE analysis_job_log (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) UNIQUE,
                    status VARCHAR(10),
                    news_count INTEGER DEFAULT 0,
                    sentiment_score INTEGER,
                    options_score INTEGER,
                    processed_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            conn.execute(text("UPDATE analysis_job SET status='idle', processed_count=0, total_count=0"))
            conn.commit()
    except Exception as e:
        logger.warning(f"Reset job error: {e}")


def _log_result(ticker: str, status: str, news: int, sentiment, options,
                fundamental=None, technical=None, committee=None):
    """Log or update ticker result."""
    try:
        from src.db.connection import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            # Check if ticker already in log
            result = conn.execute(text("SELECT 1 FROM analysis_job_log WHERE ticker = :t"), {'t': ticker})
            exists = result.fetchone() is not None

            if exists:
                # Update existing entry
                conn.execute(text("""
                    UPDATE analysis_job_log SET
                        status = :s,
                        news_count = :n,
                        sentiment_score = :sent,
                        options_score = :opt,
                        processed_at = NOW()
                    WHERE ticker = :t
                """), {'t': ticker, 's': status, 'n': news, 'sent': _to_native(sentiment), 'opt': _to_native(options)})
            else:
                # Insert new entry (for skipped tickers that weren't pre-logged)
                conn.execute(text("""
                    INSERT INTO analysis_job_log (ticker, status, news_count, sentiment_score, options_score)
                    VALUES (:t, :s, :n, :sent, :opt)
                """), {'t': ticker, 's': status, 'n': news, 'sent': _to_native(sentiment), 'opt': _to_native(options)})
                conn.execute(text("UPDATE analysis_job SET processed_count=processed_count+1, updated_at=NOW()"))

            conn.commit()
        logger.info(f"{ticker}: {status} | News:{news} Sent:{sentiment} Opts:{options} Fund:{fundamental} Tech:{technical} Committee:{committee}")
    except Exception as e:
        logger.warning(f"Log result error for {ticker}: {e}")


def _get_processed_tickers() -> list:
    """Get already processed tickers."""
    try:
        from src.db.connection import get_engine
        engine = get_engine()
        df = pd.read_sql("SELECT ticker FROM analysis_job_log", engine)
        return df['ticker'].tolist() if not df.empty else []
    except:
        return []


def _get_job_log() -> pd.DataFrame:
    """Get job log."""
    try:
        from src.db.connection import get_engine
        engine = get_engine()
        return pd.read_sql("""
                           SELECT ticker,
                                  status,
                                  news_count,
                                  sentiment_score,
                                  options_score,
                                  TO_CHAR(processed_at, 'HH24:MI:SS') as time
                           FROM analysis_job_log
                           ORDER BY processed_at DESC LIMIT 30
                           """, engine)
    except:
        return pd.DataFrame()


def _get_job_stats() -> dict:
    """Get FULL job statistics (not limited to 30 like the log display)."""
    try:
        from src.db.connection import get_engine
        engine = get_engine()
        df = pd.read_sql("""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE status = '✅') as analyzed,
                COUNT(*) FILTER (WHERE status = '⏭️') as skipped,
                COUNT(*) FILTER (WHERE status = '⚠️') as failed,
                COALESCE(SUM(news_count), 0) as news_total
            FROM analysis_job_log
        """, engine)
        if df.empty:
            return {'total': 0, 'analyzed': 0, 'skipped': 0, 'failed': 0, 'news_total': 0}
        row = df.iloc[0]
        return {
            'total': int(row['total'] or 0),
            'analyzed': int(row['analyzed'] or 0),
            'skipped': int(row['skipped'] or 0),
            'failed': int(row['failed'] or 0),
            'news_total': int(row['news_total'] or 0)
        }
    except Exception as e:
        logger.debug(f"Stats query error: {e}")
        return {'total': 0, 'analyzed': 0, 'skipped': 0, 'failed': 0, 'news_total': 0}


# =============================================================================
# MAIN TAB
# =============================================================================

def render_signals_tab():
    """Render the Signals tab with Universe-style table and stock deep dive."""

    st.markdown("### 📊 Signal Hub")

    if not SIGNAL_HUB_AVAILABLE:
        st.error("Signal Hub not available.")
        return

    # Session state initialization
    if 'signals_loaded' not in st.session_state:
        st.session_state.signals_loaded = False
    if 'show_analysis' not in st.session_state:
        st.session_state.show_analysis = False
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = None

    # Check job from DB
    _ensure_job_tables()
    job = _get_job_status()

    # Auto-show analysis if job running
    if job['status'] == 'running':
        st.session_state.show_analysis = True

    # Top action buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("📥 Load Signals", type="primary", width='stretch'):
            # Clear any cached data to force fresh database read
            if 'signals_data' in st.session_state:
                del st.session_state['signals_data']
            st.session_state.signals_loaded = True
            st.session_state.show_analysis = False
            st.session_state.selected_ticker = None
            st.session_state.force_refresh = True  # Force fresh load
            st.rerun()

    with col2:
        if st.button("🚀 Run Analysis", type="secondary", width='stretch'):
            st.session_state.show_analysis = True
            st.rerun()

    with col3:
        # Status indicator
        if job['status'] == 'running':
            st.info(f"🔄 Running: {job['processed_count']}/{job['total_count']}")
        elif job['status'] == 'completed':
            st.success(f"✅ Completed: {job['processed_count']} tickers")

    # =========================================================================
    # QUICK ADD TICKER - Analyze any stock on-demand
    # =========================================================================
    with st.expander("➕ Quick Add & Analyze Ticker", expanded=False):
        _render_quick_add_ticker()

    # =========================================================================
    # REMOVE TICKER - Remove stocks from watchlist
    # =========================================================================
    with st.expander("🗑️ Remove Ticker from Watchlist", expanded=False):
        _render_remove_ticker()

    # Show analysis panel or signals table
    if st.session_state.show_analysis:
        _render_analysis_panel()
    elif st.session_state.signals_loaded:
        _render_signals_table_view()
    else:
        st.info("👆 Click **Load Signals** to view the signals table, or **Run Analysis** to refresh all data")


def _render_quick_add_ticker():
    """
    Render the Quick Add Ticker UI.
    Allows users to analyze any ticker on-demand, even if not in the universe.
    Uses the same pipeline as the full scanner for consistency.

    Features:
    - Add new tickers to watchlist
    - Force refresh existing tickers (re-fetch everything: news, sentiment, technical, etc.)
    """
    import os

    # Initialize session state for this feature
    if 'quick_add_result' not in st.session_state:
        st.session_state.quick_add_result = None
    if 'quick_add_error' not in st.session_state:
        st.session_state.quick_add_error = None

    st.markdown("**Analyze any ticker instantly** - generates full signals using the same pipeline as the scanner.")

    # Input row
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        ticker_input = st.text_input(
            "Ticker Symbol",
            placeholder="e.g., NVAX, CERT, ADPT",
            key="quick_add_ticker_input",
            label_visibility="collapsed"
        ).upper().strip()

    with col2:
        add_permanent = st.checkbox(
            "Save to watchlist",
            value=False,
            help="If checked, adds the ticker permanently to your universe.csv and database"
        )

    with col3:
        force_refresh = st.checkbox(
            "🔄 Force Refresh",
            value=False,
            help="Re-fetch everything: news, sentiment, technical, fundamentals. Use for existing tickers that need updated data."
        )

    with col4:
        analyze_clicked = st.button(
            "🚀 Analyze",
            type="primary",
            use_container_width=True,
            disabled=not ticker_input
        )

    # Process the analysis
    if analyze_clicked and ticker_input:
        st.session_state.quick_add_result = None
        st.session_state.quick_add_error = None

        # Parse multiple tickers (comma, space, or semicolon separated)
        import re
        tickers = [t.strip().upper() for t in re.split(r'[,;\s]+', ticker_input) if t.strip()]

        if not tickers:
            st.session_state.quick_add_error = "No valid tickers entered"
        elif len(tickers) == 1:
            # Single ticker - original behavior
            action_text = "🔄 Force refreshing" if force_refresh else "🔍 Analyzing"
            with st.spinner(f"{action_text} {tickers[0]}..."):
                try:
                    result = _quick_analyze_ticker(tickers[0], add_permanent, force_refresh=force_refresh)
                    if result['success']:
                        st.session_state.quick_add_result = result
                        st.session_state.quick_add_error = None
                        # Force refresh signals table to show new/updated ticker
                        st.session_state.signals_loaded = False
                        st.rerun()
                    else:
                        st.session_state.quick_add_error = result['error']
                        st.session_state.quick_add_result = None
                except Exception as e:
                    st.session_state.quick_add_error = str(e)
                    st.session_state.quick_add_result = None
                    logger.error(f"Quick add error: {e}")
        else:
            # Multiple tickers - process each one
            results = []
            errors = []
            action_text = "Force refreshing" if force_refresh else "Analyzing"
            progress_bar = st.progress(0, text=f"{action_text} {len(tickers)} tickers...")

            for i, ticker in enumerate(tickers):
                progress_bar.progress((i + 1) / len(tickers), text=f"🔍 {action_text} {ticker} ({i+1}/{len(tickers)})...")
                try:
                    result = _quick_analyze_ticker(ticker, add_permanent, force_refresh=force_refresh)
                    if result['success']:
                        results.append(result)
                    else:
                        errors.append(f"{ticker}: {result['error']}")
                except Exception as e:
                    errors.append(f"{ticker}: {str(e)}")
                    logger.error(f"Quick add error for {ticker}: {e}")

            progress_bar.empty()

            # Store results
            if results:
                # Store the last successful result for display, but mark as multi
                st.session_state.quick_add_result = {
                    'success': True,
                    'multi': True,
                    'count': len(results),
                    'tickers': [r['ticker'] for r in results],
                    'results': results,
                    'force_refreshed': force_refresh
                }
                st.session_state.signals_loaded = False

            if errors:
                st.session_state.quick_add_error = f"Failed: {'; '.join(errors)}"
            else:
                st.session_state.quick_add_error = None

            if results:
                st.rerun()


    # Show error if any
    if st.session_state.quick_add_error:
        st.error(f"❌ {st.session_state.quick_add_error}")

    # Show result if available
    if st.session_state.quick_add_result:
        result = st.session_state.quick_add_result

        # Handle multi-ticker results
        if result.get('multi'):
            action = "🔄 Force refreshed" if result.get('force_refreshed') else "✅ Analyzed"
            st.success(f"{action} {result['count']} tickers: {', '.join(result['tickers'])}")

            # Show summary for each ticker
            for single_result in result.get('results', []):
                ticker = single_result['ticker']
                signal = single_result.get('signal')

                with st.expander(f"📊 {ticker}", expanded=True):
                    if signal:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            score = signal.today_score if signal.today_score is not None else 50
                            try:
                                score = float(score)
                                color = "🟢" if score >= 65 else "🔴" if score <= 35 else "🟡"
                                signal_val = signal.today_signal.value if signal.today_signal else "HOLD"
                                st.metric("Signal", f"{color} {signal_val}", f"{score:.0f}%")
                            except (TypeError, ValueError):
                                st.metric("Signal", "N/A")
                        with col2:
                            try:
                                if signal.current_price:
                                    st.metric("Price", f"${float(signal.current_price):.2f}")
                                else:
                                    st.metric("Price", "N/A")
                            except (TypeError, ValueError):
                                st.metric("Price", "N/A")
                        with col3:
                            st.metric("Sector", signal.sector or "Unknown")
                        with col4:
                            try:
                                if signal.technical_score is not None:
                                    st.metric("Technical", f"{float(signal.technical_score):.0f}")
                                else:
                                    st.metric("Technical", "N/A")
                            except (TypeError, ValueError):
                                st.metric("Technical", "N/A")
                    else:
                        st.info("Signal data not available")
        else:
            # Single ticker result - original behavior
            ticker = result['ticker']

            # Build status message
            status_parts = []
            if result.get('force_refreshed'):
                status_parts.append("🔄 Force refreshed")
            if result.get('added_to_watchlist'):
                status_parts.append("Added to watchlist")
            if result.get('saved_to_db'):
                status_parts.append("Saved to database")
            else:
                status_parts.append("⚠️ Not saved to DB - may not appear in signals table")

            status_msg = " | ".join(status_parts) if status_parts else "Temporary"
            st.success(f"✅ {ticker} analyzed successfully! ({status_msg})")

            # Show signal summary
            if result.get('signal'):
                signal = result['signal']

                # Signal summary cards
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    score = signal.today_score if signal.today_score is not None else 50
                    try:
                        score = float(score)
                        color = "🟢" if score >= 65 else "🔴" if score <= 35 else "🟡"
                        signal_val = signal.today_signal.value if signal.today_signal else "HOLD"
                        st.metric("Signal", f"{color} {signal_val}", f"{score:.0f}%")
                    except (TypeError, ValueError):
                        st.metric("Signal", "N/A")

                with col2:
                    try:
                        if signal.current_price:
                            st.metric("Price", f"${float(signal.current_price):.2f}")
                        else:
                            st.metric("Price", "N/A")
                    except (TypeError, ValueError):
                        st.metric("Price", "N/A")

                with col3:
                    st.metric("Sector", signal.sector or "Unknown")

                with col4:
                    try:
                        if signal.technical_score is not None:
                            st.metric("Technical", f"{float(signal.technical_score):.0f}")
                        else:
                            st.metric("Technical", "N/A")
                    except (TypeError, ValueError):
                        st.metric("Technical", "N/A")

            # Show scores breakdown
            if result.get('scores'):
                scores = result['scores']
                st.markdown("**Score Breakdown:**")
                score_cols = st.columns(5)
                score_names = ['Technical', 'Fundamental', 'Sentiment', 'Options', 'Total']
                score_keys = ['technical_score', 'fundamental_score', 'sentiment_score', 'options_flow_score', 'total_score']

                for i, (name, key) in enumerate(zip(score_names, score_keys)):
                    with score_cols[i]:
                        val = scores.get(key)
                        if val is None:
                            val = 50  # Default to neutral
                        try:
                            val = float(val)
                            color = "🟢" if val >= 65 else "🔴" if val <= 35 else "🟡"
                            st.metric(name, f"{color} {val:.0f}")
                        except (TypeError, ValueError):
                            st.metric(name, "N/A")

            # Show AI Analysis Results (if available)
            if result.get('ai_result'):
                ai_res = result['ai_result']
                st.markdown("---")
                st.markdown("### 🤖 AI Analysis")

                # AI Decision row
                ai_cols = st.columns(4)
                with ai_cols[0]:
                    action_color = "🟢" if ai_res.ai_action == "BUY" else "🔴" if ai_res.ai_action == "SELL" else "🟡"
                    st.metric("AI Action", f"{action_color} {ai_res.ai_action}")
                with ai_cols[1]:
                    conf_color = "🟢" if ai_res.ai_confidence == "HIGH" else "🟡" if ai_res.ai_confidence == "MEDIUM" else "🔴"
                    st.metric("Confidence", f"{conf_color} {ai_res.ai_confidence}")
                with ai_cols[2]:
                    trade_icon = "✅" if ai_res.trade_allowed else "❌"
                    st.metric("Trade Allowed", f"{trade_icon} {'Yes' if ai_res.trade_allowed else 'No'}")
                with ai_cols[3]:
                    if ai_res.target_price and ai_res.entry_price:
                        upside = ((ai_res.target_price - ai_res.entry_price) / ai_res.entry_price) * 100
                        st.metric("Upside", f"{upside:.1f}%")
                    else:
                        st.metric("Upside", "N/A")

                # Entry/Exit plan
                plan_cols = st.columns(4)
                with plan_cols[0]:
                    st.metric("Entry", f"${ai_res.entry_price:.2f}" if ai_res.entry_price else "N/A")
                with plan_cols[1]:
                    st.metric("Stop Loss", f"${ai_res.stop_loss:.2f}" if ai_res.stop_loss else "N/A")
                with plan_cols[2]:
                    st.metric("Target", f"${ai_res.target_price:.2f}" if ai_res.target_price else "N/A")
                with plan_cols[3]:
                    st.metric("Position", ai_res.position_size or "N/A")

                # Bull/Bear case
                case_cols = st.columns(2)
                with case_cols[0]:
                    st.markdown("**🐂 Bull Case:**")
                    for point in ai_res.bull_case[:3]:
                        st.markdown(f"• {point}")
                with case_cols[1]:
                    st.markdown("**🐻 Bear Case:**")
                    for point in ai_res.bear_case[:3]:
                        st.markdown(f"• {point}")

                # Key risks
                if ai_res.key_risks:
                    st.markdown("**⚠️ Key Risks:**")
                    for risk in ai_res.key_risks[:3]:
                        st.markdown(f"• {risk}")

                # Blocking factors
                if ai_res.blocking_factors:
                    st.warning(f"**Blocking Factors:** {', '.join(ai_res.blocking_factors)}")

            # Buttons row
            col_btn1, col_btn2, col_btn3 = st.columns(3)

            with col_btn1:
                # Button to view in signals table
                if st.button(f"📋 View in Signals Table", key=f"view_signals_{ticker}", use_container_width=True):
                    st.session_state.selected_ticker = ticker
                    st.session_state.signals_loaded = True
                    st.session_state.force_refresh = True
                    if 'signals_data' in st.session_state:
                        del st.session_state['signals_data']
                    st.rerun()

            with col_btn2:
                # Direct deep dive toggle
                deep_dive_key = f"show_deep_dive_{ticker}"
                if deep_dive_key not in st.session_state:
                    st.session_state[deep_dive_key] = False

                if st.button(f"🔍 Deep Dive", key=f"deep_dive_btn_{ticker}", use_container_width=True):
                    st.session_state[deep_dive_key] = not st.session_state[deep_dive_key]
                    st.rerun()

            with col_btn3:
                # Clear button
                if st.button("🗑️ Clear", key=f"clear_result_{ticker}", use_container_width=True):
                    st.session_state.quick_add_result = None
                    st.session_state.selected_ticker = None
                    # Clear deep dive state
                    if f"show_deep_dive_{ticker}" in st.session_state:
                        del st.session_state[f"show_deep_dive_{ticker}"]
                    st.rerun()

            # Show deep dive inline if toggled
            if st.session_state.get(f"show_deep_dive_{ticker}", False):
                st.markdown("---")
                st.markdown(f"### 🔍 Deep Dive: {ticker}")

                # Use the signal from the result if available, otherwise fetch fresh
                if result.get('signal'):
                    _render_deep_dive(result['signal'])
                else:
                    # Fetch fresh signal for deep dive
                    try:
                        from src.core import get_signal_engine
                        engine = get_signal_engine()
                        signal = engine.generate_signal(ticker, force_refresh=True)
                        if signal:
                            _render_deep_dive(signal)
                        else:
                            st.warning(f"Could not generate signal for {ticker}")
                    except Exception as e:
                        st.error(f"Error loading deep dive: {e}")


def _quick_analyze_ticker(ticker: str, add_permanent: bool = False, force_refresh: bool = False) -> dict:
    """
    Analyze a ticker using the FULL signal pipeline - same as "Run Analysis".

    This runs:
    1. Fresh news collection
    2. Sentiment analysis
    3. Options flow & squeeze scores
    4. Technical analysis
    5. Fundamental data fetch (yfinance + Finviz + IBKR)
    6. Create/Update screener_scores with ALL scores
    7. Generate signal with committee
    8. Run AI Analysis (BatchAIAnalyzer) for deep insights

    Args:
        ticker: Stock ticker symbol
        add_permanent: If True, add to universe.csv
        force_refresh: If True, skip cache and re-fetch ALL data (news, sentiment, technical, etc.)

    Returns:
        dict with 'success', 'ticker', 'signal', 'scores', 'error', 'added_to_watchlist',
        'saved_to_db', 'force_refreshed', 'ai_result'
        dict with 'success', 'ticker', 'signal', 'scores', 'error', 'added_to_watchlist', 'saved_to_db', 'force_refreshed'
    """
    import os
    import yfinance as yf
    from datetime import date as dt_date

    ticker = ticker.upper().strip()

    if force_refresh:
        logger.info(f"{ticker}: FORCE REFRESH mode - re-fetching all data")

    # Validate ticker format
    if not ticker or len(ticker) > 10 or not ticker.replace('.', '').replace('-', '').isalnum():
        return {'success': False, 'error': f"Invalid ticker format: {ticker}"}

    # Quick validation - check if ticker exists via yfinance
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        if not info or info.get('regularMarketPrice') is None:
            try:
                fast = yf_ticker.fast_info
                if not hasattr(fast, 'last_price') or fast.last_price is None:
                    return {'success': False, 'error': f"Ticker '{ticker}' not found or has no price data"}
            except:
                return {'success': False, 'error': f"Ticker '{ticker}' not found or has no price data"}
    except Exception as e:
        return {'success': False, 'error': f"Could not validate ticker '{ticker}': {str(e)}"}

    # =========================================================================
    # FETCH DATA FROM MULTIPLE SOURCES (yfinance already fetched above)
    # =========================================================================

    # Source 1: yfinance data (already in 'info' dict)
    logger.info(f"{ticker}: Fetching from multiple data sources...")

    # Source 2: Finviz data
    finviz_data = {}
    try:
        from src.data.finviz import FinvizDataFetcher
        finviz_fetcher = FinvizDataFetcher()
        finviz_data = finviz_fetcher.get_fundamentals(ticker) or {}
        if finviz_data:
            logger.info(f"{ticker}: ✓ Finviz data fetched (Inst Own: {finviz_data.get('inst_own')}%)")

        # Also get analyst ratings from Finviz
        finviz_ratings = finviz_fetcher.get_analyst_ratings(ticker) or {}
        if finviz_ratings:
            finviz_data['finviz_buy_pct'] = finviz_ratings.get('buy_pct')
            finviz_data['finviz_total_ratings'] = finviz_ratings.get('total_ratings')
            logger.info(f"{ticker}: ✓ Finviz ratings: {finviz_ratings.get('total_positive')}/{finviz_ratings.get('total_ratings')} positive")
    except ImportError:
        logger.debug(f"{ticker}: Finviz module not available")
    except Exception as e:
        logger.debug(f"{ticker}: Finviz fetch failed: {e}")

    # Source 3: IBKR data (if available)
    ibkr_data = {}
    try:
        from src.data.ibkr_client import IBKRClient
        ibkr = IBKRClient()
        if ibkr.is_connected():
            ibkr_data = ibkr.get_fundamental_data(ticker) or {}
            if ibkr_data:
                logger.info(f"{ticker}: ✓ IBKR data fetched")
    except ImportError:
        logger.debug(f"{ticker}: IBKR module not available")
    except Exception as e:
        logger.debug(f"{ticker}: IBKR fetch failed: {e}")

    # =========================================================================
    # MERGE DATA FROM ALL SOURCES (prefer: IBKR > Finviz > yfinance)
    # =========================================================================
    def get_best_value(*values):
        """Return first non-None value from multiple sources."""
        for v in values:
            if v is not None and v != '' and not (isinstance(v, float) and pd.isna(v)):
                return v
        return None

    # Check if already in universe
    already_exists = _ticker_in_universe(ticker)

    # Add to universe.csv if requested and not already there
    if add_permanent and not already_exists:
        try:
            _add_ticker_to_universe(ticker, info)
            logger.info(f"Added {ticker} to universe.csv")
        except Exception as e:
            logger.warning(f"Could not add {ticker} to universe.csv: {e}")

    # =========================================================================
    # RUN FULL ANALYSIS PIPELINE (same as "Run Analysis" button)
    # =========================================================================
    saved_to_db = False
    scores = {}
    today = dt_date.today()

    try:
        from src.db.connection import get_connection

        # -----------------------------------------------------------------
        # STEP 1: Collect Fresh News
        # -----------------------------------------------------------------
        articles = []
        try:
            from src.data.news import NewsCollector
            nc = NewsCollector()
            result = nc.collect_and_save(ticker, days_back=7, force_refresh=True)
            articles = result.get('articles', [])
            logger.info(f"{ticker}: Collected {len(articles)} news articles")
        except Exception as e:
            logger.warning(f"{ticker}: News collection failed: {e}")

        # -----------------------------------------------------------------
        # STEP 2: Analyze Sentiment
        # -----------------------------------------------------------------
        sentiment_score = None
        sentiment_data = {}
        if articles:
            try:
                from src.screener.sentiment import SentimentAnalyzer
                sa = SentimentAnalyzer()
                sentiment_data = sa.analyze_ticker_sentiment(ticker, articles)
                sentiment_score = sentiment_data.get('sentiment_score')
                logger.info(f"{ticker}: Sentiment score = {sentiment_score}")
            except Exception as e:
                logger.warning(f"{ticker}: Sentiment analysis failed: {e}")

        # -----------------------------------------------------------------
        # STEP 3: Get Options Flow & Squeeze Scores
        # -----------------------------------------------------------------
        options_score = None
        squeeze_score = None
        try:
            from src.analytics.universe_scorer import UniverseScorer
            scorer = UniverseScorer(skip_ibkr=True)
            scores_list, _ = scorer.score_and_save_universe(tickers=[ticker], max_workers=1)
            if scores_list:
                for score_obj in scores_list:
                    if score_obj.ticker == ticker:
                        options_score = score_obj.options_flow_score
                        squeeze_score = score_obj.short_squeeze_score
                        logger.info(f"{ticker}: Options={options_score}, Squeeze={squeeze_score}")
                        break
        except Exception as e:
            logger.warning(f"{ticker}: UniverseScorer failed: {e}")

        # -----------------------------------------------------------------
        # STEP 4: Get Technical Score
        # -----------------------------------------------------------------
        technical_score = None
        try:
            from src.analytics.technical_analysis import TechnicalAnalyzer
            ta = TechnicalAnalyzer()
            tech_result = ta.analyze_ticker(ticker)
            if tech_result:
                technical_score = tech_result.get('technical_score', tech_result.get('score'))
                logger.info(f"{ticker}: Technical score = {technical_score}")
        except Exception as e:
            logger.debug(f"{ticker}: Technical analysis skipped: {e}")

        # -----------------------------------------------------------------
        # STEP 5: Get Fundamental Data (MERGED FROM ALL SOURCES)
        # -----------------------------------------------------------------
        fundamental_score = None
        growth_score = None
        dividend_score = None
        gap_score = 50  # Neutral default

        try:
            # Merge fundamental data from all sources (prefer: IBKR > Finviz > yfinance)
            # Valuation metrics
            pe_ratio = get_best_value(
                ibkr_data.get('pe_ratio'),
                finviz_data.get('pe'),
                info.get('trailingPE')
            )
            forward_pe = get_best_value(
                ibkr_data.get('forward_pe'),
                finviz_data.get('forward_pe'),
                info.get('forwardPE')
            )
            peg_ratio = get_best_value(
                ibkr_data.get('peg_ratio'),
                finviz_data.get('peg'),
                info.get('pegRatio')
            )

            # Profitability metrics
            roe = get_best_value(
                ibkr_data.get('roe'),
                finviz_data.get('roe'),
                info.get('returnOnEquity')
            )
            # Convert Finviz ROE from percentage to decimal if needed
            if roe and finviz_data.get('roe') and roe > 1:
                roe = roe / 100

            roa = get_best_value(
                ibkr_data.get('roa'),
                finviz_data.get('roa'),
                info.get('returnOnAssets')
            )
            if roa and finviz_data.get('roa') and roa > 1:
                roa = roa / 100

            profit_margin = get_best_value(
                ibkr_data.get('profit_margin'),
                finviz_data.get('profit_margin'),
                info.get('profitMargins')
            )
            if profit_margin and finviz_data.get('profit_margin') and profit_margin > 1:
                profit_margin = profit_margin / 100

            # Growth metrics
            revenue_growth = get_best_value(
                ibkr_data.get('revenue_growth'),
                finviz_data.get('sales_growth_yoy'),
                info.get('revenueGrowth')
            )
            if revenue_growth and finviz_data.get('sales_growth_yoy') and abs(revenue_growth) > 1:
                revenue_growth = revenue_growth / 100

            earnings_growth = get_best_value(
                ibkr_data.get('earnings_growth'),
                finviz_data.get('eps_growth_yoy'),
                info.get('earningsGrowth')
            )
            if earnings_growth and finviz_data.get('eps_growth_yoy') and abs(earnings_growth) > 1:
                earnings_growth = earnings_growth / 100

            # Dividend
            dividend_yield = get_best_value(
                ibkr_data.get('dividend_yield'),
                finviz_data.get('dividend_yield'),
                info.get('dividendYield')
            )
            if dividend_yield and finviz_data.get('dividend_yield') and dividend_yield > 1:
                dividend_yield = dividend_yield / 100

            # Financial health
            debt_equity = get_best_value(
                ibkr_data.get('debt_equity'),
                finviz_data.get('debt_eq'),
                info.get('debtToEquity')
            )
            current_ratio = get_best_value(
                ibkr_data.get('current_ratio'),
                finviz_data.get('current_ratio'),
                info.get('currentRatio')
            )

            # Company info
            sector = get_best_value(
                info.get('sector'),
                'Unknown'
            )
            industry = get_best_value(
                info.get('industry'),
                'Unknown'
            )

            # Finviz-specific data
            inst_own = finviz_data.get('inst_own')
            insider_own = finviz_data.get('insider_own')
            short_float = finviz_data.get('short_float')
            target_price_finviz = finviz_data.get('target_price')
            beta = get_best_value(finviz_data.get('beta'), info.get('beta'))
            rsi = finviz_data.get('rsi')

            # Log data sources used
            sources_used = []
            if any(ibkr_data.values()):
                sources_used.append("IBKR")
            if any(finviz_data.values()):
                sources_used.append("Finviz")
            sources_used.append("yfinance")
            logger.info(f"{ticker}: Data sources: {', '.join(sources_used)}")

            # Enhanced fundamental score calculation using all sources
            fund_points = 50  # Start neutral

            # PE analysis
            if pe_ratio and pe_ratio > 0:
                if pe_ratio < 15:
                    fund_points += 10
                elif pe_ratio < 25:
                    fund_points += 5
                elif pe_ratio > 50:
                    fund_points -= 10

            # PEG analysis
            if peg_ratio and peg_ratio > 0:
                if peg_ratio < 1:
                    fund_points += 10
                elif peg_ratio < 2:
                    fund_points += 5
                elif peg_ratio > 3:
                    fund_points -= 5

            # Profitability analysis
            if roe and roe > 0.15:
                fund_points += 5
            if roe and roe > 0.25:
                fund_points += 5
            if profit_margin and profit_margin > 0.1:
                fund_points += 5

            # Financial health
            if current_ratio and current_ratio > 1.5:
                fund_points += 5
            if debt_equity and debt_equity < 0.5:
                fund_points += 5
            elif debt_equity and debt_equity > 2:
                fund_points -= 5

            # Institutional ownership (from Finviz)
            if inst_own is not None:
                if inst_own >= 70:
                    fund_points += 5
                elif inst_own < 20:
                    fund_points -= 5

            # Short interest (from Finviz) - bearish indicator
            if short_float is not None:
                if short_float >= 20:
                    fund_points -= 10
                elif short_float >= 10:
                    fund_points -= 5

            fundamental_score = max(0, min(100, fund_points))

            # Growth score (enhanced)
            growth_points = 50
            if revenue_growth:
                if revenue_growth > 0.2:
                    growth_points += 20
                elif revenue_growth > 0.1:
                    growth_points += 15
                elif revenue_growth > 0:
                    growth_points += 5
                elif revenue_growth < -0.1:
                    growth_points -= 15
            if earnings_growth:
                if earnings_growth > 0.2:
                    growth_points += 20
                elif earnings_growth > 0.1:
                    growth_points += 15
                elif earnings_growth > 0:
                    growth_points += 5
                elif earnings_growth < -0.1:
                    growth_points -= 15
            growth_score = max(0, min(100, growth_points))

            # Dividend score
            dividend_points = 50
            if dividend_yield:
                if dividend_yield > 0.04:
                    dividend_points += 30
                elif dividend_yield > 0.02:
                    dividend_points += 20
                elif dividend_yield > 0.01:
                    dividend_points += 10
            dividend_score = max(0, min(100, dividend_points))

            logger.info(f"{ticker}: Fundamental={fundamental_score}, Growth={growth_score}, Dividend={dividend_score}")
            if inst_own is not None:
                logger.info(f"{ticker}: Finviz - Inst Own: {inst_own}%, Short Float: {short_float}%")

            # Save fundamentals to database (matching actual schema)
            with get_connection() as conn:
                with conn.cursor() as cur:
                    try:
                        # Check if record exists for today
                        cur.execute("SELECT 1 FROM fundamentals WHERE ticker = %s AND date = %s", (ticker, today))
                        exists = cur.fetchone() is not None

                        if exists:
                            cur.execute("""
                                UPDATE fundamentals SET
                                    pe_ratio = COALESCE(%s, pe_ratio),
                                    forward_pe = COALESCE(%s, forward_pe),
                                    peg_ratio = COALESCE(%s, peg_ratio),
                                    dividend_yield = COALESCE(%s, dividend_yield),
                                    revenue_growth = COALESCE(%s, revenue_growth),
                                    earnings_growth = COALESCE(%s, earnings_growth),
                                    roe = COALESCE(%s, roe),
                                    debt_to_equity = COALESCE(%s, debt_to_equity),
                                    current_ratio = COALESCE(%s, current_ratio),
                                    sector = COALESCE(%s, sector)
                                WHERE ticker = %s AND date = %s
                            """, (
                                _to_native(pe_ratio), _to_native(forward_pe), _to_native(peg_ratio),
                                _to_native(dividend_yield), _to_native(revenue_growth), _to_native(earnings_growth),
                                _to_native(roe), _to_native(debt_equity), _to_native(current_ratio),
                                sector,
                                ticker, today
                            ))
                        else:
                            cur.execute("""
                                INSERT INTO fundamentals (ticker, date, pe_ratio, forward_pe, peg_ratio,
                                                          dividend_yield, revenue_growth, earnings_growth, roe,
                                                          debt_to_equity, current_ratio, sector)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                ticker, today,
                                _to_native(pe_ratio), _to_native(forward_pe), _to_native(peg_ratio),
                                _to_native(dividend_yield), _to_native(revenue_growth), _to_native(earnings_growth),
                                _to_native(roe), _to_native(debt_equity), _to_native(current_ratio),
                                sector
                            ))
                        conn.commit()
                        logger.info(f"{ticker}: Saved fundamentals with sector='{sector}'")
                    except Exception as db_err:
                        logger.debug(f"{ticker}: Fundamentals save failed: {db_err}")
                        conn.rollback()

        except Exception as e:
            logger.debug(f"{ticker}: Fundamental data fetch failed: {e}")

        # -----------------------------------------------------------------
        # STEP 5b: Save Price to prices table
        # -----------------------------------------------------------------
        current_price = None
        try:
            current_price = info.get('regularMarketPrice') or info.get('currentPrice')
            if current_price:
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO prices (ticker, date, close, open, high, low, volume)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (ticker, date) DO UPDATE SET
                                close = EXCLUDED.close,
                                open = EXCLUDED.open,
                                high = EXCLUDED.high,
                                low = EXCLUDED.low,
                                volume = EXCLUDED.volume
                        """, (
                            ticker, today,
                            _to_native(current_price),
                            _to_native(info.get('regularMarketOpen')),
                            _to_native(info.get('regularMarketDayHigh')),
                            _to_native(info.get('regularMarketDayLow')),
                            _to_native(info.get('regularMarketVolume'))
                        ))
                    conn.commit()
                logger.info(f"{ticker}: Saved price ${current_price:.2f} to prices table")
        except Exception as e:
            logger.debug(f"{ticker}: Price save failed: {e}")

        # -----------------------------------------------------------------
        # STEP 5c: Save Analyst Ratings (merged from yfinance + Finviz)
        # -----------------------------------------------------------------
        try:
            # Get analyst recommendations from yfinance
            recommendations = info.get('recommendationKey', '')
            target_mean = info.get('targetMeanPrice')
            target_high = info.get('targetHighPrice')
            target_low = info.get('targetLowPrice')
            num_analysts = info.get('numberOfAnalystOpinions', 0)

            buy_count = 0
            hold_count = 0
            sell_count = 0

            # Method 1: Try recommendationTrend from yfinance info
            rec_summary = info.get('recommendationTrend', {}).get('trend', [])
            if rec_summary and len(rec_summary) > 0:
                latest = rec_summary[0]
                buy_count = (latest.get('strongBuy', 0) or 0) + (latest.get('buy', 0) or 0)
                hold_count = latest.get('hold', 0) or 0
                sell_count = (latest.get('sell', 0) or 0) + (latest.get('strongSell', 0) or 0)
                logger.info(f"{ticker}: Got analyst data from yfinance recommendationTrend")

            # Method 2: If no data, try yfinance recommendations DataFrame
            if buy_count == 0 and hold_count == 0 and sell_count == 0:
                try:
                    rec_df = yf_ticker.recommendations
                    if rec_df is not None and not rec_df.empty:
                        latest = rec_df.iloc[-1]
                        buy_count = (latest.get('strongBuy', 0) or 0) + (latest.get('buy', 0) or 0)
                        hold_count = latest.get('hold', 0) or 0
                        sell_count = (latest.get('sell', 0) or 0) + (latest.get('strongSell', 0) or 0)
                        logger.info(f"{ticker}: Got analyst data from yfinance recommendations DataFrame")
                except Exception as e:
                    logger.debug(f"{ticker}: Could not get yfinance recommendations DataFrame: {e}")

            # Method 3: Use Finviz data if still no yfinance data
            total_ratings = buy_count + hold_count + sell_count
            if total_ratings == 0 and finviz_data:
                finviz_total = finviz_data.get('finviz_total_ratings', 0) or 0
                finviz_buy_pct = finviz_data.get('finviz_buy_pct', 0) or 0
                if finviz_total > 0:
                    # Estimate buy/sell counts from Finviz buy percentage
                    buy_count = int(finviz_total * finviz_buy_pct / 100)
                    sell_count = int(finviz_total * 0.1)  # Estimate 10% sell
                    hold_count = finviz_total - buy_count - sell_count
                    total_ratings = finviz_total
                    logger.info(f"{ticker}: Using Finviz analyst data - {finviz_buy_pct}% positive ({finviz_total} ratings)")

            if total_ratings == 0:
                total_ratings = num_analysts or 0

            # Calculate analyst positivity (% buy)
            analyst_positivity = (buy_count / total_ratings * 100) if total_ratings > 0 else None

            # Merge with Finviz target price if yfinance is missing
            if not target_mean and finviz_data.get('target_price'):
                target_mean = finviz_data.get('target_price')
                logger.info(f"{ticker}: Using Finviz target price: ${target_mean}")

            if total_ratings > 0 or target_mean:
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        # Check if record exists for today
                        cur.execute("SELECT 1 FROM analyst_ratings WHERE ticker = %s AND date = %s", (ticker, today))
                        exists = cur.fetchone() is not None

                        if exists:
                            cur.execute("""
                                UPDATE analyst_ratings SET
                                    analyst_buy = COALESCE(%s, analyst_buy),
                                    analyst_hold = COALESCE(%s, analyst_hold),
                                    analyst_sell = COALESCE(%s, analyst_sell),
                                    analyst_total = COALESCE(%s, analyst_total),
                                    analyst_positivity = COALESCE(%s, analyst_positivity)
                                WHERE ticker = %s AND date = %s
                            """, (
                                _to_native(buy_count) if buy_count > 0 else None,
                                _to_native(hold_count) if hold_count > 0 else None,
                                _to_native(sell_count) if sell_count > 0 else None,
                                _to_native(total_ratings) if total_ratings > 0 else None,
                                _to_native(analyst_positivity),
                                ticker, today
                            ))
                        else:
                            cur.execute("""
                                INSERT INTO analyst_ratings (ticker, date, analyst_buy, analyst_hold, analyst_sell,
                                                             analyst_total, analyst_positivity)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, (
                                ticker, today,
                                _to_native(buy_count) if buy_count > 0 else None,
                                _to_native(hold_count) if hold_count > 0 else None,
                                _to_native(sell_count) if sell_count > 0 else None,
                                _to_native(total_ratings) if total_ratings > 0 else None,
                                _to_native(analyst_positivity)
                            ))
                    conn.commit()
                logger.info(f"{ticker}: Saved analyst ratings - Buy:{buy_count} Hold:{hold_count} Sell:{sell_count} Total:{total_ratings}" + (f" Positivity:{analyst_positivity:.1f}%" if analyst_positivity else ""))
        except Exception as e:
            logger.debug(f"{ticker}: Analyst ratings save failed: {e}")

        # -----------------------------------------------------------------
        # STEP 5d: Save Price Targets (yfinance + Finviz fallback)
        # -----------------------------------------------------------------
        try:
            # Get from yfinance first
            target_mean = info.get('targetMeanPrice')
            target_high = info.get('targetHighPrice')
            target_low = info.get('targetLowPrice')
            target_median = info.get('targetMedianPrice')

            # Use Finviz target as fallback
            if not target_mean and finviz_data.get('target_price'):
                target_mean = finviz_data.get('target_price')
                logger.info(f"{ticker}: Using Finviz target price: ${target_mean}")

            if target_mean:
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        # Check if record exists for today
                        cur.execute("SELECT 1 FROM price_targets WHERE ticker = %s AND date = %s", (ticker, today))
                        exists = cur.fetchone() is not None

                        if exists:
                            cur.execute("""
                                UPDATE price_targets SET
                                    target_mean = COALESCE(%s, target_mean),
                                    target_high = COALESCE(%s, target_high),
                                    target_low = COALESCE(%s, target_low)
                                WHERE ticker = %s AND date = %s
                            """, (
                                _to_native(target_mean),
                                _to_native(target_high),
                                _to_native(target_low),
                                ticker, today
                            ))
                        else:
                            cur.execute("""
                                INSERT INTO price_targets (ticker, date, target_mean, target_high, target_low)
                                VALUES (%s, %s, %s, %s, %s)
                            """, (
                                ticker, today,
                                _to_native(target_mean),
                                _to_native(target_high),
                                _to_native(target_low)
                            ))
                    conn.commit()
                logger.info(f"{ticker}: Saved price target ${target_mean:.2f}")
        except Exception as e:
            logger.debug(f"{ticker}: Price target save failed: {e}")

        # -----------------------------------------------------------------
        # STEP 5e: Save Earnings Calendar
        # -----------------------------------------------------------------
        try:
            earnings_dates = yf_ticker.earnings_dates
            if earnings_dates is not None and not earnings_dates.empty:
                # Get next upcoming earnings date
                for idx in earnings_dates.index:
                    earnings_dt = idx.date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
                    if earnings_dt >= today:
                        with get_connection() as conn:
                            with conn.cursor() as cur:
                                try:
                                    cur.execute("""
                                        INSERT INTO earnings_calendar (ticker, earnings_date)
                                        VALUES (%s, %s)
                                        ON CONFLICT (ticker) DO UPDATE SET
                                            earnings_date = EXCLUDED.earnings_date
                                    """, (ticker, earnings_dt))
                                except:
                                    pass
                            conn.commit()
                        logger.info(f"{ticker}: Saved earnings date {earnings_dt}")
                        break
        except Exception as e:
            logger.debug(f"{ticker}: Earnings calendar save failed: {e}")

        # -----------------------------------------------------------------
        # STEP 5f: Update fundamentals with ex_dividend_date
        # -----------------------------------------------------------------
        try:
            ex_div_date = info.get('exDividendDate')
            if ex_div_date:
                # Convert timestamp to date
                if isinstance(ex_div_date, (int, float)):
                    from datetime import datetime
                    ex_div_date = datetime.fromtimestamp(ex_div_date).date()

                with get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            UPDATE fundamentals SET ex_dividend_date = %s WHERE ticker = %s
                        """, (ex_div_date, ticker))
                    conn.commit()
                logger.info(f"{ticker}: Saved ex-dividend date {ex_div_date}")
        except Exception as e:
            logger.debug(f"{ticker}: Ex-dividend date save failed: {e}")

        # -----------------------------------------------------------------
        # STEP 6: Calculate Total Score
        # -----------------------------------------------------------------
        available_scores = []
        available_weights = []

        if sentiment_score is not None:
            available_scores.append(sentiment_score * 0.25)
            available_weights.append(0.25)
        if fundamental_score is not None:
            available_scores.append(fundamental_score * 0.25)
            available_weights.append(0.25)
        if technical_score is not None:
            available_scores.append(technical_score * 0.25)
            available_weights.append(0.25)
        if options_score is not None:
            available_scores.append(options_score * 0.15)
            available_weights.append(0.15)
        if squeeze_score is not None:
            available_scores.append(squeeze_score * 0.10)
            available_weights.append(0.10)

        if available_weights:
            total_weight = sum(available_weights)
            total_score = round(sum(available_scores) / total_weight) if total_weight > 0 else 50
            total_score = max(0, min(100, total_score))
        else:
            total_score = 50

        logger.info(f"{ticker}: Total score = {total_score}")

        # -----------------------------------------------------------------
        # STEP 7: Save to screener_scores (UPSERT)
        # -----------------------------------------------------------------
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO screener_scores (
                        ticker, date, 
                        sentiment_score, sentiment_weighted, article_count,
                        fundamental_score, technical_score, growth_score, dividend_score,
                        gap_score, total_score,
                        options_flow_score, short_squeeze_score,
                        created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                    )
                    ON CONFLICT (ticker, date) DO UPDATE SET
                        sentiment_score = EXCLUDED.sentiment_score,
                        sentiment_weighted = EXCLUDED.sentiment_weighted,
                        article_count = EXCLUDED.article_count,
                        fundamental_score = EXCLUDED.fundamental_score,
                        technical_score = EXCLUDED.technical_score,
                        growth_score = EXCLUDED.growth_score,
                        dividend_score = EXCLUDED.dividend_score,
                        gap_score = EXCLUDED.gap_score,
                        total_score = EXCLUDED.total_score,
                        options_flow_score = EXCLUDED.options_flow_score,
                        short_squeeze_score = EXCLUDED.short_squeeze_score
                """, (
                    ticker, today,
                    _to_native(sentiment_score),
                    _to_native(sentiment_data.get('sentiment_weighted', sentiment_score)),
                    _to_native(len(articles)),
                    _to_native(fundamental_score),
                    _to_native(technical_score),
                    _to_native(growth_score),
                    _to_native(dividend_score),
                    _to_native(gap_score),
                    _to_native(total_score),
                    _to_native(options_score),
                    _to_native(squeeze_score),
                ))
            conn.commit()
            saved_to_db = True
            logger.info(f"{ticker}: Saved to screener_scores for {today}")

        # -----------------------------------------------------------------
        # STEP 8: Save sentiment_scores
        # -----------------------------------------------------------------
        if sentiment_score is not None:
            sentiment_class = (
                'Very Bullish' if sentiment_score >= 70 else
                'Bullish' if sentiment_score >= 55 else
                'Neutral' if sentiment_score >= 45 else
                'Bearish' if sentiment_score >= 30 else
                'Very Bearish'
            )
            try:
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO sentiment_scores
                            (ticker, date, sentiment_raw, sentiment_weighted, ai_sentiment_fast,
                             article_count, relevant_article_count, sentiment_class)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s) 
                            ON CONFLICT (ticker, date) DO UPDATE SET
                                sentiment_raw = EXCLUDED.sentiment_raw,
                                sentiment_weighted = EXCLUDED.sentiment_weighted,
                                ai_sentiment_fast = EXCLUDED.ai_sentiment_fast,
                                article_count = EXCLUDED.article_count,
                                sentiment_class = EXCLUDED.sentiment_class
                        """, (
                            ticker, today,
                            _to_native(sentiment_score),
                            _to_native(sentiment_data.get('sentiment_weighted', sentiment_score)),
                            _to_native(sentiment_score),
                            _to_native(len(articles)),
                            _to_native(sentiment_data.get('relevant_count', len(articles))),
                            sentiment_class
                        ))
                    conn.commit()
            except Exception as e:
                logger.warning(f"{ticker}: sentiment_scores save failed: {e}")

        # Build scores dict for return
        scores = {
            'sentiment_score': sentiment_score,
            'fundamental_score': fundamental_score,
            'technical_score': technical_score,
            'options_flow_score': options_score,
            'short_squeeze_score': squeeze_score,
            'growth_score': growth_score,
            'dividend_score': dividend_score,
            'total_score': total_score,
            'article_count': len(articles),
        }

    except Exception as e:
        logger.error(f"{ticker}: Full analysis pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # -----------------------------------------------------------------
    # STEP 9: Generate UnifiedSignal with committee
    # -----------------------------------------------------------------
    signal = None
    try:
        from src.core import get_signal_engine
        engine = get_signal_engine()

        # Clear cache for this ticker
        if hasattr(engine, '_cache') and ticker in engine._cache:
            del engine._cache[ticker]

        signal = engine.generate_signal(ticker, force_refresh=True)
        logger.info(f"{ticker}: Generated signal - {signal.today_signal if signal else 'None'}")
    except Exception as e:
        logger.error(f"SignalEngine failed for {ticker}: {e}")

    # -----------------------------------------------------------------
    # STEP 10: Run AI Analysis (BatchAIAnalyzer) for deep insights
    # -----------------------------------------------------------------
    ai_result = None
    try:
        from src.ai.batch_ai_analysis import BatchAIAnalyzer

        analyzer = BatchAIAnalyzer()

        # Build signal data for AI context
        signal_data = {
            'total_score': scores.get('total_score', 50) if scores else 50,
            'signal_type': signal.today_signal.value if signal and signal.today_signal else 'HOLD',
            'technical_score': technical_score,
            'fundamental_score': fundamental_score,
            'sentiment_score': sentiment_score,
        }

        # Run AI analysis (fast_mode=False for full context with all collected data)
        logger.info(f"{ticker}: Running AI analysis...")
        ai_result = analyzer.analyze_ticker(ticker, signal_data, fast_mode=False)

        if ai_result:
            # Save to database
            save_ok = analyzer.save_result(ai_result)
            if save_ok:
                logger.info(f"{ticker}: AI Analysis complete - {ai_result.ai_action} ({ai_result.ai_confidence})")
            else:
                logger.warning(f"{ticker}: AI Analysis done but save failed")
        else:
            logger.warning(f"{ticker}: AI Analysis returned no result")

    except ImportError:
        logger.debug(f"{ticker}: BatchAIAnalyzer not available - skipping AI analysis")
    except Exception as e:
        logger.warning(f"{ticker}: AI Analysis failed: {e}")

    return {
        'success': True,
        'ticker': ticker,
        'signal': signal,
        'scores': scores,
        'added_to_watchlist': add_permanent and not already_exists,
        'saved_to_db': saved_to_db,
        'already_existed': already_exists,
        'force_refreshed': force_refresh,
        'ai_result': ai_result
    }


def _ticker_in_universe(ticker: str) -> bool:
    """Check if ticker already exists in universe.csv."""
    import os

    try:
        # Find universe.csv path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        universe_file = os.path.join(project_root, 'config', 'universe.csv')

        if not os.path.exists(universe_file):
            return False

        df = pd.read_csv(universe_file)
        return ticker.upper() in df['ticker'].str.upper().values
    except Exception as e:
        logger.debug(f"Error checking universe: {e}")
        return False


def _add_ticker_to_universe(ticker: str, yf_info: dict) -> bool:
    """
    Add a ticker to universe.csv with metadata from yfinance.

    Args:
        ticker: Stock ticker symbol
        yf_info: yfinance info dict

    Returns:
        True if added successfully
    """
    import os

    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        universe_file = os.path.join(project_root, 'config', 'universe.csv')

        # Read existing universe
        if os.path.exists(universe_file):
            df = pd.read_csv(universe_file)
        else:
            df = pd.DataFrame(columns=['ticker', 'name', 'sector', 'industry'])

        # Check if already exists
        if ticker.upper() in df['ticker'].str.upper().values:
            logger.info(f"{ticker} already in universe.csv")
            return True

        # Extract info from yfinance
        name = yf_info.get('shortName') or yf_info.get('longName') or ticker
        sector = yf_info.get('sector') or 'Unknown'
        industry = yf_info.get('industry') or 'Unknown'

        # Add new row
        new_row = pd.DataFrame([{
            'ticker': ticker.upper(),
            'name': name,
            'sector': sector,
            'industry': industry
        }])

        df = pd.concat([df, new_row], ignore_index=True)

        # Save
        df.to_csv(universe_file, index=False)
        logger.info(f"Added {ticker} to universe.csv: {name} ({sector}/{industry})")

        return True

    except Exception as e:
        logger.error(f"Error adding {ticker} to universe: {e}")
        raise


def _remove_ticker_from_universe(ticker: str) -> dict:
    """
    Remove a ticker from universe.csv and from all database tables.

    Args:
        ticker: Stock ticker symbol to remove

    Returns:
        dict with 'success', 'message', 'removed_from' list
    """
    import os

    ticker = ticker.upper().strip()
    removed_from = []
    errors = []

    try:
        # 1. Remove from universe.csv
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        universe_file = os.path.join(project_root, 'config', 'universe.csv')

        if os.path.exists(universe_file):
            df = pd.read_csv(universe_file)
            original_count = len(df)
            df = df[df['ticker'].str.upper() != ticker]

            if len(df) < original_count:
                df.to_csv(universe_file, index=False)
                removed_from.append('universe.csv')
                logger.info(f"Removed {ticker} from universe.csv")
            else:
                logger.info(f"{ticker} was not in universe.csv")

        # 2. Remove from database tables (ALWAYS try, regardless of universe.csv)
        try:
            from src.db.connection import get_connection

            tables_to_clean = [
                'screener_scores',
                'news_articles',
                'options_flow',
                'options_flow_scores',
                'trading_signals',
                'signals',
                'earnings_calendar',
                'historical_scores',
                'fundamentals',
                'analyst_ratings',
                'price_targets',
                'prices',
                'sentiment_scores',
                'committee_decisions',
                'agent_votes',
                'insider_transactions',
            ]

            with get_connection() as conn:
                with conn.cursor() as cur:
                    for table in tables_to_clean:
                        try:
                            # Check if table exists first
                            cur.execute("""
                                SELECT EXISTS (
                                    SELECT FROM information_schema.tables 
                                    WHERE table_name = %s
                                )
                            """, (table,))
                            if cur.fetchone()[0]:
                                cur.execute(f"DELETE FROM {table} WHERE ticker = %s", (ticker,))
                                if cur.rowcount > 0:
                                    removed_from.append(f'{table} ({cur.rowcount} rows)')
                                    logger.info(f"Removed {cur.rowcount} rows for {ticker} from {table}")
                        except Exception as e:
                            logger.debug(f"Could not clean {table}: {e}")

                conn.commit()

        except ImportError as e:
            errors.append(f"Database import error: {e}")
            logger.error(f"Database connection import failed: {e}")
        except Exception as e:
            errors.append(f"Database cleanup: {str(e)}")
            logger.warning(f"Database cleanup error: {e}")

        if removed_from:
            return {
                'success': True,
                'message': f"Successfully removed {ticker}",
                'removed_from': removed_from,
                'errors': errors
            }
        else:
            return {
                'success': False,
                'message': f"{ticker} was not found in watchlist or database",
                'removed_from': [],
                'errors': errors
            }

    except Exception as e:
        logger.error(f"Error removing {ticker}: {e}")
        return {
            'success': False,
            'message': f"Error removing {ticker}: {str(e)}",
            'removed_from': removed_from,
            'errors': [str(e)]
        }

def _get_universe_tickers() -> List[str]:
    """Get list of all tickers in universe.csv."""
    import os

    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        universe_file = os.path.join(project_root, 'config', 'universe.csv')

        if os.path.exists(universe_file):
            df = pd.read_csv(universe_file)
            return sorted(df['ticker'].str.upper().tolist())
        return []
    except Exception as e:
        logger.debug(f"Error reading universe: {e}")
        return []


def _render_remove_ticker():
    """
    Render the Remove Ticker UI.
    Allows users to remove tickers from their watchlist/universe.
    """
    import re

    # Initialize session state
    if 'remove_result' not in st.session_state:
        st.session_state.remove_result = None
    if 'remove_error' not in st.session_state:
        st.session_state.remove_error = None

    st.markdown("**Remove tickers from your watchlist** - removes from universe.csv and cleans up database entries.")

    # Get current universe for dropdown
    universe_tickers = _get_universe_tickers()

    # Two input methods: dropdown or text input
    col1, col2 = st.columns([2, 1])

    with col1:
        # Text input for manual entry (supports multiple)
        ticker_input = st.text_input(
            "Tickers to remove",
            placeholder="e.g., ADPT, DYN or select from dropdown",
            key="remove_ticker_input",
            label_visibility="collapsed"
        ).upper().strip()

    with col2:
        # Dropdown for existing tickers
        selected_from_list = st.selectbox(
            "Or select from watchlist",
            options=[""] + universe_tickers,
            key="remove_ticker_select",
            label_visibility="collapsed"
        )

    # Combine inputs
    tickers_to_remove = []
    if ticker_input:
        tickers_to_remove = [t.strip().upper() for t in re.split(r'[,;\s]+', ticker_input) if t.strip()]
    elif selected_from_list:
        tickers_to_remove = [selected_from_list]

    col1, col2 = st.columns([1, 1])

    with col1:
        also_clean_db = st.checkbox(
            "Also clean database entries",
            value=True,
            help="Remove associated data from screener_scores, news_articles, options_flow, etc."
        )

    with col2:
        remove_clicked = st.button(
            "🗑️ Remove",
            type="secondary",
            disabled=not tickers_to_remove,
            use_container_width=True
        )

    # Process removal
    if remove_clicked and tickers_to_remove:
        st.session_state.remove_result = None
        st.session_state.remove_error = None

        results = []
        errors = []

        with st.spinner(f"Removing {len(tickers_to_remove)} ticker(s)..."):
            for ticker in tickers_to_remove:
                result = _remove_ticker_from_universe(ticker)
                if result['success']:
                    results.append(result)
                else:
                    errors.append(f"{ticker}: {result['message']}")

        if results:
            st.session_state.remove_result = {
                'count': len(results),
                'tickers': [r['message'] for r in results],
                'details': results
            }
            # Force refresh of signals table
            st.session_state.signals_loaded = False
            if 'signals_data' in st.session_state:
                del st.session_state['signals_data']

        if errors:
            st.session_state.remove_error = "; ".join(errors)

        st.rerun()

    # Show results
    if st.session_state.remove_error:
        st.error(f"❌ {st.session_state.remove_error}")

    if st.session_state.remove_result:
        result = st.session_state.remove_result
        st.success(f"✅ Removed {result['count']} ticker(s)")

        # Show details
        for detail in result.get('details', []):
            if detail.get('removed_from'):
                with st.expander(f"📋 {detail['message']}", expanded=False):
                    st.write("Removed from:")
                    for loc in detail['removed_from']:
                        st.write(f"  • {loc}")

        if st.button("🗑️ Clear", key="clear_remove_result"):
            st.session_state.remove_result = None
            st.session_state.remove_error = None
            st.rerun()


def _test_ai_connection():
    """Test if AI chat assistant is working."""
    st.markdown("### 🔧 AI Connection Test")

    with st.spinner("Testing AI connection..."):
        try:
            # Test direct OpenAI client (what batch analysis uses)
            st.write("Testing direct OpenAI client connection...")

            from openai import OpenAI
            import os
            from dotenv import load_dotenv

            load_dotenv()

            base_url = os.getenv("LLM_QWEN_BASE_URL", "http://172.23.193.91:8090/v1")
            model = os.getenv("LLM_QWEN_MODEL", "Qwen3-32B-Q6_K.gguf")

            st.write(f"   - Base URL: {base_url}")
            st.write(f"   - Model: {model}")

            client = OpenAI(
                base_url=base_url,
                api_key="not-needed"
            )

            st.write("✅ OpenAI client created")

            # Try a simple test message
            st.write("📡 Sending test message...")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say 'AI connection OK' and nothing else."}
                ],
                temperature=0.1,
                max_tokens=50
            )

            result = response.choices[0].message.content
            st.write(f"✅ Response received: {result}")

            # Also test AlphaChat if available
            st.markdown("---")
            st.write("Testing AlphaChat (for comparison)...")
            try:
                from src.ai.chat import AlphaChat
                assistant = AlphaChat()
                st.write(f"✅ AlphaChat created, available={assistant.available}")
            except Exception as e:
                st.warning(f"AlphaChat not available: {e}")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())


def _test_single_ticker_analysis():
    """Test AI analysis on a single ticker with verbose output."""
    st.markdown("### 🔧 Single Ticker Test (AAPL)")

    ticker = "AAPL"  # Use a common ticker for testing

    with st.spinner(f"Testing analysis for {ticker}..."):
        try:
            from src.ai.batch_ai_analysis import BatchAIAnalyzer, STRUCTURED_ANALYSIS_PROMPT

            st.write("✅ BatchAIAnalyzer imported")

            analyzer = BatchAIAnalyzer()
            st.write("✅ Analyzer created")

            # Test context building
            st.write(f"📄 Building context for {ticker}...")
            context = analyzer._build_context_for_ticker(ticker)

            if context:
                st.write(f"✅ Context built, length: {len(context)} chars")
                with st.expander("View Context", expanded=False):
                    st.code(context[:2000] + "..." if len(context) > 2000 else context)
            else:
                st.error("❌ Context is empty!")
                return

            # Test direct AI call
            st.write("🤖 Testing direct AI call...")

            prompt = STRUCTURED_ANALYSIS_PROMPT.format(context=context)
            st.write(f"📝 Prompt length: {len(prompt)} chars")

            # Call AI directly
            st.write("📡 Calling AI (direct)...")
            response = analyzer._call_ai_direct(prompt)

            if response:
                st.write(f"✅ Response received, length: {len(response)} chars")
                with st.expander("View Raw Response", expanded=True):
                    st.code(response[:3000] + "..." if len(response) > 3000 else response)

                # Try to parse
                st.write("🔍 Parsing JSON...")
                parsed = analyzer._parse_json_response(response, ticker)

                if parsed:
                    st.write("✅ JSON parsed successfully!")
                    st.json(parsed)
                else:
                    st.error("❌ Failed to parse JSON from response")
            else:
                st.error("❌ Empty response from AI!")

        except ImportError as e:
            st.error(f"❌ Import error: {e}")
            st.info("Make sure batch_ai_analysis.py is in src/ai/ folder")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())


def _render_prediction_tracker():
    """
    Render the Prediction Tracker dashboard showing ML learning progress.
    """
    st.markdown("""
    **Monitor your ML system's learning progress.** Track predictions, outcomes, and accuracy over time.
    """)

    try:
        from src.ml.prediction_tracker import (
            get_prediction_stats,
            get_recent_predictions,
            update_outcomes,
            ensure_predictions_table,
            reset_predictions_table
        )

        # Ensure table exists
        ensure_predictions_table()

        # Get stats
        stats = get_prediction_stats()

        # =====================================================================
        # ML PROGRESS SECTION
        # =====================================================================
        st.markdown("### 🎯 ML Reliability Progress")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Predictions", stats.total_predictions)
        with col2:
            st.metric("With Outcomes", stats.predictions_with_outcome)
        with col3:
            st.metric("Pending", stats.pending_outcomes)
        with col4:
            # Color code ML status
            status_colors = {
                "BLOCKED": "🔴",
                "DEGRADED": "🟡",
                "TRADABLE": "🟢"
            }
            status_icon = status_colors.get(stats.ml_gate_status, "⚪")
            st.metric("ML Status", f"{status_icon} {stats.ml_gate_status}")

        # Progress bars
        st.markdown("#### Progress to ML Reliability Thresholds")

        col1, col2 = st.columns(2)

        with col1:
            progress_to_degraded = min(stats.samples_for_ml / 40, 1.0)
            st.progress(progress_to_degraded, text=f"DEGRADED: {stats.samples_for_ml}/40 samples")
            if stats.samples_to_degraded > 0:
                st.caption(f"Need {stats.samples_to_degraded} more samples")
            else:
                st.caption("✅ Threshold reached!")

        with col2:
            progress_to_tradable = min(stats.samples_for_ml / 80, 1.0)
            st.progress(progress_to_tradable, text=f"TRADABLE: {stats.samples_for_ml}/80 samples")
            if stats.samples_to_tradable > 0:
                st.caption(f"Need {stats.samples_to_tradable} more samples + >55% accuracy")
            else:
                if stats.direction_accuracy >= 0.55:
                    st.caption("✅ Threshold reached!")
                else:
                    st.caption(f"⚠️ Accuracy {stats.direction_accuracy:.1%} < 55% required")

        # =====================================================================
        # ACCURACY METRICS
        # =====================================================================
        if stats.predictions_with_outcome > 0:
            st.markdown("### 📈 Accuracy Metrics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                acc_color = "🟢" if stats.direction_accuracy >= 0.55 else "🔴"
                st.metric("Direction Accuracy", f"{acc_color} {stats.direction_accuracy:.1%}")
            with col2:
                st.metric("Win Rate", f"{stats.win_rate_overall:.1%}")
            with col3:
                st.metric("Mean Abs Error", f"{stats.mean_absolute_error:.2%}")
            with col4:
                st.metric("Last 30d Accuracy", f"{stats.accuracy_last_30d:.1%}" if stats.accuracy_last_30d else "N/A")

        # =====================================================================
        # UPDATE OUTCOMES BUTTON
        # =====================================================================
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("🔄 Update Outcomes", key="update_outcomes_btn"):
                with st.spinner("Fetching actual returns..."):
                    updated = update_outcomes(days_back=30)
                    if updated > 0:
                        st.success(f"✅ Updated {updated} predictions with actual outcomes!")
                        st.rerun()
                    else:
                        st.info("No predictions ready for outcome update yet.")

        with col2:
            if st.button("🔧 Fix/Reset Table", key="reset_pred_table_btn"):
                with st.spinner("Resetting predictions table..."):
                    if reset_predictions_table():
                        st.success("✅ Table reset successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to reset table")

        with col3:
            st.caption("🔄 Updates outcomes from Yahoo Finance | 🔧 Resets table if schema issues")

        # =====================================================================
        # RECENT PREDICTIONS TABLE
        # =====================================================================
        st.markdown("### 📋 Recent Predictions")

        recent_df = get_recent_predictions(limit=20)

        if recent_df.empty:
            st.info("No predictions recorded yet. Predictions are saved when you analyze stocks with the Alpha Model.")
        else:
            # Format for display
            display_df = recent_df.copy()

            # Format columns
            if 'prediction_date' in display_df.columns:
                display_df['prediction_date'] = pd.to_datetime(display_df['prediction_date']).dt.strftime('%m-%d')

            if 'predicted_return_5d' in display_df.columns:
                display_df['predicted_return_5d'] = display_df['predicted_return_5d'].apply(
                    lambda x: f"{x:+.2%}" if pd.notna(x) else "N/A"
                )

            if 'actual_return_5d' in display_df.columns:
                display_df['actual_return_5d'] = display_df['actual_return_5d'].apply(
                    lambda x: f"{x:+.2%}" if pd.notna(x) else "Pending"
                )

            if 'predicted_probability' in display_df.columns:
                display_df['predicted_probability'] = display_df['predicted_probability'].apply(
                    lambda x: f"{x:.0%}" if pd.notna(x) else "N/A"
                )

            if 'price_at_prediction' in display_df.columns:
                display_df['price_at_prediction'] = display_df['price_at_prediction'].apply(
                    lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
                )

            # Rename columns
            col_rename = {
                'ticker': 'Ticker',
                'prediction_date': 'Date',
                'alpha_signal': 'Signal',
                'predicted_return_5d': 'Pred 5d',
                'predicted_probability': 'P(Win)',
                'actual_return_5d': 'Actual 5d',
                'result': 'Result'
            }
            display_df = display_df.rename(columns={k: v for k, v in col_rename.items() if k in display_df.columns})

            # Select columns to show
            show_cols = ['Ticker', 'Date', 'Signal', 'Pred 5d', 'P(Win)', 'Actual 5d', 'Result']
            show_cols = [c for c in show_cols if c in display_df.columns]
            display_df = display_df[show_cols]

            # Color code results
            def color_result(val):
                if val == 'WIN':
                    return 'background-color: #69F0AE; color: black'
                elif val == 'LOSS':
                    return 'background-color: #FF8A80; color: black'
                elif val == 'PENDING':
                    return 'background-color: #FFE082; color: black'
                return ''

            styled = display_df.style
            if 'Result' in display_df.columns:
                styled = styled.applymap(color_result, subset=['Result'])

            st.dataframe(styled, width='stretch', height=300)

            # Summary
            if 'result' in recent_df.columns:
                wins = len(recent_df[recent_df['result'] == 'WIN'])
                losses = len(recent_df[recent_df['result'] == 'LOSS'])
                pending = len(recent_df[recent_df['result'] == 'PENDING'])
                st.caption(f"Last 20: 🟢 {wins} wins | 🔴 {losses} losses | 🟡 {pending} pending")

    except ImportError as e:
        st.warning(f"Prediction Tracker module not found: {e}")
        st.info("Place `prediction_tracker.py` in `src/ml/` folder")
    except Exception as e:
        st.error(f"Error loading prediction tracker: {e}")
        import traceback
        st.code(traceback.format_exc())


def _render_smart_refresh_status():
    """
    Render the Smart Refresh status showing data freshness.
    """
    st.markdown("""
    **Avoid duplicate work!** Only refresh data that's actually stale.
    Different data types have different refresh intervals based on how often they change.
    """)

    try:
        from src.core.smart_refresh import (
            SmartScanner, DataType, RefreshConfig,
            get_refresh_stats, reset_freshness
        )

        scanner = SmartScanner()
        stats = scanner.get_status_summary()

        # =====================================================================
        # OVERVIEW METRICS
        # =====================================================================
        st.markdown("### 📊 Data Freshness Overview")

        col1, col2, col3, col4 = st.columns(4)

        total = stats.get('total_entries', 0)
        fresh = stats.get('fresh_count', 0)
        stale = stats.get('stale_count', 0)

        with col1:
            st.metric("Total Tracked", total)
        with col2:
            st.metric("🟢 Fresh", fresh)
        with col3:
            st.metric("🟡 Stale", stale)
        with col4:
            pct = (fresh / total * 100) if total > 0 else 0
            st.metric("Fresh %", f"{pct:.0f}%")

        # =====================================================================
        # REFRESH INTERVALS
        # =====================================================================
        st.markdown("### ⏱️ Refresh Intervals")

        config = RefreshConfig()

        intervals_data = []
        for dt in DataType:
            ttl_hours = config.ttl_hours.get(dt, 24)
            type_stats = stats.get('by_type', {}).get(dt.value, {})

            if ttl_hours >= 168:
                interval_str = f"{ttl_hours // 24} days"
            elif ttl_hours >= 24:
                interval_str = f"{ttl_hours // 24} day{'s' if ttl_hours >= 48 else ''}"
            else:
                interval_str = f"{ttl_hours} hours"

            intervals_data.append({
                "Data Type": dt.value.replace('_', ' ').title(),
                "Interval": interval_str,
                "Fresh": type_stats.get('fresh', 0),
                "Stale": type_stats.get('stale', 0),
            })

        intervals_df = pd.DataFrame(intervals_data)

        # Color code the rows
        def color_stale(val):
            if isinstance(val, (int, float)) and val > 0:
                return 'background-color: #FFCDD2'
            return ''

        styled = intervals_df.style.applymap(color_stale, subset=['Stale'])
        st.dataframe(styled, width='stretch', hide_index=True)

        # =====================================================================
        # EXPLANATION
        # =====================================================================
        st.markdown("### 💡 How It Works")

        st.markdown("""
        | Data Type | Why This Interval |
        |-----------|-------------------|
        | **News/Sentiment** | Can change rapidly, especially near earnings |
        | **Options Flow** | Intraday shifts, refreshed every 4 hours |
        | **Technical** | Based on daily prices, refreshed daily |
        | **Fundamentals** | Only changes on earnings, refreshed weekly |
        | **Insider/13F** | SEC filings are slow, refreshed weekly |
        
        **Near Earnings**: When a stock is within 7 days of earnings, news/sentiment/options refresh 4x more frequently!
        """)

        # =====================================================================
        # CONTROLS
        # =====================================================================
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("🔄 Force Full Refresh", key="force_refresh_btn"):
                with st.spinner("Resetting freshness tracking..."):
                    reset_freshness()
                    st.success("✅ All data marked for refresh on next scan!")
                    st.rerun()

        with col2:
            refresh_type = st.selectbox(
                "Reset specific type",
                options=[""] + [dt.value for dt in DataType],
                key="reset_type_select"
            )
            if refresh_type and st.button("Reset Type", key="reset_type_btn"):
                dt = DataType(refresh_type)
                reset_freshness(data_types=[dt])
                st.success(f"✅ {refresh_type} marked for refresh!")

        with col3:
            st.caption("""
            **Force Full Refresh**: Clears all freshness tracking - everything will refresh on next scan.
            **Reset Type**: Only reset a specific data type (e.g., news only).
            """)

    except ImportError as e:
        st.warning(f"Smart Refresh module not found: {e}")
        st.info("Place `smart_refresh.py` in `src/core/` folder")
    except Exception as e:
        st.error(f"Error loading smart refresh: {e}")
        import traceback
        st.code(traceback.format_exc())


def _render_ai_batch_analysis(filtered_df: pd.DataFrame):
    """
    Render the AI Batch Analysis UI section.
    Allows running AI analysis on top N signals and displays results.
    """

    # Initialize session state
    if 'ai_batch_running' not in st.session_state:
        st.session_state.ai_batch_running = False
    if 'ai_batch_progress' not in st.session_state:
        st.session_state.ai_batch_progress = {'current': 0, 'total': 0, 'ticker': ''}
    if 'ai_batch_results' not in st.session_state:
        st.session_state.ai_batch_results = None

    st.markdown("""
    **Run AI deep analysis on your top signals.** The AI will analyze each ticker and provide:
    - Action recommendation (BUY/SELL/HOLD)
    - Confidence level and position sizing
    - Entry/Exit prices and stop loss
    - Bull/Bear case and key risks
    """)

    # Speed mode info
    st.info("""💡 **Fast Mode Enabled**: Using cached data only (no API calls).  
    ⏱️ Estimated time: ~5-10 seconds per ticker (AI processing only).  
    📊 For 10 stocks ≈ 1-2 minutes | For 20 stocks ≈ 2-4 minutes""")

    # Controls row
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        top_n = st.selectbox(
            "Analyze Top N",
            options=[5, 10, 15, 20, 30],
            index=1,  # Default to 10
            key="ai_batch_top_n"
        )

    with col2:
        sort_options = ['Total', 'Sentiment', 'OptFlow', 'Fundamental', 'Technical']
        sort_by = st.selectbox(
            "Sort By",
            options=sort_options,
            key="ai_batch_sort_by"
        )

    with col3:
        filter_options = ['All', 'BUY Only', 'SELL Only', 'BUY+SELL']
        signal_filter = st.selectbox(
            "Signal Filter",
            options=filter_options,
            key="ai_batch_signal_filter"
        )

    with col4:
        st.write("")  # Spacer
        st.write("")
        run_clicked = st.button(
            "🤖 Run AI Analysis",
            type="primary",
            width='stretch',
            disabled=st.session_state.ai_batch_running,
            key="ai_batch_run_btn"
        )

    # Debug button row
    debug_col1, debug_col2 = st.columns(2)
    with debug_col1:
        if st.button("🔧 Test AI Connection", key="ai_test_connection"):
            _test_ai_connection()
    with debug_col2:
        if st.button("🔧 Test Single Ticker", key="ai_test_single"):
            _test_single_ticker_analysis()

    # Process if button clicked
    if run_clicked and not st.session_state.ai_batch_running:
        st.session_state.ai_batch_running = True
        st.session_state.ai_batch_results = None

        # Get tickers to analyze
        df_to_analyze = filtered_df.copy()

        # Apply signal filter
        if signal_filter == 'BUY Only' and 'Signal' in df_to_analyze.columns:
            df_to_analyze = df_to_analyze[df_to_analyze['Signal'].str.contains('BUY', na=False)]
        elif signal_filter == 'SELL Only' and 'Signal' in df_to_analyze.columns:
            df_to_analyze = df_to_analyze[df_to_analyze['Signal'].str.contains('SELL', na=False)]
        elif signal_filter == 'BUY+SELL' and 'Signal' in df_to_analyze.columns:
            df_to_analyze = df_to_analyze[
                df_to_analyze['Signal'].str.contains('BUY', na=False) |
                df_to_analyze['Signal'].str.contains('SELL', na=False)
            ]

        # Sort
        if sort_by in df_to_analyze.columns:
            df_to_analyze = df_to_analyze.sort_values(sort_by, ascending=False)

        # Get top N tickers
        tickers = df_to_analyze.head(top_n)['Ticker'].tolist() if 'Ticker' in df_to_analyze.columns else []

        if not tickers:
            st.warning("No tickers match the filter criteria")
            st.session_state.ai_batch_running = False
        else:
            # Build signal data dict
            signal_data = {}
            for _, row in df_to_analyze.iterrows():
                ticker = row.get('Ticker')
                if ticker:
                    signal_data[ticker] = {
                        # FIX: Use None for missing scores, not 50
                        'total_score': row.get('Total') if pd.notna(row.get('Total')) else None,
                        'signal_type': row.get('Signal', 'HOLD'),
                        'technical_score': row.get('Technical') if pd.notna(row.get('Technical')) else None,
                        'fundamental_score': row.get('Fundamental') if pd.notna(row.get('Fundamental')) else None,
                        'sentiment_score': row.get('Sentiment') if pd.notna(row.get('Sentiment')) else None,
                    }

            # Run analysis
            try:
                from src.ai.batch_ai_analysis import BatchAIAnalyzer
                import time

                analyzer = BatchAIAnalyzer()
                results = []
                errors = []

                progress_bar = st.progress(0, text="Starting AI analysis...")
                status_text = st.empty()
                eta_text = st.empty()
                error_container = st.empty()

                start_time = time.time()

                for i, ticker in enumerate(tickers):
                    ticker_start = time.time()

                    # Calculate ETA
                    elapsed = time.time() - start_time
                    if i > 0:
                        avg_per_ticker = elapsed / i
                        remaining = (len(tickers) - i) * avg_per_ticker
                        if remaining < 60:
                            eta_str = f"~{remaining:.0f} seconds remaining"
                        else:
                            eta_str = f"~{remaining/60:.1f} minutes remaining"
                    else:
                        eta_str = "Calculating ETA..."

                    progress = (i + 1) / len(tickers)
                    progress_bar.progress(progress, text=f"Analyzing {ticker}... ({i+1}/{len(tickers)})")
                    status_text.info(f"🤖 Processing: {ticker}")
                    eta_text.caption(f"⏱️ {eta_str} | Elapsed: {elapsed:.0f}s")

                    try:
                        # Use fast_mode=True for DB-only context (much faster)
                        result = analyzer.analyze_ticker(ticker, signal_data.get(ticker, {}), fast_mode=True)
                        ticker_elapsed = time.time() - ticker_start

                        if result:
                            save_ok = analyzer.save_result(result)
                            if save_ok:
                                results.append(result)
                                logger.info(f"✅ {ticker}: {result.ai_action} ({result.ai_confidence}) - {ticker_elapsed:.1f}s")
                            else:
                                errors.append(f"{ticker}: Failed to save")
                        else:
                            errors.append(f"{ticker}: No result returned")
                            logger.warning(f"❌ {ticker}: analyze_ticker returned None - {ticker_elapsed:.1f}s")
                    except Exception as e:
                        error_msg = f"{ticker}: {str(e)}"
                        errors.append(error_msg)
                        logger.error(f"Error analyzing {ticker}: {e}")
                        continue

                total_time = time.time() - start_time
                progress_bar.progress(1.0, text="Complete!")
                eta_text.empty()

                if results:
                    status_text.success(f"✅ Analyzed {len(results)} of {len(tickers)} tickers in {total_time:.0f}s")
                else:
                    status_text.warning(f"⚠️ No results obtained from {len(tickers)} tickers")

                # Show errors if any
                if errors:
                    with error_container.expander(f"⚠️ {len(errors)} errors occurred", expanded=False):
                        for err in errors:
                            st.text(err)

                st.session_state.ai_batch_results = results
                st.session_state.ai_batch_running = False

                # Force refresh to show results
                if results:
                    st.rerun()

            except ImportError as e:
                st.error(f"❌ Batch AI Analysis module not available: {e}")
                st.info("Make sure `batch_ai_analysis.py` is in `src/ai/` folder")
                st.session_state.ai_batch_running = False
            except Exception as e:
                st.error(f"❌ Analysis error: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.session_state.ai_batch_running = False

    # Display results
    st.markdown("---")
    st.markdown("### 📊 AI Analysis Results")

    # Load from database
    try:
        from src.ai.batch_ai_analysis import get_ai_analysis_for_display
        ai_df = get_ai_analysis_for_display()

        if ai_df.empty:
            st.info("No AI analysis results yet. Click **Run AI Analysis** to analyze top signals.")
        else:
            # Format for display
            display_cols = {
                'ticker': 'Ticker',
                'ai_action': 'AI Action',
                'ai_confidence': 'Confidence',
                'trade_allowed': 'Trade OK',
                'entry_price': 'Entry',
                'stop_loss': 'Stop',
                'target_price': 'Target',
                'position_size': 'Size',
                'time_horizon': 'Horizon',
                'platform_score': 'Score',
                'analysis_date': 'Analyzed'
            }

            # Select and rename columns
            available_cols = [c for c in display_cols.keys() if c in ai_df.columns]
            ai_display = ai_df[available_cols].copy()
            ai_display = ai_display.rename(columns=display_cols)

            # Format date
            if 'Analyzed' in ai_display.columns:
                ai_display['Analyzed'] = pd.to_datetime(ai_display['Analyzed']).dt.strftime('%m-%d %H:%M')

            # Format prices
            for col in ['Entry', 'Stop', 'Target']:
                if col in ai_display.columns:
                    ai_display[col] = ai_display[col].apply(
                        lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
                    )

            # Color code AI Action
            def color_action(val):
                if pd.isna(val):
                    return ''
                val = str(val).upper()
                if 'BUY' in val:
                    return 'background-color: #69F0AE; color: black'
                elif 'SELL' in val:
                    return 'background-color: #FF8A80; color: black'
                elif 'WAIT' in val:
                    return 'background-color: #FFE082; color: black'
                return ''

            def color_confidence(val):
                if pd.isna(val):
                    return ''
                val = str(val).upper()
                if val == 'HIGH':
                    return 'background-color: #00C853; color: white'
                elif val == 'MEDIUM':
                    return 'background-color: #FFD54F; color: black'
                return 'background-color: #BDBDBD'

            def color_trade_ok(val):
                if val == True:
                    return 'background-color: #69F0AE'
                elif val == False:
                    return 'background-color: #FF8A80'
                return ''

            # Apply styling
            styled = ai_display.style
            if 'AI Action' in ai_display.columns:
                styled = styled.applymap(color_action, subset=['AI Action'])
            if 'Confidence' in ai_display.columns:
                styled = styled.applymap(color_confidence, subset=['Confidence'])
            if 'Trade OK' in ai_display.columns:
                styled = styled.applymap(color_trade_ok, subset=['Trade OK'])

            # Stats row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                buy_count = len(ai_df[ai_df['ai_action'] == 'BUY']) if 'ai_action' in ai_df.columns else 0
                st.metric("🟢 BUY", buy_count)
            with col2:
                sell_count = len(ai_df[ai_df['ai_action'] == 'SELL']) if 'ai_action' in ai_df.columns else 0
                st.metric("🔴 SELL", sell_count)
            with col3:
                hold_count = len(ai_df[ai_df['ai_action'] == 'HOLD']) if 'ai_action' in ai_df.columns else 0
                st.metric("🟡 HOLD", hold_count)
            with col4:
                high_conf = len(ai_df[ai_df['ai_confidence'] == 'HIGH']) if 'ai_confidence' in ai_df.columns else 0
                st.metric("⭐ High Conf", high_conf)

            # Display table
            st.dataframe(styled, width='stretch', height=300)

            # Filter controls for AI results
            col1, col2 = st.columns(2)
            with col1:
                ai_action_filter = st.multiselect(
                    "Filter by AI Action",
                    options=['BUY', 'SELL', 'HOLD', 'WAIT_FOR_PULLBACK'],
                    key="ai_result_action_filter"
                )
            with col2:
                ai_conf_filter = st.multiselect(
                    "Filter by Confidence",
                    options=['HIGH', 'MEDIUM', 'LOW'],
                    key="ai_result_conf_filter"
                )

            # Show filtered actionable signals
            if ai_action_filter or ai_conf_filter:
                filtered_ai = ai_df.copy()
                if ai_action_filter:
                    filtered_ai = filtered_ai[filtered_ai['ai_action'].isin(ai_action_filter)]
                if ai_conf_filter:
                    filtered_ai = filtered_ai[filtered_ai['ai_confidence'].isin(ai_conf_filter)]

                if not filtered_ai.empty:
                    st.markdown(f"**Filtered Results: {len(filtered_ai)} signals**")
                    st.dataframe(filtered_ai[['ticker', 'ai_action', 'ai_confidence', 'entry_price', 'target_price']])

    except ImportError:
        st.warning("AI Batch Analysis module not installed. Place `batch_ai_analysis.py` in `src/ai/` folder.")
    except Exception as e:
        st.error(f"Error loading AI results: {e}")
        logger.error(f"AI results display error: {e}")


def _render_signals_table_view():
    """Render the signals table (Universe-style) with dropdown for deep dive."""
    from src.db.connection import get_engine

    # =========================================================================
    # MARKET OVERVIEW (SPY, QQQ, VIX)
    # =========================================================================
    try:
        overview = get_market_overview()
        if overview:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                c = "🟢" if overview.spy_change >= 0 else "🔴"
                st.metric("SPY", f"{c} {overview.spy_change:+.2f}%")
            with col2:
                c = "🟢" if overview.qqq_change >= 0 else "🔴"
                st.metric("QQQ", f"{c} {overview.qqq_change:+.2f}%")
            with col3:
                st.metric("VIX", f"{overview.vix:.1f}")
            with col4:
                st.metric("Regime", overview.regime)
    except Exception as e:
        logger.debug(f"Could not load market overview: {e}")

    # =========================================================================
    # LOAD DATA FROM DATABASE
    # =========================================================================
    try:
        engine = get_engine()
        df = pd.read_sql("""
                    WITH latest_prices AS (
                        SELECT DISTINCT ON (ticker) ticker, close as price
                        FROM prices
                        ORDER BY ticker, date DESC
                    ),
                    latest_fundamentals AS (
                        SELECT DISTINCT ON (ticker) *
                        FROM fundamentals
                        ORDER BY ticker, date DESC
                    ),
                    latest_analyst_ratings AS (
                        SELECT DISTINCT ON (ticker) *
                        FROM analyst_ratings
                        ORDER BY ticker, date DESC
                    ),
                    latest_price_targets AS (
                        SELECT DISTINCT ON (ticker) *
                        FROM price_targets
                        ORDER BY ticker, date DESC
                    )
                    SELECT 
                        s.ticker,
                        s.date as score_date,
                        f.sector,
                        f.market_cap,
                        CASE 
                            WHEN s.total_score >= 80 THEN 'STRONG BUY'
                            WHEN s.total_score >= 65 THEN 'BUY'
                            WHEN s.total_score >= 55 THEN 'WEAK BUY'
                            WHEN s.total_score <= 20 THEN 'STRONG SELL'
                            WHEN s.total_score <= 35 THEN 'SELL'
                            WHEN s.total_score <= 45 THEN 'WEAK SELL'
                            ELSE 'HOLD'
                        END as signal_type,
                        s.total_score,
                        s.sentiment_score,
                        s.options_flow_score,
                        s.short_squeeze_score,
                        s.fundamental_score,
                        s.growth_score,
                        s.dividend_score,
                        s.technical_score,
                        s.gap_score,
                        CASE 
                            WHEN s.gap_score >= 70 THEN 'BEARISH'
                            WHEN s.gap_score <= 30 THEN 'BULLISH'
                            ELSE 'None'
                        END as gap_type,
                        s.likelihood_score,
                        s.article_count,
                        lp.price,
                        pt.target_mean,
                        -- Get next upcoming earnings OR most recent past earnings (within 14 days)
                        COALESCE(
                            (SELECT earnings_date FROM earnings_calendar 
                             WHERE ticker = s.ticker AND earnings_date >= CURRENT_DATE 
                             ORDER BY earnings_date ASC LIMIT 1),
                            (SELECT earnings_date FROM earnings_calendar 
                             WHERE ticker = s.ticker AND earnings_date >= CURRENT_DATE - INTERVAL '14 days' AND earnings_date < CURRENT_DATE
                             ORDER BY earnings_date DESC LIMIT 1)
                        ) as earnings_date,
                        f.ex_dividend_date,
                        CASE WHEN lp.price > 0 AND pt.target_mean > 0 
                             THEN ROUND(((pt.target_mean - lp.price) / lp.price * 100)::numeric, 2)
                             ELSE NULL END as target_upside_pct,
                        ar.analyst_positivity,
                        ar.analyst_buy as buy_count,
                        ar.analyst_total as total_ratings,
                        f.dividend_yield,
                        f.pe_ratio,
                        f.forward_pe,
                        f.peg_ratio,
                        f.roe
                    FROM screener_scores s
                    LEFT JOIN latest_fundamentals f ON s.ticker = f.ticker
                    LEFT JOIN latest_analyst_ratings ar ON s.ticker = ar.ticker
                    LEFT JOIN latest_price_targets pt ON s.ticker = pt.ticker
                    LEFT JOIN latest_prices lp ON s.ticker = lp.ticker
                    WHERE s.date = (
                        SELECT MAX(date) FROM screener_scores ss WHERE ss.ticker = s.ticker
                    )
                    ORDER BY s.total_score DESC NULLS LAST
                """, engine)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    if df.empty:
        st.warning("No signals data. Run Analysis first.")
        return

    # =========================================================================
    # ENHANCED SCORING - OPTIONAL (expensive operation)
    # =========================================================================
    # Enhanced scoring is computed during batch analysis and stored.
    # This checkbox allows on-demand recalculation for display only.
    col_enh1, col_enh2 = st.columns([3, 1])
    with col_enh2:
        apply_enhanced = st.checkbox(
            "🔬 Live Enhanced Scoring",
            value=False,
            help="Fetches live data (MACD, earnings, PEG) - slower but most current. Leave OFF for fast loading."
        )

    if ENHANCED_SCORING_AVAILABLE and apply_enhanced:
        try:
            with st.spinner("Fetching live enhanced scores (this may take 30-60 seconds)..."):
                df = apply_enhanced_scores_to_dataframe(
                    df,
                    score_column='total_score',
                    enhanced_column='enhanced_score',
                    adjustment_column='score_adjustment',
                    fetch_price_history=True,
                )
                logger.info(f"Enhanced scoring applied to {len(df)} tickers")

                # Use enhanced score as primary if available
                if 'enhanced_score' in df.columns:
                    df['original_total'] = df['total_score']
                    df['total_score'] = df['enhanced_score']

                    # Recalculate signal type based on enhanced score
                    df['signal_type'] = df['total_score'].apply(lambda x:
                        'STRONG BUY' if x >= 80 else
                        'BUY' if x >= 65 else
                        'WEAK BUY' if x >= 55 else
                        'STRONG SELL' if x <= 20 else
                        'SELL' if x <= 35 else
                        'WEAK SELL' if x <= 45 else
                        'HOLD'
                    )
        except Exception as e:
            logger.warning(f"Enhanced scoring failed, using base scores: {e}")

    # Column mapping
    display_cols = {
        'ticker': 'Ticker',
        'score_date': 'Date',
        'sector': 'Sector',
        'market_cap': 'Cap',
        'signal_type': 'Signal',
        'total_score': 'Total',
        'score_adjustment': 'Adj',  # Enhanced scoring adjustment
        'sentiment_score': 'Sentiment',
        'options_flow_score': 'OptFlow',
        'short_squeeze_score': 'Squeeze',
        'fundamental_score': 'Fundamental',
        'growth_score': 'Growth',
        'dividend_score': 'Dividend',
        'technical_score': 'Technical',
        'gap_score': 'Gap',
        'gap_type': 'GapType',
        'likelihood_score': 'Likelihood',
        'article_count': 'Articles',
        'price': 'Price',
        'target_mean': 'TargetPrice',
        'earnings_date': 'Earnings',
        'ex_dividend_date': 'Ex-Div',
        'target_upside_pct': 'Upside%',
        'analyst_positivity': 'AnalystPos%',
        'buy_count': 'BuyRatings',
        'total_ratings': 'TotalRatings',
        'dividend_yield': 'DivYield%',
        'pe_ratio': 'PE',
        'forward_pe': 'FwdPE',
        'peg_ratio': 'PEG',
        'roe': 'ROE',
    }

    available_cols = [c for c in display_cols.keys() if c in df.columns]
    display_df = df[available_cols].copy()
    display_df = display_df.rename(columns={k: v for k, v in display_cols.items() if k in display_df.columns})

    # Format dates
    if 'Date' in display_df.columns:
        def format_score_date(val):
            if pd.isna(val) or val == '' or val is None:
                return ''
            try:
                if hasattr(val, 'strftime'):
                    return val.strftime('%m-%d')
                return str(val)[5:10] if len(str(val)) >= 10 else str(val)
            except:
                return ''
        display_df['Date'] = display_df['Date'].apply(format_score_date)

    # Format market cap as human-readable (B/T)
    if 'Cap' in display_df.columns:
        def format_market_cap(val):
            if pd.isna(val) or val == 0:
                return ''
            try:
                v = float(val)
                if v >= 1e12:
                    return f"${v / 1e12:.1f}T"
                elif v >= 1e9:
                    return f"${v / 1e9:.0f}B"
                elif v >= 1e6:
                    return f"${v / 1e6:.0f}M"
                else:
                    return f"${v:,.0f}"
            except:
                return ''

        display_df['Cap'] = display_df['Cap'].apply(format_market_cap)

    if 'Earnings' in display_df.columns:
        def format_earnings_date(val):
            if pd.isna(val) or val == '' or val is None:
                return ''
            try:
                from datetime import date
                if hasattr(val, 'date'):
                    dt = val.date() if hasattr(val, 'date') else val
                elif hasattr(val, 'strftime'):
                    dt = val
                else:
                    dt = pd.to_datetime(str(val)).date()

                today = date.today()

                # Format: MM-DD with indicator
                if dt < today:
                    # Past earnings (already reported) - add ✓
                    if val.year != today.year:
                        return f"✓{dt.strftime('%m-%d-%y')}"
                    return f"✓{dt.strftime('%m-%d')}"
                else:
                    # Future earnings (upcoming)
                    if val.year != today.year:
                        return dt.strftime('%m-%d-%y')
                    return dt.strftime('%m-%d')
            except:
                return str(val)[:10] if len(str(val)) >= 10 else str(val)
        display_df['Earnings'] = display_df['Earnings'].apply(format_earnings_date)

    if 'Ex-Div' in display_df.columns:
        display_df['Ex-Div'] = display_df['Ex-Div'].apply(lambda x: x.strftime('%m-%d') if hasattr(x, 'strftime') else '')

    # Format numeric columns - KEEP AS NUMBERS for proper sorting
    # FIX: Score columns should show empty/- for NULL, not 0
    # Separate score columns (where NULL = missing data) from count columns (where 0 is valid)
    count_columns = ['BuyRatings', 'TotalRatings', 'Articles']  # 0 is valid here
    score_columns = ['Sentiment', 'Fundamental', 'Growth', 'Dividend', 'Technical',
                     'Gap', 'Likelihood', 'Total', 'OptFlow', 'Squeeze']  # NULL = missing

    for col in count_columns:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce').fillna(0).astype(int)

    for col in score_columns:
        if col in display_df.columns:
            # Keep as numeric for sorting, but NaN will display as empty
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
            # Round first, then convert to nullable int - NaN stays as NaN for display
            display_df[col] = display_df[col].round(0).astype('Int64')  # Nullable integer type

    # Convert decimal columns to numeric (NOT strings) - required for proper sorting
    # Fill NaN with -999 for sorting purposes (will sort to bottom in descending order)
    decimal_columns = ['Price', 'TargetPrice', 'Upside%', 'AnalystPos%', 'DivYield%', 'PE', 'ROE']
    for col in decimal_columns:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
            # Round for display but keep as float
            display_df[col] = display_df[col].round(2)

    # =========================================================================
    # ADD AI WIN PROBABILITY COLUMN
    # =========================================================================
    if AI_PROB_AVAILABLE and 'Ticker' in display_df.columns:
        try:
            # Get AI probabilities for all tickers (cached for 5 min)
            ticker_scores = {}
            for _, row in display_df.iterrows():
                ticker_scores[row['Ticker']] = {
                    # FIX: Use None as default, not 50 - AI model should handle missing data
                    'sentiment_score': row.get('Sentiment') if pd.notna(row.get('Sentiment')) else None,
                    'fundamental_score': row.get('Fundamental') if pd.notna(row.get('Fundamental')) else None,
                    'technical_score': row.get('Technical') if pd.notna(row.get('Technical')) else None,
                    'options_flow_score': row.get('OptFlow') if pd.notna(row.get('OptFlow')) else None,
                    'short_squeeze_score': row.get('Squeeze') if pd.notna(row.get('Squeeze')) else None,
                    'total_score': row.get('Total') if pd.notna(row.get('Total')) else None,
                }

            ai_probs = _get_ai_probabilities_for_table(ticker_scores)

            # Add AI Prob column
            display_df['AI Prob'] = display_df['Ticker'].map(
                lambda t: ai_probs.get(t, {}).get('prob', None)
            )
            # Convert to percentage for display (keep as float for sorting)
            display_df['AI Prob'] = pd.to_numeric(display_df['AI Prob'], errors='coerce')

            # Reorder columns to put AI Prob after Total
            if 'AI Prob' in display_df.columns and 'Total' in display_df.columns:
                cols = display_df.columns.tolist()
                if 'AI Prob' in cols:
                    cols.remove('AI Prob')
                    total_idx = cols.index('Total') if 'Total' in cols else 0
                    cols.insert(total_idx + 1, 'AI Prob')
                    display_df = display_df[cols]

        except Exception as e:
            logger.debug(f"AI Prob column not added: {e}")

    # =========================================================================
    # FILTERS ROW
    # =========================================================================
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 2, 2])

    with col1:
        signal_options = ['STRONG BUY', 'BUY', 'WEAK BUY', 'HOLD', 'WEAK SELL', 'SELL', 'STRONG SELL']
        signal_filter = st.multiselect("Filter by Signal", options=signal_options, key="signal_filter")

    with col2:
        min_total = st.slider("Min Total", 0, 100, 0, key="min_total_slider")

    with col3:
        min_sentiment = st.slider("Min Sent", 0, 100, 0, key="min_sentiment_slider")

    with col4:
        # AI Strategy Filter
        ai_filter_options = [
            "All Signals",
            "🤖 AI Probability (≥55%)",
            "🤖 AI Conservative (≥60%)",
            "🤖 AI High Conviction (≥65%)"
        ]
        ai_filter = st.selectbox("AI Strategy", ai_filter_options, key="ai_filter_select")

    with col5:
        sort_options = ['Total', 'AI Prob', 'Sentiment', 'OptFlow', 'Squeeze', 'Fundamental', 'Technical', 'Upside%',
                        'Price', 'PE']
        sort_col1, sort_col2 = st.columns([3, 1])
        with sort_col1:
            sort_by = st.selectbox("Sort by", sort_options, key="sort_by_select")
        with sort_col2:
            sort_dir = st.selectbox("↕️", ["Desc", "Asc"], key="sort_dir_select")

    # Apply filters
    filtered_df = display_df.copy()
    if signal_filter and 'Signal' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Signal'].isin(signal_filter)]
    if min_total > 0 and 'Total' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Total'] >= min_total]
    if min_sentiment > 0 and 'Sentiment' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Sentiment'] >= min_sentiment]

    # Apply AI Strategy filter
    if 'AI Prob' in filtered_df.columns and ai_filter != "All Signals":
        if "≥55%" in ai_filter:
            filtered_df = filtered_df[filtered_df['AI Prob'] >= 0.55]
        elif "≥60%" in ai_filter:
            filtered_df = filtered_df[filtered_df['AI Prob'] >= 0.60]
        elif "≥65%" in ai_filter:
            filtered_df = filtered_df[filtered_df['AI Prob'] >= 0.65]

    # Sort - use dropdown sorting (column header sorting in Streamlit is unreliable)
    if sort_by in filtered_df.columns:
        ascending = (sort_dir == "Asc")
        filtered_df = filtered_df.sort_values(sort_by, ascending=ascending, na_position='last')
        # Reset index for clean display
        filtered_df = filtered_df.reset_index(drop=True)

    # =========================================================================
    # AI BATCH ANALYSIS SECTION
    # =========================================================================
    with st.expander("🤖 AI Batch Analysis - Deep Analysis on Top Signals", expanded=False):
        _render_ai_batch_analysis(filtered_df)

    # =========================================================================
    # PREDICTION TRACKER SECTION
    # =========================================================================
    with st.expander("📊 ML Learning Progress - Prediction Tracker", expanded=False):
        _render_prediction_tracker()

    # =========================================================================
    # SMART REFRESH STATUS
    # =========================================================================
    with st.expander("🔄 Smart Data Refresh - Avoid Duplicate Work", expanded=False):
        _render_smart_refresh_status()

    # =========================================================================
    # STOCK SELECTOR FOR DEEP DIVE
    # =========================================================================
    st.markdown("---")

    ticker_list = filtered_df['Ticker'].tolist() if 'Ticker' in filtered_df.columns else []

    # Check if there's a pre-selected ticker from Quick Add
    preselected = st.session_state.get('selected_ticker')

    # If preselected ticker is not in the filtered list, add it temporarily
    # (This handles newly added tickers that might not be in the current filter)
    if preselected and preselected not in ticker_list:
        ticker_list = [preselected] + ticker_list

    # Calculate default index
    default_index = 0
    if preselected and preselected in ticker_list:
        default_index = ticker_list.index(preselected) + 1  # +1 because we prepend ""

    col_select, col_refresh, col_info = st.columns([2, 1, 2])

    with col_select:
        selected = st.selectbox(
            "🔍 Select Stock for Deep Dive",
            options=[""] + ticker_list,
            index=default_index,
            key="ticker_deep_dive_select",
            help="Select a stock to see detailed analysis below the table"
        )
        if selected:
            st.session_state.selected_ticker = selected

    with col_refresh:
        if selected and st.button(f"🔄 Refresh {selected}", key="refresh_selected_btn"):
            _run_single_analysis(selected)

    with col_info:
        if selected:
            # Show quick summary
            row = filtered_df[filtered_df['Ticker'] == selected].iloc[0] if not filtered_df[filtered_df['Ticker'] == selected].empty else None
            if row is not None:
                signal = row.get('Signal', 'N/A')
                total = row.get('Total', 0)
                signal_emoji = "🟢" if "BUY" in str(signal) else "🔴" if "SELL" in str(signal) else "🟡"
                st.markdown(f"**{selected}**: {signal_emoji} {signal} | Score: {total}")

    # =========================================================================
    # STYLED TABLE
    # =========================================================================
    def color_signal(val):
        if pd.isna(val):
            return ''
        val = str(val).upper()
        if 'STRONG BUY' in val:
            return 'background-color: #00C853; color: white'
        elif 'BUY' in val:
            return 'background-color: #69F0AE'
        elif 'STRONG SELL' in val:
            return 'background-color: #FF1744; color: white'
        elif 'SELL' in val:
            return 'background-color: #FF8A80'
        return ''

    def color_score(val):
        if pd.isna(val):
            return ''
        try:
            v = float(val)
            if v >= 70:
                return 'background-color: #00C853; color: white'
            elif v >= 60:
                return 'background-color: #69F0AE'
            elif v <= 30:
                return 'background-color: #FF1744; color: white'
            elif v <= 40:
                return 'background-color: #FF8A80'
        except:
            pass
        return ''

    def color_squeeze(val):
        if pd.isna(val):
            return ''
        try:
            v = float(val)
            if v >= 70:
                return 'background-color: #FFD700; color: black'
            elif v >= 50:
                return 'background-color: #FFA500'
        except:
            pass
        return ''

    def color_ai_prob(val):
        """Color AI probability: green >= 55%, orange 45-55%, red < 45%"""
        if pd.isna(val):
            return ''
        try:
            v = float(val)
            if v >= 0.55:
                return 'background-color: #00C853; color: white'
            elif v >= 0.45:
                return 'background-color: #FFA500'
            else:
                return 'background-color: #FF1744; color: white'
        except:
            pass
        return ''

    # Apply styling

    # Apply styling (use map for pandas 2.1+ compatibility)

    # First, convert numeric columns to proper numeric types (handles "None" strings, etc.)
    numeric_cols = ['Price', 'TargetPrice', 'Upside%', 'AnalystPos%', 'DivYield%',
                    'PE', 'FwdPE', 'PEG', 'ROE', 'AI Prob',
                    'Total', 'Technical', 'Fundamental', 'Sentiment', 'OptFlow',
                    'Squeeze', 'Growth', 'Dividend', 'Gap', 'Likelihood',
                    'Articles', 'BuyRatings', 'TotalRatings']

    for col in numeric_cols:
        if col in filtered_df.columns:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')

    styled_df = filtered_df.style

    # Format numeric columns - CLEAN DECIMALS
    format_dict = {}

    # Price columns - 2 decimal places
    price_cols = ['Price', 'TargetPrice']
    for col in price_cols:
        if col in filtered_df.columns:
            format_dict[col] = '{:.2f}'

    # Percentage columns - 2 decimal places
    pct_cols = ['Upside%', 'AnalystPos%', 'DivYield%']
    for col in pct_cols:
        if col in filtered_df.columns:
            format_dict[col] = '{:.2f}'

    # Ratio columns - 2 decimal places
    ratio_cols = ['PE', 'FwdPE', 'PEG', 'ROE', 'AI Prob']
    for col in ratio_cols:
        if col in filtered_df.columns:
            format_dict[col] = '{:.2f}'

    # Score columns - integer (no decimals)
    score_cols = ['Total', 'Technical', 'Fundamental', 'Sentiment', 'OptFlow',
                  'Squeeze', 'Growth', 'Dividend', 'Gap', 'Likelihood']
    for col in score_cols:
        if col in filtered_df.columns:
            format_dict[col] = '{:.0f}'

    # Count columns - integer (no decimals)
    count_cols = ['Articles', 'BuyRatings', 'TotalRatings']
    for col in count_cols:
        if col in filtered_df.columns:
            format_dict[col] = '{:.0f}'

    # Apply format
    if format_dict:
        styled_df = styled_df.format(format_dict, na_rep='N/A')

    try:
        # Try newer pandas API first
        if 'Signal' in filtered_df.columns:
            styled_df = styled_df.map(color_signal, subset=['Signal'])
        for col in ['OptFlow', 'Sentiment', 'Fundamental', 'Technical']:
            if col in filtered_df.columns:
                styled_df = styled_df.map(color_score, subset=[col])
        if 'Squeeze' in filtered_df.columns:
            styled_df = styled_df.applymap(color_squeeze, subset=['Squeeze'])
        if 'AI Prob' in filtered_df.columns:
            styled_df = styled_df.applymap(color_ai_prob, subset=['AI Prob'])
    except AttributeError:
        # Fall back to older pandas API
        if 'Signal' in filtered_df.columns:
            styled_df = styled_df.applymap(color_signal, subset=['Signal'])
        for col in ['OptFlow', 'Sentiment', 'Fundamental', 'Technical']:
            if col in filtered_df.columns:
                styled_df = styled_df.applymap(color_score, subset=[col])
        if 'Squeeze' in filtered_df.columns:
            styled_df = styled_df.applymap(color_squeeze, subset=['Squeeze'])
        if 'AI Prob' in filtered_df.columns:
            styled_df = styled_df.applymap(color_ai_prob, subset=['AI Prob'])

        # Display table with row selection
    # Display table with row selection
    st.dataframe(styled_df, width='stretch', height=400)

    # Table stats and legend
    ai_filter_text = ""
    if 'AI Prob' in display_df.columns and ai_filter != "All Signals":
        ai_filter_text = f" | 🤖 {ai_filter}"
    st.caption(f"📊 Showing {len(filtered_df)} of {len(display_df)} tickers{ai_filter_text} | Earnings: ✓=reported | OptFlow: 50=neutral")

    # Show AI strategy counts
    if 'AI Prob' in display_df.columns:
        prob_55 = len(display_df[display_df['AI Prob'] >= 0.55])
        prob_60 = len(display_df[display_df['AI Prob'] >= 0.60])
        prob_65 = len(display_df[display_df['AI Prob'] >= 0.65])
        st.caption(f"🤖 AI Signals Today: Probability(≥55%): {prob_55} | Conservative(≥60%): {prob_60} | High Conviction(≥65%): {prob_65}")

    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "signals.csv", "text/csv", key="download_signals_csv")

    # =========================================================================
    # DEEP DIVE SECTION (if stock selected)
    # =========================================================================
    if selected:
        st.markdown("---")
        st.markdown(f"## 🔍 Deep Dive: {selected}")
        _render_stock_deep_dive(selected)


def _render_stock_deep_dive(ticker: str):
    """Render detailed analysis for a selected stock - uses full deep dive."""

    try:
        from src.core import get_signal_engine
        engine = get_signal_engine()
        signal = engine.generate_signal(ticker)
    except Exception as e:
        st.error(f"Error loading signal: {e}")
        return

    if not signal:
        st.warning(f"No signal data for {ticker}")
        return

    # Use the full deep dive renderer
    _render_deep_dive(signal)


def _render_ticker_news(ticker: str):
    """Render recent news for a ticker."""
    try:
        from src.db.connection import get_engine
        engine = get_engine()

        news_df = pd.read_sql(f"""
            SELECT headline, source, url, ai_sentiment_fast, published_at, created_at
            FROM news_articles
            WHERE ticker = '{ticker}'
            ORDER BY COALESCE(published_at, created_at) DESC
            LIMIT 8
        """, engine)

        if news_df.empty:
            st.caption("No recent news")
            return

        _display_news_items(news_df, max_items=8)

    except Exception as e:
        st.caption(f"Could not load news: {e}")
# ANALYSIS PANEL
# =============================================================================

def _render_analysis_panel():
    """Render analysis panel."""

    st.markdown("### 🚀 Full Analysis")

    # Get ALL tickers (no limit)
    tickers = _get_tickers()
    if not tickers:
        st.warning("No tickers. Add in Universe tab.")
        return

    job = _get_job_status()

    # Controls Row 1
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    # Check if we can resume
    can_resume = job['status'] in ('stopped', 'idle') and job['processed_count'] > 0

    with col1:
        btn_label = "▶️ Resume" if can_resume else "▶️ Start"
        if st.button(btn_label, disabled=(job['status'] == 'running'), width='stretch'):
            _start_job(len(tickers))
            st.rerun()

    with col2:
        if st.button("⏹️ Stop", disabled=(job['status'] != 'running'), width='stretch'):
            _stop_job()
            st.rerun()

    with col3:
        if st.button("🔄 Reset", width='stretch', help="Clear all progress and start fresh"):
            _reset_job()
            st.rerun()

    with col4:
        if job['status'] == 'running':
            st.info(f"🔄 {job['processed_count']}/{job['total_count']}")
        elif job['status'] == 'completed':
            st.success(f"✅ {job['processed_count']}/{job['total_count']}")
        elif job['status'] == 'stopped':
            if job['processed_count'] > 0:
                st.warning(f"⏸️ Paused {job['processed_count']}/{job['total_count']} - Click Resume")
            else:
                st.warning(f"⏹️ Stopped")
        else:
            if job['processed_count'] > 0:
                st.info(f"⏸️ {job['processed_count']}/{job['total_count']} - Click Resume")
            else:
                st.caption(f"Ready ({len(tickers)})")

    # Controls Row 2 - Options
    col_opt1, col_opt2, col_opt3 = st.columns([1, 1, 2])

    with col_opt1:
        skip_analyzed = st.checkbox(
            "⏭️ Skip already analyzed today",
            value=st.session_state.get('skip_analyzed_today', True),
            help="Skip tickers that already have today's data in the database. Uncheck to force re-analyze everything.",
            disabled=(job['status'] == 'running')
        )
        st.session_state['skip_analyzed_today'] = skip_analyzed

    with col_opt2:
        force_news = st.checkbox(
            "🔄 Force fresh news",
            value=st.session_state.get('force_fresh_news', False),
            help="Force fetch new articles even if cache is fresh. Only applies to tickers being analyzed.",
            disabled=(job['status'] == 'running')
        )
        st.session_state['force_fresh_news'] = force_news

    st.markdown("---")

    # Log
    col_log, col_stats = st.columns([2, 1])

    with col_log:
        st.markdown("#### 📋 Log")
        log_df = _get_job_log()
        if log_df.empty:
            st.caption("No activity")
        else:
            display = []
            for _, r in log_df.iterrows():
                sent = r.get('sentiment_score')
                opt = r.get('options_score')
                status_icon = r['status']
                display.append({
                    '': status_icon,
                    'Ticker': r['ticker'],
                    'News': r['news_count'],
                    'Sent': f"{'🟢' if sent and sent >= 60 else '🔴' if sent and sent <= 40 else '🟡'}{int(sent)}" if pd.notna(
                        sent) else '-',
                    'Opts': f"{'🟢' if opt and opt >= 60 else '🔴' if opt and opt <= 40 else '🟡'}{int(opt)}" if pd.notna(
                        opt) else '-',
                    'Time': r['time']
                })
            st.dataframe(pd.DataFrame(display), width='stretch', hide_index=True, height=400)

    with col_stats:
        st.markdown("#### 📊 Stats")
        # Use full stats query (not limited to 30 like log display)
        stats = _get_job_stats()
        st.metric("✅ Analyzed", stats['analyzed'])
        st.metric("⏭️ Skipped", stats['skipped'])
        st.metric("⚠️ Failed", stats['failed'])
        st.metric("📰 News", stats['news_total'])

    # Process next if running
    if job['status'] == 'running':
        skip_flag = st.session_state.get('skip_analyzed_today', True)
        force_news_flag = st.session_state.get('force_fresh_news', False)
        _process_next(tickers, skip_if_analyzed_today=skip_flag, force_fresh_news=force_news_flag)


def _process_next(tickers: list, skip_if_analyzed_today: bool = True, force_fresh_news: bool = False):
    """
    Process next ticker with FULL analysis - runs everything:
    1. Fresh news collection
    2. Sentiment analysis
    3. Options flow, fundamentals, technical scores
    4. Create/update TODAY's row in screener_scores
    5. Update sentiment_scores table
    6. Regenerate signal with committee

    Args:
        tickers: List of tickers to process
        skip_if_analyzed_today: If True, skip tickers that already have today's row in screener_scores
        force_fresh_news: If True, force fetch new articles even if cache is fresh
    """
    from datetime import date as dt_date
    from src.db.connection import get_connection, get_engine

    today = dt_date.today()

    # Check job status first - if processed_count >= total_count, we're done
    job = _get_job_status()
    if job['total_count'] > 0 and job['processed_count'] >= job['total_count']:
        _complete_job()
        st.success("✅ All done!")
        st.balloons()
        time.sleep(2)  # Let user see the balloons
        st.rerun()  # Refresh to show signals table
        return

    processed = _get_processed_tickers()

    # Safety check: if we've processed more than the tickers list, something is off
    # This can happen if the log table has entries from a previous run
    if len(processed) >= len(tickers):
        _complete_job()
        st.success("✅ All done!")
        st.balloons()
        time.sleep(2)  # Let user see the balloons
        st.rerun()  # Refresh to show signals table
        return

    # Find next ticker to process (that hasn't been processed in this job)
    next_ticker = None
    for t in tickers:
        if t not in processed:
            next_ticker = t
            break

    if not next_ticker:
        _complete_job()
        st.success("✅ All done!")
        st.balloons()
        time.sleep(2)  # Let user see the balloons
        st.rerun()  # Refresh to show signals table
        return

    # Check if this ticker already has TODAY's data in screener_scores
    if skip_if_analyzed_today:
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT sentiment_score, options_flow_score, fundamental_score, technical_score
                        FROM screener_scores 
                        WHERE ticker = %s AND date = %s
                    """, (next_ticker, today))
                    existing = cur.fetchone()

                    if existing and existing[0] is not None:
                        # Already has today's data - skip but log it
                        sent, opts, fund, tech = existing
                        logger.info(f"{next_ticker}: SKIPPED - already has today's data (Sent:{sent} Opts:{opts} Fund:{fund} Tech:{tech})")
                        _log_result(next_ticker, '⏭️', 0, sent, opts, fund, tech, 'SKIPPED')
                        st.rerun()
                        return
        except Exception as e:
            logger.warning(f"{next_ticker}: Could not check existing data - {e}")

    # Progress display (clamp to valid range 0.0-1.0)
    progress_val = min(1.0, max(0.0, len(processed) / len(tickers))) if tickers else 0.0
    st.progress(progress_val)
    st.info(f"🔄 Processing **{next_ticker}** ({len(processed) + 1}/{len(tickers)})")

    # =========================================================================
    # CRITICAL: Pre-log ticker as "in progress" BEFORE processing
    # This prevents infinite loops if processing fails
    # =========================================================================
    try:
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            # Check if already in log
            result = conn.execute(text("SELECT 1 FROM analysis_job_log WHERE ticker = :t"), {'t': next_ticker})
            if not result.fetchone():
                # Insert placeholder - will be updated when done
                conn.execute(text("""
                    INSERT INTO analysis_job_log (ticker, status, news_count)
                    VALUES (:t, '🔄', 0)
                """), {'t': next_ticker})
                conn.execute(text("UPDATE analysis_job SET processed_count=processed_count+1, updated_at=NOW()"))
                conn.commit()
                logger.info(f"{next_ticker}: Pre-logged as in-progress")
    except Exception as e:
        logger.warning(f"{next_ticker}: Pre-log failed: {e}")
        # If pre-log fails, we should still continue but log the error

    status = '✅'
    news_count = 0
    sentiment = None
    options = None
    fundamental = None
    technical = None
    committee_verdict = None
    today = dt_date.today()

    try:
        # =====================================================================
        # STEP 1: Collect Fresh News
        # =====================================================================
        from src.data.news import NewsCollector
        nc = NewsCollector()
        result = nc.collect_and_save(next_ticker, days_back=7, force_refresh=force_fresh_news)
        articles = result.get('articles', [])
        news_count = len(articles)

        # =====================================================================
        # STEP 2: Analyze Sentiment
        # =====================================================================
        sentiment_data = {}
        if articles:
            from src.screener.sentiment import SentimentAnalyzer
            sa = SentimentAnalyzer()
            sentiment_data = sa.analyze_ticker_sentiment(next_ticker, articles)
            sentiment = sentiment_data.get('sentiment_score')  # None if not present
        else:
            sentiment = None  # FIX: Don't use 50 as default - let it be None to indicate missing data

        # =====================================================================
        # STEP 3: Get Options Flow & Squeeze Scores (from UniverseScorer)
        # =====================================================================
        squeeze = None
        growth = None
        dividend = None

        try:
            from src.analytics.universe_scorer import UniverseScorer
            scorer = UniverseScorer(skip_ibkr=True)
            scores_list, _ = scorer.score_and_save_universe(tickers=[next_ticker], max_workers=1)

            # FIX: scores_list is a List[UniverseScores], not a dict!
            # Convert to dict for easier access
            if scores_list:
                for score_obj in scores_list:
                    if score_obj.ticker == next_ticker:
                        # UniverseScores only has: options_flow_score, short_squeeze_score
                        # It does NOT have fundamental/technical/growth/dividend
                        options = score_obj.options_flow_score
                        squeeze = score_obj.short_squeeze_score
                        break
        except Exception as e:
            logger.warning(f"{next_ticker}: Universe scorer error - {e}")

        # =====================================================================
        # STEP 3b: Get Fundamental Score (from database)
        # =====================================================================
        try:
            # FundamentalAnalyzer not implemented - scores come from screener
            pass
        except Exception as e:
            logger.debug(f"{next_ticker}: Fundamental analysis skipped - {e}")

        # =====================================================================
        # STEP 3c: Get Technical Score (from TechnicalAnalyzer)
        # =====================================================================
        try:
            from src.analytics.technical_analysis import TechnicalAnalyzer
            ta = TechnicalAnalyzer()
            tech_result = ta.analyze_ticker(next_ticker)
            if tech_result:
                technical = tech_result.get('technical_score', tech_result.get('score'))
        except Exception as e:
            logger.debug(f"{next_ticker}: Technical analysis skipped - {e}")

        # =====================================================================
        # STEP 4: Update sentiment_scores table
        # =====================================================================
        # Determine sentiment class
        # BUG FIX: Guard against None to prevent TypeError crash
        if sentiment is None:
            sentiment_class = 'Unknown'
        elif sentiment >= 70:
            sentiment_class = 'Very Bullish'
        elif sentiment >= 55:
            sentiment_class = 'Bullish'
        elif sentiment >= 45:
            sentiment_class = 'Neutral'
        elif sentiment >= 30:
            sentiment_class = 'Bearish'
        else:
            sentiment_class = 'Very Bearish'

        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO sentiment_scores
                        (ticker, date, sentiment_raw, sentiment_weighted, ai_sentiment_fast,
                         article_count, relevant_article_count, sentiment_class)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) 
                        ON CONFLICT (ticker, date) DO UPDATE SET
                            sentiment_raw = EXCLUDED.sentiment_raw,
                            sentiment_weighted = EXCLUDED.sentiment_weighted,
                            ai_sentiment_fast = EXCLUDED.ai_sentiment_fast,
                            article_count = EXCLUDED.article_count,
                            relevant_article_count = EXCLUDED.relevant_article_count,
                            sentiment_class = EXCLUDED.sentiment_class
                    """, (
                        next_ticker,
                        today,
                        _to_native(sentiment),
                        _to_native(sentiment_data.get('sentiment_weighted', sentiment)),
                        _to_native(sentiment),
                        _to_native(news_count),
                        _to_native(sentiment_data.get('relevant_count', news_count)),
                        sentiment_class
                    ))
                conn.commit()
        except Exception as e:
            logger.warning(f"{next_ticker}: sentiment_scores update error - {e}")

        # =====================================================================
        # STEP 5: Create/Update TODAY's row in screener_scores
        # =====================================================================
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    # Check if today's row exists
                    cur.execute("""
                        SELECT COUNT(*) FROM screener_scores 
                        WHERE ticker = %s AND date = %s
                    """, (next_ticker, today))
                    today_exists = cur.fetchone()[0] > 0

                    if today_exists:
                        # Update existing row
                        cur.execute("""
                            UPDATE screener_scores SET
                                sentiment_score = %s,
                                article_count = %s,
                                sentiment_weighted = %s,
                                options_flow_score = COALESCE(%s, options_flow_score),
                                short_squeeze_score = COALESCE(%s, short_squeeze_score),
                                fundamental_score = COALESCE(%s, fundamental_score),
                                technical_score = COALESCE(%s, technical_score),
                                growth_score = COALESCE(%s, growth_score),
                                dividend_score = COALESCE(%s, dividend_score)
                            WHERE ticker = %s AND date = %s
                        """, (
                            _to_native(sentiment), _to_native(news_count), _to_native(sentiment_data.get('sentiment_weighted', sentiment)),
                            _to_native(options), _to_native(squeeze), _to_native(fundamental), _to_native(technical), _to_native(growth), _to_native(dividend),
                            next_ticker, today
                        ))
                    else:
                        # Create new row for today by copying from most recent
                        cur.execute("""
                            SELECT * FROM screener_scores 
                            WHERE ticker = %s 
                            ORDER BY date DESC LIMIT 1
                        """, (next_ticker,))

                        existing_row = cur.fetchone()

                        if existing_row:
                            col_names = [desc[0] for desc in cur.description]
                            existing_data = dict(zip(col_names, existing_row))

                            # FIX Issue #4: Compute total_score fresh instead of copying stale value
                            # Use today's scores where available, fall back to existing only if needed
                            # FIX: Use None instead of 50 to indicate missing data
                            final_sentiment = sentiment if sentiment is not None else existing_data.get('sentiment_score')
                            final_fundamental = fundamental if fundamental is not None else existing_data.get('fundamental_score')
                            final_technical = technical if technical is not None else existing_data.get('technical_score')
                            final_options = options if options is not None else existing_data.get('options_flow_score')
                            final_squeeze = squeeze if squeeze is not None else existing_data.get('short_squeeze_score')
                            final_growth = growth if growth is not None else existing_data.get('growth_score')
                            final_dividend = dividend if dividend is not None else existing_data.get('dividend_score')

                            # Recompute total_score using weighted average of available scores
                            # Weights: sentiment=25%, fundamental=25%, technical=25%, options=15%, squeeze=10%
                            # FIX: Handle None values - only include scores that are available
                            available_scores = []
                            available_weights = []

                            if final_sentiment is not None:
                                available_scores.append(final_sentiment * 0.25)
                                available_weights.append(0.25)
                            if final_fundamental is not None:
                                available_scores.append(final_fundamental * 0.25)
                                available_weights.append(0.25)
                            if final_technical is not None:
                                available_scores.append(final_technical * 0.25)
                                available_weights.append(0.25)
                            if final_options is not None:
                                available_scores.append(final_options * 0.15)
                                available_weights.append(0.15)
                            if final_squeeze is not None:
                                available_scores.append(final_squeeze * 0.10)
                                available_weights.append(0.10)

                            # Calculate base_score as weighted average of available scores
                            if available_weights:
                                total_weight = sum(available_weights)
                                # BUG FIX: Use round() instead of int() to avoid systematic downward bias
                                base_score = round(sum(available_scores) / total_weight) if total_weight > 0 else None
                                if base_score is not None:
                                    base_score = max(0, min(100, base_score))  # Clamp to valid range
                            else:
                                base_score = None

                            # Apply enhanced scoring adjustments if available
                            total_score = base_score
                            if ENHANCED_SCORING_AVAILABLE:
                                try:
                                    # Build row_data for enhanced scoring
                                    enhanced_row_data = {
                                        'price': existing_data.get('price'),
                                        'target_mean': existing_data.get('target_mean'),
                                        'pe_ratio': existing_data.get('pe_ratio'),
                                        'forward_pe': existing_data.get('forward_pe'),
                                        'peg_ratio': existing_data.get('peg_ratio'),
                                        'sector': existing_data.get('sector'),
                                        'buy_count': existing_data.get('buy_count', 0),
                                        'total_ratings': existing_data.get('total_ratings', 0),
                                    }
                                    enhanced_score, adjustment, _ = get_enhanced_total_score(
                                        ticker=next_ticker,
                                        base_score=base_score,
                                        row_data=enhanced_row_data,
                                    )
                                    total_score = enhanced_score
                                    if adjustment != 0:
                                        logger.info(f"{next_ticker}: Enhanced score {base_score} → {enhanced_score} (adj: {adjustment:+d})")
                                except Exception as e:
                                    logger.debug(f"{next_ticker}: Enhanced scoring skipped: {e}")

                            # FIX Issue #5: Track data quality - count how many scores are fresh vs stale
                            fresh_count = sum([
                                1 if sentiment is not None else 0,
                                1 if fundamental is not None else 0,
                                1 if technical is not None else 0,
                                1 if options is not None else 0,
                            ])
                            if fresh_count < 2:
                                logger.warning(f"{next_ticker}: Low data quality - only {fresh_count}/4 scores are fresh")

                            cur.execute("""
                                INSERT INTO screener_scores (
                                    ticker, date, 
                                    sentiment_score, sentiment_weighted, article_count,
                                    fundamental_score, technical_score, growth_score, dividend_score,
                                    gap_score, likelihood_score, total_score,
                                    options_flow_score, short_squeeze_score, options_sentiment, squeeze_risk,
                                    created_at
                                ) VALUES (
                                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                                )
                            """, (
                                next_ticker, today,
                                _to_native(final_sentiment),
                                _to_native(sentiment_data.get('sentiment_weighted', final_sentiment)),
                                _to_native(news_count),
                                _to_native(final_fundamental),
                                _to_native(final_technical),
                                _to_native(final_growth),
                                _to_native(final_dividend),
                                _to_native(existing_data.get('gap_score', 50)),
                                _to_native(existing_data.get('likelihood_score', 50)),
                                _to_native(total_score),
                                _to_native(final_options),
                                _to_native(final_squeeze),
                                _to_native(existing_data.get('options_sentiment')),
                                _to_native(existing_data.get('squeeze_risk')),
                            ))

                            # Save enhanced scores to separate table for fast deep-dive loading
                            if ENHANCED_SCORES_DB_AVAILABLE:
                                try:
                                    compute_and_save_enhanced_scores(
                                        ticker=next_ticker,
                                        row_data=enhanced_row_data if 'enhanced_row_data' in dir() else existing_data,
                                    )
                                except Exception as e:
                                    logger.debug(f"{next_ticker}: Could not save enhanced scores: {e}")
                        else:
                            # No existing row - create new one with computed total_score
                            # FIX: Keep None values to indicate missing data, don't default to 50
                            final_sentiment = sentiment  # Keep as is (None if missing)
                            final_fundamental = fundamental
                            final_technical = technical
                            final_options = options
                            final_squeeze = squeeze
                            final_growth = growth
                            final_dividend = dividend

                            # Compute base total_score - handle None values
                            available_scores = []
                            available_weights = []

                            if final_sentiment is not None:
                                available_scores.append(final_sentiment * 0.25)
                                available_weights.append(0.25)
                            if final_fundamental is not None:
                                available_scores.append(final_fundamental * 0.25)
                                available_weights.append(0.25)
                            if final_technical is not None:
                                available_scores.append(final_technical * 0.25)
                                available_weights.append(0.25)
                            if final_options is not None:
                                available_scores.append(final_options * 0.15)
                                available_weights.append(0.15)
                            if final_squeeze is not None:
                                available_scores.append(final_squeeze * 0.10)
                                available_weights.append(0.10)

                            if available_weights:
                                total_weight = sum(available_weights)
                                # BUG FIX: Use round() instead of int() to avoid systematic downward bias
                                base_score = round(sum(available_scores) / total_weight) if total_weight > 0 else None
                                if base_score is not None:
                                    base_score = max(0, min(100, base_score))  # Clamp to valid range
                            else:
                                base_score = None

                            # Apply enhanced scoring if available (limited data for new ticker)
                            total_score = base_score
                            if ENHANCED_SCORING_AVAILABLE:
                                try:
                                    enhanced_score, adjustment, _ = get_enhanced_total_score(
                                        ticker=next_ticker,
                                        base_score=base_score,
                                        row_data={},  # No existing data
                                    )
                                    total_score = enhanced_score
                                except:
                                    pass  # Use base score if enhanced fails

                            cur.execute("""
                                INSERT INTO screener_scores (
                                    ticker, date, 
                                    sentiment_score, sentiment_weighted, article_count,
                                    fundamental_score, technical_score, growth_score, dividend_score,
                                    total_score, options_flow_score, short_squeeze_score,
                                    created_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                            """, (
                                next_ticker, today,
                                _to_native(final_sentiment),
                                _to_native(sentiment_data.get('sentiment_weighted', final_sentiment)),
                                _to_native(news_count),
                                _to_native(final_fundamental),
                                _to_native(final_technical),
                                _to_native(final_growth),
                                _to_native(final_dividend),
                                _to_native(total_score),
                                _to_native(final_options),
                                _to_native(final_squeeze),
                            ))

                    conn.commit()
                    logger.info(f"{next_ticker}: screener_scores updated for {today}")

                    # Auto-sync to historical_scores for backtesting
                    try:
                        from src.utils.historical_sync import sync_to_historical
                        sync_to_historical(next_ticker, today, {
                            'sentiment_score': final_sentiment if 'final_sentiment' in dir() else sentiment,
                            'fundamental_score': final_fundamental if 'final_fundamental' in dir() else fundamental,
                            'growth_score': final_growth if 'final_growth' in dir() else growth,
                            'dividend_score': final_dividend if 'final_dividend' in dir() else dividend,
                            'total_score': total_score,
                            'gap_score': existing_data.get('gap_score') if 'existing_data' in dir() and existing_data else None,
                            'composite_score': existing_data.get('composite_score') if 'existing_data' in dir() and existing_data else None,
                        })
                    except Exception as e:
                        logger.debug(f"{next_ticker}: historical sync skipped: {e}")
        except Exception as e:
            logger.warning(f"{next_ticker}: screener_scores update error - {e}")

        # =====================================================================
        # STEP 6: Update Earnings Calendar from yfinance
        # =====================================================================
        try:
            import yfinance as yf
            stock = yf.Ticker(next_ticker)
            ed = stock.earnings_dates

            if ed is not None and not ed.empty:
                # Get the next upcoming earnings date
                for idx in ed.index:
                    earnings_dt = idx.date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
                    if earnings_dt >= today:
                        # Found next earnings date - save to DB
                        with get_connection() as conn:
                            with conn.cursor() as cur:
                                # Try with updated_at, fall back to simpler insert
                                try:
                                    cur.execute("""
                                        INSERT INTO earnings_calendar (ticker, earnings_date, updated_at)
                                        VALUES (%s, %s, NOW())
                                        ON CONFLICT (ticker) DO UPDATE SET
                                            earnings_date = EXCLUDED.earnings_date,
                                            updated_at = NOW()
                                    """, (next_ticker, earnings_dt))
                                except Exception:
                                    # Table might not have updated_at column
                                    cur.execute("""
                                        INSERT INTO earnings_calendar (ticker, earnings_date)
                                        VALUES (%s, %s)
                                        ON CONFLICT (ticker) DO UPDATE SET
                                            earnings_date = EXCLUDED.earnings_date
                                    """, (next_ticker, earnings_dt))
                            conn.commit()
                        logger.debug(f"{next_ticker}: Updated earnings_calendar -> {earnings_dt}")
                        break
        except Exception as e:
            logger.debug(f"{next_ticker}: Could not update earnings calendar: {e}")

        # =====================================================================
        # STEP 7: Derive Signal Verdict from Total Score
        # NOTE: We skip calling generate_signal() here because it re-runs ALL
        # analysis (options, earnings, 13F) which we already computed above.
        # The verdict is simply derived from the total_score we just saved.
        # =====================================================================
        try:
            # Derive committee verdict from total_score
            if total_score is not None:
                if total_score >= 80:
                    committee_verdict = 'STRONG_BUY'
                elif total_score >= 65:
                    committee_verdict = 'BUY'
                elif total_score >= 55:
                    committee_verdict = 'WEAK_BUY'
                elif total_score <= 20:
                    committee_verdict = 'STRONG_SELL'
                elif total_score <= 35:
                    committee_verdict = 'SELL'
                elif total_score <= 45:
                    committee_verdict = 'WEAK_SELL'
                else:
                    committee_verdict = 'HOLD'
            else:
                committee_verdict = 'HOLD'

            # Save the signal to trading_signals table
            try:
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO trading_signals (ticker, date, signal_type, score, created_at)
                            VALUES (%s, %s, %s, %s, NOW())
                            ON CONFLICT (ticker, date) DO UPDATE SET
                                signal_type = EXCLUDED.signal_type,
                                score = EXCLUDED.score,
                                created_at = NOW()
                        """, (next_ticker, today, committee_verdict, total_score))
                    conn.commit()
            except Exception as e:
                logger.debug(f"{next_ticker}: trading_signals update skipped: {e}")

            # Clear SignalEngine cache so next load gets fresh data
            try:
                from src.core import get_signal_engine
                engine = get_signal_engine()
                if hasattr(engine, '_cache'):
                    for t in [next_ticker, next_ticker.upper(), next_ticker.lower()]:
                        if t in engine._cache:
                            del engine._cache[t]
            except:
                pass

            logger.info(f"{next_ticker}: Signal regenerated - Committee: {committee_verdict}")
        except Exception as e:
            logger.warning(f"{next_ticker}: Signal derivation error - {e}")
            committee_verdict = 'HOLD'

    except Exception as e:
        status = '⚠️'
        logger.error(f"Error processing {next_ticker}: {e}")

    # Log result with more details
    _log_result(next_ticker, status, news_count, sentiment, options, fundamental, technical, committee_verdict)
    st.rerun()


def _check_earnings_status(ticker: str) -> str:
    """Check if ticker is near earnings. Returns 'pre', 'post', or 'none'."""
    try:
        import yfinance as yf
        from src.db.connection import get_engine

        engine = get_engine()

        # Check database for earnings date
        df = pd.read_sql(f"""
            SELECT earnings_date FROM earnings_calendar 
            WHERE ticker = '{ticker}' 
            AND earnings_date >= CURRENT_DATE - INTERVAL '5 days'
            AND earnings_date <= CURRENT_DATE + INTERVAL '5 days'
            ORDER BY ABS(earnings_date - CURRENT_DATE) LIMIT 1
        """, engine)

        if not df.empty and df.iloc[0]['earnings_date']:
            ed = pd.to_datetime(df.iloc[0]['earnings_date']).date()
            days = (ed - date.today()).days

            if days < 0:
                return 'post'
            elif days <= 5:
                return 'pre'

        # Fallback to yfinance
        stock = yf.Ticker(ticker)
        try:
            hist = stock.earnings_history
            if hist is not None and not hist.empty:
                latest_date = pd.to_datetime(hist.index[0]).date()
                days = (latest_date - date.today()).days
                if days >= -5 and days <= 0:
                    return 'post'
        except:
            pass

    except Exception as e:
        logger.debug(f"Earnings status check error for {ticker}: {e}")

    return 'none'


def _run_earnings_aware_analysis(ticker: str, earnings_status: str) -> Dict:
    """Run analysis with earnings context."""
    from src.data.news import NewsCollector
    from src.screener.sentiment import SentimentAnalyzer
    import yfinance as yf

    result = {
        'ticker': ticker,
        'news_count': 0,
        'sentiment_score': None,
        'earnings_summary': None,
    }

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        company_name = info.get('shortName', info.get('longName', ticker))

        # Build earnings-specific search queries
        if earnings_status == 'post':
            queries = [
                f"{ticker} earnings results",
                f"{company_name} quarterly earnings",
                f"{ticker} earnings reaction",
                f"{ticker} guidance outlook",
            ]
        else:
            queries = [
                f"{ticker} earnings preview expectations",
                f"{company_name} earnings whisper",
            ]

        nc = NewsCollector()
        all_articles = []

        # Collect with earnings queries
        for query in queries:
            try:
                articles = nc.collect_ai_search(ticker, company_name=query)
                for a in articles:
                    a['ticker'] = ticker
                all_articles.extend(articles)
            except:
                pass

        # Also standard collection with force refresh
        standard = nc.collect_and_save(ticker, days_back=5, force_refresh=True)
        all_articles.extend(standard.get('articles', []))

        # Deduplicate
        seen = set()
        unique = []
        for a in all_articles:
            title = str(a.get('title', '')).lower()[:40]
            if title and title not in seen:
                seen.add(title)
                unique.append(a)

        result['news_count'] = len(unique)

        # Save and analyze sentiment
        if unique:
            nc.save_articles(unique)

            sa = SentimentAnalyzer()
            sent_result = sa.analyze_ticker_sentiment(ticker, unique)
            if sent_result:
                result['sentiment_score'] = sent_result.get('sentiment_score')

        # Get earnings result if post-earnings
        if earnings_status == 'post':
            try:
                hist = stock.earnings_history
                if hist is not None and not hist.empty:
                    latest = hist.iloc[0]
                    eps_actual = latest.get('epsActual')
                    eps_est = latest.get('epsEstimate')

                    surprise_pct = None
                    if eps_actual and eps_est and eps_est != 0:
                        surprise_pct = ((eps_actual - eps_est) / abs(eps_est)) * 100

                    # Get price reaction
                    price_hist = stock.history(period="5d")
                    reaction_pct = 0
                    if len(price_hist) >= 2:
                        reaction_pct = ((price_hist['Close'].iloc[-1] - price_hist['Close'].iloc[-2]) /
                                        price_hist['Close'].iloc[-2]) * 100

                    overall = "MISS" if (eps_actual or 0) < (eps_est or 0) else "BEAT" if (eps_actual or 0) > (
                                eps_est or 0) else "INLINE"

                    result['earnings_summary'] = {
                        'eps_actual': eps_actual,
                        'eps_estimate': eps_est,
                        'eps_surprise_pct': surprise_pct,
                        'reaction_pct': reaction_pct,
                        'overall_result': overall,
                    }
            except Exception as e:
                logger.debug(f"Earnings result error: {e}")

    except Exception as e:
        logger.error(f"Earnings-aware analysis error for {ticker}: {e}")

    return result


# =============================================================================
# SIGNALS VIEW
# =============================================================================

def _get_tickers() -> List[str]:
    """Get tickers from universe."""
    try:
        from src.db.connection import get_engine
        engine = get_engine()
        df = pd.read_sql("SELECT DISTINCT ticker FROM screener_scores ORDER BY ticker", engine)
        return df['ticker'].tolist() if not df.empty else []
    except:
        return []


def _render_signals_view(filter_type: str, sort_by: str):
    """Render signals table."""

    if 'signals_data' not in st.session_state or st.session_state.get('force_refresh'):
        with st.spinner("Loading..."):
            try:
                tickers = _get_tickers()
                engine = get_signal_engine()
                st.session_state.signals_data = engine.generate_signals_batch(tickers, max_workers=5)
                st.session_state.market_overview = get_market_overview()
                st.session_state.force_refresh = False
            except Exception as e:
                st.error(f"Error: {e}")
                return

    signals = st.session_state.signals_data
    overview = st.session_state.get('market_overview')

    # Market overview
    if overview:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            c = "🟢" if overview.spy_change >= 0 else "🔴"
            st.metric("SPY", f"{c} {overview.spy_change:+.2f}%")
        with col2:
            c = "🟢" if overview.qqq_change >= 0 else "🔴"
            st.metric("QQQ", f"{c} {overview.qqq_change:+.2f}%")
        with col3:
            st.metric("VIX", f"{overview.vix:.1f}")
        with col4:
            st.metric("Regime", overview.regime)

    st.markdown("---")

    # Table
    _render_signals_table(signals, filter_type, sort_by)


def _render_signals_table(signals: Dict[str, UnifiedSignal], filter_type: str, sort_by: str):
    """Render signals table with deep dive."""

    if not signals:
        st.warning("No signals")
        return

    signal_list = list(signals.values())

    # Filter
    if filter_type == "Stocks":
        signal_list = [s for s in signal_list if s.asset_type.value == 'stock']
    elif filter_type == "Bonds":
        signal_list = [s for s in signal_list if s.asset_type.value == 'bond']

    # Sort
    if "Today ↓" in sort_by:
        signal_list.sort(key=lambda x: x.today_score, reverse=True)
    elif "Today ↑" in sort_by:
        signal_list.sort(key=lambda x: x.today_score)
    elif "Long-term" in sort_by:
        signal_list.sort(key=lambda x: x.longterm_score, reverse=True)
    elif "Risk" in sort_by:
        signal_list.sort(key=lambda x: x.risk_score, reverse=True)

    # Table data
    table_data = []
    for s in signal_list:
        e = "🟢" if s.today_signal.value == 'bullish' else "🔴" if s.today_signal.value == 'bearish' else "🟡"
        table_data.append({
            'Ticker': s.ticker,
            'Today': f"{e} {s.today_score}%",
            'Long': f"{s.longterm_score}/100",
            'Risk': s.risk_level.value,
            'Reason': (s.signal_reason or '')[:40]
        })

    st.markdown(f"**{len(signal_list)} signals**")

    # Selector - restore ticker if we just refreshed
    tickers = [s.ticker for s in signal_list]

    # Check if we need to restore a ticker after refresh
    default_index = 0
    if '_refresh_ticker' in st.session_state:
        refresh_ticker = st.session_state.pop('_refresh_ticker')
        if refresh_ticker in tickers:
            default_index = tickers.index(refresh_ticker) + 1  # +1 because of empty string at start

    selected = st.selectbox("Details:", [""] + tickers, index=default_index, key="deep_dive")

    st.dataframe(pd.DataFrame(table_data), width='stretch', hide_index=True, height=350)

    # Deep dive
    if selected and selected in signals:
        st.markdown("---")
        _render_deep_dive(signals[selected])


def _render_deep_dive(signal: UnifiedSignal):
    """Render full deep dive with all features."""

    # Load additional data from DB
    additional_data = _load_additional_data(signal.ticker)

    # Get extended hours / current price
    extended_price_data = _get_extended_hours_price(signal.ticker)

    # Header
    c = "green" if signal.today_signal.value == 'bullish' else "red" if signal.today_signal.value == 'bearish' else "orange"
    st.markdown(f"## {signal.ticker} - {signal.company_name}")

    # PRICE BANNER - Show extended hours if available
    if extended_price_data.get('has_extended'):
        _render_price_banner(signal, extended_price_data)

    # Check for recent earnings - show alert banner
    ei = additional_data.get('earnings_intelligence')
    if ei and ei.is_post_earnings:
        _render_earnings_alert_banner(ei)

    # Signal summary row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### :{c}[{signal.today_signal.value.upper()}]")
        st.caption(f"Today: {signal.today_score}%")
    with col2:
        st.metric("Long-term", f"{signal.longterm_score}/100")
    with col3:
        risk_c = "🟢" if signal.risk_level.value == 'LOW' else "🔴" if signal.risk_level.value in ('HIGH',
                                                                                                 'EXTREME') else "🟡"
        st.metric("Risk", f"{risk_c} {signal.risk_level.value}")

    st.info(f"💡 {signal.signal_reason}")

    # =========================================================================
    # TRADE RECOMMENDATION (for BUY signals)
    # =========================================================================
    if 'BUY' in signal.today_signal.value.upper() or 'bullish' in signal.today_signal.value.lower() or signal.today_score >= 60:
        st.markdown("### 💰 Trade Recommendation")
        try:
            _render_trade_recommendation(signal)
        except Exception as e:
            st.caption(f"Trade recommendation unavailable: {e}")

    # =========================================================================
    # AI SUMMARY - Quick overview of the stock
    # =========================================================================
    st.markdown("### 📝 Quick Summary")
    try:
        _render_ai_summary(signal, additional_data)
    except Exception as e:
        st.error(f"Summary error: {e}")
        # Fallback simple summary
        st.markdown(f"""
**Overall Score: {signal.today_score}/100**

- 📰 Sentiment: {signal.sentiment_score or 'N/A'}
- 📊 Options Flow: {signal.options_score or 'N/A'}  
- 📈 Fundamental: {signal.fundamental_score or 'N/A'}
- 📉 Technical: {signal.technical_score or 'N/A'}
""")

    # Two columns layout
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Price Targets & Ratings
        st.markdown("#### 🎯 Price Targets & Ratings")
        _render_price_targets(signal, additional_data)

        # Price Chart
        st.markdown("#### 📈 Price Chart (6 Months)")
        _render_chart(signal.ticker, additional_data)

        # Component Scores
        st.markdown("#### 📊 Component Scores")
        _render_component_scores(signal)

        # Enhanced Scoring Breakdown (in expander to avoid slow page load)
        if ENHANCED_SCORING_AVAILABLE:
            with st.expander("📊 Enhanced Score Breakdown (click to load)", expanded=False):
                try:
                    render_enhancement_breakdown(signal.ticker)
                except Exception as e:
                    st.warning(f"Could not load enhancement breakdown: {e}")

        # News - with earnings filter if post-earnings
        if ei and ei.is_post_earnings:
            st.markdown("#### 📰 Earnings News & Recent Headlines")
            _render_news(signal.ticker, earnings_focus=True)
        else:
            st.markdown("#### 📰 Recent News")
            _render_news(signal.ticker, earnings_focus=False)

    with col_right:
        # Committee Decision
        st.markdown("#### 🎯 Committee Decision")
        verdict_color = "green" if 'BUY' in signal.committee_verdict else "red" if 'SELL' in signal.committee_verdict else "orange"
        st.markdown(f"**:{verdict_color}[{signal.committee_verdict}]** ({signal.committee_confidence:.0%} confidence)")
        if signal.committee_votes:
            votes_str = ", ".join([f"{k}: {v}" for k, v in signal.committee_votes.items()])
            st.caption(votes_str)

        # AI Analysis
        try:
            from src.ml.signals_tab_ai import _render_ai_analysis
            _render_ai_analysis(signal)
        except Exception as e:
            pass

        # Earnings Intelligence (IES/ECS) - shows for both pre and post earnings
        ei = additional_data.get('earnings_intelligence')
        if ei and ei.in_compute_window:
            if ei.is_post_earnings:
                st.markdown("#### 📊 Post-Earnings Analysis")
            else:
                st.markdown("#### 📊 Earnings Intelligence")
            _render_earnings_intelligence(ei)
        elif ei and ei.earnings_date:
            st.markdown("#### 📅 Earnings")
            st.caption(f"Next: {ei.earnings_date} ({ei.days_to_earnings} days)")
        elif signal.earnings_date or signal.days_to_earnings:
            st.markdown("#### 📅 Earnings")
            if signal.earnings_date:
                st.caption(f"Next: {signal.earnings_date}")
            if signal.days_to_earnings and signal.days_to_earnings < 999:
                st.caption(f"Days: {signal.days_to_earnings}")

        # Risk Assessment
        st.markdown("#### ⚠️ Risk Assessment")
        risk_color = "green" if signal.risk_level.value == 'LOW' else "red" if signal.risk_level.value in ('HIGH',
                                                                                                           'EXTREME') else "orange"
        st.markdown(f"**:{risk_color}[{signal.risk_level.value}]** (Score: {signal.risk_score})")
        if signal.risk_factors:
            for f in signal.risk_factors[:4]:
                st.caption(f"• {f}")

        # 52-Week Position
        st.markdown("#### 📍 52-Week Position")
        st.caption(f"From High: {signal.pct_from_high:+.1f}%")
        st.caption(f"From Low: {signal.pct_from_low:+.1f}%")

        # Insider Activity
        st.markdown("#### 🏦 Insider Activity")
        _render_insider_activity(additional_data)

        # Portfolio Position
        if signal.in_portfolio:
            st.markdown("#### 💼 Your Position")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Weight", f"{signal.portfolio_weight:.2%}")
                st.metric("Target", f"{signal.target_weight:.2%}")
            with col2:
                pnl_color = "green" if signal.portfolio_pnl_pct >= 0 else "red"
                st.metric("P&L", f":{pnl_color}[{signal.portfolio_pnl_pct:+.1f}%]")
                st.metric("Days Held", signal.days_held)

        # Flags
        if signal.flags:
            st.markdown("#### 🏷️ Flags")
            st.markdown(" ".join(signal.flags))

        # =====================================================================
        # PHASE 2/3: INSTITUTIONAL SIGNALS
        # =====================================================================
        if INSTITUTIONAL_SIGNALS_AVAILABLE:
            st.markdown("---")
            try:
                render_institutional_signals(
                    ticker=signal.ticker,
                    current_price=signal.current_price,
                    sector=getattr(signal, 'sector', None),
                    days_to_earnings=getattr(signal, 'days_to_earnings', 999) or 999,
                )
            except Exception as e:
                st.caption(f"Institutional signals error: {e}")

        # Run Analysis Button
        st.markdown("---")
        if st.button(f"🔄 Refresh {signal.ticker} Data", width='stretch'):
            _run_single_analysis(signal.ticker)

    # =========================================================================
    # FULL-WIDTH POST-EARNINGS REACTION ANALYSIS (Always available)
    # =========================================================================
    st.markdown("---")

    # Header with refresh button
    col_header, col_refresh = st.columns([4, 1])
    with col_header:
        st.markdown("## 📊 Post-Earnings Reaction Analysis")
    with col_refresh:
        if st.button("🔄 Refresh", key=f"refresh_reaction_{signal.ticker}", help="Force fresh analysis"):
            # Clear cache for this ticker
            cache_key = f"reaction_cache_{signal.ticker}"
            if cache_key in st.session_state:
                del st.session_state[cache_key]
            st.rerun()

    # Check if analysis is available
    if REACTION_ANALYZER_AVAILABLE:
        # Check for recent earnings using yfinance directly
        has_recent_earnings = False
        days_since_earnings = None

        try:
            import yfinance as yf
            from datetime import date
            stock = yf.Ticker(signal.ticker)
            ed = stock.earnings_dates

            if ed is not None and not ed.empty:
                today = date.today()
                for idx in ed.index:
                    try:
                        earnings_date = idx.date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
                        days_diff = (today - earnings_date).days
                        if 0 <= days_diff <= 10:  # Within 10 days
                            has_recent_earnings = True
                            days_since_earnings = days_diff
                            break
                    except:
                        continue
        except Exception as e:
            logger.debug(f"Earnings date check error: {e}")

        if has_recent_earnings:
            st.caption(f"📅 Earnings reported {days_since_earnings} day(s) ago")
            _render_full_reaction_analysis_standalone(signal.ticker)
        else:
            st.info("No recent earnings detected (within 10 days)")
            if st.button("🔍 Run Analysis Anyway", key=f"force_reaction_{signal.ticker}"):
                _render_full_reaction_analysis_standalone(signal.ticker)
    else:
        st.warning(
            "Reaction analyzer not available. Install reaction_analyzer.py in src/analytics/earnings_intelligence/")

    # =========================================================================
    # SEC FILING INSIGHTS SECTION
    # =========================================================================
    st.markdown("---")
    if SEC_INSIGHTS_AVAILABLE:
        with st.expander("📄 SEC Filing Insights", expanded=True):
            try:
                insights = get_filing_insights(signal.ticker)
                if insights.get('available'):
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        st.metric("Filing Score", f"{insights.get('score', 0):.0f}/100")
                    with col2:
                        st.metric("Rating", insights.get('score_label', 'Unknown'))
                    with col3:
                        quality = insights.get('data_quality', {})
                        st.caption(f"📅 {quality.get('freshness_days', 0)} days ago | {quality.get('filings_analyzed', 0)} filings")

                    # Factors
                    factors = insights.get('factors', {})
                    cols = st.columns(5)
                    for col, (key, label) in zip(cols, [('guidance', 'Guidance'), ('risk', 'Risk'), ('litigation', 'Litigation'), ('china', 'China'), ('ai_demand', 'AI')]):
                        with col:
                            st.metric(label, f"{factors.get(key, {}).get('score', 50):.0f}")

                    # Signals
                    col_b, col_r = st.columns(2)
                    with col_b:
                        for s in insights.get('bullish_signals', [])[:3]:
                            st.markdown(f"✅ {s}")
                    with col_r:
                        for s in insights.get('bearish_signals', [])[:3]:
                            st.markdown(f"⚠️ {s}")
                else:
                    st.info(f"No SEC filing data for {signal.ticker}. Run `python -m src.rag.sec_ingestion`")
            except Exception as e:
                st.error(f"SEC insights error: {e}")
    else:
        with st.expander("📄 SEC Filing Insights", expanded=False):
            st.info("SEC Filing Insights module not available")

    # =========================================================================
    # DUAL ANALYST SECTION
    # =========================================================================
    st.markdown("---")
    if DUAL_ANALYST_AVAILABLE:
        with st.expander("🔬 AI Dual Analysis (SQL + RAG)", expanded=False):
            cache_key = f"dual_signals_{signal.ticker}"

            if st.button("🚀 Run Dual Analysis", key=f"dual_btn_{signal.ticker}"):
                with st.spinner("Running dual analysis (15-30s)..."):
                    try:
                        service = DualAnalystService()
                        result = service.analyze_for_display(signal.ticker)
                        st.session_state[cache_key] = result
                    except Exception as e:
                        st.error(f"Failed: {e}")

            if cache_key in st.session_state:
                r = st.session_state[cache_key]

                # Agreement
                agreement = r.get('evaluation', {}).get('agreement_score', 0)
                st.progress(agreement, text=f"Agreement: {agreement:.0%}")

                # Two analysts
                col1, col2 = st.columns(2)
                ICONS = {"very_bullish": "🚀", "bullish": "📈", "neutral": "➡️", "bearish": "📉", "very_bearish": "🔻", "unknown": "❓"}

                with col1:
                    sql = r.get('sql_analyst', {})
                    icon = ICONS.get(sql.get('sentiment', 'unknown'), "❓")
                    st.markdown(f"**📊 Quant:** {icon} {sql.get('sentiment', 'unknown').replace('_', ' ').title()}")
                    st.caption(sql.get('summary', '')[:120])

                with col2:
                    rag = r.get('rag_analyst', {})
                    icon = ICONS.get(rag.get('sentiment', 'unknown'), "❓")
                    st.markdown(f"**📄 Qual:** {icon} {rag.get('sentiment', 'unknown').replace('_', ' ').title()}")
                    st.caption(rag.get('summary', '')[:120])

                # Synthesis
                syn = r.get('synthesis', {})
                icon = ICONS.get(syn.get('sentiment', 'unknown'), "❓")
                st.markdown(f"**🎯 Verdict:** {icon} {syn.get('sentiment', 'unknown').replace('_', ' ').title()} ({syn.get('confidence', 0):.0%})")
    else:
        with st.expander("🔬 AI Dual Analysis (SQL + RAG)", expanded=False):
            st.info("Dual Analyst Service not available")

    # AI Chat Section (full width)
    st.markdown("---")
    _render_ai_chat(signal, additional_data)


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

        # FIX: yfinance previousClose is often stale during weekends/holidays
        # Determine the best baseline (last regular market close):
        # - If we have pre/post market, currentPrice IS the last close, use it as baseline
        # - If no extended hours, currentPrice is the current trading price
        # - Fallback to previousClose if currentPrice is unavailable

        if pre_price > 0 or post_price > 0:
            # Extended hours active - currentPrice should be last regular close
            baseline = current_price if current_price > 0 else yf_prev_close
        else:
            # No extended hours - use yfinance previousClose
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


def _render_price_banner(signal: UnifiedSignal, extended_data: dict):
    """Render prominent price banner with extended hours info."""

    session = extended_data.get('session', '')
    ext_price = extended_data.get('extended_price', 0)
    ext_change = extended_data.get('extended_change', 0)
    ext_change_pct = extended_data.get('extended_change_pct', 0)
    prev_close = extended_data.get('regular_close', 0)

    # Determine color based on change
    if ext_change_pct <= -5:
        color = "red"
        emoji = "🔴📉"
        severity = "MAJOR LOSS"
    elif ext_change_pct <= -2:
        color = "red"
        emoji = "🔴"
        severity = "Down"
    elif ext_change_pct >= 5:
        color = "green"
        emoji = "🟢📈"
        severity = "MAJOR GAIN"
    elif ext_change_pct >= 2:
        color = "green"
        emoji = "🟢"
        severity = "Up"
    else:
        color = "orange"
        emoji = "🟡"
        severity = "Flat"

    # Create banner
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        st.markdown(f"**{session}:** ${ext_price:.2f}")

    with col2:
        st.markdown(f"**{emoji} {ext_change_pct:+.2f}%** ({severity})")

    with col3:
        st.markdown(f"Prev Close: ${prev_close:.2f}")

    # Big alert for major moves
    if abs(ext_change_pct) >= 5:
        if ext_change_pct < 0:
            st.error(f"🚨 **{session.upper()}: {signal.ticker} DOWN {abs(ext_change_pct):.1f}%** (${ext_change:+.2f})")
        else:
            st.success(f"🚀 **{session.upper()}: {signal.ticker} UP {ext_change_pct:.1f}%** (${ext_change:+.2f})")


def _render_earnings_alert_banner(ei):
    """Render prominent earnings alert banner."""

    days_since = abs(ei.days_to_earnings)

    # ECS color
    ecs_val = ei.ecs_category.value if hasattr(ei.ecs_category, 'value') else str(ei.ecs_category)

    if ecs_val in ('STRONG_MISS', 'MISS'):
        st.error(f"""
        🚨 **EARNINGS ALERT** - Reported {days_since} day(s) ago

        **Result:** {ecs_val.replace('_', ' ')} | **Reaction:** {ei.total_reaction_pct:+.1f}%
        """)
    elif ecs_val in ('STRONG_BEAT', 'BEAT'):
        st.success(f"""
        📊 **EARNINGS** - Reported {days_since} day(s) ago

        **Result:** {ecs_val.replace('_', ' ')} | **Reaction:** {ei.total_reaction_pct:+.1f}%
        """)
    else:
        st.info(f"""
        📊 **EARNINGS** - Reported {days_since} day(s) ago

        **Result:** {ecs_val.replace('_', ' ')} | **Reaction:** {ei.total_reaction_pct:+.1f}%
        """)


def _load_additional_data(ticker: str) -> dict:
    """Load additional data from database including earnings intelligence and current price."""
    data = {}
    try:
        from src.db.connection import get_engine
        engine = get_engine()

        # Get current/extended price FIRST - this is critical
        try:
            ext_price = _get_extended_hours_price(ticker)
            data['extended_price_data'] = ext_price
            data['current_price'] = ext_price.get('extended_price') or ext_price.get('regular_close', 0)
        except:
            pass

        # Price Targets
        try:
            df = pd.read_sql(f"""
                SELECT target_mean, target_high, target_low, target_upside_pct, analyst_count
                FROM price_targets WHERE ticker = '{ticker}' ORDER BY date DESC LIMIT 1
            """, engine)
            if not df.empty:
                row = df.iloc[0]
                data['target_mean'] = row.get('target_mean')
                data['target_high'] = row.get('target_high')
                data['target_low'] = row.get('target_low')
                data['target_upside'] = row.get('target_upside_pct')
                data['analyst_count'] = row.get('analyst_count')
        except:
            pass

        # Analyst Ratings
        try:
            df = pd.read_sql(f"""
                SELECT strong_buy, buy, hold, sell, strong_sell, consensus
                FROM analyst_ratings WHERE ticker = '{ticker}' ORDER BY date DESC LIMIT 1
            """, engine)
            if not df.empty:
                row = df.iloc[0]
                data['strong_buy'] = row.get('strong_buy', 0)
                data['buy'] = row.get('buy', 0)
                data['hold'] = row.get('hold', 0)
                data['sell'] = row.get('sell', 0)
                data['strong_sell'] = row.get('strong_sell', 0)
                data['consensus'] = row.get('consensus', '')
        except:
            pass

        # Insider Transactions
        try:
            df = pd.read_sql(f"""
                SELECT insider_name, transaction_type, shares_transacted, total_value, transaction_date
                FROM insider_transactions WHERE ticker = '{ticker}' 
                ORDER BY transaction_date DESC LIMIT 10
            """, engine)
            data['insider_transactions'] = df.to_dict('records') if not df.empty else []
        except:
            data['insider_transactions'] = []

        # Earnings Intelligence (IES/ECS) - with proper post-earnings handling
        try:
            from src.analytics.earnings_intelligence import enrich_screener_with_earnings
            ei = enrich_screener_with_earnings(ticker)
            data['earnings_intelligence'] = ei

            # If post-earnings, get actual results
            if ei and ei.is_post_earnings:
                import yfinance as yf
                stock = yf.Ticker(ticker)

                try:
                    hist = stock.earnings_history
                    if hist is not None and not hist.empty:
                        latest = hist.iloc[0]
                        data['earnings_actual'] = {
                            'eps_actual': latest.get('epsActual'),
                            'eps_estimate': latest.get('epsEstimate'),
                            'surprise_pct': None,
                        }
                        if data['earnings_actual']['eps_actual'] and data['earnings_actual']['eps_estimate']:
                            est = data['earnings_actual']['eps_estimate']
                            if est != 0:
                                data['earnings_actual']['surprise_pct'] = (
                                                                                  (data['earnings_actual'][
                                                                                       'eps_actual'] - est) / abs(est)
                                                                          ) * 100
                except:
                    pass

        except ImportError:
            # Try alternate path
            try:
                import sys
                sys.path.insert(0, '/mnt/user-data/outputs/signal_hub')
                from src.analytics.earnings_intelligence import enrich_screener_with_earnings
                ei = enrich_screener_with_earnings(ticker)
                data['earnings_intelligence'] = ei
            except:
                data['earnings_intelligence'] = None
        except Exception as e:
            logger.debug(f"EI error for {ticker}: {e}")
            data['earnings_intelligence'] = None

    except Exception as e:
        logger.error(f"Error loading additional data: {e}")

    return data


def _render_trade_recommendation(signal: UnifiedSignal):
    """Render trade recommendation with entry, stop loss, and target prices."""

    current_price = getattr(signal, 'current_price', 0) or 0
    if current_price <= 0:
        st.caption("Price not available for trade calculation")
        return

    # Get risk level safely
    risk_level = 'MEDIUM'
    if hasattr(signal, 'risk_level'):
        risk_level = signal.risk_level.value if hasattr(signal.risk_level, 'value') else str(signal.risk_level)

    # Get today score safely - FIX: use is None check (0 is a valid score)
    _today_score = getattr(signal, 'today_score', None)
    today_score = _today_score if _today_score is not None else 50

    # Calculate trade levels based on signal strength
    # Entry price - current price or slight pullback for strong signals
    if today_score >= 80:
        entry_price = current_price
        entry_note = "Enter at market"
    elif today_score >= 65:
        entry_price = current_price * 0.98  # 2% pullback
        entry_note = "Enter on 2% pullback"
    else:
        entry_price = current_price * 0.95  # 5% pullback
        entry_note = "Enter on 5% pullback"

    # Stop loss - based on risk level
    risk_multiplier = {
        'LOW': 5.0,
        'MEDIUM': 7.0,
        'HIGH': 10.0,
        'EXTREME': 12.0
    }.get(risk_level, 7.0)

    stop_loss = entry_price * (1 - risk_multiplier / 100)
    stop_pct = ((stop_loss - entry_price) / entry_price) * 100

    # Target price - use analyst target if available, otherwise calculate
    analyst_target = getattr(signal, 'target_price', None)
    if analyst_target and analyst_target > current_price:
        target_price = analyst_target
        target_note = "Analyst consensus"
    else:
        # Calculate based on 2.5:1 R/R
        risk_amount = entry_price - stop_loss
        target_price = entry_price + (risk_amount * 2.5)
        target_note = "2.5:1 R/R target"

    target_pct = ((target_price - entry_price) / entry_price) * 100

    # Display trade levels
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📍 Entry", f"${entry_price:.2f}")
        st.caption(entry_note)

    with col2:
        st.metric("🛑 Stop Loss", f"${stop_loss:.2f}", f"{stop_pct:.1f}%")

    with col3:
        st.metric("🎯 Target", f"${target_price:.2f}", f"+{target_pct:.1f}%")
        st.caption(target_note)

    with col4:
        rr_ratio = abs(target_pct / stop_pct) if stop_pct != 0 else 0
        rr_color = "🟢" if rr_ratio >= 2 else "🟡" if rr_ratio >= 1.5 else "🔴"
        st.metric("R/R Ratio", f"{rr_color} {rr_ratio:.1f}:1")


def _render_ai_summary(signal: UnifiedSignal, additional_data: dict):
    """Render a quick AI-generated summary of the stock."""

    # Build summary based on component scores
    summaries = []

    # Get scores safely - FIX: use is None check (0 is a valid score)
    _today = getattr(signal, 'today_score', None)
    _sent = getattr(signal, 'sentiment_score', None)
    _opts = getattr(signal, 'options_score', None)
    _fund = getattr(signal, 'fundamental_score', None)
    _tech = getattr(signal, 'technical_score', None)
    _squeeze = getattr(signal, 'squeeze_score', None)

    today_score = _today if _today is not None else 50
    sent = _sent if _sent is not None else 50
    opts = _opts if _opts is not None else 50
    fund = _fund if _fund is not None else 50
    tech = _tech if _tech is not None else 50
    squeeze = _squeeze if _squeeze is not None else 50

    # Overall signal
    if today_score >= 70:
        overall = "🟢 **Strong opportunity** - Multiple factors align positively."
    elif today_score >= 55:
        overall = "🟡 **Moderate opportunity** - Some positive factors, but mixed signals."
    elif today_score >= 45:
        overall = "⚪ **Neutral** - No clear direction, wait for better setup."
    elif today_score >= 30:
        overall = "🟠 **Caution advised** - More negative factors than positive."
    else:
        overall = "🔴 **Avoid** - Multiple warning signs present."

    summaries.append(overall)

    # Sentiment analysis
    if sent >= 70:
        summaries.append(f"📰 **Sentiment: Very Bullish ({sent})** - News and social media are overwhelmingly positive.")
    elif sent >= 55:
        summaries.append(f"📰 **Sentiment: Bullish ({sent})** - More positive than negative news coverage.")
    elif sent >= 45:
        summaries.append(f"📰 **Sentiment: Neutral ({sent})** - Mixed news coverage with no clear bias.")
    elif sent >= 30:
        summaries.append(f"📰 **Sentiment: Bearish ({sent})** - Negative news outweighs positive.")
    else:
        summaries.append(f"📰 **Sentiment: Very Bearish ({sent})** - Overwhelmingly negative press.")

    # Options flow
    if opts >= 70:
        summaries.append(f"📊 **Options Flow: Bullish ({opts})** - Smart money positioning for upside. Unusual call activity.")
    elif opts >= 55:
        summaries.append(f"📊 **Options Flow: Mildly Bullish ({opts})** - Slight bullish bias in options market.")
    elif opts >= 45:
        summaries.append(f"📊 **Options Flow: Neutral ({opts})** - No significant directional bias.")
    elif opts >= 30:
        summaries.append(f"📊 **Options Flow: Bearish ({opts})** - Put buying exceeds calls.")
    else:
        summaries.append(f"📊 **Options Flow: Very Bearish ({opts})** - Heavy put buying detected.")

    # Squeeze potential
    if squeeze >= 70:
        summaries.append(f"🚀 **Short Squeeze: High Potential ({squeeze})** - High short interest + bullish catalysts!")
    elif squeeze >= 50:
        summaries.append(f"🚀 **Short Squeeze: Moderate ({squeeze})** - Some short interest present.")
    else:
        summaries.append(f"🚀 **Short Squeeze: Low ({squeeze})** - Minimal short interest.")

    # Fundamentals
    if fund >= 70:
        summaries.append(f"📈 **Fundamentals: Strong ({fund})** - Solid financials, good growth, attractive valuation.")
    elif fund >= 55:
        summaries.append(f"📈 **Fundamentals: Good ({fund})** - Decent financial health.")
    elif fund >= 45:
        summaries.append(f"📈 **Fundamentals: Fair ({fund})** - Mixed financial picture.")
    elif fund >= 30:
        summaries.append(f"📈 **Fundamentals: Weak ({fund})** - Some concerning metrics.")
    else:
        summaries.append(f"📈 **Fundamentals: Poor ({fund})** - Significant concerns.")

    # Technical
    if tech >= 70:
        summaries.append(f"📉 **Technical: Bullish ({tech})** - Strong chart, above key MAs, momentum positive.")
    elif tech >= 55:
        summaries.append(f"📉 **Technical: Mildly Bullish ({tech})** - Some positive chart setups.")
    elif tech >= 45:
        summaries.append(f"📉 **Technical: Neutral ({tech})** - Consolidating, no clear direction.")
    elif tech >= 30:
        summaries.append(f"📉 **Technical: Bearish ({tech})** - Below key support levels.")
    else:
        summaries.append(f"📉 **Technical: Very Bearish ({tech})** - Strong downtrend.")

    # Earnings context
    ei = additional_data.get('earnings_intelligence') if additional_data else None
    if ei:
        try:
            if getattr(ei, 'is_post_earnings', False):
                reaction = getattr(ei, 'reaction_category', 'Unknown')
                summaries.append(f"📅 **Recent Earnings: {reaction}** - Stock reacted to latest report.")
            elif getattr(ei, 'days_to_earnings', 999) <= 14:
                days = ei.days_to_earnings
                summaries.append(f"📅 **Earnings in {days} days** - Increased volatility expected.")
        except:
            pass

    # Display summary in a nice box
    for summary in summaries:
        st.markdown(summary)


def _render_price_targets(signal: UnifiedSignal, additional_data: dict):
    """Render price targets section."""
    target = additional_data.get('target_mean')

    if target:
        upside = ((float(target) - signal.current_price) / signal.current_price * 100) if signal.current_price else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current", f"${signal.current_price:.2f}")
        with col2:
            color = "green" if upside > 0 else "red"
            st.metric("Target", f"${float(target):.2f}", f"{upside:+.1f}%")
        with col3:
            st.metric("Low", f"${float(additional_data.get('target_low', 0)):.2f}")
        with col4:
            st.metric("High", f"${float(additional_data.get('target_high', 0)):.2f}")

        # Analyst breakdown
        if additional_data.get('consensus'):
            cols = st.columns(5)
            labels = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']
            keys = ['strong_buy', 'buy', 'hold', 'sell', 'strong_sell']
            for i, (label, key) in enumerate(zip(labels, keys)):
                with cols[i]:
                    val = additional_data.get(key, 0) or 0
                    st.caption(f"{label}: {val}")
    else:
        st.caption("No analyst targets available")


def _render_chart(ticker: str, additional_data: dict):
    """Render Google Finance-style price chart with period selectors and key stats."""
    try:
        import yfinance as yf
        import plotly.graph_objects as go
        from datetime import datetime, timedelta

        stock = yf.Ticker(ticker)
        info = stock.info

        # Period selector
        period_options = {
            "1D": ("1d", "5m"),
            "5D": ("5d", "15m"),
            "1M": ("1mo", "1h"),
            "6M": ("6mo", "1d"),
            "YTD": ("ytd", "1d"),
            "1Y": ("1y", "1d"),
            "5Y": ("5y", "1wk"),
            "Max": ("max", "1mo"),
        }

        cols = st.columns(8)
        selected_period = st.session_state.get(f'chart_period_{ticker}', 'YTD')

        for i, period in enumerate(period_options.keys()):
            with cols[i]:
                if st.button(period, key=f"period_{ticker}_{period}",
                           width='stretch',
                           type="primary" if period == selected_period else "secondary"):
                    st.session_state[f'chart_period_{ticker}'] = period
                    selected_period = period
                    st.rerun()

        # Get data for selected period
        period_code, interval = period_options[selected_period]
        data = stock.history(period=period_code, interval=interval)

        if data.empty:
            st.caption("Chart unavailable")
            return

        # Calculate price change
        current_price = data['Close'].iloc[-1]
        start_price = data['Close'].iloc[0]
        price_change = current_price - start_price
        price_change_pct = (price_change / start_price) * 100
        is_positive = price_change >= 0

        # Color scheme
        line_color = "#00C805" if is_positive else "#FF5252"
        fill_color = "rgba(0, 200, 5, 0.1)" if is_positive else "rgba(255, 82, 82, 0.1)"

        # Price header
        st.markdown(f"""
        <div style="margin-bottom: 10px;">
            <span style="font-size: 2.2em; font-weight: bold;">{current_price:.2f}</span>
            <span style="font-size: 1em; color: #888;"> USD</span>
            <br>
            <span style="color: {line_color}; font-size: 1.1em;">
                {'+' if is_positive else ''}{price_change:.2f} ({'+' if is_positive else ''}{price_change_pct:.2f}%)
            </span>
            <span style="color: #888; font-size: 0.9em;"> {selected_period.lower()}</span>
        </div>
        """, unsafe_allow_html=True)

        # Create area chart
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Price',
            line=dict(color=line_color, width=2),
            fill='tozeroy',
            fillcolor=fill_color,
            hovertemplate='%{x}<br>$%{y:.2f}<extra></extra>'
        ))

        # Optional: Add target price line
        target = additional_data.get('target_mean')
        if target and selected_period in ['6M', 'YTD', '1Y']:
            fig.add_hline(y=float(target), line_dash="dash", line_color="#FFD700",
                          annotation_text=f"Target ${float(target):.2f}",
                          annotation_position="right")

        fig.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(
                showgrid=False,
                showticklabels=True,
                tickfont=dict(size=10, color='#888'),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.1)',
                tickfont=dict(size=10, color='#888'),
                tickprefix='$',
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            hovermode='x unified',
        )

        st.plotly_chart(fig, width='stretch')

        # Key stats grid (Google Finance style)
        _render_key_stats(stock, info, current_price)

    except Exception as e:
        st.caption(f"Chart error: {e}")


def _render_key_stats(stock, info: dict, current_price: float):
    """Render key stats in Google Finance style grid."""
    try:
        # Get values safely
        open_price = info.get('open') or info.get('regularMarketOpen', '-')
        high_price = info.get('dayHigh') or info.get('regularMarketDayHigh', '-')
        low_price = info.get('dayLow') or info.get('regularMarketDayLow', '-')
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE') or info.get('forwardPE', '-')
        high_52wk = info.get('fiftyTwoWeekHigh', '-')
        low_52wk = info.get('fiftyTwoWeekLow', '-')
        dividend_yield = info.get('dividendYield', 0)
        dividend_rate = info.get('dividendRate', 0)

        # Format market cap
        if market_cap and market_cap > 0:
            if market_cap >= 1e12:
                mkt_cap_str = f"{market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                mkt_cap_str = f"{market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                mkt_cap_str = f"{market_cap/1e6:.2f}M"
            else:
                mkt_cap_str = f"{market_cap:,.0f}"
        else:
            mkt_cap_str = "-"

        # Format dividend yield
        div_yield_str = f"{dividend_yield*100:.2f}%" if dividend_yield else "-"
        div_rate_str = f"{dividend_rate:.2f}" if dividend_rate else "-"

        # Format PE
        pe_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else str(pe_ratio)

        # Create stats grid
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div style="font-size: 0.85em;">
                <div style="color: #888;">Open</div>
                <div style="font-weight: 500;">{open_price if isinstance(open_price, str) else f'{open_price:.2f}'}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size: 0.85em; margin-top: 8px;">
                <div style="color: #888;">High</div>
                <div style="font-weight: 500;">{high_price if isinstance(high_price, str) else f'{high_price:.2f}'}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size: 0.85em; margin-top: 8px;">
                <div style="color: #888;">Low</div>
                <div style="font-weight: 500;">{low_price if isinstance(low_price, str) else f'{low_price:.2f}'}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="font-size: 0.85em;">
                <div style="color: #888;">Mkt cap</div>
                <div style="font-weight: 500;">{mkt_cap_str}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size: 0.85em; margin-top: 8px;">
                <div style="color: #888;">P/E ratio</div>
                <div style="font-weight: 500;">{pe_str}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size: 0.85em; margin-top: 8px;">
                <div style="color: #888;">52-wk high</div>
                <div style="font-weight: 500;">{high_52wk if isinstance(high_52wk, str) else f'{high_52wk:.2f}'}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style="font-size: 0.85em;">
                <div style="color: #888;">Dividend</div>
                <div style="font-weight: 500;">{div_yield_str}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size: 0.85em; margin-top: 8px;">
                <div style="color: #888;">Qtrly Div Amt</div>
                <div style="font-weight: 500;">{div_rate_str}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size: 0.85em; margin-top: 8px;">
                <div style="color: #888;">52-wk low</div>
                <div style="font-weight: 500;">{low_52wk if isinstance(low_52wk, str) else f'{low_52wk:.2f}'}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            # Price position in 52-week range
            if isinstance(high_52wk, (int, float)) and isinstance(low_52wk, (int, float)) and high_52wk > low_52wk:
                range_pct = (current_price - low_52wk) / (high_52wk - low_52wk) * 100
                range_color = "#00C805" if range_pct > 50 else "#FF5252"
                st.markdown(f"""
                <div style="font-size: 0.85em;">
                    <div style="color: #888;">52-wk range</div>
                    <div style="font-weight: 500; color: {range_color};">{range_pct:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Volume
            volume = info.get('volume', 0)
            avg_volume = info.get('averageVolume', 0)
            if volume:
                vol_str = f"{volume/1e6:.1f}M" if volume >= 1e6 else f"{volume/1e3:.0f}K"
                st.markdown(f"""
                <div style="font-size: 0.85em; margin-top: 8px;">
                    <div style="color: #888;">Volume</div>
                    <div style="font-weight: 500;">{vol_str}</div>
                </div>
                """, unsafe_allow_html=True)

            if avg_volume:
                avg_vol_str = f"{avg_volume/1e6:.1f}M" if avg_volume >= 1e6 else f"{avg_volume/1e3:.0f}K"
                st.markdown(f"""
                <div style="font-size: 0.85em; margin-top: 8px;">
                    <div style="color: #888;">Avg volume</div>
                    <div style="font-weight: 500;">{avg_vol_str}</div>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.caption(f"Stats error: {e}")


def _render_component_scores(signal: UnifiedSignal):
    """Render component scores with progress bars and detailed explanations."""

    # Score interpretation helper
    def get_score_interpretation(name: str, score: int) -> str:
        """Get detailed interpretation for each score type."""
        if name == "Technical":
            if score >= 70:
                return "Strong uptrend, above key MAs, momentum positive"
            elif score >= 55:
                return "Mild bullish setup, testing resistance"
            elif score >= 45:
                return "Consolidating, no clear direction"
            elif score >= 30:
                return "Weak, below moving averages"
            else:
                return "Strong downtrend, avoid"
        elif name == "Fundamental":
            if score >= 70:
                return "Strong financials, good growth, attractive valuation"
            elif score >= 55:
                return "Decent fundamentals, some positive metrics"
            elif score >= 45:
                return "Mixed fundamentals, fair valuation"
            elif score >= 30:
                return "Weak financials, concerning metrics"
            else:
                return "Poor fundamentals, high risk"
        elif name == "Sentiment":
            if score >= 70:
                return "Very positive news/social media sentiment"
            elif score >= 55:
                return "Bullish sentiment, positive coverage"
            elif score >= 45:
                return "Neutral sentiment, mixed coverage"
            elif score >= 30:
                return "Bearish sentiment, negative coverage"
            else:
                return "Very negative sentiment, strong concerns"
        elif name == "Options Flow":
            if score >= 70:
                return "Heavy call buying, smart money bullish"
            elif score >= 55:
                return "Slight bullish bias in options"
            elif score >= 45:
                return "Neutral options flow, no clear direction"
            elif score >= 30:
                return "Put buying exceeds calls, bearish"
            else:
                return "Heavy put buying, institutions hedging"
        elif name == "Squeeze":
            if score >= 70:
                return "HIGH squeeze potential - high SI + bullish setup"
            elif score >= 50:
                return "Moderate squeeze potential"
            else:
                return "Low squeeze risk, minimal short interest"
        elif name == "Earnings":
            if score >= 70:
                return "Strong earnings outlook, beat expected"
            elif score >= 55:
                return "Positive earnings expectations"
            elif score >= 45:
                return "Mixed earnings outlook"
            elif score >= 30:
                return "Weak earnings expected"
            else:
                return "Poor earnings outlook, miss likely"
        return ""

    components = [
        ("Technical", signal.technical_score, signal.technical_signal, signal.technical_reason),
        ("Fundamental", signal.fundamental_score, signal.fundamental_signal, signal.fundamental_reason),
        ("Sentiment", signal.sentiment_score, signal.sentiment_signal, signal.sentiment_reason),
        ("Options Flow", signal.options_score, signal.options_signal, signal.options_reason),
        ("Earnings", signal.earnings_score, signal.earnings_signal, signal.earnings_reason),
    ]

    # Add squeeze if available - FIX: use is not None check
    squeeze_score = getattr(signal, 'squeeze_score', None)
    if squeeze_score is not None:
        components.append(("Squeeze", squeeze_score, None, None))

    for name, score, sig, reason in components:
        # FIX: use is None check (0 is a valid score)
        score = score if score is not None else 50
        emoji = "🟢" if score >= 60 else "🔴" if score <= 40 else "🟡"

        col1, col2, col3 = st.columns([1.5, 2, 2])
        with col1:
            st.markdown(f"**{name}** {emoji}")
            st.caption(f"Score: {score}")
        with col2:
            st.progress(score / 100)
        with col3:
            # Show reason if available, otherwise show interpretation
            if reason:
                st.caption(reason[:80])
            else:
                interpretation = get_score_interpretation(name, score)
                st.caption(interpretation)


def _render_news(ticker: str, earnings_focus: bool = False):
    """Render recent news with sentiment. If earnings_focus, prioritize earnings news."""
    try:
        from src.db.connection import get_engine
        engine = get_engine()

        # Build query - prioritize earnings news if focused
        if earnings_focus:
            # First try to get earnings-specific news
            df_earnings = pd.read_sql(f"""
                SELECT headline, ai_sentiment_fast, url, source,
                       published_at,
                       created_at,
                       COALESCE(published_at, created_at) as article_date
                FROM news_articles 
                WHERE ticker = '{ticker}' 
                AND headline IS NOT NULL AND headline != ''
                AND (
                    LOWER(headline) LIKE '%earning%'
                    OR LOWER(headline) LIKE '%quarter%'
                    OR LOWER(headline) LIKE '%revenue%'
                    OR LOWER(headline) LIKE '%profit%'
                    OR LOWER(headline) LIKE '%eps%'
                    OR LOWER(headline) LIKE '%beat%'
                    OR LOWER(headline) LIKE '%miss%'
                    OR LOWER(headline) LIKE '%guidance%'
                    OR LOWER(headline) LIKE '%forecast%'
                    OR LOWER(headline) LIKE '%results%'
                    OR LOWER(headline) LIKE '%report%'
                )
                ORDER BY COALESCE(published_at, created_at) DESC NULLS LAST
                LIMIT 8
            """, engine)

            if not df_earnings.empty:
                st.markdown("**📊 Earnings Coverage:**")
                _display_news_items(df_earnings, max_items=5)
                st.markdown("---")

        # Get all recent news
        df = pd.read_sql(f"""
            SELECT headline, ai_sentiment_fast, url, source,
                   published_at,
                   created_at,
                   COALESCE(published_at, created_at) as article_date
            FROM news_articles 
            WHERE ticker = '{ticker}' AND headline IS NOT NULL AND headline != ''
            ORDER BY COALESCE(published_at, created_at) DESC NULLS LAST
            LIMIT 15
        """, engine)

        if df.empty:
            st.caption("No recent news. Click 'Refresh' to fetch latest.")
            return

        # Filter junk
        junk = ['stock price', 'stock quote', 'google finance', 'yahoo finance', 'msn', 'marketwatch quote']
        df = df[~df['headline'].str.lower().str.contains('|'.join(junk), na=False)]

        if earnings_focus:
            st.markdown("**📰 Other Headlines:**")

        _display_news_items(df, max_items=8)

    except Exception as e:
        st.caption(f"News unavailable: {e}")


def _display_news_items(df: pd.DataFrame, max_items: int = 6):
    """Display news items with sentiment badges, source, and publication date."""

    # Invalid/generic sources to replace with URL-based source
    INVALID_SOURCES = {'brave', 'tavily', 'ai search', 'chrome', 'firefox', 'safari', 'edge', 'opera', '', 'none', 'null', 'unknown'}

    # Source name mapping from URL domains
    URL_TO_SOURCE = {
        'seekingalpha.com': 'Seeking Alpha',
        'cnbc.com': 'CNBC',
        'yahoo.com': 'Yahoo Finance',
        'finance.yahoo.com': 'Yahoo Finance',
        'bloomberg.com': 'Bloomberg',
        'reuters.com': 'Reuters',
        'wsj.com': 'WSJ',
        'ft.com': 'Financial Times',
        'marketwatch.com': 'MarketWatch',
        'fool.com': 'Motley Fool',
        'barrons.com': 'Barrons',
        'investors.com': 'IBD',
        'benzinga.com': 'Benzinga',
        'thestreet.com': 'TheStreet',
        'forbes.com': 'Forbes',
        'morningstar.com': 'Morningstar',
        'nasdaq.com': 'Nasdaq',
        'investopedia.com': 'Investopedia',
        'tipranks.com': 'TipRanks',
        'zacks.com': 'Zacks',
        'investing.com': 'Investing.com',
        'businessinsider.com': 'Business Insider',
        'nytimes.com': 'NY Times',
        'washingtonpost.com': 'Washington Post',
        'cnn.com': 'CNN',
        'bbc.com': 'BBC',
        'theguardian.com': 'Guardian',
        'marketbeat.com': 'MarketBeat',
        'simplywall.st': 'Simply Wall St',
        'tradingview.com': 'TradingView',
        'finviz.com': 'Finviz',
        'stockanalysis.com': 'Stock Analysis',
    }

    def extract_source_from_url(url: str) -> Optional[str]:
        """Extract readable source name from URL."""
        if not url:
            return None
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]

            # Check mapping first
            for url_pattern, source_name in URL_TO_SOURCE.items():
                if url_pattern in domain:
                    return source_name

            # Fallback: capitalize domain name
            # e.g., "finance.yahoo.com" -> "Yahoo" or "example.com" -> "Example"
            parts = domain.replace('.com', '').replace('.org', '').replace('.net', '').split('.')
            if parts:
                # Take the main part (usually second-to-last or last meaningful part)
                main_part = parts[-1] if len(parts) == 1 else parts[-2] if parts[-1] in ['com', 'org', 'net', 'io'] else parts[-1]
                return main_part.capitalize()
        except:
            pass
        return None

    # Filter and sort by actual publication date (newest first)
    # Only keep articles with valid published_at for proper sorting
    df_with_dates = df[df['published_at'].notna()].copy()
    df_without_dates = df[df['published_at'].isna()].copy()

    # Sort articles with dates by published_at (newest first)
    if not df_with_dates.empty:
        df_with_dates = df_with_dates.sort_values('published_at', ascending=False)

    # Combine: dated articles first (sorted by date), then undated articles
    df = pd.concat([df_with_dates, df_without_dates], ignore_index=True)

    displayed = 0
    seen_headlines = set()  # Avoid duplicates

    for _, row in df.iterrows():
        if displayed >= max_items:
            break

        headline = row.get('headline', '') or row.get('title', '')
        if not headline or len(headline) < 10:
            continue

        # Skip duplicates (similar headlines)
        headline_lower = headline.lower()[:50]
        if headline_lower in seen_headlines:
            continue
        seen_headlines.add(headline_lower)

        # Get sentiment
        sentiment = row.get('ai_sentiment_fast')
        if sentiment is not None and pd.notna(sentiment):
            try:
                sent_val = float(sentiment)
                if sent_val >= 60:
                    emoji = "🟢"
                elif sent_val <= 40:
                    emoji = "🔴"
                else:
                    emoji = "🟡"
            except:
                emoji = "🟡"
        else:
            emoji = "🟡"

        # Get source - extract from URL if source is invalid/generic
        source = str(row.get('source', '')).strip() if row.get('source') else ''
        url = row.get('url', '')

        # If source is a search engine name or empty, extract from URL
        if not source or source.lower() in INVALID_SOURCES:
            extracted = extract_source_from_url(url)
            if extracted:
                source = extracted
            else:
                source = ""  # Don't show any source if we can't determine it

        # Truncate long source names
        if source and len(source) > 25:
            source = source[:25]

        # Get date - ONLY use published_at (actual publication date)
        # Do NOT fall back to created_at/fetched_at as those show when WE saved it
        date_str = ""
        date_value = row.get('published_at')

        if date_value is not None and pd.notna(date_value):
            try:
                dt_parsed = None

                # Method 1: If already a datetime object
                if isinstance(date_value, datetime):
                    dt_parsed = date_value
                elif isinstance(date_value, pd.Timestamp):
                    dt_parsed = date_value.to_pydatetime()
                else:
                    # Method 2: Try pandas first (handles ISO format well)
                    try:
                        dt_parsed = pd.to_datetime(date_value)
                        if isinstance(dt_parsed, pd.Timestamp):
                            dt_parsed = dt_parsed.to_pydatetime()
                    except:
                        pass

                    # Method 3: Use dateutil parser
                    if dt_parsed is None and DATEUTIL_AVAILABLE and date_parser:
                        try:
                            dt_parsed = date_parser.parse(str(date_value), fuzzy=True)
                        except:
                            pass

                # Format as actual date/time in Zurich timezone
                if dt_parsed is not None:
                    try:
                        import pytz
                        zurich_tz = pytz.timezone('Europe/Zurich')

                        if dt_parsed.tzinfo is not None:
                            dt_zurich = dt_parsed.astimezone(zurich_tz)
                        else:
                            utc_tz = pytz.UTC
                            dt_utc = utc_tz.localize(dt_parsed)
                            dt_zurich = dt_utc.astimezone(zurich_tz)

                        date_str = dt_zurich.strftime("%d.%m.%Y %H:%M")
                    except:
                        if hasattr(dt_parsed, 'tzinfo') and dt_parsed.tzinfo is not None:
                            dt_parsed = dt_parsed.replace(tzinfo=None)
                        date_str = dt_parsed.strftime("%d.%m.%Y %H:%M")

            except Exception as e:
                logger.debug(f"Date parsing error for '{date_value}': {e}")

        # Build info line - show source and date if available
        info_parts = []
        if source:
            info_parts.append(source)
        if date_str:
            info_parts.append(date_str)
        info_str = " • ".join(info_parts) if info_parts else ""

        # Display: emoji + headline (as link) + source/date
        if url:
            if info_str:
                st.markdown(f"{emoji} [{headline}]({url}) — *{info_str}*")
            else:
                st.markdown(f"{emoji} [{headline}]({url})")
        else:
            if info_str:
                st.markdown(f"{emoji} {headline} — *{info_str}*")
            else:
                st.markdown(f"{emoji} {headline}")

        displayed += 1


def _render_earnings_intelligence(ei):
    """Render Earnings Intelligence section (IES/ECS) - handles both pre and post earnings."""

    if ei.is_post_earnings:
        # POST-EARNINGS VIEW
        _render_post_earnings(ei)
    else:
        # PRE-EARNINGS VIEW
        _render_pre_earnings(ei)


def _render_post_earnings(ei):
    """Render post-earnings analysis with actual results and reaction analysis."""

    days_since = abs(ei.days_to_earnings)
    st.markdown(f"**📅 Earnings {days_since} day(s) ago**")

    # ECS Result - big and prominent
    ecs_val = ei.ecs_category.value if hasattr(ei.ecs_category, 'value') else str(ei.ecs_category)

    ecs_display = {
        'STRONG_BEAT': ('🚀', 'green', 'CRUSHED IT'),
        'BEAT': ('✅', 'green', 'BEAT'),
        'INLINE': ('➡️', 'orange', 'INLINE'),
        'MISS': ('❌', 'red', 'MISSED'),
        'STRONG_MISS': ('💥', 'red', 'BADLY MISSED'),
        'PENDING': ('⏳', 'gray', 'PENDING'),
    }

    emoji, color, label = ecs_display.get(ecs_val, ('❓', 'gray', ecs_val))

    if color == 'green':
        st.success(f"{emoji} **{label}** expectations")
    elif color == 'red':
        st.error(f"{emoji} **{label}** expectations")
    else:
        st.warning(f"{emoji} **{label}** expectations")

    # EPS Result - show actual numbers
    if ei.eps_actual is not None and ei.eps_estimate is not None:
        surprise_pct = ((ei.eps_actual - ei.eps_estimate) / abs(ei.eps_estimate)) * 100 if ei.eps_estimate != 0 else 0
        eps_color = "green" if surprise_pct > 0 else "red"
        beat_miss = "BEAT" if surprise_pct > 0 else "MISS"
        st.markdown(
            f"**EPS:** ${ei.eps_actual:.2f} vs ${ei.eps_estimate:.2f} (:{eps_color}[{beat_miss} by {abs(surprise_pct):.1f}%])")
    elif hasattr(ei, 'eps_surprise_z') and ei.eps_surprise_z != 0:
        direction = "beat" if ei.eps_surprise_z > 0 else "miss"
        st.caption(f"EPS {direction} (z: {ei.eps_surprise_z:.1f})")

    # Price Reaction - THE KEY METRIC
    st.markdown("---")
    st.markdown("**📈 Market Reaction:**")

    if ei.total_reaction_pct <= -10:
        st.error(f"💥 **{ei.total_reaction_pct:+.1f}%** - CRUSHED")
    elif ei.total_reaction_pct <= -5:
        st.warning(f"📉 **{ei.total_reaction_pct:+.1f}%** - Sold Off")
    elif ei.total_reaction_pct >= 10:
        st.success(f"🚀 **{ei.total_reaction_pct:+.1f}%** - SOARED")
    elif ei.total_reaction_pct >= 5:
        st.success(f"📈 **{ei.total_reaction_pct:+.1f}%** - Rallied")
    else:
        st.info(f"➡️ **{ei.total_reaction_pct:+.1f}%** - Flat")

    # Breakdown
    if ei.gap_pct != 0 or ei.intraday_move_pct != 0:
        col1, col2 = st.columns(2)
        with col1:
            gap_emoji = "📈" if ei.gap_pct >= 0 else "📉"
            st.caption(f"{gap_emoji} Gap: {ei.gap_pct:+.1f}%")
        with col2:
            intra_emoji = "📈" if ei.intraday_move_pct >= 0 else "📉"
            st.caption(f"{intra_emoji} Intraday: {ei.intraday_move_pct:+.1f}%")

    # EQS (Earnings Quality Score)
    st.markdown("---")
    eqs_color = "green" if ei.eqs >= 60 else "red" if ei.eqs <= 40 else "orange"
    st.markdown(f"**EQS (Quality):** :{eqs_color}[{ei.eqs:.0f}/100]")
    st.progress(ei.eqs / 100)

    if ei.eqs >= 70:
        st.caption("📊 Strong earnings quality")
    elif ei.eqs <= 40:
        st.caption("📉 Weak earnings quality")

    # =========================================================================
    # REACTION ANALYZER - WHY IT MOVED + RECOMMENDATION
    # =========================================================================
    st.markdown("---")
    st.markdown("#### 🔍 Reaction Analysis")

    if REACTION_ANALYZER_AVAILABLE:
        try:
            # Get the ticker from ei
            ticker = ei.ticker if hasattr(ei, 'ticker') else None

            if ticker:
                # Use cached version to avoid re-analyzing every time
                reaction = _get_cached_reaction_analysis(ticker)

                if reaction is None:
                    st.warning("Could not analyze reaction")
                else:
                    # WHY IT MOVED
                    st.markdown("**Why did it move?**")
                    st.markdown(f"📌 **{reaction.primary_reason}**")

                if reaction.drop_reasons:
                    with st.expander("All factors", expanded=False):
                        for reason in reaction.drop_reasons[:5]:
                            st.markdown(f"• {reason}")

                # QUANTITATIVE ASSESSMENT
                st.markdown("---")
                col1, col2, col3 = st.columns(3)

                with col1:
                    # Oversold score
                    score = reaction.oversold_score
                    if score >= 65:
                        st.metric("Oversold Score", f"🟢 {score:.0f}/100", "Oversold")
                    elif score <= 40:
                        st.metric("Oversold Score", f"🟡 {score:.0f}/100", "Fair value")
                    else:
                        st.metric("Oversold Score", f"⚪ {score:.0f}/100", "Neutral")

                with col2:
                    # Assessment
                    assessment = reaction.reaction_assessment.value.replace("_", " ")
                    st.metric("Assessment", assessment)

                with col3:
                    # Confidence
                    st.metric("Confidence", f"{reaction.confidence:.0f}%")

                # RECOMMENDATION
                st.markdown("---")
                rec = reaction.recommendation.value
                rec_colors = {
                    'STRONG_BUY': ('🟢🟢', 'success'),
                    'BUY_DIP': ('🟢', 'success'),
                    'NIBBLE': ('🟢', 'info'),
                    'WAIT': ('🟡', 'warning'),
                    'AVOID': ('🔴', 'error'),
                    'SELL': ('🔴🔴', 'error'),
                    'TAKE_PROFITS': ('🟠', 'warning'),
                }
                rec_emoji, rec_type = rec_colors.get(rec, ('⚪', 'info'))

                if rec_type == 'success':
                    st.success(f"{rec_emoji} **RECOMMENDATION: {rec}**")
                elif rec_type == 'error':
                    st.error(f"{rec_emoji} **RECOMMENDATION: {rec}**")
                elif rec_type == 'warning':
                    st.warning(f"{rec_emoji} **RECOMMENDATION: {rec}**")
                else:
                    st.info(f"{rec_emoji} **RECOMMENDATION: {rec}**")

                st.caption(reaction.recommendation_reason)

                # TRADING LEVELS (if BUY recommendation)
                if rec in ['STRONG_BUY', 'BUY_DIP', 'NIBBLE'] and reaction.entry_price:
                    st.markdown("---")
                    st.markdown("**📊 Trading Levels:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Entry", f"${reaction.entry_price:.2f}")
                    with col2:
                        st.metric("Stop", f"${reaction.stop_loss:.2f}" if reaction.stop_loss else "-")
                    with col3:
                        st.metric("Target", f"${reaction.target_price:.2f}" if reaction.target_price else "-")
                    with col4:
                        st.metric("R/R", f"{reaction.risk_reward_ratio:.1f}:1" if reaction.risk_reward_ratio else "-")

                    if reaction.suggested_position_pct > 0:
                        st.caption(f"💼 Suggested position: {reaction.suggested_position_pct:.0f}% of normal size")

                # QUANT DETAILS (expandable)
                with st.expander("📐 Quantitative Details", expanded=False):
                    qm = reaction.quant_metrics

                    if qm.implied_move_pct and qm.actual_move_pct:
                        ratio_str = f"{qm.move_vs_implied_ratio:.2f}x" if qm.move_vs_implied_ratio else ""
                        verdict = "⚠️ More than expected" if qm.move_vs_implied_ratio and qm.move_vs_implied_ratio > 1.2 else "✅ Less than expected" if qm.move_vs_implied_ratio and qm.move_vs_implied_ratio < 0.8 else "≈ As expected"
                        st.markdown(
                            f"**Options implied:** ±{abs(qm.implied_move_pct):.1f}% → Actual: {qm.actual_move_pct:+.1f}% ({ratio_str}) {verdict}")

                    if qm.rsi_14:
                        rsi_status = "🔴 OVERSOLD" if qm.rsi_14 < 30 else "🟢 OVERBOUGHT" if qm.rsi_14 > 70 else "Neutral"
                        st.markdown(f"**RSI(14):** {qm.rsi_14:.0f} ({rsi_status})")

                    if qm.reaction_percentile:
                        st.markdown(f"**Reaction percentile:** {qm.reaction_percentile:.0f}th (vs historical)")

                    if qm.sector_move_pct is not None and qm.relative_to_sector is not None:
                        st.markdown(
                            f"**Sector:** {qm.sector_move_pct:+.1f}% | Stock vs sector: {qm.relative_to_sector:+.1f}%")

                    if qm.distance_to_52w_low_pct:
                        st.markdown(f"**Distance to 52w low:** {qm.distance_to_52w_low_pct:.1f}%")

            else:
                st.caption("Ticker not available for reaction analysis")

        except Exception as e:
            logger.error(f"Reaction analysis error: {e}")
            st.caption(f"Reaction analysis unavailable: {e}")
            # Fallback to simple heuristics
            _render_simple_reaction_heuristics(ei)
    else:
        # Fallback to simple heuristics when analyzer not available
        _render_simple_reaction_heuristics(ei)

    # Risk flags
    if ei.risk_flags:
        with st.expander("📋 Risk Flags"):
            for flag in ei.risk_flags:
                st.caption(flag)


def _render_simple_reaction_heuristics(ei):
    """Simple heuristic fallback when reaction analyzer not available."""
    if ei.total_reaction_pct <= -5 and ei.eqs >= 60:
        st.warning("⚠️ **SELL THE NEWS** - Good report but stock sold off. May be oversold - watch for bounce.")
    elif ei.total_reaction_pct >= 5 and ei.eqs <= 40:
        st.warning("⚠️ **BUY THE RUMOR** - Weak report but stock rallied. May be overbought.")
    elif ei.total_reaction_pct <= -8 and ei.eqs >= 45:
        st.info("💡 **POTENTIAL OPPORTUNITY** - Decent report, big selloff. Watch for mean reversion.")
    elif ei.total_reaction_pct <= -8 and ei.eqs <= 40:
        st.error("🚫 **JUSTIFIED SELLOFF** - Bad report, bad reaction. Avoid catching falling knife.")
    elif ei.total_reaction_pct >= 5 and ei.eqs >= 70:
        st.success("✅ **CONFIRMED STRENGTH** - Good report, good reaction. Momentum play.")


def _render_full_reaction_analysis_standalone(ticker: str):
    """
    Render FULL-WIDTH centralized post-earnings reaction analysis.
    STANDALONE version - doesn't need ei object, fetches everything itself.
    """
    try:
        # Use cached version to avoid re-analyzing every time
        reaction = _get_cached_reaction_analysis(ticker)

        if reaction is None:
            st.warning("Could not analyze reaction")
            return

        # =====================================================================
        # TOP ROW: Key Metrics
        # =====================================================================
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            # Earnings Result
            result_emoji = "✅" if reaction.earnings_result == "BEAT" else "❌" if reaction.earnings_result == "MISS" else "➡️"
            st.metric("Result", f"{result_emoji} {reaction.earnings_result or 'N/A'}")

        with col2:
            # Price Reaction
            if reaction.reaction_pct:
                react_emoji = "📈" if reaction.reaction_pct >= 0 else "📉"
                st.metric("Reaction", f"{react_emoji} {reaction.reaction_pct:+.1f}%")
            else:
                st.metric("Reaction", "N/A")

        with col3:
            # Oversold Score
            score = reaction.oversold_score
            score_emoji = "🟢" if score >= 65 else "🟡" if score >= 45 else "🔴"
            st.metric("Oversold Score", f"{score_emoji} {score:.0f}/100")

        with col4:
            # Assessment
            assessment = reaction.reaction_assessment.value.replace("_", " ").title()
            st.metric("Assessment", assessment)

        with col5:
            # Recommendation
            rec = reaction.recommendation.value
            rec_emoji = {'STRONG_BUY': '🟢🟢', 'BUY_DIP': '🟢', 'NIBBLE': '🟡', 'WAIT': '🟡', 'AVOID': '🔴',
                         'SELL': '🔴🔴'}.get(rec, '⚪')
            st.metric("Action", f"{rec_emoji} {rec}")

        st.markdown("---")

        # =====================================================================
        # MAIN SECTION: Two columns - Why + Recommendation
        # =====================================================================
        col_left, col_right = st.columns([3, 2])

        with col_left:
            # WHY DID IT MOVE
            st.markdown("### 🔍 Why Did It Move?")

            st.markdown(f"**Primary Reason:** {reaction.primary_reason}")

            if reaction.drop_reasons:
                st.markdown("**All Contributing Factors:**")
                for i, reason in enumerate(reaction.drop_reasons[:6], 1):
                    st.markdown(f"{i}. {reason}")

            # Key Headlines
            if reaction.key_headlines:
                st.markdown("---")
                st.markdown("**📰 Key Headlines:**")
                for headline in reaction.key_headlines[:4]:
                    st.caption(f"• {headline[:100]}...")

        with col_right:
            # RECOMMENDATION BOX
            st.markdown("### 🎯 Recommendation")

            rec = reaction.recommendation.value
            rec_configs = {
                'STRONG_BUY': ('🟢🟢', 'success', 'Strong Buy - High conviction entry'),
                'BUY_DIP': ('🟢', 'success', 'Buy the Dip - Good entry point'),
                'NIBBLE': ('🟡', 'info', 'Nibble - Small position, wait for confirmation'),
                'WAIT': ('🟡', 'warning', 'Wait - Need more clarity'),
                'AVOID': ('🔴', 'error', 'Avoid - More downside likely'),
                'SELL': ('🔴🔴', 'error', 'Sell - Exit position'),
                'TAKE_PROFITS': ('🟠', 'warning', 'Take Profits - Reduce exposure'),
            }
            rec_emoji, rec_type, rec_label = rec_configs.get(rec, ('⚪', 'info', rec))

            if rec_type == 'success':
                st.success(f"**{rec_emoji} {rec_label}**")
            elif rec_type == 'error':
                st.error(f"**{rec_emoji} {rec_label}**")
            elif rec_type == 'warning':
                st.warning(f"**{rec_emoji} {rec_label}**")
            else:
                st.info(f"**{rec_emoji} {rec_label}**")

            st.caption(f"Confidence: {reaction.confidence:.0f}%")
            st.markdown(f"*{reaction.recommendation_reason}*")

            # Trading Levels
            if rec in ['STRONG_BUY', 'BUY_DIP', 'NIBBLE'] and reaction.entry_price:
                st.markdown("---")
                st.markdown("**📊 Trading Levels:**")

                levels_col1, levels_col2 = st.columns(2)
                with levels_col1:
                    st.metric("Entry", f"${reaction.entry_price:.2f}")
                    if reaction.stop_loss:
                        st.metric("Stop Loss", f"${reaction.stop_loss:.2f}")
                with levels_col2:
                    if reaction.target_price:
                        st.metric("Target", f"${reaction.target_price:.2f}")
                    if reaction.risk_reward_ratio:
                        st.metric("Risk/Reward", f"{reaction.risk_reward_ratio:.1f}:1")

                if reaction.suggested_position_pct > 0:
                    st.info(f"💼 Suggested: {reaction.suggested_position_pct:.0f}% of normal position size")

        st.markdown("---")

        # =====================================================================
        # PRICE INFO
        # =====================================================================
        if reaction.price_before and reaction.price_current:
            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                st.metric("Price Before", f"${reaction.price_before:.2f}")
            with pcol2:
                st.metric("Current Price", f"${reaction.price_current:.2f}")
            with pcol3:
                if reaction.reaction_pct:
                    change_color = "green" if reaction.reaction_pct >= 0 else "red"
                    st.metric("Change", f"{reaction.reaction_pct:+.1f}%")

        # =====================================================================
        # BOTTOM SECTION: Quantitative Details (Expandable)
        # =====================================================================
        with st.expander("📐 Quantitative Analysis Details", expanded=False):
            qcol1, qcol2, qcol3 = st.columns(3)

            qm = reaction.quant_metrics

            with qcol1:
                st.markdown("**Options Analysis:**")
                if qm.implied_move_pct:
                    st.markdown(f"• Implied move: ±{abs(qm.implied_move_pct):.1f}%")
                if qm.actual_move_pct:
                    st.markdown(f"• Actual move: {qm.actual_move_pct:+.1f}%")
                if qm.move_vs_implied_ratio:
                    verdict = "More than expected ⚠️" if qm.move_vs_implied_ratio > 1.2 else "Less than expected ✅" if qm.move_vs_implied_ratio < 0.8 else "As expected"
                    st.markdown(f"• Ratio: {qm.move_vs_implied_ratio:.2f}x ({verdict})")

            with qcol2:
                st.markdown("**Technical Analysis:**")
                if qm.rsi_14:
                    rsi_status = "OVERSOLD 🔴" if qm.rsi_14 < 30 else "OVERBOUGHT 🟢" if qm.rsi_14 > 70 else "Neutral"
                    st.markdown(f"• RSI(14): {qm.rsi_14:.0f} ({rsi_status})")
                if qm.distance_to_52w_low_pct:
                    st.markdown(f"• Distance to 52w low: {qm.distance_to_52w_low_pct:.1f}%")
                if qm.reaction_percentile:
                    st.markdown(f"• Reaction percentile: {qm.reaction_percentile:.0f}th")

            with qcol3:
                st.markdown("**Relative Performance:**")
                if qm.sector_move_pct is not None:
                    st.markdown(f"• Sector move: {qm.sector_move_pct:+.1f}%")
                if qm.relative_to_sector is not None:
                    rel_status = "Underperformed ⚠️" if qm.relative_to_sector < -3 else "Outperformed ✅" if qm.relative_to_sector > 3 else "Inline"
                    st.markdown(f"• vs Sector: {qm.relative_to_sector:+.1f}% ({rel_status})")

        # =====================================================================
        # EPS DETAILS (Expandable)
        # =====================================================================
        with st.expander("📊 Earnings Details", expanded=False):
            if reaction.eps_actual is not None or reaction.eps_estimate is not None:
                ecol1, ecol2, ecol3 = st.columns(3)

                with ecol1:
                    st.markdown("**EPS:**")
                    if reaction.eps_actual is not None:
                        st.markdown(f"• Actual: ${reaction.eps_actual:.2f}")
                    if reaction.eps_estimate is not None:
                        st.markdown(f"• Estimate: ${reaction.eps_estimate:.2f}")

                with ecol2:
                    if reaction.eps_actual is not None and reaction.eps_estimate is not None and reaction.eps_estimate != 0:
                        surprise = ((reaction.eps_actual - reaction.eps_estimate) / abs(reaction.eps_estimate)) * 100
                        st.markdown(f"**Surprise:** {surprise:+.1f}%")

                with ecol3:
                    st.markdown(f"**Result:** {reaction.earnings_result or 'N/A'}")
            else:
                st.caption("EPS details not available")

    except Exception as e:
        logger.error(f"Full reaction analysis error for {ticker}: {e}")
        import traceback
        st.error(f"Could not load reaction analysis: {e}")
        st.code(traceback.format_exc())


def _render_full_reaction_analysis(ticker: str, ei):
    """
    Render FULL-WIDTH centralized post-earnings reaction analysis.
    Shows everything about the earnings reaction in one place.
    """
    try:
        # Use cached version to avoid re-analyzing every time
        reaction = _get_cached_reaction_analysis(ticker)

        if reaction is None:
            st.warning("Could not analyze reaction")
            return

        # =====================================================================
        # TOP ROW: Key Metrics
        # =====================================================================
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            # ECS Result
            ecs_val = ei.ecs_category.value if hasattr(ei.ecs_category, 'value') else str(ei.ecs_category)
            ecs_emoji = {'STRONG_BEAT': '🚀', 'BEAT': '✅', 'INLINE': '➡️', 'MISS': '❌', 'STRONG_MISS': '💥'}.get(ecs_val,
                                                                                                               '❓')
            st.metric("Expectations", f"{ecs_emoji} {ecs_val}")

        with col2:
            # Price Reaction
            reaction_pct = ei.total_reaction_pct
            react_emoji = "📈" if reaction_pct >= 0 else "📉"
            st.metric("Reaction", f"{react_emoji} {reaction_pct:+.1f}%")

        with col3:
            # Oversold Score
            score = reaction.oversold_score
            score_emoji = "🟢" if score >= 65 else "🟡" if score >= 45 else "🔴"
            st.metric("Oversold Score", f"{score_emoji} {score:.0f}/100")

        with col4:
            # Assessment
            assessment = reaction.reaction_assessment.value.replace("_", " ").title()
            st.metric("Assessment", assessment)

        with col5:
            # Recommendation
            rec = reaction.recommendation.value
            rec_emoji = {'STRONG_BUY': '🟢🟢', 'BUY_DIP': '🟢', 'NIBBLE': '🟡', 'WAIT': '🟡', 'AVOID': '🔴',
                         'SELL': '🔴🔴'}.get(rec, '⚪')
            st.metric("Action", f"{rec_emoji} {rec}")

        st.markdown("---")

        # =====================================================================
        # MAIN SECTION: Two columns - Why + Recommendation
        # =====================================================================
        col_left, col_right = st.columns([3, 2])

        with col_left:
            # WHY DID IT MOVE
            st.markdown("### 🔍 Why Did It Move?")

            st.markdown(f"**Primary Reason:** {reaction.primary_reason}")

            if reaction.drop_reasons:
                st.markdown("**All Contributing Factors:**")
                for i, reason in enumerate(reaction.drop_reasons[:6], 1):
                    st.markdown(f"{i}. {reason}")

            # Key Headlines
            if reaction.key_headlines:
                st.markdown("---")
                st.markdown("**📰 Key Headlines:**")
                for headline in reaction.key_headlines[:4]:
                    st.caption(f"• {headline[:80]}...")

        with col_right:
            # RECOMMENDATION BOX
            st.markdown("### 🎯 Recommendation")

            rec = reaction.recommendation.value
            rec_configs = {
                'STRONG_BUY': ('🟢🟢', 'success', 'Strong Buy - High conviction entry'),
                'BUY_DIP': ('🟢', 'success', 'Buy the Dip - Good entry point'),
                'NIBBLE': ('🟡', 'info', 'Nibble - Small position, wait for confirmation'),
                'WAIT': ('🟡', 'warning', 'Wait - Need more clarity'),
                'AVOID': ('🔴', 'error', 'Avoid - More downside likely'),
                'SELL': ('🔴🔴', 'error', 'Sell - Exit position'),
                'TAKE_PROFITS': ('🟠', 'warning', 'Take Profits - Reduce exposure'),
            }
            rec_emoji, rec_type, rec_label = rec_configs.get(rec, ('⚪', 'info', rec))

            if rec_type == 'success':
                st.success(f"**{rec_emoji} {rec_label}**")
            elif rec_type == 'error':
                st.error(f"**{rec_emoji} {rec_label}**")
            elif rec_type == 'warning':
                st.warning(f"**{rec_emoji} {rec_label}**")
            else:
                st.info(f"**{rec_emoji} {rec_label}**")

            st.caption(f"Confidence: {reaction.confidence:.0f}%")
            st.markdown(f"*{reaction.recommendation_reason}*")

            # Trading Levels
            if rec in ['STRONG_BUY', 'BUY_DIP', 'NIBBLE'] and reaction.entry_price:
                st.markdown("---")
                st.markdown("**📊 Trading Levels:**")

                levels_col1, levels_col2 = st.columns(2)
                with levels_col1:
                    st.metric("Entry", f"${reaction.entry_price:.2f}")
                    if reaction.stop_loss:
                        st.metric("Stop Loss", f"${reaction.stop_loss:.2f}")
                with levels_col2:
                    if reaction.target_price:
                        st.metric("Target", f"${reaction.target_price:.2f}")
                    if reaction.risk_reward_ratio:
                        st.metric("Risk/Reward", f"{reaction.risk_reward_ratio:.1f}:1")

                if reaction.suggested_position_pct > 0:
                    st.info(f"💼 Suggested: {reaction.suggested_position_pct:.0f}% of normal position size")

        st.markdown("---")

        # =====================================================================
        # BOTTOM SECTION: Quantitative Details (Expandable)
        # =====================================================================
        with st.expander("📐 Quantitative Analysis Details", expanded=False):
            qcol1, qcol2, qcol3 = st.columns(3)

            qm = reaction.quant_metrics

            with qcol1:
                st.markdown("**Options Analysis:**")
                if qm.implied_move_pct:
                    st.markdown(f"• Implied move: ±{abs(qm.implied_move_pct):.1f}%")
                if qm.actual_move_pct:
                    st.markdown(f"• Actual move: {qm.actual_move_pct:+.1f}%")
                if qm.move_vs_implied_ratio:
                    verdict = "More than expected ⚠️" if qm.move_vs_implied_ratio > 1.2 else "Less than expected ✅" if qm.move_vs_implied_ratio < 0.8 else "As expected"
                    st.markdown(f"• Ratio: {qm.move_vs_implied_ratio:.2f}x ({verdict})")

            with qcol2:
                st.markdown("**Technical Analysis:**")
                if qm.rsi_14:
                    rsi_status = "OVERSOLD 🔴" if qm.rsi_14 < 30 else "OVERBOUGHT 🟢" if qm.rsi_14 > 70 else "Neutral"
                    st.markdown(f"• RSI(14): {qm.rsi_14:.0f} ({rsi_status})")
                if qm.distance_to_52w_low_pct:
                    st.markdown(f"• Distance to 52w low: {qm.distance_to_52w_low_pct:.1f}%")
                if qm.reaction_percentile:
                    st.markdown(f"• Reaction percentile: {qm.reaction_percentile:.0f}th")

            with qcol3:
                st.markdown("**Relative Performance:**")
                if qm.sector_move_pct is not None:
                    st.markdown(f"• Sector move: {qm.sector_move_pct:+.1f}%")
                if qm.relative_to_sector is not None:
                    rel_status = "Underperformed ⚠️" if qm.relative_to_sector < -3 else "Outperformed ✅" if qm.relative_to_sector > 3 else "Inline"
                    st.markdown(f"• vs Sector: {qm.relative_to_sector:+.1f}% ({rel_status})")

        # =====================================================================
        # EARNINGS DETAILS (Expandable)
        # =====================================================================
        with st.expander("📊 Earnings Details", expanded=False):
            ecol1, ecol2, ecol3 = st.columns(3)

            with ecol1:
                st.markdown("**EPS:**")
                if ei.eps_actual is not None and ei.eps_estimate is not None:
                    surprise = ((ei.eps_actual - ei.eps_estimate) / abs(
                        ei.eps_estimate)) * 100 if ei.eps_estimate != 0 else 0
                    st.markdown(f"• Actual: ${ei.eps_actual:.2f}")
                    st.markdown(f"• Estimate: ${ei.eps_estimate:.2f}")
                    st.markdown(f"• Surprise: {surprise:+.1f}%")

            with ecol2:
                st.markdown("**Quality Scores:**")
                st.markdown(f"• EQS (Quality): {ei.eqs:.0f}/100")
                if hasattr(ei, 'ies') and ei.ies:
                    st.markdown(f"• IES (Pre-ER): {ei.ies:.0f}/100")

            with ecol3:
                st.markdown("**Price Reaction:**")
                st.markdown(f"• Total: {ei.total_reaction_pct:+.1f}%")
                if ei.gap_pct:
                    st.markdown(f"• Gap: {ei.gap_pct:+.1f}%")
                if ei.intraday_move_pct:
                    st.markdown(f"• Intraday: {ei.intraday_move_pct:+.1f}%")

    except Exception as e:
        logger.error(f"Full reaction analysis error for {ticker}: {e}")
        st.warning(f"Could not load full reaction analysis: {e}")
        # Show basic info
        st.markdown(f"**Reaction:** {ei.total_reaction_pct:+.1f}%")
        st.markdown(f"**EQS:** {ei.eqs:.0f}/100")


def _render_pre_earnings(ei):
    """Render pre-earnings IES analysis."""

    # Regime badge
    regime_colors = {
        'HYPED': ('🔥', 'red', 'High expectations priced in'),
        'FEARED': ('😰', 'blue', 'Low expectations - upside potential'),
        'VOLATILE': ('⚡', 'orange', 'High uncertainty'),
        'NORMAL': ('📊', 'gray', 'Standard expectations'),
    }

    regime_val = ei.regime.value if hasattr(ei.regime, 'value') else str(ei.regime)
    emoji, color, desc = regime_colors.get(regime_val, ('📊', 'gray', ''))

    st.markdown(f"**Regime:** {emoji} :{color}[{regime_val}]")
    st.caption(desc)

    # IES Score with gauge
    ies_color = "red" if ei.ies >= 75 else "orange" if ei.ies >= 60 else "green" if ei.ies <= 35 else "gray"
    st.markdown(f"**IES:** :{ies_color}[{ei.ies:.0f}/100]")
    st.progress(ei.ies / 100)

    # IES interpretation
    if ei.ies >= 75:
        st.caption("⚠️ Very high expectations - needs blowout to beat")
    elif ei.ies >= 60:
        st.caption("📈 Elevated expectations - strong beat required")
    elif ei.ies <= 35:
        st.caption("✅ Low expectations - easier bar to clear")
    else:
        st.caption("📊 Normal expectations")

    # Days to earnings with urgency
    if ei.days_to_earnings <= 2:
        st.error(f"🚨 EARNINGS IN {ei.days_to_earnings} DAY(S)!")
    elif ei.days_to_earnings <= 5:
        st.warning(f"⚠️ Earnings in {ei.days_to_earnings} days")
    else:
        st.info(f"📅 Earnings in {ei.days_to_earnings} days")

    # Position scaling warning
    if ei.position_scale < 1.0:
        st.warning(f"⚖️ Position scale: {ei.position_scale:.0%}")
        st.caption("Reduce position size due to earnings risk")

    # Component breakdown in expander
    with st.expander("📊 IES Components"):
        components = [
            ("Pre-ER Runup", ei.pre_earnings_runup),
            ("Implied Move", ei.implied_move_percentile),
            ("Analyst Revisions", ei.analyst_revision_momentum),
            ("Options Skew", ei.options_skew_score),
            ("News Sentiment", ei.news_sentiment_score),
            ("Beat Rate", ei.historical_beat_rate),
            ("Sector Momentum", ei.sector_momentum),
        ]
        for name, score in components:
            emoji = "🟢" if score <= 40 else "🔴" if score >= 70 else "🟡"
            st.caption(f"{emoji} {name}: {score:.0f}")

    # Risk flags
    if ei.risk_flags:
        st.markdown("**Flags:**")
        for flag in ei.risk_flags[:3]:
            st.caption(flag)


def _render_insider_activity(additional_data: dict):
    """Render insider activity."""
    transactions = additional_data.get('insider_transactions', [])

    # Filter valid
    transactions = [t for t in transactions
                    if t.get('shares_transacted') and float(t.get('shares_transacted', 0)) > 0
                    and t.get('transaction_type') in ('P', 'S')]

    if not transactions:
        st.caption("No recent insider activity")
        return

    # Calculate net
    buys = [t for t in transactions if t.get('transaction_type') == 'P']
    sells = [t for t in transactions if t.get('transaction_type') == 'S']

    buy_val = sum(float(t.get('total_value', 0) or 0) for t in buys)
    sell_val = sum(float(t.get('total_value', 0) or 0) for t in sells)

    if buy_val > sell_val * 1.5:
        st.success(f"🟢 Net Buying ({len(buys)} buys, {len(sells)} sells)")
    elif sell_val > buy_val * 1.5:
        st.error(f"🔴 Net Selling ({len(buys)} buys, {len(sells)} sells)")
    else:
        st.warning(f"🟡 Mixed ({len(buys)} buys, {len(sells)} sells)")

    # Show transactions
    for t in transactions[:4]:
        tx_type = "🟢 Buy" if t.get('transaction_type') == 'P' else "🔴 Sell"
        name = str(t.get('insider_name', ''))[:15]
        val = float(t.get('total_value', 0) or 0)
        val_str = f"${val / 1000:.0f}K" if val >= 1000 else f"${val:.0f}"
        st.caption(f"{tx_type}: {name} ({val_str})")




def _get_recent_news_headlines(ticker: str) -> list:
    """Get recent news headlines for a ticker from database."""
    try:
        from src.db.connection import get_engine
        import pandas as pd
        engine = get_engine()
        query = """
            SELECT headline FROM news_articles
            WHERE ticker = %s
            AND published_at >= NOW() - INTERVAL '7 days'
            ORDER BY published_at DESC
            LIMIT 15
        """
        df = pd.read_sql(query, engine, params=(ticker,))
        return df['headline'].tolist() if not df.empty else []
    except Exception:
        return []

def _render_ai_chat(signal: UnifiedSignal, additional_data: dict):
    """Render AI chat section with pre-computed metrics (LLM should not calculate)."""

    st.markdown("### 🤖 Ask AI About This Stock")

    # Get extended hours price for context
    extended_price = _get_extended_hours_price(signal.ticker)

    # ==========================================================================
    # BUILD MARKET SNAPSHOT - single source of truth for all metrics
    # ==========================================================================
    snapshot = MarketSnapshot(
        ticker=signal.ticker,
        snapshot_ts_utc=datetime.now(timezone.utc)
    )

    # Price data
    if extended_price.get('has_extended') and extended_price.get('extended_price', 0) > 0:
        snapshot.price = extended_price['extended_price']
        snapshot.price_source = extended_price.get('session', 'unknown').lower()
    else:
        snapshot.price = signal.current_price
        snapshot.price_source = 'regular'

    snapshot.previous_close = extended_price.get('regular_close', signal.current_price)
    if snapshot.previous_close > 0:
        snapshot.price_change_pct = ((snapshot.price - snapshot.previous_close) / snapshot.previous_close) * 100

    # 52W data - calculated from snapshot price
    snapshot.week_52_high = signal.week_52_high
    snapshot.week_52_low = signal.week_52_low
    if snapshot.week_52_high > 0 and snapshot.price > 0:
        snapshot.pct_from_52w_high = ((snapshot.price - snapshot.week_52_high) / snapshot.week_52_high) * 100
    if snapshot.week_52_low > 0 and snapshot.price > 0:
        snapshot.pct_from_52w_low = ((snapshot.price - snapshot.week_52_low) / snapshot.week_52_low) * 100

    # Scores
    snapshot.sentiment_score = signal.sentiment_score if signal.sentiment_score is not None else None
    snapshot.technical_score = signal.technical_score if signal.technical_score is not None else None
    snapshot.fundamental_score = signal.fundamental_score if signal.fundamental_score is not None else None
    snapshot.options_score = signal.options_score if signal.options_score is not None else None
    snapshot.earnings_score = signal.earnings_score if signal.earnings_score is not None else None
    snapshot.total_score = signal.today_score

    # Count available components
    components = [snapshot.sentiment_score, snapshot.technical_score,
                  snapshot.fundamental_score, snapshot.options_score, snapshot.earnings_score]
    snapshot.components_available = sum(1 for c in components if c is not None)
    snapshot.components_total = 5

    # Target
    target_mean = additional_data.get('target_mean', 0)
    if target_mean and target_mean > 0:
        snapshot.analyst_target = target_mean
        if snapshot.price > 0:
            snapshot.target_upside_pct = ((target_mean / snapshot.price) - 1) * 100

    # ==========================================================================
    # COMPUTE TRADE LEVELS (deterministic - LLM should NOT recalculate)
    # ==========================================================================
    risk_level = signal.risk_level.value if hasattr(signal.risk_level, 'value') else str(signal.risk_level)
    trade_levels = compute_trade_levels(
        current_price=snapshot.price,
        signal_score=signal.today_score or 50,
        risk_level=risk_level,
        analyst_target=snapshot.analyst_target,
        max_pain=None,  # Will be filled from options context
        week_52_high=snapshot.week_52_high,
        week_52_low=snapshot.week_52_low
    )

    # ==========================================================================
    # RESOLVE SIGNAL CONFLICT (deterministic policy)
    # ==========================================================================
    platform_direction = signal.today_signal.value if hasattr(signal.today_signal, 'value') else str(signal.today_signal)
    committee_direction = signal.committee_verdict or "HOLD"

    # Get ML status from additional_data
    ml_status = additional_data.get('ml_status', 'UNKNOWN')
    if 'alpha_context' in additional_data:
        alpha_ctx = additional_data.get('alpha_context', {})
        if isinstance(alpha_ctx, dict):
            ml_status = alpha_ctx.get('ml_status', ml_status)

    policy_decision = resolve_signal_conflict(
        platform_signal=platform_direction,
        platform_score=signal.today_score or 50,
        committee_signal=committee_direction,
        committee_confidence=signal.committee_confidence or 0.5,
        components_fresh=snapshot.components_available,
        ml_status=ml_status
    )

    # Update trade levels with policy decision
    trade_levels.position_size_multiplier = policy_decision.max_size

    # ==========================================================================
    # APPLY UNCERTAINTY SHRINKAGE
    # ==========================================================================
    if snapshot.total_score is not None:
        shrunk_score, data_confidence = apply_uncertainty_shrinkage(
            raw_score=snapshot.total_score,
            components_available=snapshot.components_available,
            components_total=snapshot.components_total,
            staleness_penalty=snapshot.staleness_penalty
        )
    else:
        shrunk_score, data_confidence = 50, "NONE"

    # ==========================================================================
    # BUILD CONTEXT WITH PRE-COMPUTED METRICS
    # ==========================================================================
    price_timestamp = snapshot.snapshot_ts_utc.strftime("%Y-%m-%d %H:%M UTC")

    # Conflict description
    if policy_decision.conflict_flag:
        conflict_desc = f"⚠️ CONFLICT: {' | '.join(policy_decision.reasons)}"
    else:
        conflict_desc = "✅ ALIGNED: Platform and Committee signals agree"

    # Pre-compute conditional strings (can't nest f-strings)
    analyst_target_str = f"${snapshot.analyst_target:.2f}" if snapshot.analyst_target else "UNKNOWN"
    target_upside_str = f"{snapshot.target_upside_pct:+.1f}%" if snapshot.target_upside_pct else "UNKNOWN"
    technical_str = snapshot.technical_score if snapshot.technical_score is not None else 'UNKNOWN'
    fundamental_str = snapshot.fundamental_score if snapshot.fundamental_score is not None else 'UNKNOWN'
    sentiment_str = snapshot.sentiment_score if snapshot.sentiment_score is not None else 'UNKNOWN'
    options_str = snapshot.options_score if snapshot.options_score is not None else 'UNKNOWN'
    earnings_str = snapshot.earnings_score if snapshot.earnings_score is not None else 'UNKNOWN'

    context = f"""
=== {signal.ticker} - {signal.company_name} ===
Sector: {signal.sector}
Data as of: {price_timestamp}

============================================================
PRE-COMPUTED METRICS (use these exactly - do NOT recalculate)
============================================================
PRICE DATA:
- Current Price: ${snapshot.price:.2f} ({snapshot.price_source})
- Previous Close: ${snapshot.previous_close:.2f}
- Change: {snapshot.price_change_pct:+.2f}%

52-WEEK RANGE:
- 52W High: ${snapshot.week_52_high:.2f}
- 52W Low: ${snapshot.week_52_low:.2f}
- Distance from 52W High: {snapshot.pct_from_52w_high:+.1f}%
- Distance from 52W Low: {snapshot.pct_from_52w_low:+.1f}%

TARGET:
- Analyst Target: {analyst_target_str}
- Target Upside: {target_upside_str}

TRADE LEVELS (pre-computed):
- Entry Price: ${trade_levels.entry_price:.2f} ({trade_levels.entry_method})
- Stop Loss: ${trade_levels.stop_loss:.2f} ({trade_levels.stop_distance_pct:+.1f}%, {trade_levels.stop_method})
- Target Price: ${trade_levels.target_price:.2f} ({trade_levels.target_upside_pct:+.1f}%)
- Risk/Reward: {trade_levels.risk_reward_ratio:.2f}:1
- Position Size: {trade_levels.position_size_multiplier}x

DATA QUALITY:
- Components Available: {snapshot.components_available}/{snapshot.components_total}
- Data Completeness: {data_confidence}
- Raw Score: {snapshot.total_score}
- Uncertainty-Adjusted Score: {shrunk_score:.1f}
============================================================

SIGNALS:
- Platform Signal: {platform_direction} ({signal.today_score}%)
- Long-term: {signal.longterm_score}/100
- Risk Level: {risk_level} ({signal.risk_score})
- Reason: {signal.signal_reason}

COMPONENT SCORES:
- Technical: {technical_str} ({signal.technical_signal})
- Fundamental: {fundamental_str} ({signal.fundamental_signal})
- Sentiment: {sentiment_str} ({signal.sentiment_signal})
- Options: {options_str} ({signal.options_signal})
- Earnings: {earnings_str} ({signal.earnings_signal})

COMMITTEE: {committee_direction} ({signal.committee_confidence:.0%})
SIGNAL CONSENSUS: {conflict_desc}

============================================================
POLICY DECISION (BINDING - LLM cannot override)
============================================================
Action: {policy_decision.action}
Max Position Size: {policy_decision.max_size}x
Confidence: {policy_decision.confidence}
Reasons: {'; '.join(policy_decision.reasons) if policy_decision.reasons else 'None'}
============================================================
"""

    if signal.in_portfolio:
        context += f"""
PORTFOLIO POSITION:
- Weight: {signal.portfolio_weight:.2%}
- P&L: {signal.portfolio_pnl_pct:+.1f}%
- Days Held: {signal.days_held}
"""

    # Add Earnings Intelligence if available
    ei = additional_data.get('earnings_intelligence')
    if ei and ei.in_compute_window:
        context += f"""
EARNINGS INTELLIGENCE:
- Days to Earnings: {ei.days_to_earnings}
- IES (Implied Expectations): {ei.ies:.0f}/100
- Regime: {ei.regime.value if hasattr(ei.regime, 'value') else ei.regime}
- Position Scale: {ei.position_scale:.0%}
- In Action Window: {ei.in_action_window}
"""
        if ei.risk_flags:
            context += f"- Risk Flags: {', '.join(ei.risk_flags)}\n"

    # Add Recent News Headlines
    try:
        news_headlines = _get_recent_news_headlines(signal.ticker)
        if news_headlines:
            context += f"\nRECENT NEWS ({len(news_headlines)} articles):\n"
            for i, headline in enumerate(news_headlines[:5], 1):
                context += f"  {i}. {headline[:100]}\n"
    except Exception as e:
        logger.debug(f"News context error: {e}")

    # Add Options Flow Details
    try:
        options_context = _get_options_flow_context(signal.ticker, snapshot.price)
        if options_context:
            context += options_context
    except Exception as e:
        logger.debug(f"Options flow context error: {e}")

    # Add Insider Trading Activity
    try:
        insider_context = _get_insider_context(signal.ticker)
        if insider_context:
            context += insider_context
    except Exception as e:
        logger.debug(f"Insider context error: {e}")

    # Add Short Squeeze Analysis
    try:
        squeeze_context = _get_squeeze_context(signal.ticker)
        if squeeze_context:
            context += squeeze_context
    except Exception as e:
        logger.debug(f"Squeeze context error: {e}")

    # Add Fundamentals Data
    try:
        fundamentals_context = _get_fundamentals_context(signal.ticker)
        if fundamentals_context:
            context += fundamentals_context
    except Exception as e:
        logger.debug(f"Fundamentals context error: {e}")

    # Add Analyst Ratings
    try:
        analyst_context = _get_analyst_context(signal.ticker)
        if analyst_context:
            context += analyst_context
    except Exception as e:
        logger.debug(f"Analyst context error: {e}")

    # Add Alpha Model ML Predictions
    try:
        # Pass platform data directly from signal object to avoid DB query issues
        alpha_context = _get_alpha_model_context(
            ticker=signal.ticker,
            platform_score=signal.today_score,
            platform_signal=signal.today_signal.value,
            technical_score=signal.technical_score
        )
        if alpha_context:
            context += alpha_context
    except Exception as e:
        context += f"\n🔍 ALPHA ERROR: {e}\n"

    # Add AI Rules section
    context += """
===================================================================
🚫 AI RESPONSE RULES - MUST FOLLOW
===================================================================
1. DO NOT claim "institutional" buying/selling based on P/C ratios alone
2. DO NOT invent specific dates for OPEC meetings, Fed decisions, or other events unless explicitly provided in the data
3. DO NOT use options data that is more than 3 days old for current recommendations
4. DO NOT use max pain values that are >20% away from current stock price (data error)
5. DO report the ACTUAL conflict between Platform and Committee signals
6. DO note when data is stale and recommend verification with real-time sources
7. Dividend yield may vary ±0.1% depending on methodology - acknowledge this uncertainty
===================================================================
"""

    # Initialize session state for this ticker
    chat_key = f'ai_chat_{signal.ticker}'
    if chat_key not in st.session_state:
        st.session_state[chat_key] = {'response': '', 'last_question': ''}

    # Quick buttons - these directly trigger AI call
    st.caption("Quick questions:")
    col1, col2, col3, col4 = st.columns(4)

    quick_question = None

    with col1:
        if st.button("📊 Full Analysis", key=f"ai_full_{signal.ticker}", width='stretch'):
            quick_question = f"Give me a complete analysis of {signal.ticker} including current signals, key metrics, and your buy/hold/sell recommendation with reasoning."
    with col2:
        if st.button("🎯 Trade Idea", key=f"ai_trade_{signal.ticker}", width='stretch'):
            quick_question = f"Give me a specific trade idea for {signal.ticker} with: entry price, target price, stop loss, position size suggestion, and time horizon."
    with col3:
        if st.button("⚠️ Risks", key=f"ai_risk_{signal.ticker}", width='stretch'):
            quick_question = f"What are the main risks for {signal.ticker}? Include market risks, company-specific risks, sector risks, and any upcoming catalysts to watch."
    with col4:
        if st.button("📰 News", key=f"ai_news_{signal.ticker}", width='stretch'):
            quick_question = f"Summarize the recent news and sentiment for {signal.ticker}. What are the key headlines and how might they affect the stock price?"

    # If quick button was pressed, run AI immediately
    if quick_question:
        with st.spinner("🤔 Analyzing..."):
            response, model_used = _get_ai_response(quick_question, context, signal.ticker)
            st.session_state[chat_key]['response'] = response
            st.session_state[chat_key]['last_question'] = quick_question
            st.session_state[chat_key]['model_used'] = model_used

    st.markdown("---")

    # Custom question input (text_area for multi-line)
    st.caption("Or ask your own question:")
    question = st.text_area(
        "Your question:",
        height=80,
        key=f"ai_input_{signal.ticker}",
        placeholder=f"e.g., Should I buy {signal.ticker} at current levels?\nWhat's the best entry point?\nCompare to competitors...",
        label_visibility="collapsed"
    )

    col_submit, col_clear = st.columns([1, 1])
    with col_submit:
        if st.button("🚀 Ask AI", key=f"ai_submit_{signal.ticker}", type="primary", width='stretch'):
            if question.strip():
                with st.spinner("🤔 Thinking..."):
                    response, model_used = _get_ai_response(question, context, signal.ticker)
                    st.session_state[chat_key]['response'] = response
                    st.session_state[chat_key]['last_question'] = question
                    st.session_state[chat_key]['model_used'] = model_used
                    st.rerun()
            else:
                st.warning("Please enter a question")

    with col_clear:
        if st.button("🗑️ Clear", key=f"ai_clear_{signal.ticker}", width='stretch'):
            st.session_state[chat_key] = {'response': '', 'last_question': '', 'model_used': ''}
            st.rerun()

    # Show response if exists
    if st.session_state[chat_key]['response']:
        st.markdown("---")

        # Show which AI model answered
        model_used = st.session_state[chat_key].get('model_used', 'Unknown')
        st.markdown(f"#### 💬 AI Response")
        st.caption(f"🤖 **Model:** {model_used}")

        if st.session_state[chat_key]['last_question']:
            st.caption(f"Q: {st.session_state[chat_key]['last_question'][:100]}...")
        st.markdown(st.session_state[chat_key]['response'])

    # Show context in expander
    with st.expander("📋 Data Context (what AI sees)"):
        st.code(context, language=None)


def _get_alpha_model_context(ticker: str, platform_score: float = None,
                              platform_signal: str = None, technical_score: float = None) -> str:
    """
    Get Alpha Model ML predictions with all enhancements applied:
    - Forecast shrinkage (context-aware)
    - ML reliability gate (vol-scaled bias, EWMA accuracy)
    - Calibrated probabilities (smoothed by sample size)
    - Decision policy (binding rules for LLM)
    - Sector neutralization
    - Sentiment velocity

    Args:
        ticker: Stock ticker
        platform_score: Platform score (e.g., 71 for BUY 71%) - passed directly to avoid DB query
        platform_signal: Platform signal (e.g., "BUY") - passed directly to avoid DB query
        technical_score: Technical score - passed directly to avoid DB query
    """
    import os
    import pandas as pd
    from src.db.connection import get_engine

    prediction = None
    alpha_model_available = False

    try:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "models", "multi_factor_alpha.pkl"
        )

        if os.path.exists(model_path):
            from src.ml.multi_factor_alpha import MultiFactorAlphaModel

            model = MultiFactorAlphaModel()
            model.load(model_path)

            # Use predict_live with single ticker
            result_df = model.predict_live(tickers=[ticker])

            if not result_df.empty:
                # Get the row for this ticker and convert to dict
                row = result_df.iloc[0]
                prediction = {
                    'signal': row.get('signal', 'HOLD'),
                    'conviction': row.get('conviction', 0.5),
                    'expected_return_5d': row.get('expected_return_5d', 0),
                    'expected_return_10d': row.get('expected_return_10d', 0),
                    'expected_return_20d': row.get('expected_return_20d', 0),
                    'prob_positive_5d': row.get('prob_positive_5d', 0.5),
                    'prob_beat_market_5d': row.get('prob_beat_market_5d', 0.5),
                    'regime': row.get('regime', 'UNKNOWN'),
                    'top_bullish_factors': row.get('top_bullish_factors', []),
                    'top_bearish_factors': row.get('top_bearish_factors', []),
                }
                alpha_model_available = True
    except Exception as e:
        logger.debug(f"Alpha model prediction error for {ticker}: {e}")
        prediction = None
        alpha_model_available = False

    # If no Alpha Model prediction, create a default one so enhancements still work
    if prediction is None:
        prediction = {
            'signal': 'HOLD',
            'conviction': 0.0,
            'expected_return_5d': 0,
            'expected_return_10d': 0,
            'expected_return_20d': 0,
            'prob_positive_5d': 0.5,
            'prob_beat_market_5d': 0.5,
            'regime': 'UNKNOWN',
            'top_bullish_factors': [],
            'top_bearish_factors': [],
        }

    # Use passed-in platform data if available, otherwise query DB
    platform_data_available = platform_score is not None and platform_signal is not None

    if not platform_data_available:
        # Fallback to database query
        try:
            query = """
                SELECT total_score, signal_type, technical_score
                FROM latest_scores
                WHERE ticker = %s
            """
            df = pd.read_sql(query, get_engine(), params=(ticker,))
            if not df.empty:
                prow = df.iloc[0]
                if pd.notna(prow['total_score']) and prow['total_score'] != 0:
                    platform_score = float(prow['total_score'])
                if pd.notna(prow['signal_type']) and prow['signal_type']:
                    platform_signal = str(prow['signal_type'])
                if pd.notna(prow['technical_score']) and prow['technical_score'] != 0:
                    technical_score = float(prow['technical_score'])
                platform_data_available = platform_score is not None and platform_signal is not None
        except Exception as e:
            logger.debug(f"Could not get platform scores: {e}")

    # Apply defaults if still missing
    if platform_score is None:
        platform_score = 50
    if platform_signal is None:
        platform_signal = "HOLD"
    if technical_score is None:
        technical_score = 50

    # Try to use enhanced context
    try:
        from src.ml.alpha_enhancements import build_enhanced_alpha_context

        context = build_enhanced_alpha_context(
            ticker=ticker,
            alpha_prediction=prediction,
            platform_score=platform_score,
            platform_signal=platform_signal,
            technical_score=technical_score
        )

        # Add note if Alpha Model wasn't available
        if not alpha_model_available:
            context = f"⚠️ NOTE: Alpha Model prediction unavailable for {ticker} (new ticker or missing data)\n\n" + context

        if not platform_data_available:
            context = f"⚠️ WARNING: Platform scores unavailable for {ticker} - using conservative defaults\n\n" + context

        return context

    except ImportError:
        # Fallback if enhancements module not available
        logger.warning("alpha_enhancements module not available, using basic format")
    except Exception as e:
        logger.warning(f"Enhancement error, using fallback: {e}")

    # Fallback: basic format without enhancements
    if alpha_model_available:
        context = f"""
🧠 ALPHA MODEL (ML PREDICTION):
- Signal: {prediction.get('signal', 'N/A')} (Conviction: {prediction.get('conviction', 0):.0%})
- Expected 5d Return: {prediction.get('expected_return_5d', 0):+.2%}
- Expected 10d Return: {prediction.get('expected_return_10d', 0):+.2%}
- P(Positive 5d): {prediction.get('prob_positive_5d', 0):.1%}
- P(Beat Market): {prediction.get('prob_beat_market_5d', 0):.1%}
- Market Regime: {prediction.get('regime', 'UNKNOWN')}

⚠️ NOTE: Enhanced context unavailable - showing raw predictions
"""
    else:
        context = f"""
🧠 ALPHA MODEL: No prediction available for {ticker}
   (New ticker or not in training data)

📊 PLATFORM SIGNAL: {platform_signal} ({platform_score:.0f}%)
   Using platform signal only for analysis.
"""
    return context


def _get_options_flow_context(ticker: str, display_price: float = 0) -> str:
    """Get options flow context for AI - fetches LIVE data if DB is stale.

    Args:
        ticker: Stock symbol
        display_price: The price being displayed to user (pre-market/after-hours if available)
                      Used for consistent percentage calculations
    """
    try:
        from src.db.connection import get_engine
        import pandas as pd
        from datetime import datetime, date
        import yfinance as yf

        engine = get_engine()

        # First check DB for recent data (within 1 day)
        query = """
            SELECT total_call_volume, total_put_volume, put_call_volume_ratio,
                   put_call_oi_ratio, overall_sentiment, sentiment_score, 
                   max_pain_price, scan_date, stock_price
            FROM options_flow_daily
            WHERE ticker = %s
            ORDER BY scan_date DESC
            LIMIT 1
        """
        df = pd.read_sql(query, engine, params=(ticker,))

        # Check if data is fresh (within 1 day)
        use_live = True
        db_data = None

        if not df.empty:
            row = df.iloc[0]
            scan_date = row.get('scan_date')
            if scan_date:
                if isinstance(scan_date, str):
                    scan_dt = datetime.strptime(scan_date[:10], "%Y-%m-%d").date()
                elif hasattr(scan_date, 'date'):
                    scan_dt = scan_date.date()
                else:
                    scan_dt = scan_date

                data_age = (date.today() - scan_dt).days
                if data_age <= 1:
                    use_live = False
                    db_data = row

        # Fetch LIVE data if DB is stale or empty
        if use_live:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                stock_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

                # Get options expirations
                expirations = stock.options
                if not expirations:
                    return _format_stale_options_context(df, "No options data available")

                # Get near-term options (up to 4 expiries)
                all_calls = []
                all_puts = []
                expiries_used = []

                for expiry in expirations[:4]:
                    try:
                        opt_chain = stock.option_chain(expiry)
                        all_calls.append(opt_chain.calls)
                        all_puts.append(opt_chain.puts)
                        expiries_used.append(expiry)
                    except:
                        continue

                if not all_calls or not all_puts:
                    return _format_stale_options_context(df, "Could not fetch options data")

                calls_df = pd.concat(all_calls, ignore_index=True)
                puts_df = pd.concat(all_puts, ignore_index=True)

                # Calculate metrics
                call_vol = calls_df['volume'].sum() if 'volume' in calls_df else 0
                put_vol = puts_df['volume'].sum() if 'volume' in puts_df else 0
                call_oi = calls_df['openInterest'].sum() if 'openInterest' in calls_df else 0
                put_oi = puts_df['openInterest'].sum() if 'openInterest' in puts_df else 0

                pc_volume = put_vol / call_vol if call_vol > 0 else 1.0
                pc_oi = put_oi / call_oi if call_oi > 0 else 1.0

                # Calculate max pain for NEAREST EXPIRY ONLY (verifiable against public sources)
                nearest_expiry = expiries_used[0] if expiries_used else None
                max_pain = 0
                max_pain_expiry = "N/A"

                if nearest_expiry:
                    # Get options for just the nearest expiry
                    try:
                        nearest_chain = stock.option_chain(nearest_expiry)
                        max_pain = _calculate_max_pain_live(nearest_chain.calls, nearest_chain.puts)
                        max_pain_expiry = nearest_expiry
                    except:
                        # Fallback to aggregated if single expiry fails
                        max_pain = _calculate_max_pain_live(calls_df, puts_df)
                        max_pain_expiry = f"aggregated ({len(expiries_used)} expiries)"

                # Validate max pain
                # NOTE: AI rules say >20% = data error, so we use 15% as warning, 20% as error
                max_pain_note = ""
                if stock_price > 0 and max_pain > 0:
                    pct_diff = abs(max_pain - stock_price) / stock_price * 100
                    if pct_diff > 20:
                        max_pain_note = f" (⚠️ DATA ERROR: {pct_diff:.0f}% from price - DO NOT USE)"
                    elif pct_diff > 15:
                        max_pain_note = f" (⚠️ {pct_diff:.0f}% from price - verify)"

                # Determine sentiment
                if pc_volume < 0.5:
                    sentiment = "Bullish bias (high call volume vs puts)"
                elif pc_volume < 0.7:
                    sentiment = "Moderately bullish"
                elif pc_volume < 1.0:
                    sentiment = "Neutral to slightly bullish"
                elif pc_volume < 1.3:
                    sentiment = "Neutral to slightly bearish"
                else:
                    sentiment = "Bearish bias (high put volume vs calls)"

                # Calculate max pain distance from DISPLAY price (pre-market if available)
                price_for_calc = display_price if display_price > 0 else stock_price
                max_pain_distance = ((max_pain - price_for_calc) / price_for_calc * 100) if price_for_calc > 0 and max_pain > 0 else 0

                # Get current timestamp (timezone-aware UTC)
                from datetime import datetime, timezone
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

                context = f"""
OPTIONS FLOW (LIVE data, {timestamp}):
- Stock Price: ${stock_price:.2f} (regular close)
- Call Volume: {call_vol:,.0f}
- Put Volume: {put_vol:,.0f}
- P/C Ratio (volume): {pc_volume:.2f} - {sentiment}
- P/C Ratio (OI): {pc_oi:.2f}
- Max Pain: ${max_pain:.2f} for {max_pain_expiry} ({max_pain_distance:+.1f}% from ${price_for_calc:.2f}){max_pain_note}

⚠️ NOTES:
- P/C ratio is VOLUME-based (OI-based may differ)
- Max pain shown for nearest expiry ({max_pain_expiry}) - verify with public sources
- Low P/C does NOT prove institutional buying
"""
                return context

            except Exception as e:
                logger.debug(f"Live options fetch error: {e}")
                # Fall back to DB data with warning
                return _format_stale_options_context(df, f"Live fetch failed: {e}")

        else:
            # Use fresh DB data
            row = db_data
            # BUG FIX: Use "is None" check - P/C ratio of 0.0 is valid (extremely bearish)
            # "or 1.0" would incorrectly convert 0.0 to 1.0 (neutral)
            pc_val = row.get('put_call_volume_ratio')
            put_call = pc_val if pc_val is not None else 1.0
            pc_oi_val = row.get('put_call_oi_ratio')
            put_call_oi = pc_oi_val if pc_oi_val is not None else 1.0
            call_vol = row.get('total_call_volume', 0) or 0
            put_vol = row.get('total_put_volume', 0) or 0
            max_pain = row.get('max_pain_price', 0) or 0
            stock_price = row.get('stock_price', 0) or 0
            scan_date = row.get('scan_date', '')

            # Format date
            date_str = scan_date.strftime("%Y-%m-%d") if hasattr(scan_date, 'strftime') else str(scan_date)[:10]

            # Sentiment interpretation
            if put_call < 0.5:
                sentiment = "Bullish bias (high call volume vs puts)"
            elif put_call < 0.7:
                sentiment = "Moderately bullish"
            elif put_call < 1.0:
                sentiment = "Neutral to slightly bullish"
            elif put_call < 1.3:
                sentiment = "Neutral to slightly bearish"
            else:
                sentiment = "Bearish bias (high put volume vs calls)"

            # Calculate max pain distance from DISPLAY price (pre-market if available)
            price_for_calc = display_price if display_price > 0 else stock_price
            max_pain_distance = ((max_pain - price_for_calc) / price_for_calc * 100) if price_for_calc > 0 and max_pain > 0 else 0

            # Validate max pain
            # NOTE: AI rules say >20% = data error, so we use 15% as warning, 20% as error
            max_pain_str = f"${max_pain:.2f} ({max_pain_distance:+.1f}% from current ${price_for_calc:.2f})"
            if price_for_calc > 0 and max_pain > 0:
                pct_diff = abs(max_pain - price_for_calc) / price_for_calc * 100
                if pct_diff > 20:
                    max_pain_str = f"${max_pain:.2f} ({max_pain_distance:+.1f}% - ⚠️ DATA ERROR >20% from ${price_for_calc:.2f})"
                elif pct_diff > 15:
                    max_pain_str = f"${max_pain:.2f} ({max_pain_distance:+.1f}% - ⚠️ verify, >15% from ${price_for_calc:.2f})"

            context = f"""
OPTIONS FLOW (platform data, {date_str}):
- Call Volume: {call_vol:,.0f}
- Put Volume: {put_vol:,.0f}
- P/C Ratio (volume): {put_call:.2f} - {sentiment}
- P/C Ratio (OI): {put_call_oi:.2f}
- Max Pain: {max_pain_str}

⚠️ NOTES:
- P/C ratio is VOLUME-based (OI-based may differ)
- Low P/C does NOT prove institutional buying
"""
            return context

    except Exception as e:
        logger.debug(f"Options flow context error: {e}")
        return ""


def _calculate_max_pain_live(calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> float:
    """
    Calculate max pain from live options data.

    Max pain = strike where total payout to option holders is MINIMUM
    (i.e., where most options expire worthless, causing max pain to holders)

    At strike X:
    - Calls with strike < X are ITM: payout = (X - strike) * OI
    - Calls with strike >= X are OTM: payout = 0
    - Puts with strike > X are ITM: payout = (strike - X) * OI
    - Puts with strike <= X are OTM: payout = 0

    We want the strike X that MINIMIZES total payout.
    """
    if calls_df.empty or puts_df.empty:
        return 0

    all_strikes = sorted(set(
        list(calls_df['strike'].unique()) +
        list(puts_df['strike'].unique())
    ))

    if not all_strikes:
        return 0

    min_payout = float('inf')
    max_pain_strike = all_strikes[0]

    for test_strike in all_strikes:
        total_payout = 0

        # ITM Calls: strike < test_strike, payout = (test_strike - strike) * OI
        for _, row in calls_df.iterrows():
            strike = row['strike']
            oi = row.get('openInterest', 0) or 0
            if strike < test_strike:
                total_payout += oi * (test_strike - strike)

        # ITM Puts: strike > test_strike, payout = (strike - test_strike) * OI
        for _, row in puts_df.iterrows():
            strike = row['strike']
            oi = row.get('openInterest', 0) or 0
            if strike > test_strike:
                total_payout += oi * (strike - test_strike)

        if total_payout < min_payout:
            min_payout = total_payout
            max_pain_strike = test_strike

    return max_pain_strike


def _format_stale_options_context(df: pd.DataFrame, error_msg: str) -> str:
    """Format stale DB data with warning."""
    if df.empty:
        return f"""
OPTIONS FLOW: ⚠️ NO DATA AVAILABLE
Error: {error_msg}
"""

    row = df.iloc[0]
    scan_date = row.get('scan_date', 'Unknown')
    date_str = scan_date.strftime("%Y-%m-%d") if hasattr(scan_date, 'strftime') else str(scan_date)[:10]

    return f"""
OPTIONS FLOW (⚠️ STALE DATA from {date_str}):
- P/C Ratio (volume): {row.get('put_call_volume_ratio', 'N/A')}
- Max Pain: ${row.get('max_pain_price', 0):.2f}
⛔ DATA IS STALE - verify with real-time sources
Error: {error_msg}
"""


def _get_insider_context(ticker: str) -> str:
    """Get insider trading context for AI."""
    try:
        from src.db.connection import get_engine
        import pandas as pd

        engine = get_engine()
        query = """
            SELECT insider_name, title, transaction_type, shares, 
                   price_per_share, total_value, transaction_date
            FROM insider_transactions
            WHERE ticker = %s
            AND transaction_date >= NOW() - INTERVAL '90 days'
            ORDER BY transaction_date DESC
            LIMIT 5
        """
        df = pd.read_sql(query, engine, params=(ticker,))

        if df.empty:
            return ""

        # Calculate summary
        buys = df[df['transaction_type'] == 'P']
        sells = df[df['transaction_type'] == 'S']

        buy_value = buys['total_value'].sum() if not buys.empty else 0
        sell_value = sells['total_value'].sum() if not sells.empty else 0

        context = f"""
INSIDER ACTIVITY (90 days):
- Total Buys: {len(buys)} transactions (${buy_value:,.0f})
- Total Sells: {len(sells)} transactions (${sell_value:,.0f})
- Net: {'BUYING' if buy_value > sell_value else 'SELLING' if sell_value > buy_value else 'NEUTRAL'}
Recent Transactions:
"""
        for _, row in df.head(3).iterrows():
            tx_type = "BUY" if row['transaction_type'] == 'P' else "SELL"
            context += f"  - {tx_type}: {row['insider_name'][:20]} - ${row.get('total_value', 0):,.0f}\n"

        return context
    except Exception as e:
        logger.debug(f"Insider context error: {e}")
        return ""


def _get_squeeze_context(ticker: str) -> str:
    """Get short squeeze context for AI."""
    try:
        from src.analytics.short_squeeze import get_squeeze_report
        report = get_squeeze_report(ticker)
        if report:
            return f"\n{report}\n"
        return ""
    except Exception as e:
        logger.debug(f"Squeeze context error: {e}")
        return ""


def _get_fundamentals_context(ticker: str) -> str:
    """Get fundamentals context for AI."""
    try:
        from src.db.connection import get_engine
        import pandas as pd

        engine = get_engine()
        query = """
            SELECT market_cap, pe_ratio, forward_pe, pb_ratio, ps_ratio,
                   profit_margin, roe, roa, revenue_growth, earnings_growth,
                   dividend_yield, debt_to_equity, current_ratio, free_cash_flow
            FROM fundamentals
            WHERE ticker = %s
            ORDER BY date DESC
            LIMIT 1
        """
        df = pd.read_sql(query, engine, params=(ticker,))

        if df.empty:
            return ""

        row = df.iloc[0]

        # Helper to normalize percentage values from DB
        def normalize_pct(val):
            if val is None:
                return 0
            val = float(val)
            # If > 0.50 (50%), likely stored as percentage not decimal
            if val > 0.50:
                val = val / 100
            # If still > 0.50, divide again (handles 348% -> 3.48%)
            if val > 0.50:
                val = val / 100
            return val

        # Format market cap
        market_cap = row.get('market_cap', 0) or 0
        if market_cap >= 1e12:
            mc_str = f"${market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            mc_str = f"${market_cap/1e9:.2f}B"
        else:
            mc_str = f"${market_cap/1e6:.2f}M"

        # Normalize percentages
        profit_margin = normalize_pct(row.get('profit_margin', 0))
        roe = normalize_pct(row.get('roe', 0))
        roa = normalize_pct(row.get('roa', 0))
        revenue_growth = normalize_pct(row.get('revenue_growth', 0))
        earnings_growth = normalize_pct(row.get('earnings_growth', 0))
        dividend_yield = normalize_pct(row.get('dividend_yield', 0))

        # FIX: Debt/Equity - Yahoo/providers return as percentage, convert properly
        debt_equity_raw = row.get('debt_to_equity', 0) or 0
        # If value > 1, it's stored as percentage (e.g., 15.67 = 15.67%)
        if debt_equity_raw > 1:
            debt_equity_ratio = debt_equity_raw / 100  # Convert to ratio (0.1567)
            debt_equity_pct = debt_equity_raw  # Keep percentage (15.67%)
        else:
            debt_equity_ratio = debt_equity_raw  # Already a ratio
            debt_equity_pct = debt_equity_raw * 100  # Convert to percentage

        # Interpret leverage level for context
        if debt_equity_ratio < 0.3:
            leverage_desc = "LOW leverage - strong balance sheet"
        elif debt_equity_ratio < 0.6:
            leverage_desc = "MODERATE leverage"
        elif debt_equity_ratio < 1.0:
            leverage_desc = "ELEVATED leverage"
        else:
            leverage_desc = "HIGH leverage - monitor debt burden"

        context = f"""
FUNDAMENTALS (from platform database - may differ slightly from real-time sources):
- Market Cap: {mc_str}
- P/E Ratio: {row.get('pe_ratio', 'N/A')}
- Forward P/E: {row.get('forward_pe', 'N/A')}
- P/B Ratio: {row.get('pb_ratio', 'N/A')}
- P/S Ratio: {row.get('ps_ratio', 'N/A')}
- Profit Margin: {profit_margin*100:.1f}%
- ROE: {roe*100:.1f}%
- ROA: {roa*100:.1f}%
- Revenue Growth: {revenue_growth*100:.1f}%
- Earnings Growth: {earnings_growth*100:.1f}%
- Dividend Yield: {dividend_yield*100:.2f}%
- Debt/Equity: {debt_equity_ratio:.2f}x ({debt_equity_pct:.1f}%) - {leverage_desc}
- Current Ratio: {row.get('current_ratio', 'N/A')}
"""
        return context
    except Exception as e:
        logger.debug(f"Fundamentals context error: {e}")
        return ""


def _get_analyst_context(ticker: str) -> str:
    """Get analyst ratings context for AI."""
    try:
        from src.db.connection import get_engine
        import pandas as pd

        engine = get_engine()

        # Get analyst ratings
        query = """
            SELECT buy_count, hold_count, sell_count, 
                   strong_buy_count, strong_sell_count
            FROM analyst_ratings
            WHERE ticker = %s
            ORDER BY date DESC
            LIMIT 1
        """
        df = pd.read_sql(query, engine, params=(ticker,))

        # Get price targets
        pt_query = """
            SELECT target_mean, target_high, target_low, current_price,
                   upside_pct, analyst_count
            FROM price_targets
            WHERE ticker = %s
            ORDER BY date DESC
            LIMIT 1
        """
        pt_df = pd.read_sql(pt_query, engine, params=(ticker,))

        context = ""

        if not df.empty:
            row = df.iloc[0]
            total = (row.get('strong_buy_count', 0) or 0) + (row.get('buy_count', 0) or 0) + \
                    (row.get('hold_count', 0) or 0) + (row.get('sell_count', 0) or 0) + \
                    (row.get('strong_sell_count', 0) or 0)

            context += f"""
ANALYST RATINGS:
- Strong Buy: {row.get('strong_buy_count', 0)}
- Buy: {row.get('buy_count', 0)}
- Hold: {row.get('hold_count', 0)}
- Sell: {row.get('sell_count', 0)}
- Strong Sell: {row.get('strong_sell_count', 0)}
- Total Analysts: {total}
"""

        if not pt_df.empty:
            pt = pt_df.iloc[0]
            context += f"""
PRICE TARGETS:
- Mean Target: ${pt.get('target_mean', 0):.2f}
- High Target: ${pt.get('target_high', 0):.2f}
- Low Target: ${pt.get('target_low', 0):.2f}
- Upside: {pt.get('upside_pct', 0):.1f}%
- # Analysts: {pt.get('analyst_count', 0)}
"""

        return context
    except Exception as e:
        logger.debug(f"Analyst context error: {e}")
        return ""


def _get_ai_response(question: str, context: str, ticker: str) -> tuple:
    """
    Get AI response using configured AI model with STRICT prompting to prevent hallucinations.

    Returns:
        tuple: (response_text, model_name)
    """

    # STRICT PROMPT - prevents LLM from inventing data or doing its own math
    prompt = f"""You are a financial analyst assistant. Your role is to EXPLAIN pre-computed data, NOT to calculate or invent.

=== STRICT RULES (MUST FOLLOW) ===
1. Use ONLY numbers that appear in the DATA CONTEXT below
2. If a metric shows "UNKNOWN" or "N/A", write "UNKNOWN" - do NOT guess or estimate
3. NEVER invent dates for events (OPEC, Fed, earnings) unless explicitly listed
4. NEVER claim "institutional buying/selling" from P/C ratios alone
5. NEVER recalculate percentages - use the pre-computed values exactly as shown
6. NEVER add events, catalysts, or news not mentioned in the context
7. If asked about something not in the data, say "This information is not available in the current data"
8. For entry/stop/target, use the Decision Policy values - do not invent your own levels

=== PROHIBITED PHRASES ===
- "institutional buying/selling" (unless attributed to specific trade data)
- "smart money"
- Made-up dates like "OPEC meeting Jan X" (unless in CATALYSTS)
- Made-up indicators like "RSI shows..." (unless RSI is in data)

=== DATA CONTEXT ===
{context}

=== QUESTION ===
{question}

=== RESPONSE FORMAT ===
Provide a clear, structured response using ONLY the data above. Include:
1. Direct answer to the question
2. Supporting evidence (cite specific numbers from the data)
3. Risks and caveats
4. If any information is missing, explicitly state "Data not available" for that item

ANSWER:"""

    model_name = "Unknown"
    selected_model = st.session_state.get('ai_model', 'qwen_local')
    attempted_model = selected_model  # Track what we tried

    logger.info(f"AI Request - Selected model: {selected_model}")

    try:
        # Try to use AI settings component first (if available)
        try:
            from src.components.ai_settings import get_current_model_response, get_current_model

            model = get_current_model()
            if model:
                model_name = f"{model.icon} {model.name}"
                logger.info(f"Using ai_settings component: {model_name}")
                response = ""
                result = get_current_model_response(prompt, stream=False)
                if isinstance(result, str):
                    response = result
                else:
                    for chunk in result:
                        response += chunk

                # Clean thinking tags
                if '</think>' in response:
                    response = response.split('</think>')[-1].strip()

                return response, model_name
        except ImportError:
            logger.debug("ai_settings not installed - using direct API calls")
        except Exception as e:
            logger.warning(f"ai_settings error: {e}, falling back to direct API")

        # Direct API calls based on selected model
        import os

        # Try to get model config from ai_models_config
        try:
            from src.components.ai_models_config import get_all_models, get_model_api_id
            all_models = get_all_models()

            if selected_model in all_models:
                model_config = all_models[selected_model]
                provider = model_config.get("provider", "qwen")
                api_id = model_config.get("api_id", selected_model)
                model_name = f"{model_config.get('icon', '🤖')} {model_config.get('name', selected_model)}"
                api_key_env = model_config.get("api_key_env")

                logger.info(f"Using model config: {selected_model} -> {api_id} ({provider})")

                # OpenAI provider
                if provider == "openai":
                    api_key = os.getenv("OPENAI_API_KEY", "")
                    if not api_key:
                        return f"❌ OpenAI API key not set. Add OPENAI_API_KEY to your .env file.", f"❌ {model_name} (No API Key)"

                    try:
                        from openai import OpenAI
                        logger.info(f"Calling OpenAI API: {api_id}")
                        client = OpenAI(api_key=api_key, timeout=120)

                        # GPT-5.x and o-series use max_completion_tokens, older models use max_tokens
                        is_new_model = api_id.startswith(('gpt-5', 'o1', 'o3', 'o4'))

                        if is_new_model:
                            response = client.chat.completions.create(
                                model=api_id,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.15,
                                max_completion_tokens=3000
                            )
                        else:
                            response = client.chat.completions.create(
                                model=api_id,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.15,
                                max_tokens=3000
                            )

                        result = response.choices[0].message.content

                        if '</think>' in result:
                            result = result.split('</think>')[-1].strip()

                        logger.info(f"OpenAI response received from {api_id}")
                        return result, model_name

                    except Exception as e:
                        error_msg = str(e)[:100]
                        logger.error(f"OpenAI API error: {error_msg}")
                        return f"❌ OpenAI API Error: {error_msg}", f"❌ {model_name} (Error)"

                # Anthropic provider
                elif provider == "anthropic":
                    api_key = os.getenv("ANTHROPIC_API_KEY", "")
                    if not api_key:
                        return f"❌ Anthropic API key not set. Add ANTHROPIC_API_KEY to your .env file.", f"❌ {model_name} (No API Key)"

                    try:
                        import anthropic
                        logger.info(f"Calling Anthropic API: {api_id}")
                        client = anthropic.Anthropic(api_key=api_key)

                        response = client.messages.create(
                            model=api_id,
                            max_tokens=3000,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        result = response.content[0].text

                        logger.info(f"Anthropic response received from {api_id}")
                        return result, model_name

                    except ImportError:
                        return "❌ Anthropic package not installed. Run: pip install anthropic", f"❌ {model_name} (Package Missing)"
                    except Exception as e:
                        error_msg = str(e)[:100]
                        logger.error(f"Anthropic API error: {error_msg}")
                        return f"❌ Anthropic API Error: {error_msg}", f"❌ {model_name} (Error)"

                # Ollama provider
                elif provider == "ollama":
                    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

                    try:
                        from openai import OpenAI
                        logger.info(f"Calling Ollama API: {api_id}")
                        client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama", timeout=120)

                        response = client.chat.completions.create(
                            model=api_id,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.15
                        )
                        result = response.choices[0].message.content

                        if '</think>' in result:
                            result = result.split('</think>')[-1].strip()

                        logger.info(f"Ollama response received from {api_id}")
                        return result, model_name

                    except Exception as e:
                        error_msg = str(e)[:100]
                        logger.error(f"Ollama API error: {error_msg}")
                        return f"❌ Ollama Error: {error_msg}\n\nMake sure Ollama is running at {base_url}", f"❌ {model_name} (Error)"

                # Local Qwen provider - handle here instead of falling through
                elif provider == "qwen":
                    logger.info(f"Using local Qwen via config (selected: {selected_model})")
                    from src.ai.chat import AlphaChat

                    if 'alpha_chat' not in st.session_state:
                        st.session_state.alpha_chat = AlphaChat()

                    chat = st.session_state.alpha_chat

                    # Get model name from chat config
                    if hasattr(chat, 'config') and hasattr(chat.config, 'model'):
                        model_name = f"🏠 {chat.config.model}"
                    else:
                        model_name = "🏠 Qwen (Local)"

                    if chat.available:
                        response = ""
                        for chunk in chat.chat_stream(prompt, ticker=ticker):
                            response += chunk

                        # Clean thinking tags
                        if '</think>' in response:
                            response = response.split('</think>')[-1].strip()

                        logger.info(f"Qwen response received")
                        return response, model_name
                    else:
                        logger.error("Qwen server not available")
                        return _fallback_response(context, ticker), "❌ Qwen Unavailable"

        except ImportError:
            logger.debug("ai_models_config not available, using fallback")

        # Default fallback: Local Qwen via AlphaChat (when config not available)
        logger.info(f"Using local Qwen fallback (selected: {selected_model})")
        from src.ai.chat import AlphaChat

        if 'alpha_chat' not in st.session_state:
            st.session_state.alpha_chat = AlphaChat()

        chat = st.session_state.alpha_chat

        # Get model name from chat config
        if hasattr(chat, 'config') and hasattr(chat.config, 'model'):
            model_name = f"🏠 {chat.config.model}"
        else:
            model_name = "🏠 Qwen (Local)"

        if chat.available:
            response = ""
            for chunk in chat.chat_stream(prompt, ticker=ticker):
                response += chunk

            # Clean thinking tags
            if '</think>' in response:
                response = response.split('</think>')[-1].strip()

            logger.info(f"Qwen response received")
            return response, model_name
        else:
            logger.error("Qwen server not available")
            return _fallback_response(context, ticker), "❌ Qwen Unavailable"

    except Exception as e:
        logger.error(f"AI response error (attempted: {attempted_model}): {e}")
        return f"❌ Error with {attempted_model}: {str(e)[:100]}", f"❌ {attempted_model} (Failed)"


def _fallback_response(context: str, ticker: str) -> str:
    """Fallback when AI not available."""
    return f"""**{ticker} Summary** (AI unavailable)

Based on the data:
{context[:500]}...

*Start Qwen server for full AI analysis:*
```
llama-server.exe -m qwen3-32b-q6_k.gguf -c 32768 -ngl 99 --port 8080
```
"""


def _run_single_analysis(ticker: str):
    """
    Run FULL analysis for single ticker - refresh everything automatically.

    This function:
    1. Fetches fresh news
    2. Analyzes sentiment
    3. Gets options flow, fundamentals, technical scores
    4. Creates/updates a row for TODAY in screener_scores
    5. Clears all caches
    6. Forces page refresh to show new data
    """
    with st.spinner(f"🔄 Refreshing all data for {ticker}..."):
        try:
            from src.data.news import NewsCollector
            from src.screener.sentiment import SentimentAnalyzer
            from datetime import date as dt_date
            from src.db.connection import get_connection

            status_container = st.empty()
            results_container = st.container()
            today = dt_date.today()

            # =====================================================================
            # STEP 1: Collect Fresh News
            # =====================================================================
            status_container.info(f"📰 Step 1/6: Collecting fresh news for {ticker}...")
            nc = NewsCollector()
            result = nc.collect_and_save(ticker, days_back=7, force_refresh=True)
            articles = result.get('articles', [])
            saved_count = result.get('saved', 0)

            # =====================================================================
            # STEP 2: Analyze Sentiment
            # =====================================================================
            status_container.info(f"🧠 Step 2/6: Analyzing sentiment ({len(articles)} articles)...")
            sentiment_score = None  # FIX: Use None, not 50
            sentiment_data = {}
            if articles:
                sa = SentimentAnalyzer()
                sentiment_data = sa.analyze_ticker_sentiment(ticker, articles)
                sentiment_score = sentiment_data.get('sentiment_score')  # FIX: No default

            # =====================================================================
            # STEP 3: Get Options Flow & Other Scores
            # =====================================================================
            status_container.info(f"📊 Step 3/6: Updating options flow & fundamentals...")
            options_score = None
            squeeze_score = None
            fundamental_score = None
            technical_score = None
            growth_score = None
            dividend_score = None

            try:
                from src.analytics.universe_scorer import UniverseScorer
                scorer = UniverseScorer(skip_ibkr=True)
                scores_list, _ = scorer.score_and_save_universe(tickers=[ticker], max_workers=1)

                # FIX: scores_list is a List[UniverseScores], not a dict!
                if scores_list:
                    for score_obj in scores_list:
                        if score_obj.ticker == ticker:
                            # UniverseScores only has options/squeeze, not fundamental/technical
                            options_score = score_obj.options_flow_score
                            squeeze_score = score_obj.short_squeeze_score
                            break
            except Exception as e:
                logger.warning(f"Universe scorer error for {ticker}: {e}")

            # Get Fundamental Score separately
            try:
                # FundamentalAnalyzer not implemented - scores come from screener
                pass
            except Exception as e:
                logger.debug(f"{ticker}: Fundamental analysis skipped - {e}")

            # Get Technical Score separately
            try:
                from src.analytics.technical_analysis import TechnicalAnalyzer
                ta = TechnicalAnalyzer()
                tech_result = ta.analyze_ticker(ticker)
                if tech_result:
                    technical_score = tech_result.get('technical_score', tech_result.get('score'))
            except Exception as e:
                logger.debug(f"{ticker}: Technical analysis skipped - {e}")

            # =====================================================================
            # STEP 4: Update sentiment_scores table (UPSERT)
            # =====================================================================
            status_container.info(f"💾 Step 4/6: Saving sentiment scores...")

            # Determine sentiment class
            if sentiment_score >= 70:
                sentiment_class = 'Very Bullish'
            elif sentiment_score >= 55:
                sentiment_class = 'Bullish'
            elif sentiment_score >= 45:
                sentiment_class = 'Neutral'
            elif sentiment_score >= 30:
                sentiment_class = 'Bearish'
            else:
                sentiment_class = 'Very Bearish'

            with get_connection() as conn:
                with conn.cursor() as cur:
                    # UPSERT sentiment_scores
                    cur.execute("""
                        INSERT INTO sentiment_scores
                        (ticker, date, sentiment_raw, sentiment_weighted, ai_sentiment_fast,
                         article_count, relevant_article_count, sentiment_class)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) 
                        ON CONFLICT (ticker, date) DO UPDATE SET
                            sentiment_raw = EXCLUDED.sentiment_raw,
                            sentiment_weighted = EXCLUDED.sentiment_weighted,
                            ai_sentiment_fast = EXCLUDED.ai_sentiment_fast,
                            article_count = EXCLUDED.article_count,
                            relevant_article_count = EXCLUDED.relevant_article_count,
                            sentiment_class = EXCLUDED.sentiment_class
                    """, (
                        ticker,
                        today,
                        sentiment_score,
                        sentiment_data.get('sentiment_weighted', sentiment_score),
                        sentiment_score,
                        len(articles),
                        sentiment_data.get('relevant_count', len(articles)),
                        sentiment_class
                    ))
                conn.commit()

            # =====================================================================
            # STEP 5: Create/Update TODAY's row in screener_scores
            # =====================================================================
            status_container.info(f"💾 Step 5/6: Updating screener_scores for today...")

            with get_connection() as conn:
                with conn.cursor() as cur:
                    # Check if today's row exists
                    cur.execute("""
                        SELECT COUNT(*) FROM screener_scores 
                        WHERE ticker = %s AND date = %s
                    """, (ticker, today))
                    today_exists = cur.fetchone()[0] > 0

                    if today_exists:
                        # Update existing row for today
                        logger.info(f"{ticker}: Updating existing screener_scores row for {today}")

                        # First get current values to compute total_score
                        cur.execute("""
                            SELECT sentiment_score, fundamental_score, technical_score, 
                                   options_flow_score, short_squeeze_score
                            FROM screener_scores WHERE ticker = %s AND date = %s
                        """, (ticker, today))
                        current = cur.fetchone()

                        # Compute fresh total_score using new values where available
                        # BUG FIX: Use "is not None" - score of 0 is valid!
                        final_sent = sentiment_score if sentiment_score is not None else (current[0] if current else 50)
                        final_fund = fundamental_score if fundamental_score is not None else (current[1] if current else 50)
                        final_tech = technical_score if technical_score is not None else (current[2] if current else 50)
                        final_opts = options_score if options_score is not None else (current[3] if current else 50)
                        final_sqz = squeeze_score if squeeze_score is not None else (current[4] if current else 50)

                        # Weighted average: sent=25%, fund=25%, tech=25%, opts=15%, sqz=10%
                        # BUG FIX: Use round() instead of int() to avoid systematic downward bias
                        base_total = round(
                            final_sent * 0.25 + final_fund * 0.25 + final_tech * 0.25 +
                            final_opts * 0.15 + final_sqz * 0.10
                        )
                        # Clamp to valid range
                        base_total = max(0, min(100, base_total))

                        # Apply enhanced scoring if available
                        computed_total = base_total
                        if ENHANCED_SCORING_AVAILABLE:
                            try:
                                enhanced_score, adjustment, _ = get_enhanced_total_score(
                                    ticker=ticker,
                                    base_score=base_total,
                                    row_data={},
                                )
                                computed_total = enhanced_score
                                if adjustment != 0:
                                    logger.info(f"{ticker}: Enhanced score {base_total} → {computed_total} (adj: {adjustment:+d})")
                            except:
                                pass

                        cur.execute("""
                            UPDATE screener_scores SET
                                sentiment_score = %s,
                                article_count = %s,
                                sentiment_weighted = %s,
                                options_flow_score = COALESCE(%s, options_flow_score),
                                short_squeeze_score = COALESCE(%s, short_squeeze_score),
                                fundamental_score = COALESCE(%s, fundamental_score),
                                technical_score = COALESCE(%s, technical_score),
                                growth_score = COALESCE(%s, growth_score),
                                dividend_score = COALESCE(%s, dividend_score),
                                total_score = %s
                            WHERE ticker = %s AND date = %s
                        """, (
                            _to_native(sentiment_score),
                            _to_native(len(articles)),
                            _to_native(sentiment_data.get('sentiment_weighted', sentiment_score)),
                            _to_native(options_score),
                            _to_native(squeeze_score),
                            _to_native(fundamental_score),
                            _to_native(technical_score),
                            _to_native(growth_score),
                            _to_native(dividend_score),
                            _to_native(computed_total),
                            ticker,
                            today
                        ))
                    else:
                        # Create new row for today by copying from most recent row
                        logger.info(f"{ticker}: Creating new screener_scores row for {today} (copying from most recent)")

                        # First, get the most recent row to copy base values
                        cur.execute("""
                            SELECT * FROM screener_scores 
                            WHERE ticker = %s 
                            ORDER BY date DESC LIMIT 1
                        """, (ticker,))

                        existing_row = cur.fetchone()

                        if existing_row:
                            # Get column names
                            col_names = [desc[0] for desc in cur.description]
                            existing_data = dict(zip(col_names, existing_row))

                            # FIX Issue #4: Compute total_score fresh
                            # FIX: Use None instead of 50 for missing data
                            final_sent = sentiment_score if sentiment_score is not None else existing_data.get('sentiment_score')
                            final_fund = fundamental_score if fundamental_score is not None else existing_data.get('fundamental_score')
                            final_tech = technical_score if technical_score is not None else existing_data.get('technical_score')
                            final_opts = options_score if options_score is not None else existing_data.get('options_flow_score')
                            final_sqz = squeeze_score if squeeze_score is not None else existing_data.get('short_squeeze_score')
                            final_growth = growth_score if growth_score is not None else existing_data.get('growth_score')
                            final_div = dividend_score if dividend_score is not None else existing_data.get('dividend_score')

                            # Weighted average: only include available scores
                            available_scores = []
                            available_weights = []
                            if final_sent is not None:
                                available_scores.append(final_sent * 0.25)
                                available_weights.append(0.25)
                            if final_fund is not None:
                                available_scores.append(final_fund * 0.25)
                                available_weights.append(0.25)
                            if final_tech is not None:
                                available_scores.append(final_tech * 0.25)
                                available_weights.append(0.25)
                            if final_opts is not None:
                                available_scores.append(final_opts * 0.15)
                                available_weights.append(0.15)
                            if final_sqz is not None:
                                available_scores.append(final_sqz * 0.10)
                                available_weights.append(0.10)

                            if available_weights:
                                total_weight = sum(available_weights)
                                # BUG FIX: Use round() instead of int() to avoid systematic downward bias
                                base_total = round(sum(available_scores) / total_weight) if total_weight > 0 else None
                                if base_total is not None:
                                    base_total = max(0, min(100, base_total))  # Clamp to valid range
                            else:
                                base_total = None

                            # Apply enhanced scoring if available
                            computed_total = base_total
                            if ENHANCED_SCORING_AVAILABLE:
                                try:
                                    enhanced_row_data = {
                                        'price': existing_data.get('price'),
                                        'target_mean': existing_data.get('target_mean'),
                                        'pe_ratio': existing_data.get('pe_ratio'),
                                        'forward_pe': existing_data.get('forward_pe'),
                                        'peg_ratio': existing_data.get('peg_ratio'),
                                        'sector': existing_data.get('sector'),
                                        'buy_count': existing_data.get('buy_count', 0),
                                        'total_ratings': existing_data.get('total_ratings', 0),
                                    }
                                    enhanced_score, adjustment, _ = get_enhanced_total_score(
                                        ticker=ticker,
                                        base_score=base_total,
                                        row_data=enhanced_row_data,
                                    )
                                    computed_total = enhanced_score
                                    if adjustment != 0:
                                        logger.info(f"{ticker}: Enhanced score {base_total} → {computed_total} (adj: {adjustment:+d})")
                                except:
                                    pass

                            # FIX Issue #5: Track data freshness
                            # BUG FIX: Use "is not None" instead of truthiness
                            # Because score of 0 is valid but would be treated as "not fresh"
                            fresh_count = sum([
                                1 if sentiment_score is not None else 0,
                                1 if fundamental_score is not None else 0,
                                1 if technical_score is not None else 0,
                                1 if options_score is not None else 0,
                            ])
                            if fresh_count < 2:
                                logger.warning(f"{ticker}: Low data quality - only {fresh_count}/4 scores are fresh")

                            # Insert new row with COMPUTED total_score
                            cur.execute("""
                                INSERT INTO screener_scores (
                                    ticker, date, 
                                    sentiment_score, sentiment_weighted, article_count,
                                    fundamental_score, technical_score, growth_score, dividend_score,
                                    gap_score, likelihood_score, total_score,
                                    options_flow_score, short_squeeze_score, options_sentiment, squeeze_risk,
                                    created_at
                                ) VALUES (
                                    %s, %s,
                                    %s, %s, %s,
                                    %s, %s, %s, %s,
                                    %s, %s, %s,
                                    %s, %s, %s, %s,
                                    NOW()
                                )
                            """, (
                                ticker, today,
                                _to_native(final_sent),
                                _to_native(sentiment_data.get('sentiment_weighted', final_sent)),
                                _to_native(len(articles)),
                                _to_native(final_fund),
                                _to_native(final_tech),
                                _to_native(final_growth),
                                _to_native(final_div),
                                _to_native(existing_data.get('gap_score', 50)),
                                _to_native(existing_data.get('likelihood_score', 50)),
                                _to_native(computed_total),
                                _to_native(final_opts),
                                _to_native(final_sqz),
                                _to_native(existing_data.get('options_sentiment')),
                                _to_native(existing_data.get('squeeze_risk')),
                            ))
                            logger.info(f"{ticker}: Created new row for {today} with fresh total_score={computed_total}")

                            # Save enhanced scores for fast deep-dive loading
                            if ENHANCED_SCORES_DB_AVAILABLE:
                                try:
                                    compute_and_save_enhanced_scores(ticker=ticker, row_data=existing_data)
                                except Exception as e:
                                    logger.debug(f"{ticker}: Could not save enhanced scores: {e}")
                        else:
                            # No existing row at all - create a new one with computed total_score
                            logger.warning(f"{ticker}: No existing screener_scores rows, creating new row")

                            # Use available values or default to 50
                            # BUG FIX: Use "is not None" - score of 0 is valid!
                            final_sent = sentiment_score if sentiment_score is not None else 50
                            final_fund = fundamental_score if fundamental_score is not None else 50
                            final_tech = technical_score if technical_score is not None else 50
                            final_opts = options_score if options_score is not None else 50
                            final_sqz = squeeze_score if squeeze_score is not None else 50
                            final_growth = growth_score if growth_score is not None else 50
                            final_div = dividend_score if dividend_score is not None else 50

                            # Compute base total_score
                            # BUG FIX: Use round() instead of int() to avoid systematic downward bias
                            base_total = round(
                                final_sent * 0.25 + final_fund * 0.25 + final_tech * 0.25 +
                                final_opts * 0.15 + final_sqz * 0.10
                            )
                            # Clamp to valid range
                            base_total = max(0, min(100, base_total))

                            # Apply enhanced scoring if available
                            computed_total = base_total
                            if ENHANCED_SCORING_AVAILABLE:
                                try:
                                    enhanced_score, adjustment, _ = get_enhanced_total_score(
                                        ticker=ticker,
                                        base_score=base_total,
                                        row_data={},
                                    )
                                    computed_total = enhanced_score
                                except:
                                    pass

                            cur.execute("""
                                INSERT INTO screener_scores (
                                    ticker, date, 
                                    sentiment_score, sentiment_weighted, article_count,
                                    fundamental_score, technical_score, growth_score, dividend_score,
                                    total_score, options_flow_score, short_squeeze_score,
                                    created_at
                                ) VALUES (
                                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                                )
                            """, (
                                ticker, today,
                                _to_native(final_sent),
                                _to_native(sentiment_data.get('sentiment_weighted', final_sent)),
                                _to_native(len(articles)),
                                _to_native(final_fund),
                                _to_native(final_tech),
                                _to_native(final_growth),
                                _to_native(final_div),
                                _to_native(computed_total),
                                _to_native(final_opts),
                                _to_native(final_sqz),
                            ))

                    conn.commit()
                    logger.info(f"{ticker}: screener_scores updated for {today} - sentiment={sentiment_score}, options={options_score}")

                    # Auto-sync to historical_scores for backtesting
                    try:
                        from src.utils.historical_sync import sync_to_historical
                        sync_to_historical(ticker, today, {
                            'sentiment_score': final_sent,
                            'fundamental_score': final_fund,
                            'growth_score': final_growth,
                            'dividend_score': final_div,
                            'total_score': computed_total,
                        })
                    except Exception as e:
                        logger.debug(f"{ticker}: historical sync skipped: {e}")

            # =====================================================================
            # STEP 6: Update Earnings Calendar from yfinance
            # =====================================================================
            status_container.info(f"📅 Step 6/8: Updating earnings calendar...")

            try:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                ed = stock.earnings_dates

                if ed is not None and not ed.empty:
                    # Get the next upcoming earnings date
                    for idx in ed.index:
                        earnings_dt = idx.date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
                        if earnings_dt >= today:
                            # Found next earnings date - save to DB
                            with get_connection() as conn:
                                with conn.cursor() as cur:
                                    try:
                                        cur.execute("""
                                            INSERT INTO earnings_calendar (ticker, earnings_date, updated_at)
                                            VALUES (%s, %s, NOW())
                                            ON CONFLICT (ticker) DO UPDATE SET
                                                earnings_date = EXCLUDED.earnings_date,
                                                updated_at = NOW()
                                        """, (ticker, earnings_dt))
                                    except Exception:
                                        # Table might not have updated_at column
                                        cur.execute("""
                                            INSERT INTO earnings_calendar (ticker, earnings_date)
                                            VALUES (%s, %s)
                                            ON CONFLICT (ticker) DO UPDATE SET
                                                earnings_date = EXCLUDED.earnings_date
                                        """, (ticker, earnings_dt))
                                conn.commit()
                            logger.info(f"{ticker}: Updated earnings_calendar -> {earnings_dt}")
                            break
            except Exception as e:
                logger.debug(f"{ticker}: Could not update earnings calendar: {e}")

            # =====================================================================
            # STEP 7: Regenerate Signal with Committee
            # =====================================================================
            status_container.info(f"🤖 Step 7/8: Regenerating signal with committee...")

            new_signal = None
            try:
                from src.core import get_signal_engine
                engine = get_signal_engine()

                # Clear this ticker from cache first
                if hasattr(engine, '_cache'):
                    for t in [ticker, ticker.upper(), ticker.lower()]:
                        if t in engine._cache:
                            del engine._cache[t]

                # Now regenerate the signal - this will reload from DB and recalculate committee
                new_signal = engine.generate_signal(ticker)
                logger.info(f"{ticker}: Regenerated signal - Today:{new_signal.today_signal} Committee:{new_signal.committee_verdict}")
            except Exception as e:
                logger.warning(f"Could not regenerate signal: {e}")

            # =====================================================================
            # STEP 8: Clear ALL caches and force refresh
            # =====================================================================
            status_container.info(f"🗑️ Step 8/8: Clearing all caches...")

            # Clear session state caches
            keys_to_clear = ['signals_data', 'market_overview']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]

            # Clear ticker-specific caches
            for key in list(st.session_state.keys()):
                if ticker in str(key) or ticker.upper() in str(key):
                    del st.session_state[key]
                    logger.info(f"Cleared session state: {key}")

            # Set flags for refresh
            st.session_state.force_refresh = True
            st.session_state.signals_loaded = True
            st.session_state['_refresh_ticker'] = ticker

            status_container.empty()

            # Show results summary
            with results_container:
                st.success(f"✅ **{ticker} Fully Updated for {today}!**")

                # Row 1: Basic metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📰 Articles", f"{len(articles)}", f"+{saved_count} new")
                with col2:
                    sent_emoji = "🟢" if sentiment_score >= 60 else "🔴" if sentiment_score <= 40 else "🟡"
                    st.metric("🧠 Sentiment", f"{sent_emoji} {sentiment_score}")
                with col3:
                    if options_score:
                        opt_emoji = "🟢" if options_score >= 60 else "🔴" if options_score <= 40 else "🟡"
                        st.metric("📊 Options", f"{opt_emoji} {options_score}")
                    else:
                        st.metric("📊 Options", "N/A")
                with col4:
                    if fundamental_score:
                        fund_emoji = "🟢" if fundamental_score >= 60 else "🔴" if fundamental_score <= 40 else "🟡"
                        st.metric("📈 Fundamental", f"{fund_emoji} {fundamental_score}")
                    else:
                        st.metric("📈 Fundamental", "N/A")

                # Row 2: Signal and Committee
                # Row 2: Signal and Committee
                # Row 2: Signal and Committee
                if new_signal:
                    st.markdown("---")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        today_signal = new_signal.today_signal.value if hasattr(new_signal.today_signal,
                                                                                'value') else str(
                            new_signal.today_signal)
                        signal_emoji = "🟢" if "BUY" in str(today_signal) else "🔴" if "SELL" in str(
                            today_signal) else "🟡"
                        st.metric("📈 Today Signal", signal_emoji + " " + str(today_signal))
                    with col2:
                        verdict = str(new_signal.committee_verdict) if new_signal.committee_verdict else "N/A"
                        verdict_emoji = "🟢" if "BUY" in verdict else "🔴" if "SELL" in verdict else "🟡"
                        st.metric("🗳️ Committee", verdict_emoji + " " + verdict)
                    with col3:
                        agreement = new_signal.committee_agreement or 0
                        try:
                            agreement_num = float(agreement) if not isinstance(agreement, str) else float(
                                agreement.replace('%', '').strip())
                            st.metric("📊 Agreement", "{:.0f}%".format(agreement_num * 100))
                        except (ValueError, AttributeError):
                            st.metric("📊 Agreement", str(agreement) if agreement else "N/A")
                    with col4:
                        try:
                            score_val = int(new_signal.today_score) if new_signal.today_score else 0
                            st.metric("📊 Today Score", str(score_val))
                        except (ValueError, TypeError):
                            st.metric("📊 Today Score", str(new_signal.today_score) if new_signal.today_score else "N/A")

            # Rerun to show updated data
            time.sleep(0.3)
            st.rerun()

        except Exception as e:
            st.error(f"Error refreshing {ticker}: {e}")
            import traceback
            st.code(traceback.format_exc())