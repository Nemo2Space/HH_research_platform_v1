"""
Alpha Model Enhancements - Production Version

Implements all signal quality improvements:
1. Forecast Shrinkage (continuous, context-aware)
2. ML Reliability Gate (vol-scaled bias, soft N gate, EWMA accuracy)
3. Catalyst Detection (CatalystScore 0-3)
4. Decision Policy Layer (strict binding)
5. Probability Calibration (smoothed by sample size)

Location: src/ml/alpha_enhancements.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

try:
    from src.db.connection import get_engine, get_connection
    from src.utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    from sqlalchemy import create_engine
    from contextlib import contextmanager
    import os
    from dotenv import load_dotenv
    load_dotenv()
    _engine = None
    def get_engine():
        global _engine
        if _engine is None:
            host = os.getenv('POSTGRES_HOST', 'localhost')
            port = os.getenv('POSTGRES_PORT', '5432')
            db = os.getenv('POSTGRES_DB', 'alpha_platform')
            user = os.getenv('POSTGRES_USER', 'alpha')
            password = os.getenv('POSTGRES_PASSWORD', '')
            _engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db}')
        return _engine
    @contextmanager
    def get_connection():
        conn = get_engine().connect()
        try:
            yield conn
        finally:
            conn.close()

@dataclass
class EnhancementConfig:
    shrinkage_k_base: float = 0.35
    shrinkage_k_catalyst_max: float = 0.35
    shrinkage_k_regime_penalty: float = 0.10
    shrinkage_k_uncertainty_max: float = 0.15
    shrinkage_k_floor: float = 0.0
    shrinkage_k_cap: float = 0.80
    min_samples_hard_block: int = 40
    min_samples_full_weight: int = 200
    accuracy_hard_block: float = 0.50
    accuracy_degraded: float = 0.55
    accuracy_strong_signal: float = 0.58
    bias_scaled_tradable: float = 0.20
    bias_scaled_strong: float = 0.10
    bias_scaled_block: float = 0.25
    earnings_days_strong: int = 3
    earnings_days_medium: int = 10
    iv_rank_strong: float = 75
    iv_rank_medium: float = 60
    options_volume_multiplier: float = 3.0
    vix_high_threshold: float = 25
    # Note: base_rate is calculated from actual data, not configured

CONFIG = EnhancementConfig()

class CatalystLevel(Enum):
    NONE = 0
    WEAK = 1
    MEDIUM = 2
    STRONG = 3

@dataclass
class CatalystInfo:
    score: CatalystLevel
    earnings_days: Optional[int] = None
    iv_rank: Optional[float] = None
    implied_move_pct: Optional[float] = None
    options_volume_ratio: Optional[float] = None
    vix_level: Optional[float] = None  # None = unknown, not assumed
    news_keywords_detected: bool = False
    verified_event: Optional[str] = None
    reasons: List[str] = field(default_factory=list)

    def is_high_regime(self) -> bool:
        if self.vix_level is None:
            return False  # Unknown VIX = don't assume high regime
        return self.vix_level >= CONFIG.vix_high_threshold

    def vix_status(self) -> str:
        if self.vix_level is None:
            return "UNKNOWN"
        return f"{self.vix_level:.1f}" + (" ⚠️ HIGH" if self.is_high_regime() else "")

def detect_catalysts(
    ticker: str,
    days_to_earnings: Optional[int] = None,
    iv_rank: Optional[float] = None,
    implied_move_pct: Optional[float] = None,
    options_volume_ratio: Optional[float] = None,
    vix_level: Optional[float] = None,  # None = unknown
    news_headlines: Optional[List[str]] = None,
    verified_event: Optional[str] = None
) -> CatalystInfo:
    reasons = []
    max_score = 0

    if days_to_earnings is not None:
        if days_to_earnings <= CONFIG.earnings_days_strong:
            max_score = max(max_score, 3)
            reasons.append(f"Earnings in {days_to_earnings}d (STRONG)")
        elif days_to_earnings <= CONFIG.earnings_days_medium:
            max_score = max(max_score, 2)
            reasons.append(f"Earnings in {days_to_earnings}d (MEDIUM)")

    if iv_rank is not None:
        if iv_rank >= CONFIG.iv_rank_strong:
            max_score = max(max_score, 3)
            reasons.append(f"IV Rank {iv_rank:.0f}% (STRONG)")
        elif iv_rank >= CONFIG.iv_rank_medium:
            max_score = max(max_score, 2)
            reasons.append(f"IV Rank {iv_rank:.0f}% (MEDIUM)")

    if implied_move_pct is not None:
        if implied_move_pct >= 8.0:
            max_score = max(max_score, 3)
            reasons.append(f"Implied move {implied_move_pct:.1f}% (STRONG)")
        elif implied_move_pct >= 5.0:
            max_score = max(max_score, 2)
            reasons.append(f"Implied move {implied_move_pct:.1f}% (MEDIUM)")

    if options_volume_ratio is not None and options_volume_ratio >= CONFIG.options_volume_multiplier:
        max_score = max(max_score, 2)
        reasons.append(f"Options vol {options_volume_ratio:.1f}x avg (MEDIUM)")

    if verified_event:
        max_score = max(max_score, 3)
        reasons.append(f"Verified: {verified_event} (STRONG)")

    news_keywords_detected = False
    if news_headlines:
        kws = ['fda', 'approval', 'merger', 'acquisition', 'buyout', 'guidance', 'preannounce']
        for h in news_headlines:
            if any(k in h.lower() for k in kws):
                news_keywords_detected = True
                break
        if news_keywords_detected and max_score < 1:
            max_score = 1
            reasons.append("News keywords (WEAK)")

    score_map = {0: CatalystLevel.NONE, 1: CatalystLevel.WEAK, 2: CatalystLevel.MEDIUM, 3: CatalystLevel.STRONG}
    return CatalystInfo(
        score=score_map.get(max_score, CatalystLevel.NONE),
        earnings_days=days_to_earnings, iv_rank=iv_rank,
        implied_move_pct=implied_move_pct, options_volume_ratio=options_volume_ratio,
        vix_level=vix_level, news_keywords_detected=news_keywords_detected,
        verified_event=verified_event, reasons=reasons
    )

def get_catalyst_info_from_db(ticker: str) -> CatalystInfo:
    days_to_earnings = None
    iv_rank = None
    options_volume_ratio = None
    implied_move_pct = None
    vix_level = None  # Unknown until fetched from DB
    news_headlines = []

    try:
        engine = get_engine()

        # Get earnings date
        try:
            df = pd.read_sql("SELECT earnings_date FROM earnings_calendar WHERE ticker = %s AND earnings_date >= CURRENT_DATE ORDER BY earnings_date LIMIT 1", engine, params=(ticker,))
            if not df.empty and df.iloc[0]['earnings_date']:
                ed = df.iloc[0]['earnings_date']
                if isinstance(ed, str):
                    ed = datetime.strptime(ed, '%Y-%m-%d').date()
                days_to_earnings = (ed - date.today()).days
        except Exception as e:
            logger.debug(f"Could not get earnings date: {e}")

        # Get IV rank
        try:
            df = pd.read_sql("SELECT iv_rank, iv_percentile FROM options_summary WHERE ticker = %s ORDER BY date DESC LIMIT 1", engine, params=(ticker,))
            if not df.empty:
                iv_rank = df.iloc[0].get('iv_rank') or df.iloc[0].get('iv_percentile')
        except Exception as e:
            logger.debug(f"Could not get IV rank: {e}")

        # Get options volume ratio (unusual activity detection)
        try:
            df = pd.read_sql("""
                SELECT total_volume, 
                       AVG(total_volume) OVER (ORDER BY date ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING) as avg_vol_30d
                FROM options_flow WHERE ticker = %s ORDER BY date DESC LIMIT 1
            """, engine, params=(ticker,))
            if not df.empty and df.iloc[0].get('avg_vol_30d'):
                avg_vol = df.iloc[0]['avg_vol_30d']
                if avg_vol > 0:
                    options_volume_ratio = df.iloc[0]['total_volume'] / avg_vol
        except Exception as e:
            logger.debug(f"Could not get options volume: {e}")

        # Get VIX level
        try:
            df = pd.read_sql("SELECT close FROM market_data WHERE ticker IN ('^VIX', 'VIX') ORDER BY date DESC LIMIT 1", engine)
            if not df.empty:
                vix_level = float(df.iloc[0]['close'])
            else:
                logger.debug("No VIX data found in database")
        except Exception as e:
            logger.debug(f"Could not get VIX: {e}")

        # Get recent news headlines
        try:
            df = pd.read_sql("""
                SELECT headline FROM news 
                WHERE ticker = %s AND published_at >= NOW() - INTERVAL '3 days'
                ORDER BY published_at DESC LIMIT 10
            """, engine, params=(ticker,))
            if not df.empty:
                news_headlines = df['headline'].tolist()
        except Exception as e:
            logger.debug(f"Could not get news: {e}")

        # Estimate implied move from IV if available (approximation)
        if iv_rank is not None and days_to_earnings is not None and days_to_earnings <= 10:
            # Rough estimate: implied move ≈ IV * sqrt(days/252) * adjustment
            # This is a simplification - proper calculation needs ATM straddle price
            try:
                df = pd.read_sql("SELECT implied_volatility FROM options_summary WHERE ticker = %s ORDER BY date DESC LIMIT 1", engine, params=(ticker,))
                if not df.empty and df.iloc[0].get('implied_volatility'):
                    iv = df.iloc[0]['implied_volatility']
                    # Estimate: IV * sqrt(days/252) * 100 for percentage
                    implied_move_pct = iv * np.sqrt(days_to_earnings / 252) * 100
            except Exception as e:
                logger.debug(f"Could not estimate implied move: {e}")

    except Exception as e:
        logger.debug(f"Catalyst DB error: {e}")

    return detect_catalysts(
        ticker=ticker,
        days_to_earnings=days_to_earnings,
        iv_rank=iv_rank,
        implied_move_pct=implied_move_pct,
        options_volume_ratio=options_volume_ratio,
        vix_level=vix_level,
        news_headlines=news_headlines
    )

@dataclass
class ReliabilityMetrics:
    sample_count_5d: int = 0
    sample_count_10d: int = 0
    accuracy_ewma_5d: Optional[float] = None  # None = not calculated yet
    accuracy_ewma_10d: Optional[float] = None
    strong_buy_accuracy: Optional[float] = None
    strong_sell_accuracy: Optional[float] = None
    bias_scaled_5d: Optional[float] = None
    bias_raw_5d: Optional[float] = None
    base_rate_5d: Optional[float] = None  # None = not measured, don't use placeholder
    base_rate_10d: Optional[float] = None
    last_updated: datetime = None

    def get_reliability_weight(self, horizon: int = 5) -> float:
        n = self.sample_count_5d if horizon == 5 else self.sample_count_10d
        if n < CONFIG.min_samples_hard_block:
            return 0.0

        acc = self.accuracy_ewma_5d if horizon == 5 else self.accuracy_ewma_10d
        if acc is None:
            return 0.0  # No accuracy data = can't trust

        w_sample = min(1.0, n / CONFIG.min_samples_full_weight)

        if acc < CONFIG.accuracy_hard_block:
            return 0.0
        elif acc < CONFIG.accuracy_degraded:
            w_acc = (acc - CONFIG.accuracy_hard_block) / (CONFIG.accuracy_degraded - CONFIG.accuracy_hard_block)
        else:
            w_acc = 1.0

        bias = self.bias_scaled_5d
        if bias is None:
            w_bias = 0.5  # Unknown bias = reduce confidence
        elif bias > CONFIG.bias_scaled_block:
            return 0.0
        elif bias > CONFIG.bias_scaled_tradable:
            w_bias = 1.0 - (bias - CONFIG.bias_scaled_tradable) / (CONFIG.bias_scaled_block - CONFIG.bias_scaled_tradable)
        else:
            w_bias = 1.0
        return w_sample * w_acc * w_bias

    def get_status(self, horizon: int = 5) -> str:
        w = self.get_reliability_weight(horizon)
        n = self.sample_count_5d if horizon == 5 else self.sample_count_10d
        acc = self.accuracy_ewma_5d if horizon == 5 else self.accuracy_ewma_10d
        if n < CONFIG.min_samples_hard_block:
            return "BLOCKED_INSUFFICIENT_DATA"
        elif acc is None:
            return "BLOCKED_NO_ACCURACY_DATA"
        elif w == 0:
            return "BLOCKED_LOW_ACCURACY" if acc < CONFIG.accuracy_hard_block else "BLOCKED_HIGH_BIAS"
        elif w < 0.5:
            return "DEGRADED"
        elif acc >= CONFIG.accuracy_strong_signal:
            return "STRONG_ELIGIBLE"
        return "TRADABLE"

    def is_strong_signal_eligible(self, horizon: int = 5) -> bool:
        n = self.sample_count_5d if horizon == 5 else self.sample_count_10d
        acc = self.accuracy_ewma_5d if horizon == 5 else self.accuracy_ewma_10d
        bias = self.bias_scaled_5d
        if acc is None or bias is None:
            return False
        return n >= CONFIG.min_samples_hard_block and acc >= CONFIG.accuracy_strong_signal and bias <= CONFIG.bias_scaled_strong

    def get_base_rate(self, horizon: int = 5) -> Optional[float]:
        """Returns None if base rate not measured - caller must handle."""
        if horizon == 5:
            return self.base_rate_5d
        return self.base_rate_10d

def calculate_ewma_accuracy(returns: pd.Series, predictions: pd.Series, span: int = 60) -> float:
    if len(returns) < 10:
        return 0.50
    correct = ((predictions > 0) & (returns > 0)) | ((predictions < 0) & (returns < 0))
    ewma = correct.ewm(span=span, min_periods=10).mean()
    return float(ewma.iloc[-1]) if len(ewma) > 0 else 0.50

def get_reliability_metrics(days_back: int = 90) -> ReliabilityMetrics:
    metrics = ReliabilityMetrics(last_updated=datetime.now())
    try:
        engine = get_engine()
        # Use date arithmetic that works with parameterization
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        query = """SELECT ticker, prediction_date, predicted_return_5d, predicted_return_10d,
                   actual_return_5d, actual_return_10d, alpha_signal, absolute_error_5d
                   FROM alpha_predictions WHERE actual_return_5d IS NOT NULL
                   AND prediction_date >= %s ORDER BY prediction_date"""
        df = pd.read_sql(query, engine, params=(cutoff_date,))
        if df.empty:
            return metrics
        metrics.sample_count_5d = len(df[df['actual_return_5d'].notna()])
        metrics.sample_count_10d = len(df[df['actual_return_10d'].notna()])
        if metrics.sample_count_5d >= 10:
            df_5d = df[df['actual_return_5d'].notna()].copy()
            metrics.accuracy_ewma_5d = calculate_ewma_accuracy(df_5d['actual_return_5d'], df_5d['predicted_return_5d'])
        strong_buy = df[df['alpha_signal'] == 'STRONG_BUY']
        if len(strong_buy) >= 5:
            metrics.strong_buy_accuracy = (strong_buy['actual_return_5d'] > 0).mean()
        strong_sell = df[df['alpha_signal'] == 'STRONG_SELL']
        if len(strong_sell) >= 5:
            metrics.strong_sell_accuracy = (strong_sell['actual_return_5d'] < 0).mean()
        if metrics.sample_count_5d >= 20:
            errors = df['absolute_error_5d'].dropna()
            metrics.bias_raw_5d = float(errors.mean())
            actual_mad = df['actual_return_5d'].dropna().abs().median()
            metrics.bias_scaled_5d = abs(metrics.bias_raw_5d) / actual_mad if actual_mad > 0 else abs(metrics.bias_raw_5d) * 100
        if metrics.sample_count_5d >= 30:
            metrics.base_rate_5d = float((df['actual_return_5d'] > 0).mean())
    except Exception as e:
        logger.error(f"Reliability metrics error: {e}")
    return metrics

@dataclass
class ShrinkageResult:
    k_factor: float
    pred_5d_raw: float
    pred_5d_shrunk: float
    pred_10d_raw: float
    pred_10d_shrunk: float
    pred_20d_raw: float
    pred_20d_shrunk: float
    k_base: float
    k_catalyst: float
    k_regime_penalty: float
    k_uncertainty_penalty: float
    reasons: List[str] = field(default_factory=list)

def compute_shrinkage_factor(catalyst_info: CatalystInfo, reliability: ReliabilityMetrics) -> Tuple[float, List[str]]:
    reasons = []
    reliability_weight = reliability.get_reliability_weight(5)
    if reliability_weight == 0:
        reasons.append("ML unreliable (k=0)")
        return 0.0, reasons
    k_base = CONFIG.shrinkage_k_base
    reasons.append(f"Base k={k_base:.2f}")
    k_catalyst = (catalyst_info.score.value / 3.0) * CONFIG.shrinkage_k_catalyst_max
    if k_catalyst > 0:
        reasons.append(f"Catalyst +{k_catalyst:.2f}")
    k_regime = CONFIG.shrinkage_k_regime_penalty if catalyst_info.is_high_regime() else 0.0
    if k_regime > 0:
        reasons.append(f"VIX penalty -{k_regime:.2f}")
    k_uncertainty = (1.0 - reliability_weight) * CONFIG.shrinkage_k_uncertainty_max if reliability_weight < 1.0 else 0.0
    if k_uncertainty > 0:
        reasons.append(f"Reliability penalty -{k_uncertainty:.2f}")
    k = max(CONFIG.shrinkage_k_floor, min(CONFIG.shrinkage_k_cap, k_base + k_catalyst - k_regime - k_uncertainty))
    reasons.append(f"Final k={k:.2f}")
    return k, reasons

def apply_forecast_shrinkage(pred_5d: float, pred_10d: float, pred_20d: float, catalyst_info: CatalystInfo, reliability: ReliabilityMetrics) -> ShrinkageResult:
    k, reasons = compute_shrinkage_factor(catalyst_info, reliability)
    k_catalyst = (catalyst_info.score.value / 3.0) * CONFIG.shrinkage_k_catalyst_max
    k_regime = CONFIG.shrinkage_k_regime_penalty if catalyst_info.is_high_regime() else 0.0
    rw = reliability.get_reliability_weight(5)
    k_unc = (1.0 - rw) * CONFIG.shrinkage_k_uncertainty_max if rw < 1.0 else 0.0
    return ShrinkageResult(k_factor=k, pred_5d_raw=pred_5d, pred_5d_shrunk=pred_5d * k,
                          pred_10d_raw=pred_10d, pred_10d_shrunk=pred_10d * k,
                          pred_20d_raw=pred_20d, pred_20d_shrunk=pred_20d * k,
                          k_base=CONFIG.shrinkage_k_base, k_catalyst=k_catalyst,
                          k_regime_penalty=k_regime, k_uncertainty_penalty=k_unc, reasons=reasons)

@dataclass
class CalibrationBin:
    pred_range: Tuple[float, float]
    actual_win_rate: float
    sample_count: int

class ProbabilityCalibrator:
    def __init__(self):
        self.calibration_bins: List[CalibrationBin] = []
        self.total_samples: int = 0
        self.loaded = False

    def load_calibration(self, days_back: int = 180):
        try:
            # Use date arithmetic that works with parameterization
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            df = pd.read_sql("""SELECT predicted_return_5d * 100 as pred_pct, actual_return_5d,
                               CASE WHEN actual_return_5d > 0 THEN 1 ELSE 0 END as is_win
                               FROM alpha_predictions WHERE actual_return_5d IS NOT NULL
                               AND prediction_date >= %s""", get_engine(), params=(cutoff_date,))
            if len(df) < 30:
                self.loaded = False
                return
            self.total_samples = len(df)
            bins = [(-100, -5), (-5, -3), (-3, -1), (-1, 0), (0, 1), (1, 3), (3, 5), (5, 100)]
            self.calibration_bins = []
            for bmin, bmax in bins:
                subset = df[(df['pred_pct'] >= bmin) & (df['pred_pct'] < bmax)]
                if len(subset) >= 5:
                    self.calibration_bins.append(CalibrationBin((bmin, bmax), float(subset['is_win'].mean()), len(subset)))
            self.loaded = len(self.calibration_bins) >= 3
        except Exception as e:
            logger.error(f"Calibration error: {e}")
            self.loaded = False

    def calibrate_probability(self, pred_5d_pct: float, raw_prob: float, base_rate: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Calibrate probability. If base_rate is None (not measured), uses raw calibrated without smoothing.
        Returns: (p_calibrated, p_smoothed, weight)
        """
        if not self.loaded:
            return raw_prob, raw_prob, 0.0
        p_calibrated = raw_prob
        bin_n = 0
        for b in self.calibration_bins:
            if b.pred_range[0] <= pred_5d_pct < b.pred_range[1]:
                p_calibrated = b.actual_win_rate
                bin_n = b.sample_count
                break
        w = min(1.0, self.total_samples / 200.0)
        if bin_n < 20:
            w *= (bin_n / 20.0)

        # If base_rate not measured, skip smoothing
        if base_rate is None:
            return p_calibrated, p_calibrated, w

        p_smoothed = w * p_calibrated + (1.0 - w) * base_rate
        return p_calibrated, p_smoothed, w

_calibrator = None
def get_calibrator() -> ProbabilityCalibrator:
    global _calibrator
    if _calibrator is None:
        _calibrator = ProbabilityCalibrator()
        _calibrator.load_calibration()
    return _calibrator


# =============================================================================
# 6. SECTOR NEUTRALIZATION
# =============================================================================

@dataclass
class SectorContext:
    """Sector-relative analysis for a ticker."""
    ticker: str
    sector: str
    sector_rank: Optional[int] = None  # Rank within sector (1 = best)
    sector_count: Optional[int] = None  # Total stocks in sector
    sector_percentile: Optional[float] = None  # Percentile within sector
    is_sector_leader: bool = False  # Top 20% of sector
    is_sector_laggard: bool = False  # Bottom 20% of sector
    sector_avg_score: Optional[float] = None
    ticker_score: Optional[float] = None
    relative_strength: Optional[float] = None  # Z-score vs sector


def get_sector_context(ticker: str) -> SectorContext:
    """
    Get sector-relative context for a ticker.

    This helps identify if a stock is a sector leader or just riding sector momentum.
    """
    context = SectorContext(ticker=ticker, sector="Unknown")

    try:
        engine = get_engine()

        # Get ticker's sector and score
        ticker_query = """
            SELECT ticker, sector, total_score 
            FROM latest_scores 
            WHERE ticker = %s
        """
        ticker_df = pd.read_sql(ticker_query, engine, params=(ticker,))

        if ticker_df.empty:
            return context

        sector = ticker_df.iloc[0]['sector']
        ticker_score = ticker_df.iloc[0]['total_score']
        context.sector = sector or "Unknown"
        context.ticker_score = ticker_score

        if not sector:
            return context

        # Get all stocks in the same sector
        sector_query = """
            SELECT ticker, total_score 
            FROM latest_scores 
            WHERE sector = %s AND total_score IS NOT NULL
            ORDER BY total_score DESC
        """
        sector_df = pd.read_sql(sector_query, engine, params=(sector,))

        if len(sector_df) < 3:
            return context

        context.sector_count = len(sector_df)
        context.sector_avg_score = float(sector_df['total_score'].mean())

        # Calculate rank (1 = best)
        sector_df['rank'] = range(1, len(sector_df) + 1)
        ticker_row = sector_df[sector_df['ticker'] == ticker]
        if not ticker_row.empty:
            context.sector_rank = int(ticker_row.iloc[0]['rank'])
            context.sector_percentile = (1 - (context.sector_rank / context.sector_count)) * 100

            # Determine if leader or laggard
            context.is_sector_leader = context.sector_percentile >= 80
            context.is_sector_laggard = context.sector_percentile <= 20

        # Calculate relative strength (Z-score vs sector)
        sector_std = sector_df['total_score'].std()
        if sector_std > 0 and ticker_score is not None:
            context.relative_strength = (ticker_score - context.sector_avg_score) / sector_std

    except Exception as e:
        logger.debug(f"Could not get sector context: {e}")

    return context


def get_sector_context_string(context: SectorContext) -> str:
    """Format sector context for AI."""
    if context.sector == "Unknown" or context.sector_count is None:
        return ""

    lines = [
        f"📊 SECTOR CONTEXT ({context.sector}):",
        f"   Rank: #{context.sector_rank} of {context.sector_count} in sector",
        f"   Percentile: {context.sector_percentile:.0f}th",
        f"   Score: {context.ticker_score:.1f} vs Sector Avg: {context.sector_avg_score:.1f}",
    ]

    if context.relative_strength is not None:
        lines.append(f"   Relative Strength (Z): {context.relative_strength:+.2f}")

    if context.is_sector_leader:
        lines.append("   ⭐ SECTOR LEADER - Outperforming peers")
    elif context.is_sector_laggard:
        lines.append("   ⚠️ SECTOR LAGGARD - Underperforming peers")
    else:
        lines.append("   📈 Mid-pack within sector")

    return "\n".join(lines)


# =============================================================================
# 7. SENTIMENT VELOCITY (Rate of Change)
# =============================================================================

@dataclass
class SentimentVelocity:
    """Tracks rate of change of sentiment."""
    ticker: str
    current_score: Optional[float] = None
    score_3d_ago: Optional[float] = None
    score_7d_ago: Optional[float] = None
    velocity_3d: Optional[float] = None  # Change over 3 days
    velocity_7d: Optional[float] = None  # Change over 7 days
    velocity_z: Optional[float] = None   # Z-score of velocity vs history
    is_accelerating: bool = False        # Velocity increasing
    is_decelerating: bool = False        # Velocity decreasing
    signal: str = "NEUTRAL"              # ACCELERATING_BULLISH, DECELERATING, etc.


def get_sentiment_velocity(ticker: str) -> SentimentVelocity:
    """
    Calculate sentiment velocity (rate of change).

    Key insight: "Accelerating sentiment + flat price = ignition phase"
    """
    velocity = SentimentVelocity(ticker=ticker)

    try:
        engine = get_engine()

        # Get sentiment history (need a table that stores daily sentiment)
        # Try sentiment_scores or stock_scores table
        query = """
            SELECT date, sentiment_score as score
            FROM (
                SELECT date, 
                       COALESCE(sentiment_score, total_score * 0.3) as sentiment_score
                FROM latest_scores_history
                WHERE ticker = %s
                ORDER BY date DESC
                LIMIT 10
            ) sub
            ORDER BY date ASC
        """

        try:
            df = pd.read_sql(query, engine, params=(ticker,))
        except:
            # Fallback: try using signal_history or similar
            query = """
                SELECT created_at::date as date, 
                       (total_score - 50) as score  
                FROM unified_signals
                WHERE ticker = %s
                ORDER BY created_at DESC
                LIMIT 10
            """
            try:
                df = pd.read_sql(query, engine, params=(ticker,))
            except:
                return velocity

        if len(df) < 3:
            return velocity

        df = df.sort_values('date')

        # Current and historical scores
        velocity.current_score = float(df.iloc[-1]['score'])

        if len(df) >= 4:
            velocity.score_3d_ago = float(df.iloc[-4]['score']) if len(df) >= 4 else None
            velocity.velocity_3d = velocity.current_score - velocity.score_3d_ago

        if len(df) >= 8:
            velocity.score_7d_ago = float(df.iloc[-8]['score']) if len(df) >= 8 else None
            velocity.velocity_7d = velocity.current_score - velocity.score_7d_ago

        # Calculate velocity Z-score (how unusual is current velocity)
        if len(df) >= 5:
            daily_changes = df['score'].diff().dropna()
            if len(daily_changes) >= 3 and daily_changes.std() > 0:
                recent_velocity = daily_changes.iloc[-3:].mean()
                velocity.velocity_z = (recent_velocity - daily_changes.mean()) / daily_changes.std()

        # Determine acceleration
        if velocity.velocity_3d is not None and velocity.velocity_7d is not None:
            # If 3d velocity > 7d velocity (per day), we're accelerating
            v3_per_day = velocity.velocity_3d / 3
            v7_per_day = velocity.velocity_7d / 7
            velocity.is_accelerating = v3_per_day > v7_per_day + 0.5
            velocity.is_decelerating = v3_per_day < v7_per_day - 0.5

        # Generate signal
        if velocity.velocity_3d is not None:
            if velocity.velocity_3d > 5 and velocity.is_accelerating:
                velocity.signal = "🚀 ACCELERATING_BULLISH"
            elif velocity.velocity_3d > 3:
                velocity.signal = "📈 IMPROVING"
            elif velocity.velocity_3d < -5 and velocity.is_decelerating:
                velocity.signal = "📉 ACCELERATING_BEARISH"
            elif velocity.velocity_3d < -3:
                velocity.signal = "⚠️ DETERIORATING"
            else:
                velocity.signal = "➡️ STABLE"

    except Exception as e:
        logger.debug(f"Could not calculate sentiment velocity: {e}")

    return velocity


def get_sentiment_velocity_string(velocity: SentimentVelocity) -> str:
    """Format sentiment velocity for AI."""
    if velocity.current_score is None:
        return ""

    lines = [f"📈 SENTIMENT VELOCITY:"]

    if velocity.velocity_3d is not None:
        lines.append(f"   3-Day Change: {velocity.velocity_3d:+.1f} points")
    if velocity.velocity_7d is not None:
        lines.append(f"   7-Day Change: {velocity.velocity_7d:+.1f} points")
    if velocity.velocity_z is not None:
        lines.append(f"   Velocity Z-Score: {velocity.velocity_z:+.2f}")

    lines.append(f"   Signal: {velocity.signal}")

    if velocity.is_accelerating and velocity.velocity_3d and velocity.velocity_3d > 0:
        lines.append("   💡 IGNITION PATTERN: Rising sentiment - watch for price breakout")
    elif velocity.is_decelerating and velocity.velocity_3d and velocity.velocity_3d < 0:
        lines.append("   ⚠️ EXHAUSTION PATTERN: Fading momentum - watch for reversal")

    return "\n".join(lines)


class TradeAction(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    WAIT_FOR_TRIGGER = "WAIT_FOR_TRIGGER"
    NO_TRADE = "NO_TRADE"

@dataclass
class DecisionPolicy:
    trade_allowed: bool
    action: TradeAction
    size_cap: float
    confidence_band: str
    stop_type: str
    entry_styles_allowed: List[str]
    time_horizon: str
    primary_reason: str
    blocking_factors: List[str]
    required_triggers: List[str]
    human_override_allowed: bool = False

    def to_context_string(self) -> str:
        size_str = f"{self.size_cap:.2f}x MAXIMUM" if self.size_cap > 0 else "N/A (no trade)"
        lines = ["=" * 67, "🎯 DECISION POLICY (BINDING)", "=" * 67, "",
                 f"Trade Allowed: {'✅ YES' if self.trade_allowed else '❌ NO'}",
                 f"Action: {self.action.value}",
                 f"Size Cap: {size_str}",
                 f"Confidence: {self.confidence_band}",
                 f"Stop Type: {self.stop_type}",
                 f"Entry Options: {', '.join(self.entry_styles_allowed)}",
                 "", f"Primary Reason: {self.primary_reason}"]
        if self.blocking_factors:
            lines.extend(["", "⚠️ BLOCKING FACTORS:"] + [f"   • {f}" for f in self.blocking_factors])
        if self.required_triggers:
            lines.extend(["", "📍 REQUIRED TRIGGERS:"] + [f"   • {t}" for t in self.required_triggers])
        lines.extend(["", "=" * 67, "LLM RULES:",
                     "  ✅ MAY: Explain, suggest timing, add risks, downgrade, reduce size",
                     "  ❌ MAY NOT: Upgrade action, exceed size cap, ignore blockers", "=" * 67])
        return "\n".join(lines)

def compute_decision_policy(platform_score: float, platform_signal: str, alpha_signal: str,
                           alpha_pred_5d_shrunk: float, alpha_prob_calibrated: float,
                           reliability: ReliabilityMetrics, catalyst_info: CatalystInfo,
                           technical_score: float = 50, human_override_allowed: bool = False) -> DecisionPolicy:
    blocking_factors = []
    required_triggers = []
    reliability_weight = reliability.get_reliability_weight(5)

    # FIX: Track ML blocked status and COMPLETELY ignore alpha values when blocked
    ml_blocked = reliability_weight == 0
    ml_status = reliability.get_status(5)

    if ml_blocked:
        blocking_factors.append(f"ML {ml_status}")
        # CRITICAL FIX: Reset alpha values to neutral when ML is blocked
        # This prevents the AI from using unreliable ML predictions
        alpha_signal = "HOLD"  # Ignore ML signal completely
        alpha_pred_5d_shrunk = 0.0  # Zero out prediction
        alpha_prob_calibrated = 0.50  # Neutral probability
    elif reliability_weight < 0.5:
        blocking_factors.append(f"ML degraded ({reliability_weight:.0%})")

    strong_eligible = reliability.is_strong_signal_eligible(5)
    if alpha_signal in ['STRONG_BUY', 'STRONG_SELL'] and not strong_eligible:
        blocking_factors.append("STRONG signals blocked")
    if catalyst_info.is_high_regime():
        blocking_factors.append(f"High VIX ({catalyst_info.vix_level:.0f})")

    signal_map = {'STRONG_SELL': -2, 'SELL': -1, 'HOLD': 0, 'BUY': 1, 'STRONG_BUY': 2}
    platform_dir = signal_map.get(platform_signal, 0)
    alpha_dir = signal_map.get(alpha_signal, 0)
    conflict_level = abs(platform_dir - alpha_dir)

    # FIX: When ML blocked, use platform signal ONLY with no conflict calculation
    if ml_blocked:
        if platform_signal in ['BUY', 'SELL']:
            base_action = TradeAction(platform_signal)
        else:
            base_action = TradeAction.HOLD
        primary_reason = f"ML blocked ({ml_status}) - using platform signal only ({platform_signal})"
        conflict_level = 0  # No conflict when we completely ignore ML
    elif conflict_level >= 3:
        base_action = TradeAction.NO_TRADE
        primary_reason = f"Hard conflict: {platform_signal} vs {alpha_signal}"
        required_triggers.append("Wait for alignment")
    elif conflict_level == 2:
        base_action = TradeAction.WAIT_FOR_TRIGGER
        primary_reason = f"Moderate conflict: {platform_signal} vs {alpha_signal}"
        required_triggers.extend(["Break key level", "Volume confirmation"])
    elif conflict_level == 1:
        # STRICT: Platform is ceiling - cannot escalate HOLD to BUY/SELL
        # Can only de-risk (BUY to HOLD, SELL to HOLD)
        if platform_signal == 'HOLD':
            # Platform says HOLD, Alpha says BUY or SELL -> WAIT_FOR_TRIGGER (no escalation)
            base_action = TradeAction.WAIT_FOR_TRIGGER
            primary_reason = f"Mild conflict: Platform {platform_signal} vs Alpha {alpha_signal} - waiting for confirmation"
            if alpha_dir > 0:
                required_triggers.append("Platform upgrade to BUY")
            else:
                required_triggers.append("Platform downgrade to SELL")
        elif platform_signal in ['BUY', 'SELL']:
            # Platform allows trade, but Alpha disagrees on direction or wants HOLD
            # Use platform signal (Alpha can de-risk but not flip)
            base_action = TradeAction(platform_signal)
            primary_reason = f"Mild conflict: Using platform {platform_signal} (Alpha: {alpha_signal})"
        else:
            base_action = TradeAction.HOLD
            primary_reason = f"Mild conflict: Defaulting to HOLD"
    else:
        if alpha_signal == 'STRONG_BUY':
            base_action = TradeAction.STRONG_BUY if strong_eligible else TradeAction.BUY
        elif alpha_signal == 'STRONG_SELL':
            base_action = TradeAction.STRONG_SELL if strong_eligible else TradeAction.SELL
        elif alpha_signal in ['BUY', 'SELL', 'HOLD']:
            base_action = TradeAction(alpha_signal)
        else:
            base_action = TradeAction.HOLD
        primary_reason = f"Aligned: {platform_signal} ≈ {alpha_signal}"

    base_size = 1.0
    if conflict_level >= 2: base_size *= 0.25
    elif conflict_level == 1: base_size *= 0.5
    if reliability_weight < 1.0: base_size *= (0.5 + 0.5 * reliability_weight)
    if catalyst_info.is_high_regime(): base_size *= 0.5
    if alpha_prob_calibrated < 0.55: base_size *= 0.5
    if technical_score < 40 and alpha_dir > 0:
        base_size *= 0.7
        blocking_factors.append("Technical bearish")
    elif technical_score > 60 and alpha_dir < 0:
        base_size *= 0.7
        blocking_factors.append("Technical bullish")

    # Determine confidence band early for conflict resolution
    confidence_band = "LOW" if len(blocking_factors) >= 2 or conflict_level >= 2 else ("MEDIUM" if len(blocking_factors) >= 1 or conflict_level == 1 or alpha_prob_calibrated < 0.60 else "HIGH")

    # FIX: Additional conflict resolution for LOW confidence scenarios
    # If confidence is LOW with multiple blocking factors, be more conservative
    if confidence_band == "LOW" and len(blocking_factors) >= 2:
        if base_action in [TradeAction.BUY, TradeAction.SELL]:
            # Downgrade to WAIT_FOR_TRIGGER instead of outright trade
            base_action = TradeAction.WAIT_FOR_TRIGGER
            primary_reason = f"Downgraded to WAIT: LOW confidence + {len(blocking_factors)} blocking factors"
            required_triggers.append("Await signal alignment or blocking factor resolution")

    # Determine trade_allowed
    trade_allowed = base_action not in [TradeAction.NO_TRADE, TradeAction.WAIT_FOR_TRIGGER, TradeAction.HOLD]
    if len(blocking_factors) >= 3:
        trade_allowed = False
        if base_action not in [TradeAction.NO_TRADE, TradeAction.HOLD]:
            base_action = TradeAction.WAIT_FOR_TRIGGER

    # P0 Fix: Size cap = 0 when trade not allowed
    if trade_allowed:
        size_cap = max(0.25, min(1.0, base_size))
    else:
        size_cap = 0.0  # No trade = no size

    stop_type = "WIDE" if catalyst_info.is_high_regime() or conflict_level >= 2 else ("NORMAL" if conflict_level == 1 or blocking_factors else "TIGHT")
    entry_styles = ["BREAKOUT", "PULLBACK"] if base_action == TradeAction.WAIT_FOR_TRIGGER else (["LIMIT", "PULLBACK", "STAGED"] if conflict_level >= 1 else ["MARKET", "LIMIT", "PULLBACK"])

    return DecisionPolicy(trade_allowed=trade_allowed, action=base_action, size_cap=round(size_cap, 2),
                         confidence_band=confidence_band, stop_type=stop_type, entry_styles_allowed=entry_styles,
                         time_horizon="SWING", primary_reason=primary_reason, blocking_factors=blocking_factors,
                         required_triggers=required_triggers, human_override_allowed=human_override_allowed)

def build_enhanced_alpha_context(ticker: str, alpha_prediction: Dict[str, Any], platform_score: float,
                                  platform_signal: str, technical_score: float = 50,
                                  news_headlines: Optional[List[str]] = None) -> str:
    catalyst_info = get_catalyst_info_from_db(ticker)
    if news_headlines:
        catalyst_info = detect_catalysts(ticker, catalyst_info.earnings_days, catalyst_info.iv_rank,
                                         catalyst_info.options_volume_ratio, catalyst_info.vix_level, news_headlines)
    reliability = get_reliability_metrics(90)

    # UNIT HANDLING: Model returns decimals (0.02 for 2%), convert to percentage
    # Guard: if value > 1, it's already a percentage
    pred_5d_raw = alpha_prediction.get('predicted_return_5d', 0)
    pred_10d_raw = alpha_prediction.get('predicted_return_10d', 0)
    pred_20d_raw = alpha_prediction.get('expected_return_20d', 0)

    if abs(pred_5d_raw) < 1:  # Decimal format, convert to %
        pred_5d_raw *= 100
    if abs(pred_10d_raw) < 1:
        pred_10d_raw *= 100
    if abs(pred_20d_raw) < 1:
        pred_20d_raw *= 100

    shrinkage = apply_forecast_shrinkage(pred_5d_raw, pred_10d_raw, pred_20d_raw, catalyst_info, reliability)
    calibrator = get_calibrator()
    raw_prob = alpha_prediction.get('prob_positive_5d', 0.5)
    p_calibrated, p_smoothed, cal_w = calibrator.calibrate_probability(pred_5d_raw, raw_prob, reliability.get_base_rate(5))
    policy = compute_decision_policy(platform_score, platform_signal, alpha_prediction.get('signal', 'HOLD'),
                                     shrinkage.pred_5d_shrunk, p_smoothed, reliability, catalyst_info, technical_score)

    parts = [policy.to_context_string()]

    # Catalyst section - handle None VIX
    vix_str = f"{catalyst_info.vix_level:.1f}" if catalyst_info.vix_level is not None else "UNKNOWN"
    vix_warn = " ⚠️ HIGH" if catalyst_info.is_high_regime() else ""
    parts.append(f"""
📅 CATALYST: Score {catalyst_info.score.value}/3 ({catalyst_info.score.name})
   VIX: {vix_str}{vix_warn}
   {'Earnings: ' + str(catalyst_info.earnings_days) + 'd' if catalyst_info.earnings_days else ''}
   {'; '.join(catalyst_info.reasons) if catalyst_info.reasons else ''}""")

    # Reliability section - handle None values
    acc_str = f"{reliability.accuracy_ewma_5d:.1%}" if reliability.accuracy_ewma_5d is not None else "N/A"
    bias_str = f"{reliability.bias_scaled_5d:.3f}" if reliability.bias_scaled_5d is not None else "N/A"
    ml_status = reliability.get_status(5)
    ml_blocked = ml_status.startswith("BLOCKED")

    parts.append(f"""
📊 ML RELIABILITY ({reliability.sample_count_5d} samples):
   Status: {ml_status} | Weight: {reliability.get_reliability_weight(5):.0%}
   Accuracy (EWMA): {acc_str} | Bias: {bias_str}
   Strong-Eligible: {'✅' if reliability.is_strong_signal_eligible(5) else '❌'}""")

    # Only show shrinkage/calibration details if ML is not blocked
    if not ml_blocked:
        parts.append(f"""
📉 SHRINKAGE: k={shrinkage.k_factor:.2f}
   Base: {shrinkage.k_base:.2f} | Catalyst: +{shrinkage.k_catalyst:.2f} | VIX: -{shrinkage.k_regime_penalty:.2f} | Uncertainty: -{shrinkage.k_uncertainty_penalty:.2f}
   Raw 5d: {shrinkage.pred_5d_raw:+.2f}% → Shrunk: {shrinkage.pred_5d_shrunk:+.2f}%""")

        # Calibration section - show availability status
        calibration_status = "✅ Active" if calibrator.loaded else "⚠️ UNAVAILABLE (using raw)"
        if raw_prob > p_smoothed + 0.15:
            confidence_note = "⚠️ Overconfident - reduce trust"
        elif not calibrator.loaded:
            confidence_note = "⚠️ Not calibrated - reduce trust"
        else:
            confidence_note = "✅ Calibrated"

        parts.append(f"""
📈 CALIBRATION ({calibration_status}):
   Raw P(Win): {raw_prob:.1%} → Calibrated: {p_calibrated:.1%} → Smoothed: {p_smoothed:.1%}
   {confidence_note}""")

        parts.append(f"""
🧠 ALPHA MODEL (USE THESE VALUES):
   Signal: {alpha_prediction.get('signal', 'N/A')} (Strength: {alpha_prediction.get('conviction', 0):.0%})
   Expected 5d: {shrinkage.pred_5d_shrunk:+.2f}% | P(Win): {p_smoothed:.1%}
   Regime: {alpha_prediction.get('regime', 'UNKNOWN')}""")
    else:
        # ML is blocked - show clear message, don't display meaningless predictions
        parts.append(f"""
🧠 ALPHA MODEL: ⛔ BLOCKED ({ml_status})
   ML predictions are NOT reliable for this ticker (insufficient historical data).
   Decision based on PLATFORM SIGNAL ONLY: {platform_signal} ({platform_score:.0f}%)
   ⚠️ Ignore any ML signal/regime values - they have no statistical validity.""")
    bullish = alpha_prediction.get('top_bullish_factors', [])[:2]
    bearish = alpha_prediction.get('top_bearish_factors', [])[:2]
    if bullish:
        parts.append("   Bullish: " + ", ".join([f"{f}:+{c:.2f}" for f, c in bullish]))
    if bearish:
        parts.append("   Bearish: " + ", ".join([f"{f}:{c:.2f}" for f, c in bearish]))

    # Add sector context
    sector_ctx = get_sector_context(ticker)
    sector_str = get_sector_context_string(sector_ctx)
    if sector_str:
        parts.append("")
        parts.append(sector_str)

    # Add sentiment velocity
    sent_velocity = get_sentiment_velocity(ticker)
    velocity_str = get_sentiment_velocity_string(sent_velocity)
    if velocity_str:
        parts.append("")
        parts.append(velocity_str)

    return "\n".join(parts)


# Convenience function for chat.py integration
def get_enhanced_alpha_context(ticker: str, alpha_prediction: Dict, platform_score: float,
                               platform_signal: str, technical_score: float = 50) -> str:
    return build_enhanced_alpha_context(ticker, alpha_prediction, platform_score, platform_signal, technical_score)


if __name__ == "__main__":
    print("\n" + "="*60 + "\nALPHA ENHANCEMENTS TEST\n" + "="*60)
    rel = get_reliability_metrics(90)
    print(f"Reliability: {rel.get_status(5)} | Weight: {rel.get_reliability_weight(5):.0%}")
    cat = detect_catalysts("AMD", days_to_earnings=5, vix_level=22)
    print(f"Catalyst: {cat.score.value}/3 | Reasons: {cat.reasons}")
    shr = apply_forecast_shrinkage(10.0, 15.0, 25.0, cat, rel)
    print(f"Shrinkage: k={shr.k_factor:.2f} | Raw +10% → Shrunk {shr.pred_5d_shrunk:+.2f}%")
    pol = compute_decision_policy(59, "HOLD", "STRONG_BUY", shr.pred_5d_shrunk, 0.62, rel, cat, 38)
    print(f"Policy: {pol.action.value} | Size: {pol.size_cap}x | Trade: {'YES' if pol.trade_allowed else 'NO'}")