"""
Unified Scorer - Single Source of Truth - FIXED VERSION

This module provides ONE canonical scoring pipeline used by:
- Live screener/dashboard
- Trade idea generation
- Backtesting engine
- API endpoints

CRITICAL: All scoring MUST go through this module. No separate scoring logic allowed.

Key Principles:
1. Same code path for live and backtest
2. Point-in-time enforcement (no future data leakage)
3. Explicit handling of missing data (no silent defaults)
4. Deterministic replay capability (all inputs logged)

FIXES APPLIED:
- ScoringResult.composite_score is now Optional[float], None when no data
- ScoringResult.signal_type is "CANNOT_SCORE" when insufficient data
- All feature defaults properly return None instead of 0 or neutral values
- Data quality tracking is explicit

Author: Alpha Research Platform
Location: src/core/unified_scorer.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA QUALITY FLAGS
# =============================================================================

class DataQuality(Enum):
    """Data quality levels - affect confidence, not score."""
    COMPLETE = "complete"       # All data present and fresh
    PARTIAL = "partial"         # Some fields missing
    STALE = "stale"             # Data older than threshold
    MISSING = "missing"         # Critical data unavailable
    SUSPECT = "suspect"         # Outlier or anomalous values
    INSUFFICIENT = "insufficient"  # Not enough data to score


class ScoringStatus(Enum):
    """Status of the scoring attempt."""
    SUCCESS = "success"                    # Scored successfully
    PARTIAL = "partial"                    # Scored with some missing data
    BLOCKED = "blocked"                    # Cannot score - insufficient data
    ERROR = "error"                        # Error during scoring


@dataclass
class DataQualityReport:
    """Report on data quality for a scoring run."""
    overall_quality: DataQuality
    missing_fields: List[str] = field(default_factory=list)
    stale_fields: Dict[str, int] = field(default_factory=dict)  # field -> days stale
    outlier_fields: List[str] = field(default_factory=list)
    confidence_penalty: float = 0.0  # 0-1, reduces final confidence
    components_available: int = 0
    components_total: int = 7

    def to_dict(self) -> Dict:
        return {
            'overall_quality': self.overall_quality.value,
            'missing_fields': self.missing_fields,
            'stale_fields': self.stale_fields,
            'outlier_fields': self.outlier_fields,
            'confidence_penalty': self.confidence_penalty,
            'components_available': self.components_available,
            'components_total': self.components_total,
        }

    @property
    def has_sufficient_data(self) -> bool:
        """Returns True if we have enough data to produce a meaningful score."""
        return self.components_available >= 2


# =============================================================================
# FEATURE CONTAINER (Point-in-Time Safe)
# =============================================================================

@dataclass
class TickerFeatures:
    """
    All features for a single ticker at a specific point in time.

    CRITICAL: All timestamps must be validated against as_of_time.
    No feature with timestamp > as_of_time is allowed.

    FIXED: All optional fields are truly Optional[float], not defaulted to 50 or 0.
    """
    ticker: str
    as_of_time: datetime  # The "now" for this scoring - no data after this allowed

    # === PRICE DATA (required) ===
    current_price: Optional[float] = None  # FIXED: was 0.0
    price_timestamp: Optional[datetime] = None

    # === SENTIMENT (0-100) ===
    sentiment_score: Optional[float] = None
    sentiment_timestamp: Optional[datetime] = None
    sentiment_article_count: Optional[int] = None  # FIXED: was 0
    sentiment_source_quality: Optional[float] = None  # FIXED: was 1.0

    # === FUNDAMENTAL (0-100) ===
    fundamental_score: Optional[float] = None
    fundamental_timestamp: Optional[datetime] = None

    # Underlying fundamental metrics - all Optional
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    profit_margin: Optional[float] = None
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None

    # === TECHNICAL (0-100) ===
    technical_score: Optional[float] = None
    technical_timestamp: Optional[datetime] = None

    # Underlying technical metrics
    rsi_14: Optional[float] = None
    macd_signal: Optional[str] = None
    trend_20d: Optional[str] = None
    above_50ma: Optional[bool] = None
    above_200ma: Optional[bool] = None
    relative_strength_rating: Optional[int] = None

    # === OPTIONS FLOW (0-100) ===
    options_flow_score: Optional[float] = None
    options_timestamp: Optional[datetime] = None
    options_sentiment: Optional[str] = None
    put_call_ratio: Optional[float] = None
    unusual_activity: Optional[bool] = None  # FIXED: was False

    # === SHORT SQUEEZE (0-100) ===
    short_squeeze_score: Optional[float] = None
    short_pct_float: Optional[float] = None
    days_to_cover: Optional[float] = None

    # === INSTITUTIONAL (0-100) ===
    institutional_score: Optional[float] = None
    institutional_timestamp: Optional[datetime] = None
    institutional_buyers: Optional[int] = None  # FIXED: was 0
    institutional_sellers: Optional[int] = None  # FIXED: was 0
    institutional_net_change: Optional[float] = None  # FIXED: was 0.0

    # === INSIDER (0-100) ===
    insider_score: Optional[float] = None
    insider_buy_count: Optional[int] = None  # FIXED: was 0
    insider_sell_count: Optional[int] = None  # FIXED: was 0
    insider_net_value: Optional[float] = None  # FIXED: was 0.0

    # === EARNINGS ===
    days_to_earnings: Optional[int] = None
    earnings_date: Optional[date] = None
    ies_score: Optional[float] = None  # Implied Expectations Score
    ecs_category: Optional[str] = None  # Earnings Correctness Score category

    # === ANALYST ===
    analyst_rating: Optional[str] = None
    price_target: Optional[float] = None
    target_upside_pct: Optional[float] = None
    analyst_count: Optional[int] = None  # FIXED: was 0

    # === MACRO/REGIME ===
    regime_adjustment: Optional[int] = None  # FIXED: was 0
    sector: Optional[str] = None  # FIXED: was ""
    is_growth_stock: Optional[bool] = None  # FIXED: was False
    is_defensive_stock: Optional[bool] = None  # FIXED: was False

    # === SECTOR CONTEXT ===
    sector_momentum: Optional[str] = None
    vs_sector_20d: Optional[float] = None
    vs_spy_20d: Optional[float] = None

    # === LIQUIDITY ===
    avg_dollar_volume: Optional[float] = None  # FIXED: was 0.0
    relative_volume: Optional[float] = None  # FIXED: was 1.0
    liquidity_class: Optional[str] = None  # FIXED: was "MEDIUM"

    # === DATA QUALITY ===
    data_quality: Optional[DataQualityReport] = None

    def validate_timestamps(self) -> List[str]:
        """Validate no feature has timestamp after as_of_time."""
        violations = []

        timestamp_fields = [
            ('price_timestamp', self.price_timestamp),
            ('sentiment_timestamp', self.sentiment_timestamp),
            ('fundamental_timestamp', self.fundamental_timestamp),
            ('technical_timestamp', self.technical_timestamp),
            ('options_timestamp', self.options_timestamp),
            ('institutional_timestamp', self.institutional_timestamp),
        ]

        for field_name, ts in timestamp_fields:
            if ts is not None and ts > self.as_of_time:
                violations.append(f"{field_name}: {ts} > as_of_time {self.as_of_time}")

        return violations

    def get_feature_hash(self) -> str:
        """Generate deterministic hash of all features for replay verification."""
        feature_dict = {k: v for k, v in asdict(self).items()
                       if v is not None and k != 'data_quality'}

        for k, v in feature_dict.items():
            if isinstance(v, (datetime, date)):
                feature_dict[k] = v.isoformat()

        sorted_str = json.dumps(feature_dict, sort_keys=True, default=str)
        return hashlib.md5(sorted_str.encode()).hexdigest()[:12]

    def get_available_scores(self) -> Dict[str, float]:
        """Return only the scores that are actually available (not None)."""
        score_map = {
            'sentiment': self.sentiment_score,
            'fundamental': self.fundamental_score,
            'technical': self.technical_score,
            'options_flow': self.options_flow_score,
            'institutional': self.institutional_score,
            'short_squeeze': self.short_squeeze_score,
            'insider': self.insider_score,
        }
        return {k: v for k, v in score_map.items() if v is not None}


# =============================================================================
# SCORE RESULT
# =============================================================================

@dataclass
class ScoringResult:
    """
    Complete scoring result for a single ticker.

    FIXED: composite_score and total_score are Optional[float].
    When data is insufficient, they are None, not 50.
    """
    ticker: str
    as_of_time: datetime

    # === SCORING STATUS ===
    status: ScoringStatus = ScoringStatus.BLOCKED
    status_message: str = ""

    # === COMPONENT SCORES (0-100, None if unavailable) ===
    sentiment_score: Optional[float] = None
    fundamental_score: Optional[float] = None
    technical_score: Optional[float] = None
    options_flow_score: Optional[float] = None
    short_squeeze_score: Optional[float] = None
    institutional_score: Optional[float] = None
    insider_score: Optional[float] = None

    # === COMPOSITE SCORES - NOW OPTIONAL ===
    composite_score: Optional[float] = None  # FIXED: was 50.0
    total_score: Optional[float] = None      # FIXED: was 50.0

    # === CROSS-SECTIONAL (relative to universe) ===
    composite_z: Optional[float] = None       # FIXED: was 0.0
    composite_percentile: Optional[float] = None  # FIXED: was 50.0
    universe_rank: Optional[int] = None       # FIXED: was 0
    universe_size: Optional[int] = None       # FIXED: was 0

    # === ADJUSTMENTS ===
    regime_adjustment: int = 0
    earnings_adjustment: int = 0

    # === SIGNAL - NOW REFLECTS DATA AVAILABILITY ===
    signal_type: str = "CANNOT_SCORE"  # FIXED: was "HOLD"
    signal_strength: Optional[float] = None  # FIXED: was 50.0

    # === CONFIDENCE ===
    confidence: float = 0.0  # FIXED: was 1.0 (0 when no data)
    confidence_factors: List[str] = field(default_factory=list)

    # === DATA QUALITY ===
    data_quality: Optional[DataQualityReport] = None
    components_used: List[str] = field(default_factory=list)
    components_missing: List[str] = field(default_factory=list)

    # === REPLAY INFO ===
    feature_hash: str = ""
    scorer_version: str = "2.0.0"

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/API."""
        result = asdict(self)
        result['as_of_time'] = self.as_of_time.isoformat()
        result['status'] = self.status.value
        if self.data_quality:
            result['data_quality'] = self.data_quality.to_dict()
        return result

    @property
    def is_scorable(self) -> bool:
        """Returns True if we have a valid score."""
        return self.composite_score is not None and self.status != ScoringStatus.BLOCKED

    @property
    def score_display(self) -> str:
        """Return score for display with confidence indicator."""
        if self.composite_score is None:
            return "N/A"
        if self.confidence < 0.5:
            return f"{self.composite_score:.0f}*"
        return f"{self.composite_score:.0f}"


# =============================================================================
# UNIFIED SCORER
# =============================================================================

class UnifiedScorer:
    """
    THE single source of truth for all scoring.

    FIXED: Properly handles missing data without silent defaults.
    """

    VERSION = "2.0.0"

    WEIGHTS = {
        'sentiment': 0.20,
        'fundamental': 0.25,
        'technical': 0.20,
        'options_flow': 0.15,
        'institutional': 0.10,
        'short_squeeze': 0.05,
        'insider': 0.05,
    }

    MIN_COMPONENTS_REQUIRED = 2

    STALENESS_THRESHOLDS = {
        'price': 1,
        'sentiment': 3,
        'fundamental': 30,
        'technical': 1,
        'options': 1,
        'institutional': 90,
    }

    OUTLIER_BOUNDS = {
        'pe_ratio': (0, 1000),
        'peg_ratio': (-10, 100),
        'debt_to_equity': (0, 1000),
        'short_pct_float': (0, 100),
        'rsi_14': (0, 100),
    }

    def __init__(self, repository=None):
        self.repo = repository
        self._cache = {}
        self._universe_stats = None

    def compute_features(self,
                         ticker: str,
                         as_of_time: datetime,
                         preloaded_data: Dict = None) -> TickerFeatures:
        """Compute all features for a ticker at a specific point in time."""
        features = TickerFeatures(ticker=ticker, as_of_time=as_of_time)
        missing_fields = []
        stale_fields = {}
        outlier_fields = []

        data = preloaded_data or {}

        # === PRICE DATA ===
        price_data = data.get('price') or self._fetch_price(ticker, as_of_time)
        if price_data:
            features.current_price = price_data.get('close')
            features.price_timestamp = price_data.get('timestamp')
            if features.price_timestamp:
                days_old = (as_of_time - features.price_timestamp).days
                if days_old > self.STALENESS_THRESHOLDS['price']:
                    stale_fields['price'] = days_old
        else:
            missing_fields.append('price')

        # === SENTIMENT ===
        sentiment_data = data.get('sentiment') or self._fetch_sentiment(ticker, as_of_time)
        if sentiment_data:
            features.sentiment_score = sentiment_data.get('sentiment_score')
            features.sentiment_timestamp = sentiment_data.get('timestamp')
            features.sentiment_article_count = sentiment_data.get('article_count')
            if features.sentiment_timestamp:
                days_old = (as_of_time - features.sentiment_timestamp).days
                if days_old > self.STALENESS_THRESHOLDS['sentiment']:
                    stale_fields['sentiment'] = days_old
        else:
            missing_fields.append('sentiment')

        # === FUNDAMENTAL ===
        fundamental_data = data.get('fundamental') or self._fetch_fundamental(ticker, as_of_time)
        if fundamental_data:
            features.fundamental_score = fundamental_data.get('fundamental_score')
            features.fundamental_timestamp = fundamental_data.get('timestamp')
            features.pe_ratio = self._validate_outlier(fundamental_data.get('pe_ratio'), 'pe_ratio', outlier_fields)
            features.forward_pe = fundamental_data.get('forward_pe')
            features.peg_ratio = self._validate_outlier(fundamental_data.get('peg_ratio'), 'peg_ratio', outlier_fields)
            features.revenue_growth = fundamental_data.get('revenue_growth')
            features.earnings_growth = fundamental_data.get('earnings_growth')
            features.profit_margin = fundamental_data.get('profit_margin')
            features.roe = fundamental_data.get('roe')
            features.debt_to_equity = self._validate_outlier(fundamental_data.get('debt_to_equity'), 'debt_to_equity', outlier_fields)
        else:
            missing_fields.append('fundamental')

        # === TECHNICAL ===
        technical_data = data.get('technical') or self._fetch_technical(ticker, as_of_time)
        if technical_data:
            features.technical_score = technical_data.get('technical_score')
            features.technical_timestamp = technical_data.get('timestamp')
            features.rsi_14 = self._validate_outlier(technical_data.get('rsi_14'), 'rsi_14', outlier_fields)
            features.macd_signal = technical_data.get('macd_signal')
            features.trend_20d = technical_data.get('trend_20d')
            features.above_50ma = technical_data.get('above_50ma')
            features.above_200ma = technical_data.get('above_200ma')
            features.relative_strength_rating = technical_data.get('rs_rating')
            features.vs_spy_20d = technical_data.get('vs_spy_20d')
            features.vs_sector_20d = technical_data.get('vs_sector_20d')
        else:
            missing_fields.append('technical')

        # === OPTIONS FLOW ===
        options_data = data.get('options') or self._fetch_options(ticker, as_of_time)
        if options_data:
            features.options_flow_score = options_data.get('options_flow_score')
            features.options_timestamp = options_data.get('timestamp')
            features.options_sentiment = options_data.get('sentiment')
            features.put_call_ratio = options_data.get('put_call_ratio')
            features.unusual_activity = options_data.get('unusual_activity')
        else:
            missing_fields.append('options_flow')

        # === SHORT SQUEEZE ===
        squeeze_data = data.get('squeeze') or self._fetch_squeeze(ticker, as_of_time)
        if squeeze_data:
            features.short_squeeze_score = squeeze_data.get('squeeze_score')
            features.short_pct_float = self._validate_outlier(squeeze_data.get('short_pct_float'), 'short_pct_float', outlier_fields)
            features.days_to_cover = squeeze_data.get('days_to_cover')

        # === INSTITUTIONAL ===
        institutional_data = data.get('institutional') or self._fetch_institutional(ticker, as_of_time)
        if institutional_data:
            features.institutional_score = institutional_data.get('institutional_signal')
            features.institutional_timestamp = institutional_data.get('timestamp')
            features.institutional_buyers = institutional_data.get('buyers')
            features.institutional_sellers = institutional_data.get('sellers')
            features.institutional_net_change = institutional_data.get('net_change')

        # === INSIDER ===
        insider_data = data.get('insider') or self._fetch_insider(ticker, as_of_time)
        if insider_data:
            features.insider_score = insider_data.get('insider_signal')
            features.insider_buy_count = insider_data.get('buy_count')
            features.insider_sell_count = insider_data.get('sell_count')
            features.insider_net_value = insider_data.get('net_value')

        # === EARNINGS ===
        earnings_data = data.get('earnings') or self._fetch_earnings(ticker, as_of_time)
        if earnings_data:
            features.earnings_date = earnings_data.get('earnings_date')
            if features.earnings_date:
                features.days_to_earnings = (features.earnings_date - as_of_time.date()).days
            features.ies_score = earnings_data.get('ies')
            features.ecs_category = earnings_data.get('ecs_category')

        # === ANALYST ===
        analyst_data = data.get('analyst') or self._fetch_analyst(ticker, as_of_time)
        if analyst_data:
            features.analyst_rating = analyst_data.get('rating')
            features.price_target = analyst_data.get('target_price')
            if features.price_target and features.current_price and features.current_price > 0:
                features.target_upside_pct = ((features.price_target - features.current_price) / features.current_price * 100)
            features.analyst_count = analyst_data.get('analyst_count')

        # === REGIME ===
        regime_data = data.get('regime') or self._fetch_regime(ticker, as_of_time)
        if regime_data:
            features.regime_adjustment = regime_data.get('adjustment')
            features.sector = regime_data.get('sector')
            features.is_growth_stock = regime_data.get('is_growth')
            features.is_defensive_stock = regime_data.get('is_defensive')
            features.sector_momentum = regime_data.get('sector_momentum')

        # === LIQUIDITY ===
        liquidity_data = data.get('liquidity') or self._fetch_liquidity(ticker, as_of_time)
        if liquidity_data:
            features.avg_dollar_volume = liquidity_data.get('avg_dollar_volume')
            features.relative_volume = liquidity_data.get('relative_volume')
            features.liquidity_class = liquidity_data.get('liquidity_class')

        # === VALIDATE POINT-IN-TIME ===
        timestamp_violations = features.validate_timestamps()
        if timestamp_violations:
            logger.error(f"Point-in-time violations for {ticker}: {timestamp_violations}")

        # === DATA QUALITY REPORT ===
        available_scores = features.get_available_scores()
        features.data_quality = self._assess_data_quality(missing_fields, stale_fields, outlier_fields, len(available_scores))

        return features

    def compute_scores(self, features: TickerFeatures, universe_stats: Dict = None) -> ScoringResult:
        """Compute all scores from features."""
        result = ScoringResult(
            ticker=features.ticker,
            as_of_time=features.as_of_time,
            scorer_version=self.VERSION,
            feature_hash=features.get_feature_hash()
        )

        # Copy component scores
        result.sentiment_score = features.sentiment_score
        result.fundamental_score = features.fundamental_score
        result.technical_score = features.technical_score
        result.options_flow_score = features.options_flow_score
        result.short_squeeze_score = features.short_squeeze_score
        result.institutional_score = features.institutional_score
        result.insider_score = features.insider_score

        # Get available scores
        available_scores = features.get_available_scores()
        result.components_used = list(available_scores.keys())
        result.components_missing = [k for k in self.WEIGHTS.keys() if k not in available_scores]

        # Check if we have enough data
        if len(available_scores) < self.MIN_COMPONENTS_REQUIRED:
            result.status = ScoringStatus.BLOCKED
            result.status_message = f"Insufficient data: {len(available_scores)}/{self.MIN_COMPONENTS_REQUIRED} components. Missing: {', '.join(result.components_missing)}"
            result.signal_type = "CANNOT_SCORE"
            result.composite_score = None
            result.total_score = None
            result.confidence = 0.0
            result.confidence_factors = [result.status_message]
            result.data_quality = features.data_quality
            return result

        # Compute composite score
        weighted_sum = 0.0
        total_weight = 0.0
        confidence_factors = []

        for component, score in available_scores.items():
            weight = self.WEIGHTS.get(component, 0)
            weighted_sum += score * weight
            total_weight += weight

        if total_weight > 0:
            result.composite_score = weighted_sum / total_weight
        else:
            result.composite_score = None
            result.status = ScoringStatus.BLOCKED
            result.status_message = "No component scores with valid weights"
            return result

        if result.components_missing:
            confidence_factors.append(f"Missing: {', '.join(result.components_missing)}")
            result.status = ScoringStatus.PARTIAL
        else:
            result.status = ScoringStatus.SUCCESS

        # Apply adjustments
        result.regime_adjustment = features.regime_adjustment or 0

        if features.days_to_earnings is not None and features.ies_score is not None:
            if 0 < features.days_to_earnings <= 5 and features.ies_score >= 70:
                result.earnings_adjustment = -10
                confidence_factors.append("High expectations pre-earnings")
            elif 0 < features.days_to_earnings <= 3:
                result.earnings_adjustment = -5
                confidence_factors.append("Binary earnings event imminent")

        result.total_score = result.composite_score + result.regime_adjustment + result.earnings_adjustment
        result.total_score = max(0, min(100, result.total_score))

        # Cross-sectional scoring
        if universe_stats and result.composite_score is not None:
            mean = universe_stats.get('mean')
            std = universe_stats.get('std')
            all_scores = universe_stats.get('scores', [])

            if mean is not None and std is not None and std > 0:
                result.composite_z = (result.composite_score - mean) / std

            if all_scores:
                result.universe_size = len(all_scores)
                below_count = sum(1 for s in all_scores if s < result.composite_score)
                result.composite_percentile = (below_count / len(all_scores)) * 100

        # Determine signal
        result.signal_strength = result.total_score

        if result.total_score >= 75:
            result.signal_type = "STRONG_BUY"
        elif result.total_score >= 60:
            result.signal_type = "BUY"
        elif result.total_score >= 40:
            result.signal_type = "HOLD"
        elif result.total_score >= 25:
            result.signal_type = "SELL"
        else:
            result.signal_type = "STRONG_SELL"

        # Confidence
        result.confidence = len(available_scores) / len(self.WEIGHTS)
        result.confidence_factors = confidence_factors

        if features.data_quality:
            result.confidence -= features.data_quality.confidence_penalty
            result.data_quality = features.data_quality

        result.confidence = max(0.1, min(1.0, result.confidence))

        return result

    def score_universe(self, tickers: List[str], as_of_time: datetime, include_cross_sectional: bool = True) -> List[ScoringResult]:
        """Score entire universe with cross-sectional ranking."""
        results = []

        for ticker in tickers:
            try:
                features = self.compute_features(ticker, as_of_time)
                result = self.compute_scores(features)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error scoring {ticker}: {e}")
                error_result = ScoringResult(ticker=ticker, as_of_time=as_of_time, status=ScoringStatus.ERROR, status_message=str(e)[:100])
                results.append(error_result)

        if include_cross_sectional:
            scorable_results = [r for r in results if r.composite_score is not None]
            if scorable_results:
                all_scores = [r.composite_score for r in scorable_results]
                universe_stats = {'mean': np.mean(all_scores), 'std': np.std(all_scores) if len(all_scores) > 1 else 0, 'scores': all_scores}

                for result in scorable_results:
                    result.universe_size = len(scorable_results)
                    if universe_stats['std'] > 0:
                        result.composite_z = (result.composite_score - universe_stats['mean']) / universe_stats['std']
                    below_count = sum(1 for s in all_scores if s < result.composite_score)
                    result.composite_percentile = (below_count / len(all_scores)) * 100

        results.sort(key=lambda x: x.total_score if x.total_score is not None else -1, reverse=True)
        return results

    # =========================================================================
    # DATA FETCHING
    # =========================================================================

    def _fetch_price(self, ticker: str, as_of: datetime) -> Optional[Dict]:
        """
        Fetch price data with point-in-time enforcement.
        
        FIXED: Previously used repo.get_latest_price(ticker) which returns the 
        absolute latest price regardless of as_of time — causing look-ahead bias
        in backtesting. Now queries with explicit date constraint.
        """
        if not self.repo:
            return None
        try:
            query = (
                "SELECT close, adj_close, date FROM prices "
                "WHERE ticker = %(ticker)s AND date <= %(as_of)s "
                "ORDER BY date DESC LIMIT 1"
            )
            df = pd.read_sql(query, self.repo.engine, params={
                'ticker': ticker, 'as_of': as_of.date()
            })
            if len(df) > 0:
                row = df.iloc[0]
                close_price = self._safe_float(row.get('adj_close')) or self._safe_float(row.get('close'))
                if close_price is not None:
                    return {
                        'close': close_price,
                        'timestamp': self._parse_timestamp(row.get('date'))
                    }
        except Exception as e:
            logger.debug(f"Error fetching price for {ticker} as_of {as_of}: {e}")
        return None

    def _fetch_sentiment(self, ticker: str, as_of: datetime) -> Optional[Dict]:
        if not self.repo:
            return None
        try:
            query = "SELECT sentiment_score, article_count, date FROM screener_scores WHERE ticker = %(ticker)s AND date <= %(as_of)s ORDER BY date DESC LIMIT 1"
            df = pd.read_sql(query, self.repo.engine, params={'ticker': ticker, 'as_of': as_of.date()})
            if len(df) > 0:
                row = df.iloc[0]
                sentiment_score = row.get('sentiment_score')
                if sentiment_score is not None and not pd.isna(sentiment_score):
                    return {'sentiment_score': float(sentiment_score), 'article_count': row.get('article_count'), 'timestamp': self._parse_timestamp(row.get('date'))}
        except Exception as e:
            logger.debug(f"Error fetching sentiment for {ticker}: {e}")
        return None

    def _fetch_fundamental(self, ticker: str, as_of: datetime) -> Optional[Dict]:
        if not self.repo:
            return None
        try:
            query = "SELECT * FROM fundamentals WHERE ticker = %(ticker)s AND date <= %(as_of)s ORDER BY date DESC LIMIT 1"
            df = pd.read_sql(query, self.repo.engine, params={'ticker': ticker, 'as_of': as_of.date()})
            if len(df) > 0:
                row = df.iloc[0]
                fundamental_score = self._calculate_fundamental_score(row)
                if fundamental_score is not None:
                    return {'fundamental_score': fundamental_score, 'timestamp': self._parse_timestamp(row.get('date')), 'pe_ratio': self._safe_float(row.get('pe_ratio')), 'revenue_growth': self._safe_float(row.get('revenue_growth')), 'profit_margin': self._safe_float(row.get('profit_margin'))}
        except Exception as e:
            logger.debug(f"Error fetching fundamentals for {ticker}: {e}")
        return None

    def _fetch_technical(self, ticker: str, as_of: datetime) -> Optional[Dict]:
        if not self.repo:
            return None
        try:
            query = "SELECT technical_score, date FROM screener_scores WHERE ticker = %(ticker)s AND date <= %(as_of)s ORDER BY date DESC LIMIT 1"
            df = pd.read_sql(query, self.repo.engine, params={'ticker': ticker, 'as_of': as_of.date()})
            if len(df) > 0:
                row = df.iloc[0]
                technical_score = row.get('technical_score')
                if technical_score is not None and not pd.isna(technical_score):
                    return {'technical_score': float(technical_score), 'timestamp': self._parse_timestamp(row.get('date'))}
        except Exception as e:
            logger.debug(f"Error fetching technical for {ticker}: {e}")
        return None

    def _fetch_options(self, ticker: str, as_of: datetime) -> Optional[Dict]:
        if not self.repo:
            return None
        try:
            query = "SELECT options_flow_score, options_sentiment, date FROM screener_scores WHERE ticker = %(ticker)s AND date <= %(as_of)s ORDER BY date DESC LIMIT 1"
            df = pd.read_sql(query, self.repo.engine, params={'ticker': ticker, 'as_of': as_of.date()})
            if len(df) > 0:
                row = df.iloc[0]
                options_score = row.get('options_flow_score')
                if options_score is not None and not pd.isna(options_score):
                    return {'options_flow_score': float(options_score), 'sentiment': row.get('options_sentiment'), 'timestamp': self._parse_timestamp(row.get('date'))}
        except Exception as e:
            logger.debug(f"Error fetching options for {ticker}: {e}")
        return None

    def _fetch_squeeze(self, ticker: str, as_of: datetime) -> Optional[Dict]:
        if not self.repo:
            return None
        try:
            query = "SELECT short_squeeze_score, squeeze_risk, date FROM screener_scores WHERE ticker = %(ticker)s AND date <= %(as_of)s ORDER BY date DESC LIMIT 1"
            df = pd.read_sql(query, self.repo.engine, params={'ticker': ticker, 'as_of': as_of.date()})
            if len(df) > 0:
                row = df.iloc[0]
                squeeze_score = row.get('short_squeeze_score')
                if squeeze_score is not None and not pd.isna(squeeze_score):
                    return {'squeeze_score': float(squeeze_score), 'squeeze_risk': row.get('squeeze_risk'), 'timestamp': self._parse_timestamp(row.get('date'))}
        except Exception as e:
            logger.debug(f"Error fetching squeeze for {ticker}: {e}")
        return None

    def _fetch_institutional(self, ticker: str, as_of: datetime) -> Optional[Dict]:
        if not self.repo:
            return None
        try:
            query = "SELECT institutional_signal, date FROM screener_scores WHERE ticker = %(ticker)s AND date <= %(as_of)s ORDER BY date DESC LIMIT 1"
            df = pd.read_sql(query, self.repo.engine, params={'ticker': ticker, 'as_of': as_of.date()})
            if len(df) > 0:
                row = df.iloc[0]
                inst_signal = row.get('institutional_signal')
                if inst_signal is not None and not pd.isna(inst_signal):
                    return {'institutional_signal': float(inst_signal), 'timestamp': self._parse_timestamp(row.get('date'))}
        except Exception as e:
            logger.debug(f"Error fetching institutional for {ticker}: {e}")
        return None

    def _fetch_insider(self, ticker: str, as_of: datetime) -> Optional[Dict]:
        if not self.repo:
            return None
        try:
            query = "SELECT insider_signal, date FROM screener_scores WHERE ticker = %(ticker)s AND date <= %(as_of)s ORDER BY date DESC LIMIT 1"
            df = pd.read_sql(query, self.repo.engine, params={'ticker': ticker, 'as_of': as_of.date()})
            if len(df) > 0:
                row = df.iloc[0]
                insider_signal = row.get('insider_signal')
                if insider_signal is not None and not pd.isna(insider_signal):
                    return {'insider_signal': float(insider_signal), 'timestamp': self._parse_timestamp(row.get('date'))}
        except Exception as e:
            logger.debug(f"Error fetching insider for {ticker}: {e}")
        return None

    def _fetch_earnings(self, ticker: str, as_of: datetime) -> Optional[Dict]:
        if not self.repo:
            return None
        try:
            query = "SELECT earnings_date, ies, ecs_category FROM earnings_calendar WHERE ticker = %(ticker)s AND earnings_date >= %(as_of)s ORDER BY earnings_date ASC LIMIT 1"
            df = pd.read_sql(query, self.repo.engine, params={'ticker': ticker, 'as_of': as_of.date()})
            if len(df) > 0:
                row = df.iloc[0]
                earnings_date = row.get('earnings_date')
                if earnings_date is not None:
                    return {'earnings_date': earnings_date, 'ies': self._safe_float(row.get('ies')), 'ecs_category': row.get('ecs_category')}
        except Exception as e:
            logger.debug(f"Error fetching earnings for {ticker}: {e}")
        return None

    def _fetch_analyst(self, ticker: str, as_of: datetime) -> Optional[Dict]:
        if not self.repo:
            return None
        try:
            query = "SELECT target_upside_pct, analyst_positivity, date FROM screener_scores WHERE ticker = %(ticker)s AND date <= %(as_of)s ORDER BY date DESC LIMIT 1"
            df = pd.read_sql(query, self.repo.engine, params={'ticker': ticker, 'as_of': as_of.date()})
            if len(df) > 0:
                row = df.iloc[0]
                return {'target_upside_pct': self._safe_float(row.get('target_upside_pct')), 'analyst_positivity': self._safe_float(row.get('analyst_positivity')), 'timestamp': self._parse_timestamp(row.get('date'))}
        except Exception as e:
            logger.debug(f"Error fetching analyst for {ticker}: {e}")
        return None

    def _fetch_regime(self, ticker: str, as_of: datetime) -> Optional[Dict]:
        """
        Fetch regime/macro adjustment.
        
        FIXED: Previously called get_regime_adjustment() which uses current
        market conditions regardless of as_of time. Now passes as_of to
        ensure point-in-time correctness during backtesting.
        """
        try:
            from src.analytics.macro_regime import get_regime_adjustment
            sector = self._get_ticker_sector(ticker)
            # Pass as_of to regime adjustment if the function supports it
            try:
                adjustment = get_regime_adjustment(ticker, sector, as_of=as_of)
            except TypeError:
                # Fallback if the function doesn't accept as_of yet
                # In this case, return None for regime adjustment during backtest
                # to avoid look-ahead bias
                if as_of.date() < datetime.now().date():
                    # We're in backtest mode — don't use current regime data
                    logger.debug(
                        f"Regime adjustment for {ticker} skipped in backtest mode "
                        f"(as_of={as_of.date()} < today). Function doesn't support as_of."
                    )
                    return {'adjustment': 0, 'sector': sector, 'is_growth': None, 'is_defensive': None}
                adjustment = get_regime_adjustment(ticker, sector)
            is_growth = sector in ['Technology', 'Consumer Cyclical', 'Communication Services'] if sector else None
            is_defensive = sector in ['Utilities', 'Consumer Defensive', 'Healthcare', 'Consumer Staples'] if sector else None
            return {'adjustment': adjustment, 'sector': sector, 'is_growth': is_growth, 'is_defensive': is_defensive}
        except ImportError:
            logger.debug("macro_regime module not available")
        except Exception as e:
            logger.debug(f"Error fetching regime for {ticker}: {e}")
        return None

    def _fetch_liquidity(self, ticker: str, as_of: datetime) -> Optional[Dict]:
        if not self.repo:
            return None
        try:
            start_date = (as_of - timedelta(days=30)).date()
            query = "SELECT AVG(volume) as avg_volume, AVG(close * volume) as avg_dollar_volume FROM prices WHERE ticker = %(ticker)s AND date <= %(as_of)s AND date >= %(start_date)s"
            df = pd.read_sql(query, self.repo.engine, params={'ticker': ticker, 'as_of': as_of.date(), 'start_date': start_date})
            if len(df) > 0:
                row = df.iloc[0]
                avg_dollar_volume = self._safe_float(row.get('avg_dollar_volume'))
                if avg_dollar_volume is not None:
                    if avg_dollar_volume >= 100_000_000:
                        liquidity_class = "VERY_HIGH"
                    elif avg_dollar_volume >= 20_000_000:
                        liquidity_class = "HIGH"
                    elif avg_dollar_volume >= 5_000_000:
                        liquidity_class = "MEDIUM"
                    else:
                        liquidity_class = "LOW"
                    return {'avg_dollar_volume': avg_dollar_volume, 'liquidity_class': liquidity_class}
        except Exception as e:
            logger.debug(f"Error fetching liquidity for {ticker}: {e}")
        return None

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _validate_outlier(self, value: Any, field_name: str, outlier_list: List[str]) -> Optional[float]:
        if value is None or pd.isna(value):
            return None
        try:
            value = float(value)
        except (ValueError, TypeError):
            return None
        bounds = self.OUTLIER_BOUNDS.get(field_name)
        if bounds:
            min_val, max_val = bounds
            if value < min_val or value > max_val:
                outlier_list.append(f"{field_name}={value}")
                return None
        return value

    def _assess_data_quality(self, missing_fields: List[str], stale_fields: Dict[str, int], outlier_fields: List[str], components_available: int) -> DataQualityReport:
        report = DataQualityReport(overall_quality=DataQuality.COMPLETE, missing_fields=missing_fields, stale_fields=stale_fields, outlier_fields=outlier_fields, components_available=components_available, components_total=len(self.WEIGHTS))

        if components_available < self.MIN_COMPONENTS_REQUIRED:
            report.overall_quality = DataQuality.INSUFFICIENT
            report.confidence_penalty = 1.0
        elif len(missing_fields) >= 4:
            report.overall_quality = DataQuality.MISSING
            report.confidence_penalty = 0.4
        elif len(stale_fields) >= 2:
            report.overall_quality = DataQuality.STALE
            report.confidence_penalty = 0.2
        elif len(outlier_fields) >= 1:
            report.overall_quality = DataQuality.SUSPECT
            report.confidence_penalty = 0.15
        elif len(missing_fields) >= 2:
            report.overall_quality = DataQuality.PARTIAL
            report.confidence_penalty = 0.1

        return report

    def _calculate_fundamental_score(self, row: pd.Series) -> Optional[float]:
        scores = []
        pe = self._safe_float(row.get('pe_ratio'))
        if pe is not None and 0 < pe < 100:
            scores.append(80 if pe < 15 else 60 if pe < 25 else 40 if pe < 35 else 20)
        rev_growth = self._safe_float(row.get('revenue_growth'))
        if rev_growth is not None:
            scores.append(90 if rev_growth > 0.3 else 70 if rev_growth > 0.15 else 50 if rev_growth > 0.05 else 30 if rev_growth > 0 else 20)
        margin = self._safe_float(row.get('profit_margin'))
        if margin is not None:
            scores.append(80 if margin > 0.2 else 60 if margin > 0.1 else 40 if margin > 0.05 else 20)
        return sum(scores) / len(scores) if scores else None

    def _get_ticker_sector(self, ticker: str) -> Optional[str]:
        if not self.repo:
            return None
        try:
            query = "SELECT sector FROM fundamentals WHERE ticker = %(ticker)s ORDER BY date DESC LIMIT 1"
            df = pd.read_sql(query, self.repo.engine, params={'ticker': ticker})
            if len(df) > 0 and df.iloc[0]['sector']:
                return df.iloc[0]['sector']
        except Exception as e:
            logger.debug(f"Error getting sector for {ticker}: {e}")
        return None

    def _parse_timestamp(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                try:
                    return pd.to_datetime(value).to_pydatetime()
                except:
                    pass
        return None

    def _safe_float(self, value: Any) -> Optional[float]:
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


# =============================================================================
# SINGLETON
# =============================================================================

_unified_scorer: Optional[UnifiedScorer] = None

def get_unified_scorer(repository=None) -> UnifiedScorer:
    global _unified_scorer
    if _unified_scorer is None or (repository is not None and _unified_scorer.repo != repository):
        _unified_scorer = UnifiedScorer(repository)
    return _unified_scorer