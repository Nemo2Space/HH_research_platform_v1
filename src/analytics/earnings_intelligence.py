"""
Earnings Intelligence Module - FIXED VERSION

Solves the "great report, bad trade" problem by implementing:
- IES (Implied Expectations Score): Estimates what's priced in before earnings
- ECS (Expectations Clearance Score): Whether results cleared the bar
- Regime Classification: HYPED, FEARED, VOLATILE, NORMAL
- Position Scaling: Risk-adjusted position sizing

CRITICAL FIX: All component calculations now return Optional[float] with explicit
status tracking. No more silent 50.0 defaults that mask missing data.

Author: Alpha Research Platform
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
from datetime import datetime, date, timedelta
import yfinance as yf

from src.utils.logging import get_logger

logger = get_logger(__name__)


class IESRegime(Enum):
    """Expectation regime classification."""
    HYPED = "HYPED"          # IES > 75, market expects blowout
    FEARED = "FEARED"        # IES < 25, market expects disaster
    VOLATILE = "VOLATILE"    # High implied move, uncertain
    NORMAL = "NORMAL"        # Standard expectations
    UNKNOWN = "UNKNOWN"      # Not enough data to classify


class ECSCategory(Enum):
    """Expectations Clearance Score categories."""
    STRONG_BEAT = "STRONG_BEAT"    # Crushed elevated expectations
    BEAT = "BEAT"                   # Cleared the bar
    INLINE = "INLINE"               # Met expectations
    MISS = "MISS"                   # Failed to clear
    STRONG_MISS = "STRONG_MISS"    # Badly missed low expectations
    PENDING = "PENDING"             # Pre-earnings
    UNKNOWN = "UNKNOWN"             # Cannot determine


class ComponentStatus(Enum):
    """Status of each IES component calculation."""
    CALCULATED = "CALCULATED"       # Successfully computed from real data
    NO_DATA = "NO_DATA"             # Data source returned empty
    ERROR = "ERROR"                  # Exception during calculation
    INSUFFICIENT = "INSUFFICIENT"   # Not enough data points
    NOT_APPLICABLE = "N/A"          # Component doesn't apply


@dataclass
class ComponentResult:
    """Result of a single component calculation."""
    value: Optional[float]  # None if not calculated
    status: ComponentStatus
    message: str = ""       # Explains why if not CALCULATED

    @property
    def is_available(self) -> bool:
        return self.value is not None and self.status == ComponentStatus.CALCULATED


@dataclass
class EarningsEnrichment:
    """Complete earnings intelligence for a ticker."""
    ticker: str

    # Core scores - NOW OPTIONAL
    ies: Optional[float] = None         # 0-100, Implied Expectations Score (None = not calculated)
    eqs: Optional[float] = None         # 0-100, Earnings Quality Score (post-earnings)
    regime: IESRegime = IESRegime.UNKNOWN
    ecs_category: ECSCategory = ECSCategory.PENDING

    # Timing
    earnings_date: Optional[date] = None
    days_to_earnings: Optional[int] = None  # None = unknown, not 999
    in_compute_window: bool = False   # Within 10 days (for display)
    in_action_window: bool = False    # Within 5 days (for trading decisions)
    is_post_earnings: bool = False    # Earnings already happened (within 7 days)

    # IES Components - NOW OPTIONAL (None = not calculated)
    pre_earnings_runup: Optional[float] = None
    implied_move_percentile: Optional[float] = None
    analyst_revision_momentum: Optional[float] = None
    options_skew_score: Optional[float] = None
    news_sentiment_score: Optional[float] = None
    historical_beat_rate: Optional[float] = None
    sector_momentum: Optional[float] = None

    # Component status tracking - NEW
    component_status: Dict[str, str] = field(default_factory=dict)
    # e.g., {'pre_earnings_runup': 'CALCULATED', 'options_skew': 'NO_DATA: No options available'}

    # Post-earnings actual results
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    eps_surprise_pct: Optional[float] = None  # Changed from 0.0
    revenue_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_surprise_pct: Optional[float] = None  # Changed from 0.0

    # Z-scores for ECS calculation
    eps_surprise_z: Optional[float] = None
    revenue_surprise_z: Optional[float] = None
    guidance_z: Optional[float] = None
    event_z: Optional[float] = None           # Blended surprise z-score

    # Market reaction (post-earnings)
    gap_pct: Optional[float] = None           # Open vs previous close
    intraday_move_pct: Optional[float] = None # Close vs open on ER day
    total_reaction_pct: Optional[float] = None # Close vs previous close

    # Trading adjustments
    position_scale: float = 1.0    # 0.2 to 1.0
    score_adjustment: int = 0      # -20 to +20
    risk_flags: List[str] = field(default_factory=list)

    # Data quality - ENHANCED
    data_quality: str = "UNKNOWN"      # HIGH, MEDIUM, LOW, INSUFFICIENT, UNKNOWN
    components_available: int = 0       # Count of successfully calculated components
    components_total: int = 7           # Total possible components
    missing_inputs: List[str] = field(default_factory=list)

    # Confidence indicator - NEW
    ies_confidence: float = 0.0         # 0-1, based on data availability

    def to_dict(self) -> dict:
        return {
            'ticker': self.ticker,
            'ies': self.ies,
            'ies_confidence': self.ies_confidence,
            'eqs': self.eqs,
            'regime': self.regime.value,
            'ecs_category': self.ecs_category.value,
            'earnings_date': self.earnings_date.isoformat() if self.earnings_date else None,
            'days_to_earnings': self.days_to_earnings,
            'is_post_earnings': self.is_post_earnings,
            'in_action_window': self.in_action_window,
            'eps_actual': self.eps_actual,
            'eps_estimate': self.eps_estimate,
            'total_reaction_pct': self.total_reaction_pct,
            'position_scale': self.position_scale,
            'score_adjustment': self.score_adjustment,
            'risk_flags': self.risk_flags,
            'data_quality': self.data_quality,
            'components_available': self.components_available,
            'components_total': self.components_total,
            'component_status': self.component_status,
        }

    @property
    def has_sufficient_data(self) -> bool:
        """Returns True if enough components are available for meaningful IES."""
        return self.components_available >= 3

    @property
    def ies_display(self) -> str:
        """Returns IES for display with confidence indicator."""
        if self.ies is None:
            return "N/A (insufficient data)"
        if self.ies_confidence < 0.5:
            return f"{self.ies:.0f}* (low confidence)"
        return f"{self.ies:.0f}"


class EarningsIntelligence:
    """
    Main class for computing IES/ECS and providing earnings intelligence.

    FIXED: All component calculations now properly handle missing data
    by returning None instead of silent 50.0 defaults.
    """

    # IES component weights (must sum to 100)
    IES_WEIGHTS = {
        'pre_earnings_runup': 20,
        'implied_move_percentile': 15,
        'analyst_revision_momentum': 15,
        'options_skew_score': 20,
        'news_sentiment_score': 10,
        'historical_beat_rate': 10,
        'sector_momentum': 10,
    }

    # ECS thresholds based on IES regime
    ECS_THRESHOLDS = {
        # IES range: (beat_threshold, strong_beat_threshold)
        'high': (1.5, 2.5),    # IES >= 75: need huge beat
        'normal': (1.0, 2.0),  # IES 35-75: standard beat
        'low': (0.5, 1.5),     # IES < 35: easy bar
    }

    def __init__(self):
        self._cache: Dict[str, Tuple[EarningsEnrichment, datetime]] = {}
        self._cache_ttl = timedelta(hours=1)

    def enrich(self, ticker: str, force_refresh: bool = False) -> EarningsEnrichment:
        """Get complete earnings intelligence for a ticker."""

        # Check cache
        if not force_refresh and ticker in self._cache:
            cached, timestamp = self._cache[ticker]
            if datetime.now() - timestamp < self._cache_ttl:
                return cached

        enrichment = EarningsEnrichment(ticker=ticker)

        try:
            # Get earnings date (could be future or recent past)
            earnings_result = self._get_next_earnings(ticker)
            enrichment.earnings_date = earnings_result[0]
            enrichment.days_to_earnings = earnings_result[1]

            # Handle case where we couldn't get earnings date
            if enrichment.days_to_earnings is None:
                enrichment.missing_inputs.append("earnings_date: Could not determine")
                enrichment.in_compute_window = False
                enrichment.data_quality = "UNKNOWN"
                self._cache[ticker] = (enrichment, datetime.now())
                return enrichment

            # Determine window type
            days_to = enrichment.days_to_earnings
            if days_to < 0 and days_to >= -7:
                # Post-earnings window (within last 7 days)
                enrichment.is_post_earnings = True
                enrichment.in_compute_window = True
                enrichment.in_action_window = False
            elif days_to >= 0 and days_to <= 10:
                # Pre-earnings window
                enrichment.is_post_earnings = False
                enrichment.in_compute_window = True
                enrichment.in_action_window = days_to <= 5
            else:
                # Outside earnings window
                enrichment.is_post_earnings = False
                enrichment.in_compute_window = False
                enrichment.in_action_window = False

            if not enrichment.in_compute_window:
                # Outside earnings window - IES not applicable
                enrichment.data_quality = "N/A"
                enrichment.component_status['all'] = "NOT_APPLICABLE: Outside earnings window"
                self._cache[ticker] = (enrichment, datetime.now())
                return enrichment

            if enrichment.is_post_earnings:
                # POST-EARNINGS: Calculate reaction and ECS
                self._calc_post_earnings_reaction(enrichment)
                enrichment.risk_flags = self._generate_post_earnings_flags(enrichment)
            else:
                # PRE-EARNINGS: Calculate IES components
                self._calculate_all_components(enrichment, ticker)

                # Compute composite IES only if we have enough data
                self._compute_ies(enrichment)

                # Determine regime
                enrichment.regime = self._classify_regime(enrichment)

                # Calculate position scaling and adjustments
                enrichment.position_scale = self._calc_position_scale(enrichment)
                enrichment.score_adjustment = self._calc_score_adjustment(enrichment)
                enrichment.risk_flags = self._generate_risk_flags(enrichment)

            # Assess data quality
            enrichment.data_quality = self._assess_data_quality(enrichment)

        except Exception as e:
            logger.error(f"Error enriching {ticker}: {e}")
            enrichment.data_quality = "ERROR"
            enrichment.missing_inputs.append(f"Error: {str(e)[:100]}")

        self._cache[ticker] = (enrichment, datetime.now())
        return enrichment

    def _calculate_all_components(self, enrichment: EarningsEnrichment, ticker: str):
        """Calculate all IES components with proper error tracking."""

        # Pre-earnings runup
        result = self._calc_pre_earnings_runup(ticker)
        enrichment.pre_earnings_runup = result.value
        enrichment.component_status['pre_earnings_runup'] = f"{result.status.value}: {result.message}" if result.message else result.status.value

        # Implied move percentile
        result = self._calc_implied_move_percentile(ticker)
        enrichment.implied_move_percentile = result.value
        enrichment.component_status['implied_move_percentile'] = f"{result.status.value}: {result.message}" if result.message else result.status.value

        # Analyst revisions
        result = self._calc_analyst_revisions(ticker)
        enrichment.analyst_revision_momentum = result.value
        enrichment.component_status['analyst_revision_momentum'] = f"{result.status.value}: {result.message}" if result.message else result.status.value

        # Options skew
        result = self._calc_options_skew(ticker)
        enrichment.options_skew_score = result.value
        enrichment.component_status['options_skew_score'] = f"{result.status.value}: {result.message}" if result.message else result.status.value

        # News sentiment
        result = self._calc_news_sentiment(ticker)
        enrichment.news_sentiment_score = result.value
        enrichment.component_status['news_sentiment_score'] = f"{result.status.value}: {result.message}" if result.message else result.status.value

        # Historical beat rate
        result = self._calc_historical_beat_rate(ticker)
        enrichment.historical_beat_rate = result.value
        enrichment.component_status['historical_beat_rate'] = f"{result.status.value}: {result.message}" if result.message else result.status.value

        # Sector momentum
        result = self._calc_sector_momentum(ticker)
        enrichment.sector_momentum = result.value
        enrichment.component_status['sector_momentum'] = f"{result.status.value}: {result.message}" if result.message else result.status.value

        # Count available components
        components = [
            enrichment.pre_earnings_runup,
            enrichment.implied_move_percentile,
            enrichment.analyst_revision_momentum,
            enrichment.options_skew_score,
            enrichment.news_sentiment_score,
            enrichment.historical_beat_rate,
            enrichment.sector_momentum,
        ]
        enrichment.components_available = sum(1 for c in components if c is not None)
        enrichment.components_total = len(components)

    def _generate_post_earnings_flags(self, e: EarningsEnrichment) -> List[str]:
        """Generate flags for post-earnings stocks."""
        flags = []

        if e.days_to_earnings is not None:
            days_since = abs(e.days_to_earnings)
            flags.append(f"📅 Earnings {days_since}d ago")

        # ECS result
        if e.ecs_category and e.ecs_category != ECSCategory.UNKNOWN:
            ecs_val = e.ecs_category.value
            if ecs_val == 'STRONG_BEAT':
                flags.append("🚀 STRONG BEAT - Crushed expectations")
            elif ecs_val == 'BEAT':
                flags.append("✅ BEAT - Cleared the bar")
            elif ecs_val == 'INLINE':
                flags.append("➡️ INLINE - Met expectations")
            elif ecs_val == 'MISS':
                flags.append("❌ MISS - Failed to clear")
            elif ecs_val == 'STRONG_MISS':
                flags.append("💥 STRONG MISS - Badly missed")

        # Price reaction - only if we have the data
        if e.total_reaction_pct is not None:
            if e.total_reaction_pct <= -10:
                flags.append(f"📉 CRUSHED: {e.total_reaction_pct:+.1f}% reaction")
            elif e.total_reaction_pct <= -5:
                flags.append(f"📉 Sold off: {e.total_reaction_pct:+.1f}%")
            elif e.total_reaction_pct >= 10:
                flags.append(f"📈 SOARED: {e.total_reaction_pct:+.1f}%")
            elif e.total_reaction_pct >= 5:
                flags.append(f"📈 Rallied: {e.total_reaction_pct:+.1f}%")
            else:
                flags.append(f"➡️ Flat: {e.total_reaction_pct:+.1f}%")

        # EPS result - only if we have both actual and estimate
        if e.eps_actual is not None and e.eps_estimate is not None and e.eps_estimate != 0:
            surprise_pct = ((e.eps_actual - e.eps_estimate) / abs(e.eps_estimate)) * 100
            if surprise_pct > 0:
                flags.append(f"💰 EPS: ${e.eps_actual:.2f} vs ${e.eps_estimate:.2f} (+{surprise_pct:.1f}%)")
            else:
                flags.append(f"💸 EPS: ${e.eps_actual:.2f} vs ${e.eps_estimate:.2f} ({surprise_pct:.1f}%)")

        # Beaten down opportunity? - only if we have the data
        if e.total_reaction_pct is not None and e.eqs is not None:
            if e.total_reaction_pct <= -8 and e.eqs >= 50:
                flags.append("💡 OVERSOLD? Good report but sold off - potential opportunity")
            if e.total_reaction_pct <= -5 and e.eqs >= 70:
                flags.append("⚠️ SELL THE NEWS - Great report but stock down")

        return flags

    def _get_next_earnings(self, ticker: str) -> Tuple[Optional[date], Optional[int]]:
        """Get next earnings date and days until. Returns (None, None) if unknown."""
        try:
            # Try database first
            from src.db.connection import get_engine
            engine = get_engine()

            # Check for upcoming earnings
            df = pd.read_sql(f"""
                SELECT earnings_date FROM earnings_calendar 
                WHERE ticker = '{ticker}' AND earnings_date >= CURRENT_DATE
                ORDER BY earnings_date LIMIT 1
            """, engine)

            if not df.empty and df.iloc[0]['earnings_date']:
                ed = pd.to_datetime(df.iloc[0]['earnings_date']).date()
                days = (ed - date.today()).days
                return ed, days

            # Check for recent past earnings (within last 7 days)
            df_past = pd.read_sql(f"""
                SELECT earnings_date FROM earnings_calendar 
                WHERE ticker = '{ticker}' 
                AND earnings_date >= CURRENT_DATE - INTERVAL '7 days'
                AND earnings_date < CURRENT_DATE
                ORDER BY earnings_date DESC LIMIT 1
            """, engine)

            if not df_past.empty and df_past.iloc[0]['earnings_date']:
                ed = pd.to_datetime(df_past.iloc[0]['earnings_date']).date()
                days = (ed - date.today()).days  # Will be negative
                return ed, days

        except Exception as e:
            logger.debug(f"Database earnings date error for {ticker}: {e}")

        # Fallback to yfinance
        try:
            stock = yf.Ticker(ticker)
            cal = stock.calendar
            if cal is not None and 'Earnings Date' in cal:
                ed = pd.to_datetime(cal['Earnings Date']).date()
                days = (ed - date.today()).days
                return ed, days
        except Exception as e:
            logger.debug(f"yfinance earnings date error for {ticker}: {e}")

        # Could not determine earnings date
        return None, None

    def _calc_post_earnings_reaction(self, e: EarningsEnrichment):
        """Calculate post-earnings price reaction and ECS."""
        try:
            stock = yf.Ticker(e.ticker)

            # Get recent price history
            hist = stock.history(period="10d")
            if len(hist) < 2:
                e.missing_inputs.append("price_history: Insufficient data")
                return

            # Find earnings day in price data
            earnings_idx = None
            if e.earnings_date:
                for i, dt in enumerate(hist.index):
                    if dt.date() == e.earnings_date or dt.date() == e.earnings_date + timedelta(days=1):
                        earnings_idx = i
                        break

            if earnings_idx is None or earnings_idx == 0:
                # Use most recent gap as proxy
                earnings_idx = len(hist) - 1

            # Calculate gap (open vs previous close)
            if earnings_idx > 0:
                prev_close = hist['Close'].iloc[earnings_idx - 1]
                er_open = hist['Open'].iloc[earnings_idx]
                er_close = hist['Close'].iloc[earnings_idx]

                e.gap_pct = ((er_open - prev_close) / prev_close) * 100
                e.intraday_move_pct = ((er_close - er_open) / er_open) * 100
                e.total_reaction_pct = ((er_close - prev_close) / prev_close) * 100

            # Get earnings surprise data
            try:
                earnings_hist = stock.earnings_history
                if earnings_hist is not None and not earnings_hist.empty:
                    latest = earnings_hist.iloc[0]

                    # EPS surprise
                    eps_est = latest.get('epsEstimate')
                    eps_act = latest.get('epsActual')

                    if eps_est is not None and eps_act is not None and eps_est != 0:
                        e.eps_actual = float(eps_act)
                        e.eps_estimate = float(eps_est)
                        e.eps_surprise_z = ((eps_act - eps_est) / abs(eps_est)) * 10
                        e.eps_surprise_pct = ((eps_act - eps_est) / abs(eps_est)) * 100

                    # Revenue if available
                    rev_est = latest.get('revenueEstimate')
                    rev_act = latest.get('revenueActual')
                    if rev_est is not None and rev_act is not None and rev_est != 0:
                        e.revenue_actual = float(rev_act)
                        e.revenue_estimate = float(rev_est)
                        e.revenue_surprise_z = ((rev_act - rev_est) / abs(rev_est)) * 10
                        e.revenue_surprise_pct = ((rev_act - rev_est) / abs(rev_est)) * 100
            except Exception as ex:
                logger.debug(f"Earnings history error for {e.ticker}: {ex}")
                e.missing_inputs.append(f"earnings_history: {str(ex)[:50]}")

            # Compute blended event_z only if we have data
            if e.eps_surprise_z is not None:
                if e.revenue_surprise_z is not None:
                    e.event_z = e.eps_surprise_z * 0.6 + e.revenue_surprise_z * 0.4
                else:
                    e.event_z = e.eps_surprise_z

            # Determine ECS category
            self._determine_ecs(e)

            # Calculate EQS (Earnings Quality Score)
            self._calc_eqs(e)

        except Exception as ex:
            logger.debug(f"Post-earnings reaction error for {e.ticker}: {ex}")
            e.missing_inputs.append(f"post_earnings_calc: {str(ex)[:50]}")

    def _determine_ecs(self, e: EarningsEnrichment):
        """Determine ECS category based on event_z and IES."""

        if e.event_z is None:
            e.ecs_category = ECSCategory.UNKNOWN
            return

        # Get thresholds based on pre-earnings IES
        # Note: For post-earnings, we might not have IES computed
        if e.ies is not None:
            if e.ies >= 75:
                thresholds = self.ECS_THRESHOLDS['high']
            elif e.ies <= 35:
                thresholds = self.ECS_THRESHOLDS['low']
            else:
                thresholds = self.ECS_THRESHOLDS['normal']
        else:
            # Default to normal thresholds if IES not available
            thresholds = self.ECS_THRESHOLDS['normal']

        beat_threshold, strong_beat_threshold = thresholds

        if e.event_z >= strong_beat_threshold:
            e.ecs_category = ECSCategory.STRONG_BEAT
        elif e.event_z >= beat_threshold:
            e.ecs_category = ECSCategory.BEAT
        elif e.event_z >= -beat_threshold:
            e.ecs_category = ECSCategory.INLINE
        elif e.event_z >= -strong_beat_threshold:
            e.ecs_category = ECSCategory.MISS
        else:
            e.ecs_category = ECSCategory.STRONG_MISS

    def _calc_eqs(self, e: EarningsEnrichment):
        """Calculate Earnings Quality Score (0-100) for post-earnings."""
        score = 50  # Start neutral

        # EPS surprise component (-15 to +15)
        if e.eps_surprise_z is not None:
            if e.eps_surprise_z > 2:
                score += 15
            elif e.eps_surprise_z > 1:
                score += 10
            elif e.eps_surprise_z > 0:
                score += 5
            elif e.eps_surprise_z > -1:
                score += 0
            elif e.eps_surprise_z > -2:
                score -= 10
            else:
                score -= 15

        # Revenue surprise component (-15 to +15)
        if e.revenue_surprise_z is not None:
            if e.revenue_surprise_z > 2:
                score += 15
            elif e.revenue_surprise_z > 1:
                score += 10
            elif e.revenue_surprise_z > 0:
                score += 5
            elif e.revenue_surprise_z > -1:
                score += 0
            else:
                score -= 15

        # Market reaction component (-30 to +30)
        if e.total_reaction_pct is not None:
            if e.total_reaction_pct > 5:
                score += 30
            elif e.total_reaction_pct > 0:
                score += 15
            elif e.total_reaction_pct > -5:
                score += 0
            elif e.total_reaction_pct > -10:
                score -= 15
            else:
                score -= 30

        e.eqs = max(0, min(100, score))

    # =========================================================================
    # COMPONENT CALCULATIONS - FIXED TO RETURN ComponentResult
    # =========================================================================

    def _calc_pre_earnings_runup(self, ticker: str) -> ComponentResult:
        """Calculate pre-earnings price runup score (0-100)."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")

            if hist is None or len(hist) < 5:
                return ComponentResult(
                    value=None,
                    status=ComponentStatus.INSUFFICIENT,
                    message="Less than 5 days of price history"
                )

            # Calculate 20-day return
            pct_change = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100

            # Convert to 0-100 score
            # -10% = 0, 0% = 50, +10% = 100
            score = 50 + (pct_change * 5)
            score = max(0, min(100, score))

            return ComponentResult(
                value=score,
                status=ComponentStatus.CALCULATED,
                message=f"20d return: {pct_change:+.1f}%"
            )

        except Exception as e:
            return ComponentResult(
                value=None,
                status=ComponentStatus.ERROR,
                message=str(e)[:100]
            )

    def _calc_implied_move_percentile(self, ticker: str) -> ComponentResult:
        """Calculate implied move percentile vs history (0-100)."""
        try:
            stock = yf.Ticker(ticker)

            # Get ATM options
            if not stock.options:
                return ComponentResult(
                    value=None,
                    status=ComponentStatus.NO_DATA,
                    message="No options available for this ticker"
                )

            exp = stock.options[0]
            chain = stock.option_chain(exp)

            current_price = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice')
            if not current_price:
                return ComponentResult(
                    value=None,
                    status=ComponentStatus.NO_DATA,
                    message="Could not get current price"
                )

            # Find ATM straddle
            calls = chain.calls
            puts = chain.puts

            if calls.empty or puts.empty:
                return ComponentResult(
                    value=None,
                    status=ComponentStatus.NO_DATA,
                    message="Options chain is empty"
                )

            atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
            atm_put = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]

            call_price = atm_call['lastPrice'].values[0]
            put_price = atm_put['lastPrice'].values[0]

            if call_price == 0 and put_price == 0:
                return ComponentResult(
                    value=None,
                    status=ComponentStatus.NO_DATA,
                    message="ATM options have no price data"
                )

            straddle_price = call_price + put_price
            implied_move_pct = (straddle_price / current_price) * 100

            # Compare to typical earnings move (assume 5% is average)
            # 2.5% = 0, 5% = 50, 10% = 100
            score = (implied_move_pct / 10) * 100
            score = max(0, min(100, score))

            return ComponentResult(
                value=score,
                status=ComponentStatus.CALCULATED,
                message=f"Implied move: {implied_move_pct:.1f}%"
            )

        except Exception as e:
            return ComponentResult(
                value=None,
                status=ComponentStatus.ERROR,
                message=str(e)[:100]
            )

    def _calc_analyst_revisions(self, ticker: str) -> ComponentResult:
        """Calculate analyst revision momentum (0-100)."""
        try:
            from src.db.connection import get_engine
            engine = get_engine()

            # Get recent price target changes
            df = pd.read_sql(f"""
                SELECT target_mean, date FROM price_targets
                WHERE ticker = '{ticker}'
                ORDER BY date DESC LIMIT 5
            """, engine)

            if df.empty:
                return ComponentResult(
                    value=None,
                    status=ComponentStatus.NO_DATA,
                    message="No price target data in database"
                )

            if len(df) < 2:
                return ComponentResult(
                    value=None,
                    status=ComponentStatus.INSUFFICIENT,
                    message="Need at least 2 price target records"
                )

            # Calculate trend
            recent = df.iloc[0]['target_mean']
            older = df.iloc[-1]['target_mean']

            if older is None or older == 0:
                return ComponentResult(
                    value=None,
                    status=ComponentStatus.NO_DATA,
                    message="Invalid price target values"
                )

            pct_change = ((recent - older) / older) * 100

            # -10% = 0, 0% = 50, +10% = 100
            score = 50 + (pct_change * 5)
            score = max(0, min(100, score))

            return ComponentResult(
                value=score,
                status=ComponentStatus.CALCULATED,
                message=f"Target revision: {pct_change:+.1f}%"
            )

        except Exception as e:
            return ComponentResult(
                value=None,
                status=ComponentStatus.ERROR,
                message=str(e)[:100]
            )

    def _calc_options_skew(self, ticker: str) -> ComponentResult:
        """Calculate options skew score (0-100). High = bullish skew."""
        try:
            stock = yf.Ticker(ticker)

            if not stock.options:
                return ComponentResult(
                    value=None,
                    status=ComponentStatus.NO_DATA,
                    message="No options available"
                )

            exp = stock.options[0]
            chain = stock.option_chain(exp)

            # Put/call volume ratio
            call_vol = chain.calls['volume'].sum()
            put_vol = chain.puts['volume'].sum()

            if pd.isna(call_vol) or call_vol == 0:
                return ComponentResult(
                    value=None,
                    status=ComponentStatus.NO_DATA,
                    message="No call volume data"
                )

            if pd.isna(put_vol):
                put_vol = 0

            pc_ratio = put_vol / call_vol

            # PC < 0.5 = bullish (high score), PC > 1.5 = bearish (low score)
            # Inverted because high IES = high expectations
            score = 100 - (pc_ratio * 40)
            score = max(0, min(100, score))

            return ComponentResult(
                value=score,
                status=ComponentStatus.CALCULATED,
                message=f"P/C ratio: {pc_ratio:.2f}"
            )

        except Exception as e:
            return ComponentResult(
                value=None,
                status=ComponentStatus.ERROR,
                message=str(e)[:100]
            )

    def _calc_news_sentiment(self, ticker: str) -> ComponentResult:
        """Get recent news sentiment score (0-100)."""
        try:
            from src.db.connection import get_engine
            engine = get_engine()

            df = pd.read_sql(f"""
                SELECT AVG(ai_sentiment_fast) as avg_sent, COUNT(*) as article_count
                FROM news_articles
                WHERE ticker = '{ticker}' 
                AND fetched_at > NOW() - INTERVAL '7 days'
                AND ai_sentiment_fast IS NOT NULL
            """, engine)

            if df.empty:
                return ComponentResult(
                    value=None,
                    status=ComponentStatus.NO_DATA,
                    message="No news articles in database"
                )

            avg_sent = df.iloc[0]['avg_sent']
            article_count = df.iloc[0]['article_count']

            if pd.isna(avg_sent) or article_count == 0:
                return ComponentResult(
                    value=None,
                    status=ComponentStatus.NO_DATA,
                    message="No sentiment scores in recent articles"
                )

            return ComponentResult(
                value=float(avg_sent),
                status=ComponentStatus.CALCULATED,
                message=f"Based on {article_count} articles"
            )

        except Exception as e:
            return ComponentResult(
                value=None,
                status=ComponentStatus.ERROR,
                message=str(e)[:100]
            )

    def _calc_historical_beat_rate(self, ticker: str) -> ComponentResult:
        """Calculate historical earnings beat rate (0-100)."""
        try:
            stock = yf.Ticker(ticker)

            # Get earnings history
            earnings = stock.earnings_history
            if earnings is None or earnings.empty:
                return ComponentResult(
                    value=None,
                    status=ComponentStatus.NO_DATA,
                    message="No earnings history available"
                )

            # Check for required columns
            if 'epsActual' not in earnings.columns or 'epsEstimate' not in earnings.columns:
                return ComponentResult(
                    value=None,
                    status=ComponentStatus.NO_DATA,
                    message="Earnings history missing EPS data"
                )

            # Filter out rows with missing data
            valid_earnings = earnings.dropna(subset=['epsActual', 'epsEstimate'])

            if len(valid_earnings) == 0:
                return ComponentResult(
                    value=None,
                    status=ComponentStatus.NO_DATA,
                    message="No valid EPS comparisons"
                )

            # Calculate beat rate
            beats = (valid_earnings['epsActual'] > valid_earnings['epsEstimate']).sum()
            total = len(valid_earnings)

            beat_rate = (beats / total) * 100

            return ComponentResult(
                value=beat_rate,
                status=ComponentStatus.CALCULATED,
                message=f"{beats}/{total} beats ({beat_rate:.0f}%)"
            )

        except Exception as e:
            return ComponentResult(
                value=None,
                status=ComponentStatus.ERROR,
                message=str(e)[:100]
            )

    def _calc_sector_momentum(self, ticker: str) -> ComponentResult:
        """Calculate sector momentum score (0-100)."""
        try:
            # TODO: Use actual sector ETF instead of SPY
            # For now, using SPY as market proxy
            spy = yf.Ticker("SPY")
            hist = spy.history(period="1mo")

            if hist is None or len(hist) < 5:
                return ComponentResult(
                    value=None,
                    status=ComponentStatus.INSUFFICIENT,
                    message="Insufficient SPY price history"
                )

            pct_change = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100

            # -5% = 0, 0% = 50, +5% = 100
            score = 50 + (pct_change * 10)
            score = max(0, min(100, score))

            return ComponentResult(
                value=score,
                status=ComponentStatus.CALCULATED,
                message=f"SPY 1m return: {pct_change:+.1f}%"
            )

        except Exception as e:
            return ComponentResult(
                value=None,
                status=ComponentStatus.ERROR,
                message=str(e)[:100]
            )

    def _compute_ies(self, e: EarningsEnrichment):
        """
        Compute composite IES from available components.

        FIXED: Only uses components that are actually available.
        Sets ies to None if insufficient data.
        """
        components = {
            'pre_earnings_runup': e.pre_earnings_runup,
            'implied_move_percentile': e.implied_move_percentile,
            'analyst_revision_momentum': e.analyst_revision_momentum,
            'options_skew_score': e.options_skew_score,
            'news_sentiment_score': e.news_sentiment_score,
            'historical_beat_rate': e.historical_beat_rate,
            'sector_momentum': e.sector_momentum,
        }

        # Filter to available components only
        available = {k: v for k, v in components.items() if v is not None}

        if len(available) < 3:
            # Insufficient data - don't compute IES
            e.ies = None
            e.ies_confidence = 0.0
            e.missing_inputs.append(f"IES: Only {len(available)}/7 components available (need 3+)")
            return

        # Compute weighted average using only available components
        weighted_sum = 0.0
        total_weight = 0.0

        for component_name, value in available.items():
            weight = self.IES_WEIGHTS[component_name]
            weighted_sum += value * weight
            total_weight += weight

        if total_weight > 0:
            # Normalize by actual weight used
            e.ies = weighted_sum / total_weight
            e.ies = max(0, min(100, e.ies))

            # Confidence is proportion of weights we actually have
            e.ies_confidence = total_weight / 100.0
        else:
            e.ies = None
            e.ies_confidence = 0.0

    def _classify_regime(self, e: EarningsEnrichment) -> IESRegime:
        """Classify the expectation regime."""

        # Can't classify without IES
        if e.ies is None:
            return IESRegime.UNKNOWN

        if e.ies >= 75:
            return IESRegime.HYPED
        elif e.ies <= 25:
            return IESRegime.FEARED
        elif e.implied_move_percentile is not None and e.implied_move_percentile >= 80:
            return IESRegime.VOLATILE
        else:
            return IESRegime.NORMAL

    def _calc_position_scale(self, e: EarningsEnrichment) -> float:
        """Calculate position scaling factor (0.2 to 1.0)."""

        if not e.in_action_window:
            return 1.0

        # If we don't have IES, be conservative
        if e.ies is None:
            return 0.5  # Conservative default when data is missing

        # Scale down based on IES extremity
        ies_factor = 1 - (abs(e.ies - 50) / 100)  # Extreme = lower

        # Scale down based on implied move (if available)
        if e.implied_move_percentile is not None:
            move_factor = 1 - (e.implied_move_percentile / 200)
        else:
            move_factor = 0.8  # Conservative when unknown

        scale = ies_factor * move_factor
        return max(0.2, min(1.0, scale))

    def _calc_score_adjustment(self, e: EarningsEnrichment) -> int:
        """Calculate score adjustment (-20 to +20)."""

        if not e.in_action_window:
            return 0

        # If regime is unknown due to missing data, apply small penalty
        if e.regime == IESRegime.UNKNOWN:
            return -5  # Small penalty for uncertainty

        # Penalize extreme expectations
        if e.regime == IESRegime.HYPED:
            return -15  # High expectations = harder to beat
        elif e.regime == IESRegime.FEARED:
            return +10  # Low expectations = easier to beat
        elif e.regime == IESRegime.VOLATILE:
            return -10  # Uncertain = risky

        return 0

    def _generate_risk_flags(self, e: EarningsEnrichment) -> List[str]:
        """Generate risk flags based on earnings intelligence."""

        flags = []

        if e.in_action_window and e.days_to_earnings is not None:
            flags.append(f"⚠️ Earnings in {e.days_to_earnings}d")

        # Data quality warning
        if e.ies is None:
            flags.append("⚠️ INSUFFICIENT DATA: IES could not be calculated")
        elif e.ies_confidence < 0.5:
            flags.append(f"⚠️ LOW CONFIDENCE: IES based on {e.components_available}/{e.components_total} components")

        if e.regime == IESRegime.UNKNOWN:
            flags.append("❓ UNKNOWN REGIME: Insufficient data to classify")
        elif e.regime == IESRegime.HYPED:
            flags.append("🔥 HYPED: High expectations priced in")
        elif e.regime == IESRegime.FEARED:
            flags.append("😰 FEARED: Low expectations - potential upside")
        elif e.regime == IESRegime.VOLATILE:
            flags.append("⚡ VOLATILE: High implied move")

        if e.pre_earnings_runup is not None and e.pre_earnings_runup >= 80:
            flags.append("📈 Run-up: Stock up significantly pre-earnings")

        if e.implied_move_percentile is not None and e.implied_move_percentile >= 80:
            flags.append(f"📊 Implied move at {e.implied_move_percentile:.0f}th percentile")

        if e.position_scale < 0.5:
            flags.append(f"⚖️ Reduce position size to {e.position_scale:.0%}")

        return flags

    def _assess_data_quality(self, e: EarningsEnrichment) -> str:
        """Assess data quality based on available inputs."""

        if e.components_available == 0:
            return "INSUFFICIENT"
        elif e.components_available <= 2:
            return "LOW"
        elif e.components_available <= 4:
            return "MEDIUM"
        else:
            return "HIGH"


# Singleton instance
_earnings_intelligence = None


def get_earnings_intelligence() -> EarningsIntelligence:
    """Get singleton instance."""
    global _earnings_intelligence
    if _earnings_intelligence is None:
        _earnings_intelligence = EarningsIntelligence()
    return _earnings_intelligence


def enrich_screener_with_earnings(ticker: str) -> EarningsEnrichment:
    """Convenience function to enrich a ticker with earnings intelligence."""
    return get_earnings_intelligence().enrich(ticker)


# ============================================================================
# Test
# ============================================================================
if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    print(f"\n{'='*60}")
    print(f"EARNINGS INTELLIGENCE TEST: {ticker}")
    print(f"{'='*60}\n")

    result = enrich_screener_with_earnings(ticker)

    print(f"Earnings Date: {result.earnings_date}")
    print(f"Days to Earnings: {result.days_to_earnings}")
    print(f"In Action Window: {result.in_action_window}")
    print(f"Is Post-Earnings: {result.is_post_earnings}")
    print()
    print(f"IES: {result.ies_display}")
    print(f"IES Confidence: {result.ies_confidence:.1%}")
    print(f"Regime: {result.regime.value}")
    print(f"Data Quality: {result.data_quality}")
    print(f"Components: {result.components_available}/{result.components_total}")
    print()
    print("Component Status:")
    for comp, status in result.component_status.items():
        value = getattr(result, comp, None)
        value_str = f"{value:.1f}" if value is not None else "N/A"
        print(f"  {comp}: {value_str} ({status})")
    print()
    print("Risk Flags:")
    for flag in result.risk_flags:
        print(f"  {flag}")
    print()
    if result.missing_inputs:
        print("Missing Inputs:")
        for mi in result.missing_inputs:
            print(f"  - {mi}")