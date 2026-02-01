"""
Earnings Intelligence Module

Solves the "great report, bad trade" problem by implementing:
- IES (Implied Expectations Score): Estimates what's priced in before earnings
- ECS (Expectations Clearance Score): Whether results cleared the bar
- Regime Classification: HYPED, FEARED, VOLATILE, NORMAL
- Position Scaling: Risk-adjusted position sizing

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


class ECSCategory(Enum):
    """Expectations Clearance Score categories."""
    STRONG_BEAT = "STRONG_BEAT"    # Crushed elevated expectations
    BEAT = "BEAT"                   # Cleared the bar
    INLINE = "INLINE"               # Met expectations
    MISS = "MISS"                   # Failed to clear
    STRONG_MISS = "STRONG_MISS"    # Badly missed low expectations
    PENDING = "PENDING"             # Pre-earnings


@dataclass
class EarningsEnrichment:
    """Complete earnings intelligence for a ticker."""
    ticker: str

    # Core scores
    ies: float = 50.0              # 0-100, Implied Expectations Score
    eqs: float = 50.0              # 0-100, Earnings Quality Score (post-earnings)
    regime: IESRegime = IESRegime.NORMAL
    ecs_category: ECSCategory = ECSCategory.PENDING

    # Timing
    earnings_date: Optional[date] = None
    days_to_earnings: int = 999
    in_compute_window: bool = False   # Within 10 days (for display)
    in_action_window: bool = False    # Within 5 days (for trading decisions)
    is_post_earnings: bool = False    # Earnings already happened (within 7 days)

    # IES Components (0-100 each)
    pre_earnings_runup: float = 50.0
    implied_move_percentile: float = 50.0
    analyst_revision_momentum: float = 50.0
    options_skew_score: float = 50.0
    news_sentiment_score: float = 50.0
    historical_beat_rate: float = 50.0
    sector_momentum: float = 50.0

    # Post-earnings actual results
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    eps_surprise_pct: float = 0.0
    revenue_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_surprise_pct: float = 0.0

    # Z-scores for ECS calculation
    eps_surprise_z: float = 0.0
    revenue_surprise_z: float = 0.0
    guidance_z: float = 0.0
    event_z: float = 0.0           # Blended surprise z-score

    # Market reaction (post-earnings)
    gap_pct: float = 0.0           # Open vs previous close
    intraday_move_pct: float = 0.0 # Close vs open on ER day
    total_reaction_pct: float = 0.0 # Close vs previous close

    # Trading adjustments
    position_scale: float = 1.0    # 0.2 to 1.0
    score_adjustment: int = 0      # -20 to +20
    risk_flags: List[str] = field(default_factory=list)

    # Data quality
    data_quality: str = "LOW"      # HIGH, MEDIUM, LOW
    missing_inputs: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'ticker': self.ticker,
            'ies': self.ies,
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
        }


class EarningsIntelligence:
    """
    Main class for computing IES/ECS and providing earnings intelligence.
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
        missing_inputs = []

        try:
            # Get earnings date (could be future or recent past)
            earnings_date, days_to = self._get_next_earnings(ticker)
            enrichment.earnings_date = earnings_date
            enrichment.days_to_earnings = days_to

            # Determine window type
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
                # Outside earnings window - return defaults
                enrichment.data_quality = "HIGH"
                self._cache[ticker] = (enrichment, datetime.now())
                return enrichment

            if enrichment.is_post_earnings:
                # POST-EARNINGS: Calculate reaction and ECS
                self._calc_post_earnings_reaction(enrichment)

                # Generate post-earnings flags
                enrichment.risk_flags = self._generate_post_earnings_flags(enrichment)

            else:
                # PRE-EARNINGS: Calculate IES components
                enrichment.pre_earnings_runup = self._calc_pre_earnings_runup(ticker)
                enrichment.implied_move_percentile = self._calc_implied_move_percentile(ticker)
                enrichment.analyst_revision_momentum = self._calc_analyst_revisions(ticker)
                enrichment.options_skew_score = self._calc_options_skew(ticker)
                enrichment.news_sentiment_score = self._calc_news_sentiment(ticker)
                enrichment.historical_beat_rate = self._calc_historical_beat_rate(ticker)
                enrichment.sector_momentum = self._calc_sector_momentum(ticker)

                # Compute composite IES
                enrichment.ies = self._compute_ies(enrichment)

                # Determine regime
                enrichment.regime = self._classify_regime(enrichment)

                # Calculate position scaling and adjustments
                enrichment.position_scale = self._calc_position_scale(enrichment)
                enrichment.score_adjustment = self._calc_score_adjustment(enrichment)
                enrichment.risk_flags = self._generate_risk_flags(enrichment)

            # Assess data quality
            enrichment.missing_inputs = missing_inputs
            enrichment.data_quality = self._assess_data_quality(enrichment)

        except Exception as e:
            logger.error(f"Error enriching {ticker}: {e}")
            enrichment.data_quality = "LOW"
            enrichment.missing_inputs.append(f"Error: {str(e)[:50]}")

        self._cache[ticker] = (enrichment, datetime.now())
        return enrichment

    def _generate_post_earnings_flags(self, e: EarningsEnrichment) -> List[str]:
        """Generate flags for post-earnings stocks."""
        flags = []

        days_since = abs(e.days_to_earnings)
        flags.append(f"📅 Earnings {days_since}d ago")

        # ECS result
        if e.ecs_category:
            ecs_val = e.ecs_category.value if hasattr(e.ecs_category, 'value') else str(e.ecs_category)
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

        # Price reaction
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

        # EPS result
        if e.eps_actual and e.eps_estimate:
            surprise_pct = ((e.eps_actual - e.eps_estimate) / abs(e.eps_estimate)) * 100
            if surprise_pct > 0:
                flags.append(f"💰 EPS: ${e.eps_actual:.2f} vs ${e.eps_estimate:.2f} (+{surprise_pct:.1f}%)")
            else:
                flags.append(f"💸 EPS: ${e.eps_actual:.2f} vs ${e.eps_estimate:.2f} ({surprise_pct:.1f}%)")

        # Beaten down opportunity?
        if e.total_reaction_pct <= -8 and e.eqs >= 50:
            flags.append("💡 OVERSOLD? Good report but sold off - potential opportunity")

        # Sell the news?
        if e.total_reaction_pct <= -5 and e.eqs >= 70:
            flags.append("⚠️ SELL THE NEWS - Great report but stock down")

        return flags

    def _get_next_earnings(self, ticker: str) -> Tuple[Optional[date], int]:
        """Get next earnings date and days until. Also checks recent past earnings."""
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

            # Fallback to yfinance
            stock = yf.Ticker(ticker)
            cal = stock.calendar
            if cal is not None and 'Earnings Date' in cal:
                ed = pd.to_datetime(cal['Earnings Date']).date()
                days = (ed - date.today()).days
                return ed, days

        except Exception as e:
            logger.debug(f"Earnings date error for {ticker}: {e}")

        return None, 999

    def _calc_post_earnings_reaction(self, e: EarningsEnrichment):
        """Calculate post-earnings price reaction and ECS."""
        try:
            stock = yf.Ticker(e.ticker)

            # Get recent price history
            hist = stock.history(period="10d")
            if len(hist) < 2:
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
            earnings_hist = stock.earnings_history
            if earnings_hist is not None and not earnings_hist.empty:
                latest = earnings_hist.iloc[0]

                # EPS surprise
                if 'epsEstimate' in latest and latest['epsEstimate'] and latest['epsEstimate'] != 0:
                    eps_actual = latest.get('epsActual', 0) or 0
                    eps_est = latest['epsEstimate']
                    e.eps_surprise_z = ((eps_actual - eps_est) / abs(eps_est)) * 10

                # Store actual vs estimate
                e.eps_actual = latest.get('epsActual')
                e.eps_estimate = latest.get('epsEstimate')

                # Revenue if available
                if 'revenueActual' in latest and 'revenueEstimate' in latest:
                    rev_actual = latest.get('revenueActual', 0) or 0
                    rev_est = latest.get('revenueEstimate', 1)
                    if rev_est:
                        e.revenue_surprise_z = ((rev_actual - rev_est) / abs(rev_est)) * 10

            # Compute blended event_z
            e.event_z = e.eps_surprise_z * 0.6 + e.revenue_surprise_z * 0.4

            # Determine ECS category
            self._determine_ecs(e)

            # Calculate EQS (Earnings Quality Score)
            self._calc_eqs(e)

        except Exception as ex:
            logger.debug(f"Post-earnings reaction error for {e.ticker}: {ex}")

    def _determine_ecs(self, e: EarningsEnrichment):
        """Determine ECS category based on event_z and IES."""

        # Get thresholds based on pre-earnings IES
        if e.ies >= 75:
            thresholds = self.ECS_THRESHOLDS['high']
        elif e.ies <= 35:
            thresholds = self.ECS_THRESHOLDS['low']
        else:
            thresholds = self.ECS_THRESHOLDS['normal']

        beat_thresh, strong_beat_thresh = thresholds

        if e.event_z >= strong_beat_thresh:
            e.ecs_category = ECSCategory.STRONG_BEAT
        elif e.event_z >= beat_thresh:
            e.ecs_category = ECSCategory.BEAT
        elif e.event_z >= -beat_thresh:
            e.ecs_category = ECSCategory.INLINE
        elif e.event_z >= -strong_beat_thresh:
            e.ecs_category = ECSCategory.MISS
        else:
            e.ecs_category = ECSCategory.STRONG_MISS

    def _calc_eqs(self, e: EarningsEnrichment):
        """Calculate Earnings Quality Score (0-100)."""
        # EQS measures absolute quality, not relative to expectations

        score = 50  # Base

        # EPS surprise component (0-40 points)
        if e.eps_surprise_z > 2:
            score += 40
        elif e.eps_surprise_z > 1:
            score += 30
        elif e.eps_surprise_z > 0:
            score += 20
        elif e.eps_surprise_z > -1:
            score += 0
        else:
            score -= 20

        # Revenue surprise component (0-30 points)
        if e.revenue_surprise_z > 1:
            score += 30
        elif e.revenue_surprise_z > 0:
            score += 15
        elif e.revenue_surprise_z > -1:
            score += 0
        else:
            score -= 15

        # Market reaction component (-30 to +30)
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

    def _calc_pre_earnings_runup(self, ticker: str) -> float:
        """Calculate pre-earnings price runup score (0-100)."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")

            if len(hist) < 5:
                return 50.0

            # Calculate 20-day return
            pct_change = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100

            # Convert to 0-100 score
            # -10% = 0, 0% = 50, +10% = 100
            score = 50 + (pct_change * 5)
            return max(0, min(100, score))

        except:
            return 50.0

    def _calc_implied_move_percentile(self, ticker: str) -> float:
        """Calculate implied move percentile vs history (0-100)."""
        try:
            stock = yf.Ticker(ticker)

            # Get ATM options
            if not stock.options:
                return 50.0

            exp = stock.options[0]
            chain = stock.option_chain(exp)

            current_price = stock.info.get('currentPrice', 0)
            if not current_price:
                return 50.0

            # Find ATM straddle
            calls = chain.calls
            puts = chain.puts

            atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
            atm_put = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]

            straddle_price = atm_call['lastPrice'].values[0] + atm_put['lastPrice'].values[0]
            implied_move_pct = (straddle_price / current_price) * 100

            # Compare to typical earnings move (assume 5% is average)
            # 2.5% = 0, 5% = 50, 10% = 100
            score = (implied_move_pct / 10) * 100
            return max(0, min(100, score))

        except:
            return 50.0

    def _calc_analyst_revisions(self, ticker: str) -> float:
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

            if len(df) < 2:
                return 50.0

            # Calculate trend
            recent = df.iloc[0]['target_mean']
            older = df.iloc[-1]['target_mean']

            if older == 0:
                return 50.0

            pct_change = ((recent - older) / older) * 100

            # -10% = 0, 0% = 50, +10% = 100
            score = 50 + (pct_change * 5)
            return max(0, min(100, score))

        except:
            return 50.0

    def _calc_options_skew(self, ticker: str) -> float:
        """Calculate options skew score (0-100). High = bullish skew."""
        try:
            stock = yf.Ticker(ticker)

            if not stock.options:
                return 50.0

            exp = stock.options[0]
            chain = stock.option_chain(exp)

            # Put/call volume ratio
            call_vol = chain.calls['volume'].sum()
            put_vol = chain.puts['volume'].sum()

            if call_vol == 0:
                return 50.0

            pc_ratio = put_vol / call_vol

            # PC < 0.5 = bullish (high score), PC > 1.5 = bearish (low score)
            # Inverted because high IES = high expectations
            score = 100 - (pc_ratio * 40)
            return max(0, min(100, score))

        except:
            return 50.0

    def _calc_news_sentiment(self, ticker: str) -> float:
        """Get recent news sentiment score (0-100)."""
        try:
            from src.db.connection import get_engine
            engine = get_engine()

            df = pd.read_sql(f"""
                SELECT AVG(ai_sentiment_fast) as avg_sent
                FROM news_articles
                WHERE ticker = '{ticker}' 
                AND fetched_at > NOW() - INTERVAL '7 days'
                AND ai_sentiment_fast IS NOT NULL
            """, engine)

            if df.empty or pd.isna(df.iloc[0]['avg_sent']):
                return 50.0

            return float(df.iloc[0]['avg_sent'])

        except:
            return 50.0

    def _calc_historical_beat_rate(self, ticker: str) -> float:
        """Calculate historical earnings beat rate (0-100)."""
        try:
            stock = yf.Ticker(ticker)

            # Get earnings history
            earnings = stock.earnings_history
            if earnings is None or earnings.empty:
                return 50.0

            # Calculate beat rate
            beats = (earnings['epsActual'] > earnings['epsEstimate']).sum()
            total = len(earnings)

            if total == 0:
                return 50.0

            beat_rate = (beats / total) * 100
            return beat_rate

        except:
            return 50.0

    def _calc_sector_momentum(self, ticker: str) -> float:
        """Calculate sector momentum score (0-100)."""
        try:
            # Simplified: use SPY as proxy
            spy = yf.Ticker("SPY")
            hist = spy.history(period="1mo")

            if len(hist) < 5:
                return 50.0

            pct_change = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100

            # -5% = 0, 0% = 50, +5% = 100
            score = 50 + (pct_change * 10)
            return max(0, min(100, score))

        except:
            return 50.0

    def _compute_ies(self, e: EarningsEnrichment) -> float:
        """Compute composite IES from components."""

        components = {
            'pre_earnings_runup': e.pre_earnings_runup,
            'implied_move_percentile': e.implied_move_percentile,
            'analyst_revision_momentum': e.analyst_revision_momentum,
            'options_skew_score': e.options_skew_score,
            'news_sentiment_score': e.news_sentiment_score,
            'historical_beat_rate': e.historical_beat_rate,
            'sector_momentum': e.sector_momentum,
        }

        weighted_sum = sum(
            components[k] * (self.IES_WEIGHTS[k] / 100)
            for k in components
        )

        # Clamp to 0-100
        return max(0, min(100, weighted_sum))

    def _classify_regime(self, e: EarningsEnrichment) -> IESRegime:
        """Classify the expectation regime."""

        if e.ies >= 75:
            return IESRegime.HYPED
        elif e.ies <= 25:
            return IESRegime.FEARED
        elif e.implied_move_percentile >= 80:
            return IESRegime.VOLATILE
        else:
            return IESRegime.NORMAL

    def _calc_position_scale(self, e: EarningsEnrichment) -> float:
        """Calculate position scaling factor (0.2 to 1.0)."""

        if not e.in_action_window:
            return 1.0

        # Scale down based on IES extremity
        ies_factor = 1 - (abs(e.ies - 50) / 100)  # Extreme = lower

        # Scale down based on implied move
        move_factor = 1 - (e.implied_move_percentile / 200)

        scale = ies_factor * move_factor
        return max(0.2, min(1.0, scale))

    def _calc_score_adjustment(self, e: EarningsEnrichment) -> int:
        """Calculate score adjustment (-20 to +20)."""

        if not e.in_action_window:
            return 0

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

        if e.in_action_window:
            flags.append(f"⚠️ Earnings in {e.days_to_earnings}d")

        if e.regime == IESRegime.HYPED:
            flags.append("🔥 HYPED: High expectations priced in")
        elif e.regime == IESRegime.FEARED:
            flags.append("😰 FEARED: Low expectations - potential upside")
        elif e.regime == IESRegime.VOLATILE:
            flags.append("⚡ VOLATILE: High implied move")

        if e.pre_earnings_runup >= 80:
            flags.append(f"📈 Run-up: Stock up significantly pre-earnings")

        if e.implied_move_percentile >= 80:
            flags.append(f"📊 Implied move at {e.implied_move_percentile:.0f}th percentile")

        if e.position_scale < 0.5:
            flags.append(f"⚖️ Reduce position size to {e.position_scale:.0%}")

        return flags

    def _assess_data_quality(self, e: EarningsEnrichment) -> str:
        """Assess data quality based on available inputs."""

        # Count how many components are at default 50
        defaults = sum(1 for v in [
            e.pre_earnings_runup,
            e.implied_move_percentile,
            e.analyst_revision_momentum,
            e.options_skew_score,
            e.news_sentiment_score,
            e.historical_beat_rate,
            e.sector_momentum,
        ] if v == 50.0)

        if defaults <= 2:
            return "HIGH"
        elif defaults <= 4:
            return "MEDIUM"
        else:
            return "LOW"


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