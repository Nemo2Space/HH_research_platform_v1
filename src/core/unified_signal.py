"""
Unified Signal Model

The single source of truth for any ticker's signal.
Combines all analyzers into one comprehensive signal.

Author: Alpha Research Platform
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from enum import Enum
import json


class SignalStrength(Enum):
    """Signal strength levels."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class AssetType(Enum):
    """Asset type for smart routing."""
    STOCK = "STOCK"
    BOND_ETF = "BOND_ETF"
    ETF = "ETF"
    UNKNOWN = "UNKNOWN"


# Bond ETF tickers for smart detection
BOND_ETFS = {'TLT', 'ZROZ', 'EDV', 'TMF', 'TBT', 'SHY', 'IEF', 'BND', 'AGG', 'LQD', 'HYG', 'JNK'}


@dataclass
class ComponentScore:
    """Individual component score with metadata."""
    name: str
    score: int  # 0-100
    signal: str  # BUY/HOLD/SELL
    weight: float  # Weight in final calculation
    reason: str = ""  # One-line explanation

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UnifiedSignal:
    """
    The complete signal for any ticker.

    This is the single source of truth that combines:
    - Technical analysis
    - Fundamental analysis
    - Sentiment/News
    - Options flow
    - Earnings intelligence
    - Committee decision
    - Bond analysis (for bond ETFs)
    """

    # ========================================
    # IDENTIFICATION
    # ========================================
    ticker: str
    company_name: str = ""
    sector: str = ""
    asset_type: AssetType = AssetType.STOCK

    # ========================================
    # OVERALL SIGNAL
    # ========================================
    # Today signal - short term (will it go up today/this week?)
    today_signal: SignalStrength = SignalStrength.HOLD
    today_score: int = 50  # 0-100 (confidence it goes up)

    # Long-term signal - investment quality
    longterm_signal: SignalStrength = SignalStrength.HOLD
    longterm_score: int = 50  # 0-100

    # Risk assessment (separate from signal)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    risk_score: int = 50  # 0-100 (higher = more risky)
    risk_factors: List[str] = field(default_factory=list)

    # Why this signal (one sentence)
    signal_reason: str = ""

    # ========================================
    # COMPONENT SCORES (all 0-100)
    # ========================================
    technical_score: int = 50
    technical_signal: str = "HOLD"
    technical_reason: str = ""

    fundamental_score: int = 50
    fundamental_signal: str = "HOLD"
    fundamental_reason: str = ""

    sentiment_score: int = 50
    sentiment_signal: str = "HOLD"
    sentiment_reason: str = ""

    options_score: int = 50
    options_signal: str = "HOLD"
    options_reason: str = ""

    earnings_score: int = 50
    earnings_signal: str = "HOLD"
    earnings_reason: str = ""

    # Bond-specific (only for bond ETFs)
    bond_score: int = 50
    bond_signal: str = "HOLD"
    bond_reason: str = ""

    # ========================================
    # INSTITUTIONAL SIGNALS (Phase 2-4)
    # ========================================
    # GEX/Gamma Analysis
    gex_score: int = 50
    gex_signal: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL, PINNED
    gex_regime: str = "NEUTRAL"  # POSITIVE_GEX, NEGATIVE_GEX, NEUTRAL
    gex_reason: str = ""

    # Dark Pool Flow
    dark_pool_score: int = 50
    dark_pool_signal: str = "NEUTRAL"  # ACCUMULATION, DISTRIBUTION, NEUTRAL
    institutional_bias: str = "NEUTRAL"  # BUYING, SELLING, NEUTRAL
    dark_pool_reason: str = ""

    # Cross-Asset Context
    cross_asset_score: int = 50
    cross_asset_signal: str = "NEUTRAL"  # RISK_ON, RISK_OFF, NEUTRAL
    cycle_phase: str = ""  # EARLY_CYCLE, MID_CYCLE, LATE_CYCLE, RECESSION
    cross_asset_reason: str = ""

    # Sentiment NLP (AI-powered)
    sentiment_nlp_score: int = 50
    sentiment_nlp_signal: str = "NEUTRAL"
    sentiment_nlp_reason: str = ""

    # Earnings Whisper
    whisper_score: int = 50
    whisper_signal: str = "NEUTRAL"  # BEAT_EXPECTED, MISS_EXPECTED, NEUTRAL
    whisper_reason: str = ""

    # Insider Transactions (Form 4 - 2 day lag)
    insider_score: int = 50
    insider_signal: str = "NEUTRAL"  # STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
    insider_ceo_bought: bool = False
    insider_cfo_bought: bool = False
    insider_cluster_buying: bool = False
    insider_cluster_selling: bool = False
    insider_net_value: float = 0.0
    insider_reason: str = ""

    # 13F Institutional Holdings (45+ day lag)
    inst_13f_score: int = 50
    inst_13f_signal: str = "NEUTRAL"
    inst_buffett_owns: bool = False
    inst_buffett_added: bool = False
    inst_activist_involved: bool = False
    inst_notable_holders: int = 0
    inst_13f_reason: str = ""

    # ========================================
    # COMMITTEE DECISION
    # ========================================
    committee_verdict: str = "HOLD"
    committee_confidence: float = 0.5
    committee_votes: Dict[str, str] = field(default_factory=dict)
    # e.g., {'fundamental': 'BUY', 'technical': 'BUY', 'sentiment': 'HOLD'}
    committee_agreement: float = 0.0  # 0-1, how much they agree

    # ========================================
    # CATALYSTS & EVENTS
    # ========================================
    next_catalyst: str = ""  # "ER 28d", "FOMC 14d", etc.
    next_catalyst_date: Optional[date] = None
    days_to_catalyst: Optional[int] = None

    # Earnings specific
    earnings_date: Optional[date] = None
    days_to_earnings: Optional[int] = None
    last_earnings_result: str = ""  # "BEAT", "MISS", "INLINE"
    ies_score: int = 0  # Implied Expectations Score
    ecs_category: str = ""  # Expectations Clearance Score

    # ========================================
    # PRICE & TARGETS
    # ========================================
    current_price: float = 0
    target_price: float = 0
    stop_loss: float = 0
    upside_pct: float = 0
    downside_pct: float = 0
    risk_reward: float = 0

    # 52-week context
    week_52_high: float = 0
    week_52_low: float = 0
    pct_from_high: float = 0
    pct_from_low: float = 0

    # ========================================
    # PORTFOLIO CONTEXT
    # ========================================
    in_portfolio: bool = False
    portfolio_weight: float = 0
    target_weight: float = 0
    portfolio_pnl: float = 0
    portfolio_pnl_pct: float = 0
    days_held: int = 0

    # ========================================
    # TRADE IDEA (if generated)
    # ========================================
    has_trade_idea: bool = False
    trade_action: str = ""  # BUY, SELL, HOLD
    trade_conviction: str = ""  # HIGH, MEDIUM, LOW
    trade_thesis: str = ""
    trade_timeframe: str = ""  # "Swing 2-4w", "Position 1-3m"

    # ========================================
    # FLAGS & ALERTS
    # ========================================
    flags: List[str] = field(default_factory=list)
    # e.g., ["🔥 Hot", "📊 ER Beat", "⚠️ Volatile", "🟢 Insider Buying"]

    # ========================================
    # METADATA
    # ========================================
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    data_quality: str = "UNKNOWN"  # HIGH, MEDIUM, LOW, UNKNOWN
    components_available: List[str] = field(default_factory=list)
    # e.g., ["technical", "fundamental", "sentiment", "options"]

    # ========================================
    # HELPER METHODS
    # ========================================

    @classmethod
    def detect_asset_type(cls, ticker: str) -> AssetType:
        """Detect if ticker is stock, bond ETF, or other ETF."""
        ticker_upper = ticker.upper()
        if ticker_upper in BOND_ETFS:
            return AssetType.BOND_ETF
        # Could add more ETF detection logic here
        return AssetType.STOCK

    def get_signal_emoji(self) -> str:
        """Get emoji for today's signal."""
        mapping = {
            SignalStrength.STRONG_BUY: "🟢🟢",
            SignalStrength.BUY: "🟢",
            SignalStrength.HOLD: "🟡",
            SignalStrength.SELL: "🔴",
            SignalStrength.STRONG_SELL: "🔴🔴",
        }
        return mapping.get(self.today_signal, "⚪")

    def get_risk_emoji(self) -> str:
        """Get emoji for risk level."""
        mapping = {
            RiskLevel.LOW: "🟢",
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.HIGH: "🟠",
            RiskLevel.EXTREME: "🔴",
        }
        return mapping.get(self.risk_level, "⚪")

    def get_stars(self) -> str:
        """Get star rating for long-term score."""
        if self.longterm_score >= 90:
            return "⭐⭐⭐⭐⭐"
        elif self.longterm_score >= 75:
            return "⭐⭐⭐⭐"
        elif self.longterm_score >= 60:
            return "⭐⭐⭐"
        elif self.longterm_score >= 45:
            return "⭐⭐"
        else:
            return "⭐"

    def get_component_scores(self) -> List[ComponentScore]:
        """Get all component scores as list."""
        components = []

        if self.asset_type == AssetType.BOND_ETF:
            # Bond-specific components
            components.append(ComponentScore(
                name="Bond Analysis",
                score=self.bond_score,
                signal=self.bond_signal,
                weight=0.40,
                reason=self.bond_reason
            ))
        else:
            # Stock components
            components.append(ComponentScore(
                name="Technical",
                score=self.technical_score,
                signal=self.technical_signal,
                weight=0.20,
                reason=self.technical_reason
            ))
            components.append(ComponentScore(
                name="Fundamental",
                score=self.fundamental_score,
                signal=self.fundamental_signal,
                weight=0.20,
                reason=self.fundamental_reason
            ))
            components.append(ComponentScore(
                name="Sentiment",
                score=self.sentiment_score,
                signal=self.sentiment_signal,
                weight=0.15,
                reason=self.sentiment_reason
            ))
            components.append(ComponentScore(
                name="Options Flow",
                score=self.options_score,
                signal=self.options_signal,
                weight=0.15,
                reason=self.options_reason
            ))
            components.append(ComponentScore(
                name="Earnings",
                score=self.earnings_score,
                signal=self.earnings_signal,
                weight=0.15,
                reason=self.earnings_reason
            ))

        return components

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON/DB storage."""
        data = asdict(self)
        # Convert enums to strings
        data['today_signal'] = self.today_signal.value
        data['risk_level'] = self.risk_level.value
        data['longterm_signal'] = self.longterm_signal.value
        data['asset_type'] = self.asset_type.value
        # Convert dates
        if self.next_catalyst_date:
            data['next_catalyst_date'] = self.next_catalyst_date.isoformat()
        if self.earnings_date:
            data['earnings_date'] = self.earnings_date.isoformat()
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> 'UnifiedSignal':
        """Create from dictionary."""
        # Convert string enums back
        if isinstance(data.get('today_signal'), str):
            data['today_signal'] = SignalStrength(data['today_signal'])
        if isinstance(data.get('longterm_signal'), str):
            data['longterm_signal'] = SignalStrength(data['longterm_signal'])
        if isinstance(data.get('risk_level'), str):
            data['risk_level'] = RiskLevel(data['risk_level'])
        if isinstance(data.get('asset_type'), str):
            data['asset_type'] = AssetType(data['asset_type'])
        # Convert dates
        if data.get('next_catalyst_date') and isinstance(data['next_catalyst_date'], str):
            data['next_catalyst_date'] = date.fromisoformat(data['next_catalyst_date'])
        if data.get('earnings_date') and isinstance(data['earnings_date'], str):
            data['earnings_date'] = date.fromisoformat(data['earnings_date'])
        if data.get('created_at') and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('updated_at') and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class MarketOverview:
    """
    Market-wide summary for the main signals page header.
    """
    # Regime
    regime: str = "NEUTRAL"  # RISK_ON, RISK_OFF, NEUTRAL
    regime_score: int = 50  # 0-100
    regime_description: str = ""

    # Key metrics
    vix: float = 0
    vix_change: float = 0
    vix_trend: str = ""  # "Rising", "Falling", "Stable"

    # Market performance
    spy_change: float = 0
    qqq_change: float = 0
    iwm_change: float = 0

    # Sector leadership
    leading_sector: str = ""
    leading_sector_change: float = 0
    lagging_sector: str = ""
    lagging_sector_change: float = 0

    # Economic calendar
    economic_events_today: List[Dict] = field(default_factory=list)
    has_high_impact_today: bool = False
    next_fed_meeting: Optional[date] = None
    days_to_fed: int = 999

    # AI Summary
    ai_summary: str = ""

    # Metadata
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.next_fed_meeting:
            data['next_fed_meeting'] = self.next_fed_meeting.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data


@dataclass
class SignalSnapshot:
    """
    Historical snapshot of a signal for performance tracking.
    Saved daily to database.
    """
    ticker: str
    snapshot_date: date

    # Signal at this point in time
    today_signal: str
    today_score: int
    longterm_score: int
    risk_level: str

    # Price at snapshot
    price_at_snapshot: float

    # Component scores
    technical_score: int
    fundamental_score: int
    sentiment_score: int
    options_score: int
    earnings_score: int

    # For later comparison
    price_1d_later: Optional[float] = None
    price_7d_later: Optional[float] = None
    price_30d_later: Optional[float] = None

    # Was signal correct?
    correct_1d: Optional[bool] = None
    correct_7d: Optional[bool] = None
    correct_30d: Optional[bool] = None

    # Full signal JSON for deep investigation
    full_signal_json: str = ""

    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        data = asdict(self)
        data['snapshot_date'] = self.snapshot_date.isoformat()
        data['created_at'] = self.created_at.isoformat()
        return data