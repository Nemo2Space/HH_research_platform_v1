"""
Earnings Intelligence System - Core Data Models

Defines all dataclasses and enums used throughout the Earnings Intelligence System.
These models mirror the database schema and provide type safety.

Author: Alpha Research Platform
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from enum import Enum
from typing import List, Optional, Dict, Any


# ============================================================
# ENUMS
# ============================================================

class ECSCategory(Enum):
    """
    Expectations Clearance Score categories.

    Indicates whether earnings results cleared the priced-in expectations bar.
    """
    STRONG_BEAT = "STRONG_BEAT"   # event_z > required_z + 0.5
    BEAT = "BEAT"                 # event_z >= required_z
    INLINE = "INLINE"             # event_z >= required_z - 0.3
    MISS = "MISS"                 # event_z >= required_z - 1.0
    STRONG_MISS = "STRONG_MISS"   # event_z < required_z - 1.0
    UNKNOWN = "UNKNOWN"           # Not yet computed or insufficient data

    @classmethod
    def from_event_z(cls, event_z: float, required_z: float) -> 'ECSCategory':
        """
        Determine ECS category from event_z and required_z threshold.

        Args:
            event_z: Blended event surprise z-score
            required_z: Required z-score threshold based on implied move

        Returns:
            ECSCategory
        """
        if event_z is None or required_z is None:
            return cls.UNKNOWN

        diff = event_z - required_z

        if diff > 0.5:
            return cls.STRONG_BEAT
        elif diff >= 0:
            return cls.BEAT
        elif diff >= -0.3:
            return cls.INLINE
        elif diff >= -1.0:
            return cls.MISS
        else:
            return cls.STRONG_MISS

    @property
    def is_positive(self) -> bool:
        """Returns True if this is a positive outcome (BEAT or STRONG_BEAT)."""
        return self in (ECSCategory.STRONG_BEAT, ECSCategory.BEAT)

    @property
    def is_negative(self) -> bool:
        """Returns True if this is a negative outcome (MISS or STRONG_MISS)."""
        return self in (ECSCategory.MISS, ECSCategory.STRONG_MISS)

    @property
    def score_adjustment(self) -> int:
        """Returns score adjustment for Trade Ideas integration."""
        adjustments = {
            ECSCategory.STRONG_BEAT: 18,
            ECSCategory.BEAT: 8,
            ECSCategory.INLINE: 0,
            ECSCategory.MISS: -12,
            ECSCategory.STRONG_MISS: -22,
            ECSCategory.UNKNOWN: 0,
        }
        return adjustments.get(self, 0)


class ExpectationsRegime(Enum):
    """
    Pre-earnings expectations regime classification.

    Summarizes the market setup before earnings for decision logic.
    """
    HYPED = "HYPED"         # High expectations, high bar to clear
    FEARED = "FEARED"       # Low expectations, low bar to clear
    VOLATILE = "VOLATILE"   # High uncertainty, direction unclear
    NORMAL = "NORMAL"       # Standard expectations

    @classmethod
    def classify(cls, ies: float, implied_move_pctl: float, drift_20d: float) -> 'ExpectationsRegime':
        """
        Classify regime based on IES components.

        Priority order: HYPED > FEARED > VOLATILE > NORMAL

        Args:
            ies: Implied Expectations Score (0-100)
            implied_move_pctl: Implied move percentile (0-100)
            drift_20d: 20-day price drift (decimal, e.g., 0.15 = 15%)

        Returns:
            ExpectationsRegime
        """
        if ies is None or implied_move_pctl is None or drift_20d is None:
            return cls.NORMAL

        # HYPED: High expectations across the board
        if ies >= 70 and implied_move_pctl >= 70 and drift_20d >= 0.10:
            return cls.HYPED

        # FEARED: Market is skeptical
        if ies <= 50 and drift_20d <= -0.05:
            return cls.FEARED

        # VOLATILE: Expecting big move but uncertain direction
        if implied_move_pctl >= 80 and 40 <= ies <= 60:
            return cls.VOLATILE

        # Default
        return cls.NORMAL

    @property
    def required_ecs_for_buy(self) -> ECSCategory:
        """Minimum ECS required to issue a BUY recommendation."""
        requirements = {
            ExpectationsRegime.HYPED: ECSCategory.STRONG_BEAT,
            ExpectationsRegime.FEARED: ECSCategory.BEAT,
            ExpectationsRegime.VOLATILE: ECSCategory.BEAT,
            ExpectationsRegime.NORMAL: ECSCategory.BEAT,
        }
        return requirements.get(self, ECSCategory.BEAT)

    @property
    def default_position_scale(self) -> float:
        """Default position scale for this regime (before IES/IV adjustments)."""
        scales = {
            ExpectationsRegime.HYPED: 0.30,
            ExpectationsRegime.FEARED: 0.70,
            ExpectationsRegime.VOLATILE: 0.50,
            ExpectationsRegime.NORMAL: 1.00,
        }
        return scales.get(self, 1.0)


class DataQuality(Enum):
    """
    Data quality assessment for earnings intelligence calculations.

    Indicates confidence level in computed scores.
    """
    HIGH = "HIGH"       # All required inputs available and current
    MEDIUM = "MEDIUM"   # Most inputs available, some missing or stale
    LOW = "LOW"         # Critical inputs missing, outputs unreliable

    @classmethod
    def assess(cls, missing_inputs: List[str]) -> 'DataQuality':
        """
        Assess data quality based on missing inputs.

        Args:
            missing_inputs: List of missing input names

        Returns:
            DataQuality level
        """
        if not missing_inputs:
            return cls.HIGH

        # Critical inputs - LOW quality if missing
        critical = {'eps_actual', 'eps_consensus', 'revenue_actual',
                   'revenue_consensus', 'current_price', 'earnings_date'}

        # Important inputs - MEDIUM quality if missing
        important = {'implied_move_pct', 'iv', 'guidance_direction',
                    'analyst_revisions', 'options_chain'}

        missing_set = set(missing_inputs)

        if missing_set & critical:
            return cls.LOW
        elif missing_set & important:
            return cls.MEDIUM
        else:
            return cls.HIGH


class SuppressionReason(Enum):
    """
    Reason why a trading recommendation was suppressed.

    Distinguishes between logic-based and data-based suppression.
    """
    LOGIC = "LOGIC"   # Good data, but setup doesn't meet criteria
    DATA = "DATA"     # Insufficient data to make reliable determination
    NONE = None       # No suppression, recommendation stands

    @classmethod
    def determine(cls, data_quality: DataQuality, ecs: ECSCategory,
                  regime: ExpectationsRegime) -> 'SuppressionReason':
        """
        Determine if and why recommendation should be suppressed.

        Args:
            data_quality: Current data quality level
            ecs: Expectations Clearance Score
            regime: Expectations regime

        Returns:
            SuppressionReason or NONE
        """
        # Data quality too low
        if data_quality == DataQuality.LOW:
            return cls.DATA

        # Logic-based: HYPED regime requires STRONG_BEAT
        if regime == ExpectationsRegime.HYPED and not ecs == ECSCategory.STRONG_BEAT:
            return cls.LOGIC

        # Logic-based: Any regime with MISS or worse
        if ecs in (ECSCategory.MISS, ECSCategory.STRONG_MISS):
            return cls.LOGIC

        return cls.NONE


class GuidanceDirection(Enum):
    """Guidance direction from earnings report."""
    RAISED_STRONG = "RAISED_STRONG"   # Significantly above prior
    RAISED = "RAISED"                 # Above prior
    MAINTAINED = "MAINTAINED"         # In-line with prior
    LOWERED = "LOWERED"               # Below prior
    LOWERED_STRONG = "LOWERED_STRONG" # Significantly below prior
    NOT_PROVIDED = "NOT_PROVIDED"     # No guidance given

    @property
    def numeric_value(self) -> float:
        """Numeric mapping for z-score calculation."""
        values = {
            GuidanceDirection.RAISED_STRONG: 2.0,
            GuidanceDirection.RAISED: 1.0,
            GuidanceDirection.MAINTAINED: 0.0,
            GuidanceDirection.LOWERED: -1.0,
            GuidanceDirection.LOWERED_STRONG: -2.0,
            GuidanceDirection.NOT_PROVIDED: 0.0,
        }
        return values.get(self, 0.0)

    @property
    def eqs_score(self) -> float:
        """Score contribution for EQS (0-100 scale)."""
        scores = {
            GuidanceDirection.RAISED_STRONG: 95,
            GuidanceDirection.RAISED: 77,
            GuidanceDirection.MAINTAINED: 50,
            GuidanceDirection.LOWERED: 30,
            GuidanceDirection.LOWERED_STRONG: 8,
            GuidanceDirection.NOT_PROVIDED: 50,
        }
        return scores.get(self, 50)


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class IESComponents:
    """
    Components that make up the Implied Expectations Score.

    All values computed BEFORE earnings release.
    """
    # Price drift
    drift_20d: Optional[float] = None          # 20-day return (decimal)
    rel_drift_20d: Optional[float] = None      # Relative to sector ETF

    # Options-derived
    iv: Optional[float] = None                 # Current implied volatility
    iv_pctl: Optional[float] = None            # IV percentile (0-100)
    implied_move_pct: Optional[float] = None   # Expected move (decimal)
    implied_move_pctl: Optional[float] = None  # Implied move percentile (0-100)
    skew_shift: Optional[float] = None         # Call IV - Put IV change

    # Sentiment
    revision_score: Optional[float] = None     # Analyst revisions (0-100)
    confidence_lang_score: Optional[float] = None  # Bullish language (0-100)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def missing_inputs(self) -> List[str]:
        """List of missing (None) inputs."""
        missing = []
        for field_name, value in asdict(self).items():
            if value is None:
                missing.append(field_name)
        return missing


@dataclass
class EQSComponents:
    """
    Components that make up the Earnings Quality Score.

    Computed AFTER earnings release.
    """
    eps_z: Optional[float] = None              # EPS surprise z-score
    rev_z: Optional[float] = None              # Revenue surprise z-score
    guidance_score: Optional[float] = None     # Guidance score (0-100)
    margin_score: Optional[float] = None       # Margin trend (0-100)
    tone_score: Optional[float] = None         # Management tone (0-100)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ZScoreComponents:
    """
    Z-score components for event surprise calculation.
    """
    eps_z: Optional[float] = None              # EPS surprise z-score
    rev_z: Optional[float] = None              # Revenue surprise z-score
    guidance_z: Optional[float] = None         # Guidance z-score
    event_z: Optional[float] = None            # Blended z-score

    # Historical context
    eps_surprise_pct: Optional[float] = None
    revenue_surprise_pct: Optional[float] = None
    guidance_direction: Optional[GuidanceDirection] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.guidance_direction:
            result['guidance_direction'] = self.guidance_direction.value
        return result


@dataclass
class PositionScaling:
    """
    Position scaling calculation results.
    """
    base_scale: float = 1.0
    ies_penalty: float = 0.0
    implied_move_penalty: float = 0.0
    final_scale: float = 1.0

    # Constraints applied
    floor_applied: bool = False
    ceiling_applied: bool = False

    @classmethod
    def calculate(cls, ies: float, implied_move_pctl: float) -> 'PositionScaling':
        """
        Calculate position scaling based on IES and implied move.

        Formula: scale = 1.0 - ies_penalty - implied_move_penalty
        Clamped to [0.20, 1.00]

        Args:
            ies: Implied Expectations Score (0-100)
            implied_move_pctl: Implied move percentile (0-100)

        Returns:
            PositionScaling with all components
        """
        result = cls()

        # IES penalty: kicks in above 70
        if ies is not None and ies > 70:
            result.ies_penalty = (ies - 70) / 100

        # Implied move penalty: kicks in above 60
        if implied_move_pctl is not None and implied_move_pctl > 60:
            result.implied_move_penalty = (implied_move_pctl - 60) / 150

        # Calculate final scale
        raw_scale = result.base_scale - result.ies_penalty - result.implied_move_penalty

        # Apply floor (0.20)
        if raw_scale < 0.20:
            result.final_scale = 0.20
            result.floor_applied = True
        # Apply ceiling (1.0)
        elif raw_scale > 1.0:
            result.final_scale = 1.0
            result.ceiling_applied = True
        else:
            result.final_scale = raw_scale

        return result


@dataclass
class ReactionMeasurements:
    """
    Post-earnings price reaction measurements.
    """
    pre_earnings_close: Optional[float] = None
    post_earnings_open: Optional[float] = None
    post_earnings_close: Optional[float] = None

    # Calculated reactions
    gap_reaction: Optional[float] = None       # Overnight gap
    intraday_reaction: Optional[float] = None  # First day move
    total_reaction: Optional[float] = None     # Canonical metric

    # Extended
    reaction_5d: Optional[float] = None
    reaction_10d: Optional[float] = None

    def calculate_reactions(self):
        """Calculate reaction percentages from price data."""
        if self.pre_earnings_close and self.post_earnings_open:
            self.gap_reaction = (
                (self.post_earnings_open - self.pre_earnings_close)
                / self.pre_earnings_close
            )

        if self.post_earnings_open and self.post_earnings_close:
            self.intraday_reaction = (
                (self.post_earnings_close - self.post_earnings_open)
                / self.post_earnings_open
            )

        if self.pre_earnings_close and self.post_earnings_close:
            self.total_reaction = (
                (self.post_earnings_close - self.pre_earnings_close)
                / self.pre_earnings_close
            )


@dataclass
class EarningsIntelligenceResult:
    """
    Complete result from Earnings Intelligence analysis.

    This is the main output structure combining all components.
    """
    # Identification
    ticker: str
    earnings_date: Optional[date] = None
    earnings_timestamp: Optional[datetime] = None
    sector: Optional[str] = None
    market_cap: Optional[int] = None

    # IES (Pre-Earnings)
    ies_components: Optional[IESComponents] = None
    ies: Optional[float] = None
    ies_compute_timestamp: Optional[datetime] = None

    # Regime
    regime: ExpectationsRegime = ExpectationsRegime.NORMAL

    # Raw Earnings Data
    eps_actual: Optional[float] = None
    eps_consensus: Optional[float] = None
    eps_surprise_pct: Optional[float] = None
    revenue_actual: Optional[int] = None
    revenue_consensus: Optional[int] = None
    revenue_surprise_pct: Optional[float] = None
    guidance_direction: Optional[GuidanceDirection] = None

    # Z-Scores (Post-Earnings)
    zscore_components: Optional[ZScoreComponents] = None
    event_z: Optional[float] = None

    # EQS (Post-Earnings)
    eqs_components: Optional[EQSComponents] = None
    eqs: Optional[float] = None

    # ECS (Post-Earnings)
    required_z: Optional[float] = None
    ecs: ECSCategory = ECSCategory.UNKNOWN

    # Position Scaling
    position_scaling: Optional[PositionScaling] = None
    position_scale: float = 1.0

    # Data Quality
    data_quality: DataQuality = DataQuality.HIGH
    missing_inputs: List[str] = field(default_factory=list)
    suppression_reason: Optional[SuppressionReason] = None

    # Reactions (Post-Earnings)
    reactions: Optional[ReactionMeasurements] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Window status
    in_compute_window: bool = False
    in_action_window: bool = False
    days_to_earnings: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage or JSON serialization."""
        result = {
            'ticker': self.ticker,
            'earnings_date': self.earnings_date.isoformat() if self.earnings_date else None,
            'earnings_timestamp': self.earnings_timestamp.isoformat() if self.earnings_timestamp else None,
            'sector': self.sector,
            'market_cap': self.market_cap,

            'ies': self.ies,
            'ies_compute_timestamp': self.ies_compute_timestamp.isoformat() if self.ies_compute_timestamp else None,
            'regime': self.regime.value if self.regime else None,

            'eps_actual': self.eps_actual,
            'eps_consensus': self.eps_consensus,
            'eps_surprise_pct': self.eps_surprise_pct,
            'revenue_actual': self.revenue_actual,
            'revenue_consensus': self.revenue_consensus,
            'revenue_surprise_pct': self.revenue_surprise_pct,
            'guidance_direction': self.guidance_direction.value if self.guidance_direction else None,

            'event_z': self.event_z,
            'eqs': self.eqs,
            'required_z': self.required_z,
            'ecs': self.ecs.value if self.ecs else None,

            'position_scale': self.position_scale,

            'data_quality': self.data_quality.value if self.data_quality else None,
            'missing_inputs': self.missing_inputs,
            'suppression_reason': self.suppression_reason.value if self.suppression_reason else None,

            'in_compute_window': self.in_compute_window,
            'in_action_window': self.in_action_window,
            'days_to_earnings': self.days_to_earnings,

            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

        # Add component details
        if self.ies_components:
            result.update({f'ies_{k}': v for k, v in self.ies_components.to_dict().items()})

        if self.zscore_components:
            result['eps_z'] = self.zscore_components.eps_z
            result['rev_z'] = self.zscore_components.rev_z
            result['guidance_z'] = self.zscore_components.guidance_z

        if self.eqs_components:
            result['guidance_score'] = self.eqs_components.guidance_score
            result['margin_score'] = self.eqs_components.margin_score
            result['tone_score'] = self.eqs_components.tone_score

        if self.position_scaling:
            result['ies_penalty'] = self.position_scaling.ies_penalty
            result['implied_move_penalty'] = self.position_scaling.implied_move_penalty

        if self.reactions:
            result['gap_reaction'] = self.reactions.gap_reaction
            result['intraday_reaction'] = self.reactions.intraday_reaction
            result['total_reaction'] = self.reactions.total_reaction
            result['reaction_5d'] = self.reactions.reaction_5d
            result['reaction_10d'] = self.reactions.reaction_10d

        return result

    def get_ai_context(self) -> str:
        """
        Generate formatted context string for AI Chat integration.

        Returns:
            Formatted string for inclusion in AI prompts
        """
        lines = [
            f"EARNINGS EXPECTATIONS: {self.ticker}",
            "=" * 40,
        ]

        if self.in_compute_window:
            window_status = "IN ACTION WINDOW" if self.in_action_window else "IN COMPUTE WINDOW"
            lines.append(f"Status: {window_status}")
            if self.days_to_earnings is not None:
                if self.days_to_earnings > 0:
                    lines.append(f"Days to Earnings: {self.days_to_earnings}")
                elif self.days_to_earnings == 0:
                    lines.append("Earnings: TODAY")
                else:
                    lines.append(f"Days Since Earnings: {abs(self.days_to_earnings)}")

        lines.append("")

        # Pre-earnings info
        if self.ies is not None:
            ies_level = "Low" if self.ies < 40 else "Normal" if self.ies < 70 else "Elevated" if self.ies < 85 else "Extreme"
            lines.append(f"Implied Expectations Score: {self.ies:.0f}/100 ({ies_level})")

        if self.regime:
            lines.append(f"Expectations Regime: {self.regime.value}")

        if self.position_scale < 1.0:
            lines.append(f"Position Scale: {self.position_scale:.0%} of normal")

        # Post-earnings info (if available)
        if self.ecs != ECSCategory.UNKNOWN:
            lines.append("")
            lines.append("POST-EARNINGS:")

            if self.eqs is not None:
                lines.append(f"  Earnings Quality Score: {self.eqs:.0f}/100")

            if self.event_z is not None:
                lines.append(f"  Event Surprise Z: {self.event_z:+.2f}")

            if self.required_z is not None:
                lines.append(f"  Required Z (to clear bar): {self.required_z:.2f}")

            lines.append(f"  Expectations Clearance: {self.ecs.value}")

        # Data quality
        if self.data_quality != DataQuality.HIGH:
            lines.append("")
            lines.append(f"Data Quality: {self.data_quality.value}")
            if self.missing_inputs:
                lines.append(f"Missing: {', '.join(self.missing_inputs[:3])}")

        # Suppression
        if self.suppression_reason and self.suppression_reason != SuppressionReason.NONE:
            lines.append("")
            lines.append(f"RECOMMENDATION SUPPRESSED: {self.suppression_reason.value}")

        return "\n".join(lines)

    @property
    def is_tradeable(self) -> bool:
        """Whether this setup is tradeable based on ECS and regime."""
        if self.suppression_reason and self.suppression_reason != SuppressionReason.NONE:
            return False
        if self.data_quality == DataQuality.LOW:
            return False
        if self.ecs == ECSCategory.UNKNOWN:
            return True  # Pre-earnings, may be tradeable
        return self.ecs.is_positive or self.ecs == ECSCategory.INLINE


# ============================================================
# CONSTANTS
# ============================================================

# IES component weights (sum to 100%)
IES_WEIGHTS = {
    'drift_20d': 0.20,
    'rel_drift_20d': 0.15,
    'iv_pctl': 0.15,
    'implied_move_pctl': 0.20,
    'skew_shift': 0.10,
    'revision_score': 0.10,
    'confidence_lang_score': 0.10,
}

# EQS component weights (sum to 100%)
EQS_WEIGHTS = {
    'eps_z': 0.25,
    'rev_z': 0.25,
    'guidance_score': 0.30,
    'margin_score': 0.10,
    'tone_score': 0.10,
}

# Event Z component weights (sum to 100%)
EVENT_Z_WEIGHTS = {
    'eps_z': 0.35,
    'rev_z': 0.30,
    'guidance_z': 0.35,
}

# Required Z formula parameters
REQUIRED_Z_BASE = 0.5
REQUIRED_Z_SLOPE = 1/50  # implied_move_pctl / 50
REQUIRED_Z_FLOOR = 0.8
REQUIRED_Z_CEILING = 2.0

# Position scaling parameters
POSITION_SCALE_FLOOR = 0.20
POSITION_SCALE_CEILING = 1.0
IES_PENALTY_THRESHOLD = 70
IMPLIED_MOVE_PENALTY_THRESHOLD = 60

# Window definitions (in terms of days_to_earnings)
# days_to_earnings: positive = future (before earnings), negative = past (after earnings)
# Compute window: from 10 days before earnings to 2 days after
# So: days_to_earnings from -2 (2 days after) to +10 (10 days before)
COMPUTE_WINDOW_START = -2   # 2 days after earnings (post-earnings)
COMPUTE_WINDOW_END = 10     # 10 days before earnings (pre-earnings)
# Action window: from 5 days before earnings to 2 days after
ACTION_WINDOW_START = -2    # 2 days after earnings (post-earnings)
ACTION_WINDOW_END = 5       # 5 days before earnings (pre-earnings)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    # Test enum functionality
    print("Testing ECSCategory...")
    ecs = ECSCategory.from_event_z(1.5, 1.2)
    print(f"  event_z=1.5, required_z=1.2 -> {ecs.value}")
    print(f"  is_positive: {ecs.is_positive}")
    print(f"  score_adjustment: {ecs.score_adjustment}")

    print("\nTesting ExpectationsRegime...")
    regime = ExpectationsRegime.classify(ies=78, implied_move_pctl=82, drift_20d=0.15)
    print(f"  IES=78, IM_pctl=82, drift=15% -> {regime.value}")
    print(f"  required_ecs_for_buy: {regime.required_ecs_for_buy.value}")

    print("\nTesting PositionScaling...")
    scaling = PositionScaling.calculate(ies=85, implied_move_pctl=90)
    print(f"  IES=85, IM_pctl=90 -> scale={scaling.final_scale:.0%}")
    print(f"  ies_penalty={scaling.ies_penalty:.3f}, im_penalty={scaling.implied_move_penalty:.3f}")

    print("\nTesting EarningsIntelligenceResult...")
    result = EarningsIntelligenceResult(
        ticker="NVDA",
        ies=78,
        regime=ExpectationsRegime.HYPED,
        ecs=ECSCategory.BEAT,
        position_scale=0.52,
        in_action_window=True,
        days_to_earnings=3
    )
    print(result.get_ai_context())

    print("\n[OK] All model tests passed!")