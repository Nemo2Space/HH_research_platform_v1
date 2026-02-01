from dataclasses import dataclass
from typing import Dict, Optional


# ---------------------------------------------------------------------
# Risk / trading constraints (v1 defaults)
# ---------------------------------------------------------------------
@dataclass
class RiskConstraints:
    # Portfolio constraints
    max_position_weight: float = 0.50     # 10% max per position
    max_sector_weight: float = 0.50       # 50% max per sector (adjustable)

    # Cash policy
    cash_target: float = 0.045            # 4.5% target cash
    cash_min: float = -1.00               # -50% min cash (allows significant margin)
    cash_max: float = 0.15                # 15% max cash

    # Rebalance drift triggers
    drift_abs_threshold: float = 0.05     # 5% absolute weight drift
    drift_rel_threshold: float = 0.25     # 25% relative drift for small targets
    drift_small_target_cutoff: float = 0.05  # 5% target cutoff

    # Turnover / execution limits
    max_turnover_per_cycle: float = 1.00  # 100% of NAV per rebalance cycle (no limit)
    max_trades_per_cycle: int = 100       # max trades per cycle
    min_order_notional_usd: float = 100.0

    # Liquidity / spread gates
    max_order_adv_pct: float = 0.02       # 2% of ADV per order
    max_spread_bps_etf: float = 20.0
    max_spread_bps_stock: float = 40.0

    # Safety toggles
    block_if_market_data_missing: bool = True
    block_if_prices_stale: bool = True
    
    # NEW: Allow override of hard gates
    allow_gate_override: bool = True      # If True, user can override blocked gates


DEFAULT_CONSTRAINTS = RiskConstraints()

# RELAXED constraints - use when you want to bypass most gates
RELAXED_CONSTRAINTS = RiskConstraints(
    max_position_weight=0.20,      # 20%
    max_sector_weight=1.00,        # 100% (no sector limit)
    cash_min=-0.50,                # -50% (allow significant margin)
    cash_max=0.50,                 # 50%
    max_turnover_per_cycle=1.00,   # 100%
    max_trades_per_cycle=200,
    allow_gate_override=True,
)


# ---------------------------------------------------------------------
# Strategy profile config
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyProfile:
    key: str
    name: str
    description: str

    # Target portfolio construction
    max_holdings: int = 25
    min_weight: float = 0.01  # ignore targets smaller than 1%

    # Signal blend
    w_total_score: float = 1.0
    w_signal_strength: float = 0.0
    w_committee_conviction: float = 0.0

    # Optional knobs
    prefer_etfs: bool = False
    sector_balance: bool = False

    # Override constraints (optional)
    constraints_override: Optional[RiskConstraints] = None


STRATEGIES: Dict[str, StrategyProfile] = {
    "core_passive_drift": StrategyProfile(
        key="core_passive_drift",
        name="Core Passive + Drift Control",
        description="Conservative rebalancing focused on drift + hard risk constraints.",
        max_holdings=25,
        min_weight=0.01,
        w_total_score=0.30,
        w_signal_strength=0.20,
        w_committee_conviction=0.50,
        prefer_etfs=True,
        sector_balance=False,
    ),
    "quality_lowvol": StrategyProfile(
        key="quality_lowvol",
        name="Quality + Low Vol",
        description="Bias toward quality and stability; aims for lower turnover.",
        max_holdings=25,
        min_weight=0.01,
        w_total_score=0.60,
        w_signal_strength=0.25,
        w_committee_conviction=0.15,
        prefer_etfs=False,
        sector_balance=True,
    ),
    "momentum": StrategyProfile(
        key="momentum",
        name="Momentum (Medium-Term)",
        description="Higher turnover; uses signal strength more aggressively.",
        max_holdings=30,
        min_weight=0.01,
        w_total_score=0.30,
        w_signal_strength=0.55,
        w_committee_conviction=0.15,
        prefer_etfs=False,
        sector_balance=False,
    ),
    "ai_composite": StrategyProfile(
        key="ai_composite",
        name="AI Composite (Platform Full Score)",
        description="Uses your platform combined scoring with conservative execution gates.",
        max_holdings=30,
        min_weight=0.01,
        w_total_score=0.50,
        w_signal_strength=0.25,
        w_committee_conviction=0.25,
        prefer_etfs=False,
        sector_balance=True,
    ),
    "ai_probability": StrategyProfile(
        key="ai_probability",
        name="AI Probability",
        description="Uses probability and EV if available.",
        max_holdings=30,
        min_weight=0.01,
        w_total_score=0.0,
        w_signal_strength=0.0,
        w_committee_conviction=0.0,
        constraints_override=None,
    ),
    "ai_conservative": StrategyProfile(
        key="ai_conservative",
        name="AI Conservative",
        description="Stricter AI probability+EV thresholds.",
        max_holdings=20,
        min_weight=0.02,
        w_total_score=0.0,
        w_signal_strength=0.0,
        w_committee_conviction=0.0,
        constraints_override=None,
    ),
}
