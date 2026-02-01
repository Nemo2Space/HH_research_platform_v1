from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# -----------------------------
# Core portfolio state
# -----------------------------
@dataclass(frozen=True)
class Position:
    symbol: str
    sec_type: str
    currency: str
    exchange: str
    con_id: int
    quantity: float
    avg_cost: Optional[float] = None
    market_price: Optional[float] = None
    market_value: Optional[float] = None


@dataclass(frozen=True)
class OrderStatus:
    order_id: Optional[int]
    perm_id: Optional[int]
    symbol: Optional[str]
    action: Optional[str]
    quantity: Optional[float]
    order_type: Optional[str]
    lmt_price: Optional[float]
    status: Optional[str]
    account: Optional[str] = None


@dataclass(frozen=True)
class PortfolioSnapshot:
    ts_utc: datetime
    account: str

    currency: Optional[str] = None
    net_liquidation: Optional[float] = None
    total_cash: Optional[float] = None
    available_funds: Optional[float] = None
    buying_power: Optional[float] = None

    positions: List[Position] = field(default_factory=list)
    open_orders: List[OrderStatus] = field(default_factory=list)


# -----------------------------
# Signals (normalized view)
# -----------------------------
@dataclass(frozen=True)
class SignalRow:
    symbol: str

    # normalized scores [0..1] or [-1..+1] depending on your system
    expected_return_score: Optional[float] = None
    risk_flag_score: Optional[float] = None
    conviction_score: Optional[float] = None

    # original/raw info for explainability
    raw: Dict[str, Any] = field(default_factory=dict)

    # data freshness
    asof_utc: Optional[datetime] = None
    source: Optional[str] = None


@dataclass(frozen=True)
class SignalSnapshot:
    ts_utc: datetime
    rows: Dict[str, SignalRow] = field(default_factory=dict)  # key: symbol


# -----------------------------
# Targets / plan / gates
# -----------------------------
@dataclass(frozen=True)
class TargetWeights:
    ts_utc: datetime
    strategy_key: str
    weights: Dict[str, float] = field(default_factory=dict)  # symbol -> target weight (0..1)
    cash_target: Optional[float] = None
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class OrderTicket:
    symbol: str
    action: str  # "BUY" / "SELL"
    quantity: float
    # optional details
    order_type: str = "MKT"  # "MKT" / "LMT" etc.
    limit_price: Optional[float] = None
    tif: str = "DAY"
    reason: Optional[str] = None


@dataclass(frozen=True)
class TradePlan:
    ts_utc: datetime
    account: str
    strategy_key: str

    # summary
    nav: Optional[float] = None
    cash: Optional[float] = None
    turnover_est: Optional[float] = None  # fraction of NAV
    num_trades: int = 0

    # details
    current_weights: Dict[str, float] = field(default_factory=dict)
    target_weights: Dict[str, float] = field(default_factory=dict)
    drift_by_symbol: Dict[str, float] = field(default_factory=dict)

    orders: List[OrderTicket] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class GateReport:
    ts_utc: datetime
    blocked: bool
    block_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # helpful derived flags
    requires_human_action: bool = True  # always true if blocked; can also be true for warnings


@dataclass(frozen=True)
class ExecutionResult:
    ts_utc: datetime
    account: str
    strategy_key: str
    submitted: bool
    submitted_orders: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# -----------------------------
# Decision context (inputs bundled)
# -----------------------------
@dataclass(frozen=True)
class DecisionContext:
    ts_utc: datetime
    account: str
    strategy_key: str

    portfolio: PortfolioSnapshot
    signals: SignalSnapshot
