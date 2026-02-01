"""
Decision/Risk Layer - Phase 3

The gatekeeper that converts ML probabilities into actual trade decisions.
This is often more important than model improvements.

Key principle: Don't trade on probability alone. Trade on Expected Value + Risk.

EV = p * avg_win - (1-p) * avg_loss - costs

Location: src/ml/decision_layer.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class TradeDecision(Enum):
    """Possible trade decisions."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NO_TRADE = "NO_TRADE"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class RejectionReason(Enum):
    """Why a trade was rejected."""
    LOW_PROBABILITY = "Probability below threshold"
    NEGATIVE_EV = "Expected value is negative"
    LOW_EV = "Expected value below minimum"
    SECTOR_EXPOSURE = "Would exceed sector exposure limit"
    POSITION_LIMIT = "Would exceed position limit"
    DRAWDOWN_GUARD = "Portfolio in drawdown protection mode"
    LIQUIDITY = "Insufficient liquidity"
    SPREAD_TOO_WIDE = "Bid-ask spread too wide"
    EARNINGS_BLACKOUT = "Earnings within blackout window"
    HIGH_VIX = "VIX too high for new positions"
    CORRELATION = "Too correlated with existing positions"
    DAILY_LIMIT = "Daily trade limit reached"


@dataclass
class RiskLimits:
    """Risk management limits."""
    max_position_pct: float = 0.05
    max_sector_pct: float = 0.25
    max_positions: int = 30
    max_drawdown_pct: float = 0.10
    drawdown_reduce_pct: float = 0.05
    drawdown_halt_pct: float = 0.08
    max_vix_for_new: float = 35
    reduce_size_vix: float = 25
    earnings_blackout_days: int = 2
    max_correlation: float = 0.7
    max_daily_trades: int = 10
    max_daily_exposure_change: float = 0.10
    min_avg_volume: int = 500000
    max_spread_pct: float = 0.005


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    shares: int
    dollar_amount: float
    position_pct: float
    method: str
    stop_loss_pct: float
    dollar_risk: float
    risk_reward_ratio: float
    adjustments: List[str] = field(default_factory=list)


@dataclass
class TradeRecommendation:
    """Complete trade recommendation from decision layer."""
    ticker: str
    decision: TradeDecision
    ml_probability: float
    ml_confidence: str
    expected_value: float
    expected_return_pct: float
    position_size: Optional[PositionSizeResult]
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward_ratio: float
    max_loss_pct: float
    recommended_horizon: int
    approved: bool
    rejection_reasons: List[RejectionReason] = field(default_factory=list)
    explanation: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'decision': self.decision.value,
            'approved': self.approved,
            'ml_probability': round(self.ml_probability, 3),
            'expected_value': round(self.expected_value, 4),
            'entry_price': round(self.entry_price, 2),
            'stop_loss': round(self.stop_loss, 2),
            'target_price': round(self.target_price, 2),
            'risk_reward_ratio': round(self.risk_reward_ratio, 2),
            'position_pct': round(self.position_size.position_pct, 3) if self.position_size else 0,
            'shares': self.position_size.shares if self.position_size else 0,
            'rejection_reasons': [r.value for r in self.rejection_reasons],
            'explanation': self.explanation
        }


@dataclass
class PortfolioState:
    """Current portfolio state for risk calculations."""
    total_value: float
    cash: float
    positions: Dict[str, Dict]
    sector_exposure: Dict[str, float]
    daily_pnl: float
    drawdown_pct: float
    peak_value: float
    trades_today: int
    exposure_change_today: float
    current_vix: float

    @classmethod
    def empty(cls, starting_capital: float = 100000) -> 'PortfolioState':
        return cls(
            total_value=starting_capital,
            cash=starting_capital,
            positions={},
            sector_exposure={},
            daily_pnl=0,
            drawdown_pct=0,
            peak_value=starting_capital,
            trades_today=0,
            exposure_change_today=0,
            current_vix=20
        )


class PositionSizer:
    """Calculates optimal position size using Kelly, Volatility Targeting, or Equal Weight."""

    def __init__(self,
                 default_method: str = 'VOLATILITY_TARGET',
                 kelly_fraction: float = 0.25,
                 target_volatility: float = 0.02,
                 max_position_pct: float = 0.05):
        self.default_method = default_method
        self.kelly_fraction = kelly_fraction
        self.target_volatility = target_volatility
        self.max_position_pct = max_position_pct

    def calculate(self, portfolio: PortfolioState, probability: float,
                  avg_win: float, avg_loss: float, current_price: float,
                  stock_volatility: float = 0.02, stop_loss_pct: float = 0.05) -> PositionSizeResult:
        adjustments = []

        if self.default_method == 'KELLY':
            position_pct = self._kelly_size(probability, avg_win, avg_loss)
            method = 'KELLY'
        elif self.default_method == 'VOLATILITY_TARGET':
            position_pct = self._volatility_target_size(stock_volatility)
            method = 'VOLATILITY_TARGET'
        else:
            position_pct = self._equal_weight_size(portfolio)
            method = 'EQUAL_WEIGHT'

        if position_pct > self.max_position_pct:
            position_pct = self.max_position_pct
            adjustments.append(f"Capped at max {self.max_position_pct:.1%}")

        if portfolio.drawdown_pct > 0.05:
            reduction = min(0.5, portfolio.drawdown_pct * 5)
            position_pct *= (1 - reduction)
            adjustments.append(f"Reduced {reduction:.0%} for drawdown")

        if portfolio.current_vix > 25:
            vix_reduction = min(0.5, (portfolio.current_vix - 25) / 20)
            position_pct *= (1 - vix_reduction)
            adjustments.append(f"Reduced {vix_reduction:.0%} for high VIX")

        if probability < 0.6:
            prob_reduction = (0.6 - probability) * 2
            position_pct *= (1 - prob_reduction)
            adjustments.append(f"Reduced {prob_reduction:.0%} for lower probability")

        dollar_amount = portfolio.total_value * position_pct
        shares = int(dollar_amount / current_price)
        actual_dollar = shares * current_price
        actual_pct = actual_dollar / portfolio.total_value if portfolio.total_value > 0 else 0
        dollar_risk = actual_dollar * stop_loss_pct
        expected_gain = actual_dollar * avg_win
        expected_loss = actual_dollar * avg_loss
        risk_reward = expected_gain / expected_loss if expected_loss > 0 else 0

        return PositionSizeResult(
            shares=shares, dollar_amount=actual_dollar, position_pct=actual_pct,
            method=method, stop_loss_pct=stop_loss_pct, dollar_risk=dollar_risk,
            risk_reward_ratio=risk_reward, adjustments=adjustments
        )

    def _kelly_size(self, p: float, avg_win: float, avg_loss: float) -> float:
        if avg_win <= 0:
            return 0.01
        kelly = (p * avg_win - (1 - p) * avg_loss) / avg_win
        kelly *= self.kelly_fraction
        return max(0.01, min(kelly, self.max_position_pct))

    def _volatility_target_size(self, stock_volatility: float) -> float:
        if stock_volatility <= 0:
            return 0.02
        return min(self.target_volatility / stock_volatility, self.max_position_pct)

    def _equal_weight_size(self, portfolio: PortfolioState) -> float:
        n_positions = len(portfolio.positions) + 1
        return min(1.0 / n_positions, self.max_position_pct)


class DecisionLayer:
    """The gatekeeper for all trades. Converts ML predictions into actionable decisions."""

    def __init__(self, limits: RiskLimits = None, min_ev: float = 0.001,
                 min_probability: float = 0.55, cost_pct: float = 0.0015):
        self.limits = limits or RiskLimits()
        self.min_ev = min_ev
        self.min_probability = min_probability
        self.cost_pct = cost_pct
        self.position_sizer = PositionSizer()

    def evaluate(self, ticker: str, ml_prediction: Dict,
                 portfolio: PortfolioState, stock_data: Dict) -> TradeRecommendation:
        rejection_reasons = []
        warnings = []

        probability = ml_prediction.get('prob_win_5d', 0.5)
        confidence = ml_prediction.get('confidence', 'MEDIUM')
        horizon = ml_prediction.get('recommended_horizon', 5)

        current_price = stock_data.get('price', 100)
        sector = stock_data.get('sector', 'Unknown')
        volatility = stock_data.get('volatility', 0.02)
        avg_volume = stock_data.get('avg_volume', 1000000)
        spread_pct = stock_data.get('spread_pct', 0.001)
        earnings_date = stock_data.get('earnings_date')
        avg_win = stock_data.get('avg_win', 0.03)
        avg_loss = stock_data.get('avg_loss', 0.02)

        # Calculate EV
        ev = probability * avg_win - (1 - probability) * avg_loss - self.cost_pct
        expected_return_pct = ev * 100

        # Check thresholds
        if probability < self.min_probability:
            rejection_reasons.append(RejectionReason.LOW_PROBABILITY)
        if ev < 0:
            rejection_reasons.append(RejectionReason.NEGATIVE_EV)
        elif ev < self.min_ev:
            rejection_reasons.append(RejectionReason.LOW_EV)

        # Risk limit checks
        if portfolio.current_vix > self.limits.max_vix_for_new:
            rejection_reasons.append(RejectionReason.HIGH_VIX)
        if portfolio.drawdown_pct > self.limits.drawdown_halt_pct:
            rejection_reasons.append(RejectionReason.DRAWDOWN_GUARD)

        current_sector_exp = portfolio.sector_exposure.get(sector, 0)
        if current_sector_exp > self.limits.max_sector_pct:
            rejection_reasons.append(RejectionReason.SECTOR_EXPOSURE)

        if len(portfolio.positions) >= self.limits.max_positions and ticker not in portfolio.positions:
            rejection_reasons.append(RejectionReason.POSITION_LIMIT)
        if portfolio.trades_today >= self.limits.max_daily_trades:
            rejection_reasons.append(RejectionReason.DAILY_LIMIT)
        if avg_volume < self.limits.min_avg_volume:
            rejection_reasons.append(RejectionReason.LIQUIDITY)
        if spread_pct > self.limits.max_spread_pct:
            rejection_reasons.append(RejectionReason.SPREAD_TOO_WIDE)

        if earnings_date:
            days_to_earnings = (earnings_date - date.today()).days
            if 0 <= days_to_earnings <= self.limits.earnings_blackout_days:
                rejection_reasons.append(RejectionReason.EARNINGS_BLACKOUT)

        # Position sizing
        position_size = None
        if not rejection_reasons:
            position_size = self.position_sizer.calculate(
                portfolio, probability, avg_win, avg_loss,
                current_price, volatility, 0.05
            )

        # Entry/exit levels
        stop_loss_pct = min(0.07, max(0.03, volatility * 2))
        stop_loss = current_price * (1 - stop_loss_pct)
        target_pct = avg_win * 1.5
        target_price = current_price * (1 + target_pct)
        risk = current_price - stop_loss
        reward = target_price - current_price
        risk_reward = reward / risk if risk > 0 else 0

        # Decision
        approved = len(rejection_reasons) == 0
        if approved:
            if probability >= 0.70 and ev >= 0.005:
                decision = TradeDecision.STRONG_BUY
            elif probability >= 0.55 and ev >= 0.001:
                decision = TradeDecision.BUY
            else:
                decision = TradeDecision.NO_TRADE
                approved = False
        else:
            decision = TradeDecision.NO_TRADE

        # Explanation
        if approved:
            explanation = f"APPROVED: {ticker} {probability:.1%} prob, {expected_return_pct:.2f}% EV, {position_size.shares} shares"
        else:
            explanation = f"REJECTED: {', '.join(r.value for r in rejection_reasons)}"

        return TradeRecommendation(
            ticker=ticker, decision=decision, ml_probability=probability,
            ml_confidence=confidence, expected_value=ev,
            expected_return_pct=expected_return_pct, position_size=position_size,
            entry_price=current_price, stop_loss=stop_loss, target_price=target_price,
            risk_reward_ratio=risk_reward, max_loss_pct=stop_loss_pct,
            recommended_horizon=horizon, approved=approved,
            rejection_reasons=rejection_reasons, explanation=explanation, warnings=warnings
        )

    def get_portfolio_risk_summary(self, portfolio: PortfolioState) -> Dict:
        return {
            'total_value': portfolio.total_value,
            'invested_pct': 1 - (portfolio.cash / portfolio.total_value) if portfolio.total_value > 0 else 0,
            'position_count': len(portfolio.positions),
            'drawdown_pct': portfolio.drawdown_pct,
            'current_vix': portfolio.current_vix,
            'risk_status': self._get_risk_status(portfolio)
        }

    def _get_risk_status(self, portfolio: PortfolioState) -> str:
        if portfolio.drawdown_pct > self.limits.drawdown_halt_pct:
            return "🔴 HALT"
        elif portfolio.current_vix > self.limits.max_vix_for_new:
            return "🔴 HALT - VIX"
        elif portfolio.drawdown_pct > self.limits.drawdown_reduce_pct:
            return "🟡 CAUTION"
        else:
            return "🟢 NORMAL"