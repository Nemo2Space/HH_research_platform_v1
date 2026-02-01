"""
Transaction Cost Model

Realistic execution cost modeling for backtests and live trading.

Components:
1. Spread cost (bid-ask spread)
2. Slippage (market impact of order)
3. Market impact (price move from large orders)
4. Timing cost (delay costs)

Cost varies by:
- Market cap (mega-cap cheaper than small-cap)
- Volatility (higher vol = wider spreads)
- Time of day (open/close more expensive)
- Volume (higher volume = lower impact)

Author: Alpha Research Platform
Location: src/backtest/transaction_costs.py
"""

import numpy as np
from datetime import datetime, time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketCapTier(Enum):
    """Market cap classification for cost estimation."""
    MEGA = "mega"       # > $200B
    LARGE = "large"     # $10B - $200B
    MID = "mid"         # $2B - $10B
    SMALL = "small"     # $300M - $2B
    MICRO = "micro"     # < $300M


class TradeTiming(Enum):
    """Time of day for cost adjustment."""
    OPEN = "open"       # First 30 minutes
    CORE = "core"       # 10:00 - 15:30
    CLOSE = "close"     # Last 30 minutes
    OVERNIGHT = "overnight"  # After hours


@dataclass
class TransactionCost:
    """Complete transaction cost breakdown."""
    ticker: str
    trade_value: float
    direction: str  # 'BUY' or 'SELL'
    
    # Component costs (as decimal, e.g., 0.0003 = 3 bps)
    spread_cost: float
    slippage_cost: float
    market_impact_cost: float
    timing_cost: float
    
    # Total
    total_cost_pct: float  # As percentage
    total_cost_bps: float  # As basis points
    total_cost_dollars: float
    
    # Context
    market_cap_tier: str
    avg_daily_volume: float
    trade_pct_of_adv: float
    volatility: float
    trade_time: str
    
    def __repr__(self):
        return (f"TransactionCost({self.ticker}: {self.total_cost_bps:.1f} bps = "
                f"${self.total_cost_dollars:.2f})")


class TransactionCostModel:
    """
    Models realistic transaction costs for equity trades.
    
    Based on academic research and industry practice:
    - Spread: Bid-ask spread cost (half spread for one-way)
    - Slippage: Execution price vs. decision price
    - Market Impact: Almgren-Chriss model for large orders
    - Timing: Costs from execution delay
    
    References:
    - Almgren, R. & Chriss, N. (2000) "Optimal execution of portfolio transactions"
    - Kissell, R. (2013) "The Science of Algorithmic Trading and Portfolio Management"
    """
    
    # Base spread by market cap tier (half-spread, one-way, in bps)
    BASE_SPREAD_BPS = {
        MarketCapTier.MEGA: 1.0,    # 1 bp for mega-caps (AAPL, MSFT, etc.)
        MarketCapTier.LARGE: 2.0,   # 2 bps for large-caps
        MarketCapTier.MID: 4.0,     # 4 bps for mid-caps
        MarketCapTier.SMALL: 8.0,   # 8 bps for small-caps
        MarketCapTier.MICRO: 20.0,  # 20 bps for micro-caps
    }
    
    # Base slippage by market cap tier (in bps)
    BASE_SLIPPAGE_BPS = {
        MarketCapTier.MEGA: 0.5,
        MarketCapTier.LARGE: 1.0,
        MarketCapTier.MID: 2.0,
        MarketCapTier.SMALL: 4.0,
        MarketCapTier.MICRO: 10.0,
    }
    
    # Market impact parameters (Almgren-Chriss)
    IMPACT_ALPHA = 0.1       # Permanent impact coefficient
    IMPACT_BETA = 0.5        # Temporary impact coefficient
    IMPACT_GAMMA = 0.6       # Volume exponent
    IMPACT_ETA = 0.01        # Volatility multiplier
    
    # Timing adjustment by time of day
    TIMING_MULTIPLIER = {
        TradeTiming.OPEN: 1.5,      # 50% higher at open
        TradeTiming.CORE: 1.0,      # Base cost during core hours
        TradeTiming.CLOSE: 1.3,     # 30% higher at close
        TradeTiming.OVERNIGHT: 2.0,  # 100% higher after hours
    }
    
    # Volatility adjustment thresholds
    VOL_LOW = 0.15    # Annualized vol < 15%
    VOL_HIGH = 0.40   # Annualized vol > 40%
    
    def __init__(self, 
                 default_volatility: float = 0.25,
                 default_adv: float = 10_000_000):
        """
        Initialize transaction cost model.
        
        Args:
            default_volatility: Default annualized volatility if unknown
            default_adv: Default average daily volume in dollars if unknown
        """
        self.default_volatility = default_volatility
        self.default_adv = default_adv
    
    def classify_market_cap(self, market_cap: float) -> MarketCapTier:
        """Classify stock by market cap."""
        if market_cap >= 200_000_000_000:
            return MarketCapTier.MEGA
        elif market_cap >= 10_000_000_000:
            return MarketCapTier.LARGE
        elif market_cap >= 2_000_000_000:
            return MarketCapTier.MID
        elif market_cap >= 300_000_000:
            return MarketCapTier.SMALL
        else:
            return MarketCapTier.MICRO
    
    def get_trade_timing(self, trade_time: datetime = None) -> TradeTiming:
        """Determine trade timing category."""
        if trade_time is None:
            return TradeTiming.CORE
        
        t = trade_time.time()
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        if t < market_open or t >= market_close:
            return TradeTiming.OVERNIGHT
        elif t < time(10, 0):
            return TradeTiming.OPEN
        elif t >= time(15, 30):
            return TradeTiming.CLOSE
        else:
            return TradeTiming.CORE
    
    def calculate_spread_cost(self,
                              market_cap_tier: MarketCapTier,
                              volatility: float) -> float:
        """
        Calculate half-spread cost in decimal.
        
        Spread widens with volatility.
        """
        base_spread = self.BASE_SPREAD_BPS[market_cap_tier] / 10000
        
        # Adjust for volatility
        vol_multiplier = 1.0
        if volatility > self.VOL_HIGH:
            vol_multiplier = 1.5 + (volatility - self.VOL_HIGH) * 2
        elif volatility < self.VOL_LOW:
            vol_multiplier = 0.8
        
        return base_spread * vol_multiplier
    
    def calculate_slippage_cost(self,
                                market_cap_tier: MarketCapTier,
                                volatility: float) -> float:
        """
        Calculate slippage cost in decimal.
        
        Slippage is the difference between decision price and execution price.
        """
        base_slippage = self.BASE_SLIPPAGE_BPS[market_cap_tier] / 10000
        
        # Adjust for volatility
        vol_multiplier = 1.0 + max(0, (volatility - 0.25) * 2)
        
        return base_slippage * vol_multiplier
    
    def calculate_market_impact(self,
                                trade_value: float,
                                avg_daily_volume: float,
                                volatility: float) -> float:
        """
        Calculate market impact cost using Almgren-Chriss model.
        
        Market impact increases with:
        - Trade size relative to ADV
        - Volatility
        
        Formula: impact = eta * sigma * (V / ADV)^gamma
        
        Where:
        - eta: impact coefficient
        - sigma: volatility
        - V: trade value
        - ADV: average daily volume
        - gamma: volume exponent (typically 0.5-0.6)
        """
        if avg_daily_volume <= 0:
            avg_daily_volume = self.default_adv
        
        # Trade size as fraction of ADV
        participation_rate = trade_value / avg_daily_volume
        
        # Almgren-Chriss temporary impact
        impact = (
            self.IMPACT_ETA * 
            volatility * 
            (participation_rate ** self.IMPACT_GAMMA)
        )
        
        # Cap at reasonable maximum (10%)
        return min(0.10, impact)
    
    def calculate_timing_cost(self,
                              base_cost: float,
                              trade_timing: TradeTiming) -> float:
        """
        Calculate additional cost from trade timing.
        
        Open and close are more expensive due to:
        - Higher volatility
        - Wider spreads
        - More informed trading
        """
        multiplier = self.TIMING_MULTIPLIER[trade_timing]
        return base_cost * (multiplier - 1.0)  # Additional cost only
    
    def estimate_cost(self,
                      ticker: str,
                      trade_value: float,
                      direction: str = 'BUY',
                      market_cap: float = None,
                      avg_daily_volume: float = None,
                      volatility: float = None,
                      trade_time: datetime = None) -> TransactionCost:
        """
        Estimate total transaction cost for a trade.
        
        Args:
            ticker: Stock symbol
            trade_value: Dollar value of trade
            direction: 'BUY' or 'SELL'
            market_cap: Market capitalization (optional)
            avg_daily_volume: Average daily dollar volume (optional)
            volatility: Annualized volatility (optional)
            trade_time: Time of trade (optional)
            
        Returns:
            TransactionCost with complete breakdown
        """
        # Use defaults if not provided
        volatility = volatility or self.default_volatility
        avg_daily_volume = avg_daily_volume or self.default_adv
        
        # Classify market cap
        if market_cap:
            market_cap_tier = self.classify_market_cap(market_cap)
        else:
            # Assume large-cap if unknown
            market_cap_tier = MarketCapTier.LARGE
        
        # Determine trade timing
        trade_timing = self.get_trade_timing(trade_time)
        
        # Calculate components
        spread_cost = self.calculate_spread_cost(market_cap_tier, volatility)
        slippage_cost = self.calculate_slippage_cost(market_cap_tier, volatility)
        market_impact = self.calculate_market_impact(
            trade_value, avg_daily_volume, volatility
        )
        
        # Base cost before timing
        base_cost = spread_cost + slippage_cost + market_impact
        
        # Timing cost
        timing_cost = self.calculate_timing_cost(base_cost, trade_timing)
        
        # Total
        total_cost_decimal = spread_cost + slippage_cost + market_impact + timing_cost
        total_cost_pct = total_cost_decimal * 100
        total_cost_bps = total_cost_decimal * 10000
        total_cost_dollars = trade_value * total_cost_decimal
        
        # Trade as % of ADV
        trade_pct_of_adv = (trade_value / avg_daily_volume * 100) if avg_daily_volume > 0 else 0
        
        return TransactionCost(
            ticker=ticker,
            trade_value=trade_value,
            direction=direction,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            market_impact_cost=market_impact,
            timing_cost=timing_cost,
            total_cost_pct=total_cost_pct,
            total_cost_bps=total_cost_bps,
            total_cost_dollars=total_cost_dollars,
            market_cap_tier=market_cap_tier.value,
            avg_daily_volume=avg_daily_volume,
            trade_pct_of_adv=trade_pct_of_adv,
            volatility=volatility,
            trade_time=trade_timing.value
        )
    
    def estimate_round_trip_cost(self,
                                 ticker: str,
                                 trade_value: float,
                                 **kwargs) -> float:
        """
        Estimate round-trip cost (buy + sell) in basis points.
        
        Useful for backtesting where we need total cost of a trade.
        """
        entry_cost = self.estimate_cost(ticker, trade_value, 'BUY', **kwargs)
        exit_cost = self.estimate_cost(ticker, trade_value, 'SELL', **kwargs)
        
        return entry_cost.total_cost_bps + exit_cost.total_cost_bps


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_cost_model = None

def get_cost_model() -> TransactionCostModel:
    """Get singleton cost model instance."""
    global _cost_model
    if _cost_model is None:
        _cost_model = TransactionCostModel()
    return _cost_model


def estimate_trade_cost(ticker: str,
                        trade_value: float,
                        market_cap: float = None,
                        avg_daily_volume: float = None,
                        volatility: float = None) -> TransactionCost:
    """Quick access to cost estimation."""
    return get_cost_model().estimate_cost(
        ticker, trade_value,
        market_cap=market_cap,
        avg_daily_volume=avg_daily_volume,
        volatility=volatility
    )


def get_round_trip_cost_bps(ticker: str,
                            trade_value: float,
                            market_cap: float = None) -> float:
    """Get round-trip cost in basis points."""
    return get_cost_model().estimate_round_trip_cost(
        ticker, trade_value, market_cap=market_cap
    )


# =============================================================================
# COST LOOKUP TABLE (for fast backtesting)
# =============================================================================

# Pre-computed costs by market cap tier (round-trip, in bps)
ROUND_TRIP_COST_TABLE = {
    'mega': 5,      # ~5 bps for mega-caps
    'large': 10,    # ~10 bps for large-caps
    'mid': 20,      # ~20 bps for mid-caps
    'small': 40,    # ~40 bps for small-caps
    'micro': 80,    # ~80 bps for micro-caps
    'default': 15,  # Default if unknown
}


def get_quick_cost_bps(market_cap_tier: str = 'default') -> float:
    """
    Get quick round-trip cost estimate from lookup table.
    
    Use this for fast backtesting when you don't need precision.
    """
    return ROUND_TRIP_COST_TABLE.get(market_cap_tier.lower(), 
                                      ROUND_TRIP_COST_TABLE['default'])


# =============================================================================
# BACKTEST INTEGRATION
# =============================================================================

def apply_transaction_costs(returns: np.ndarray,
                            trade_mask: np.ndarray,
                            cost_bps: float = 15) -> np.ndarray:
    """
    Apply transaction costs to a return series.
    
    Args:
        returns: Array of returns (as percentages, e.g., 2.5 for 2.5%)
        trade_mask: Boolean array indicating when trades occurred
        cost_bps: Round-trip cost in basis points
        
    Returns:
        Cost-adjusted returns array
    """
    cost_pct = cost_bps / 100  # Convert bps to percentage
    
    adjusted_returns = returns.copy()
    
    # Subtract cost whenever a trade occurred
    adjusted_returns[trade_mask] -= cost_pct
    
    return adjusted_returns


def adjust_backtest_returns(trades: list,
                            cost_model: TransactionCostModel = None) -> list:
    """
    Adjust trade returns for transaction costs.
    
    Args:
        trades: List of trade dicts with 'ticker', 'entry_price', 'exit_price', 'return_pct'
        cost_model: TransactionCostModel instance (uses default if None)
        
    Returns:
        Trades with 'return_pct_net' added
    """
    model = cost_model or get_cost_model()
    
    for trade in trades:
        # Estimate trade value (approximate)
        trade_value = trade.get('entry_price', 100) * 100  # Assume 100 shares
        
        # Get round-trip cost
        cost_bps = model.estimate_round_trip_cost(
            trade.get('ticker', 'UNKNOWN'),
            trade_value,
            market_cap=trade.get('market_cap'),
            volatility=trade.get('volatility')
        )
        
        # Adjust return
        raw_return = trade.get('return_pct', 0)
        cost_pct = cost_bps / 100
        trade['return_pct_net'] = raw_return - cost_pct
        trade['transaction_cost_bps'] = cost_bps
    
    return trades


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    model = TransactionCostModel()
    
    # Test mega-cap trade
    cost = model.estimate_cost(
        ticker="AAPL",
        trade_value=100_000,
        market_cap=3_000_000_000_000,  # $3T
        avg_daily_volume=500_000_000,   # $500M ADV
        volatility=0.25
    )
    
    print(f"\nMega-cap trade (AAPL):")
    print(f"  Trade value: ${cost.trade_value:,.0f}")
    print(f"  Spread: {cost.spread_cost * 10000:.1f} bps")
    print(f"  Slippage: {cost.slippage_cost * 10000:.1f} bps")
    print(f"  Market impact: {cost.market_impact_cost * 10000:.1f} bps")
    print(f"  Timing: {cost.timing_cost * 10000:.1f} bps")
    print(f"  TOTAL: {cost.total_cost_bps:.1f} bps (${cost.total_cost_dollars:.2f})")
    
    # Test small-cap trade
    cost2 = model.estimate_cost(
        ticker="XYZ",
        trade_value=50_000,
        market_cap=500_000_000,      # $500M
        avg_daily_volume=2_000_000,   # $2M ADV
        volatility=0.45
    )
    
    print(f"\nSmall-cap trade (XYZ):")
    print(f"  Trade value: ${cost2.trade_value:,.0f}")
    print(f"  TOTAL: {cost2.total_cost_bps:.1f} bps (${cost2.total_cost_dollars:.2f})")
    
    # Round-trip comparison
    print(f"\nRound-trip costs:")
    print(f"  AAPL: {model.estimate_round_trip_cost('AAPL', 100_000, market_cap=3e12):.1f} bps")
    print(f"  XYZ: {model.estimate_round_trip_cost('XYZ', 50_000, market_cap=500e6):.1f} bps")
