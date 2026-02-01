"""
HH Research Platform - Sukuk Signals (Phase 1)

Generates trading signals for USD sukuk using IBKR data.
Phase 1: Basic signals without YTM/duration calculations.

Key features:
- Liquidity gates (bid/ask spread, staleness)
- Portfolio constraints (issuer exposure, position limits)
- Maturity ladder bias
- Carry proxy using coupon rate
"""

from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from src.utils.logging import get_logger

try:
    from src.analytics.sukuk_models import (
        SukukInstrument, SukukUniverse, SukukLiveData, SukukSignal,
        Quote, DataQuality, SukukAction, MaturityBucket, RiskLimits
    )
except ImportError:
    from sukuk_models import (
        SukukInstrument, SukukUniverse, SukukLiveData, SukukSignal,
        Quote, DataQuality, SukukAction, MaturityBucket, RiskLimits
    )

logger = get_logger(__name__)


@dataclass
class RatesRegime:
    """
    Current rates environment for context.
    Passed from the main bond analytics module.
    """
    curve_shape: str  # "NORMAL", "FLAT", "INVERTED"
    rates_level: str  # "LOW", "NORMAL", "HIGH"
    fed_stance: str   # "HAWKISH", "NEUTRAL", "DOVISH"
    spread_10y_2y: float
    yield_10y: float
    fed_funds_rate: float


@dataclass
class PortfolioContext:
    """
    Current portfolio holdings for constraint checking.
    """
    # Current sukuk weights by ISIN
    sukuk_weights: Dict[str, float]  # ISIN -> weight %

    # Issuer exposure by bucket
    issuer_exposure: Dict[str, float]  # issuer_bucket -> total weight %

    # Total sukuk allocation
    total_sukuk_pct: float

    # Total portfolio value (for sizing)
    portfolio_value: float

    @classmethod
    def empty(cls) -> 'PortfolioContext':
        """Create empty portfolio context."""
        return cls(
            sukuk_weights={},
            issuer_exposure={},
            total_sukuk_pct=0.0,
            portfolio_value=0.0
        )


class SukukSignalGenerator:
    """
    Generates Phase 1 trading signals for sukuk.

    Signal Logic (Phase 1):
    - BUY: Good liquidity, under position limits, favorable regime
    - HOLD: Already at/near limit, or mixed signals
    - WATCH: Poor liquidity, stale data, or unfavorable conditions
    - AVOID: Matured, no data, or violated hard limits
    """

    def __init__(self, risk_limits: RiskLimits):
        """
        Initialize signal generator.

        Args:
            risk_limits: Portfolio risk constraints
        """
        self.risk_limits = risk_limits

    def generate_signals(self,
                        live_data: Dict[int, SukukLiveData],
                        rates_regime: Optional[RatesRegime] = None,
                        portfolio: Optional[PortfolioContext] = None
                        ) -> List[SukukSignal]:
        """
        Generate signals for all sukuk with live data.

        Args:
            live_data: Dict of conid -> SukukLiveData
            rates_regime: Current rates environment
            portfolio: Current portfolio holdings

        Returns:
            List of SukukSignal
        """
        if portfolio is None:
            portfolio = PortfolioContext.empty()

        signals = []

        for conid, data in live_data.items():
            signal = self._generate_single_signal(data, rates_regime, portfolio)
            signals.append(signal)

        # Sort by conviction (highest first), then by action priority
        action_priority = {
            SukukAction.BUY: 0,
            SukukAction.HOLD: 1,
            SukukAction.WATCH: 2,
            SukukAction.AVOID: 3
        }

        signals.sort(key=lambda s: (action_priority[s.action], -s.conviction))

        return signals

    def _generate_single_signal(self,
                                data: SukukLiveData,
                                rates_regime: Optional[RatesRegime],
                                portfolio: PortfolioContext
                                ) -> SukukSignal:
        """
        Generate signal for a single sukuk.
        """
        instrument = data.instrument

        # =====================================================================
        # HARD GATES - These result in AVOID or WATCH
        # =====================================================================

        # Gate 1: Matured
        if instrument.is_matured:
            return SukukSignal.watch_signal(
                instrument,
                f"MATURED on {instrument.maturity}",
                data
            )

        # Gate 2: No valid conid
        if instrument.conid <= 0:
            return SukukSignal.watch_signal(
                instrument,
                "Missing IBKR conid - update universe JSON",
                data
            )

        # Gate 3: No valid price
        if data.data_quality == DataQuality.MISSING or not data.quote.has_valid_price:
            return SukukSignal.watch_signal(
                instrument,
                "No valid price data from IBKR",
                data
            )

        # Gate 4: Stale data
        if data.quote.stale_seconds and data.quote.stale_seconds > self.risk_limits.max_stale_seconds:
            return SukukSignal(
                instrument=instrument,
                live_data=data,
                action=SukukAction.WATCH,
                conviction=10,
                size_cap_pct=0,
                reason=f"Stale data ({data.quote.stale_seconds}s > {self.risk_limits.max_stale_seconds}s limit)",
                data_quality=DataQuality.STALE,
                price=data.price_pct,
                carry_proxy=data.carry_proxy,
                ttm_years=data.ttm_years,
                bid_ask_bps=data.bid_ask_bps,
            )

        # Gate 5: Wide spread (illiquid)
        if data.bid_ask_bps and data.bid_ask_bps > self.risk_limits.min_bid_ask_bps:
            return SukukSignal(
                instrument=instrument,
                live_data=data,
                action=SukukAction.WATCH,
                conviction=20,
                size_cap_pct=0,
                reason=f"Wide spread ({data.bid_ask_bps:.0f} bps > {self.risk_limits.min_bid_ask_bps} limit)",
                data_quality=DataQuality.DEGRADED,
                price=data.price_pct,
                carry_proxy=data.carry_proxy,
                ttm_years=data.ttm_years,
                bid_ask_bps=data.bid_ask_bps,
            )

        # =====================================================================
        # PORTFOLIO CONSTRAINT CHECKS
        # =====================================================================

        # Current position weight
        current_weight = portfolio.sukuk_weights.get(instrument.isin, 0.0)

        # Issuer exposure
        issuer_exposure = portfolio.issuer_exposure.get(instrument.issuer_bucket, 0.0)

        # Check single position limit
        at_position_limit = current_weight >= self.risk_limits.max_single_position_pct

        # Check issuer limit
        at_issuer_limit = issuer_exposure >= self.risk_limits.max_issuer_pct

        # Check total sukuk limit
        at_total_limit = portfolio.total_sukuk_pct >= self.risk_limits.max_total_sukuk_pct

        # Calculate remaining capacity
        position_room = max(0, self.risk_limits.max_single_position_pct - current_weight)
        issuer_room = max(0, self.risk_limits.max_issuer_pct - issuer_exposure)
        total_room = max(0, self.risk_limits.max_total_sukuk_pct - portfolio.total_sukuk_pct)

        # Effective size cap is minimum of all constraints
        size_cap = min(position_room, issuer_room, total_room, instrument.weight * 100)

        # =====================================================================
        # SIGNAL LOGIC
        # =====================================================================

        # Start with base conviction from data quality
        conviction = self._base_conviction(data)

        # Adjust for rates regime
        regime_adj = self._regime_adjustment(rates_regime, instrument.maturity_bucket)
        conviction += regime_adj

        # Adjust for carry proxy (higher coupon = more attractive)
        carry_adj = self._carry_adjustment(data.carry_proxy)
        conviction += carry_adj

        # Adjust for maturity bucket preference
        maturity_adj = self._maturity_adjustment(instrument.maturity_bucket, rates_regime)
        conviction += maturity_adj

        # Cap conviction at 100
        conviction = min(100, max(0, conviction))

        # Determine action
        if at_position_limit or at_issuer_limit or at_total_limit:
            # At limit - HOLD
            limit_reason = []
            if at_position_limit:
                limit_reason.append(f"position limit ({current_weight:.1f}%)")
            if at_issuer_limit:
                limit_reason.append(f"issuer limit ({issuer_exposure:.1f}%)")
            if at_total_limit:
                limit_reason.append(f"total sukuk limit ({portfolio.total_sukuk_pct:.1f}%)")

            return SukukSignal(
                instrument=instrument,
                live_data=data,
                action=SukukAction.HOLD,
                conviction=conviction,
                size_cap_pct=0,
                reason=f"At {', '.join(limit_reason)}",
                data_quality=data.data_quality,
                price=data.price_pct,
                carry_proxy=data.carry_proxy,
                ttm_years=data.ttm_years,
                bid_ask_bps=data.bid_ask_bps,
                current_weight_pct=current_weight,
                issuer_exposure_pct=issuer_exposure,
            )

        elif conviction >= 60 and size_cap > 0:
            # Strong signal with room - BUY
            return SukukSignal(
                instrument=instrument,
                live_data=data,
                action=SukukAction.BUY,
                conviction=conviction,
                size_cap_pct=size_cap,
                reason=self._buy_reason(data, rates_regime, conviction),
                data_quality=data.data_quality,
                price=data.price_pct,
                carry_proxy=data.carry_proxy,
                ttm_years=data.ttm_years,
                bid_ask_bps=data.bid_ask_bps,
                current_weight_pct=current_weight,
                issuer_exposure_pct=issuer_exposure,
            )

        elif conviction >= 40:
            # Moderate signal - HOLD/WATCH
            return SukukSignal(
                instrument=instrument,
                live_data=data,
                action=SukukAction.HOLD,
                conviction=conviction,
                size_cap_pct=size_cap,
                reason=f"Moderate conviction ({conviction}%) - monitoring",
                data_quality=data.data_quality,
                price=data.price_pct,
                carry_proxy=data.carry_proxy,
                ttm_years=data.ttm_years,
                bid_ask_bps=data.bid_ask_bps,
                current_weight_pct=current_weight,
                issuer_exposure_pct=issuer_exposure,
            )

        else:
            # Weak signal - WATCH
            return SukukSignal(
                instrument=instrument,
                live_data=data,
                action=SukukAction.WATCH,
                conviction=conviction,
                size_cap_pct=0,
                reason=f"Low conviction ({conviction}%) - wait for better conditions",
                data_quality=data.data_quality,
                price=data.price_pct,
                carry_proxy=data.carry_proxy,
                ttm_years=data.ttm_years,
                bid_ask_bps=data.bid_ask_bps,
                current_weight_pct=current_weight,
                issuer_exposure_pct=issuer_exposure,
            )

    def _base_conviction(self, data: SukukLiveData) -> int:
        """Base conviction from data quality."""
        if data.data_quality == DataQuality.OK:
            return 50
        elif data.data_quality == DataQuality.DEGRADED:
            return 35
        elif data.data_quality == DataQuality.STALE:
            return 20
        else:
            return 10

    def _regime_adjustment(self, regime: Optional[RatesRegime],
                          maturity_bucket: MaturityBucket) -> int:
        """Adjust conviction based on rates regime."""
        if regime is None:
            return 0

        adj = 0

        # Dovish Fed is good for bonds
        if regime.fed_stance == "DOVISH":
            adj += 15
        elif regime.fed_stance == "HAWKISH":
            adj -= 10

        # Normal curve is favorable
        if regime.curve_shape == "NORMAL":
            adj += 5
        elif regime.curve_shape == "INVERTED":
            # Inverted curve hurts long duration more
            if maturity_bucket in [MaturityBucket.LONG, MaturityBucket.ULTRA]:
                adj -= 15
            else:
                adj -= 5

        return adj

    def _carry_adjustment(self, coupon: float) -> int:
        """Adjust conviction based on carry proxy (coupon)."""
        # Higher coupon = better carry
        if coupon >= 5.0:
            return 10
        elif coupon >= 4.0:
            return 5
        elif coupon >= 3.0:
            return 0
        else:
            return -5

    def _maturity_adjustment(self, bucket: MaturityBucket,
                            regime: Optional[RatesRegime]) -> int:
        """Adjust conviction based on maturity bucket."""
        # Phase 1: Prefer shorter maturities unless regime is clearly bullish
        is_bullish = regime and regime.fed_stance == "DOVISH"

        if bucket == MaturityBucket.SHORT:
            return 10  # Always prefer short for safety
        elif bucket == MaturityBucket.MEDIUM:
            return 5
        elif bucket == MaturityBucket.LONG:
            return 5 if is_bullish else -5
        else:  # ULTRA
            return 10 if is_bullish else -10

    def _buy_reason(self, data: SukukLiveData,
                   regime: Optional[RatesRegime],
                   conviction: int) -> str:
        """Generate human-readable buy reason."""
        reasons = []

        # Liquidity
        if data.bid_ask_bps and data.bid_ask_bps < 100:
            reasons.append("tight spread")
        elif data.bid_ask_bps and data.bid_ask_bps < 150:
            reasons.append("acceptable liquidity")

        # Carry
        if data.carry_proxy >= 5.0:
            reasons.append(f"attractive carry ({data.carry_proxy:.2f}%)")
        elif data.carry_proxy >= 4.0:
            reasons.append(f"decent carry ({data.carry_proxy:.2f}%)")

        # Regime
        if regime and regime.fed_stance == "DOVISH":
            reasons.append("dovish Fed stance")

        # Maturity
        bucket = data.instrument.maturity_bucket
        if bucket == MaturityBucket.SHORT:
            reasons.append("short duration (lower risk)")
        elif bucket == MaturityBucket.MEDIUM:
            reasons.append("medium duration")

        if reasons:
            return f"{conviction}% conviction: {', '.join(reasons)}"
        return f"{conviction}% conviction"


# =============================================================================
# NUMERIC BLOCK BUILDER FOR LLM
# =============================================================================

def build_sukuk_numeric_block(signals: List[SukukSignal],
                              rates_regime: Optional[RatesRegime] = None
                              ) -> str:
    """
    Build the numeric block for sukuk to inject into AI context.

    This follows the same pattern as the bond ETF numeric block.
    """
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("SUKUK DATA BLOCK - DO NOT EDIT OR REFORMAT")
    lines.append("=" * 60)
    lines.append("")

    # Summary by action
    buy_count = sum(1 for s in signals if s.action == SukukAction.BUY)
    hold_count = sum(1 for s in signals if s.action == SukukAction.HOLD)
    watch_count = sum(1 for s in signals if s.action == SukukAction.WATCH)

    lines.append(f"SUKUK SIGNALS SUMMARY:")
    lines.append(f"  BUY: {buy_count} | HOLD: {hold_count} | WATCH: {watch_count}")
    lines.append("")

    # Detail for each signal
    lines.append("INDIVIDUAL SUKUK:")
    for sig in signals:
        inst = sig.instrument

        # Price display
        price_str = f"${sig.price:.2f}" if sig.price else "N/A"
        spread_str = f"{sig.bid_ask_bps:.0f}bps" if sig.bid_ask_bps else "N/A"

        # Get yield from live_data if available
        yield_str = "N/A"
        if sig.live_data and sig.live_data.ask_yield:
            yield_str = f"{sig.live_data.ask_yield:.2f}%"
        elif inst.cached_yield:
            yield_str = f"{inst.cached_yield:.2f}%"

        lines.append(f"  {inst.name}:")
        lines.append(f"    ISIN: {inst.isin}")
        lines.append(f"    Issuer: {inst.issuer} ({inst.issuer_bucket})")
        lines.append(f"    Maturity: {inst.maturity} ({sig.ttm_years:.1f}y)")
        lines.append(f"    Coupon: {inst.coupon_rate_pct:.2f}%")
        lines.append(f"    Price: {price_str} | Yield: {yield_str} | Spread: {spread_str}")
        lines.append(f"    Signal: {sig.action.value} ({sig.conviction}% conviction)")
        lines.append(f"    Size Cap: {sig.size_cap_pct:.1f}%")
        lines.append(f"    Reason: {sig.reason}")
        lines.append(f"    Data Quality: {sig.data_quality.value}")

        if sig.current_weight_pct > 0:
            lines.append(f"    Current Weight: {sig.current_weight_pct:.1f}%")
        if sig.issuer_exposure_pct > 0:
            lines.append(f"    Issuer Exposure: {sig.issuer_exposure_pct:.1f}%")
        lines.append("")

    # Rates context if available
    if rates_regime:
        lines.append("RATES CONTEXT:")
        lines.append(f"  Curve Shape: {rates_regime.curve_shape}")
        lines.append(f"  Fed Stance: {rates_regime.fed_stance}")
        lines.append(f"  10Y Yield: {rates_regime.yield_10y:.2f}%")
        lines.append(f"  10Y-2Y Spread: {rates_regime.spread_10y_2y:+.2f}%")
        lines.append("")

    lines.append("=" * 60)
    lines.append("END SUKUK DATA BLOCK")
    lines.append("=" * 60)

    return "\n".join(lines)


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

def generate_sukuk_analysis(
    universe_path: str,
    rates_regime: Optional[RatesRegime] = None,
    portfolio: Optional[PortfolioContext] = None,
    ibkr_host: str = "127.0.0.1",
    ibkr_port: int = 7496
) -> Dict[str, Any]:
    """
    High-level function to generate complete sukuk analysis.

    Args:
        universe_path: Path to sukuk_universe_usd.json
        rates_regime: Current rates environment
        portfolio: Current portfolio holdings
        ibkr_host: IBKR TWS host
        ibkr_port: IBKR TWS port

    Returns:
        Dict with signals, numeric_block, warnings
    """
    try:
        from src.analytics.sukuk_data import load_sukuk_universe, fetch_sukuk_market_data
    except ImportError:
        from sukuk_data import load_sukuk_universe, fetch_sukuk_market_data

    # Load universe
    universe = load_sukuk_universe(universe_path)
    logger.info(f"Loaded {len(universe.instruments)} sukuk, "
               f"{len(universe.active_instruments)} active")

    # Fetch market data
    live_data, fetch_warnings = fetch_sukuk_market_data(
        universe,
        host=ibkr_host,
        port=ibkr_port
    )

    # Generate signals
    generator = SukukSignalGenerator(universe.risk_limits)
    signals = generator.generate_signals(live_data, rates_regime, portfolio)

    # Build numeric block
    numeric_block = build_sukuk_numeric_block(signals, rates_regime)

    return {
        'signals': signals,
        'universe': universe,
        'live_data': live_data,
        'numeric_block': numeric_block,
        'warnings': fetch_warnings,
        'generated_at': datetime.now().isoformat(),
    }