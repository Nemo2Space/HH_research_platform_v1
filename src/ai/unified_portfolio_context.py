import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Options flow integration (optional - graceful fallback if not available)
try:
    from src.analytics.options_flow import OptionsFlowAnalyzer

    OPTIONS_FLOW_AVAILABLE = True
except ImportError:
    OPTIONS_FLOW_AVAILABLE = False


@dataclass
class ActualHolding:
    """Actual position from IBKR"""
    symbol: str
    shares: float = 0
    value: float = 0
    weight: float = 0
    avg_cost: float = 0
    unrealized_pnl: float = 0
    open_order_shares: float = 0  # Pending buy/sell
    open_order_action: str = ""  # BUY or SELL


@dataclass
class TargetHolding:
    """Target position from ETF Creator"""
    symbol: str
    target_weight: float = 0
    reasoning: str = ""
    scores: Dict[str, Any] = field(default_factory=dict)
    sector: str = ""
    risks: List[str] = field(default_factory=list)


@dataclass
class RebalanceStatus:
    """Last rebalance status from Rebalancer"""
    symbol: str
    action: str = ""  # BUY, SELL, HOLD
    target_quantity: int = 0
    executed_quantity: int = 0
    status: str = ""  # FILLED, PARTIAL, FAILED, PENDING
    execution_accuracy: float = 0
    rebalance_date: str = ""


@dataclass
class UnifiedHolding:
    """Unified view of a holding from all sources"""
    symbol: str
    name: str = ""
    sector: str = ""

    # Actual (IBKR)
    actual_shares: float = 0
    actual_value: float = 0
    actual_weight: float = 0
    avg_cost: float = 0
    unrealized_pnl: float = 0
    open_order_shares: float = 0
    open_order_action: str = ""

    # Target (ETF Creator)
    target_weight: float = 0
    selection_reasoning: str = ""
    scores: Dict[str, Any] = field(default_factory=dict)
    risks: List[str] = field(default_factory=list)

    # Last Rebalance
    last_action: str = ""
    last_status: str = ""
    last_execution_accuracy: float = 0
    last_rebalance_date: str = ""

    # Calculated
    drift: float = 0  # actual_weight - target_weight
    holding_status: str = ""  # "✓ Held", "⚠️ Underweight", "❌ Not held", "⚠️ Not in target"

    # Source flags
    in_ibkr: bool = False
    in_etf_creator: bool = False
    in_rebalancer: bool = False


@dataclass
class UnifiedPortfolioSummary:
    """Summary of the unified portfolio"""
    total_value: float = 0
    total_holdings_actual: int = 0
    total_holdings_target: int = 0
    last_rebalance_date: str = ""
    last_rebalance_accuracy: float = 0

    # Calculated
    holdings_on_target: int = 0  # In both IBKR and ETF Creator
    holdings_not_held: int = 0  # In ETF Creator but not IBKR
    holdings_extra: int = 0  # In IBKR but not ETF Creator

    total_drift: float = 0  # Sum of absolute drifts
    avg_drift: float = 0
    max_drift_symbol: str = ""
    max_drift_value: float = 0

    sector_allocations: Dict[str, Dict[str, float]] = field(default_factory=dict)


class UnifiedPortfolioLoader:
    """
    Loads and merges portfolio data from all three sources.

    Usage:
        loader = UnifiedPortfolioLoader()
        context = loader.get_unified_context(ibkr_positions, account_summary)
    """

    def __init__(self,
                 etf_creator_dir: str = None,
                 rebalancer_dir: str = None):
        """Initialize with paths to JSON directories."""
        self.etf_creator_dir = etf_creator_dir or os.getenv(
            'ETF_CREATOR_JSON_DIR',
            r'C:\Develop\Latest_2025\create_etf_v5\json'
        )
        self.rebalancer_dir = rebalancer_dir or os.getenv(
            'REBALANCER_JSON_DIR',
            r'C:\Develop\Latest_2025\rebalancing_portfolio_website_v45_allWorks\json'
        )

        self._etf_creator_data = None
        self._rebalancer_data = None
        self._unified_holdings: Dict[str, UnifiedHolding] = {}
        self._summary: UnifiedPortfolioSummary = None

    def _find_latest_file(self, directory: str, prefix: str) -> Optional[str]:
        """Find the latest JSON file matching prefix."""
        try:
            if not os.path.exists(directory):
                logger.warning(f"Directory not found: {directory}")
                return None

            # First try *_latest.json
            for f in os.listdir(directory):
                if f.startswith(prefix) and f.endswith('_latest.json'):
                    return os.path.join(directory, f)

            # Fall back to most recent timestamped file
            files = []
            for f in os.listdir(directory):
                if f.startswith(prefix) and f.endswith('.json') and '_latest' not in f:
                    files.append(f)

            if files:
                files.sort(reverse=True)
                return os.path.join(directory, files[0])

            return None

        except Exception as e:
            logger.error(f"Error finding latest file: {e}")
            return None

    def _load_json_safe(self, filepath: str) -> Optional[Dict]:
        """Safely load JSON file."""
        try:
            if not filepath or not os.path.exists(filepath):
                return None

            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Error loading JSON {filepath}: {e}")
            return None

    def _load_etf_creator(self) -> Dict[str, TargetHolding]:
        """Load ETF Creator data and convert to TargetHolding dict."""
        targets = {}

        for prefix in ['portfolio_SC', 'portfolio_', 'etf_']:
            filepath = self._find_latest_file(self.etf_creator_dir, prefix)
            if filepath:
                data = self._load_json_safe(filepath)
                if data:
                    self._etf_creator_data = data
                    logger.info(f"Loaded ETF Creator from {filepath}")
                    break

        if not self._etf_creator_data:
            return targets

        holdings = self._etf_creator_data.get('holdings', [])
        for h in holdings:
            symbol = h.get('symbol', '').upper()
            if not symbol:
                continue

            targets[symbol] = TargetHolding(
                symbol=symbol,
                target_weight=h.get('weight', 0),
                reasoning=h.get('reasoning', ''),
                scores=h.get('scores', {}),
                sector=h.get('sector', ''),
                risks=h.get('risks', [])
            )

        return targets

    def _load_rebalancer(self) -> Dict[str, RebalanceStatus]:
        """Load Rebalancer data and convert to RebalanceStatus dict."""
        statuses = {}

        for prefix in ['rebalance_', 'rebal_']:
            filepath = self._find_latest_file(self.rebalancer_dir, prefix)
            if filepath:
                data = self._load_json_safe(filepath)
                if data:
                    self._rebalancer_data = data
                    logger.info(f"Loaded Rebalancer from {filepath}")
                    break

        if not self._rebalancer_data:
            return statuses

        # Get rebalance date from metadata
        rebalance_date = self._rebalancer_data.get('metadata', {}).get('generated_at', '')[:10]

        holdings = self._rebalancer_data.get('holdings', [])
        for h in holdings:
            symbol = h.get('symbol', '').upper()
            if not symbol:
                continue

            trade = h.get('trade', {})
            statuses[symbol] = RebalanceStatus(
                symbol=symbol,
                action=trade.get('action', ''),
                target_quantity=trade.get('target_quantity', 0),
                executed_quantity=trade.get('executed_quantity', 0),
                status=trade.get('status', ''),
                execution_accuracy=trade.get('execution_accuracy', 0),
                rebalance_date=rebalance_date
            )

        return statuses

    def _parse_ibkr_positions(self, positions: List[Dict],
                              account_summary: Optional[Dict] = None) -> Tuple[Dict[str, ActualHolding], float]:
        """Parse IBKR positions into ActualHolding dict."""
        actuals = {}
        total_value = 0

        if not positions:
            return actuals, total_value

        # Calculate total portfolio value
        for pos in positions:
            # Handle multiple possible field names
            value = (
                    pos.get('market_value', 0) or  # IBKR format
                    pos.get('marketValue', 0) or  # Alternative format
                    pos.get('value', 0) or
                    0
            )
            total_value += value

        # Override with account summary if available
        if account_summary:
            total_value = account_summary.get('net_liquidation', total_value) or total_value

        logger.info(f"Total portfolio value: ${total_value:,.2f}")

        # Parse each position
        for pos in positions:
            symbol = pos.get('symbol', '').upper()
            if not symbol:
                continue

            # Handle multiple possible field names for each field
            value = (
                    pos.get('market_value', 0) or
                    pos.get('marketValue', 0) or
                    pos.get('value', 0) or
                    0
            )

            shares = (
                    pos.get('quantity', 0) or  # IBKR format
                    pos.get('position', 0) or  # Alternative format
                    pos.get('shares', 0) or
                    0
            )

            avg_cost = (
                    pos.get('avg_cost', 0) or  # IBKR format
                    pos.get('avgCost', 0) or  # Alternative format
                    pos.get('averageCost', 0) or
                    0
            )

            unrealized_pnl = (
                    pos.get('unrealized_pnl', 0) or  # IBKR format
                    pos.get('unrealizedPNL', 0) or  # Alternative format
                    pos.get('unrealizedPnl', 0) or
                    0
            )

            weight = (value / total_value * 100) if total_value > 0 else 0

            actuals[symbol] = ActualHolding(
                symbol=symbol,
                shares=shares,
                value=value,
                weight=weight,
                avg_cost=avg_cost,
                unrealized_pnl=unrealized_pnl,
                open_order_shares=0,
                open_order_action=""
            )

        logger.info(f"Parsed {len(actuals)} IBKR positions")
        return actuals, total_value

    def _merge_holdings(self,
                        actuals: Dict[str, ActualHolding],
                        targets: Dict[str, TargetHolding],
                        rebalances: Dict[str, RebalanceStatus]) -> Dict[str, UnifiedHolding]:
        """Merge all three sources into unified holdings."""
        unified = {}

        # Get all unique symbols
        all_symbols = set(actuals.keys()) | set(targets.keys()) | set(rebalances.keys())

        for symbol in all_symbols:
            actual = actuals.get(symbol)
            target = targets.get(symbol)
            rebalance = rebalances.get(symbol)

            # Create unified holding
            holding = UnifiedHolding(symbol=symbol)

            # Fill from IBKR (actual)
            if actual:
                holding.in_ibkr = True
                holding.actual_shares = actual.shares
                holding.actual_value = actual.value
                holding.actual_weight = actual.weight
                holding.avg_cost = actual.avg_cost
                holding.unrealized_pnl = actual.unrealized_pnl
                holding.open_order_shares = actual.open_order_shares
                holding.open_order_action = actual.open_order_action

            # Fill from ETF Creator (target)
            if target:
                holding.in_etf_creator = True
                holding.target_weight = target.target_weight
                holding.selection_reasoning = target.reasoning
                holding.scores = target.scores
                holding.sector = target.sector
                holding.risks = target.risks

            # Fill from Rebalancer
            if rebalance:
                holding.in_rebalancer = True
                holding.last_action = rebalance.action
                holding.last_status = rebalance.status
                holding.last_execution_accuracy = rebalance.execution_accuracy
                holding.last_rebalance_date = rebalance.rebalance_date

            # Calculate drift
            holding.drift = holding.actual_weight - holding.target_weight

            # Determine holding status
            if holding.in_ibkr and holding.in_etf_creator:
                if abs(holding.drift) < 0.5:
                    holding.holding_status = "✓ On target"
                elif holding.drift < 0:
                    holding.holding_status = f"⚠️ Underweight ({holding.drift:+.1f}%)"
                else:
                    holding.holding_status = f"⚠️ Overweight ({holding.drift:+.1f}%)"
            elif holding.in_ibkr and not holding.in_etf_creator:
                holding.holding_status = "⚠️ Not in target (consider selling)"
            elif not holding.in_ibkr and holding.in_etf_creator:
                holding.holding_status = f"❌ Not held (target: {holding.target_weight:.1f}%)"
            else:
                holding.holding_status = "? Unknown"

            unified[symbol] = holding

        return unified

    def _calculate_summary(self, unified: Dict[str, UnifiedHolding],
                           total_value: float) -> UnifiedPortfolioSummary:
        """Calculate portfolio summary from unified holdings."""
        summary = UnifiedPortfolioSummary(
            total_value=total_value,
            sector_allocations={}
        )

        # Count holdings
        for symbol, holding in unified.items():
            if holding.in_ibkr:
                summary.total_holdings_actual += 1
            if holding.in_etf_creator:
                summary.total_holdings_target += 1

            if holding.in_ibkr and holding.in_etf_creator:
                summary.holdings_on_target += 1
            elif holding.in_etf_creator and not holding.in_ibkr:
                summary.holdings_not_held += 1
            elif holding.in_ibkr and not holding.in_etf_creator:
                summary.holdings_extra += 1

            # Track drift
            abs_drift = abs(holding.drift)
            summary.total_drift += abs_drift
            if abs_drift > abs(summary.max_drift_value):
                summary.max_drift_value = holding.drift
                summary.max_drift_symbol = symbol

            # Track sector allocations
            sector = holding.sector or "Unknown"
            if sector not in summary.sector_allocations:
                summary.sector_allocations[sector] = {"actual": 0, "target": 0}
            summary.sector_allocations[sector]["actual"] += holding.actual_weight
            summary.sector_allocations[sector]["target"] += holding.target_weight

        # Calculate averages
        if unified:
            summary.avg_drift = summary.total_drift / len(unified)

        # Get rebalance info from rebalancer data
        if self._rebalancer_data:
            meta = self._rebalancer_data.get('metadata', {})
            summary.last_rebalance_date = meta.get('generated_at', '')[:10] if meta.get('generated_at') else ''

            sum_data = self._rebalancer_data.get('summary', {})
            summary.last_rebalance_accuracy = sum_data.get('average_accuracy', 0)

        return summary

    def load_unified_portfolio(self,
                               ibkr_positions: List[Dict] = None,
                               account_summary: Dict = None) -> Tuple[
        Dict[str, UnifiedHolding], UnifiedPortfolioSummary]:
        """
        Load and merge all portfolio sources.

        Args:
            ibkr_positions: List of position dicts from IBKR
            account_summary: Account summary dict from IBKR

        Returns:
            Tuple of (unified_holdings, summary)
        """
        # Load all sources
        targets = self._load_etf_creator()
        rebalances = self._load_rebalancer()
        actuals, total_value = self._parse_ibkr_positions(ibkr_positions or [], account_summary)

        logger.info(f"Loaded: {len(actuals)} actual, {len(targets)} target, {len(rebalances)} rebalance")

        # Merge
        self._unified_holdings = self._merge_holdings(actuals, targets, rebalances)

        # Calculate summary
        self._summary = self._calculate_summary(self._unified_holdings, total_value)

        return self._unified_holdings, self._summary

    def get_holding_context(self, symbol: str) -> str:
        """Get detailed context for a specific holding."""
        symbol = symbol.upper()

        if symbol not in self._unified_holdings:
            return f"No data found for {symbol} in portfolio."

        h = self._unified_holdings[symbol]
        lines = [f"\n{'=' * 50}", f"{symbol} - UNIFIED PORTFOLIO VIEW", f"{'=' * 50}"]

        # Status
        lines.append(f"\nSTATUS: {h.holding_status}")

        # Actual position
        lines.append(f"\n📊 ACTUAL (Live from IBKR):")
        if h.in_ibkr:
            lines.append(f"   Shares: {h.actual_shares:,.0f}")
            lines.append(f"   Value: ${h.actual_value:,.2f}")
            lines.append(f"   Weight: {h.actual_weight:.2f}%")
            lines.append(f"   Avg Cost: ${h.avg_cost:.2f}")
            lines.append(f"   Unrealized P&L: ${h.unrealized_pnl:+,.2f}")
            if h.open_order_shares > 0:
                lines.append(f"   Open Order: {h.open_order_action} {h.open_order_shares:,.0f} shares")
        else:
            lines.append("   Not currently held")

        # Target
        lines.append(f"\n🎯 TARGET (From ETF Creator):")
        if h.in_etf_creator:
            lines.append(f"   Target Weight: {h.target_weight:.2f}%")
            lines.append(f"   Sector: {h.sector}")
            if h.selection_reasoning:
                lines.append(f"   Selection Reason: {h.selection_reasoning}")
            if h.scores:
                score_str = ", ".join([f"{k}={v}" for k, v in h.scores.items() if v])
                if score_str:
                    lines.append(f"   Scores: {score_str}")
            if h.risks:
                lines.append(f"   Risks: {'; '.join(h.risks[:2])}")
        else:
            lines.append("   Not in target portfolio")

        # Drift
        lines.append(f"\n📐 DRIFT:")
        lines.append(f"   Actual vs Target: {h.drift:+.2f}%")
        if h.in_ibkr and h.in_etf_creator:
            if abs(h.drift) < 0.5:
                lines.append("   Assessment: Well aligned")
            elif h.drift < -1:
                lines.append("   Assessment: Significantly underweight - consider buying")
            elif h.drift > 1:
                lines.append("   Assessment: Significantly overweight - consider trimming")

        # Last rebalance
        lines.append(f"\n🔄 LAST REBALANCE:")
        if h.in_rebalancer:
            lines.append(f"   Date: {h.last_rebalance_date}")
            lines.append(f"   Action: {h.last_action}")
            lines.append(f"   Status: {h.last_status}")
            lines.append(f"   Execution Accuracy: {h.last_execution_accuracy:.1f}%")
        else:
            lines.append("   No rebalance data")

        # Options flow (if available)
        options_context = self.get_options_flow_context(symbol)
        if options_context:
            lines.append(options_context)

        return "\n".join(lines)

    def get_options_flow_context(self, symbol: str) -> str:
        """
        Get options flow analysis for a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            Formatted options flow context string
        """
        if not OPTIONS_FLOW_AVAILABLE:
            return ""

        try:
            analyzer = OptionsFlowAnalyzer()
            summary = analyzer.analyze_ticker(symbol.upper())

            if not summary or (summary.total_call_volume == 0 and summary.total_put_volume == 0):
                return f"\n🔮 OPTIONS FLOW: No options data available for {symbol}"

            lines = [
                f"\n{'=' * 50}",
                f"🔮 OPTIONS FLOW ANALYSIS: {symbol.upper()}",
                f"{'=' * 50}",
                f"",
                f"📊 VOLUME & SENTIMENT:",
                f"   Put/Call Volume Ratio: {summary.put_call_volume_ratio:.2f}",
                f"   Overall Sentiment: {summary.overall_sentiment} (score: {summary.sentiment_score:.0f})",
                f"   Call Volume: {summary.total_call_volume:,} | Put Volume: {summary.total_put_volume:,}",
                f"",
                f"💰 KEY LEVELS:",
                f"   Stock Price: ${summary.stock_price:.2f}",
                f"   Max Pain: ${summary.max_pain_price:.2f}",
            ]

            # Add interpretation
            if summary.max_pain_price > 0:
                max_pain_diff = ((summary.max_pain_price - summary.stock_price) / summary.stock_price) * 100
                if abs(max_pain_diff) < 2:
                    lines.append(f"   Max Pain Assessment: Price is near max pain (within 2%)")
                elif max_pain_diff > 0:
                    lines.append(f"   Max Pain Assessment: Max pain is {max_pain_diff:.1f}% above current price")
                else:
                    lines.append(f"   Max Pain Assessment: Max pain is {abs(max_pain_diff):.1f}% below current price")

            # IV info
            lines.append(f"")
            lines.append(f"📈 IMPLIED VOLATILITY:")
            lines.append(f"   Avg Call IV: {summary.avg_call_iv * 100:.1f}%")
            lines.append(f"   Avg Put IV: {summary.avg_put_iv * 100:.1f}%")
            lines.append(f"   IV Skew: {summary.iv_skew * 100:.1f}% (positive = more fear/hedging)")

            # Unusual activity summary
            if summary.alerts:
                high_alerts = [a for a in summary.alerts if
                               a.severity == 'HIGH' and abs(a.distance_from_strike_pct) < 15]
                if high_alerts:
                    lines.append(f"")
                    lines.append(f"⚠️ UNUSUAL ACTIVITY ({len(high_alerts)} high-priority alerts near the money):")
                    for alert in high_alerts[:3]:
                        lines.append(
                            f"   • {alert.option_type} ${alert.strike:.0f} ({alert.expiry}): {alert.description[:60]}")

            # Sentiment interpretation for AI
            lines.append(f"")
            lines.append(f"🤖 AI INTERPRETATION:")
            if summary.overall_sentiment == 'BULLISH':
                lines.append(f"   Options flow suggests bullish institutional positioning.")
                if summary.put_call_volume_ratio < 0.5:
                    lines.append(f"   Very low put/call ratio indicates strong call buying activity.")
            elif summary.overall_sentiment == 'BEARISH':
                lines.append(f"   Options flow suggests bearish institutional positioning.")
                if summary.put_call_volume_ratio > 1.5:
                    lines.append(f"   High put/call ratio indicates significant put buying or hedging.")
            else:
                lines.append(f"   Options flow is neutral - no strong directional bias detected.")

            return "\n".join(lines)

        except Exception as e:
            logger.debug(f"Error getting options flow for {symbol}: {e}")
            return ""

    def get_unified_context(self,
                            ibkr_positions: List[Dict] = None,
                            account_summary: Dict = None,
                            max_holdings: int = 20) -> str:
        """
        Get full unified portfolio context for AI.

        Args:
            ibkr_positions: List of position dicts from IBKR
            account_summary: Account summary dict from IBKR
            max_holdings: Max number of holdings to include in detail

        Returns:
            Formatted context string for AI
        """
        # Load if not already loaded
        if not self._unified_holdings:
            self.load_unified_portfolio(ibkr_positions, account_summary)

        s = self._summary
        lines = []

        # Header
        lines.append("=" * 60)
        lines.append("UNIFIED PORTFOLIO (Live IBKR + Target + Rebalance)")
        lines.append("=" * 60)

        # Summary
        lines.append(f"\n📈 PORTFOLIO SUMMARY:")
        lines.append(f"   Total Value: ${s.total_value:,.2f} (from IBKR)")
        lines.append(f"   Holdings: {s.total_holdings_actual} actual / {s.total_holdings_target} target")
        lines.append(
            f"   On Target: {s.holdings_on_target} | Not Held: {s.holdings_not_held} | Extra: {s.holdings_extra}")
        lines.append(f"   Last Rebalance: {s.last_rebalance_date} ({s.last_rebalance_accuracy:.1f}% accuracy)")

        # Drift summary
        lines.append(f"\n📐 DRIFT SUMMARY:")
        lines.append(f"   Average Drift: {s.avg_drift:.2f}%")
        lines.append(f"   Max Drift: {s.max_drift_symbol} ({s.max_drift_value:+.2f}%)")

        # Sector allocation
        if s.sector_allocations:
            lines.append(f"\n📊 SECTOR ALLOCATION:")
            sorted_sectors = sorted(s.sector_allocations.items(),
                                    key=lambda x: x[1]['actual'], reverse=True)
            for sector, weights in sorted_sectors[:8]:
                actual = weights['actual']
                target = weights['target']
                drift = actual - target
                drift_icon = "✓" if abs(drift) < 2 else ("⬆️" if drift > 0 else "⬇️")
                lines.append(f"   {sector}: {actual:.1f}% actual | {target:.1f}% target | {drift:+.1f}% {drift_icon}")

        # Holdings detail
        lines.append(f"\n📋 HOLDINGS DETAIL (Top {max_holdings}):")

        # Sort by actual weight (held positions first)
        sorted_holdings = sorted(
            self._unified_holdings.values(),
            key=lambda x: (x.in_ibkr, x.actual_weight),
            reverse=True
        )[:max_holdings]

        for i, h in enumerate(sorted_holdings, 1):
            # Compact format
            actual_str = f"{h.actual_weight:.1f}%" if h.in_ibkr else "0%"
            target_str = f"{h.target_weight:.1f}%" if h.in_etf_creator else "N/A"
            drift_str = f"{h.drift:+.1f}%" if h.in_ibkr and h.in_etf_creator else "N/A"

            status_icon = "✓" if "On target" in h.holding_status else "⚠️" if "⚠️" in h.holding_status else "❌"

            lines.append(f"\n{i}. {h.symbol} {status_icon}")
            lines.append(f"   Actual: {actual_str} | Target: {target_str} | Drift: {drift_str}")

            if h.in_ibkr:
                lines.append(
                    f"   Shares: {h.actual_shares:,.0f} | Value: ${h.actual_value:,.2f} | P&L: ${h.unrealized_pnl:+,.2f}")

            if h.selection_reasoning:
                reason_short = h.selection_reasoning[:100] + "..." if len(
                    h.selection_reasoning) > 100 else h.selection_reasoning
                lines.append(f"   Reason: {reason_short}")

            if h.last_action and h.last_action != "HOLD":
                lines.append(f"   Last Trade: {h.last_action} - {h.last_status}")

        # Attention needed
        needs_attention = [h for h in self._unified_holdings.values()
                           if (not h.in_ibkr and h.in_etf_creator) or  # Not held but should be
                           (h.in_ibkr and not h.in_etf_creator) or  # Held but shouldn't be
                           (abs(h.drift) > 2)]  # Large drift

        if needs_attention:
            lines.append(f"\n⚠️ NEEDS ATTENTION ({len(needs_attention)} items):")
            for h in sorted(needs_attention, key=lambda x: abs(x.drift), reverse=True)[:5]:
                lines.append(f"   - {h.symbol}: {h.holding_status}")

        # Rebalance recommendation
        lines.append(f"\n💡 REBALANCE RECOMMENDATION:")
        if s.holdings_not_held > 5 or s.avg_drift > 2:
            lines.append("   🔴 Rebalancing strongly recommended")
            lines.append(f"   - {s.holdings_not_held} target positions not held")
            lines.append(f"   - Average drift: {s.avg_drift:.1f}%")
        elif s.holdings_not_held > 0 or s.avg_drift > 1:
            lines.append("   🟡 Consider rebalancing soon")
        else:
            lines.append("   🟢 Portfolio is well-aligned")

        return "\n".join(lines)

    def get_needs_attention(self) -> str:
        """Get list of holdings that need attention."""
        if not self._unified_holdings:
            return "No portfolio data loaded."

        issues = []

        # Not held but should be
        not_held = [h for h in self._unified_holdings.values()
                    if not h.in_ibkr and h.in_etf_creator]
        if not_held:
            issues.append("NOT HELD (should buy):")
            for h in sorted(not_held, key=lambda x: x.target_weight, reverse=True)[:5]:
                issues.append(f"  - {h.symbol}: Target {h.target_weight:.1f}%")

        # Held but not in target
        extra = [h for h in self._unified_holdings.values()
                 if h.in_ibkr and not h.in_etf_creator]
        if extra:
            issues.append("\nEXTRA HOLDINGS (consider selling):")
            for h in sorted(extra, key=lambda x: x.actual_weight, reverse=True)[:5]:
                issues.append(f"  - {h.symbol}: {h.actual_weight:.1f}% (${h.actual_value:,.0f})")

        # Large drift
        large_drift = [h for h in self._unified_holdings.values()
                       if h.in_ibkr and h.in_etf_creator and abs(h.drift) > 2]
        if large_drift:
            issues.append("\nLARGE DRIFT (needs rebalancing):")
            for h in sorted(large_drift, key=lambda x: abs(x.drift), reverse=True)[:5]:
                action = "buy more" if h.drift < 0 else "trim"
                issues.append(f"  - {h.symbol}: {h.drift:+.1f}% drift ({action})")

        # Failed trades
        failed = [h for h in self._unified_holdings.values()
                  if h.last_status == "FAILED"]
        if failed:
            issues.append("\nFAILED TRADES:")
            for h in failed[:5]:
                issues.append(f"  - {h.symbol}: {h.last_action} failed")

        if not issues:
            return "✓ No issues found - portfolio is well-aligned!"

        return "\n".join(issues)


# Singleton instance
_unified_loader = None


def get_unified_portfolio_loader() -> UnifiedPortfolioLoader:
    """Get singleton instance of unified portfolio loader."""
    global _unified_loader
    if _unified_loader is None:
        _unified_loader = UnifiedPortfolioLoader()
    return _unified_loader


def get_unified_context(ibkr_positions: List[Dict] = None,
                        account_summary: Dict = None,
                        symbol: str = None) -> str:
    """
    Quick access to unified portfolio context.

    Args:
        ibkr_positions: Live positions from IBKR
        account_summary: Account summary from IBKR
        symbol: Optional specific ticker

    Returns:
        Formatted context string
    """
    loader = get_unified_portfolio_loader()
    loader.load_unified_portfolio(ibkr_positions, account_summary)

    if symbol:
        return loader.get_holding_context(symbol)
    else:
        return loader.get_unified_context(ibkr_positions, account_summary)


def get_options_flow(symbol: str) -> str:
    """
    Get options flow analysis for any ticker (standalone function).

    This can be called without loading portfolio data - useful for
    analyzing tickers that aren't in the portfolio.

    Args:
        symbol: Stock ticker

    Returns:
        Formatted options flow context string
    """
    if not OPTIONS_FLOW_AVAILABLE:
        return f"Options flow analysis not available (module not installed)"

    loader = get_unified_portfolio_loader()
    return loader.get_options_flow_context(symbol)


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    # Mock IBKR data for testing
    mock_positions = [
        {"symbol": "NVDA", "position": 50, "marketValue": 22500, "avgCost": 400, "unrealizedPNL": 2500},
        {"symbol": "MSFT", "position": 30, "marketValue": 12000, "avgCost": 380, "unrealizedPNL": 600},
        {"symbol": "AAPL", "position": 40, "marketValue": 8000, "avgCost": 190, "unrealizedPNL": 400},
        {"symbol": "XYZ", "position": 100, "marketValue": 5000, "avgCost": 48, "unrealizedPNL": 200},  # Not in target
    ]

    mock_summary = {
        "net_liquidation": 320000
    }

    print("\n" + "=" * 60)
    print("UNIFIED PORTFOLIO CONTEXT TEST")
    print("=" * 60)

    loader = UnifiedPortfolioLoader()
    context = loader.get_unified_context(mock_positions, mock_summary)
    print(context)

    print("\n" + "=" * 60)
    print("NVDA SPECIFIC CONTEXT")
    print("=" * 60)
    print(loader.get_holding_context("NVDA"))

    print("\n" + "=" * 60)
    print("NEEDS ATTENTION")
    print("=" * 60)
    print(loader.get_needs_attention())