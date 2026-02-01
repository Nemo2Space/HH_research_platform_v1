
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PortfolioSummary:
    """Summary of portfolio state for AI context"""
    total_holdings: int = 0
    total_value: float = 0.0
    top_10_concentration: float = 0.0
    sector_distribution: Dict[str, float] = None
    last_rebalance_date: str = ""
    last_rebalance_accuracy: float = 0.0
    drift_summary: str = ""
    etf_sources: List[str] = None


class PortfolioContextLoader:
    """
    Loads and formats portfolio context for AI Chat.

    Reads from:
    - ETF Creator: json/portfolio_*_latest.json (stock selection reasoning)
    - Rebalancer: json/rebalance_*_latest.json (trade execution details)

    Usage:
        loader = PortfolioContextLoader()
        context = loader.get_full_context()
        # Pass context to AI system prompt
    """

    def __init__(self,
                 etf_creator_dir: str = None,
                 rebalancer_dir: str = None):
        """
        Initialize with paths to JSON directories.

        Args:
            etf_creator_dir: Path to ETF Creator json folder
            rebalancer_dir: Path to Rebalancer json folder
        """
        # Default paths - adjust based on your project structure
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
        self._cache_time = None
        self._cache_ttl = 300  # 5 minutes

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
                files.sort(reverse=True)  # Most recent timestamp first
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

    def load_etf_creator_data(self) -> Optional[Dict]:
        """Load latest ETF Creator portfolio data."""
        # Try multiple prefixes
        for prefix in ['portfolio_SC', 'portfolio_', 'etf_']:
            filepath = self._find_latest_file(self.etf_creator_dir, prefix)
            if filepath:
                data = self._load_json_safe(filepath)
                if data:
                    logger.info(f"Loaded ETF Creator data from {filepath}")
                    self._etf_creator_data = data
                    return data

        logger.warning("No ETF Creator data found")
        return None

    def load_rebalancer_data(self) -> Optional[Dict]:
        """Load latest Rebalancer data."""
        # Try multiple prefixes
        for prefix in ['rebalance_', 'rebal_']:
            filepath = self._find_latest_file(self.rebalancer_dir, prefix)
            if filepath:
                data = self._load_json_safe(filepath)
                if data:
                    logger.info(f"Loaded Rebalancer data from {filepath}")
                    self._rebalancer_data = data
                    return data

        logger.warning("No Rebalancer data found")
        return None

    def get_holding_context(self, symbol: str) -> str:
        """
        Get context for a specific holding.

        Args:
            symbol: Stock ticker

        Returns:
            Formatted context string for the holding
        """
        context_parts = []
        symbol = symbol.upper()

        # Load data if needed
        if not self._etf_creator_data:
            self.load_etf_creator_data()
        if not self._rebalancer_data:
            self.load_rebalancer_data()

        # ETF Creator context (why selected, scores)
        if self._etf_creator_data:
            holdings = self._etf_creator_data.get('holdings', [])
            for h in holdings:
                if h.get('symbol', '').upper() == symbol:
                    context_parts.append(f"\n{symbol} SELECTION REASONING:")

                    # Weight info
                    weight = h.get('weight', 0)
                    context_parts.append(f"- Target Weight: {weight:.2f}%")

                    # Scores
                    scores = h.get('scores', {})
                    if scores:
                        context_parts.append(f"- Sentiment: {scores.get('sentiment', 'N/A')}")
                        context_parts.append(f"- Fundamental: {scores.get('fundamental', 'N/A')}")
                        context_parts.append(f"- Growth: {scores.get('growth', 'N/A')}")

                    # Weight breakdown
                    breakdown = h.get('weight_breakdown', {})
                    if breakdown:
                        context_parts.append(
                            f"- Weight Components: Market Cap={breakdown.get('market_cap_weight', 0):.2f}%, Sentiment={breakdown.get('sentiment_weight', 0):.2f}%")

                    # AI reasoning
                    reasoning = h.get('reasoning', '')
                    if reasoning:
                        context_parts.append(f"- Selection Reason: {reasoning}")

                    # Risks
                    risks = h.get('risks', [])
                    if risks:
                        context_parts.append(f"- Risks: {'; '.join(risks[:3])}")

                    break

        # Rebalancer context (trade execution)
        if self._rebalancer_data:
            holdings = self._rebalancer_data.get('holdings', [])
            for h in holdings:
                if h.get('symbol', '').upper() == symbol:
                    context_parts.append(f"\n{symbol} REBALANCE STATUS:")

                    # Drift
                    drift = h.get('drift', {})
                    if drift:
                        context_parts.append(f"- Target Weight: {drift.get('target_weight', 0):.2f}%")
                        context_parts.append(f"- Actual Weight: {drift.get('actual_weight_after', 0):.2f}%")
                        context_parts.append(f"- Drift: {drift.get('drift_after', 0):+.2f}%")

                    # Trade
                    trade = h.get('trade', {})
                    if trade:
                        action = trade.get('action', 'HOLD')
                        accuracy = trade.get('execution_accuracy', 0)
                        context_parts.append(f"- Last Action: {action}")
                        context_parts.append(f"- Execution Accuracy: {accuracy:.1f}%")
                        context_parts.append(f"- Status: {trade.get('status', 'UNKNOWN')}")

                    # Reasoning
                    reasoning = h.get('reasoning', '')
                    if reasoning:
                        context_parts.append(f"- Trade Reason: {reasoning}")

                    break

        if not context_parts:
            return f"No portfolio data available for {symbol}"

        return "\n".join(context_parts)

    def get_portfolio_summary(self) -> PortfolioSummary:
        """Get high-level portfolio summary."""
        summary = PortfolioSummary(
            sector_distribution={},
            etf_sources=[]
        )

        # Load data
        if not self._etf_creator_data:
            self.load_etf_creator_data()
        if not self._rebalancer_data:
            self.load_rebalancer_data()

        # From ETF Creator
        if self._etf_creator_data:
            meta = self._etf_creator_data.get('metadata', {})
            sum_data = self._etf_creator_data.get('summary', {})

            summary.total_holdings = sum_data.get('total_holdings', 0)
            summary.top_10_concentration = sum_data.get('top_10_concentration', 0)
            summary.etf_sources = meta.get('etf_sources', [])

            # Sector distribution
            sectors = sum_data.get('sector_distribution', {})
            if isinstance(sectors, list):
                summary.sector_distribution = {s['name']: s['value'] for s in sectors}
            elif isinstance(sectors, dict):
                summary.sector_distribution = sectors

        # From Rebalancer
        if self._rebalancer_data:
            meta = self._rebalancer_data.get('metadata', {})
            sum_data = self._rebalancer_data.get('summary', {})

            summary.total_value = sum_data.get('portfolio_value_after', 0)
            summary.last_rebalance_date = meta.get('generated_at', '')[:10] if meta.get('generated_at') else ''
            summary.last_rebalance_accuracy = sum_data.get('average_accuracy', 0)

            # Calculate drift summary
            holdings = self._rebalancer_data.get('holdings', [])
            if holdings:
                drifts = []
                for h in holdings:
                    drift_data = h.get('drift', {})
                    drift_val = abs(drift_data.get('drift_after', 0))
                    drifts.append(drift_val)

                avg_drift = sum(drifts) / len(drifts) if drifts else 0
                max_drift = max(drifts) if drifts else 0

                if avg_drift < 0.5:
                    summary.drift_summary = f"Well aligned (avg drift {avg_drift:.2f}%)"
                elif avg_drift < 1.0:
                    summary.drift_summary = f"Minor drift (avg {avg_drift:.2f}%, max {max_drift:.2f}%)"
                else:
                    summary.drift_summary = f"Needs rebalancing (avg drift {avg_drift:.2f}%, max {max_drift:.2f}%)"

        return summary

    def get_top_holdings_context(self, n: int = 10) -> str:
        """Get context for top N holdings by weight."""
        if not self._rebalancer_data:
            self.load_rebalancer_data()
        if not self._etf_creator_data:
            self.load_etf_creator_data()

        # Use rebalancer data for actual weights
        holdings = []
        if self._rebalancer_data:
            holdings = self._rebalancer_data.get('holdings', [])
        elif self._etf_creator_data:
            holdings = self._etf_creator_data.get('holdings', [])

        if not holdings:
            return "No holdings data available"

        # Sort by weight
        sorted_holdings = sorted(
            holdings,
            key=lambda x: x.get('drift', {}).get('actual_weight_after', 0) or x.get('weight', 0),
            reverse=True
        )[:n]

        lines = [f"TOP {n} HOLDINGS:"]
        for i, h in enumerate(sorted_holdings, 1):
            symbol = h.get('symbol', 'N/A')

            # Get weight
            drift = h.get('drift', {})
            weight = drift.get('actual_weight_after', 0) or h.get('weight', 0)

            # Get action from trade
            trade = h.get('trade', {})
            action = trade.get('action', '')
            status = trade.get('status', '')

            line = f"{i}. {symbol}: {weight:.2f}%"
            if action and action != 'HOLD':
                line += f" [{action} - {status}]"

            lines.append(line)

        return "\n".join(lines)

    def get_failed_trades_context(self) -> str:
        """Get context about failed trades from last rebalance."""
        if not self._rebalancer_data:
            self.load_rebalancer_data()

        if not self._rebalancer_data:
            return "No rebalance data available"

        failed = self._rebalancer_data.get('failed_holdings', [])
        if not failed:
            # Check holdings for failed status
            holdings = self._rebalancer_data.get('holdings', [])
            failed = [h for h in holdings if h.get('trade', {}).get('status') == 'FAILED']

        if not failed:
            return "No failed trades in last rebalance"

        lines = ["FAILED TRADES:"]
        for h in failed:
            symbol = h.get('symbol', 'N/A')
            trade = h.get('trade', {})
            action = trade.get('action', 'N/A')
            target_qty = trade.get('target_quantity', 0)
            reason = h.get('reasoning', 'Unknown reason')

            lines.append(f"- {symbol}: {action} {target_qty} shares - {reason[:100]}")

        return "\n".join(lines)

    def get_sector_analysis_context(self) -> str:
        """Get sector allocation analysis."""
        summary = self.get_portfolio_summary()

        if not summary.sector_distribution:
            return "No sector data available"

        lines = ["SECTOR ALLOCATION:"]

        # Sort by weight descending
        sorted_sectors = sorted(
            summary.sector_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for sector, weight in sorted_sectors:
            bar = "█" * int(weight / 3)  # Simple bar chart
            lines.append(f"- {sector}: {weight:.1f}% {bar}")

        # Add concentration warning if needed
        top_sector = sorted_sectors[0] if sorted_sectors else ('', 0)
        if top_sector[1] > 35:
            lines.append(f"\n⚠️ High concentration in {top_sector[0]} ({top_sector[1]:.1f}%)")

        return "\n".join(lines)

    def get_rebalance_needed_context(self) -> str:
        """Analyze if rebalancing is needed."""
        if not self._rebalancer_data:
            self.load_rebalancer_data()

        if not self._rebalancer_data:
            return "No rebalance data available to analyze drift"

        holdings = self._rebalancer_data.get('holdings', [])
        if not holdings:
            return "No holdings data"

        # Analyze drift
        high_drift = []
        moderate_drift = []

        for h in holdings:
            symbol = h.get('symbol', '')
            drift = h.get('drift', {})
            drift_val = abs(drift.get('drift_after', 0))

            if drift_val > 2.0:
                high_drift.append((symbol, drift_val))
            elif drift_val > 1.0:
                moderate_drift.append((symbol, drift_val))

        lines = ["REBALANCE ANALYSIS:"]

        summary = self._rebalancer_data.get('summary', {})
        last_date = self._rebalancer_data.get('metadata', {}).get('generated_at', '')[:10]

        lines.append(f"- Last rebalance: {last_date}")
        lines.append(f"- Average accuracy: {summary.get('average_accuracy', 0):.1f}%")

        if high_drift:
            lines.append(f"\n⚠️ HIGH DRIFT ({len(high_drift)} stocks):")
            for symbol, drift in sorted(high_drift, key=lambda x: x[1], reverse=True)[:5]:
                lines.append(f"  - {symbol}: {drift:+.2f}% from target")

        if moderate_drift:
            lines.append(f"\n⚡ MODERATE DRIFT ({len(moderate_drift)} stocks):")
            for symbol, drift in sorted(moderate_drift, key=lambda x: x[1], reverse=True)[:5]:
                lines.append(f"  - {symbol}: {drift:+.2f}% from target")

        # Recommendation
        if len(high_drift) > 5:
            lines.append("\n🔴 RECOMMENDATION: Rebalancing strongly recommended")
        elif len(high_drift) > 0 or len(moderate_drift) > 10:
            lines.append("\n🟡 RECOMMENDATION: Consider rebalancing soon")
        else:
            lines.append("\n🟢 RECOMMENDATION: Portfolio is well-balanced")

        return "\n".join(lines)

    def get_full_context(self, include_holdings: bool = True,
                         include_sectors: bool = True,
                         include_rebalance: bool = True,
                         max_holdings: int = 15) -> str:
        """
        Get full portfolio context for AI system prompt.

        Args:
            include_holdings: Include top holdings details
            include_sectors: Include sector analysis
            include_rebalance: Include rebalance status
            max_holdings: Max number of holdings to include

        Returns:
            Formatted context string
        """
        parts = []

        # Header
        parts.append("=" * 50)
        parts.append("PORTFOLIO CONTEXT (from ETF Creator & Rebalancer)")
        parts.append("=" * 50)

        # Summary
        summary = self.get_portfolio_summary()
        parts.append(f"\nPORTFOLIO SUMMARY:")
        parts.append(f"- Total Holdings: {summary.total_holdings}")
        parts.append(f"- Portfolio Value: ${summary.total_value:,.2f}")
        parts.append(f"- Top 10 Concentration: {summary.top_10_concentration:.1f}%")
        parts.append(f"- Last Rebalance: {summary.last_rebalance_date}")
        parts.append(f"- Rebalance Accuracy: {summary.last_rebalance_accuracy:.1f}%")
        parts.append(f"- Drift Status: {summary.drift_summary}")
        if summary.etf_sources:
            parts.append(f"- ETF Sources: {', '.join(summary.etf_sources)}")

        # Sectors
        if include_sectors:
            parts.append(f"\n{self.get_sector_analysis_context()}")

        # Top holdings
        if include_holdings:
            parts.append(f"\n{self.get_top_holdings_context(max_holdings)}")

        # Rebalance status
        if include_rebalance:
            parts.append(f"\n{self.get_rebalance_needed_context()}")

        # Failed trades
        failed_context = self.get_failed_trades_context()
        if "No failed" not in failed_context:
            parts.append(f"\n{failed_context}")

        return "\n".join(parts)


# Singleton instance for easy import
_portfolio_loader = None


def get_portfolio_context_loader() -> PortfolioContextLoader:
    """Get singleton instance of portfolio context loader."""
    global _portfolio_loader
    if _portfolio_loader is None:
        _portfolio_loader = PortfolioContextLoader()
    return _portfolio_loader


def get_portfolio_context(symbol: str = None) -> str:
    """
    Quick access to portfolio context.

    Args:
        symbol: Optional specific ticker to get context for

    Returns:
        Formatted context string
    """
    loader = get_portfolio_context_loader()

    if symbol:
        return loader.get_holding_context(symbol)
    else:
        return loader.get_full_context()


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    loader = PortfolioContextLoader()

    print("\n" + "=" * 60)
    print("PORTFOLIO CONTEXT TEST")
    print("=" * 60)

    # Test full context
    context = loader.get_full_context()
    print(context)

    # Test specific holding
    print("\n" + "=" * 60)
    print("NVDA CONTEXT")
    print("=" * 60)
    print(loader.get_holding_context("NVDA"))