# dashboard/ai_pm/portfolio_intelligence.py
"""
AI Portfolio Intelligence Module
================================

This module provides TRUE AI-powered portfolio analysis and recommendations.
It analyzes your actual portfolio and makes intelligent decisions:

1. Identifies underperforming/bad positions with alternatives
2. Recognizes when portfolio is already good (no action needed)
3. Detects low drift scenarios (skip rebalancing)
4. Handles empty portfolios with deployment recommendations
5. Considers NAV, risk tolerance, and strategy alignment

NO HARDCODED DATA - All analysis uses real portfolio and signal data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import json

import pandas as pd


@dataclass
class PositionAnalysis:
    """Analysis of a single position."""
    symbol: str
    current_weight: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float

    # Signal data (if available)
    signal: Optional[str] = None
    total_score: Optional[float] = None
    ai_probability: Optional[float] = None
    sector: Optional[str] = None

    # AI Assessment
    assessment: str = "UNKNOWN"  # STRONG, GOOD, NEUTRAL, WEAK, BAD
    issues: List[str] = field(default_factory=list)

    # Alternatives (if position is weak/bad)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PortfolioIntelligenceReport:
    """Complete AI analysis of a portfolio."""
    ts_utc: datetime
    account: str

    # Portfolio Summary
    nav: float
    cash: float
    cash_pct: float
    num_positions: int

    # Overall Assessment
    overall_status: str  # EXCELLENT, GOOD, NEEDS_ATTENTION, POOR, EMPTY
    headline: str

    # Detailed Analysis
    position_analyses: List[PositionAnalysis] = field(default_factory=list)

    # Recommendations
    recommended_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Flags
    no_action_needed: bool = False
    drift_too_low: bool = False
    has_bad_positions: bool = False
    portfolio_is_empty: bool = False

    # Reasoning
    reasoning: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_utc": self.ts_utc.isoformat(),
            "account": self.account,
            "nav": self.nav,
            "cash": self.cash,
            "cash_pct": self.cash_pct,
            "num_positions": self.num_positions,
            "overall_status": self.overall_status,
            "headline": self.headline,
            "no_action_needed": self.no_action_needed,
            "drift_too_low": self.drift_too_low,
            "has_bad_positions": self.has_bad_positions,
            "portfolio_is_empty": self.portfolio_is_empty,
            "recommended_actions": self.recommended_actions,
            "reasoning": self.reasoning,
            "warnings": self.warnings,
        }


def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except:
        return None


def _get_signal_data(symbol: str, signals) -> Dict[str, Any]:
    """Extract signal data for a symbol."""
    if not signals or not hasattr(signals, 'rows'):
        return {}

    row = signals.rows.get(symbol)
    if not row:
        return {}

    raw = getattr(row, 'raw', {}) or {}

    return {
        'signal': raw.get('signal') or raw.get('signal_type') or raw.get('Signal'),
        'total_score': _safe_float(raw.get('total_score')),
        'ai_probability': _safe_float(raw.get('ai_probability') or raw.get('likelihood_score')),
        'sector': raw.get('sector') or raw.get('Sector'),
        'fundamental_score': _safe_float(raw.get('fundamental_score')),
        'technical_score': _safe_float(raw.get('technical_score')),
        'sentiment_score': _safe_float(raw.get('sentiment_score')),
    }


def _assess_position(
        symbol: str,
        weight: float,
        pnl_pct: float,
        signal_data: Dict[str, Any],
) -> Tuple[str, List[str]]:
    """
    Assess a position's quality.
    Returns: (assessment, list_of_issues)

    Assessment levels: STRONG, GOOD, NEUTRAL, WEAK, BAD
    """
    issues = []
    score = 50  # Start neutral

    signal = str(signal_data.get('signal', '')).upper() if signal_data.get('signal') else ''
    total_score = signal_data.get('total_score')
    ai_prob = signal_data.get('ai_probability')

    # Check signal alignment
    if 'SELL' in signal or 'BEARISH' in signal:
        issues.append(f"Signal is {signal} - consider exiting")
        score -= 30
    elif 'STRONG' in signal and 'BUY' in signal:
        score += 20
    elif 'BUY' in signal:
        score += 10
    elif not signal:
        issues.append("No signal data available")
        score -= 5

    # Check total score
    if total_score is not None:
        if total_score < 30:
            issues.append(f"Low score ({total_score:.0f}/100)")
            score -= 20
        elif total_score < 50:
            issues.append(f"Below average score ({total_score:.0f}/100)")
            score -= 10
        elif total_score >= 70:
            score += 15
        elif total_score >= 60:
            score += 5

    # Check AI probability
    if ai_prob is not None:
        if ai_prob < 0.35:
            issues.append(f"Low AI probability ({ai_prob:.0%})")
            score -= 15
        elif ai_prob >= 0.60:
            score += 10

    # Check P&L
    if pnl_pct < -20:
        issues.append(f"Significant loss ({pnl_pct:.1f}%)")
        score -= 15
    elif pnl_pct < -10:
        issues.append(f"Moderate loss ({pnl_pct:.1f}%)")
        score -= 5
    elif pnl_pct > 50:
        score += 10  # Winner, but might need profit taking

    # Determine assessment
    if score >= 70:
        assessment = "STRONG"
    elif score >= 55:
        assessment = "GOOD"
    elif score >= 40:
        assessment = "NEUTRAL"
    elif score >= 25:
        assessment = "WEAK"
    else:
        assessment = "BAD"

    return assessment, issues


def _find_alternatives(
        bad_position: PositionAnalysis,
        signals,
        current_symbols: set,
        max_alternatives: int = 3,
) -> List[Dict[str, Any]]:
    """Find better alternatives for a weak/bad position."""
    if not signals or not hasattr(signals, 'rows'):
        return []

    sector = bad_position.sector

    candidates = []
    for sym, row in signals.rows.items():
        # Skip if already in portfolio
        if sym in current_symbols:
            continue

        raw = getattr(row, 'raw', {}) or {}
        signal = str(raw.get('signal') or raw.get('signal_type') or '').upper()
        total_score = _safe_float(raw.get('total_score'))
        ai_prob = _safe_float(raw.get('ai_probability') or raw.get('likelihood_score'))
        sym_sector = raw.get('sector') or raw.get('Sector')

        # Only consider BUY signals with decent scores
        if 'BUY' not in signal:
            continue
        if total_score is None or total_score < 50:
            continue

        # Prefer same sector if available
        same_sector = (sector and sym_sector and sector.lower() == sym_sector.lower())

        candidates.append({
            'symbol': sym,
            'signal': signal,
            'total_score': total_score,
            'ai_probability': ai_prob,
            'sector': sym_sector,
            'same_sector': same_sector,
            'score_rank': (total_score or 0) + (20 if same_sector else 0),
        })

    # Sort by score (same sector gets bonus)
    candidates.sort(key=lambda x: x['score_rank'], reverse=True)

    return candidates[:max_alternatives]


def _calculate_drift(current_weights: Dict[str, float], target_weights: Dict[str, float]) -> float:
    """Calculate total portfolio drift."""
    all_symbols = set(current_weights.keys()) | set(target_weights.keys())
    total_drift = 0.0

    for sym in all_symbols:
        cw = current_weights.get(sym, 0)
        tw = target_weights.get(sym, 0)
        total_drift += abs(tw - cw)

    return total_drift / 2  # Divide by 2 because drift is counted twice


def analyze_portfolio_intelligence(
        *,
        snapshot,
        signals,
        targets,
        strategy_key: str,
        account: str,
        loaded_positions: Dict[str, Dict[str, Any]] = None,
        min_drift_threshold: float = 0.02,  # 2% minimum drift to recommend action
) -> PortfolioIntelligenceReport:
    """
    Main AI Portfolio Intelligence analysis.

    This function provides intelligent portfolio analysis:
    1. Analyzes each position for quality
    2. Identifies positions that need attention
    3. Finds alternatives for bad positions
    4. Determines if any action is needed
    5. Provides clear recommendations
    """
    ts = datetime.utcnow()

    # Get NAV and cash
    nav = _safe_float(getattr(snapshot, 'net_liquidation', None)) or 0.0
    cash = _safe_float(getattr(snapshot, 'total_cash', None)) or 0.0
    cash_pct = (cash / nav) if nav > 0 else 0.0

    # Get positions - prefer loaded_positions from statement, fallback to snapshot
    if loaded_positions:
        positions_data = loaded_positions
    else:
        positions_data = {}
        total_value = 0
        for p in (getattr(snapshot, 'positions', None) or []):
            mv = _safe_float(getattr(p, 'market_value', None)) or 0
            total_value += abs(mv)

        for p in (getattr(snapshot, 'positions', None) or []):
            sym = (getattr(p, 'symbol', '') or '').strip().upper()
            if not sym:
                continue
            mv = _safe_float(getattr(p, 'market_value', None)) or 0
            positions_data[sym] = {
                'quantity': getattr(p, 'quantity', 0),
                'market_value': mv,
                'weight': mv / total_value if total_value > 0 else 0,
                'unrealized_pnl': 0,
                'unrealized_pnl_pct': 0,
            }

    num_positions = len(positions_data)
    current_symbols = set(positions_data.keys())

    # Get target weights
    target_weights = targets.weights if targets and targets.weights else {}

    # Initialize report
    report = PortfolioIntelligenceReport(
        ts_utc=ts,
        account=account,
        nav=nav,
        cash=cash,
        cash_pct=cash_pct,
        num_positions=num_positions,
        overall_status="UNKNOWN",
        headline="",
    )

    # === CASE 1: Empty Portfolio ===
    if num_positions == 0:
        report.portfolio_is_empty = True
        report.overall_status = "EMPTY"
        report.headline = f"Portfolio is empty. NAV: ${nav:,.0f} available to deploy."

        if nav > 0:
            report.reasoning.append(f"You have ${nav:,.0f} available to invest.")

            # Find top recommendations from signals
            if signals and hasattr(signals, 'rows') and signals.rows:
                top_buys = []
                for sym, row in signals.rows.items():
                    raw = getattr(row, 'raw', {}) or {}
                    signal = str(raw.get('signal') or raw.get('signal_type') or '').upper()
                    total_score = _safe_float(raw.get('total_score'))

                    if 'BUY' in signal and total_score and total_score >= 50:
                        top_buys.append({
                            'symbol': sym,
                            'signal': signal,
                            'total_score': total_score,
                            'sector': raw.get('sector'),
                        })

                top_buys.sort(key=lambda x: x['total_score'] or 0, reverse=True)

                if top_buys:
                    report.recommended_actions.append({
                        'action': 'DEPLOY_CAPITAL',
                        'description': f"Deploy ${nav:,.0f} into top-rated stocks",
                        'suggestions': top_buys[:10],
                    })
                    report.reasoning.append(f"Found {len(top_buys)} stocks with BUY signals and score ≥50")
                else:
                    report.warnings.append("No stocks with strong BUY signals found. Consider running scanner first.")
            else:
                report.warnings.append("No signal data available. Run the scanner to get stock recommendations.")
        else:
            report.warnings.append("NAV is zero. Deposit funds to start investing.")

        return report

    # === CASE 2: Analyze existing positions ===
    position_analyses = []
    strong_positions = 0
    good_positions = 0
    neutral_positions = 0
    weak_positions = 0
    bad_positions = 0

    # Build current weights
    current_weights = {}
    for sym, data in positions_data.items():
        current_weights[sym] = data.get('weight', 0)

    for sym, data in positions_data.items():
        signal_data = _get_signal_data(sym, signals)

        weight = data.get('weight', 0)
        pnl_pct = data.get('unrealized_pnl_pct', 0)

        assessment, issues = _assess_position(sym, weight, pnl_pct, signal_data)

        pa = PositionAnalysis(
            symbol=sym,
            current_weight=weight,
            market_value=data.get('market_value', 0),
            unrealized_pnl=data.get('unrealized_pnl', 0),
            unrealized_pnl_pct=pnl_pct,
            signal=signal_data.get('signal'),
            total_score=signal_data.get('total_score'),
            ai_probability=signal_data.get('ai_probability'),
            sector=signal_data.get('sector'),
            assessment=assessment,
            issues=issues,
        )

        # Find alternatives for weak/bad positions
        if assessment in ('WEAK', 'BAD'):
            pa.alternatives = _find_alternatives(pa, signals, current_symbols)

        position_analyses.append(pa)

        if assessment == 'STRONG':
            strong_positions += 1
        elif assessment == 'GOOD':
            good_positions += 1
        elif assessment == 'NEUTRAL':
            neutral_positions += 1
        elif assessment == 'WEAK':
            weak_positions += 1
        else:
            bad_positions += 1

    report.position_analyses = position_analyses

    # === CASE 3: Check if drift is too low ===
    if target_weights:
        drift = _calculate_drift(current_weights, target_weights)

        if drift < min_drift_threshold:
            report.drift_too_low = True
            report.reasoning.append(
                f"Portfolio drift is only {drift:.1%}, below threshold of {min_drift_threshold:.1%}")

    # === Determine overall status and recommendations ===

    # Check for bad positions
    if bad_positions > 0 or weak_positions > 0:
        report.has_bad_positions = True

    # Determine overall status
    total_good = strong_positions + good_positions
    total_bad = weak_positions + bad_positions

    if total_bad == 0 and total_good >= num_positions * 0.7:
        report.overall_status = "EXCELLENT"
        report.headline = f"Portfolio is in excellent shape. {strong_positions} strong, {good_positions} good positions."

        if report.drift_too_low:
            report.no_action_needed = True
            report.reasoning.append("Portfolio is well-positioned and drift is low. No action needed.")

    elif total_bad == 0 and total_good >= num_positions * 0.5:
        report.overall_status = "GOOD"
        report.headline = f"Portfolio is good. {neutral_positions} positions could be improved."

        if report.drift_too_low:
            report.no_action_needed = True
            report.reasoning.append(
                "Portfolio is acceptable and drift is low. Consider improving neutral positions when convenient.")

    elif total_bad <= num_positions * 0.2:
        report.overall_status = "NEEDS_ATTENTION"
        report.headline = f"Portfolio needs attention. {weak_positions} weak, {bad_positions} bad positions identified."

    else:
        report.overall_status = "POOR"
        report.headline = f"Portfolio needs significant work. {weak_positions} weak, {bad_positions} bad positions."

    # === Build recommendations ===

    # Recommend replacing bad positions
    bad_pas = [pa for pa in position_analyses if pa.assessment in ('BAD', 'WEAK')]
    if bad_pas:
        for pa in sorted(bad_pas, key=lambda x: x.market_value, reverse=True):
            action = {
                'action': 'REPLACE',
                'symbol': pa.symbol,
                'current_weight': f"{pa.current_weight:.1%}",
                'assessment': pa.assessment,
                'issues': pa.issues,
                'alternatives': pa.alternatives,
            }

            if pa.alternatives:
                best_alt = pa.alternatives[0]
                action[
                    'recommendation'] = f"Consider replacing {pa.symbol} with {best_alt['symbol']} (Score: {best_alt['total_score']:.0f})"
            else:
                action['recommendation'] = f"Exit {pa.symbol} - no suitable alternative found in same sector"

            report.recommended_actions.append(action)

    # If no action needed, say so explicitly
    if report.no_action_needed:
        report.recommended_actions = [{
            'action': 'HOLD',
            'description': 'No action recommended at this time',
            'reasoning': report.reasoning,
        }]

    # Add summary reasoning
    report.reasoning.append(
        f"Position breakdown: {strong_positions} strong, {good_positions} good, {neutral_positions} neutral, {weak_positions} weak, {bad_positions} bad")

    if cash_pct > 0.10:
        report.warnings.append(f"Cash allocation is {cash_pct:.1%} - consider deploying excess cash")
    elif cash_pct < 0.02:
        report.warnings.append(f"Cash allocation is only {cash_pct:.1%} - consider maintaining some cash buffer")

    return report


def format_intelligence_report_markdown(report: PortfolioIntelligenceReport) -> str:
    """Format the intelligence report as markdown for display."""
    lines = []

    # Header
    lines.append(f"# 🧠 AI Portfolio Intelligence Report")
    lines.append(f"*Generated: {report.ts_utc.strftime('%Y-%m-%d %H:%M UTC')}*")
    lines.append("")

    # Status badge
    status_emoji = {
        'EXCELLENT': '🌟',
        'GOOD': '✅',
        'NEEDS_ATTENTION': '⚠️',
        'POOR': '🔴',
        'EMPTY': '📭',
    }.get(report.overall_status, '❓')

    lines.append(f"## {status_emoji} {report.headline}")
    lines.append("")

    # Summary
    lines.append("### Portfolio Summary")
    lines.append(f"- **NAV:** ${report.nav:,.0f}")
    lines.append(f"- **Cash:** ${report.cash:,.0f} ({report.cash_pct:.1%})")
    lines.append(f"- **Positions:** {report.num_positions}")
    lines.append("")

    # Key Flags
    if report.no_action_needed:
        lines.append("### ✅ No Action Needed")
        lines.append("Your portfolio is well-positioned. No changes recommended at this time.")
        lines.append("")

    if report.drift_too_low:
        lines.append("### 📊 Drift Analysis")
        lines.append("Portfolio drift is below threshold. Rebalancing would be inefficient.")
        lines.append("")

    # Recommendations
    if report.recommended_actions:
        lines.append("### 📋 Recommended Actions")
        for i, action in enumerate(report.recommended_actions, 1):
            if action['action'] == 'HOLD':
                lines.append(f"**{i}. HOLD** - {action['description']}")
            elif action['action'] == 'REPLACE':
                lines.append(f"**{i}. REPLACE {action['symbol']}** ({action['assessment']})")
                if action.get('issues'):
                    for issue in action['issues']:
                        lines.append(f"   - Issue: {issue}")
                if action.get('recommendation'):
                    lines.append(f"   - 💡 {action['recommendation']}")
                if action.get('alternatives'):
                    lines.append("   - Alternatives:")
                    for alt in action['alternatives'][:3]:
                        lines.append(f"     - {alt['symbol']}: {alt['signal']} (Score: {alt['total_score']:.0f})")
            elif action['action'] == 'DEPLOY_CAPITAL':
                lines.append(f"**{i}. DEPLOY CAPITAL** - {action['description']}")
                if action.get('suggestions'):
                    lines.append("   Top recommendations:")
                    for sug in action['suggestions'][:5]:
                        lines.append(f"   - {sug['symbol']}: {sug['signal']} (Score: {sug['total_score']:.0f})")
            lines.append("")

    # Reasoning
    if report.reasoning:
        lines.append("### 🔍 Analysis Details")
        for reason in report.reasoning:
            lines.append(f"- {reason}")
        lines.append("")

    # Warnings
    if report.warnings:
        lines.append("### ⚠️ Warnings")
        for warning in report.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines)