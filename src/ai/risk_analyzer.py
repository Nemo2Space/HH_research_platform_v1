"""
Alpha Platform - Risk Analyzer Module (v2)

Provides portfolio risk analysis including:
- Concentration risk (position weights, top holdings)
- Sector exposure analysis (with Yahoo Finance fallback)
- Portfolio beta calculation
- Correlation warnings between holdings
- Position alerts (SELL signals, big gainers/losers)

Author: Alpha Platform
Location: src/ai/risk_analyzer.py
"""

"""
Test script for Risk Analyzer module.
"""

"""
Test script for Risk Analyzer module.
"""

"""
Test script for Risk Analyzer module.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.db.connection import get_engine
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RiskAlert:
    """Represents a risk alert."""
    alert_type: str  # concentration, signal_conflict, profit_taking, stop_loss, correlation
    severity: str    # high, medium, low
    symbol: str
    message: str
    value: float     # The metric value (e.g., weight %, P&L %)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_type': self.alert_type,
            'severity': self.severity,
            'symbol': self.symbol,
            'message': self.message,
            'value': self.value
        }


@dataclass
class PortfolioRiskMetrics:
    """Complete portfolio risk metrics."""
    # Concentration
    top_5_concentration: float
    top_10_concentration: float
    largest_position: Tuple[str, float]  # (symbol, weight%)
    concentration_alerts: List[RiskAlert]

    # Sector exposure
    sector_breakdown: Dict[str, float]  # sector -> weight%
    sector_alerts: List[RiskAlert]

    # Beta
    portfolio_beta: float
    beta_interpretation: str

    # Correlations
    high_correlations: List[Tuple[str, str, float]]  # (sym1, sym2, correlation)
    correlation_alerts: List[RiskAlert]

    # Signal conflicts
    signal_conflicts: List[Dict[str, Any]]

    # P&L alerts
    profit_taking_candidates: List[Dict[str, Any]]
    stop_loss_candidates: List[Dict[str, Any]]

    # Overall
    risk_score: str  # A, B, C, D, F
    risk_level: str  # Low, Medium, High
    total_alerts: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'top_5_concentration': self.top_5_concentration,
            'top_10_concentration': self.top_10_concentration,
            'largest_position': self.largest_position,
            'concentration_alerts': [a.to_dict() for a in self.concentration_alerts],
            'sector_breakdown': self.sector_breakdown,
            'sector_alerts': [a.to_dict() for a in self.sector_alerts],
            'portfolio_beta': self.portfolio_beta,
            'beta_interpretation': self.beta_interpretation,
            'high_correlations': self.high_correlations,
            'correlation_alerts': [a.to_dict() for a in self.correlation_alerts],
            'signal_conflicts': self.signal_conflicts,
            'profit_taking_candidates': self.profit_taking_candidates,
            'stop_loss_candidates': self.stop_loss_candidates,
            'risk_score': self.risk_score,
            'risk_level': self.risk_level,
            'total_alerts': self.total_alerts
        }


class RiskAnalyzer:
    """
    Analyzes portfolio risk metrics.

    Usage:
        analyzer = RiskAnalyzer()
        metrics = analyzer.analyze_portfolio(positions, account_summary)
    """

    # Risk thresholds (configurable)
    MAX_POSITION_WEIGHT = 10.0      # Max single position weight %
    MAX_SECTOR_WEIGHT = 30.0        # Max sector weight %
    PROFIT_TAKING_THRESHOLD = 40.0  # Suggest profit taking above this P&L %
    STOP_LOSS_THRESHOLD = -20.0     # Suggest review below this P&L %
    CORRELATION_WARNING = 0.70      # Warn if correlation above this

    def __init__(self):
        self.engine = get_engine()
        self._sector_cache = {}  # Cache for Yahoo Finance lookups

    def analyze_portfolio(self,
                          positions: List[Dict[str, Any]],
                          account_summary: Optional[Dict[str, Any]] = None) -> PortfolioRiskMetrics:
        """
        Perform complete risk analysis on portfolio.

        Args:
            positions: List of position dicts from IBKR
                       Each has: symbol, quantity, avg_cost, current_price,
                                 market_value, unrealized_pnl, unrealized_pnl_pct
            account_summary: Optional account summary with net_liquidation

        Returns:
            PortfolioRiskMetrics with all risk data
        """
        if not positions:
            return self._empty_metrics()

        # Filter to stocks only (exclude options, futures, etc.)
        stock_positions = [p for p in positions if p.get('sec_type', 'STK') == 'STK']

        if not stock_positions:
            return self._empty_metrics()

        # Calculate total portfolio value from stock positions
        total_value = sum(p.get('market_value', 0) for p in stock_positions)

        # Add weight to each position
        for p in stock_positions:
            p['weight'] = (p.get('market_value', 0) / total_value * 100) if total_value > 0 else 0

        # Sort by weight descending
        positions_sorted = sorted(stock_positions, key=lambda x: x.get('weight', 0), reverse=True)

        # 1. Concentration analysis
        concentration_result = self._analyze_concentration(positions_sorted, total_value)

        # 2. Sector exposure (with Yahoo Finance fallback)
        sector_result = self._analyze_sector_exposure(positions_sorted)

        # 3. Portfolio beta
        beta_result = self._calculate_portfolio_beta(positions_sorted)

        # 4. Correlation analysis
        correlation_result = self._analyze_correlations(positions_sorted)

        # 5. Signal conflicts
        signal_conflicts = self._check_signal_conflicts(positions_sorted)

        # 6. P&L alerts
        profit_candidates, stop_candidates = self._check_pnl_alerts(positions_sorted)

        # 7. Calculate overall risk score
        all_alerts = (
            concentration_result['alerts'] +
            sector_result['alerts'] +
            correlation_result['alerts']
        )

        # Add signal conflicts and P&L alerts to count
        total_alerts = len(all_alerts) + len(signal_conflicts) + len(profit_candidates) + len(stop_candidates)

        risk_score, risk_level = self._calculate_risk_score(
            concentration_result, sector_result, beta_result,
            signal_conflicts, profit_candidates, stop_candidates
        )

        return PortfolioRiskMetrics(
            top_5_concentration=concentration_result['top_5'],
            top_10_concentration=concentration_result['top_10'],
            largest_position=concentration_result['largest'],
            concentration_alerts=concentration_result['alerts'],
            sector_breakdown=sector_result['breakdown'],
            sector_alerts=sector_result['alerts'],
            portfolio_beta=beta_result['beta'],
            beta_interpretation=beta_result['interpretation'],
            high_correlations=correlation_result['high_correlations'],
            correlation_alerts=correlation_result['alerts'],
            signal_conflicts=signal_conflicts,
            profit_taking_candidates=profit_candidates,
            stop_loss_candidates=stop_candidates,
            risk_score=risk_score,
            risk_level=risk_level,
            total_alerts=total_alerts
        )

    def _analyze_concentration(self, positions: List[Dict], total_value: float) -> Dict[str, Any]:
        """Analyze position concentration risk."""
        alerts = []

        # Top 5 and Top 10 concentration
        weights = [p.get('weight', 0) for p in positions]
        top_5 = sum(weights[:5]) if len(weights) >= 5 else sum(weights)
        top_10 = sum(weights[:10]) if len(weights) >= 10 else sum(weights)

        # Largest position
        largest = (positions[0]['symbol'], positions[0]['weight']) if positions else ('N/A', 0)

        # Check for overweight positions
        for p in positions:
            weight = p.get('weight', 0)
            if weight > self.MAX_POSITION_WEIGHT:
                if weight > 20:
                    severity = 'high'
                elif weight > 15:
                    severity = 'high'
                else:
                    severity = 'medium'

                alerts.append(RiskAlert(
                    alert_type='concentration',
                    severity=severity,
                    symbol=p['symbol'],
                    message=f"{p['symbol']} is {weight:.1f}% of portfolio (max recommended: {self.MAX_POSITION_WEIGHT}%)",
                    value=weight
                ))

        # Warn if top 5 is too concentrated
        if top_5 > 60:
            alerts.append(RiskAlert(
                alert_type='concentration',
                severity='high',
                symbol='TOP5',
                message=f"Top 5 positions = {top_5:.1f}% (very concentrated)",
                value=top_5
            ))
        elif top_5 > 50:
            alerts.append(RiskAlert(
                alert_type='concentration',
                severity='medium',
                symbol='TOP5',
                message=f"Top 5 positions = {top_5:.1f}% (concentrated)",
                value=top_5
            ))

        return {
            'top_5': round(top_5, 1),
            'top_10': round(top_10, 1),
            'largest': (largest[0], round(largest[1], 1)),
            'alerts': alerts
        }

    def _get_sector_from_yahoo(self, symbol: str) -> str:
        """Get sector from Yahoo Finance."""
        if symbol in self._sector_cache:
            return self._sector_cache[symbol]

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            sector = info.get('sector', 'Unknown')
            self._sector_cache[symbol] = sector
            return sector
        except Exception as e:
            logger.debug(f"Could not get sector for {symbol}: {e}")
            self._sector_cache[symbol] = 'Unknown'
            return 'Unknown'

    def _analyze_sector_exposure(self, positions: List[Dict]) -> Dict[str, Any]:
        """Analyze sector exposure with Yahoo Finance fallback."""
        alerts = []
        sector_weights = {}

        symbols = [p['symbol'] for p in positions]

        # First try database
        sector_map = {}
        try:
            if symbols:
                # Fixed - use tuple for IN clause
                query = """
                        SELECT DISTINCT \
                        ON (ticker) ticker, sector
                        FROM fundamentals
                        WHERE ticker = ANY (%(symbols)s)
                          AND sector IS NOT NULL
                          AND sector != ''
                        ORDER BY ticker, date DESC \
                        """
                df = pd.read_sql(query, self.engine, params={"symbols": symbols})
                sector_map = dict(zip(df['ticker'], df['sector']))
        except Exception as e:
            logger.warning(f"Could not fetch sectors from DB: {e}")

        # For symbols not in DB, try Yahoo Finance
        missing_symbols = [s for s in symbols if s not in sector_map or not sector_map.get(s)]

        if missing_symbols:
            logger.info(f"Fetching sectors from Yahoo for {len(missing_symbols)} symbols...")
            for symbol in missing_symbols[:30]:  # Limit to avoid rate limiting
                sector = self._get_sector_from_yahoo(symbol)
                if sector and sector != 'Unknown':
                    sector_map[symbol] = sector

        # Calculate sector weights
        for p in positions:
            sector = sector_map.get(p['symbol'], 'Unknown')
            weight = p.get('weight', 0)
            sector_weights[sector] = sector_weights.get(sector, 0) + weight

        # Round and sort
        sector_breakdown = {k: round(v, 1) for k, v in sector_weights.items()}
        sector_breakdown = dict(sorted(sector_breakdown.items(), key=lambda x: x[1], reverse=True))

        # Check for overweight sectors (excluding Unknown)
        for sector, weight in sector_breakdown.items():
            if sector != 'Unknown' and weight > self.MAX_SECTOR_WEIGHT:
                severity = 'high' if weight > 45 else 'medium'
                alerts.append(RiskAlert(
                    alert_type='sector_concentration',
                    severity=severity,
                    symbol=sector,
                    message=f"{sector} is {weight:.1f}% of portfolio (max: {self.MAX_SECTOR_WEIGHT}%)",
                    value=weight
                ))

        return {
            'breakdown': sector_breakdown,
            'alerts': alerts
        }

    def _calculate_portfolio_beta(self, positions: List[Dict]) -> Dict[str, Any]:
        """Calculate weighted portfolio beta using Yahoo Finance."""
        symbols = [p['symbol'] for p in positions]
        beta_map = {}

        # Try database first
        try:
            if symbols:
                query = """
                        SELECT DISTINCT \
                        ON (ticker) ticker, beta
                        FROM fundamentals
                        WHERE ticker = ANY (%(symbols)s)
                          AND beta IS NOT NULL
                        ORDER BY ticker, date DESC
                        """
                df = pd.read_sql(query, self.engine, params={"symbols": symbols})
                beta_map = dict(zip(df['ticker'], df['beta']))
        except Exception as e:
            logger.debug(f"Could not fetch betas from DB: {e}")
        # Fallback to Yahoo Finance for missing
        missing = [s for s in symbols if s not in beta_map]
        if missing:
            try:
                import yfinance as yf
                for symbol in missing[:20]:  # Limit API calls
                    try:
                        ticker = yf.Ticker(symbol)
                        beta = ticker.info.get('beta')
                        if beta is not None:
                            beta_map[symbol] = float(beta)
                    except:
                        pass
            except:
                pass

        # Calculate weighted beta
        total_weight = 0
        weighted_beta = 0

        for p in positions:
            beta = beta_map.get(p['symbol'], 1.0)  # Default to 1.0
            weight = p.get('weight', 0) / 100
            weighted_beta += beta * weight
            total_weight += weight

        portfolio_beta = weighted_beta / total_weight if total_weight > 0 else 1.0

        # Interpretation
        if portfolio_beta < 0.8:
            interpretation = "Defensive (less volatile than market)"
        elif portfolio_beta <= 1.2:
            interpretation = "Neutral (moves with market)"
        elif portfolio_beta <= 1.5:
            interpretation = "Aggressive (more volatile than market)"
        else:
            interpretation = "Very Aggressive (high volatility)"

        return {
            'beta': round(portfolio_beta, 2),
            'interpretation': interpretation
        }

    def _analyze_correlations(self, positions: List[Dict]) -> Dict[str, Any]:
        """Analyze correlations between top holdings."""
        alerts = []
        high_correlations = []

        # Only analyze top 15 positions
        top_positions = positions[:15]
        symbols = [p['symbol'] for p in top_positions]

        if len(symbols) < 2:
            return {'high_correlations': [], 'alerts': []}

        try:
            # Get price history
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=180)

            # Fixed - use tuple for IN clause
            query = """
                    SELECT ticker, date, adj_close
                    FROM prices
                    WHERE ticker = ANY (%(symbols)s)
                      AND date >= %(start_date)s
                      AND date <= %(end_date)s
                    ORDER BY ticker, date \
                    """
            params = {"symbols": symbols, "start_date": start_date, "end_date": end_date}
            df = pd.read_sql(query, self.engine, params=params)

            if df.empty or len(df) < 50:
                return {'high_correlations': [], 'alerts': []}

            # Pivot and calculate returns
            pivot = df.pivot(index='date', columns='ticker', values='adj_close')
            returns = pivot.pct_change(fill_method=None).dropna()
            if len(returns) < 20:
                return {'high_correlations': [], 'alerts': []}

            # Correlation matrix
            corr_matrix = returns.corr()

            # Find high correlations
            checked = set()
            for sym1 in corr_matrix.columns:
                for sym2 in corr_matrix.columns:
                    if sym1 >= sym2:
                        continue
                    pair = (sym1, sym2)
                    if pair in checked:
                        continue
                    checked.add(pair)

                    corr = corr_matrix.loc[sym1, sym2]
                    if pd.notna(corr) and corr >= self.CORRELATION_WARNING:
                        high_correlations.append((sym1, sym2, round(corr, 2)))

                        if corr >= 0.85:
                            alerts.append(RiskAlert(
                                alert_type='correlation',
                                severity='medium',
                                symbol=f"{sym1}/{sym2}",
                                message=f"{sym1} & {sym2} are {corr:.0%} correlated (double exposure)",
                                value=corr * 100
                            ))

            high_correlations.sort(key=lambda x: x[2], reverse=True)

        except Exception as e:
            logger.warning(f"Correlation analysis failed: {e}")

        return {
            'high_correlations': high_correlations[:10],
            'alerts': alerts
        }

    def _check_signal_conflicts(self, positions: List[Dict]) -> List[Dict[str, Any]]:
        """Check for positions that conflict with current signals."""
        conflicts = []
        symbols = [p['symbol'] for p in positions]

        if not symbols:
            return conflicts

        try:
            query = """
                    SELECT DISTINCT \
                    ON (ticker) ticker, signal_type, signal_strength, date
                    FROM trading_signals
                    WHERE ticker = ANY (%(symbols)s)
                    ORDER BY ticker, date DESC
                    """
            df = pd.read_sql(query, self.engine, params={"symbols": symbols})
            signal_map = {row['ticker']: row for _, row in df.iterrows()}

            for p in positions:
                symbol = p['symbol']
                signal_data = signal_map.get(symbol)

                if signal_data is None:
                    continue

                signal_type = signal_data['signal_type']
                signal_strength = signal_data['signal_strength']

                # Flag SELL signals
                if signal_strength <= -3:
                    days_old = (datetime.now().date() - signal_data['date']).days
                    conflicts.append({
                        'symbol': symbol,
                        'signal': signal_type,
                        'strength': signal_strength,
                        'weight': round(p.get('weight', 0), 1),
                        'pnl_pct': round(p.get('unrealized_pnl_pct', 0), 1),
                        'days_with_signal': days_old,
                        'message': f"{symbol} has {signal_type} signal ({days_old}d)"
                    })

        except Exception as e:
            logger.warning(f"Signal conflict check failed: {e}")

        return conflicts

    def _check_pnl_alerts(self, positions: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Check for profit taking and stop loss candidates."""
        profit_candidates = []
        stop_candidates = []

        for p in positions:
            pnl_pct = p.get('unrealized_pnl_pct', 0)
            symbol = p['symbol']

            if pnl_pct >= self.PROFIT_TAKING_THRESHOLD:
                profit_candidates.append({
                    'symbol': symbol,
                    'pnl_pct': round(pnl_pct, 1),
                    'pnl_amount': round(p.get('unrealized_pnl', 0), 2),
                    'weight': round(p.get('weight', 0), 1),
                    'message': f"{symbol} +{pnl_pct:.0f}% - consider profits"
                })

            elif pnl_pct <= self.STOP_LOSS_THRESHOLD:
                stop_candidates.append({
                    'symbol': symbol,
                    'pnl_pct': round(pnl_pct, 1),
                    'pnl_amount': round(p.get('unrealized_pnl', 0), 2),
                    'weight': round(p.get('weight', 0), 1),
                    'message': f"{symbol} {pnl_pct:.0f}% - review/stop-loss"
                })

        return profit_candidates, stop_candidates

    def _calculate_risk_score(self, concentration: Dict, sector: Dict,
                              beta: Dict, signal_conflicts: List,
                              profit_candidates: List, stop_candidates: List) -> Tuple[str, str]:
        """Calculate overall portfolio risk score."""
        score = 100

        # Concentration penalties
        largest_weight = concentration['largest'][1]
        if largest_weight > 20:
            score -= 25
        elif largest_weight > 15:
            score -= 15
        elif largest_weight > 10:
            score -= 10

        if concentration['top_5'] > 60:
            score -= 15
        elif concentration['top_5'] > 50:
            score -= 10

        # Sector penalties
        for sector_name, weight in sector['breakdown'].items():
            if sector_name != 'Unknown' and weight > 45:
                score -= 15
            elif sector_name != 'Unknown' and weight > 35:
                score -= 10

        # Beta penalties
        if beta['beta'] > 1.5:
            score -= 15
        elif beta['beta'] > 1.3:
            score -= 10

        # Signal conflict penalties (serious!)
        score -= len(signal_conflicts) * 8

        # Stop loss penalties
        score -= len(stop_candidates) * 5

        # Ensure in range
        score = max(0, min(100, score))

        # Grade
        if score >= 85:
            grade, level = 'A', 'Low'
        elif score >= 75:
            grade, level = 'B+', 'Low'
        elif score >= 65:
            grade, level = 'B', 'Medium'
        elif score >= 55:
            grade, level = 'C+', 'Medium'
        elif score >= 45:
            grade, level = 'C', 'Medium-High'
        elif score >= 35:
            grade, level = 'D', 'High'
        else:
            grade, level = 'F', 'High'

        return grade, level

    def _empty_metrics(self) -> PortfolioRiskMetrics:
        """Return empty metrics."""
        return PortfolioRiskMetrics(
            top_5_concentration=0,
            top_10_concentration=0,
            largest_position=('N/A', 0),
            concentration_alerts=[],
            sector_breakdown={},
            sector_alerts=[],
            portfolio_beta=1.0,
            beta_interpretation="No positions",
            high_correlations=[],
            correlation_alerts=[],
            signal_conflicts=[],
            profit_taking_candidates=[],
            stop_loss_candidates=[],
            risk_score='N/A',
            risk_level='N/A',
            total_alerts=0
        )

    def get_all_alerts(self, metrics: PortfolioRiskMetrics) -> List[Dict[str, Any]]:
        """Get all alerts as flat list for display."""
        alerts = []

        # Signal conflicts first (most important)
        for c in metrics.signal_conflicts:
            alerts.append({
                'type': 'signal_conflict',
                'severity': 'high',
                'symbol': c['symbol'],
                'message': c['message'],
                'icon': '🚨'
            })

        # Stop loss candidates
        for s in metrics.stop_loss_candidates:
            alerts.append({
                'type': 'stop_loss',
                'severity': 'high',
                'symbol': s['symbol'],
                'message': s['message'],
                'icon': '⛔'
            })

        # Concentration alerts
        for a in metrics.concentration_alerts:
            alerts.append({
                'type': 'concentration',
                'severity': a.severity,
                'symbol': a.symbol,
                'message': a.message,
                'icon': '⚖️'
            })

        # Sector alerts
        for a in metrics.sector_alerts:
            alerts.append({
                'type': 'sector',
                'severity': a.severity,
                'symbol': a.symbol,
                'message': a.message,
                'icon': '📊'
            })

        # Correlation alerts
        for a in metrics.correlation_alerts:
            alerts.append({
                'type': 'correlation',
                'severity': a.severity,
                'symbol': a.symbol,
                'message': a.message,
                'icon': '🔗'
            })

        # Profit taking (lower priority)
        for p in metrics.profit_taking_candidates:
            alerts.append({
                'type': 'profit_taking',
                'severity': 'low',
                'symbol': p['symbol'],
                'message': p['message'],
                'icon': '💰'
            })

        return alerts

    def generate_risk_summary(self, metrics: PortfolioRiskMetrics) -> str:
        """Generate text summary for AI context."""
        lines = []

        lines.append(f"PORTFOLIO RISK SUMMARY")
        lines.append(f"Risk Score: {metrics.risk_score} | Risk Level: {metrics.risk_level}")
        lines.append(f"Total Alerts: {metrics.total_alerts}")
        lines.append("")

        # Concentration
        lines.append(f"CONCENTRATION:")
        lines.append(f"  Top 5 positions: {metrics.top_5_concentration}%")
        lines.append(f"  Top 10 positions: {metrics.top_10_concentration}%")
        lines.append(f"  Largest: {metrics.largest_position[0]} ({metrics.largest_position[1]}%)")

        # Sectors (top 5 only)
        lines.append(f"\nSECTOR EXPOSURE:")
        for sector, weight in list(metrics.sector_breakdown.items())[:5]:
            flag = "⚠️" if weight > 30 else ""
            lines.append(f"  {flag}{sector}: {weight}%")

        # Beta
        lines.append(f"\nPORTFOLIO BETA: {metrics.portfolio_beta}")
        lines.append(f"  {metrics.beta_interpretation}")

        # Critical alerts
        if metrics.signal_conflicts:
            lines.append(f"\n🚨 SIGNAL CONFLICTS ({len(metrics.signal_conflicts)}):")
            for c in metrics.signal_conflicts[:5]:
                lines.append(f"  • {c['symbol']}: {c['signal']} signal")

        if metrics.stop_loss_candidates:
            lines.append(f"\n⛔ STOP-LOSS REVIEW ({len(metrics.stop_loss_candidates)}):")
            for s in metrics.stop_loss_candidates[:5]:
                lines.append(f"  • {s['symbol']}: {s['pnl_pct']}%")

        if metrics.profit_taking_candidates:
            lines.append(f"\n💰 PROFIT TAKING ({len(metrics.profit_taking_candidates)}):")
            for p in metrics.profit_taking_candidates[:5]:
                lines.append(f"  • {p['symbol']}: +{p['pnl_pct']}%")

        if metrics.concentration_alerts:
            lines.append(f"\n⚖️ CONCENTRATION WARNINGS:")
            for a in metrics.concentration_alerts[:3]:
                lines.append(f"  • {a.message}")

        return "\n".join(lines)

    def generate_daily_questions(self, metrics: PortfolioRiskMetrics,
                                  positions: List[Dict]) -> List[str]:
        """Generate daily questions for the AI to ask the user."""
        questions = []

        # Largest position question
        if metrics.largest_position[1] > 15:
            sym, weight = metrics.largest_position
            questions.append(
                f"Would you buy {sym} today at current price? "
                f"It's {weight:.0f}% of your portfolio."
            )

        # Signal conflict questions
        for c in metrics.signal_conflicts[:2]:
            questions.append(
                f"{c['symbol']} has had a {c['signal']} signal for {c['days_with_signal']} days. "
                f"What's your plan?"
            )

        # Big winner question
        for p in metrics.profit_taking_candidates[:1]:
            questions.append(
                f"{p['symbol']} is up {p['pnl_pct']:.0f}%. "
                f"Have you considered taking partial profits?"
            )

        # Big loser question
        for s in metrics.stop_loss_candidates[:1]:
            questions.append(
                f"{s['symbol']} is down {abs(s['pnl_pct']):.0f}%. "
                f"Is your original thesis still valid?"
            )

        # Sector concentration question
        for sector, weight in metrics.sector_breakdown.items():
            if sector != 'Unknown' and weight > 40:
                questions.append(
                    f"Your {sector} exposure is {weight:.0f}%. "
                    f"Are you comfortable with this concentration?"
                )
                break

        return questions[:5]  # Max 5 questions


# Convenience function
def analyze_portfolio_risk(positions: List[Dict],
                           account_summary: Optional[Dict] = None) -> PortfolioRiskMetrics:
    """Quick function to analyze portfolio risk."""
    analyzer = RiskAnalyzer()
    return analyzer.analyze_portfolio(positions, account_summary)
