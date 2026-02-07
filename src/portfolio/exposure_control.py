"""
Exposure Control Module

Portfolio exposure management for institutional-grade risk control.

Controls:
1. Beta exposure (market sensitivity)
2. Sector concentration limits
3. Correlation clustering (hidden concentration)
4. Volatility targeting
5. Factor exposure monitoring

For mega-caps, many "alpha-looking" signals are actually:
- Nasdaq beta
- Rates/real-yield exposure
- USD sensitivity
- AI/semis factor exposure

This module helps separate true alpha from factor bets.

Author: Alpha Research Platform
Location: src/portfolio/exposure_control.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ExposureStatus(Enum):
    """Status of exposure relative to limits."""
    OK = "ok"
    WARNING = "warning"
    BREACH = "breach"


@dataclass
class FactorExposure:
    """Exposure to a specific factor."""
    factor_name: str
    exposure: float          # Beta or loading to factor
    contribution_pct: float  # Contribution to portfolio variance
    status: ExposureStatus
    limit: float
    
    def is_significant(self, threshold: float = 0.1) -> bool:
        """Check if exposure is significant."""
        return abs(self.exposure) >= threshold


@dataclass
class SectorExposure:
    """Exposure to a sector."""
    sector: str
    weight_pct: float
    position_count: int
    largest_position: str
    largest_position_pct: float
    status: ExposureStatus
    limit: float


@dataclass
class CorrelationCluster:
    """Group of correlated positions."""
    cluster_id: int
    tickers: List[str]
    avg_correlation: float
    combined_weight_pct: float
    effective_positions: float  # Weighted effective N
    risk_contribution_pct: float
    warning: Optional[str] = None


@dataclass 
class ExposureReport:
    """Complete exposure report for a portfolio."""
    as_of_time: datetime
    
    # Market exposure — FIXED: Optional (None when data unavailable)
    portfolio_beta: Optional[float]
    beta_status: ExposureStatus
    
    # Volatility — FIXED: Optional (None when data unavailable)
    portfolio_volatility: Optional[float]
    target_volatility: float
    vol_scaling_factor: Optional[float]  # FIXED: Optional
    
    # Sector exposures
    sector_exposures: List[SectorExposure]
    sector_breaches: List[str]
    
    # Factor exposures
    factor_exposures: List[FactorExposure]
    dominant_factors: List[str]
    
    # Correlation clusters
    correlation_clusters: List[CorrelationCluster]
    hidden_concentrations: List[str]
    
    # Concentration
    hhi_index: float  # Herfindahl-Hirschman Index
    effective_positions: float  # 1/HHI
    top_10_weight: float
    
    # Risk metrics — FIXED: Optional (None when vol data unavailable)
    var_95_pct: Optional[float]
    max_drawdown_estimate: Optional[float]
    
    # Recommendations
    warnings: List[str] = field(default_factory=list)
    required_actions: List[str] = field(default_factory=list)
    
    def is_compliant(self) -> bool:
        """Check if all exposures within limits."""
        return (
            self.beta_status != ExposureStatus.BREACH and
            len(self.sector_breaches) == 0 and
            all(f.status != ExposureStatus.BREACH for f in self.factor_exposures)
        )


@dataclass
class ExposureLimits:
    """Configuration for exposure limits."""
    # Beta limits
    max_beta: float = 1.5
    min_beta: float = 0.5
    target_beta: float = 1.0
    
    # Volatility targeting
    target_volatility: float = 0.15  # 15% annualized
    max_volatility: float = 0.25
    min_volatility: float = 0.08
    
    # Sector limits
    max_sector_weight: float = 0.30  # 30% max in any sector
    max_single_position: float = 0.10  # 10% max in any position
    
    # Concentration
    max_hhi: float = 0.15  # Equivalent to ~7 effective positions minimum
    min_positions: int = 10
    max_top_10_weight: float = 0.60  # Top 10 positions max 60%
    
    # Correlation
    max_cluster_weight: float = 0.40  # Max weight in correlated cluster
    correlation_threshold: float = 0.70  # Threshold for "high correlation"
    
    # Factor limits
    factor_limits: Dict[str, float] = field(default_factory=lambda: {
        'nasdaq_beta': 1.5,
        'rates_sensitivity': 0.5,
        'usd_sensitivity': 0.3,
        'momentum': 0.5,
        'value': 0.5,
        'size': 0.5,
    })


# =============================================================================
# EXPOSURE CONTROLLER
# =============================================================================

class ExposureController:
    """
    Controls and monitors portfolio exposures.
    
    Usage:
        controller = ExposureController(limits)
        
        # Check current exposures
        report = controller.analyze_portfolio(positions)
        
        # Get position sizing with exposure constraints
        size = controller.get_constrained_position_size(
            ticker, proposed_weight, current_positions
        )
        
        # Suggest rebalancing trades
        trades = controller.get_rebalancing_trades(positions)
    """
    
    # Sector classification for common mega-caps
    SECTOR_OVERRIDES = {
        'AAPL': 'Technology',
        'MSFT': 'Technology',
        'GOOGL': 'Technology',
        'AMZN': 'Consumer Discretionary',
        'NVDA': 'Technology',
        'META': 'Technology',
        'TSLA': 'Consumer Discretionary',
        'JPM': 'Financials',
        'V': 'Financials',
        'JNJ': 'Healthcare',
        'UNH': 'Healthcare',
        'XOM': 'Energy',
        'PG': 'Consumer Staples',
    }
    
    # Factor proxies
    FACTOR_PROXIES = {
        'nasdaq_beta': 'QQQ',
        'rates_sensitivity': 'TLT',
        'usd_sensitivity': 'UUP',
        'momentum': 'MTUM',
        'value': 'VTV',
        'size': 'IWM',
    }
    
    def __init__(self, limits: ExposureLimits = None):
        """
        Initialize exposure controller.
        
        Args:
            limits: ExposureLimits configuration
        """
        self.limits = limits or ExposureLimits()
        self._price_cache = {}
        self._returns_cache = None
        
    def analyze_portfolio(self,
                          positions: List[Dict],
                          historical_days: int = 252) -> ExposureReport:
        """
        Analyze current portfolio exposures.
        
        Args:
            positions: List of position dicts with:
                - symbol: ticker
                - weight: portfolio weight (0-1)
                - market_value: dollar value
                - sector: sector classification
            historical_days: Days of history for calculations
            
        Returns:
            ExposureReport with complete exposure analysis
        """
        report = ExposureReport(
            as_of_time=datetime.now(),
            portfolio_beta=None,        # FIXED: was 1.0
            beta_status=ExposureStatus.OK,
            portfolio_volatility=None,  # FIXED: was 0.15
            target_volatility=self.limits.target_volatility,
            vol_scaling_factor=None,    # FIXED: was 1.0
            sector_exposures=[],
            sector_breaches=[],
            factor_exposures=[],
            dominant_factors=[],
            correlation_clusters=[],
            hidden_concentrations=[],
            hhi_index=0,
            effective_positions=0,
            top_10_weight=0,
            var_95_pct=None,          # FIXED: was 0
            max_drawdown_estimate=None, # FIXED: was 0
        )
        
        if not positions:
            return report
        
        # Extract data
        symbols = [p.get('symbol', '') for p in positions]
        weights = np.array([p.get('weight', 0) for p in positions])
        
        # Normalize weights if they don't sum to 1
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        # =========================================================================
        # 1. MARKET BETA
        # =========================================================================
        try:
            computed_beta = self._calculate_portfolio_beta(
                symbols, weights, historical_days
            )
            report.portfolio_beta = computed_beta
            
            if computed_beta is None:
                report.beta_status = ExposureStatus.WARNING
                report.warnings.append(
                    "Beta: Data not available — cannot compute market sensitivity"
                )
            elif computed_beta > self.limits.max_beta:
                report.beta_status = ExposureStatus.BREACH
                report.warnings.append(
                    f"Beta {computed_beta:.2f} exceeds max {self.limits.max_beta}"
                )
            elif computed_beta < self.limits.min_beta:
                report.beta_status = ExposureStatus.WARNING
                report.warnings.append(
                    f"Beta {computed_beta:.2f} below min {self.limits.min_beta}"
                )
        except Exception as e:
            logger.warning(f"Error calculating beta: {e}")
        
        # =========================================================================
        # 2. VOLATILITY
        # =========================================================================
        try:
            computed_vol = self._calculate_portfolio_volatility(
                symbols, weights, historical_days
            )
            report.portfolio_volatility = computed_vol
            
            if computed_vol is None:
                report.warnings.append(
                    "Volatility: Data not available — cannot compute portfolio risk"
                )
            else:
                # Calculate scaling factor to hit target vol
                if computed_vol > 0:
                    report.vol_scaling_factor = (
                        self.limits.target_volatility / computed_vol
                    )
                    
                if computed_vol > self.limits.max_volatility:
                    report.warnings.append(
                        f"Volatility {computed_vol:.1%} exceeds max "
                        f"{self.limits.max_volatility:.1%}"
                    )
                    if report.vol_scaling_factor is not None:
                        report.required_actions.append(
                            f"Reduce position sizes by {1 - report.vol_scaling_factor:.1%}"
                        )
        except Exception as e:
            logger.warning(f"Error calculating volatility: {e}")
        
        # =========================================================================
        # 3. SECTOR EXPOSURE
        # =========================================================================
        sector_weights = {}
        sector_positions = {}
        
        for i, p in enumerate(positions):
            sector = p.get('sector', 'Unknown')
            # Use override if available
            symbol = p.get('symbol', '')
            if symbol in self.SECTOR_OVERRIDES:
                sector = self.SECTOR_OVERRIDES[symbol]
            
            weight = weights[i]
            
            if sector not in sector_weights:
                sector_weights[sector] = 0
                sector_positions[sector] = []
            
            sector_weights[sector] += weight
            sector_positions[sector].append((symbol, weight))
        
        for sector, weight in sector_weights.items():
            positions_in_sector = sector_positions[sector]
            positions_in_sector.sort(key=lambda x: x[1], reverse=True)
            
            largest = positions_in_sector[0] if positions_in_sector else ('', 0)
            
            status = ExposureStatus.OK
            if weight > self.limits.max_sector_weight:
                status = ExposureStatus.BREACH
                report.sector_breaches.append(sector)
            elif weight > self.limits.max_sector_weight * 0.8:
                status = ExposureStatus.WARNING
            
            report.sector_exposures.append(SectorExposure(
                sector=sector,
                weight_pct=weight * 100,
                position_count=len(positions_in_sector),
                largest_position=largest[0],
                largest_position_pct=largest[1] * 100,
                status=status,
                limit=self.limits.max_sector_weight * 100
            ))
        
        # Sort by weight
        report.sector_exposures.sort(key=lambda x: x.weight_pct, reverse=True)
        
        if report.sector_breaches:
            report.required_actions.append(
                f"Reduce exposure to: {', '.join(report.sector_breaches)}"
            )
        
        # =========================================================================
        # 4. FACTOR EXPOSURES
        # =========================================================================
        try:
            for factor, proxy in self.FACTOR_PROXIES.items():
                exposure = self._calculate_factor_exposure(
                    symbols, weights, proxy, historical_days
                )
                
                if exposure is None:
                    continue  # Skip factors we can't compute
                
                limit = self.limits.factor_limits.get(factor, 1.0)
                
                status = ExposureStatus.OK
                if abs(exposure) > limit:
                    status = ExposureStatus.BREACH
                elif abs(exposure) > limit * 0.8:
                    status = ExposureStatus.WARNING
                
                factor_exp = FactorExposure(
                    factor_name=factor,
                    exposure=exposure,
                    contribution_pct=0,  # Would need variance decomposition
                    status=status,
                    limit=limit
                )
                
                report.factor_exposures.append(factor_exp)
                
                if factor_exp.is_significant():
                    report.dominant_factors.append(f"{factor}: {exposure:.2f}")
        except Exception as e:
            logger.warning(f"Error calculating factor exposures: {e}")
        
        # =========================================================================
        # 5. CORRELATION CLUSTERS
        # =========================================================================
        try:
            report.correlation_clusters = self._find_correlation_clusters(
                symbols, weights, historical_days
            )
            
            for cluster in report.correlation_clusters:
                if cluster.combined_weight_pct > self.limits.max_cluster_weight * 100:
                    report.hidden_concentrations.append(
                        f"Correlated group ({', '.join(cluster.tickers[:3])}...): "
                        f"{cluster.combined_weight_pct:.1f}%"
                    )
                    cluster.warning = f"Exceeds {self.limits.max_cluster_weight:.0%} cluster limit"
        except Exception as e:
            logger.warning(f"Error finding correlation clusters: {e}")
        
        # =========================================================================
        # 6. CONCENTRATION METRICS
        # =========================================================================
        # HHI (Herfindahl-Hirschman Index)
        report.hhi_index = (weights ** 2).sum()
        report.effective_positions = 1 / report.hhi_index if report.hhi_index > 0 else 0
        
        # Top 10 weight
        sorted_weights = np.sort(weights)[::-1]
        report.top_10_weight = sorted_weights[:10].sum()
        
        if report.hhi_index > self.limits.max_hhi:
            report.warnings.append(
                f"Portfolio too concentrated: HHI={report.hhi_index:.3f}, "
                f"effective positions={report.effective_positions:.1f}"
            )
        
        if report.top_10_weight > self.limits.max_top_10_weight:
            report.warnings.append(
                f"Top 10 concentration: {report.top_10_weight:.1%} "
                f"(max {self.limits.max_top_10_weight:.1%})"
            )
        
        # =========================================================================
        # 7. RISK METRICS
        # =========================================================================
        try:
            if report.portfolio_volatility is not None:
                # VaR estimate (parametric)
                report.var_95_pct = report.portfolio_volatility * 1.65 / np.sqrt(252)
                # Max drawdown estimate (based on vol and horizon)
                report.max_drawdown_estimate = report.portfolio_volatility * 2.5
            else:
                report.var_95_pct = None
                report.max_drawdown_estimate = None
        except Exception as e:
            logger.warning(f"Error calculating risk metrics: {e}")
        
        return report
    
    def get_constrained_position_size(self,
                                      ticker: str,
                                      proposed_weight: float,
                                      current_positions: List[Dict],
                                      sector: str = None) -> Tuple[float, List[str]]:
        """
        Get position size constrained by exposure limits.
        
        Args:
            ticker: Stock to size
            proposed_weight: Proposed weight (0-1)
            current_positions: Current portfolio positions
            sector: Sector of the stock
            
        Returns:
            Tuple of (constrained_weight, list_of_constraints_applied)
        """
        constraints = []
        final_weight = proposed_weight
        
        # 1. Single position limit
        if final_weight > self.limits.max_single_position:
            final_weight = self.limits.max_single_position
            constraints.append(f"Capped at single position limit: {self.limits.max_single_position:.1%}")
        
        # 2. Sector limit
        if sector:
            current_sector_weight = sum(
                p.get('weight', 0) for p in current_positions
                if p.get('sector') == sector and p.get('symbol') != ticker
            )
            max_additional = self.limits.max_sector_weight - current_sector_weight
            
            if final_weight > max_additional and max_additional > 0:
                final_weight = max_additional
                constraints.append(f"Capped by sector limit: {sector} at {self.limits.max_sector_weight:.1%}")
        
        # 3. Concentration limit
        current_hhi = sum(p.get('weight', 0) ** 2 for p in current_positions)
        new_hhi = current_hhi + final_weight ** 2
        
        if new_hhi > self.limits.max_hhi:
            # Solve for max weight that keeps HHI under limit
            max_weight = np.sqrt(self.limits.max_hhi - current_hhi)
            if max_weight < final_weight and max_weight > 0:
                final_weight = max_weight
                constraints.append(f"Capped by concentration limit: HHI < {self.limits.max_hhi:.3f}")
        
        return final_weight, constraints
    
    def get_vol_adjusted_weights(self,
                                 positions: List[Dict],
                                 target_vol: float = None) -> Dict[str, float]:
        """
        Get weights adjusted to hit target volatility.
        
        Args:
            positions: Current positions
            target_vol: Target volatility (uses limits.target_volatility if None)
            
        Returns:
            Dict of ticker -> adjusted weight
        """
        target_vol = target_vol or self.limits.target_volatility
        
        # Calculate current portfolio vol
        symbols = [p.get('symbol', '') for p in positions]
        weights = np.array([p.get('weight', 0) for p in positions])
        
        current_vol = self._calculate_portfolio_volatility(symbols, weights)
        
        if current_vol <= 0:
            return {p.get('symbol'): p.get('weight', 0) for p in positions}
        
        # Scaling factor
        scale = target_vol / current_vol
        scale = min(1.5, max(0.5, scale))  # Cap scaling at 50% up/down
        
        return {
            p.get('symbol'): p.get('weight', 0) * scale
            for p in positions
        }
    
    # =========================================================================
    # CALCULATION HELPERS
    # =========================================================================
    
    def _get_returns(self, symbols: List[str], days: int) -> pd.DataFrame:
        """Fetch return data for symbols."""
        try:
            import yfinance as yf
            
            # Add SPY for beta calculations
            all_symbols = list(set(symbols + ['SPY']))
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 30)
            
            data = yf.download(
                all_symbols,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )['Close']
            
            if isinstance(data, pd.Series):
                data = data.to_frame(name=all_symbols[0])
            
            returns = data.pct_change().dropna()
            return returns
            
        except Exception as e:
            logger.error(f"Error fetching returns: {e}")
            return pd.DataFrame()
    
    def _calculate_portfolio_beta(self,
                                   symbols: List[str],
                                   weights: np.ndarray,
                                   days: int = 252) -> Optional[float]:
        """
        Calculate portfolio beta vs SPY.
        
        FIXED: Returns None when data is unavailable instead of defaulting to 1.0.
        A hardcoded beta=1.0 is dangerous because it hides actual market sensitivity.
        """
        returns = self._get_returns(symbols, days)
        
        if returns.empty or 'SPY' not in returns.columns:
            return None  # FIXED: was return 1.0
        
        spy_returns = returns['SPY']
        
        # Calculate weighted portfolio returns
        portfolio_returns = pd.Series(0, index=returns.index)
        
        for i, symbol in enumerate(symbols):
            if symbol in returns.columns:
                portfolio_returns += returns[symbol] * weights[i]
        
        # Calculate beta
        cov = portfolio_returns.cov(spy_returns)
        var = spy_returns.var()
        
        if var is None or var <= 0 or np.isnan(var):
            return None  # FIXED: was return 1.0
        
        beta = cov / var
        if np.isnan(beta) or np.isinf(beta):
            return None
        
        return float(beta)
    
    def _calculate_portfolio_volatility(self,
                                         symbols: List[str],
                                         weights: np.ndarray,
                                         days: int = 252) -> Optional[float]:
        """
        Calculate annualized portfolio volatility.
        
        FIXED: Returns None when data unavailable instead of defaulting to 0.15 (15%).
        A hardcoded volatility masks actual risk levels.
        """
        returns = self._get_returns(symbols, days)
        
        if returns.empty:
            return None  # FIXED: was return 0.15
        
        # Calculate weighted portfolio returns
        portfolio_returns = pd.Series(0, index=returns.index)
        
        for i, symbol in enumerate(symbols):
            if symbol in returns.columns:
                portfolio_returns += returns[symbol] * weights[i]
        
        # Annualized volatility
        daily_vol = portfolio_returns.std()
        
        if daily_vol is None or np.isnan(daily_vol):
            return None
        
        annual_vol = daily_vol * np.sqrt(252)
        
        return float(annual_vol)
    
    def _calculate_factor_exposure(self,
                                    symbols: List[str],
                                    weights: np.ndarray,
                                    factor_proxy: str,
                                    days: int = 252) -> Optional[float]:
        """
        Calculate portfolio exposure to a factor.
        
        FIXED: Returns None instead of 0.0 when data unavailable.
        """
        returns = self._get_returns(symbols + [factor_proxy], days)
        
        if returns.empty or factor_proxy not in returns.columns:
            return None  # FIXED: was return 0.0
        
        factor_returns = returns[factor_proxy]
        
        # Calculate weighted portfolio returns
        portfolio_returns = pd.Series(0, index=returns.index)
        
        for i, symbol in enumerate(symbols):
            if symbol in returns.columns:
                portfolio_returns += returns[symbol] * weights[i]
        
        # Calculate beta to factor
        cov = portfolio_returns.cov(factor_returns)
        var = factor_returns.var()
        
        return cov / var if var > 0 else None
    
    def _find_correlation_clusters(self,
                                    symbols: List[str],
                                    weights: np.ndarray,
                                    days: int = 252) -> List[CorrelationCluster]:
        """Find clusters of highly correlated positions."""
        returns = self._get_returns(symbols, days)
        
        if returns.empty or len(symbols) < 2:
            return []
        
        # Calculate correlation matrix
        corr_matrix = returns[symbols].corr()
        
        # Simple clustering: group positions with correlation > threshold
        clusters = []
        visited = set()
        cluster_id = 0
        
        for i, sym1 in enumerate(symbols):
            if sym1 in visited or sym1 not in corr_matrix.columns:
                continue
            
            cluster_members = [sym1]
            cluster_weights = [weights[i]]
            visited.add(sym1)
            
            for j, sym2 in enumerate(symbols):
                if sym2 in visited or sym2 not in corr_matrix.columns:
                    continue
                
                if sym1 in corr_matrix.columns and sym2 in corr_matrix.columns:
                    corr = corr_matrix.loc[sym1, sym2]
                    
                    if corr >= self.limits.correlation_threshold:
                        cluster_members.append(sym2)
                        cluster_weights.append(weights[j])
                        visited.add(sym2)
            
            if len(cluster_members) >= 2:
                avg_corr = 0
                n_pairs = 0
                
                for m1 in cluster_members:
                    for m2 in cluster_members:
                        if m1 != m2 and m1 in corr_matrix.columns and m2 in corr_matrix.columns:
                            avg_corr += corr_matrix.loc[m1, m2]
                            n_pairs += 1
                
                if n_pairs > 0:
                    avg_corr /= n_pairs
                
                clusters.append(CorrelationCluster(
                    cluster_id=cluster_id,
                    tickers=cluster_members,
                    avg_correlation=avg_corr,
                    combined_weight_pct=sum(cluster_weights) * 100,
                    effective_positions=len(cluster_members),
                    risk_contribution_pct=0  # Would need variance decomposition
                ))
                
                cluster_id += 1
        
        # Sort by weight
        clusters.sort(key=lambda x: x.combined_weight_pct, reverse=True)
        
        return clusters


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_controller_instance = None


def get_exposure_controller(limits: ExposureLimits = None) -> ExposureController:
    """Get singleton exposure controller."""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = ExposureController(limits)
    return _controller_instance


def analyze_portfolio_exposure(positions: List[Dict]) -> ExposureReport:
    """Quick access to portfolio exposure analysis."""
    return get_exposure_controller().analyze_portfolio(positions)


def get_position_constraints(ticker: str,
                             proposed_weight: float,
                             current_positions: List[Dict],
                             sector: str = None) -> Tuple[float, List[str]]:
    """Get constrained position size."""
    return get_exposure_controller().get_constrained_position_size(
        ticker, proposed_weight, current_positions, sector
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test with sample portfolio
    positions = [
        {"symbol": "AAPL", "weight": 0.15, "sector": "Technology"},
        {"symbol": "MSFT", "weight": 0.12, "sector": "Technology"},
        {"symbol": "GOOGL", "weight": 0.10, "sector": "Technology"},
        {"symbol": "NVDA", "weight": 0.08, "sector": "Technology"},
        {"symbol": "AMZN", "weight": 0.08, "sector": "Consumer Discretionary"},
        {"symbol": "META", "weight": 0.07, "sector": "Technology"},
        {"symbol": "JPM", "weight": 0.06, "sector": "Financials"},
        {"symbol": "JNJ", "weight": 0.05, "sector": "Healthcare"},
        {"symbol": "V", "weight": 0.05, "sector": "Financials"},
        {"symbol": "PG", "weight": 0.04, "sector": "Consumer Staples"},
        {"symbol": "XOM", "weight": 0.04, "sector": "Energy"},
        {"symbol": "HD", "weight": 0.04, "sector": "Consumer Discretionary"},
        {"symbol": "UNH", "weight": 0.04, "sector": "Healthcare"},
        {"symbol": "KO", "weight": 0.04, "sector": "Consumer Staples"},
        {"symbol": "DIS", "weight": 0.04, "sector": "Communication Services"},
    ]
    
    controller = ExposureController()
    report = controller.analyze_portfolio(positions)
    
    print(f"\n{'='*60}")
    print("EXPOSURE REPORT")
    print(f"{'='*60}")
    print(f"\nMarket Exposure:")
    print(f"  Beta: {report.portfolio_beta:.2f} ({report.beta_status.value})")
    print(f"  Volatility: {report.portfolio_volatility:.1%}")
    print(f"  Vol Scaling Factor: {report.vol_scaling_factor:.2f}")
    
    print(f"\nSector Exposures:")
    for exp in report.sector_exposures[:5]:
        print(f"  {exp.sector}: {exp.weight_pct:.1f}% ({exp.status.value})")
    
    print(f"\nFactor Exposures:")
    for exp in report.factor_exposures:
        if exp.is_significant():
            print(f"  {exp.factor_name}: {exp.exposure:.2f} ({exp.status.value})")
    
    print(f"\nCorrelation Clusters:")
    for cluster in report.correlation_clusters[:3]:
        print(f"  {', '.join(cluster.tickers[:3])}: {cluster.combined_weight_pct:.1f}% "
              f"(avg corr: {cluster.avg_correlation:.2f})")
    
    print(f"\nConcentration:")
    print(f"  HHI: {report.hhi_index:.4f}")
    print(f"  Effective Positions: {report.effective_positions:.1f}")
    print(f"  Top 10 Weight: {report.top_10_weight:.1%}")
    
    print(f"\nRisk Metrics:")
    print(f"  VaR 95% (daily): {report.var_95_pct:.2%}")
    print(f"  Max Drawdown Estimate: {report.max_drawdown_estimate:.1%}")
    
    if report.warnings:
        print(f"\n⚠️ Warnings:")
        for w in report.warnings:
            print(f"  - {w}")
    
    if report.required_actions:
        print(f"\n🚨 Required Actions:")
        for a in report.required_actions:
            print(f"  - {a}")
    
    print(f"\n✅ Compliant: {report.is_compliant()}")
