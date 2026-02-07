"""
Institutional Upgrade Integration Layer

This module shows how to integrate the new unified architecture
into the existing HH Research Platform.

Integration Steps:
1. Replace signal_engine.py calls with UnifiedScorer
2. Update backtest engine.py to use same scoring + costs
3. Add exposure controls to trade idea generation
4. Add PIT validation to data loading

Author: Alpha Research Platform
Location: src/core/integration.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

# Import new modules
from src.core.unified_scorer import (
    UnifiedScorer, 
    TickerFeatures, 
    ScoringResult,
    get_unified_scorer,
    score_ticker,
    score_universe
)
from src.backtest.transaction_costs import (
    TransactionCostModel,
    get_cost_model,
    get_round_trip_cost_bps,
    ROUND_TRIP_COST_TABLE
)
from src.portfolio.exposure_control import (
    ExposureController,
    ExposureLimits,
    ExposureReport,
    get_exposure_controller,
    analyze_portfolio_exposure
)
from src.data.pit_validator import (
    PITValidator,
    BacktestIntegrityChecker,
    get_pit_validator,
    validate_features_pit,
    check_backtest_integrity
)

logger = logging.getLogger(__name__)


# =============================================================================
# INTEGRATION ADAPTERS
# =============================================================================

class SignalEngineAdapter:
    """
    Adapter to replace existing signal_engine.py functionality
    with the new UnifiedScorer.
    
    This allows gradual migration - you can switch functions one at a time.
    """
    
    def __init__(self, repository=None):
        """
        Initialize adapter.
        
        Args:
            repository: Existing Repository instance for data access
        """
        self.repo = repository
        self.scorer = UnifiedScorer(repository)
        
    def generate_signal(self, 
                        ticker: str,
                        include_cross_sectional: bool = True) -> Dict[str, Any]:
        """
        Generate signal for a ticker.
        
        This replaces signal_engine.generate_unified_signal()
        
        Returns dict compatible with existing UnifiedSignal format.
        """
        as_of = datetime.now()
        
        # Get features and scores from unified scorer
        features = self.scorer.compute_features(ticker, as_of)
        
        # Get universe scores for cross-sectional if needed
        universe_stats = None
        if include_cross_sectional:
            universe_stats = self._get_universe_stats(as_of)
        
        result = self.scorer.compute_scores(features, universe_stats)
        
        # Convert to existing format for compatibility
        return self._convert_to_legacy_format(features, result)
    
    def generate_signals_batch(self, 
                               tickers: List[str],
                               include_cross_sectional: bool = True) -> List[Dict]:
        """
        Generate signals for multiple tickers.
        
        This replaces signal_engine.generate_all_signals()
        """
        as_of = datetime.now()
        results = self.scorer.score_universe(tickers, as_of, include_cross_sectional)
        
        return [
            self._convert_to_legacy_format(
                self.scorer.compute_features(r.ticker, as_of), r
            )
            for r in results
        ]
    
    def _get_universe_stats(self, as_of: datetime) -> Dict:
        """Get universe statistics for cross-sectional scoring."""
        # Load universe tickers
        if self.repo:
            tickers = self.repo.get_universe()
        else:
            tickers = []
        
        if not tickers:
            return None
        
        # Score all tickers (cached)
        scores = []
        for ticker in tickers[:50]:  # Sample for efficiency
            try:
                features = self.scorer.compute_features(ticker, as_of)
                result = self.scorer.compute_scores(features)
                scores.append(result.composite_score)
            except Exception:
                pass
        
        if not scores:
            return None
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores
        }
    
    def _convert_to_legacy_format(self, 
                                  features: TickerFeatures,
                                  result: ScoringResult) -> Dict:
        """Convert new format to existing UnifiedSignal-compatible format."""
        return {
            # Basic info
            'ticker': result.ticker,
            'company_name': '',  # Would need to fetch
            'sector': features.sector,
            'asset_type': 'STOCK',
            
            # Scores
            'today_score': int(result.total_score),
            'today_signal': result.signal_type,
            'longterm_score': int(result.composite_score),
            'longterm_signal': result.signal_type,
            
            # Component scores
            'sentiment_score': features.sentiment_score,
            'fundamental_score': features.fundamental_score,
            'technical_score': features.technical_score,
            'options_score': features.options_flow_score,
            'institutional_score': features.institutional_score,
            
            # Cross-sectional
            'composite_z': result.composite_z,
            'universe_rank': result.universe_rank,
            'universe_percentile': result.composite_percentile,
            
            # Adjustments
            'regime_adjustment': result.regime_adjustment,
            'earnings_adjustment': result.earnings_adjustment,
            
            # Confidence
            'confidence': result.confidence,
            'data_quality': result.data_quality.overall_quality.value if result.data_quality else 'unknown',
            
            # Price info
            'current_price': features.current_price,
            
            # Earnings
            'days_to_earnings': features.days_to_earnings,
            'ies_score': features.ies_score,
            
            # Metadata
            'generated_at': result.as_of_time.isoformat(),
            'scorer_version': result.scorer_version,
            'feature_hash': result.feature_hash,
        }


class BacktestEngineAdapter:
    """
    Adapter to add transaction costs and exposure controls
    to the existing backtest engine.
    
    This can be used to wrap existing engine.py functionality.
    """
    
    def __init__(self):
        self.cost_model = TransactionCostModel()
        self.pit_validator = PITValidator()
        self.integrity_checker = BacktestIntegrityChecker()
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate backtest data before running.
        
        Returns integrity report - fail if critical issues.
        """
        return self.integrity_checker.check_historical_scores(df)
    
    def apply_costs_to_trades(self, 
                              trades: List[Dict],
                              market_cap_lookup: Dict[str, float] = None) -> List[Dict]:
        """
        Apply transaction costs to trade results.
        
        Args:
            trades: List of trade dicts with 'ticker', 'return_pct', 'entry_price'
            market_cap_lookup: Optional dict of ticker -> market_cap
            
        Returns:
            Trades with 'return_pct_net' and 'cost_bps' added
        """
        market_cap_lookup = market_cap_lookup or {}
        
        for trade in trades:
            ticker = trade.get('ticker', 'UNKNOWN')
            entry_price = trade.get('entry_price', 100)
            
            # Estimate trade value
            trade_value = entry_price * 100  # Assume 100 shares
            
            # Get market cap for cost estimation
            market_cap = market_cap_lookup.get(ticker)
            
            # Calculate round-trip cost
            cost_bps = self.cost_model.estimate_round_trip_cost(
                ticker, trade_value, market_cap=market_cap
            )
            
            # Adjust return
            raw_return = trade.get('return_pct', 0)
            cost_pct = cost_bps / 100
            
            trade['return_pct_net'] = raw_return - cost_pct
            trade['cost_bps'] = cost_bps
        
        return trades
    
    def get_quick_cost_adjustment(self, 
                                  returns: np.ndarray,
                                  market_cap_tier: str = 'large') -> np.ndarray:
        """
        Quick cost adjustment for return arrays.
        
        Use this for fast backtesting when you don't need per-trade precision.
        """
        cost_bps = ROUND_TRIP_COST_TABLE.get(market_cap_tier.lower(), 15)
        cost_pct = cost_bps / 100
        
        return returns - cost_pct


class TradeIdeasAdapter:
    """
    Adapter to add exposure controls to trade idea generation.
    
    Integrates with existing trade_ideas.py
    """
    
    def __init__(self, limits: ExposureLimits = None):
        self.exposure_controller = ExposureController(limits or ExposureLimits())
        self.scorer = UnifiedScorer()
    
    def filter_by_exposure(self,
                           candidates: List[Dict],
                           current_positions: List[Dict]) -> List[Dict]:
        """
        Filter trade candidates based on exposure limits.
        
        Args:
            candidates: List of trade candidates from trade_ideas.py
            current_positions: Current portfolio positions
            
        Returns:
            Filtered candidates that won't breach exposure limits
        """
        # Get current exposure report
        exposure_report = self.exposure_controller.analyze_portfolio(current_positions)
        
        filtered = []
        
        for candidate in candidates:
            ticker = candidate.get('ticker', '')
            sector = candidate.get('sector', '')
            proposed_weight = candidate.get('proposed_weight', 0.05)
            
            # Check if adding this position would breach limits
            constrained_weight, constraints = self.exposure_controller.get_constrained_position_size(
                ticker, proposed_weight, current_positions, sector
            )
            
            if constrained_weight > 0:
                candidate['constrained_weight'] = constrained_weight
                candidate['weight_constraints'] = constraints
                filtered.append(candidate)
            else:
                logger.info(f"Filtered out {ticker}: would breach exposure limits")
        
        return filtered
    
    def rank_with_exposure_adjustment(self,
                                      candidates: List[Dict],
                                      current_positions: List[Dict]) -> List[Dict]:
        """
        Re-rank candidates considering exposure impact.
        
        Candidates that improve diversification get bonus points.
        """
        exposure_report = self.exposure_controller.analyze_portfolio(current_positions)
        
        for candidate in candidates:
            sector = candidate.get('sector', '')
            
            # Base score from trade ideas — skip if no AI score available
            base_score = candidate.get('ai_score')
            if base_score is None:
                candidate['exposure_adjusted_score'] = None
                candidate['exposure_adjustment'] = 0
                continue
            adjustment = 0
            
            # Bonus for sectors with low exposure
            for exp in exposure_report.sector_exposures:
                if exp.sector == sector:
                    if exp.weight_pct < 10:  # Underweight sector
                        adjustment += 5
                    elif exp.weight_pct > 25:  # Overweight sector
                        adjustment -= 10
                    break
            
            # Bonus for low correlation with existing positions
            # (simplified - would need actual correlation calculation)
            
            candidate['exposure_adjusted_score'] = base_score + adjustment
            candidate['exposure_adjustment'] = adjustment
        
        # Re-sort by adjusted score (None-scored candidates sorted to bottom)
        candidates.sort(key=lambda x: x.get('exposure_adjusted_score') or 0, reverse=True)
        
        return candidates


# =============================================================================
# MIGRATION HELPERS
# =============================================================================

def migrate_to_unified_scoring():
    """
    Instructions for migrating to unified scoring.
    
    Step 1: Replace imports
    -----------------------
    OLD:
        from src.screener.signal_engine import SignalEngine
        engine = SignalEngine()
        signal = engine.generate_unified_signal(ticker)
    
    NEW:
        from src.core.integration import SignalEngineAdapter
        adapter = SignalEngineAdapter(repository)
        signal = adapter.generate_signal(ticker)
    
    Step 2: Update backtest engine
    ------------------------------
    In engine.py, add after calculating returns:
    
        from src.core.integration import BacktestEngineAdapter
        adapter = BacktestEngineAdapter()
        
        # Validate data first
        integrity = adapter.validate_data(df)
        if not integrity['is_valid']:
            raise ValueError(f"Data integrity issues: {integrity['issues']}")
        
        # Apply costs to trades
        trades = adapter.apply_costs_to_trades(trades)
    
    Step 3: Add exposure controls to trade ideas
    --------------------------------------------
    In trade_ideas.py generate_ideas():
    
        from src.core.integration import TradeIdeasAdapter
        adapter = TradeIdeasAdapter()
        
        # Filter by exposure
        candidates = adapter.filter_by_exposure(candidates, portfolio_positions)
        
        # Rank with exposure adjustment
        candidates = adapter.rank_with_exposure_adjustment(candidates, portfolio_positions)
    
    Step 4: Verify consistency
    --------------------------
    Run both old and new scoring on same data, compare results:
    
        from src.core.unified_scorer import score_ticker
        old_signal = old_engine.generate_unified_signal('AAPL')
        new_signal = score_ticker('AAPL')
        
        # Compare scores
        assert abs(old_signal['today_score'] - new_signal.total_score) < 5
    """
    pass


def validate_migration(ticker: str, old_engine, repository) -> Dict:
    """
    Validate that new scoring matches old scoring within tolerance.
    
    Returns comparison report.
    """
    adapter = SignalEngineAdapter(repository)
    
    # Get old signal
    old_signal = old_engine.generate_unified_signal(ticker)
    
    # Get new signal
    new_signal = adapter.generate_signal(ticker)
    
    # Compare
    comparisons = {
        'ticker': ticker,
        'matches': True,
        'differences': []
    }
    
    fields_to_compare = [
        ('today_score', 'today_score', 5),
        ('sentiment_score', 'sentiment_score', 3),
        ('fundamental_score', 'fundamental_score', 3),
        ('technical_score', 'technical_score', 3),
    ]
    
    for old_field, new_field, tolerance in fields_to_compare:
        old_val = old_signal.get(old_field, 50)
        new_val = new_signal.get(new_field, 50)
        
        if old_val is None:
            old_val = 50
        if new_val is None:
            new_val = 50
        
        diff = abs(old_val - new_val)
        
        if diff > tolerance:
            comparisons['matches'] = False
            comparisons['differences'].append({
                'field': old_field,
                'old': old_val,
                'new': new_val,
                'diff': diff
            })
    
    return comparisons


# =============================================================================
# UNIFIED PIPELINE FUNCTION
# =============================================================================

def run_unified_pipeline(tickers: List[str],
                         current_positions: List[Dict] = None,
                         include_costs: bool = True,
                         include_exposure: bool = True) -> Dict[str, Any]:
    """
    Run the complete unified pipeline:
    1. Score all tickers with unified scorer
    2. Apply exposure constraints
    3. Estimate transaction costs
    4. Return ranked opportunities
    
    This is the "single source of truth" function that should be used
    by live dashboard, trade ideas, AND backtesting.
    
    Args:
        tickers: List of tickers to analyze
        current_positions: Current portfolio positions
        include_costs: Whether to estimate transaction costs
        include_exposure: Whether to apply exposure constraints
        
    Returns:
        Dict with:
        - 'opportunities': List of ranked opportunities
        - 'exposure_report': Current exposure analysis
        - 'data_quality': Data quality summary
    """
    current_positions = current_positions or []
    as_of = datetime.now()
    
    # Initialize components
    scorer = UnifiedScorer()
    cost_model = TransactionCostModel() if include_costs else None
    exposure_controller = ExposureController() if include_exposure else None
    
    # Score all tickers
    scores = scorer.score_universe(tickers, as_of, include_cross_sectional=True)
    
    # Build opportunities list
    opportunities = []
    
    for result in scores:
        opp = {
            'ticker': result.ticker,
            'score': result.total_score,
            'signal': result.signal_type,
            'confidence': result.confidence,
            'rank': result.universe_rank,
            'percentile': result.composite_percentile,
            'z_score': result.composite_z,
        }
        
        # Add component scores
        opp['components'] = {
            'sentiment': result.sentiment_score,
            'fundamental': result.fundamental_score,
            'technical': result.technical_score,
            'options': result.options_flow_score,
            'institutional': result.institutional_score,
        }
        
        # Add adjustments
        opp['adjustments'] = {
            'regime': result.regime_adjustment,
            'earnings': result.earnings_adjustment,
        }
        
        # Add cost estimate if enabled
        if cost_model:
            # Assume $10K position for cost estimation
            cost = cost_model.estimate_cost(result.ticker, 10000)
            opp['cost_bps'] = cost.total_cost_bps
            opp['cost_tier'] = cost.market_cap_tier
        
        opportunities.append(opp)
    
    # Apply exposure constraints if enabled
    if exposure_controller and current_positions:
        exposure_report = exposure_controller.analyze_portfolio(current_positions)
        
        # Filter and adjust opportunities
        for opp in opportunities:
            ticker = opp['ticker']
            sector = opp.get('sector', '')
            
            constrained_weight, constraints = exposure_controller.get_constrained_position_size(
                ticker, 0.05, current_positions, sector
            )
            
            opp['max_weight'] = constrained_weight
            opp['weight_constraints'] = constraints
    else:
        exposure_report = None
    
    # Sort by score
    opportunities.sort(key=lambda x: x['score'], reverse=True)
    
    # Data quality summary
    data_quality = {
        'total_scored': len(scores),
        'high_confidence': sum(1 for s in scores if s.confidence >= 0.8),
        'low_confidence': sum(1 for s in scores if s.confidence < 0.5),
    }
    
    return {
        'opportunities': opportunities,
        'exposure_report': exposure_report,
        'data_quality': data_quality,
        'as_of': as_of.isoformat(),
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test the unified pipeline
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    test_positions = [
        {'symbol': 'AAPL', 'weight': 0.15, 'sector': 'Technology'},
        {'symbol': 'MSFT', 'weight': 0.12, 'sector': 'Technology'},
    ]
    
    print("Running unified pipeline...")
    result = run_unified_pipeline(
        test_tickers,
        current_positions=test_positions,
        include_costs=True,
        include_exposure=True
    )
    
    print(f"\nScored {result['data_quality']['total_scored']} tickers")
    print(f"High confidence: {result['data_quality']['high_confidence']}")
    print(f"Low confidence: {result['data_quality']['low_confidence']}")
    
    print("\nTop opportunities:")
    for opp in result['opportunities'][:5]:
        print(f"  {opp['ticker']}: {opp['score']:.0f} ({opp['signal']}) "
              f"[{opp.get('cost_bps', 0):.0f} bps, max {opp.get('max_weight', 0.05):.1%}]")
    
    if result['exposure_report']:
        print(f"\nExposure Report:")
        print(f"  Beta: {result['exposure_report'].portfolio_beta:.2f}")
        print(f"  Volatility: {result['exposure_report'].portfolio_volatility:.1%}")
        print(f"  Compliant: {result['exposure_report'].is_compliant()}")
