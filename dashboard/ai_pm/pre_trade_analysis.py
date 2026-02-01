
# dashboard/ai_pm/pre_trade_analysis.py
"""
AI Pre-Trade Analysis
Analyzes proposed trades before execution using:
- Earnings calendar (proximity to announcements)
- Market regime (VIX, trend)
- Macro events (war, Fed, crisis)
- Stock signals (sentiment, technicals)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class WarningType(Enum):
    EARNINGS_IMMINENT = "EARNINGS_IMMINENT"
    EARNINGS_SOON = "EARNINGS_SOON"
    HIGH_VIX = "HIGH_VIX"
    BEAR_REGIME = "BEAR_REGIME"
    MACRO_EVENT = "MACRO_EVENT"
    NEGATIVE_SENTIMENT = "NEGATIVE_SENTIMENT"
    WEAK_SIGNAL = "WEAK_SIGNAL"
    OVERBOUGHT = "OVERBOUGHT"
    OVERSOLD = "OVERSOLD"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"


@dataclass
class TradeWarning:
    """Warning for a specific trade"""
    symbol: str
    warning_type: WarningType
    risk_level: RiskLevel
    message: str
    details: str = ""
    recommendation: str = ""  # PROCEED, WAIT, REDUCE_SIZE, CANCEL


@dataclass
class MarketCondition:
    """Current market conditions"""
    vix_level: float = 0.0
    vix_status: str = "NORMAL"  # NORMAL, ELEVATED, HIGH, EXTREME
    regime: str = "NEUTRAL"  # BULL, NEUTRAL, BEAR, CRISIS
    regime_confidence: int = 0
    active_events: List[str] = field(default_factory=list)
    overall_risk: RiskLevel = RiskLevel.LOW
    
    def to_dict(self) -> dict:
        return {
            'vix_level': self.vix_level,
            'vix_status': self.vix_status,
            'regime': self.regime,
            'regime_confidence': self.regime_confidence,
            'active_events': self.active_events,
            'overall_risk': self.overall_risk.value,
        }


@dataclass 
class PreTradeAnalysis:
    """Complete pre-trade analysis result"""
    timestamp: datetime
    market_condition: MarketCondition
    warnings: List[TradeWarning]
    trade_count: int
    blocked_count: int
    warning_count: int
    proceed_count: int
    overall_recommendation: str  # PROCEED, CAUTION, WAIT, ABORT
    summary: str
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'market_condition': self.market_condition.to_dict(),
            'warnings': [
                {
                    'symbol': w.symbol,
                    'type': w.warning_type.value,
                    'risk': w.risk_level.value,
                    'message': w.message,
                    'recommendation': w.recommendation,
                }
                for w in self.warnings
            ],
            'trade_count': self.trade_count,
            'blocked_count': self.blocked_count,
            'warning_count': self.warning_count,
            'proceed_count': self.proceed_count,
            'overall_recommendation': self.overall_recommendation,
            'summary': self.summary,
        }


class PreTradeAnalyzer:
    """Analyzes proposed trades before execution"""
    
    # Thresholds
    EARNINGS_IMMINENT_DAYS = 3  # Critical - earnings within 3 days
    EARNINGS_SOON_DAYS = 7  # Warning - earnings within 7 days
    VIX_ELEVATED = 20
    VIX_HIGH = 25
    VIX_EXTREME = 35
    
    def __init__(self):
        self._earnings_cache: Dict[str, Optional[datetime]] = {}
        self._regime_cache: Optional[Tuple[datetime, MarketCondition]] = None
    
    def analyze_trades(
        self,
        proposed_orders: List[dict],
        signals_snapshot: Optional[object] = None,
        force_refresh: bool = False,
    ) -> PreTradeAnalysis:
        """
        Analyze proposed trades and return warnings/recommendations
        
        Args:
            proposed_orders: List of OrderTicket dicts with symbol, action, quantity
            signals_snapshot: SignalSnapshot with current signals
            force_refresh: Force refresh of cached data
            
        Returns:
            PreTradeAnalysis with warnings and recommendations
        """
        timestamp = datetime.now()
        warnings: List[TradeWarning] = []
        
        # Get market conditions (cached, fast)
        market_condition = self._get_market_conditions(force_refresh)
        
        # Add market-wide warnings
        if market_condition.vix_status == "EXTREME":
            warnings.append(TradeWarning(
                symbol="MARKET",
                warning_type=WarningType.HIGH_VIX,
                risk_level=RiskLevel.CRITICAL,
                message=f"VIX at {market_condition.vix_level:.1f} - Extreme volatility",
                details="Market is in panic mode. Consider delaying non-urgent trades.",
                recommendation="WAIT",
            ))
        elif market_condition.vix_status == "HIGH":
            warnings.append(TradeWarning(
                symbol="MARKET",
                warning_type=WarningType.HIGH_VIX,
                risk_level=RiskLevel.HIGH,
                message=f"VIX at {market_condition.vix_level:.1f} - High volatility",
                details="Elevated market risk. Consider reducing position sizes.",
                recommendation="REDUCE_SIZE",
            ))
        
        if market_condition.regime == "BEAR":
            warnings.append(TradeWarning(
                symbol="MARKET",
                warning_type=WarningType.BEAR_REGIME,
                risk_level=RiskLevel.MEDIUM,
                message="Bear market regime detected",
                details="Market trend is negative. Be cautious with new long positions.",
                recommendation="CAUTION",
            ))
        elif market_condition.regime == "CRISIS":
            warnings.append(TradeWarning(
                symbol="MARKET",
                warning_type=WarningType.BEAR_REGIME,
                risk_level=RiskLevel.CRITICAL,
                message="Crisis regime detected",
                details="Major market stress. Consider defensive positioning.",
                recommendation="WAIT",
            ))
        
        # Add macro event warnings
        for event in market_condition.active_events[:3]:
            warnings.append(TradeWarning(
                symbol="MARKET",
                warning_type=WarningType.MACRO_EVENT,
                risk_level=RiskLevel.MEDIUM,
                message=f"Active macro event: {event}",
                details="Monitor this event for portfolio impact.",
                recommendation="CAUTION",
            ))
        
        # Get signals for all symbols - USE SIGNALS FOR EARNINGS DATA (already loaded, fast!)
        signals_map = {}
        if signals_snapshot and hasattr(signals_snapshot, 'rows'):
            signals_map = signals_snapshot.rows or {}
        
        blocked_count = 0
        warning_count = 0
        
        for order in proposed_orders:
            symbol = order.get('symbol', '').upper()
            action = order.get('action', 'BUY')
            
            if not symbol:
                continue
            
            signal_row = signals_map.get(symbol)
            
            # Check earnings FROM SIGNALS (fast - data already loaded)
            earnings_warning = self._check_earnings_from_signal(symbol, action, signal_row)
            if earnings_warning:
                warnings.append(earnings_warning)
                if earnings_warning.risk_level == RiskLevel.CRITICAL:
                    blocked_count += 1
                else:
                    warning_count += 1
            
            # Check signals
            signal_warning = self._check_signals(symbol, action, signal_row)
            if signal_warning:
                warnings.append(signal_warning)
                warning_count += 1
        
        # Calculate overall recommendation
        proceed_count = len(proposed_orders) - blocked_count - warning_count
        
        if blocked_count > 0 or market_condition.overall_risk == RiskLevel.CRITICAL:
            overall_recommendation = "WAIT"
        elif warning_count > len(proposed_orders) * 0.3 or market_condition.overall_risk == RiskLevel.HIGH:
            overall_recommendation = "CAUTION"
        elif warning_count > 0:
            overall_recommendation = "PROCEED_WITH_CAUTION"
        else:
            overall_recommendation = "PROCEED"
        
        # Generate summary
        summary = self._generate_summary(
            len(proposed_orders), blocked_count, warning_count, 
            market_condition, warnings
        )
        
        return PreTradeAnalysis(
            timestamp=timestamp,
            market_condition=market_condition,
            warnings=warnings,
            trade_count=len(proposed_orders),
            blocked_count=blocked_count,
            warning_count=warning_count,
            proceed_count=proceed_count,
            overall_recommendation=overall_recommendation,
            summary=summary,
        )
    
    def _get_market_conditions(self, force_refresh: bool = False) -> MarketCondition:
        """Get current market conditions from macro regime detector - FAST version"""
        
        # Check cache (valid for 10 minutes to reduce API calls)
        if not force_refresh and self._regime_cache:
            cache_time, cached_condition = self._regime_cache
            if (datetime.now() - cache_time).seconds < 600:  # 10 min cache
                return cached_condition
        
        condition = MarketCondition()
        
        # Try to get regime (with timeout protection)
        try:
            from src.analytics.macro_regime import get_current_regime
            
            regime_result = get_current_regime()
            if regime_result:
                condition.regime = regime_result.regime.value if hasattr(regime_result.regime, 'value') else str(regime_result.regime)
                condition.regime_confidence = getattr(regime_result, 'confidence', 50)
                
                # Try to get VIX from regime result (faster than calling detector)
                if hasattr(regime_result, 'vix') and regime_result.vix:
                    condition.vix_level = float(regime_result.vix)
                elif hasattr(regime_result, 'indicators'):
                    for ind in (regime_result.indicators or []):
                        if hasattr(ind, 'name') and 'vix' in ind.name.lower():
                            condition.vix_level = float(ind.value) if ind.value else 0
                            break
                
                # Set VIX status
                if condition.vix_level > 0:
                    if condition.vix_level >= self.VIX_EXTREME:
                        condition.vix_status = "EXTREME"
                    elif condition.vix_level >= self.VIX_HIGH:
                        condition.vix_status = "HIGH"
                    elif condition.vix_level >= self.VIX_ELEVATED:
                        condition.vix_status = "ELEVATED"
                    else:
                        condition.vix_status = "NORMAL"
        except Exception as e:
            logger.debug(f"Could not get macro regime: {e}")
        
        # Skip macro events for speed (they're informational only)
        # Can be enabled later if needed
        
        # Calculate overall risk
        if condition.vix_status == "EXTREME" or "CRISIS" in condition.regime.upper():
            condition.overall_risk = RiskLevel.CRITICAL
        elif condition.vix_status == "HIGH" or "BEAR" in condition.regime.upper():
            condition.overall_risk = RiskLevel.HIGH
        elif condition.vix_status == "ELEVATED":
            condition.overall_risk = RiskLevel.MEDIUM
        else:
            condition.overall_risk = RiskLevel.LOW
        
        # Cache result
        self._regime_cache = (datetime.now(), condition)
        
        return condition
    
    def _get_earnings_dates(self, symbols: List[str]) -> Dict[str, Optional[datetime]]:
        """Get upcoming earnings dates for symbols - DEPRECATED
        
        NOTE: This method is slow as it fetches from external API.
        Use _check_earnings_from_signal() instead which uses pre-loaded signal data.
        """
        # Return empty - we now use signals data instead
        return {s: None for s in symbols}
    
    def _check_earnings_from_signal(
        self, 
        symbol: str, 
        action: str, 
        signal_row: Optional[object]
    ) -> Optional[TradeWarning]:
        """Check if earnings is imminent - uses data from signals (fast!)"""
        
        if not signal_row:
            return None
        
        raw = getattr(signal_row, 'raw', {}) or {}
        
        # Get days_to_earnings directly from signals (already calculated)
        days_until = raw.get('days_to_earnings')
        earnings_date_str = raw.get('earnings_date')
        
        if days_until is None:
            return None
        
        try:
            days_until = int(float(days_until))
        except (ValueError, TypeError):
            return None
        
        # Format date for display
        date_display = earnings_date_str if earnings_date_str else "soon"
        
        if days_until <= self.EARNINGS_IMMINENT_DAYS and days_until >= 0:
            return TradeWarning(
                symbol=symbol,
                warning_type=WarningType.EARNINGS_IMMINENT,
                risk_level=RiskLevel.CRITICAL if action == "BUY" else RiskLevel.HIGH,
                message=f"Earnings in {days_until} days ({date_display})",
                details="Stock may gap significantly after earnings. High risk of adverse move.",
                recommendation="WAIT" if action == "BUY" else "PROCEED",
            )
        elif days_until <= self.EARNINGS_SOON_DAYS and days_until >= 0:
            return TradeWarning(
                symbol=symbol,
                warning_type=WarningType.EARNINGS_SOON,
                risk_level=RiskLevel.MEDIUM,
                message=f"Earnings in {days_until} days ({date_display})",
                details="Consider waiting until after earnings or reducing position size.",
                recommendation="CAUTION",
            )
        
        return None
    
    def _check_earnings(
        self, 
        symbol: str, 
        action: str, 
        earnings_date: Optional[datetime]
    ) -> Optional[TradeWarning]:
        """Check if earnings is imminent - DEPRECATED: use _check_earnings_from_signal"""
        
        if not earnings_date:
            return None
        
        now = datetime.now()
        if hasattr(earnings_date, 'tzinfo') and earnings_date.tzinfo:
            from datetime import timezone
            now = datetime.now(timezone.utc)
        
        days_until = (earnings_date - now).days
        
        if days_until <= self.EARNINGS_IMMINENT_DAYS:
            return TradeWarning(
                symbol=symbol,
                warning_type=WarningType.EARNINGS_IMMINENT,
                risk_level=RiskLevel.CRITICAL if action == "BUY" else RiskLevel.HIGH,
                message=f"Earnings in {days_until} days ({earnings_date.strftime('%Y-%m-%d')})",
                details="Stock may gap significantly after earnings. High risk of adverse move.",
                recommendation="WAIT" if action == "BUY" else "PROCEED",
            )
        elif days_until <= self.EARNINGS_SOON_DAYS:
            return TradeWarning(
                symbol=symbol,
                warning_type=WarningType.EARNINGS_SOON,
                risk_level=RiskLevel.MEDIUM,
                message=f"Earnings in {days_until} days ({earnings_date.strftime('%Y-%m-%d')})",
                details="Consider waiting until after earnings or reducing position size.",
                recommendation="CAUTION",
            )
        
        return None
    
    def _check_signals(
        self, 
        symbol: str, 
        action: str, 
        signal_row: Optional[object]
    ) -> Optional[TradeWarning]:
        """Check signal quality for the trade"""
        
        if not signal_row:
            return None
        
        raw = getattr(signal_row, 'raw', {}) or {}
        
        # Get signal type - actual field name is 'signal_type'
        signal_type = str(raw.get('signal_type', raw.get('signal', ''))).upper()
        
        # Get scores - actual field names from signals table
        total_score = raw.get('total_score') or raw.get('composite_score')
        sentiment = raw.get('sentiment_score_score') or raw.get('sentiment_score') or raw.get('sentiment_weighted')
        
        # Get additional useful fields
        analyst_positivity = raw.get('analyst_positivity')
        target_upside = raw.get('target_upside_pct')
        days_to_earnings = raw.get('days_to_earnings')
        ai_probability = raw.get('ai_probability')
        
        # Check for conflicting signals
        if action == "BUY" and signal_type in ['SELL', 'STRONG_SELL', 'WEAK_SELL']:
            return TradeWarning(
                symbol=symbol,
                warning_type=WarningType.WEAK_SIGNAL,
                risk_level=RiskLevel.HIGH,
                message=f"BUY order conflicts with {signal_type} signal",
                details=f"Current signal is {signal_type} with score {total_score}.",
                recommendation="RECONSIDER",
            )
        
        if action == "SELL" and signal_type in ['STRONG_BUY']:
            return TradeWarning(
                symbol=symbol,
                warning_type=WarningType.WEAK_SIGNAL,
                risk_level=RiskLevel.MEDIUM,
                message=f"SELL order conflicts with {signal_type} signal",
                details=f"Current signal is {signal_type}. Consider keeping position.",
                recommendation="RECONSIDER",
            )
        
        # Check for low AI probability on buys
        if action == "BUY" and ai_probability is not None:
            try:
                prob_val = float(ai_probability)
                if prob_val < 0.4:  # Less than 40% AI probability
                    return TradeWarning(
                        symbol=symbol,
                        warning_type=WarningType.WEAK_SIGNAL,
                        risk_level=RiskLevel.MEDIUM,
                        message=f"Low AI probability ({prob_val:.0%})",
                        details="AI model gives low probability for this position.",
                        recommendation="CAUTION",
                    )
            except:
                pass
        
        # Check for negative sentiment on buys
        if action == "BUY" and sentiment is not None:
            try:
                sent_val = float(sentiment)
                if sent_val < 30:
                    return TradeWarning(
                        symbol=symbol,
                        warning_type=WarningType.NEGATIVE_SENTIMENT,
                        risk_level=RiskLevel.MEDIUM,
                        message=f"Negative sentiment ({sent_val:.0f}/100)",
                        details="Recent news sentiment is negative.",
                        recommendation="CAUTION",
                    )
            except:
                pass
        
        # Check for negative target upside on buys
        if action == "BUY" and target_upside is not None:
            try:
                upside_val = float(target_upside)
                if upside_val < -5:  # Analysts expect >5% downside
                    return TradeWarning(
                        symbol=symbol,
                        warning_type=WarningType.WEAK_SIGNAL,
                        risk_level=RiskLevel.MEDIUM,
                        message=f"Negative analyst target ({upside_val:.1f}%)",
                        details="Analyst price targets suggest downside.",
                        recommendation="CAUTION",
                    )
            except:
                pass
        
        # Check for low analyst positivity on buys
        if action == "BUY" and analyst_positivity is not None:
            try:
                pos_val = float(analyst_positivity)
                if pos_val < 30:  # Less than 30% positive ratings
                    return TradeWarning(
                        symbol=symbol,
                        warning_type=WarningType.WEAK_SIGNAL,
                        risk_level=RiskLevel.MEDIUM,
                        message=f"Low analyst support ({pos_val:.0f}% positive)",
                        details="Most analysts are neutral or negative.",
                        recommendation="CAUTION",
                    )
            except:
                pass
        
        return None
    
    def _generate_summary(
        self,
        trade_count: int,
        blocked_count: int,
        warning_count: int,
        market_condition: MarketCondition,
        warnings: List[TradeWarning],
    ) -> str:
        """Generate human-readable summary"""
        
        lines = []
        
        # Market summary
        lines.append(f"**Market Conditions:** {market_condition.regime} regime, VIX {market_condition.vix_level:.1f} ({market_condition.vix_status})")
        
        if market_condition.active_events:
            lines.append(f"**Active Events:** {', '.join(market_condition.active_events[:3])}")
        
        # Trade summary
        lines.append(f"**Trades:** {trade_count} proposed | {blocked_count} blocked | {warning_count} warnings | {trade_count - blocked_count - warning_count} clear")
        
        # Key warnings
        critical_warnings = [w for w in warnings if w.risk_level == RiskLevel.CRITICAL]
        if critical_warnings:
            lines.append(f"**⚠️ Critical:** {len(critical_warnings)} trades need attention")
            for w in critical_warnings[:3]:
                lines.append(f"  - {w.symbol}: {w.message}")
        
        return "\n".join(lines)


# Singleton instance
_analyzer: Optional[PreTradeAnalyzer] = None

def get_pre_trade_analyzer() -> PreTradeAnalyzer:
    """Get singleton analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = PreTradeAnalyzer()
    return _analyzer


def analyze_proposed_trades(
    orders: List[dict],
    signals_snapshot: Optional[object] = None,
) -> PreTradeAnalysis:
    """Convenience function to analyze trades"""
    return get_pre_trade_analyzer().analyze_trades(orders, signals_snapshot)
