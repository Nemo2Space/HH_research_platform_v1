"""
Alpha Decay Tracker

Measures how signals decay over time to determine:
1. Optimal holding periods
2. Signal half-life by type
3. When to exit positions
4. Urgency of execution

For mega-caps, many signals decay within hours. This module
tracks decay curves and recommends horizons.

Author: Alpha Research Platform
Location: src/analytics/alpha_decay.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of signals for decay analysis."""
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    OPTIONS_FLOW = "options_flow"
    EARNINGS = "earnings"
    FUNDAMENTAL = "fundamental"
    COMPOSITE = "composite"


class DecaySpeed(Enum):
    """Classification of decay speed."""
    FAST = "fast"         # Half-life < 1 day
    MEDIUM = "medium"     # Half-life 1-5 days
    SLOW = "slow"         # Half-life 5-20 days
    PERSISTENT = "persistent"  # Half-life > 20 days


@dataclass
class DecayProfile:
    """Decay profile for a signal type."""
    signal_type: SignalType
    half_life_hours: float  # Time for signal to lose half its predictive power
    decay_speed: DecaySpeed
    
    # IC (Information Coefficient) at different horizons
    ic_1h: float = 0.0
    ic_6h: float = 0.0
    ic_24h: float = 0.0
    ic_48h: float = 0.0
    ic_5d: float = 0.0
    ic_10d: float = 0.0
    
    # Optimal horizons
    optimal_entry_window: str = "0-2h"
    optimal_exit_window: str = "24-48h"
    
    # Sample size
    sample_count: int = 0
    last_updated: Optional[datetime] = None
    
    def get_ic_at_horizon(self, hours: float) -> float:
        """Interpolate IC at a specific horizon."""
        # Simple linear interpolation
        horizons = [1, 6, 24, 48, 120, 240]
        ics = [self.ic_1h, self.ic_6h, self.ic_24h, self.ic_48h, self.ic_5d, self.ic_10d]
        
        if hours <= horizons[0]:
            return ics[0]
        if hours >= horizons[-1]:
            return ics[-1]
        
        for i in range(len(horizons) - 1):
            if horizons[i] <= hours <= horizons[i + 1]:
                # Linear interpolation
                ratio = (hours - horizons[i]) / (horizons[i + 1] - horizons[i])
                return ics[i] + ratio * (ics[i + 1] - ics[i])
        
        return 0.0
    
    def to_dict(self) -> Dict:
        return {
            'signal_type': self.signal_type.value,
            'half_life_hours': self.half_life_hours,
            'decay_speed': self.decay_speed.value,
            'ic_1h': self.ic_1h,
            'ic_6h': self.ic_6h,
            'ic_24h': self.ic_24h,
            'ic_48h': self.ic_48h,
            'ic_5d': self.ic_5d,
            'ic_10d': self.ic_10d,
            'optimal_entry_window': self.optimal_entry_window,
            'optimal_exit_window': self.optimal_exit_window,
            'sample_count': self.sample_count,
        }


@dataclass
class SignalSnapshot:
    """A snapshot of a signal at a point in time."""
    ticker: str
    signal_type: SignalType
    signal_value: float  # Normalized 0-100 or z-score
    timestamp: datetime
    
    # Forward returns at various horizons (filled in later)
    return_1h: Optional[float] = None
    return_6h: Optional[float] = None
    return_24h: Optional[float] = None
    return_48h: Optional[float] = None
    return_5d: Optional[float] = None
    return_10d: Optional[float] = None


class AlphaDecayTracker:
    """
    Tracks signal decay over time.
    
    Usage:
        tracker = AlphaDecayTracker()
        
        # Record signal snapshots
        tracker.record_signal('AAPL', SignalType.SENTIMENT, 75, datetime.now())
        
        # Later, fill in forward returns
        tracker.record_forward_returns('AAPL', timestamp, {
            '1h': 0.2, '6h': 0.5, '24h': 1.1, ...
        })
        
        # Calculate decay profiles
        profiles = tracker.calculate_decay_profiles()
        
        # Get recommendation for a signal
        rec = tracker.get_horizon_recommendation(SignalType.SENTIMENT)
    """
    
    # Default decay profiles (can be updated from data)
    DEFAULT_PROFILES = {
        SignalType.SENTIMENT: DecayProfile(
            signal_type=SignalType.SENTIMENT,
            half_life_hours=6,
            decay_speed=DecaySpeed.FAST,
            ic_1h=0.08,
            ic_6h=0.05,
            ic_24h=0.02,
            ic_48h=0.01,
            ic_5d=0.00,
            ic_10d=-0.01,
            optimal_entry_window="0-2h",
            optimal_exit_window="6-24h",
        ),
        SignalType.OPTIONS_FLOW: DecayProfile(
            signal_type=SignalType.OPTIONS_FLOW,
            half_life_hours=12,
            decay_speed=DecaySpeed.FAST,
            ic_1h=0.10,
            ic_6h=0.07,
            ic_24h=0.04,
            ic_48h=0.02,
            ic_5d=0.01,
            ic_10d=0.00,
            optimal_entry_window="0-4h",
            optimal_exit_window="12-48h",
        ),
        SignalType.TECHNICAL: DecayProfile(
            signal_type=SignalType.TECHNICAL,
            half_life_hours=48,
            decay_speed=DecaySpeed.MEDIUM,
            ic_1h=0.04,
            ic_6h=0.04,
            ic_24h=0.04,
            ic_48h=0.03,
            ic_5d=0.02,
            ic_10d=0.01,
            optimal_entry_window="0-24h",
            optimal_exit_window="2-5d",
        ),
        SignalType.EARNINGS: DecayProfile(
            signal_type=SignalType.EARNINGS,
            half_life_hours=24,
            decay_speed=DecaySpeed.MEDIUM,
            ic_1h=0.15,
            ic_6h=0.10,
            ic_24h=0.05,
            ic_48h=0.03,
            ic_5d=0.01,
            ic_10d=0.00,
            optimal_entry_window="0-6h",
            optimal_exit_window="24-72h",
        ),
        SignalType.FUNDAMENTAL: DecayProfile(
            signal_type=SignalType.FUNDAMENTAL,
            half_life_hours=240,  # 10 days
            decay_speed=DecaySpeed.SLOW,
            ic_1h=0.01,
            ic_6h=0.01,
            ic_24h=0.02,
            ic_48h=0.02,
            ic_5d=0.02,
            ic_10d=0.02,
            optimal_entry_window="0-5d",
            optimal_exit_window="10-30d",
        ),
        SignalType.COMPOSITE: DecayProfile(
            signal_type=SignalType.COMPOSITE,
            half_life_hours=24,
            decay_speed=DecaySpeed.MEDIUM,
            ic_1h=0.06,
            ic_6h=0.05,
            ic_24h=0.04,
            ic_48h=0.03,
            ic_5d=0.02,
            ic_10d=0.01,
            optimal_entry_window="0-6h",
            optimal_exit_window="1-5d",
        ),
    }
    
    def __init__(self, db_engine=None):
        """
        Initialize tracker.
        
        Args:
            db_engine: SQLAlchemy engine for persistence
        """
        self.db_engine = db_engine
        self.profiles = self.DEFAULT_PROFILES.copy()
        
        # In-memory storage for snapshots
        self.snapshots: List[SignalSnapshot] = []
        self.max_snapshots = 10000
    
    def record_signal(self,
                      ticker: str,
                      signal_type: SignalType,
                      signal_value: float,
                      timestamp: datetime = None):
        """
        Record a signal snapshot.
        
        Args:
            ticker: Stock symbol
            signal_type: Type of signal
            signal_value: Signal value (0-100 or z-score)
            timestamp: Signal timestamp
        """
        snapshot = SignalSnapshot(
            ticker=ticker,
            signal_type=signal_type,
            signal_value=signal_value,
            timestamp=timestamp or datetime.now()
        )
        
        self.snapshots.append(snapshot)
        
        # Trim if too many
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]
    
    def record_forward_returns(self,
                               ticker: str,
                               signal_timestamp: datetime,
                               returns: Dict[str, float]):
        """
        Fill in forward returns for a signal snapshot.
        
        Args:
            ticker: Stock symbol
            signal_timestamp: Original signal timestamp
            returns: Dict of horizon -> return (e.g., {'1h': 0.2, '24h': 1.5})
        """
        # Find matching snapshot
        for snapshot in self.snapshots:
            if (snapshot.ticker == ticker and 
                abs((snapshot.timestamp - signal_timestamp).total_seconds()) < 60):
                
                snapshot.return_1h = returns.get('1h')
                snapshot.return_6h = returns.get('6h')
                snapshot.return_24h = returns.get('24h')
                snapshot.return_48h = returns.get('48h')
                snapshot.return_5d = returns.get('5d')
                snapshot.return_10d = returns.get('10d')
                break
    
    def calculate_decay_profiles(self, 
                                 min_samples: int = 100) -> Dict[SignalType, DecayProfile]:
        """
        Calculate decay profiles from recorded snapshots.
        
        Args:
            min_samples: Minimum samples required for calculation
            
        Returns:
            Updated decay profiles by signal type
        """
        # Group snapshots by signal type
        by_type: Dict[SignalType, List[SignalSnapshot]] = {}
        
        for snapshot in self.snapshots:
            if snapshot.return_24h is not None:  # Must have at least some returns
                if snapshot.signal_type not in by_type:
                    by_type[snapshot.signal_type] = []
                by_type[snapshot.signal_type].append(snapshot)
        
        # Calculate IC for each type and horizon
        for signal_type, snapshots in by_type.items():
            if len(snapshots) < min_samples:
                continue
            
            # Extract signal values and returns
            signals = np.array([s.signal_value for s in snapshots])
            
            # Normalize signals to z-scores if needed
            if signals.std() > 0:
                signals = (signals - signals.mean()) / signals.std()
            
            # Calculate IC at each horizon
            ic_values = {}
            
            for horizon, attr in [
                ('1h', 'return_1h'),
                ('6h', 'return_6h'),
                ('24h', 'return_24h'),
                ('48h', 'return_48h'),
                ('5d', 'return_5d'),
                ('10d', 'return_10d'),
            ]:
                returns = np.array([
                    getattr(s, attr) or 0 for s in snapshots
                ])
                
                # IC = correlation between signal and forward return
                if len(returns) > 10 and returns.std() > 0:
                    ic = np.corrcoef(signals, returns)[0, 1]
                    if not np.isnan(ic):
                        ic_values[horizon] = ic
            
            # Estimate half-life from IC decay
            if ic_values:
                half_life = self._estimate_half_life(ic_values)
                decay_speed = self._classify_decay_speed(half_life)
                
                self.profiles[signal_type] = DecayProfile(
                    signal_type=signal_type,
                    half_life_hours=half_life,
                    decay_speed=decay_speed,
                    ic_1h=ic_values.get('1h', 0),
                    ic_6h=ic_values.get('6h', 0),
                    ic_24h=ic_values.get('24h', 0),
                    ic_48h=ic_values.get('48h', 0),
                    ic_5d=ic_values.get('5d', 0),
                    ic_10d=ic_values.get('10d', 0),
                    optimal_entry_window=self._get_optimal_entry(half_life),
                    optimal_exit_window=self._get_optimal_exit(half_life),
                    sample_count=len(snapshots),
                    last_updated=datetime.now(),
                )
        
        return self.profiles
    
    def get_decay_profile(self, signal_type: SignalType) -> DecayProfile:
        """Get decay profile for a signal type."""
        return self.profiles.get(signal_type, self.DEFAULT_PROFILES.get(signal_type))
    
    def get_horizon_recommendation(self, signal_type: SignalType) -> Dict[str, Any]:
        """
        Get trading horizon recommendation for a signal type.
        
        Returns:
            Dict with entry window, exit window, urgency, and expected IC
        """
        profile = self.get_decay_profile(signal_type)
        
        # Calculate urgency based on half-life
        if profile.half_life_hours < 6:
            urgency = "IMMEDIATE"
            urgency_score = 90
        elif profile.half_life_hours < 24:
            urgency = "HIGH"
            urgency_score = 70
        elif profile.half_life_hours < 72:
            urgency = "MEDIUM"
            urgency_score = 50
        else:
            urgency = "LOW"
            urgency_score = 30
        
        return {
            'signal_type': signal_type.value,
            'half_life_hours': profile.half_life_hours,
            'decay_speed': profile.decay_speed.value,
            'urgency': urgency,
            'urgency_score': urgency_score,
            'optimal_entry_window': profile.optimal_entry_window,
            'optimal_exit_window': profile.optimal_exit_window,
            'expected_ic_1h': profile.ic_1h,
            'expected_ic_24h': profile.ic_24h,
            'expected_ic_5d': profile.ic_5d,
            'recommendation': self._generate_recommendation(profile),
        }
    
    def get_composite_recommendation(self,
                                     signal_weights: Dict[SignalType, float]) -> Dict[str, Any]:
        """
        Get blended recommendation for multiple signal types.
        
        Args:
            signal_weights: Dict of signal_type -> weight
            
        Returns:
            Blended recommendation
        """
        total_weight = sum(signal_weights.values())
        if total_weight == 0:
            return self.get_horizon_recommendation(SignalType.COMPOSITE)
        
        # Weight-average the half-lives
        blended_half_life = sum(
            self.get_decay_profile(sig).half_life_hours * weight
            for sig, weight in signal_weights.items()
        ) / total_weight
        
        # Weight-average the ICs
        blended_ic_24h = sum(
            self.get_decay_profile(sig).ic_24h * weight
            for sig, weight in signal_weights.items()
        ) / total_weight
        
        # Determine urgency from fastest-decaying signal
        min_half_life = min(
            self.get_decay_profile(sig).half_life_hours
            for sig in signal_weights.keys()
        )
        
        if min_half_life < 6:
            urgency = "IMMEDIATE"
        elif min_half_life < 24:
            urgency = "HIGH"
        elif min_half_life < 72:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"
        
        return {
            'blended_half_life_hours': blended_half_life,
            'fastest_signal': min(
                signal_weights.keys(),
                key=lambda s: self.get_decay_profile(s).half_life_hours
            ).value,
            'urgency': urgency,
            'blended_ic_24h': blended_ic_24h,
            'entry_window': self._get_optimal_entry(min_half_life),
            'exit_window': self._get_optimal_exit(blended_half_life),
        }
    
    def get_decay_report(self) -> str:
        """Generate text report of decay profiles."""
        lines = [
            "=" * 60,
            "ALPHA DECAY REPORT",
            "=" * 60,
            "",
        ]
        
        for signal_type in SignalType:
            profile = self.get_decay_profile(signal_type)
            rec = self.get_horizon_recommendation(signal_type)
            
            lines.append(f"📊 {signal_type.value.upper()}")
            lines.append(f"   Half-life: {profile.half_life_hours:.0f} hours ({profile.decay_speed.value})")
            lines.append(f"   Urgency: {rec['urgency']} ({rec['urgency_score']})")
            lines.append(f"   Entry window: {profile.optimal_entry_window}")
            lines.append(f"   Exit window: {profile.optimal_exit_window}")
            lines.append(f"   IC curve: 1h={profile.ic_1h:.2f}, 24h={profile.ic_24h:.2f}, 5d={profile.ic_5d:.2f}")
            if profile.sample_count > 0:
                lines.append(f"   Based on {profile.sample_count} samples")
            lines.append("")
        
        return "\n".join(lines)
    
    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================
    
    def _estimate_half_life(self, ic_values: Dict[str, float]) -> float:
        """Estimate half-life from IC decay curve."""
        # Map horizons to hours
        horizon_hours = {
            '1h': 1,
            '6h': 6,
            '24h': 24,
            '48h': 48,
            '5d': 120,
            '10d': 240,
        }
        
        # Find initial IC
        initial_ic = ic_values.get('1h', 0.05)
        if initial_ic <= 0:
            initial_ic = 0.05
        
        half_ic = initial_ic / 2
        
        # Find horizon where IC drops to half
        hours_list = sorted([(horizon_hours[k], v) for k, v in ic_values.items()])
        
        for i in range(len(hours_list) - 1):
            h1, ic1 = hours_list[i]
            h2, ic2 = hours_list[i + 1]
            
            if ic1 >= half_ic >= ic2:
                # Interpolate
                if ic1 != ic2:
                    ratio = (ic1 - half_ic) / (ic1 - ic2)
                    return h1 + ratio * (h2 - h1)
        
        # If IC never drops to half, use longer half-life
        if hours_list and hours_list[-1][1] > half_ic:
            return 240  # 10 days
        
        return 24  # Default
    
    def _classify_decay_speed(self, half_life_hours: float) -> DecaySpeed:
        """Classify decay speed from half-life."""
        if half_life_hours < 24:
            return DecaySpeed.FAST
        elif half_life_hours < 120:
            return DecaySpeed.MEDIUM
        elif half_life_hours < 480:
            return DecaySpeed.SLOW
        else:
            return DecaySpeed.PERSISTENT
    
    def _get_optimal_entry(self, half_life_hours: float) -> str:
        """Get optimal entry window based on half-life."""
        if half_life_hours < 6:
            return "0-1h"
        elif half_life_hours < 24:
            return "0-4h"
        elif half_life_hours < 72:
            return "0-24h"
        else:
            return "0-3d"
    
    def _get_optimal_exit(self, half_life_hours: float) -> str:
        """Get optimal exit window based on half-life."""
        if half_life_hours < 6:
            return "2-6h"
        elif half_life_hours < 24:
            return "6-24h"
        elif half_life_hours < 72:
            return "1-3d"
        else:
            return "5-20d"
    
    def _generate_recommendation(self, profile: DecayProfile) -> str:
        """Generate text recommendation for a profile."""
        if profile.decay_speed == DecaySpeed.FAST:
            return (
                f"Execute immediately. Signal loses half its value in "
                f"{profile.half_life_hours:.0f} hours. Consider intraday exit."
            )
        elif profile.decay_speed == DecaySpeed.MEDIUM:
            return (
                f"Execute within {profile.optimal_entry_window}. "
                f"Plan to exit within {profile.optimal_exit_window}."
            )
        elif profile.decay_speed == DecaySpeed.SLOW:
            return (
                f"Can wait for better entry. Signal persists for "
                f"{profile.half_life_hours/24:.0f}+ days. Swing trade timeframe."
            )
        else:
            return (
                f"Long-term signal. Position for multi-week holding period."
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_tracker_instance = None


def get_alpha_decay_tracker() -> AlphaDecayTracker:
    """Get singleton alpha decay tracker."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = AlphaDecayTracker()
    return _tracker_instance


def get_signal_urgency(signal_type: SignalType) -> Dict[str, Any]:
    """Quick access to get signal urgency recommendation."""
    return get_alpha_decay_tracker().get_horizon_recommendation(signal_type)


def get_execution_window(signal_type: SignalType) -> Tuple[str, str]:
    """Get optimal entry and exit windows for a signal type."""
    profile = get_alpha_decay_tracker().get_decay_profile(signal_type)
    return profile.optimal_entry_window, profile.optimal_exit_window


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    tracker = AlphaDecayTracker()
    
    # Print default decay profiles
    print(tracker.get_decay_report())
    
    # Get recommendations for each signal type
    print("\n" + "=" * 60)
    print("HORIZON RECOMMENDATIONS")
    print("=" * 60)
    
    for signal_type in SignalType:
        rec = tracker.get_horizon_recommendation(signal_type)
        print(f"\n{signal_type.value}:")
        print(f"  Urgency: {rec['urgency']} ({rec['urgency_score']})")
        print(f"  Entry: {rec['optimal_entry_window']}")
        print(f"  Exit: {rec['optimal_exit_window']}")
        print(f"  {rec['recommendation']}")
    
    # Test composite recommendation
    print("\n" + "=" * 60)
    print("COMPOSITE RECOMMENDATION")
    print("=" * 60)
    
    weights = {
        SignalType.SENTIMENT: 0.3,
        SignalType.OPTIONS_FLOW: 0.25,
        SignalType.TECHNICAL: 0.25,
        SignalType.FUNDAMENTAL: 0.2,
    }
    
    composite = tracker.get_composite_recommendation(weights)
    print(f"\nBlended half-life: {composite['blended_half_life_hours']:.0f} hours")
    print(f"Fastest signal: {composite['fastest_signal']}")
    print(f"Urgency: {composite['urgency']}")
    print(f"Entry: {composite['entry_window']}")
    print(f"Exit: {composite['exit_window']}")
