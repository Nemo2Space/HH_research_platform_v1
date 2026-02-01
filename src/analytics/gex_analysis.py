"""
GEX/Gamma Exposure Module - Phase 2

Calculates Gamma Exposure (GEX) to predict price pinning and directional moves.

Key concepts:
- Market makers are short options → they must hedge
- When GEX is HIGH POSITIVE: MM buy dips, sell rips → SUPPRESSED volatility
- When GEX is NEGATIVE: MM sell dips, buy rips → AMPLIFIED volatility
- Price tends to "pin" near max gamma strikes on expiration

Signals:
- High positive GEX = range-bound, mean reversion
- Negative GEX = trending, momentum
- GEX flip = potential volatility expansion

Author: Alpha Research Platform
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

from src.utils.logging import get_logger

logger = get_logger(__name__)


class GEXRegime(Enum):
    """GEX-based market regime."""
    VERY_POSITIVE = "VERY_POSITIVE"  # Strong dealer long gamma → vol suppression
    POSITIVE = "POSITIVE"             # Dealer long gamma → stable
    NEUTRAL = "NEUTRAL"               # Balanced
    NEGATIVE = "NEGATIVE"             # Dealer short gamma → vol amplification
    VERY_NEGATIVE = "VERY_NEGATIVE"   # Strong short gamma → potential explosion


class GammaProfile(Enum):
    """Gamma distribution profile."""
    PINNED = "PINNED"           # Strong gamma at current price → likely to stay
    BREAKOUT_UP = "BREAKOUT_UP" # Gamma wall above → resistance, but if broken...
    BREAKOUT_DN = "BREAKOUT_DN" # Gamma wall below → support, but if broken...
    TRENDING = "TRENDING"       # Low gamma → free to move
    VOLATILE = "VOLATILE"       # Negative gamma → amplified moves


@dataclass
class GEXAnalysis:
    """GEX analysis for a single ticker."""
    ticker: str
    as_of_date: date

    # Core GEX metrics
    total_gex: float = 0.0           # Net gamma exposure ($ per 1% move)
    gex_regime: GEXRegime = GEXRegime.NEUTRAL
    gamma_profile: GammaProfile = GammaProfile.TRENDING

    # Key levels
    max_gamma_strike: float = 0.0    # Strike with highest gamma (pin level)
    call_wall: float = 0.0           # Highest call OI strike (resistance)
    put_wall: float = 0.0            # Highest put OI strike (support)
    zero_gamma_level: float = 0.0    # Price where GEX flips sign

    # Directional signals
    gamma_tilt: float = 0.0          # Positive = bullish, negative = bearish
    vol_trigger_up: float = 0.0      # Price above which vol expands
    vol_trigger_dn: float = 0.0      # Price below which vol expands

    # Options data
    total_call_oi: int = 0
    total_put_oi: int = 0
    put_call_oi_ratio: float = 1.0
    weighted_iv: float = 0.0
    iv_skew: float = 0.0             # Put IV - Call IV (positive = fear)

    # Expiration analysis
    nearest_expiry: str = ""
    days_to_expiry: int = 0
    gamma_at_expiry: float = 0.0     # Gamma concentrated at nearest expiry

    # Signals
    signal: str = "NEUTRAL"          # BULLISH, BEARISH, NEUTRAL, PINNED
    signal_strength: int = 50        # 0-100
    expected_move: float = 0.0       # Expected % move based on gamma

    # Warnings
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ticker': self.ticker,
            'total_gex': round(self.total_gex / 1e6, 2),  # In millions
            'gex_regime': self.gex_regime.value,
            'gamma_profile': self.gamma_profile.value,
            'max_gamma_strike': self.max_gamma_strike,
            'call_wall': self.call_wall,
            'put_wall': self.put_wall,
            'signal': self.signal,
            'signal_strength': self.signal_strength,
            'put_call_oi_ratio': round(self.put_call_oi_ratio, 2),
            'expected_move': round(self.expected_move, 2),
            'warnings': self.warnings,
        }


class GEXCalculator:
    """
    Calculates Gamma Exposure from options data.
    """

    # Constants for gamma calculation
    SHARES_PER_CONTRACT = 100

    def __init__(self):
        self._cache = {}
        self._cache_time = {}
        self._cache_duration = timedelta(minutes=15)

    def analyze_gex(self, ticker: str, current_price: float = None) -> GEXAnalysis:
        """
        Analyze GEX for a ticker.

        Args:
            ticker: Stock symbol
            current_price: Current stock price (fetched if not provided)

        Returns:
            GEXAnalysis with all metrics
        """
        analysis = GEXAnalysis(
            ticker=ticker,
            as_of_date=date.today(),
        )

        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)

            # Get current price
            if current_price is None:
                hist = stock.history(period="1d")
                if hist.empty:
                    analysis.warnings.append("Could not fetch current price")
                    return analysis
                current_price = hist['Close'].iloc[-1]

            # Get options expirations
            try:
                expirations = stock.options
            except Exception:
                analysis.warnings.append("No options data available")
                return analysis

            if not expirations:
                analysis.warnings.append("No options expirations found")
                return analysis

            # Focus on near-term expirations (next 45 days)
            today = date.today()
            near_expirations = []
            for exp in expirations[:6]:  # First 6 expirations
                try:
                    exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                    days = (exp_date - today).days
                    if 0 < days <= 45:
                        near_expirations.append((exp, days))
                except:
                    pass

            if not near_expirations:
                near_expirations = [(expirations[0], 7)]  # Use first available

            analysis.nearest_expiry = near_expirations[0][0]
            analysis.days_to_expiry = near_expirations[0][1]

            # Aggregate options data across expirations
            all_calls = []
            all_puts = []

            for exp, days in near_expirations:
                try:
                    opt_chain = stock.option_chain(exp)

                    calls = opt_chain.calls.copy()
                    puts = opt_chain.puts.copy()

                    calls['expiry'] = exp
                    calls['days'] = days
                    puts['expiry'] = exp
                    puts['days'] = days

                    all_calls.append(calls)
                    all_puts.append(puts)
                except Exception as e:
                    logger.debug(f"{ticker}: Error fetching {exp} options: {e}")

            if not all_calls:
                analysis.warnings.append("Could not fetch options chains")
                return analysis

            calls_df = pd.concat(all_calls, ignore_index=True)
            puts_df = pd.concat(all_puts, ignore_index=True)

            # Calculate GEX
            self._calculate_gex_metrics(analysis, calls_df, puts_df, current_price)

            # Determine regime and profile
            self._determine_regime(analysis, current_price)

            # Generate signals
            self._generate_signals(analysis, current_price)

        except Exception as e:
            logger.error(f"{ticker}: GEX analysis error: {e}")
            analysis.warnings.append(f"Analysis error: {str(e)}")

        return analysis

    def _calculate_gex_metrics(self, analysis: GEXAnalysis,
                                calls_df: pd.DataFrame,
                                puts_df: pd.DataFrame,
                                current_price: float):
        """Calculate core GEX metrics from options data."""

        # Total OI
        analysis.total_call_oi = int(calls_df['openInterest'].sum())
        analysis.total_put_oi = int(puts_df['openInterest'].sum())

        if analysis.total_call_oi > 0:
            analysis.put_call_oi_ratio = analysis.total_put_oi / analysis.total_call_oi

        # Find walls (highest OI strikes)
        call_oi_by_strike = calls_df.groupby('strike')['openInterest'].sum()
        put_oi_by_strike = puts_df.groupby('strike')['openInterest'].sum()

        if not call_oi_by_strike.empty:
            # Call wall - highest OI above current price
            above_price = call_oi_by_strike[call_oi_by_strike.index > current_price]
            if not above_price.empty:
                analysis.call_wall = float(above_price.idxmax())

        if not put_oi_by_strike.empty:
            # Put wall - highest OI below current price
            below_price = put_oi_by_strike[put_oi_by_strike.index < current_price]
            if not below_price.empty:
                analysis.put_wall = float(below_price.idxmax())

        # Calculate gamma at each strike
        # Simplified gamma calculation (without full Black-Scholes)
        # Gamma is highest ATM and decays as you move away

        def estimate_gamma(row, is_call: bool):
            """Estimate gamma contribution."""
            strike = row['strike']
            oi = row.get('openInterest', 0)
            days = row.get('days', 7)

            if oi == 0 or days == 0:
                return 0

            # Distance from current price (normalized)
            moneyness = (strike - current_price) / current_price

            # Gamma is highest ATM, decays with distance
            # Using simplified normal approximation
            gamma_factor = np.exp(-0.5 * (moneyness / 0.05) ** 2)

            # Time decay factor (gamma increases as expiry approaches)
            time_factor = 1 / np.sqrt(max(days, 1))

            # Gamma in dollar terms (per 1% move)
            gamma = oi * self.SHARES_PER_CONTRACT * current_price * 0.01 * gamma_factor * time_factor

            # Calls contribute positive gamma, puts contribute negative
            # (from dealer's perspective - they are short options)
            # Actually, dealers delta-hedge:
            # - Long gamma (from being short puts) = stabilizing
            # - Short gamma (from being short calls) = destabilizing
            # This is simplified; actual sign depends on dealer positioning

            return gamma

        # Calculate total gamma
        calls_df['gamma'] = calls_df.apply(lambda r: estimate_gamma(r, True), axis=1)
        puts_df['gamma'] = puts_df.apply(lambda r: estimate_gamma(r, False), axis=1)

        # Net GEX (simplified model)
        # Positive GEX = dealers will buy dips/sell rips (stabilizing)
        # This is a proxy - real GEX requires dealer position data
        call_gamma = calls_df['gamma'].sum()
        put_gamma = puts_df['gamma'].sum()

        # Puts create positive gamma for dealers (they sell puts, buy stock to hedge)
        # Calls create negative gamma for dealers (they sell calls, sell stock to hedge)
        analysis.total_gex = put_gamma - call_gamma * 0.5  # Simplified

        # Find max gamma strike
        all_gamma = pd.concat([
            calls_df.groupby('strike')['gamma'].sum(),
            puts_df.groupby('strike')['gamma'].sum()
        ]).groupby(level=0).sum()

        if not all_gamma.empty:
            analysis.max_gamma_strike = float(all_gamma.abs().idxmax())

        # IV metrics
        if 'impliedVolatility' in calls_df.columns:
            atm_calls = calls_df[
                (calls_df['strike'] >= current_price * 0.95) &
                (calls_df['strike'] <= current_price * 1.05)
            ]
            if not atm_calls.empty:
                analysis.weighted_iv = float(atm_calls['impliedVolatility'].mean() * 100)

        # IV Skew (25-delta put vs call)
        otm_puts = puts_df[puts_df['strike'] < current_price * 0.95]
        otm_calls = calls_df[calls_df['strike'] > current_price * 1.05]

        if not otm_puts.empty and not otm_calls.empty:
            if 'impliedVolatility' in otm_puts.columns:
                put_iv = otm_puts['impliedVolatility'].mean()
                call_iv = otm_calls['impliedVolatility'].mean()
                analysis.iv_skew = float((put_iv - call_iv) * 100)

        # Gamma tilt (bullish/bearish based on OI distribution)
        if analysis.call_wall > 0 and analysis.put_wall > 0:
            call_dist = (analysis.call_wall - current_price) / current_price
            put_dist = (current_price - analysis.put_wall) / current_price
            analysis.gamma_tilt = (put_dist - call_dist) * 100  # Positive = bullish

    def _determine_regime(self, analysis: GEXAnalysis, current_price: float):
        """Determine GEX regime and gamma profile."""

        gex = analysis.total_gex

        # GEX regime thresholds (in millions)
        gex_m = gex / 1e6

        if gex_m > 500:
            analysis.gex_regime = GEXRegime.VERY_POSITIVE
        elif gex_m > 100:
            analysis.gex_regime = GEXRegime.POSITIVE
        elif gex_m > -100:
            analysis.gex_regime = GEXRegime.NEUTRAL
        elif gex_m > -500:
            analysis.gex_regime = GEXRegime.NEGATIVE
        else:
            analysis.gex_regime = GEXRegime.VERY_NEGATIVE

        # Gamma profile based on positioning relative to walls
        if analysis.max_gamma_strike > 0:
            dist_to_max = abs(current_price - analysis.max_gamma_strike) / current_price

            if dist_to_max < 0.02:  # Within 2% of max gamma
                analysis.gamma_profile = GammaProfile.PINNED
            elif current_price < analysis.max_gamma_strike:
                analysis.gamma_profile = GammaProfile.BREAKOUT_UP
            else:
                analysis.gamma_profile = GammaProfile.BREAKOUT_DN

        if analysis.gex_regime in [GEXRegime.NEGATIVE, GEXRegime.VERY_NEGATIVE]:
            analysis.gamma_profile = GammaProfile.VOLATILE

        # Vol trigger levels
        if analysis.call_wall > 0:
            analysis.vol_trigger_up = analysis.call_wall
        if analysis.put_wall > 0:
            analysis.vol_trigger_dn = analysis.put_wall

    def _generate_signals(self, analysis: GEXAnalysis, current_price: float):
        """Generate trading signals from GEX analysis."""

        strength = 50
        signal = "NEUTRAL"

        # GEX regime impact
        if analysis.gex_regime == GEXRegime.VERY_POSITIVE:
            signal = "PINNED"
            strength = 70
            analysis.expected_move = 1.0  # Low expected move
        elif analysis.gex_regime == GEXRegime.VERY_NEGATIVE:
            signal = "VOLATILE"
            strength = 30
            analysis.expected_move = 5.0  # High expected move

        # Gamma tilt (directional bias)
        if analysis.gamma_tilt > 5:
            signal = "BULLISH"
            strength += 10
        elif analysis.gamma_tilt < -5:
            signal = "BEARISH"
            strength += 10

        # Put/call ratio
        if analysis.put_call_oi_ratio > 1.5:
            if signal == "BULLISH":
                strength += 10  # Contrarian bullish
            analysis.warnings.append(f"High put/call ratio ({analysis.put_call_oi_ratio:.2f}) - fear elevated")
        elif analysis.put_call_oi_ratio < 0.5:
            if signal == "BEARISH":
                strength += 10  # Contrarian bearish
            analysis.warnings.append(f"Low put/call ratio ({analysis.put_call_oi_ratio:.2f}) - complacency")

        # IV skew
        if analysis.iv_skew > 10:
            analysis.warnings.append(f"High IV skew ({analysis.iv_skew:.1f}%) - put protection demand")

        # Near expiry gamma
        if analysis.days_to_expiry <= 2:
            analysis.warnings.append("Expiry pinning likely - gamma very high")
            if analysis.gamma_profile == GammaProfile.PINNED:
                strength += 15

        analysis.signal = signal
        analysis.signal_strength = min(100, max(0, strength))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_calculator = None

def get_gex_calculator() -> GEXCalculator:
    """Get singleton calculator instance."""
    global _calculator
    if _calculator is None:
        _calculator = GEXCalculator()
    return _calculator


def analyze_gex(ticker: str, current_price: float = None) -> GEXAnalysis:
    """
    Analyze GEX for a ticker.

    Usage:
        analysis = analyze_gex('SPY')
        print(f"GEX Regime: {analysis.gex_regime.value}")
        print(f"Max Gamma: ${analysis.max_gamma_strike}")
        print(f"Signal: {analysis.signal}")
    """
    calc = get_gex_calculator()
    return calc.analyze_gex(ticker, current_price)


def get_market_gex() -> Dict[str, GEXAnalysis]:
    """
    Get GEX analysis for major indices/ETFs.

    Usage:
        gex = get_market_gex()
        print(f"SPY GEX: {gex['SPY'].gex_regime.value}")
    """
    calc = get_gex_calculator()

    tickers = ['SPY', 'QQQ', 'IWM']
    results = {}

    for ticker in tickers:
        try:
            results[ticker] = calc.analyze_gex(ticker)
        except Exception as e:
            logger.error(f"{ticker}: GEX analysis failed: {e}")

    return results


def get_gex_table(tickers: List[str]) -> pd.DataFrame:
    """
    Get GEX analysis as DataFrame.

    Usage:
        df = get_gex_table(['AAPL', 'NVDA', 'TSLA'])
        print(df)
    """
    calc = get_gex_calculator()

    data = []
    for ticker in tickers:
        try:
            analysis = calc.analyze_gex(ticker)
            data.append({
                'Ticker': ticker,
                'GEX ($M)': round(analysis.total_gex / 1e6, 1),
                'Regime': analysis.gex_regime.value,
                'Profile': analysis.gamma_profile.value,
                'Max Gamma': f"${analysis.max_gamma_strike:.0f}",
                'Call Wall': f"${analysis.call_wall:.0f}",
                'Put Wall': f"${analysis.put_wall:.0f}",
                'P/C Ratio': round(analysis.put_call_oi_ratio, 2),
                'Signal': analysis.signal,
            })
        except Exception as e:
            logger.debug(f"{ticker}: GEX failed: {e}")

    return pd.DataFrame(data)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing GEX Analysis...")

    analysis = analyze_gex('SPY')

    print(f"\nSPY GEX Analysis:")
    print(f"  Total GEX: ${analysis.total_gex / 1e6:.1f}M")
    print(f"  Regime: {analysis.gex_regime.value}")
    print(f"  Profile: {analysis.gamma_profile.value}")
    print(f"  Max Gamma Strike: ${analysis.max_gamma_strike:.0f}")
    print(f"  Call Wall: ${analysis.call_wall:.0f}")
    print(f"  Put Wall: ${analysis.put_wall:.0f}")
    print(f"  Signal: {analysis.signal} (strength: {analysis.signal_strength})")

    if analysis.warnings:
        print(f"  Warnings: {', '.join(analysis.warnings)}")