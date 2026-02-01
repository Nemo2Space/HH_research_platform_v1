"""
Enhanced Scoring Module

Adds institutional-grade scoring factors:
1. Sector-relative PE valuation
2. Price target integration
3. MACD momentum indicator
4. Insider trading pattern detection
5. Analyst revision tracking
6. Volume profile analysis
7. Earnings surprise history

Author: Alpha Research Platform
Version: 2024-12-23
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import database connection
try:
    from src.db.connection import get_engine, get_connection
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("Database connection not available")


# =============================================================================
# SECTOR REFERENCE DATA
# =============================================================================

SECTOR_PE_MEDIANS = {
    'Technology': 28,
    'Healthcare': 22,
    'Financials': 12,
    'Financial Services': 12,
    'Consumer Cyclical': 18,
    'Consumer Discretionary': 18,
    'Industrials': 20,
    'Communication Services': 18,
    'Consumer Defensive': 22,
    'Consumer Staples': 22,
    'Energy': 10,
    'Utilities': 16,
    'Real Estate': 35,
    'Basic Materials': 14,
    'Materials': 14,
}

SECTOR_GROWTH_EXPECTATIONS = {
    'Technology': 0.15,        # 15% expected growth
    'Healthcare': 0.10,
    'Financials': 0.05,
    'Financial Services': 0.05,
    'Consumer Cyclical': 0.08,
    'Consumer Discretionary': 0.08,
    'Industrials': 0.06,
    'Communication Services': 0.08,
    'Consumer Defensive': 0.04,
    'Consumer Staples': 0.04,
    'Energy': 0.03,
    'Utilities': 0.03,
    'Real Estate': 0.04,
    'Basic Materials': 0.05,
    'Materials': 0.05,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EnhancedScores:
    """Container for all enhanced scoring components."""
    ticker: str

    # Valuation
    pe_relative_score: int = 0
    price_target_score: int = 0
    peg_score: int = 0

    # Momentum
    macd_score: int = 0
    volume_score: int = 0

    # Smart Money
    insider_score: int = 0
    institutional_score: int = 0

    # Analyst
    revision_score: int = 0

    # Earnings
    earnings_surprise_score: int = 0

    # Combined
    enhancement_total: int = 0

    # Details for display
    details: Dict[str, str] = field(default_factory=dict)

    def calculate_total(self):
        """Calculate total enhancement score."""
        self.enhancement_total = (
            self.pe_relative_score +
            self.price_target_score +
            self.peg_score +
            self.macd_score +
            self.volume_score +
            self.insider_score +
            self.revision_score +
            self.earnings_surprise_score
        )
        return self.enhancement_total


# =============================================================================
# 1. SECTOR-RELATIVE PE SCORING
# =============================================================================

def score_pe_relative(pe: float, sector: str, forward_pe: float = None) -> Tuple[int, str]:
    """
    Score PE ratio relative to sector median.

    Returns:
        Tuple of (score adjustment, reason string)
    """
    if not pe or pe <= 0:
        return 0, "PE not available"

    # Get sector median (default to market average if unknown)
    sector_median = SECTOR_PE_MEDIANS.get(sector, 20)

    # Calculate discount/premium to sector
    pe_ratio_to_sector = pe / sector_median

    score = 0
    reason = ""

    if pe_ratio_to_sector < 0.6:
        score = 15
        reason = f"Deep value: PE {pe:.1f} vs sector {sector_median} ({(1-pe_ratio_to_sector)*100:.0f}% discount)"
    elif pe_ratio_to_sector < 0.8:
        score = 10
        reason = f"Value: PE {pe:.1f} vs sector {sector_median} ({(1-pe_ratio_to_sector)*100:.0f}% discount)"
    elif pe_ratio_to_sector < 0.95:
        score = 5
        reason = f"Slight discount: PE {pe:.1f} vs sector {sector_median}"
    elif pe_ratio_to_sector > 1.5:
        score = -10
        reason = f"Expensive: PE {pe:.1f} vs sector {sector_median} ({(pe_ratio_to_sector-1)*100:.0f}% premium)"
    elif pe_ratio_to_sector > 1.25:
        score = -5
        reason = f"Premium: PE {pe:.1f} vs sector {sector_median}"
    else:
        reason = f"Fair value: PE {pe:.1f} vs sector {sector_median}"

    # Forward PE bonus (if growing into valuation)
    if forward_pe and forward_pe > 0 and pe > 0:
        if forward_pe < pe * 0.8:
            score += 5
            reason += " | Forward PE improving"

    return score, reason


def score_peg_ratio(peg: float, sector: str, pe: float = None, growth_rate: float = None, ticker: str = None) -> Tuple[int, str]:
    """
    Score PEG ratio (PE / Growth rate).
    PEG < 1 is generally considered undervalued.

    If PEG not provided, try to fetch from yfinance or calculate.
    """
    # Try to calculate PEG if not provided
    if (not peg or peg <= 0) and pe and pe > 0 and growth_rate and growth_rate > 0:
        peg = pe / (growth_rate * 100)  # growth_rate as decimal, e.g., 0.15 = 15%

    # Try to fetch PEG from yfinance if still not available
    if (not peg or peg <= 0) and ticker:
        try:
            import yfinance as yf
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stock = yf.Ticker(ticker)
                info = stock.info
                peg = info.get('pegRatio')
        except:
            pass

    if not peg or peg <= 0:
        return 0, "PEG not available"

    score = 0

    if peg < 0.5:
        score = 10
        reason = f"Very attractive PEG {peg:.2f}"
    elif peg < 1.0:
        score = 5
        reason = f"Attractive PEG {peg:.2f}"
    elif peg < 1.5:
        score = 2
        reason = f"Fair PEG {peg:.2f}"
    elif peg > 2.5:
        score = -5
        reason = f"High PEG {peg:.2f}"
    elif peg > 3.5:
        score = -10
        reason = f"Very high PEG {peg:.2f}"
    else:
        reason = f"Moderate PEG {peg:.2f}"

    return score, reason


# =============================================================================
# 2. PRICE TARGET INTEGRATION
# =============================================================================

def score_price_target(current_price: float, target_price: float,
                       buy_count: int = 0, total_ratings: int = 0) -> Tuple[int, str]:
    """
    Score based on analyst price targets and ratings.

    Args:
        current_price: Current stock price
        target_price: Consensus price target
        buy_count: Number of buy ratings
        total_ratings: Total analyst ratings
    """
    if not current_price or current_price <= 0 or not target_price or target_price <= 0:
        return 0, "Price target not available"

    upside_pct = ((target_price - current_price) / current_price) * 100

    score = 0
    reasons = []

    # Upside potential scoring
    if upside_pct > 40:
        score += 15
        reasons.append(f"Strong upside {upside_pct:.0f}%")
    elif upside_pct > 25:
        score += 10
        reasons.append(f"Good upside {upside_pct:.0f}%")
    elif upside_pct > 10:
        score += 5
        reasons.append(f"Modest upside {upside_pct:.0f}%")
    elif upside_pct < -15:
        score -= 10
        reasons.append(f"Downside risk {upside_pct:.0f}%")
    elif upside_pct < -5:
        score -= 5
        reasons.append(f"Limited upside {upside_pct:.0f}%")

    # Analyst consensus scoring
    if total_ratings and total_ratings > 0:
        buy_pct = (buy_count / total_ratings) * 100 if buy_count else 0

        if buy_pct >= 80:
            score += 5
            reasons.append(f"Strong buy consensus ({buy_pct:.0f}%)")
        elif buy_pct >= 60:
            score += 3
            reasons.append(f"Majority buy ({buy_pct:.0f}%)")
        elif buy_pct <= 20:
            score -= 5
            reasons.append(f"Weak buy consensus ({buy_pct:.0f}%)")

    return score, " | ".join(reasons) if reasons else "Neutral price target"


# =============================================================================
# 3. MACD MOMENTUM INDICATOR
# =============================================================================

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26,
                   signal: int = 9) -> Tuple[float, float, float]:
    """
    Calculate MACD indicator.

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    if len(prices) < slow + signal:
        return 0, 0, 0

    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]


def score_macd(prices: pd.Series) -> Tuple[int, str]:
    """
    Score based on MACD signals.
    """
    if prices is None or len(prices) < 35:
        return 0, "Insufficient data for MACD"

    macd, signal, histogram = calculate_macd(prices)

    # Also check previous values for crossover detection
    if len(prices) >= 36:
        prev_prices = prices.iloc[:-1]
        prev_macd, prev_signal, prev_hist = calculate_macd(prev_prices)
    else:
        prev_macd, prev_signal, prev_hist = macd, signal, histogram

    score = 0
    reasons = []

    # Current position
    if macd > signal:
        score += 3
        reasons.append("MACD bullish")
    elif macd < signal:
        score -= 3
        reasons.append("MACD bearish")

    # Histogram momentum
    if histogram > 0 and histogram > prev_hist:
        score += 2
        reasons.append("momentum increasing")
    elif histogram < 0 and histogram < prev_hist:
        score -= 2
        reasons.append("momentum decreasing")

    # Crossover detection (strong signals)
    if prev_macd <= prev_signal and macd > signal:
        score += 5
        reasons.append("bullish crossover!")
    elif prev_macd >= prev_signal and macd < signal:
        score -= 5
        reasons.append("bearish crossover!")

    # Zero line cross
    if prev_macd < 0 and macd > 0:
        score += 3
        reasons.append("crossed above zero")
    elif prev_macd > 0 and macd < 0:
        score -= 3
        reasons.append("crossed below zero")

    return score, " | ".join(reasons) if reasons else "MACD neutral"


# =============================================================================
# 4. VOLUME PROFILE ANALYSIS
# =============================================================================

def score_volume_profile(volumes: pd.Series, prices: pd.Series) -> Tuple[int, str]:
    """
    Score based on volume patterns indicating accumulation or distribution.
    """
    if volumes is None or len(volumes) < 20:
        return 0, "Insufficient volume data"

    score = 0
    reasons = []

    # Recent volume vs average
    avg_volume = volumes.iloc[-20:].mean()
    recent_volume = volumes.iloc[-5:].mean()
    volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

    # Price change over same period
    if len(prices) >= 5:
        price_change = (prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5]
    else:
        price_change = 0

    # Accumulation: Rising price + rising volume
    if volume_ratio > 1.5 and price_change > 0.02:
        score += 8
        reasons.append(f"Strong accumulation (vol +{(volume_ratio-1)*100:.0f}%)")
    elif volume_ratio > 1.2 and price_change > 0.01:
        score += 4
        reasons.append("Accumulation pattern")

    # Distribution: Falling price + rising volume
    elif volume_ratio > 1.5 and price_change < -0.02:
        score -= 8
        reasons.append(f"Distribution (vol +{(volume_ratio-1)*100:.0f}%)")
    elif volume_ratio > 1.2 and price_change < -0.01:
        score -= 4
        reasons.append("Distribution pattern")

    # Low volume warning
    elif volume_ratio < 0.5:
        reasons.append("Low volume (caution)")

    # On-Balance Volume trend (simplified)
    if len(prices) >= 10 and len(volumes) >= 10:
        obv = []
        obv_val = 0
        for i in range(1, min(len(prices), len(volumes))):
            if prices.iloc[i] > prices.iloc[i-1]:
                obv_val += volumes.iloc[i]
            elif prices.iloc[i] < prices.iloc[i-1]:
                obv_val -= volumes.iloc[i]
            obv.append(obv_val)

        if len(obv) >= 10:
            obv_trend = obv[-1] - obv[-10]
            if obv_trend > 0 and price_change > 0:
                score += 2
                reasons.append("OBV confirms uptrend")
            elif obv_trend < 0 and price_change < 0:
                score -= 2
                reasons.append("OBV confirms downtrend")
            elif obv_trend > 0 and price_change < 0:
                score += 3
                reasons.append("OBV divergence (bullish)")
            elif obv_trend < 0 and price_change > 0:
                score -= 3
                reasons.append("OBV divergence (bearish)")

    return score, " | ".join(reasons) if reasons else "Volume neutral"


# =============================================================================
# 5. INSIDER TRADING PATTERN DETECTION
# =============================================================================

def score_insider_trading(ticker: str, days_back: int = 90) -> Tuple[int, str]:
    """
    Score based on insider trading patterns.

    Patterns detected:
    - Cluster buying (multiple insiders buying)
    - Large purchases by executives
    - Selling before earnings (red flag)
    - Consistent buying over time
    """
    if not DB_AVAILABLE:
        return 0, "Database not available"

    try:
        query = """
            SELECT 
                insider_name,
                insider_title,
                transaction_type,
                shares,
                price,
                value,
                transaction_date
            FROM insider_transactions
            WHERE ticker = %s
              AND transaction_date >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY transaction_date DESC
        """

        df = pd.read_sql(query, get_engine(), params=(ticker, days_back))

        if df.empty:
            return 0, "No recent insider activity"

        score = 0
        reasons = []

        # Normalize transaction types
        df['is_buy'] = df['transaction_type'].str.lower().str.contains('buy|purchase|acquisition|award', na=False)
        df['is_sell'] = df['transaction_type'].str.lower().str.contains('sell|sale|dispose', na=False)

        # Count buys and sells
        buy_transactions = df[df['is_buy']]
        sell_transactions = df[df['is_sell']]

        total_buy_value = buy_transactions['value'].sum() if not buy_transactions.empty else 0
        total_sell_value = sell_transactions['value'].sum() if not sell_transactions.empty else 0

        num_buyers = buy_transactions['insider_name'].nunique() if not buy_transactions.empty else 0
        num_sellers = sell_transactions['insider_name'].nunique() if not sell_transactions.empty else 0

        # Pattern 1: Cluster buying (multiple insiders buying)
        if num_buyers >= 3:
            score += 15
            reasons.append(f"Cluster buying: {num_buyers} insiders")
        elif num_buyers >= 2:
            score += 8
            reasons.append(f"Multiple buyers: {num_buyers} insiders")
        elif num_buyers == 1 and total_buy_value > 100000:
            score += 5
            reasons.append("Significant insider purchase")

        # Pattern 2: Large executive purchases
        if not buy_transactions.empty:
            exec_titles = ['CEO', 'CFO', 'COO', 'President', 'Chairman', 'Director']
            exec_buys = buy_transactions[
                buy_transactions['insider_title'].str.contains('|'.join(exec_titles), case=False, na=False)
            ]
            if not exec_buys.empty:
                exec_value = exec_buys['value'].sum()
                if exec_value > 500000:
                    score += 10
                    reasons.append(f"Executive buying ${exec_value/1000:.0f}K")
                elif exec_value > 100000:
                    score += 5
                    reasons.append(f"Executive purchase ${exec_value/1000:.0f}K")

        # Pattern 3: Heavy selling (warning)
        if num_sellers >= 3 and total_sell_value > total_buy_value * 2:
            score -= 10
            reasons.append(f"Heavy insider selling: {num_sellers} insiders")
        elif total_sell_value > total_buy_value * 3 and total_sell_value > 500000:
            score -= 8
            reasons.append(f"Significant insider selling ${total_sell_value/1e6:.1f}M")

        # Pattern 4: Net buying ratio
        if total_buy_value > 0 and total_sell_value > 0:
            buy_sell_ratio = total_buy_value / total_sell_value
            if buy_sell_ratio > 2:
                score += 5
                reasons.append(f"Buy/sell ratio: {buy_sell_ratio:.1f}x")
            elif buy_sell_ratio < 0.3:
                score -= 5
                reasons.append(f"Sell/buy ratio: {1/buy_sell_ratio:.1f}x")
        elif total_buy_value > 0 and total_sell_value == 0:
            score += 3
            reasons.append("Only buying, no selling")

        # Summary
        if not reasons:
            reasons.append(f"Buys: {len(buy_transactions)}, Sells: {len(sell_transactions)}")

        return score, " | ".join(reasons)

    except Exception as e:
        logger.warning(f"Error scoring insider trading for {ticker}: {e}")
        return 0, f"Error: {str(e)[:50]}"


# =============================================================================
# 6. ANALYST REVISION TRACKING
# =============================================================================

def score_analyst_revisions(ticker: str, days_back: int = 30) -> Tuple[int, str]:
    """
    Score based on analyst ratings and price target data.
    Works even with single-row per ticker (uses current consensus strength).
    """
    if not DB_AVAILABLE:
        return 0, "Database not available"

    score = 0
    reasons = []

    try:
        # Get current analyst ratings
        ar_query = """
            SELECT 
                analyst_buy, analyst_hold, analyst_sell, analyst_total,
                analyst_positivity, consensus_rating
            FROM analyst_ratings
            WHERE ticker = %s
            ORDER BY date DESC
            LIMIT 1
        """

        ar_df = pd.read_sql(ar_query, get_engine(), params=(ticker,))

        if not ar_df.empty:
            row = ar_df.iloc[0]
            total = row.get('analyst_total') or 0
            buys = row.get('analyst_buy') or 0
            sells = row.get('analyst_sell') or 0
            positivity = row.get('analyst_positivity') or 0
            consensus = row.get('consensus_rating') or ''

            if total >= 3:  # Need at least 3 analysts for meaningful consensus
                buy_pct = (buys / total) * 100
                sell_pct = (sells / total) * 100

                # Score based on buy/sell ratio
                if buy_pct >= 80:
                    score += 5
                    reasons.append(f"Strong Buy consensus ({buy_pct:.0f}%)")
                elif buy_pct >= 60:
                    score += 3
                    reasons.append(f"Majority Buy ({buy_pct:.0f}%)")
                elif sell_pct >= 50:
                    score -= 5
                    reasons.append(f"Majority Sell ({sell_pct:.0f}%)")
                elif sell_pct >= 30:
                    score -= 2
                    reasons.append(f"High sell ratings ({sell_pct:.0f}%)")

                # Positivity bonus
                if positivity >= 80:
                    score += 2
                    reasons.append(f"{positivity:.0f}% positive")
                elif positivity <= 40:
                    score -= 2
                    reasons.append(f"Only {positivity:.0f}% positive")
    except Exception as e:
        logger.debug(f"Error getting analyst ratings for {ticker}: {e}")

    # Also check price targets for upside signal
    try:
        pt_query = """
            SELECT target_mean, target_high, target_low, current_price, target_upside_pct
            FROM price_targets
            WHERE ticker = %s
            ORDER BY date DESC
            LIMIT 1
        """
        pt_df = pd.read_sql(pt_query, get_engine(), params=(ticker,))

        if not pt_df.empty:
            row = pt_df.iloc[0]
            upside = row.get('target_upside_pct')
            target_high = row.get('target_high')
            target_low = row.get('target_low')
            target_mean = row.get('target_mean')

            # Wide target spread = high uncertainty
            if target_high and target_low and target_mean and target_mean > 0:
                spread = ((target_high - target_low) / target_mean) * 100
                if spread > 100:
                    score -= 2
                    reasons.append(f"High uncertainty (spread {spread:.0f}%)")
    except Exception as e:
        logger.debug(f"Error getting price targets for {ticker}: {e}")

    if not reasons:
        return 0, "Analyst data analyzed"

    return score, " | ".join(reasons)


# =============================================================================
# 7. EARNINGS SURPRISE SCORING
# =============================================================================

def score_earnings_surprise(ticker: str) -> Tuple[int, str]:
    """
    Score based on earnings beat/miss history.

    Priority:
    1. Use existing reaction_analyzer (has best data integration)
    2. Fall back to yfinance earnings_history
    """
    score = 0
    reasons = []

    # METHOD 1: Try to use existing reaction_analyzer (best integration)
    try:
        from src.analytics.earnings_intelligence.reaction_analyzer import analyze_post_earnings
        result = analyze_post_earnings(ticker)

        if result and result.eps_beat is not None:
            # Get beat/miss from reaction analyzer
            if result.eps_beat:
                score += 7
                reasons.append("Latest Q: BEAT")
            else:
                score -= 5
                reasons.append("Latest Q: MISSED")

            # Add surprise percentage if available
            if result.eps_actual and result.eps_estimate and result.eps_estimate != 0:
                surprise_pct = ((result.eps_actual - result.eps_estimate) / abs(result.eps_estimate)) * 100
                if surprise_pct > 10:
                    score += 3
                    reasons.append(f"+{surprise_pct:.1f}% surprise")
                elif surprise_pct < -10:
                    score -= 3
                    reasons.append(f"{surprise_pct:.1f}% surprise")

            # Consider stock reaction (oversold after beat = opportunity)
            if hasattr(result, 'is_oversold') and result.is_oversold and result.eps_beat:
                score += 5
                reasons.append("Oversold after beat")

            return score, " | ".join(reasons) if reasons else "Earnings analyzed"
    except ImportError:
        pass  # reaction_analyzer not available
    except Exception as e:
        logger.debug(f"reaction_analyzer error for {ticker}: {e}")

    # METHOD 2: Fall back to yfinance earnings_history
    try:
        import yfinance as yf
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stock = yf.Ticker(ticker)
            hist = stock.earnings_history

        if hist is not None and not hist.empty:
            beats = 0
            misses = 0
            total_quarters = 0
            latest_surprise_pct = None

            for idx, row in hist.iterrows():
                actual = row.get('epsActual')
                estimate = row.get('epsEstimate')

                if pd.notna(actual) and pd.notna(estimate) and estimate != 0:
                    total_quarters += 1
                    surprise_pct = ((actual - estimate) / abs(estimate)) * 100

                    if total_quarters == 1:
                        latest_surprise_pct = surprise_pct

                    if actual > estimate:
                        beats += 1
                    elif actual < estimate:
                        misses += 1

            if total_quarters > 0:
                beat_rate = (beats / total_quarters) * 100

                if beat_rate >= 100:
                    score += 10
                    reasons.append(f"Perfect beat streak ({beats}/{total_quarters})")
                elif beat_rate >= 75:
                    score += 7
                    reasons.append(f"Strong beat history ({beats}/{total_quarters})")
                elif beat_rate >= 50:
                    score += 3
                    reasons.append(f"Beat rate {beat_rate:.0f}%")
                elif beat_rate <= 25:
                    score -= 5
                    reasons.append(f"Frequent misses ({misses}/{total_quarters})")

                if latest_surprise_pct is not None:
                    if latest_surprise_pct > 10:
                        score += 3
                        reasons.append(f"Last Q: +{latest_surprise_pct:.1f}%")
                    elif latest_surprise_pct < -10:
                        score -= 3
                        reasons.append(f"Last Q: {latest_surprise_pct:.1f}%")

                return score, " | ".join(reasons) if reasons else f"Beat rate: {beat_rate:.0f}%"
    except Exception as e:
        logger.debug(f"yfinance earnings error for {ticker}: {e}")

    return 0, "Earnings history unavailable"


# =============================================================================
# MAIN ENHANCED SCORING FUNCTION
# =============================================================================

def get_enhanced_scores(
    ticker: str,
    current_price: float = None,
    target_price: float = None,
    pe_ratio: float = None,
    forward_pe: float = None,
    peg_ratio: float = None,
    sector: str = None,
    buy_count: int = 0,
    total_ratings: int = 0,
    price_history: pd.Series = None,
    volume_history: pd.Series = None,
) -> EnhancedScores:
    """
    Calculate all enhanced scores for a ticker.

    Args:
        ticker: Stock ticker symbol
        current_price: Current stock price
        target_price: Analyst price target
        pe_ratio: Current PE ratio
        forward_pe: Forward PE ratio
        peg_ratio: PEG ratio
        sector: Stock sector
        buy_count: Number of buy ratings
        total_ratings: Total analyst ratings
        price_history: Historical prices (pandas Series)
        volume_history: Historical volumes (pandas Series)

    Returns:
        EnhancedScores dataclass with all component scores
    """
    scores = EnhancedScores(ticker=ticker)

    # Auto-fetch missing data from database if needed
    if DB_AVAILABLE and (not pe_ratio or not sector or not peg_ratio):
        try:
            fund_query = """
                SELECT pe_ratio, forward_pe, peg_ratio, sector, revenue_growth, earnings_growth
                FROM fundamentals
                WHERE ticker = %s
                ORDER BY date DESC LIMIT 1
            """
            fund_df = pd.read_sql(fund_query, get_engine(), params=(ticker,))
            if not fund_df.empty:
                fund_row = fund_df.iloc[0]
                if not pe_ratio:
                    pe_ratio = fund_row.get('pe_ratio')
                if not forward_pe:
                    forward_pe = fund_row.get('forward_pe')
                if not peg_ratio:
                    peg_ratio = fund_row.get('peg_ratio')
                if not sector:
                    sector = fund_row.get('sector')
                # Get growth rate for PEG calculation fallback
                growth_rate = fund_row.get('revenue_growth') or fund_row.get('earnings_growth')
        except Exception as e:
            logger.debug(f"Could not fetch fundamentals for {ticker}: {e}")
            growth_rate = None
    else:
        growth_rate = None

    # 1. Sector-relative PE
    if pe_ratio and sector:
        scores.pe_relative_score, reason = score_pe_relative(pe_ratio, sector, forward_pe)
        scores.details['pe_relative'] = reason

    # 2. PEG ratio (pass ticker for yfinance fallback)
    scores.peg_score, reason = score_peg_ratio(peg_ratio, sector, pe_ratio, growth_rate, ticker)
    scores.details['peg'] = reason

    # 3. Price target
    if current_price and target_price:
        scores.price_target_score, reason = score_price_target(
            current_price, target_price, buy_count, total_ratings
        )
        scores.details['price_target'] = reason

    # 4. MACD momentum
    if price_history is not None and len(price_history) >= 35:
        scores.macd_score, reason = score_macd(price_history)
        scores.details['macd'] = reason

    # 5. Volume profile
    if volume_history is not None and price_history is not None:
        scores.volume_score, reason = score_volume_profile(volume_history, price_history)
        scores.details['volume'] = reason

    # 6. Insider trading
    scores.insider_score, reason = score_insider_trading(ticker)
    scores.details['insider'] = reason

    # 7. Analyst revisions
    scores.revision_score, reason = score_analyst_revisions(ticker)
    scores.details['revisions'] = reason

    # 8. Earnings surprise
    scores.earnings_surprise_score, reason = score_earnings_surprise(ticker)
    scores.details['earnings_surprise'] = reason

    # Calculate total
    scores.calculate_total()

    return scores


def get_enhanced_score_adjustment(
    ticker: str,
    current_price: float = None,
    target_price: float = None,
    pe_ratio: float = None,
    forward_pe: float = None,
    peg_ratio: float = None,
    sector: str = None,
    buy_count: int = 0,
    total_ratings: int = 0,
    price_history: pd.Series = None,
    volume_history: pd.Series = None,
    max_adjustment: int = 30,
) -> Tuple[int, Dict[str, Any]]:
    """
    Get the score adjustment to add to the base total_score.

    Returns:
        Tuple of (clamped adjustment, details dict)
    """
    scores = get_enhanced_scores(
        ticker=ticker,
        current_price=current_price,
        target_price=target_price,
        pe_ratio=pe_ratio,
        forward_pe=forward_pe,
        peg_ratio=peg_ratio,
        sector=sector,
        buy_count=buy_count,
        total_ratings=total_ratings,
        price_history=price_history,
        volume_history=volume_history,
    )

    # Clamp the adjustment to prevent extreme swings
    adjustment = max(-max_adjustment, min(max_adjustment, scores.enhancement_total))

    details = {
        'raw_adjustment': scores.enhancement_total,
        'clamped_adjustment': adjustment,
        'components': {
            'pe_relative': scores.pe_relative_score,
            'peg': scores.peg_score,
            'price_target': scores.price_target_score,
            'macd': scores.macd_score,
            'volume': scores.volume_score,
            'insider': scores.insider_score,
            'revisions': scores.revision_score,
            'earnings_surprise': scores.earnings_surprise_score,
        },
        'reasons': scores.details,
    }

    return adjustment, details


# =============================================================================
# CONVENIENCE FUNCTION FOR SIGNALS TAB
# =============================================================================

def calculate_enhanced_total_score(
    base_score: int,
    ticker: str,
    row_data: Dict[str, Any],
    price_history: pd.Series = None,
    volume_history: pd.Series = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Calculate enhanced total score from base score + enhancements.


    Args:
        base_score: Original total_score from weighted components
        ticker: Stock ticker
        row_data: Dict with price, target, PE, sector, etc.
        price_history: Price series for MACD
        volume_history: Volume series for accumulation/distribution

    Returns:
        Tuple of (enhanced_total_score, enhancement_details)
    """
    adjustment, details = get_enhanced_score_adjustment(
        ticker=ticker,
        current_price=row_data.get('price') or row_data.get('Price'),
        target_price=row_data.get('target_mean') or row_data.get('TargetPrice'),
        pe_ratio=row_data.get('pe_ratio') or row_data.get('PE'),
        forward_pe=row_data.get('forward_pe'),
        peg_ratio=row_data.get('peg_ratio'),
        sector=row_data.get('sector') or row_data.get('Sector'),
        buy_count=row_data.get('buy_count') or row_data.get('BuyRatings') or 0,
        total_ratings=row_data.get('total_ratings') or row_data.get('TotalRatings') or 0,
        price_history=price_history,
        volume_history=volume_history,
        max_adjustment=30,
    )

    # Apply adjustment to base score
    enhanced_score = max(0, min(100, base_score + adjustment))

    details['base_score'] = base_score
    details['enhanced_score'] = enhanced_score

    return enhanced_score, details


# =============================================================================
# TEST / CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Scoring Module")
    parser.add_argument("ticker", help="Stock ticker to analyze")
    parser.add_argument("--base-score", type=int, default=50, help="Base score to enhance")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Enhanced Scoring Analysis: {args.ticker}")
    print(f"{'='*60}")

    # Get scores
    scores = get_enhanced_scores(args.ticker)

    print(f"\n📊 Component Scores:")
    print(f"   PE Relative:      {scores.pe_relative_score:+3d}  {scores.details.get('pe_relative', '')}")
    print(f"   PEG Ratio:        {scores.peg_score:+3d}  {scores.details.get('peg', '')}")
    print(f"   Price Target:     {scores.price_target_score:+3d}  {scores.details.get('price_target', '')}")
    print(f"   MACD:             {scores.macd_score:+3d}  {scores.details.get('macd', '')}")
    print(f"   Volume:           {scores.volume_score:+3d}  {scores.details.get('volume', '')}")
    print(f"   Insider Trading:  {scores.insider_score:+3d}  {scores.details.get('insider', '')}")
    print(f"   Analyst Revisions:{scores.revision_score:+3d}  {scores.details.get('revisions', '')}")
    print(f"   Earnings Surprise:{scores.earnings_surprise_score:+3d}  {scores.details.get('earnings_surprise', '')}")
    print(f"\n   {'─'*50}")
    print(f"   TOTAL ADJUSTMENT: {scores.enhancement_total:+3d}")
    print(f"\n   Base Score:     {args.base_score}")
    print(f"   Enhanced Score: {max(0, min(100, args.base_score + scores.enhancement_total))}")
    print(f"{'='*60}\n")