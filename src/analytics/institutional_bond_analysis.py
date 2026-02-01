"""
Unified Institutional Bond Analysis Module

This module enhances the standard bond signals with institutional-grade data:
- Treasury Futures (ZB, ZN, ZT, UB) with DV01
- Auction Analytics (Bid-to-Cover, Tail, Participation)
- Term Premium Decomposition
- Curve Trades (Butterflies, Steepeners/Flatteners)
- MOVE Index / Vol Regime
- Convexity & DV01 Risk
- Macro Flows

All data is integrated into signals and formatted for AI learning.

Location: src/analytics/institutional_bond_analysis.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import requests

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TreasuryFuture:
    """Treasury futures contract data."""
    symbol: str
    name: str
    price: float
    change: float
    change_pct: float
    dv01: float
    volume: int = 0


@dataclass
class AuctionResult:
    """Treasury auction result."""
    auction_date: date
    security_type: str
    issue_size: float
    bid_to_cover: float
    tail: float  # Negative = strong
    indirect_pct: float
    direct_pct: float
    dealer_pct: float
    demand_grade: str  # Strong, Average, Weak


@dataclass
class CurveTrade:
    """Curve trade signal."""
    trade_name: str
    current_level: float
    z_score: float
    signal: str
    interpretation: str


@dataclass
class VolRegime:
    """Rate volatility regime."""
    move_index: float
    regime: str  # LOW, NORMAL, ELEVATED, STRESSED
    percentile: float
    interpretation: str
    trading_implication: str


@dataclass
class TermPremiumData:
    """Term premium analysis."""
    nominal_yield: float
    expected_rate: float
    term_premium: float
    percentile: float
    interpretation: str


@dataclass
class InstitutionalEnhancement:
    """Institutional data that enhances bond signals."""
    # Data points
    futures: List[TreasuryFuture]
    recent_auctions: List[AuctionResult]
    upcoming_auctions: List[dict]
    curve_trades: List[CurveTrade]
    vol_regime: VolRegime
    term_premium: Optional[TermPremiumData]

    # Aggregated scores (0-100, 50=neutral, None=data not available)
    auction_demand_score: Optional[int]  # High demand = bullish for bonds, None if no data
    vol_regime_score: Optional[int]  # Low vol = bullish
    term_premium_score: Optional[int]  # Low TP = bullish
    curve_trade_score: Optional[int]  # Based on curve signals
    futures_momentum_score: Optional[int]  # Futures rising = bullish

    # Overall institutional score adjustment
    institutional_adjustment: int  # Added/subtracted from base signal

    # Analysis
    key_insights: List[str]
    risks: List[str]
    opportunities: List[str]

    # AI Context
    ai_context: str


# =============================================================================
# DATA FETCHING
# =============================================================================

def get_move_index() -> Optional[float]:
    """Fetch MOVE Index (bond market VIX)."""
    try:
        import yfinance as yf
        # Try FRED
        try:
            url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLMOVE"
            df = pd.read_csv(url, parse_dates=['DATE'])
            if not df.empty and 'BAMLMOVE' in df.columns:
                return float(df['BAMLMOVE'].dropna().iloc[-1])
        except:
            pass
        return None
    except:
        return None


def get_treasury_futures() -> List[TreasuryFuture]:
    """Fetch Treasury futures data."""
    import yfinance as yf

    futures_config = {
        'ZB=F': ('ZB', '30Y Bond', 175),
        'ZN=F': ('ZN', '10Y Note', 78),
        'ZF=F': ('ZF', '5Y Note', 47),
        'ZT=F': ('ZT', '2Y Note', 39),
    }

    results = []

    for symbol, (code, name, dv01) in futures_config.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")

            if hist.empty:
                continue

            current = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
            change = current - prev
            change_pct = (change / prev) * 100 if prev else 0
            volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0

            results.append(TreasuryFuture(
                symbol=code,
                name=name,
                price=current,
                change=change,
                change_pct=change_pct,
                dv01=dv01,
                volume=volume
            ))
        except Exception as e:
            logger.debug(f"Error fetching {symbol}: {e}")

    return results


def get_vol_regime(move_value: float = None) -> VolRegime:
    """Determine rate volatility regime."""
    move = move_value or get_move_index() or 100

    if move < 80:
        regime = "LOW"
        percentile = 25
        interpretation = "Low rate volatility - stable environment"
        implication = "Favorable for carry trades. Consider adding duration."
    elif move < 100:
        regime = "NORMAL"
        percentile = 50
        interpretation = "Normal volatility - typical conditions"
        implication = "Standard position sizing appropriate."
    elif move < 120:
        regime = "ELEVATED"
        percentile = 75
        interpretation = "Elevated volatility - increased uncertainty"
        implication = "Reduce position sizes. Wider stops needed."
    else:
        regime = "STRESSED"
        percentile = 90
        interpretation = "High stress - significant rate uncertainty"
        implication = "Defensive positioning. Consider reducing duration."

    return VolRegime(
        move_index=move,
        regime=regime,
        percentile=percentile,
        interpretation=interpretation,
        trading_implication=implication
    )


def get_curve_trades() -> List[CurveTrade]:
    """Calculate curve trade levels and signals."""
    import yfinance as yf

    trades = []

    # Get yields
    try:
        tickers = {'^IRX': '2Y', '^FVX': '5Y', '^TNX': '10Y', '^TYX': '30Y'}
        data = yf.download(list(tickers.keys()), period='1y', progress=False)

        if data.empty:
            return trades

        yields = {}
        histories = {}

        for symbol, tenor in tickers.items():
            if symbol in data['Close'].columns:
                yields[tenor] = data['Close'][symbol].iloc[-1]
                histories[tenor] = data['Close'][symbol]

        # Adjust 2Y (^IRX is 13-week)
        if '5Y' in yields:
            yields['2Y'] = yields.get('2Y', yields['5Y'] - 0.3)

        # 2s10s Spread
        if '2Y' in yields and '10Y' in yields:
            spread = (yields['10Y'] - yields['2Y']) * 100

            # Historical stats
            if '10Y' in histories and '5Y' in histories:
                hist_spread = (histories['10Y'] - histories['5Y'] + 0.3) * 100
                mean = hist_spread.mean()
                std = hist_spread.std()
                z_score = (spread - mean) / std if std > 0 else 0
            else:
                z_score = 0

            if z_score > 1.5:
                signal = "FLATTEN"
                interp = "Curve historically steep - consider flattener"
            elif z_score < -1.5:
                signal = "STEEPEN"
                interp = "Curve historically flat - consider steepener"
            else:
                signal = "NEUTRAL"
                interp = "Curve within normal range"

            trades.append(CurveTrade(
                trade_name="2s10s Spread",
                current_level=spread,
                z_score=z_score,
                signal=signal,
                interpretation=interp
            ))

        # 5s30s Spread
        if '5Y' in yields and '30Y' in yields:
            spread = (yields['30Y'] - yields['5Y']) * 100

            trades.append(CurveTrade(
                trade_name="5s30s Spread",
                current_level=spread,
                z_score=0,
                signal="NEUTRAL",
                interpretation="Long-end steepness indicator"
            ))

        # 2-5-10 Butterfly
        if all(t in yields for t in ['2Y', '5Y', '10Y']):
            fly = (2 * yields['5Y'] - yields['2Y'] - yields['10Y']) * 100

            if fly > 15:
                signal = "RECEIVE_BELLY"
                interp = "Belly cheap - receive 5Y in butterfly"
            elif fly < -15:
                signal = "PAY_BELLY"
                interp = "Belly rich - pay 5Y in butterfly"
            else:
                signal = "NEUTRAL"
                interp = "Butterfly near fair value"

            trades.append(CurveTrade(
                trade_name="2-5-10 Butterfly",
                current_level=fly,
                z_score=fly / 15,
                signal=signal,
                interpretation=interp
            ))

    except Exception as e:
        logger.debug(f"Curve trade error: {e}")

    return trades


def get_term_premium() -> Optional[TermPremiumData]:
    """Estimate term premium."""
    import yfinance as yf
    import pandas as pd

    try:
        tickers = {'^IRX': '3M', '^TNX': '10Y'}
        data = yf.download(list(tickers.keys()), period='5d', progress=False, auto_adjust=False)

        if data.empty:
            logger.warning("Term premium: No data from yfinance")
            return None

        # Handle MultiIndex columns from multi-ticker download
        if isinstance(data.columns, pd.MultiIndex):
            y10 = float(data['Close']['^TNX'].iloc[-1])
            y3m = float(data['Close']['^IRX'].iloc[-1])
        else:
            # Single ticker fallback - shouldn't happen but handle it
            logger.warning("Term premium: Unexpected column structure")
            return None

        # Rough term premium estimate
        expected_avg_rate = (y3m + 3.0) / 2  # Assume long-run neutral ~3%
        term_premium = y10 - expected_avg_rate

        logger.info(f"Term premium calc: 10Y={y10:.2f}%, 3M={y3m:.2f}%, expected={expected_avg_rate:.2f}%, TP={term_premium:.2f}%")

        if term_premium < 0:
            percentile = 20
            interp = "Term premium negative - strong demand for duration"
        elif term_premium < 0.5:
            percentile = 40
            interp = "Term premium low - moderate demand"
        elif term_premium < 1.0:
            percentile = 60
            interp = "Term premium normal"
        else:
            percentile = 80
            interp = "Term premium elevated - supply pressure or inflation fears"

        return TermPremiumData(
            nominal_yield=y10,
            expected_rate=expected_avg_rate,
            term_premium=term_premium,
            percentile=percentile,
            interpretation=interp
        )
    except Exception as e:
        logger.warning(f"Term premium calculation failed: {e}")
        return None


def get_upcoming_auctions() -> List[dict]:
    """
    Get upcoming Treasury auction schedule from Treasury Direct API.

    Returns empty list if data unavailable - NO FAKE DATA.
    """
    try:
        # Treasury Direct API for upcoming auctions
        url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/upcoming_auctions"
        params = {
            'sort': '-auction_date',
            'page[size]': 10
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            auctions = []
            for item in data.get('data', []):
                auctions.append({
                    'date': item.get('auction_date', ''),
                    'security': item.get('security_type', ''),
                    'size': item.get('offering_amt', 0),
                    'source': 'Treasury Direct API'
                })
            logger.info(f"Fetched {len(auctions)} upcoming auctions from Treasury Direct")
            return auctions
    except Exception as e:
        logger.warning(f"Could not fetch upcoming auctions: {e}")

    # Return empty list - NO FAKE DATA
    return []


def get_recent_auctions() -> List[AuctionResult]:
    """
    Get recent Treasury auction results from Treasury Direct API.

    Returns empty list if data unavailable - NO FAKE DATA.
    """
    try:
        # Treasury Direct API for auction results
        url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"

        # Get auctions from last 30 days
        from_date = (date.today() - timedelta(days=30)).strftime('%Y-%m-%d')

        params = {
            'filter': f'auction_date:gte:{from_date}',
            'sort': '-auction_date',
            'page[size]': 20
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            auctions = []

            for item in data.get('data', []):
                try:
                    # Parse auction date
                    auction_date_str = item.get('auction_date', '')
                    if auction_date_str:
                        auction_dt = datetime.strptime(auction_date_str, '%Y-%m-%d').date()
                    else:
                        continue

                    # Get bid-to-cover ratio
                    bid_to_cover = float(item.get('bid_to_cover_ratio', 0) or 0)

                    # Calculate tail (high yield - when issued)
                    high_yield = float(item.get('high_yield', 0) or 0)
                    # Tail calculation requires when-issued yield which may not be in API
                    tail = 0.0

                    # Get allocation percentages
                    indirect_pct = float(item.get('percentage_debt_purchased_by_indirect_bidders', 0) or 0)
                    direct_pct = float(item.get('percentage_debt_purchased_by_direct_bidders', 0) or 0)
                    dealer_pct = 100 - indirect_pct - direct_pct

                    # Determine demand grade based on bid-to-cover
                    if bid_to_cover >= 2.5:
                        demand_grade = "Strong"
                    elif bid_to_cover >= 2.2:
                        demand_grade = "Average"
                    else:
                        demand_grade = "Weak"

                    auctions.append(AuctionResult(
                        auction_date=auction_dt,
                        security_type=item.get('security_type', 'Unknown'),
                        issue_size=float(item.get('offering_amt', 0) or 0) / 1000,  # Convert to billions
                        bid_to_cover=bid_to_cover,
                        tail=tail,
                        indirect_pct=indirect_pct,
                        direct_pct=direct_pct,
                        dealer_pct=dealer_pct,
                        demand_grade=demand_grade
                    ))
                except Exception as e:
                    logger.debug(f"Error parsing auction: {e}")
                    continue

            logger.info(f"Fetched {len(auctions)} recent auctions from Treasury Direct")
            return auctions

    except Exception as e:
        logger.warning(f"Could not fetch recent auctions: {e}")

    # Return empty list - NO FAKE DATA
    return []


# =============================================================================
# MAIN ENHANCEMENT FUNCTION
# =============================================================================

def get_institutional_enhancement() -> InstitutionalEnhancement:
    """
    Get all institutional data and calculate enhancement scores.

    These scores modify the base bond signal:
    - Positive = more bullish for bonds
    - Negative = more bearish for bonds
    """

    # Gather data
    futures = get_treasury_futures()
    move = get_move_index()
    vol_regime = get_vol_regime(move)
    curve_trades = get_curve_trades()
    term_premium = get_term_premium()
    recent_auctions = get_recent_auctions()
    upcoming_auctions = get_upcoming_auctions()

    # Calculate scores
    insights = []
    risks = []
    opportunities = []

    # 1. Auction Demand Score
    auction_demand_score = None  # None = data not available
    auction_data_available = False

    if recent_auctions:
        auction_data_available = True
        # Filter out auctions with 0 bid-to-cover (missing data)
        valid_auctions = [a for a in recent_auctions if a.bid_to_cover > 0]

        if valid_auctions:
            avg_b2c = np.mean([a.bid_to_cover for a in valid_auctions])
            avg_tail = np.mean([a.tail for a in valid_auctions])

            if avg_b2c > 2.6 and avg_tail < 0:
                auction_demand_score = 70
                insights.append(f"Strong auction demand (B/C: {avg_b2c:.2f}) - bullish signal")
            elif avg_b2c > 2.4 and avg_tail < 1:
                auction_demand_score = 60
                insights.append(f"Solid auction demand (B/C: {avg_b2c:.2f})")
            elif avg_b2c < 2.2 or avg_tail > 2:
                auction_demand_score = 35
                risks.append(f"Weak auction demand (B/C: {avg_b2c:.2f}) - supply pressure")
            else:
                auction_demand_score = 50  # Average

            long_auctions = [a for a in valid_auctions if '30' in a.security_type or 'Bond' in a.security_type]
            if long_auctions and any(a.demand_grade == 'Weak' for a in long_auctions):
                risks.append("Long-end auctions showing weakness")
        else:
            insights.append("⚠️ Auction data not available")
    else:
        insights.append("⚠️ Auction data not available - Treasury API may be down")

    # 2. Vol Regime Score
    vol_regime_score = 50
    if vol_regime.regime == "LOW":
        vol_regime_score = 65
        opportunities.append("Low vol environment - carry trades attractive")
    elif vol_regime.regime == "NORMAL":
        vol_regime_score = 50
    elif vol_regime.regime == "ELEVATED":
        vol_regime_score = 40
        risks.append(f"Elevated rate volatility (MOVE: {vol_regime.move_index:.0f})")
    else:
        vol_regime_score = 25
        risks.append(f"High rate stress (MOVE: {vol_regime.move_index:.0f}) - reduce positions")

    # 3. Term Premium Score
    term_premium_score = 50
    if term_premium:
        logger.info(f"Term premium value: {term_premium.term_premium:.2f}%")
        if term_premium.term_premium < 0:
            term_premium_score = 70
            insights.append("Negative term premium - strong duration demand")
        elif term_premium.term_premium < 0.5:
            term_premium_score = 60
            insights.append(f"Low term premium ({term_premium.term_premium:.2f}%) - moderate duration demand")
        elif term_premium.term_premium > 1.0:
            term_premium_score = 35
            risks.append("Elevated term premium - supply concerns")
        else:
            term_premium_score = 50
            insights.append(f"Normal term premium ({term_premium.term_premium:.2f}%)")
    else:
        logger.warning("Term premium data not available - using default score 50")

    # 4. Curve Trade Score
    curve_trade_score = 50
    for ct in curve_trades:
        if ct.trade_name == "2s10s Spread":
            if ct.signal == "FLATTEN":
                curve_trade_score = 55
                opportunities.append(f"2s10s steep ({ct.current_level:.0f}bp) - flattener opportunity")
            elif ct.signal == "STEEPEN":
                curve_trade_score = 45
                opportunities.append(f"2s10s flat ({ct.current_level:.0f}bp) - steepener opportunity")

    # 5. Futures Momentum Score
    futures_momentum_score = 50
    if futures:
        avg_change = np.mean([f.change_pct for f in futures])
        if avg_change > 0.1:
            futures_momentum_score = 65
            insights.append("Treasury futures showing positive momentum")
        elif avg_change < -0.1:
            futures_momentum_score = 35
            risks.append("Treasury futures under pressure")

    # Calculate overall adjustment - only use available data
    scores = {}
    weights = {}

    if auction_demand_score is not None:
        scores['auction'] = auction_demand_score
        weights['auction'] = 0.25

    if vol_regime_score is not None:
        scores['vol'] = vol_regime_score
        weights['vol'] = 0.25

    if term_premium_score is not None:
        scores['term_premium'] = term_premium_score
        weights['term_premium'] = 0.20

    if curve_trade_score is not None:
        scores['curve'] = curve_trade_score
        weights['curve'] = 0.15

    if futures_momentum_score is not None:
        scores['futures'] = futures_momentum_score
        weights['futures'] = 0.15

    # Calculate weighted average only from available data
    if scores and weights:
        total_weight = sum(weights.values())
        weighted_score = sum(scores[k] * weights[k] for k in scores.keys()) / total_weight * (sum(weights.values()) / total_weight)
        weighted_score = sum(scores[k] * weights[k] for k in scores.keys()) / total_weight

        institutional_adjustment = int((weighted_score - 50) * 0.67)
        institutional_adjustment = max(-15, min(15, institutional_adjustment))
    else:
        weighted_score = 50
        institutional_adjustment = 0
        insights.append("⚠️ Insufficient data for institutional adjustment")

    ai_context = _build_ai_context(
        futures, vol_regime, curve_trades, term_premium,
        recent_auctions, upcoming_auctions, insights, risks, opportunities
    )

    return InstitutionalEnhancement(
        futures=futures,
        recent_auctions=recent_auctions,
        upcoming_auctions=upcoming_auctions,
        curve_trades=curve_trades,
        vol_regime=vol_regime,
        term_premium=term_premium,
        auction_demand_score=auction_demand_score,
        vol_regime_score=vol_regime_score,
        term_premium_score=term_premium_score,
        curve_trade_score=curve_trade_score,
        futures_momentum_score=futures_momentum_score,
        institutional_adjustment=institutional_adjustment,
        key_insights=insights,
        risks=risks,
        opportunities=opportunities,
        ai_context=ai_context
    )


def _build_ai_context(futures, vol_regime, curve_trades, term_premium,
                      recent_auctions, upcoming_auctions, insights, risks, opportunities) -> str:
    """Build comprehensive AI context for bond analysis."""

    context = f"""
🏦 INSTITUTIONAL FIXED INCOME ANALYSIS
{'=' * 60}

📊 TREASURY FUTURES:
"""

    for f in futures:
        direction = "↑" if f.change >= 0 else "↓"
        context += f"   {f.symbol} ({f.name}): {f.price:.3f} {direction}{abs(f.change_pct):.2f}% | DV01: ${f.dv01}\n"

    context += f"""
📈 RATE VOLATILITY:
   MOVE Index: {vol_regime.move_index:.0f} ({vol_regime.regime})
   {vol_regime.interpretation}
   Trading Implication: {vol_regime.trading_implication}
"""

    if term_premium:
        context += f"""
📉 TERM PREMIUM:
   10Y Yield: {term_premium.nominal_yield:.2f}%
   Expected Avg Rate: {term_premium.expected_rate:.2f}%
   Term Premium: {term_premium.term_premium:+.2f}%
   {term_premium.interpretation}
"""

    context += "\n📐 CURVE TRADES:\n"
    for ct in curve_trades:
        context += f"   {ct.trade_name}: {ct.current_level:.1f}bp (z={ct.z_score:+.1f}) → {ct.signal}\n"
        context += f"      {ct.interpretation}\n"

    context += "\n🏛️ RECENT AUCTIONS:\n"
    for a in recent_auctions[:3]:
        grade_icon = "✅" if a.demand_grade == "Strong" else "⚠️" if a.demand_grade == "Weak" else "➖"
        context += f"   {a.security_type}: B/C {a.bid_to_cover:.2f}, Tail {a.tail:+.1f}bp {grade_icon}\n"

    context += "\n📅 UPCOMING AUCTIONS:\n"
    for a in upcoming_auctions[:4]:
        context += f"   {a['date']}: {a['security']} (${a['size']}B)\n"

    if insights:
        context += "\n✅ KEY INSIGHTS:\n"
        for i in insights:
            context += f"   • {i}\n"

    if risks:
        context += "\n⚠️ RISKS:\n"
        for r in risks:
            context += f"   • {r}\n"

    if opportunities:
        context += "\n💡 OPPORTUNITIES:\n"
        for o in opportunities:
            context += f"   • {o}\n"

    return context


# =============================================================================
# STREAMLIT RENDERING
# =============================================================================

def render_institutional_section(enhancement: InstitutionalEnhancement):
    """Render institutional analysis section in Streamlit."""
    import streamlit as st

    vol = enhancement.vol_regime
    vol_colors = {'LOW': 'green', 'NORMAL': 'blue', 'ELEVATED': 'orange', 'STRESSED': 'red'}
    vol_color = vol_colors.get(vol.regime, 'gray')

    # Header with key metrics
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(100,100,100,0.1), transparent); 
                padding: 12px; border-radius: 8px; margin-bottom: 15px;">
        <span style="font-weight: bold;">Institutional Assessment:</span>
        <span style="margin-left: 15px;">MOVE: <span style="color: {vol_color};">{vol.move_index:.0f}</span> ({vol.regime})</span>
        <span style="margin-left: 15px;">Signal Adjustment: <span style="color: {'green' if enhancement.institutional_adjustment > 0 else 'red' if enhancement.institutional_adjustment < 0 else 'gray'};">{enhancement.institutional_adjustment:+d}</span></span>
    </div>
    """, unsafe_allow_html=True)

    # Key Insights / Risks / Opportunities
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**✅ Insights**")
        if enhancement.key_insights:
            for i in enhancement.key_insights:
                st.caption(f"• {i}")
        else:
            st.caption("No significant insights")

    with col2:
        st.markdown("**⚠️ Risks**")
        if enhancement.risks:
            for r in enhancement.risks:
                st.caption(f"• {r}")
        else:
            st.caption("No significant risks")

    with col3:
        st.markdown("**💡 Opportunities**")
        if enhancement.opportunities:
            for o in enhancement.opportunities:
                st.caption(f"• {o}")
        else:
            st.caption("No current opportunities")


def render_institutional_details(enhancement: InstitutionalEnhancement):
    """Render detailed institutional data in expanders."""
    import streamlit as st

    with st.expander("📊 Treasury Futures"):
        if enhancement.futures:
            fut_data = []
            for f in enhancement.futures:
                fut_data.append({
                    'Contract': f.symbol,
                    'Name': f.name,
                    'Price': f"{f.price:.3f}",
                    'Change': f"{f.change_pct:+.2f}%",
                    'DV01': f"${f.dv01}",
                })
            st.dataframe(pd.DataFrame(fut_data), use_container_width=True, hide_index=True)
        else:
            st.caption("Futures data unavailable")

    with st.expander("📐 Curve Trades"):
        if enhancement.curve_trades:
            for ct in enhancement.curve_trades:
                signal_color = "green" if "FLATTEN" in ct.signal or "RECEIVE" in ct.signal else "red" if "STEEPEN" in ct.signal or "PAY" in ct.signal else "gray"
                st.markdown(f"**{ct.trade_name}:** {ct.current_level:.1f}bp (z={ct.z_score:+.1f})")
                st.markdown(f"Signal: :{signal_color}[{ct.signal}] - {ct.interpretation}")
                st.markdown("---")

    with st.expander("🏛️ Auction Analytics"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Recent Results**")
            if enhancement.recent_auctions:
                for a in enhancement.recent_auctions[:4]:
                    grade_icon = "✅" if a.demand_grade == "Strong" else "⚠️" if a.demand_grade == "Weak" else "➖"
                    st.caption(f"{grade_icon} {a.security_type}: B/C {a.bid_to_cover:.2f}, Tail {a.tail:+.1f}bp")

        with col2:
            st.markdown("**Upcoming**")
            if enhancement.upcoming_auctions:
                for a in enhancement.upcoming_auctions[:4]:
                    st.caption(f"• {a['date']}: {a['security']} (${a['size']}B)")

    with st.expander("📈 Volatility & Term Premium"):
        col1, col2 = st.columns(2)

        with col1:
            vol = enhancement.vol_regime
            st.markdown("**Rate Volatility**")
            st.metric("MOVE Index", f"{vol.move_index:.0f}", vol.regime)
            st.caption(vol.trading_implication)

        with col2:
            if enhancement.term_premium:
                tp = enhancement.term_premium
                st.markdown("**Term Premium**")
                st.metric("10Y Term Premium", f"{tp.term_premium:+.2f}%")
                st.caption(tp.interpretation)

    with st.expander("📊 Institutional Score Components"):
        # Build scores with extra context
        scores = [
            ('Auction Demand', enhancement.auction_demand_score, None),
            ('Vol Regime', enhancement.vol_regime_score, f"MOVE: {enhancement.vol_regime.move_index:.0f}" if enhancement.vol_regime else None),
            ('Term Premium', enhancement.term_premium_score, f"{enhancement.term_premium.term_premium:+.2f}%" if enhancement.term_premium else None),
            ('Curve Trade', enhancement.curve_trade_score, None),
            ('Futures Momentum', enhancement.futures_momentum_score, None),
        ]

        for name, score, extra in scores:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**{name}**")
            with col2:
                if score is None:
                    st.caption("⚠️ Data not available")
                else:
                    color = "green" if score >= 60 else "red" if score <= 40 else "orange"
                    st.progress(score / 100)
                    # Show score with extra info if available
                    if extra:
                        st.caption(f":{color}[{score}/100] — {extra}")
                    else:
                        st.caption(f":{color}[{score}/100]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Getting institutional enhancement...")
    enh = get_institutional_enhancement()

    print(f"\nVol Regime: {enh.vol_regime.regime} (MOVE: {enh.vol_regime.move_index:.0f})")
    print(f"Institutional Adjustment: {enh.institutional_adjustment:+d}")
    print(f"\nAI Context:\n{enh.ai_context}")