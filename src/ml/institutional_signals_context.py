"""
Institutional Signals Context Builder for AI Chat

This module gathers all institutional signals and formats them
with explanations so the AI model can properly interpret and use the data.

Signals included:
- GEX/Gamma Analysis (options market maker positioning)
- Dark Pool Flow (institutional accumulation/distribution)
- Cross-Asset Signals (macro environment, sector rotation)
- Sentiment NLP (AI-powered news analysis)
- Earnings Whisper (beat/miss prediction)
- Insider Transactions (Form 4 - 2 day lag)
- 13F Institutional Holdings (45+ day lag)
- Pairs/Correlation (hedging candidates)

Location: src/ml/institutional_signals_context.py
"""

from typing import Dict, Any, Optional
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ============================================================================
# IMPORTS - Phase 2 & 3 modules
# ============================================================================

# Phase 2: GEX/Gamma
try:
    from src.analytics.gex_analysis import analyze_gex, GEXRegime
    GEX_AVAILABLE = True
except ImportError:
    GEX_AVAILABLE = False

# Phase 2: Dark Pool
try:
    from src.analytics.dark_pool import analyze_dark_pool, DarkPoolSentiment
    DARK_POOL_AVAILABLE = True
except ImportError:
    DARK_POOL_AVAILABLE = False

# Phase 2: Cross-Asset
try:
    from src.analytics.cross_asset import get_cross_asset_signals, CrossAssetSignal
    CROSS_ASSET_AVAILABLE = True
except ImportError:
    CROSS_ASSET_AVAILABLE = False

# Phase 3: Sentiment NLP
try:
    from src.analytics.sentiment_nlp import (
        analyze_news_sentiment,
        is_llm_available,
        SentimentLevel
    )
    SENTIMENT_NLP_AVAILABLE = True
except ImportError:
    SENTIMENT_NLP_AVAILABLE = False

# Phase 3: Earnings Whisper
try:
    from src.analytics.earnings_whisper import (
        get_earnings_whisper,
        EarningsPrediction,
        WhisperSignal
    )
    EARNINGS_WHISPER_AVAILABLE = True
except ImportError:
    EARNINGS_WHISPER_AVAILABLE = False

# Form 4 Insider Tracker
try:
    from src.analytics.insider_tracker import get_insider_signal
    INSIDER_TRACKER_AVAILABLE = True
except ImportError:
    INSIDER_TRACKER_AVAILABLE = False

# 13F Institutional Tracker
try:
    from src.analytics.institutional_13f_tracker import get_institutional_ownership
    INSTITUTIONAL_13F_AVAILABLE = True
except ImportError:
    INSTITUTIONAL_13F_AVAILABLE = False

# 13F Institutional Tracker
try:
    from src.analytics.institutional_13f_tracker import get_institutional_ownership
    INSTITUTIONAL_13F_AVAILABLE = True
except ImportError:
    INSTITUTIONAL_13F_AVAILABLE = False

# Pairs/Correlation Finder
try:
    from src.analytics.pairs_correlation import find_hedges, find_correlated_pairs
    PAIRS_CORRELATION_AVAILABLE = True
except ImportError:
    PAIRS_CORRELATION_AVAILABLE = False


# Cache for cross-asset (same for all tickers)
# Cache for cross-asset (same for all tickers)
_cross_asset_cache = None


def get_institutional_signals_context(ticker: str,
                                       current_price: float = None,
                                       sector: str = None,
                                       days_to_earnings: int = 999,
                                       news_headlines: list = None) -> str:
    """
    Gather all institutional signals and format as context for AI.

    Args:
        ticker: Stock symbol
        current_price: Current stock price
        sector: Stock sector
        days_to_earnings: Days until next earnings
        news_headlines: List of recent news headlines (optional)

    Returns:
        Formatted string with all signals and explanations for AI context
    """
    sections = []

    # Get current price if not provided
    if current_price is None or current_price <= 0:
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            current_price = stock.history(period='1d')['Close'].iloc[-1]
        except Exception:
            current_price = 0

    # ========================================================================
    # GEX/GAMMA ANALYSIS
    # ========================================================================
    if GEX_AVAILABLE and current_price > 0:
        try:
            gex = analyze_gex(ticker, current_price)

            gex_section = f"""
GEX/GAMMA ANALYSIS (Options Market Maker Positioning):
-------------------------------------------------------
What this tells you: GEX (Gamma Exposure) shows where market makers must hedge.
- Positive GEX = Market makers buy dips, sell rips → suppresses volatility, price tends to be pinned
- Negative GEX = Market makers sell dips, buy rips → amplifies moves, more volatile

Current Data:
- GEX Regime: {gex.gex_regime.value}
- Signal: {gex.signal} (strength: {gex.signal_strength}/100)
- Max Gamma Strike: ${gex.max_gamma_strike:.0f} (price gravitates here near expiration)
- Call Wall (resistance): ${gex.call_wall:.0f} (heavy call OI = hard to break above)
- Put Wall (support): ${gex.put_wall:.0f} (heavy put OI = hard to break below)
"""
            # Add pin proximity
            if gex.max_gamma_strike > 0:
                dist_to_pin = abs(current_price - gex.max_gamma_strike) / current_price * 100
                if dist_to_pin < 3:
                    gex_section += f"⚠️ PINNED: Price is {dist_to_pin:.1f}% from max gamma - expect range-bound action until options expire\n"

            # Add interpretation
            if gex.signal == "BULLISH":
                gex_section += "Interpretation: Options positioning is BULLISH - call buying dominates, upside pressure.\n"
            elif gex.signal == "BEARISH":
                gex_section += "Interpretation: Options positioning is BEARISH - put buying dominates, downside hedging.\n"
            elif gex.signal == "PINNED":
                gex_section += "Interpretation: Price likely PINNED near max gamma - avoid directional bets until post-expiration.\n"

            if gex.warnings:
                gex_section += f"Warnings: {'; '.join(gex.warnings)}\n"

            sections.append(gex_section)

        except Exception as e:
            logger.debug(f"GEX context error: {e}")

    # ========================================================================
    # DARK POOL ANALYSIS
    # ========================================================================
    if DARK_POOL_AVAILABLE:
        try:
            dp = analyze_dark_pool(ticker)

            dp_section = f"""
DARK POOL ANALYSIS (Institutional/Smart Money Flow):
----------------------------------------------------
What this tells you: Dark pools are private exchanges where institutions trade large blocks.
- ACCUMULATION = institutions quietly buying (bullish)
- DISTRIBUTION = institutions quietly selling (bearish)
- Block trades (10K+ shares) reveal smart money positioning

Current Data:
- Sentiment: {dp.sentiment.value}
- Dark Pool Score: {dp.sentiment_score}/100 (higher = more bullish institutional activity)
- Institutional Bias: {dp.institutional_bias}
- Block Buy Volume: {dp.block_buy_volume:,} shares
- Block Sell Volume: {dp.block_sell_volume:,} shares
"""
            # Add interpretation
            if dp.institutional_bias == "BUYING":
                dp_section += "Interpretation: Institutions are NET BUYERS - smart money accumulating. Bullish signal.\n"
            elif dp.institutional_bias == "SELLING":
                dp_section += "Interpretation: Institutions are NET SELLERS - smart money distributing. Bearish signal.\n"
            else:
                dp_section += "Interpretation: Institutional flow is NEUTRAL - no clear smart money direction.\n"

            # Net block analysis
            net_blocks = dp.block_buy_volume - dp.block_sell_volume
            if net_blocks > 100000:
                dp_section += f"⚡ Large net block buying: {net_blocks:,} shares - significant accumulation\n"
            elif net_blocks < -100000:
                dp_section += f"⚡ Large net block selling: {abs(net_blocks):,} shares - significant distribution\n"

            if dp.warnings:
                dp_section += f"Warnings: {'; '.join(dp.warnings)}\n"

            sections.append(dp_section)

        except Exception as e:
            logger.debug(f"Dark Pool context error: {e}")

    # ========================================================================
    # CROSS-ASSET ANALYSIS
    # ========================================================================
    if CROSS_ASSET_AVAILABLE:
        try:
            global _cross_asset_cache
            if _cross_asset_cache is None:
                _cross_asset_cache = get_cross_asset_signals()

            xa = _cross_asset_cache

            xa_section = f"""
CROSS-ASSET ANALYSIS (Macro Market Environment):
------------------------------------------------
What this tells you: How other asset classes (bonds, gold, dollar, etc.) affect stocks.
This is the same for all stocks - it shows the overall market environment.

Current Data:
- Primary Signal: {xa.primary_signal.value}
- Risk Appetite: {xa.risk_appetite} (RISK_ON = buy growth, RISK_OFF = buy defensive)
- Economic Cycle: {xa.cycle_phase.value}
- Favored Sectors: {', '.join(xa.favored_sectors[:5]) if xa.favored_sectors else 'None'}
- Avoid Sectors: {', '.join(xa.avoid_sectors[:3]) if xa.avoid_sectors else 'None'}
"""
            # Add sector-specific context
            if sector:
                if xa.favored_sectors and sector in xa.favored_sectors:
                    xa_section += f"✅ {sector} is FAVORED by cross-asset signals - tailwind for this stock.\n"
                elif xa.avoid_sectors and sector in xa.avoid_sectors:
                    xa_section += f"⚠️ {sector} is UNFAVORED by cross-asset signals - headwind for this stock.\n"

            # Add cycle interpretation
            if xa.cycle_phase.value == "EARLY_CYCLE":
                xa_section += "Cycle Guidance: Early cycle favors cyclicals, small caps, high-beta growth.\n"
            elif xa.cycle_phase.value == "MID_CYCLE":
                xa_section += "Cycle Guidance: Mid cycle favors quality growth, tech, industrials.\n"
            elif xa.cycle_phase.value == "LATE_CYCLE":
                xa_section += "Cycle Guidance: Late cycle favors energy, materials, defensives. Be cautious on duration.\n"
            elif xa.cycle_phase.value == "RECESSION":
                xa_section += "Cycle Guidance: Recession - favor utilities, staples, healthcare. Avoid cyclicals.\n"

            sections.append(xa_section)

        except Exception as e:
            logger.debug(f"Cross-asset context error: {e}")

    # ========================================================================
    # SENTIMENT NLP ANALYSIS
    # ========================================================================
    if SENTIMENT_NLP_AVAILABLE and news_headlines:
        try:
            sentiment = analyze_news_sentiment(ticker, news_headlines)

            nlp_section = f"""
SENTIMENT NLP ANALYSIS (AI-Powered News Analysis):
--------------------------------------------------
What this tells you: AI analysis of recent news headlines, not just keyword matching.
Analyzes tone, context, and implications.

Current Data:
- Sentiment Level: {sentiment.sentiment_level.value}
- Sentiment Score: {sentiment.sentiment_score}/100 (50 = neutral, >60 = bullish, <40 = bearish)
- Management/Source Tone: {sentiment.management_tone.value}
- Confidence: {sentiment.confidence:.0%}
"""
            if sentiment.key_positives:
                nlp_section += f"Key Positives: {'; '.join(sentiment.key_positives[:3])}\n"
            if sentiment.key_negatives:
                nlp_section += f"Key Negatives: {'; '.join(sentiment.key_negatives[:3])}\n"
            if sentiment.summary:
                nlp_section += f"Summary: {sentiment.summary}\n"

            # Add interpretation
            if sentiment.sentiment_score >= 70:
                nlp_section += "Interpretation: News sentiment is STRONGLY BULLISH - positive catalysts in play.\n"
            elif sentiment.sentiment_score >= 55:
                nlp_section += "Interpretation: News sentiment is MODERATELY BULLISH - generally positive coverage.\n"
            elif sentiment.sentiment_score <= 30:
                nlp_section += "Interpretation: News sentiment is STRONGLY BEARISH - negative catalysts/concerns.\n"
            elif sentiment.sentiment_score <= 45:
                nlp_section += "Interpretation: News sentiment is MODERATELY BEARISH - some concerns in coverage.\n"

            sections.append(nlp_section)

        except Exception as e:
            logger.debug(f"Sentiment NLP context error: {e}")

    # ========================================================================
    # EARNINGS WHISPER ANALYSIS
    # ========================================================================
    if EARNINGS_WHISPER_AVAILABLE:
        try:
            whisper = get_earnings_whisper(ticker)

            if whisper.days_to_earnings <= 60:  # Only relevant if earnings within 60 days
                whisper_section = f"""
EARNINGS WHISPER (Pre-Earnings Prediction):
-------------------------------------------
What this tells you: Predicts whether company will beat/miss earnings BEFORE the announcement.
Based on: analyst revision trends, options positioning, historical beat patterns.

Current Data:
- Earnings Date: {whisper.earnings_date} ({whisper.days_to_earnings} days away)
- Prediction: {whisper.prediction.value}
- Beat Probability: {whisper.beat_probability:.0f}%
- Expected Surprise: {whisper.expected_surprise_pct:+.1f}% (positive = beat by this %)
- Signal: {whisper.signal.value} (strength: {whisper.signal_strength}/100)

Component Breakdown:
- Revision Score: {whisper.revision_score}/100 (are analysts raising estimates? >50 = yes)
- Options Score: {whisper.options_score}/100 (is options market bullish? >50 = yes)
- Historical Score: {whisper.historical_score}/100 (does company usually beat? >50 = yes)
"""
                # Add interpretation
                if whisper.beat_probability >= 70:
                    whisper_section += "Interpretation: HIGH probability of earnings beat - multiple factors aligned bullish.\n"
                    whisper_section += "Trading Guidance: Consider holding through earnings or adding pre-earnings.\n"
                elif whisper.beat_probability >= 55:
                    whisper_section += "Interpretation: LIKELY to beat earnings - positive bias but not certain.\n"
                    whisper_section += "Trading Guidance: Moderate confidence for earnings hold.\n"
                elif whisper.beat_probability <= 30:
                    whisper_section += "Interpretation: HIGH probability of earnings MISS - multiple red flags.\n"
                    whisper_section += "Trading Guidance: Consider reducing position or hedging before earnings.\n"
                elif whisper.beat_probability <= 45:
                    whisper_section += "Interpretation: ELEVATED RISK of miss - be cautious.\n"
                    whisper_section += "Trading Guidance: Consider trimming or waiting until after earnings.\n"

                if whisper.warnings:
                    whisper_section += f"⚠️ Warnings: {'; '.join(whisper.warnings)}\n"

                # High expectations warning
                if whisper.high_expectations:
                    whisper_section += "⚠️ HIGH EXPECTATIONS: Multiple consecutive beats mean the bar is high.\n"

                sections.append(whisper_section)

        except Exception as e:
            logger.debug(f"Earnings whisper context error: {e}")

    # ========================================================================
    # INSIDER TRANSACTIONS (Form 4)
    # ========================================================================
    if INSIDER_TRACKER_AVAILABLE:
        try:
            insider = get_insider_signal(ticker)

            insider_section = f"""
INSIDER TRANSACTIONS (Form 4 - 2 Day Lag):
------------------------------------------
What this tells you: CEOs, CFOs, Directors buying/selling their own company stock.
This is HIGHLY actionable - insiders know their company best.
Filing deadline: 2 business days after transaction.

Current Data (Last 90 Days):
- Signal: {insider.signal} (strength: {insider.signal_strength}/100)
- Total Buys: {insider.total_buys} (${insider.buy_value:,.0f})
- Total Sells: {insider.total_sells} (${insider.sell_value:,.0f})
- Net Value: ${insider.net_value:,.0f} (positive = more buying)
- Unique Buyers: {insider.unique_buyers}
- Unique Sellers: {insider.unique_sellers}

Key Insider Activity:
- CEO Bought: {'YES ✅' if insider.ceo_bought else 'No'}
- CFO Bought: {'YES ✅' if insider.cfo_bought else 'No'}
- CEO Sold: {'YES ⚠️' if insider.ceo_sold else 'No'}
- CFO Sold: {'YES ⚠️' if insider.cfo_sold else 'No'}
- Cluster Buying (3+ insiders): {'YES 🔥' if insider.cluster_buying else 'No'}
- Cluster Selling: {'YES 🚨' if insider.cluster_selling else 'No'}
"""
            # Interpretation
            if insider.ceo_bought or insider.cfo_bought:
                insider_section += "Interpretation: C-SUITE BUYING - Very bullish signal. They know the company best.\n"
            if insider.cluster_buying:
                insider_section += "Interpretation: CLUSTER BUYING - Multiple insiders agree. Strong conviction.\n"
            if insider.cluster_selling:
                insider_section += "Interpretation: CLUSTER SELLING - Multiple insiders exiting. Concerning signal.\n"
            if insider.net_value > 500000:
                insider_section += f"Interpretation: Strong NET BUYING (${insider.net_value:,.0f}) - Insiders putting money where mouth is.\n"
            elif insider.net_value < -500000:
                insider_section += f"Interpretation: Strong NET SELLING (${abs(insider.net_value):,.0f}) - Insiders taking profits or concerned.\n"

            sections.append(insider_section)

        except Exception as e:
            logger.debug(f"Insider context error: {e}")

    # ========================================================================
    # 13F INSTITUTIONAL HOLDINGS
    # ========================================================================
    if INSTITUTIONAL_13F_AVAILABLE:
        try:
            ownership = get_institutional_ownership(ticker)

            if ownership.num_institutions > 0:
                inst_section = f"""
13F INSTITUTIONAL HOLDINGS (45+ Day Lag):
-----------------------------------------
What this tells you: Which major hedge funds own this stock.
IMPORTANT: Data is 45-105 days old. They may have sold since filing.
Best used for: Long-term conviction signals, not short-term timing.

Current Data:
- Signal: {ownership.signal} (strength: {ownership.signal_strength}/100)
- Notable Holders: {ownership.num_institutions}
- Total Shares: {ownership.total_shares:,}
- Buffett Owns: {'YES 🎯' if ownership.buffett_owns else 'No'}
- Buffett Added: {'YES 🔥' if ownership.buffett_added else 'No'}
- Activist Involved: {'YES 📢' if ownership.activist_involved else 'No'}
"""
                if ownership.new_buyers:
                    inst_section += f"New Buyers This Quarter: {', '.join(ownership.new_buyers)}\n"
                if ownership.added_by:
                    inst_section += f"Added Position: {', '.join(ownership.added_by)}\n"
                if ownership.reduced_by:
                    inst_section += f"Reduced Position: {', '.join(ownership.reduced_by)}\n"
                if ownership.sold_by:
                    inst_section += f"Sold Entirely: {', '.join(ownership.sold_by)}\n"

                # Interpretation
                if ownership.buffett_added:
                    inst_section += "Interpretation: BUFFETT ADDED - Warren Buffett increased position. High conviction value signal.\n"
                elif ownership.buffett_owns:
                    inst_section += "Interpretation: BUFFETT HOLDS - In Berkshire portfolio. Long-term quality indicator.\n"
                if ownership.activist_involved:
                    inst_section += "Interpretation: ACTIVIST INVOLVED - Expect potential catalyst/changes. Can be volatile.\n"
                if len(ownership.new_buyers) >= 2:
                    inst_section += "Interpretation: Multiple new institutional buyers - building interest.\n"

                sections.append(inst_section)

        except Exception as e:
            logger.debug(f"13F context error: {e}")

        # ========================================================================
        # PAIRS/CORRELATION ANALYSIS
        # ========================================================================


    if PAIRS_CORRELATION_AVAILABLE:
        try:
            # Find hedge candidates
            hedges = find_hedges(ticker, max_candidates=5)

            if hedges:
                pairs_section = f"""
        PAIRS & CORRELATION ANALYSIS:
        -----------------------------
        What this tells you: Stocks/ETFs that can hedge {ticker} or are correlated with it.
        Negative correlation = good hedge. Use for risk management.
    
        Hedge Candidates for {ticker}:
        """
                for h in hedges[:5]:
                    hedge_type = "STRONG HEDGE" if h.correlation < -0.6 else "MODERATE HEDGE" if h.correlation < -0.3 else "WEAK HEDGE"
                    pairs_section += f"- {h.ticker}: correlation={h.correlation:.2f} ({hedge_type}), hedge_ratio={h.beta:.2f}\n"

                pairs_section += """
        How to use:
        - Negative correlation means the hedge moves opposite to the stock
        - Hedge ratio tells you how many shares of hedge per share of stock
        - SH, SDS, SPXU are inverse S&P ETFs (broad market hedge)
        - TLT, GLD are bonds/gold (flight to safety hedge)
        """
                sections.append(pairs_section)

        except Exception as e:
            logger.debug(f"Pairs/correlation context error: {e}")

    # ========================================================================
    # COMBINE ALL SECTIONS
    # ========================================================================
    # ========================================================================
    # COMBINE ALL SECTIONS
    # ========================================================================
    if not sections:
        return ""

    header = """
================================================================================
INSTITUTIONAL SIGNALS (Phase 2 & 3 Advanced Analytics)
================================================================================
These are professional-grade signals used by institutional traders.
Use them to inform your analysis and recommendations.
"""

    return header + "\n".join(sections)


def get_trading_implications(ticker: str, current_price: float = None, sector: str = None) -> str:
    """
    Get a summary of trading implications from all institutional signals.

    Returns a condensed actionable summary.
    """
    implications = []
    bullish_count = 0
    bearish_count = 0

    if current_price is None:
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            current_price = stock.history(period='1d')['Close'].iloc[-1]
        except:
            current_price = 0

    # GEX
    if GEX_AVAILABLE and current_price > 0:
        try:
            gex = analyze_gex(ticker, current_price)
            if gex.signal == "BULLISH":
                bullish_count += 1
                implications.append("GEX: Bullish options positioning")
            elif gex.signal == "BEARISH":
                bearish_count += 1
                implications.append("GEX: Bearish options positioning")
            elif gex.signal == "PINNED":
                implications.append(f"GEX: Price pinned near ${gex.max_gamma_strike:.0f}")
        except:
            pass

    # Dark Pool
    if DARK_POOL_AVAILABLE:
        try:
            dp = analyze_dark_pool(ticker)
            if dp.institutional_bias == "BUYING":
                bullish_count += 1
                implications.append("Dark Pool: Institutional accumulation")
            elif dp.institutional_bias == "SELLING":
                bearish_count += 1
                implications.append("Dark Pool: Institutional distribution")
        except:
            pass

    # Cross-Asset
    if CROSS_ASSET_AVAILABLE:
        try:
            global _cross_asset_cache
            if _cross_asset_cache is None:
                _cross_asset_cache = get_cross_asset_signals()
            xa = _cross_asset_cache

            if xa.risk_appetite == "RISK_ON":
                bullish_count += 1
                implications.append("Cross-Asset: Risk-on environment")
            elif xa.risk_appetite == "RISK_OFF":
                bearish_count += 1
                implications.append("Cross-Asset: Risk-off environment")

            if sector and xa.favored_sectors and sector in xa.favored_sectors:
                bullish_count += 1
                implications.append(f"Cross-Asset: {sector} sector favored")
        except:
            pass

    # Earnings Whisper
    if EARNINGS_WHISPER_AVAILABLE:
        try:
            whisper = get_earnings_whisper(ticker)
            if whisper.days_to_earnings <= 30:
                if whisper.beat_probability >= 65:
                    bullish_count += 1
                    implications.append(f"Whisper: {whisper.beat_probability:.0f}% beat probability")
                elif whisper.beat_probability <= 35:
                    bearish_count += 1
                    implications.append(f"Whisper: {100-whisper.beat_probability:.0f}% miss probability")
        except:
            pass

    # Insider Transactions
    if INSIDER_TRACKER_AVAILABLE:
        try:
            insider = get_insider_signal(ticker)
            if insider.ceo_bought or insider.cfo_bought:
                bullish_count += 2  # C-suite buying is very significant
                implications.append("Insider: C-suite buying")
            elif insider.cluster_buying:
                bullish_count += 1
                implications.append("Insider: Cluster buying (3+ insiders)")
            elif insider.cluster_selling:
                bearish_count += 1
                implications.append("Insider: Cluster selling")
            elif insider.signal_strength >= 65:
                bullish_count += 1
                implications.append(f"Insider: Net buying (${insider.net_value:,.0f})")
            elif insider.signal_strength <= 35:
                bearish_count += 1
                implications.append(f"Insider: Net selling (${abs(insider.net_value):,.0f})")
        except:
            pass

    # 13F Institutional
    if INSTITUTIONAL_13F_AVAILABLE:
        try:
            ownership = get_institutional_ownership(ticker)
            if ownership.buffett_added:
                bullish_count += 2  # Buffett adding is very significant
                implications.append("13F: Buffett added position")
            elif ownership.buffett_owns:
                bullish_count += 1
                implications.append("13F: Buffett holds")
            elif len(ownership.new_buyers) >= 2:
                bullish_count += 1
                implications.append(f"13F: {len(ownership.new_buyers)} new institutional buyers")
            elif len(ownership.sold_by) >= 2:
                bearish_count += 1
                implications.append(f"13F: {len(ownership.sold_by)} institutions sold")
        except:
            pass

    # Build summary
    if not implications:
        return ""

    net_signal = bullish_count - bearish_count
    if net_signal >= 2:
        bias = "BULLISH"
    elif net_signal <= -2:
        bias = "BEARISH"
    else:
        bias = "MIXED"

    summary = f"""
INSTITUTIONAL SIGNAL SUMMARY:
- Net Bias: {bias} ({bullish_count} bullish, {bearish_count} bearish signals)
- Key Points: {'; '.join(implications[:4])}
"""
    return summary