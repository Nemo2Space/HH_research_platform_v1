"""
Signal Engine

The central engine that combines ALL analyzers to produce UnifiedSignal.
Runs in background or on-demand to generate signals for all tickers.

FIXES APPLIED:
1. _scores_from_db flag now properly set after loading from DB
2. SPY/QQQ/VIX fetched via yfinance (not broken .get() on dataclass)

VERSION: 2024-12-23-FIXED
Author: Alpha Research Platform
"""

import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# ============================================================================
# SELF-VERIFICATION: Log which file is loaded (remove after confirming)
# ============================================================================
import logging
_self_check_logger = logging.getLogger("signal_engine.verify")
_self_check_logger.info(f"✅ signal_engine.py loaded from: {__file__}")
_self_check_logger.info(f"   VERSION: 2024-12-23-FIXED (_scores_from_db flag, SPY/QQQ/VIX fix)")
# ============================================================================

from src.utils.logging import get_logger
from src.db.connection import get_engine, get_connection

from src.core.unified_signal import (
    UnifiedSignal,
    MarketOverview,
    SignalSnapshot,
    SignalStrength,
    RiskLevel,
    AssetType,
    BOND_ETFS,
)

logger = get_logger(__name__)

# ============================================================
# IMPORT ALL ANALYZERS (graceful fallbacks)
# ============================================================

# Technical Analysis
try:
    from src.analytics.technical_analysis import TechnicalAnalyzer
    TECHNICAL_AVAILABLE = True
except ImportError:
    TECHNICAL_AVAILABLE = False
    logger.warning("Technical analysis not available")

# Options Flow
try:
    from src.analytics.options_flow import OptionsFlowAnalyzer
    OPTIONS_AVAILABLE = True
except ImportError:
    OPTIONS_AVAILABLE = False
    logger.warning("Options flow not available")

# Short Squeeze
try:
    from src.analytics.short_squeeze import ShortSqueezeDetector
    SQUEEZE_AVAILABLE = True
except ImportError:
    SQUEEZE_AVAILABLE = False
    logger.warning("Short squeeze detector not available")

# Market Context
try:
    from src.analytics.market_context import MarketContextAnalyzer, get_market_context
    MARKET_CONTEXT_AVAILABLE = True
except ImportError:
    MARKET_CONTEXT_AVAILABLE = False
    logger.warning("Market context not available")

# ============================================================
# INSTITUTIONAL SIGNAL IMPORTS (Phase 2-4)
# ============================================================

# GEX/Gamma Analysis
try:
    from src.analytics.gex_analysis import analyze_gex
    GEX_AVAILABLE = True
except ImportError:
    GEX_AVAILABLE = False

# Dark Pool Flow
try:
    from src.analytics.dark_pool import analyze_dark_pool
    DARK_POOL_AVAILABLE = True
except ImportError:
    DARK_POOL_AVAILABLE = False

# Cross-Asset Signals
try:
    from src.analytics.cross_asset import get_cross_asset_signals
    CROSS_ASSET_AVAILABLE = True
except ImportError:
    CROSS_ASSET_AVAILABLE = False

# Sentiment NLP
try:
    from src.analytics.sentiment_nlp import analyze_news_sentiment
    SENTIMENT_NLP_AVAILABLE = True
except ImportError:
    SENTIMENT_NLP_AVAILABLE = False

# Earnings Whisper
try:
    from src.analytics.earnings_whisper import get_earnings_whisper
    EARNINGS_WHISPER_AVAILABLE = True
except ImportError:
    EARNINGS_WHISPER_AVAILABLE = False

# Insider Tracker (Form 4)
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
# Macro Regime
try:
    from src.analytics.macro_regime import get_current_regime, MacroRegimeAnalyzer
    MACRO_REGIME_AVAILABLE = True
except ImportError:
    MACRO_REGIME_AVAILABLE = False
    logger.warning("Macro regime not available")

# Earnings Intelligence
try:
    from src.analytics.earnings_intelligence import (
        get_earnings_info,
        calculate_ies,
        calculate_ecs,
        enrich_screener_with_earnings,
    )
    EARNINGS_AVAILABLE = True
except ImportError:
    EARNINGS_AVAILABLE = False
    logger.warning("Earnings intelligence not available")

# Bond Signals
try:
    from src.analytics.bond_signals import (
        BondSignalGenerator,
        get_bond_signal,
        get_treasury_yields,
    )
    BONDS_AVAILABLE = True
except ImportError:
    BONDS_AVAILABLE = False
    logger.warning("Bond signals not available")

# Economic Calendar
try:
    from src.analytics.economic_calendar import EconomicCalendarFetcher
    ECONOMIC_CALENDAR_AVAILABLE = True
except ImportError:
    ECONOMIC_CALENDAR_AVAILABLE = False
    logger.warning("Economic calendar not available")

# Sentiment
try:
    from src.analytics.sentiment import SentimentAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    logger.warning("Sentiment analyzer not available")

# Phase 0: Unified Scorer (single source of truth)
try:
    from src.core.unified_scorer import UnifiedScorer, TickerFeatures, get_unified_scorer
    UNIFIED_SCORER_AVAILABLE = True
except ImportError:
    UNIFIED_SCORER_AVAILABLE = False
    logger.warning("Unified Scorer not available")

class SignalEngine:
    """
    Central engine for generating unified signals.

    Combines all available analyzers and produces a single
    UnifiedSignal per ticker.
    """

    def __init__(self):
        self._cache: Dict[str, UnifiedSignal] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=15)

        # Initialize analyzers
        self._init_analyzers()

        # Phase 0: Unified Scorer for consistent scoring
        if UNIFIED_SCORER_AVAILABLE:
            self.unified_scorer = get_unified_scorer()
        else:
            self.unified_scorer = None

        logger.info("SignalEngine initialized")
        logger.info(f"Available components: {self._get_available_components()}")

    def _init_analyzers(self):
        """Initialize all available analyzers."""
        self.technical_analyzer = TechnicalAnalyzer() if TECHNICAL_AVAILABLE else None
        self.options_analyzer = OptionsFlowAnalyzer() if OPTIONS_AVAILABLE else None
        self.squeeze_detector = ShortSqueezeDetector() if SQUEEZE_AVAILABLE else None
        self.bond_generator = BondSignalGenerator() if BONDS_AVAILABLE else None
        self.economic_calendar = EconomicCalendarFetcher() if ECONOMIC_CALENDAR_AVAILABLE else None

    def _get_available_components(self) -> List[str]:
        """Get list of available analysis components."""
        components = []
        if TECHNICAL_AVAILABLE:
            components.append("technical")
        if OPTIONS_AVAILABLE:
            components.append("options")
        if SQUEEZE_AVAILABLE:
            components.append("squeeze")
        if MARKET_CONTEXT_AVAILABLE:
            components.append("market_context")
        if MACRO_REGIME_AVAILABLE:
            components.append("macro_regime")
        if EARNINGS_AVAILABLE:
            components.append("earnings")
        if BONDS_AVAILABLE:
            components.append("bonds")
        if ECONOMIC_CALENDAR_AVAILABLE:
            components.append("economic_calendar")
        if SENTIMENT_AVAILABLE:
            components.append("sentiment")
        if UNIFIED_SCORER_AVAILABLE:
            components.append("unified_scorer")
        return components
    # ============================================================
    # MAIN SIGNAL GENERATION
    # ============================================================

    def generate_signal(self, ticker: str, force_refresh: bool = False) -> UnifiedSignal:
        """
        Generate unified signal for a single ticker.

        Args:
            ticker: Stock/ETF symbol
            force_refresh: Bypass cache

        Returns:
            UnifiedSignal with all components
        """
        ticker = ticker.upper()

        # Check cache
        if not force_refresh and ticker in self._cache:
            cached = self._cache[ticker]
            age = (datetime.now() - cached.updated_at).total_seconds()
            if age < self._cache_duration.total_seconds():
                return cached

        logger.info(f"Generating signal for {ticker}...")

        # Detect asset type
        asset_type = UnifiedSignal.detect_asset_type(ticker)

        # Route to appropriate generator
        if asset_type == AssetType.BOND_ETF:
            signal = self._generate_bond_signal(ticker)
        else:
            signal = self._generate_stock_signal(ticker)

        # Cache result
        self._cache[ticker] = signal

        return signal

    def generate_signals_batch(self, tickers: List[str], max_workers: int = 5) -> Dict[str, UnifiedSignal]:
        """
        Generate signals for multiple tickers in parallel.

        Args:
            tickers: List of symbols
            max_workers: Parallel threads

        Returns:
            Dict of ticker -> UnifiedSignal
        """
        signals = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.generate_signal, ticker): ticker
                for ticker in tickers
            }

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    signals[ticker] = future.result()
                except Exception as e:
                    logger.error(f"Error generating signal for {ticker}: {e}")
                    # Create minimal signal on error
                    signals[ticker] = UnifiedSignal(
                        ticker=ticker,
                        data_quality="ERROR",
                        signal_reason=f"Error: {str(e)}"
                    )

        return signals

    # ============================================================
    # STOCK SIGNAL GENERATION
    # ============================================================

    def _generate_stock_signal(self, ticker: str) -> UnifiedSignal:
        """Generate signal for a stock ticker."""
        signal = UnifiedSignal(
            ticker=ticker,
            asset_type=AssetType.STOCK,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        components_used = []

        # 1. Get ALL scores from screener_scores table (same as Universe tab)
        has_db_scores = self._load_scores_from_db(signal)

        if has_db_scores:
            components_used.extend(["technical", "fundamental", "sentiment", "options"])

        # 2. Get basic info (company name, price, sector)
        self._load_basic_info(signal)

        # 3. Only run live analysis if DB doesn't have data
        if not has_db_scores:
            # Technical Analysis
            if TECHNICAL_AVAILABLE:
                try:
                    self._add_technical_analysis(signal)
                    if signal.technical_score != 50:
                        components_used.append("technical")
                except Exception as e:
                    logger.debug(f"{ticker}: Technical analysis error: {e}")

            # Fundamental Analysis (from DB or yfinance)
            try:
                self._add_fundamental_analysis(signal)
                if signal.fundamental_score != 50:
                    components_used.append("fundamental")
            except Exception as e:
                logger.debug(f"{ticker}: Fundamental analysis error: {e}")

        # 4. Earnings Intelligence (always check - has dates)
        if EARNINGS_AVAILABLE:
            try:
                self._add_earnings_analysis(signal)
                if signal.earnings_date or signal.earnings_score != 50:
                    components_used.append("earnings")
            except Exception as e:
                logger.debug(f"{ticker}: Earnings analysis error: {e}")

            # 5. Portfolio context
            try:
                self._add_portfolio_context(signal)
            except Exception as e:
                logger.debug(f"{ticker}: Portfolio context error: {e}")

        # ============================================================
        # 5.5 INSTITUTIONAL SIGNALS (Phase 2-4)
        # ============================================================

        # GEX/Gamma Analysis
        if GEX_AVAILABLE:
            try:
                gex = analyze_gex(ticker, signal.current_price)
                signal.gex_score = gex.signal_strength
                signal.gex_signal = gex.signal
                signal.gex_regime = gex.regime
                signal.gex_reason = f"{gex.regime} regime, {gex.signal} signal"
                components_used.append("gex")
            except Exception as e:
                logger.debug(f"{ticker}: GEX analysis error: {e}")

        # Dark Pool Flow
        if DARK_POOL_AVAILABLE:
            try:
                dp = analyze_dark_pool(ticker)
                signal.dark_pool_score = dp.score
                signal.dark_pool_signal = dp.sentiment
                signal.institutional_bias = dp.institutional_bias
                signal.dark_pool_reason = f"{dp.sentiment} ({dp.institutional_bias})"
                components_used.append("dark_pool")
            except Exception as e:
                logger.debug(f"{ticker}: Dark pool analysis error: {e}")

        # Cross-Asset Context (market-wide, not ticker-specific)
        if CROSS_ASSET_AVAILABLE:
            try:
                ca = get_cross_asset_signals()
                signal.cross_asset_score = ca.signal_strength
                signal.cross_asset_signal = ca.risk_signal
                signal.cycle_phase = ca.cycle_phase
                signal.cross_asset_reason = f"{ca.risk_signal}, {ca.cycle_phase}"
                components_used.append("cross_asset")
            except Exception as e:
                logger.debug(f"{ticker}: Cross-asset analysis error: {e}")

        # Sentiment NLP (AI-powered news analysis)
        if SENTIMENT_NLP_AVAILABLE:
            try:
                # Get recent news headlines for this ticker
                headlines = self._get_recent_headlines(ticker)
                if headlines:
                    nlp_result = analyze_news_sentiment(headlines, ticker)
                    signal.sentiment_nlp_score = nlp_result.score
                    signal.sentiment_nlp_signal = nlp_result.sentiment.value
                    signal.sentiment_nlp_reason = nlp_result.summary[:100] if nlp_result.summary else ""
                    components_used.append("sentiment_nlp")
            except Exception as e:
                logger.debug(f"{ticker}: Sentiment NLP error: {e}")

        # Earnings Whisper
        if EARNINGS_WHISPER_AVAILABLE and signal.days_to_earnings and signal.days_to_earnings <= 30:
            try:
                whisper = get_earnings_whisper(ticker)
                signal.whisper_score = whisper.signal_strength
                signal.whisper_signal = whisper.signal.value
                signal.whisper_reason = f"{whisper.prediction.value}: {whisper.reasoning[:50]}" if whisper.reasoning else whisper.prediction.value
                components_used.append("whisper")
            except Exception as e:
                logger.debug(f"{ticker}: Earnings whisper error: {e}")

        # Insider Tracker (Form 4)
        if INSIDER_TRACKER_AVAILABLE:
            try:
                insider = get_insider_signal(ticker)
                signal.insider_score = insider.signal_strength
                signal.insider_signal = insider.signal
                signal.insider_ceo_bought = insider.ceo_bought
                signal.insider_cfo_bought = insider.cfo_bought
                signal.insider_cluster_buying = insider.cluster_buying
                signal.insider_cluster_selling = insider.cluster_selling
                signal.insider_net_value = insider.net_value

                reasons = []
                if insider.ceo_bought:
                    reasons.append("CEO bought")
                if insider.cfo_bought:
                    reasons.append("CFO bought")
                if insider.cluster_buying:
                    reasons.append("cluster buying")
                if insider.cluster_selling:
                    reasons.append("cluster selling")
                signal.insider_reason = ", ".join(reasons) if reasons else f"Net: ${insider.net_value:,.0f}"
                components_used.append("insider")
            except Exception as e:
                logger.debug(f"{ticker}: Insider tracker error: {e}")

        # 13F Institutional Holdings
        if INSTITUTIONAL_13F_AVAILABLE:
            try:
                inst = get_institutional_ownership(ticker)
                signal.inst_13f_score = inst.signal_strength
                signal.inst_13f_signal = inst.signal
                signal.inst_buffett_owns = inst.buffett_owns
                signal.inst_buffett_added = inst.buffett_added
                signal.inst_activist_involved = inst.activist_involved
                signal.inst_notable_holders = inst.num_institutions

                reasons = []
                if inst.buffett_added:
                    reasons.append("Buffett added")
                elif inst.buffett_owns:
                    reasons.append("Buffett owns")
                if inst.activist_involved:
                    reasons.append("activist involved")
                if inst.new_buyers:
                    reasons.append(f"{len(inst.new_buyers)} new buyers")
                signal.inst_13f_reason = ", ".join(
                    reasons) if reasons else f"{inst.num_institutions} notable holders"
                components_used.append("inst_13f")
            except Exception as e:
                logger.debug(f"{ticker}: 13F tracker error: {e}")

        # ============================================================

        # 6. Calculate final scores
        self._calculate_final_scores(signal)

        # 7. Determine risk level
        self._calculate_risk(signal)

        # 8. Generate signal reason
        self._generate_signal_reason(signal)

        # 9. Add flags
        self._add_flags(signal)

        # Set metadata
        signal.components_available = list(set(components_used))
        signal.data_quality = "HIGH" if len(components_used) >= 3 else "MEDIUM" if len(components_used) >= 1 else "LOW"

        return signal

    def _load_scores_from_db(self, signal: UnifiedSignal) -> bool:
        """Load all scores from screener_scores table. Returns True if found."""
        try:
            engine = get_engine()
            import pandas as pd

            # Query matches actual table schema from Universe tab
            query = """
                SELECT DISTINCT ON (ticker)
                    ticker, 
                    sentiment_score,
                    fundamental_score,
                    technical_score,
                    growth_score,
                    dividend_score,
                    total_score,
                    article_count,
                    options_flow_score,
                    short_squeeze_score,
                    options_sentiment,
                    squeeze_risk,
                    date
                FROM screener_scores
                WHERE ticker = %s
                ORDER BY ticker, date DESC
            """
            df = pd.read_sql(query, engine, params=(signal.ticker,))

            if df.empty:
                logger.debug(f"{signal.ticker}: No data in screener_scores table")
                return False

            row = df.iloc[0]

            # Load all scores
            signal.sentiment_score = int(row.get('sentiment_score') or 50)
            signal.fundamental_score = int(row.get('fundamental_score') or 50)
            signal.technical_score = int(row.get('technical_score') or 50)
            signal.options_score = int(row.get('options_flow_score') or 50)

            logger.info(f"{signal.ticker}: DB scores - Sent:{signal.sentiment_score} Fund:{signal.fundamental_score} Tech:{signal.technical_score} Opts:{signal.options_score}")

            # Set signals based on scores
            signal.sentiment_signal = "BUY" if signal.sentiment_score >= 60 else "SELL" if signal.sentiment_score <= 40 else "HOLD"
            signal.fundamental_signal = "BUY" if signal.fundamental_score >= 60 else "SELL" if signal.fundamental_score <= 40 else "HOLD"
            signal.technical_signal = "BUY" if signal.technical_score >= 60 else "SELL" if signal.technical_score <= 40 else "HOLD"
            signal.options_signal = "BUY" if signal.options_score >= 60 else "SELL" if signal.options_score <= 40 else "HOLD"

            # Set reasons with actual values
            articles = int(row.get('article_count') or 0)
            signal.sentiment_reason = f"Score {signal.sentiment_score}" + (f" ({articles} articles)" if articles > 0 else "")
            signal.fundamental_reason = f"Score {signal.fundamental_score}"
            signal.technical_reason = f"Score {signal.technical_score}"
            signal.options_reason = str(row.get('options_sentiment') or f"Score {signal.options_score}")

            # Squeeze flag
            squeeze = int(row.get('short_squeeze_score') or 0)
            if squeeze >= 70:
                signal.flags.append(f"🚀 Squeeze {squeeze}")

            # CRITICAL FIX: Set flag so analyzers don't overwrite DB scores
            signal._scores_from_db = True

            return True

        except Exception as e:
            logger.warning(f"{signal.ticker}: Error loading from DB: {e}")
            return False

    def _get_recent_headlines(self, ticker: str, days_back: int = 3) -> List[str]:
        """Get recent news headlines for sentiment NLP analysis."""
        try:
            engine = get_engine()
            import pandas as pd

            query = """
                    SELECT headline
                    FROM news_articles
                    WHERE ticker = %s
                      AND date >= CURRENT_DATE - INTERVAL '%s days'
                    ORDER BY date DESC
                        LIMIT 10 \
                    """
            df = pd.read_sql(query, engine, params=(ticker, days_back))

            if not df.empty:
                return df['headline'].tolist()
            return []
        except Exception as e:
            logger.debug(f"Could not get headlines for {ticker}: {e}")
            return []

    # ============================================================
    # BOND SIGNAL GENERATION
    # ============================================================

    def _generate_bond_signal(self, ticker: str) -> UnifiedSignal:
        """Generate signal for a bond ETF."""
        signal = UnifiedSignal(
            ticker=ticker,
            asset_type=AssetType.BOND_ETF,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Set basic info
        bond_names = {
            'TLT': '20+ Year Treasury Bond ETF',
            'ZROZ': 'Zero Coupon 25+ Year Treasury',
            'EDV': 'Extended Duration Treasury',
            'TMF': '3x Long 20+ Year Treasury',
            'TBT': '2x Short 20+ Year Treasury',
            'SHY': '1-3 Year Treasury Bond ETF',
            'IEF': '7-10 Year Treasury Bond ETF',
            'BND': 'Total Bond Market ETF',
            'AGG': 'Core US Aggregate Bond ETF',
            'LQD': 'Investment Grade Corporate Bond',
            'HYG': 'High Yield Corporate Bond',
            'JNK': 'High Yield Bond ETF',
        }
        signal.company_name = bond_names.get(ticker.upper(), f"{ticker} - Bond ETF")

        # Try to get price from yfinance
        try:
            import yfinance as yf
            etf = yf.Ticker(ticker)
            info = etf.info
            signal.current_price = info.get('currentPrice', info.get('regularMarketPrice', 0)) or 0
            signal.week_52_high = info.get('fiftyTwoWeekHigh', 0) or 0
            signal.week_52_low = info.get('fiftyTwoWeekLow', 0) or 0
        except:
            pass

        if not BONDS_AVAILABLE:
            signal.data_quality = "LOW"
            signal.signal_reason = "Bond analysis module not available"
            signal.bond_reason = "Run screener to enable bond analysis"
            return signal

        try:
            # Get bond signal from bond_signals module
            bond_result = get_bond_signal(ticker)

            if bond_result is None:
                signal.data_quality = "LOW"
                signal.signal_reason = "Could not generate bond signal"
                return signal

            # Map to unified signal
            if bond_result.current_price:
                signal.current_price = bond_result.current_price
            signal.target_price = bond_result.target_price
            signal.stop_loss = bond_result.stop_loss
            signal.upside_pct = bond_result.upside_pct
            signal.downside_pct = bond_result.downside_pct
            signal.risk_reward = bond_result.risk_reward

            # Bond-specific scores
            signal.bond_score = bond_result.score
            signal.bond_signal = bond_result.signal.value if bond_result.signal else "HOLD"
            signal.bond_reason = bond_result.recommendation or "No recommendation"

            # Map component scores
            signal.technical_score = bond_result.technical_score if hasattr(bond_result, 'technical_score') else 50
            signal.technical_signal = "BUY" if signal.technical_score > 60 else "SELL" if signal.technical_score < 40 else "HOLD"

            # Map bond score to today/longterm
            signal.today_score = bond_result.score
            signal.longterm_score = bond_result.score

            # Map signal strength
            signal_map = {
                "STRONG_BUY": SignalStrength.STRONG_BUY,
                "BUY": SignalStrength.BUY,
                "HOLD": SignalStrength.HOLD,
                "SELL": SignalStrength.SELL,
                "STRONG_SELL": SignalStrength.STRONG_SELL,
            }
            signal_value = bond_result.signal.value if bond_result.signal else "HOLD"
            signal.today_signal = signal_map.get(signal_value, SignalStrength.HOLD)
            signal.longterm_signal = signal.today_signal

            # Set signal reason
            signal.signal_reason = bond_result.recommendation or f"Bond score: {bond_result.score}"

            # Add economic calendar context
            if hasattr(bond_result, 'economic_impact') and bond_result.economic_impact:
                impact = bond_result.economic_impact
                if hasattr(impact, 'fomc_this_week') and impact.fomc_this_week:
                    signal.next_catalyst = f"FOMC {impact.days_to_fomc}d"
                    signal.days_to_catalyst = impact.days_to_fomc
                if hasattr(impact, 'cpi_today') and impact.cpi_today:
                    signal.flags.append(f"📊 CPI {impact.cpi_result}")

            # Flags from factors
            if hasattr(bond_result, 'bull_factors') and bond_result.bull_factors:
                for factor in bond_result.bull_factors[:2]:
                    signal.flags.append(f"✅ {factor}")
            if hasattr(bond_result, 'bear_factors') and bond_result.bear_factors:
                for factor in bond_result.bear_factors[:1]:
                    signal.flags.append(f"⚠️ {factor}")

            signal.data_quality = "HIGH"
            signal.components_available = ["bonds", "technical", "economic"]

        except Exception as e:
            logger.error(f"Error generating bond signal for {ticker}: {e}")
            signal.data_quality = "ERROR"
            signal.signal_reason = f"Error loading bond data"
            signal.bond_reason = str(e)[:50]

        return signal

    # ============================================================
    # COMPONENT LOADERS
    # ============================================================

    def _load_basic_info(self, signal: UnifiedSignal):
        """Load basic ticker info (company name, price, sector) - NOT scores."""
        try:
            engine = get_engine()
            import pandas as pd

            # Get sector from fundamentals
            fund_query = """
                SELECT sector FROM fundamentals 
                WHERE ticker = %s 
                ORDER BY date DESC LIMIT 1
            """
            fund_df = pd.read_sql(fund_query, engine, params=(signal.ticker,))
            if not fund_df.empty:
                signal.sector = fund_df.iloc[0].get('sector', 'Unknown') or 'Unknown'

            # FIX: Always fetch LIVE price and 52W data from yfinance
            # The prices table may be stale, causing incorrect 52W calculations
            try:
                import yfinance as yf
                stock = yf.Ticker(signal.ticker)
                info = stock.info

                signal.company_name = info.get('shortName', info.get('longName', signal.ticker))

                if not signal.sector or signal.sector == 'Unknown':
                    signal.sector = info.get('sector', 'Unknown')

                # FIX: Always use live price from yfinance (not stale DB)
                live_price = info.get('currentPrice') or info.get('regularMarketPrice', 0) or 0
                if live_price > 0:
                    signal.current_price = float(live_price)

                # 52W high/low from yfinance (always live)
                signal.week_52_high = info.get('fiftyTwoWeekHigh', 0) or 0
                signal.week_52_low = info.get('fiftyTwoWeekLow', 0) or 0

            except Exception as e:
                logger.debug(f"{signal.ticker}: yfinance error: {e}")
                if not signal.company_name:
                    signal.company_name = signal.ticker
                # Fallback to DB price only if yfinance fails
                if signal.current_price == 0:
                    price_query = """
                        SELECT close FROM prices 
                        WHERE ticker = %s 
                        ORDER BY date DESC LIMIT 1
                    """
                    price_df = pd.read_sql(price_query, engine, params=(signal.ticker,))
                    if not price_df.empty:
                        signal.current_price = float(price_df.iloc[0]['close'] or 0)

            # Calculate percent from high/low
            if signal.week_52_high > 0 and signal.current_price > 0:
                signal.pct_from_high = ((signal.current_price - signal.week_52_high) / signal.week_52_high) * 100
            if signal.week_52_low > 0 and signal.current_price > 0:
                signal.pct_from_low = ((signal.current_price - signal.week_52_low) / signal.week_52_low) * 100

        except Exception as e:
            logger.debug(f"{signal.ticker}: Error loading basic info: {e}")
            if not signal.company_name:
                signal.company_name = signal.ticker

    def _add_technical_analysis(self, signal: UnifiedSignal):
        """Add technical analysis component."""
        # Skip if already loaded from screener_scores
        if hasattr(signal, '_scores_from_db') and signal._scores_from_db and signal.technical_score != 50:
            return

        if not self.technical_analyzer:
            return

        try:
            result = self.technical_analyzer.analyze(signal.ticker)

            # Map technical score (assume result has score 0-100)
            if hasattr(result, 'score'):
                signal.technical_score = int(result.score)
            elif hasattr(result, 'technical_score'):
                signal.technical_score = int(result.technical_score)
            else:
                # Calculate from individual indicators
                signal.technical_score = 50

            # Determine signal
            if signal.technical_score >= 70:
                signal.technical_signal = "BUY"
                signal.technical_reason = "Strong technical setup"
            elif signal.technical_score >= 55:
                signal.technical_signal = "BUY"
                signal.technical_reason = "Positive technicals"
            elif signal.technical_score <= 30:
                signal.technical_signal = "SELL"
                signal.technical_reason = "Weak technical setup"
            elif signal.technical_score <= 45:
                signal.technical_signal = "SELL"
                signal.technical_reason = "Negative technicals"
            else:
                signal.technical_signal = "HOLD"
                signal.technical_reason = "Neutral technicals"

        except Exception as e:
            logger.debug(f"{signal.ticker}: Technical analysis error: {e}")

    def _add_fundamental_analysis(self, signal: UnifiedSignal):
        """Add fundamental analysis from database."""
        # Skip if already loaded from screener_scores
        if hasattr(signal, '_scores_from_db') and signal._scores_from_db and signal.fundamental_score != 50:
            return

        try:
            engine = get_engine()
            query = """
                SELECT 
                    pe_ratio, forward_pe, peg_ratio,
                    revenue_growth, earnings_growth,
                    profit_margin, roe, debt_to_equity
                FROM fundamentals
                WHERE ticker = %s
                ORDER BY date DESC
                LIMIT 1
            """
            import pandas as pd
            df = pd.read_sql(query, engine, params=(signal.ticker,))

            if df.empty:
                return

            row = df.iloc[0]

            # Simple scoring logic
            score = 50
            reasons = []

            # PE ratio
            pe = row.get('pe_ratio')
            if pe and pe > 0:
                if pe < 15:
                    score += 10
                    reasons.append("Low P/E")
                elif pe > 40:
                    score -= 10
                    reasons.append("High P/E")

            # Revenue growth
            rev_growth = row.get('revenue_growth')
            if rev_growth:
                if rev_growth > 0.20:
                    score += 15
                    reasons.append("Strong growth")
                elif rev_growth > 0.10:
                    score += 5
                elif rev_growth < 0:
                    score -= 10
                    reasons.append("Declining revenue")

            # Profit margin
            margin = row.get('profit_margin')
            if margin:
                if margin > 0.20:
                    score += 10
                    reasons.append("High margins")
                elif margin < 0:
                    score -= 15
                    reasons.append("Unprofitable")

            # Debt
            debt = row.get('debt_to_equity')
            if debt and debt > 2:
                score -= 10
                reasons.append("High debt")

            signal.fundamental_score = max(0, min(100, score))

            if signal.fundamental_score >= 60:
                signal.fundamental_signal = "BUY"
            elif signal.fundamental_score <= 40:
                signal.fundamental_signal = "SELL"
            else:
                signal.fundamental_signal = "HOLD"

            signal.fundamental_reason = ", ".join(reasons[:2]) if reasons else "Average fundamentals"

        except Exception as e:
            logger.debug(f"{signal.ticker}: Fundamental analysis error: {e}")

    def _add_sentiment_analysis(self, signal: UnifiedSignal):
        """Add sentiment analysis from database."""
        # Skip if already loaded from screener_scores
        if hasattr(signal, '_scores_from_db') and signal._scores_from_db and signal.sentiment_score != 50:
            return

        try:
            engine = get_engine()
            query = """
                SELECT 
                    sentiment_score, article_count
                FROM screener_scores
                WHERE ticker = %s
                ORDER BY score_date DESC
                LIMIT 1
            """
            import pandas as pd
            df = pd.read_sql(query, engine, params=(signal.ticker,))

            if df.empty:
                return

            row = df.iloc[0]

            # Sentiment score from screener (already 0-100)
            sent_score = row.get('sentiment_score', 50)
            signal.sentiment_score = int(sent_score) if sent_score else 50

            article_count = row.get('article_count', 0)

            if signal.sentiment_score >= 65:
                signal.sentiment_signal = "BUY"
                signal.sentiment_reason = f"Positive sentiment ({article_count} articles)"
            elif signal.sentiment_score <= 35:
                signal.sentiment_signal = "SELL"
                signal.sentiment_reason = f"Negative sentiment ({article_count} articles)"
            else:
                signal.sentiment_signal = "HOLD"
                signal.sentiment_reason = f"Mixed sentiment ({article_count} articles)"

        except Exception as e:
            logger.debug(f"{signal.ticker}: Sentiment analysis error: {e}")

    def _add_options_analysis(self, signal: UnifiedSignal):
        """Add options flow analysis."""
        # Skip if already loaded from screener_scores
        if hasattr(signal, '_scores_from_db') and signal._scores_from_db and signal.options_score != 50:
            return

        if not self.options_analyzer:
            return

        try:
            result = self.options_analyzer.analyze(signal.ticker)

            if hasattr(result, 'sentiment_score'):
                signal.options_score = int(result.sentiment_score)
            elif hasattr(result, 'score'):
                signal.options_score = int(result.score)
            else:
                signal.options_score = 50

            if signal.options_score >= 70:
                signal.options_signal = "BUY"
                signal.options_reason = "Bullish options flow"
            elif signal.options_score <= 30:
                signal.options_signal = "SELL"
                signal.options_reason = "Bearish options flow"
            else:
                signal.options_signal = "HOLD"
                signal.options_reason = "Neutral options flow"

        except Exception as e:
            logger.debug(f"{signal.ticker}: Options analysis error: {e}")

    def _add_earnings_analysis(self, signal: UnifiedSignal):
        """Add earnings intelligence."""
        try:
            # Get earnings info
            earnings_info = get_earnings_info(signal.ticker)

            if earnings_info and earnings_info.earnings_date:
                signal.earnings_date = earnings_info.earnings_date
                signal.days_to_earnings = earnings_info.days_to_earnings

                if signal.days_to_earnings and signal.days_to_earnings > 0:
                    signal.next_catalyst = f"ER {signal.days_to_earnings}d"
                    signal.next_catalyst_date = signal.earnings_date
                    signal.days_to_catalyst = signal.days_to_earnings

            # Get IES (pre-earnings)
            if signal.days_to_earnings and 0 < signal.days_to_earnings <= 14:
                try:
                    ies_result = calculate_ies(signal.ticker)
                    if ies_result and ies_result.ies:
                        signal.ies_score = int(ies_result.ies)
                        signal.earnings_score = 100 - signal.ies_score  # Lower expectations = bullish

                        if signal.ies_score >= 70:
                            signal.earnings_signal = "SELL"
                            signal.earnings_reason = "High expectations priced in"
                        elif signal.ies_score <= 30:
                            signal.earnings_signal = "BUY"
                            signal.earnings_reason = "Low expectations, easy to beat"
                        else:
                            signal.earnings_signal = "HOLD"
                            signal.earnings_reason = "Normal expectations"
                except:
                    pass

            # Get ECS (post-earnings)
            if signal.days_to_earnings and signal.days_to_earnings < 0 and signal.days_to_earnings >= -14:
                try:
                    ecs_result = calculate_ecs(signal.ticker)
                    if ecs_result:
                        signal.ecs_category = ecs_result.ecs_category.value if ecs_result.ecs_category else ""
                        signal.last_earnings_result = signal.ecs_category

                        if signal.ecs_category in ["STRONG_BEAT", "BEAT"]:
                            signal.earnings_score = 75
                            signal.earnings_signal = "BUY"
                            signal.earnings_reason = f"Recent {signal.ecs_category}"
                            signal.flags.append(f"📊 ER {signal.ecs_category}")
                        elif signal.ecs_category in ["STRONG_MISS", "MISS"]:
                            signal.earnings_score = 25
                            signal.earnings_signal = "SELL"
                            signal.earnings_reason = f"Recent {signal.ecs_category}"
                            signal.flags.append(f"📉 ER {signal.ecs_category}")
                except:
                    pass

        except Exception as e:
            logger.debug(f"{signal.ticker}: Earnings analysis error: {e}")

    def _add_portfolio_context(self, signal: UnifiedSignal):
        """Add portfolio context if ticker is held."""
        try:
            engine = get_engine()

            # Check if in portfolio
            query = """
                SELECT 
                    ticker, weight, target_weight,
                    unrealized_pnl, unrealized_pnl_pct,
                    days_held
                FROM portfolio_positions
                WHERE ticker = %s
                LIMIT 1
            """
            import pandas as pd
            df = pd.read_sql(query, engine, params=(signal.ticker,))

            if not df.empty:
                row = df.iloc[0]
                signal.in_portfolio = True
                signal.portfolio_weight = row.get('weight', 0)
                signal.target_weight = row.get('target_weight', 0)
                signal.portfolio_pnl = row.get('unrealized_pnl', 0)
                signal.portfolio_pnl_pct = row.get('unrealized_pnl_pct', 0)
                signal.days_held = row.get('days_held', 0)

        except Exception as e:
            logger.debug(f"{signal.ticker}: Portfolio context error: {e}")

    # ============================================================
    # SCORE CALCULATION
    # ============================================================

    def _calculate_final_scores(self, signal: UnifiedSignal):
        """
        Calculate final today and long-term scores.
        Phase 0: Uses UnifiedScorer for consistent weights across platform.
        """

        # Phase 0: Use UnifiedScorer if available for single source of truth
        if UNIFIED_SCORER_AVAILABLE and self.unified_scorer:
            try:
                self._calculate_scores_unified(signal)
                return
            except Exception as e:
                logger.debug(f"{signal.ticker}: UnifiedScorer failed, using fallback: {e}")

        # Fallback to original scoring logic
        self._calculate_scores_legacy(signal)

    def _calculate_scores_unified(self, signal: UnifiedSignal):
        """
        Calculate scores using UnifiedScorer (Phase 0).
        Ensures backtest and live scoring use identical weights.
        """
        from datetime import datetime

        # Build TickerFeatures from signal data
        features = TickerFeatures(
            ticker=signal.ticker,
            as_of_time=datetime.now(),
            current_price=signal.current_price or 0,
            sentiment_score=signal.sentiment_score,
            fundamental_score=signal.fundamental_score,
            technical_score=signal.technical_score,
            options_flow_score=signal.options_score,
            short_squeeze_score=signal.squeeze_score if hasattr(signal, 'squeeze_score') else 0,
            # Add earnings if available
            earnings_date=signal.earnings_date if signal.earnings_date else None,
        )

        # Compute scores using unified weights
        result = self.unified_scorer.compute_scores(features)

        # Map back to signal
        signal.today_score = int(result.total_score)
        signal.longterm_score = int(result.total_score)  # Could differentiate later

        # Determine signal strengths from unified result
        signal.today_signal = self._score_to_signal(signal.today_score)
        signal.longterm_signal = self._score_to_signal(signal.longterm_score)

        # Use unified signal type if available
        if hasattr(result, 'signal_type'):
            signal_map = {
                'STRONG_BUY': SignalStrength.STRONG_BUY,
                'BUY': SignalStrength.BUY,
                'HOLD': SignalStrength.HOLD,
                'SELL': SignalStrength.SELL,
                'STRONG_SELL': SignalStrength.STRONG_SELL,
            }
            signal.today_signal = signal_map.get(result.signal_type, signal.today_signal)

        # Store confidence from unified scorer
        if hasattr(result, 'confidence'):
            signal.confidence = result.confidence

        # Committee votes (component signals)
        signal.committee_votes = {
            'technical': signal.technical_signal,
            'fundamental': signal.fundamental_signal,
            'sentiment': signal.sentiment_signal,
            'options': signal.options_signal,
            'earnings': signal.earnings_signal,
        }

        # Calculate agreement
        votes = [v for v in signal.committee_votes.values() if v]
        if votes:
            bullish = sum(1 for v in votes if v in ['BUY', 'STRONG_BUY'])
            bearish = sum(1 for v in votes if v in ['SELL', 'STRONG_SELL'])
            total = len(votes)

            if bullish == total:
                signal.committee_agreement = "UNANIMOUS_BULLISH"
            elif bearish == total:
                signal.committee_agreement = "UNANIMOUS_BEARISH"
            elif bullish > bearish:
                signal.committee_agreement = "MAJORITY_BULLISH"
            elif bearish > bullish:
                signal.committee_agreement = "MAJORITY_BEARISH"
            else:
                signal.committee_agreement = "SPLIT"

        logger.debug(f"{signal.ticker}: UnifiedScorer -> Today={signal.today_score} Signal={result.signal_type}")

    def _calculate_scores_legacy(self, signal: UnifiedSignal):
        """
        Original scoring logic (fallback).
        Kept for backwards compatibility.
        """
        # Track if we have real data (not just default 50s)
        has_real_data = any([
            signal.technical_score != 50,
            signal.sentiment_score != 50,
            signal.fundamental_score != 50,
            signal.options_score != 50,
            signal.earnings_score != 50,
        ])

        # Weights for today signal (short-term)
        today_weights = {
            'technical': 0.30,
            'sentiment': 0.25,
            'options': 0.25,
            'earnings': 0.20,
        }

        # Weights for long-term signal
        longterm_weights = {
            'fundamental': 0.40,
            'technical': 0.25,
            'sentiment': 0.15,
            'earnings': 0.20,
        }

        # Calculate today score - include all components
        today_total = 0
        today_weight_sum = 0

        # Always include if we have real data, or if score is not default
        if signal.technical_score != 50 or has_real_data:
            today_total += signal.technical_score * today_weights['technical']
            today_weight_sum += today_weights['technical']
        if signal.sentiment_score != 50 or has_real_data:
            today_total += signal.sentiment_score * today_weights['sentiment']
            today_weight_sum += today_weights['sentiment']
        if signal.options_score != 50 or has_real_data:
            today_total += signal.options_score * today_weights['options']
            today_weight_sum += today_weights['options']
        if signal.earnings_score != 50:  # Only include if we have earnings data
            today_total += signal.earnings_score * today_weights['earnings']
            today_weight_sum += today_weights['earnings']

        if today_weight_sum > 0:
            signal.today_score = int(today_total / today_weight_sum)
        else:
            signal.today_score = 50  # Default neutral

        # Calculate long-term score
        longterm_total = 0
        longterm_weight_sum = 0

        if signal.fundamental_score != 50 or has_real_data:
            longterm_total += signal.fundamental_score * longterm_weights['fundamental']
            longterm_weight_sum += longterm_weights['fundamental']
        if signal.technical_score != 50 or has_real_data:
            longterm_total += signal.technical_score * longterm_weights['technical']
            longterm_weight_sum += longterm_weights['technical']
        if signal.sentiment_score != 50 or has_real_data:
            longterm_total += signal.sentiment_score * longterm_weights['sentiment']
            longterm_weight_sum += longterm_weights['sentiment']
        if signal.earnings_score != 50:
            longterm_total += signal.earnings_score * longterm_weights['earnings']
            longterm_weight_sum += longterm_weights['earnings']

        if longterm_weight_sum > 0:
            signal.longterm_score = int(longterm_total / longterm_weight_sum)
        else:
            signal.longterm_score = 50

        logger.debug(f"{signal.ticker}: Legacy -> Today={signal.today_score} Longterm={signal.longterm_score}")

        # Determine signal strengths
        signal.today_signal = self._score_to_signal(signal.today_score)
        signal.longterm_signal = self._score_to_signal(signal.longterm_score)

        # Committee votes
        signal.committee_votes = {
            'technical': signal.technical_signal,
            'fundamental': signal.fundamental_signal,
            'sentiment': signal.sentiment_signal,
            'options': signal.options_signal,
            'earnings': signal.earnings_signal,
        }

        # Calculate agreement
        votes = [v for v in signal.committee_votes.values() if v]
        if votes:
            bullish = sum(1 for v in votes if v in ['BUY', 'STRONG_BUY'])
            bearish = sum(1 for v in votes if v in ['SELL', 'STRONG_SELL'])
            total = len(votes)

            if bullish == total:
                signal.committee_agreement = "UNANIMOUS_BULLISH"
            elif bearish == total:
                signal.committee_agreement = "UNANIMOUS_BEARISH"
            elif bullish > bearish:
                signal.committee_agreement = "MAJORITY_BULLISH"
            elif bearish > bullish:
                signal.committee_agreement = "MAJORITY_BEARISH"
            else:
                signal.committee_agreement = "SPLIT"

    def _score_to_signal(self, score: int) -> SignalStrength:
        """Convert score to signal strength."""
        if score >= 80:
            return SignalStrength.STRONG_BUY
        elif score >= 60:
            return SignalStrength.BUY
        elif score <= 20:
            return SignalStrength.STRONG_SELL
        elif score <= 40:
            return SignalStrength.SELL
        else:
            return SignalStrength.HOLD

    def _calculate_risk(self, signal: UnifiedSignal):
        """Calculate risk level and factors."""
        risk_score = 50
        risk_factors = []

        # Earnings risk
        if signal.days_to_earnings and 0 < signal.days_to_earnings <= 7:
            risk_score += 20
            risk_factors.append(f"Earnings in {signal.days_to_earnings} days")

        # Near 52-week high
        if signal.pct_from_high and signal.pct_from_high > -5:
            risk_score += 10
            risk_factors.append("Near 52-week high")

        # Near 52-week low
        if signal.pct_from_low and signal.pct_from_low < 10:
            risk_score += 10
            risk_factors.append("Near 52-week low")

        # High debt
        # (would need fundamental data)

        # Set risk level
        signal.risk_score = min(100, risk_score)
        signal.risk_factors = risk_factors

        if risk_score >= 80:
            signal.risk_level = RiskLevel.EXTREME
        elif risk_score >= 60:
            signal.risk_level = RiskLevel.HIGH
        elif risk_score >= 40:
            signal.risk_level = RiskLevel.MEDIUM
        else:
            signal.risk_level = RiskLevel.LOW

    def _generate_signal_reason(self, signal: UnifiedSignal):
        """Generate one-sentence signal reason."""
        reasons = []
        neutral_info = []

        # Technical
        if signal.technical_score >= 65:
            reasons.append("strong technicals")
        elif signal.technical_score <= 35:
            reasons.append("weak technicals")
        elif signal.technical_score != 50:
            neutral_info.append(f"tech {signal.technical_score}")

        # Sentiment
        if signal.sentiment_score >= 65:
            reasons.append("positive sentiment")
        elif signal.sentiment_score <= 35:
            reasons.append("negative sentiment")
        elif signal.sentiment_score != 50:
            neutral_info.append(f"sent {signal.sentiment_score}")

        # Options
        if signal.options_score >= 65:
            reasons.append("bullish options flow")
        elif signal.options_score <= 35:
            reasons.append("bearish options flow")
        elif signal.options_score != 50:
            neutral_info.append(f"opts {signal.options_score}")

        # Fundamentals
        if signal.fundamental_score >= 65:
            reasons.append("solid fundamentals")
        elif signal.fundamental_score <= 35:
            reasons.append("weak fundamentals")
        elif signal.fundamental_score != 50:
            neutral_info.append(f"fund {signal.fundamental_score}")

        # Earnings
        if signal.earnings_reason and "beat" in signal.earnings_reason.lower():
            reasons.append("recent earnings beat")
        elif signal.earnings_reason and "miss" in signal.earnings_reason.lower():
            reasons.append("recent earnings miss")
        elif signal.ies_score >= 70:
            reasons.append("high expectations priced in")
        elif signal.ies_score > 0 and signal.ies_score <= 30:
            reasons.append("low expectations")

        # Upcoming catalyst
        if signal.next_catalyst:
            neutral_info.append(signal.next_catalyst)

        if reasons:
            if signal.today_score >= 65:
                signal.signal_reason = f"Bullish: {', '.join(reasons[:3])}"
            elif signal.today_score <= 35:
                signal.signal_reason = f"Bearish: {', '.join(reasons[:3])}"
            else:
                signal.signal_reason = f"Mixed: {', '.join(reasons[:3])}"
        elif neutral_info:
            # No strong signals but have some data
            signal.signal_reason = f"Neutral ({', '.join(neutral_info[:3])})"
        else:
            # No data from DB - use price position
            if signal.pct_from_high is not None and signal.pct_from_high > -10:
                signal.signal_reason = f"Near 52w high ({signal.pct_from_high:+.1f}%)"
            elif signal.pct_from_low is not None and signal.pct_from_low < 20:
                signal.signal_reason = f"Near 52w low ({signal.pct_from_low:+.1f}%)"
            elif signal.data_quality == "LOW":
                signal.signal_reason = "Run screener to get analysis"
            else:
                signal.signal_reason = "Neutral - no strong signals"

    def _add_flags(self, signal: UnifiedSignal):
        """Add visual flags based on signal characteristics."""
        # Hot stock (high score + volume)
        if signal.today_score >= 80:
            signal.flags.append("🔥 Hot")

        # Earnings flags already added in earnings analysis

        # Volatile
        if signal.risk_level in [RiskLevel.HIGH, RiskLevel.EXTREME]:
            if "⚠️ Volatile" not in signal.flags:
                signal.flags.append("⚠️ High Risk")

        # In portfolio
        if signal.in_portfolio:
            signal.flags.append("💼 Held")

    # ============================================================
    # MARKET OVERVIEW
    # ============================================================

    def get_market_overview(self) -> MarketOverview:
        """Generate market-wide overview for signals page header."""
        overview = MarketOverview()

        # Macro regime
        if MACRO_REGIME_AVAILABLE:
            try:
                regime = get_current_regime()
                overview.regime = regime.regime.value if regime else "NEUTRAL"
                overview.regime_score = regime.regime_score if regime else 50
            except:
                pass

        # Market context - get VIX from MarketContext
        if MARKET_CONTEXT_AVAILABLE:
            try:
                context = get_market_context(run_ai_analysis=False)  # Skip AI analysis for speed
                if context:
                    # Access dataclass attributes directly (not .get())
                    overview.vix = context.vix_level if context.vix_level else 0
                    overview.regime = context.market_regime if context.market_regime else overview.regime
            except Exception as e:
                logger.debug(f"Error getting market context: {e}")

        # Fetch SPY/QQQ changes directly (MarketContext doesn't have daily changes)
        try:
            import yfinance as yf

            # Get SPY change
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="2d")
            if len(spy_hist) >= 2:
                prev_close = spy_hist['Close'].iloc[-2]
                curr_close = spy_hist['Close'].iloc[-1]
                overview.spy_change = ((curr_close - prev_close) / prev_close) * 100

            # Get QQQ change
            qqq = yf.Ticker("QQQ")
            qqq_hist = qqq.history(period="2d")
            if len(qqq_hist) >= 2:
                prev_close = qqq_hist['Close'].iloc[-2]
                curr_close = qqq_hist['Close'].iloc[-1]
                overview.qqq_change = ((curr_close - prev_close) / prev_close) * 100

            # Get VIX if not already set
            if overview.vix == 0:
                vix = yf.Ticker("^VIX")
                vix_hist = vix.history(period="2d")
                if len(vix_hist) >= 1:
                    overview.vix = vix_hist['Close'].iloc[-1]

        except Exception as e:
            logger.debug(f"Error fetching market data: {e}")

        # Economic calendar
        if self.economic_calendar:
            try:
                calendar = self.economic_calendar.get_calendar()
                overview.has_high_impact_today = calendar.high_impact_today
                overview.days_to_fed = calendar.days_to_fed

                # Today's events
                for event in calendar.today_events[:5]:
                    overview.economic_events_today.append({
                        'name': event.event_name,
                        'time': event.event_time,
                        'actual': event.actual,
                        'forecast': event.forecast,
                    })
            except:
                pass

        # AI Summary (simplified)
        if overview.regime_score >= 60:
            overview.ai_summary = "Risk-on environment. Favor growth stocks and cyclicals."
        elif overview.regime_score <= 40:
            overview.ai_summary = "Risk-off environment. Consider defensive positions and bonds."
        else:
            overview.ai_summary = "Mixed signals. Be selective with new positions."

        overview.updated_at = datetime.now()

        return overview

    # ============================================================
    # SNAPSHOT MANAGEMENT
    # ============================================================

    def save_signal_snapshot(self, signal: UnifiedSignal) -> bool:
        """Save signal snapshot to database for historical tracking."""
        try:
            snapshot = SignalSnapshot(
                ticker=signal.ticker,
                snapshot_date=date.today(),
                today_signal=signal.today_signal.value,
                today_score=signal.today_score,
                longterm_score=signal.longterm_score,
                risk_level=signal.risk_level.value,
                price_at_snapshot=signal.current_price,
                technical_score=signal.technical_score,
                fundamental_score=signal.fundamental_score,
                sentiment_score=signal.sentiment_score,
                options_score=signal.options_score,
                earnings_score=signal.earnings_score,
                full_signal_json=signal.to_json(),
            )

            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO signal_snapshots 
                        (ticker, snapshot_date, today_signal, today_score, longterm_score,
                         risk_level, price_at_snapshot, technical_score, fundamental_score,
                         sentiment_score, options_score, earnings_score, full_signal_json)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker, snapshot_date) DO UPDATE SET
                            today_signal = EXCLUDED.today_signal,
                            today_score = EXCLUDED.today_score,
                            full_signal_json = EXCLUDED.full_signal_json
                    """, (
                        snapshot.ticker,
                        snapshot.snapshot_date,
                        snapshot.today_signal,
                        snapshot.today_score,
                        snapshot.longterm_score,
                        snapshot.risk_level,
                        snapshot.price_at_snapshot,
                        snapshot.technical_score,
                        snapshot.fundamental_score,
                        snapshot.sentiment_score,
                        snapshot.options_score,
                        snapshot.earnings_score,
                        snapshot.full_signal_json,
                    ))
                    conn.commit()

            return True

        except Exception as e:
            logger.error(f"Error saving signal snapshot: {e}")
            return False


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

_engine = None

def get_signal_engine() -> SignalEngine:
    """Get singleton SignalEngine instance."""
    global _engine
    if _engine is None:
        _engine = SignalEngine()
    return _engine

def generate_signal(ticker: str) -> UnifiedSignal:
    """Generate signal for a ticker."""
    engine = get_signal_engine()
    return engine.generate_signal(ticker)

def generate_signals(tickers: List[str]) -> Dict[str, UnifiedSignal]:
    """Generate signals for multiple tickers."""
    engine = get_signal_engine()
    return engine.generate_signals_batch(tickers)

def get_market_overview() -> MarketOverview:
    """Get market overview."""
    engine = get_signal_engine()
    return engine.get_market_overview()