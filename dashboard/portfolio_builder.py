"""
AI Portfolio Builder V2 - With Deterministic Engine

This module provides AI-assisted portfolio construction with:
1. LLM extracts structured INTENT from user request (JSON only)
2. Python engine builds portfolio DETERMINISTICALLY
3. All constraints enforced in CODE
4. Shariah compliance filtering supported

Author: HH Research Platform
Location: dashboard/portfolio_builder.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# IMPORT PORTFOLIO ENGINE
# =============================================================================
try:
    from dashboard.portfolio_engine import (
        PortfolioEngine,
        PortfolioIntent,
        PortfolioResult,
        RiskLevel,
        PortfolioObjective,
        parse_llm_intent,
        get_intent_extraction_prompt,
        RISK_CONSTRAINTS,
        OBJECTIVE_WEIGHTS,
        check_shariah_compliance_from_data,
    )
    ENGINE_AVAILABLE = True
except ImportError:
    try:
        from portfolio_engine import (
            PortfolioEngine,
            PortfolioIntent,
            PortfolioResult,
            RiskLevel,
            PortfolioObjective,
            parse_llm_intent,
            get_intent_extraction_prompt,
            RISK_CONSTRAINTS,
            OBJECTIVE_WEIGHTS,
            check_shariah_compliance_from_data,
        )
        ENGINE_AVAILABLE = True
    except ImportError:
        ENGINE_AVAILABLE = False
        logger.warning("Portfolio engine not available")


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def get_db_connection():
    """Get database connection."""
    try:
        from src.utils.db import get_db_connection as _get_db_connection
        return _get_db_connection
    except ImportError:
        try:
            from src.db.connection import get_connection
            return get_connection
        except ImportError:
            import psycopg2
            def _connect():
                return psycopg2.connect(
                    host="localhost", port=5432, dbname="alpha_platform",
                    user="alpha", password="alpha_secure_2024"
                )
            return _connect


def _load_sectors_from_cache() -> Dict[str, str]:
    """Load sectors from sector_cache table."""
    try:
        get_conn = get_db_connection()
        query = "SELECT ticker, sector FROM sector_cache WHERE sector IS NOT NULL"

        try:
            with get_conn() as conn:
                df = pd.read_sql(query, conn)
        except TypeError:
            conn = get_conn()
            try:
                df = pd.read_sql(query, conn)
            finally:
                conn.close()

        if df.empty:
            return {}
        return dict(zip(df['ticker'], df['sector']))
    except Exception as e:
        # Table might not exist, that's OK
        logger.debug(f"Sector cache not available: {e}")
        return {}


def _save_sectors_to_cache(sector_map: Dict[str, str]):
    """Save sectors to sector_cache table."""
    if not sector_map:
        return

    try:
        get_conn = get_db_connection()

        # Create table if not exists and upsert
        create_sql = """
        CREATE TABLE IF NOT EXISTS sector_cache (
            ticker VARCHAR(20) PRIMARY KEY,
            sector VARCHAR(100),
            industry VARCHAR(100),
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        upsert_sql = """
        INSERT INTO sector_cache (ticker, sector, updated_at)
        VALUES (%s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (ticker) DO UPDATE SET 
            sector = COALESCE(EXCLUDED.sector, fundamentals.sector),
            updated_at = CURRENT_TIMESTAMP
        """

        try:
            with get_conn() as conn:
                cur = conn.cursor()
                cur.execute(create_sql)
                for ticker, sector in sector_map.items():
                    cur.execute(upsert_sql, (ticker, sector))
                conn.commit()
                cur.close()
        except TypeError:
            conn = get_conn()
            try:
                cur = conn.cursor()
                cur.execute(create_sql)
                for ticker, sector in sector_map.items():
                    cur.execute(upsert_sql, (ticker, sector))
                conn.commit()
                cur.close()
            finally:
                conn.close()

        logger.info(f"Saved {len(sector_map)} sectors to cache")
    except Exception as e:
        logger.warning(f"Could not save sector cache: {e}")


# =============================================================================
# PORTFOLIO SAVING/LOADING
# =============================================================================

def _ensure_portfolio_tables():
    """Create portfolio storage tables if they don't exist."""
    try:
        get_conn = get_db_connection()

        create_portfolios_sql = """
        CREATE TABLE IF NOT EXISTS saved_portfolios (
            id SERIAL PRIMARY KEY,
            name VARCHAR(200) NOT NULL,
            description TEXT,
            objective VARCHAR(50),
            risk_level VARCHAR(50),
            total_value NUMERIC(15,2),
            num_holdings INTEGER,
            num_sectors INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        create_holdings_sql = """
        CREATE TABLE IF NOT EXISTS saved_portfolio_holdings (
            id SERIAL PRIMARY KEY,
            portfolio_id INTEGER REFERENCES saved_portfolios(id) ON DELETE CASCADE,
            ticker VARCHAR(20) NOT NULL,
            weight_pct NUMERIC(8,4),
            shares NUMERIC(15,4),
            value NUMERIC(15,2),
            sector VARCHAR(100),
            score NUMERIC(8,2),
            signal_type VARCHAR(20),
            rationale TEXT
        )
        """

        try:
            with get_conn() as conn:
                cur = conn.cursor()
                cur.execute(create_portfolios_sql)
                cur.execute(create_holdings_sql)
                conn.commit()
                cur.close()
        except TypeError:
            conn = get_conn()
            try:
                cur = conn.cursor()
                cur.execute(create_portfolios_sql)
                cur.execute(create_holdings_sql)
                conn.commit()
                cur.close()
            finally:
                conn.close()

    except Exception as e:
        logger.error(f"Could not create portfolio tables: {e}")


def save_portfolio(name: str, result: 'PortfolioResult', description: str = None) -> Optional[int]:
    """
    Save a portfolio result to database.

    Args:
        name: Portfolio name
        result: PortfolioResult from engine
        description: Optional description

    Returns:
        Portfolio ID if saved, None if failed
    """
    if not result or not result.success:
        logger.error("Cannot save failed portfolio")
        return None

    _ensure_portfolio_tables()

    try:
        get_conn = get_db_connection()

        intent = result.intent

        insert_portfolio_sql = """
        INSERT INTO saved_portfolios 
            (name, description, objective, risk_level, total_value, num_holdings, num_sectors)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """

        insert_holding_sql = """
        INSERT INTO saved_portfolio_holdings 
            (portfolio_id, ticker, weight_pct, shares, value, sector, score, signal_type, rationale, conviction)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        portfolio_id = None

        try:
            with get_conn() as conn:
                cur = conn.cursor()

                # Insert portfolio
                cur.execute(insert_portfolio_sql, (
                    name,
                    description or f"{intent.objective} portfolio - {intent.risk_level} risk",
                    intent.objective,
                    intent.risk_level,
                    result.total_value,
                    result.num_holdings,
                    result.num_sectors
                ))
                portfolio_id = cur.fetchone()[0]

                # Insert holdings
                for h in result.holdings:
                    cur.execute(insert_holding_sql, (
                        portfolio_id,
                        str(h.ticker),
                        float(h.weight_pct),
                        int(h.shares),
                        float(h.value),
                        str(h.sector) if h.sector else None,
                        float(h.composite_score),
                        h.ai_decision.signal_type if h.ai_decision and hasattr(h.ai_decision, "signal_type") else None,
                        (h.bull_case[0] if h.bull_case else None) or (h.ai_decision.one_line_summary if h.ai_decision else None),
                        str(h.conviction) if h.conviction else 'MEDIUM'
                    ))

                conn.commit()
                cur.close()

        except TypeError:
            conn = get_conn()
            try:
                cur = conn.cursor()

                cur.execute(insert_portfolio_sql, (
                    name,
                    description or f"{intent.objective} portfolio - {intent.risk_level} risk",
                    intent.objective,
                    intent.risk_level,
                    result.total_value,
                    result.num_holdings,
                    result.num_sectors
                ))
                portfolio_id = cur.fetchone()[0]

                for h in result.holdings:
                    cur.execute(insert_holding_sql, (
                        portfolio_id,
                        str(h.ticker),
                        float(h.weight_pct),
                        int(h.shares),
                        float(h.value),
                        str(h.sector) if h.sector else None,
                        float(h.composite_score),
                        h.ai_decision.signal_type if h.ai_decision and hasattr(h.ai_decision, "signal_type") else None,
                        (h.bull_case[0] if h.bull_case else None) or (h.ai_decision.one_line_summary if h.ai_decision else None),
                        str(h.conviction) if h.conviction else 'MEDIUM'
                    ))

                conn.commit()
                cur.close()
            finally:
                conn.close()

        logger.info(f"Saved portfolio '{name}' with ID {portfolio_id}")
        return portfolio_id

    except Exception as e:
        logger.error(f"Failed to save portfolio: {e}")
        return None


def get_saved_portfolios() -> pd.DataFrame:
    """Get list of all saved portfolios."""
    _ensure_portfolio_tables()

    try:
        get_conn = get_db_connection()

        query = """
        SELECT id, name, description, objective, risk_level, 
               total_value, num_holdings, num_sectors, created_at
        FROM saved_portfolios
        ORDER BY created_at DESC
        """

        try:
            with get_conn() as conn:
                df = pd.read_sql(query, conn)
        except TypeError:
            conn = get_conn()
            try:
                df = pd.read_sql(query, conn)
            finally:
                conn.close()

        return df

    except Exception as e:
        logger.error(f"Failed to get saved portfolios: {e}")
        return pd.DataFrame()


def load_portfolio(portfolio_id: int) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    # Convert numpy.int64 to native Python int
    portfolio_id = int(portfolio_id)
    """
    Load a saved portfolio by ID.

    Returns:
        Tuple of (portfolio_info dict, holdings DataFrame)
    """
    try:
        get_conn = get_db_connection()

        portfolio_query = """
        SELECT * FROM saved_portfolios WHERE id = %s
        """

        holdings_query = """
SELECT ticker, weight_pct, shares, value, sector, score, signal_type, rationale, conviction
                FROM saved_portfolio_holdings
                WHERE portfolio_id = %s
                ORDER BY weight_pct DESC
                """

        try:
            with get_conn() as conn:
                # Get portfolio info
                portfolio_df = pd.read_sql(portfolio_query, conn, params=(portfolio_id,))
                holdings_df = pd.read_sql(holdings_query, conn, params=(portfolio_id,))
        except TypeError:
            conn = get_conn()
            try:
                portfolio_df = pd.read_sql(portfolio_query, conn, params=(portfolio_id,))
                holdings_df = pd.read_sql(holdings_query, conn, params=(portfolio_id,))
            finally:
                conn.close()

        if portfolio_df.empty:
            return None, None

        portfolio_info = portfolio_df.iloc[0].to_dict()
        return portfolio_info, holdings_df

    except Exception as e:
        logger.error(f"Failed to load portfolio: {e}")
        return None, None


def delete_portfolio(portfolio_id: int) -> bool:
    """Delete a saved portfolio."""
    try:
        get_conn = get_db_connection()

        delete_sql = "DELETE FROM saved_portfolios WHERE id = %s"

        try:
            with get_conn() as conn:
                cur = conn.cursor()
                cur.execute(delete_sql, (portfolio_id,))
                conn.commit()
                cur.close()
        except TypeError:
            conn = get_conn()
            try:
                cur = conn.cursor()
                cur.execute(delete_sql, (portfolio_id,))
                conn.commit()
                cur.close()
            finally:
                conn.close()

        logger.info(f"Deleted portfolio {portfolio_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to delete portfolio: {e}")
        return False


def update_sector_cache(tickers: List[str] = None):
    """
    Fetch sectors from yfinance and save to cache.
    Run this once to populate the cache, then loading is instant.

    Usage:
        from dashboard.portfolio_builder import update_sector_cache
        update_sector_cache()  # Updates all missing
    """
    import yfinance as yf

    if tickers is None:
        # Get tickers missing sectors
        try:
            df = get_latest_stock_universe()
            missing_mask = df['sector'].isna() | (df['sector'] == '') | (df['sector'] == 'Unknown')
            tickers = df.loc[missing_mask, 'ticker'].tolist()
        except:
            logger.error("Could not get tickers list")
            return

    if not tickers:
        logger.info("No tickers need sector updates")
        return

    logger.info(f"Fetching sectors for {len(tickers)} tickers from yfinance...")

    sector_map = {}
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i+50]
        logger.info(f"  Batch {i//50 + 1}/{(len(tickers)-1)//50 + 1}: {len(batch)} tickers")
        try:
            tickers_obj = yf.Tickers(' '.join(batch))
            for ticker in batch:
                try:
                    info = tickers_obj.tickers[ticker].info
                    sector = info.get('sector')
                    if sector:
                        sector_map[ticker] = sector
                except:
                    pass
        except Exception as e:
            logger.warning(f"Batch failed: {e}")

    if sector_map:
        _save_sectors_to_cache(sector_map)
        logger.info(f"✅ Updated {len(sector_map)} sectors in cache")
    else:
        logger.warning("No sectors fetched")


def _fetch_current_prices(tickers: List[str]) -> Dict[str, float]:
    """Fetch current prices for tickers from yfinance."""
    import yfinance as yf

    prices = {}

    # Batch fetch
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i+50]
        try:
            # Use download for faster batch fetching
            data = yf.download(batch, period='1d', progress=False, threads=True)

            if not data.empty:
                # Get the latest close price
                if 'Close' in data.columns:
                    if len(batch) == 1:
                        # Single ticker - data is Series
                        price = data['Close'].iloc[-1]
                        if pd.notna(price):
                            prices[batch[0]] = float(price)
                    else:
                        # Multiple tickers - data has multi-level columns
                        for ticker in batch:
                            try:
                                if ticker in data['Close'].columns:
                                    price = data['Close'][ticker].iloc[-1]
                                    if pd.notna(price):
                                        prices[ticker] = float(price)
                            except:
                                pass
        except Exception as e:
            logger.warning(f"Price fetch batch failed: {e}")

    return prices


def _load_prices_from_cache() -> Dict[str, float]:
    """Load prices from price_cache table (if exists)."""
    try:
        get_conn = get_db_connection()
        query = """
        SELECT ticker, price FROM price_cache 
        WHERE updated_at > NOW() - INTERVAL '1 day'
        """

        try:
            with get_conn() as conn:
                df = pd.read_sql(query, conn)
        except TypeError:
            conn = get_conn()
            try:
                df = pd.read_sql(query, conn)
            finally:
                conn.close()

        if df.empty:
            return {}
        return dict(zip(df['ticker'], df['price']))
    except:
        return {}


def _save_prices_to_cache(prices: Dict[str, float]):
    """Save prices to price_cache table."""
    if not prices:
        return

    try:
        get_conn = get_db_connection()

        create_sql = """
        CREATE TABLE IF NOT EXISTS price_cache (
            ticker VARCHAR(20) PRIMARY KEY,
            price NUMERIC(15,4),
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        upsert_sql = """
        INSERT INTO price_cache (ticker, price, updated_at)
        VALUES (%s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (ticker) DO UPDATE SET 
            price = EXCLUDED.price,
            updated_at = CURRENT_TIMESTAMP
        """

        try:
            with get_conn() as conn:
                cur = conn.cursor()
                cur.execute(create_sql)
                for ticker, price in prices.items():
                    cur.execute(upsert_sql, (ticker, price))
                conn.commit()
                cur.close()
        except TypeError:
            conn = get_conn()
            try:
                cur = conn.cursor()
                cur.execute(create_sql)
                for ticker, price in prices.items():
                    cur.execute(upsert_sql, (ticker, price))
                conn.commit()
                cur.close()
            finally:
                conn.close()

        logger.info(f"Saved {len(prices)} prices to cache")
    except Exception as e:
        logger.warning(f"Could not save price cache: {e}")


def _load_company_names_from_cache() -> Dict[str, str]:
    """Load company names from name_cache table."""
    try:
        get_conn = get_db_connection()
        query = "SELECT ticker, company_name FROM name_cache WHERE company_name IS NOT NULL"

        try:
            with get_conn() as conn:
                df = pd.read_sql(query, conn)
        except TypeError:
            conn = get_conn()
            try:
                df = pd.read_sql(query, conn)
            finally:
                conn.close()

        if df.empty:
            return {}
        return dict(zip(df['ticker'], df['company_name']))
    except:
        return {}


def _fetch_company_names(tickers: List[str]) -> Dict[str, str]:
    """Fetch company names from yfinance."""
    import yfinance as yf

    names = {}

    for i in range(0, len(tickers), 50):
        batch = tickers[i:i+50]
        try:
            tickers_obj = yf.Tickers(' '.join(batch))
            for ticker in batch:
                try:
                    info = tickers_obj.tickers[ticker].info
                    name = info.get('shortName') or info.get('longName') or info.get('name')
                    if name:
                        names[ticker] = name
                except:
                    pass
        except Exception as e:
            logger.warning(f"Batch name fetch failed: {e}")

    return names


def _save_company_names_to_cache(names: Dict[str, str]):
    """Save company names to name_cache table."""
    if not names:
        return

    try:
        get_conn = get_db_connection()

        create_sql = """
        CREATE TABLE IF NOT EXISTS name_cache (
            ticker VARCHAR(20) PRIMARY KEY,
            company_name VARCHAR(200),
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        upsert_sql = """
        INSERT INTO name_cache (ticker, company_name, updated_at)
        VALUES (%s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (ticker) DO UPDATE SET 
            company_name = EXCLUDED.company_name,
            updated_at = CURRENT_TIMESTAMP
        """

        try:
            with get_conn() as conn:
                cur = conn.cursor()
                cur.execute(create_sql)
                for ticker, name in names.items():
                    cur.execute(upsert_sql, (ticker, name))
                conn.commit()
                cur.close()
        except TypeError:
            conn = get_conn()
            try:
                cur = conn.cursor()
                cur.execute(create_sql)
                for ticker, name in names.items():
                    cur.execute(upsert_sql, (ticker, name))
                conn.commit()
                cur.close()
            finally:
                conn.close()

        logger.info(f"Saved {len(names)} company names to cache")
    except Exception as e:
        logger.warning(f"Could not save name cache: {e}")


def get_latest_stock_universe(force_refresh: bool = False) -> pd.DataFrame:
    """Load the latest stock universe with all scores and fundamentals.

    Uses session state caching - first load fetches sectors/prices from yfinance,
    subsequent loads are instant within the same session.

    Args:
        force_refresh: If True, bypass cache and reload from database
    """
    import streamlit as st

    # Check session state cache first (instant if already loaded)
    if not force_refresh and 'stock_universe_df' in st.session_state:
        cached_df = st.session_state.stock_universe_df
        if cached_df is not None and not cached_df.empty:
            logger.info(f"Using cached universe: {len(cached_df)} stocks")
            return cached_df.copy()

    # Query with explicit aliases to avoid column conflicts
    query_basic = """
WITH latest_scores AS (
    SELECT DISTINCT ON (ticker)
        ticker, date as score_date,
        sentiment_score, fundamental_score, growth_score,
        technical_score, dividend_score, options_flow_score,
        short_squeeze_score, options_sentiment, squeeze_risk,
        days_to_earnings, target_upside_pct, analyst_positivity,
        total_score
    FROM screener_scores
    ORDER BY ticker, date DESC
),
latest_fundamentals AS (
    SELECT DISTINCT ON (ticker)
        ticker, market_cap, pe_ratio, pb_ratio,
        profit_margin, gross_margin, roe,
        revenue_growth, dividend_yield,
        current_ratio, debt_to_equity, free_cash_flow, sector
    FROM fundamentals
    ORDER BY ticker, date DESC
),
latest_ai AS (
    SELECT DISTINCT ON (ticker)
        ticker, ai_action, ai_confidence,
        bull_case as ai_bull_case, bear_case as ai_bear_case,
        key_risks as ai_key_risks, one_line_summary
    FROM ai_analysis
    ORDER BY ticker, analysis_date DESC
),
latest_ai_rec AS (
    SELECT DISTINCT ON (ticker)
        ticker, ai_probability
    FROM ai_recommendations
    ORDER BY ticker, created_at DESC
),
latest_committee AS (
    SELECT DISTINCT ON (ticker)
        ticker, verdict as committee_verdict,
        conviction as committee_conviction,
        rationale as committee_rationale
    FROM committee_decisions
    ORDER BY ticker, date DESC
),
latest_alpha AS (
    SELECT DISTINCT ON (ticker)
        ticker, predicted_probability as alpha_probability,
        alpha_signal as alpha_pred_signal
    FROM alpha_predictions
    ORDER BY ticker, prediction_date DESC
),
latest_enhanced AS (
    SELECT DISTINCT ON (ticker)
        ticker, insider_score, revision_score, earnings_surprise_score
    FROM enhanced_scores
    ORDER BY ticker, date DESC
),
latest_signals AS (
    SELECT DISTINCT ON (ticker)
        ticker, 
        signal_type as trading_signal_type,
        signal_strength as trading_signal_strength,
        signal_reason as trading_signal_reason
    FROM trading_signals
    ORDER BY ticker, date DESC
),
latest_earnings AS (
    SELECT DISTINCT ON (ticker)
        ticker, earnings_date as next_earnings_date,
        eps_estimate, guidance_direction
    FROM earnings_calendar
    WHERE earnings_date >= CURRENT_DATE
    ORDER BY ticker, earnings_date ASC
),
latest_fda AS (
    SELECT DISTINCT ON (ticker)
        ticker,
        expected_date as fda_expected_date,
        drug_name as fda_drug_name,
        catalyst_type as fda_catalyst_type,
        indication as fda_indication,
        priority as fda_priority,
        date_confirmed as fda_date_confirmed
    FROM fda_calendar
    WHERE expected_date >= CURRENT_DATE
    ORDER BY ticker, expected_date ASC
),
agent_fundamental AS (
    SELECT DISTINCT ON (ticker)
        ticker, buy_prob as agent_fundamental_buy_prob,
        confidence as agent_fundamental_confidence,
        rationale as agent_fundamental_rationale
    FROM agent_votes WHERE agent_role = 'fundamental'
    ORDER BY ticker, date DESC
),
agent_sentiment AS (
    SELECT DISTINCT ON (ticker)
        ticker, buy_prob as agent_sentiment_buy_prob,
        confidence as agent_sentiment_confidence,
        rationale as agent_sentiment_rationale
    FROM agent_votes WHERE agent_role = 'sentiment'
    ORDER BY ticker, date DESC
),
agent_technical AS (
    SELECT DISTINCT ON (ticker)
        ticker, buy_prob as agent_technical_buy_prob,
        confidence as agent_technical_confidence,
        rationale as agent_technical_rationale
    FROM agent_votes WHERE agent_role = 'technical'
    ORDER BY ticker, date DESC
),
agent_valuation AS (
    SELECT DISTINCT ON (ticker)
        ticker, buy_prob as agent_valuation_buy_prob,
        confidence as agent_valuation_confidence,
        rationale as agent_valuation_rationale
    FROM agent_votes WHERE agent_role = 'valuation'
    ORDER BY ticker, date DESC
)
SELECT 
    s.*,
    f.market_cap, f.pe_ratio, f.pb_ratio, f.profit_margin, f.gross_margin,
    f.roe, f.revenue_growth, f.dividend_yield, f.current_ratio,
    f.debt_to_equity, f.free_cash_flow, f.sector,
    ai.ai_action, ai.ai_confidence, ai.ai_bull_case, ai.ai_bear_case,
    ai.ai_key_risks, ai.one_line_summary,
    ar.ai_probability,
    c.committee_verdict, c.committee_conviction, c.committee_rationale,
    ap.alpha_probability, ap.alpha_pred_signal,
    e.insider_score, e.revision_score, e.earnings_surprise_score,
    sig.trading_signal_type, sig.trading_signal_strength, sig.trading_signal_reason,
    earn.next_earnings_date, earn.eps_estimate, earn.guidance_direction,
    fda.fda_expected_date, fda.fda_drug_name, fda.fda_catalyst_type, 
    fda.fda_indication, fda.fda_priority, fda.fda_date_confirmed,
    af.agent_fundamental_buy_prob, af.agent_fundamental_confidence, af.agent_fundamental_rationale,
    asent.agent_sentiment_buy_prob, asent.agent_sentiment_confidence, asent.agent_sentiment_rationale,
    at.agent_technical_buy_prob, at.agent_technical_confidence, at.agent_technical_rationale,
    av.agent_valuation_buy_prob, av.agent_valuation_confidence, av.agent_valuation_rationale
FROM latest_scores s
LEFT JOIN latest_fundamentals f ON s.ticker = f.ticker
LEFT JOIN latest_ai ai ON s.ticker = ai.ticker
LEFT JOIN latest_ai_rec ar ON s.ticker = ar.ticker
LEFT JOIN latest_committee c ON s.ticker = c.ticker
LEFT JOIN latest_alpha ap ON s.ticker = ap.ticker
LEFT JOIN latest_enhanced e ON s.ticker = e.ticker
LEFT JOIN latest_signals sig ON s.ticker = sig.ticker
LEFT JOIN latest_earnings earn ON s.ticker = earn.ticker
LEFT JOIN latest_fda fda ON s.ticker = fda.ticker
LEFT JOIN agent_fundamental af ON s.ticker = af.ticker
LEFT JOIN agent_sentiment asent ON s.ticker = asent.ticker
LEFT JOIN agent_technical at ON s.ticker = at.ticker
LEFT JOIN agent_valuation av ON s.ticker = av.ticker
"""

    try:
        get_conn = get_db_connection()

        try:
            with get_conn() as conn:
                df = pd.read_sql(query_basic, conn)
        except TypeError:
            conn = get_conn()
            try:
                df = pd.read_sql(query_basic, conn)
            finally:
                conn.close()

        # Post-processing: filter out ETFs and invalid tickers
        etf_tickers = {
            'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'VEA', 'VWO', 'EFA', 'EEM',
            'TLT', 'IEF', 'SHY', 'BND', 'AGG', 'LQD', 'HYG', 'JNK', 'TIP', 'ZROZ',
            'GLD', 'SLV', 'IAU', 'GDX', 'GDXJ', 'USO', 'UNG', 'DBC', 'GOAU',
            'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE',
            'VNQ', 'REIT', 'ARKK', 'ARKG', 'ARKW', 'ARKF', 'ARKQ',
            'SPSK', 'HLAL', 'UMMA', 'SPRE'  # Islamic ETFs
        }

        # Filter out ETFs
        if not df.empty:
            before = len(df)
            df = df[~df['ticker'].isin(etf_tickers)]
            logger.info(f"Filtered {before - len(df)} ETFs from universe")

        # Fill missing sectors - check cache, then fetch from yfinance if needed
        if 'sector' in df.columns:
            missing_sector_mask = df['sector'].isna() | (df['sector'] == '') | (df['sector'] == 'None')
            missing_count = missing_sector_mask.sum()

            if missing_count > 0:
                # Try to load from sector_cache table first
                cached_sectors = {}
                try:
                    cached_sectors = _load_sectors_from_cache()
                    if cached_sectors:
                        for ticker, sector in cached_sectors.items():
                            mask = (df['ticker'] == ticker) & (df['sector'].isna() | (df['sector'] == '') | (df['sector'] == 'None'))
                            df.loc[mask, 'sector'] = sector

                        # Recount missing
                        missing_sector_mask = df['sector'].isna() | (df['sector'] == '') | (df['sector'] == 'None')
                        still_missing = missing_sector_mask.sum()
                        if still_missing < missing_count:
                            logger.info(f"Applied {missing_count - still_missing} sectors from cache")
                except Exception as e:
                    logger.warning(f"Could not load sector cache: {e}")

                # If still missing, fetch from yfinance (first time only)
                missing_sector_mask = df['sector'].isna() | (df['sector'] == '') | (df['sector'] == 'None')
                still_missing_count = missing_sector_mask.sum()

                if still_missing_count > 0:
                    missing_tickers = df.loc[missing_sector_mask, 'ticker'].tolist()
                    logger.info(f"Fetching sectors for {still_missing_count} stocks from yfinance (one-time)...")

                    try:
                        import yfinance as yf

                        sector_map = {}
                        for i in range(0, len(missing_tickers), 50):
                            batch = missing_tickers[i:i+50]
                            try:
                                tickers_obj = yf.Tickers(' '.join(batch))
                                for ticker in batch:
                                    try:
                                        info = tickers_obj.tickers[ticker].info
                                        sector = info.get('sector')
                                        if sector:
                                            sector_map[ticker] = sector
                                    except:
                                        pass
                            except Exception as e:
                                logger.warning(f"Batch sector fetch failed: {e}")

                        # Apply fetched sectors
                        if sector_map:
                            for ticker, sector in sector_map.items():
                                df.loc[df['ticker'] == ticker, 'sector'] = sector
                            logger.info(f"Fetched {len(sector_map)} sectors from yfinance")

                            # Save to cache for next time
                            try:
                                _save_sectors_to_cache(sector_map)
                            except Exception as e:
                                logger.warning(f"Could not save sector cache: {e}")
                    except Exception as e:
                        logger.warning(f"yfinance sector fetch failed: {e}")

            # Fill remaining None sectors as "Unknown"
            still_missing = df['sector'].isna() | (df['sector'] == '') | (df['sector'] == 'None')
            if still_missing.sum() > 0:
                df.loc[still_missing, 'sector'] = 'Unknown'
                logger.info(f"Set {still_missing.sum()} stocks to 'Unknown' sector")

        # Fetch current prices if not available
        if 'price' not in df.columns and 'close' not in df.columns and 'last_price' not in df.columns:
            # Initialize price column
            df['price'] = None

            # Try cache first
            cached_prices = _load_prices_from_cache()
            if cached_prices:
                for ticker, price in cached_prices.items():
                    df.loc[df['ticker'] == ticker, 'price'] = price
                cached_count = df['price'].notna().sum()
                if cached_count > 0:
                    logger.info(f"Loaded {cached_count} prices from cache")

            # Fetch missing prices
            missing_price_tickers = df.loc[df['price'].isna(), 'ticker'].tolist()

            if missing_price_tickers:
                logger.info(f"Fetching prices for {len(missing_price_tickers)} stocks from yfinance...")
                try:
                    prices = _fetch_current_prices(missing_price_tickers)
                    if prices:
                        for ticker, price in prices.items():
                            df.loc[df['ticker'] == ticker, 'price'] = price
                        logger.info(f"Fetched prices for {len(prices)} stocks")

                        # Save to cache
                        try:
                            _save_prices_to_cache(prices)
                        except Exception as e:
                            logger.warning(f"Could not cache prices: {e}")
                except Exception as e:
                    logger.warning(f"Could not fetch prices: {e}")

        # Fetch company names if not available
        if 'company_name' not in df.columns and 'name' not in df.columns:
            df['company_name'] = None

            # Try to load from sector_cache (which may have names too)
            cached_names = _load_company_names_from_cache()
            if cached_names:
                for ticker, name in cached_names.items():
                    df.loc[df['ticker'] == ticker, 'company_name'] = name
                cached_count = df['company_name'].notna().sum()
                if cached_count > 0:
                    logger.info(f"Loaded {cached_count} company names from cache")

            # Fetch missing names
            missing_name_tickers = df.loc[df['company_name'].isna(), 'ticker'].tolist()

            if missing_name_tickers:
                logger.info(f"Fetching company names for {len(missing_name_tickers)} stocks...")
                try:
                    names = _fetch_company_names(missing_name_tickers)
                    if names:
                        for ticker, name in names.items():
                            df.loc[df['ticker'] == ticker, 'company_name'] = name
                        logger.info(f"Fetched names for {len(names)} stocks")

                        # Save to cache
                        try:
                            _save_company_names_to_cache(names)
                        except Exception as e:
                            logger.warning(f"Could not cache names: {e}")
                except Exception as e:
                    logger.warning(f"Could not fetch company names: {e}")

        # Add empty columns for Shariah if not present
        for col in ['industry', 'total_debt', 'cash', 'total_assets']:
            if col not in df.columns:
                df[col] = None

        # Cache to session state for instant reload
        st.session_state.stock_universe_df = df.copy()
        logger.info(f"Cached {len(df)} stocks to session state")

        return df

    except Exception as e:
        logger.error(f"Error loading stock universe: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def get_sector_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Get summary statistics by sector."""
    if df.empty or 'sector' not in df.columns:
        return pd.DataFrame()
    summary = df.groupby('sector').agg({
        'ticker': 'count', 'total_score': 'mean', 'sentiment_score': 'mean',
        'fundamental_score': 'mean', 'market_cap': 'sum', 'dividend_yield': 'mean',
        'pe_ratio': 'mean',
    }).round(2)
    summary.columns = ['Count', 'Avg Score', 'Avg Sentiment', 'Avg Fundamental',
                       'Total Market Cap', 'Avg Div Yield', 'Avg PE']
    return summary.sort_values('Avg Score', ascending=False)


def get_top_stocks_by_category(df: pd.DataFrame, n: int = 10) -> Dict[str, pd.DataFrame]:
    """Get top stocks for different investment categories."""
    categories = {}
    if df.empty:
        return categories

    if 'total_score' in df.columns:
        categories['best_overall'] = df.nlargest(n, 'total_score')[
            ['ticker', 'sector', 'total_score', 'trading_signal_type', 'market_cap']].copy()

    if 'pe_ratio' in df.columns and 'fundamental_score' in df.columns:
        value_df = df[df['pe_ratio'].notna() & (df['pe_ratio'] > 0) & (df['pe_ratio'] < 30)].copy()
        if not value_df.empty:
            value_df['value_score'] = value_df['fundamental_score'] - (value_df['pe_ratio'] / 5)
            categories['best_value'] = value_df.nlargest(n, 'value_score')[
                ['ticker', 'sector', 'pe_ratio', 'fundamental_score', 'market_cap']]

    if 'growth_score' in df.columns:
        growth_df = df[df['growth_score'].notna()].copy()
        if not growth_df.empty:
            categories['best_growth'] = growth_df.nlargest(n, 'growth_score')[
                ['ticker', 'sector', 'growth_score', 'revenue_growth']]

    if 'dividend_yield' in df.columns:
        div_df = df[df['dividend_yield'].notna() & (df['dividend_yield'] > 0)].copy()
        if not div_df.empty:
            categories['best_dividend'] = div_df.nlargest(n, 'dividend_yield')[
                ['ticker', 'sector', 'dividend_yield', 'dividend_score']]

    if 'technical_score' in df.columns:
        momentum_df = df.copy()
        opt_flow = momentum_df['options_flow_score'].fillna(50) if 'options_flow_score' in momentum_df.columns else 50
        momentum_df['momentum_score'] = (momentum_df['technical_score'].fillna(50) + opt_flow) / 2
        categories['best_momentum'] = momentum_df.nlargest(n, 'momentum_score')[
            ['ticker', 'sector', 'technical_score', 'options_flow_score', 'trading_signal_type']]

    if 'trading_signal_type' in df.columns and 'signal_strength' in df.columns:
        buy_df = df[df['trading_signal_type'].isin(['STRONG_BUY', 'BUY'])].copy()
        if not buy_df.empty:
            categories['buy_signals'] = buy_df.nlargest(n, 'signal_strength')[
                ['ticker', 'sector', 'trading_signal_type', 'signal_strength', 'total_score']]

    return categories


# =============================================================================
# LLM INTERFACE
# =============================================================================

def get_llm_client(model_id: str = None):
    """
    Get LLM client using the same model selected in sidebar.

    Tries in order:
    1. ai_settings component (sidebar's model selector)
    2. ai_models_config (model registry)
    3. Direct fallback to local Qwen
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    model_name = "Unknown Model"

    # Try to use ai_settings component first (same as sidebar)
    try:
        from src.components.ai_settings import get_current_model_response, get_current_model

        model = get_current_model()
        if model:
            # Handle both object and dict style model configs
            if hasattr(model, 'icon'):
                model_name = f"{model.icon} {model.name}"
            elif isinstance(model, dict):
                model_name = f"{model.get('icon', '🤖')} {model.get('name', 'Unknown')}"
            else:
                model_name = str(model)

            logger.info(f"Using ai_settings model: {model_name}")

            class AISettingsWrapper:
                def __init__(self, name):
                    self.model_name = name

                def chat(self, messages):
                    try:
                        # Convert to single prompt for ai_settings
                        prompt = ""
                        for m in messages:
                            if m["role"] == "system":
                                prompt += f"System: {m['content']}\n\n"
                            elif m["role"] == "user":
                                prompt += f"User: {m['content']}\n\n"
                            elif m["role"] == "assistant":
                                prompt += f"Assistant: {m['content']}\n\n"

                        response = ""
                        # Try with stream parameter first, then without
                        try:
                            result = get_current_model_response(prompt, stream=False)
                        except TypeError:
                            # If stream parameter not supported, call without it
                            result = get_current_model_response(prompt)

                        if isinstance(result, str):
                            response = result
                        else:
                            for chunk in result:
                                response += chunk

                        # Clean thinking tags if present
                        if '</think>' in response:
                            response = response.split('</think>')[-1].strip()

                        return response
                    except Exception as e:
                        logger.error(f"ai_settings chat error: {e}")
                        raise RuntimeError(f"LLM call failed: {e}")  # Raise instead of returning error string

            return AISettingsWrapper(model_name)

    except ImportError:
        logger.debug("ai_settings not available, trying ai_models_config")
    except Exception as e:
        logger.warning(f"ai_settings error: {e}, trying fallback")

    # Try ai_models_config
    try:
        from src.components.ai_models_config import get_all_models

        all_models = get_all_models()

        # Use model_id if provided, else get from session state
        if not model_id:
            import streamlit as st
            model_id = st.session_state.get('ai_model', 'qwen_local')

        if model_id in all_models:
            config = all_models[model_id]
            provider = config.get("provider", "qwen")
            api_id = config.get("api_id", model_id)
            model_name = f"{config.get('icon', '🤖')} {config.get('name', model_id)}"
            api_key_env = config.get("api_key_env")

            logger.info(f"Using ai_models_config: {model_name} ({provider})")

            if provider == "openai":
                from openai import OpenAI
                api_key = os.getenv(api_key_env) if api_key_env else os.getenv("OPENAI_API_KEY")
                client = OpenAI(api_key=api_key, timeout=120)

                class OpenAIWrapper:
                    def __init__(self, client, model, name):
                        self.client = client
                        self.model = model
                        self.model_name = name

                    def chat(self, messages):
                        try:
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                temperature=0.3,
                                max_tokens=4000
                            )
                            return response.choices[0].message.content
                        except Exception as e:
                            logger.error(f"OpenAI chat error: {e}")
                            return f"Error: {e}"

                return OpenAIWrapper(client, api_id, model_name)

            elif provider == "anthropic":
                import anthropic
                api_key = os.getenv(api_key_env) if api_key_env else os.getenv("ANTHROPIC_API_KEY")
                client = anthropic.Anthropic(api_key=api_key)

                class AnthropicWrapper:
                    def __init__(self, client, model, name):
                        self.client = client
                        self.model = model
                        self.model_name = name

                    def chat(self, messages):
                        try:
                            system_msg = ""
                            chat_msgs = []
                            for m in messages:
                                if m["role"] == "system":
                                    system_msg = m["content"]
                                else:
                                    chat_msgs.append({"role": m["role"], "content": m["content"]})

                            response = self.client.messages.create(
                                model=self.model,
                                max_tokens=4000,
                                system=system_msg,
                                messages=chat_msgs
                            )
                            return response.content[0].text
                        except Exception as e:
                            logger.error(f"Anthropic chat error: {e}")
                            return f"Error: {e}"

                return AnthropicWrapper(client, api_id, model_name)

            else:
                # Qwen/local OpenAI-compatible
                from openai import OpenAI
                base_url = config.get("base_url") or os.getenv(config.get("base_url_env", "LLM_QWEN_BASE_URL"), "http://172.23.193.91:8090/v1")
                api_key = os.getenv(api_key_env) if api_key_env else "not-needed"

                client = OpenAI(base_url=base_url, api_key=api_key, timeout=120)

                class QwenWrapper:
                    def __init__(self, client, model, name):
                        self.client = client
                        self.model = model
                        self.model_name = name

                    def chat(self, messages):
                        try:
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                temperature=0.3,
                                max_tokens=4000
                            )
                            return response.choices[0].message.content
                        except Exception as e:
                            logger.error(f"Qwen chat error: {e}")
                            return f"Error: {e}"

                return QwenWrapper(client, api_id, model_name)

    except ImportError:
        logger.debug("ai_models_config not available, using fallback")
    except Exception as e:
        logger.warning(f"ai_models_config error: {e}, using fallback")

    # Fallback to direct local Qwen
    try:
        from openai import OpenAI

        base_url = os.getenv("LLM_QWEN_BASE_URL", "http://172.23.193.91:8090/v1")
        model = os.getenv("LLM_QWEN_MODEL", "Qwen3-32B-Q6_K.gguf")
        model_name = "🏠 Qwen3-32B (Local)"

        client = OpenAI(base_url=base_url, api_key="not-needed", timeout=120)

        class FallbackWrapper:
            def __init__(self, client, model, name):
                self.client = client
                self.model = model
                self.model_name = name

            def chat(self, messages):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=4000
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    logger.error(f"Fallback chat error: {e}")
                    return f"Error: {e}"

        logger.info(f"Using fallback: {model_name}")
        return FallbackWrapper(client, model, model_name)

    except Exception as e:
        logger.error(f"All LLM clients failed: {e}")
        return None


def get_ai_response(user_request: str, df: pd.DataFrame, history: List[Dict] = None, model_id: str = None) -> str:
    """
    Smart AI response:
    - If user asks to BUILD a portfolio -> use engine
    - If user asks questions -> use LLM conversationally

    Args:
        user_request: User's message
        df: Stock universe DataFrame
        history: Chat history
        model_id: Selected AI model ID (from AI_MODELS)

    Returns:
        Response string with model attribution
    """
    if not ENGINE_AVAILABLE:
        return "Portfolio engine not available. Make sure portfolio_engine.py exists."

    # Detect if this is a BUILD request or a QUESTION
    build_keywords = [
        'build', 'create', 'construct', 'make', 'generate', 'design',
        'portfolio with', 'portfolio of', 'etf with', 'stocks with',
        'diversified', 'dividend portfolio', 'growth portfolio', 'value portfolio',
        'shariah', 'halal', 'islamic', 'compliant portfolio',
        'aggressive portfolio', 'conservative portfolio', 'balanced portfolio',
        '$100', '$500', '$1000', '$10000', '$100000', '$1m', 'million',
        # AI/Tech theme keywords
        'ai companies', 'ai stocks', 'ai portfolio', 'artificial intelligence',
        'semiconductor', 'chip companies', 'gpu stocks', 'tech portfolio',
        'cybersecurity', 'fintech', 'cloud computing'
    ]

    request_lower = user_request.lower()
    is_build_request = any(kw in request_lower for kw in build_keywords)

    # Also check if it starts with action words
    action_starts = ['build', 'create', 'make', 'generate', 'design', 'construct', 'give me', 'i want', 'i need']
    if any(request_lower.strip().startswith(a) for a in action_starts):
        is_build_request = True

    if is_build_request:
        # BUILD MODE: Use engine
        return _build_portfolio_response(user_request, df, history, model_id)
    else:
        # CHAT MODE: Use LLM conversationally
        return _chat_response(user_request, df, history, model_id)


def _build_portfolio_response(user_request: str, df: pd.DataFrame, history: List[Dict] = None, model_id: str = None) -> str:
    """Build portfolio using engine. NO FALLBACKS - fail explicitly if LLM fails."""
    try:
        client = get_llm_client(model_id)
        if not client:
            return "❌ **LLM client not available.** Check your LLM server connection and try again."

        model_name = getattr(client, 'model_name', 'Unknown Model')

        tickers = df['ticker'].dropna().astype(str).tolist()
        sectors = df['sector'].dropna().astype(str).unique().tolist()

        intent_prompt = get_intent_extraction_prompt(
            user_request=user_request, sectors=sectors, tickers=tickers
        )

        messages = [
            {"role": "system", "content": "You extract structured investment intent. Output ONLY a JSON object, no other text."},
            {"role": "user", "content": intent_prompt}
        ]

        try:
            llm_output = client.chat(messages)
        except Exception as e:
            return f"❌ **LLM call failed:** {e}\n\nPlease check your LLM server is running and try again."

        # Log LLM output for debugging
        logger.info(f"LLM intent output ({model_name}): {llm_output[:500]}...")

        # Check for error responses
        if llm_output.startswith("Error:") or not llm_output.strip():
            return f"❌ **LLM returned an error or empty response:**\n\n```\n{llm_output}\n```\n\nPlease check your LLM server and try again."

        intent, intent_errors = parse_llm_intent(llm_output, valid_tickers=tickers, valid_sectors=sectors)

        # CRITICAL: If no JSON found, DO NOT proceed with defaults
        if "No JSON found in LLM output" in intent_errors:
            return f"""❌ **Failed to parse LLM response as JSON.**

The LLM did not return valid JSON. Raw output:

```
{llm_output[:1000]}{'...' if len(llm_output) > 1000 else ''}
```

**Possible causes:**
1. LLM server returned an error
2. Prompt was too long/complex for the model
3. Model is not configured correctly

Please try:
1. Simplify your request
2. Check LLM server status
3. Try a different model"""

        # Log parsed intent
        logger.info(f"Parsed intent: max_holdings={intent.max_holdings}, objective={intent.objective}, risk={intent.risk_level}, restrict_to_tickers={intent.restrict_to_tickers}")

        # If restrict_to_tickers is True but no tickers found, fail
        if intent.restrict_to_tickers and not intent.tickers_include:
            return f"""❌ **You specified a restricted ticker list, but no valid tickers were parsed.**

Please ensure your tickers are in the database. Use Quick Add to analyze them first.

Parsed errors: {intent_errors}"""

        engine = PortfolioEngine(df)
        result = engine.build_portfolio(intent, user_request=user_request)

        md_parts = []

        # Add model attribution at the top
        md_parts.append(f"*Analyzed by {model_name}*\n")

        if intent_errors:
            md_parts.append("### ⚠️ Intent Parse Notes\n")
            for e in intent_errors:
                md_parts.append(f"- {e}")
            md_parts.append("")

        md_parts.append(result.to_markdown())

        # Store result context for follow-up questions
        import streamlit as st
        st.session_state.last_portfolio_result = result
        st.session_state.last_portfolio_intent = intent
        st.session_state.last_portfolio_model = model_name

        return "\n".join(md_parts)

    except Exception as e:
        logger.error(f"AI portfolio error: {e}")
        import traceback
        return f"❌ **Error building portfolio:** {e}\n\n```\n{traceback.format_exc()}\n```"


def _chat_response(user_request: str, df: pd.DataFrame, history: List[Dict] = None, model_id: str = None) -> str:
    """Conversational response about portfolios."""
    try:
        client = get_llm_client(model_id)
        if not client:
            return "❌ LLM client not available. Check your LLM server connection."

        model_name = getattr(client, 'model_name', 'Unknown Model')

        import streamlit as st

        # Build context about last portfolio if exists
        context_parts = []

        if hasattr(st.session_state, 'last_portfolio_result') and st.session_state.last_portfolio_result:
            result = st.session_state.last_portfolio_result
            intent = getattr(st.session_state, 'last_portfolio_intent', None)

            context_parts.append("=== LAST BUILT PORTFOLIO ===")
            context_parts.append(f"Objective: {intent.objective if intent else 'N/A'}")
            context_parts.append(f"Risk Level: {intent.risk_level if intent else 'N/A'}")
            context_parts.append(f"Total Value: ${result.total_value:,.0f}")
            context_parts.append(f"Holdings: {result.num_holdings}")
            context_parts.append(f"Sectors: {result.num_sectors}")
            context_parts.append(f"Max Position: {result.max_position_pct:.1f}%")
            context_parts.append(f"Max Sector: {result.max_sector_pct:.1f}%")
            context_parts.append(f"Cash Buffer: ${result.cash_value:,.0f}")
            context_parts.append("")

            # Weight methodology
            context_parts.append("=== WEIGHT CALCULATION METHOD ===")
            if intent and intent.equal_weight:
                context_parts.append("Method: EQUAL WEIGHT (each stock gets same %)")
            else:
                context_parts.append("Method: SCORE-PROPORTIONAL (higher score = higher weight)")
                context_parts.append("Formula: weight = (stock_score / sum_of_all_scores) * investable_pct")

            context_parts.append("")
            context_parts.append("Position caps enforced iteratively:")
            context_parts.append(f"- Max position cap: {result.constraints_used.get('max_position_pct', 'N/A')}%")
            context_parts.append(f"- Max sector cap: {result.constraints_used.get('max_sector_pct', 'N/A')}%")
            context_parts.append("Excess weight redistributed to uncapped positions.")
            context_parts.append("")

            # Top holdings for context
            context_parts.append("=== TOP 10 HOLDINGS ===")
            for h in sorted(result.holdings, key=lambda x: -x.weight_pct)[:10]:
                context_parts.append(f"- {h.ticker}: {h.weight_pct:.1f}% (score={h.composite_score:.0f}, sector={h.sector})")

            context_parts.append("")
            context_parts.append("=== CONSTRAINTS USED ===")
            for k, v in result.constraints_used.items():
                context_parts.append(f"- {k}: {v}")

        # Universe summary
        context_parts.append("")
        context_parts.append(f"=== STOCK UNIVERSE ===")
        context_parts.append(f"Total stocks available: {len(df)}")
        if 'sector' in df.columns:
            context_parts.append(f"Sectors: {df['sector'].nunique()}")
        if 'trading_signal_type' in df.columns:
            buy_count = len(df[df['trading_signal_type'].isin(['BUY', 'STRONG_BUY'])])
            context_parts.append(f"BUY signals: {buy_count}")

        context = "\n".join(context_parts)

        # Build conversation
        system_prompt = f"""You are a helpful portfolio advisor AI. Answer questions about portfolio construction, 
stock selection, and investment strategy. Be specific and reference the actual data when available.

CONTEXT:
{context}

SCORING METHODOLOGY:
- Composite score combines: market_cap, fundamental_score, technical_score, sentiment, signal_strength
- Weights depend on objective (growth emphasizes technical/momentum, value emphasizes fundamentals, etc.)
- Score is normalized 0-100

WEIGHT CALCULATION:
- For score-proportional: weight_pct = (stock_score / total_scores) * investable_percent
- Position caps are enforced iteratively with excess redistributed
- Sector caps prevent over-concentration

Be concise but informative. If asked about specific stocks or methodology, give detailed answers."""

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        if history:
            for msg in history[-6:]:  # Last 6 messages for context
                messages.append({"role": msg["role"], "content": msg["content"][:1000]})

        messages.append({"role": "user", "content": user_request})

        response = client.chat(messages)

        # Add model attribution
        return f"{response}\n\n---\n*Answered by {model_name}*"

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return f"❌ Error: {e}"


# =============================================================================
# PORTFOLIO TEMPLATES
# =============================================================================

PORTFOLIO_TEMPLATES = {
    "balanced_etf": {
        "name": "Balanced All-Weather ETF",
        "description": "15-25 stocks diversified across sectors",
        "intent": PortfolioIntent(objective="balanced", risk_level="moderate", portfolio_value=100000) if ENGINE_AVAILABLE else None
    },
    "growth_portfolio": {
        "name": "High Growth Portfolio",
        "description": "10-20 aggressive growth stocks",
        "intent": PortfolioIntent(objective="growth", risk_level="aggressive", portfolio_value=100000) if ENGINE_AVAILABLE else None
    },
    "dividend_income": {
        "name": "Dividend Income Portfolio",
        "description": "15-25 high yield stocks",
        "intent": PortfolioIntent(objective="income", risk_level="conservative", portfolio_value=100000, min_dividend_yield=2.0) if ENGINE_AVAILABLE else None
    },
    "value_picks": {
        "name": "Deep Value Portfolio",
        "description": "10-20 undervalued stocks",
        "intent": PortfolioIntent(objective="value", risk_level="moderate", portfolio_value=100000) if ENGINE_AVAILABLE else None
    },
    "momentum_trades": {
        "name": "Momentum Trading Portfolio",
        "description": "10-15 high momentum stocks",
        "intent": PortfolioIntent(objective="momentum", risk_level="aggressive", portfolio_value=100000) if ENGINE_AVAILABLE else None
    },
    "quality_compounders": {
        "name": "Quality Compounders",
        "description": "15-25 high quality stocks",
        "intent": PortfolioIntent(objective="quality", risk_level="conservative", portfolio_value=100000) if ENGINE_AVAILABLE else None
    },
    "shariah_compliant": {
        "name": "Shariah Compliant Portfolio",
        "description": "15-25 halal stocks excluding financials, gambling, alcohol",
        "intent": PortfolioIntent(objective="shariah", risk_level="moderate", portfolio_value=100000, shariah_compliant=True) if ENGINE_AVAILABLE else None
    },
}


def build_portfolio_from_intent(intent: PortfolioIntent, df: pd.DataFrame, user_request: str = "") -> Tuple[Optional[PortfolioResult], List[str]]:
    """Build portfolio from explicit intent."""
    if not ENGINE_AVAILABLE:
        return None, ["Portfolio engine not available"]
    try:
        engine = PortfolioEngine(df)
        result = engine.build_portfolio(intent, user_request=user_request)
        return result, result.warnings + result.errors
    except Exception as e:
        return None, [f"Engine error: {e}"]


def build_portfolio_ai_context(df: pd.DataFrame) -> str:
    """Legacy function."""
    if df.empty:
        return "No stock data available."
    lines = [f"Total Stocks: {len(df)}"]
    if 'sector' in df.columns:
        lines.append(f"Sectors: {df['sector'].nunique()}")
    return "\n".join(lines)


def build_portfolio_instructions() -> str:
    """Legacy function."""
    return "Portfolio construction handled by deterministic engine."