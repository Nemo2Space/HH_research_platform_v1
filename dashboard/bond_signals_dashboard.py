"""
Unified Bond & Sukuk Trading Tab - Complete Dashboard

Full integration of:
- Bond signals (TLT, ZROZ, EDV, etc.)
- USD Sukuk signals (HYSK holdings via IBKR)
- Institutional analysis (Futures, Auctions, MOVE, Curve Trades)
- News fetching with sentiment analysis
- Technical analysis
- AI Chat integration
- AI Predictions (yield momentum, calendar effects, cross-asset signals)
- All data fed to AI for comprehensive advice

Author: HH Research Platform
Location: dashboard/bond_signals_dashboard.py
"""

import streamlit as st
import pandas as pd
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
import pickle
import os
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# HELPER FUNCTIONS FOR FORMATTING AND SANITIZATION
# =============================================================================

def fmt_usd(x: float) -> str:
    """Format a number as USD currency."""
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "$0.00"


def fmt_pct(x: float) -> str:
    """Format a number as percentage with sign."""
    try:
        return f"{float(x):+.1f}%"
    except Exception:
        return "+0.0%"


def sanitize_bond_ai_answer(text: str, allow_polymarket: bool = False, numeric_block: str = None) -> str:
    """Sanitize AI output to remove hallucinations and fix formatting.

    Args:
        text: Raw AI response
        allow_polymarket: If False, removes all Polymarket mentions
        numeric_block: If provided, can replace corrupted sections with this clean data

    Returns:
        Cleaned response text
    """
    if not text:
        return text

    out = text

    # Remove Polymarket mentions if not allowed
    if not allow_polymarket:
        out = re.sub(r'(?im)^.*polymarket.*$\n?', '', out)
        out = re.sub(r'(?i)\bpolymarket\b[^.\n]*[.\n]?', '', out)

    # =========================================================================
    # REMOVE HALLUCINATED OPTIONS DATA
    # =========================================================================
    out = re.sub(r'(?i)put[/-]?call\s*ratio\s*[=:]\s*[\d.]+[^.\n]*[.\n]?', '', out)
    out = re.sub(r'(?i)\(put[/-]?call\s*ratio\s*[=:]\s*[\d.]+[^)]*\)', '', out)
    out = re.sub(r'(?i)options\s*flow[^.\n]*put[/-]?call[^.\n]*[.\n]?', '', out)
    out = re.sub(r'(?i)neutral\s*options\s*flow[^.\n]*[.\n]?', '', out)
    out = re.sub(r'(?i)no\s*strong\s*options\s*flow[^.\n]*[.\n]?', '', out)

    # =========================================================================
    # FIX DOUBLE-MERGED TARGET PATTERNS (most critical)
    # Pattern: 89.97(+2.389.97(+2.387.93) -> $89.97 (+2.3% upside from $87.93)
    # =========================================================================
    out = re.sub(
        r'\$?(\d+\.\d+)\s*\(\+(\d+\.\d)\s*\1\s*\(\+\2\s*(\d+\.\d+)\)',
        r'$\1 (+\2% from $\3)',
        out
    )

    # Simpler version: 89.97(+2.387.93) -> $89.97 (+2.3% from $87.93)
    out = re.sub(
        r'\$?(\d+\.\d+)\s*\(\+(\d+\.\d)(\d+\.\d+)\)',
        r'$\1 (+\2% from $\3)',
        out
    )

    # =========================================================================
    # FIX "and" MERGED PATTERNS
    # Pattern: (87.93)andZROZ(87.93)andZROZ(65.02) -> $87.93 (TLT) and $65.02 (ZROZ)
    # =========================================================================
    out = re.sub(
        r'\(\$?(\d+\.\d+)\)\s*and\s*ZROZ\s*\(\$?\1\)\s*and\s*ZROZ\s*\(\$?(\d+\.\d+)\)',
        r'$\1 (TLT) and $\2 (ZROZ)',
        out
    )
    out = re.sub(
        r'\$?(\d+\.\d+)\s*\)\s*and\s*ZROZ\s*\(\s*\$?\1\s*\)\s*and\s*ZROZ\s*\(\s*\$?(\d+\.\d+)',
        r'\1 (TLT) and $\2 (ZROZ)',
        out
    )
    # Generic: $(X)andY$(X)andY$(Z) pattern
    out = re.sub(
        r'\(\$?(\d+\.\d+)\)\s*and\s*[A-Z]+\s*\(\$?\1\)\s*and\s*[A-Z]+\s*\(\$?(\d+\.\d+)\)',
        r'$\1 and $\2',
        out
    )

    # =========================================================================
    # FIX DUPLICATE ENTRY ZONE PATTERNS (all dash types)
    # Pattern: 85.29–85.29–87.05 -> 85.29–87.05
    # Must handle: hyphen(-), en-dash(–), em-dash(—), and mixed
    # =========================================================================
    # With dollar signs
    out = re.sub(r'\$(\d+\.?\d*)\s*[-–—]\s*\$?\1\s*[-–—]\s*\$?(\d+\.?\d*)', r'$\1–$\2', out)
    # Without dollar signs
    out = re.sub(r'(\d+\.\d+)\s*[-–—]\s*\1\s*[-–—]\s*(\d+\.\d+)', r'\1–\2', out)
    # Integer version
    out = re.sub(r'(\d+)\s*[-–—]\s*\1\s*[-–—]\s*(\d+)', r'\1–\2', out)

    # =========================================================================
    # FIX "vs" DUPLICATIONS (including "vsentry" patterns)
    # =========================================================================
    # Pattern: 87.93vsentry87.93vsentry85.29 -> $87.93 vs entry $85.29
    out = re.sub(
        r'\$?(\d+\.\d+)\s*vs\s*entry\s*\$?\1\s*vs\s*entry\s*\$?(\d+\.\d+)',
        r'$\1 vs entry $\2',
        out
    )
    out = re.sub(r'(\d+\.\d+)\s*vs\s*\1\s*vs\s*(\d+\.\d+)', r'\1 vs \2', out)
    out = re.sub(r'vs\s+(\d+\.\d+)\s*vs\s*(\d+\.\d+)', r'\1 vs \2', out)

    # =========================================================================
    # FIX MARKDOWN CORRUPTION AROUND PRICES
    # =========================================================================
    def clean_price_line(match):
        return re.sub(r'\*+', '', match.group(0))
    out = re.sub(r'^.*\$\d+.*$', clean_price_line, out, flags=re.MULTILINE)

    out = re.sub(r'\$(\d+\.?\d*)\*+\(([^:]+):\*+\$\1\*+\([^:]+:\*+\$(\d+\.?\d*)', r'$\1 (\2: $\3)', out)
    out = re.sub(r'\$(\d+\.?\d*)\s*\(([^:]+):\s*\$\1\s*\([^:]+:', r'$\1 (\2:', out)

    # Remove stray asterisks around dollar amounts
    out = re.sub(r'\*+(\$\d+\.?\d*)\*+', r'\1', out)
    out = re.sub(r'(\$\d+\.?\d*)\*+', r'\1', out)
    out = re.sub(r'\*+(\$\d+\.?\d*)', r'\1', out)

    # Clean corrupted ** patterns
    out = re.sub(r'\*\*\*+', '**', out)
    out = re.sub(r'\*\*\s*\*\*', '', out)
    out = re.sub(r'\$(\d+\.?\d*)\s*\*+\s*\(', r'$\1 (', out)

    # Collapse excessive whitespace
    out = re.sub(r'\n{3,}', '\n\n', out).strip()

    # =========================================================================
    # DETECT SEVERE CORRUPTION AND OFFER NUMERIC BLOCK RESTORATION
    # =========================================================================
    severe_corruption_patterns = [
        r'\d+\.\d+\s*\(\+\d+\.\d\d+\.\d+\s*\(\+',  # Double merged: 89.97(+2.389.97(+2.3
        r'\d+\.\d+\s*[-–—]\s*\d+\.\d+\s*[-–—]\s*\d+\.\d+\s*[-–—]',  # Triple entry
    ]

    if numeric_block:
        for pattern in severe_corruption_patterns:
            if re.search(pattern, out):
                out += f"\n\n---\n⚠️ Some numbers above may be corrupted. Reference data:\n{numeric_block}"
                break

    return out

# =============================================================================
# BOND PREDICTION ENGINE IMPORT
# =============================================================================
try:
    from src.analytics.bond_prediction_engine import (
        BondPredictionEngine,
        get_calendar_alerts,
        get_rate_signals,
        predict_bond,
    )
    PREDICTION_ENGINE_AVAILABLE = True
except ImportError:
    PREDICTION_ENGINE_AVAILABLE = False
    logger.debug("Bond prediction engine not available")

# =============================================================================
# MACRO EVENT ENGINE IMPORT
# =============================================================================
try:
    from src.analytics.macro_event_engine import (
        get_macro_engine,
        get_macro_factors,
        get_macro_context,
        refresh_macro_data,
    )
    MACRO_ENGINE_AVAILABLE = True
except ImportError:
    MACRO_ENGINE_AVAILABLE = False
    logger.debug("Macro event engine not available")

# =============================================================================
# SUKUK MODULE IMPORT
# =============================================================================
try:
    from src.analytics.sukuk_data import load_sukuk_universe, fetch_sukuk_market_data
    from src.analytics.sukuk_signals import (
        SukukSignalGenerator, build_sukuk_numeric_block, RatesRegime
    )
    from src.analytics.sukuk_models import SukukAction, DataQuality as SukukDataQuality
    SUKUK_MODULE_AVAILABLE = True
except ImportError:
    SUKUK_MODULE_AVAILABLE = False
    logger.debug("Sukuk module not available")

# =============================================================================
# HELPER FUNCTIONS FOR DATA RESOLUTION
# =============================================================================

def _resolve_yields_for_dashboard(generator, bond_intelligence, sample_signal):
    """
    Always return real yields for the dashboard/AI context.
    Priority:
      1) bond_intelligence.yields (if present)
      2) generator.get_treasury_yields() (authoritative)
      3) sample_signal.fundamental (last resort, but NEVER invent 2Y/spread as 0)
    """
    from types import SimpleNamespace

    # 1) bond_intelligence yields (best source)
    try:
        if bond_intelligence and getattr(bond_intelligence, "yields", None):
            y = bond_intelligence.yields
            # Ensure it has the key fields
            if getattr(y, 'yield_10y', None) and getattr(y, 'yield_2y', None):
                return y
    except Exception:
        pass

    # 2) generator.get_treasury_yields() (authoritative fallback)
    try:
        if generator:
            y = generator.get_treasury_yields()
            if y and getattr(y, "yield_10y", 0) and getattr(y, "yield_2y", 0):
                return y
    except Exception:
        pass

    # 3) last resort: sample_signal fundamental (NO ZEROS)
    if sample_signal and hasattr(sample_signal, "fundamental"):
        f = sample_signal.fundamental
        yield_10y = float(getattr(f, "yield_10y", 0) or 0)
        yield_30y = float(getattr(f, "yield_30y", 0) or 0)

        # If we don't have 2Y, set to None - DO NOT fabricate
        return SimpleNamespace(
            yield_10y=yield_10y,
            yield_30y=yield_30y,
            yield_2y=None,  # Explicitly None, not 0
            spread_10y_2y=None,  # Explicitly None, not 0
            sources={"fallback": "sample_signal.fundamental (2Y/spread unavailable)"},
            data_quality="DEGRADED",
            warnings=["2Y and 10Y-2Y spread unavailable in fallback mode"],
            yield_2y_date="",
            yield_10y_date="",
            yield_30y_date="",
            spread_10y_2y_date="",
        )

    return None


def _extract_first_fenced_block(text: str) -> str:
    """Extract the first fenced code block from text for sanitizer restoration."""
    if not text:
        return None
    m = re.search(r"```[\s\S]*?```", text)
    return m.group(0) if m else None


# =============================================================================
# FILE-BASED CACHE FOR PERSISTENCE ACROSS BROWSER REFRESHES
# =============================================================================

CACHE_DIR = Path("data/cache")
BOND_CACHE_FILE = CACHE_DIR / "bond_analysis_cache.pkl"

def _serialize_signal(sig) -> dict:
    """Convert a BondSignal object to a picklable dictionary."""
    try:
        return {
            'ticker': sig.ticker,
            'name': sig.name,
            'current_price': sig.current_price,
            'previous_close': sig.previous_close,
            'day_change_pct': sig.day_change_pct,
            'volume': sig.volume,
            'avg_volume': sig.avg_volume,
            'composite_score': sig.composite_score,
            'signal_value': sig.signal.value if sig.signal else 'HOLD',
            'confidence': sig.confidence,
            'target_price': sig.target_price,
            'upside_pct': sig.upside_pct,
            'stop_loss': sig.stop_loss,
            'bull_case': list(sig.bull_case) if sig.bull_case else [],
            'bear_case': list(sig.bear_case) if sig.bear_case else [],
            'key_levels': dict(sig.key_levels) if sig.key_levels else {},
            'analysis_time': sig.analysis_time.isoformat() if sig.analysis_time else None,
            # Technical
            'technical': {
                'current_price': sig.technical.current_price if sig.technical else 0,
                'rsi_14': sig.technical.rsi_14 if sig.technical else None,
                'rsi_signal': sig.technical.rsi_signal if sig.technical else '',
                'macd_crossover': sig.technical.macd_crossover if sig.technical else '',
                'bb_position': sig.technical.bb_position if sig.technical else '',
                'trend_value': sig.technical.trend.value if sig.technical and sig.technical.trend else 'Neutral',
                'sma_50': sig.technical.sma_50 if sig.technical else None,
                'sma_200': sig.technical.sma_200 if sig.technical else None,
                'vwap': sig.technical.vwap if sig.technical else None,
                'support_1': sig.technical.support_1 if sig.technical else None,
                'technical_score': sig.technical.technical_score if sig.technical else None,
            } if sig.technical else {},
            # Fundamental
            'fundamental': {
                'yield_10y': sig.fundamental.yield_10y if sig.fundamental else 0,
                'yield_30y': sig.fundamental.yield_30y if sig.fundamental else 0,
                'price_target_12m': sig.fundamental.price_target_12m if sig.fundamental else None,
                'upside_pct': sig.fundamental.upside_pct if sig.fundamental else None,
                'fundamental_score': sig.fundamental.fundamental_score if sig.fundamental else None,
                'modified_duration': sig.fundamental.modified_duration if sig.fundamental else 0,
            } if sig.fundamental else {},
            # Flow
            'flow': {
                'flow_score': sig.flow.flow_score if sig.flow else None,
                'auction_signal': sig.flow.auction_signal if sig.flow else '',
            } if sig.flow else {},
            # Macro
            'macro': {
                'macro_score': sig.macro.macro_score if sig.macro else None,
                'fed_policy_signal': sig.macro.fed_policy_signal if sig.macro else '',
            } if sig.macro else {},
            # Sentiment
            'sentiment': {
                'sentiment_score': sig.sentiment.sentiment_score if sig.sentiment else None,
                'news_sentiment_label': sig.sentiment.news_sentiment_label if sig.sentiment else '',
            } if sig.sentiment else {},
        }
    except Exception as e:
        logger.error(f"Error serializing signal: {e}")
        return {}

def _deserialize_signal(data: dict):
    """Convert a dictionary back to a signal-like object for display."""
    from types import SimpleNamespace

    # Create nested namespaces for technical, fundamental, etc.
    technical = SimpleNamespace(**data.get('technical', {}))
    # Add trend as namespace
    technical.trend = SimpleNamespace(value=data.get('technical', {}).get('trend_value', 'Neutral'))

    fundamental = SimpleNamespace(**data.get('fundamental', {}))
    flow = SimpleNamespace(**data.get('flow', {}))
    macro = SimpleNamespace(**data.get('macro', {}))
    sentiment = SimpleNamespace(**data.get('sentiment', {}))

    # Create signal namespace
    signal_obj = SimpleNamespace(value=data.get('signal_value', 'HOLD'))

    # Parse analysis_time
    analysis_time = None
    if data.get('analysis_time'):
        try:
            analysis_time = datetime.fromisoformat(data['analysis_time'])
        except:
            analysis_time = datetime.now()

    return SimpleNamespace(
        ticker=data.get('ticker', ''),
        name=data.get('name', ''),
        current_price=data.get('current_price', 0),
        previous_close=data.get('previous_close', 0),
        day_change_pct=data.get('day_change_pct', 0),
        volume=data.get('volume', 0),
        avg_volume=data.get('avg_volume', 0),
        composite_score=data.get('composite_score', 50),
        signal=signal_obj,
        confidence=data.get('confidence', 'Medium'),
        target_price=data.get('target_price', 0),
        upside_pct=data.get('upside_pct', 0),
        stop_loss=data.get('stop_loss'),
        bull_case=data.get('bull_case', []),
        bear_case=data.get('bear_case', []),
        key_levels=data.get('key_levels', {}),
        analysis_time=analysis_time,
        technical=technical,
        fundamental=fundamental,
        flow=flow,
        macro=macro,
        sentiment=sentiment,
    )

def save_bond_cache(data: dict) -> bool:
    """Save ALL bond analysis data to JSON file for persistence."""
    import json

    json_file = CACHE_DIR / "bond_analysis_cache.json"

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        serialized_data = {
            'cache_time': datetime.now().isoformat(),
            'signals': {},
            'yields': None,
            'institutional': None,
            'news_result': None,
            'bond_intelligence': None,
        }

        # Serialize each signal - use safe attribute access
        if 'signals' in data and data['signals']:
            for ticker, sig in data['signals'].items():
                try:
                    # Get signal value safely
                    signal_value = 'HOLD'
                    if hasattr(sig, 'signal') and sig.signal:
                        signal_value = str(getattr(sig.signal, 'value', 'HOLD'))

                    # Get technical data safely
                    tech_score = 50
                    rsi_14 = None
                    trend_value = 'Neutral'
                    macd = ''
                    bb_position = ''
                    sma_50 = None
                    sma_200 = None
                    vwap = None
                    support_1 = None
                    rsi_signal = ''

                    tech = getattr(sig, 'technical', None)
                    if tech:
                        tech_score = int(getattr(tech, 'technical_score', 50) or 50)
                        rsi_14 = getattr(tech, 'rsi_14', None)
                        if rsi_14 is not None:
                            rsi_14 = float(rsi_14)
                        if hasattr(tech, 'trend') and tech.trend:
                            trend_value = str(getattr(tech.trend, 'value', 'Neutral'))
                        macd = str(getattr(tech, 'macd_crossover', '') or '')
                        bb_position = str(getattr(tech, 'bb_position', '') or '')
                        rsi_signal = str(getattr(tech, 'rsi_signal', '') or '')
                        sma_50 = float(getattr(tech, 'sma_50', 0) or 0) if getattr(tech, 'sma_50', None) else None
                        sma_200 = float(getattr(tech, 'sma_200', 0) or 0) if getattr(tech, 'sma_200', None) else None
                        vwap = float(getattr(tech, 'vwap', 0) or 0) if getattr(tech, 'vwap', None) else None
                        support_1 = float(getattr(tech, 'support_1', 0) or 0) if getattr(tech, 'support_1', None) else None

                    # Get other scores safely
                    fund = getattr(sig, 'fundamental', None)
                    fund_score = int(getattr(fund, 'fundamental_score', 50) or 50) if fund else 50
                    duration = float(getattr(fund, 'modified_duration', 0) or 0) if fund else 0
                    fund_yield_10y = float(getattr(fund, 'yield_10y', 0) or 0) if fund else 0
                    fund_yield_30y = float(getattr(fund, 'yield_30y', 0) or 0) if fund else 0

                    flow = getattr(sig, 'flow', None)
                    flow_score = int(getattr(flow, 'flow_score', 50) or 50) if flow else 50

                    macro = getattr(sig, 'macro', None)
                    macro_score = int(getattr(macro, 'macro_score', 50) or 50) if macro else 50

                    sent = getattr(sig, 'sentiment', None)
                    sent_score = int(getattr(sent, 'sentiment_score', 50) or 50) if sent else 50

                    serialized_data['signals'][ticker] = {
                        'ticker': str(ticker),
                        'name': str(getattr(sig, 'name', '') or ''),
                        'current_price': float(getattr(sig, 'current_price', 0) or 0),
                        'target_price': float(getattr(sig, 'target_price', 0) or 0),
                        'upside_pct': float(getattr(sig, 'upside_pct', 0) or 0),
                        'composite_score': int(getattr(sig, 'composite_score', 50) or 50),
                        'signal_value': signal_value,
                        'confidence': str(getattr(sig, 'confidence', 'Medium') or 'Medium'),
                        'bull_case': [str(x) for x in (getattr(sig, 'bull_case', []) or [])],
                        'bear_case': [str(x) for x in (getattr(sig, 'bear_case', []) or [])],
                        'technical_score': tech_score,
                        'fundamental_score': fund_score,
                        'flow_score': flow_score,
                        'macro_score': macro_score,
                        'sentiment_score': sent_score,
                        'rsi_14': rsi_14,
                        'rsi_signal': rsi_signal,
                        'macd_crossover': macd,
                        'bb_position': bb_position,
                        'trend_value': trend_value,
                        'sma_50': sma_50,
                        'sma_200': sma_200,
                        'vwap': vwap,
                        'support_1': support_1,
                        'modified_duration': duration,
                        'yield_10y': fund_yield_10y,
                        'yield_30y': fund_yield_30y,
                    }
                except Exception as e:
                    logger.error(f"Error serializing {ticker}: {e}")

        # Serialize yields (include audit metadata)
        if 'yields' in data and data['yields']:
            try:
                y = data['yields']
                serialized_data['yields'] = {
                    'yield_30y': float(getattr(y, 'yield_30y', 0) or 0),
                    'yield_10y': float(getattr(y, 'yield_10y', 0) or 0),
                    'yield_5y': float(getattr(y, 'yield_5y', 0) or 0),
                    'yield_2y': float(getattr(y, 'yield_2y', 0) or 0),
                    'yield_3m': float(getattr(y, 'yield_3m', 0) or 0),
                    'spread_10y_2y': float(getattr(y, 'spread_10y_2y', 0) or 0),
                    'spread_10y_3m': float(getattr(y, 'spread_10y_3m', 0) or 0),

                    # As-of dates for each yield
                    'yield_2y_date': str(getattr(y, 'yield_2y_date', '') or ''),
                    'yield_10y_date': str(getattr(y, 'yield_10y_date', '') or ''),
                    'yield_30y_date': str(getattr(y, 'yield_30y_date', '') or ''),
                    'spread_10y_2y_date': str(getattr(y, 'spread_10y_2y_date', '') or ''),

                    # Audit fields for data provenance
                    'fetched_at': str(getattr(y, 'fetched_at', '') or ''),
                    'data_quality': str(getattr(y, 'data_quality', 'OK') or 'OK'),
                    'sources': dict(getattr(y, 'sources', {}) or {}),
                    'warnings': list(getattr(y, 'warnings', []) or []),
                }
            except Exception as e:
                logger.error(f"Error serializing yields: {e}")

        # Serialize institutional enhancement
        if 'institutional' in data and data['institutional']:
            try:
                inst = data['institutional']
                serialized_data['institutional'] = {
                    'auction_demand_score': int(getattr(inst, 'auction_demand_score', 50) or 50),
                    'vol_regime_score': int(getattr(inst, 'vol_regime_score', 50) or 50),
                    'term_premium_score': int(getattr(inst, 'term_premium_score', 50) or 50),
                    'curve_trade_score': int(getattr(inst, 'curve_trade_score', 50) or 50),
                    'futures_momentum_score': int(getattr(inst, 'futures_momentum_score', 50) or 50),
                    'institutional_adjustment': int(getattr(inst, 'institutional_adjustment', 0) or 0),
                    'key_insights': list(getattr(inst, 'key_insights', []) or []),
                    'risks': list(getattr(inst, 'risks', []) or []),
                    'opportunities': list(getattr(inst, 'opportunities', []) or []),
                }
                # Term premium details
                tp = getattr(inst, 'term_premium', None)
                if tp:
                    serialized_data['institutional']['term_premium_value'] = float(getattr(tp, 'term_premium', 0) or 0)
                    serialized_data['institutional']['term_premium_interpretation'] = str(getattr(tp, 'interpretation', '') or '')
                # Vol regime details
                vr = getattr(inst, 'vol_regime', None)
                if vr:
                    serialized_data['institutional']['move_index'] = float(getattr(vr, 'move_index', 0) or 0)
                    serialized_data['institutional']['vol_regime'] = str(getattr(vr, 'regime', '') or '')
                    serialized_data['institutional']['vol_trading_implication'] = str(getattr(vr, 'trading_implication', '') or '')
            except Exception as e:
                logger.error(f"Error serializing institutional: {e}")

        # Serialize news result
        if 'news_result' in data and data['news_result']:
            try:
                news = data['news_result']
                # Handle overall_score - might be 0-1 or 0-100
                raw_score = getattr(news, 'overall_score', 0.5)
                if raw_score is not None:
                    if isinstance(raw_score, float) and raw_score <= 1:
                        # It's in 0-1 range, store as-is (will be used with .0% format)
                        overall_score = float(raw_score)
                    else:
                        # It's 0-100, convert to 0-1
                        overall_score = float(raw_score) / 100
                else:
                    overall_score = 0.5

                # Get original sentiment
                original_sentiment = str(getattr(news, 'overall_sentiment', '') or '')

                serialized_data['news_result'] = {
                    'overall_score': overall_score,  # Store as 0-1 range
                    'overall_sentiment': original_sentiment,  # Store original sentiment
                    'bullish_count': int(getattr(news, 'bullish_count', 0) or 0),
                    'bearish_count': int(getattr(news, 'bearish_count', 0) or 0),
                    'neutral_count': int(getattr(news, 'neutral_count', 0) or 0),
                    'themes': list(getattr(news, 'themes', []) or []),
                    'articles': [],
                }

                logger.info(f"Saving news: score={overall_score:.2f}, sentiment={original_sentiment}, bullish={serialized_data['news_result']['bullish_count']}, bearish={serialized_data['news_result']['bearish_count']}")

                # Serialize articles
                for art in (getattr(news, 'articles', []) or [])[:10]:  # Max 10 articles
                    serialized_data['news_result']['articles'].append({
                        'title': str(getattr(art, 'title', '') or ''),
                        'source': str(getattr(art, 'source', '') or ''),
                        'sentiment': str(getattr(art, 'sentiment', '') or ''),
                        'score': int(getattr(art, 'score', 0) or 0),
                        'url': str(getattr(art, 'url', '') or ''),
                        'category': str(getattr(art, 'category', 'General') or 'General'),
                        'description': str(getattr(art, 'description', '') or ''),
                    })
            except Exception as e:
                logger.error(f"Error serializing news: {e}")

        # Serialize bond intelligence
        if 'bond_intelligence' in data and data['bond_intelligence']:
            try:
                bi = data['bond_intelligence']
                serialized_data['bond_intelligence'] = {}

                # Fed policy
                fp = getattr(bi, 'fed_policy', None)
                if fp:
                    serialized_data['bond_intelligence']['fed_policy'] = {
                        'current_rate_lower': float(getattr(fp, 'current_rate_lower', 0) or 0),
                        'current_rate_upper': float(getattr(fp, 'current_rate_upper', 0) or 0),
                        'effective_rate': float(getattr(fp, 'effective_rate', 0) or 0),
                        'last_decision': str(getattr(fp, 'last_decision', '') or ''),
                        'data_source': str(getattr(fp, 'data_source', '') or ''),
                    }

                # Rate probabilities
                rp = getattr(bi, 'rate_probabilities', None)
                if rp:
                    serialized_data['bond_intelligence']['rate_probabilities'] = {
                        'next_meeting_probs': dict(getattr(rp, 'next_meeting_probs', {}) or {}),
                        'cuts_priced_in_2025': float(getattr(rp, 'cuts_priced_in_2025', 0) or 0),
                        'data_source': str(getattr(rp, 'data_source', '') or ''),
                    }

                # Yields
                yld = getattr(bi, 'yields', None)
                if yld:
                    serialized_data['bond_intelligence']['yields'] = {
                        'yield_10y': float(getattr(yld, 'yield_10y', 0) or 0),
                        'yield_30y': float(getattr(yld, 'yield_30y', 0) or 0),
                        'yield_2y': float(getattr(yld, 'yield_2y', 0) or 0),
                        'spread_10y_2y': float(getattr(yld, 'spread_10y_2y', 0) or 0),
                        'spread_10y_fed': float(getattr(yld, 'spread_10y_fed', 0) or 0),
                    }

                # AI context
                serialized_data['bond_intelligence']['ai_context'] = str(getattr(bi, 'ai_context', '') or '')
            except Exception as e:
                logger.error(f"Error serializing bond_intelligence: {e}")

        # Write JSON file
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serialized_data, f, indent=2)

        file_size = json_file.stat().st_size
        n_signals = len(serialized_data.get('signals', {}))
        has_inst = serialized_data.get('institutional') is not None
        has_news = serialized_data.get('news_result') is not None
        logger.info(f"Saved bond cache: {json_file} ({file_size} bytes, {n_signals} signals, inst={has_inst}, news={has_news})")

        return file_size > 50

    except Exception as e:
        logger.error(f"Failed to save bond cache: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def load_bond_cache() -> dict:
    """Load bond analysis data from JSON file."""
    import json
    from types import SimpleNamespace

    json_file = CACHE_DIR / "bond_analysis_cache.json"

    try:
        if not json_file.exists():
            logger.debug("No cache file exists")
            return {}

        if json_file.stat().st_size < 50:
            logger.warning("Cache file too small, deleting")
            json_file.unlink()
            return {}

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data or 'signals' not in data or not data['signals']:
            logger.warning("Cache has no signals")
            return {}

        # Convert signals back to namespace objects
        signals = {}
        for ticker, s in data['signals'].items():
            tech = SimpleNamespace(
                current_price=s.get('current_price', 0),
                rsi_14=s.get('rsi_14'),
                rsi_signal=s.get('rsi_signal', ''),
                macd_crossover=s.get('macd_crossover', ''),
                bb_position=s.get('bb_position', ''),
                trend=SimpleNamespace(value=s.get('trend_value', 'Neutral')),
                sma_50=s.get('sma_50'),
                sma_200=s.get('sma_200'),
                vwap=s.get('vwap'),
                support_1=s.get('support_1'),
                technical_score=s.get('technical_score', 50),
            )

            fund = SimpleNamespace(
                fundamental_score=s.get('fundamental_score', 50),
                modified_duration=s.get('modified_duration', 0),
                yield_10y=s.get('yield_10y', 0),
                yield_30y=s.get('yield_30y', 0),
                price_target_12m=s.get('target_price', 0),
                upside_pct=s.get('upside_pct', 0),
            )

            flow = SimpleNamespace(flow_score=s.get('flow_score', 50), auction_signal='')
            macro = SimpleNamespace(macro_score=s.get('macro_score', 50), fed_policy_signal='')
            sentiment = SimpleNamespace(sentiment_score=s.get('sentiment_score', 50), news_sentiment_label='')

            signals[ticker] = SimpleNamespace(
                ticker=s.get('ticker', ticker),
                name=s.get('name', ''),
                current_price=s.get('current_price', 0),
                previous_close=0,
                day_change_pct=0,
                volume=0,
                avg_volume=0,
                composite_score=s.get('composite_score', 50),
                signal=SimpleNamespace(value=s.get('signal_value', 'HOLD')),
                confidence=s.get('confidence', 'Medium'),
                target_price=s.get('target_price', 0),
                upside_pct=s.get('upside_pct', 0),
                stop_loss=None,
                bull_case=s.get('bull_case', []),
                bear_case=s.get('bear_case', []),
                key_levels={},
                analysis_time=datetime.now(),
                technical=tech,
                fundamental=fund,
                flow=flow,
                macro=macro,
                sentiment=sentiment,
            )

        # Convert yields (restore audit metadata)
        yields = None
        if data.get('yields'):
            y = data['yields']
            yields = SimpleNamespace(
                yield_30y=y.get('yield_30y', 0),
                yield_10y=y.get('yield_10y', 0),
                yield_5y=y.get('yield_5y', 0),
                yield_2y=y.get('yield_2y', 0),
                yield_3m=y.get('yield_3m', 0),
                spread_10y_2y=y.get('spread_10y_2y', 0),
                spread_10y_3m=y.get('spread_10y_3m', 0),

                # As-of dates
                yield_2y_date=y.get('yield_2y_date', ''),
                yield_10y_date=y.get('yield_10y_date', ''),
                yield_30y_date=y.get('yield_30y_date', ''),
                spread_10y_2y_date=y.get('spread_10y_2y_date', ''),

                # Audit metadata
                fetched_at=y.get('fetched_at', ''),
                data_quality=y.get('data_quality', 'OK'),
                sources=y.get('sources', {}) or {},
                warnings=y.get('warnings', []) or [],
            )

        # Convert institutional
        institutional = None
        if data.get('institutional'):
            inst = data['institutional']
            # Create term_premium namespace
            term_premium_ns = None
            if inst.get('term_premium_value') is not None:
                term_premium_ns = SimpleNamespace(
                    term_premium=inst.get('term_premium_value', 0),
                    interpretation=inst.get('term_premium_interpretation', ''),
                    nominal_yield=0,
                    expected_rate=0,
                    percentile=50,
                )
            # Create vol_regime namespace
            vol_regime_ns = SimpleNamespace(
                move_index=inst.get('move_index', 0),
                regime=inst.get('vol_regime', 'NORMAL'),
                trading_implication=inst.get('vol_trading_implication', ''),
            )
            institutional = SimpleNamespace(
                auction_demand_score=inst.get('auction_demand_score', 50),
                vol_regime_score=inst.get('vol_regime_score', 50),
                term_premium_score=inst.get('term_premium_score', 50),
                curve_trade_score=inst.get('curve_trade_score', 50),
                futures_momentum_score=inst.get('futures_momentum_score', 50),
                institutional_adjustment=inst.get('institutional_adjustment', 0),
                key_insights=inst.get('key_insights', []),
                risks=inst.get('risks', []),
                opportunities=inst.get('opportunities', []),
                term_premium=term_premium_ns,
                vol_regime=vol_regime_ns,
                # Additional fields for render functions
                futures=[],
                curve_trades=[],
                recent_auctions=[],
                upcoming_auctions=[],
            )

        # Convert news_result
        news_result = None
        if data.get('news_result'):
            nr = data['news_result']
            articles = []
            for art in nr.get('articles', []):
                articles.append(SimpleNamespace(
                    title=art.get('title', ''),
                    source=art.get('source', ''),
                    sentiment=art.get('sentiment', ''),
                    score=art.get('score', 0),
                    url=art.get('url', ''),
                    published_at=None,  # Date not preserved in cache
                    category=art.get('category', 'General'),
                    description=art.get('description', ''),
                ))

            # Get saved score (already in 0-1 range)
            overall_score = nr.get('overall_score', 0.5)

            # Use saved sentiment, or derive from bullish/bearish counts as fallback
            overall_sentiment = nr.get('overall_sentiment', '')
            if not overall_sentiment:
                # Derive from counts instead of score
                bullish = nr.get('bullish_count', 0)
                bearish = nr.get('bearish_count', 0)
                if bullish > bearish:
                    overall_sentiment = "BULLISH"
                elif bearish > bullish:
                    overall_sentiment = "BEARISH"
                else:
                    overall_sentiment = "NEUTRAL"

            logger.info(f"Loading news: score={overall_score:.2f}, sentiment={overall_sentiment}, bullish={nr.get('bullish_count', 0)}, bearish={nr.get('bearish_count', 0)}")

            news_result = SimpleNamespace(
                overall_score=overall_score,  # Already in 0-1 range
                overall_sentiment=overall_sentiment,
                bullish_count=nr.get('bullish_count', 0),
                bearish_count=nr.get('bearish_count', 0),
                neutral_count=nr.get('neutral_count', 0),
                themes=nr.get('themes', []),
                articles=articles,
            )

        # Convert bond_intelligence
        bond_intelligence = None
        if data.get('bond_intelligence'):
            bi = data['bond_intelligence']

            fed_policy = None
            if bi.get('fed_policy'):
                fp = bi['fed_policy']
                fed_policy = SimpleNamespace(
                    current_rate_lower=fp.get('current_rate_lower', 0),
                    current_rate_upper=fp.get('current_rate_upper', 0),
                    effective_rate=fp.get('effective_rate', 0),
                    last_decision=fp.get('last_decision', ''),
                    data_source=fp.get('data_source', ''),
                    recent_fed_news=[],
                )

            rate_probabilities = None
            if bi.get('rate_probabilities'):
                rp = bi['rate_probabilities']
                rate_probabilities = SimpleNamespace(
                    next_meeting_probs=rp.get('next_meeting_probs', {}),
                    cuts_priced_in_2025=rp.get('cuts_priced_in_2025', 0),
                    data_source=rp.get('data_source', ''),
                )

            bi_yields = None
            if bi.get('yields'):
                yld = bi['yields']
                bi_yields = SimpleNamespace(
                    yield_10y=yld.get('yield_10y', 0),
                    yield_30y=yld.get('yield_30y', 0),
                    yield_2y=yld.get('yield_2y', 0),
                    yield_5y=0,
                    yield_3m=0,
                    spread_10y_2y=yld.get('spread_10y_2y', 0),
                    spread_10y_3m=0,
                    spread_10y_fed=yld.get('spread_10y_fed', 0),
                )

            bond_intelligence = SimpleNamespace(
                fed_policy=fed_policy,
                rate_probabilities=rate_probabilities,
                yields=bi_yields,
                ai_context=bi.get('ai_context', ''),
            )

        # Parse cache time
        cache_time = datetime.now()
        if data.get('cache_time'):
            try:
                cache_time = datetime.fromisoformat(data['cache_time'])
            except:
                pass

        age_mins = (datetime.now() - cache_time).seconds // 60
        has_inst = institutional is not None
        has_news = news_result is not None
        logger.info(f"Loaded bond cache: {len(signals)} signals, {age_mins}m old, inst={has_inst}, news={has_news}")

        return {
            'signals': signals,
            'yields': yields,
            'cache_time': cache_time,
            'yield_hist': pd.DataFrame(),
            'institutional': institutional,
            'news_result': news_result,
            'bond_intelligence': bond_intelligence,
            'economic_impact': None,
        }

    except Exception as e:
        logger.error(f"Failed to load bond cache: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return {}


def get_cache_info() -> tuple:
    """Get cache file info. Returns (exists, age_minutes, num_signals)."""
    import json

    json_file = CACHE_DIR / "bond_analysis_cache.json"

    try:
        if json_file.exists() and json_file.stat().st_size > 50:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if data and 'signals' in data and data['signals']:
                n_signals = len(data['signals'])
                cache_time_str = data.get('cache_time')

                if cache_time_str:
                    try:
                        cache_time = datetime.fromisoformat(cache_time_str)
                        age_mins = (datetime.now() - cache_time).seconds // 60
                        return True, age_mins, n_signals
                    except:
                        return True, 0, n_signals
    except:
        pass

    return False, 0, 0

# =============================================================================
# IMPORTS - All modules
# =============================================================================

# Institutional enhancement
try:
    from src.analytics.institutional_bond_analysis import (
        get_institutional_enhancement,
        render_institutional_section,
        render_institutional_details,
        InstitutionalEnhancement
    )
    INSTITUTIONAL_AVAILABLE = True
except ImportError as e:
    INSTITUTIONAL_AVAILABLE = False
    logger.debug(f"Institutional bond analysis not available: {e}")

# Bond news
try:
    from src.analytics.bond_news import (
        get_bond_news,
        BondNewsResult,
        BondNewsArticle
    )
    NEWS_AVAILABLE = True
except ImportError as e:
    NEWS_AVAILABLE = False
    logger.debug(f"Bond news module not available: {e}")

# Bond market intelligence (Fed policy, rates, term premium)
try:
    from src.analytics.bond_market_intelligence import (
        get_bond_market_intelligence,
        BondMarketIntelligence
    )
    INTELLIGENCE_AVAILABLE = True
except ImportError as e:
    INTELLIGENCE_AVAILABLE = False
    logger.debug(f"Bond market intelligence not available: {e}")

# AI Chat
try:
    from src.ai.chat import AlphaChat
    AI_CHAT_AVAILABLE = True
except ImportError as e:
    AI_CHAT_AVAILABLE = False
    logger.debug(f"AI Chat not available: {e}")


def render_bond_trading_tab():
    """Render the unified Bond Trading tab with all features."""

    st.markdown("### 🏦 Bond/Rates & Sukuk Trading")
    st.caption("Institutional-grade bond & sukuk analysis with news, sentiment, and AI advisor")

    # Check if institutional bond signal module is available
    try:
        from src.analytics.bond_signals_institutional import (
            InstitutionalBondSignalGenerator,
            BondSignal,
            get_institutional_bond_signals,
        )
        BOND_MODULE_AVAILABLE = True
    except ImportError as e:
        st.error(f"Institutional bond signals module not found: {e}")
        st.info("Make sure src/analytics/bond_signals_institutional.py exists")
        return

    # Initialize generator
    if 'bond_generator' not in st.session_state:
        st.session_state.bond_generator = InstitutionalBondSignalGenerator()

    generator = st.session_state.bond_generator

    # Initialize chat history for bonds
    if 'bond_chat_history' not in st.session_state:
        st.session_state.bond_chat_history = []

    # ==========================================================================
    # CHECK FOR CACHED DATA (Session State + File)
    # ==========================================================================

    # Check session state first
    has_session_cache = (
        'bond_signals' in st.session_state and
        'bond_yields' in st.session_state and
        st.session_state.bond_signals is not None and
        len(st.session_state.get('bond_signals', {})) > 0
    )

    # Check file cache
    file_cache_exists, file_cache_age, file_cache_signals = get_cache_info()

    logger.debug(f"Cache check: session={has_session_cache}, file={file_cache_exists} ({file_cache_signals} signals, {file_cache_age}m old)")

    # ==========================================================================
    # CONTROL BUTTONS - Two options: Load Last vs Run Fresh
    # ==========================================================================
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        load_last_clicked = st.button("📂 Load Last Analysis", key="load_cached_bonds", type="secondary")

    with col2:
        run_fresh_clicked = st.button("🔄 Run Fresh Analysis", key="load_fresh_bonds", type="primary")

    with col3:
        if has_session_cache and 'bond_load_time' in st.session_state:
            load_time = st.session_state.bond_load_time
            mins_ago = (datetime.now() - load_time).seconds // 60
            n_signals = len(st.session_state.get('bond_signals', {}))
            st.caption(f"✅ Session: {n_signals} signals, {mins_ago}m ago")
        elif file_cache_exists:
            st.caption(f"💾 File cache: {file_cache_signals} signals, {file_cache_age}m old")
        else:
            st.caption("📂 No cache | 🔄 Run Fresh to fetch data")

    # ==========================================================================
    # HANDLE BUTTON CLICKS
    # ==========================================================================

    if run_fresh_clicked:
        # Clear all session cache and fetch fresh
        for key in ['bond_signals', 'bond_yields', 'bond_yield_hist',
                    'institutional_enhancement', 'bond_news_result',
                    'bond_intelligence', 'bond_economic_impact']:
            st.session_state.pop(key, None)
        st.session_state.bond_force_fresh = True
        st.rerun()

    if load_last_clicked:
        # First try session cache
        if has_session_cache:
            st.toast("⚡ Using session cache")
            logger.info(f"Load Last: Using session cache with {len(st.session_state.bond_signals)} signals")
        # Then try file cache
        elif file_cache_exists:
            st.toast("💾 Loading from file cache...")
            file_data = load_bond_cache()

            if file_data and 'signals' in file_data:
                # Restore to session state
                st.session_state.bond_signals = file_data.get('signals', {})
                st.session_state.bond_yields = file_data.get('yields')
                st.session_state.bond_yield_hist = file_data.get('yield_hist', pd.DataFrame())
                st.session_state.institutional_enhancement = file_data.get('institutional')
                st.session_state.bond_news_result = file_data.get('news_result')
                st.session_state.bond_intelligence = file_data.get('bond_intelligence')
                st.session_state.bond_economic_impact = file_data.get('economic_impact')
                st.session_state.bond_load_time = file_data.get('cache_time', datetime.now())

                logger.info(f"Load Last: Restored {len(st.session_state.bond_signals)} signals from file cache")
                st.success(f"✅ Loaded {len(st.session_state.bond_signals)} signals from cache")
                has_session_cache = True  # Now we have data
                st.rerun()  # Refresh to show loaded data
            else:
                st.warning("⚠️ File cache corrupt or empty. Click '🔄 Run Fresh Analysis'.")
                _render_quick_summary()
                return
        else:
            st.warning("⚠️ No cached data available. Click '🔄 Run Fresh Analysis' to fetch data.")
            logger.warning("Load Last clicked but no cache available")
            _render_quick_summary()
            return

    # Check if we need to fetch fresh (from button click)
    need_fresh_fetch = st.session_state.pop('bond_force_fresh', False)

    # ==========================================================================
    # DETERMINE WHAT TO DO
    # ==========================================================================

    if has_session_cache and not need_fresh_fetch:
        # FAST PATH: Use session cache
        signals = st.session_state.bond_signals
        yields = st.session_state.bond_yields
        institutional = st.session_state.get('institutional_enhancement')
        news_result = st.session_state.get('bond_news_result')
        economic_impact = st.session_state.get('bond_economic_impact')
        bond_intelligence = st.session_state.get('bond_intelligence')
        yield_hist = st.session_state.get('bond_yield_hist', pd.DataFrame())

        sample_signal = list(signals.values())[0] if signals else None

    elif need_fresh_fetch:
        # SLOW PATH: Fetch fresh data from all sources
        with st.spinner("🔄 Fetching fresh data from all sources (yields, news, institutional, Fed policy)..."):
            try:
                # 1. Institutional enhancement (auction demand, vol regime, etc.)
                institutional = None
                auction_score = None
                if INSTITUTIONAL_AVAILABLE:
                    try:
                        institutional = get_institutional_enhancement()
                        auction_score = institutional.auction_demand_score
                    except Exception as e:
                        logger.warning(f"Could not get institutional data: {e}")

                # 2. Bond news with sentiment
                news_result = None
                if NEWS_AVAILABLE:
                    try:
                        news_result = get_bond_news(
                            max_articles=20,
                            days_back=3,
                            force_refresh=True,
                            analyze_sentiment=True
                        )
                    except Exception as e:
                        logger.warning(f"Could not fetch bond news: {e}")

                # 3. Bond market intelligence (Fed policy, rates, term premium)
                bond_intelligence = None
                rate_probs = None
                if INTELLIGENCE_AVAILABLE:
                    try:
                        bond_intelligence = get_bond_market_intelligence(force_refresh=True)
                        # Extract rate probabilities for signal generator
                        if bond_intelligence and bond_intelligence.rate_probabilities:
                            rate_probs = {
                                'hold': bond_intelligence.rate_probabilities.next_meeting_probs.get('hold', 0),
                                'cut': bond_intelligence.rate_probabilities.next_meeting_probs.get('cut', 0),
                                'cuts_priced_in_2025': bond_intelligence.rate_probabilities.cuts_priced_in_2025,
                            }
                    except Exception as e:
                        logger.warning(f"Could not get bond intelligence: {e}")

                # 4. Generate institutional-grade signals with ALL data
                signals = generator.generate_all_signals(
                    auction_score=auction_score,
                    news_score=news_result.overall_score if news_result else None,
                    news_bullish=news_result.bullish_count if news_result else 0,
                    news_bearish=news_result.bearish_count if news_result else 0,
                    news_themes=news_result.themes if news_result else [],
                    rate_probs=rate_probs,
                )

                # 5. Get yields from a sample signal or fetch separately
                sample_signal = list(signals.values())[0] if signals else None

                # Create yields object - NEVER use zeros as fallback
                yields = _resolve_yields_for_dashboard(generator, bond_intelligence, sample_signal)

                # Economic impact (optional)
                economic_impact = None

                # Initialize yield_hist as empty DataFrame
                yield_hist = pd.DataFrame()

                # Store ALL in session state for persistence
                st.session_state.bond_signals = signals
                st.session_state.bond_yields = yields
                st.session_state.bond_yield_hist = yield_hist
                st.session_state.institutional_enhancement = institutional
                st.session_state.bond_news_result = news_result
                st.session_state.bond_economic_impact = economic_impact
                st.session_state.bond_intelligence = bond_intelligence
                st.session_state.bond_load_time = datetime.now()

                # Also save to file cache for persistence across browser refreshes
                cache_data = {
                    'signals': signals,
                    'yields': yields,
                    'yield_hist': yield_hist,
                    'institutional': institutional,
                    'news_result': news_result,
                    'economic_impact': economic_impact,
                    'bond_intelligence': bond_intelligence,
                }
                if save_bond_cache(cache_data):
                    st.success("✅ Fresh data loaded and saved to cache!")
                else:
                    st.success("✅ Fresh institutional data loaded!")

            except Exception as e:
                st.error(f"Error loading bond data: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

        sample_signal = list(signals.values())[0] if signals else None

    else:
        # No data and no button clicked - show welcome screen
        st.info("👆 Click **'🔄 Run Fresh Analysis'** to fetch bond market data")
        _render_quick_summary()
        return

    # Get bond_intelligence from session if not in local scope
    if 'bond_intelligence' not in dir() or bond_intelligence is None:
        bond_intelligence = st.session_state.get('bond_intelligence')

    # ==========================================================================
    # FETCH MOVE INDEX (single authoritative source)
    # ==========================================================================
    move = None
    try:
        if generator and hasattr(generator, 'get_move_index'):
            move = generator.get_move_index()
            st.session_state.bond_move = move
    except Exception as e:
        logger.warning(f"MOVE fetch error: {e}")
        move = {"value": None, "quality": "ERROR", "warning": str(e)}

    # ==========================================================================
    # BUILD COMPLETE AI CONTEXT
    # ==========================================================================
    complete_ai_context = _build_complete_ai_context(
        yields, yield_hist, signals, sample_signal,
        economic_impact, institutional, news_result, generator,
        bond_intelligence,
        portfolio_weights=st.session_state.get('portfolio_weights'),
        move=move
    )
    st.session_state.bond_ai_context = complete_ai_context

    # ==========================================================================
    # MAIN LAYOUT - Analysis on left, AI Chat on right
    # ==========================================================================
    col_main, col_chat = st.columns([2, 1])

    with col_main:
        # SECTION 1: UNIFIED MARKET SUMMARY
        _render_unified_summary(yields, sample_signal, institutional, news_result, bond_intelligence)

        # SECTION 2: TREASURY YIELDS & CURVE
        st.markdown("---")
        _render_yields_section(yields, yield_hist)

        # SECTION 3: INSTITUTIONAL INTELLIGENCE (with expanders)
        if institutional:
            st.markdown("---")
            st.markdown("### 🏦 Institutional Intelligence")
            render_institutional_section(institutional)
            render_institutional_details(institutional)

        # SECTION 4: NEWS (in expander, same style as institutional)
        if news_result and news_result.articles:
            _render_news_expander(news_result)
        elif NEWS_AVAILABLE:
            with st.expander("📰 Bond Market News (No articles found)"):
                st.warning("No bond news articles found. Check if NEWSAPI_KEY is set in your .env file.")
                st.caption("Expected: NEWSAPI_KEY=your_api_key_here")
        else:
            with st.expander("📰 Bond Market News (Module not available)"):
                st.info("Bond news module not loaded. Make sure src/analytics/bond_news.py exists.")
                st.caption("Copy bond_news.py to src/analytics/ folder")

        # SECTION 5: BOND ETF SIGNALS
        st.markdown("---")
        _render_signals_section(signals, institutional)

        # SECTION 6: AI PREDICTIONS (NEW)
        st.markdown("---")
        _render_ai_predictions_section()

        # SECTION 7: USD SUKUK (NEW)
        if SUKUK_MODULE_AVAILABLE:
            st.markdown("---")
            _render_sukuk_section(yields)

        # SECTION 8: DETAILED ANALYSIS
        st.markdown("---")
        _render_detailed_analysis(signals, institutional)

    # ==========================================================================
    # AI CHAT COLUMN
    # ==========================================================================
    with col_chat:
        _render_ai_chat(complete_ai_context)

    # ==========================================================================
    # EDUCATION & FULL AI CONTEXT
    # ==========================================================================
    st.markdown("---")
    with st.expander("🤖 Complete AI Context"):
        st.caption("All data fed to AI for analysis")
        st.code(complete_ai_context, language=None)

    with st.expander("📚 Bond & Sukuk Trading Education"):
        _render_education()


def _render_quick_summary():
    """Show quick summary before full analysis."""
    st.markdown("### Quick Treasury Check")

    try:
        import yfinance as yf
        tickers = ['^TNX', '^TYX', '^IRX']
        data = yf.download(tickers, period='2d', progress=False)

        if not data.empty:
            close = data['Close'].iloc[-1]
            prev = data['Close'].iloc[-2] if len(data) > 1 else close

            col1, col2, col3 = st.columns(3)
            with col1:
                y10 = close.get('^TNX', 0)
                chg = y10 - prev.get('^TNX', y10)
                st.metric("10-Year Yield", f"{y10:.2f}%", f"{chg:+.2f}%")
            with col2:
                y30 = close.get('^TYX', 0)
                chg = y30 - prev.get('^TYX', y30)
                st.metric("30-Year Yield", f"{y30:.2f}%", f"{chg:+.2f}%")
            with col3:
                y2 = close.get('^IRX', 0)
                spread = y10 - y2
                st.metric("10Y-2Y Spread", f"{spread:+.2f}%",
                         "Inverted" if spread < 0 else "Normal",
                         delta_color="inverse" if spread < 0 else "normal")
    except:
        st.info("Load full analysis for detailed data")


def _render_unified_summary(yields, sample_signal, institutional, news_result, bond_intelligence=None):
    """Render unified market summary with all inputs."""

    st.markdown("### 📋 Market Summary")

    # Calculate adjusted score
    base_score = sample_signal.composite_score if sample_signal else 50
    adjusted_score = base_score

    adjustments = []

    # Institutional adjustment (already -15 to +15)
    if institutional:
        adjusted_score += institutional.institutional_adjustment
        if institutional.institutional_adjustment != 0:
            adjustments.append(f"Institutional: {institutional.institutional_adjustment:+d}")

    # News sentiment adjustment - ±20 range
    if news_result and news_result.overall_score != 0.5:
        # 72% bullish = (0.72 - 0.5) * 40 = +8.8 → +9
        news_adj = int((news_result.overall_score - 0.5) * 40)
        news_adj = max(-20, min(20, news_adj))
        adjusted_score += news_adj
        if news_adj != 0:
            adjustments.append(f"News: {news_adj:+d}")

    # Bond intelligence adjustment (term premium, Fed outlook)
    if bond_intelligence and hasattr(bond_intelligence, 'yields'):
        intel_adj = 0

        # If 10Y trading significantly above Fed funds, bonds may be cheap (bullish)
        if bond_intelligence.yields.spread_10y_fed > 0.5:
            intel_adj += 5  # Bullish - high term premium
        elif bond_intelligence.yields.spread_10y_fed < -0.3:
            intel_adj -= 5  # Bearish - cuts heavily priced in

        # If curve deeply inverted, eventual rally expected (bullish long-term)
        if bond_intelligence.yields.spread_10y_2y < -0.5:
            intel_adj += 3  # Recession signal - flight to safety coming

        if intel_adj != 0:
            adjusted_score += intel_adj
            adjustments.append(f"TermPremium: {intel_adj:+d}")

    adjusted_score = max(0, min(100, adjusted_score))

    # Determine stance
    if adjusted_score >= 65:
        stance, stance_emoji, stance_color = "BULLISH", "🟢", "green"
    elif adjusted_score <= 35:
        stance, stance_emoji, stance_color = "BEARISH", "🔴", "red"
    else:
        stance, stance_emoji, stance_color = "NEUTRAL", "🟡", "orange"

    # Build summary
    analysis_parts = [f"**Overall Assessment: {stance_emoji} {stance}** (Score: {adjusted_score}/100)"]

    if adjustments:
        analysis_parts.append(f"*Base: {base_score} | Adjustments: {', '.join(adjustments)}*")

    analysis_parts.append("")

    # Yield trend from technical analysis
    if sample_signal and sample_signal.technical and sample_signal.technical.trend:
        trend = sample_signal.technical.trend.value.lower().replace('_', ' ')
        if 'up' in trend:
            analysis_parts.append(f"📈 **Price trend {trend}** - Bullish momentum")
        elif 'down' in trend:
            analysis_parts.append(f"📉 **Price trend {trend}** - Bearish momentum")
        else:
            analysis_parts.append(f"➡️ **Price stable**")

    # Curve from bond intelligence if available
    if bond_intelligence and hasattr(bond_intelligence, 'yields') and bond_intelligence.yields:
        spread = bond_intelligence.yields.spread_10y_2y
        if spread < 0:
            analysis_parts.append(f"⚠️ **Curve inverted** ({spread:+.2f}%) - Recession signal")
        elif spread > 0.5:
            analysis_parts.append(f"📈 **Curve steep** ({spread:+.2f}%)")

    # Institutional
    if institutional:
        vol = institutional.vol_regime
        if vol.regime in ["ELEVATED", "STRESSED"]:
            analysis_parts.append(f"⚠️ **High volatility** (MOVE: {vol.move_index:.0f})")
        elif vol.regime == "LOW":
            analysis_parts.append(f"✅ **Low volatility** (MOVE: {vol.move_index:.0f})")

    # News sentiment
    if news_result:
        sent_emoji = "🟢" if news_result.overall_sentiment == "BULLISH" else "🔴" if news_result.overall_sentiment == "BEARISH" else "🟡"
        analysis_parts.append(f"📰 **News Sentiment:** {sent_emoji} {news_result.overall_sentiment} ({news_result.overall_score:.0%})")
        if news_result.themes:
            analysis_parts.append(f"   Key themes: {', '.join(news_result.themes[:3])}")

    # Recommendation
    analysis_parts.append("")
    if adjusted_score >= 65:
        analysis_parts.append("**💡 Recommendation:** Favorable for long bonds. Consider TLT, ZROZ.")
    elif adjusted_score <= 35:
        analysis_parts.append("**💡 Recommendation:** Avoid long bonds. Consider SHY or TBT.")
    else:
        analysis_parts.append("**💡 Recommendation:** Mixed signals - maintain current allocation.")

    st.info("\n\n".join(analysis_parts))


def _render_news_expander(news_result: 'BondNewsResult'):
    """Render bond news in an expander, similar to Deep Dive style."""

    # Summary bar outside expander
    sent_emoji = "🟢" if news_result.overall_sentiment == "BULLISH" else "🔴" if news_result.overall_sentiment == "BEARISH" else "🟡"
    themes_text = f" | Themes: {', '.join(news_result.themes[:3])}" if news_result.themes else ""
    sources_text = f" | Sources: {news_result.sources_succeeded}/{news_result.sources_tried}" if hasattr(news_result, 'sources_tried') else ""

    with st.expander(f"📰 Bond Market News ({len(news_result.articles)} articles) - {sent_emoji} {news_result.overall_sentiment} ({news_result.overall_score:.0%}){themes_text}{sources_text}"):

        # Sentiment summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sentiment", f"{sent_emoji} {news_result.overall_sentiment}")
        with col2:
            st.metric("Score", f"{news_result.overall_score:.0%}")
        with col3:
            st.metric("🟢 Bullish", f"{news_result.bullish_count}")
        with col4:
            st.metric("🔴 Bearish", f"{news_result.bearish_count}")

        # Sources info
        if hasattr(news_result, 'sources_tried') and news_result.sources_tried > 0:
            st.caption(f"📡 Data from {news_result.sources_succeeded}/{news_result.sources_tried} sources (Google News, Finnhub, NewsAPI, AI Search)")

        st.markdown("---")

        # News articles with Deep Dive style
        for article in news_result.articles[:15]:
            sentiment_icon = "🟢" if article.sentiment == "BULLISH" else "🔴" if article.sentiment == "BEARISH" else "🟡"

            # Format date
            if article.published_at:
                pub_date = article.published_at.strftime('%b %d, %H:%M')
            else:
                pub_date = ''

            # Category icon
            category_icons = {
                'Fed Policy': '🏛️',
                'Inflation': '📊',
                'Auctions': '🏦',
                'Yields': '📈',
                'Macro': '🌍',
                'General': '📰'
            }
            cat_icon = category_icons.get(article.category, '📰')

            # Display article with hyperlink
            st.markdown(f"""
**{sentiment_icon} [{article.title}]({article.url})**  
<small>{cat_icon} {article.category} • **{article.source}** • {pub_date}</small>
""", unsafe_allow_html=True)

            if article.description:
                desc = article.description[:180] + "..." if len(article.description) > 180 else article.description
                st.caption(desc)

            st.markdown("")  # Spacing


def _render_news_section(news_result: 'BondNewsResult'):
    """Render bond news section with hyperlinks."""

    st.markdown("### 📰 Bond Market News")

    # Sentiment summary bar
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sent_emoji = "🟢" if news_result.overall_sentiment == "BULLISH" else "🔴" if news_result.overall_sentiment == "BEARISH" else "🟡"
        st.metric("Sentiment", f"{sent_emoji} {news_result.overall_sentiment}")
    with col2:
        st.metric("Score", f"{news_result.overall_score:.0%}")
    with col3:
        st.metric("Bullish", f"{news_result.bullish_count}", delta=None)
    with col4:
        st.metric("Bearish", f"{news_result.bearish_count}", delta=None)

    if news_result.themes:
        st.caption(f"**Key Themes:** {' • '.join(news_result.themes)}")

    st.markdown("---")

    # News articles with hyperlinks
    for article in news_result.articles[:12]:
        sentiment_icon = "🟢" if article.sentiment == "BULLISH" else "🔴" if article.sentiment == "BEARISH" else "🟡"

        # Format date
        pub_date = article.published_at.strftime('%b %d, %H:%M') if article.published_at else ''

        # Category badge
        category_colors = {
            'Fed Policy': '🏛️',
            'Inflation': '📊',
            'Auctions': '🏦',
            'Yields': '📈',
            'Macro': '🌍',
            'General': '📰'
        }
        cat_icon = category_colors.get(article.category, '📰')

        st.markdown(f"""
        {sentiment_icon} **[{article.title}]({article.url})**  
        {cat_icon} {article.category} • {article.source} • {pub_date}
        """)

        if article.description:
            st.caption(article.description[:150] + "..." if len(article.description) > 150 else article.description)

        st.markdown("")


def _render_yields_section(yields, yield_hist):
    """Render treasury yields section."""

    st.markdown("### 📊 Treasury Yields")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("30-Year", f"{yields.yield_30y:.2f}%")
    with col2:
        st.metric("10-Year", f"{yields.yield_10y:.2f}%")
    with col3:
        st.metric("2-Year", f"{yields.yield_2y:.2f}%")
    with col4:
        spread_color = "normal" if yields.spread_10y_2y > 0 else "inverse"
        st.metric("10Y-2Y Spread", f"{yields.spread_10y_2y:+.2f}%",
                  delta="Normal" if yields.spread_10y_2y > 0 else "Inverted",
                  delta_color=spread_color)

    # Yield curve chart
    if not yield_hist.empty:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("30Y Yield (60 days)", "Yield Curve"))

        fig.add_trace(
            go.Scatter(x=yield_hist.index, y=yield_hist['yield_30y'],
                       mode='lines', name='30Y Yield',
                       line=dict(color='#1f77b4', width=2)),
            row=1, col=1
        )

        ma_20 = yield_hist['yield_30y'].rolling(20).mean()
        fig.add_trace(
            go.Scatter(x=yield_hist.index, y=ma_20,
                       mode='lines', name='20-day MA',
                       line=dict(color='orange', width=1, dash='dash')),
            row=1, col=1
        )

        maturities = ['2Y', '10Y', '30Y']
        current_yields = [yields.yield_2y, yields.yield_10y, yields.yield_30y]
        curve_color = '#28a745' if yields.spread_10y_2y > 0 else '#dc3545'

        fig.add_trace(
            go.Scatter(x=maturities, y=current_yields,
                       mode='lines+markers', name='Current Curve',
                       line=dict(color=curve_color, width=3),
                       marker=dict(size=10)),
            row=1, col=2
        )

        fig.update_layout(height=280, showlegend=False, margin=dict(t=30, b=20))
        st.plotly_chart(fig, width='stretch')


# =============================================================================
# AI PREDICTIONS SECTION (NEW)
# =============================================================================

def _render_ai_predictions_section():
    """
    Render AI-powered bond predictions with yield momentum,
    calendar effects, cross-asset signals, and macro factors.
    Only shows value-add data, not duplicate ETF predictions.
    """

    st.markdown("### 🔮 Market Intelligence")
    st.caption("Calendar alerts • Yield momentum • Cross-asset signals • Macro factors")

    # =================================================================
    # ROW 0: MACRO ALERTS (NEW - Always show first if elevated)
    # =================================================================
    if MACRO_ENGINE_AVAILABLE:
        try:
            factors = get_macro_factors()
            elevated = factors.get_elevated_factors(threshold=65)

            if elevated:
                st.markdown("#### ⚠️ Macro Alerts")
                alert_cols = st.columns(min(len(elevated), 4))
                for i, (factor, value) in enumerate(elevated[:4]):
                    with alert_cols[i]:
                        factor_display = factor.replace('_', ' ').title()
                        if value >= 75:
                            st.error(f"🔴 **{factor_display}**: {value}")
                        else:
                            st.warning(f"🟡 **{factor_display}**: {value}")
        except Exception as e:
            logger.debug(f"Macro factors unavailable: {e}")

    if not PREDICTION_ENGINE_AVAILABLE:
        st.warning("Bond prediction engine not available. Make sure `bond_prediction_engine.py` is in `src/analytics/`")
        return

    try:
        # Get or create prediction engine
        if 'bond_pred_engine' not in st.session_state:
            st.session_state.bond_pred_engine = BondPredictionEngine()

        engine = st.session_state.bond_pred_engine

        # =================================================================
        # ROW 1: CALENDAR ALERTS (Always visible at top)
        # =================================================================
        alerts = get_calendar_alerts()
        if alerts:
            with st.container():
                alert_cols = st.columns(min(len(alerts), 4))  # Max 4 alerts
                for i, alert in enumerate(alerts[:4]):
                    with alert_cols[i]:
                        if "FOMC" in alert or "EXTREME" in alert:
                            st.error(alert, icon="⚠️")
                        elif "ex-dividend" in alert.lower():
                            st.success(alert, icon="💰")
                        elif "First" in alert:
                            st.info(alert, icon="📅")
                        else:
                            st.info(alert, icon="ℹ️")

        # =================================================================
        # ROW 2: YIELD MOMENTUM + CROSS-ASSET (Side by side)
        # =================================================================
        col_momentum, col_cross = st.columns(2)

        with col_momentum:
            st.markdown("#### 📈 Yield Momentum")
            try:
                momentum = engine.analyze_yield_momentum()

                # Direction with color
                direction_config = {
                    'RISING_FAST': ('🔴🔴', 'Bearish for bonds - yields surging'),
                    'RISING': ('🔴', 'Bearish for bonds - yields climbing'),
                    'STABLE': ('🟡', 'Neutral - yields stable'),
                    'FALLING': ('🟢', 'Bullish for bonds - yields dropping'),
                    'FALLING_FAST': ('🟢🟢', 'Very bullish - yields plunging'),
                }
                emoji, desc = direction_config.get(
                    momentum.direction.value, ('🟡', 'Unknown')
                )

                # Metrics row
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(
                        "Direction",
                        f"{emoji} {momentum.direction.value.replace('_', ' ')}"
                    )
                with m2:
                    change_5d = momentum.change_5d_30y * 100 if momentum.change_5d_30y else 0
                    st.metric("5D Change", f"{change_5d:+.0f}bps")
                with m3:
                    forecast = momentum.forecast_5d_bps if momentum.forecast_5d_bps else 0
                    delta_color = "inverse" if forecast > 0 else "normal"  # Red if yields rising
                    st.metric("5D Forecast", f"{forecast:+.0f}bps", delta_color=delta_color)

                st.caption(f"30Y: {momentum.current_yield_30y:.2f}% | 10Y: {momentum.current_yield_10y:.2f}%")
                st.info(f"💡 {desc}")

            except Exception as e:
                st.warning(f"Could not load yield momentum: {e}")

        with col_cross:
            st.markdown("#### 🌐 Cross-Asset Signals")
            try:
                cross = engine.analyze_cross_assets()

                # VIX signal
                vix_config = {
                    'HIGH_FEAR': ('🔴', 'High fear - flight to bonds likely'),
                    'ELEVATED': ('🟠', 'Elevated caution'),
                    'NORMAL': ('🟡', 'Normal volatility'),
                    'COMPLACENT': ('🟢', 'Complacent - risk-on mode'),
                }
                vix_emoji, vix_desc = vix_config.get(cross.vix_signal, ('🟡', 'Unknown'))

                # Regime
                regime_config = {
                    'RISK_OFF': ('🛡️', 'Flight to safety - bullish for bonds'),
                    'RISK_ON': ('🚀', 'Risk appetite - bearish for bonds'),
                    'NEUTRAL': ('➡️', 'No clear direction'),
                }
                regime_emoji, regime_desc = regime_config.get(cross.risk_regime, ('➡️', 'Unknown'))

                c1, c2 = st.columns(2)
                with c1:
                    vix_val = f"{cross.vix_level:.1f}" if cross.vix_level else "N/A"
                    st.metric("VIX", vix_val, f"{vix_emoji} {cross.vix_signal}")
                with c2:
                    st.metric("Risk Regime", f"{regime_emoji} {cross.risk_regime}")

                st.info(f"💡 {regime_desc}")

            except Exception as e:
                st.warning(f"Could not load cross-asset signals: {e}")

        # =================================================================
        # ROW 3: MACRO FACTOR GAUGES (NEW)
        # =================================================================
        if MACRO_ENGINE_AVAILABLE:
            with st.expander("🌍 Macro Factor Scores", expanded=False):
                _render_macro_factors_compact()

        # =================================================================
        # ROW 4: SECTOR RATE IMPACTS (Collapsible)
        # =================================================================
        with st.expander("📊 How Current Rates Affect Equity Sectors"):
            try:
                rate_signals = get_rate_signals()
                sector_impacts = rate_signals.get('sector_impacts', {})

                if sector_impacts:
                    sorted_sectors = sorted(sector_impacts.items(), key=lambda x: x[1], reverse=True)

                    col_win, col_lose = st.columns(2)

                    with col_win:
                        st.markdown("**📈 Benefiting from rate environment**")
                        for sector, impact in sorted_sectors:
                            if impact > 0:
                                st.success(f"{sector}: +{impact:.1%}")

                    with col_lose:
                        st.markdown("**📉 Pressured by rate environment**")
                        for sector, impact in reversed(sorted_sectors):
                            if impact < 0:
                                st.error(f"{sector}: {impact:.1%}")

                    st.caption(f"Rate direction: {rate_signals.get('rate_direction', 'N/A')} | "
                              f"Duration recommendation: {rate_signals.get('duration_recommendation', 'N/A')}")
                else:
                    st.info("Sector impacts not available")

            except Exception as e:
                st.warning(f"Could not load sector impacts: {e}")

    except Exception as e:
        st.error(f"Error initializing prediction engine: {e}")
        import traceback
        st.code(traceback.format_exc())


def _render_macro_factors_compact():
    """Render compact macro factor gauges for bond dashboard."""

    if not MACRO_ENGINE_AVAILABLE:
        st.info("Macro engine not available")
        return

    try:
        factors = get_macro_factors()

        # Bond-relevant factors
        bond_factors = [
            ('Inflation Pressure', factors.inflation_pressure, 'Higher inflation → bearish bonds'),
            ('Recession Risk', factors.recession_risk, 'Recession → bullish bonds (flight to safety)'),
            ('Risk-Off Sentiment', factors.risk_off_sentiment, 'Risk-off → bullish bonds'),
            ('Geopolitical Tension', factors.geopolitical_tension, 'Geopolitical risk → bullish bonds'),
            ('Oil Supply Shock', factors.oil_supply_shock, 'Oil shock → inflation → bearish bonds'),
            ('FX Stress', factors.fx_stress, 'FX crisis → USD strength → bullish US bonds'),
        ]

        col1, col2 = st.columns(2)

        for i, (name, value, tooltip) in enumerate(bond_factors):
            col = col1 if i % 2 == 0 else col2

            with col:
                # Color based on value
                if value >= 70:
                    emoji = "🔴"
                elif value >= 55:
                    emoji = "🟡"
                else:
                    emoji = "🟢"

                st.metric(name, f"{emoji} {value}", help=tooltip)

        st.caption("Scale: 0-100 (50 = neutral)")

        # Refresh button
        if st.button("🔄 Refresh Macro Data", key="refresh_macro_bond"):
            with st.spinner("Fetching macro news..."):
                try:
                    new_events, _ = refresh_macro_data()
                    st.success(f"Found {new_events} new events")
                    st.rerun()
                except Exception as e:
                    st.error(f"Refresh failed: {e}")

    except Exception as e:
        st.warning(f"Could not load macro factors: {e}")


def _render_signals_section(signals, institutional):
    """Render bond ETF signals table."""

    st.markdown("### 📈 Bond ETF Signals")

    # Note: Institutional adjustment is already integrated into composite_score
    if institutional:
        st.caption(f"Institutional data integrated: Auction={institutional.auction_demand_score or 'N/A'}, Vol={institutional.vol_regime_score or 'N/A'}")

    if not signals:
        st.warning("No signals available. Try running fresh analysis.")
        return

    signal_data = []
    for ticker, sig in signals.items():
        # Score already includes all adjustments (technical, fundamental, flow, macro, sentiment)
        adjusted_score = sig.composite_score

        signal_emoji = {
            'STRONG BUY': '🟢🟢', 'BUY': '🟢', 'HOLD': '🟡',
            'SELL': '🔴', 'STRONG SELL': '🔴🔴'
        }.get(sig.signal.value, '⚪')

        signal_data.append({
            'Ticker': ticker,
            'Name': sig.name[:22] + '...' if len(sig.name) > 22 else sig.name,
            'Signal_Display': f"{signal_emoji} {sig.signal.value}",
            'Score': adjusted_score,
            'Price': f"${sig.current_price:.2f}",
            'Target': f"${sig.target_price:.2f}",
            'Upside': f"{sig.upside_pct:+.1f}%",
        })

    if not signal_data:
        st.warning("No signal data generated.")
        return

    df = pd.DataFrame(signal_data)

    # Rename for display
    df = df.rename(columns={'Signal_Display': 'Signal'})

    def color_signal(val):
        if not isinstance(val, str):
            return ''
        if 'STRONG BUY' in val:
            return 'background-color: #d4edda; color: #155724; font-weight: bold'
        elif 'BUY' in val and 'STRONG' not in val:
            return 'background-color: #d4edda; color: #155724'
        elif 'STRONG SELL' in val:
            return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
        elif 'SELL' in val:
            return 'background-color: #f8d7da; color: #721c24'
        return ''

    # Use map instead of deprecated applymap
    if 'Signal' in df.columns:
        styled = df.style.map(color_signal, subset=['Signal'])
        st.dataframe(styled, width='stretch', hide_index=True)
    else:
        st.dataframe(df, width='stretch', hide_index=True)


def _render_detailed_analysis(signals, institutional):
    """Render detailed analysis for selected ticker."""

    st.markdown("### 🔍 Detailed Analysis")

    if not signals:
        st.warning("No signals available for detailed analysis.")
        return

    selected_ticker = st.selectbox("Select Bond ETF", options=list(signals.keys()), key="bond_detail_ticker")

    if selected_ticker and selected_ticker in signals:
        sig = signals[selected_ticker]

        # Score already includes all adjustments
        adjusted_score = sig.composite_score

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=adjusted_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"{selected_ticker} Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#28a745' if adjusted_score >= 60 else '#dc3545' if adjusted_score < 40 else '#ffc107'},
                    'steps': [
                        {'range': [0, 40], 'color': '#f8d7da'},
                        {'range': [40, 60], 'color': '#ffffcc'},
                        {'range': [60, 100], 'color': '#d4edda'},
                    ]
                }
            ))
            fig.update_layout(height=220, margin=dict(t=40, b=0))
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.markdown(f"**{selected_ticker}** - {sig.name}")
            st.markdown(f"Signal: **{sig.signal.value}** | Score: {adjusted_score}/100 | Confidence: {sig.confidence}")
            st.markdown(f"Price: ${sig.current_price:.2f} → Target: ${sig.target_price:.2f} ({sig.upside_pct:+.1f}%)")

            # Get duration from signal's fundamental data
            duration = sig.fundamental.modified_duration if sig.fundamental else 'N/A'
            st.markdown(f"Duration: {duration}y")

        # Component scores from new institutional model
        with st.expander("📊 Component Scores (JPM/BlackRock Model)"):
            components = [
                ('Technical', sig.technical.technical_score),
                ('Fundamental', sig.fundamental.fundamental_score),
                ('Flow/Auction', sig.flow.flow_score),
                ('Macro/Fed', sig.macro.macro_score),
                ('Sentiment', sig.sentiment.sentiment_score),
            ]

            for name, value in components:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.caption(name)
                with col2:
                    if value is not None:
                        color = "green" if value >= 60 else "red" if value <= 40 else "orange"
                        st.progress(value / 100)
                        st.caption(f":{color}[{value}/100]")
                    else:
                        st.caption("⚠️ Data not available")

        # Technical indicators
        with st.expander("📈 Technical Indicators"):
            tech = sig.technical
            t_col1, t_col2 = st.columns(2)

            with t_col1:
                rsi_str = f"{tech.rsi_14:.1f}" if tech.rsi_14 else "N/A"
                st.markdown(f"**RSI(14):** {rsi_str} ({tech.rsi_signal})")
                st.markdown(f"**MACD:** {tech.macd_crossover}")
                st.markdown(f"**Bollinger:** {tech.bb_position}")
                st.markdown(f"**Trend:** {tech.trend.value}")

            with t_col2:
                sma50_str = f"${tech.sma_50:.2f}" if tech.sma_50 else "$0.00"
                sma200_str = f"${tech.sma_200:.2f}" if tech.sma_200 else "$0.00"
                vwap_str = f"${tech.vwap:.2f}" if tech.vwap else "$0.00"
                support_str = f"${tech.support_1:.2f}" if tech.support_1 else "$0.00"
                st.markdown(f"**SMA 50:** {sma50_str}")
                st.markdown(f"**SMA 200:** {sma200_str}")
                st.markdown(f"**VWAP:** {vwap_str}")
                st.markdown(f"**Support 1:** {support_str}")

        # Bull/Bear cases
        with st.expander("🎯 Bull & Bear Case"):
            b_col1, b_col2 = st.columns(2)

            with b_col1:
                st.markdown("**🟢 Bull Case:**")
                for point in sig.bull_case:
                    st.markdown(f"• {point}")
                if not sig.bull_case:
                    st.caption("No strong bullish factors")

            with b_col2:
                st.markdown("**🔴 Bear Case:**")
                for point in sig.bear_case:
                    st.markdown(f"• {point}")
                if not sig.bear_case:
                    st.caption("No strong bearish factors")


def _render_ai_chat(complete_ai_context: str):
    """Render AI chat for bonds."""

    st.markdown("### 🤖 Bond AI Advisor")

    if not AI_CHAT_AVAILABLE:
        st.warning("AI Chat not available. Check src/ai/chat.py import.")
        return

    # Initialize chat
    if 'bond_chat' not in st.session_state:
        try:
            st.session_state.bond_chat = AlphaChat()
        except Exception as e:
            st.error(f"Could not initialize AI: {e}")
            return

    chat = st.session_state.bond_chat

    # Quick question buttons
    st.caption("Quick questions:")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Full Analysis", key="q1", width='stretch'):
            st.session_state.bond_pending_question = "Give me a complete analysis of the current bond market. Should I buy bond ETFs like TLT or ZROZ today?"
    with col2:
        if st.button("🎯 Trade Idea", key="q2", width='stretch'):
            st.session_state.bond_pending_question = "Based on all the data, what's your best trade idea for bonds right now?"

    col3, col4 = st.columns(2)
    with col3:
        if st.button("⚠️ Key Risks", key="q3", width='stretch'):
            st.session_state.bond_pending_question = "What are the key risks for bond investors right now?"
    with col4:
        if st.button("📰 News Impact", key="q4", width='stretch'):
            st.session_state.bond_pending_question = "How is the recent news affecting bonds? What should I watch?"

    # Chat history
    st.markdown("---")

    chat_container = st.container()

    with chat_container:
        for msg in st.session_state.bond_chat_history[-6:]:  # Last 6 messages
            role = "user" if msg["role"] == "user" else "assistant"
            with st.chat_message(role):
                st.write(msg["content"][:500] + "..." if len(msg["content"]) > 500 else msg["content"])

    # Chat input
    prompt = st.chat_input("Ask about bonds, yields, Fed policy...")

    # Handle pending question from buttons
    if st.session_state.get('bond_pending_question'):
        prompt = st.session_state.bond_pending_question
        st.session_state.bond_pending_question = None

    if prompt:
        st.session_state.bond_chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                # Add bond context to prompt - increase limit to capture prices
                enhanced_prompt = f"""[BOND MARKET CONTEXT]
{complete_ai_context[:5000]}

[USER QUESTION]
{prompt}

CRITICAL FORMATTING RULES:
1. DO NOT use ** or * markdown around prices or numbers
2. Entry zones: "$85-$87" format only (NO duplicates like "$85-$85-$87")
3. Use regular hyphens (-) not en-dashes (–)
4. State upside reference: "from current" or "from entry"

FORBIDDEN DATA (DO NOT INVENT):
❌ Put/Call ratios - NEVER say "Put/Call ratio = 0.62" or any number
❌ Options flow - NEVER say "neutral options flow" or similar
❌ Polymarket probabilities
❌ Treasury auction dates

If you don't have data for something, don't mention it at all.
Plain text prices, no markdown."""

                for chunk in chat.chat_stream(enhanced_prompt):
                    full_response += chunk
                    # Clean thinking tags
                    display = full_response
                    if '</think>' in display:
                        display = display.split('</think>')[-1].strip()
                    response_placeholder.markdown(display + "▌")

                # Final clean
                if '</think>' in full_response:
                    full_response = full_response.split('</think>')[-1].strip()

                # Sanitize: remove Polymarket hallucinations and fix formatting
                # Extract numeric block from context for potential restoration
                numeric_block = _extract_first_fenced_block(complete_ai_context)
                full_response = sanitize_bond_ai_answer(
                    full_response,
                    allow_polymarket=False,
                    numeric_block=numeric_block
                )

                response_placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"Error: {str(e)}"
                response_placeholder.error(full_response)

        st.session_state.bond_chat_history.append({"role": "assistant", "content": full_response})


def _build_complete_ai_context(yields, yield_hist, signals, sample_signal,
                               economic_impact, institutional, news_result, generator,
                               bond_intelligence=None, portfolio_weights=None, move=None) -> str:
    """Build comprehensive AI context with all bond data.

    DESIGN: All numeric data is hard-rendered in a fenced code block.
    The LLM is strictly instructed NOT to reformat or recalculate anything
    from that block. This prevents corrupted output like "89.97(+2.387.93)".

    Args:
        move: Dict from generator.get_move_index() with keys:
              value, asof, source, quality, warning
    """

    # =========================================================================
    # 1. EXTRACT AS-OF DATES FROM SOURCES
    # =========================================================================
    def _extract_date(src: str) -> str:
        """Extract date like '2026-01-09' from 'FRED:DGS10 (2026-01-09)'"""
        try:
            match = re.search(r'\((\d{4}-\d{2}-\d{2})\)', str(src))
            return match.group(1) if match else "unknown"
        except:
            return "unknown"

    sources = getattr(yields, 'sources', {}) or {}
    spread_date = getattr(yields, 'spread_10y_2y_date', '') or _extract_date(sources.get('spread_10y_2y', '')) or "unknown"
    y10_date = getattr(yields, 'yield_10y_date', '') or _extract_date(sources.get('yield_10y', '')) or "unknown"
    y2_date = getattr(yields, 'yield_2y_date', '') or _extract_date(sources.get('yield_2y', '')) or "unknown"
    y30_date = getattr(yields, 'yield_30y_date', '') or _extract_date(sources.get('yield_30y', '')) or "unknown"

    # Data quality
    dq = str(getattr(yields, 'data_quality', 'OK') or 'OK')

    # =========================================================================
    # DATA VALIDATION - Flag suspicious values
    # =========================================================================
    data_warnings = []

    # Check for default/suspicious values
    spread_val = getattr(yields, 'spread_10y_2y', 0) or 0
    if abs(spread_val) < 0.01:  # Spread exactly 0.00 is suspicious
        data_warnings.append("SPREAD=0.00 likely indicates stale/missing data")
        dq = "DEGRADED"

    # Check MOVE index - use passed move parameter (single authoritative source)
    move_val = None
    move_asof = None
    move_quality = "MISSING"
    if move and isinstance(move, dict):
        move_val = move.get("value")
        move_asof = move.get("asof", "unknown")
        move_quality = move.get("quality", "MISSING")
        if move.get("warning"):
            data_warnings.append(f"MOVE: {move['warning']}")
        if move_quality == "SUSPICIOUS":
            data_warnings.append(f"MOVE value ({move_val}) outside normal range")
    else:
        data_warnings.append("MOVE index not fetched")

    # Check yield values
    y10_val = getattr(yields, 'yield_10y', 0) or 0
    y2_val = getattr(yields, 'yield_2y', 0) or 0
    if y10_val == 0 or y2_val == 0:
        data_warnings.append("Zero yields indicate missing data")
        dq = "DEGRADED"

    # Curve shape from analytics (authoritative)
    curve_shape = "NORMAL"
    if sample_signal and hasattr(sample_signal, 'curve_shape'):
        if hasattr(sample_signal.curve_shape, 'value'):
            curve_shape = sample_signal.curve_shape.value
        else:
            curve_shape = str(sample_signal.curve_shape)

    # =========================================================================
    # 2. PRE-COMPUTE PORTFOLIO LABELS (in code, NOT by LLM)
    # =========================================================================
    portfolio_labels = {}
    if portfolio_weights:
        for ticker in signals.keys():
            actual = portfolio_weights.get(ticker, {}).get('actual', 0)
            target = portfolio_weights.get(ticker, {}).get('target', None)

            if target is None:
                label = "NO_TARGET"
            elif actual > target * 1.1:
                label = "OVERWEIGHT"
            elif actual < target * 0.9:
                label = "UNDERWEIGHT"
            else:
                label = "ON_TARGET"

            portfolio_labels[ticker] = {'actual': actual, 'target': target, 'label': label}

    # =========================================================================
    # 3. BUILD HARD-RENDERED NUMERIC BLOCK (fenced - LLM cannot modify)
    # =========================================================================
    lines = []
    lines.append("```")
    lines.append("=" * 60)
    lines.append("NUMERIC DATA BLOCK - DO NOT EDIT OR REFORMAT")
    lines.append("=" * 60)
    lines.append("")

    # Treasury Yields with as-of dates
    lines.append("TREASURY YIELDS:")
    lines.append(f"  30Y: {yields.yield_30y:.2f}%  (DGS30, as-of {y30_date})")
    lines.append(f"  10Y: {yields.yield_10y:.2f}%  (DGS10, as-of {y10_date})")
    lines.append(f"  2Y:  {yields.yield_2y:.2f}%  (DGS2, as-of {y2_date})")
    lines.append("")

    # Yield Curve with as-of date
    lines.append("YIELD CURVE:")
    lines.append(f"  10Y-2Y Spread: {yields.spread_10y_2y:+.2f}%  (as-of {spread_date})")
    lines.append(f"  Curve Shape: {curve_shape}")
    lines.append("")

    # ETF Prices and Targets - each field on separate line
    lines.append("ETF DATA:")
    for ticker, sig in signals.items():
        current = sig.current_price
        target = sig.target_price
        upside = ((target / current) - 1) * 100 if current > 0 else 0

        # Entry zone calculation
        entry_low = round(current * 0.97, 2)
        entry_high = round(current * 0.99, 2)
        entry_mid = (entry_low + entry_high) / 2
        stop = round(entry_mid * 0.95, 2)
        upside_from_entry = ((target / entry_mid) - 1) * 100 if entry_mid > 0 else 0

        lines.append(f"  {ticker}:")
        lines.append(f"    current_price: ${current:.2f}")
        lines.append(f"    model_target (MODEL): ${target:.2f}")
        lines.append(f"    upside_from_current: {upside:+.1f}%")
        lines.append(f"    entry_zone (RULE: 1-3% pullback): ${entry_low:.2f} to ${entry_high:.2f}")
        lines.append(f"    stop_loss (RULE: 5% below entry): ${stop:.2f}")
        lines.append(f"    upside_from_entry: {upside_from_entry:+.1f}%")

        # Portfolio weight with pre-computed label
        if ticker in portfolio_labels:
            pw = portfolio_labels[ticker]
            if pw['target'] is not None:
                lines.append(f"    portfolio: {pw['actual']:.1f}% actual vs {pw['target']:.1f}% target = {pw['label']}")
            else:
                lines.append(f"    portfolio: {pw['actual']:.1f}% actual, {pw['label']}")
        lines.append("")

    # Fed Funds Rate
    if bond_intelligence and hasattr(bond_intelligence, 'fed_policy'):
        fp = bond_intelligence.fed_policy
        fed_low = getattr(fp, 'target_lower', 0) or 0
        fed_high = getattr(fp, 'target_upper', 0) or 0
        effr = getattr(fp, 'effective_rate', 0) or 0
        lines.append("FED FUNDS:")
        lines.append(f"  Target Range: {fed_low:.2f}% to {fed_high:.2f}%")
        lines.append(f"  EFFR: {effr:.2f}%")
        lines.append("")

    # Inflation data (if available)
    if sample_signal and hasattr(sample_signal, 'inflation_data') and sample_signal.inflation_data:
        inf = sample_signal.inflation_data
        if inf.get('data_available'):
            lines.append("INFLATION:")
            if inf.get('cpi_yoy') is not None:
                lines.append(f"  CPI YoY: {inf['cpi_yoy']:.1f}%  (as-of {inf.get('cpi_date', 'unknown')})")
            if inf.get('core_pce_yoy') is not None:
                lines.append(f"  Core PCE YoY: {inf['core_pce_yoy']:.1f}%  (as-of {inf.get('core_pce_date', 'unknown')})")
            lines.append("")
        else:
            lines.append("INFLATION: NOT_AVAILABLE")
            lines.append("")
    else:
        lines.append("INFLATION: NOT_LOADED")
        lines.append("")

    # MOVE Index - use validated value from passed move parameter
    lines.append("RATE VOLATILITY (MOVE):")
    if move_val is not None and move_quality != "MISSING":
        lines.append(f"  MOVE: {move_val:.1f} ({move_quality}) [source: ^MOVE, as-of: {move_asof}]")
    else:
        lines.append("  MOVE: NOT_AVAILABLE")
    lines.append("")

    # Add data warnings if any
    if data_warnings:
        lines.append("⚠️ DATA WARNINGS:")
        for w in data_warnings:
            lines.append(f"  - {w}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("END NUMERIC DATA BLOCK")
    lines.append("=" * 60)
    lines.append("```")

    numeric_block = "\n".join(lines)

    # =========================================================================
    # 4. BUILD CONTEXT WITH INSTRUCTIONS
    # =========================================================================
    warnings_str = "\n".join(f"  ⚠️ {w}" for w in data_warnings) if data_warnings else ""

    context = f"""
{'='*70}
BOND MARKET ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*70}

Data Quality: {dq}
{warnings_str}

{numeric_block}

⛔ CRITICAL: The numeric block above is FINAL and FROZEN.
   - Do NOT rewrite or merge numbers (no "89.97(+2.387.93)" patterns)
   - Do NOT recalculate percentages
   - Copy values exactly as shown if needed
   - Entry zone and Target are SEPARATE concepts

"""

    # Add non-numeric narrative from bond intelligence
    if bond_intelligence and hasattr(bond_intelligence, 'ai_context'):
        # Filter out any numeric data from AI context to avoid duplication
        context += "MARKET NARRATIVE:\n"
        context += bond_intelligence.ai_context
        context += "\n\n"

    # Economic events (non-numeric)
    if economic_impact:
        context += "TODAY'S EVENTS:\n"
        events = []
        if economic_impact.cpi_today:
            events.append("CPI Report TODAY")
        if economic_impact.fomc_today:
            events.append("FOMC Decision TODAY")
        if economic_impact.fomc_this_week and not economic_impact.fomc_today:
            events.append(f"FOMC in {economic_impact.days_to_fomc} days")
        if economic_impact.today_summary:
            events.append(economic_impact.today_summary)
        context += "  " + "; ".join(events) if events else "  No major events"
        context += "\n\n"

    # =========================================================================
    # 4b. SUKUK DATA (if available)
    # =========================================================================
    sukuk_context = st.session_state.get('sukuk_ai_context', '')
    if sukuk_context:
        context += sukuk_context
        context += "\n\n"

    # =========================================================================
    # 5. AI INSTRUCTIONS (concise)
    # =========================================================================
    context += f"""
{'='*70}
AI INSTRUCTIONS
{'='*70}

READ THE NUMERIC BLOCK ABOVE. All values are final.

FORMATTING RULES:
1. Do NOT merge numbers. Write: "Price: $87.93, Target: $89.97 (+2.3%)"
   NOT: "89.97(+2.387.93)"
2. Entry zone and Target are DIFFERENT:
   - Entry zone = where to BUY (e.g., "$85-$87")
   - Target = where to SELL (e.g., "$90")
3. No asterisks around prices. Write "$87.93" not "**$87.93**"
4. Use hyphen (-) not en-dash (−)

PORTFOLIO LABELS (pre-computed, do not change):
- {', '.join(f"{t}: {p['label']}" for t, p in portfolio_labels.items()) if portfolio_labels else 'No portfolio data'}
- If label is NO_TARGET, do NOT say "underweight" or "overweight"

DATA NOT AVAILABLE (DO NOT INVENT - these are FORBIDDEN):
❌ Options flow / Put-Call ratios - We do NOT have bond options data
   NEVER say "Put/Call ratio = X" or "neutral options flow"
❌ Polymarket / CME FedWatch probabilities  
❌ Treasury auction calendar / specific dates
❌ Bid-to-cover ratios / auction demand metrics

If MOVE_INDEX shows "NOT_AVAILABLE" or equals 100.0:
  - Do NOT cite MOVE index as if it's real
  - Say "volatility data unavailable" instead

If spread = 0.00%:
  - This indicates missing data, not a flat curve
  - Acknowledge data quality issue

CURVE INTERPRETATION:
- Curve is "{curve_shape}" - do not re-label
- A positive curve is CONSISTENT WITH easing expectations (conditional, not guaranteed)
- Do NOT say "curve favors long duration" as absolute fact
"""

    return context


# =============================================================================
# SUKUK SECTION
# =============================================================================

def _render_sukuk_section(yields):
    """Render USD Sukuk section with live IBKR data and signals."""

    st.markdown("### 🕌 USD Sukuk")
    st.caption("Islamic bonds - HYSK ETF holdings with IBKR live data")

    # Check if sukuk module is available
    if not SUKUK_MODULE_AVAILABLE:
        with st.expander("🕌 USD Sukuk (Module not available)"):
            st.info("Sukuk module not loaded. Make sure src/analytics/sukuk_*.py files exist.")
        return

    # Path to sukuk universe
    sukuk_universe_path = "config/HYSK-holdings_ibkr.json"

    # Check if file exists
    if not os.path.exists(sukuk_universe_path):
        with st.expander("🕌 USD Sukuk (No data file)"):
            st.warning(f"Sukuk universe file not found: {sukuk_universe_path}")
            st.caption("Copy HYSK-holdings_ibkr.json to config/ folder")
        return

    # Check for cached sukuk data in session
    has_sukuk_cache = (
        'sukuk_signals' in st.session_state and
        st.session_state.sukuk_signals is not None and
        len(st.session_state.sukuk_signals) > 0
    )

    # Refresh button
    col1, col2 = st.columns([1, 3])
    with col1:
        refresh_sukuk = st.button("🔄 Refresh Sukuk", key="refresh_sukuk")
    with col2:
        if has_sukuk_cache:
            n_signals = len(st.session_state.sukuk_signals)
            st.caption(f"✅ {n_signals} sukuk loaded")
        else:
            st.caption("Click refresh to load sukuk data")

    # Load sukuk data if needed
    if refresh_sukuk or not has_sukuk_cache:
        with st.spinner("Loading sukuk data from IBKR..."):
            try:
                # Load universe
                universe = load_sukuk_universe(sukuk_universe_path)

                # Fetch live data
                live_data, warnings = fetch_sukuk_market_data(
                    universe,
                    host="127.0.0.1",
                    port=7496,
                    use_cached_fallback=True
                )

                # Build rates regime from yields
                rates_regime = None
                if yields:
                    spread = getattr(yields, 'spread_10y_2y', 0) or 0
                    y10 = getattr(yields, 'yield_10y', 4.5) or 4.5

                    # Determine curve shape
                    if spread > 0.5:
                        curve_shape = "STEEP"
                    elif spread > 0:
                        curve_shape = "NORMAL"
                    elif spread > -0.5:
                        curve_shape = "FLAT"
                    else:
                        curve_shape = "INVERTED"

                    # Determine Fed stance (simplified)
                    fed_stance = "NEUTRAL"
                    if y10 < 4.0:
                        fed_stance = "DOVISH"
                    elif y10 > 5.0:
                        fed_stance = "HAWKISH"

                    rates_regime = RatesRegime(
                        curve_shape=curve_shape,
                        rates_level="NORMAL",
                        fed_stance=fed_stance,
                        spread_10y_2y=spread,
                        yield_10y=y10,
                        fed_funds_rate=5.0
                    )

                # Generate signals
                generator = SukukSignalGenerator(universe.risk_limits)
                signals = generator.generate_signals(live_data, rates_regime=rates_regime)

                # Build numeric block for AI
                sukuk_block = build_sukuk_numeric_block(signals, rates_regime)

                # Store in session
                st.session_state.sukuk_universe = universe
                st.session_state.sukuk_live_data = live_data
                st.session_state.sukuk_signals = signals
                st.session_state.sukuk_ai_context = sukuk_block
                st.session_state.sukuk_warnings = warnings

                if warnings:
                    for w in warnings:
                        st.warning(w)

            except Exception as e:
                st.error(f"Error loading sukuk data: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # Get cached data
    signals = st.session_state.get('sukuk_signals', [])

    if not signals:
        st.info("No sukuk signals available. Click refresh to load.")
        return

    # Summary metrics
    buy_signals = [s for s in signals if s.action == SukukAction.BUY]
    hold_signals = [s for s in signals if s.action == SukukAction.HOLD]
    watch_signals = [s for s in signals if s.action == SukukAction.WATCH]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🟢 BUY", len(buy_signals))
    with col2:
        st.metric("🟡 HOLD", len(hold_signals))
    with col3:
        st.metric("⚪ WATCH", len(watch_signals))
    with col4:
        # Average conviction for BUY signals
        avg_conviction = sum(s.conviction for s in buy_signals) / len(buy_signals) if buy_signals else 0
        st.metric("Avg Conviction", f"{avg_conviction:.0f}%")

    # Signals table
    with st.expander("📊 Sukuk Signals", expanded=True):
        if signals:
            df_data = []
            for sig in signals:
                inst = sig.instrument
                price_str = f"${sig.price:.2f}" if sig.price else "N/A"
                yield_str = f"{inst.cached_yield:.2f}%" if inst.cached_yield else "N/A"

                df_data.append({
                    "Name": inst.name,
                    "Issuer": inst.issuer_bucket,
                    "Price": price_str,
                    "Coupon": f"{inst.coupon_rate_pct:.2f}%",
                    "Yield": yield_str,
                    "TTM": f"{sig.ttm_years:.1f}y",
                    "Signal": sig.action.value,
                    "Conviction": f"{sig.conviction}%",
                    "Size Cap": f"{sig.size_cap_pct:.1f}%",
                })

            df = pd.DataFrame(df_data)

            # Style the dataframe
            def color_signal(val):
                if val == "BUY":
                    return "background-color: #28a745; color: white"
                elif val == "HOLD":
                    return "background-color: #ffc107; color: black"
                elif val == "WATCH":
                    return "background-color: #6c757d; color: white"
                return ""

            styled_df = df.style.applymap(color_signal, subset=["Signal"])
            st.dataframe(styled_df, width='stretch', hide_index=True)

    # Top opportunities
    if buy_signals:
        with st.expander("🎯 Top Opportunities"):
            for sig in buy_signals[:5]:
                inst = sig.instrument
                price_str = f"${sig.price:.2f}" if sig.price else "N/A"
                yield_str = f"{inst.cached_yield:.2f}%" if inst.cached_yield else "N/A"

                st.markdown(f"""
                **{inst.name}** ({inst.issuer})
                - Price: {price_str} | Yield: {yield_str} | Coupon: {inst.coupon_rate_pct:.2f}%
                - TTM: {sig.ttm_years:.1f}y | Conviction: {sig.conviction}%
                - Reason: {sig.reason}
                """)
                st.markdown("---")


def _render_education():
    """Render education section."""
    st.markdown("""
    ### Score Interpretation
    - **75-100**: Strong BUY - bonds likely to rally
    - **60-74**: BUY - favorable conditions
    - **40-59**: HOLD - wait for clearer signals
    - **25-39**: SELL - bearish conditions
    - **0-24**: Strong SELL

    ### Key Relationships
    | Factor | Bullish for Bonds | Bearish for Bonds |
    |--------|-------------------|-------------------|
    | **Yields** | Falling | Rising |
    | **Fed** | Cutting rates / Dovish | Hiking rates / Hawkish |
    | **Inflation** | Falling | Rising |
    | **Economy** | Weakening | Strengthening |
    | **MOVE Index** | < 80 (Low) | > 120 (High) |
    | **Auctions** | Strong demand | Weak demand |

    ### ETF Selection
    | ETF | Best For |
    |-----|----------|
    | **TLT** | Core long bond position |
    | **ZROZ** | Maximum duration/convexity |
    | **EDV** | Low-cost long duration |
    | **TMF** | Leveraged short-term bets |
    | **SHY** | Cash parking, low risk |
    
    ### Sukuk (Islamic Bonds)
    | Issuer Type | Examples |
    |-------------|----------|
    | **KSA Sovereign** | PIF, SECO, SRC |
    | **UAE Banks** | DIB, ADIB |
    | **Qatar Banks** | QIB |
    | **Indonesia** | INDOIS (sovereign) |
    | **Corporates** | DP World, Aldar, ESIC |
    """)


if __name__ == "__main__":
    st.set_page_config(page_title="Bond & Sukuk Trading", layout="wide")
    render_bond_trading_tab()