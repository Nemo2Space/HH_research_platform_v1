"""
Options Flow Analyzer

Detects unusual options activity that may signal institutional moves:
- Volume > Open Interest (new positions being opened)
- Volume significantly above average
- Put/Call ratio extremes
- High implied volatility spikes

Data source: IBKR (real-time) with Yahoo Finance fallback (delayed)

Author: Alpha Research Platform
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.logging import get_logger

# Database connection (optional - graceful fallback)
try:
    from src.db.connection import get_connection, get_engine
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

# IBKR Options fetcher (real-time data)
try:
    from src.broker.ibkr_options import (
        OptionsDataFetcher,
        DataSource,
        fetch_options_chain,
        OptionsChainResult,
    )
    IBKR_OPTIONS_AVAILABLE = True
except ImportError:
    IBKR_OPTIONS_AVAILABLE = False

    # Define fallback DataSource enum
    from enum import Enum
    class DataSource(Enum):
        IBKR = "IBKR"
        YAHOO = "YAHOO"
        UNKNOWN = "UNKNOWN"

logger = get_logger(__name__)

@dataclass
class OptionsAlert:
    """Single unusual options activity alert."""
    ticker: str
    alert_type: str  # 'UNUSUAL_VOLUME', 'HIGH_OI_RATIO', 'IV_SPIKE', 'PUT_CALL_SKEW'
    direction: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    severity: str  # 'HIGH', 'MEDIUM', 'LOW'

    # Details
    option_type: str  # 'CALL' or 'PUT'
    strike: float
    expiry: str

    # Metrics
    volume: int
    open_interest: int
    volume_oi_ratio: float
    implied_volatility: float

    # Context
    stock_price: float
    distance_from_strike_pct: float
    days_to_expiry: int

    # Interpretation
    description: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptionsFlowSummary:
    """Summary of options flow for a ticker."""
    ticker: str
    stock_price: float

    # Aggregate metrics
    total_call_volume: int
    total_put_volume: int
    total_call_oi: int
    total_put_oi: int

    # Ratios
    put_call_volume_ratio: float
    put_call_oi_ratio: float

    # Sentiment
    overall_sentiment: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    sentiment_score: float  # -100 to +100

    # Unusual activity
    alerts: List[OptionsAlert] = field(default_factory=list)

    # Max pain - NOTE: aggregated across near-term expiries, not a specific date
    max_pain_price: float = 0.0
    max_pain_expiry_context: str = "aggregated across near-term expiries"  # FIX: Add expiry context

    # IV metrics
    avg_call_iv: float = 0.0
    avg_put_iv: float = 0.0
    iv_skew: float = 0.0  # Put IV - Call IV

    # Data source tracking
    data_source: str = "UNKNOWN"  # "IBKR" (real-time) or "YAHOO" (delayed)
    data_timestamp: str = ""  # When data was fetched



class OptionsFlowAnalyzer:
    """
    Analyze options flow for unusual activity.

    Detection rules:
    1. Volume > Open Interest: New positions being opened
    2. Volume > 5x avg daily volume: Unusual activity
    3. Put/Call ratio > 1.5 or < 0.5: Sentiment extreme
    4. IV spike > 50%: Volatility event expected

    Filtering:
    - Only near-the-money options (within 20% of strike)
    - Excludes deep ITM options with distorted IV
    - Minimum volume/OI thresholds
    """

    # Thresholds for unusual activity
    VOLUME_OI_THRESHOLD = 1.0  # Volume > OI
    VOLUME_SPIKE_THRESHOLD = 3.0  # Volume > 3x normal
    PUT_CALL_BULLISH_THRESHOLD = 0.5  # Below = bullish
    PUT_CALL_BEARISH_THRESHOLD = 1.5  # Above = bearish
    IV_HIGH_THRESHOLD = 0.5  # 50% IV considered high
    IV_MAX_THRESHOLD = 2.0  # 200% IV - likely distorted, filter out
    MIN_VOLUME_THRESHOLD = 100  # Minimum volume to consider
    MIN_OI_THRESHOLD = 50  # Minimum OI to consider

    # Moneyness filters (key improvement)
    MAX_ITM_PCT = 20  # Filter out options more than 20% ITM
    MAX_OTM_PCT = 30  # Filter out options more than 30% OTM

    def __init__(self):
        self.cache = {}  # Cache options data

    def get_options_chain(self, ticker: str, max_expiries: int = 4,
                          ibkr_host: str = "127.0.0.1",
                          ibkr_port: int = 7496,
                          skip_ibkr: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, float, str]:
        """
        Get options chain data for a ticker.

        Tries IBKR real-time data first, falls back to Yahoo Finance.

        Args:
            ticker: Stock symbol
            max_expiries: Maximum number of expiration dates to fetch
            ibkr_host: IBKR TWS/Gateway host
            ibkr_port: IBKR port (7496 for TWS, 4001 for Gateway)
            skip_ibkr: If True, skip IBKR entirely and go straight to Yahoo (for scanner speed)

        Returns:
            Tuple of (calls_df, puts_df, stock_price, data_source)
        """
        # Try IBKR/Yahoo unified fetcher
        if IBKR_OPTIONS_AVAILABLE and not skip_ibkr:
            try:
                result = fetch_options_chain(ticker, max_expiries, ibkr_host, ibkr_port)

                if not result.calls.empty or not result.puts.empty:
                    source = result.data_source.value  # Actual source: "IBKR" or "YAHOO"

                    # Check if IBKR returned data but with zero volume/OI
                    if source == "IBKR":
                        call_vol = result.calls['volume'].sum() if 'volume' in result.calls.columns else 0
                        call_oi = result.calls['openInterest'].sum() if 'openInterest' in result.calls.columns else 0
                        put_vol = result.puts['volume'].sum() if 'volume' in result.puts.columns else 0
                        put_oi = result.puts['openInterest'].sum() if 'openInterest' in result.puts.columns else 0

                        if call_vol == 0 and put_vol == 0 and call_oi == 0 and put_oi == 0:
                            logger.warning(
                                f"{ticker}: IBKR returned zero volume/OI - falling back to Yahoo for volume data")
                            # Don't return yet - fall through to Yahoo fallback below
                        else:
                            return result.calls, result.puts, result.stock_price, source
                    else:
                        return result.calls, result.puts, result.stock_price, source

            except Exception as e:
                logger.warning(f"Options fetch failed for {ticker}: {e}")

        # Fallback to Yahoo Finance (HARD timeout via subprocess.run)
        logger.info(f"\U0001f4ca {ticker}: Using Yahoo Finance options (15-20 min delayed)")
        logger.info(f"{ticker}: >>> SUBPROCESS VERSION ACTIVE <<<")
        try:
            import json
            import subprocess
            import sys

            cmd = [
                sys.executable,
                "-m",
                "src.analytics.yahoo_options_subprocess",
                ticker,
                str(int(max_expiries)),
            ]

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
                )
                try:
                    stdout_data, stderr_data = proc.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
                    logger.warning(f"{ticker}: Yahoo options timed out after 3s (subprocess killed)")
                    return pd.DataFrame(), pd.DataFrame(), 0.0, "YAHOO"
            except Exception as e:
                logger.warning(f"{ticker}: Yahoo subprocess launch failed: {e}")
                return pd.DataFrame(), pd.DataFrame(), 0.0, "YAHOO"

            stdout = (stdout_data or "").strip()


            lines = [ln for ln in stdout.splitlines() if ln.strip()]
            last = lines[-1] if lines else stdout

            try:
                payload = json.loads(last)
            except Exception:
                logger.debug(f"{ticker}: Yahoo subprocess JSON parse failed. tail={last[-500:]}")
                return pd.DataFrame(), pd.DataFrame(), 0.0, "YAHOO"

            if not payload.get("ok"):
                logger.debug(f"{ticker}: Yahoo subprocess failed: {payload.get('error')}")
                return pd.DataFrame(), pd.DataFrame(), 0.0, "YAHOO"

            stock_price = float(payload.get("stock_price") or 0.0)

            calls_split = payload.get("calls_split")
            puts_split = payload.get("puts_split")

            if not calls_split or not puts_split:
                return pd.DataFrame(), pd.DataFrame(), stock_price, "YAHOO"

            from io import StringIO
            calls_df = pd.read_json(StringIO(calls_split), orient="split")
            puts_df = pd.read_json(StringIO(puts_split), orient="split")

            return calls_df, puts_df, stock_price, "YAHOO"

        except Exception as e:
            logger.warning(f"{ticker}: Yahoo fallback failed: {e}")
            return pd.DataFrame(), pd.DataFrame(), 0.0, "YAHOO"

    def detect_unusual_activity(self, ticker: str, calls_df: pd.DataFrame,
                                puts_df: pd.DataFrame, stock_price: float) -> List[OptionsAlert]:
        """Detect unusual options activity."""
        alerts = []

        if calls_df.empty and puts_df.empty:
            return alerts

        # Process calls
        for _, row in calls_df.iterrows():
            alert = self._check_option_unusual(ticker, row, 'CALL', stock_price)
            if alert:
                alerts.append(alert)

        # Process puts
        for _, row in puts_df.iterrows():
            alert = self._check_option_unusual(ticker, row, 'PUT', stock_price)
            if alert:
                alerts.append(alert)

        # Sort by severity
        severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        alerts.sort(key=lambda x: severity_order.get(x.severity, 3))

        return alerts

    def _check_option_unusual(self, ticker: str, row: pd.Series,
                              option_type: str, stock_price: float) -> Optional[OptionsAlert]:
        """Check if a single option contract has unusual activity."""

        # Helper to safely convert to number, handling NaN
        def safe_float(val, default=0):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            try:
                return float(val)
            except:
                return default

        def safe_int(val, default=0):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            try:
                return int(val)
            except:
                return default

        volume = safe_int(row.get('volume', 0), 0)
        oi = safe_int(row.get('openInterest', 0), 0)
        iv = safe_float(row.get('impliedVolatility', 0), 0)
        strike = safe_float(row.get('strike', 0), 0)
        expiry = row.get('expiry', '')
        days_to_exp = safe_int(row.get('daysToExpiry', 0), 0)

        # Skip if below minimum thresholds
        if volume < self.MIN_VOLUME_THRESHOLD:
            return None

        # Calculate metrics
        volume_oi_ratio = volume / oi if oi > 0 else float('inf') if volume > 0 else 0
        distance_from_strike = ((strike - stock_price) / stock_price * 100) if stock_price > 0 else 0

        # === MONEYNESS FILTER (Key improvement to reduce noise) ===
        # For CALL: distance > 0 means OTM, distance < 0 means ITM
        # For PUT: distance > 0 means ITM, distance < 0 means OTM
        if option_type == 'CALL':
            is_itm = distance_from_strike < 0
            itm_pct = abs(distance_from_strike) if is_itm else 0
            otm_pct = distance_from_strike if not is_itm else 0
        else:  # PUT
            is_itm = distance_from_strike > 0
            itm_pct = distance_from_strike if is_itm else 0
            otm_pct = abs(distance_from_strike) if not is_itm else 0

        # Filter out deep ITM options (often have distorted IV)
        if itm_pct > self.MAX_ITM_PCT:
            return None

        # Filter out very far OTM options (low relevance)
        if otm_pct > self.MAX_OTM_PCT:
            return None

        # Filter out extremely high IV (likely data errors or illiquid)
        if iv > self.IV_MAX_THRESHOLD:
            return None

        # Determine if unusual
        is_unusual = False
        alert_type = ''
        severity = 'LOW'
        description_parts = []

        # Check 1: Volume > Open Interest (new positions)
        if volume_oi_ratio > self.VOLUME_OI_THRESHOLD and oi >= self.MIN_OI_THRESHOLD:
            is_unusual = True
            alert_type = 'UNUSUAL_VOLUME'
            description_parts.append(f"Volume ({volume:,}) exceeds OI ({oi:,}) by {volume_oi_ratio:.1f}x")

            if volume_oi_ratio > 5:
                severity = 'HIGH'
            elif volume_oi_ratio > 2:
                severity = 'MEDIUM'

        # Check 2: High implied volatility
        if iv > self.IV_HIGH_THRESHOLD:
            is_unusual = True
            if not alert_type:
                alert_type = 'IV_SPIKE'
            description_parts.append(f"High IV: {iv * 100:.0f}%")

            if iv > 1.0:  # > 100% IV
                severity = 'HIGH'
            elif iv > 0.75:
                severity = 'MEDIUM' if severity != 'HIGH' else 'HIGH'

        # Check 3: Near-the-money with high volume
        if abs(distance_from_strike) < 5 and volume > 500:
            if not is_unusual:
                is_unusual = True
                alert_type = 'HIGH_VOLUME_ATM'
            description_parts.append(f"High ATM volume: {volume:,}")
            severity = 'MEDIUM' if severity != 'HIGH' else 'HIGH'

        if not is_unusual:
            return None

        # Determine direction
        if option_type == 'CALL':
            direction = 'BULLISH'
        else:
            direction = 'BEARISH'

        # Create alert
        return OptionsAlert(
            ticker=ticker,
            alert_type=alert_type,
            direction=direction,
            severity=severity,
            option_type=option_type,
            strike=strike,
            expiry=expiry,
            volume=volume,  # Already safe_int
            open_interest=oi,  # Already safe_int
            volume_oi_ratio=volume_oi_ratio,
            implied_volatility=iv,
            stock_price=stock_price,
            distance_from_strike_pct=distance_from_strike,
            days_to_expiry=days_to_exp,
            description="; ".join(description_parts)
        )

    def calculate_put_call_ratio(self, calls_df: pd.DataFrame,
                                 puts_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate put/call ratios."""

        call_volume = calls_df['volume'].fillna(0).sum() if not calls_df.empty else 0
        put_volume = puts_df['volume'].fillna(0).sum() if not puts_df.empty else 0
        call_oi = calls_df['openInterest'].fillna(0).sum() if not calls_df.empty else 0
        put_oi = puts_df['openInterest'].fillna(0).sum() if not puts_df.empty else 0

        return {
            'call_volume': int(call_volume),
            'put_volume': int(put_volume),
            'call_oi': int(call_oi),
            'put_oi': int(put_oi),
            'pc_volume_ratio': put_volume / call_volume if call_volume > 0 else 0,
            'pc_oi_ratio': put_oi / call_oi if call_oi > 0 else 0
        }

    def calculate_max_pain(self, calls_df: pd.DataFrame,
                           puts_df: pd.DataFrame) -> float:
        """
        Calculate max pain price.
        Max pain = strike where option holders lose the most money.
        """
        if calls_df.empty or puts_df.empty:
            return 0

        # Get all unique strikes
        all_strikes = sorted(set(
            list(calls_df['strike'].unique()) +
            list(puts_df['strike'].unique())
        ))

        if not all_strikes:
            return 0

        min_pain = float('inf')
        max_pain_strike = all_strikes[0]

        for test_strike in all_strikes:
            total_pain = 0

            # Pain for call holders (lose money if stock < strike)
            for _, row in calls_df.iterrows():
                strike = row['strike']
                oi = row.get('openInterest', 0) or 0
                if test_strike < strike:
                    # Calls expire worthless - loss = premium (approximated by OI * intrinsic loss)
                    total_pain += oi * (strike - test_strike)

            # Pain for put holders (lose money if stock > strike)
            for _, row in puts_df.iterrows():
                strike = row['strike']
                oi = row.get('openInterest', 0) or 0
                if test_strike > strike:
                    # Puts expire worthless
                    total_pain += oi * (test_strike - strike)

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_strike

        return max_pain_strike

    def analyze_ticker(self, ticker: str, skip_ibkr: bool = False) -> OptionsFlowSummary:
        """
        Full options flow analysis for a ticker.

        Args:
            ticker: Stock symbol
            skip_ibkr: If True, skip IBKR connection attempt (faster for scanner)

        Returns:
            OptionsFlowSummary with all metrics and alerts
        """
        from datetime import datetime

        # Get options data
        calls_df, puts_df, stock_price, data_source = self.get_options_chain(ticker, skip_ibkr=skip_ibkr)

        data_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        if calls_df.empty and puts_df.empty:
            return OptionsFlowSummary(
                ticker=ticker,
                stock_price=stock_price,
                total_call_volume=0,
                total_put_volume=0,
                total_call_oi=0,
                total_put_oi=0,
                put_call_volume_ratio=0,
                put_call_oi_ratio=0,
                overall_sentiment='NEUTRAL',
                sentiment_score=0,
                data_source=data_source,
                data_timestamp=data_timestamp,
            )

        # Get expiry context for max pain clarity
        expiry_dates = []
        if 'expiry' in calls_df.columns:
            expiry_dates = sorted(calls_df['expiry'].unique().tolist())
        elif 'expiry' in puts_df.columns:
            expiry_dates = sorted(puts_df['expiry'].unique().tolist())

        if expiry_dates:
            if len(expiry_dates) == 1:
                expiry_context = f"{expiry_dates[0]} expiry"
            else:
                expiry_context = f"aggregated ({expiry_dates[0]} to {expiry_dates[-1]})"
        else:
            expiry_context = "aggregated across near-term expiries"

        # Calculate ratios
        ratios = self.calculate_put_call_ratio(calls_df, puts_df)

        # Detect unusual activity
        alerts = self.detect_unusual_activity(ticker, calls_df, puts_df, stock_price)

        # Calculate max pain
        max_pain = self.calculate_max_pain(calls_df, puts_df)

        # Calculate IV metrics
        avg_call_iv = calls_df['impliedVolatility'].mean() if not calls_df.empty else 0
        avg_put_iv = puts_df['impliedVolatility'].mean() if not puts_df.empty else 0
        iv_skew = avg_put_iv - avg_call_iv

        # Determine overall sentiment
        pc_ratio = ratios['pc_volume_ratio']
        pc_oi_ratio = ratios['pc_oi_ratio']  # Also check OI ratio for guardrails
        bullish_alerts = sum(1 for a in alerts if a.direction == 'BULLISH' and a.severity in ['HIGH', 'MEDIUM'])
        bearish_alerts = sum(1 for a in alerts if a.direction == 'BEARISH' and a.severity in ['HIGH', 'MEDIUM'])

        if pc_ratio < self.PUT_CALL_BULLISH_THRESHOLD or bullish_alerts > bearish_alerts + 2:
            sentiment = 'BULLISH'
            sentiment_score = min(100, 50 + (0.5 - pc_ratio) * 100 + bullish_alerts * 10)
        elif pc_ratio > self.PUT_CALL_BEARISH_THRESHOLD or bearish_alerts > bullish_alerts + 2:
            sentiment = 'BEARISH'
            sentiment_score = max(-100, -50 - (pc_ratio - 1.5) * 50 - bearish_alerts * 10)
        else:
            sentiment = 'NEUTRAL'
            sentiment_score = (bullish_alerts - bearish_alerts) * 10

        # ==========================================
        # GUARDRAIL: Cap score when P/C ratios contradict sentiment
        # ==========================================
        # This prevents extreme scores when P/C ratios contradict alerts

        # BULLISH GUARDRAILS (cap high scores when indicators are bearish)
        # If P/C volume is near neutral (0.9-1.1) AND P/C OI is put-heavy (>1.0)
        if 0.9 <= pc_ratio <= 1.1 and pc_oi_ratio >= 1.0:
            if sentiment_score > 60:
                logger.info(f"{ticker}: Capping options sentiment from {sentiment_score:.0f} to 60 "
                           f"(P/C volume={pc_ratio:.2f} is neutral, OI={pc_oi_ratio:.2f} is put-heavy)")
                sentiment_score = 60
                sentiment = 'NEUTRAL'  # Downgrade from BULLISH

        # If P/C OI > 1.3 (strongly put-heavy), cap bullish scores
        if pc_oi_ratio > 1.3:
            if sentiment_score > 50:
                logger.info(f"{ticker}: Capping options sentiment from {sentiment_score:.0f} to 50 "
                           f"(P/C OI={pc_oi_ratio:.2f} indicates bearish positioning)")
                sentiment_score = 50
                if sentiment == 'BULLISH':
                    sentiment = 'NEUTRAL'

        # If P/C volume > 1.0 (more puts than calls), don't allow strong bullish
        if pc_ratio > 1.0:
            if sentiment_score > 55:
                logger.info(f"{ticker}: Capping options sentiment from {sentiment_score:.0f} to 55 "
                           f"(P/C volume={pc_ratio:.2f} > 1.0)")
                sentiment_score = 55
                if sentiment == 'BULLISH':
                    sentiment = 'NEUTRAL'

        # BEARISH GUARDRAILS (cap low scores when indicators are bullish)
        # If P/C volume < 0.8 (more calls than puts), don't allow extreme bearish
        if pc_ratio < 0.8:
            if sentiment_score < -50:
                logger.info(f"{ticker}: Raising options sentiment from {sentiment_score:.0f} to -50 "
                           f"(P/C volume={pc_ratio:.2f} is bullish)")
                sentiment_score = -50
                if sentiment == 'BEARISH':
                    sentiment = 'NEUTRAL'

        # If P/C volume < 0.6 (strongly call-heavy), floor at -20
        if pc_ratio < 0.6:
            if sentiment_score < -20:
                logger.info(f"{ticker}: Raising options sentiment from {sentiment_score:.0f} to -20 "
                           f"(P/C volume={pc_ratio:.2f} is strongly bullish)")
                sentiment_score = -20
                if sentiment == 'BEARISH':
                    sentiment = 'NEUTRAL'

        # If P/C OI < 0.8 (more call OI than put OI), don't allow extreme bearish
        if pc_oi_ratio < 0.8:
            if sentiment_score < -30:
                logger.info(f"{ticker}: Raising options sentiment from {sentiment_score:.0f} to -30 "
                           f"(P/C OI={pc_oi_ratio:.2f} is bullish)")
                sentiment_score = -30
                if sentiment == 'BEARISH':
                    sentiment = 'NEUTRAL'
        # ==========================================

        return OptionsFlowSummary(
            ticker=ticker,
            stock_price=stock_price,
            total_call_volume=ratios['call_volume'],
            total_put_volume=ratios['put_volume'],
            total_call_oi=ratios['call_oi'],
            total_put_oi=ratios['put_oi'],
            put_call_volume_ratio=ratios['pc_volume_ratio'],
            put_call_oi_ratio=ratios['pc_oi_ratio'],
            overall_sentiment=sentiment,
            sentiment_score=sentiment_score,
            alerts=alerts,
            max_pain_price=max_pain,
            max_pain_expiry_context=expiry_context,
            avg_call_iv=avg_call_iv,
            avg_put_iv=avg_put_iv,
            iv_skew=iv_skew,
            data_source=data_source,
            data_timestamp=data_timestamp,
        )

    def scan_universe(self, tickers: List[str], max_workers: int = 5) -> List[OptionsFlowSummary]:
        """
        Scan multiple tickers for unusual options activity.

        Args:
            tickers: List of ticker symbols
            max_workers: Number of parallel threads

        Returns:
            List of OptionsFlowSummary sorted by alert count
        """
        results = []

        logger.info(f"Scanning options flow for {len(tickers)} tickers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.analyze_ticker, t): t for t in tickers}

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result.alerts or result.overall_sentiment != 'NEUTRAL':
                        results.append(result)
                except Exception as e:
                    logger.debug(f"Error analyzing {ticker}: {e}")

        # Sort by number of alerts (most active first)
        results.sort(key=lambda x: len(x.alerts), reverse=True)

        logger.info(f"Found {len(results)} tickers with notable options activity")

        return results

    def save_alerts_to_db(self, summary: OptionsFlowSummary) -> int:
        """
        Save options flow alerts to database.

        Args:
            summary: OptionsFlowSummary from analyze_ticker

        Returns:
            Number of alerts saved
        """
        if not DB_AVAILABLE:
            logger.warning("Database not available - cannot save alerts")
            return 0

        if not summary.alerts:
            return 0

        # Helper to convert numpy types to Python native types
        def to_native(val):
            if val is None:
                return None
            if hasattr(val, 'item'):  # numpy scalar
                return val.item()
            return val

        saved = 0
        today = date.today()

        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    for alert in summary.alerts:
                        try:
                            cur.execute("""
                                        INSERT INTO options_flow_alerts (ticker, alert_date, alert_time,
                                                                         alert_type, direction, severity,
                                                                         option_type, strike, expiry,
                                                                         volume, open_interest, volume_oi_ratio,
                                                                         implied_volatility,
                                                                         stock_price, distance_from_strike_pct,
                                                                         days_to_expiry,
                                                                         description,
                                                                         put_call_volume_ratio, put_call_oi_ratio,
                                                                         overall_sentiment, sentiment_score,
                                                                         max_pain_price)
                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                                %s, %s, %s, %s) ON CONFLICT (ticker, alert_date, option_type, strike, expiry, alert_type)
                                DO
                                        UPDATE SET
                                            volume = EXCLUDED.volume,
                                            open_interest = EXCLUDED.open_interest,
                                            volume_oi_ratio = EXCLUDED.volume_oi_ratio,
                                            implied_volatility = EXCLUDED.implied_volatility,
                                            alert_time = EXCLUDED.alert_time
                                        """, (
                                            alert.ticker,
                                            today,
                                            datetime.now(),
                                            alert.alert_type,
                                            alert.direction,
                                            alert.severity,
                                            alert.option_type,
                                            to_native(alert.strike),
                                            alert.expiry,
                                            to_native(alert.volume),
                                            to_native(alert.open_interest),
                                            to_native(alert.volume_oi_ratio),
                                            to_native(alert.implied_volatility),
                                            to_native(alert.stock_price),
                                            to_native(alert.distance_from_strike_pct),
                                            to_native(alert.days_to_expiry),
                                            alert.description,
                                            to_native(summary.put_call_volume_ratio),
                                            to_native(summary.put_call_oi_ratio),
                                            summary.overall_sentiment,
                                            to_native(summary.sentiment_score),
                                            to_native(summary.max_pain_price)
                                        ))
                            saved += 1
                        except Exception as e:
                            logger.debug(f"Error saving alert: {e}")

                    conn.commit()

            logger.info(f"Saved {saved} alerts for {summary.ticker}")
            return saved

        except Exception as e:
            logger.error(f"Error saving alerts to DB: {e}")
            return 0

    def save_daily_summary(self, summary: OptionsFlowSummary) -> bool:
        """
        Save daily options flow summary to database.

        Args:
            summary: OptionsFlowSummary from analyze_ticker

        Returns:
            True if saved successfully
        """
        if not DB_AVAILABLE:
            return False

        today = date.today()
        high_alerts = sum(1 for a in summary.alerts if a.severity == 'HIGH')
        medium_alerts = sum(1 for a in summary.alerts if a.severity == 'MEDIUM')
        low_alerts = sum(1 for a in summary.alerts if a.severity == 'LOW')

        # Helper to convert numpy types to Python native types
        def to_native(val):
            if val is None:
                return None
            if hasattr(val, 'item'):  # numpy scalar
                return val.item()
            return val

        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                                INSERT INTO options_flow_daily (ticker, scan_date,
                                                                stock_price, total_call_volume, total_put_volume,
                                                                total_call_oi, total_put_oi,
                                                                put_call_volume_ratio, put_call_oi_ratio,
                                                                avg_call_iv, avg_put_iv, iv_skew,
                                                                overall_sentiment, sentiment_score, max_pain_price,
                                                                high_alerts, medium_alerts, low_alerts)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (ticker, scan_date)
                        DO
                                UPDATE SET
                                    stock_price = EXCLUDED.stock_price,
                                    total_call_volume = EXCLUDED.total_call_volume,
                                    total_put_volume = EXCLUDED.total_put_volume,
                                    put_call_volume_ratio = EXCLUDED.put_call_volume_ratio,
                                    overall_sentiment = EXCLUDED.overall_sentiment,
                                    sentiment_score = EXCLUDED.sentiment_score,
                                    high_alerts = EXCLUDED.high_alerts,
                                    medium_alerts = EXCLUDED.medium_alerts,
                                    low_alerts = EXCLUDED.low_alerts
                                """, (
                                    summary.ticker,
                                    today,
                                    to_native(summary.stock_price),
                                    to_native(summary.total_call_volume),
                                    to_native(summary.total_put_volume),
                                    to_native(summary.total_call_oi),
                                    to_native(summary.total_put_oi),
                                    to_native(summary.put_call_volume_ratio),
                                    to_native(summary.put_call_oi_ratio),
                                    to_native(summary.avg_call_iv),
                                    to_native(summary.avg_put_iv),
                                    to_native(summary.iv_skew),
                                    summary.overall_sentiment,
                                    to_native(summary.sentiment_score),
                                    to_native(summary.max_pain_price),
                                    high_alerts,
                                    medium_alerts,
                                    low_alerts
                                ))
                    conn.commit()

            return True

        except Exception as e:
            logger.error(f"Error saving daily summary: {e}")
            return False

    def analyze_and_save(self, ticker: str) -> OptionsFlowSummary:
        """
        Analyze ticker and save results to database.

        Args:
            ticker: Stock symbol

        Returns:
            OptionsFlowSummary
        """
        summary = self.analyze_ticker(ticker)

        if summary and DB_AVAILABLE:
            self.save_alerts_to_db(summary)
            self.save_daily_summary(summary)

        return summary

    def scan_and_save(self, tickers: List[str], max_workers: int = 5) -> List[OptionsFlowSummary]:
        """
        Scan multiple tickers and save all results to database.

        Args:
            tickers: List of ticker symbols
            max_workers: Number of parallel threads

        Returns:
            List of OptionsFlowSummary with notable activity
        """
        results = []
        total_alerts_saved = 0

        logger.info(f"Scanning and saving options flow for {len(tickers)} tickers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.analyze_and_save, t): t for t in tickers}

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result.alerts or result.overall_sentiment != 'NEUTRAL':
                        results.append(result)
                        total_alerts_saved += len(result.alerts)
                except Exception as e:
                    logger.debug(f"Error analyzing {ticker}: {e}")

        results.sort(key=lambda x: len(x.alerts), reverse=True)

        logger.info(f"Scan complete: {len(results)} tickers with activity, {total_alerts_saved} alerts saved")

        return results

    @staticmethod
    def get_recent_alerts(days: int = 7, severity: str = None, ticker: str = None) -> pd.DataFrame:
        """
        Get recent alerts from database.

        Args:
            days: Number of days to look back
            severity: Filter by severity ('HIGH', 'MEDIUM', 'LOW')
            ticker: Filter by ticker

        Returns:
            DataFrame of recent alerts
        """
        if not DB_AVAILABLE:
            return pd.DataFrame()

        query = """
                SELECT ticker, \
                       alert_date, \
                       alert_type, \
                       direction, \
                       severity, \
                       option_type, \
                       strike, \
                       expiry, \
                       volume, \
                       open_interest, \
                       volume_oi_ratio, \
                       implied_volatility * 100 as iv_pct, \
                       stock_price, \
                       overall_sentiment, \
                       sentiment_score, \
                       description
                FROM options_flow_alerts
                WHERE alert_date >= CURRENT_DATE - INTERVAL '%s days' \
                """
        params = [days]

        if severity:
            query += " AND severity = %s"
            params.append(severity)

        if ticker:
            query += " AND ticker = %s"
            params.append(ticker.upper())

        query += " ORDER BY alert_date DESC, severity ASC, volume DESC"

        try:
            # Use string formatting for interval (safe since days is int)
            final_query = query % (days,) if not severity and not ticker else query
            return pd.read_sql(final_query, get_engine(), params=params[1:] if params[1:] else None)
        except Exception as e:
            logger.error(f"Error fetching alerts: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_sentiment_history(ticker: str, days: int = 30) -> pd.DataFrame:
        """
        Get historical sentiment for a ticker.

        Args:
            ticker: Stock symbol
            days: Number of days

        Returns:
            DataFrame with daily sentiment history
        """
        if not DB_AVAILABLE:
            return pd.DataFrame()

        query = """
                SELECT scan_date, \
                       stock_price, \
                       put_call_volume_ratio, \
                       overall_sentiment, \
                       sentiment_score, \
                       max_pain_price, \
                       high_alerts, \
                       medium_alerts
                FROM options_flow_daily
                WHERE ticker = %s
                  AND scan_date >= CURRENT_DATE - INTERVAL '%s days'
                ORDER BY scan_date ASC \
                """

        try:
            return pd.read_sql(query % ('%s', days), get_engine(), params=[ticker.upper()])
        except Exception as e:
            logger.error(f"Error fetching sentiment history: {e}")
            return pd.DataFrame()

    def get_flow_for_ai(self, ticker: str) -> str:
        """
        Get options flow summary formatted for AI context.

        Args:
            ticker: Stock symbol

        Returns:
            Formatted string for AI consumption
        """
        summary = self.analyze_ticker(ticker)

        lines = [
            f"\n{'=' * 50}",
            f"OPTIONS FLOW ANALYSIS: {ticker}",
            f"{'=' * 50}",
            f"Stock Price: ${summary.stock_price:.2f}",
            f"Max Pain: ${summary.max_pain_price:.2f} ({summary.max_pain_expiry_context})",
            f"Data Source: {summary.data_source}{' - ' + summary.data_timestamp if summary.data_timestamp else ''}",
            f"",
            f"📊 VOLUME & OPEN INTEREST:",
            f"   Call Volume: {summary.total_call_volume:,}",
            f"   Put Volume: {summary.total_put_volume:,}",
            f"   Call OI: {summary.total_call_oi:,}",
            f"   Put OI: {summary.total_put_oi:,}",
            f"",
            f"📈 RATIOS (volume-based, platform data):",
            f"   Put/Call Volume: {summary.put_call_volume_ratio:.2f}",
            f"   Put/Call OI: {summary.put_call_oi_ratio:.2f}",
            f"   IV Skew: {summary.iv_skew * 100:.1f}%",
            f"",
            f"🎯 SENTIMENT: {summary.overall_sentiment} (score: {summary.sentiment_score:.0f})",
            f"⚠️ Note: Ratios reflect volume, not trade size. Cannot confirm institutional positioning.",
        ]

        if summary.alerts:
            lines.append(f"\n⚠️ UNUSUAL ACTIVITY ({len(summary.alerts)} alerts):")
            for alert in summary.alerts[:5]:  # Top 5 alerts
                lines.append(
                    f"   [{alert.severity}] {alert.option_type} ${alert.strike} {alert.expiry}: "
                    f"{alert.description}"
                )
        else:
            lines.append(f"\n✅ No unusual activity detected")

        return "\n".join(lines)


# Convenience functions
def get_options_flow_analyzer() -> OptionsFlowAnalyzer:
    """Get options flow analyzer instance."""
    return OptionsFlowAnalyzer()


def analyze_options_flow(ticker: str) -> OptionsFlowSummary:
    """Quick analysis for a single ticker."""
    analyzer = OptionsFlowAnalyzer()
    return analyzer.analyze_ticker(ticker)


if __name__ == "__main__":
    # Test
    analyzer = OptionsFlowAnalyzer()

    # Test single ticker
    print("\n=== Testing AAPL Options Flow ===")
    summary = analyzer.analyze_ticker("AAPL")
    print(analyzer.get_flow_for_ai("AAPL"))

    # Test universe scan
    print("\n=== Scanning Top Tickers ===")
    test_tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD"]
    results = analyzer.scan_universe(test_tickers)

    for r in results[:3]:
        print(f"\n{r.ticker}: {r.overall_sentiment} - {len(r.alerts)} alerts")