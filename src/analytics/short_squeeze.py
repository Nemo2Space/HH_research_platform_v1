"""
Short Squeeze Detector - FIXED VERSION

FIXES APPLIED:
- RSI calculation returns Optional[float], None on error (not 50)
- All numeric fields are Optional
- Data availability is tracked explicitly

Author: Alpha Research Platform
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.logging import get_logger

try:
    from src.analytics.options_flow import OptionsFlowAnalyzer

    OPTIONS_FLOW_AVAILABLE = True
except ImportError:
    OPTIONS_FLOW_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class ShortSqueezeData:
    """Short squeeze analysis data - FIXED with proper Optional handling."""
    ticker: str

    # Price info - all Optional
    current_price: Optional[float] = None
    price_change_5d: Optional[float] = None
    price_change_1m: Optional[float] = None

    # Short interest data - all Optional
    short_interest: Optional[int] = None
    short_percent_of_float: Optional[float] = None
    short_percent_of_shares: Optional[float] = None
    days_to_cover: Optional[float] = None
    short_prior_month: Optional[int] = None
    short_change_mom: Optional[float] = None

    # Float and volume - all Optional
    float_shares: Optional[int] = None
    shares_outstanding: Optional[int] = None
    avg_volume: Optional[int] = None
    avg_volume_10d: Optional[int] = None
    relative_volume: Optional[float] = None

    # Options flow
    options_sentiment: str = "NOT_ANALYZED"
    options_sentiment_score: Optional[float] = None
    put_call_ratio: Optional[float] = None
    unusual_call_activity: Optional[bool] = None
    call_volume: Optional[int] = None
    put_volume: Optional[int] = None

    # Technical indicators - FIXED: Optional instead of 0/False
    rsi_14: Optional[float] = None
    above_20ma: Optional[bool] = None
    above_50ma: Optional[bool] = None
    near_52w_high: Optional[bool] = None

    # Squeeze score - Optional when insufficient data
    squeeze_score: Optional[float] = None
    squeeze_risk: str = "UNKNOWN"

    # Analysis
    squeeze_factors: List[str] = field(default_factory=list)
    warning_factors: List[str] = field(default_factory=list)

    # Data quality tracking
    data_available: Dict[str, bool] = field(default_factory=dict)
    calculation_errors: List[str] = field(default_factory=list)

    @property
    def has_sufficient_data(self) -> bool:
        return self.short_percent_of_float is not None and self.current_price is not None

    @property
    def score_display(self) -> str:
        if self.squeeze_score is None:
            return "N/A"
        missing = self.options_sentiment == "NOT_ANALYZED" or self.rsi_14 is None
        return f"{self.squeeze_score:.0f}{'*' if missing else ''}"


class ShortSqueezeDetector:
    """Detects stocks with high short squeeze potential."""

    HIGH_SHORT_INTEREST = 15
    EXTREME_SHORT_INTEREST = 25
    HIGH_DAYS_TO_COVER = 5
    EXTREME_DAYS_TO_COVER = 10

    def __init__(self):
        self.options_analyzer = OptionsFlowAnalyzer() if OPTIONS_FLOW_AVAILABLE else None

    def analyze_ticker(self, ticker: str) -> ShortSqueezeData:
        """Analyze a single ticker for short squeeze potential."""
        ticker = ticker.upper()
        data = ShortSqueezeData(ticker=ticker)

        try:
            # Use subprocess wrapper to avoid curl_cffi/Streamlit deadlock
            from src.analytics.yf_subprocess import get_stock_info_and_history
            _yf_data = get_stock_info_and_history(ticker, history_period="3mo")
            info = _yf_data.get("info", {})
            _hist_from_subprocess = _yf_data.get("history")

            if not info:
                data.calculation_errors.append("Could not fetch stock info")
                return data

            # Basic info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if current_price is not None:
                data.current_price = float(current_price)
                data.data_available['current_price'] = True

            data.shares_outstanding = info.get('sharesOutstanding')
            data.float_shares = info.get('floatShares')
            data.avg_volume = info.get('averageVolume')
            data.avg_volume_10d = info.get('averageVolume10days')

            today_volume = info.get('volume')
            if today_volume and data.avg_volume and data.avg_volume > 0:
                data.relative_volume = today_volume / data.avg_volume

            # Short interest
            data.short_interest = info.get('sharesShort')
            data.short_prior_month = info.get('sharesShortPriorMonth')

            short_pct = info.get('shortPercentOfFloat')
            if short_pct is not None:
                data.short_percent_of_float = short_pct * 100 if 0 < short_pct < 1 else float(short_pct)
                data.data_available['short_percent_of_float'] = True
            else:
                data.data_available['short_percent_of_float'] = False
                data.calculation_errors.append("Short interest data not available")

            if data.short_interest and data.shares_outstanding and data.shares_outstanding > 0:
                data.short_percent_of_shares = (data.short_interest / data.shares_outstanding) * 100

            if data.avg_volume and data.avg_volume > 0 and data.short_interest:
                data.days_to_cover = data.short_interest / data.avg_volume

            if data.short_prior_month and data.short_prior_month > 0 and data.short_interest:
                data.short_change_mom = ((data.short_interest - data.short_prior_month) / data.short_prior_month) * 100

            # Price data
            try:
                hist = _hist_from_subprocess
                if hist is not None and len(hist) > 0:
                    current = hist['Close'].iloc[-1]

                    if len(hist) >= 5:
                        price_5d_ago = hist['Close'].iloc[-5]
                        if price_5d_ago > 0:
                            data.price_change_5d = ((current - price_5d_ago) / price_5d_ago) * 100

                    if len(hist) >= 21:
                        price_1m_ago = hist['Close'].iloc[-21]
                        if price_1m_ago > 0:
                            data.price_change_1m = ((current - price_1m_ago) / price_1m_ago) * 100

                    # RSI - FIXED: returns None on error
                    data.rsi_14 = self._calculate_rsi(hist['Close'], 14)
                    data.data_available['rsi'] = data.rsi_14 is not None

                    if len(hist) >= 20:
                        ma_20 = hist['Close'].iloc[-20:].mean()
                        data.above_20ma = current > ma_20

                    if len(hist) >= 50:
                        ma_50 = hist['Close'].iloc[-50:].mean()
                        data.above_50ma = current > ma_50

                    high_52w = info.get('fiftyTwoWeekHigh')
                    if high_52w and high_52w > 0:
                        pct_from_high = ((high_52w - current) / high_52w) * 100
                        data.near_52w_high = pct_from_high <= 10
            except Exception as e:
                data.calculation_errors.append(f"Price history error: {str(e)[:50]}")

            # Options flow
            if self.options_analyzer:
                try:
                    options_data = self.options_analyzer.analyze_ticker(ticker, skip_ibkr=True)
                    if options_data:
                        data.options_sentiment = options_data.overall_sentiment or "NEUTRAL"
                        data.options_sentiment_score = options_data.sentiment_score
                        data.put_call_ratio = options_data.put_call_volume_ratio
                        data.call_volume = options_data.total_call_volume
                        data.put_volume = options_data.total_put_volume
                        data.data_available['options'] = True
                        if options_data.alerts:
                            high_calls = [a for a in options_data.alerts if
                                          a.option_type == 'CALL' and a.severity == 'HIGH']
                            data.unusual_call_activity = len(high_calls) > 0
                    else:
                        data.options_sentiment = "NO_DATA"
                        data.data_available['options'] = False
                except Exception as e:
                    data.options_sentiment = "ERROR"
                    data.data_available['options'] = False
            else:
                data.options_sentiment = "NOT_AVAILABLE"
                data.data_available['options'] = False

            # Calculate score
            data = self._calculate_squeeze_score(data)

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            data.calculation_errors.append(f"General error: {str(e)[:100]}")

        return data

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate RSI - FIXED: returns None on error instead of 50."""
        try:
            if prices is None or len(prices) < period + 1:
                return None

            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            if loss.iloc[-1] == 0:
                return 100.0 if gain.iloc[-1] > 0 else None

            rs = gain.iloc[-1] / loss.iloc[-1]
            rsi = 100 - (100 / (1 + rs))

            return float(rsi) if not pd.isna(rsi) else None
        except:
            return None

    def _calculate_squeeze_score(self, data: ShortSqueezeData) -> ShortSqueezeData:
        """Calculate squeeze score - FIXED: returns None if insufficient data."""

        if not data.has_sufficient_data:
            data.squeeze_score = None
            data.squeeze_risk = "UNKNOWN"
            data.warning_factors.append("Insufficient data to calculate squeeze score")
            return data

        score = 0
        factors = []
        warnings = []
        max_possible = 0

        # Short interest (0-35 points)
        if data.short_percent_of_float is not None:
            max_possible += 35
            if data.short_percent_of_float >= self.EXTREME_SHORT_INTEREST:
                score += 30
                factors.append(f"🔥 Extreme short interest: {data.short_percent_of_float:.1f}%")
            elif data.short_percent_of_float >= self.HIGH_SHORT_INTEREST:
                score += 20
                factors.append(f"⚠️ High short interest: {data.short_percent_of_float:.1f}%")
            elif data.short_percent_of_float >= 10:
                score += 10
            elif data.short_percent_of_float >= 5:
                score += 5
            else:
                warnings.append(f"Low short interest: {data.short_percent_of_float:.1f}%")

            if data.short_change_mom is not None and data.short_change_mom > 10:
                score += 5
                factors.append(f"📈 Short interest increasing: +{data.short_change_mom:.1f}%")

        # Days to cover (0-20 points)
        if data.days_to_cover is not None:
            max_possible += 20
            if data.days_to_cover >= self.EXTREME_DAYS_TO_COVER:
                score += 20
                factors.append(f"🔥 Extreme days to cover: {data.days_to_cover:.1f}")
            elif data.days_to_cover >= self.HIGH_DAYS_TO_COVER:
                score += 15
            elif data.days_to_cover >= 3:
                score += 8
            elif data.days_to_cover >= 1:
                score += 3

        # Options flow (0-25 points)
        if data.options_sentiment in ["BULLISH", "BEARISH", "NEUTRAL"]:
            max_possible += 25
            if data.options_sentiment == "BULLISH":
                score += 10
                factors.append("📞 Bullish options flow")
                if data.unusual_call_activity:
                    score += 10
                    factors.append("🚀 Unusual call activity")
                if data.put_call_ratio and data.put_call_ratio < 0.5:
                    score += 5
            elif data.options_sentiment == "BEARISH":
                score -= 5
                warnings.append("Bearish options flow")

        # Price momentum (0-15 points)
        if data.price_change_5d is not None:
            max_possible += 15
            if data.price_change_5d > 10:
                score += 10
                factors.append(f"🚀 Strong momentum: +{data.price_change_5d:.1f}%")
            elif data.price_change_5d > 5:
                score += 6
            elif data.price_change_5d > 0:
                score += 3
            elif data.price_change_5d < -10:
                warnings.append(f"Negative momentum: {data.price_change_5d:.1f}%")

            if data.relative_volume and data.relative_volume > 2:
                score += 5
                factors.append(f"📊 Volume surge: {data.relative_volume:.1f}x")

        # Technical setup (0-15 points)
        if data.above_20ma is not None or data.above_50ma is not None:
            max_possible += 15
            if data.above_20ma and data.above_50ma:
                score += 8
                factors.append("Above key MAs")
            elif data.above_20ma:
                score += 4

        if data.near_52w_high:
            score += 5
            factors.append("Near 52-week high")

        if data.rsi_14 is not None:
            if 50 <= data.rsi_14 <= 70:
                score += 2
            elif data.rsi_14 > 70:
                warnings.append(f"Overbought RSI: {data.rsi_14:.0f}")

        # Normalize and finalize
        if max_possible > 0:
            data.squeeze_score = min(100, max(0, (score / max_possible) * 100))
        else:
            data.squeeze_score = None
            data.squeeze_risk = "UNKNOWN"
            data.warning_factors = warnings
            data.squeeze_factors = factors
            return data

        if data.squeeze_score >= 70:
            data.squeeze_risk = "EXTREME"
        elif data.squeeze_score >= 50:
            data.squeeze_risk = "HIGH"
        elif data.squeeze_score >= 30:
            data.squeeze_risk = "MEDIUM"
        else:
            data.squeeze_risk = "LOW"

        data.squeeze_factors = factors
        data.warning_factors = warnings
        return data

    def scan_universe(self, tickers: List[str], max_workers: int = 5) -> List[ShortSqueezeData]:
        """Scan multiple tickers for short squeeze potential."""
        results = []
        logger.info(f"Scanning {len(tickers)} tickers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.analyze_ticker, t): t for t in tickers}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result.squeeze_score is not None or result.short_percent_of_float is not None:
                        results.append(result)
                except Exception as e:
                    logger.debug(f"Error: {e}")

        results.sort(key=lambda x: x.squeeze_score if x.squeeze_score is not None else -1, reverse=True)
        return results

    def get_squeeze_report(self, ticker: str) -> str:
        """Get formatted squeeze report."""
        data = self.analyze_ticker(ticker)
        lines = [f"=== SHORT SQUEEZE: {ticker} ===", ""]

        if data.squeeze_score is not None:
            lines.append(f"Score: {data.squeeze_score:.0f}/100 ({data.squeeze_risk})")
        else:
            lines.append(f"Score: N/A ({data.squeeze_risk})")

        lines.append("")
        lines.append(
            f"Short % Float: {data.short_percent_of_float:.1f}%" if data.short_percent_of_float else "Short % Float: N/A")
        lines.append(f"Days to Cover: {data.days_to_cover:.1f}" if data.days_to_cover else "Days to Cover: N/A")
        lines.append(f"RSI(14): {data.rsi_14:.1f}" if data.rsi_14 else "RSI(14): N/A")
        lines.append(f"Options: {data.options_sentiment}")
        lines.append("")

        if data.squeeze_factors:
            lines.append("BULLISH:")
            for f in data.squeeze_factors:
                lines.append(f"  • {f}")
        if data.warning_factors:
            lines.append("WARNINGS:")
            for w in data.warning_factors:
                lines.append(f"  • {w}")
        if data.calculation_errors:
            lines.append("ERRORS:")
            for e in data.calculation_errors:
                lines.append(f"  ⚠️ {e}")

        return "\n".join(lines)


_squeeze_detector: Optional[ShortSqueezeDetector] = None


def get_squeeze_detector() -> ShortSqueezeDetector:
    global _squeeze_detector
    if _squeeze_detector is None:
        _squeeze_detector = ShortSqueezeDetector()
    return _squeeze_detector