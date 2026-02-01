"""
Telegram Alerts Module

Sends trading alerts to Telegram:
- Signal changes (BUY → SELL)
- Regime changes (Risk-On ↔ Risk-Off)
- Earnings approaching
- Bond opportunities
- Trade ideas
- Portfolio risk alerts

Author: Alpha Research Platform
"""

import os
import json
import requests
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from src.analytics.bond_signals_analytics import get_all_bond_signals
from src.utils.logging import get_logger
from src.db.connection import get_engine, get_connection

logger = get_logger(__name__)


class AlertType(Enum):
    """Types of alerts."""
    SIGNAL_CHANGE = "SIGNAL_CHANGE"
    REGIME_CHANGE = "REGIME_CHANGE"
    EARNINGS_ALERT = "EARNINGS_ALERT"
    BOND_OPPORTUNITY = "BOND_OPPORTUNITY"
    TRADE_IDEA = "TRADE_IDEA"
    PORTFOLIO_RISK = "PORTFOLIO_RISK"
    DAILY_SUMMARY = "DAILY_SUMMARY"
    CUSTOM = "CUSTOM"


@dataclass
class Alert:
    """Alert data structure."""
    alert_type: AlertType
    title: str
    message: str
    ticker: str = None
    priority: str = "MEDIUM"  # HIGH, MEDIUM, LOW
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class TelegramAlerter:
    """
    Sends alerts via Telegram Bot API.
    """

    def __init__(self, bot_token: str = None, chat_id: str = None):
        """
        Initialize Telegram alerter.

        Args:
            bot_token: Telegram bot API token
            chat_id: Telegram chat ID to send messages to
        """
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')

        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("Telegram alerter initialized")

        self.engine = get_engine()
        self._last_regime = None
        self._last_signals = {}

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message via Telegram.

        Args:
            text: Message text (supports HTML formatting)
            parse_mode: Parse mode (HTML or Markdown)

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.warning("Telegram not enabled - message not sent")
            print(f"[TELEGRAM DISABLED] Would send:\n{text}")
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode
            }

            response = requests.post(url, json=payload, timeout=10)

            if response.status_code == 200:
                logger.info("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False

    def send_alert(self, alert: Alert) -> bool:
        """
        Send a formatted alert.

        Args:
            alert: Alert object

        Returns:
            True if sent successfully
        """
        # Format based on alert type
        emoji_map = {
            AlertType.SIGNAL_CHANGE: "🔔",
            AlertType.REGIME_CHANGE: "🌍",
            AlertType.EARNINGS_ALERT: "📊",
            AlertType.BOND_OPPORTUNITY: "🏦",
            AlertType.TRADE_IDEA: "💡",
            AlertType.PORTFOLIO_RISK: "⚠️",
            AlertType.DAILY_SUMMARY: "📋",
            AlertType.CUSTOM: "📌",
        }

        priority_emoji = {
            "HIGH": "🔴",
            "MEDIUM": "🟡",
            "LOW": "🟢"
        }

        emoji = emoji_map.get(alert.alert_type, "📌")
        priority = priority_emoji.get(alert.priority, "")

        text = f"""{emoji} <b>{alert.title}</b> {priority}

{alert.message}

<i>{alert.timestamp.strftime('%Y-%m-%d %H:%M')}</i>"""

        return self.send_message(text)

    # ================================================================
    # SIGNAL CHANGE ALERTS
    # ================================================================

    def check_signal_changes(self) -> List[Alert]:
        """
        Check for signal changes since last check.

        Returns:
            List of alerts for changed signals
        """
        alerts = []

        try:
            query = """
                    SELECT DISTINCT \
                    ON (ticker)
                        ticker, signal_type, signal_strength, date
                    FROM trading_signals
                    WHERE date >= CURRENT_DATE - INTERVAL '1 day'
                    ORDER BY ticker, date DESC \
                    """

            import pandas as pd
            df = pd.read_sql(query, self.engine)

            for _, row in df.iterrows():
                ticker = row['ticker']
                new_signal = row['signal_type']

                if ticker in self._last_signals:
                    old_signal = self._last_signals[ticker]

                    # Check for significant change
                    if self._is_significant_change(old_signal, new_signal):
                        alert = Alert(
                            alert_type=AlertType.SIGNAL_CHANGE,
                            title=f"{ticker} Signal Change",
                            message=f"<b>{ticker}</b> changed from <code>{old_signal}</code> → <code>{new_signal}</code>\n\nStrength: {row['signal_strength']:.0f}",
                            ticker=ticker,
                            priority="HIGH" if new_signal in ['STRONG_BUY', 'STRONG_SELL'] else "MEDIUM"
                        )
                        alerts.append(alert)

                self._last_signals[ticker] = new_signal

        except Exception as e:
            logger.error(f"Error checking signal changes: {e}")

        return alerts

    def _is_significant_change(self, old: str, new: str) -> bool:
        """Check if signal change is significant enough to alert."""
        # Alert on direction changes
        buy_signals = ['BUY', 'STRONG_BUY']
        sell_signals = ['SELL', 'STRONG_SELL']

        old_is_buy = old in buy_signals
        new_is_buy = new in buy_signals
        old_is_sell = old in sell_signals
        new_is_sell = new in sell_signals

        # Direction reversal
        if (old_is_buy and new_is_sell) or (old_is_sell and new_is_buy):
            return True

        # From HOLD to strong signal
        if old == 'HOLD' and new in ['STRONG_BUY', 'STRONG_SELL']:
            return True

        return False

    def alert_signal_change(self, ticker: str, old_signal: str, new_signal: str,
                            strength: float = None) -> bool:
        """
        Send alert for a specific signal change.

        Args:
            ticker: Stock ticker
            old_signal: Previous signal
            new_signal: New signal
            strength: Signal strength
        """
        direction = "📈" if new_signal in ['BUY', 'STRONG_BUY'] else "📉" if new_signal in ['SELL',
                                                                                          'STRONG_SELL'] else "➡️"

        message = f"""{direction} <b>{ticker}</b>

Signal: <code>{old_signal}</code> → <code>{new_signal}</code>
{f"Strength: {strength:.0f}/100" if strength else ""}

Check the platform for details."""

        alert = Alert(
            alert_type=AlertType.SIGNAL_CHANGE,
            title=f"{ticker} Signal: {new_signal}",
            message=message,
            ticker=ticker,
            priority="HIGH"
        )

        return self.send_alert(alert)

    # ================================================================
    # REGIME CHANGE ALERTS
    # ================================================================

    def check_regime_change(self) -> Optional[Alert]:
        """
        Check if macro regime has changed.

        Returns:
            Alert if regime changed, None otherwise
        """
        try:
            from src.analytics.macro_regime import get_current_regime

            current = get_current_regime()
            current_regime = current.regime.value

            if self._last_regime and self._last_regime != current_regime:
                alert = Alert(
                    alert_type=AlertType.REGIME_CHANGE,
                    title="Macro Regime Change",
                    message=f"""Regime shifted from <code>{self._last_regime}</code> → <code>{current_regime}</code>

Score: {current.regime_score}/100
Confidence: {current.confidence:.0%}

<b>Implications:</b>
Growth stocks: {current.growth_adjustment:+d} points
Defensive stocks: {current.defensive_adjustment:+d} points""",
                    priority="HIGH"
                )

                self._last_regime = current_regime
                return alert

            self._last_regime = current_regime
            return None

        except Exception as e:
            logger.error(f"Error checking regime: {e}")
            return None

    def alert_regime_change(self, old_regime: str, new_regime: str,
                            score: int, details: str = "") -> bool:
        """Send regime change alert."""
        emoji = "🟢" if "RISK_ON" in new_regime else "🔴" if "RISK_OFF" in new_regime else "🟡"

        message = f"""{emoji} Regime: <code>{old_regime}</code> → <code>{new_regime}</code>

Score: {score}/100
{details}

<b>Action:</b> Review portfolio allocation"""

        alert = Alert(
            alert_type=AlertType.REGIME_CHANGE,
            title="🌍 Macro Regime Shift",
            message=message,
            priority="HIGH"
        )

        return self.send_alert(alert)

    # ================================================================
    # EARNINGS ALERTS
    # ================================================================

    def check_upcoming_earnings(self, days_ahead: int = 3) -> List[Alert]:
        """
        Check for earnings in the next N days.

        Args:
            days_ahead: Days to look ahead

        Returns:
            List of earnings alerts
        """
        alerts = []

        try:
            from src.analytics.earnings_analyzer import get_upcoming_earnings

            # Get portfolio tickers
            query = "SELECT DISTINCT ticker FROM screener_scores WHERE date >= CURRENT_DATE - INTERVAL '7 days'"
            import pandas as pd
            df = pd.read_sql(query, self.engine)
            tickers = df['ticker'].tolist() if not df.empty else None

            upcoming = get_upcoming_earnings(tickers, days_ahead)

            for item in upcoming:
                alert = Alert(
                    alert_type=AlertType.EARNINGS_ALERT,
                    title=f"📊 {item.ticker} Earnings Soon",
                    message=f"""<b>{item.ticker}</b> - {item.company_name}

📅 Date: {item.earnings_date}
⏰ Time: {item.time_of_day}
📊 EPS Estimate: ${item.eps_estimate:.2f if item.eps_estimate else 'N/A'}
⏳ Days Until: {item.days_until}

<b>Action:</b> Review position before earnings""",
                    ticker=item.ticker,
                    priority="HIGH" if item.days_until <= 1 else "MEDIUM"
                )
                alerts.append(alert)

        except Exception as e:
            logger.error(f"Error checking earnings: {e}")

        return alerts

    def alert_earnings_approaching(self, ticker: str, company: str,
                                   earnings_date: str, days_until: int) -> bool:
        """Send earnings approaching alert."""
        urgency = "🔴 TOMORROW" if days_until <= 1 else f"⏳ {days_until} days"

        message = f"""<b>{ticker}</b> - {company}

📅 Earnings: {earnings_date}
{urgency}

<b>Action:</b> Review your position and set stops"""

        alert = Alert(
            alert_type=AlertType.EARNINGS_ALERT,
            title=f"📊 {ticker} Earnings Alert",
            message=message,
            ticker=ticker,
            priority="HIGH" if days_until <= 1 else "MEDIUM"
        )

        return self.send_alert(alert)

    # ================================================================
    # BOND ALERTS
    # ================================================================

    def check_bond_opportunities(self, threshold_high: int = 70,
                                 threshold_low: int = 30) -> List[Alert]:
        """
        Check for bond trading opportunities.

        Args:
            threshold_high: Score above this = strong buy
            threshold_low: Score below this = strong sell

        Returns:
            List of bond alerts
        """
        alerts = []

        try:

            signals = get_all_bond_signals()

            for ticker, sig in signals.items():
                if sig.score >= threshold_high:
                    alert = Alert(
                        alert_type=AlertType.BOND_OPPORTUNITY,
                        title=f"🏦 {ticker} Buy Signal",
                        message=f"""<b>{ticker}</b> Score: {sig.score}/100

Signal: <code>{sig.signal.value}</code>
Price: ${sig.current_price:.2f}
Target: ${sig.target_price:.2f} ({sig.upside_pct:+.1f}%)

Yield Trend: {sig.yield_trend.value}
Curve: {sig.curve_shape.value}""",
                        ticker=ticker,
                        priority="HIGH"
                    )
                    alerts.append(alert)

                elif sig.score <= threshold_low:
                    alert = Alert(
                        alert_type=AlertType.BOND_OPPORTUNITY,
                        title=f"🏦 {ticker} Sell Signal",
                        message=f"""<b>{ticker}</b> Score: {sig.score}/100

Signal: <code>{sig.signal.value}</code>
⚠️ Consider reducing position

Yield Trend: {sig.yield_trend.value}""",
                        ticker=ticker,
                        priority="MEDIUM"
                    )
                    alerts.append(alert)

        except Exception as e:
            logger.error(f"Error checking bond opportunities: {e}")

        return alerts

    # ================================================================
    # TRADE IDEAS ALERTS
    # ================================================================

    def alert_trade_idea(self, ticker: str, action: str, score: int,
                         price: float, target: float, reasoning: str) -> bool:
        """Send trade idea alert."""
        emoji = "📈" if "BUY" in action else "📉"

        message = f"""{emoji} <b>{ticker}</b> - {action}

Score: {score}/100
Price: ${price:.2f}
Target: ${target:.2f} ({(target / price - 1) * 100:+.1f}%)

<b>Reasoning:</b>
{reasoning[:200]}..."""

        alert = Alert(
            alert_type=AlertType.TRADE_IDEA,
            title=f"💡 Trade Idea: {ticker}",
            message=message,
            ticker=ticker,
            priority="HIGH" if score >= 75 else "MEDIUM"
        )

        return self.send_alert(alert)

    # ================================================================
    # PORTFOLIO RISK ALERTS
    # ================================================================

    def alert_portfolio_risk(self, risk_type: str, value: float,
                             threshold: float, details: str = "") -> bool:
        """Send portfolio risk alert."""
        message = f"""⚠️ <b>{risk_type}</b>

Current: {value:.1f}%
Threshold: {threshold:.1f}%

{details}

<b>Action:</b> Review risk exposure immediately"""

        alert = Alert(
            alert_type=AlertType.PORTFOLIO_RISK,
            title=f"⚠️ Risk Alert: {risk_type}",
            message=message,
            priority="HIGH"
        )

        return self.send_alert(alert)

    # ================================================================
    # DAILY SUMMARY
    # ================================================================

    def send_daily_summary(self) -> bool:
        """
        Send daily market summary.

        Returns:
            True if sent successfully
        """
        try:
            # Get regime
            regime_info = ""
            try:
                from src.analytics.macro_regime import get_current_regime
                regime = get_current_regime()
                regime_emoji = "🟢" if "RISK_ON" in regime.regime.value else "🔴" if "RISK_OFF" in regime.regime.value else "🟡"
                regime_info = f"{regime_emoji} Regime: {regime.regime.value} ({regime.regime_score}/100)"
            except:
                pass

            # Get signal counts
            signal_info = ""
            try:
                import pandas as pd
                query = """
                        SELECT signal_type, COUNT(*) as cnt
                        FROM trading_signals
                        WHERE date = CURRENT_DATE
                        GROUP BY signal_type \
                        """
                df = pd.read_sql(query, self.engine)
                if not df.empty:
                    counts = df.set_index('signal_type')['cnt'].to_dict()
                    buys = counts.get('BUY', 0) + counts.get('STRONG_BUY', 0)
                    sells = counts.get('SELL', 0) + counts.get('STRONG_SELL', 0)
                    holds = counts.get('HOLD', 0)
                    signal_info = f"📊 Signals: {buys} Buy | {holds} Hold | {sells} Sell"
            except:
                pass

            # Get yields
            yield_info = ""
            try:
                from src.analytics.bond_signals_analytics import get_treasury_yields
                yields = get_treasury_yields()
                yield_info = f"🏦 Yields: 10Y={yields.yield_10y:.2f}% | 30Y={yields.yield_30y:.2f}%"
            except:
                pass

            # Get upcoming earnings
            earnings_info = ""
            try:
                from src.analytics.earnings_analyzer import get_upcoming_earnings
                upcoming = get_upcoming_earnings(None, 7)
                if upcoming:
                    tickers = [e.ticker for e in upcoming[:5]]
                    earnings_info = f"📅 Earnings this week: {', '.join(tickers)}"
            except:
                pass

            message = f"""<b>📋 Daily Market Summary</b>
{datetime.now().strftime('%A, %B %d, %Y')}

{regime_info}

{signal_info}

{yield_info}

{earnings_info}

<i>Good luck trading! 🚀</i>"""

            alert = Alert(
                alert_type=AlertType.DAILY_SUMMARY,
                title="📋 Daily Summary",
                message=message,
                priority="LOW"
            )

            return self.send_alert(alert)

        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
            return False

    # ================================================================
    # RUN ALL CHECKS
    # ================================================================

    def run_all_checks(self) -> int:
        """
        Run all alert checks and send alerts.

        Returns:
            Number of alerts sent
        """
        alerts_sent = 0

        # Check signal changes
        signal_alerts = self.check_signal_changes()
        for alert in signal_alerts:
            if self.send_alert(alert):
                alerts_sent += 1

        # Check regime change
        regime_alert = self.check_regime_change()
        if regime_alert:
            if self.send_alert(regime_alert):
                alerts_sent += 1

        # Check earnings (only in morning)
        if datetime.now().hour < 10:
            earnings_alerts = self.check_upcoming_earnings(days_ahead=3)
            for alert in earnings_alerts:
                if self.send_alert(alert):
                    alerts_sent += 1

        # Check bond opportunities
        bond_alerts = self.check_bond_opportunities()
        for alert in bond_alerts:
            if self.send_alert(alert):
                alerts_sent += 1

        logger.info(f"Sent {alerts_sent} alerts")
        return alerts_sent


# ============================================================
# Convenience Functions
# ============================================================

_alerter = None


def get_alerter() -> TelegramAlerter:
    """Get singleton alerter instance."""
    global _alerter
    if _alerter is None:
        _alerter = TelegramAlerter()
    return _alerter


def send_alert(message: str) -> bool:
    """Quick function to send a simple message."""
    alerter = get_alerter()
    return alerter.send_message(message)


def send_signal_alert(ticker: str, old_signal: str, new_signal: str,
                      strength: float = None) -> bool:
    """Send signal change alert."""
    alerter = get_alerter()
    return alerter.alert_signal_change(ticker, old_signal, new_signal, strength)


def send_regime_alert(old_regime: str, new_regime: str, score: int) -> bool:
    """Send regime change alert."""
    alerter = get_alerter()
    return alerter.alert_regime_change(old_regime, new_regime, score)


def send_earnings_alert(ticker: str, company: str, date: str, days: int) -> bool:
    """Send earnings alert."""
    alerter = get_alerter()
    return alerter.alert_earnings_approaching(ticker, company, date, days)


def send_daily_summary() -> bool:
    """Send daily summary."""
    alerter = get_alerter()
    return alerter.send_daily_summary()


def run_alert_checks() -> int:
    """Run all alert checks."""
    alerter = get_alerter()
    return alerter.run_all_checks()


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    print("Testing Telegram Alerts...\n")

    alerter = TelegramAlerter()

    if alerter.enabled:
        # Test simple message
        print("Sending test message...")
        success = alerter.send_message("🧪 <b>Test Alert</b>\n\nYour Alpha Research Platform alerts are working!")
        print(f"Result: {'✅ Success' if success else '❌ Failed'}")

        # Test daily summary
        print("\nSending daily summary...")
        success = alerter.send_daily_summary()
        print(f"Result: {'✅ Success' if success else '❌ Failed'}")
    else:
        print("Telegram not configured. Add to .env:")
        print("TELEGRAM_BOT_TOKEN=your_token")
        print("TELEGRAM_CHAT_ID=your_chat_id")