"""
Earnings Intelligence - AI Chat Integration

Provides earnings intelligence context for AI chat integration.
Automatically enriches ticker queries with IES/EQS/ECS analysis.

This module provides:
1. get_earnings_context_for_ai(ticker) - Main function for chat integration
2. get_earnings_keywords() - Keywords for context detection
3. format_earnings_for_chat(ticker) - Formatted earnings summary

Usage in chat.py:
    from src.analytics.earnings_intelligence.chat_integration import (
        get_earnings_context_for_ai,
        needs_earnings_context,
    )

    # In _get_ticker_context():
    if needs_earnings_context(message):
        earnings_context = get_earnings_context_for_ai(ticker)
        if earnings_context:
            context_parts.append(earnings_context)

Author: Alpha Research Platform
Phase: 11 of Earnings Intelligence System
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any

from src.utils.logging import get_logger

# Import from previous phases
from src.analytics.earnings_intelligence.windows import (
    get_earnings_info,
    is_in_compute_window,
    is_in_action_window,
)

from src.analytics.earnings_intelligence.ies_calculator import (
    calculate_ies,
    IESCalculationResult,
)

from src.analytics.earnings_intelligence.eqs_calculator import (
    calculate_eqs,
    EQSCalculationResult,
)

from src.analytics.earnings_intelligence.ecs_calculator import (
    calculate_ecs,
    ECSCalculationResult,
    get_full_earnings_analysis,
)

from src.analytics.earnings_intelligence.screener_integration import (
    enrich_screener_with_earnings,
    EarningsEnrichment,
)

from src.analytics.earnings_intelligence.models import (
    ECSCategory,
    ExpectationsRegime,
    DataQuality,
)

logger = get_logger(__name__)


# ============================================================================
# KEYWORDS FOR CONTEXT DETECTION
# ============================================================================

EARNINGS_KEYWORDS = [
    # Direct earnings terms
    'earnings', 'earning', 'eps', 'revenue', 'quarterly', 'quarter',
    'report', 'results', 'beat', 'miss', 'surprise', 'guidance',
    'forecast', 'outlook', 'call', 'transcript',

    # Expectations terms
    'expectations', 'expected', 'priced in', 'priced-in', 'consensus',
    'estimates', 'estimate', 'analyst', 'analysts',

    # IES/EQS/ECS terms
    'ies', 'eqs', 'ecs', 'implied expectations', 'clearance',
    'cleared', 'bar', 'threshold',

    # Trading around earnings
    'before earnings', 'after earnings', 'pre-earnings', 'post-earnings',
    'earnings play', 'earnings trade', 'hold through', 'sell before',

    # Time-based
    'when is', 'when are', 'upcoming', 'next earnings', 'last earnings',
    'days until', 'days to',
]


def needs_earnings_context(message: str) -> bool:
    """
    Check if message needs earnings context.

    Args:
        message: User message

    Returns:
        True if earnings context would be helpful
    """
    message_lower = message.lower()
    return any(kw in message_lower for kw in EARNINGS_KEYWORDS)


def get_earnings_keywords() -> List[str]:
    """Get list of earnings-related keywords for chat detection."""
    return EARNINGS_KEYWORDS.copy()


# ============================================================================
# MAIN AI CONTEXT FUNCTIONS
# ============================================================================

def get_earnings_context_for_ai(ticker: str,
                                 include_full_analysis: bool = False) -> str:
    """
    Get earnings intelligence context formatted for AI chat.

    This is the main function called from chat.py to inject earnings
    context into the AI conversation.

    Args:
        ticker: Stock symbol
        include_full_analysis: Include detailed component breakdown

    Returns:
        Formatted context string for AI consumption
    """
    try:
        # Get earnings window info first (fast)
        earnings_info = get_earnings_info(ticker)

        if earnings_info is None:
            return ""

        lines = []

        # Header
        lines.append(f"📊 EARNINGS INTELLIGENCE: {ticker}")
        lines.append("-" * 40)

        # Earnings date and window
        if earnings_info.earnings_date:
            lines.append(f"Earnings Date: {earnings_info.earnings_date}")

            if earnings_info.days_to_earnings is not None:
                days = earnings_info.days_to_earnings
                if days > 0:
                    lines.append(f"Days Until Earnings: {days}")
                    if days <= 5:
                        lines.append("⚠️ IN ACTION WINDOW - Earnings approaching!")
                elif days == 0:
                    lines.append("⚠️ EARNINGS TODAY!")
                else:
                    lines.append(f"Days Since Earnings: {abs(days)}")

        # Check if we're in compute window (worth calculating IES)
        in_compute = is_in_compute_window(ticker)
        in_action = is_in_action_window(ticker)

        if in_compute or in_action:
            # Calculate IES for pre-earnings context
            try:
                ies_result = calculate_ies(ticker)

                if ies_result.ies is not None:
                    lines.append("")
                    lines.append("PRE-EARNINGS SETUP:")

                    # IES level interpretation
                    ies = ies_result.ies
                    if ies >= 70:
                        ies_desc = "HIGH expectations priced in"
                    elif ies >= 50:
                        ies_desc = "MODERATE expectations"
                    else:
                        ies_desc = "LOW expectations (easier to beat)"

                    lines.append(f"  IES: {ies:.0f}/100 - {ies_desc}")
                    lines.append(f"  Regime: {ies_result.regime.value}")

                    if ies_result.implied_move_pctl:
                        lines.append(f"  Implied Move Pctl: {ies_result.implied_move_pctl:.0f}th")

                    lines.append(f"  Position Scale: {ies_result.position_scale:.0%}")

                    # Trading guidance
                    lines.append("")
                    lines.append("TRADING GUIDANCE:")

                    if ies_result.regime == ExpectationsRegime.HYPED:
                        lines.append("  ⚠️ HYPED regime - even a beat may disappoint")
                        lines.append("  Consider reducing position before earnings")
                    elif ies_result.regime == ExpectationsRegime.FEARED:
                        lines.append("  📉 FEARED regime - low bar to clear")
                        lines.append("  Potential for positive surprise")
                    elif ies_result.regime == ExpectationsRegime.VOLATILE:
                        lines.append("  ⚡ VOLATILE - big move expected, uncertain direction")
                        lines.append("  Consider options strategies")
                    else:
                        lines.append("  Standard expectations setup")

            except Exception as e:
                logger.debug(f"{ticker}: Error calculating IES for chat: {e}")

        # Check for recent earnings (post-earnings analysis)
        # Always calculate ECS for tickers with recent earnings (within 30 days)
        if earnings_info.days_to_earnings is not None and earnings_info.days_to_earnings < 0:
            if earnings_info.days_to_earnings >= -30:  # Within 30 days after earnings
                try:
                    # First try EQS alone (faster, less dependencies)
                    try:
                        eqs_result = calculate_eqs(ticker)

                        if eqs_result and eqs_result.eqs is not None:
                            lines.append("")
                            lines.append("POST-EARNINGS RESULTS:")

                            if eqs_result.eps_surprise_pct is not None:
                                eps_emoji = "✅" if eqs_result.eps_surprise_pct > 0 else "❌"
                                lines.append(f"  {eps_emoji} EPS Surprise: {eqs_result.eps_surprise_pct:+.1f}%")

                            if eqs_result.revenue_surprise_pct is not None:
                                rev_emoji = "✅" if eqs_result.revenue_surprise_pct > 0 else "❌"
                                lines.append(f"  {rev_emoji} Revenue Surprise: {eqs_result.revenue_surprise_pct:+.1f}%")

                            lines.append(f"  EQS: {eqs_result.eqs:.0f}/100")

                            if eqs_result.event_z is not None:
                                lines.append(f"  Event Z: {eqs_result.event_z:+.2f}")
                    except Exception as e:
                        logger.debug(f"{ticker}: EQS calculation error: {e}")

                    # Then try full ECS
                    ecs_result = calculate_ecs(ticker)

                    if ecs_result.ecs_category != ECSCategory.UNKNOWN:
                        lines.append("")
                        lines.append("EXPECTATIONS CLEARANCE:")

                        # ECS result
                        emoji = _get_ecs_emoji(ecs_result.ecs_category)
                        lines.append(f"  {emoji} ECS: {ecs_result.ecs_category.value}")
                        lines.append(f"  Cleared Bar: {'YES ✅' if ecs_result.cleared_bar else 'NO ❌'}")

                        if ecs_result.required_z is not None:
                            lines.append(f"  Required Z: {ecs_result.required_z:.2f}")

                        # Score adjustment
                        if ecs_result.score_adjustment != 0:
                            lines.append(f"  Score Adjustment: {ecs_result.score_adjustment:+d}")

                except Exception as e:
                    logger.debug(f"{ticker}: Error calculating ECS for chat: {e}")

        # Include full analysis if requested
        if include_full_analysis:
            lines.append("")
            lines.append("DETAILED ANALYSIS:")
            try:
                analysis = get_full_earnings_analysis(ticker)

                if analysis.get('pre_earnings', {}).get('ies'):
                    lines.append(f"  IES Components:")
                    pre = analysis['pre_earnings']
                    if pre.get('drift_20d'):
                        lines.append(f"    Drift 20d: {pre['drift_20d']:+.1%}")
                    if pre.get('iv_pctl'):
                        lines.append(f"    IV Pctl: {pre['iv_pctl']:.0f}")

            except Exception as e:
                logger.debug(f"{ticker}: Error getting full analysis: {e}")

        # Data quality note
        lines.append("")
        lines.append(f"Data Quality: {earnings_info.data_quality if hasattr(earnings_info, 'data_quality') else 'UNKNOWN'}")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"{ticker}: Error getting earnings context for AI: {e}")
        return ""


def get_quick_earnings_status(ticker: str) -> str:
    """
    Get a quick one-line earnings status for a ticker.

    Useful for including in ticker summaries without full analysis.

    Args:
        ticker: Stock symbol

    Returns:
        One-line status string
    """
    try:
        earnings_info = get_earnings_info(ticker)

        if earnings_info is None or earnings_info.earnings_date is None:
            return ""

        days = earnings_info.days_to_earnings

        if days is None:
            return ""

        if days == 0:
            return f"⚠️ {ticker}: EARNINGS TODAY"
        elif days > 0 and days <= 5:
            return f"⚠️ {ticker}: Earnings in {days} days"
        elif days > 5 and days <= 14:
            return f"📅 {ticker}: Earnings in {days} days"
        elif days < 0 and days >= -5:
            return f"📊 {ticker}: Reported {abs(days)} days ago"

        return ""

    except Exception as e:
        logger.debug(f"{ticker}: Error getting quick status: {e}")
        return ""


def format_earnings_for_chat(ticker: str) -> str:
    """
    Format earnings intelligence as a conversational response.

    This is used when the user explicitly asks about earnings.

    Args:
        ticker: Stock symbol

    Returns:
        Formatted conversational response
    """
    try:
        earnings_info = get_earnings_info(ticker)

        if earnings_info is None:
            return f"I don't have earnings calendar data for {ticker}."

        lines = []

        # Start with earnings date
        if earnings_info.earnings_date:
            days = earnings_info.days_to_earnings

            if days is not None:
                if days > 0:
                    lines.append(f"**{ticker}** reports earnings on **{earnings_info.earnings_date}** ({days} days away).")
                elif days == 0:
                    lines.append(f"**{ticker}** reports earnings **TODAY** ({earnings_info.earnings_date})!")
                else:
                    lines.append(f"**{ticker}** reported earnings on **{earnings_info.earnings_date}** ({abs(days)} days ago).")
        else:
            lines.append(f"No confirmed earnings date for {ticker}.")
            return "\n".join(lines)

        # Pre-earnings analysis
        if earnings_info.days_to_earnings is not None and earnings_info.days_to_earnings >= 0:
            try:
                ies_result = calculate_ies(ticker)

                if ies_result.ies is not None:
                    lines.append("")
                    lines.append("**Pre-Earnings Analysis:**")

                    # IES interpretation
                    ies = ies_result.ies
                    if ies >= 70:
                        lines.append(f"- IES: {ies:.0f}/100 - Expectations are **HIGH**. The stock needs exceptional results.")
                    elif ies >= 50:
                        lines.append(f"- IES: {ies:.0f}/100 - Expectations are **MODERATE**. A solid beat is needed.")
                    else:
                        lines.append(f"- IES: {ies:.0f}/100 - Expectations are **LOW**. Easier to surprise positively.")

                    # Regime
                    regime = ies_result.regime
                    if regime == ExpectationsRegime.HYPED:
                        lines.append("- ⚠️ **HYPED** regime - Even good results may disappoint. Consider caution.")
                    elif regime == ExpectationsRegime.FEARED:
                        lines.append("- 📉 **FEARED** regime - Bar is low. Potential for positive surprise.")
                    elif regime == ExpectationsRegime.VOLATILE:
                        lines.append("- ⚡ **VOLATILE** - Big move expected but direction uncertain.")

                    # Position guidance
                    if ies_result.position_scale < 0.8:
                        lines.append(f"- Suggested position scale: **{ies_result.position_scale:.0%}** of normal (reduced due to risk)")

            except Exception as e:
                logger.debug(f"{ticker}: Error in pre-earnings analysis: {e}")

        # Post-earnings analysis
        elif earnings_info.days_to_earnings is not None and earnings_info.days_to_earnings < 0:
            try:
                ecs_result = calculate_ecs(ticker)

                if ecs_result.ecs_category != ECSCategory.UNKNOWN:
                    lines.append("")
                    lines.append("**Post-Earnings Analysis:**")

                    # ECS result
                    cat = ecs_result.ecs_category
                    if cat == ECSCategory.STRONG_BEAT:
                        lines.append("- 🚀 **STRONG BEAT** - Results significantly exceeded expectations!")
                    elif cat == ECSCategory.BEAT:
                        lines.append("- ✅ **BEAT** - Results cleared the expectations bar.")
                    elif cat == ECSCategory.INLINE:
                        lines.append("- ➡️ **INLINE** - Results roughly matched what was priced in.")
                    elif cat == ECSCategory.MISS:
                        lines.append("- ⚠️ **MISS** - Results fell short of priced-in expectations.")
                    elif cat == ECSCategory.STRONG_MISS:
                        lines.append("- 🔴 **STRONG MISS** - Results significantly disappointed.")

                    # Surprise details
                    if ecs_result.eps_surprise_pct is not None:
                        direction = "beat" if ecs_result.eps_surprise_pct > 0 else "missed"
                        lines.append(f"- EPS {direction} by {abs(ecs_result.eps_surprise_pct):.1f}%")

                    # Score impact
                    adj = ecs_result.score_adjustment
                    if adj != 0:
                        lines.append(f"- Score adjustment: **{adj:+d}** points")

            except Exception as e:
                logger.debug(f"{ticker}: Error in post-earnings analysis: {e}")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"{ticker}: Error formatting earnings for chat: {e}")
        return f"Error retrieving earnings data for {ticker}."


# ============================================================================
# BATCH FUNCTIONS FOR PORTFOLIO CONTEXT
# ============================================================================

def get_earnings_alerts_for_portfolio(tickers: List[str]) -> str:
    """
    Get earnings alerts for a portfolio of tickers.

    Returns a summary of stocks approaching earnings.

    Args:
        tickers: List of ticker symbols

    Returns:
        Formatted alerts string
    """
    alerts = []

    for ticker in tickers:
        try:
            earnings_info = get_earnings_info(ticker)

            if earnings_info is None or earnings_info.days_to_earnings is None:
                continue

            days = earnings_info.days_to_earnings

            # Alert for stocks within 7 days of earnings
            if 0 <= days <= 7:
                if days == 0:
                    alerts.append(f"⚠️ {ticker}: EARNINGS TODAY!")
                else:
                    alerts.append(f"📅 {ticker}: Earnings in {days} day{'s' if days > 1 else ''}")

            # Alert for recent earnings (within 3 days after)
            elif -3 <= days < 0:
                try:
                    ecs_result = calculate_ecs(ticker)
                    if ecs_result.ecs_category != ECSCategory.UNKNOWN:
                        emoji = _get_ecs_emoji(ecs_result.ecs_category)
                        alerts.append(f"{emoji} {ticker}: {ecs_result.ecs_category.value} ({abs(days)} day{'s' if abs(days) > 1 else ''} ago)")
                except:
                    pass

        except Exception as e:
            logger.debug(f"{ticker}: Error checking earnings: {e}")

    if alerts:
        return "🗓️ EARNINGS ALERTS:\n" + "\n".join(alerts)

    return ""


def get_high_risk_earnings_tickers(tickers: List[str]) -> List[str]:
    """
    Get tickers with high-risk earnings setups.

    Args:
        tickers: List of ticker symbols

    Returns:
        List of high-risk tickers
    """
    high_risk = []

    for ticker in tickers:
        try:
            enrichment = enrich_screener_with_earnings(ticker)

            if enrichment.in_action_window:
                high_risk.append(ticker)
            elif enrichment.ies is not None and enrichment.ies >= 70:
                high_risk.append(ticker)
            elif enrichment.regime == ExpectationsRegime.HYPED:
                high_risk.append(ticker)

        except Exception as e:
            logger.debug(f"{ticker}: Error checking risk: {e}")

    return high_risk


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_ecs_emoji(category: ECSCategory) -> str:
    """Get emoji for ECS category."""
    emojis = {
        ECSCategory.STRONG_BEAT: "🚀",
        ECSCategory.BEAT: "✅",
        ECSCategory.INLINE: "➡️",
        ECSCategory.MISS: "⚠️",
        ECSCategory.STRONG_MISS: "🔴",
        ECSCategory.UNKNOWN: "❓",
    }
    return emojis.get(category, "❓")


# ============================================================================
# INTEGRATION HELPER FOR CHAT.PY
# ============================================================================

def get_earnings_for_ai(ticker: str) -> str:
    """
    Drop-in replacement for existing earnings context function.

    This function matches the interface expected by chat.py's
    _get_ticker_context method.

    Args:
        ticker: Stock symbol

    Returns:
        Formatted context string
    """
    return get_earnings_context_for_ai(ticker)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("Phase 11: AI Chat Integration Test")
    print("=" * 60)

    test_tickers = ["AAPL", "NVDA", "TSLA"]

    print("\n1. Testing earnings context for AI...")
    for ticker in test_tickers[:2]:
        print(f"\n--- {ticker} ---")
        context = get_earnings_context_for_ai(ticker)
        if context:
            print(context)
        else:
            print(f"  No earnings context for {ticker}")

    print("\n" + "=" * 60)
    print("2. Testing quick status...")
    for ticker in test_tickers:
        status = get_quick_earnings_status(ticker)
        if status:
            print(f"  {status}")
        else:
            print(f"  {ticker}: No upcoming earnings")

    print("\n" + "=" * 60)
    print("3. Testing conversational format...")
    response = format_earnings_for_chat("NVDA")
    print(response)

    print("\n" + "=" * 60)
    print("4. Testing portfolio alerts...")
    alerts = get_earnings_alerts_for_portfolio(test_tickers)
    if alerts:
        print(alerts)
    else:
        print("  No earnings alerts for portfolio")

    print("\n" + "=" * 60)
    print("[OK] AI Chat Integration tests complete!")