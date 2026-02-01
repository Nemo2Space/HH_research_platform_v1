"""
Economic News Analyzer Module
==============================
AI-powered analysis of economic data releases using local Qwen model.

Provides:
- Surprise calculation (BEAT/MISS/IN-LINE)
- AI interpretation of what numbers mean
- Market impact assessment (BULLISH/BEARISH/NEUTRAL)
- Trading implications and sector impact

Author: Alpha Research Platform
"""

import os
import requests
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv

from src.utils.logging import get_logger

load_dotenv()
logger = get_logger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
LLM_BASE_URL = os.getenv("LLM_QWEN_BASE_URL", "http://172.23.193.91:8090/v1")
LLM_MODEL = os.getenv("LLM_QWEN_MODEL", "Qwen3-32B-Q6_K.gguf")


@dataclass
class EventAnalysis:
    """Analysis of a single economic event."""
    event_name: str
    event_time: str
    actual: Optional[str]
    forecast: Optional[str]
    previous: Optional[str]

    # Calculated
    surprise: str  # "BEAT", "MISS", "IN-LINE", "PENDING"
    surprise_pct: Optional[float]

    # AI Analysis
    market_impact: str  # "BULLISH", "BEARISH", "NEUTRAL"
    ai_analysis: str  # Full AI response

    # Parsed from AI
    interpretation: str = ""
    trading_implications: str = ""
    sectors_affected: str = ""


@dataclass
class MarketAssessment:
    """Overall market assessment from all events."""
    timestamp: datetime
    overall_signal: str  # "BULLISH", "BEARISH", "MIXED"
    summary: str
    key_takeaways: List[str]
    trading_strategy: List[str]
    risks: List[str]
    sector_impact: Dict[str, str]  # sector -> "+"/"-"/"neutral"
    full_analysis: str

    # Event breakdown
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    events_analyzed: List[EventAnalysis] = field(default_factory=list)


class EconomicNewsAnalyzer:
    """
    AI-powered analyzer for economic news releases.
    Uses local Qwen model for interpretation.
    """

    def __init__(self):
        self._llm_url = f"{LLM_BASE_URL}/chat/completions"
        self._model = LLM_MODEL
        self._cache = {}

    def _call_llm(self, prompt: str, max_tokens: int = 1500) -> str:
        """Call local Qwen model."""
        try:
            response = requests.post(
                self._llm_url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": self._model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert financial analyst specializing in macroeconomic data interpretation. Provide concise, actionable analysis. Be direct and specific. No thinking tags."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                    "stream": False
                },
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                # Clean up any thinking tags if present
                if '<think>' in content:
                    content = content.split('</think>')[-1].strip()
                return content
            else:
                logger.error(f"LLM error: {response.status_code}")
                return ""

        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to LLM at {self._llm_url}")
            return ""
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ""

    def _parse_value(self, value_str: str) -> Optional[float]:
        """Parse economic value string to float."""
        if not value_str:
            return None
        try:
            clean = value_str.strip()
            multiplier = 1
            if clean.endswith('%'):
                clean = clean[:-1]
            elif clean.endswith('K'):
                clean = clean[:-1]
                multiplier = 1000
            elif clean.endswith('M'):
                clean = clean[:-1]
                multiplier = 1000000
            elif clean.endswith('B'):
                clean = clean[:-1]
                multiplier = 1000000000
            return float(clean) * multiplier
        except:
            return None

    def _calculate_surprise(self, actual: str, forecast: str) -> tuple:
        """Calculate surprise vs forecast."""
        if not actual:
            return "PENDING", None

        actual_val = self._parse_value(actual)
        forecast_val = self._parse_value(forecast)

        if actual_val is None or forecast_val is None or forecast_val == 0:
            return "N/A", None

        pct_diff = ((actual_val - forecast_val) / abs(forecast_val)) * 100

        if abs(pct_diff) < 1:
            return "IN-LINE", pct_diff
        elif pct_diff > 0:
            return "BEAT", pct_diff
        else:
            return "MISS", pct_diff

    def analyze_event(self, event_name: str, event_time: str,
                      actual: str, forecast: str, previous: str = None) -> EventAnalysis:
        """
        Analyze a single economic event.

        Args:
            event_name: Name of the event
            event_time: Time of release (local)
            actual: Actual released value (or None if pending)
            forecast: Expected value
            previous: Previous period value

        Returns:
            EventAnalysis with AI interpretation
        """
        surprise, surprise_pct = self._calculate_surprise(actual, forecast)

        # If no actual data yet, return pending analysis
        if not actual or surprise == "PENDING":
            return EventAnalysis(
                event_name=event_name,
                event_time=event_time,
                actual=actual,
                forecast=forecast,
                previous=previous,
                surprise="PENDING",
                surprise_pct=None,
                market_impact="PENDING",
                ai_analysis="Data not yet released. Check back after the release time.",
                interpretation="Awaiting data release.",
                trading_implications="",
                sectors_affected=""
            )

        # Build prompt for AI
        surprise_str = f"{surprise_pct:+.1f}%" if surprise_pct is not None else "N/A"
        prompt = f"""Analyze this economic data release concisely:

EVENT: {event_name}
ACTUAL: {actual}
EXPECTED: {forecast}
PREVIOUS: {previous or 'N/A'}
SURPRISE: {surprise} ({surprise_str} vs forecast)

Provide analysis in this EXACT format (be concise, 2-3 sentences each):

INTERPRETATION:
[What this means for the economy]

MARKET IMPACT:
[One word: BULLISH, BEARISH, or NEUTRAL]

WHY:
[Why this is bullish/bearish/neutral for stocks]

TRADING IMPLICATIONS:
[2-3 bullet points for traders]

SECTORS AFFECTED:
[List 2-3 sectors with + or - impact]
"""

        ai_response = self._call_llm(prompt, max_tokens=800)

        # Parse market impact
        market_impact = "NEUTRAL"
        if "MARKET IMPACT:" in ai_response:
            impact_line = ai_response.split("MARKET IMPACT:")[1].split("\n")[0]
            if "BULLISH" in impact_line.upper():
                market_impact = "BULLISH"
            elif "BEARISH" in impact_line.upper():
                market_impact = "BEARISH"

        # Parse sections
        interpretation = ""
        trading_implications = ""
        sectors_affected = ""

        if "INTERPRETATION:" in ai_response:
            parts = ai_response.split("INTERPRETATION:")[1]
            interpretation = parts.split("MARKET IMPACT:")[0].strip() if "MARKET IMPACT:" in parts else parts[:200]

        if "TRADING IMPLICATIONS:" in ai_response:
            parts = ai_response.split("TRADING IMPLICATIONS:")[1]
            trading_implications = parts.split("SECTORS")[0].strip() if "SECTORS" in parts else parts[:300]

        if "SECTORS AFFECTED:" in ai_response:
            sectors_affected = ai_response.split("SECTORS AFFECTED:")[1].strip()[:200]

        return EventAnalysis(
            event_name=event_name,
            event_time=event_time,
            actual=actual,
            forecast=forecast,
            previous=previous,
            surprise=surprise,
            surprise_pct=surprise_pct,
            market_impact=market_impact,
            ai_analysis=ai_response,
            interpretation=interpretation,
            trading_implications=trading_implications,
            sectors_affected=sectors_affected
        )

    def analyze_all_events(self, events: List[Dict]) -> MarketAssessment:
        """
        Analyze all events and provide overall market assessment.

        Args:
            events: List of event dicts with keys: name, time, actual, forecast, previous

        Returns:
            MarketAssessment with overall analysis
        """
        # Filter to released events only
        released_events = [e for e in events if e.get('actual')]

        if not released_events:
            return MarketAssessment(
                timestamp=datetime.now(),
                overall_signal="PENDING",
                summary="No economic data released yet. Analysis will be available after data releases.",
                key_takeaways=[],
                trading_strategy=[],
                risks=[],
                sector_impact={},
                full_analysis="Waiting for data releases...",
                events_analyzed=[]
            )

        # Build events text for AI
        events_text = ""
        for e in released_events:
            surprise, pct = self._calculate_surprise(e.get('actual', ''), e.get('forecast', ''))
            pct_str = f"{pct:+.1f}%" if pct is not None else "N/A"
            events_text += f"""
• {e.get('name', 'Unknown')}
  Actual: {e.get('actual')} | Expected: {e.get('forecast', 'N/A')} | Previous: {e.get('previous', 'N/A')}
  Surprise: {surprise} ({pct_str})
"""

        prompt = f"""You are a senior market strategist. Analyze today's economic releases.

TODAY'S DATA:
{events_text}

Provide analysis in this EXACT format:

OVERALL ASSESSMENT:
[One word: BULLISH, BEARISH, or MIXED]

SUMMARY:
[3-4 sentences on the overall economic picture]

KEY TAKEAWAYS:
- [Point 1]
- [Point 2]
- [Point 3]

TRADING STRATEGY:
- [Action 1]
- [Action 2]

RISKS TO WATCH:
- [Risk 1]
- [Risk 2]

SECTOR IMPACT:
- [Sector 1] (+/-)
- [Sector 2] (+/-)
- [Sector 3] (+/-)
"""

        ai_response = self._call_llm(prompt, max_tokens=1500)

        # Parse overall signal
        overall_signal = "MIXED"
        if "OVERALL ASSESSMENT:" in ai_response:
            assessment_line = ai_response.split("OVERALL ASSESSMENT:")[1].split("\n")[0]
            if "BULLISH" in assessment_line.upper():
                overall_signal = "BULLISH"
            elif "BEARISH" in assessment_line.upper():
                overall_signal = "BEARISH"

        # Parse summary
        summary = ""
        if "SUMMARY:" in ai_response:
            parts = ai_response.split("SUMMARY:")[1]
            summary = parts.split("KEY TAKEAWAYS:")[0].strip() if "KEY TAKEAWAYS:" in parts else parts[:500]

        # Parse key takeaways
        key_takeaways = []
        if "KEY TAKEAWAYS:" in ai_response:
            parts = ai_response.split("KEY TAKEAWAYS:")[1]
            section = parts.split("TRADING STRATEGY:")[0] if "TRADING STRATEGY:" in parts else parts[:500]
            for line in section.split("\n"):
                line = line.strip()
                if line.startswith("-") or line.startswith("•"):
                    key_takeaways.append(line[1:].strip())

        # Parse trading strategy
        trading_strategy = []
        if "TRADING STRATEGY:" in ai_response:
            parts = ai_response.split("TRADING STRATEGY:")[1]
            section = parts.split("RISKS")[0] if "RISKS" in parts else parts[:500]
            for line in section.split("\n"):
                line = line.strip()
                if line.startswith("-") or line.startswith("•"):
                    trading_strategy.append(line[1:].strip())

        # Parse risks
        risks = []
        if "RISKS TO WATCH:" in ai_response:
            parts = ai_response.split("RISKS TO WATCH:")[1]
            section = parts.split("SECTOR")[0] if "SECTOR" in parts else parts[:500]
            for line in section.split("\n"):
                line = line.strip()
                if line.startswith("-") or line.startswith("•"):
                    risks.append(line[1:].strip())

        # Parse sector impact
        sector_impact = {}
        if "SECTOR IMPACT:" in ai_response:
            section = ai_response.split("SECTOR IMPACT:")[1][:500]
            for line in section.split("\n"):
                line = line.strip()
                if "(+)" in line or "+" in line:
                    sector = line.replace("-", "").replace("(+)", "").replace("+", "").strip()
                    if sector:
                        sector_impact[sector] = "+"
                elif "(-)" in line or line.endswith("-"):
                    sector = line.replace("-", "").replace("(-)", "").strip()
                    if sector:
                        sector_impact[sector] = "-"

        return MarketAssessment(
            timestamp=datetime.now(),
            overall_signal=overall_signal,
            summary=summary,
            key_takeaways=key_takeaways[:5],
            trading_strategy=trading_strategy[:3],
            risks=risks[:3],
            sector_impact=sector_impact,
            full_analysis=ai_response,
            events_analyzed=[]
        )

    def get_quick_summary(self, events: List[Dict]) -> str:
        """Get a one-line quick summary of market impact."""
        released = [e for e in events if e.get('actual')]
        if not released:
            return "⏳ Awaiting data releases..."

        bullish = 0
        bearish = 0

        for e in released:
            surprise, _ = self._calculate_surprise(e.get('actual', ''), e.get('forecast', ''))
            name_lower = e.get('name', '').lower()

            # Simple heuristic
            is_higher_good = not any(x in name_lower for x in ['unemployment', 'jobless', 'cpi', 'inflation', 'ppi'])

            if surprise == "BEAT":
                if is_higher_good:
                    bullish += 1
                else:
                    bearish += 1
            elif surprise == "MISS":
                if is_higher_good:
                    bearish += 1
                else:
                    bullish += 1

        if bullish > bearish:
            return f"🟢 BULLISH ({bullish} beats, {bearish} misses)"
        elif bearish > bullish:
            return f"🔴 BEARISH ({bearish} misses, {bullish} beats)"
        else:
            return f"🟡 MIXED ({bullish} beats, {bearish} misses)"


# ============================================================
# Singleton instance
# ============================================================
_analyzer = None

def get_news_analyzer() -> EconomicNewsAnalyzer:
    """Get singleton analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = EconomicNewsAnalyzer()
    return _analyzer


def analyze_economic_event(event_name: str, event_time: str,
                           actual: str, forecast: str, previous: str = None) -> EventAnalysis:
    """Convenience function to analyze a single event."""
    return get_news_analyzer().analyze_event(event_name, event_time, actual, forecast, previous)


def analyze_all_economic_events(events: List[Dict]) -> MarketAssessment:
    """Convenience function to analyze all events."""
    return get_news_analyzer().analyze_all_events(events)


def get_market_quick_summary(events: List[Dict]) -> str:
    """Get quick one-line market summary."""
    return get_news_analyzer().get_quick_summary(events)