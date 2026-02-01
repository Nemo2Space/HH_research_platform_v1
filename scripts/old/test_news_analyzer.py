"""
TEST: AI-Powered Economic News Analyzer
========================================
Uses your local Qwen model for real AI analysis.

Run: python test_ai_news_analyzer.py
"""

import requests
import json
import os
from typing import Optional, List, Dict
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env
load_dotenv()

# ============================================================
# CONFIGURATION - From your .env
# ============================================================
LLM_BASE_URL = os.getenv("LLM_QWEN_BASE_URL", "http://172.23.193.91:8090/v1")
LLM_MODEL = os.getenv("LLM_QWEN_MODEL", "Qwen3-32B-Q6_K.gguf")


@dataclass
class AIAnalysis:
    """AI analysis result."""
    event_name: str
    actual: str
    forecast: str
    previous: str

    # Basic stats (rule-based)
    surprise: str
    surprise_pct: float

    # AI analysis
    ai_interpretation: str
    market_impact: str
    trading_implications: str


def call_qwen(prompt: str, max_tokens: int = 1500) -> str:
    """Call Qwen model via OpenAI-compatible API."""
    url = f"{LLM_BASE_URL}/chat/completions"

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "model": LLM_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert financial analyst specializing in macroeconomic data interpretation. Provide concise, actionable analysis. Be direct and specific."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "stream": False
            },
            timeout=120  # 2 min timeout for large model
        )

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code} - {response.text[:200]}"

    except requests.exceptions.ConnectionError:
        return f"ERROR: Cannot connect to LLM at {url}"
    except requests.exceptions.Timeout:
        return "ERROR: Request timed out (model may be busy)"
    except Exception as e:
        return f"ERROR: {str(e)}"


def parse_value(value_str: str) -> Optional[float]:
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
        return float(clean) * multiplier
    except:
        return None


def calculate_surprise(actual: str, forecast: str) -> tuple:
    """Calculate surprise vs forecast."""
    actual_val = parse_value(actual)
    forecast_val = parse_value(forecast)

    if actual_val is None or forecast_val is None or forecast_val == 0:
        return "N/A", 0

    pct_diff = ((actual_val - forecast_val) / abs(forecast_val)) * 100

    if abs(pct_diff) < 1:
        return "IN-LINE", pct_diff
    elif pct_diff > 0:
        return "BEAT", pct_diff
    else:
        return "MISS", pct_diff


def analyze_single_event(event_name: str, actual: str, forecast: str,
                         previous: str = None) -> AIAnalysis:
    """Analyze a single economic event using AI."""

    # Calculate basic stats
    surprise, surprise_pct = calculate_surprise(actual, forecast)

    # Build AI prompt
    prompt = f"""Analyze this economic data release:

EVENT: {event_name}
ACTUAL: {actual}
EXPECTED (Forecast): {forecast}
PREVIOUS: {previous or 'N/A'}
SURPRISE: {surprise} ({surprise_pct:+.1f}% vs forecast)

Provide a concise analysis in this exact format:

INTERPRETATION:
[2-3 sentences explaining what this number means for the economy]

MARKET IMPACT:
[One word: BULLISH, BEARISH, or NEUTRAL]

WHY:
[1-2 sentences explaining why this is bullish/bearish/neutral for stocks]

TRADING IMPLICATIONS:
[2-3 bullet points on what traders should consider]

SECTORS AFFECTED:
[List 2-3 sectors most impacted, with + or - to indicate positive/negative impact]
"""

    # Call AI
    ai_response = call_qwen(prompt)

    # Extract market impact from AI response
    market_impact = "NEUTRAL"
    if "MARKET IMPACT:" in ai_response:
        impact_line = ai_response.split("MARKET IMPACT:")[1].split("\n")[0].strip()
        if "BULLISH" in impact_line.upper():
            market_impact = "BULLISH"
        elif "BEARISH" in impact_line.upper():
            market_impact = "BEARISH"

    # Extract trading implications
    trading_implications = ""
    if "TRADING IMPLICATIONS:" in ai_response:
        parts = ai_response.split("TRADING IMPLICATIONS:")[1]
        if "SECTORS" in parts:
            trading_implications = parts.split("SECTORS")[0].strip()
        else:
            trading_implications = parts.strip()

    return AIAnalysis(
        event_name=event_name,
        actual=actual,
        forecast=forecast,
        previous=previous,
        surprise=surprise,
        surprise_pct=surprise_pct,
        ai_interpretation=ai_response,
        market_impact=market_impact,
        trading_implications=trading_implications
    )


def analyze_all_events(events: List[Dict]) -> str:
    """Analyze all events and get overall market view."""

    # Format events for AI
    events_text = ""
    for e in events:
        if e.get('actual'):
            surprise, pct = calculate_surprise(e['actual'], e.get('forecast', ''))
            events_text += f"""
• {e['name']}
  Actual: {e['actual']} | Expected: {e.get('forecast', 'N/A')} | Previous: {e.get('previous', 'N/A')}
  Surprise: {surprise} ({pct:+.1f}%)
"""

    prompt = f"""You are a senior market strategist. Analyze today's economic data releases and provide an overall market assessment.

TODAY'S ECONOMIC DATA:
{events_text}

Provide analysis in this format:

📊 OVERALL ASSESSMENT:
[BULLISH / BEARISH / MIXED - with emoji 🟢/🔴/🟡]

📈 SUMMARY:
[3-4 sentences summarizing the overall economic picture from today's data]

💡 KEY TAKEAWAYS:
[3-4 bullet points of the most important insights]

🎯 TRADING STRATEGY:
[2-3 actionable recommendations based on this data]

⚠️ RISKS TO WATCH:
[2-3 risks or contrary signals to be aware of]

🏭 SECTOR IMPACT:
[List sectors with likely positive (+) or negative (-) impact]
"""

    return call_qwen(prompt, max_tokens=2000)


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🤖 AI-POWERED ECONOMIC NEWS ANALYZER")
    print("=" * 70)
    print(f"\nLLM URL: {LLM_BASE_URL}")
    print(f"Model: {LLM_MODEL}")
    print("-" * 70)

    # Test connection
    print("\n🔗 Testing connection...")
    test_response = call_qwen("Respond with exactly: CONNECTION OK", max_tokens=20)

    if "ERROR" in test_response or "Error" in test_response:
        print(f"\n❌ {test_response}")
        exit(1)
    elif not test_response.strip():
        print(f"⚠️ Empty response from model (but connected)")
    else:
        print(f"✅ Connected! Response: {test_response.strip()[:50]}")

    # Test with sample data
    test_events = [
        {
            'name': 'Unemployment Rate (Nov)',
            'actual': '4.2%',
            'forecast': '4.1%',
            'previous': '4.1%'
        },
        {
            'name': 'Nonfarm Payrolls (Nov)',
            'actual': '227K',
            'forecast': '220K',
            'previous': '36K'
        },
        {
            'name': 'Retail Sales (MoM) (Nov)',
            'actual': '0.7%',
            'forecast': '0.6%',
            'previous': '0.5%'
        },
        {
            'name': 'S&P Global Manufacturing PMI (Dec)',
            'actual': '48.3',
            'forecast': '49.4',
            'previous': '49.7'
        },
    ]

    # Analyze individual events
    print("\n" + "=" * 70)
    print("INDIVIDUAL EVENT ANALYSIS (AI-Powered)")
    print("=" * 70)

    for event in test_events[:2]:  # First 2 for speed
        print(f"\n🔍 Analyzing: {event['name']}...")

        import time
        start = time.time()

        analysis = analyze_single_event(
            event['name'],
            event['actual'],
            event['forecast'],
            event['previous']
        )

        elapsed = time.time() - start
        print(f"   ⏱️ Analysis took {elapsed:.1f}s")
        print("-" * 70)

        # Show basic stats
        emoji = {'BULLISH': '🟢', 'BEARISH': '🔴', 'NEUTRAL': '🟡'}.get(analysis.market_impact, '⚪')
        print(f"\n{emoji} {analysis.event_name}")
        print(f"   Actual: {analysis.actual} | Exp: {analysis.forecast} | Prev: {analysis.previous}")
        print(f"   Surprise: {analysis.surprise} ({analysis.surprise_pct:+.1f}%)")

        # Show AI analysis
        print(f"\n📝 AI ANALYSIS:")
        print(analysis.ai_interpretation)
        print("-" * 70)

    # Overall analysis
    print("\n" + "=" * 70)
    print("OVERALL MARKET ASSESSMENT (AI-Powered)")
    print("=" * 70)

    print("\n🔍 Analyzing all events together...\n")
    overall = analyze_all_events(test_events)
    print(overall)

    print("\n" + "=" * 70)
    print("✅ AI ANALYSIS COMPLETE")
    print("=" * 70)