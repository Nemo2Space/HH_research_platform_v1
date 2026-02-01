"""
Earnings Transcript Analyzer

Auto-fetches and analyzes earnings call transcripts and SEC filings.
Integrates with signals to adjust scores based on earnings sentiment.

Features:
- Fetch transcripts from SEC EDGAR (8-K, 10-Q, 10-K)
- AI analysis of earnings calls (guidance, tone, surprises)
- Score adjustment based on earnings sentiment
- Support for portfolio-only or full universe

Author: Alpha Research Platform
"""

import os
import re
import json
import requests
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import pandas as pd
import yfinance as yf

from src.utils.logging import get_logger
from src.db.connection import get_engine, get_connection

logger = get_logger(__name__)

# SEC EDGAR base URL
SEC_EDGAR_BASE = "https://www.sec.gov"
SEC_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
SEC_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"

# Headers required by SEC (they block requests without proper User-Agent)
SEC_HEADERS = {
    "User-Agent": "AlphaResearchPlatform contact@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json"
}


@dataclass
class EarningsResult:
    """Structured earnings analysis result."""
    ticker: str
    filing_date: date
    filing_type: str  # 8-K, 10-Q, 10-K
    fiscal_period: str  # Q1 2024, FY 2024

    # Quantitative
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    eps_surprise_pct: Optional[float] = None
    revenue_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_surprise_pct: Optional[float] = None

    # Guidance
    guidance_direction: str = "MAINTAINED"  # RAISED, LOWERED, MAINTAINED, NOT_PROVIDED
    guidance_summary: str = ""

    # AI Analysis
    overall_sentiment: str = "NEUTRAL"  # VERY_BULLISH, BULLISH, NEUTRAL, BEARISH, VERY_BEARISH
    sentiment_score: int = 50  # 0-100
    management_tone: str = "NEUTRAL"  # CONFIDENT, CAUTIOUS, DEFENSIVE, OPTIMISTIC
    key_highlights: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)

    # Score Impact
    score_adjustment: int = 0  # -20 to +20
    adjustment_reason: str = ""

    # Raw data
    transcript_url: str = ""
    transcript_text: str = ""
    analysis_date: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EarningsCalendarItem:
    """Upcoming earnings event."""
    ticker: str
    company_name: str
    earnings_date: date
    time_of_day: str  # BMO (Before Market Open), AMC (After Market Close)
    eps_estimate: Optional[float] = None
    revenue_estimate: Optional[float] = None
    days_until: int = 0


class EarningsTranscriptAnalyzer:
    """
    Fetches and analyzes earnings transcripts from SEC EDGAR.
    """

    def __init__(self, llm_client=None):
        """
        Initialize analyzer.

        Args:
            llm_client: Optional LLM client for AI analysis (OpenAI compatible)
        """
        self.engine = get_engine()
        self.llm_client = llm_client
        self._cik_cache = {}
        self._load_cik_mapping()

        # Try to initialize LLM if not provided
        if not self.llm_client:
            self._init_llm()

    def _init_llm(self):
        """Initialize LLM client for analysis."""
        try:
            from openai import OpenAI

            # Try local LLM first
            local_url = os.getenv("LOCAL_LLM_URL", "http://localhost:8090/v1")
            self.llm_client = OpenAI(
                base_url=local_url,
                api_key="not-needed"
            )
            # Test connection
            self.llm_client.models.list()
            logger.info(f"Connected to local LLM at {local_url}")
        except Exception as e:
            logger.warning(f"Local LLM not available: {e}")
            # Fall back to OpenAI if available
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=api_key)
                logger.info("Using OpenAI for earnings analysis")
            else:
                logger.warning("No LLM available - will use rule-based analysis only")
                self.llm_client = None

    def _load_cik_mapping(self):
        """Load ticker to CIK mapping from SEC."""
        try:
            # Try to load from cache file first
            cache_file = "data/sec_cik_mapping.json"
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    self._cik_cache = json.load(f)
                logger.info(f"Loaded {len(self._cik_cache)} CIK mappings from cache")
                return

            # Fetch from SEC
            response = requests.get(SEC_COMPANY_TICKERS, headers=SEC_HEADERS, timeout=30)
            if response.status_code == 200:
                data = response.json()
                for key, item in data.items():
                    ticker = item.get('ticker', '').upper()
                    cik = str(item.get('cik_str', '')).zfill(10)
                    if ticker and cik:
                        self._cik_cache[ticker] = cik

                # Save cache
                os.makedirs("data", exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump(self._cik_cache, f)
                logger.info(f"Loaded {len(self._cik_cache)} CIK mappings from SEC")
        except Exception as e:
            logger.error(f"Error loading CIK mapping: {e}")

    def get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK number for a ticker."""
        return self._cik_cache.get(ticker.upper())

    def get_upcoming_earnings(self, tickers: List[str] = None,
                              days_ahead: int = 14) -> List[EarningsCalendarItem]:
        """
        Get upcoming earnings dates for tickers.

        Args:
            tickers: List of tickers (None = get from universe)
            days_ahead: How many days to look ahead

        Returns:
            List of EarningsCalendarItem
        """
        if tickers is None:
            # Get from database
            query = "SELECT DISTINCT ticker FROM screener_scores WHERE date >= CURRENT_DATE - INTERVAL '7 days'"
            df = pd.read_sql(query, self.engine)
            tickers = df['ticker'].tolist()

        upcoming = []
        today = date.today()
        cutoff = today + timedelta(days=days_ahead)

        logger.info(f"Checking earnings for {len(tickers)} tickers (next {days_ahead} days)")

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                earnings_date = None
                eps_estimate = None
                company_name = ticker

                # Method 1: Try calendar (dict format)
                try:
                    calendar = stock.calendar
                    if calendar is not None and isinstance(calendar, dict):
                        ed = calendar.get('Earnings Date')
                        if ed:
                            # ed is usually a list with one or two dates
                            if isinstance(ed, list) and len(ed) > 0:
                                earnings_date = ed[0]  # Already datetime.date
                            elif isinstance(ed, date):
                                earnings_date = ed

                        # Get estimate from calendar
                        eps_estimate = calendar.get('Earnings Average')

                except Exception as e:
                    logger.debug(f"{ticker} calendar error: {e}")

                # Method 2: Try earnings_dates DataFrame (more reliable)
                if earnings_date is None:
                    try:
                        earnings_dates_df = stock.earnings_dates
                        if earnings_dates_df is not None and not earnings_dates_df.empty:
                            # Get future earnings dates
                            future_dates = earnings_dates_df[
                                earnings_dates_df.index >= pd.Timestamp.now(tz='America/New_York')
                                ]
                            if not future_dates.empty:
                                # Get the next one
                                next_earnings = future_dates.index[0]
                                earnings_date = next_earnings.date()

                                # Get estimate
                                if 'EPS Estimate' in future_dates.columns:
                                    est = future_dates['EPS Estimate'].iloc[0]
                                    if pd.notna(est):
                                        eps_estimate = float(est)
                    except Exception as e:
                        logger.debug(f"{ticker} earnings_dates error: {e}")

                # Get company name
                try:
                    info = stock.info
                    company_name = info.get('shortName', ticker)
                except:
                    pass

                # Check if within range
                if earnings_date and today <= earnings_date <= cutoff:
                    days_until = (earnings_date - today).days

                    upcoming.append(EarningsCalendarItem(
                        ticker=ticker,
                        company_name=company_name,
                        earnings_date=earnings_date,
                        time_of_day="AMC/BMO",  # Usually not specified
                        eps_estimate=eps_estimate,
                        days_until=days_until
                    ))
                    logger.debug(f"{ticker}: Earnings on {earnings_date} ({days_until} days)")

            except Exception as e:
                logger.debug(f"Error getting earnings for {ticker}: {e}")
                continue

        # Sort by date
        upcoming.sort(key=lambda x: x.earnings_date)
        logger.info(f"Found {len(upcoming)} upcoming earnings in next {days_ahead} days")
        return upcoming

    def get_recent_earnings(self, ticker: str, days_back: int = 90) -> List[Dict]:
        """
        Get recent earnings filings from SEC EDGAR.

        Args:
            ticker: Stock ticker
            days_back: How many days to look back

        Returns:
            List of filing metadata
        """
        cik = self.get_cik(ticker)
        if not cik:
            logger.warning(f"No CIK found for {ticker}")
            return []

        try:
            # SEC EDGAR API for company filings
            url = f"{SEC_EDGAR_BASE}/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K&dateb=&owner=include&count=10&output=json"

            response = requests.get(url, headers=SEC_HEADERS, timeout=30)

            if response.status_code != 200:
                # Try alternative endpoint
                url = f"{SEC_EDGAR_BASE}/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-&dateb=&owner=include&count=10&output=json"
                response = requests.get(url, headers=SEC_HEADERS, timeout=30)

            filings = []

            # Parse response (SEC returns various formats)
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Process filings
                    if 'filings' in data:
                        for filing in data['filings'].get('recent', {}).get('form', []):
                            filings.append(filing)
                except:
                    # HTML response - parse differently
                    pass

            return filings

        except Exception as e:
            logger.error(f"Error fetching SEC filings for {ticker}: {e}")
            return []

    def fetch_filing_text(self, ticker: str, filing_url: str) -> str:
        """
        Fetch the text content of a filing.

        Args:
            ticker: Stock ticker
            filing_url: URL to the filing

        Returns:
            Text content of the filing
        """
        try:
            response = requests.get(filing_url, headers=SEC_HEADERS, timeout=60)
            if response.status_code == 200:
                # Clean HTML/XML
                text = response.text
                # Basic HTML tag removal
                text = re.sub(r'<[^>]+>', ' ', text)
                # Clean whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                return text[:50000]  # Limit to 50k chars for LLM
            return ""
        except Exception as e:
            logger.error(f"Error fetching filing: {e}")
            return ""

    def fetch_earnings_from_yfinance(self, ticker: str) -> Dict[str, Any]:
        """
        Get earnings data from Yahoo Finance (faster than SEC).

        Args:
            ticker: Stock ticker

        Returns:
            Earnings data dict
        """
        try:
            stock = yf.Ticker(ticker)

            result = {
                'ticker': ticker,
                'has_data': False
            }

            # Get earnings dates and estimates from calendar
            try:
                calendar = stock.calendar
                if calendar is not None:
                    if isinstance(calendar, pd.DataFrame) and not calendar.empty:
                        # Extract earnings estimates if available
                        if 'Earnings Average' in calendar.index:
                            result['eps_estimate'] = calendar.loc['Earnings Average'].iloc[0]
                        if 'Revenue Average' in calendar.index:
                            result['revenue_estimate'] = calendar.loc['Revenue Average'].iloc[0]
            except Exception as e:
                logger.debug(f"Calendar error for {ticker}: {e}")

            # Get actual EPS from quarterly income statement
            try:
                quarterly_income = stock.quarterly_income_stmt
                if quarterly_income is not None and not quarterly_income.empty:
                    # Get Basic EPS from income statement
                    if 'Basic EPS' in quarterly_income.index:
                        eps_row = quarterly_income.loc['Basic EPS']
                        if len(eps_row) > 0:
                            result['eps_actual'] = float(eps_row.iloc[0])
                            result['has_data'] = True
                    elif 'Diluted EPS' in quarterly_income.index:
                        eps_row = quarterly_income.loc['Diluted EPS']
                        if len(eps_row) > 0:
                            result['eps_actual'] = float(eps_row.iloc[0])
                            result['has_data'] = True

                    # Get Net Income
                    if 'Net Income' in quarterly_income.index:
                        result['net_income'] = float(quarterly_income.loc['Net Income'].iloc[0])

                    # Get Total Revenue
                    if 'Total Revenue' in quarterly_income.index:
                        result['revenue'] = float(quarterly_income.loc['Total Revenue'].iloc[0])
                    elif 'Operating Revenue' in quarterly_income.index:
                        result['revenue'] = float(quarterly_income.loc['Operating Revenue'].iloc[0])
            except Exception as e:
                logger.debug(f"Income statement error for {ticker}: {e}")

            # Calculate surprise if we have both actual and estimate
            if result.get('eps_actual') and result.get('eps_estimate'):
                estimate = result['eps_estimate']
                actual = result['eps_actual']
                if estimate != 0:
                    result['eps_surprise_pct'] = ((actual - estimate) / abs(estimate)) * 100
                    result['has_data'] = True

            # Try to get analyst estimates if not available
            if not result.get('eps_estimate'):
                try:
                    # Try earnings_estimate attribute
                    earnings_est = stock.earnings_estimate
                    if earnings_est is not None and not earnings_est.empty:
                        if 'avg' in earnings_est.columns:
                            # Get current quarter estimate
                            result['eps_estimate'] = float(earnings_est['avg'].iloc[0])
                except:
                    pass

                # Try analyst_price_targets for context
                try:
                    if not result.get('eps_estimate'):
                        info = stock.info
                        # forwardEps can give us an idea
                        if info.get('forwardEps'):
                            result['forward_eps'] = info.get('forwardEps')
                except:
                    pass

            # Method 3: Use earnings_dates for historical surprise data
            try:
                earnings_dates_df = stock.earnings_dates
                if earnings_dates_df is not None and not earnings_dates_df.empty:
                    # Get the most recent reported earnings (not future)
                    past_earnings = earnings_dates_df[
                        earnings_dates_df['Reported EPS'].notna()
                    ]
                    if not past_earnings.empty:
                        latest = past_earnings.iloc[0]

                        # Update with actual data
                        if pd.notna(latest.get('Reported EPS')):
                            result['eps_actual'] = float(latest['Reported EPS'])
                            result['has_data'] = True

                        if pd.notna(latest.get('EPS Estimate')):
                            result['eps_estimate'] = float(latest['EPS Estimate'])

                        if pd.notna(latest.get('Surprise(%)')):
                            result['eps_surprise_pct'] = float(latest['Surprise(%)'])

                        # Get report date from index
                        result['report_date'] = str(latest.name.date())
            except Exception as e:
                logger.debug(f"earnings_dates error for {ticker}: {e}")

            # Recalculate surprise if we now have estimate but no surprise
            if result.get('eps_actual') and result.get('eps_estimate') and not result.get('eps_surprise_pct'):
                estimate = result['eps_estimate']
                actual = result['eps_actual']
                if estimate != 0:
                    result['eps_surprise_pct'] = ((actual - estimate) / abs(estimate)) * 100
                    result['has_data'] = True

            # Get company info for context
            try:
                info = stock.info
                result['company_name'] = info.get('shortName', ticker)
                result['sector'] = info.get('sector', '')
                result['forward_pe'] = info.get('forwardPE')
                result['trailing_pe'] = info.get('trailingPE')
            except Exception as e:
                logger.debug(f"Info error for {ticker}: {e}")
                result['company_name'] = ticker

            return result

        except Exception as e:
            logger.error(f"Error fetching earnings from yfinance for {ticker}: {e}")
            return {'ticker': ticker, 'has_data': False, 'error': str(e)}

    def analyze_earnings(self, ticker: str,
                         earnings_data: Dict = None,
                         transcript_text: str = None) -> EarningsResult:
        """
        Analyze earnings using AI.

        Args:
            ticker: Stock ticker
            earnings_data: Optional pre-fetched earnings data
            transcript_text: Optional transcript text to analyze

        Returns:
            EarningsResult with analysis
        """
        # Fetch data if not provided
        if earnings_data is None:
            earnings_data = self.fetch_earnings_from_yfinance(ticker)

        result = EarningsResult(
            ticker=ticker,
            filing_date=date.today(),
            filing_type="EARNINGS",
            fiscal_period=f"Q{((date.today().month - 1) // 3) + 1} {date.today().year}"
        )

        # Fill in quantitative data
        if earnings_data.get('has_data'):
            result.eps_actual = earnings_data.get('eps_actual')
            result.eps_estimate = earnings_data.get('eps_estimate')
            result.eps_surprise_pct = earnings_data.get('eps_surprise_pct')
            result.revenue_actual = earnings_data.get('revenue')

        # Calculate initial score based on surprise
        surprise_pct = result.eps_surprise_pct or 0

        if surprise_pct >= 10:
            result.sentiment_score = 85
            result.overall_sentiment = "VERY_BULLISH"
            result.score_adjustment = 15
        elif surprise_pct >= 5:
            result.sentiment_score = 70
            result.overall_sentiment = "BULLISH"
            result.score_adjustment = 10
        elif surprise_pct >= 0:
            result.sentiment_score = 55
            result.overall_sentiment = "NEUTRAL"
            result.score_adjustment = 5
        elif surprise_pct >= -5:
            result.sentiment_score = 40
            result.overall_sentiment = "BEARISH"
            result.score_adjustment = -5
        else:
            result.sentiment_score = 25
            result.overall_sentiment = "VERY_BEARISH"
            result.score_adjustment = -15

        # If we have transcript text, do AI analysis
        if transcript_text and self.llm_client:
            ai_analysis = self._analyze_with_llm(ticker, transcript_text, earnings_data)
            if ai_analysis:
                result.guidance_direction = ai_analysis.get('guidance_direction', 'MAINTAINED')
                result.guidance_summary = ai_analysis.get('guidance_summary', '')
                result.management_tone = ai_analysis.get('management_tone', 'NEUTRAL')
                result.key_highlights = ai_analysis.get('key_highlights', [])
                result.concerns = ai_analysis.get('concerns', [])

                # Adjust score based on guidance
                if result.guidance_direction == "RAISED":
                    result.score_adjustment += 5
                    result.sentiment_score = min(100, result.sentiment_score + 10)
                elif result.guidance_direction == "LOWERED":
                    result.score_adjustment -= 5
                    result.sentiment_score = max(0, result.sentiment_score - 10)

        # Set adjustment reason
        reasons = []
        if result.eps_surprise_pct:
            if result.eps_surprise_pct > 0:
                reasons.append(f"EPS beat by {result.eps_surprise_pct:.1f}%")
            else:
                reasons.append(f"EPS missed by {abs(result.eps_surprise_pct):.1f}%")
        if result.guidance_direction != "MAINTAINED":
            reasons.append(f"Guidance {result.guidance_direction.lower()}")

        result.adjustment_reason = "; ".join(reasons) if reasons else "Earnings analysis"

        return result

    def _analyze_with_llm(self, ticker: str, transcript: str,
                          earnings_data: Dict) -> Optional[Dict]:
        """
        Use LLM to analyze earnings transcript.

        Args:
            ticker: Stock ticker
            transcript: Transcript text
            earnings_data: Quantitative earnings data

        Returns:
            Analysis dict or None
        """
        if not self.llm_client:
            return None

        prompt = f"""Analyze this earnings call transcript for {ticker}.

EARNINGS DATA:
- EPS Actual: {earnings_data.get('eps_actual', 'N/A')}
- EPS Estimate: {earnings_data.get('eps_estimate', 'N/A')}
- EPS Surprise: {earnings_data.get('eps_surprise_pct', 'N/A')}%

TRANSCRIPT (excerpt):
{transcript[:8000]}

Provide analysis in this exact JSON format:
{{
    "guidance_direction": "RAISED" or "LOWERED" or "MAINTAINED" or "NOT_PROVIDED",
    "guidance_summary": "Brief summary of forward guidance",
    "management_tone": "CONFIDENT" or "CAUTIOUS" or "DEFENSIVE" or "OPTIMISTIC",
    "key_highlights": ["highlight 1", "highlight 2", "highlight 3"],
    "concerns": ["concern 1", "concern 2"]
}}

Return ONLY the JSON, no other text."""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",  # or local model
                messages=[
                    {"role": "system",
                     "content": "You are a financial analyst expert at earnings call analysis. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]

            return json.loads(content)

        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return None

    def analyze_universe(self, tickers: List[str] = None,
                         portfolio_only: bool = False,
                         max_workers: int = 3) -> List[EarningsResult]:
        """
        Analyze earnings for multiple tickers.

        Args:
            tickers: List of tickers (None = auto from DB)
            portfolio_only: If True, only analyze portfolio holdings
            max_workers: Parallel workers

        Returns:
            List of EarningsResult
        """
        if tickers is None:
            if portfolio_only:
                # Get from portfolio
                query = """
                        SELECT DISTINCT ticker
                        FROM portfolio_positions
                        WHERE quantity > 0 \
                        """
            else:
                # Get full universe
                query = """
                        SELECT DISTINCT ticker
                        FROM screener_scores
                        WHERE date >= CURRENT_DATE - INTERVAL '7 days' \
                        """
            try:
                df = pd.read_sql(query, self.engine)
                tickers = df['ticker'].tolist()
            except:
                tickers = []

        logger.info(f"Analyzing earnings for {len(tickers)} tickers")

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.analyze_earnings, ticker): ticker
                for ticker in tickers
            }

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing {ticker}: {e}")

        return results

    def save_earnings_analysis(self, result: EarningsResult) -> bool:
        """
        Save earnings analysis to database.

        Args:
            result: EarningsResult to save

        Returns:
            True if saved successfully
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                                INSERT INTO earnings_analysis (ticker, filing_date, filing_type, fiscal_period,
                                                               eps_actual, eps_estimate, eps_surprise_pct,
                                                               revenue_actual, revenue_estimate, revenue_surprise_pct,
                                                               guidance_direction, guidance_summary,
                                                               overall_sentiment, sentiment_score, management_tone,
                                                               key_highlights, concerns,
                                                               score_adjustment, adjustment_reason,
                                                               transcript_url, analysis_date)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (ticker, filing_date) DO
                                UPDATE SET
                                    sentiment_score = EXCLUDED.sentiment_score,
                                    score_adjustment = EXCLUDED.score_adjustment,
                                    guidance_direction = EXCLUDED.guidance_direction,
                                    analysis_date = EXCLUDED.analysis_date
                                """, (
                                    result.ticker, result.filing_date, result.filing_type,
                                    result.fiscal_period, result.eps_actual, result.eps_estimate,
                                    result.eps_surprise_pct, result.revenue_actual,
                                    result.revenue_estimate, result.revenue_surprise_pct,
                                    result.guidance_direction, result.guidance_summary,
                                    result.overall_sentiment, result.sentiment_score,
                                    result.management_tone, json.dumps(result.key_highlights),
                                    json.dumps(result.concerns), result.score_adjustment,
                                    result.adjustment_reason, result.transcript_url,
                                    result.analysis_date
                                ))
                    conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving earnings analysis: {e}")
            return False

    def update_signal_with_earnings(self, ticker: str,
                                    earnings_result: EarningsResult) -> bool:
        """
        Update screener scores with earnings adjustment.

        Args:
            ticker: Stock ticker
            earnings_result: Earnings analysis result

        Returns:
            True if updated
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    # Get current score
                    cur.execute("""
                                SELECT total_score
                                FROM screener_scores
                                WHERE ticker = %s
                                ORDER BY date DESC LIMIT 1
                                """, (ticker,))

                    row = cur.fetchone()
                    if row:
                        current_score = row[0] or 50
                        # Apply adjustment (capped at -20 to +20)
                        adjustment = max(-20, min(20, earnings_result.score_adjustment))
                        new_score = max(0, min(100, current_score + adjustment))

                        # Update score
                        cur.execute("""
                                    UPDATE screener_scores
                                    SET total_score         = %s,
                                        earnings_adjustment = %s
                                    WHERE ticker = %s
                                      AND date = (SELECT MAX (date) FROM screener_scores WHERE ticker = %s)
                                    """, (new_score, adjustment, ticker, ticker))

                        conn.commit()
                        logger.info(
                            f"{ticker}: Score adjusted {current_score} → {new_score} ({adjustment:+d}) due to earnings")
                        return True
            return False
        except Exception as e:
            logger.error(f"Error updating signal for {ticker}: {e}")
            return False

    def get_earnings_context_for_ai(self, ticker: str) -> str:
        """
        Get earnings analysis formatted for AI Chat.

        Args:
            ticker: Stock ticker

        Returns:
            Formatted string for AI context
        """
        try:
            query = """
                    SELECT * \
                    FROM earnings_analysis
                    WHERE ticker = %s
                    ORDER BY filing_date DESC LIMIT 1 \
                    """
            df = pd.read_sql(query, self.engine, params=(ticker,))

            if df.empty:
                return ""

            row = df.iloc[0]

            # Safely get values with defaults
            eps_actual = row.get('eps_actual')
            eps_estimate = row.get('eps_estimate')
            eps_surprise = row.get('eps_surprise_pct')
            sentiment_score = row.get('sentiment_score') or 50
            score_adjustment = row.get('score_adjustment') or 0

            # Format EPS line
            eps_actual_str = f"${eps_actual:.2f}" if eps_actual is not None else "N/A"
            eps_estimate_str = f"${eps_estimate:.2f}" if eps_estimate is not None else "N/A"
            eps_surprise_str = f"{eps_surprise:+.1f}%" if eps_surprise is not None else "N/A"

            context = f"""
📊 EARNINGS ANALYSIS: {ticker}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 Report Date: {row.get('filing_date', 'N/A')}
📈 Period: {row.get('fiscal_period', 'N/A')}

💰 RESULTS:
   EPS: {eps_actual_str} vs {eps_estimate_str} est ({eps_surprise_str} surprise)

🔮 GUIDANCE: {row.get('guidance_direction', 'N/A')}
   {row.get('guidance_summary', '') or ''}

🎯 AI ASSESSMENT:
   Sentiment: {row.get('overall_sentiment', 'N/A')} (score: {sentiment_score}/100)
   Management Tone: {row.get('management_tone', 'N/A')}
   Score Impact: {score_adjustment:+d} points

✅ KEY HIGHLIGHTS:
"""
            highlights_raw = row.get('key_highlights', '[]')
            highlights = json.loads(highlights_raw) if highlights_raw else []
            for h in highlights[:3]:
                context += f"   • {h}\n"

            concerns_raw = row.get('concerns', '[]')
            concerns = json.loads(concerns_raw) if concerns_raw else []
            if concerns:
                context += "\n⚠️ CONCERNS:\n"
                for c in concerns[:3]:
                    context += f"   • {c}\n"

            return context

        except Exception as e:
            logger.error(f"Error getting earnings context for {ticker}: {e}")
            return ""


# ============================================================
# Database Migration
# ============================================================
EARNINGS_TABLE_SQL = """
                     CREATE TABLE IF NOT EXISTS earnings_analysis \
                     ( \
                         id \
                         SERIAL \
                         PRIMARY \
                         KEY, \
                         ticker \
                         VARCHAR \
                     ( \
                         20 \
                     ) NOT NULL,
                         filing_date DATE NOT NULL,
                         filing_type VARCHAR \
                     ( \
                         20 \
                     ),
                         fiscal_period VARCHAR \
                     ( \
                         20 \
                     ),

                         eps_actual FLOAT,
                         eps_estimate FLOAT,
                         eps_surprise_pct FLOAT,
                         revenue_actual FLOAT,
                         revenue_estimate FLOAT,
                         revenue_surprise_pct FLOAT,

                         guidance_direction VARCHAR \
                     ( \
                         20 \
                     ),
                         guidance_summary TEXT,

                         overall_sentiment VARCHAR \
                     ( \
                         20 \
                     ),
                         sentiment_score INT,
                         management_tone VARCHAR \
                     ( \
                         20 \
                     ),
                         key_highlights JSONB,
                         concerns JSONB,

                         score_adjustment INT DEFAULT 0,
                         adjustment_reason TEXT,

                         transcript_url TEXT,
                         transcript_text TEXT,
                         analysis_date TIMESTAMP DEFAULT NOW \
                     ( \
                     ),

                         created_at TIMESTAMP DEFAULT NOW \
                     ( \
                     ), \
                         UNIQUE \
                     ( \
                         ticker, \
                         filing_date \
                     )
                         );

                     CREATE INDEX IF NOT EXISTS idx_earnings_ticker ON earnings_analysis(ticker);
                     CREATE INDEX IF NOT EXISTS idx_earnings_date ON earnings_analysis(filing_date);

-- Add earnings_adjustment column to screener_scores if not exists
                     DO \
                     $$
                     BEGIN 
    IF \
                     NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'screener_scores' AND column_name = 'earnings_adjustment'
    ) THEN
                     ALTER TABLE screener_scores \
                         ADD COLUMN earnings_adjustment INT DEFAULT 0;
                     END IF;
                     END $$; \
                     """


def run_migration():
    """Run database migration for earnings tables."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(EARNINGS_TABLE_SQL)
                conn.commit()
        logger.info("Earnings analysis table created/updated")
        return True
    except Exception as e:
        logger.error(f"Migration error: {e}")
        return False


# ============================================================
# Convenience Functions
# ============================================================

def get_earnings_analyzer() -> EarningsTranscriptAnalyzer:
    """Get singleton analyzer instance."""
    return EarningsTranscriptAnalyzer()


def analyze_ticker_earnings(ticker: str) -> EarningsResult:
    """Analyze earnings for a single ticker."""
    analyzer = get_earnings_analyzer()
    return analyzer.analyze_earnings(ticker)


def get_upcoming_earnings(tickers: List[str] = None, days_ahead: int = 14) -> List[EarningsCalendarItem]:
    """Get upcoming earnings for tickers."""
    analyzer = get_earnings_analyzer()
    return analyzer.get_upcoming_earnings(tickers, days_ahead)


def get_earnings_for_ai(ticker: str) -> str:
    """Get earnings context for AI chat."""
    analyzer = get_earnings_analyzer()
    return analyzer.get_earnings_context_for_ai(ticker)


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    # Run migration
    print("Running migration...")
    run_migration()

    # Test analyzer
    print("\nTesting analyzer...")
    analyzer = EarningsTranscriptAnalyzer()

    # Test single ticker
    result = analyzer.analyze_earnings("AAPL")
    print(f"\nAAPL Earnings Analysis:")
    print(f"  EPS Surprise: {result.eps_surprise_pct}%")
    print(f"  Sentiment: {result.overall_sentiment} ({result.sentiment_score}/100)")
    print(f"  Score Adjustment: {result.score_adjustment:+d}")
    print(f"  Reason: {result.adjustment_reason}")

    # Test upcoming earnings
    print("\n\nUpcoming Earnings (next 14 days):")
    upcoming = analyzer.get_upcoming_earnings(['AAPL', 'MSFT', 'GOOGL', 'NVDA'], days_ahead=30)
    for item in upcoming:
        print(f"  {item.ticker}: {item.earnings_date} ({item.days_until} days)")