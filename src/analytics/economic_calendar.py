"""
Economic Calendar Module - Using investpy (investing.com data)

Fetches LIVE economic calendar data from investing.com via investpy library.
Shows HIGH IMPACT events only in local timezone (Zurich).

Install: pip install investpy

Author: Alpha Research Platform
"""

import os
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EconomicEvent:
    """Single economic event."""
    event_date: date
    event_time: str  # Local time (Zurich)
    event_type: str  # 'FED', 'CPI', 'JOBS', 'GDP', 'PMI', etc.
    event_name: str
    country: str
    impact: str  # 'HIGH', 'MEDIUM', 'LOW'
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None


@dataclass
class EconomicCalendar:
    """Economic calendar data."""
    last_updated: datetime
    events: List[EconomicEvent] = field(default_factory=list)

    # Quick lookups
    next_fed_meeting: Optional[date] = None
    days_to_fed: int = 999
    fed_meeting_this_week: bool = False
    last_fed_meeting: Optional[date] = None

    next_cpi: Optional[date] = None
    days_to_cpi: int = 999

    next_jobs: Optional[date] = None
    days_to_jobs: int = 999

    high_impact_today: bool = False
    high_impact_this_week: bool = False
    today_events: List[EconomicEvent] = field(default_factory=list)
    week_events: List[EconomicEvent] = field(default_factory=list)


class EconomicCalendarFetcher:
    """
    Fetches economic calendar from investing.com via investpy.
    """

    def __init__(self):
        self._cache = None
        self._cache_time = None
        self._cache_duration = timedelta(minutes=30)  # Cache for 30 min

        # Check if investpy is available
        try:
            import investpy
            self._investpy_available = True
            logger.info("investpy library available for economic calendar")
        except ImportError:
            self._investpy_available = False
            logger.error("investpy not installed! Run: pip install investpy")

    def get_calendar(self, force_refresh: bool = False) -> EconomicCalendar:
        """
        Get economic calendar data.

        Args:
            force_refresh: Force refresh from API (use for refresh button)

        Returns:
            EconomicCalendar with events and quick lookups
        """
        # Check cache (skip if force_refresh)
        if not force_refresh and self._cache and self._cache_time:
            if datetime.now() - self._cache_time < self._cache_duration:
                return self._cache

        calendar = EconomicCalendar(last_updated=datetime.now())
        today = date.today()

        # Fetch from investpy
        events = self._fetch_from_investpy(today)

        if not events:
            logger.warning("No events fetched from investpy")

        calendar.events = events

        # Process events for quick lookups
        self._process_events(calendar, today)

        # Cache result
        self._cache = calendar
        self._cache_time = datetime.now()

        return calendar

    def refresh(self) -> EconomicCalendar:
        """Force refresh calendar data (for refresh button)."""
        logger.info("Refreshing economic calendar...")
        return self.get_calendar(force_refresh=True)

    def _fetch_from_investpy(self, today: date) -> List[EconomicEvent]:
        """Fetch economic calendar from investing.com via investpy."""
        events = []

        if not self._investpy_available:
            logger.error("investpy not available")
            return events

        try:
            import investpy

            # Get past 14 days (for last Fed meeting) and next 30 days
            from_date = (today - timedelta(days=14)).strftime('%d/%m/%Y')
            to_date = (today + timedelta(days=30)).strftime('%d/%m/%Y')

            logger.info(f"Fetching economic calendar: {from_date} to {to_date}")

            # Fetch US economic calendar (times are in local timezone)
            df = investpy.economic_calendar(
                countries=['united states'],
                from_date=from_date,
                to_date=to_date
            )

            if df is None or df.empty:
                logger.warning("investpy returned empty calendar")
                return events

            logger.info(f"Fetched {len(df)} events from investing.com")

            # Process each row
            for _, row in df.iterrows():
                try:
                    # Parse date
                    event_date_str = str(row.get('date', ''))
                    event_date = None
                    for fmt in ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y']:
                        try:
                            event_date = datetime.strptime(event_date_str, fmt).date()
                            break
                        except:
                            continue

                    if event_date is None:
                        continue

                    # Get time (already in local/Zurich time)
                    event_time = str(row.get('time', ''))
                    if event_time == 'nan' or not event_time:
                        event_time = ''

                    # Get event name and clean it
                    event_name = str(row.get('event', '')).strip()

                    # Map importance to impact
                    importance = str(row.get('importance', 'low')).lower()
                    impact = 'HIGH' if importance == 'high' else 'MEDIUM' if importance == 'medium' else 'LOW'

                    # Classify event type
                    event_type = self._classify_event_type(event_name)

                    # Get values
                    actual = str(row.get('actual', '')) if pd.notna(row.get('actual')) else None
                    forecast = str(row.get('forecast', '')) if pd.notna(row.get('forecast')) else None
                    previous = str(row.get('previous', '')) if pd.notna(row.get('previous')) else None

                    # Clean None strings
                    if actual == 'None': actual = None
                    if forecast == 'None': forecast = None
                    if previous == 'None': previous = None

                    event = EconomicEvent(
                        event_date=event_date,
                        event_time=event_time,
                        event_type=event_type,
                        event_name=event_name,
                        country='US',
                        impact=impact,
                        actual=actual,
                        forecast=forecast,
                        previous=previous,
                    )
                    events.append(event)

                except Exception as e:
                    logger.debug(f"Error parsing event row: {e}")
                    continue

            logger.info(f"Parsed {len(events)} US economic events")

        except Exception as e:
            logger.error(f"Error fetching from investpy: {e}")
            import traceback
            logger.error(traceback.format_exc())

        return events

    def _classify_event_type(self, event_name: str) -> str:
        """Classify event type from event name."""
        name_lower = event_name.lower()

        # Fed/FOMC
        if any(x in name_lower for x in ['fomc', 'federal funds', 'fed rate', 'fed speaks', 'powell', 'waller', 'williams', 'bostic']):
            return 'FED'

        # Jobs/Employment
        if any(x in name_lower for x in ['nonfarm', 'payroll', 'unemployment', 'jobless', 'employment', 'adp']):
            return 'JOBS'

        # CPI/Inflation
        if any(x in name_lower for x in ['cpi', 'consumer price', 'inflation', 'pce']):
            return 'CPI'

        # GDP
        if 'gdp' in name_lower:
            return 'GDP'

        # PMI
        if 'pmi' in name_lower:
            return 'PMI'

        # Retail
        if 'retail' in name_lower:
            return 'RETAIL'

        # PPI
        if 'ppi' in name_lower or 'producer price' in name_lower:
            return 'PPI'

        # ISM
        if 'ism' in name_lower:
            return 'ISM'

        # Housing
        if any(x in name_lower for x in ['housing', 'home', 'building permit', 'existing home', 'new home']):
            return 'HOUSING'

        # Consumer sentiment
        if any(x in name_lower for x in ['consumer confidence', 'michigan', 'sentiment']):
            return 'SENTIMENT'

        return 'OTHER'

    def _process_events(self, calendar: EconomicCalendar, today: date):
        """Process events to populate quick lookup fields."""
        week_end = today + timedelta(days=7)

        for event in calendar.events:
            # Today's events (HIGH impact only for today_events)
            if event.event_date == today:
                if event.impact == 'HIGH':
                    calendar.today_events.append(event)
                    calendar.high_impact_today = True

            # This week's HIGH impact events
            if today <= event.event_date <= week_end:
                if event.impact == 'HIGH':
                    calendar.week_events.append(event)
                    calendar.high_impact_this_week = True

            # Fed meetings/events (look for actual decisions, not speeches)
            if event.event_type == 'FED':
                name_lower = event.event_name.lower()
                # FOMC decisions contain "rate" or "federal funds"
                if any(x in name_lower for x in ['federal funds', 'interest rate decision', 'fomc statement']):
                    if event.event_date > today:
                        if calendar.next_fed_meeting is None or event.event_date < calendar.next_fed_meeting:
                            calendar.next_fed_meeting = event.event_date
                            calendar.days_to_fed = (event.event_date - today).days
                            calendar.fed_meeting_this_week = calendar.days_to_fed <= 7

                    if event.event_date <= today:
                        if calendar.last_fed_meeting is None or event.event_date > calendar.last_fed_meeting:
                            calendar.last_fed_meeting = event.event_date

            # Next CPI
            if event.event_type == 'CPI' and event.event_date > today:
                if 'cpi' in event.event_name.lower() and 'core' not in event.event_name.lower():
                    if calendar.next_cpi is None or event.event_date < calendar.next_cpi:
                        calendar.next_cpi = event.event_date
                        calendar.days_to_cpi = (event.event_date - today).days

            # Next Jobs report (NFP specifically)
            if event.event_type == 'JOBS' and event.event_date > today:
                if 'nonfarm' in event.event_name.lower():
                    if calendar.next_jobs is None or event.event_date < calendar.next_jobs:
                        calendar.next_jobs = event.event_date
                        calendar.days_to_jobs = (event.event_date - today).days

        # Sort today's events by time
        calendar.today_events.sort(key=lambda x: x.event_time or '99:99')

    def get_calendar_summary(self) -> str:
        """
        Get formatted calendar summary.
        Shows HIGH IMPACT events only in Zurich time.
        """
        cal = self.get_calendar()
        today = date.today()

        lines = [
            f"📅 ECONOMIC CALENDAR - {today.strftime('%A, %B %d, %Y')}",
            f"   Times shown in Zurich (local)",
            f"{'='*60}",
            ""
        ]

        # Today's HIGH impact events only
        if cal.today_events:
            lines.append(f"📌 TODAY'S HIGH IMPACT ({len(cal.today_events)} events):")
            lines.append("")

            current_time = ""
            for event in cal.today_events:
                # Group by time
                if event.event_time and event.event_time != current_time:
                    current_time = event.event_time
                    lines.append(f"   ⏰ {event.event_time}")

                # Format values
                values = []
                if event.actual:
                    values.append(f"Actual: {event.actual}")
                if event.forecast:
                    values.append(f"Exp: {event.forecast}")
                if event.previous:
                    values.append(f"Prev: {event.previous}")

                value_str = f" [{', '.join(values)}]" if values else ""

                # Clean event name
                name = event.event_name.replace('  ', ' ')
                if len(name) > 45:
                    name = name[:42] + "..."

                lines.append(f"      🔴 {name}{value_str}")

            lines.append("")
        else:
            lines.append("📌 No high-impact US events today")
            lines.append("")

        # Key dates
        lines.append("📆 KEY DATES:")

        if cal.last_fed_meeting:
            days_since = (today - cal.last_fed_meeting).days
            lines.append(f"   Last Fed: {cal.last_fed_meeting.strftime('%b %d')} ({days_since}d ago)")

        if cal.next_fed_meeting:
            lines.append(f"   Next Fed: {cal.next_fed_meeting.strftime('%b %d')} ({cal.days_to_fed}d)")

        if cal.next_cpi:
            lines.append(f"   Next CPI: {cal.next_cpi.strftime('%b %d')} ({cal.days_to_cpi}d)")

        if cal.next_jobs:
            lines.append(f"   Next NFP: {cal.next_jobs.strftime('%b %d')} ({cal.days_to_jobs}d)")

        # Upcoming HIGH impact this week (future days only)
        future_high = [e for e in cal.week_events if e.event_date > today]

        if future_high:
            lines.append("")
            lines.append("🗓️ UPCOMING HIGH IMPACT:")
            seen = set()
            for event in sorted(future_high, key=lambda x: (x.event_date, x.event_time or '')):
                key = (event.event_date, event.event_name)
                if key not in seen:
                    seen.add(key)
                    name = event.event_name[:35]
                    lines.append(f"   {event.event_date.strftime('%a %d')} {event.event_time or ''} - {name}")
                if len(seen) >= 5:
                    break

        # Last updated
        lines.append("")
        lines.append(f"⟳ Last updated: {cal.last_updated.strftime('%H:%M:%S')}")

        return "\n".join(lines)

    def get_today_events_for_analysis(self) -> List[Dict]:
        """
        Get today's high impact events in a format suitable for AI analysis.
        Returns list of dicts with event details.
        """
        cal = self.get_calendar()

        events_data = []
        for event in cal.today_events:
            events_data.append({
                'name': event.event_name,
                'time': event.event_time,
                'type': event.event_type,
                'actual': event.actual,
                'forecast': event.forecast,
                'previous': event.previous,
                'surprise': self._calculate_surprise(event)
            })

        return events_data

    def _calculate_surprise(self, event: EconomicEvent) -> Optional[str]:
        """Calculate if actual beat/missed expectations."""
        if not event.actual or not event.forecast:
            return None

        try:
            # Parse values (remove %, K, M, B suffixes)
            actual_str = event.actual.replace('%', '').replace('K', '').replace('M', '').replace('B', '').strip()
            forecast_str = event.forecast.replace('%', '').replace('K', '').replace('M', '').replace('B', '').strip()

            actual_val = float(actual_str)
            forecast_val = float(forecast_str)

            diff = actual_val - forecast_val
            pct_diff = (diff / abs(forecast_val)) * 100 if forecast_val != 0 else 0

            if pct_diff > 5:
                return "BEAT (strong)"
            elif pct_diff > 0:
                return "BEAT"
            elif pct_diff < -5:
                return "MISS (significant)"
            elif pct_diff < 0:
                return "MISS"
            else:
                return "IN-LINE"

        except:
            return None


# ============================================================
# Convenience functions
# ============================================================
_calendar_fetcher = None

def get_calendar_fetcher() -> EconomicCalendarFetcher:
    """Get the singleton calendar fetcher instance."""
    global _calendar_fetcher
    if _calendar_fetcher is None:
        _calendar_fetcher = EconomicCalendarFetcher()
    return _calendar_fetcher

def get_economic_calendar(force_refresh: bool = False) -> EconomicCalendar:
    """Get economic calendar."""
    return get_calendar_fetcher().get_calendar(force_refresh=force_refresh)

def get_calendar_summary() -> str:
    """Get formatted calendar summary (HIGH impact only, Zurich time)."""
    return get_calendar_fetcher().get_calendar_summary()

def refresh_calendar() -> EconomicCalendar:
    """Force refresh calendar data."""
    return get_calendar_fetcher().refresh()

def is_high_impact_day() -> bool:
    """Check if today has high impact events."""
    cal = get_economic_calendar()
    return cal.high_impact_today

def days_to_next_fed() -> int:
    """Get days until next Fed meeting."""
    cal = get_economic_calendar()
    return cal.days_to_fed

def get_events_for_analysis() -> List[Dict]:
    """Get today's events in format for AI analysis."""
    return get_calendar_fetcher().get_today_events_for_analysis()


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    print("Testing Economic Calendar (HIGH impact only, Zurich time)...\n")

    fetcher = EconomicCalendarFetcher()

    # Get summary
    print(fetcher.get_calendar_summary())

    # Get events for analysis
    print(f"\n{'='*60}")
    print("Events for AI Analysis:")
    print("-" * 60)
    for e in fetcher.get_today_events_for_analysis():
        print(f"  {e['name']}")
        print(f"    Actual: {e['actual']}, Expected: {e['forecast']}, Previous: {e['previous']}")
        if e['surprise']:
            print(f"    Surprise: {e['surprise']}")
        print()