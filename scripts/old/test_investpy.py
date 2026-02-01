

import investpy
from datetime import date, timedelta
from dataclasses import dataclass
from typing import Optional, List

# ============================================================
# CONFIGURATION - Adjust for your timezone
# ============================================================
# Zurich (UTC+1) to ET (UTC-5) = -6 hours in winter
# Set to 0 if you want times in your local timezone
ET_OFFSET_HOURS = -6


def convert_to_et(time_str: str, offset: int = ET_OFFSET_HOURS) -> str:
    """Convert time string to Eastern Time."""
    if not time_str or offset == 0:
        return time_str

    try:
        parts = time_str.split(':')
        if len(parts) >= 2:
            hour = int(parts[0])
            minute = int(parts[1])

            hour = hour + offset

            if hour < 0:
                hour += 24
            elif hour >= 24:
                hour -= 24

            return f"{hour:02d}:{minute:02d}"
    except:
        pass

    return time_str


@dataclass
class EconomicEvent:
    event_date: date
    event_time: str
    event_name: str
    impact: str
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None


def fetch_calendar(target_date: date = None) -> List[EconomicEvent]:
    """Fetch economic calendar from investing.com."""
    if target_date is None:
        target_date = date.today()

    events = []

    from_date = (target_date - timedelta(days=7)).strftime('%d/%m/%Y')
    to_date = (target_date + timedelta(days=7)).strftime('%d/%m/%Y')

    print(f"Fetching calendar: {from_date} to {to_date}")

    df = investpy.economic_calendar(
        countries=['united states'],
        from_date=from_date,
        to_date=to_date
    )

    if df is None or df.empty:
        print("No data returned!")
        return events

    print(f"Got {len(df)} raw events")

    for _, row in df.iterrows():
        try:
            # Parse date
            event_date_str = str(row.get('date', ''))
            event_date = None
            for fmt in ['%d/%m/%Y', '%Y-%m-%d']:
                try:
                    event_date = __import__('datetime').datetime.strptime(event_date_str, fmt).date()
                    break
                except:
                    continue

            if event_date is None:
                continue

            # Get and convert time
            event_time = str(row.get('time', ''))
            if event_time and event_time != 'nan':
                event_time = convert_to_et(event_time)
            else:
                event_time = ''

            # Impact
            importance = str(row.get('importance', 'low')).lower()
            impact = 'HIGH' if importance == 'high' else 'MEDIUM' if importance == 'medium' else 'LOW'

            # Values
            actual = str(row.get('actual', '')) if row.get('actual') and str(row.get('actual')) != 'nan' else None
            forecast = str(row.get('forecast', '')) if row.get('forecast') and str(row.get('forecast')) != 'nan' else None
            previous = str(row.get('previous', '')) if row.get('previous') and str(row.get('previous')) != 'nan' else None

            events.append(EconomicEvent(
                event_date=event_date,
                event_time=event_time,
                event_name=str(row.get('event', '')).strip(),
                impact=impact,
                actual=actual,
                forecast=forecast,
                previous=previous
            ))

        except Exception as e:
            continue

    return events


def print_calendar_summary(events: List[EconomicEvent], target_date: date = None):
    """Print formatted calendar."""
    if target_date is None:
        target_date = date.today()

    print(f"\n{'='*70}")
    print(f"📅 ECONOMIC CALENDAR - {target_date.strftime('%A, %B %d, %Y')}")
    print(f"{'='*70}")

    # Today's events
    today_events = [e for e in events if e.event_date == target_date]
    today_high_med = [e for e in today_events if e.impact in ['HIGH', 'MEDIUM']]

    if today_high_med:
        high_count = len([e for e in today_high_med if e.impact == 'HIGH'])
        med_count = len([e for e in today_high_med if e.impact == 'MEDIUM'])

        print(f"\n📌 TODAY'S EVENTS ({high_count} high, {med_count} medium impact):\n")

        # Sort by time
        today_high_med.sort(key=lambda x: x.event_time or '99:99')

        current_time = ""
        for event in today_high_med:
            emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(event.impact, '⚪')

            if event.event_time and event.event_time != current_time:
                current_time = event.event_time
                print(f"   ⏰ {event.event_time} ET")

            # Values
            values = []
            if event.forecast:
                values.append(f"Exp: {event.forecast}")
            if event.previous:
                values.append(f"Prev: {event.previous}")
            if event.actual:
                values.append(f"Act: {event.actual}")

            value_str = f" [{', '.join(values)}]" if values else ""
            name = event.event_name[:45] + "..." if len(event.event_name) > 48 else event.event_name

            print(f"      {emoji} {name}{value_str}")
    else:
        print("\n📌 No major US economic events today\n")

    # Key upcoming dates
    print(f"\n📆 KEY DATES:")

    # Find Fed, CPI, NFP dates
    for event in events:
        name_lower = event.event_name.lower()

        if event.event_date <= target_date:
            if 'fed' in name_lower or 'fomc' in name_lower:
                days_ago = (target_date - event.event_date).days
                if days_ago <= 14:
                    print(f"   Last Fed: {event.event_date.strftime('%b %d')} ({days_ago}d ago)")
                    break

    for event in sorted(events, key=lambda x: x.event_date):
        if event.event_date > target_date:
            name_lower = event.event_name.lower()

            if ('fed' in name_lower or 'fomc' in name_lower) and 'rate' in name_lower:
                days = (event.event_date - target_date).days
                print(f"   Next Fed: {event.event_date.strftime('%b %d')} ({days}d)")
                break

    for event in sorted(events, key=lambda x: x.event_date):
        if event.event_date > target_date and 'cpi' in event.event_name.lower():
            days = (event.event_date - target_date).days
            print(f"   Next CPI: {event.event_date.strftime('%b %d')} ({days}d)")
            break

    for event in sorted(events, key=lambda x: x.event_date):
        if event.event_date > target_date and 'nonfarm' in event.event_name.lower():
            days = (event.event_date - target_date).days
            print(f"   Next NFP: {event.event_date.strftime('%b %d')} ({days}d)")
            break


# ============================================================
# MAIN TEST
# ============================================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    print("Testing Economic Calendar...")
    print(f"Timezone offset: {ET_OFFSET_HOURS} hours (to convert to ET)")
    print("-" * 70)

    events = fetch_calendar()

    print(f"\nParsed {len(events)} events")

    # Verify key events have correct times
    today = date.today()
    today_events = [e for e in events if e.event_date == today]

    print(f"\n--- VERIFICATION ---")
    nfp = [e for e in today_events if 'nonfarm' in e.event_name.lower()]
    if nfp:
        print(f"NFP time: {nfp[0].event_time} (should be 08:30 ET)")

    pmi = [e for e in today_events if 'manufacturing pmi' in e.event_name.lower()]
    if pmi:
        print(f"Manufacturing PMI time: {pmi[0].event_time} (should be 09:45 ET)")

    # Print summary
    print_calendar_summary(events)

    print(f"\n{'='*70}")
    print("✅ TEST COMPLETE")
    print(f"{'='*70}")