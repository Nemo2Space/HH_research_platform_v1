

import warnings

warnings.filterwarnings('ignore')

import investpy
from datetime import datetime, date, timedelta

today = date.today()

print("=" * 60)
print(f"📅 ECONOMIC CALENDAR - {today.strftime('%A, %B %d, %Y')}")
print(f"   Times shown in Zurich (local)")
print("=" * 60)

# Fetch
df = investpy.economic_calendar(
    countries=['united states'],
    from_date=(today - timedelta(days=14)).strftime('%d/%m/%Y'),
    to_date=(today + timedelta(days=30)).strftime('%d/%m/%Y')
)

# Filter HIGH impact only
high_impact = df[df['importance'] == 'high'].copy()

# Today's HIGH impact
today_str = today.strftime('%d/%m/%Y')
today_high = high_impact[high_impact['date'] == today_str].sort_values('time')

print(f"\n📌 TODAY'S HIGH IMPACT ({len(today_high)} events):")
print("")

if len(today_high) > 0:
    current_time = ""
    for _, row in today_high.iterrows():
        time = row['time']
        event = row['event'].replace('  ', ' ')
        if len(event) > 45:
            event = event[:42] + "..."

        actual = row['actual'] if row['actual'] and str(row['actual']) != 'nan' else None
        forecast = row['forecast'] if row['forecast'] and str(row['forecast']) != 'nan' else None
        previous = row['previous'] if row['previous'] and str(row['previous']) != 'nan' else None

        if time != current_time:
            current_time = time
            print(f"   ⏰ {time}")

        # Values
        values = []
        if actual:
            values.append(f"Actual: {actual}")
        if forecast:
            values.append(f"Exp: {forecast}")
        if previous:
            values.append(f"Prev: {previous}")

        value_str = f" [{', '.join(values)}]" if values else ""
        print(f"      🔴 {event}{value_str}")
else:
    print("   No high-impact events today")

# Upcoming HIGH impact
print(f"\n🗓️ UPCOMING HIGH IMPACT (next 7 days):")
count = 0
for _, row in high_impact.iterrows():
    try:
        event_date = datetime.strptime(row['date'], '%d/%m/%Y').date()
        if event_date > today and event_date <= today + timedelta(days=7):
            name = row['event'][:40]
            print(f"   {event_date.strftime('%a %d')} {row['time']} - {name}")
            count += 1
            if count >= 8:
                break
    except:
        continue

print(f"\n⟳ Last updated: {datetime.now().strftime('%H:%M:%S')}")
print("=" * 60)