# Deep dive into the data sources

# 1. Check earnings functions
print("="*60)
print("EARNINGS ANALYZER - Data access functions:")
print("="*60)
with open('../src/analytics/earnings_analyzer.py', 'r', encoding='utf-8') as f:
    content = f.read()
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'def ' in line and ('get_' in line.lower() or 'fetch' in line.lower() or 'upcoming' in line.lower() or 'next' in line.lower()):
            print(f"{i+1}: {line.strip()[:100]}")

# 2. Check macro regime functions
print("\n" + "="*60)
print("MACRO REGIME - Data access:")
print("="*60)
with open('../src/analytics/macro_regime.py', 'r', encoding='utf-8') as f:
    content = f.read()
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'def ' in line and ('detect' in line.lower() or 'get_' in line.lower() or 'vix' in line.lower()):
            print(f"{i+1}: {line.strip()[:100]}")

# 3. Check economic calendar
print("\n" + "="*60)
print("ECONOMIC CALENDAR:")
print("="*60)
with open('../src/analytics/economic_calendar.py', 'r', encoding='utf-8') as f:
    content = f.read()
    lines = content.split('\n')
    for i, line in enumerate(lines[:200]):
        if line.strip().startswith(('def ', 'class ')):
            print(f"{i+1}: {line.strip()[:100]}")

# 4. Check macro event engine
print("\n" + "="*60)
print("MACRO EVENT ENGINE - Event functions:")
print("="*60)
with open('../src/analytics/macro_event_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'def ' in line and ('get_' in line.lower() or 'fetch' in line.lower() or 'event' in line.lower()):
            print(f"{i+1}: {line.strip()[:100]}")
