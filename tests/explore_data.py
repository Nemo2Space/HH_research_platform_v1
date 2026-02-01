# Check the structure of key data sources

files_to_check = [
    ('dashboard/earnings_tab.py', 'earnings'),
    ('dashboard/macro_event_widget.py', 'news/events'),
    ('dashboard/macro_regime_tab.py', 'vix/regime'),
    ('dashboard/ai_pm/signal_adapter.py', 'signals'),
]

for filepath, desc in files_to_check:
    print(f"\n{'='*60}")
    print(f"FILE: {filepath} ({desc})")
    print('='*60)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Show imports and key function definitions
        for i, line in enumerate(lines[:100]):
            if line.strip().startswith(('def ', 'class ', 'import ', 'from ')):
                print(f"{i+1}: {line.rstrip()[:100]}")
    except Exception as e:
        print(f"Error: {e}")
