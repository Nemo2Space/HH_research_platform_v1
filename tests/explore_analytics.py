# Check src/analytics for the actual data sources

import os

analytics_path = '../src/analytics'
if os.path.exists(analytics_path):
    print("Files in src/analytics:")
    for f in os.listdir(analytics_path):
        if f.endswith('.py'):
            print(f"  - {f}")
    
    # Check earnings_analyzer
    earnings_file = os.path.join(analytics_path, 'earnings_analyzer.py')
    if os.path.exists(earnings_file):
        print(f"\n{'='*60}")
        print("EARNINGS ANALYZER - Key functions:")
        print('='*60)
        with open(earnings_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()[:150]):
                if line.strip().startswith(('def ', 'class ')):
                    print(f"{i+1}: {line.rstrip()[:100]}")
    
    # Check macro_event_engine
    macro_file = os.path.join(analytics_path, 'macro_event_engine.py')
    if os.path.exists(macro_file):
        print(f"\n{'='*60}")
        print("MACRO EVENT ENGINE - Key functions:")
        print('='*60)
        with open(macro_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()[:150]):
                if line.strip().startswith(('def ', 'class ')):
                    print(f"{i+1}: {line.rstrip()[:100]}")
    
    # Check macro_regime
    regime_file = os.path.join(analytics_path, 'macro_regime.py')
    if os.path.exists(regime_file):
        print(f"\n{'='*60}")
        print("MACRO REGIME - Key functions:")
        print('='*60)
        with open(regime_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()[:150]):
                if line.strip().startswith(('def ', 'class ')):
                    print(f"{i+1}: {line.rstrip()[:100]}")
else:
    print("src/analytics not found, checking other locations...")
    for root, dirs, files in os.walk('..'):
        if 'earnings' in root.lower() or 'macro' in root.lower():
            print(f"Found: {root}")
