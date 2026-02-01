"""
Module Duplication Diagnostic Tool

Run this script to:
1. Find all duplicate Python files in your project
2. Show which module paths are actually being imported
3. Identify potential import conflicts

Usage:
    python check_duplicates.py

Or add to your app.py startup:
    from check_duplicates import audit_imports
    audit_imports()
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import hashlib


def find_duplicate_files(root_dir: str = ".") -> dict:
    """
    Find all Python files with the same name in different directories.

    Returns:
        Dict mapping filename -> list of full paths
    """
    file_locations = defaultdict(list)

    # Critical module names to watch for
    critical_modules = {
        'signal_engine.py',
        'sentiment.py',
        'universe_scorer.py',
        'repository.py',
        'news.py',
        'engine.py',
        'app.py',
        'connection.py',
        'analytics_tab.py',
        'portfolio_tab.py',
        'deep_dive_tab.py',
        'earnings_tab.py',
    }

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip common non-source directories
        dirnames[:] = [d for d in dirnames if d not in {
            '__pycache__', '.git', 'venv', 'env', '.venv',
            'node_modules', '.idea', '.vscode', 'dist', 'build',
            'downloads', 'backup', 'backups', 'old', 'archive'
        }]

        for filename in filenames:
            if filename.endswith('.py'):
                full_path = os.path.join(dirpath, filename)
                file_locations[filename].append(full_path)

    # Filter to only files with duplicates
    duplicates = {k: v for k, v in file_locations.items() if len(v) > 1}

    # Highlight critical duplicates
    critical_duplicates = {k: v for k, v in duplicates.items() if k in critical_modules}

    return duplicates, critical_duplicates


def get_file_hash(filepath: str) -> str:
    """Get MD5 hash of file contents to compare if files are identical."""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    except:
        return "ERROR"


def audit_imports():
    """
    Log which actual files are being imported for critical modules.
    Call this at app startup to verify correct imports.
    """
    print("\n" + "="*70)
    print("🔍 MODULE IMPORT AUDIT")
    print("="*70)

    critical_modules = [
        ('signal_engine', 'src.core.signal_engine', 'src.signal_engine'),
        ('signals_tab', 'src.tabs.signals_tab', 'signals_tab'),
        ('sentiment', 'src.analytics.sentiment', 'sentiment'),
        ('universe_scorer', 'src.analytics.universe_scorer', 'universe_scorer'),
        ('repository', 'src.db.repository', 'repository'),
        ('engine (backtest)', 'src.backtest.engine', 'src.analytics.backtesting.engine'),
    ]

    for name, *import_paths in critical_modules:
        loaded_path = None
        loaded_from = None

        for import_path in import_paths:
            try:
                # Try to import and get the file path
                parts = import_path.split('.')
                module = __import__(import_path)
                for part in parts[1:]:
                    module = getattr(module, part)

                if hasattr(module, '__file__') and module.__file__:
                    loaded_path = module.__file__
                    loaded_from = import_path
                    break
            except (ImportError, AttributeError):
                continue

        if loaded_path:
            print(f"✅ {name:20} → {loaded_path}")
        else:
            print(f"❌ {name:20} → NOT FOUND (tried: {import_paths})")

    print("="*70 + "\n")


def main():
    print("\n" + "="*70)
    print("🔍 DUPLICATE FILE SCANNER")
    print("="*70)

    # Get project root (assume script is in project root or src/)
    root = os.getcwd()
    print(f"Scanning: {root}\n")

    duplicates, critical = find_duplicate_files(root)

    # Report critical duplicates first
    if critical:
        print("🔴 CRITICAL DUPLICATES FOUND:")
        print("-"*70)
        for filename, paths in sorted(critical.items()):
            print(f"\n  {filename}:")
            for path in paths:
                file_hash = get_file_hash(path)
                size = os.path.getsize(path)
                print(f"    [{file_hash}] {size:>6} bytes  {path}")
        print()
    else:
        print("✅ No critical module duplicates found.\n")

    # Report other duplicates
    other_duplicates = {k: v for k, v in duplicates.items() if k not in critical}
    if other_duplicates:
        print(f"🟡 OTHER DUPLICATES ({len(other_duplicates)} files):")
        print("-"*70)
        for filename, paths in sorted(other_duplicates.items()):
            if len(paths) <= 3:
                print(f"  {filename}: {', '.join(paths)}")
            else:
                print(f"  {filename}: {len(paths)} copies")
        print()

    # Show import audit
    audit_imports()

    # Recommendations
    print("="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70)

    if critical:
        print("""
1. IMMEDIATE: Identify which copy is the "canonical" version
   - Usually the one in src/... is canonical
   - Compare file hashes - if different, one has newer fixes

2. RENAME deprecated copies:
   - signal_engine.py → signal_engine_DEPRECATED.py
   - Add at top: raise RuntimeError("Use src.core.signal_engine")

3. UPDATE all imports to use canonical path:
   - from src.core.signal_engine import SignalEngine
   - NOT: from signal_engine import SignalEngine

4. DELETE __pycache__ folders after cleanup:
   - find . -type d -name __pycache__ -exec rm -rf {} +
   - Or on Windows: Get-ChildItem -Recurse -Directory -Name __pycache__ | Remove-Item -Recurse
""")
    else:
        print("\n✅ Your project structure looks clean!")

    return len(critical) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)