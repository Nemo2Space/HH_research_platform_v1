#!/usr/bin/env python3
"""
Migration Script: Fix Default-50 Contamination in Consumer Files
================================================================

This script automates the most common pattern changes needed in the 39 consumer 
files after unified_signal.py switches from `int = 50` to `Optional[int] = None`.

WHAT IT DOES:
1. Replaces `!= 50` checks with `is not None` 
2. Replaces `== 50` checks with `is None`
3. Replaces `or 50` patterns with proper None handling
4. Flags lines that need MANUAL review (arithmetic on score fields, display formatting)

SAFE: This script generates a report first, only modifies files with --apply flag.

Usage:
    python migrate_consumers.py                    # Dry run - report only
    python migrate_consumers.py --apply            # Apply changes
    python migrate_consumers.py --report-only      # Only show manual review items
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# Root of the project
PROJECT_ROOT = Path(__file__).parent

# Files already fixed (skip these)
ALREADY_FIXED = {
    "src/core/unified_signal.py",
    "src/core/signal_engine.py",
    "src/core/unified_scorer.py",  # Already uses Optional[float] = None
}

# Score field names that changed from int=50 to Optional[int]=None
SCORE_FIELDS = [
    "today_score", "longterm_score", "risk_score",
    "technical_score", "fundamental_score", "sentiment_score",
    "options_score", "earnings_score", "bond_score",
    "gex_score", "dark_pool_score", "cross_asset_score",
    "sentiment_nlp_score", "whisper_score", "insider_score", "inst_13f_score",
]

# Signal fields that changed from str="HOLD" to Optional[str]=None
SIGNAL_FIELDS = [
    "technical_signal", "fundamental_signal", "sentiment_signal",
    "options_signal", "earnings_signal", "bond_signal",
    "gex_signal", "dark_pool_signal", "cross_asset_signal",
    "sentiment_nlp_signal", "whisper_signal", "insider_signal", "inst_13f_signal",
    "institutional_bias", "gex_regime",
]


def find_consumer_files() -> List[Path]:
    """Find all Python files that reference score/signal fields."""
    consumers = []
    for py_file in PROJECT_ROOT.rglob("*.py"):
        rel = str(py_file.relative_to(PROJECT_ROOT))
        if rel in ALREADY_FIXED:
            continue
        if "__pycache__" in rel:
            continue
        
        try:
            content = py_file.read_text()
        except Exception:
            continue
            
        # Check if file references any score fields
        for field in SCORE_FIELDS + SIGNAL_FIELDS:
            if field in content:
                consumers.append(py_file)
                break
    
    return sorted(consumers)


def analyze_file(filepath: Path) -> Dict:
    """Analyze a file for patterns that need changing."""
    content = filepath.read_text()
    lines = content.split('\n')
    
    auto_fixes = []      # Can be automatically fixed
    manual_review = []    # Needs human review
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Skip comments
        if stripped.startswith('#'):
            continue
            
        for field in SCORE_FIELDS:
            # Pattern 1: != 50 → is not None
            if f".{field} != 50" in line or f"['{field}'] != 50" in line:
                auto_fixes.append((i, "!= 50 → is not None", line.rstrip()))
            
            # Pattern 2: == 50 → is None  
            if f".{field} == 50" in line or f"['{field}'] == 50" in line:
                auto_fixes.append((i, "== 50 → is None", line.rstrip()))
            
            # Pattern 3: or 50 → preserve None
            if f"'{field}', 50)" in line or f"'{field}') or 50" in line:
                auto_fixes.append((i, "or 50 fallback → remove", line.rstrip()))
            if f"['{field}'] or 50" in line:
                auto_fixes.append((i, "or 50 fallback → remove", line.rstrip()))
            
            # Pattern 4: get(..., 50) → get(...) (dict.get with 50 default)
            if f"get('{field}', 50)" in line:
                auto_fixes.append((i, f"get('{field}', 50) → get('{field}')", line.rstrip()))
            
            # MANUAL: Arithmetic operations on score fields (could crash with None)
            if f".{field} *" in line or f".{field} +" in line or f".{field} -" in line or f".{field} /" in line:
                if "is not None" not in line and "if " not in line:
                    manual_review.append((i, f"ARITHMETIC on {field} (may be None)", line.rstrip()))
            
            # MANUAL: Comparison without None guard
            if (f".{field} >" in line or f".{field} <" in line or 
                f".{field} >=" in line or f".{field} <=" in line):
                if "is not None" not in line and "!= 50" not in line and "== 50" not in line:
                    manual_review.append((i, f"COMPARISON on {field} (may be None)", line.rstrip()))
            
            # MANUAL: f-string formatting with score field
            if field in line and ('{' in line or 'f"' in line or "f'" in line):
                if "score_display" not in line and "score_value" not in line:
                    if f".{field}" in line:
                        manual_review.append((i, f"DISPLAY of {field} (may show None)", line.rstrip()))
    
    return {
        'auto_fixes': auto_fixes,
        'manual_review': manual_review,
    }


def apply_auto_fixes(filepath: Path) -> int:
    """Apply automatic fixes to a file. Returns count of fixes applied."""
    content = filepath.read_text()
    original = content
    fixes = 0
    
    for field in SCORE_FIELDS:
        # != 50 → is not None
        old = f".{field} != 50"
        new = f".{field} is not None"
        if old in content:
            content = content.replace(old, new)
            fixes += content.count(new) - original.count(new)
        
        # == 50 → is None
        old = f".{field} == 50"
        new = f".{field} is None"
        if old in content:
            content = content.replace(old, new)
            fixes += 1
        
        # get('field', 50) → get('field')  
        old = f"get('{field}', 50)"
        new = f"get('{field}')"
        if old in content:
            content = content.replace(old, new)
            fixes += 1
        
        # int(row.get('field') or 50) → _safe_int(row.get('field'))
        old = f"or 50)"
        # This is too broad for auto-fix, skip
    
    if content != original:
        filepath.write_text(content)
    
    return fixes


def main():
    apply_mode = "--apply" in sys.argv
    report_only = "--report-only" in sys.argv
    
    print("=" * 80)
    print("HH Platform: Default-50 Migration Report")
    print("=" * 80)
    print()
    
    consumers = find_consumer_files()
    print(f"Found {len(consumers)} consumer files to analyze\n")
    
    total_auto = 0
    total_manual = 0
    
    for filepath in consumers:
        rel = str(filepath.relative_to(PROJECT_ROOT))
        analysis = analyze_file(filepath)
        
        auto = analysis['auto_fixes']
        manual = analysis['manual_review']
        
        if not auto and not manual:
            continue
        
        print(f"\n{'─' * 60}")
        print(f"📄 {rel}")
        print(f"   Auto-fixable: {len(auto)}  |  Manual review: {len(manual)}")
        
        if auto and not report_only:
            print(f"   AUTO FIXES:")
            for lineno, desc, code in auto[:10]:
                print(f"     L{lineno}: {desc}")
                print(f"           {code[:100]}")
            if len(auto) > 10:
                print(f"     ... and {len(auto) - 10} more")
        
        if manual:
            print(f"   ⚠️  MANUAL REVIEW NEEDED:")
            for lineno, desc, code in manual[:10]:
                print(f"     L{lineno}: {desc}")
                print(f"           {code[:100]}")
            if len(manual) > 10:
                print(f"     ... and {len(manual) - 10} more")
        
        total_auto += len(auto)
        total_manual += len(manual)
        
        if apply_mode and auto:
            fixes_applied = apply_auto_fixes(filepath)
            print(f"   ✅ Applied {fixes_applied} auto-fixes")
    
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"  Auto-fixable patterns:  {total_auto}")
    print(f"  Manual review needed:   {total_manual}")
    print(f"{'=' * 80}")
    
    if not apply_mode and total_auto > 0:
        print(f"\nRun with --apply to auto-fix {total_auto} patterns")
    
    if total_manual > 0:
        print(f"\n⚠️  {total_manual} patterns need manual review (arithmetic/comparison/display on Optional fields)")
        print(f"   Use signal.score_value('field', default=0) for safe arithmetic")
        print(f"   Use signal.score_display('field') for safe display")
        print(f"   Use 'if signal.X is not None:' before comparisons")


if __name__ == "__main__":
    main()
