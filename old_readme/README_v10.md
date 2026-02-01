# Bond Signals Dashboard

**Unified Bond Trading Analysis Dashboard for Treasury ETFs**

A comprehensive Streamlit dashboard providing institutional-grade bond analysis with AI-powered insights, technical analysis, news sentiment, and real-time market data.

---

## Overview

This dashboard integrates multiple data sources and analysis components to generate actionable trading signals for Treasury bond ETFs (TLT, ZROZ, EDV, etc.). It combines fundamental analysis, technical indicators, institutional flow data, and news sentiment into a unified scoring system.

## Features

### 📊 Core Analysis
- **Bond Signal Generation**: Composite scoring system combining technical, fundamental, flow, macro, and sentiment factors
- **Treasury Yield Analysis**: Real-time 10Y, 30Y yields with historical data and trend analysis
- **Technical Indicators**: RSI, MACD, Bollinger Bands, SMA 50/200, VWAP, support/resistance levels

### 🏛️ Institutional Intelligence
- **Treasury Futures Analysis**: COT positioning, open interest trends
- **Auction Data**: Recent Treasury auction results and bid-to-cover ratios
- **MOVE Index**: Bond market volatility tracking
- **Curve Trades**: Steepener/Flattener analysis

### 📰 News & Sentiment
- **News Aggregation**: Fetches relevant bond market news from multiple sources
- **AI Sentiment Analysis**: Categorizes articles as Bullish/Bearish/Neutral
- **Theme Extraction**: Identifies key market themes across articles

### 🤖 AI Integration
- **Comprehensive Context Building**: All data fed to AI for analysis
- **Interactive Chat**: Ask questions about the current market setup
- **AI Advisor**: Get actionable recommendations based on complete market picture

### 💾 Caching System
- **JSON-based Persistence**: Analysis cached to disk for instant reload
- **Browser Refresh Support**: "Load Last Analysis" restores previous session
- **Efficient Serialization**: Complex objects properly serialized/deserialized

---

## Installation

### Requirements
```
streamlit
pandas
plotly
numpy
```

### File Location
```
dashboard/bond_signals_dashboard.py
```

### Data Directory
```
data/cache/bond_analysis_cache.json
```

---

## Usage

### Running the Dashboard
```bash
streamlit run dashboard/bond_signals_dashboard.py
```

### Main Workflow
1. Click **"Run Fresh Analysis"** to fetch live data and generate signals
2. Review the unified summary, yields, signals, and news sections
3. Use **"AI Chat"** to ask questions about the market
4. **"Load Last Analysis"** restores cached data after browser refresh

---

## Architecture

### File Structure (1,823 lines)

```
bond_dashboard_complete.py
├── Cache System (lines 28-660)
│   ├── _serialize_signal()      - Convert BondSignal to dict
│   ├── _deserialize_signal()    - Reconstruct BondSignal from dict
│   ├── save_bond_cache()        - Save analysis to JSON
│   ├── load_bond_cache()        - Load analysis from JSON
│   └── get_cache_info()         - Get cache age/timestamp
│
├── Main Dashboard (lines 736-1077)
│   └── render_bond_trading_tab() - Main entry point
│
├── Display Components (lines 1078-1557)
│   ├── _render_quick_summary()      - Compact summary view
│   ├── _render_unified_summary()    - Detailed summary panel
│   ├── _render_news_expander()      - Expandable news section
│   ├── _render_news_section()       - Full news display
│   ├── _render_yields_section()     - Yield charts and data
│   ├── _render_signals_section()    - Signal table
│   └── _render_detailed_analysis()  - Deep dive analysis
│
├── AI Integration (lines 1558-1789)
│   ├── _render_ai_chat()            - Chat interface
│   └── _build_complete_ai_context() - Context builder
│
└── Education (lines 1790-1823)
    └── _render_education()          - Learning resources
```

### Data Flow
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Data Sources   │────▶│   Generators     │────▶│   Dashboard     │
│                 │     │                  │     │                 │
│ • Yahoo Finance │     │ • BondSignalGen  │     │ • Unified View  │
│ • News APIs     │     │ • YieldGenerator │     │ • Signal Table  │
│ • Treasury Data │     │ • NewsAnalyzer   │     │ • AI Chat       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │   JSON Cache     │
                        │                  │
                        │ • Signals        │
                        │ • Yields         │
                        │ • News + Articles│
                        │ • Intelligence   │
                        └──────────────────┘
```

---

## Cache System Details

### Serialization
The cache system handles complex nested objects:

| Component | Saved Fields |
|-----------|--------------|
| **Signals** | ticker, price, composite_score, technical metrics, bull/bear cases |
| **News** | overall_score, overall_sentiment, article counts, themes, articles[] |
| **Articles** | title, source, sentiment, score, url, category, description |
| **Intelligence** | fed_policy, rate_probabilities, futures, auctions |

### Key Implementation Notes
- News sentiment is saved/loaded explicitly (not derived from score)
- Article attributes include: `title`, `source`, `sentiment`, `score`, `url`, `category`, `description`, `published_at`
- Fallback logic derives sentiment from bullish/bearish counts if not saved

---

## Component Scores

The composite signal uses weighted scoring:

| Component | Weight | Description |
|-----------|--------|-------------|
| Technical | 30% | RSI, MACD, Trend, Support/Resistance |
| Fundamental | 25% | Yield analysis, Duration, Target prices |
| Flow | 20% | Institutional positioning, Auction demand |
| Macro | 15% | Fed policy, Economic outlook |
| Sentiment | 10% | News sentiment, Market mood |

---

## Signal Interpretation

| Signal | Composite Score | Action |
|--------|----------------|--------|
| 🟢 STRONG BUY | > 75 | High conviction long |
| 🟢 BUY | 60-75 | Moderate long |
| 🟡 HOLD | 40-60 | No action |
| 🔴 SELL | 25-40 | Moderate short/reduce |
| 🔴 STRONG SELL | < 25 | High conviction short |

---

## Troubleshooting

### Cache Issues
```powershell
# Clear cache to force fresh analysis
Remove-Item "data\cache\bond_analysis_cache.json" -Force

# Clear Python cache
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
```

### Common Problems

| Issue | Solution |
|-------|----------|
| News sentiment shows wrong value | Delete cache, run fresh analysis |
| "Load Last Analysis" fails | Check cache file exists and is valid JSON |
| Articles not displaying | Ensure `published_at` attribute exists (may be None for cached) |

---

## Recent Updates (Dec 2025)

### Cache System Overhaul
- ✅ Converted from Pickle to JSON for better compatibility
- ✅ Fixed news sentiment save/load (stores `overall_sentiment` explicitly)
- ✅ Added all article attributes to cache (url, category, description)
- ✅ Fixed `published` → `published_at` attribute naming
- ✅ Added debug logging for save/load operations

### Bug Fixes
- Fixed "BEARISH (0%)" showing when 14/14 articles were bullish
- Fixed article display errors due to missing attributes
- Improved error handling in serialization

---

## Dependencies

```python
# Core
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
import json
from pathlib import Path
from types import SimpleNamespace

# Project
from src.utils.logging import get_logger
from src.bond_signals.generator import BondSignalGenerator
# ... other project imports
```

---

## Author

**Alpha Research Platform**  
HH Research - Hedge Fund Grade Trading Infrastructure

---

## Version

**v2.1** - December 2025  
Complete cache persistence with JSON serialization