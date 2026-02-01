# AI Portfolio Manager - Technical Documentation

## Overview

The AI Portfolio Manager is a Streamlit-based trading system that integrates with Interactive Brokers (IBKR) via TWS/Gateway. It enables automated portfolio rebalancing based on target weights from saved JSON portfolios or AI-generated signals.

**Key Capabilities:**
- Connect to IBKR TWS/Gateway
- Load target portfolios from JSON files
- Calculate trade orders to rebalance current holdings to target weights
- Execute market orders via IBKR API
- Post-execution verification comparing orders vs targets
- Risk gates and constraints enforcement

---

## Architecture

```
dashboard/
├── app.py                    # Main Streamlit app entry point
├── portfolio_tab.py          # Portfolio tab orchestrator
└── ai_pm/                    # AI Portfolio Manager module
    ├── __init__.py
    ├── ui_tab.py             # Main UI rendering (LARGEST FILE ~180KB)
    ├── ibkr_gateway.py       # IBKR connection wrapper
    ├── execution_engine.py   # Order execution logic
    ├── execution_verify.py   # Post-execution verification
    ├── trade_planner.py      # Trade plan generation
    ├── risk_engine.py        # Risk gates evaluation
    ├── signal_adapter.py     # Signal loading from platform
    ├── models.py             # Data models (dataclasses)
    └── config.py             # Configuration and constraints
```

---

## Key Files

### 1. `dashboard/app.py`
**Purpose:** Main Streamlit application entry point

**Critical Imports at Top:**
```python
import nest_asyncio
nest_asyncio.apply()

from ib_insync import util
util.startLoop()  # CRITICAL: Runs ib_insync event loop in background thread
```

**Why:** Without `nest_asyncio.apply()` and `util.startLoop()`, IBKR API calls will freeze/deadlock in Streamlit's threaded environment.

---

### 2. `dashboard/ai_pm/ui_tab.py`
**Purpose:** Main UI rendering for AI Portfolio Manager tab (~180KB, largest file)

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `render_ai_portfolio_manager_tab()` | Main entry point for the tab |
| `_to_portfolio_snapshot(gw, account)` | Fetches current portfolio state from IBKR |
| `_fetch_price_map_yahoo_first(symbols, ib)` | Fetches prices via Yahoo Finance (fast) |
| `build_target_weights(...)` | Builds target weights from signals or saved portfolio |

**Session State Keys:**
- `ai_pm_target_weights` - Dict of {symbol: weight} for target portfolio
- `ai_pm_target_portfolio` - Metadata about loaded portfolio
- `ai_pm_price_map` - Cached prices
- `ai_pm_price_cache_time` - Cache timestamp
- `ai_pm_last_verification` - Post-execution verification data
- `ai_pm_connection_ok` - IBKR connection status
- `ai_pm_gateway` - IbkrGateway instance

**Critical Issue - IBKR Calls That Freeze:**
These IBKR calls will freeze in Streamlit threads:
- `ib.qualifyContracts()` - DISABLED in execution_engine.py
- `ib.reqTickers()` - DISABLED in price fetch
- `ib.reqMktData()` - DISABLED in price fetch
- `ib.sleep()` - Use `time.sleep()` instead

---

### 3. `dashboard/ai_pm/ibkr_gateway.py`
**Purpose:** IBKR connection wrapper

**Key Methods:**
```python
class IbkrGateway:
    def connect(self, host='127.0.0.1', port=7496, timeout_sec=10)
    def disconnect()
    def is_connected() -> bool
    def list_accounts() -> List[str]
    def ping() -> bool
```

**Client ID:** Uses environment variable `IBKR_CLIENT_ID` or random ID

---

### 4. `dashboard/ai_pm/execution_engine.py`
**Purpose:** Order execution logic

**Main Function:**
```python
def execute_trade_plan(
    *,
    ib: IB,
    snapshot: PortfolioSnapshot,
    plan: TradePlan,
    account: str,
    constraints: RiskConstraints,
    dry_run: bool = True,
    kill_switch: bool = False,
    auto_trade_enabled: bool = False,
    armed: bool = False,
    price_map: Optional[Dict[str, float]] = None,  # Pre-fetched prices
    skip_live_quotes: bool = True,  # Skip slow IBKR quotes
) -> ExecutionResult
```

**Critical Fixes Applied:**
1. **Skip qualifyContracts** - Causes thread deadlock
2. **Use passed price_map** - Don't rebuild from scratch
3. **Add ib.sleep(3) after orders** - Allow event loop to transmit orders
4. **Order slicing disabled** - `max_slice_nav_pct=1.00`

**Order Placement Flow:**
```python
contract = Stock(sym, 'SMART', 'USD')
# NO qualifyContracts - causes freeze
order = MarketOrder(action, qty)
order.account = account
order.tif = 'DAY'
trade = ib.placeOrder(contract, order)
# After all orders:
ib.sleep(3)  # CRITICAL: Transmit orders to IBKR
```

---

### 5. `dashboard/ai_pm/execution_verify.py`
**Purpose:** Post-execution verification

**Main Function:**
```python
def verify_execution(
    ib, snapshot, plan, targets, price_map
) -> PortfolioVerification
```

**Output Fields:**
- `total_open_orders` - Orders visible in TWS
- `total_accuracy` - Overall portfolio accuracy %
- `projected_invested` - Total value after fills
- `projected_cash` - Cash after fills
- `missing_orders` - Symbols with missing orders
- `extra_orders` - Unexpected orders
- `symbols_verified` - List of OrderVerification per symbol

---

### 6. `dashboard/ai_pm/trade_planner.py`
**Purpose:** Generate trade plan from current vs target

**Main Function:**
```python
def build_trade_plan(
    snapshot: PortfolioSnapshot,
    targets: TargetWeights,
    constraints: RiskConstraints,
    price_map: Dict[str, float],
    capital_to_deploy: Optional[float] = None,
) -> TradePlan
```

---

### 7. `dashboard/ai_pm/risk_engine.py`
**Purpose:** Evaluate risk gates before execution

**Main Function:**
```python
def evaluate_trade_plan_gates(
    snapshot, signals, plan, constraints
) -> GateResult
```

**Gates Evaluated:**
- Turnover limits
- Position weight limits
- Sector concentration
- Cash constraints

---

### 8. `dashboard/ai_pm/config.py`
**Purpose:** Configuration and constraints

**Key Settings:**
```python
@dataclass
class RiskConstraints:
    max_position_weight: float = 0.20    # 20% max per position
    max_sector_weight: float = 1.00      # 100% (disabled)
    cash_min: float = -1.00              # -100% (no cash constraint)
    cash_max: float = 1.00
    max_turnover_per_cycle: float = 1.00
    max_trades_per_cycle: int = 200

DEFAULT_CONSTRAINTS = RiskConstraints()
```

---

### 9. `dashboard/ai_pm/models.py`
**Purpose:** Data models (dataclasses)

**Key Models:**
```python
@dataclass
class Position:
    symbol: str
    quantity: float
    market_price: Optional[float]
    market_value: Optional[float]
    avg_cost: Optional[float]
    ...

@dataclass
class PortfolioSnapshot:
    account: str
    net_liquidation: float
    cash: float
    positions: List[Position]
    open_orders: List[OpenOrder]
    ...

@dataclass
class TargetWeights:
    weights: Dict[str, float]  # symbol -> weight (0.0-1.0)
    strategy_key: str
    ts_utc: datetime
    notes: List[str]

@dataclass
class TradePlan:
    orders: List[PlannedOrder]
    strategy_key: str
    ...

@dataclass
class ExecutionResult:
    submitted: bool
    submitted_orders: List[Dict]
    errors: List[str]
    notes: List[str]
    ...
```

---

## Data Flow

### 1. Portfolio Loading
```
JSON File → st.session_state['ai_pm_target_weights'] → targets: TargetWeights
```

### 2. Snapshot Capture
```
IBKR TWS → gw.ib.accountValues() → NAV, Cash
         → gw.ib.positions() → Current holdings
         → gw.ib.openTrades() → Open orders
         → PortfolioSnapshot
```

### 3. Price Fetching
```
Yahoo Finance (batch) → price_map: Dict[str, float]
(IBKR fallback DISABLED - causes freeze)
```

### 4. Trade Planning
```
PortfolioSnapshot + TargetWeights + price_map → TradePlan
```

### 5. Execution
```
TradePlan → execute_trade_plan() → IBKR placeOrder() → ExecutionResult
```

### 6. Verification
```
IBKR openTrades() + snapshot + targets → PortfolioVerification
```

---

## Known Issues & Fixes

### Issue 1: Streamlit Freezes on IBKR Calls
**Cause:** ib_insync uses asyncio which deadlocks in Streamlit threads

**Fix:**
```python
# At top of app.py
import nest_asyncio
nest_asyncio.apply()

from ib_insync import util
util.startLoop()
```

### Issue 2: qualifyContracts() Freezes
**Cause:** Async call in thread context

**Fix:** Skip qualification - IBKR accepts basic Stock() contracts
```python
contract = Stock(sym, 'SMART', 'USD')
# NO ib.qualifyContracts(contract)
trade = ib.placeOrder(contract, order)
```

### Issue 3: Orders Not Visible in TWS
**Cause:** Event loop not processing after placeOrder()

**Fix:**
```python
trade = ib.placeOrder(contract, order)
# ... after all orders ...
ib.sleep(3)  # Allow event loop to transmit
```

### Issue 4: Price Fetch Timeout (80s)
**Cause:** IBKR reqMktData() one-by-one

**Fix:** Use Yahoo Finance batch fetch, disable IBKR fallback

### Issue 5: Cash Constraint Blocking All Orders
**Cause:** cash_min was -0.20 (20% minimum cash)

**Fix:** Set cash_min = -1.00 (no constraint)

### Issue 6: Duplicate Orders
**Cause:** Order slicing (2% NAV max per order)

**Fix:** Set max_slice_nav_pct = 1.00 (disabled)

---

## Testing

### Standalone Test (Outside Streamlit)
```python
# test_full_flow.py
import nest_asyncio
nest_asyncio.apply()

from ib_insync import util
util.startLoop()

from dashboard.ai_pm.ibkr_gateway import IbkrGateway
from dashboard.ai_pm.execution_engine import execute_trade_plan
# ... full test
```

### Threading Test (Simulates Streamlit)
```python
# test_streamlit_thread.py
import threading

def simulate_confirm():
    # ... execution code ...

thread = threading.Thread(target=simulate_confirm)
thread.start()
thread.join(timeout=60)
```

### Cancel All Orders
```python
gw.ib.reqGlobalCancel()
gw.ib.sleep(3)
```

---

## JSON Portfolio Format

```json
[
    {"symbol": "AAPL", "weight": 5.0},
    {"symbol": "MSFT", "weight": 4.5},
    ...
]
```
- `weight` is percentage (5.0 = 5%)
- Loaded via file uploader or from `json/` directory

---

## UI Controls

| Control | Purpose |
|---------|---------|
| **Reconnect** | Re-establish IBKR connection |
| **Account Dropdown** | Select trading account |
| **Run Now** | Execute full planning cycle |
| **Confirm & Send** | Send orders to IBKR |
| **Cancel All Orders** | Global cancel via reqGlobalCancel() |
| **Clear Verification** | Clear verification panel |

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `IBKR_CLIENT_ID` | Client ID for IBKR connection | Random |

---

## File Locations

| Path | Purpose |
|------|---------|
| `json/` | Saved portfolio JSON files |
| `audit/` | Execution audit logs |
| `dashboard/ai_pm/` | AI PM module |

---

## Important: What NOT to Do

1. **Don't call ib.qualifyContracts()** - Will freeze
2. **Don't call ib.reqMktData() in loops** - Will freeze
3. **Don't call ib.reqTickers()** - Will freeze
4. **Don't use ib.sleep() in UI code** - Use time.sleep()
5. **Don't forget ib.sleep() after placeOrder()** - Orders won't transmit
6. **Don't remove nest_asyncio.apply()** - Everything