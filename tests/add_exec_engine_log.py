with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add logging at the start of execute_trade_plan
old_start = '''def execute_trade_plan(
        *,
        ib: IB,
        snapshot: PortfolioSnapshot,
        plan: TradePlan,
        account: str,
        constraints: RiskConstraints = DEFAULT_CONSTRAINTS,
        dry_run: bool = True,
        kill_switch: bool = False,
        auto_trade_enabled: bool = False,
        armed: bool = False,
        limit_buffer_bps: float = 10.0,
        max_slice_nav_pct: float = 0.02,  # slice orders above 2% NAV
        price_map: Optional[Dict[str, float]] = None,  # Pre-fetched prices (Yahoo) to avoid slow IBKR quotes
        skip_live_quotes: bool = True,  # Skip slow IBKR reqMktData calls
) -> ExecutionResult:
    """
    Execute orders for a plan.

    Policy implemented:
      - Missing live quote => skip that symbol (do not place blind orders)
      - Cash below cash_min:
          - manual mode: return without executing (caller should show recommendation)
          - auto+armed: execute SELL-FIRST (SELL orders only), defer buys
      - Turnover cap is enforced upstream by gates; this function does not override that.
    """
    ts = datetime.utcnow()'''

new_start = '''def execute_trade_plan(
        *,
        ib: IB,
        snapshot: PortfolioSnapshot,
        plan: TradePlan,
        account: str,
        constraints: RiskConstraints = DEFAULT_CONSTRAINTS,
        dry_run: bool = True,
        kill_switch: bool = False,
        auto_trade_enabled: bool = False,
        armed: bool = False,
        limit_buffer_bps: float = 10.0,
        max_slice_nav_pct: float = 0.02,  # slice orders above 2% NAV
        price_map: Optional[Dict[str, float]] = None,  # Pre-fetched prices (Yahoo) to avoid slow IBKR quotes
        skip_live_quotes: bool = True,  # Skip slow IBKR reqMktData calls
) -> ExecutionResult:
    """
    Execute orders for a plan.

    Policy implemented:
      - Missing live quote => skip that symbol (do not place blind orders)
      - Cash below cash_min:
          - manual mode: return without executing (caller should show recommendation)
          - auto+armed: execute SELL-FIRST (SELL orders only), defer buys
      - Turnover cap is enforced upstream by gates; this function does not override that.
    """
    import logging
    import sys
    _log = logging.getLogger(__name__)
    _log.info(f"execute_trade_plan: START dry_run={dry_run}, orders={len(plan.orders) if plan and plan.orders else 0}")
    print(f"DEBUG execute_trade_plan: START dry_run={dry_run}", flush=True)
    sys.stdout.flush()
    
    ts = datetime.utcnow()'''

if old_start in content:
    content = content.replace(old_start, new_start)
    print("✅ Added logging at start of execute_trade_plan")
else:
    print("❌ Could not find start of execute_trade_plan")

# Add logging before placeOrder
old_place = '''tr: Trade = ib.placeOrder(contract, ib_order)'''

new_place = '''_log.info(f"execute_trade_plan: placing order {sym} {o.action} {qty}")
                tr: Trade = ib.placeOrder(contract, ib_order)
                _log.info(f"execute_trade_plan: order placed for {sym}")'''

if old_place in content:
    content = content.replace(old_place, new_place)
    print("✅ Added logging around placeOrder")
else:
    print("❌ Could not find placeOrder")

with open('../dashboard/ai_pm/execution_engine.py', 'w', encoding='utf-8') as f:
    f.write(content)
