with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the execute_trade_plan function signature and add price_map parameter
old_signature = '''def execute_trade_plan(
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
) -> ExecutionResult:'''

new_signature = '''def execute_trade_plan(
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
) -> ExecutionResult:'''

if old_signature in content:
    content = content.replace(old_signature, new_signature)
    print("✅ Updated function signature")
else:
    print("❌ Could not find function signature")

# Now find where it fetches quotes and add the option to skip
old_quote_fetch = '''            quote = _fetch_quote(ib, contract, timeout_sec=2.0)

            # If quote is totally missing, SKIP (your policy: missing prices -> skip)
            if quote.get("bid") is None and quote.get("ask") is None and quote.get("last") is None and quote.get("mid") is None:
                notes.append(f"{sym}: missing live quote (bid/ask/last unavailable); skipped execution.")
                continue'''

new_quote_fetch = '''            # Use pre-fetched price if available (much faster than IBKR reqMktData)
            if skip_live_quotes and price_map and sym in price_map and price_map[sym]:
                pre_price = float(price_map[sym])
                quote = {"bid": pre_price * 0.999, "ask": pre_price * 1.001, "last": pre_price, "mid": pre_price}
            else:
                quote = _fetch_quote(ib, contract, timeout_sec=2.0)

            # If quote is totally missing, SKIP (your policy: missing prices -> skip)
            if quote.get("bid") is None and quote.get("ask") is None and quote.get("last") is None and quote.get("mid") is None:
                notes.append(f"{sym}: missing live quote (bid/ask/last unavailable); skipped execution.")
                continue'''

if old_quote_fetch in content:
    content = content.replace(old_quote_fetch, new_quote_fetch)
    print("✅ Updated quote fetching to use pre-fetched prices")
else:
    print("❌ Could not find quote fetch code")

with open('../dashboard/ai_pm/execution_engine.py', 'w', encoding='utf-8') as f:
    f.write(content)
