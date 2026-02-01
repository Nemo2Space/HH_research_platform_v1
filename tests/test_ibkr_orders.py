"""
Place ONE test order to a chosen IBKR account (no cancel).
- Uses clientId=0 by default (best chance to appear in TWS UI).
- Places a SAFE far-away LIMIT so it should not fill.

Run examples:
  python test_ibkr_orders.py --account U24043540
  python test_ibkr_orders.py --account U6454299 --symbol MSFT --limit-price 1.00
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime

from ib_insync import IB, util, Stock, LimitOrder, MarketOrder


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sleep_with_ib(ib: IB, seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        ib.sleep(0.1)


def print_trade(prefix: str, trade) -> None:
    o = trade.order
    s = trade.orderStatus
    c = trade.contract
    print(
        f"{prefix} sym={getattr(c,'symbol','?')} conId={getattr(c,'conId','?')} "
        f"orderId={o.orderId} permId={o.permId} clientId={o.clientId} acct={o.account} "
        f"type={o.orderType} {o.action} qty={o.totalQuantity} tif={o.tif} "
        f"status={s.status} filled={s.filled} remaining={s.remaining} whyHeld='{s.whyHeld}' "
        f"orderRef={getattr(o,'orderRef','')}"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7496)

    # IMPORTANT: clientId=0 binds best into TWS UI for many setups
    p.add_argument("--client-id", type=int, default=0)

    # choose target account here
    p.add_argument("--account", required=True)

    p.add_argument("--symbol", default="INTC")
    p.add_argument("--exchange", default="SMART")
    p.add_argument("--currency", default="USD")

    p.add_argument("--order-type", choices=["LMT", "MKT"], default="LMT")
    p.add_argument("--action", choices=["BUY", "SELL"], default="BUY")
    p.add_argument("--qty", type=float, default=1.0)

    # far-away price so it should not fill
    p.add_argument("--limit-price", type=float, default=1.00)

    p.add_argument("--tif", default="DAY")
    p.add_argument("--outside-rth", action="store_true", default=True)

    args = p.parse_args()

    util.startLoop()
    ib = IB()

    def on_error(req_id, code, msg, contract):
        extra = ""
        if contract is not None:
            extra = f" contract={getattr(contract,'symbol','')}:{getattr(contract,'conId','')}"
        print(f"[{ts()}] ERROR reqId={req_id} code={code} msg='{msg}'{extra}")

    def on_open_order(trade):
        print_trade(f"[{ts()}] OPEN_ORDER_EVENT:", trade)

    def on_order_status(trade):
        print_trade(f"[{ts()}] ORDER_STATUS_EVENT:", trade)

    ib.errorEvent += on_error
    ib.openOrderEvent += on_open_order
    ib.orderStatusEvent += on_order_status

    print("=" * 80)
    print("IBKR Place One Order (NO CANCEL)")
    print(f"Time: {datetime.now()}")
    print("=" * 80)

    ib.connect(args.host, args.port, clientId=args.client_id, timeout=10)
    print(f"✅ Connected host={args.host} port={args.port} clientId={args.client_id}")
    print(f"   ServerVersion={ib.client.serverVersion()}")
    accts = ib.managedAccounts()
    print(f"   ManagedAccounts={accts}")

    if accts and args.account not in accts:
        print(f"❌ Account '{args.account}' not in ManagedAccounts.")
        ib.disconnect()
        return 2

    # (Optional) request all open orders so you can see existing ones in console
    try:
        ib.reqAllOpenOrders()
        sleep_with_ib(ib, 1.5)
    except Exception:
        pass

    contract = Stock(args.symbol, args.exchange, args.currency)
    q = ib.qualifyContracts(contract)
    if not q:
        print(f"❌ Could not qualify {args.symbol}")
        ib.disconnect()
        return 3
    qc = q[0]
    print(f"✅ Qualified sym={qc.symbol} conId={qc.conId} primaryExch={qc.primaryExchange}")

    if args.order_type == "MKT":
        order = MarketOrder(args.action, args.qty, tif=args.tif)
    else:
        order = LimitOrder(args.action, args.qty, args.limit_price, tif=args.tif)

    order.account = args.account
    order.outsideRth = bool(args.outside_rth)
    order.transmit = True
    order.orderRef = f"API_ONE_{args.account}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    trade = ib.placeOrder(qc, order)
    sleep_with_ib(ib, 2.0)

    print(f"\n[{ts()}] ✅ ORDER PLACED (NOT CANCELLED)")
    print_trade("  PLACED:", trade)
    print(f"\n>>> In TWS, check Activity / Orders for account {args.account} and orderRef: {order.orderRef} <<<\n")

    ib.disconnect()
    print(f"[{ts()}] ✅ Disconnected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
