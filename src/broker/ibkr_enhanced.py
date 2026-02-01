"""
Enhanced IBKR Data Fetcher
==========================

Additional account data including:
- Dividends received per stock
- Realized P&L
- Margin requirements
- Account activity

Add to: src/broker/ibkr_enhanced.py

Author: HH Research Platform
"""

import math
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class DividendInfo:
    """Dividend information for a position."""
    symbol: str
    ex_date: Optional[str] = None
    pay_date: Optional[str] = None
    amount: float = 0.0
    currency: str = "USD"


@dataclass
class EnhancedAccountData:
    """Extended account information beyond basic summary."""
    account_id: str

    # Core metrics (from basic summary)
    net_liquidation: float = 0.0
    total_cash: float = 0.0
    buying_power: float = 0.0
    gross_position_value: float = 0.0

    # Additional metrics
    available_funds: float = 0.0
    excess_liquidity: float = 0.0
    full_maint_margin_req: float = 0.0
    full_init_margin_req: float = 0.0
    cushion: float = 0.0  # Excess liquidity as % of net liq

    # P&L metrics
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    # Interest & Dividends (if available)
    accrued_dividend: float = 0.0

    # Day trading
    day_trades_remaining: int = -1  # -1 means unlimited (not PDT restricted)
    sma: float = 0.0  # Special Memorandum Account

    # Leverage
    leverage: float = 0.0  # Gross position / net liq

    # Timestamp
    as_of: str = ""


def get_enhanced_account_summary(ib, account_id: str) -> Optional[EnhancedAccountData]:
    """
    Get enhanced account summary with all available metrics.

    Args:
        ib: Connected IB instance from ib_insync
        account_id: Account ID to query

    Returns:
        EnhancedAccountData or None if error
    """
    try:
        # Request account summary with all tags
        summary_items = ib.accountSummary(account_id)

        # Parse into dict
        summary_dict = {}
        for item in summary_items:
            if item.currency == "USD" or item.tag in ['DayTradesRemaining', 'Cushion']:
                try:
                    summary_dict[item.tag] = float(item.value)
                except ValueError:
                    summary_dict[item.tag] = item.value

        net_liq = summary_dict.get("NetLiquidation", 0)
        gross_pos = summary_dict.get("GrossPositionValue", 0)

        return EnhancedAccountData(
            account_id=account_id,
            net_liquidation=net_liq,
            total_cash=summary_dict.get("TotalCashValue", 0),
            buying_power=summary_dict.get("BuyingPower", 0),
            gross_position_value=gross_pos,
            available_funds=summary_dict.get("AvailableFunds", 0),
            excess_liquidity=summary_dict.get("ExcessLiquidity", 0),
            full_maint_margin_req=summary_dict.get("FullMaintMarginReq", 0),
            full_init_margin_req=summary_dict.get("FullInitMarginReq", 0),
            cushion=summary_dict.get("Cushion", 0) * 100 if summary_dict.get("Cushion") else 0,
            realized_pnl=summary_dict.get("RealizedPnL", 0),
            unrealized_pnl=summary_dict.get("UnrealizedPnL", 0),
            accrued_dividend=summary_dict.get("AccruedDividend", 0),
            day_trades_remaining=int(summary_dict.get("DayTradesRemaining", -1)),
            sma=summary_dict.get("SMA", 0),
            leverage=round(gross_pos / net_liq, 2) if net_liq > 0 else 0,
            as_of=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    except Exception as e:
        print(f"Error getting enhanced account summary: {e}")
        return None


def get_position_dividends(ib, positions: List[Dict]) -> Dict[str, DividendInfo]:
    """
    Get dividend information for positions.
    Note: IBKR API has limited dividend data - this fetches what's available.

    Args:
        ib: Connected IB instance
        positions: List of position dicts with 'symbol' key

    Returns:
        Dict of symbol -> DividendInfo
    """
    from ib_insync import Stock

    dividends = {}

    for pos in positions:
        symbol = pos.get('symbol', '')
        if not symbol or symbol in ['USD', 'CASH']:
            continue

        try:
            contract = Stock(symbol, "SMART", "USD")
            qualified = ib.qualifyContracts(contract)

            if qualified:
                # Get contract details which may include dividend info
                details = ib.reqContractDetails(qualified[0])

                if details:
                    detail = details[0]
                    dividends[symbol] = DividendInfo(
                        symbol=symbol,
                        # Note: Contract details don't always have dividend info
                        # This is a placeholder - real dividend data often comes from
                        # account statements or third-party APIs
                    )
        except Exception as e:
            print(f"Error getting dividend for {symbol}: {e}")

    return dividends


def get_executions_and_dividends(ib, account_id: str, days_back: int = 30) -> Tuple[List[Dict], List[Dict]]:
    """
    Get recent executions and dividend payments from account activity.

    Args:
        ib: Connected IB instance
        account_id: Account ID
        days_back: Number of days to look back

    Returns:
        Tuple of (executions, dividends)
    """
    from ib_insync import ExecutionFilter
    from datetime import datetime, timedelta

    executions = []
    dividends = []

    try:
        # Get executions from the last N days
        exec_filter = ExecutionFilter(
            acctCode=account_id,
            time=(datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d-00:00:00")
        )

        fills = ib.reqExecutions(exec_filter)
        ib.sleep(1)

        for fill in fills:
            exec_data = {
                'symbol': fill.contract.symbol,
                'side': fill.execution.side,
                'shares': fill.execution.shares,
                'price': fill.execution.price,
                'time': str(fill.execution.time),
                'commission': fill.commissionReport.commission if fill.commissionReport else 0,
                'realized_pnl': fill.commissionReport.realizedPNL if fill.commissionReport else 0,
            }
            executions.append(exec_data)

    except Exception as e:
        print(f"Error getting executions: {e}")

    return executions, dividends


def fetch_dividends_from_yahoo(symbols: List[str], days_back: int = 365) -> Dict[str, Dict]:
    """
    Fetch dividend history from Yahoo Finance as fallback.
    More reliable than IBKR API for historical dividends.

    Args:
        symbols: List of stock symbols
        days_back: How far back to look

    Returns:
        Dict of symbol -> dividend info
    """
    import yfinance as yf
    from datetime import datetime, timedelta

    result = {}
    start_date = datetime.now() - timedelta(days=days_back)

    for symbol in symbols:
        if not symbol or symbol in ['USD', 'CASH']:
            continue

        try:
            ticker = yf.Ticker(symbol)

            # Get dividend history
            divs = ticker.dividends

            if divs is not None and len(divs) > 0:
                # Filter to recent dividends
                recent_divs = divs[divs.index >= start_date.strftime('%Y-%m-%d')]

                if len(recent_divs) > 0:
                    # Sum up dividends received
                    total_dividends = recent_divs.sum()
                    last_div_date = recent_divs.index[-1].strftime('%Y-%m-%d')
                    last_div_amount = recent_divs.iloc[-1]

                    # Get dividend yield
                    info = ticker.fast_info
                    div_yield = getattr(info, 'dividend_yield', None)
                    if div_yield:
                        div_yield = div_yield * 100  # Convert to percentage

                    result[symbol] = {
                        'total_dividends_1y': round(total_dividends, 4),
                        'last_dividend_date': last_div_date,
                        'last_dividend_amount': round(last_div_amount, 4),
                        'dividend_yield_pct': round(div_yield, 2) if div_yield else None,
                        'dividend_count': len(recent_divs),
                        'annual_dividend': round(last_div_amount * 4, 2) if len(recent_divs) >= 1 else None,  # Estimate
                    }
                else:
                    result[symbol] = {
                        'total_dividends_1y': 0,
                        'dividend_yield_pct': None,
                        'note': 'No recent dividends'
                    }
            else:
                result[symbol] = {
                    'total_dividends_1y': 0,
                    'dividend_yield_pct': None,
                    'note': 'No dividend data'
                }

        except Exception as e:
            result[symbol] = {
                'error': str(e)
            }

    return result


def calculate_portfolio_dividend_income(
    positions: List[Dict],
    dividend_data: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Calculate estimated annual dividend income for portfolio.

    Args:
        positions: List of position dicts with 'symbol' and 'quantity'
        dividend_data: Dict from fetch_dividends_from_yahoo

    Returns:
        Summary dict with income estimates
    """
    total_annual_income = 0.0
    position_income = []

    for pos in positions:
        symbol = pos.get('symbol', '')
        qty = pos.get('quantity', 0)

        if symbol in dividend_data and qty > 0:
            div_info = dividend_data[symbol]
            annual_div = div_info.get('annual_dividend') or div_info.get('total_dividends_1y', 0)

            if annual_div and annual_div > 0:
                position_annual = annual_div * qty
                total_annual_income += position_annual

                position_income.append({
                    'symbol': symbol,
                    'shares': qty,
                    'div_per_share': annual_div,
                    'annual_income': round(position_annual, 2),
                    'yield_pct': div_info.get('dividend_yield_pct'),
                    'last_div_date': div_info.get('last_dividend_date'),
                })

    # Sort by income
    position_income.sort(key=lambda x: x['annual_income'], reverse=True)

    return {
        'total_annual_income': round(total_annual_income, 2),
        'monthly_income': round(total_annual_income / 12, 2),
        'quarterly_income': round(total_annual_income / 4, 2),
        'position_breakdown': position_income,
        'dividend_paying_positions': len(position_income),
    }


# =============================================================================
# CONVENIENCE FUNCTION FOR PORTFOLIO TAB
# =============================================================================

def load_enhanced_ibkr_data(
    account_id: str,
    host: str = "127.0.0.1",
    port: int = 7496,
    fetch_live_prices: bool = False,
    fetch_dividends: bool = True
) -> Dict[str, Any]:
    """
    Load all IBKR data including enhanced metrics and dividends.

    This is an enhanced version of load_ibkr_data_cached that includes
    additional account information.
    """
    import random
    import nest_asyncio
    nest_asyncio.apply()

    result = {
        "accounts": [],
        "summary": None,
        "enhanced_summary": None,
        "positions": [],
        "open_orders": [],
        "dividend_data": {},
        "dividend_summary": {},
        "error": None,
        "timestamp": datetime.now().isoformat(),
        "prices_live": fetch_live_prices
    }

    try:
        from ib_insync import IB

        ib = IB()
        client_id = random.randint(10000, 99999)
        ib.connect(host=host, port=port, clientId=client_id, timeout=10)

        if not ib.isConnected():
            result["error"] = "Failed to connect to IBKR"
            return result

        # Get accounts
        accounts = ib.managedAccounts()
        result["accounts"] = accounts

        if not accounts:
            result["error"] = "No accounts found"
            ib.disconnect()
            return result

        target_account = account_id if account_id in accounts else accounts[0]

        # Get enhanced summary
        enhanced = get_enhanced_account_summary(ib, target_account)
        if enhanced:
            result["enhanced_summary"] = {
                "account_id": enhanced.account_id,
                "net_liquidation": enhanced.net_liquidation,
                "total_cash": enhanced.total_cash,
                "buying_power": enhanced.buying_power,
                "gross_position_value": enhanced.gross_position_value,
                "available_funds": enhanced.available_funds,
                "excess_liquidity": enhanced.excess_liquidity,
                "full_maint_margin_req": enhanced.full_maint_margin_req,
                "full_init_margin_req": enhanced.full_init_margin_req,
                "cushion": enhanced.cushion,
                "realized_pnl": enhanced.realized_pnl,
                "unrealized_pnl": enhanced.unrealized_pnl,
                "accrued_dividend": enhanced.accrued_dividend,
                "day_trades_remaining": enhanced.day_trades_remaining,
                "sma": enhanced.sma,
                "leverage": enhanced.leverage,
                "as_of": enhanced.as_of,
            }
            # Also set basic summary for backward compatibility
            result["summary"] = {
                "account_id": enhanced.account_id,
                "net_liquidation": enhanced.net_liquidation,
                "total_cash": enhanced.total_cash,
                "buying_power": enhanced.buying_power,
                "gross_position_value": enhanced.gross_position_value,
            }

        # Get positions
        positions = ib.positions()
        position_list = []
        symbols = []

        for pos in positions:
            if pos.account == target_account:
                symbol = pos.contract.symbol
                symbols.append(symbol)
                avg_cost = pos.avgCost
                qty = pos.position

                position_list.append({
                    "symbol": symbol,
                    "quantity": qty,
                    "avg_cost": avg_cost,
                    "current_price": avg_cost,  # Will update if fetching live
                    "market_value": qty * avg_cost,
                    "unrealized_pnl": 0,
                    "unrealized_pnl_pct": 0,
                    "account": pos.account,
                    "sec_type": pos.contract.secType,
                    "currency": pos.contract.currency,
                })

        result["positions"] = position_list

        # Fetch dividends from Yahoo (more reliable)
        if fetch_dividends and symbols:
            try:
                div_data = fetch_dividends_from_yahoo(symbols, days_back=365)
                result["dividend_data"] = div_data

                # Calculate portfolio income
                div_summary = calculate_portfolio_dividend_income(position_list, div_data)
                result["dividend_summary"] = div_summary
            except Exception as e:
                print(f"Error fetching dividends: {e}")

        ib.disconnect()
        return result

    except Exception as e:
        result["error"] = str(e)
        return result


if __name__ == "__main__":
    # Test
    print("Testing Enhanced IBKR Data Fetcher...")

    data = load_enhanced_ibkr_data("", fetch_dividends=True)

    if data.get("error"):
        print(f"Error: {data['error']}")
    else:
        print(f"Accounts: {data['accounts']}")

        if data.get("enhanced_summary"):
            es = data["enhanced_summary"]
            print(f"\nEnhanced Summary:")
            print(f"  NAV: ${es['net_liquidation']:,.2f}")
            print(f"  Leverage: {es['leverage']:.2f}x")
            print(f"  Cushion: {es['cushion']:.1f}%")
            print(f"  Realized P&L: ${es['realized_pnl']:,.2f}")

        if data.get("dividend_summary"):
            ds = data["dividend_summary"]
            print(f"\nDividend Summary:")
            print(f"  Annual Income: ${ds['total_annual_income']:,.2f}")
            print(f"  Monthly Income: ${ds['monthly_income']:,.2f}")
            print(f"  Dividend Positions: {ds['dividend_paying_positions']}")