"""
Alpha Platform - IBKR Utility Module

Provides clean, reusable functions for connecting to Interactive Brokers
and fetching account/position data with live prices.

Place this file in: src/broker/ibkr_utils.py
"""

# Fix for Streamlit async issues - MUST be at the very top
import asyncio
import nest_asyncio
nest_asyncio.apply()

import threading
import time
import random
import math
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Position:
    """Represents a portfolio position."""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    account: str
    sec_type: str = "STK"
    currency: str = "USD"
    exchange: str = "SMART"


@dataclass
class AccountSummary:
    """Represents account summary information."""
    account_id: str
    net_liquidation: float
    total_cash: float
    buying_power: float
    gross_position_value: float
    available_funds: float
    excess_liquidity: float
    currency: str = "USD"
@dataclass
class OpenOrder:
    """Represents an open order."""
    order_id: int
    perm_id: int
    symbol: str
    action: str  # BUY or SELL
    quantity: float
    order_type: str  # LMT, MKT, STP, etc.
    limit_price: Optional[float]
    status: str  # Submitted, PreSubmitted, Filled, Cancelled, etc.
    filled: float
    remaining: float
    avg_fill_price: float
    account: str
    time_placed: Optional[str] = None



class IBKRConnection:
    """
    Manages IBKR connection using ib_insync library.

    Usage:
        conn = IBKRConnection()
        if conn.connect():
            accounts = conn.get_accounts()
            positions = conn.get_positions(accounts[0])
            conn.disconnect()
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7496,
                 client_id: Optional[int] = None, timeout: int = 10):
        """
        Initialize IBKR connection parameters.

        Args:
            host: TWS/Gateway host (default: localhost)
            port: TWS/Gateway port (7496 for TWS, 4001 for Gateway)
            client_id: Unique client ID (random if not specified)
            timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.client_id = client_id or random.randint(10000, 99999)
        self.timeout = timeout
        self.ib = None
        self._connected = False

    def cancel_order(self, perm_id: int) -> Tuple[bool, str]:
        """
        Cancel a specific order by permanent ID.
        Note: Orders placed from TWS or other clients may require global cancel.

        Returns:
            Tuple of (success, message)
        """
        if not self.is_connected():
            return False, "Not connected to IBKR"

        try:
            self.ib.reqAllOpenOrders()
            self.ib.sleep(0.5)

            for trade in self.ib.openTrades():
                if trade.order.permId == perm_id:
                    # If orderId is 0, the order was placed from another client
                    if trade.order.orderId == 0:
                        # Use global cancel as workaround
                        self.ib.reqGlobalCancel()
                        self.ib.sleep(1)
                        return True, f"Global cancel sent (includes PermID: {perm_id})"
                    else:
                        self.ib.cancelOrder(trade.order)
                        self.ib.sleep(0.5)
                        return True, f"Cancel request sent (PermID: {perm_id})"

            return False, f"Order with PermID {perm_id} not found"

        except Exception as e:
            return False, f"Error cancelling order: {e}"

    def cancel_all_orders(self, account_id: Optional[str] = None) -> Tuple[int, str]:
        """
        Cancel all open orders.

        Returns:
            Tuple of (cancelled_count, message)
        """
        if not self.is_connected():
            return 0, "Not connected to IBKR"

        try:
            self.ib.reqAllOpenOrders()
            self.ib.sleep(0.5)

            # Count orders before cancel
            trades = list(self.ib.openTrades())
            if account_id:
                trades = [t for t in trades if getattr(t.order, 'account', '') == account_id]

            count = len(trades)

            if count == 0:
                return 0, "No open orders to cancel"

            # Use global cancel - most reliable method
            self.ib.reqGlobalCancel()
            self.ib.sleep(1)

            return count, f"Cancel request sent for {count} order(s)"

        except Exception as e:
            return 0, f"Error cancelling orders: {e}"

    def close_position(self, symbol: str, account_id: str) -> Tuple[bool, str]:
        """
        Close a position by placing a market order in opposite direction.

        Returns:
            Tuple of (success, message)
        """
        if not self.is_connected():
            return False, "Not connected to IBKR"

        try:
            from ib_insync import Stock, MarketOrder

            # Find the position
            positions = self.ib.positions()
            target_pos = None

            for pos in positions:
                if pos.contract.symbol == symbol and pos.account == account_id:
                    target_pos = pos
                    break

            if not target_pos:
                return False, f"Position {symbol} not found"

            quantity = abs(target_pos.position)
            if quantity == 0:
                return False, f"No position in {symbol}"

            # Determine action (opposite of current position)
            action = "SELL" if target_pos.position > 0 else "BUY"

            # Create contract and order
            contract = Stock(symbol, "SMART", target_pos.contract.currency)
            self.ib.qualifyContracts(contract)

            order = MarketOrder(action, quantity)
            order.account = account_id

            # Place the order
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(0.5)

            return True, f"Closing {symbol}: {action} {quantity:.0f} shares"

        except Exception as e:
            return False, f"Error closing position: {e}"


    def get_cached_price(self, symbol: str, max_age_minutes: int = 60) -> Optional[float]:
        """
        Get price from cache if available and not too old.
        """
        import sqlite3
        try:
            with sqlite3.connect('portfolio_prices.db') as conn:
                cursor = conn.cursor()
                cursor.execute("""
                               SELECT price, timestamp
                               FROM prices
                               WHERE symbol = ?
                               ORDER BY timestamp DESC LIMIT 1
                               """, (symbol,))
                result = cursor.fetchone()

                if result:
                    price, timestamp_str = result
                    cached_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    age_minutes = (datetime.now() - cached_time).total_seconds() / 60

                    if age_minutes <= max_age_minutes:
                        return price
        except:
            pass
        return None

    def save_price_to_cache(self, symbol: str, price: float):
        """
        Save price to cache.
        """
        import sqlite3
        try:
            with sqlite3.connect('portfolio_prices.db') as conn:
                cursor = conn.cursor()
                cursor.execute("""
                               CREATE TABLE IF NOT EXISTS prices
                               (
                                   symbol
                                   TEXT,
                                   price
                                   REAL,
                                   timestamp
                                   DATETIME,
                                   PRIMARY
                                   KEY
                               (
                                   symbol
                               )
                                   )
                               """)
                cursor.execute("""
                    INSERT OR REPLACE INTO prices (symbol, price, timestamp)
                    VALUES (?, ?, datetime('now', 'localtime'))
                """, (symbol, price))
                conn.commit()
        except Exception as e:
            print(f"Error saving price to cache: {e}")


    def get_open_orders(self, account_id: Optional[str] = None) -> List[OpenOrder]:
        """
        Get all open orders, optionally filtered by account.

        Args:
            account_id: Optional account ID to filter orders

        Returns:
            List of OpenOrder objects
        """
        if not self.is_connected():
            return []

        try:
            # Request ALL open orders
            self.ib.reqAllOpenOrders()
            self.ib.sleep(1)

            # Get open trades (includes order + status)
            trades = list(self.ib.openTrades())

            # Debug: print what we found
            print(f"DEBUG: Found {len(trades)} trades")

            orders = []
            for trade in trades:
                # Filter by account if specified
                order_account = getattr(trade.order, 'account', '')
                if account_id and order_account != account_id:
                    continue

                # Extract order details
                contract = trade.contract
                order = trade.order
                order_status = trade.orderStatus

                orders.append(OpenOrder(
                    order_id=order.orderId,
                    perm_id=order.permId,
                    symbol=contract.symbol,
                    action=order.action,
                    quantity=order.totalQuantity,
                    order_type=order.orderType,
                    limit_price=order.lmtPrice if order.orderType == 'LMT' else None,
                    status=order_status.status,
                    filled=order_status.filled,
                    remaining=order_status.remaining,
                    avg_fill_price=order_status.avgFillPrice,
                    account=order_account,
                    time_placed=str(trade.log[0].time) if trade.log else None,
                ))

            return orders

        except Exception as e:
            print(f"Error getting open orders: {e}")
            return []


    def connect(self) -> bool:
        """
        Establish connection to IBKR TWS/Gateway.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            from ib_insync import IB

            self.ib = IB()
            self.ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=self.timeout
            )
            self._connected = self.ib.isConnected()

            if self._connected:
                # Small delay to ensure connection is stable
                time.sleep(0.5)

            return self._connected

        except Exception as e:
            print(f"IBKR Connection Error: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from IBKR."""
        if self.ib and self._connected:
            try:
                self.ib.disconnect()
            except:
                pass
        self._connected = False

    def is_connected(self) -> bool:
        """Check if currently connected."""
        if self.ib:
            return self.ib.isConnected()
        return False

    def get_accounts(self) -> List[str]:
        """
        Get list of available account IDs.

        Returns:
            List of account ID strings
        """
        if not self.is_connected():
            return []

        try:
            # managedAccounts() returns list of account IDs
            accounts = self.ib.managedAccounts()
            return accounts if accounts else []
        except Exception as e:
            print(f"Error getting accounts: {e}")
            return []

    def get_account_summary(self, account_id: str) -> Optional[AccountSummary]:
        """
        Get account summary for a specific account.

        Args:
            account_id: The account ID to query

        Returns:
            AccountSummary object or None if error
        """
        if not self.is_connected():
            return None

        try:
            # Request account summary
            summary_items = self.ib.accountSummary(account_id)

            # Parse summary items into dict
            summary_dict = {}
            for item in summary_items:
                if item.currency == "USD":
                    summary_dict[item.tag] = float(item.value)

            return AccountSummary(
                account_id=account_id,
                net_liquidation=summary_dict.get("NetLiquidation", 0),
                total_cash=summary_dict.get("TotalCashValue", 0),
                buying_power=summary_dict.get("BuyingPower", 0),
                gross_position_value=summary_dict.get("GrossPositionValue", 0),
                available_funds=summary_dict.get("AvailableFunds", 0),
                excess_liquidity=summary_dict.get("ExcessLiquidity", 0),
            )

        except Exception as e:
            print(f"Error getting account summary: {e}")
            return None

    def get_live_price(self, symbol: str, exchange: str = "SMART",
                       currency: str = "USD") -> Optional[float]:
        """
        Get live price for a single symbol using ib_insync.
        """
        if not self.is_connected():
            return None

        try:
            from ib_insync import Stock

            contract = Stock(symbol, exchange, currency)
            qualified = self.ib.qualifyContracts(contract)
            if not qualified:
                return None

            qualified_contract = qualified[0]
            ticker = self.ib.reqMktData(qualified_contract)
            self.ib.sleep(1)

            price = None
            if ticker.last and not math.isnan(ticker.last) and ticker.last > 0:
                price = ticker.last
            elif ticker.marketPrice() and not math.isnan(ticker.marketPrice()) and ticker.marketPrice() > 0:
                price = ticker.marketPrice()
            elif ticker.close and not math.isnan(ticker.close) and ticker.close > 0:
                price = ticker.close
            elif ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > 0:
                price = (ticker.bid + ticker.ask) / 2

            self.ib.cancelMktData(qualified_contract)
            return price

        except Exception as e:
            print(f"Error getting live price for {symbol}: {e}")
            return None

    def get_live_prices_batch(self, symbols: List[str], exchange: str = "SMART",
                              currency: str = "USD", use_cache: bool = True,
                              cache_max_age: int = 60) -> Dict[str, float]:
        """
        Get live prices for multiple symbols efficiently.
        Uses cache for recently fetched prices.

        Args:
            symbols: List of symbols
            exchange: Exchange (default SMART)
            currency: Currency (default USD)
            use_cache: Whether to use cached prices
            cache_max_age: Max age in minutes for cached prices
        """
        if not self.is_connected():
            return {}

        prices = {}
        symbols_to_fetch = []

        # Check cache first
        if use_cache:
            for symbol in symbols:
                cached_price = self.get_cached_price(symbol, cache_max_age)
                if cached_price:
                    prices[symbol] = cached_price
                else:
                    symbols_to_fetch.append(symbol)

            if prices:
                print(f"Using {len(prices)} cached prices, fetching {len(symbols_to_fetch)} new")
        else:
            symbols_to_fetch = symbols

        if not symbols_to_fetch:
            return prices

        try:
            from ib_insync import Stock

            # Process in smaller batches
            batch_size = 20

            for i in range(0, len(symbols_to_fetch), batch_size):
                batch_symbols = symbols_to_fetch[i:i + batch_size]
                contracts = [Stock(symbol, exchange, currency) for symbol in batch_symbols]

                try:
                    qualified_contracts = self.ib.qualifyContracts(*contracts)
                except:
                    qualified_contracts = []

                if not qualified_contracts:
                    continue

                tickers = {}
                for contract in qualified_contracts:
                    try:
                        ticker = self.ib.reqMktData(contract, '', False, False)
                        tickers[contract.symbol] = (contract, ticker)
                    except:
                        pass

                self.ib.sleep(2)

                for symbol, (contract, ticker) in tickers.items():
                    price = None

                    if ticker.last and not math.isnan(ticker.last) and ticker.last > 0:
                        price = ticker.last
                    elif ticker.close and not math.isnan(ticker.close) and ticker.close > 0:
                        price = ticker.close
                    elif ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > 0:
                        price = (ticker.bid + ticker.ask) / 2
                    elif hasattr(ticker, 'marketPrice'):
                        mp = ticker.marketPrice()
                        if mp and not math.isnan(mp) and mp > 0:
                            price = mp

                    if price:
                        prices[symbol] = price
                        # Save to cache
                        self.save_price_to_cache(symbol, price)

                    try:
                        self.ib.cancelMktData(contract)
                    except:
                        pass

            # Fallback: Get historical data for missing symbols
            missing = [s for s in symbols_to_fetch if s not in prices]
            if missing:
                print(f"Fetching historical prices for {len(missing)} missing symbols...")
                for symbol in missing:
                    price = self.get_price_with_fallback(symbol, exchange, currency)
                    if price:
                        prices[symbol] = price
                        self.save_price_to_cache(symbol, price)

        except Exception as e:
            print(f"Error getting batch prices: {e}")

        return prices

    def get_price_with_fallback(self, symbol: str, exchange: str = "SMART",
                                currency: str = "USD") -> Optional[float]:
        """
        Get price with fallback. Yahoo Finance FIRST (fast), then IBKR.
        """
        # PRIMARY: Try Yahoo Finance first (faster and doesn't require connection)
        try:
            from src.utils.price_fetcher import get_price as yahoo_get_price
            price = yahoo_get_price(symbol)
            if price and price > 0:
                return price
        except Exception:
            pass
        
        # FALLBACK: IBKR if Yahoo fails
        if not self.is_connected():
            return None

        try:
            from ib_insync import Stock

            contract = Stock(symbol, exchange, currency)
            qualified = self.ib.qualifyContracts(contract)

            if not qualified:
                return None

            qualified_contract = qualified[0]

            # Try live price first
            ticker = self.ib.reqMktData(qualified_contract, '', False, False)
            self.ib.sleep(1)

            price = None
            if ticker.last and not math.isnan(ticker.last) and ticker.last > 0:
                price = ticker.last
            elif ticker.close and not math.isnan(ticker.close) and ticker.close > 0:
                price = ticker.close
            elif ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > 0:
                price = (ticker.bid + ticker.ask) / 2

            self.ib.cancelMktData(qualified_contract)

            # If no live price, try historical data
            if not price:
                bars = self.ib.reqHistoricalData(
                    qualified_contract,
                    endDateTime='',
                    durationStr='1 D',
                    barSizeSetting='1 day',
                    whatToShow='ADJUSTED_LAST',
                    useRTH=True,
                    formatDate=1
                )
                if bars:
                    price = bars[-1].close

            return price

        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return None

    def get_positions(self, account_id: Optional[str] = None) -> List[Position]:
        """
        Get all positions with cached prices for P&L calculation.
        """
        if not self.is_connected():
            return []

        try:
            raw_positions = self.ib.positions()

            positions = []
            symbols = []

            for pos in raw_positions:
                if account_id and pos.account != account_id:
                    continue
                if pos.position == 0:
                    continue
                symbols.append(pos.contract.symbol)

            # Get cached prices for all symbols
            cached_prices = {}
            for symbol in symbols:
                cached = self.get_cached_price(symbol, max_age_minutes=120)  # 2 hour cache
                if cached:
                    cached_prices[symbol] = cached

            for pos in raw_positions:
                if account_id and pos.account != account_id:
                    continue
                if pos.position == 0:
                    continue

                symbol = pos.contract.symbol
                sec_type = pos.contract.secType
                currency = pos.contract.currency
                avg_cost = pos.avgCost
                quantity = pos.position

                # Use cached price if available, otherwise avg_cost
                current_price = cached_prices.get(symbol, avg_cost)
                market_value = quantity * current_price
                cost_basis = quantity * avg_cost
                unrealized_pnl = market_value - cost_basis
                unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis != 0 else 0

                positions.append(Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_cost=avg_cost,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    account=pos.account,
                    sec_type=sec_type,
                    currency=currency,
                ))

            return positions

        except Exception as e:
            print(f"Error getting positions: {e}")
            return []

    def get_positions_with_prices(self, account_id: Optional[str] = None,
                                   progress_callback=None) -> List[Position]:
        """
        Get positions with live prices.

        Args:
            account_id: Optional account ID to filter positions
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of Position objects with live prices
        """
        if not self.is_connected():
            return []

        # First get positions (fast)
        positions = self.get_positions(account_id)

        if not positions:
            return []

        # Get symbols list
        symbols = [p.symbol for p in positions if p.sec_type == "STK"]

        # Get live prices in batch
        if progress_callback:
            progress_callback(0, len(symbols))

        live_prices = self.get_live_prices_batch(symbols)

        if progress_callback:
            progress_callback(len(symbols), len(symbols))

        # Update positions with live prices
        updated_positions = []
        for pos in positions:
            current_price = live_prices.get(pos.symbol, pos.avg_cost)

            # Calculate P&L
            market_value = pos.quantity * current_price
            cost_basis = pos.quantity * pos.avg_cost
            unrealized_pnl = market_value - cost_basis
            unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis != 0 else 0

            updated_positions.append(Position(
                symbol=pos.symbol,
                quantity=pos.quantity,
                avg_cost=pos.avg_cost,
                current_price=current_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                account=pos.account,
                sec_type=pos.sec_type,
                currency=pos.currency,
            ))

        return updated_positions


# =============================================================================
# CONVENIENCE FUNCTIONS (for use without class instantiation)
# =============================================================================

def get_ibkr_accounts(host: str = "127.0.0.1", port: int = 7496) -> Tuple[List[str], Optional[str]]:
    """
    Quick function to get available IBKR accounts.

    Returns:
        Tuple of (account_list, error_message)
    """
    conn = IBKRConnection(host=host, port=port)
    try:
        if not conn.connect():
            return [], "Failed to connect to IBKR TWS/Gateway"

        accounts = conn.get_accounts()
        if not accounts:
            return [], "No accounts found"

        return accounts, None

    except Exception as e:
        return [], str(e)
    finally:
        conn.disconnect()


def get_ibkr_account_summary(account_id: str, host: str = "127.0.0.1",
                              port: int = 7496) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Quick function to get account summary.

    Returns:
        Tuple of (summary_dict, error_message)
    """
    conn = IBKRConnection(host=host, port=port)
    try:
        if not conn.connect():
            return None, "Failed to connect to IBKR"

        summary = conn.get_account_summary(account_id)
        if not summary:
            return None, "Failed to get account summary"

        return {
            "account_id": summary.account_id,
            "net_liquidation": summary.net_liquidation,
            "total_cash": summary.total_cash,
            "buying_power": summary.buying_power,
            "gross_position_value": summary.gross_position_value,
            "available_funds": summary.available_funds,
            "excess_liquidity": summary.excess_liquidity,
        }, None

    except Exception as e:
        return None, str(e)
    finally:
        conn.disconnect()


def get_ibkr_positions(account_id: Optional[str] = None, host: str = "127.0.0.1",
                       port: int = 7496, fetch_prices: bool = False) -> Tuple[List[Dict], Optional[str]]:
    """
    Quick function to get positions.

    Args:
        account_id: Optional account filter
        host: IBKR host
        port: IBKR port
        fetch_prices: If True, fetch live prices (slower but accurate)

    Returns:
        Tuple of (positions_list, error_message)
    """
    conn = IBKRConnection(host=host, port=port)
    try:
        if not conn.connect():
            return [], "Failed to connect to IBKR"

        if fetch_prices:
            positions = conn.get_positions_with_prices(account_id)
        else:
            positions = conn.get_positions(account_id)

        # Convert to dict list for easy JSON serialization
        return [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "avg_cost": p.avg_cost,
                "current_price": p.current_price,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl,
                "unrealized_pnl_pct": p.unrealized_pnl_pct,
                "account": p.account,
                "sec_type": p.sec_type,
                "currency": p.currency,
            }
            for p in positions
        ], None

    except Exception as e:
        return [], str(e)
    finally:
        conn.disconnect()


# =============================================================================
# STREAMLIT CACHED FUNCTIONS
# =============================================================================

def load_ibkr_data_cached(account_id: str, host: str = "127.0.0.1", port: int = 7496,
                          fetch_live_prices: bool = False):
    """
    Load all IBKR data in one connection.
    """
    result = {
        "accounts": [],
        "summary": None,
        "positions": [],
        "open_orders": [],
        "error": None,
        "timestamp": datetime.now().isoformat(),
        "prices_live": fetch_live_prices
    }

    conn = IBKRConnection(host=host, port=port)
    try:
        if not conn.connect():
            result["error"] = "Failed to connect to IBKR TWS/Gateway. Is it running?"
            return result

        accounts = conn.get_accounts()
        result["accounts"] = accounts

        if not accounts:
            result["error"] = "No accounts found"
            return result

        target_account = account_id if account_id in accounts else accounts[0]

        summary = conn.get_account_summary(target_account)
        if summary:
            result["summary"] = {
                "account_id": summary.account_id,
                "net_liquidation": summary.net_liquidation,
                "total_cash": summary.total_cash,
                "buying_power": summary.buying_power,
                "gross_position_value": summary.gross_position_value,
            }

        # Get positions with or without live prices
        if fetch_live_prices:
            positions = conn.get_positions_with_prices(target_account)
        else:
            positions = conn.get_positions(target_account)

        result["positions"] = [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "avg_cost": p.avg_cost,
                "current_price": p.current_price,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl,
                "unrealized_pnl_pct": p.unrealized_pnl_pct,
                "account": p.account,
                "sec_type": p.sec_type,
            }
            for p in positions
        ]

        # Get open orders
        open_orders = conn.get_open_orders(target_account)
        result["open_orders"] = [
            {
                "order_id": o.order_id,
                "perm_id": o.perm_id,
                "symbol": o.symbol,
                "action": o.action,
                "quantity": o.quantity,
                "order_type": o.order_type,
                "limit_price": o.limit_price,
                "status": o.status,
                "filled": o.filled,
                "remaining": o.remaining,
                "avg_fill_price": o.avg_fill_price,
                "account": o.account,
                "time_placed": o.time_placed,
            }
            for o in open_orders
        ]

        return result

    except Exception as e:
        result["error"] = str(e)
        return result
    finally:
        conn.disconnect()


if __name__ == "__main__":
    # Test the module
    print("Testing IBKR Connection...")

    accounts, error = get_ibkr_accounts()
    if error:
        print(f"Error: {error}")
    else:
        print(f"Found accounts: {accounts}")

        if accounts:
            # Test account summary
            summary, error = get_ibkr_account_summary(accounts[0])
            if summary:
                print(f"\nAccount Summary for {accounts[0]}:")
                print(f"  Net Liquidation: ${summary['net_liquidation']:,.2f}")
                print(f"  Total Cash: ${summary['total_cash']:,.2f}")
                print(f"  Buying Power: ${summary['buying_power']:,.2f}")

            # Test positions without live prices
            print("\n--- Positions (without live prices) ---")
            positions, error = get_ibkr_positions(accounts[0], fetch_prices=False)
            if positions:
                print(f"Found {len(positions)} positions")
                for p in positions[:3]:
                    print(f"  {p['symbol']}: {p['quantity']} @ ${p['avg_cost']:.2f}")

            # Test positions WITH live prices
            print("\n--- Positions (with live prices) ---")
            positions, error = get_ibkr_positions(accounts[0], fetch_prices=True)
            if positions:
                print(f"Found {len(positions)} positions with live prices")
                for p in positions[:5]:
                    print(f"  {p['symbol']}: {p['quantity']} @ ${p['current_price']:.2f} "
                          f"(Avg: ${p['avg_cost']:.2f}, P&L: ${p['unrealized_pnl']:+,.2f})")
