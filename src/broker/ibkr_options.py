"""
IBKR Real-Time Options Data Fetcher

Fetches real-time options chains from Interactive Brokers TWS/Gateway.
Provides live Greeks (Delta, Gamma, Vega, Theta) and accurate pricing.

Falls back to Yahoo Finance if IBKR connection fails.

Location: src/broker/ibkr_options.py
Author: Alpha Research Platform
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import threading

from src.utils.logging import get_logger
import atexit

logger = get_logger(__name__)

# Cleanup on exit
def _cleanup_connections():
    global _fetcher
    if _fetcher is not None:
        try:
            _fetcher.close()
        except:
            pass

atexit.register(_cleanup_connections)

# Try to import IBKR API
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.common import TickerId, TickAttrib
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    logger.warning("IBKR API not available - will use Yahoo Finance fallback")


class DataSource(Enum):
    """Options data source."""
    IBKR = "IBKR"
    YAHOO = "YAHOO"
    UNKNOWN = "UNKNOWN"


@dataclass
class OptionsChainResult:
    """Result from options chain fetch."""
    ticker: str
    calls: pd.DataFrame
    puts: pd.DataFrame
    stock_price: float
    data_source: DataSource
    timestamp: datetime = field(default_factory=datetime.now)
    error: str = ""


# =============================================================================
# IBKR OPTIONS CONNECTION (only defined if IBKR API available)
# =============================================================================

if IBKR_AVAILABLE:

    class IBKROptionsConnection(EWrapper, EClient):
        """
        IBKR connection for fetching real-time options data.
        """

        def __init__(self, host: str = "127.0.0.1", port: int = 7496, client_id: int = 10):
            EClient.__init__(self, self)
            self.host = host
            self.port = port
            self.client_id = client_id

            # Data storage
            self.options_data: Dict[int, Dict] = {}
            self.contract_details: Dict[int, List] = {}
            self.stock_prices: Dict[str, float] = {}

            # Request tracking
            self.next_req_id = 1000
            self.pending_requests: Dict[int, threading.Event] = {}

            # Connection state
            self._connected = False
            self._connection_event = threading.Event()

        def connect_and_run(self) -> bool:
            """Connect to IBKR and start message processing."""
            try:
                self.connect(self.host, self.port, self.client_id)

                # Start message processing thread
                thread = threading.Thread(target=self.run, daemon=True)
                thread.start()

                # Wait for connection
                if self._connection_event.wait(timeout=5):
                    logger.info(f"Connected to IBKR at {self.host}:{self.port}")
                    return True
                else:
                    logger.warning("IBKR connection timeout")
                    return False

            except Exception as e:
                logger.error(f"Failed to connect to IBKR: {e}")
                return False

        def nextValidId(self, orderId: int):
            """Called when connected - sets next valid order ID."""
            self._connected = True
            self._connection_event.set()
            self.next_req_id = orderId

        def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""):
            """Handle errors from IBKR."""
            if errorCode in [2104, 2106, 2158]:
                return

            if errorCode == 200:
                logger.debug(f"No options found for request {reqId}")
            elif errorCode >= 1000:
                logger.warning(f"IBKR Error {errorCode}: {errorString}")

            if reqId in self.pending_requests:
                self.pending_requests[reqId].set()

        def contractDetails(self, reqId: int, contractDetails):
            """Receive contract details (options chain)."""
            if reqId not in self.contract_details:
                self.contract_details[reqId] = []
            self.contract_details[reqId].append(contractDetails)

        def contractDetailsEnd(self, reqId: int):
            """Contract details request complete."""
            if reqId in self.pending_requests:
                self.pending_requests[reqId].set()

        def tickPrice(self, reqId: int, tickType: int, price: float, attrib: TickAttrib):
            """Receive price tick."""
            if reqId not in self.options_data:
                self.options_data[reqId] = {}

            if tickType == 1:
                self.options_data[reqId]['bid'] = price
            elif tickType == 2:
                self.options_data[reqId]['ask'] = price
            elif tickType == 4:
                self.options_data[reqId]['last'] = price
            elif tickType == 6:
                self.options_data[reqId]['high'] = price
            elif tickType == 7:
                self.options_data[reqId]['low'] = price
            elif tickType == 9:
                self.options_data[reqId]['close'] = price

        def tickSize(self, reqId: int, tickType: int, size: int):
            """Receive size tick."""
            if reqId not in self.options_data:
                self.options_data[reqId] = {}

            if tickType == 0:
                self.options_data[reqId]['bidSize'] = size
            elif tickType == 3:
                self.options_data[reqId]['askSize'] = size
            elif tickType == 5:
                self.options_data[reqId]['lastSize'] = size
            elif tickType == 8:
                self.options_data[reqId]['volume'] = size

        def tickOptionComputation(self, reqId: int, tickType: int, tickAttrib: int,
                                   impliedVol: float, delta: float, optPrice: float,
                                   pvDividend: float, gamma: float, vega: float,
                                   theta: float, undPrice: float):
            """Receive options Greeks."""
            if reqId not in self.options_data:
                self.options_data[reqId] = {}

            if tickType in [10, 11, 12, 13]:
                self.options_data[reqId]['impliedVolatility'] = impliedVol if impliedVol > 0 else None
                self.options_data[reqId]['delta'] = delta if abs(delta) <= 1 else None
                self.options_data[reqId]['gamma'] = gamma if gamma >= 0 else None
                self.options_data[reqId]['vega'] = vega
                self.options_data[reqId]['theta'] = theta
                self.options_data[reqId]['undPrice'] = undPrice

        def tickSnapshotEnd(self, reqId: int):
            """Snapshot request complete."""
            if reqId in self.pending_requests:
                self.pending_requests[reqId].set()

        def get_stock_price(self, ticker: str, timeout: float = 5.0) -> float:
            """Get current stock price from IBKR."""
            contract = Contract()
            contract.symbol = ticker
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"

            req_id = self.next_req_id
            self.next_req_id += 1

            self.pending_requests[req_id] = threading.Event()
            self.options_data[req_id] = {}

            self.reqMktData(req_id, contract, "", True, False, [])

            self.pending_requests[req_id].wait(timeout=timeout)

            data = self.options_data.get(req_id, {})
            price = data.get('last') or data.get('close') or data.get('bid', 0)

            self.cancelMktData(req_id)

            return price

        def get_options_chain(self, ticker: str, max_expiries: int = 2,
                              timeout: float = 15.0,
                              strike_range_pct: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
            """
            Get options chain from IBKR with smart filtering.

            Only fetches near-the-money options for speed and relevance.

            Args:
                ticker: Stock symbol
                max_expiries: Max expiration dates (default 2 for speed)
                timeout: Timeout in seconds
                strike_range_pct: Only strikes within this % of stock price (default 15%)
            """
            stock_price = self.get_stock_price(ticker)

            if stock_price <= 0:
                logger.warning(f"Could not get stock price for {ticker}")
                return pd.DataFrame(), pd.DataFrame(), 0

            # Calculate strike range (near-the-money only)
            min_strike = stock_price * (1 - strike_range_pct)
            max_strike = stock_price * (1 + strike_range_pct)

            logger.info(f"IBKR: {ticker} @ ${stock_price:.2f}, fetching strikes ${min_strike:.0f}-${max_strike:.0f}")

            contract = Contract()
            contract.symbol = ticker
            contract.secType = "OPT"
            contract.exchange = "SMART"
            contract.currency = "USD"

            req_id = self.next_req_id
            self.next_req_id += 1

            self.pending_requests[req_id] = threading.Event()
            self.contract_details[req_id] = []

            self.reqContractDetails(req_id, contract)

            if not self.pending_requests[req_id].wait(timeout=timeout):
                logger.warning(f"Timeout getting options chain for {ticker}")
                return pd.DataFrame(), pd.DataFrame(), stock_price

            contracts = self.contract_details.get(req_id, [])

            if not contracts:
                logger.warning(f"No options contracts found for {ticker}")
                return pd.DataFrame(), pd.DataFrame(), stock_price

            # Filter 1: Near-the-money strikes only
            contracts = [c for c in contracts if min_strike <= c.contract.strike <= max_strike]

            # Filter 2: Near-term expiries only
            expiries = sorted(set(c.contract.lastTradeDateOrContractMonth for c in contracts))
            expiries = expiries[:max_expiries]

            filtered_contracts = [
                c for c in contracts
                if c.contract.lastTradeDateOrContractMonth in expiries
            ]

            logger.info(f"IBKR: Filtered to {len(filtered_contracts)} contracts (from {len(self.contract_details[req_id])} total)")

            if not filtered_contracts:
                return pd.DataFrame(), pd.DataFrame(), stock_price

            calls_data = []
            puts_data = []

            # Process in batches
            batch_size = 50

            for i in range(0, len(filtered_contracts), batch_size):
                batch = filtered_contracts[i:i+batch_size]
                batch_req_ids = []

                for cd in batch:
                    opt_contract = cd.contract

                    opt_req_id = self.next_req_id
                    self.next_req_id += 1
                    batch_req_ids.append((opt_req_id, cd))

                    self.pending_requests[opt_req_id] = threading.Event()
                    self.options_data[opt_req_id] = {}

                    # Request Greeks
                    self.reqMktData(opt_req_id, opt_contract, "100,101,104,106", True, False, [])

                # Wait for batch
                time.sleep(0.3)

                # Collect results
                for opt_req_id, cd in batch_req_ids:
                    if opt_req_id in self.pending_requests:
                        self.pending_requests[opt_req_id].wait(timeout=1.5)

                    data = self.options_data.get(opt_req_id, {})
                    opt_contract = cd.contract

                    expiry_raw = opt_contract.lastTradeDateOrContractMonth
                    if len(expiry_raw) == 8:
                        expiry = f"{expiry_raw[:4]}-{expiry_raw[4:6]}-{expiry_raw[6:8]}"
                    else:
                        expiry = expiry_raw

                    try:
                        exp_date = datetime.strptime(expiry, '%Y-%m-%d')
                        days_to_exp = (exp_date - datetime.now()).days
                    except:
                        days_to_exp = 30

                    row = {
                        'strike': opt_contract.strike,
                        'expiry': expiry,
                        'daysToExpiry': days_to_exp,
                        'bid': data.get('bid', 0),
                        'ask': data.get('ask', 0),
                        'lastPrice': data.get('last', 0),
                        'volume': data.get('volume', 0),
                        'openInterest': 0,
                        'impliedVolatility': data.get('impliedVolatility'),
                        'delta': data.get('delta'),
                        'gamma': data.get('gamma'),
                        'vega': data.get('vega'),
                        'theta': data.get('theta'),
                        'inTheMoney': (opt_contract.right == 'C' and opt_contract.strike < stock_price) or
                                      (opt_contract.right == 'P' and opt_contract.strike > stock_price),
                    }

                    if opt_contract.right == 'C':
                        calls_data.append(row)
                    else:
                        puts_data.append(row)

                    self.cancelMktData(opt_req_id)

            calls_df = pd.DataFrame(calls_data) if calls_data else pd.DataFrame()
            puts_df = pd.DataFrame(puts_data) if puts_data else pd.DataFrame()

            logger.info(f"IBKR: Got {len(calls_df)} calls, {len(puts_df)} puts for {ticker}")

            return calls_df, puts_df, stock_price

        def disconnect_safe(self):
            """Safely disconnect from IBKR."""
            try:
                if self._connected:
                    self.disconnect()
                    self._connected = False
                    logger.debug("IBKR options connection closed")
            except:
                pass
            finally:
                # Force thread cleanup
                import sys
                if hasattr(self, 'conn') and self.conn:
                    self.conn = None

# =============================================================================
# YAHOO FINANCE FALLBACK
# =============================================================================

def get_options_chain_yahoo(ticker: str, max_expiries: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Get options chain from Yahoo Finance (fallback)."""
    import yfinance as yf

    try:
        stock = yf.Ticker(ticker)

        info = stock.info
        stock_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose', 0)

        if not stock_price:
            hist = stock.history(period='1d')
            if not hist.empty:
                stock_price = hist['Close'].iloc[-1]

        expiries = stock.options

        if not expiries:
            logger.warning(f"Yahoo: No options available for {ticker}")
            return pd.DataFrame(), pd.DataFrame(), stock_price

        expiries_to_fetch = expiries[:max_expiries]

        all_calls = []
        all_puts = []

        for expiry in expiries_to_fetch:
            try:
                opt_chain = stock.option_chain(expiry)

                calls = opt_chain.calls.copy()
                puts = opt_chain.puts.copy()

                calls['expiry'] = expiry
                puts['expiry'] = expiry

                exp_date = datetime.strptime(expiry, '%Y-%m-%d')
                days_to_exp = (exp_date - datetime.now()).days
                calls['daysToExpiry'] = days_to_exp
                puts['daysToExpiry'] = days_to_exp

                all_calls.append(calls)
                all_puts.append(puts)

            except Exception as e:
                logger.debug(f"Yahoo: Error fetching {ticker} options for {expiry}: {e}")
                continue

        calls_df = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
        puts_df = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()

        logger.info(f"Yahoo: Got {len(calls_df)} calls, {len(puts_df)} puts for {ticker}")

        return calls_df, puts_df, stock_price

    except Exception as e:
        logger.error(f"Yahoo: Error getting options for {ticker}: {e}")
        return pd.DataFrame(), pd.DataFrame(), 0


# =============================================================================
# UNIFIED OPTIONS FETCHER
# =============================================================================

class OptionsDataFetcher:
    """
    Unified options data fetcher.
    Tries IBKR first for real-time data, falls back to Yahoo Finance.
    """

    def __init__(self, ibkr_host: str = "127.0.0.1", ibkr_port: int = 7496):
        self.ibkr_host = ibkr_host
        self.ibkr_port = ibkr_port
        self._ibkr_conn = None
        self._ibkr_available = IBKR_AVAILABLE
        self._last_source = DataSource.UNKNOWN

    @property
    def last_source(self) -> DataSource:
        """Get the data source used in the last fetch."""
        return self._last_source

    def _get_ibkr_connection(self):
        """Get or create IBKR connection."""
        if not self._ibkr_available:
            return None

        if self._ibkr_conn is None or not self._ibkr_conn._connected:
            try:
                self._ibkr_conn = IBKROptionsConnection(
                    host=self.ibkr_host,
                    port=self.ibkr_port,
                    client_id=10
                )
                if not self._ibkr_conn.connect_and_run():
                    self._ibkr_conn = None
            except Exception as e:
                logger.warning(f"Could not connect to IBKR for options: {e}")
                self._ibkr_conn = None

        return self._ibkr_conn

    def get_options_chain(self, ticker: str, max_expiries: int = 2) -> OptionsChainResult:
        """Get options chain - tries IBKR first, falls back to Yahoo."""
        ticker = ticker.upper()

        # Try IBKR first
        if self._ibkr_available:
            conn = self._get_ibkr_connection()

            if conn and conn._connected:
                try:
                    calls_df, puts_df, stock_price = conn.get_options_chain(ticker, max_expiries)

                    # Disconnect immediately after getting data
                    conn.disconnect_safe()
                    self._ibkr_conn = None

                    if not calls_df.empty or not puts_df.empty:
                        self._last_source = DataSource.IBKR
                        logger.info(f"✅ {ticker}: Using IBKR real-time options data")

                        return OptionsChainResult(
                            ticker=ticker,
                            calls=calls_df,
                            puts=puts_df,
                            stock_price=stock_price,
                            data_source=DataSource.IBKR,
                        )
                except Exception as e:
                    logger.warning(f"IBKR options fetch failed for {ticker}: {e}")
                    if self._ibkr_conn:
                        self._ibkr_conn.disconnect_safe()
                        self._ibkr_conn = None

        # Fallback to Yahoo Finance
        logger.info(f"📊 {ticker}: Using Yahoo Finance options data (delayed)")

        calls_df, puts_df, stock_price = get_options_chain_yahoo(ticker, max_expiries)
        self._last_source = DataSource.YAHOO

        return OptionsChainResult(
            ticker=ticker,
            calls=calls_df,
            puts=puts_df,
            stock_price=stock_price,
            data_source=DataSource.YAHOO,
        )

    def close(self):
        """Close connections."""
        if self._ibkr_conn:
            self._ibkr_conn.disconnect_safe()
            self._ibkr_conn = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_fetcher: Optional[OptionsDataFetcher] = None

def get_options_fetcher(ibkr_host: str = "127.0.0.1", ibkr_port: int = 7496) -> OptionsDataFetcher:
    """Get singleton options fetcher."""
    global _fetcher
    if _fetcher is None:
        _fetcher = OptionsDataFetcher(ibkr_host, ibkr_port)
    return _fetcher


def fetch_options_chain(ticker: str, max_expiries: int = 4,
                        ibkr_host: str = "127.0.0.1",
                        ibkr_port: int = 7496) -> OptionsChainResult:
    """
    Fetch options chain for a ticker.
    Tries IBKR first, falls back to Yahoo.
    """
    fetcher = get_options_fetcher(ibkr_host, ibkr_port)
    return fetcher.get_options_chain(ticker, max_expiries)


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    print(f"\n{'='*60}")
    print(f"Fetching Options Chain for {ticker}")
    print(f"{'='*60}")

    result = fetch_options_chain(ticker)

    print(f"\nData Source: {result.data_source.value}")
    print(f"Stock Price: ${result.stock_price:.2f}")
    print(f"Calls: {len(result.calls)} contracts")
    print(f"Puts: {len(result.puts)} contracts")

    if not result.calls.empty:
        print(f"\nTop 5 Calls by Volume:")
        top_calls = result.calls.nlargest(5, 'volume')[['strike', 'expiry', 'volume', 'impliedVolatility']]
        print(top_calls.to_string(index=False))

    print(f"\n{'='*60}")