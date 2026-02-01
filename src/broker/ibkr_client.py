"""
IBKR Client - TWS/Gateway Connection
=====================================

Connects to Interactive Brokers TWS or Gateway to fetch:
- Account list
- Positions
- Account summary
- Real-time prices

Requires: pip install ib_insync

Author: HH Research Platform
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

# Try to import ib_insync
try:
    from ib_insync import IB, Stock, Contract, util
    IB_INSYNC_AVAILABLE = True
except ImportError:
    IB_INSYNC_AVAILABLE = False
    logger.warning("ib_insync not installed. Run: pip install ib_insync")


@dataclass
class Position:
    """Position data class."""
    symbol: str
    position: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    account: str


@dataclass
class AccountSummary:
    """Account summary data class."""
    account_id: str
    net_liquidation: float
    total_cash: float
    buying_power: float
    gross_position_value: float
    available_funds: float
    excess_liquidity: float
    currency: str = "USD"


class IBKRClient:
    """
    IBKR TWS/Gateway client for fetching account data.

    Usage:
        client = IBKRClient(host="127.0.0.1", port=7496)
        if client.connect():
            accounts = client.get_managed_accounts()
            positions = client.get_positions(accounts[0])
            summary = client.get_account_summary(accounts[0])
            client.disconnect()
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7496, client_id: int = 1):
        """
        Initialize IBKR client.

        Args:
            host: TWS/Gateway host (default: 127.0.0.1)
            port: TWS port (7496 for TWS, 4001 for Gateway)
            client_id: Client ID for connection
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib: Optional[IB] = None
        self._connected = False

    def connect(self, timeout: int = 10) -> bool:
        """
        Connect to TWS/Gateway.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if connected successfully
        """
        if not IB_INSYNC_AVAILABLE:
            logger.error("ib_insync not available")
            return False

        try:
            self.ib = IB()
            self.ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=timeout,
                readonly=True  # Read-only for safety
            )

            # Wait a moment for account data to arrive
            self.ib.sleep(0.5)

            self._connected = self.ib.isConnected()

            if self._connected:
                logger.info(f"Connected to IBKR at {self.host}:{self.port}")
            else:
                logger.warning("Connection established but not confirmed")

            return self._connected

        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from TWS/Gateway."""
        if self.ib and self._connected:
            try:
                self.ib.disconnect()
                logger.info("Disconnected from IBKR")
            except Exception as e:
                logger.debug(f"Disconnect error (ignored): {e}")
            finally:
                self._connected = False

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected and self.ib and self.ib.isConnected()

    def get_managed_accounts(self) -> List[str]:
        """
        Get list of managed accounts.

        Returns:
            List of account IDs (e.g., ['U1234567', 'U7654321'])
        """
        if not self.is_connected():
            logger.warning("Not connected to IBKR")
            return []

        try:
            # Give TWS time to send account list
            self.ib.sleep(0.3)

            accounts = self.ib.managedAccounts()

            if accounts:
                logger.info(f"Found {len(accounts)} account(s): {accounts}")
                return list(accounts)
            else:
                logger.warning("No accounts returned from TWS")
                return []

        except Exception as e:
            logger.error(f"Error getting managed accounts: {e}")
            return []

    def get_positions(self, account_id: str = None) -> List[Dict[str, Any]]:
        """
        Get positions for an account.

        Args:
            account_id: Account ID (optional, uses first account if not specified)

        Returns:
            List of position dictionaries
        """
        if not self.is_connected():
            logger.warning("Not connected to IBKR")
            return []

        try:
            # Request positions
            self.ib.reqPositions()
            self.ib.sleep(1)  # Wait for positions to arrive

            positions = self.ib.positions()

            result = []
            for pos in positions:
                # Filter by account if specified
                if account_id and pos.account != account_id:
                    continue

                # Get contract info
                contract = pos.contract
                symbol = contract.symbol

                # Calculate values
                avg_cost = pos.avgCost
                shares = pos.position

                position_dict = {
                    'symbol': symbol,
                    'position': shares,
                    'avgCost': avg_cost,
                    'marketValue': 0,  # Will be updated with live price
                    'unrealizedPNL': 0,
                    'realizedPNL': 0,
                    'account': pos.account,
                    'secType': contract.secType,
                    'exchange': contract.exchange,
                    'currency': contract.currency,
                }

                result.append(position_dict)

            logger.info(f"Found {len(result)} positions for account {account_id or 'all'}")
            return result

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def get_account_summary(self, account_id: str = None) -> Dict[str, Any]:
        """
        Get account summary.

        Args:
            account_id: Account ID (uses first if not specified)

        Returns:
            Account summary dictionary
        """
        if not self.is_connected():
            logger.warning("Not connected to IBKR")
            return {}

        try:
            # If no account specified, use first available
            if not account_id:
                accounts = self.get_managed_accounts()
                if not accounts:
                    return {}
                account_id = accounts[0]

            # Request account summary
            tags = [
                'NetLiquidation',
                'TotalCashValue',
                'BuyingPower',
                'GrossPositionValue',
                'AvailableFunds',
                'ExcessLiquidity',
                'Currency'
            ]

            summary = self.ib.accountSummary(account_id)
            self.ib.sleep(0.5)

            # Parse results
            result = {
                'account_id': account_id,
                'net_liquidation': 0,
                'total_cash': 0,
                'buying_power': 0,
                'gross_position_value': 0,
                'available_funds': 0,
                'excess_liquidity': 0,
                'currency': 'USD'
            }

            for item in summary:
                if item.account != account_id:
                    continue

                tag = item.tag
                value = item.value

                try:
                    if tag == 'NetLiquidation':
                        result['net_liquidation'] = float(value)
                    elif tag == 'TotalCashValue':
                        result['total_cash'] = float(value)
                    elif tag == 'BuyingPower':
                        result['buying_power'] = float(value)
                    elif tag == 'GrossPositionValue':
                        result['gross_position_value'] = float(value)
                    elif tag == 'AvailableFunds':
                        result['available_funds'] = float(value)
                    elif tag == 'ExcessLiquidity':
                        result['excess_liquidity'] = float(value)
                    elif tag == 'Currency':
                        result['currency'] = str(value)
                except (ValueError, TypeError):
                    pass

            logger.info(f"Account {account_id}: NAV=${result['net_liquidation']:,.2f}")
            return result

        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return {}

    def get_portfolio(self, account_id: str = None) -> List[Dict[str, Any]]:
        """
        Get portfolio with P&L data.

        Args:
            account_id: Account ID

        Returns:
            List of portfolio items with P&L
        """
        if not self.is_connected():
            return []

        try:
            # If no account specified, use first
            if not account_id:
                accounts = self.get_managed_accounts()
                if not accounts:
                    return []
                account_id = accounts[0]

            # Get portfolio items (includes P&L)
            portfolio = self.ib.portfolio(account_id)
            self.ib.sleep(0.5)

            result = []
            for item in portfolio:
                contract = item.contract

                result.append({
                    'symbol': contract.symbol,
                    'secType': contract.secType,
                    'position': item.position,
                    'marketPrice': item.marketPrice,
                    'marketValue': item.marketValue,
                    'averageCost': item.averageCost,
                    'unrealizedPNL': item.unrealizedPNL,
                    'realizedPNL': item.realizedPNL,
                    'account': account_id,
                })

            logger.info(f"Portfolio has {len(result)} items")
            return result

        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            return []

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_ibkr_accounts(host: str = "127.0.0.1", port: int = 7496) -> tuple:
    """
    Quick function to get IBKR accounts.

    Returns:
        Tuple of (accounts_list, error_message)
    """
    if not IB_INSYNC_AVAILABLE:
        return [], "ib_insync not installed. Run: pip install ib_insync"

    try:
        client = IBKRClient(host=host, port=port)
        if not client.connect(timeout=5):
            return [], "Failed to connect to TWS/Gateway"

        accounts = client.get_managed_accounts()
        client.disconnect()

        if not accounts:
            return [], "Connected but no accounts found"

        return accounts, None

    except Exception as e:
        return [], str(e)


def get_ibkr_portfolio(account_id: str, host: str = "127.0.0.1", port: int = 7496) -> Dict:
    """
    Quick function to get portfolio data.

    Returns:
        Dict with 'positions', 'summary', 'error' keys
    """
    if not IB_INSYNC_AVAILABLE:
        return {
            'positions': [],
            'summary': {},
            'error': "ib_insync not installed"
        }

    try:
        client = IBKRClient(host=host, port=port)
        if not client.connect(timeout=5):
            return {
                'positions': [],
                'summary': {},
                'error': "Failed to connect to TWS/Gateway"
            }

        positions = client.get_portfolio(account_id)
        summary = client.get_account_summary(account_id)
        client.disconnect()

        return {
            'positions': positions,
            'summary': summary,
            'error': None
        }

    except Exception as e:
        return {
            'positions': [],
            'summary': {},
            'error': str(e)
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing IBKR Client...")
    print(f"ib_insync available: {IB_INSYNC_AVAILABLE}")

    if IB_INSYNC_AVAILABLE:
        # Test connection
        accounts, error = get_ibkr_accounts()

        if error:
            print(f"Error: {error}")
        else:
            print(f"Found accounts: {accounts}")

            if accounts:
                # Test portfolio
                portfolio = get_ibkr_portfolio(accounts[0])
                print(f"Positions: {len(portfolio['positions'])}")
                print(f"Summary: {portfolio['summary']}")