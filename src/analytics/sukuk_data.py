"""
HH Research Platform - Sukuk IBKR Data Layer

Fetches live market data for sukuk bonds from IBKR.
Uses conid-based lookup for bonds (secType=BOND).

Based on patterns from get_ibkr_data_from_ISIN_Symbol_Name_v5.py
"""

import math
import time
import random
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

# Apply nest_asyncio at import time
import nest_asyncio
nest_asyncio.apply()

import ib_insync
from ib_insync import IB, Contract

from src.utils.logging import get_logger

# Import models
try:
    from src.analytics.sukuk_models import (
        SukukInstrument, SukukUniverse, Quote, SukukLiveData,
        DataQuality, RiskLimits, DEFAULT_ISSUER_MAPPING
    )
except ImportError:
    from sukuk_models import (
        SukukInstrument, SukukUniverse, Quote, SukukLiveData,
        DataQuality, RiskLimits, DEFAULT_ISSUER_MAPPING
    )

logger = get_logger(__name__)


class SukukIBKRClient:
    """
    IBKR client for fetching sukuk bond market data.

    Uses conid-based bond contracts with fallback to historical data.
    Based on patterns from get_ibkr_data_from_ISIN_Symbol_Name_v5.py
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7496,
                 client_id: Optional[int] = None, timeout: int = 15):
        """
        Initialize IBKR client for sukuk data.

        Args:
            host: TWS/Gateway host
            port: TWS port (7496) or Gateway port (4001)
            client_id: Unique client ID (random if not specified)
            timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.ib = IB()
        self._connected = False

    def connect(self, retries: int = 3) -> bool:
        """
        Connect to IBKR with retry logic.

        Args:
            retries: Number of connection attempts

        Returns:
            True if connected successfully
        """
        retry_delay = 2

        for attempt in range(retries):
            try:
                if not self.ib.isConnected():
                    # Generate random client ID to avoid conflicts
                    cid = self.client_id or random.randint(1, 9999)
                    logger.info(f"Connecting to IBKR on {self.host}:{self.port} with Client ID {cid}")

                    self.ib.connect(self.host, self.port, clientId=cid, timeout=self.timeout)

                    # Wait a bit to ensure connection is established
                    time.sleep(1)

                    if self.ib.isConnected():
                        logger.info("Successfully connected to IBKR")
                        self._connected = True
                        return True
                else:
                    logger.info("Already connected to IBKR")
                    self._connected = True
                    return True

            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2

        logger.error(f"Failed to connect to IBKR after {retries} attempts")
        self._connected = False
        return False

    def disconnect(self):
        """Disconnect from IBKR."""
        try:
            if self.ib.isConnected():
                logger.info("Disconnecting from IBKR")
                self.ib.disconnect()
                time.sleep(1)  # Give it time to disconnect properly
        except Exception as e:
            logger.error(f"Error during disconnection: {e}")
        self._connected = False

    def is_connected(self) -> bool:
        """Check if connected."""
        return self.ib.isConnected()

    def _create_bond_contract(self, conid: int, currency: str = "USD",
                               exchange: str = "SMART") -> Optional[Contract]:
        """
        Create IBKR Contract for a bond by conid.

        Args:
            conid: IBKR contract ID
            currency: Bond currency
            exchange: Exchange (usually SMART)

        Returns:
            Qualified Contract or None
        """
        try:
            contract = Contract(
                conId=conid,
                secType="BOND",
                exchange=exchange,
                currency=currency
            )

            # Qualify the contract to get full details
            qualified = self.ib.qualifyContracts(contract)
            ib_insync.util.sleep(0.3)

            if qualified and len(qualified) > 0:
                return qualified[0]
            else:
                logger.warning(f"Failed to qualify bond contract conid={conid}")
                return None

        except Exception as e:
            logger.error(f"Error creating bond contract conid={conid}: {e}")
            return None

    def get_quote(self, conid: int, currency: str = "USD",
                  exchange: str = "SMART") -> Quote:
        """
        Get live quote for a bond by conid.

        Uses reqTickers with fallback to historical data.
        Pattern from get_ibkr_data_from_ISIN_Symbol_Name_v5.py

        Args:
            conid: IBKR contract ID
            currency: Bond currency
            exchange: Exchange

        Returns:
            Quote object (may have None values if data unavailable)
        """
        if not self.is_connected():
            logger.warning("Not connected to IBKR")
            return Quote(source="IBKR_NOT_CONNECTED")

        contract = self._create_bond_contract(conid, currency, exchange)
        if not contract:
            return Quote(source="IBKR_CONTRACT_ERROR")

        try:
            # Use ib_insync.util.sleep like the working script
            ib_insync.util.sleep(0.5)

            # Request ticker
            tickers = self.ib.reqTickers(contract)

            if tickers and len(tickers) > 0:
                ticker = tickers[0]

                # Get prices - check for NaN
                bid = ticker.bid if ticker.bid and not math.isnan(ticker.bid) else None
                ask = ticker.ask if ticker.ask and not math.isnan(ticker.ask) else None
                last = ticker.last if ticker.last and not math.isnan(ticker.last) else None
                close = ticker.close if ticker.close and not math.isnan(ticker.close) else None

                # If last is NaN or 0, try historical fallback
                if last is None or last == 0:
                    historical_close = self._get_historical_close(contract)
                    if historical_close:
                        close = historical_close
                        logger.debug(f"Using historical close for conid={conid}: {close}")

                return Quote(
                    bid=bid,
                    ask=ask,
                    last=last,
                    close=close,
                    asof_ts=datetime.now(),
                    source="IBKR"
                )
            else:
                # No ticker data - try historical
                historical_close = self._get_historical_close(contract)
                if historical_close:
                    return Quote(
                        close=historical_close,
                        asof_ts=datetime.now(),
                        source="IBKR_HISTORICAL"
                    )

        except Exception as e:
            logger.error(f"Error fetching quote for conid={conid}: {e}")

        return Quote(source="IBKR_ERROR")

    def _get_historical_close(self, contract: Contract, days: int = 1) -> Optional[float]:
        """
        Get historical close price as fallback.
        Pattern from get_ibkr_data_from_ISIN_Symbol_Name_v5.py

        Args:
            contract: Qualified IBKR contract
            days: Number of days to look back

        Returns:
            Close price or None
        """
        try:
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=f'{days} D',
                barSizeSetting='1 day',
                whatToShow='MIDPOINT',
                useRTH=True,
                formatDate=1
            )

            if bars and len(bars) > 0:
                return float(bars[-1].close)

        except Exception as e:
            logger.debug(f"Historical data fallback failed: {e}")

        return None

    def get_sukuk_live_data(self, instrument: SukukInstrument) -> SukukLiveData:
        """
        Get live data for a sukuk instrument.

        Args:
            instrument: SukukInstrument definition

        Returns:
            SukukLiveData with quote and validation
        """
        quote = self.get_quote(
            conid=instrument.conid,
            currency=instrument.currency,
            exchange=instrument.exchange
        )

        live_data = SukukLiveData(
            instrument=instrument,
            quote=quote
        )

        # Run validation
        live_data.validate()

        return live_data

    def get_all_sukuk_data(self, universe: SukukUniverse,
                          delay_between: float = 0.2) -> Dict[int, SukukLiveData]:
        """
        Fetch live data for all sukuk in universe.

        Args:
            universe: SukukUniverse with instruments
            delay_between: Delay between requests (rate limiting)

        Returns:
            Dict mapping conid -> SukukLiveData
        """
        results: Dict[int, SukukLiveData] = {}

        active = universe.active_instruments
        logger.info(f"Fetching data for {len(active)} active sukuk")

        for i, instrument in enumerate(active):
            try:
                live_data = self.get_sukuk_live_data(instrument)
                results[instrument.conid] = live_data

                # Log progress
                price = live_data.price_pct
                spread = live_data.bid_ask_bps
                quality = live_data.data_quality.value

                price_str = f"${price:.2f}" if price else "N/A"
                spread_str = f"{spread:.0f}bps" if spread else "N/A"

                logger.info(f"[{i+1}/{len(active)}] {instrument.name}: {price_str} | {spread_str} | {quality}")

                if delay_between > 0:
                    ib_insync.util.sleep(delay_between)

            except Exception as e:
                logger.error(f"Error fetching {instrument.name}: {e}")
                # Create degraded entry with cached data
                results[instrument.conid] = SukukLiveData.from_cached(instrument)
                results[instrument.conid].warnings.append(str(e))

        return results


def load_sukuk_universe(json_path: str) -> SukukUniverse:
    """
    Load sukuk universe from JSON file.

    Supports two formats:
    1. Full universe JSON with metadata, risk_limits, issuer_mapping, sukuk
    2. Direct HYSK-holdings_ibkr.json format (list of sukuk)

    Args:
        json_path: Path to JSON file

    Returns:
        SukukUniverse object
    """
    import json

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Check if it's a list (direct HYSK format) or dict (universe format)
    if isinstance(data, list):
        # Direct HYSK format - list of sukuk
        return SukukUniverse.from_hysk_json(data)
    else:
        # Full universe format with metadata
        return SukukUniverse.from_json(data)


def fetch_sukuk_market_data(universe: SukukUniverse,
                            host: str = "127.0.0.1",
                            port: int = 7496,
                            use_cached_fallback: bool = True) -> Tuple[Dict[int, SukukLiveData], List[str]]:
    """
    High-level function to fetch all sukuk market data.

    Args:
        universe: SukukUniverse with instruments
        host: IBKR host
        port: IBKR port
        use_cached_fallback: If True, use cached prices from JSON when IBKR unavailable

    Returns:
        Tuple of (data dict, warnings list)
    """
    warnings = []
    data = {}

    client = SukukIBKRClient(host=host, port=port)

    try:
        if not client.connect():
            warnings.append("Failed to connect to IBKR")

            # Use cached data if available
            if use_cached_fallback:
                warnings.append("Using cached prices from JSON")
                for instrument in universe.active_instruments:
                    data[instrument.conid] = SukukLiveData.from_cached(instrument)
                return data, warnings
            else:
                return data, warnings

        data = client.get_all_sukuk_data(universe)

        # Summarize quality
        ok_count = sum(1 for d in data.values() if d.data_quality == DataQuality.OK)
        degraded_count = sum(1 for d in data.values() if d.data_quality in [
            DataQuality.DEGRADED, DataQuality.STALE, DataQuality.SUSPICIOUS
        ])
        missing_count = sum(1 for d in data.values() if d.data_quality == DataQuality.MISSING)

        logger.info(f"Sukuk data fetch complete: {ok_count} OK, {degraded_count} degraded, {missing_count} missing")

        if missing_count > 0:
            warnings.append(f"{missing_count} sukuk with missing data")
        if degraded_count > 0:
            warnings.append(f"{degraded_count} sukuk with degraded data quality")

    except Exception as e:
        warnings.append(f"Data fetch error: {e}")
        logger.error(f"Sukuk data fetch error: {e}")

        # Use cached data on error
        if use_cached_fallback:
            warnings.append("Using cached prices from JSON")
            for instrument in universe.active_instruments:
                if instrument.conid not in data:
                    data[instrument.conid] = SukukLiveData.from_cached(instrument)

    finally:
        client.disconnect()

    return data, warnings