
# dashboard/ai_pm/stock_lookup.py
"""
IBKR Stock Lookup Utility
- Searches existing JSON files for stock data
- Fetches from IBKR if not found
- Provides full IBKR contract attributes
"""

import os
import json
import glob
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class IBKRStock:
    """Full IBKR stock attributes"""
    symbol: str
    name: str = ""
    originalName: str = ""
    similarity: float = 100.0
    minMultiplier: int = 1
    isin: str = ""
    sector: str = ""
    country: str = "USA"
    weight: float = 0.0
    conid: int = 0
    secType: str = "STK"
    currency: str = "USD"
    exchange: str = "SMART"
    primary_exchange: str = ""
    found_in_file: str = ""
    found_in_json: bool = False
    input_index: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class IBKRStockLookup:
    """Lookup IBKR stock data from JSON files or IBKR API"""
    
    def __init__(self, json_dirs: List[str] = None):
        """
        Initialize with directories to search for JSON files
        
        Args:
            json_dirs: List of directories containing IBKR JSON files
        """
        self.json_dirs = json_dirs or ['json', 'json/IBKR', './json']
        self._cache: Dict[str, IBKRStock] = {}
        self._loaded_files: List[str] = []
        
    def load_all_json_files(self) -> int:
        """Load all JSON files from configured directories into cache"""
        count = 0
        for dir_path in self.json_dirs:
            if not os.path.exists(dir_path):
                continue
            
            # Find all JSON files
            patterns = [
                os.path.join(dir_path, '*.json'),
                os.path.join(dir_path, '*_IBKR.json'),
            ]
            
            for pattern in patterns:
                for json_file in glob.glob(pattern):
                    if json_file in self._loaded_files:
                        continue
                    try:
                        count += self._load_json_file(json_file)
                        self._loaded_files.append(json_file)
                    except Exception as e:
                        logger.warning(f"Error loading {json_file}: {e}")
        
        logger.info(f"Loaded {count} stocks from {len(self._loaded_files)} JSON files")
        return count
    
    def _load_json_file(self, filepath: str) -> int:
        """Load a single JSON file into cache"""
        count = 0
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return 0  # Skip empty files
                data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.debug(f"Skipping invalid JSON {filepath}: {e}")
            return 0
        except Exception as e:
            logger.debug(f"Error reading {filepath}: {e}")
            return 0
        
        if isinstance(data, list):
            for item in data:
                stock = self._parse_stock_item(item, filepath)
                if stock and stock.symbol:
                    # Only cache stocks with valid ISIN (real IBKR data)
                    if stock.isin and len(stock.isin) >= 10:
                        self._cache[stock.symbol.upper()] = stock
                        count += 1
        elif isinstance(data, dict) and 'holdings' in data:
            for item in data['holdings']:
                stock = self._parse_stock_item(item, filepath)
                if stock and stock.symbol:
                    # Only cache stocks with valid ISIN (real IBKR data)
                    if stock.isin and len(stock.isin) >= 10:
                        self._cache[stock.symbol.upper()] = stock
                        count += 1
        
        return count
    
    def _parse_stock_item(self, item: Dict, source_file: str) -> Optional[IBKRStock]:
        """Parse a stock item from JSON into IBKRStock"""
        symbol = item.get('symbol') or item.get('ticker') or item.get('Symbol')
        if not symbol:
            return None
        
        # Safe conversion helpers for None values
        def safe_float(val, default=0.0):
            if val is None:
                return default
            try:
                return float(val)
            except (TypeError, ValueError):
                return default
        
        def safe_int(val, default=0):
            if val is None:
                return default
            try:
                return int(val)
            except (TypeError, ValueError):
                return default
        
        def safe_str(val, default=''):
            return str(val) if val is not None else default
        
        return IBKRStock(
            symbol=symbol.upper(),
            name=safe_str(item.get('name') or item.get('Name'), ''),
            originalName=safe_str(item.get('originalName') or item.get('original_name'), ''),
            similarity=safe_float(item.get('similarity'), 100.0),
            minMultiplier=safe_int(item.get('minMultiplier'), 1),
            isin=safe_str(item.get('isin') or item.get('ISIN'), ''),
            sector=safe_str(item.get('sector') or item.get('Sector'), ''),
            country=safe_str(item.get('country'), 'USA'),
            weight=safe_float(item.get('weight'), 0.0),
            conid=safe_int(item.get('conid') or item.get('conId'), 0),
            secType=safe_str(item.get('secType'), 'STK'),
            currency=safe_str(item.get('currency'), 'USD'),
            exchange=safe_str(item.get('exchange'), 'SMART'),
            primary_exchange=safe_str(item.get('primary_exchange') or item.get('primaryExchange'), ''),
            found_in_file=source_file,
            found_in_json=True,
        )
    
    def lookup(self, symbol: str) -> Optional[IBKRStock]:
        """
        Look up stock by symbol
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            IBKRStock if found, None otherwise
        """
        symbol = symbol.upper().strip()
        
        # Check cache first
        if symbol in self._cache:
            return self._cache[symbol]
        
        # Load all files if cache is empty
        if not self._cache:
            self.load_all_json_files()
        
        return self._cache.get(symbol)
    
    def lookup_multiple(self, symbols: List[str]) -> Dict[str, Optional[IBKRStock]]:
        """Look up multiple symbols at once"""
        results = {}
        for symbol in symbols:
            results[symbol.upper()] = self.lookup(symbol)
        return results
    
    def fetch_from_ibkr(self, symbol: str, ib_connection=None) -> Optional[IBKRStock]:
        """
        Fetch stock data from IBKR if not in cache
        
        Args:
            symbol: Stock ticker
            ib_connection: Optional existing IB connection
            
        Returns:
            IBKRStock with IBKR data, or None if not found
        """
        try:
            from ib_insync import IB, Stock
            import random
            
            # Use provided connection or create new one
            ib = ib_connection
            should_disconnect = False
            
            if ib is None or not ib.isConnected():
                ib = IB()
                client_id = random.randint(1000, 9999)
                ib.connect('127.0.0.1', 7496, clientId=client_id, timeout=10)
                should_disconnect = True
            
            try:
                # Create and qualify contract
                contract = Stock(symbol.upper(), 'SMART', 'USD')
                qualified = ib.qualifyContracts(contract)
                
                if qualified:
                    q = qualified[0]
                    stock = IBKRStock(
                        symbol=q.symbol,
                        name=q.localSymbol or q.symbol,
                        originalName=q.symbol,
                        similarity=100.0,
                        conid=q.conId,
                        secType=q.secType,
                        currency=q.currency,
                        exchange=q.exchange,
                        primary_exchange=q.primaryExchange,
                        found_in_json=False,
                    )
                    
                    # Cache the result
                    self._cache[symbol.upper()] = stock
                    return stock
                    
            finally:
                if should_disconnect and ib.isConnected():
                    ib.disconnect()
                    
        except Exception as e:
            logger.error(f"Error fetching {symbol} from IBKR: {e}")
        
        return None
    
    def get_or_fetch(self, symbol: str, ib_connection=None) -> Optional[IBKRStock]:
        """
        Get stock from cache or fetch from IBKR
        
        Args:
            symbol: Stock ticker
            ib_connection: Optional IB connection for fetching
            
        Returns:
            IBKRStock or None
        """
        # Try cache first
        stock = self.lookup(symbol)
        if stock:
            return stock
        
        # Fetch from IBKR
        return self.fetch_from_ibkr(symbol, ib_connection)


# Global instance
_stock_lookup = None

def get_stock_lookup() -> IBKRStockLookup:
    """Get the global stock lookup instance"""
    global _stock_lookup
    if _stock_lookup is None:
        _stock_lookup = IBKRStockLookup()
    return _stock_lookup


def lookup_stock(symbol: str) -> Optional[Dict]:
    """Convenience function to lookup a stock"""
    stock = get_stock_lookup().lookup(symbol)
    return stock.to_dict() if stock else None


def lookup_or_fetch_stock(symbol: str, ib_connection=None) -> Optional[Dict]:
    """Convenience function to lookup or fetch a stock"""
    stock = get_stock_lookup().get_or_fetch(symbol, ib_connection)
    return stock.to_dict() if stock else None
