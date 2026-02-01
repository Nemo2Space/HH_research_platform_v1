# Add to src/broker/yahoo_prices.py (new file)

import yfinance as yf
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def get_batch_prices(symbols: List[str]) -> Dict[str, float]:
    """
    Fetch current prices for multiple symbols in ONE call.
    Much faster than IBKR individual requests.
    
    Args:
        symbols: List of stock tickers
        
    Returns:
        Dict of {symbol: price}
    """
    if not symbols:
        return {}
    
    prices = {}
    
    try:
        # Remove duplicates and clean
        clean_symbols = list(set([s.upper().strip() for s in symbols if s]))
        
        # Batch fetch - ONE API call for all symbols
        tickers = yf.Tickers(" ".join(clean_symbols))
        
        for symbol in clean_symbols:
            try:
                ticker = tickers.tickers.get(symbol)
                if ticker:
                    # Try fast info first
                    info = ticker.fast_info
                    price = getattr(info, 'last_price', None) or getattr(info, 'previous_close', None)
                    
                    if price and price > 0:
                        prices[symbol] = float(price)
                        
            except Exception as e:
                logger.debug(f"Could not get price for {symbol}: {e}")
                
    except Exception as e:
        logger.error(f"Batch price fetch failed: {e}")
    
    logger.info(f"Fetched {len(prices)} prices from Yahoo Finance")
    return prices


def update_positions_with_yahoo_prices(positions: List[Dict]) -> List[Dict]:
    """
    Update position market values using Yahoo Finance prices.
    
    Args:
        positions: List of position dicts (with 'symbol' and 'position' keys)
        
    Returns:
        Positions with updated marketValue
    """
    if not positions:
        return positions
    
    # Get all symbols
    symbols = [p.get('symbol', '') for p in positions if p.get('symbol')]
    
    # Batch fetch prices
    prices = get_batch_prices(symbols)
    
    # Update positions
    for pos in positions:
        symbol = pos.get('symbol', '')
        if symbol in prices:
            shares = pos.get('position', 0) or pos.get('shares', 0) or 0
            price = prices[symbol]
            pos['lastPrice'] = price
            pos['marketValue'] = shares * price
    
    return positions