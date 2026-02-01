"""
Centralized Price Fetching Utility
Yahoo Finance FIRST (fast), IBKR fallback (slow but real-time)

Price Types:
- 'live': Current/real-time price (pre/post market aware, falls back to close)
- 'close': Yesterday's close price (for calculations, faster batch)
- 'auto': Live during market hours, close otherwise

Usage:
    from src.utils.price_fetcher import get_prices, get_price
    
    # Live price (real-time, includes pre/post market)
    price = get_price('AAPL', price_type='live')
    
    # Close price (yesterday's close - for backtesting/calculations)
    price = get_price('AAPL', price_type='close')
    
    # Batch prices (much faster for close)
    prices = get_prices(['AAPL', 'MSFT', 'GOOGL'], price_type='live')
"""
import logging
from typing import Dict, List, Optional, Literal
from datetime import datetime

logger = logging.getLogger(__name__)

# Module-level cache
_price_cache: Dict[str, tuple] = {}  # key -> (price, timestamp)
_CACHE_TTL_LIVE = 60  # 1 minute for live prices
_CACHE_TTL_CLOSE = 300  # 5 minutes for close prices


def is_market_open() -> bool:
    """Check if US stock market is currently open."""
    try:
        import pytz
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        
        # Weekend check
        if now.weekday() >= 5:
            return False
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    except:
        return False


def get_prices(
    symbols: List[str],
    price_type: Literal['live', 'close', 'auto'] = 'live',
    use_cache: bool = True,
    ib=None,
) -> Dict[str, float]:
    """
    Get prices for multiple symbols. Yahoo Finance FIRST, IBKR fallback.
    
    Args:
        symbols: List of ticker symbols
        price_type: 'live' (real-time), 'close' (yesterday), or 'auto'
        use_cache: Whether to use cached prices
        ib: Optional ib_insync IB connection for fallback
        
    Returns:
        Dict mapping symbol to price
    """
    if not symbols:
        return {}
    
    syms = [s.strip().upper() for s in symbols if s]
    syms = list(set(syms))
    
    # Determine actual price type for 'auto'
    actual_type = price_type
    if price_type == 'auto':
        actual_type = 'live' if is_market_open() else 'close'
    
    result = {}
    need_fetch = []
    now = datetime.now()
    
    # Cache TTL depends on price type
    cache_ttl = _CACHE_TTL_LIVE if actual_type == 'live' else _CACHE_TTL_CLOSE
    
    # Check cache first
    if use_cache:
        for sym in syms:
            cache_key = f"{sym}_{actual_type}"
            if cache_key in _price_cache:
                price, ts = _price_cache[cache_key]
                if (now - ts).total_seconds() < cache_ttl:
                    result[sym] = price
                else:
                    need_fetch.append(sym)
            else:
                need_fetch.append(sym)
    else:
        need_fetch = syms
    
    if not need_fetch:
        return result
    
    # Fetch prices based on type
    if actual_type == 'live':
        fetched = _fetch_yahoo_live(need_fetch)
    else:
        fetched = _fetch_yahoo_close(need_fetch)
    
    # Cache and add to result
    for sym, price in fetched.items():
        result[sym] = price
        cache_key = f"{sym}_{actual_type}"
        _price_cache[cache_key] = (price, now)
    
    # FALLBACK: IBKR for missing (only if few and ib available)
    still_missing = [s for s in need_fetch if s not in result]
    
    if still_missing and ib and len(still_missing) <= 20:
        ibkr_prices = _fetch_ibkr_batch(still_missing, ib)
        for sym, price in ibkr_prices.items():
            result[sym] = price
            cache_key = f"{sym}_{actual_type}"
            _price_cache[cache_key] = (price, now)
    
    return result


def get_price(
    symbol: str,
    price_type: Literal['live', 'close', 'auto'] = 'live',
    use_cache: bool = True,
    ib=None,
) -> Optional[float]:
    """Get price for a single symbol."""
    prices = get_prices([symbol], price_type=price_type, use_cache=use_cache, ib=ib)
    return prices.get(symbol.upper())


def _fetch_yahoo_live(symbols: List[str]) -> Dict[str, float]:
    """
    Fetch LIVE/real-time prices from Yahoo Finance.
    
    Price priority:
    1. regularMarketPrice (current trading price)
    2. preMarketPrice (if pre-market and available)
    3. postMarketPrice (if after-hours and available)  
    4. previousClose (always available - fallback)
    
    Always returns a valid price (falls back to previousClose if needed).
    """
    result = {}
    
    if not symbols:
        return result
    
    try:
        import yfinance as yf
        
        for sym in symbols:
            try:
                ticker = yf.Ticker(sym)
                price = None
                
                # Try fast_info first (faster)
                try:
                    fi = ticker.fast_info
                    for field in ['lastPrice', 'last_price', 'regularMarketPrice', 'previousClose', 'previous_close']:
                        val = fi.get(field)
                        if val and val > 0:
                            price = float(val)
                            break
                    
                    if price:
                        result[sym] = price
                        continue
                except:
                    pass
                
                # Fallback to info (slower but more reliable)
                try:
                    info = ticker.info
                    
                    price_fields = [
                        'regularMarketPrice',
                        'currentPrice',
                        'preMarketPrice',
                        'postMarketPrice',
                        'regularMarketPreviousClose',
                        'previousClose',
                    ]
                    
                    for field in price_fields:
                        val = info.get(field)
                        if val is not None:
                            try:
                                val_float = float(val)
                                if val_float > 0:
                                    price = val_float
                                    break
                            except (ValueError, TypeError):
                                continue
                    
                    if price:
                        result[sym] = price
                except:
                    pass
                    
            except Exception as e:
                logger.debug(f"Yahoo live error for {sym}: {e}")
        
        # Final fallback: batch close for any still missing
        missing = [s for s in symbols if s not in result]
        if missing:
            close_prices = _fetch_yahoo_close(missing)
            for sym, px in close_prices.items():
                if sym not in result:
                    result[sym] = px
                
    except ImportError:
        logger.warning("yfinance not installed")
    except Exception as e:
        logger.warning(f"Yahoo Finance error: {e}")
    
    return result


def _fetch_yahoo_close(symbols: List[str]) -> Dict[str, float]:
    """
    Fetch CLOSE prices from Yahoo Finance (batch - faster).
    Uses yf.download for efficient batch fetching.
    """
    result = {}
    
    if not symbols:
        return result
    
    try:
        import yfinance as yf
        
        for i in range(0, len(symbols), 100):
            chunk = symbols[i:i+100]
            
            try:
                yf_data = yf.download(
                    ' '.join(chunk),
                    period='2d',
                    progress=False,
                    threads=True,
                )
                
                if yf_data.empty:
                    continue
                
                for sym in chunk:
                    try:
                        val = None
                        if ('Close', sym) in yf_data.columns:
                            val = yf_data[('Close', sym)].iloc[-1]
                        elif 'Close' in yf_data.columns and len(chunk) == 1:
                            close_col = yf_data['Close']
                            if hasattr(close_col, 'iloc'):
                                val = close_col.iloc[-1]
                        elif 'Close' in yf_data.columns:
                            close_df = yf_data['Close']
                            if hasattr(close_df, 'columns') and sym in close_df.columns:
                                val = close_df[sym].iloc[-1]
                        
                        if val is not None and float(val) > 0:
                            result[sym] = float(val)
                    except:
                        pass
            except Exception as e:
                logger.debug(f"Yahoo batch error: {e}")
                
    except ImportError:
        logger.warning("yfinance not installed")
    except Exception as e:
        logger.warning(f"Yahoo Finance error: {e}")
    
    return result


def _fetch_ibkr_batch(symbols: List[str], ib) -> Dict[str, float]:
    """Fetch prices from IBKR (fallback, slower but real-time)."""
    result = {}
    
    if not symbols or not ib:
        return result
    
    try:
        from ib_insync import Stock
        
        contracts = [Stock(sym, 'SMART', 'USD') for sym in symbols]
        
        try:
            qualified = ib.qualifyContracts(*contracts)
            tickers = ib.reqTickers(*qualified)
            
            for t in (tickers or []):
                try:
                    sym = (t.contract.symbol or "").upper()
                    price = None
                    
                    for attr in ('marketPrice', 'last', 'close'):
                        try:
                            val = getattr(t, attr, None)
                            if callable(val):
                                val = val()
                            if val and val > 0:
                                price = float(val)
                                break
                        except:
                            pass
                    
                    if price and price > 0:
                        result[sym] = price
                except:
                    pass
        except Exception as e:
            logger.debug(f"IBKR batch error: {e}")
            
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"IBKR error: {e}")
    
    return result


def clear_cache():
    """Clear the price cache."""
    global _price_cache
    _price_cache = {}


def get_cache_stats() -> Dict:
    """Get cache statistics."""
    now = datetime.now()
    live_valid = sum(1 for k, (_, ts) in _price_cache.items() 
                     if '_live' in k and (now - ts).total_seconds() < _CACHE_TTL_LIVE)
    close_valid = sum(1 for k, (_, ts) in _price_cache.items() 
                      if '_close' in k and (now - ts).total_seconds() < _CACHE_TTL_CLOSE)
    return {
        'total': len(_price_cache),
        'live_valid': live_valid,
        'close_valid': close_valid,
    }
