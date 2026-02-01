
# dashboard/ai_pm/json_portfolio_editor.py
"""
JSON Portfolio Editor
- Load portfolio JSON files
- Replace stocks with full IBKR attributes
- Save updated portfolio JSON
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from copy import deepcopy

logger = logging.getLogger(__name__)


class JSONPortfolioEditor:
    """Edit portfolio JSON files with stock replacements"""
    
    def __init__(self, json_path: str = None):
        self.json_path = json_path
        self.original_data: List[Dict] = []
        self.modified_data: List[Dict] = []
        self.replacements: List[Dict] = []  # Track all replacements made
        
    def load(self, json_path: str = None) -> bool:
        """Load portfolio JSON file"""
        path = json_path or self.json_path
        if not path:
            logger.error("No JSON path provided")
            return False
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                self.original_data = data
            elif isinstance(data, dict) and 'holdings' in data:
                self.original_data = data['holdings']
            else:
                logger.error(f"Unknown JSON structure in {path}")
                return False
            
            self.modified_data = deepcopy(self.original_data)
            self.json_path = path
            self.replacements = []
            
            logger.info(f"Loaded {len(self.original_data)} holdings from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            return False
    
    def get_holding_by_symbol(self, symbol: str) -> Optional[Dict]:
        """Get a holding by symbol"""
        symbol = symbol.upper().strip()
        for holding in self.modified_data:
            h_sym = (holding.get('symbol') or holding.get('ticker') or '').upper()
            if h_sym == symbol:
                return holding
        return None
    
    def get_holding_index(self, symbol: str) -> int:
        """Get index of holding by symbol, returns -1 if not found"""
        symbol = symbol.upper().strip()
        for i, holding in enumerate(self.modified_data):
            h_sym = (holding.get('symbol') or holding.get('ticker') or '').upper()
            if h_sym == symbol:
                return i
        return -1
    
    def replace_stock(
        self, 
        old_symbol: str, 
        new_stock_data: Dict,
        transfer_weight: bool = True
    ) -> Tuple[bool, str]:
        """
        Replace a stock in the portfolio
        
        Args:
            old_symbol: Symbol to replace
            new_stock_data: Full IBKR stock data for replacement
            transfer_weight: If True, transfer weight from old to new stock
            
        Returns:
            Tuple of (success, message)
        """
        old_symbol = old_symbol.upper().strip()
        new_symbol = (new_stock_data.get('symbol') or '').upper().strip()
        
        if not new_symbol:
            return False, "New stock data missing symbol"
        
        # Find old stock
        old_index = self.get_holding_index(old_symbol)
        if old_index == -1:
            return False, f"Stock {old_symbol} not found in portfolio"
        
        old_holding = self.modified_data[old_index]
        old_weight = float(old_holding.get('weight', 0))
        
        # Check if new stock already exists
        new_index = self.get_holding_index(new_symbol)
        
        if new_index != -1 and new_index != old_index:
            # New stock exists - add weight to it
            if transfer_weight:
                self.modified_data[new_index]['weight'] = (
                    float(self.modified_data[new_index].get('weight', 0)) + old_weight
                )
            # Remove old stock
            self.modified_data.pop(old_index)
            
            self.replacements.append({
                'action': 'merged',
                'old_symbol': old_symbol,
                'new_symbol': new_symbol,
                'weight_transferred': old_weight,
                'timestamp': datetime.now().isoformat(),
            })
            
            return True, f"Merged {old_symbol} into existing {new_symbol} (weight: {old_weight:.2f}%)"
        
        else:
            # Replace old stock with new stock data
            new_holding = {
                'name': new_stock_data.get('name', new_symbol),
                'originalName': new_stock_data.get('originalName', new_stock_data.get('name', new_symbol)),
                'similarity': new_stock_data.get('similarity', 100.0),
                'symbol': new_symbol,
                'minMultiplier': new_stock_data.get('minMultiplier', 1),
                'isin': new_stock_data.get('isin', ''),
                'sector': new_stock_data.get('sector', old_holding.get('sector', '')),
                'country': new_stock_data.get('country', 'USA'),
                'weight': old_weight if transfer_weight else float(new_stock_data.get('weight', 0)),
                'conid': new_stock_data.get('conid', 0),
                'secType': new_stock_data.get('secType', 'STK'),
                'currency': new_stock_data.get('currency', 'USD'),
                'exchange': new_stock_data.get('exchange', 'SMART'),
                'primary_exchange': new_stock_data.get('primary_exchange', ''),
            }
            
            # Preserve any extra fields from original
            for key in old_holding:
                if key not in new_holding:
                    new_holding[key] = old_holding[key]
            
            self.modified_data[old_index] = new_holding
            
            self.replacements.append({
                'action': 'replaced',
                'old_symbol': old_symbol,
                'new_symbol': new_symbol,
                'weight': old_weight if transfer_weight else new_holding['weight'],
                'timestamp': datetime.now().isoformat(),
            })
            
            return True, f"Replaced {old_symbol} with {new_symbol} (weight: {new_holding['weight']:.2f}%)"
    
    def remove_stock(self, symbol: str) -> Tuple[bool, str]:
        """Remove a stock from portfolio"""
        symbol = symbol.upper().strip()
        index = self.get_holding_index(symbol)
        
        if index == -1:
            return False, f"Stock {symbol} not found"
        
        removed = self.modified_data.pop(index)
        weight = float(removed.get('weight', 0))
        
        self.replacements.append({
            'action': 'removed',
            'old_symbol': symbol,
            'weight_removed': weight,
            'timestamp': datetime.now().isoformat(),
        })
        
        return True, f"Removed {symbol} (weight: {weight:.2f}%)"
    
    def add_stock(self, stock_data: Dict, weight: float = None) -> Tuple[bool, str]:
        """Add a new stock to portfolio"""
        symbol = (stock_data.get('symbol') or '').upper().strip()
        if not symbol:
            return False, "Stock data missing symbol"
        
        # Check if already exists
        if self.get_holding_index(symbol) != -1:
            return False, f"Stock {symbol} already exists in portfolio"
        
        new_holding = {
            'name': stock_data.get('name', symbol),
            'originalName': stock_data.get('originalName', stock_data.get('name', symbol)),
            'similarity': stock_data.get('similarity', 100.0),
            'symbol': symbol,
            'minMultiplier': stock_data.get('minMultiplier', 1),
            'isin': stock_data.get('isin', ''),
            'sector': stock_data.get('sector', ''),
            'country': stock_data.get('country', 'USA'),
            'weight': weight if weight is not None else float(stock_data.get('weight', 0)),
            'conid': stock_data.get('conid', 0),
            'secType': stock_data.get('secType', 'STK'),
            'currency': stock_data.get('currency', 'USD'),
            'exchange': stock_data.get('exchange', 'SMART'),
            'primary_exchange': stock_data.get('primary_exchange', ''),
        }
        
        self.modified_data.append(new_holding)
        
        self.replacements.append({
            'action': 'added',
            'new_symbol': symbol,
            'weight': new_holding['weight'],
            'timestamp': datetime.now().isoformat(),
        })
        
        return True, f"Added {symbol} (weight: {new_holding['weight']:.2f}%)"
    
    def normalize_weights(self, target_sum: float = 100.0) -> None:
        """Normalize all weights to sum to target"""
        current_sum = sum(float(h.get('weight', 0)) for h in self.modified_data)
        if current_sum <= 0:
            return
        
        factor = target_sum / current_sum
        for holding in self.modified_data:
            holding['weight'] = round(float(holding.get('weight', 0)) * factor, 4)
    
    def get_summary(self) -> Dict:
        """Get summary of current portfolio state"""
        total_weight = sum(float(h.get('weight', 0)) for h in self.modified_data)
        sectors = {}
        for h in self.modified_data:
            sector = h.get('sector', 'Unknown')
            sectors[sector] = sectors.get(sector, 0) + float(h.get('weight', 0))
        
        return {
            'total_holdings': len(self.modified_data),
            'total_weight': total_weight,
            'sectors': sectors,
            'replacements_made': len(self.replacements),
        }
    
    def save(self, output_path: str = None, add_timestamp: bool = True) -> Tuple[bool, str]:
        """
        Save modified portfolio to JSON
        
        Args:
            output_path: Path to save to (default: original path with _modified suffix)
            add_timestamp: Add timestamp to filename
            
        Returns:
            Tuple of (success, saved_path or error message)
        """
        if output_path is None:
            base, ext = os.path.splitext(self.json_path)
            if add_timestamp:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"{base}_modified_{timestamp}{ext}"
            else:
                output_path = f"{base}_modified{ext}"
        
        try:
            # Sort by weight descending
            sorted_data = sorted(
                self.modified_data, 
                key=lambda x: float(x.get('weight', 0)), 
                reverse=True
            )
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sorted_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Saved modified portfolio to {output_path}")
            return True, output_path
            
        except Exception as e:
            logger.error(f"Error saving JSON: {e}")
            return False, str(e)
    
    def get_changes_report(self) -> str:
        """Get a human-readable report of all changes"""
        if not self.replacements:
            return "No changes made."
        
        lines = [f"Portfolio Changes ({len(self.replacements)} modifications):"]
        lines.append("-" * 50)
        
        for i, r in enumerate(self.replacements, 1):
            action = r['action']
            if action == 'replaced':
                lines.append(f"{i}. REPLACED: {r['old_symbol']} -> {r['new_symbol']} (weight: {r['weight']:.2f}%)")
            elif action == 'merged':
                lines.append(f"{i}. MERGED: {r['old_symbol']} into {r['new_symbol']} (+{r['weight_transferred']:.2f}%)")
            elif action == 'removed':
                lines.append(f"{i}. REMOVED: {r['old_symbol']} (was {r['weight_removed']:.2f}%)")
            elif action == 'added':
                lines.append(f"{i}. ADDED: {r['new_symbol']} ({r['weight']:.2f}%)")
        
        return "\n".join(lines)


# Convenience functions
def load_portfolio_json(json_path: str) -> JSONPortfolioEditor:
    """Load a portfolio JSON and return editor"""
    editor = JSONPortfolioEditor()
    editor.load(json_path)
    return editor
