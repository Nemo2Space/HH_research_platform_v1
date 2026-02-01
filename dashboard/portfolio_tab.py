"""
Portfolio Tab V7 - IBKR Activity Statement Integration
=======================================================

Features:
- IBKR Activity Statement CSV parsing (replaces FlexQuery)
- Performance analytics with waterfall chart
- True Profit calculation (NAV - Net Deposits)
- Income breakdown (Dividends, Interest, Withholding Tax)
- Per-stock dividend tracking
- Working privacy toggle
- IBKR live data integration
- AI Portfolio Builder
- Trading signals alignment

Author: HH Research Platform
"""

import csv
import glob
import io
import os
import re
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.ai_pm.ui_tab import render_ai_portfolio_manager_tab

# =============================================================================
# FIX IMPORT PATH
# =============================================================================
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Pairs/Correlation Analysis
try:
    from src.analytics.pairs_correlation import (
        analyze_portfolio_correlations,
        find_hedges,
        get_correlation_heatmap,
    )
    PAIRS_CORRELATION_AVAILABLE = True
except ImportError:
    PAIRS_CORRELATION_AVAILABLE = False

# Enhanced IBKR data with dividends
try:
    from src.broker.ibkr_enhanced import (
        fetch_dividends_from_yahoo,
        calculate_portfolio_dividend_income,
    )
    ENHANCED_IBKR_AVAILABLE = True
except ImportError:
    ENHANCED_IBKR_AVAILABLE = False

# AI Portfolio Builder
try:
    from dashboard.portfolio_builder import (
        get_latest_stock_universe,
        get_sector_summary,
        get_top_stocks_by_category,
        build_portfolio_ai_context,
        build_portfolio_instructions,
        get_ai_response,
        build_portfolio_from_intent,
        save_portfolio,
        get_saved_portfolios,
        load_portfolio,
        delete_portfolio,
        PORTFOLIO_TEMPLATES,
        PORTFOLIO_TEMPLATES,
    )
    PORTFOLIO_BUILDER_AVAILABLE = True
except ImportError:
    PORTFOLIO_BUILDER_AVAILABLE = False

# Enhanced Portfolio Display
try:
    from dashboard.enhanced_portfolio_display import (
        render_comprehensive_stock_table,
        render_save_portfolio_section
    )
    ENHANCED_DISPLAY_AVAILABLE = True
except ImportError:
    ENHANCED_DISPLAY_AVAILABLE = False

# Portfolio Backtesting
try:
    from dashboard.backtest_tab import render_backtest_tab
    BACKTEST_AVAILABLE = True
except ImportError:
    BACKTEST_AVAILABLE = False

# Portfolio Database
try:
    from src.db.portfolio_db import PortfolioRepository
    PORTFOLIO_DB_AVAILABLE = True
except ImportError:
    PORTFOLIO_DB_AVAILABLE = False


# =============================================================================
# IBKR ACTIVITY STATEMENT DATA CLASSES
# =============================================================================

@dataclass
class AccountInfo:
    """Account identification information."""
    name: str = ""
    account_id: str = ""
    account_type: str = ""
    base_currency: str = "USD"


@dataclass
class NAVBreakdown:
    """Net Asset Value breakdown by asset type."""
    cash: float = 0.0
    stock: float = 0.0
    bonds: float = 0.0
    options: float = 0.0
    interest_accruals: float = 0.0
    dividend_accruals: float = 0.0
    total: float = 0.0
    twr_percent: float = 0.0


@dataclass
class ChangeInNAV:
    """Change in NAV components over the statement period."""
    starting_value: float = 0.0
    ending_value: float = 0.0
    mark_to_market: float = 0.0
    deposits_withdrawals: float = 0.0
    dividends: float = 0.0
    withholding_tax: float = 0.0
    change_in_div_accruals: float = 0.0
    interest: float = 0.0
    change_in_interest_accruals: float = 0.0
    other_fees: float = 0.0
    commissions: float = 0.0


@dataclass
class OpenPosition:
    """An open position in the portfolio."""
    symbol: str = ""
    description: str = ""
    quantity: float = 0.0
    cost_basis: float = 0.0
    cost_price: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    asset_class: str = "Stocks"


@dataclass
class Trade:
    """A single trade record."""
    date: datetime = None
    symbol: str = ""
    description: str = ""
    quantity: float = 0.0
    price: float = 0.0
    proceeds: float = 0.0
    commission: float = 0.0
    realized_pnl: float = 0.0
    code: str = ""  # O=Open, C=Close
    asset_class: str = "Stocks"


@dataclass
class DividendRecord:
    """A dividend payment record."""
    date: datetime = None
    symbol: str = ""
    description: str = ""
    amount: float = 0.0


@dataclass
class InterestRecord:
    """An interest payment record."""
    date: datetime = None
    description: str = ""
    amount: float = 0.0


@dataclass
class WithholdingTaxRecord:
    """A withholding tax record."""
    date: datetime = None
    symbol: str = ""
    description: str = ""
    amount: float = 0.0


@dataclass
class DepositWithdrawal:
    """A deposit or withdrawal record."""
    date: datetime = None
    description: str = ""
    amount: float = 0.0


@dataclass
class FeeRecord:
    """A fee record."""
    date: datetime = None
    description: str = ""
    amount: float = 0.0


@dataclass
class MTMPerformance:
    """Mark-to-Market performance for a single symbol."""
    symbol: str = ""
    description: str = ""
    prior_period_value: float = 0.0
    cost_basis: float = 0.0
    realized_st: float = 0.0
    realized_lt: float = 0.0
    unrealized_st: float = 0.0
    unrealized_lt: float = 0.0
    transfers: float = 0.0
    mtm_pnl: float = 0.0


@dataclass
class ParsedStatement:
    """Complete parsed IBKR Activity Statement."""
    account_info: AccountInfo = field(default_factory=AccountInfo)
    period_start: str = ""
    period_end: str = ""
    nav: NAVBreakdown = field(default_factory=NAVBreakdown)
    change_in_nav: ChangeInNAV = field(default_factory=ChangeInNAV)
    open_positions: List[OpenPosition] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    dividends: List[DividendRecord] = field(default_factory=list)
    interest: List[InterestRecord] = field(default_factory=list)
    withholding_tax: List[WithholdingTaxRecord] = field(default_factory=list)
    deposits_withdrawals: List[DepositWithdrawal] = field(default_factory=list)
    fees: List[FeeRecord] = field(default_factory=list)
    mtm_performance: List[MTMPerformance] = field(default_factory=list)
    parse_errors: List[str] = field(default_factory=list)


# =============================================================================
# IBKR ACTIVITY STATEMENT PARSER
# =============================================================================

class IBKRStatementParser:
    """
    Parser for IBKR Activity Statement CSV files.

    IBKR Activity Statements are section-based CSV files where:
    - First column identifies the section type
    - Second column is either 'Header', 'Data', 'SubTotal', or 'Total'
    - Remaining columns contain the data
    """

    def __init__(self):
        self.statement = ParsedStatement()
        self.current_section = None
        self.section_headers = {}

    def parse(self, content: str) -> ParsedStatement:
        """Parse IBKR Activity Statement CSV content."""
        self.statement = ParsedStatement()
        self.current_section = None
        self.section_headers = {}

        try:
            # Handle both string and file-like objects
            if hasattr(content, 'read'):
                content = content.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')

            reader = csv.reader(io.StringIO(content))

            for row_num, row in enumerate(reader, 1):
                try:
                    if len(row) >= 2:
                        self._process_row(row)
                except Exception as e:
                    self.statement.parse_errors.append(f"Row {row_num}: {str(e)}")

            # Post-processing
            self._calculate_position_returns()

        except Exception as e:
            self.statement.parse_errors.append(f"Parse error: {str(e)}")

        return self.statement

    def _process_row(self, row: List[str]):
        """Route row to appropriate section handler."""
        section = row[0].strip()
        row_type = row[1].strip() if len(row) > 1 else ""

        if row_type == "Header":
            self.current_section = section
            self.section_headers[section] = row[2:]
            return

        if section == "Statement":
            self._parse_statement_info(row)
        elif section == "Account Information":
            self._parse_account_info(row)
        elif section == "Net Asset Value":
            self._parse_nav(row, row_type)
        elif section == "Change in NAV":
            self._parse_change_in_nav(row, row_type)
        elif section == "Mark-to-Market Performance Summary":
            self._parse_mtm_performance(row, row_type)
        elif section == "Open Positions":
            self._parse_open_position(row, row_type)
        elif section == "Trades":
            self._parse_trade(row, row_type)
        elif section == "Dividends":
            self._parse_dividend(row, row_type)
        elif section == "Withholding Tax":
            self._parse_withholding_tax(row, row_type)
        elif section == "Interest":
            self._parse_interest(row, row_type)
        elif section == "Deposits & Withdrawals":
            self._parse_deposit_withdrawal(row, row_type)
        elif section == "Fees":
            self._parse_fee(row, row_type)

    def _parse_number(self, value: str, default: float = 0.0) -> float:
        """Parse a number from string, handling various formats."""
        if not value or value in ('--', '-', 'N/A', ''):
            return default
        try:
            # Remove quotes, commas, currency symbols
            cleaned = value.replace('"', '').replace(',', '').replace('$', '').strip()
            if cleaned in ('--', '-', 'N/A', ''):
                return default
            return float(cleaned)
        except (ValueError, TypeError):
            return default

    def _parse_date(self, value: str) -> Optional[datetime]:
        """Parse date from various formats."""
        if not value or value in ('--', '-', 'N/A', ''):
            return None
        try:
            # Try common formats
            for fmt in ['%Y-%m-%d', '%Y-%m-%d, %H:%M:%S', '%m/%d/%Y', '%d-%b-%Y']:
                try:
                    return datetime.strptime(value.strip(), fmt)
                except ValueError:
                    continue
            return None
        except Exception:
            return None

    def _parse_statement_info(self, row: List[str]):
        """Parse statement period information."""
        if len(row) >= 3:
            field_name = row[1].strip()
            value = row[2].strip() if len(row) > 2 else ""

            if field_name == "Period":
                # Format: "March 21, 2025 - January 14, 2026"
                if " - " in value:
                    parts = value.split(" - ")
                    self.statement.period_start = parts[0].strip()
                    self.statement.period_end = parts[1].strip() if len(parts) > 1 else ""

    def _parse_account_info(self, row: List[str]):
        """Parse account information."""
        if len(row) >= 3 and row[1].strip() == "Data":
            field_name = row[2].strip() if len(row) > 2 else ""
            value = row[3].strip() if len(row) > 3 else ""

            if field_name == "Name":
                self.statement.account_info.name = value
            elif field_name == "Account":
                self.statement.account_info.account_id = value
            elif field_name == "Account Type":
                self.statement.account_info.account_type = value
            elif field_name == "Base Currency":
                self.statement.account_info.base_currency = value

    def _parse_nav(self, row: List[str], row_type: str):
        """Parse Net Asset Value breakdown + TWR% (handles the 3-column TWR row)."""

        if row_type not in ("Data", "Total"):
            return

        # --- TWR row special-case (3 columns) ---
        # Example: ["Net Asset Value", "Data", "-57.908479365%"]
        if row_type == "Data" and len(row) >= 3:
            v = (row[2] or "").strip()
            if "%" in v:
                twr_val = v.replace("%", "").strip()
                twr_num = self._parse_number(twr_val)
                if twr_num is not None:
                    self.statement.nav.twr_percent = twr_num
                return

        # --- Normal NAV rows ---
        if len(row) < 7:
            return

        asset_class = (row[2] or "").strip()

        # IBKR format used in your file:
        # [2]=Asset Class, [6]=Current Total
        current_total = self._parse_number(row[6]) if len(row) > 6 else None
        if current_total is None:
            return

        if asset_class == "Total" or row_type == "Total":
            self.statement.nav.total = current_total
        elif asset_class == "Cash":
            self.statement.nav.cash = current_total
        elif asset_class == "Stock":
            self.statement.nav.stock = current_total
        elif asset_class == "Bond":
            self.statement.nav.bonds = current_total
        elif asset_class == "Options":
            self.statement.nav.options = current_total
        elif "Interest" in asset_class:
            self.statement.nav.interest_accruals = current_total
        elif "Dividend" in asset_class:
            self.statement.nav.dividend_accruals = current_total


    def _parse_change_in_nav(self, row: List[str], row_type: str):
        """Parse Change in NAV section."""
        if row_type != "Data" or len(row) < 4:
            return

        field_name = row[2].strip() if len(row) > 2 else ""
        value = self._parse_number(row[3]) if len(row) > 3 else 0.0

        # Map IBKR field names to our attributes
        # Note: IBKR uses "Deposits & Withdrawals" as combined field
        if field_name == "Starting Value":
            self.statement.change_in_nav.starting_value = value
        elif field_name == "Ending Value":
            self.statement.change_in_nav.ending_value = value
        elif field_name == "Mark-to-Market":
            self.statement.change_in_nav.mark_to_market = value
        elif field_name in ("Deposits & Withdrawals", "Deposits", "Withdrawals"):
            # Combined deposits/withdrawals field
            self.statement.change_in_nav.deposits_withdrawals += value
        elif field_name == "Dividends":
            self.statement.change_in_nav.dividends = value
        elif field_name == "Withholding Tax":
            self.statement.change_in_nav.withholding_tax = value
        elif field_name == "Change in Dividend Accruals":
            self.statement.change_in_nav.change_in_div_accruals = value
        elif field_name == "Interest":
            self.statement.change_in_nav.interest = value
        elif field_name == "Change in Interest Accruals":
            self.statement.change_in_nav.change_in_interest_accruals = value
        elif field_name in ("Other Fees", "Transaction Fees", "Sales Tax"):
            # Accumulate all fee types
            self.statement.change_in_nav.other_fees += value
        elif field_name == "Commissions":
            self.statement.change_in_nav.commissions = value

    def _parse_mtm_performance(self, row: List[str], row_type: str):
        """Parse Mark-to-Market Performance Summary."""
        if row_type != "Data" or len(row) < 5:
            return

        # Check if this is a Summary row (has symbol data)
        asset_class = row[2].strip() if len(row) > 2 else ""
        if asset_class in ("", "Stocks", "Equity and Index Options", "Bond", "Total"):
            # This is likely a subtotal row, skip
            if row_type == "SubTotal" or (len(row) > 3 and row[3].strip() == ""):
                return

        # Headers: Asset Class, Symbol, Description, Prior Period Value, Cost Basis,
        #          Realized S/T, Realized L/T, Unrealized S/T, Unrealized L/T, Transfers, MTM P/L
        if len(row) >= 12:
            mtm = MTMPerformance(
                symbol=row[3].strip() if len(row) > 3 else "",
                description=row[4].strip() if len(row) > 4 else "",
                prior_period_value=self._parse_number(row[5]) if len(row) > 5 else 0.0,
                cost_basis=self._parse_number(row[6]) if len(row) > 6 else 0.0,
                realized_st=self._parse_number(row[7]) if len(row) > 7 else 0.0,
                realized_lt=self._parse_number(row[8]) if len(row) > 8 else 0.0,
                unrealized_st=self._parse_number(row[9]) if len(row) > 9 else 0.0,
                unrealized_lt=self._parse_number(row[10]) if len(row) > 10 else 0.0,
                transfers=self._parse_number(row[11]) if len(row) > 11 else 0.0,
                mtm_pnl=self._parse_number(row[12]) if len(row) > 12 else 0.0,
            )
            if mtm.symbol:
                self.statement.mtm_performance.append(mtm)

    def _parse_open_position(self, row: List[str], row_type: str):
        """Parse open positions."""
        if row_type != "Data" or len(row) < 12:
            return

        # Headers: DataDiscriminator, Asset Category, Currency, Symbol, Quantity, Mult, Cost Price, Cost Basis, Close Price, Value, Unrealized P/L, Code
        # Indices: [2]=DataDiscriminator, [3]=Asset Category, [4]=Currency, [5]=Symbol, [6]=Quantity, [7]=Mult, [8]=Cost Price, [9]=Cost Basis, [10]=Close Price, [11]=Value, [12]=Unrealized P/L

        # Only process Summary rows (consolidated positions)
        data_discriminator = row[2].strip() if len(row) > 2 else ""
        if data_discriminator != "Summary":
            return

        asset_class = row[3].strip() if len(row) > 3 else "Stocks"
        symbol = row[5].strip() if len(row) > 5 else ""

        if not symbol:
            return

        pos = OpenPosition(
            symbol=symbol,
            asset_class=asset_class,
            quantity=self._parse_number(row[6]) if len(row) > 6 else 0.0,
            cost_price=self._parse_number(row[8]) if len(row) > 8 else 0.0,
            cost_basis=self._parse_number(row[9]) if len(row) > 9 else 0.0,
            current_price=self._parse_number(row[10]) if len(row) > 10 else 0.0,
            market_value=self._parse_number(row[11]) if len(row) > 11 else 0.0,
            unrealized_pnl=self._parse_number(row[12]) if len(row) > 12 else 0.0,
        )

        # Only add if we have meaningful data
        if pos.quantity != 0 or pos.market_value != 0:
            self.statement.open_positions.append(pos)

    def _parse_trade(self, row: List[str], row_type: str):
        """Parse trade records (robust to single-account CSV without Account column)."""
        if row_type != "Data" or len(row) < 10:
            return

        # Only process Order rows (individual trades)
        data_discriminator = row[2].strip() if len(row) > 2 else ""
        if data_discriminator != "Order":
            return

        # Single-account CSV format (confirmed by your file):
        # ['Trades','Data','Order','Asset Category','Currency','Symbol','Date/Time','Quantity','T. Price','C. Price','Proceeds','Comm/Fee','Basis','Realized P/L','MTM P/L','Code']
        # Indices:                  3              4          5        6           7         8         9        10         11       12      13            14       15
        asset_class = row[3].strip() if len(row) > 3 else "Stocks"

        symbol = row[5].strip() if len(row) > 5 else ""
        date_str = row[6].strip() if len(row) > 6 else ""

        # If the CSV ever includes Account column (older/consolidated variants), shift one to the right:
        # ... Currency, Account, Symbol, Date/Time, ...
        # Detect: if symbol looks like an account (U########) or date-like content, shift.
        if symbol.startswith("U") and symbol[1:].isdigit():
            symbol = row[6].strip() if len(row) > 6 else ""
            date_str = row[7].strip() if len(row) > 7 else ""
            qty_idx = 8
            price_idx = 9
            proceeds_idx = 11
            comm_idx = 12
            realized_idx = 14
            code_idx = 16
        else:
            qty_idx = 7
            price_idx = 8
            proceeds_idx = 10
            comm_idx = 11
            realized_idx = 13
            code_idx = 15

        if not symbol:
            return

        trade = Trade(
            symbol=symbol,
            asset_class=asset_class,
            date=self._parse_date(date_str) if date_str else None,
            quantity=self._parse_number(row[qty_idx]) if len(row) > qty_idx else 0.0,
            price=self._parse_number(row[price_idx]) if len(row) > price_idx else 0.0,
            proceeds=self._parse_number(row[proceeds_idx]) if len(row) > proceeds_idx else 0.0,
            commission=self._parse_number(row[comm_idx]) if len(row) > comm_idx else 0.0,
            realized_pnl=self._parse_number(row[realized_idx]) if len(row) > realized_idx else 0.0,
            code=row[code_idx].strip() if len(row) > code_idx else "",
        )

        self.statement.trades.append(trade)

    def _parse_dividend(self, row: List[str], row_type: str):
        """Parse dividend records (single-account + consolidated safe)."""
        if row_type not in ("Data",) or len(row) < 5:
            return

        # Headers (single-account): Currency, Date, Description, Amount
        # Row: ['Dividends','Data',Currency,Date,Description,Amount]
        currency = row[2].strip() if len(row) > 2 else ""
        date_str = row[3].strip() if len(row) > 3 else ""
        description = row[4].strip() if len(row) > 4 else ""
        amount = self._parse_number(row[5]) if len(row) > 5 else 0.0

        # Consolidated variants sometimes insert Account after Currency:
        # ['Dividends','Data',Currency,Account,Date,Description,Amount]
        if date_str.startswith("U") and date_str[1:].isdigit():
            date_str = row[4].strip() if len(row) > 4 else ""
            description = row[5].strip() if len(row) > 5 else ""
            amount = self._parse_number(row[6]) if len(row) > 6 else 0.0

        # Extract symbol from description: "SYMBOL(....) ..."
        symbol = ""
        if description:
            match = re.match(r"^([A-Z]{1,10})\s*\(", description)
            if match:
                symbol = match.group(1)
            else:
                parts = description.split()
                if parts and re.match(r"^[A-Z]{1,10}$", parts[0]):
                    symbol = parts[0]

        if amount != 0:
            div = DividendRecord(
                date=self._parse_date(date_str) if date_str else None,
                symbol=symbol,
                description=description,
                amount=amount,
            )
            self.statement.dividends.append(div)

    def _parse_withholding_tax(self, row: List[str], row_type: str):
        """Parse withholding tax records."""
        if row_type not in ("Data",) or len(row) < 5:
            return

        # Headers: Currency, Date, Description, Amount, Code
        # Row: ['Withholding Tax','Data',Currency,Date,Description,Amount,Code]
        date_str = row[3].strip() if len(row) > 3 else ""
        description = row[4].strip() if len(row) > 4 else ""
        amount = self._parse_number(row[5]) if len(row) > 5 else 0.0

        # Consolidated variants may include Account after Currency
        if date_str.startswith("U") and date_str[1:].isdigit():
            date_str = row[4].strip() if len(row) > 4 else ""
            description = row[5].strip() if len(row) > 5 else ""
            amount = self._parse_number(row[6]) if len(row) > 6 else 0.0

        # Extract symbol for dividend-related WHT lines: "SYMBOL(..."
        symbol = ""
        if description:
            m = re.match(r"^([A-Z]{1,10})\s*\(", description)
            if m:
                symbol = m.group(1)

        if amount != 0:
            wht = WithholdingTaxRecord(
                date=self._parse_date(date_str) if date_str else None,
                symbol=symbol,
                description=description,
                amount=amount,
            )
            self.statement.withholding_tax.append(wht)

    def _parse_interest(self, row: List[str], row_type: str):
        """Parse interest records."""
        if row_type not in ("Data",) or len(row) < 5:
            return

        # Headers: Currency, Date, Description, Amount
        # Row: ['Interest','Data',Currency,Date,Description,Amount]
        date_str = row[3].strip() if len(row) > 3 else ""
        description = row[4].strip() if len(row) > 4 else ""
        amount = self._parse_number(row[5]) if len(row) > 5 else 0.0

        # Consolidated variants may include Account after Currency
        if date_str.startswith("U") and date_str[1:].isdigit():
            date_str = row[4].strip() if len(row) > 4 else ""
            description = row[5].strip() if len(row) > 5 else ""
            amount = self._parse_number(row[6]) if len(row) > 6 else 0.0

        if amount != 0:
            interest = InterestRecord(
                date=self._parse_date(date_str) if date_str else None,
                description=description,
                amount=amount,
            )
            self.statement.interest.append(interest)

    def _parse_deposit_withdrawal(self, row: List[str], row_type: str):
        """Parse deposit and withdrawal records."""
        if row_type not in ("Data",) or len(row) < 5:
            return

        # Headers: Currency, Settle Date, Description, Amount
        # Row: ['Deposits & Withdrawals','Data',Currency,Settle Date,Description,Amount]
        date_str = row[3].strip() if len(row) > 3 else ""
        description = row[4].strip() if len(row) > 4 else ""
        amount = self._parse_number(row[5]) if len(row) > 5 else 0.0

        # Consolidated variants may include Account after Currency
        if date_str.startswith("U") and date_str[1:].isdigit():
            date_str = row[4].strip() if len(row) > 4 else ""
            description = row[5].strip() if len(row) > 5 else ""
            amount = self._parse_number(row[6]) if len(row) > 6 else 0.0

        if amount != 0:
            dw = DepositWithdrawal(
                date=self._parse_date(date_str) if date_str else None,
                description=description,
                amount=amount,
            )
            self.statement.deposits_withdrawals.append(dw)

    def _parse_fee(self, row: List[str], row_type: str):
        """Parse fee records (handles Fees section with Subtitle column)."""
        if row_type not in ("Data",) or len(row) < 6:
            return

        # Fees header (your file):
        # ['Fees','Header','Subtitle','Currency','Date','Description','Amount']
        # Fees row:
        # ['Fees','Data',Subtitle,Currency,Date,Description,Amount]
        subtitle = row[2].strip() if len(row) > 2 else ""
        currency = row[3].strip() if len(row) > 3 else ""
        date_str = row[4].strip() if len(row) > 4 else ""
        description = row[5].strip() if len(row) > 5 else ""
        amount = self._parse_number(row[6]) if len(row) > 6 else 0.0

        # Consolidated variants may insert Account after Currency; handle if date looks like U########
        if date_str.startswith("U") and date_str[1:].isdigit():
            date_str = row[5].strip() if len(row) > 5 else ""
            description = row[6].strip() if len(row) > 6 else ""
            amount = self._parse_number(row[7]) if len(row) > 7 else 0.0

        if amount != 0:
            fee = FeeRecord(
                date=self._parse_date(date_str) if date_str else None,
                description=(f"{subtitle}: {description}" if subtitle else description),
                amount=amount,
            )
            self.statement.fees.append(fee)

    def _calculate_position_returns(self):
        """Calculate return percentages for positions."""
        for pos in self.statement.open_positions:
            if pos.cost_basis != 0:
                pos.unrealized_pnl_pct = (pos.unrealized_pnl / abs(pos.cost_basis)) * 100


# =============================================================================
# FILE UTILITIES
# =============================================================================

def get_statement_dir() -> str:
    """Get the directory for storing statement files."""
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "statements")
    os.makedirs(config_dir, exist_ok=True)
    return config_dir


def find_statement_files(account_id: str = None) -> List[str]:
    """Find all statement files, optionally filtered by account.

    For "All" account, looks for MULTI*.csv or consolidated files.
    For specific accounts, looks for files containing the account ID.
    """
    statement_dir = get_statement_dir()
    pattern = os.path.join(statement_dir, "*.csv")
    files = glob.glob(pattern)

    if account_id:
        if account_id == "All":
            # Look for MULTI or consolidated files
            files = [f for f in files if "MULTI" in os.path.basename(f).upper() or
                     "CONSOLIDATED" in os.path.basename(f).upper() or
                     "ALL" in os.path.basename(f).upper()]
        else:
            # Look for files containing the specific account ID
            files = [f for f in files if account_id in os.path.basename(f)]

    return sorted(files, reverse=True)


def get_cached_data_path(account_id: str) -> str:
    """Get path for cached parsed statement data."""
    config_dir = get_statement_dir()
    return os.path.join(config_dir, f"{account_id}_parsed.json")


def load_statement_data(account_id: str) -> Optional[ParsedStatement]:
    """Load and parse the most recent statement for an account."""
    files = find_statement_files(account_id)

    # Debug: Log what files were found
    import logging
    logging.info(f"[load_statement_data] Looking for account: {account_id}")
    logging.info(f"[load_statement_data] Statement dir: {get_statement_dir()}")
    logging.info(f"[load_statement_data] Files found: {files}")

    if not files:
        return None

    # Load the most recent file
    latest_file = files[0]
    try:
        with open(latest_file, 'r', encoding='utf-8-sig') as f:  # Handle BOM
            content = f.read()
        parser = IBKRStatementParser()
        statement = parser.parse(content)
        logging.info(f"[load_statement_data] Parsed: NAV={statement.nav.total}, Deposits={statement.change_in_nav.deposits_withdrawals}")
        return statement
    except Exception as e:
        logging.error(f"[load_statement_data] Error: {e}")
        return None


def load_statement_from_db(account_id: str) -> Optional[ParsedStatement]:
    """
    Load portfolio data from database and create a ParsedStatement.
    This is the preferred method - faster and more reliable than CSV parsing.

    For "All" or "CONSOLIDATED", combines data from all accounts.

    NOTE: When MULTI CSV was imported, all deposits went to one account.
    For individual accounts, we calculate based on:
    - External deposits (excluding internal transfers)
    - For accounts with only internal transfers, use the transfer amount as "capital allocated"
    """
    if not PORTFOLIO_DB_AVAILABLE:
        return None

    try:
        repo = PortfolioRepository()

        # Handle consolidated "All" account - combine all accounts
        if account_id in ("All", "CONSOLIDATED"):
            return _load_consolidated_from_db(repo)

        # Get account info for specific account
        account = repo.get_account(account_id)
        if not account:
            return None

        # Get summary data
        summary = repo.get_account_summary(account_id)
        if not summary:
            return None

        # Create ParsedStatement from database data
        statement = ParsedStatement()

        # Account Info
        statement.account_info = AccountInfo(
            name=account.get('name', ''),
            account_id=account_id,
            account_type=account.get('account_type', ''),
            base_currency=account.get('base_currency', 'USD'),
        )

        # NAV from database - use NAVBreakdown class
        nav_data = summary.get('nav')
        if nav_data:
            statement.nav = NAVBreakdown(
                cash=float(nav_data.get('nav_cash') or 0),
                stock=float(nav_data.get('nav_stock') or 0),
                bonds=float(nav_data.get('nav_bonds') or 0),
                options=float(nav_data.get('nav_options') or 0),
                total=float(nav_data.get('nav_total') or 0),
                twr_percent=float(nav_data.get('twr_percent') or 0),
            )
            # Set period from NAV date
            nav_date = nav_data.get('date')
            if nav_date:
                statement.period_end = str(nav_date)

        # Load deposits/withdrawals and calculate correct capital basis
        dw_df = repo.get_deposits_withdrawals(account_id)
        external_deposits = 0.0
        internal_transfers_in = 0.0
        internal_transfers_out = 0.0

        for _, row in dw_df.iterrows():
            desc = row.get('description', '')
            amount = float(row['amount'])

            # Categorize the deposit
            # Check for self-transfers (bad data from MULTI CSV) - ignore these
            is_self_transfer = f'From Account {account_id}' in desc or f'To Account {account_id}' in desc

            if is_self_transfer:
                # Skip self-transfers - these are artifacts from MULTI CSV import
                pass
            elif 'Transfer In From Account' in desc:
                internal_transfers_in += amount  # Positive amount coming in
            elif 'Transfer Out To Account' in desc:
                internal_transfers_out += amount  # Negative amount going out
            else:
                external_deposits += amount

            statement.deposits_withdrawals.append(DepositWithdrawal(
                date=row['date'],
                description=desc,
                amount=amount,
            ))

        # Calculate correct capital basis for this specific account:
        # Capital = External deposits + Internal transfers (net)
        # For U17994267: $2.13M external - $717K transferred out = $1.414M
        # For U20993660: $0 external + $717K transferred in = $717K
        capital_basis = external_deposits + internal_transfers_in + internal_transfers_out

        # Change in NAV
        statement.change_in_nav = ChangeInNAV(
            starting_value=0,
            ending_value=float(nav_data.get('nav_total') or 0) if nav_data else 0,
            deposits_withdrawals=capital_basis,
            dividends=summary.get('total_dividends', 0),
            interest=summary.get('total_interest', 0),
            withholding_tax=summary.get('total_withholding_tax', 0),
        )

        # Load dividends - use DividendRecord class
        div_df = repo.get_dividends(account_id)
        for _, row in div_df.iterrows():
            statement.dividends.append(DividendRecord(
                date=row['date'],
                symbol=row.get('symbol', ''),
                description=row.get('description', ''),
                amount=float(row['amount']),
            ))

        # Load positions - use OpenPosition class
        pos_df = repo.get_positions(account_id)
        for _, row in pos_df.iterrows():
            statement.open_positions.append(OpenPosition(
                symbol=row['symbol'],
                description=row.get('description', ''),
                quantity=float(row['quantity']),
                cost_basis=float(row.get('cost_basis') or 0),
                cost_price=float(row.get('cost_price') or 0),
                current_price=float(row.get('current_price') or 0),
                market_value=float(row.get('market_value') or 0),
                unrealized_pnl=float(row.get('unrealized_pnl') or 0),
                unrealized_pnl_pct=float(row.get('unrealized_pnl_pct') or 0),
                asset_class=row.get('asset_class', 'Stocks'),
            ))

        # Load trades
        trades_df = repo.get_trades(account_id)
        for _, row in trades_df.iterrows():
            statement.trades.append(Trade(
                date=row['date'],
                symbol=row['symbol'],
                description=row.get('description', ''),
                quantity=float(row['quantity']),
                price=float(row['price']),
                proceeds=float(row.get('proceeds') or 0),
                commission=float(row.get('commission') or 0),
                realized_pnl=float(row.get('realized_pnl') or 0),
                asset_class=row.get('asset_class', 'Stocks'),
            ))

        return statement

    except Exception as e:
        import logging
        logging.error(f"[load_statement_from_db] Error loading {account_id}: {e}")
        return None


def _load_consolidated_from_db(repo) -> Optional[ParsedStatement]:
    """
    Load consolidated data by combining all accounts in the database.
    """
    try:
        # Get all accounts
        accounts = repo.get_all_accounts()
        if not accounts:
            return None

        # Initialize totals
        total_nav = 0.0
        total_cash = 0.0
        total_stock = 0.0
        total_bonds = 0.0
        total_options = 0.0
        total_deposits = 0.0
        total_dividends = 0.0
        total_interest = 0.0
        total_wht = 0.0

        all_positions = []
        all_deposits_withdrawals = []
        all_dividends = []
        all_trades = []
        latest_date = None

        # Combine data from all accounts
        for acc in accounts:
            acc_id = acc['account_id']

            # Get NAV
            nav = repo.get_latest_nav(acc_id)
            if nav:
                total_nav += float(nav.get('nav_total') or 0)
                total_cash += float(nav.get('nav_cash') or 0)
                total_stock += float(nav.get('nav_stock') or 0)
                total_bonds += float(nav.get('nav_bonds') or 0)
                total_options += float(nav.get('nav_options') or 0)
                nav_date = nav.get('date')
                if nav_date and (latest_date is None or nav_date > latest_date):
                    latest_date = nav_date

            # Get totals
            total_deposits += repo.get_total_deposits(acc_id)
            total_dividends += repo.get_total_dividends(acc_id)
            total_interest += repo.get_total_interest(acc_id)
            total_wht += repo.get_total_withholding_tax(acc_id)

            # Get positions
            pos_df = repo.get_positions(acc_id)
            for _, row in pos_df.iterrows():
                all_positions.append(OpenPosition(
                    symbol=row['symbol'],
                    description=row.get('description', ''),
                    quantity=float(row['quantity']),
                    cost_basis=float(row.get('cost_basis') or 0),
                    cost_price=float(row.get('cost_price') or 0),
                    current_price=float(row.get('current_price') or 0),
                    market_value=float(row.get('market_value') or 0),
                    unrealized_pnl=float(row.get('unrealized_pnl') or 0),
                    unrealized_pnl_pct=float(row.get('unrealized_pnl_pct') or 0),
                    asset_class=row.get('asset_class', 'Stocks'),
                ))

            # Get deposits/withdrawals (exclude internal transfers)
            dw_df = repo.get_deposits_withdrawals(acc_id)
            for _, row in dw_df.iterrows():
                desc = row.get('description', '')
                # Skip internal transfers between IBKR accounts
                # But keep "Electronic Fund Transfer" which is external
                is_internal_transfer = ('Transfer In From Account' in desc or
                                        'Transfer Out To Account' in desc or
                                        'Transfer In To Account' in desc or
                                        'Transfer Out From Account' in desc)
                if is_internal_transfer:
                    continue
                all_deposits_withdrawals.append(DepositWithdrawal(
                    date=row['date'],
                    description=desc,
                    amount=float(row['amount']),
                ))

            # Get dividends
            div_df = repo.get_dividends(acc_id)
            for _, row in div_df.iterrows():
                all_dividends.append(DividendRecord(
                    date=row['date'],
                    symbol=row.get('symbol', ''),
                    description=row.get('description', ''),
                    amount=float(row['amount']),
                ))

            # Get trades
            trades_df = repo.get_trades(acc_id)
            for _, row in trades_df.iterrows():
                all_trades.append(Trade(
                    date=row['date'],
                    symbol=row['symbol'],
                    description=row.get('description', ''),
                    quantity=float(row['quantity']),
                    price=float(row['price']),
                    proceeds=float(row.get('proceeds') or 0),
                    commission=float(row.get('commission') or 0),
                    realized_pnl=float(row.get('realized_pnl') or 0),
                    asset_class=row.get('asset_class', 'Stocks'),
                ))

        # Recalculate net deposits excluding internal transfers
        net_deposits = sum(dw.amount for dw in all_deposits_withdrawals)

        # Create consolidated statement
        statement = ParsedStatement()

        statement.account_info = AccountInfo(
            name="Consolidated Portfolio",
            account_id="All",
            account_type="Consolidated",
            base_currency="USD",
        )

        statement.nav = NAVBreakdown(
            cash=total_cash,
            stock=total_stock,
            bonds=total_bonds,
            options=total_options,
            total=total_nav,
            twr_percent=0,  # Can't simply add TWR
        )

        if latest_date:
            statement.period_end = str(latest_date)

        statement.change_in_nav = ChangeInNAV(
            starting_value=0,
            ending_value=total_nav,
            deposits_withdrawals=net_deposits,
            dividends=total_dividends,
            interest=total_interest,
            withholding_tax=total_wht,
        )

        statement.open_positions = all_positions
        statement.deposits_withdrawals = all_deposits_withdrawals
        statement.dividends = all_dividends
        statement.trades = all_trades

        return statement

    except Exception as e:
        import logging
        logging.error(f"[_load_consolidated_from_db] Error: {e}")
        return None


def save_uploaded_statement(uploaded_file, account_id: str = None) -> bool:
    """Save an uploaded statement file."""
    try:
        statement_dir = get_statement_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Parse to get account ID if not provided
        content = uploaded_file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8-sig')  # Handle BOM

        parser = IBKRStatementParser()
        parsed = parser.parse(content)

        if not account_id:
            account_id = parsed.account_info.account_id or "unknown"

        # Check if this is a multi-account/consolidated statement
        # Look for "Accounts Included" or "(Custom Consolidated)" in account ID
        is_multi = False
        if "Consolidated" in account_id or "MULTI" in account_id.upper():
            is_multi = True
            account_id = "MULTI"

        # Also check the raw content for "Accounts Included"
        if "Accounts Included" in content:
            is_multi = True
            account_id = "MULTI"

        filename = f"{account_id}_{timestamp}.csv"
        filepath = os.path.join(statement_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        import logging
        logging.info(f"[save_uploaded_statement] Saved to: {filepath}")
        logging.info(f"[save_uploaded_statement] Is multi-account: {is_multi}")

        return True
    except Exception as e:
        import logging
        logging.error(f"[save_uploaded_statement] Error: {e}")
        return False


# =============================================================================
# UI HELPERS
# =============================================================================

def inject_css():
    """Inject custom CSS for styling."""
    st.markdown("""
    <style>
    .perf-card {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin: 5px 0;
    }
    .perf-card.positive { border-left: 4px solid #00c853; }
    .perf-card.negative { border-left: 4px solid #ff5252; }
    .perf-card.neutral { border-left: 4px solid #888; }
    </style>
    """, unsafe_allow_html=True)


def fmt(value: float, fmt_type: str = "money", hide: bool = False) -> str:
    """Format values with privacy support."""
    if hide:
        return "•••••••"

    if value is None:
        return "N/A"

    if fmt_type == "money":
        return f"${value:,.2f}"
    elif fmt_type == "money_int":
        return f"${value:,.0f}"
    elif fmt_type == "signed":
        prefix = "+" if value >= 0 else ""
        return f"{prefix}${value:,.2f}"
    elif fmt_type == "pct":
        prefix = "+" if value >= 0 else ""
        return f"{prefix}{value:.2f}%"
    elif fmt_type == "number":
        return f"{value:,.2f}"
    else:
        return str(value)


# =============================================================================
# YAHOO PRICE FETCHING
# =============================================================================

@st.cache_data(ttl=300)
def fetch_yahoo_prices(symbols: List[str]) -> Dict[str, Dict]:
    """Fetch current prices from Yahoo Finance."""
    if not symbols:
        return {}

    try:
        import yfinance as yf

        prices = {}
        # Batch fetch
        tickers = yf.Tickers(' '.join(symbols))

        for symbol in symbols:
            try:
                ticker = tickers.tickers.get(symbol)
                if ticker:
                    info = ticker.info
                    hist = ticker.history(period='2d')

                    current_price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
                    prev_close = info.get('previousClose') or info.get('regularMarketPreviousClose') or current_price

                    if not hist.empty and len(hist) >= 1:
                        current_price = hist['Close'].iloc[-1]
                        if len(hist) >= 2:
                            prev_close = hist['Close'].iloc[-2]

                    day_change = current_price - prev_close
                    day_change_pct = (day_change / prev_close * 100) if prev_close != 0 else 0

                    prices[symbol] = {
                        'current_price': current_price,
                        'prev_close': prev_close,
                        'day_change': day_change,
                        'day_change_pct': day_change_pct
                    }
            except Exception:
                continue

        return prices
    except ImportError:
        return {}
    except Exception:
        return {}


def update_positions_with_prices(positions: List[Dict], prices: Dict[str, Dict]) -> List[Dict]:
    """Update positions with live prices from Yahoo."""
    for pos in positions:
        symbol = pos.get('symbol', '')
        if symbol in prices:
            price_data = prices[symbol]
            pos['current_price'] = price_data['current_price']
            pos['day_change'] = price_data['day_change']
            pos['day_change_pct'] = price_data['day_change_pct']

            # Recalculate market value and P&L with new price
            qty = pos.get('quantity', 0)
            avg_cost = pos.get('avg_cost', 0)

            pos['market_value'] = qty * price_data['current_price']
            pos['unrealized_pnl'] = pos['market_value'] - (qty * avg_cost)
            if qty * avg_cost != 0:
                pos['unrealized_pnl_pct'] = (pos['unrealized_pnl'] / (qty * avg_cost)) * 100

    return positions


# =============================================================================
# IBKR LIVE DATA INTEGRATION
# =============================================================================

@st.cache_data(ttl=60)
def load_ibkr_data(account_id: str, host: str = "127.0.0.1", port: int = 7496,
                  live: bool = False) -> Dict[str, Any]:
    """Load portfolio data from IBKR TWS/Gateway."""
    try:
        from src.broker.ibkr_client import IBKRClient

        client = IBKRClient(host=host, port=port)
        if not client.connect():
            return {"error": "Failed to connect to IBKR", "positions": [], "summary": {}}

        positions = client.get_positions(account_id)
        summary = client.get_account_summary(account_id)

        client.disconnect()

        return {
            "positions": positions,
            "summary": summary,
            "error": None
        }
    except ImportError:
        return {"error": "IBKR client not available", "positions": [], "summary": {}}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "positions": [], "summary": []}


@st.cache_data(ttl=60)
def get_accounts_only(host: str = "127.0.0.1", port: int = 7496) -> tuple:
    """Get list of available IBKR accounts."""
    try:
        from src.broker.ibkr_client import IBKRClient

        client = IBKRClient(host=host, port=port)
        if not client.connect():
            return [], "Failed to connect"

        accounts = client.get_managed_accounts()
        client.disconnect()

        return accounts, None
    except ImportError:
        return [], "IBKR client not available"
    except Exception as e:
        return [], str(e)


def _create_statement_from_live_ibkr(ibkr_data: Dict[str, Any], account: str, timestamp: datetime = None,
                                     historical_statement: Optional[ParsedStatement] = None) -> Optional[ParsedStatement]:
    """
    Create a ParsedStatement from live IBKR data.
    This allows the portfolio display to show live data using the same rendering logic.

    IBKRClient returns:
    - positions: list[dict] with keys: symbol, position, avgCost, marketValue, unrealizedPNL, etc.
    - summary: dict with keys: net_liquidation, total_cash, buying_power, gross_position_value, etc.

    If historical_statement is provided, we merge:
    - Live: NAV, positions, cash (current values)
    - Historical: deposits/withdrawals, dividends, interest, trades, fees (for True Profit calculation)

    NOTE: marketValue might be 0 if prices weren't fetched - we use gross_position_value from summary instead.
    """
    if not ibkr_data:
        return None

    positions = ibkr_data.get("positions", [])
    summary = ibkr_data.get("summary", {})

    if not positions and not summary:
        return None

    # Create account info
    account_info = AccountInfo(
        name="Live Portfolio",
        account_id=account,
        account_type="Live",
        base_currency=summary.get("currency", "USD") if isinstance(summary, dict) else "USD",
    )

    # Get NAV values from summary
    # IBKRClient returns lowercase keys: net_liquidation, total_cash, etc.
    nav_total = 0.0
    cash_value = 0.0
    stock_value = 0.0

    if isinstance(summary, dict):
        # Try lowercase first (IBKRClient format), then uppercase (ib_insync format)
        nav_total = float(summary.get("net_liquidation", 0) or summary.get("NetLiquidation", 0) or 0)
        cash_value = float(summary.get("total_cash", 0) or summary.get("TotalCashValue", 0) or
                          summary.get("available_funds", 0) or summary.get("AvailableFunds", 0) or 0)
        stock_value = float(summary.get("gross_position_value", 0) or summary.get("GrossPositionValue", 0) or 0)
    elif hasattr(summary, '__iter__') and not isinstance(summary, (str, dict)):
        # Summary might be a list of account values (ib_insync format)
        for item in summary:
            if hasattr(item, 'tag'):
                tag = item.tag
                val = float(item.value or 0)
                if tag == "NetLiquidation":
                    nav_total = val
                elif tag == "TotalCashValue":
                    cash_value = val
                elif tag == "GrossPositionValue":
                    stock_value = val

    # Create open positions
    open_positions = []
    calculated_stock_value = 0.0

    # Calculate per-position market value based on gross_position_value if individual marketValues are 0
    total_position_count = len([p for p in positions if isinstance(p, dict)])

    for pos in positions:
        if isinstance(pos, dict):
            # Dict-style position from IBKRClient
            # Keys: symbol, position, avgCost, marketValue, unrealizedPNL, realizedPNL, account, secType, exchange, currency
            symbol = pos.get('symbol', '')
            qty = float(pos.get('position', 0) or pos.get('quantity', 0) or 0)
            avg_cost = float(pos.get('avgCost', 0) or pos.get('avg_cost', 0) or pos.get('cost_price', 0) or 0)
            market_value = float(pos.get('marketValue', 0) or pos.get('market_value', 0) or 0)
            unrealized_pnl = float(pos.get('unrealizedPNL', 0) or pos.get('unrealized_pnl', 0) or 0)

            # If market_value is 0, estimate from avgCost (not ideal but better than showing 0)
            # The real market value comes from gross_position_value in summary
            if market_value == 0 and qty != 0 and avg_cost > 0:
                # Use avgCost as estimate - not accurate but shows something
                market_value = qty * avg_cost

            # Calculate market_price from market_value
            market_price = market_value / qty if qty != 0 else avg_cost

            # Calculate cost_basis
            cost_basis = qty * avg_cost if avg_cost > 0 else 0

            # Use provided unrealized_pnl or calculate
            if unrealized_pnl == 0 and cost_basis > 0 and market_value > 0:
                unrealized_pnl = market_value - cost_basis

            unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis != 0 else 0

            open_positions.append(OpenPosition(
                symbol=symbol,
                description=pos.get('description', '') or symbol,
                quantity=qty,
                cost_basis=cost_basis,
                cost_price=avg_cost,
                current_price=market_price if market_price > 0 else avg_cost,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                asset_class=pos.get('asset_class', '') or pos.get('secType', 'STK') or 'Stocks',
            ))
            calculated_stock_value += market_value

        elif hasattr(pos, 'contract') and hasattr(pos, 'position'):
            # ib_insync Position object
            contract = pos.contract
            symbol = getattr(contract, 'symbol', '')
            qty = float(getattr(pos, 'position', 0) or 0)
            avg_cost = float(getattr(pos, 'avgCost', 0) or 0)
            market_price = float(getattr(pos, 'marketPrice', 0) or 0)
            market_value = float(getattr(pos, 'marketValue', 0) or qty * market_price)

            cost_basis = qty * avg_cost
            unrealized_pnl = market_value - cost_basis
            unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis != 0 else 0

            open_positions.append(OpenPosition(
                symbol=symbol,
                description=getattr(contract, 'localSymbol', symbol),
                quantity=qty,
                cost_basis=cost_basis,
                cost_price=avg_cost,
                current_price=market_price if market_price > 0 else avg_cost,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                asset_class=getattr(contract, 'secType', 'STK'),
            ))
            calculated_stock_value += market_value

    # Use gross_position_value from summary (more accurate than calculated)
    # Only if calculated is way off
    if stock_value > 0 and abs(stock_value - calculated_stock_value) > 1000:
        # Adjust positions proportionally to match gross_position_value
        if calculated_stock_value > 0:
            adjustment_factor = stock_value / calculated_stock_value
            for pos in open_positions:
                pos.market_value = pos.market_value * adjustment_factor
                pos.current_price = pos.market_value / pos.quantity if pos.quantity != 0 else pos.cost_price
                pos.unrealized_pnl = pos.market_value - pos.cost_basis
                pos.unrealized_pnl_pct = (pos.unrealized_pnl / pos.cost_basis * 100) if pos.cost_basis != 0 else 0

    # Use stock_value from summary if available
    final_stock_value = stock_value if stock_value > 0 else calculated_stock_value

    # Create NAV breakdown
    nav = NAVBreakdown(
        cash=cash_value,
        stock=final_stock_value,
        total=nav_total if nav_total > 0 else (cash_value + final_stock_value),
    )

    # Create change in NAV (minimal - just ending value)
    change_in_nav = ChangeInNAV(
        ending_value=nav.total,
    )

    # Create the statement
    ts = timestamp or datetime.now()

    # If we have historical statement, merge the historical data (deposits, dividends, etc.)
    # with live data (current NAV, positions)
    if historical_statement:
        # Use historical data for:
        # - deposits_withdrawals (needed for True Profit calculation)
        # - dividends, interest, withholding_tax (income breakdown)
        # - trades, fees (trade history)
        # - change_in_nav starting_value and other fields

        # Update change_in_nav with historical data but use live ending value
        historical_change = historical_statement.change_in_nav
        change_in_nav = ChangeInNAV(
            starting_value=historical_change.starting_value,
            ending_value=nav.total,  # Use live NAV
            mark_to_market=historical_change.mark_to_market,
            deposits_withdrawals=historical_change.deposits_withdrawals,
            dividends=historical_change.dividends,
            withholding_tax=historical_change.withholding_tax,
            change_in_div_accruals=historical_change.change_in_div_accruals,
            interest=historical_change.interest,
            change_in_interest_accruals=historical_change.change_in_interest_accruals,
            other_fees=historical_change.other_fees,
            commissions=historical_change.commissions,
        )

        # Use historical account info but mark as live
        hist_acc = historical_statement.account_info
        account_info = AccountInfo(
            name=hist_acc.name if hist_acc.name else "Live Portfolio",
            account_id=account,
            account_type="Live + Historical",
            base_currency=hist_acc.base_currency or "USD",
        )

        statement = ParsedStatement(
            account_info=account_info,
            period_start=historical_statement.period_start or ts.strftime("%Y-%m-%d"),
            period_end=ts.strftime("%Y-%m-%d"),  # Today for live data
            nav=nav,
            change_in_nav=change_in_nav,
            open_positions=open_positions,  # Use live positions
            trades=historical_statement.trades,
            dividends=historical_statement.dividends,
            interest=historical_statement.interest,
            withholding_tax=historical_statement.withholding_tax,
            deposits_withdrawals=historical_statement.deposits_withdrawals,
            fees=historical_statement.fees,
            mtm_performance=historical_statement.mtm_performance,
        )
    else:
        statement = ParsedStatement(
            account_info=account_info,
            period_start=ts.strftime("%Y-%m-%d"),
            period_end=ts.strftime("%Y-%m-%d"),
            nav=nav,
            change_in_nav=change_in_nav,
            open_positions=open_positions,
        )

    return statement


# portfolio_tab.py — replace this method entirely
def get_trading_signals() -> Dict[str, Dict]:
    """Get latest trading signals from the platform DB (Repository)."""
    try:
        from src.db.repository import Repository

        repo = Repository()
        df = repo.get_latest_signals()
        if df is None or df.empty:
            return {}

        key_col = "ticker" if "ticker" in df.columns else ("symbol" if "symbol" in df.columns else None)
        if not key_col:
            return {}

        out: Dict[str, Dict] = {}
        for _, row in df.iterrows():
            k = row.get(key_col)
            if pd.isna(k) or k is None:
                continue
            out[str(k)] = row.to_dict()
        return out
    except Exception:
        return {}


def get_signal_status(qty: float, signal: str) -> Tuple[str, str]:
    """Determine signal alignment status (handles both 'STRONG BUY' and 'STRONG_BUY' formats)."""
    if not signal:
        return ("gray", "No Signal")

    s = str(signal).strip().upper()
    s = s.replace("_", " ")  # normalize DB values like STRONG_BUY -> STRONG BUY

    is_long = qty > 0
    is_bullish = s in ("STRONG BUY", "BUY", "WEAK BUY")
    is_bearish = s in ("STRONG SELL", "SELL", "WEAK SELL")

    if (is_long and is_bullish) or ((not is_long) and is_bearish):
        return ("#00c853", "Aligned")
    elif (is_long and is_bearish) or ((not is_long) and is_bullish):
        return ("#ff5252", "Conflict")
    else:
        return ("#888", "Neutral")


# =============================================================================
# STATEMENT ANALYTICS FUNCTIONS
# =============================================================================

def aggregate_dividends_by_symbol(dividends: List[DividendRecord]) -> Dict[str, float]:
    """Aggregate dividend amounts by symbol."""
    by_symbol = {}
    for div in dividends:
        if div.symbol:
            by_symbol[div.symbol] = by_symbol.get(div.symbol, 0) + div.amount
    return by_symbol


def aggregate_trades_by_symbol(trades: List[Trade]) -> Dict[str, Dict]:
    """
    Aggregates trades by symbol so that:
    - A symbol is included only if it has at least one CLOSED leg in the period.
    - Qty Bought / Bought $ are computed from ALL legs in-period (O + C),
      while Realized P&L / Commissions / Timestamp / #Trades are computed from CLOSED legs only.

    IMPORTANT LIMITATION:
    If the statement period contains ONLY the closing sells (and the opening buys happened before the period),
    then Qty Bought / Bought $ will still be 0 because those buy legs are not present in the CSV export.
    """
    by_symbol: Dict[str, Dict] = {}
    has_close: Dict[str, bool] = {}

    def _is_close_leg(t: Trade) -> bool:
        code = (t.code or "").upper()
        return ("C" in code) or ((t.realized_pnl or 0.0) != 0.0)

    # 1) detect which symbols have a close in this period
    for t in trades:
        if not t.symbol:
            continue
        if _is_close_leg(t):
            has_close[t.symbol] = True

    # 2) aggregate for those symbols, using all legs for qty/proceeds
    for t in trades:
        if not t.symbol or not has_close.get(t.symbol, False):
            continue

        if t.symbol not in by_symbol:
            by_symbol[t.symbol] = {
                "symbol": t.symbol,
                "timestamp": None,   # last close timestamp
                "trades": 0,         # count of CLOSED legs
                "bought": 0.0,
                "sold": 0.0,
                "qty_bought": 0.0,
                "qty_sold": 0.0,
                "realized_pnl": 0.0, # CLOSED legs only
                "commissions": 0.0,  # CLOSED legs only
            }

        s = by_symbol[t.symbol]

        qty = float(t.quantity or 0.0)
        proceeds = float(t.proceeds or 0.0)

        # Include ALL legs for qty/proceeds
        if qty > 0:
            s["qty_bought"] += qty
            s["bought"] += abs(proceeds)
        elif qty < 0:
            s["qty_sold"] += abs(qty)
            s["sold"] += abs(proceeds)

        # CLOSED-only metrics
        if _is_close_leg(t):
            s["trades"] += 1
            s["realized_pnl"] += float(t.realized_pnl or 0.0)
            s["commissions"] += abs(float(t.commission or 0.0))

            if t.date:
                if s["timestamp"] is None or t.date > s["timestamp"]:
                    s["timestamp"] = t.date

    # format timestamp
    for v in by_symbol.values():
        ts = v.get("timestamp")
        v["timestamp"] = ts.strftime("%Y-%m-%d, %H:%M:%S") if ts else ""

    return by_symbol

@st.cache_data(ttl=3600)
def _fetch_benchmark_adjclose(ticker: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    import yfinance as yf

    # Pad to ensure we capture the first/last trading day around holidays/weekends
    start = (start_dt - timedelta(days=20)).date().isoformat()
    end = (end_dt + timedelta(days=20)).date().isoformat()

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)

    if df is None or df.empty:
        return pd.DataFrame()

    # Handle yfinance MultiIndex columns (sometimes returned depending on version/settings)
    # Examples:
    #   - single-level: ["Open","High","Low","Close","Adj Close","Volume"]
    #   - multi-level:  [("Adj Close","SPY"), ("Close","SPY"), ...]
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer ("Adj Close", ticker) if present; else first column under "Adj Close"; else try "Close"
        if ("Adj Close", ticker) in df.columns:
            adj = df[("Adj Close", ticker)].copy()
        elif "Adj Close" in df.columns.get_level_values(0):
            adj = df.xs("Adj Close", axis=1, level=0).iloc[:, 0].copy()
        elif ("Close", ticker) in df.columns:
            adj = df[("Close", ticker)].copy()
        elif "Close" in df.columns.get_level_values(0):
            adj = df.xs("Close", axis=1, level=0).iloc[:, 0].copy()
        else:
            return pd.DataFrame()
        out = pd.DataFrame({"Adj Close": adj})
    else:
        if "Adj Close" in df.columns:
            out = df[["Adj Close"]].copy()
        elif "Close" in df.columns:
            out = df[["Close"]].rename(columns={"Close": "Adj Close"}).copy()
        else:
            return pd.DataFrame()

    out.index = pd.to_datetime(out.index)

    # Reindex to business days to make plotting smoother; fill gaps safely
    bidx = pd.date_range(out.index.min().normalize(), out.index.max().normalize(), freq="B")
    out = out.reindex(bidx)

    out["Adj Close"] = pd.to_numeric(out["Adj Close"], errors="coerce")
    out["Adj Close"] = out["Adj Close"].ffill().bfill()

    # Final cleanup
    out = out.dropna()
    return out

def _render_benchmark_tab(statement: ParsedStatement, hide_values: bool):
    st.markdown("### 📉 Benchmark Comparison")

    # -----------------------------
    # Determine statement date range
    # -----------------------------
    raw_start = (statement.period_start or "").strip()
    raw_end = (statement.period_end or "").strip()

    def _try_parse(s: str) -> Optional[datetime]:
        if not s:
            return None
        for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        return None

    stmt_start = _try_parse(raw_start)
    stmt_end = _try_parse(raw_end)

    if not stmt_start or not stmt_end:
        stmt_start, stmt_end = _infer_statement_period_dates(statement)

    if not stmt_start or not stmt_end:
        st.warning("Could not infer a valid date range from the statement (no dated activity found).")
        return

    # Convert to datetime if needed (database may return date objects)
    from datetime import date as date_type
    if isinstance(stmt_start, date_type) and not isinstance(stmt_start, datetime):
        stmt_start = datetime.combine(stmt_start, datetime.min.time())
    if isinstance(stmt_end, date_type) and not isinstance(stmt_end, datetime):
        stmt_end = datetime.combine(stmt_end, datetime.min.time())

    stmt_start = stmt_start.replace(hour=0, minute=0, second=0, microsecond=0)
    stmt_end = stmt_end.replace(hour=0, minute=0, second=0, microsecond=0)

    # -----------------------------
    # Inputs
    # -----------------------------
    c1, c2, c3, c4 = st.columns([2.2, 1, 1, 0.8])

    with c1:
        bench = st.text_input(
            "Benchmark ticker (e.g., SPY, QQQ, VT, TLT)",
            value=st.session_state.get("benchmark_ticker", "SPY"),
            key="benchmark_ticker_input",
        ).strip().upper()

    with c2:
        from_date = st.date_input(
            "From",
            value=st.session_state.get("benchmark_from", stmt_start.date()),
            min_value=stmt_start.date(),
            max_value=stmt_end.date(),
            key="benchmark_from_input",
        )

    with c3:
        to_date = st.date_input(
            "To",
            value=st.session_state.get("benchmark_to", stmt_end.date()),
            min_value=stmt_start.date(),
            max_value=stmt_end.date(),
            key="benchmark_to_input",
        )

    with c4:
        st.write("")
        if st.button("Compare", width='stretch', key="benchmark_compare_btn"):
            st.session_state["benchmark_ticker"] = bench
            st.session_state["benchmark_from"] = from_date
            st.session_state["benchmark_to"] = to_date

    bench = st.session_state.get("benchmark_ticker", bench).strip().upper()
    from_date = st.session_state.get("benchmark_from", from_date)
    to_date = st.session_state.get("benchmark_to", to_date)

    if not bench:
        st.info("Enter a benchmark ticker.")
        return

    if from_date > to_date:
        from_date, to_date = to_date, from_date

    start_dt = datetime.combine(from_date, datetime.min.time())
    end_dt = datetime.combine(to_date, datetime.min.time())

    # -----------------------------
    # Portfolio return (use validated calculation)
    # -----------------------------
    portfolio_return, portfolio_label = _compute_portfolio_return_for_benchmark(statement)

    if portfolio_return is None:
        st.warning("Portfolio return unavailable: missing Starting/Ending NAV and no valid data in statement.")
        return

    # -----------------------------
    # Benchmark series
    # -----------------------------
    bench_df = _fetch_benchmark_adjclose(bench, start_dt, end_dt)
    if bench_df.empty:
        st.error(f"No data returned for benchmark: {bench}")
        return

    bench_df = bench_df[(bench_df.index >= start_dt) & (bench_df.index <= end_dt)].copy()
    if bench_df.empty:
        st.error(f"No benchmark prices found within {start_dt.date()} to {end_dt.date()}.")
        return

    bench_series = bench_df["Adj Close"]
    if isinstance(bench_series, pd.DataFrame):
        bench_series = bench_series.iloc[:, 0]
    bench_series = pd.to_numeric(bench_series, errors="coerce").dropna()

    if bench_series.empty or len(bench_series) < 2:
        st.error("Benchmark has too few valid price points in the selected range. Expand the date range.")
        return

    start_px = float(bench_series.iloc[0])
    end_px = float(bench_series.iloc[-1])
    if start_px == 0.0:
        st.error("Benchmark start price is zero; cannot compute benchmark return.")
        return

    bench_return = (end_px / start_px) - 1.0
    bench_norm = (bench_series / start_px) * 100.0

    # -----------------------------
    # Metrics
    # -----------------------------
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Selected Period", f"{from_date.isoformat()} → {to_date.isoformat()}")
    with k2:
        st.metric(portfolio_label, f"{portfolio_return*100:+.2f}%" if not hide_values else "•••")
    with k3:
        st.metric(f"Benchmark ({bench})", f"{bench_return*100:+.2f}%" if not hide_values else "•••")
    with k4:
        ex = portfolio_return - bench_return
        st.metric("Excess Return", f"{ex*100:+.2f}%" if not hide_values else "•••")

    if hide_values:
        st.info("Privacy is enabled; chart is hidden. Disable to view the chart.")
        return

    # -----------------------------
    # Portfolio visible series (line across dates)
    # -----------------------------
    port_end = 100.0 * (1.0 + portfolio_return)
    port_series = pd.Series(
        np.linspace(100.0, port_end, num=len(bench_norm)),
        index=bench_norm.index,
        name=portfolio_label,
    )

    # -----------------------------
    # Debug (expandable)
    # -----------------------------
    with st.expander("Debug: benchmark/portfolio series", expanded=False):
        st.write(
            {
                "bench": bench,
                "range": f"{bench_norm.index.min().date()} → {bench_norm.index.max().date()}",
                "bench_points": int(len(bench_norm)),
                "bench_min": float(bench_norm.min()),
                "bench_max": float(bench_norm.max()),
                "port_min": float(port_series.min()),
                "port_max": float(port_series.max()),
                "bench_head": bench_norm.head(3).to_dict(),
                "bench_tail": bench_norm.tail(3).to_dict(),
                "port_head": port_series.head(3).to_dict(),
                "port_tail": port_series.tail(3).to_dict(),
            }
        )
        st.dataframe(pd.DataFrame({"bench_norm": bench_norm, "port_series": port_series}).head(10))

    # -----------------------------
    # Plot
    # -----------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bench_norm.index, y=bench_norm.values, mode="lines", name=f"{bench} (Adj Close)"))
    fig.add_trace(go.Scatter(x=port_series.index, y=port_series.values, mode="lines", name=portfolio_label))

    bench_min = float(bench_norm.min())
    bench_max = float(bench_norm.max())
    port_min = float(port_series.min())
    port_max = float(port_series.max())

    y_min = min(bench_min, port_min)
    y_max = max(bench_max, port_max)

    pad = max(0.5, (y_max - y_min) * 0.20)
    fig.update_yaxes(range=[y_min - pad, y_max + pad])

    fig.update_layout(
        title=f"Normalized Performance (Start = 100): {portfolio_label} vs {bench}",
        template="plotly_dark",
        height=520,
        yaxis_title="Index (100 = start)",
        margin=dict(l=30, r=20, t=50, b=30),
    )

    st.plotly_chart(
        fig,
        width='stretch',
        key=f"benchmark_chart_{bench}_{bench_norm.index.min().date()}_{bench_norm.index.max().date()}",
    )

def get_statement_summary_metrics(statement: ParsedStatement) -> Dict[str, Any]:
    """
    Summary metrics used by the Overview cards + waterfall.

    If deposits/withdrawals in the statement period are zero (common for YTD exports),
    treat Starting NAV as the invested base for True Profit calculations and display.
    """
    nav = statement.nav
    change = statement.change_in_nav

    # Primary source: Change in NAV (already reconciled by IBKR)
    total_dividends = float(change.dividends or 0.0)
    total_interest = float(change.interest or 0.0)
    total_withholding = float(change.withholding_tax or 0.0)  # often already negative in IBKR CSV
    total_fees = float((change.other_fees or 0.0) + (change.commissions or 0.0))  # often already negative

    # Fallback: sum records if Change in NAV section is empty
    if total_dividends == 0.0 and statement.dividends:
        total_dividends = float(sum(d.amount for d in statement.dividends))
    if total_interest == 0.0 and statement.interest:
        total_interest = float(sum(i.amount for i in statement.interest))
    if total_withholding == 0.0 and statement.withholding_tax:
        total_withholding = float(sum(w.amount for w in statement.withholding_tax))
    if total_fees == 0.0 and statement.fees:
        total_fees = float(sum(f.amount for f in statement.fees))

    # Starting / ending values
    starting_value = float(change.starting_value or 0.0)
    ending_value = float(change.ending_value or 0.0)
    if ending_value == 0.0:
        ending_value = float(nav.total or 0.0)

    # Net deposits (raw)
    net_deposits_raw = float(change.deposits_withdrawals or 0.0)
    if net_deposits_raw == 0.0 and statement.deposits_withdrawals:
        net_deposits_raw = float(sum(d.amount for d in statement.deposits_withdrawals))

    # Display/invested base: use raw deposits if non-zero, else starting NAV
    used_starting_as_deposits = False
    invested_base = net_deposits_raw
    if abs(invested_base) < 1e-9:
        if starting_value > 0.0:
            invested_base = starting_value
            used_starting_as_deposits = True
        elif ending_value > 0.0:
            # last-resort baseline (keeps UI non-zero even if starting is missing)
            invested_base = ending_value
            used_starting_as_deposits = True

    # True profit relative to invested base
    true_profit = (ending_value - invested_base) if invested_base != 0.0 else 0.0
    true_profit_pct = (true_profit / invested_base * 100.0) if invested_base != 0.0 else 0.0

    # Capital gain (market appreciation) = Mark-to-Market (from Change in NAV)
    capital_gain = float(change.mark_to_market or 0.0)

    return {
        "starting_value": starting_value,
        "ending_value": ending_value,

        # Keep both: raw cashflows and the UI baseline
        "net_deposits_raw": net_deposits_raw,
        "net_deposits": invested_base,
        "used_starting_as_deposits": used_starting_as_deposits,

        "true_profit": true_profit,
        "true_profit_pct": true_profit_pct,
        "capital_gain": capital_gain,

        "dividends": total_dividends,
        "interest": total_interest,
        "withholding_tax": total_withholding,
        "fees": total_fees,

        "twr": nav.twr_percent,
        "period_start": statement.period_start,
        "period_end": statement.period_end,
    }


# =============================================================================
# CHART RENDERERS
# =============================================================================

def render_waterfall_chart(metrics: Dict[str, Any], hide: bool = False):
    """Render performance attribution waterfall chart."""
    if hide:
        st.info("🔒 Chart hidden - disable privacy mode to view")
        return

    deposits_display = float(metrics.get("net_deposits", 0.0) or 0.0)
    used_starting = bool(metrics.get("used_starting_as_deposits", False))
    deposits_label = "Starting NAV" if used_starting else "Net Deposits"

    capital_gain = float(metrics.get("capital_gain", 0.0) or 0.0)
    dividends = float(metrics.get("dividends", 0.0) or 0.0)
    interest = float(metrics.get("interest", 0.0) or 0.0)

    # Withholding is often negative already; fees may be negative as well
    withholding_tax = float(metrics.get("withholding_tax", 0.0) or 0.0)
    fees = float(metrics.get("fees", 0.0) or 0.0)

    # Make "Tax & Fees" a single negative contribution
    tax_fees = withholding_tax - abs(fees)

    ending_value = float(metrics.get("ending_value", 0.0) or 0.0)

    fig = go.Figure(
        go.Waterfall(
            name="",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "total"],
            x=[deposits_label, "Capital Gain", "Dividends", "Interest", "Tax & Fees", "NAV"],
            y=[deposits_display, capital_gain, dividends, interest, tax_fees, 0],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#ff5252"}},
            increasing={"marker": {"color": "#00c853"}},
            totals={"marker": {"color": "#4ecdc4"}},
            text=[
                fmt(deposits_display, "money_int"),
                fmt(capital_gain, "signed"),
                fmt(dividends, "signed"),
                fmt(interest, "signed"),
                fmt(tax_fees, "signed"),
                fmt(ending_value, "money_int"),
            ],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="How Your Portfolio Grew",
        showlegend=False,
        height=400,
        template="plotly_dark",
        yaxis_tickformat="$,0f",
    )

    st.plotly_chart(fig, width='stretch', key="waterfall_growth_chart")


def render_allocation_chart(cash: float, stocks: float, bonds: float = 0, options: float = 0, hide: bool = False):
    """Render asset allocation pie chart."""
    if hide:
        st.info("🔒 Chart hidden - disable privacy mode to view")
        return

    labels = []
    values = []

    if stocks > 0:
        labels.append("Stocks")
        values.append(stocks)
    if bonds > 0:
        labels.append("Bonds")
        values.append(bonds)
    if options > 0:
        labels.append("Options")
        values.append(options)
    if cash > 0:
        labels.append("Cash")
        values.append(cash)

    if not values:
        st.info("No allocation data")
        return

    fig = px.pie(
        values=values,
        names=labels,
        hole=0.4,
        color_discrete_sequence=['#4ecdc4', '#ff6b6b', '#feca57', '#48dbfb']
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        showlegend=False,
        height=300,
        margin=dict(t=20, b=20),
        template='plotly_dark'
    )

    st.plotly_chart(fig, width='stretch')


def render_top_dividends_chart(dividends_by_symbol: Dict[str, float], hide: bool = False, top_n: int = 10):
    """Render top dividend payers bar chart."""
    if hide:
        st.info("🔒 Chart hidden - disable privacy mode to view")
        return

    if not dividends_by_symbol:
        st.info("No dividend data")
        return

    # Sort and get top N
    sorted_divs = sorted(dividends_by_symbol.items(), key=lambda x: x[1], reverse=True)[:top_n]

    df = pd.DataFrame(sorted_divs, columns=['Symbol', 'Dividends'])
    df = df.sort_values('Dividends')

    fig = px.bar(
        df,
        x='Dividends',
        y='Symbol',
        orientation='h',
        color='Dividends',
        color_continuous_scale='Greens'
    )

    fig.update_layout(
        title=f"Top {min(top_n, len(sorted_divs))} Dividend Payers",
        showlegend=False,
        height=350,
        template='plotly_dark',
        xaxis_tickformat='$,.0f',
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, width='stretch')


def render_positions_pnl_chart(positions: List[Dict], hide: bool = False):
    """Render P&L by position bar chart."""
    if hide:
        st.info("🔒 Chart hidden - disable privacy mode to view")
        return

    if not positions:
        st.info("No position data")
        return

    df = pd.DataFrame(positions)
    if 'unrealized_pnl' not in df.columns:
        st.info("No P&L data available")
        return

    # Get top 5 winners and losers
    top = df.nlargest(5, 'unrealized_pnl')
    bottom = df.nsmallest(5, 'unrealized_pnl')

    pnl = pd.concat([top, bottom])[['symbol', 'unrealized_pnl']].drop_duplicates()
    pnl = pnl.sort_values('unrealized_pnl')
    pnl['color'] = pnl['unrealized_pnl'].apply(lambda x: 'green' if x >= 0 else 'red')

    fig = px.bar(
        pnl,
        x='unrealized_pnl',
        y='symbol',
        orientation='h',
        color='color',
        color_discrete_map={'green': '#00c853', 'red': '#ff5252'}
    )

    fig.update_layout(
        title="Top Winners & Losers",
        showlegend=False,
        height=350,
        margin=dict(t=40, b=40),
        xaxis_tickformat="$,.0f",
        template='plotly_dark'
    )

    st.plotly_chart(fig, width='stretch')


# =============================================================================
# AI PORTFOLIO BUILDER TAB
# =============================================================================

# portfolio_tab.py — replace this method entirely
def _render_portfolio_builder_tab():
    """Render the AI Portfolio Builder sub-tab."""
    if not PORTFOLIO_BUILDER_AVAILABLE:
        st.error("Portfolio Builder module not available")
        return

    st.markdown("#### 🎯 Build Custom Portfolios with AI")
    st.caption("Uses your stock universe + portfolio engine + AI chat (if configured).")

    # -----------------------------
    # Session state
    # -----------------------------
    ss = st.session_state
    if "portfolio_chat_history" not in ss:
        ss.portfolio_chat_history = []
    if "portfolio_builder_df" not in ss:
        ss.portfolio_builder_df = None
    if "portfolio_builder_context" not in ss:
        ss.portfolio_builder_context = None  # kept for compatibility, not required
    if "portfolio_builder_last_result" not in ss:
        ss.portfolio_builder_last_result = None
    if "portfolio_builder_last_errors" not in ss:
        ss.portfolio_builder_last_errors = []
    if "portfolio_builder_loaded_info" not in ss:
        ss.portfolio_builder_loaded_info = None
    if "portfolio_builder_loaded_holdings" not in ss:
        ss.portfolio_builder_loaded_holdings = None

    # -----------------------------
    # Header controls
    # -----------------------------
    c1, c2, c3, c4 = st.columns([1.6, 1, 0.5, 2])

    with c1:
        if st.button("📊 Load Stock Universe", type="primary", key="pb_load_universe_btn"):
            msg = st.empty()
            msg.info("Loading stock universe...")
            df = get_latest_stock_universe()
            if df is None or df.empty:
                msg.error("No stock data found. Run your screener / data pipeline first.")
            else:
                ss.portfolio_builder_df = df
                ss.portfolio_builder_context = build_portfolio_ai_context(df)
                msg.success(f"✅ Loaded {len(df)} stocks")
                ss.portfolio_builder_last_result = None
                ss.portfolio_builder_last_errors = []
                ss.portfolio_builder_loaded_info = None
                ss.portfolio_builder_loaded_holdings = None

    with c2:
        if st.button("🔄 Refresh", key="pb_refresh_universe_btn"):
            with st.spinner("Refreshing data..."):
                df = get_latest_stock_universe(force_refresh=True)
                if df is None or df.empty:
                    st.error("No stock data found.")
                else:
                    ss.portfolio_builder_df = df
                    ss.portfolio_builder_context = build_portfolio_ai_context(df)
                    ss.portfolio_builder_last_result = None
                    ss.portfolio_builder_last_errors = []
                    ss.portfolio_builder_loaded_info = None
                    ss.portfolio_builder_loaded_holdings = None
                    st.success(f"✅ Refreshed {len(df)} stocks")
                    st.rerun()

    with c3:
        if st.button("🗑️", key="pb_clear_chat_btn", help="Clear chat"):
            ss.portfolio_chat_history = []
            st.rerun()

    with c4:
        if ss.portfolio_builder_df is not None and isinstance(ss.portfolio_builder_df, pd.DataFrame) and not ss.portfolio_builder_df.empty:
            df = ss.portfolio_builder_df
            sector_col = "sector" if "sector" in df.columns else None
            price_col = "price" if "price" in df.columns else None
            n_sectors = int(df[sector_col].nunique()) if sector_col else 0
            n_with_price = int(df[price_col].notna().sum()) if price_col else 0
            st.caption(f"✅ {len(df)} stocks | {n_sectors} sectors | {n_with_price} with prices")

    if ss.portfolio_builder_df is None or not isinstance(ss.portfolio_builder_df, pd.DataFrame) or ss.portfolio_builder_df.empty:
        st.info("Click **Load Stock Universe** to enable the AI Portfolio Builder.")
        return

    df = ss.portfolio_builder_df

    # -----------------------------
    # Render current loaded/built portfolio (top)
    # -----------------------------
    def _result_to_df(result_obj) -> pd.DataFrame:
        rows = []
        for h in getattr(result_obj, "holdings", []) or []:
            rows.append({
                "ticker": getattr(h, "ticker", None),
                "weight_pct": getattr(h, "weight_pct", None),
                "shares": getattr(h, "shares", None),
                "value": getattr(h, "value", None),
                "sector": getattr(h, "sector", None),
                "score": getattr(h, "score", None),
                "signal_type": getattr(h, "signal_type", None),
                "rationale": getattr(h, "rationale", None),
            })
        out = pd.DataFrame(rows)
        if not out.empty and "weight_pct" in out.columns:
            out = out.sort_values("weight_pct", ascending=False)
        return out

    if ss.portfolio_builder_last_errors:
        with st.expander("⚠️ Portfolio build issues", expanded=True):
            for e in ss.portfolio_builder_last_errors:
                st.error(str(e))

    if ss.portfolio_builder_loaded_info is not None and ss.portfolio_builder_loaded_holdings is not None:
        info = ss.portfolio_builder_loaded_info
        holdings_df = ss.portfolio_builder_loaded_holdings

        st.markdown("##### 📂 Loaded Portfolio")
        top = st.columns(5)
        top[0].metric("Name", str(info.get("name", "")))
        top[1].metric("Objective", str(info.get("objective", "")))
        top[2].metric("Risk", str(info.get("risk_level", "")))
        top[3].metric("# Holdings", int(info.get("num_holdings", 0) or 0))
        top[4].metric("Total Value", f"{float(info.get('total_value', 0) or 0):,.2f}")

        if isinstance(holdings_df, pd.DataFrame) and not holdings_df.empty:
            st.dataframe(holdings_df, width='stretch', hide_index=True)
        st.markdown("---")

    elif ss.portfolio_builder_last_result is not None and getattr(ss.portfolio_builder_last_result, "success", False):
        result = ss.portfolio_builder_last_result
        st.markdown("##### ✅ Built Portfolio (Latest)")
        top = st.columns(5)
        top[0].metric("Objective", str(getattr(result, "objective", "")))
        top[1].metric("Risk", str(getattr(result, "risk_level", "")))
        top[2].metric("# Holdings", int(getattr(result, "num_holdings", 0) or 0))
        top[3].metric("# Sectors", int(getattr(result, "num_sectors", 0) or 0))
        top[4].metric("Investable %", f"{float(getattr(result, 'investable_percent', 0) or 0):.1f}%")

        holdings_df = _result_to_df(result)
        if not holdings_df.empty:
            st.dataframe(holdings_df, width='stretch', hide_index=True)

        with st.expander("💾 Save this portfolio", expanded=False):
            name = st.text_input("Name", key="pb_save_name")
            desc = st.text_area("Description (optional)", key="pb_save_desc")
            if st.button("Save", key="pb_save_btn"):
                if not name.strip():
                    st.error("Name is required.")
                else:
                    pid = save_portfolio(name.strip(), result, description=(desc.strip() if desc else None))
                    if pid is None:
                        st.error("Save failed.")
                    else:
                        st.success(f"Saved (ID={pid}).")
        st.markdown("---")

    # -----------------------------
    # Main layout
    # -----------------------------
    col_main, col_chat = st.columns([2, 1])

    with col_main:
        st.markdown("##### 📈 Universe Summary")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Stocks", len(df))

        if "total_score" in df.columns:
            m2.metric("Avg Score", f"{float(df['total_score'].mean()):.1f}")
        else:
            m2.metric("Avg Score", "N/A")

        if "signal_type" in df.columns:
            buy_count = int(df["signal_type"].isin(["STRONG_BUY", "BUY"]).sum())
            sell_count = int(df["signal_type"].isin(["STRONG_SELL", "SELL"]).sum())
            m3.metric("🟢 Buy Signals", buy_count)
            m4.metric("🔴 Sell Signals", sell_count)
        else:
            m3.metric("🟢 Buy Signals", "N/A")
            m4.metric("🔴 Sell Signals", "N/A")

        if "sector" in df.columns:
            m5.metric("Sectors", int(df["sector"].nunique()))
        else:
            m5.metric("Sectors", "N/A")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 By Sector", "🏆 Top Stocks", "📋 Full List", "🎨 Templates"])

        with tab1:
            sector_summary = get_sector_summary(df)
            if isinstance(sector_summary, pd.DataFrame) and not sector_summary.empty:
                st.dataframe(sector_summary, width='stretch')
                fig = px.bar(
                    sector_summary.reset_index(),
                    x="sector",
                    y="Count",
                    color="Avg Score",
                    color_continuous_scale="RdYlGn",
                    title="Stocks by Sector (colored by avg score)"
                )
                fig.update_layout(height=350, template="plotly_dark")
                st.plotly_chart(fig, width='stretch', key="pb_sector_bar")
            else:
                st.info("No sector data available")

        with tab2:
            categories = get_top_stocks_by_category(df, n=10)
            if isinstance(categories, dict):
                for cat_name, cat_df in categories.items():
                    if isinstance(cat_df, pd.DataFrame) and not cat_df.empty:
                        with st.expander(f"📈 {cat_name}", expanded=(cat_name == "Top Rated")):
                            st.dataframe(cat_df, width='stretch', hide_index=True)

        with tab3:
            search = st.text_input("🔍 Search stocks", key="pb_universe_search")
            display_df = df.copy()

            sym_col = "ticker" if "ticker" in display_df.columns else ("symbol" if "symbol" in display_df.columns else None)
            name_col = "name" if "name" in display_df.columns else None

            if search and sym_col:
                mask = display_df[sym_col].astype(str).str.contains(search.upper(), na=False)
                if name_col:
                    mask |= display_df[name_col].astype(str).str.contains(search, case=False, na=False)
                display_df = display_df[mask]

            display_cols = []
            for c in ["ticker", "symbol", "name", "sector", "total_score", "signal_type", "price"]:
                if c in display_df.columns:
                    display_cols.append(c)

            if display_cols:
                st.dataframe(display_df[display_cols].head(200), width='stretch', hide_index=True)
                if len(display_df) > 200:
                    st.caption(f"Showing 200 of {len(display_df)} rows")
            else:
                st.dataframe(display_df.head(200), width='stretch', hide_index=True)

        with tab4:
            st.markdown("##### 📋 Portfolio Templates")

            for template_key, template_config in PORTFOLIO_TEMPLATES.items():
                display_name = template_config.get("name", template_key)
                with st.expander(f"📦 {display_name}"):
                    st.markdown(f"**Description:** {template_config.get('description', 'N/A')}")
                    intent = template_config.get("intent")

                    if intent is None:
                        st.warning("Template intent not available (portfolio engine disabled).")
                        continue

                    if st.button(f"Build {display_name}", key=f"pb_build_{template_key}"):
                        with st.spinner("Building portfolio..."):
                            result, errors = build_portfolio_from_intent(intent, df)

                        ss.portfolio_builder_loaded_info = None
                        ss.portfolio_builder_loaded_holdings = None

                        ss.portfolio_builder_last_result = result
                        ss.portfolio_builder_last_errors = errors or []

                        if result is not None and getattr(result, "success", False):
                            ss.portfolio_chat_history.append({
                                "role": "assistant",
                                "content": f"Built template **{display_name}** with {len(getattr(result, 'holdings', []) or [])} holdings."
                            })
                        else:
                            ss.portfolio_chat_history.append({
                                "role": "assistant",
                                "content": f"Failed to build **{display_name}**. Check errors in the page."
                            })
                        st.rerun()

    with col_chat:
        st.markdown("##### 💬 Chat with AI")

        chat_container = st.container(height=420)
        with chat_container:
            for msg in ss.portfolio_chat_history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    st.markdown(f"**You:** {content}")
                else:
                    st.markdown(f"**AI:** {content}")

        user_input = st.chat_input("Ask about portfolios...", key="pb_chat_input")
        if user_input:
            ss.portfolio_chat_history.append({"role": "user", "content": user_input})
            with st.spinner("Thinking..."):
                response = get_ai_response(
                    user_request=user_input,
                    df=df,
                    history=ss.portfolio_chat_history,
                    model_id=None
                )
            ss.portfolio_chat_history.append({"role": "assistant", "content": response})
            st.rerun()




        st.markdown("---")
        # JSON Import Section
        st.markdown("##### 📁 Import Portfolio from JSON")
        
        # Initialize session state for JSON data
        if 'json_portfolio_data' not in ss:
            ss.json_portfolio_data = None
        
        uploaded_json = st.file_uploader("Upload JSON file", type=['json'], key="pb_json_upload")
        
        # Parse and store in session state when file is uploaded
        if uploaded_json is not None:
            try:
                import json as json_lib
                uploaded_json.seek(0)  # Reset file pointer
                data = json_lib.load(uploaded_json)
                ss.json_portfolio_data = data  # Store in session state
                
                if isinstance(data, list) and len(data) > 0:
                    # Parse JSON - extract tickers and weights and store in session state
                    # Support both 'ticker' and 'symbol' keys (IBKR format uses 'symbol')
                    ss.json_tickers = []
                    ss.json_weights = {}
                    ss.json_names = {}
                    
                    for item in data:
                        # Get ticker from either 'ticker' or 'symbol' field
                        ticker = item.get('ticker') or item.get('symbol') or ''
                        ticker = ticker.upper().strip()
                        
                        if ticker and ticker.isalpha():  # Skip numeric symbols like '2454'
                            ss.json_tickers.append(ticker)
                            ss.json_weights[ticker] = float(item.get('weight', 0) or 0)
                            # Get name from 'name' or 'originalName'
                            ss.json_names[ticker] = item.get('name') or item.get('originalName') or ''
                    
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_tickers = []
                    for t in ss.json_tickers:
                        if t not in seen:
                            seen.add(t)
                            unique_tickers.append(t)
                    ss.json_tickers = unique_tickers
                    
                    st.success(f"✅ Found {len(ss.json_tickers)} tickers in JSON")
                    
                    # Check which tickers are in the database
                    if ss.portfolio_builder_df is not None:
                        available = set(ss.portfolio_builder_df['ticker'].str.upper())
                        ss.json_found_tickers = [t for t in ss.json_tickers if t in available]
                        ss.json_missing_tickers = [t for t in ss.json_tickers if t not in available]
                        
                        # Show summary
                        col_info1, col_info2, col_info3 = st.columns(3)
                        col_info1.metric("Total in JSON", len(ss.json_tickers))
                        col_info2.metric("Found in DB", len(ss.json_found_tickers))
                        col_info3.metric("Missing", len(ss.json_missing_tickers))
                        
                        if ss.json_missing_tickers:
                            with st.expander(f"⚠️ {len(ss.json_missing_tickers)} tickers not in database", expanded=True):
                                missing_df = pd.DataFrame([
                                    {'Ticker': t, 'Name': ss.json_names.get(t, ''), 'Weight': ss.json_weights.get(t, 0)}
                                    for t in ss.json_missing_tickers
                                ])
                                st.dataframe(missing_df, hide_index=True)
                                
                                if st.button("➕ Add Missing Tickers to Database", key="pb_add_missing"):
                                    import yfinance as yf
                                    import psycopg2
                                    
                                    progress = st.progress(0)
                                    status = st.empty()
                                    added = []
                                    failed = []
                                    
                                    try:
                                        conn = psycopg2.connect(
                                            host='localhost', port=5432, 
                                            dbname='alpha_platform', user='alpha', password='alpha_secure_2024'
                                        )
                                        cur = conn.cursor()
                                        
                                        for idx, ticker in enumerate(ss.json_missing_tickers):
                                            status.text(f"Fetching {ticker}...")
                                            progress.progress((idx + 1) / len(ss.json_missing_tickers))
                                            
                                            try:
                                                stock = yf.Ticker(ticker)
                                                info = stock.info
                                                
                                                if info.get('symbol'):
                                                    cur.execute("""
                                                        INSERT INTO fundamentals (ticker, sector, market_cap, pe_ratio, dividend_yield, date)
                                                        VALUES (%s, %s, %s, %s, %s, CURRENT_DATE)
                                                        ON CONFLICT (ticker, date) DO UPDATE SET
                                                            sector = EXCLUDED.sector,
                                                            market_cap = EXCLUDED.market_cap,
                                                            pe_ratio = EXCLUDED.pe_ratio,
                                                            dividend_yield = EXCLUDED.dividend_yield
                                                    """, (
                                                        ticker,
                                                        info.get('sector', 'Unknown'),
                                                        info.get('marketCap'),
                                                        info.get('trailingPE'),
                                                        info.get('dividendYield')
                                                    ))
                                                    added.append(ticker)
                                                else:
                                                    failed.append(f"{ticker}: Not found")
                                            except Exception as fetch_err:
                                                failed.append(f"{ticker}: {str(fetch_err)[:30]}")
                                        
                                        conn.commit()
                                        conn.close()
                                        
                                        status.empty()
                                        progress.empty()
                                        
                                        if added:
                                            st.success(f"✅ Added {len(added)} tickers: {', '.join(added)}")
                                            st.info("🔄 Click 'Refresh' to reload stock universe with new tickers")
                                        if failed:
                                            st.warning(f"⚠️ Failed: {', '.join(failed[:5])}")
                                    except Exception as db_err:
                                        st.error(f"Database error: {db_err}")
                        
                        # Build portfolio buttons
                        if ss.json_found_tickers:
                            st.markdown("---")
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                build_btn = st.button("🚀 Build Portfolio from JSON", key="pb_build_from_json", type="primary")
                            
                            with col2:
                                # Preview table
                                preview_df = pd.DataFrame([
                                    {'Ticker': t, 'Weight': f"{ss.json_weights.get(t, 0):.1f}%", 'Status': '✅ In DB' if t in ss.json_found_tickers else '❌ Missing'}
                                    for t in ss.json_tickers
                                ])
                                st.dataframe(preview_df, hide_index=True, height=200)
                            
                            if build_btn:
                                with st.spinner("Building portfolio..."):
                                    try:
                                        from dashboard.portfolio_engine import PortfolioIntent, PortfolioEngine
                                        
                                        intent = PortfolioIntent(
                                            objective='custom',
                                            risk_level='moderate',
                                            portfolio_value=100000,
                                            tickers_include=ss.json_found_tickers,
                                            restrict_to_tickers=True,
                                            fully_invested=True,
                                            equal_weight=False,
                                            max_position_pct=15
                                        )
                                        
                                        engine = PortfolioEngine(ss.portfolio_builder_df)
                                        result = engine.build_portfolio(intent, user_request=f"Custom Portfolio from JSON ({len(ss.json_found_tickers)} stocks)")
                                        
                                        if result and result.success:
                                            # Apply custom weights from JSON
                                            for h in result.holdings:
                                                ticker_upper = h.ticker.upper()
                                                if ticker_upper in ss.json_weights:
                                                    h.weight_pct = ss.json_weights[ticker_upper]
                                                    h.value = 100000 * h.weight_pct / 100
                                            
                                            # Store in session state - BOTH keys for display
                                            ss.portfolio_builder_last_result = result
                                            ss.last_portfolio_result = result
                                            ss.portfolio_builder_loaded_info = None
                                            ss.portfolio_builder_loaded_holdings = None
                                            ss.portfolio_builder_last_errors = []
                                            
                                            st.success(f"✅ Built portfolio with {result.num_holdings} stocks!")
                                            st.rerun()
                                        else:
                                            warnings = getattr(result, 'warnings', []) if result else ['Build failed']
                                            st.error(f"Failed: {', '.join(warnings)}")
                                    except Exception as build_err:
                                        st.error(f"Build error: {build_err}")
                                        import traceback
                                        st.code(traceback.format_exc())
                        else:
                            st.error("No valid tickers found in database. Add missing tickers first.")
                    else:
                        st.warning("⚠️ Load stock universe first to validate tickers")
                else:
                    st.error("Invalid JSON format - expected array of objects with 'ticker' and 'weight'")
            except Exception as e:
                st.error(f"JSON parse error: {e}")
        
        st.markdown("---")
        
        # Strategy-Based Portfolio Builder
        st.markdown("##### 🎯 Build by Strategy")
        
        # Define available strategies with display names
        STRATEGY_OPTIONS = {
            "momentum": "📈 Momentum - AI-validated trend following",
            "tech_growth": "💻 Tech Growth - High-growth tech with AI validation", 
            "aggressive_growth": "🚀 Aggressive Growth - Maximum growth with AI signals",
            "biotech_growth": "🧬 Biotech Growth - FDA/clinical catalyst focus",
            "value": "💎 Value - AI-identified undervalued companies",
            "income": "💰 Income - Dividend-focused with AI quality check",
            "balanced": "⚖️ Balanced - AI-optimized multi-factor",
            "conservative": "🛡️ Conservative - Capital preservation",
            "quality": "✨ Quality Factor - High-quality companies",
            "shariah": "☪️ Shariah Compliant - Islamic finance screening",
        }
        
        col_strat1, col_strat2 = st.columns([2, 1])
        
        with col_strat1:
            selected_strategy = st.selectbox(
                "Select Strategy",
                options=list(STRATEGY_OPTIONS.keys()),
                format_func=lambda x: STRATEGY_OPTIONS[x],
                key="pb_strategy_select"
            )
        
        with col_strat2:
            strategy_value = st.number_input(
                "Portfolio Value ($)",
                min_value=10000,
                max_value=10000000,
                value=100000,
                step=10000,
                key="pb_strategy_value"
            )
        
        col_strat3, col_strat4, col_strat5 = st.columns(3)
        
        with col_strat3:
            strategy_risk = st.selectbox(
                "Risk Level",
                options=["conservative", "moderate", "aggressive"],
                index=1,
                key="pb_strategy_risk"
            )
        
        with col_strat4:
            strategy_holdings = st.slider(
                "Max Holdings",
                min_value=10,
                max_value=50,
                value=25,
                key="pb_strategy_holdings"
            )
        
        with col_strat5:
            strategy_max_pos = st.slider(
                "Max Position %",
                min_value=3,
                max_value=15,
                value=8,
                key="pb_strategy_max_pos"
            )
        
        if st.button("🚀 Build Strategy Portfolio", key="pb_build_strategy", type="primary"):
            if ss.portfolio_builder_df is None:
                st.error("Load stock universe first!")
            else:
                with st.spinner(f"Building {STRATEGY_OPTIONS[selected_strategy].split(' - ')[0]} portfolio..."):
                    try:
                        from dashboard.portfolio_engine import PortfolioIntent, PortfolioEngine
                        
                        intent = PortfolioIntent(
                            objective=selected_strategy,
                            risk_level=strategy_risk,
                            portfolio_value=strategy_value,
                            max_holdings=strategy_holdings,
                            max_position_pct=strategy_max_pos,
                            fully_invested=True,
                            equal_weight=False
                        )
                        
                        engine = PortfolioEngine(ss.portfolio_builder_df)
                        result = engine.build_portfolio(intent, user_request=f"{STRATEGY_OPTIONS[selected_strategy]} portfolio")
                        
                        if result and result.success:
                            ss.portfolio_builder_last_result = result
                            ss.last_portfolio_result = result
                            ss.portfolio_builder_loaded_info = None
                            ss.portfolio_builder_loaded_holdings = None
                            ss.portfolio_builder_last_errors = []
                            
                            st.success(f"✅ Built {result.num_holdings} holdings using {STRATEGY_OPTIONS[selected_strategy].split(' - ')[0]}")
                            st.rerun()
                        else:
                            warnings = getattr(result, 'warnings', []) if result else ['Build failed']
                            st.error(f"Failed: {', '.join(str(w) for w in warnings)}")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        st.markdown("---")
        
        st.markdown("##### 💾 Saved Portfolios")

        saved_df = get_saved_portfolios()
        if isinstance(saved_df, pd.DataFrame) and not saved_df.empty:
            for _, row in saved_df.iterrows():
                pid = int(row.get("id"))
                pname = str(row.get("name", f"Portfolio {pid}"))
                cL, cR = st.columns([4, 1])
                with cL:
                    if st.button(f"📂 {pname}", key=f"pb_load_{pid}"):
                        info, holdings = load_portfolio(pid)
                        if info is None or holdings is None:
                            st.error("Load failed.")
                        else:
                            ss.portfolio_builder_loaded_info = info
                            ss.portfolio_builder_loaded_holdings = holdings
                            ss.portfolio_builder_last_errors = []
                            
                            # Re-build portfolio using engine for proper display
                            try:
                                from dashboard.portfolio_engine import PortfolioIntent, PortfolioEngine
                                
                                # Get tickers from saved holdings
                                saved_tickers = holdings['ticker'].tolist() if 'ticker' in holdings.columns else []
                                saved_weights = dict(zip(holdings['ticker'], holdings['weight_pct'])) if 'ticker' in holdings.columns and 'weight_pct' in holdings.columns else {}
                                
                                if saved_tickers and ss.portfolio_builder_df is not None:
                                    intent = PortfolioIntent(
                                        objective=info.get('objective', 'custom'),
                                        risk_level=info.get('risk_level', 'moderate'),
                                        portfolio_value=float(info.get('total_value', 100000) or 100000),
                                        tickers_include=saved_tickers,
                                        restrict_to_tickers=True,
                                        fully_invested=True,
                                        equal_weight=False,
                                        max_position_pct=15
                                    )
                                    
                                    engine = PortfolioEngine(ss.portfolio_builder_df)
                                    result = engine.build_portfolio(intent, user_request=f"Loaded: {pname}")
                                    
                                    if result and result.success:
                                        # Apply saved weights
                                        for h in result.holdings:
                                            if h.ticker in saved_weights:
                                                h.weight_pct = float(saved_weights[h.ticker])
                                                h.value = float(info.get('total_value', 100000) or 100000) * h.weight_pct / 100
                                        
                                        ss.portfolio_builder_last_result = result
                                        ss.last_portfolio_result = result
                                    else:
                                        ss.portfolio_builder_last_result = None
                                        ss.last_portfolio_result = None
                                else:
                                    ss.portfolio_builder_last_result = None
                                    ss.last_portfolio_result = None
                            except Exception as conv_err:
                                import logging
                                logging.error(f"Could not rebuild saved portfolio: {conv_err}")
                                ss.portfolio_builder_last_result = None
                                ss.last_portfolio_result = None
                            
                            st.rerun()
                with cR:
                    if st.button("🗑️", key=f"pb_del_{pid}"):
                        delete_portfolio(pid)
                        st.rerun()
        else:
            st.caption("No saved portfolios")

    # ===========================================================================
    # ENHANCED PORTFOLIO DISPLAY - OUTSIDE COLUMNS (FULL WIDTH)
    # ===========================================================================
    if ENHANCED_DISPLAY_AVAILABLE and hasattr(st.session_state, 'last_portfolio_result'):
        result = st.session_state.last_portfolio_result

        if result and hasattr(result, 'success') and result.success:
            if hasattr(result, 'holdings') and result.holdings:
                st.markdown("---")
                st.markdown("## 📊 Portfolio Analysis")

                try:
                    render_comprehensive_stock_table(result)
                except Exception as e:
                    st.error(f"Error displaying portfolio table: {e}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

                try:
                    render_save_portfolio_section(result)
                except Exception as e:
                    st.error(f"Error displaying save section: {e}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())


# =============================================================================
# MAIN RENDER FUNCTIONS
# =============================================================================

def render_portfolio_tab():
    """Render the Portfolio tab with main tabs: Current Portfolio and AI Builder."""

    st.subheader("💼 Portfolio Management")

    # Initialize tab selection in session state
    if 'portfolio_active_tab' not in st.session_state:
        st.session_state.portfolio_active_tab = 0

    # Tab selector using radio (more reliable than st.tabs for state persistence)
    tab_options = ["📊 Current Portfolio", "🎯 AI Portfolio Builder", "🤖 AI Portfolio Manager", "📈 Backtest Portfolio"]

    # Create tab-like radio buttons
    selected_tab = st.radio(
        "Select View",
        tab_options,
        index=st.session_state.portfolio_active_tab,
        horizontal=True,
        key="portfolio_tab_selector",
        label_visibility="collapsed"
    )

    # Update session state
    st.session_state.portfolio_active_tab = tab_options.index(selected_tab)

    st.markdown("---")

    # Render selected tab content
    if selected_tab == "📊 Current Portfolio":
        _render_current_portfolio()
    elif selected_tab == "🎯 AI Portfolio Builder":
        _render_portfolio_builder_tab()
    elif selected_tab == "🤖 AI Portfolio Manager":
        render_ai_portfolio_manager_tab()
    elif selected_tab == "📈 Backtest Portfolio":
        if BACKTEST_AVAILABLE:
            render_backtest_tab()
        else:
            st.error("""
            Portfolio Backtester not available.
            
            **Required:**
            1. Install yfinance: `pip install yfinance`
            2. Place `portfolio_backtester.py` in `dashboard/` folder
            3. Place `backtest_tab.py` in `dashboard/` folder
            """)
    else:
        render_ai_portfolio_manager_tab()


def _render_current_portfolio():
    """Render the current IBKR portfolio with Activity Statement data."""

    # Initialize session state
    if 'last_upload' not in st.session_state:
        st.session_state['last_upload'] = None
    if 'parsed_statement' not in st.session_state:
        st.session_state['parsed_statement'] = None

    inject_css()

    # -------------------------------------------------------------------------
    # Sidebar Settings
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### ⚙️ Display Settings")
        hide_values = st.toggle("🙈 Hide Values", key="privacy_mode", help="Hide sensitive financial values")

        st.markdown("---")
        st.markdown("### 📁 IBKR Statement")

        uploaded = st.file_uploader("Upload Activity Statement CSV", type=['csv'], key="statement_upload")

    # -------------------------------------------------------------------------
    # Connection Settings
    # -------------------------------------------------------------------------
    with st.expander("⚙️ IBKR Connection", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            host = st.text_input("Host", value="127.0.0.1", key="host")
        with c2:
            port = st.number_input("Port", value=7496, key="port")
        with c3:
            live = st.checkbox("🔴 IBKR Live", value=False, key="live")

    # -------------------------------------------------------------------------
    # Account Selection
    # -------------------------------------------------------------------------
    col1, col2 = st.columns([3, 1])
    with col1:
        accounts, error = get_accounts_only(host, port)

        if error:
            st.warning(f"⚠️ IBKR: {error}")
            accounts = ["Manual"]
        if not accounts:
            accounts = ["Manual"]

        # Add "All" option if multiple accounts
        if len(accounts) > 1 and "All" not in accounts:
            accounts = ["All"] + list(accounts)

        account = st.selectbox("📊 Account", accounts, key="account_sel")

        # Show warning if "All" is selected
        if account == "All":
            st.info("📌 Select a specific account to load live data. 'All' shows combined data from uploaded statements only.")

    # Detect account change - clear statement if account changed
    previous_account = st.session_state.get('current_account')
    if previous_account and previous_account != account:
        # Account changed - clear the old statement and live data
        st.session_state['parsed_statement'] = None
        st.session_state['ibkr_live_data'] = None
        st.session_state['ibkr_live_account'] = None
        st.session_state['last_upload'] = None  # Allow re-upload
    st.session_state['current_account'] = account

    with col2:
        st.write("")
        if st.button("🔄 Refresh", width='stretch'):
            st.cache_data.clear()

            # Handle "All" account selection - load from database
            if account == "All":
                statement = None
                # Try database first (consolidated account U17994267)
                if PORTFOLIO_DB_AVAILABLE:
                    statement = load_statement_from_db("U17994267")
                    if statement:
                        st.session_state['parsed_statement'] = statement
                        st.session_state['ibkr_live_data'] = None
                        st.success("✅ Loaded consolidated data from database")

                # Fallback to CSV
                if not statement:
                    statement = load_statement_data("All")
                    if statement:
                        st.session_state['parsed_statement'] = statement
                        st.session_state['ibkr_live_data'] = None
                        st.success("✅ Loaded consolidated statement from CSV")
                    else:
                        st.warning("⚠️ No consolidated data found. Import MULTI CSV to database.")
                st.rerun()

            # Fetch live IBKR data on refresh for specific account
            if account and account not in ("Manual", "All"):
                with st.spinner("Fetching live IBKR data..."):
                    ibkr_data = load_ibkr_data(account, host, port, live=True)

                    if ibkr_data and not ibkr_data.get("error"):
                        # Store live data in session state
                        st.session_state['ibkr_live_data'] = ibkr_data
                        st.session_state['ibkr_live_account'] = account
                        st.session_state['ibkr_live_timestamp'] = datetime.now()
                        st.success("✅ Live data loaded from IBKR")
                    elif ibkr_data and ibkr_data.get("error"):
                        st.warning(f"⚠️ IBKR: {ibkr_data.get('error')}")
            st.session_state['parsed_statement'] = None
            st.rerun()

    # Handle file upload (with duplicate prevention) - also import to database
    if uploaded:
        upload_key = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get('last_upload') != upload_key:
            # Parse the uploaded file
            content = uploaded.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8-sig')  # Handle BOM

            parser = IBKRStatementParser()
            parsed = parser.parse(content)

            st.session_state['parsed_statement'] = parsed
            st.session_state['last_upload'] = upload_key

            # Save file
            uploaded.seek(0)
            if save_uploaded_statement(uploaded, account if account not in ("Manual", "All") else None):
                st.success(f"✅ Loaded: {uploaded.name} ({len(parsed.deposits_withdrawals)} deposits, {len(parsed.dividends)} dividends)")

            # Also import to database if available
            if PORTFOLIO_DB_AVAILABLE:
                try:
                    repo = PortfolioRepository()
                    results = repo.import_statement(parsed, uploaded.name)
                    st.info(f"💾 Imported to database: {results.get('account_id')}")
                except Exception as e:
                    st.warning(f"⚠️ Database import failed: {e}")

            st.rerun()

    # -------------------------------------------------------------------------
    # Load Data - Priority: Live IBKR + DB History > DB Only > Session State > CSV Files
    # -------------------------------------------------------------------------

    # First, check for live IBKR data in session state (from Refresh button)
    ibkr_data = st.session_state.get('ibkr_live_data')
    ibkr_live_account = st.session_state.get('ibkr_live_account')
    ibkr_live_timestamp = st.session_state.get('ibkr_live_timestamp')

    # Only use live data if it's for the current account
    if ibkr_live_account != account:
        ibkr_data = None

    # Also load live data if checkbox is checked
    if account not in ("Manual", "All") and live and not ibkr_data:
        with st.spinner("Loading IBKR data..."):
            ibkr_data = load_ibkr_data(account, host, port, live)
            if ibkr_data.get("error"):
                st.warning(f"IBKR: {ibkr_data['error']}")
                ibkr_data = None

    # If we have live IBKR data, use it with database history
    statement = None
    if ibkr_data and not ibkr_data.get("error"):
        # Try to load historical data from DATABASE first (preferred)
        historical_statement = None
        if PORTFOLIO_DB_AVAILABLE:
            historical_statement = load_statement_from_db(account)
            if historical_statement:
                st.caption(f"💾 Using database history: {len(historical_statement.deposits_withdrawals)} deposits, {len(historical_statement.dividends)} dividends")

        # Fallback to CSV if database not available
        if not historical_statement:
            historical_statement = load_statement_data(account)
            if historical_statement:
                st.caption(f"📂 Using CSV history: {len(historical_statement.deposits_withdrawals)} deposits")

        if not historical_statement:
            st.caption(f"⚠️ No historical data for {account} - import CSV to database for True Profit")

        # Create live statement, optionally merged with historical data
        statement = _create_statement_from_live_ibkr(ibkr_data, account, ibkr_live_timestamp, historical_statement)

        if statement:
            st.session_state['parsed_statement'] = statement
            if historical_statement:
                st.success(f"🔴 **LIVE DATA** from IBKR + History (as of {ibkr_live_timestamp.strftime('%H:%M:%S') if ibkr_live_timestamp else 'now'})")
            else:
                st.info(f"🔴 **LIVE DATA** from IBKR (as of {ibkr_live_timestamp.strftime('%H:%M:%S') if ibkr_live_timestamp else 'now'})")

    # If no live data, check session state
    if statement is None:
        statement = st.session_state.get('parsed_statement')
        # Verify session statement matches current account
        if statement and hasattr(statement, 'account_info'):
            stmt_account = statement.account_info.account_id
            # Clean account ID for comparison
            stmt_account_clean = stmt_account.split(" (")[0].strip()
            if stmt_account_clean != account and account != "All":
                statement = None  # Clear mismatched statement

    # If still no statement, try to load from DATABASE (preferred) or CSV files
    if statement is None and account != "Manual":
        # Try database first
        if PORTFOLIO_DB_AVAILABLE:
            # For "All", use consolidated function; otherwise use specific account
            db_account = account  # load_statement_from_db handles "All" internally
            statement = load_statement_from_db(db_account)
            if statement:
                st.session_state['parsed_statement'] = statement
                st.caption(f"💾 Loaded from database: {statement.account_info.account_id}")

        # Fallback to CSV files
        if not statement:
            statement = load_statement_data(account)
            if statement:
                st.session_state['parsed_statement'] = statement
                st.caption(f"📂 Loaded from CSV: {statement.account_info.account_id}")

    # -------------------------------------------------------------------------
    # Display Data
    # -------------------------------------------------------------------------
    if statement is None:
        st.info("👆 Upload an IBKR Activity Statement CSV or click **Refresh** to load live data from IBKR")
        st.markdown("""
        ### How to get your Activity Statement:
        
        1. Log into **IBKR Account Management**
        2. Go to **Reports → Statements**
        3. Select **Activity** statement type
        4. Choose your date range
        5. Export as **CSV**
        6. Upload here
        
        **OR** click the **🔄 Refresh** button to load live portfolio data directly from IBKR.
        
        The Activity Statement includes:
        - Net Asset Value breakdown
        - Open positions
        - Trades
        - Dividends received
        - Interest income
        - Deposits & Withdrawals
        """)
        return

    # Account Info Header
    acc_info = statement.account_info
    st.markdown(f"### 📊 {acc_info.name or 'Portfolio'}")
    st.caption(f"Account: {acc_info.account_id} | Period: {statement.period_start} to {statement.period_end}")

    # Show parse errors if any
    if statement.parse_errors:
        with st.expander(f"⚠️ {len(statement.parse_errors)} Parse Warnings", expanded=False):
            for err in statement.parse_errors[:10]:
                st.text(err)

    # Get summary metrics
    metrics = get_statement_summary_metrics(statement)

    # -------------------------------------------------------------------------
    # Performance Summary
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📊 Portfolio Performance")

    # Row 1: Key metrics
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        true_profit = metrics['true_profit']
        true_profit_pct = metrics['true_profit_pct']
        color = "#00c853" if true_profit >= 0 else "#ff5252"
        st.markdown(f"""
        <div class="perf-card {'positive' if true_profit >= 0 else 'negative'}">
            <div style="font-size: 0.85em; color: #888;">💰 True Profit</div>
            <div style="font-size: 1.5em; font-weight: bold; color: {color};">
                {fmt(true_profit_pct, 'pct', hide_values)}
            </div>
            <div style="color: {color};">{fmt(true_profit, 'signed', hide_values)}</div>
            <div style="font-size: 0.75em; color: #666;">NAV - Deposits</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="perf-card neutral">
            <div style="font-size: 0.85em; color: #888;">📊 Current NAV</div>
            <div style="font-size: 1.5em; font-weight: bold;">{fmt(metrics['ending_value'], 'money_int', hide_values)}</div>
            <div style="font-size: 0.75em; color: #666;">Total portfolio value</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="perf-card neutral">
            <div style="font-size: 0.85em; color: #888;">💵 Net Invested</div>
            <div style="font-size: 1.5em; font-weight: bold;">{fmt(metrics['net_deposits'], 'money_int', hide_values)}</div>
            <div style="font-size: 0.75em; color: #666;">Deposits - Withdrawals</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        capital_gain = metrics['capital_gain']
        cap_pct = (capital_gain / metrics['net_deposits'] * 100) if metrics['net_deposits'] != 0 else 0
        cap_color = "#00c853" if capital_gain >= 0 else "#ff5252"
        st.markdown(f"""
        <div class="perf-card {'positive' if capital_gain >= 0 else 'negative'}">
            <div style="font-size: 0.85em; color: #888;">📈 Capital Gain</div>
            <div style="font-size: 1.5em; font-weight: bold; color: {cap_color};">
                {fmt(cap_pct, 'pct', hide_values)}
            </div>
            <div style="color: {cap_color};">{fmt(capital_gain, 'signed', hide_values)}</div>
            <div style="font-size: 0.75em; color: #666;">Price appreciation</div>
        </div>
        """, unsafe_allow_html=True)

    # Row 2: Income breakdown
    st.markdown("---")
    st.markdown("### 💰 Income Breakdown")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        div_color = "#00c853" if metrics['dividends'] > 0 else "#888"
        st.markdown(f"""
        <div class="perf-card {'positive' if metrics['dividends'] > 0 else 'neutral'}">
            <div style="font-size: 0.85em; color: #888;">🎯 Dividends</div>
            <div style="font-size: 1.3em; font-weight: bold; color: {div_color};">{fmt(metrics['dividends'], 'money', hide_values)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        int_color = "#00c853" if metrics['interest'] > 0 else "#888"
        st.markdown(f"""
        <div class="perf-card {'positive' if metrics['interest'] > 0 else 'neutral'}">
            <div style="font-size: 0.85em; color: #888;">🏦 Interest</div>
            <div style="font-size: 1.3em; font-weight: bold; color: {int_color};">{fmt(metrics['interest'], 'money', hide_values)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="perf-card negative">
            <div style="font-size: 0.85em; color: #888;">🏛️ Withholding Tax</div>
            <div style="font-size: 1.3em; font-weight: bold; color: #ff5252;">{fmt(metrics['withholding_tax'], 'money', hide_values)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="perf-card negative">
            <div style="font-size: 0.85em; color: #888;">💳 Fees & Commissions</div>
            <div style="font-size: 1.3em; font-weight: bold; color: #ff5252;">{fmt(-abs(metrics['fees']), 'money', hide_values)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        total_income = metrics['dividends'] + metrics['interest'] + metrics['withholding_tax'] - abs(metrics['fees'])
        inc_color = "#00c853" if total_income > 0 else "#ff5252"
        st.markdown(f"""
        <div class="perf-card {'positive' if total_income > 0 else 'negative'}">
            <div style="font-size: 0.85em; color: #888;">📊 Net Income</div>
            <div style="font-size: 1.3em; font-weight: bold; color: {inc_color};">{fmt(total_income, 'money', hide_values)}</div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Tabs for detailed views
    # -------------------------------------------------------------------------
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📈 Overview", "📋 Positions", "💹 Trades", "📉 Benchmark", "🎯 Dividends", "💵 Cash Flows"]
    )

    with tab1:
        # (unchanged)
        col1, col2 = st.columns([2, 1])
        with col1:
            render_waterfall_chart(metrics, hide_values)
        with col2:
            st.markdown("#### 📊 Performance Summary")
            dep = metrics['net_deposits']
            st.markdown(f"""
            | Component | Amount | % of Deposits |
            |-----------|--------|---------------|
            | 💵 Deposits | {fmt(dep, 'money_int', hide_values)} | 100% |
            | 📈 Capital Gain | {fmt(metrics['capital_gain'], 'signed', hide_values)} | {metrics['capital_gain']/dep*100 if dep else 0:+.1f}% |
            | 🎯 Dividends | {fmt(metrics['dividends'], 'money', hide_values)} | {metrics['dividends']/dep*100 if dep else 0:+.2f}% |
            | 🏦 Interest | {fmt(metrics['interest'], 'money', hide_values)} | {metrics['interest']/dep*100 if dep else 0:+.2f}% |
            | 🏛️ Taxes/Fees | {fmt(metrics['withholding_tax'] - abs(metrics['fees']), 'money', hide_values)} | {(metrics['withholding_tax'] - abs(metrics['fees']))/dep*100 if dep else 0:+.2f}% |
            | **📊 NAV** | **{fmt(metrics['ending_value'], 'money_int', hide_values)}** | **{metrics['true_profit_pct']:+.1f}%** |
            """)
            if metrics['twr'] != 0:
                st.markdown("---")
                st.metric("TWR (IBKR)", f"{metrics['twr']:.2f}%" if not hide_values else "•••")
                st.caption("Time-Weighted Return from IBKR")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🎨 Asset Allocation")
            nav = statement.nav
            render_allocation_chart(nav.cash, nav.stock, nav.bonds, nav.options, hide_values)
        with col2:
            st.markdown("#### 🏆 Top Dividend Payers")
            divs_by_symbol = aggregate_dividends_by_symbol(statement.dividends)
            render_top_dividends_chart(divs_by_symbol, hide_values)

    with tab2:
        _render_positions_tab(statement, ibkr_data, hide_values)

    with tab3:
        _render_trades_tab(statement, hide_values)

    with tab4:
        _render_benchmark_tab(statement, hide_values)

    with tab5:
        _render_dividends_tab(statement, hide_values)

    with tab6:
        _render_cash_flows_tab(statement, hide_values)




def _compute_portfolio_return_for_benchmark(statement: ParsedStatement) -> Tuple[Optional[float], str]:
    """
    Returns (portfolio_return_decimal, label).
    Robust fallback order:
      1) Use IBKR TWR% if available AND reasonable
      2) Use True Profit % (ending - deposits) / deposits if deposits > 0
      3) Use (ending - starting - net_deposits) / starting if starting > 0

    TWR validation: If TWR sign differs significantly from True Profit sign, TWR is likely wrong
    (common issue with consolidated multi-account statements due to internal transfers).
    """
    nav = statement.nav
    change = statement.change_in_nav

    ending = float(change.ending_value or 0.0) or float(nav.total or 0.0)
    net_deposits = float(change.deposits_withdrawals or 0.0)
    starting = float(change.starting_value or 0.0)

    # Calculate True Profit % for comparison
    true_profit = ending - net_deposits
    true_profit_pct = (true_profit / abs(net_deposits)) if net_deposits != 0 else 0.0

    twr = float(nav.twr_percent or 0.0)

    # Validate TWR: if TWR shows large loss but True Profit shows gain (or vice versa), TWR is wrong
    # This happens with consolidated accounts due to internal transfers
    twr_decimal = twr / 100.0

    if twr != 0.0:
        # Check if TWR and True Profit have wildly different signs/magnitudes
        # If True Profit is positive but TWR is very negative (or vice versa), TWR is wrong
        twr_positive = twr_decimal > 0
        profit_positive = true_profit_pct > 0

        # If signs match and magnitudes are somewhat similar, use TWR
        if twr_positive == profit_positive:
            # Signs match - use TWR
            return twr_decimal, "Portfolio (IBKR TWR)"
        elif abs(twr_decimal) < 0.1 and abs(true_profit_pct) < 0.1:
            # Both are small, minor discrepancy - use TWR
            return twr_decimal, "Portfolio (IBKR TWR)"
        else:
            # Signs differ significantly - TWR is wrong, use True Profit %
            # This is common with consolidated multi-account statements
            pass  # Fall through to use True Profit calculation

    # Use True Profit % as the primary metric when TWR is unavailable or invalid
    if net_deposits != 0 and ending != 0:
        return true_profit_pct, "Portfolio (True Profit %)"

    # Fallback: if starting value exists, use period return
    if starting > 0 and ending != 0:
        return (ending - starting - net_deposits) / starting, "Portfolio (Period Return)"

    return None, "Portfolio (return unavailable)"




def _render_positions_tab(statement: ParsedStatement, ibkr_data: Optional[Dict], hide_values: bool):
    """Render the positions tab."""
    positions = statement.open_positions

    # Get dividend totals by symbol
    divs_by_symbol = aggregate_dividends_by_symbol(statement.dividends)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 📋 Open Positions ({len(positions)})")
    with col2:
        if positions:
            # Download button
            pos_data = [asdict(p) for p in positions]
            df = pd.DataFrame(pos_data)
            st.download_button(
                "📥 CSV",
                df.to_csv(index=False),
                f"positions_{statement.account_info.account_id}.csv",
                "text/csv",
                width='stretch'
            )

    if not positions:
        st.info("No open positions")
        return

    # Convert to list of dicts and enrich with dividends
    pos_list = []
    for pos in positions:
        p = asdict(pos)
        p['dividends_received'] = divs_by_symbol.get(pos.symbol, 0)
        p['total_return'] = pos.unrealized_pnl + p['dividends_received']
        p['total_return_pct'] = (p['total_return'] / abs(pos.cost_basis) * 100) if pos.cost_basis != 0 else 0
        pos_list.append(p)

    # Create DataFrame
    df = pd.DataFrame(pos_list)

    # Calculate weights
    total_mv = df['market_value'].sum()
    df['weight'] = (df['market_value'] / total_mv * 100).round(2) if total_mv > 0 else 0

    # Get trading signals
    signals = get_trading_signals()
    df['signal'] = df['symbol'].apply(lambda s: signals.get(s, {}).get('signal_type', ''))
    df['signal_status'] = df.apply(
        lambda r: get_signal_status(r['quantity'], r['signal'])[1],
        axis=1
    )

    # Filters
    c1, c2, c3 = st.columns(3)
    with c1:
        search = st.text_input("🔍 Search", key="pos_search")
    with c2:
        status = st.selectbox("Status", ["All", "Aligned", "Conflicts", "No Signal"], key="pos_status")
    with c3:
        sort_by = st.selectbox("Sort By", ["Market Value", "P&L $", "Return %", "Dividends", "Total Return"], key="pos_sort")

    # Apply filters
    fdf = df.copy()
    if search:
        fdf = fdf[fdf['symbol'].str.contains(search.upper(), na=False)]
    if status == "Aligned":
        fdf = fdf[fdf['signal_status'] == 'Aligned']
    elif status == "Conflicts":
        fdf = fdf[fdf['signal_status'] == 'Conflict']
    elif status == "No Signal":
        fdf = fdf[fdf['signal_status'] == 'No Signal']

    # Sort
    sort_map = {
        "Market Value": "market_value",
        "P&L $": "unrealized_pnl",
        "Return %": "unrealized_pnl_pct",
        "Dividends": "dividends_received",
        "Total Return": "total_return"
    }
    sort_col = sort_map.get(sort_by, "market_value")
    fdf = fdf.sort_values(sort_col, ascending=False)

    # Summary metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Cost", fmt(fdf['cost_basis'].sum(), 'money_int', hide_values))
    with c2:
        st.metric("Market Value", fmt(fdf['market_value'].sum(), 'money_int', hide_values))
    with c3:
        st.metric("Unrealized P&L", fmt(fdf['unrealized_pnl'].sum(), 'signed', hide_values))
    with c4:
        st.metric("Dividends", fmt(fdf['dividends_received'].sum(), 'money', hide_values))
    with c5:
        st.metric("Total Return", fmt(fdf['total_return'].sum(), 'signed', hide_values))

    # Display table
    display_cols = ['symbol', 'quantity', 'cost_price', 'current_price', 'cost_basis', 'market_value',
                   'unrealized_pnl', 'unrealized_pnl_pct', 'dividends_received', 'total_return',
                   'total_return_pct', 'weight', 'signal', 'signal_status']
    display_cols = [c for c in display_cols if c in fdf.columns]
    display_df = fdf[display_cols].copy()

    col_names = {
        'symbol': 'Symbol', 'quantity': 'Qty', 'cost_price': 'Avg Cost', 'current_price': 'Current',
        'cost_basis': 'Cost Basis', 'market_value': 'Mkt Value', 'unrealized_pnl': 'Unreal P&L',
        'unrealized_pnl_pct': 'Return %', 'dividends_received': 'Dividends',
        'total_return': 'Total Return', 'total_return_pct': 'Total %',
        'weight': 'Weight %', 'signal': 'Signal', 'signal_status': 'Status'
    }
    display_df.columns = [col_names.get(c, c) for c in display_df.columns]

    if hide_values:
        for col in ['Avg Cost', 'Current', 'Cost Basis', 'Mkt Value', 'Unreal P&L', 'Return %',
                   'Dividends', 'Total Return', 'Total %', 'Weight %']:
            if col in display_df.columns:
                display_df[col] = '•••'
        st.dataframe(display_df, width='stretch', height=400, hide_index=True)
    else:
        st.dataframe(
            display_df, width='stretch', height=400, hide_index=True,
            column_config={
                "Avg Cost": st.column_config.NumberColumn(format="$%.2f"),
                "Current": st.column_config.NumberColumn(format="$%.2f"),
                "Cost Basis": st.column_config.NumberColumn(format="$%.0f"),
                "Mkt Value": st.column_config.NumberColumn(format="$%.0f"),
                "Unreal P&L": st.column_config.NumberColumn(format="$%.0f"),
                "Return %": st.column_config.NumberColumn(format="%.2f%%"),
                "Dividends": st.column_config.NumberColumn(format="$%.2f"),
                "Total Return": st.column_config.NumberColumn(format="$%.0f"),
                "Total %": st.column_config.NumberColumn(format="%.2f%%"),
                "Weight %": st.column_config.NumberColumn(format="%.1f%%"),
            }
        )

    # Charts
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🥧 Top Holdings")
        top = fdf.nlargest(10, 'market_value')[['symbol', 'weight']].copy()
        other = 100 - top['weight'].sum()
        if other > 0:
            top = pd.concat([top, pd.DataFrame({'symbol': ['Other'], 'weight': [other]})])

        if not hide_values:
            fig = px.pie(top, values='weight', names='symbol', hole=0.4)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(showlegend=False, height=350, margin=dict(t=20, b=20), template='plotly_dark')
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("🔒 Hidden")

    with col2:
        st.markdown("#### 📊 P&L by Position")
        render_positions_pnl_chart(fdf.to_dict('records'), hide_values)

    # Signal summary
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ Aligned", len(fdf[fdf['signal_status'] == 'Aligned']))
    c2.metric("⚠️ Conflicts", len(fdf[fdf['signal_status'] == 'Conflict']))
    c3.metric("➖ No Signal", len(fdf[fdf['signal_status'] == 'No Signal']))
    coverage = (len(fdf) - len(fdf[fdf['signal_status'] == 'No Signal'])) / max(len(fdf), 1) * 100
    c4.metric("📊 Coverage", f"{coverage:.0f}%")


def _render_trades_tab(statement: ParsedStatement, hide_values: bool):
    """
    Trades tab:
    - Shows only symbols that have at least one CLOSED leg in the period.
    - Uses ALL in-period legs for Qty Bought/Sold and Bought/Sold $.
    - Uses CLOSED legs only for realized P&L, commissions, timestamp, and trade-count.
    """
    trades = statement.trades or []

    def _is_close_leg(t: Trade) -> bool:
        code = (t.code or "").upper()
        return ("C" in code) or ((t.realized_pnl or 0.0) != 0.0)

    # Identify closed symbols and closed trades
    closed_trades = [t for t in trades if t.symbol and _is_close_leg(t)]
    closed_symbols = set(t.symbol for t in closed_trades)

    st.markdown(f"### 💹 Trade History ({len(closed_trades)} trades)")

    if not closed_trades:
        st.info("No closed trades in this period")
        return

    # Aggregate using ALL legs but only for symbols that have a close
    trades_for_agg = [t for t in trades if t.symbol in closed_symbols]
    trades_by_symbol = aggregate_trades_by_symbol(trades_for_agg)

    # Summary metrics (from closed legs only, consistent with the headline)
    total_realized = sum((t.realized_pnl or 0.0) for t in closed_trades)
    total_comm = sum(abs(t.commission or 0.0) for t in closed_trades)
    net_pnl = total_realized - total_comm

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Realized P&L", fmt(total_realized, "signed", hide_values))
    with c2:
        st.metric("Commissions", fmt(-abs(total_comm), "money", hide_values))
    with c3:
        st.metric("Net P&L", fmt(net_pnl, "signed", hide_values))
    with c4:
        winners = sum(1 for t in trades_by_symbol.values() if t["realized_pnl"] > 0)
        win_rate = winners / len(trades_by_symbol) * 100 if trades_by_symbol else 0
        st.metric("Win Rate", f"{win_rate:.0f}%")

    # Trades table
    df = pd.DataFrame(list(trades_by_symbol.values()))
    if df.empty:
        st.info("No trade data to display")
        return

    df["net_pnl"] = df["realized_pnl"] - df["commissions"]
    df = df.sort_values("realized_pnl", ascending=False)

    display_cols = [
        "timestamp", "symbol", "trades", "qty_bought", "qty_sold",
        "bought", "sold", "realized_pnl", "commissions", "net_pnl"
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    col_names = {
        "timestamp": "Timestamp",
        "symbol": "Symbol",
        "trades": "# Trades",
        "qty_bought": "Qty Bought",
        "qty_sold": "Qty Sold",
        "bought": "Bought $",
        "sold": "Sold $",
        "realized_pnl": "Realized P&L",
        "commissions": "Commissions",
        "net_pnl": "Net P&L",
    }

    display_df = df[display_cols].copy()
    display_df.columns = [col_names.get(c, c) for c in display_df.columns]

    if hide_values:
        for col in ["Bought $", "Sold $", "Realized P&L", "Commissions", "Net P&L"]:
            if col in display_df.columns:
                display_df[col] = "•••"

    st.dataframe(display_df, width='stretch', hide_index=True)

    # Individual closed trades
    with st.expander("📜 All Individual Closed Trades"):
        trade_data = []
        for t in closed_trades:
            trade_data.append({
                "Timestamp": t.date.strftime("%Y-%m-%d, %H:%M:%S") if t.date else "N/A",
                "Symbol": t.symbol,
                "Qty": t.quantity,
                "Price": t.price,
                "Proceeds": t.proceeds,
                "Commission": t.commission,
                "Realized P&L": t.realized_pnl,
                "Type": "BUY" if (t.quantity or 0) > 0 else "SELL",
                "Code": t.code or "",
            })

        tdf = pd.DataFrame(trade_data)
        if hide_values:
            for col in ["Price", "Proceeds", "Commission", "Realized P&L"]:
                if col in tdf.columns:
                    tdf[col] = "•••"
        st.dataframe(tdf, width='stretch', hide_index=True)


def _infer_statement_period_dates(statement: ParsedStatement) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Fallback period inference when statement.period_start / period_end are empty.
    Uses min/max timestamps found in any dated activity section.
    """
    dates: List[datetime] = []

    for t in (statement.trades or []):
        if t.date:
            dates.append(t.date)

    for d in (statement.dividends or []):
        if d.date:
            dates.append(d.date)

    for i in (statement.interest or []):
        if i.date:
            dates.append(i.date)

    for w in (statement.withholding_tax or []):
        if w.date:
            dates.append(w.date)

    for dw in (statement.deposits_withdrawals or []):
        if dw.date:
            dates.append(dw.date)

    for f in (statement.fees or []):
        if f.date:
            dates.append(f.date)

    if not dates:
        return None, None

    return min(dates), max(dates)

def _render_dividends_tab(statement: ParsedStatement, hide_values: bool):
    """Render dividend details per symbol."""

    dividends = statement.dividends

    st.markdown(f"### 🎯 Dividends ({len(dividends)} payments)")

    if not dividends:
        st.info("No dividend records found")
        return

    rows = []
    for d in dividends:
        rows.append({
            "Date": d.date.strftime("%Y-%m-%d") if d.date else "",
            "Symbol": d.symbol,
            "Description": d.description,
            "Amount": d.amount,
        })

    df = pd.DataFrame(rows)

    total = df["Amount"].sum()

    st.dataframe(df, width='stretch', hide_index=True)
    st.markdown(f"**Total Dividends:** {fmt(total, 'money', hide_values)}")


def _render_cash_flows_tab(statement: ParsedStatement, hide_values: bool):
    """Render the cash flows tab."""

    # Deposits & Withdrawals
    st.markdown("### 💵 Deposits & Withdrawals")

    dw = statement.deposits_withdrawals
    if dw:
        total_deposits = sum(d.amount for d in dw if d.amount > 0)
        total_withdrawals = sum(d.amount for d in dw if d.amount < 0)
        net = total_deposits + total_withdrawals

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Deposits", fmt(total_deposits, 'money', hide_values))
        with c2:
            st.metric("Withdrawals", fmt(total_withdrawals, 'money', hide_values))
        with c3:
            st.metric("Net", fmt(net, 'signed', hide_values))

        dw_data = []
        for d in dw:
            dw_data.append({
                'Date': d.date.strftime('%Y-%m-%d') if d.date else 'N/A',
                'Type': 'Deposit' if d.amount > 0 else 'Withdrawal',
                'Amount': d.amount,
                'Description': d.description[:40] + '...' if len(d.description) > 40 else d.description
            })

        if dw_data:
            dwdf = pd.DataFrame(dw_data)
            dwdf = dwdf.sort_values('Date', ascending=False)
            if hide_values:
                dwdf['Amount'] = '•••'
            st.dataframe(dwdf, width='stretch', hide_index=True)
    else:
        st.info("No deposits or withdrawals in this period")

    # Interest
    st.markdown("---")
    st.markdown("### 🏦 Interest")

    interest = statement.interest
    if interest:
        total_interest = sum(i.amount for i in interest)
        st.metric("Total Interest", fmt(total_interest, 'money', hide_values))

        int_data = []
        for i in interest:
            int_data.append({
                'Date': i.date.strftime('%Y-%m-%d') if i.date else 'N/A',
                'Amount': i.amount,
                'Description': i.description[:50] + '...' if len(i.description) > 50 else i.description
            })

        if int_data:
            idf = pd.DataFrame(int_data)
            idf = idf.sort_values('Date', ascending=False)
            if hide_values:
                idf['Amount'] = '•••'
            st.dataframe(idf, width='stretch', hide_index=True)
    else:
        st.info("No interest in this period")

    # Fees
    st.markdown("---")
    st.markdown("### 💳 Fees")

    fees = statement.fees
    if fees:
        total_fees = sum(f.amount for f in fees)
        st.metric("Total Fees", fmt(total_fees, 'money', hide_values))

        fee_data = []
        for f in fees:
            fee_data.append({
                'Date': f.date.strftime('%Y-%m-%d') if f.date else 'N/A',
                'Amount': f.amount,
                'Description': f.description[:50] + '...' if len(f.description) > 50 else f.description
            })

        if fee_data:
            fdf = pd.DataFrame(fee_data)
            fdf = fdf.sort_values('Date', ascending=False)
            if hide_values:
                fdf['Amount'] = '•••'
            st.dataframe(fdf, width='stretch', hide_index=True)
    else:
        st.info("No fees in this period")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    st.set_page_config(page_title="Portfolio", layout="wide")
    render_portfolio_tab()