"""
Standalone IBKR CSV Import Script
=================================

Run from project root:
    python scripts/import_ibkr_csv.py MULTI_20250321_20260123.csv
"""

import sys
import os
import csv
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.portfolio_db import PortfolioRepository


# =============================================================================
# DATA CLASSES (copied from portfolio_tab.py to avoid import issues)
# =============================================================================

@dataclass
class AccountInfo:
    name: str = ""
    account_id: str = ""
    account_type: str = ""
    base_currency: str = "USD"

@dataclass
class NAV:
    cash: float = 0.0
    stock: float = 0.0
    bonds: float = 0.0
    options: float = 0.0
    total: float = 0.0
    twr_percent: float = 0.0

@dataclass
class ChangeInNAV:
    starting_value: float = 0.0
    ending_value: float = 0.0
    deposits_withdrawals: float = 0.0
    dividends: float = 0.0
    interest: float = 0.0
    withholding_tax: float = 0.0
    fees: float = 0.0
    commissions: float = 0.0
    other_fees: float = 0.0
    mark_to_market: float = 0.0

@dataclass
class DepositWithdrawal:
    date: datetime = None
    description: str = ""
    amount: float = 0.0

@dataclass
class Dividend:
    date: datetime = None
    symbol: str = ""
    description: str = ""
    amount: float = 0.0

@dataclass
class Interest:
    date: datetime = None
    description: str = ""
    amount: float = 0.0

@dataclass
class WithholdingTax:
    date: datetime = None
    symbol: str = ""
    description: str = ""
    amount: float = 0.0

@dataclass
class Fee:
    date: datetime = None
    description: str = ""
    amount: float = 0.0

@dataclass
class Trade:
    date: datetime = None
    symbol: str = ""
    description: str = ""
    quantity: float = 0.0
    price: float = 0.0
    proceeds: float = 0.0
    commission: float = 0.0
    realized_pnl: float = 0.0
    asset_class: str = ""

@dataclass
class Position:
    symbol: str = ""
    description: str = ""
    quantity: float = 0.0
    cost_basis: float = 0.0
    cost_price: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    asset_class: str = ""

@dataclass
class ParsedStatement:
    account_info: AccountInfo = field(default_factory=AccountInfo)
    period_start: str = ""
    period_end: str = ""
    nav: NAV = field(default_factory=NAV)
    change_in_nav: ChangeInNAV = field(default_factory=ChangeInNAV)
    deposits_withdrawals: List[DepositWithdrawal] = field(default_factory=list)
    dividends: List[Dividend] = field(default_factory=list)
    interest: List[Interest] = field(default_factory=list)
    withholding_tax: List[WithholdingTax] = field(default_factory=list)
    fees: List[Fee] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    open_positions: List[Position] = field(default_factory=list)


# =============================================================================
# PARSER (simplified from portfolio_tab.py)
# =============================================================================

class IBKRStatementParser:
    """Parser for IBKR Activity Statement CSV files."""

    def __init__(self):
        self.statement = ParsedStatement()
        self._current_section = None

    def parse(self, content: str) -> ParsedStatement:
        """Parse CSV content and return ParsedStatement."""
        self.statement = ParsedStatement()

        reader = csv.reader(content.splitlines())
        for row in reader:
            if len(row) < 2:
                continue

            section = row[0].strip()
            row_type = row[1].strip() if len(row) > 1 else ""

            if row_type == "Header":
                self._current_section = section
                continue

            if row_type == "Data":
                self._parse_row(section, row)

        return self.statement

    def _parse_row(self, section: str, row: List[str]):
        """Route row to appropriate parser based on section."""
        parsers = {
            "Statement": self._parse_statement,
            "Account Information": self._parse_account_info,
            "Net Asset Value": self._parse_nav,
            "Change in NAV": self._parse_change_in_nav,
            "Deposits & Withdrawals": self._parse_deposit_withdrawal,
            "Dividends": self._parse_dividend,
            "Interest": self._parse_interest,
            "Withholding Tax": self._parse_withholding_tax,
            "Fees": self._parse_fee,
            "Trades": self._parse_trade,
            "Open Positions": self._parse_position,
        }

        parser = parsers.get(section)
        if parser:
            try:
                parser(row)
            except Exception as e:
                pass  # Skip errors silently

    def _parse_number(self, value: str) -> float:
        """Parse number, handling commas and currency symbols."""
        if not value:
            return 0.0
        clean = re.sub(r'[,$%]', '', str(value).strip())
        try:
            return float(clean)
        except:
            return 0.0

    def _parse_date(self, value: str) -> Optional[datetime]:
        """Parse date string."""
        if not value:
            return None
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%b-%y", "%B %d, %Y"):
            try:
                return datetime.strptime(value.strip(), fmt)
            except:
                continue
        return None

    def _parse_statement(self, row: List[str]):
        """Parse statement metadata."""
        if len(row) < 4:
            return
        field_name = row[2].strip()
        field_value = row[3].strip()

        if field_name == "Period":
            # Parse "March 21, 2025 - January 23, 2026"
            parts = field_value.split(" - ")
            if len(parts) == 2:
                self.statement.period_start = parts[0].strip()
                self.statement.period_end = parts[1].strip()

    def _parse_account_info(self, row: List[str]):
        """Parse account information."""
        if len(row) < 4:
            return
        field_name = row[2].strip()
        field_value = row[3].strip()

        if field_name == "Name":
            self.statement.account_info.name = field_value
        elif field_name == "Account":
            self.statement.account_info.account_id = field_value
        elif field_name == "Account Type":
            self.statement.account_info.account_type = field_value
        elif field_name == "Base Currency":
            self.statement.account_info.base_currency = field_value

    def _parse_nav(self, row: List[str]):
        """Parse NAV data."""
        if len(row) < 3:
            return

        # Check for TWR row (contains % sign)
        if len(row) >= 3 and "%" in str(row[2]):
            twr_str = row[2].strip().replace('%', '')
            self.statement.nav.twr_percent = self._parse_number(twr_str)
            return

        if len(row) < 7:
            return

        asset_class = row[2].strip().lower()
        # NAV format: Asset Class, Prior Total, Current Long, Current Short, Current Total, Change
        # Columns:    [2]         [3]          [4]           [5]            [6]            [7]
        current_total = self._parse_number(row[6]) if len(row) > 6 else 0

        if "cash" in asset_class:
            self.statement.nav.cash = current_total
        elif "stock" in asset_class:
            self.statement.nav.stock = current_total
        elif "bond" in asset_class:
            self.statement.nav.bonds = current_total
        elif "option" in asset_class:
            self.statement.nav.options = current_total
        elif asset_class == "total":
            self.statement.nav.total = current_total

    def _parse_change_in_nav(self, row: List[str]):
        """Parse Change in NAV."""
        if len(row) < 4:
            return
        field_name = row[2].strip()
        field_value = self._parse_number(row[3])

        mapping = {
            "Starting Value": "starting_value",
            "Ending Value": "ending_value",
            "Deposits & Withdrawals": "deposits_withdrawals",
            "Dividends": "dividends",
            "Interest": "interest",
            "Withholding Tax": "withholding_tax",
            "Other Fees": "other_fees",
            "Commissions": "commissions",
            "Mark-to-Market": "mark_to_market",
        }

        attr = mapping.get(field_name)
        if attr:
            setattr(self.statement.change_in_nav, attr, field_value)

    def _parse_deposit_withdrawal(self, row: List[str]):
        """Parse deposit/withdrawal."""
        if len(row) < 6:
            return

        # Skip Total rows
        if row[2].strip() == "Total":
            return

        # Standard: Currency, Settle Date, Description, Amount
        # Multi-account: Currency, Account, Settle Date, Description, Amount
        date_str = row[3].strip()

        # Check if this is multi-account format (date_str starts with U)
        if date_str.startswith("U") and len(date_str) > 1 and date_str[1:2].isdigit():
            date_str = row[4].strip() if len(row) > 4 else ""
            description = row[5].strip() if len(row) > 5 else ""
            amount = self._parse_number(row[6]) if len(row) > 6 else 0
        else:
            description = row[4].strip() if len(row) > 4 else ""
            amount = self._parse_number(row[5]) if len(row) > 5 else 0

        # Skip if no valid date
        parsed_date = self._parse_date(date_str)
        if amount != 0 and parsed_date:
            self.statement.deposits_withdrawals.append(DepositWithdrawal(
                date=parsed_date,
                description=description,
                amount=amount,
            ))

    def _parse_dividend(self, row: List[str]):
        """Parse dividend."""
        if len(row) < 6:
            return

        # Skip Total rows
        if row[2].strip() == "Total":
            return

        # Standard: Currency, Date, Description, Amount
        # Multi-account: Currency, Account, Date, Description, Amount
        date_str = row[3].strip()

        if date_str.startswith("U") and len(date_str) > 1 and date_str[1:2].isdigit():
            date_str = row[4].strip() if len(row) > 4 else ""
            description = row[5].strip() if len(row) > 5 else ""
            amount = self._parse_number(row[6]) if len(row) > 6 else 0
        else:
            description = row[4].strip() if len(row) > 4 else ""
            amount = self._parse_number(row[5]) if len(row) > 5 else 0

        # Extract symbol from description
        symbol = ""
        if description:
            match = re.match(r'^([A-Z]{1,5})\s*\(', description)
            if match:
                symbol = match.group(1)

        # Skip if no valid date
        parsed_date = self._parse_date(date_str)
        if amount != 0 and parsed_date:
            self.statement.dividends.append(Dividend(
                date=parsed_date,
                symbol=symbol,
                description=description,
                amount=amount,
            ))

    def _parse_interest(self, row: List[str]):
        """Parse interest."""
        if len(row) < 6:
            return

        # Skip Total rows
        if row[2].strip() == "Total":
            return

        date_str = row[3].strip()

        if date_str.startswith("U") and len(date_str) > 1 and date_str[1:2].isdigit():
            date_str = row[4].strip() if len(row) > 4 else ""
            description = row[5].strip() if len(row) > 5 else ""
            amount = self._parse_number(row[6]) if len(row) > 6 else 0
        else:
            description = row[4].strip() if len(row) > 4 else ""
            amount = self._parse_number(row[5]) if len(row) > 5 else 0

        # Skip if no valid date
        parsed_date = self._parse_date(date_str)
        if amount != 0 and parsed_date:
            self.statement.interest.append(Interest(
                date=parsed_date,
                description=description,
                amount=amount,
            ))

    def _parse_withholding_tax(self, row: List[str]):
        """Parse withholding tax."""
        if len(row) < 6:
            return

        # Skip Total rows
        if row[2].strip() == "Total":
            return

        date_str = row[3].strip()

        if date_str.startswith("U") and len(date_str) > 1 and date_str[1:2].isdigit():
            date_str = row[4].strip() if len(row) > 4 else ""
            description = row[5].strip() if len(row) > 5 else ""
            amount = self._parse_number(row[6]) if len(row) > 6 else 0
        else:
            description = row[4].strip() if len(row) > 4 else ""
            amount = self._parse_number(row[5]) if len(row) > 5 else 0

        # Extract symbol
        symbol = ""
        if description:
            match = re.match(r'^([A-Z]{1,5})\s*\(', description)
            if match:
                symbol = match.group(1)

        # Skip if no valid date
        parsed_date = self._parse_date(date_str)
        if amount != 0 and parsed_date:
            self.statement.withholding_tax.append(WithholdingTax(
                date=parsed_date,
                symbol=symbol,
                description=description,
                amount=amount,
            ))

    def _parse_fee(self, row: List[str]):
        """Parse fee."""
        if len(row) < 7:
            return

        # Skip Total rows
        if row[2].strip() == "Total":
            return

        date_str = row[4].strip()

        if date_str.startswith("U") and len(date_str) > 1 and date_str[1:2].isdigit():
            date_str = row[5].strip() if len(row) > 5 else ""
            description = row[6].strip() if len(row) > 6 else ""
            amount = self._parse_number(row[7]) if len(row) > 7 else 0
        else:
            description = row[5].strip() if len(row) > 5 else ""
            amount = self._parse_number(row[6]) if len(row) > 6 else 0

        # Skip if no valid date
        parsed_date = self._parse_date(date_str)
        if amount != 0 and parsed_date:
            self.statement.fees.append(Fee(
                date=parsed_date,
                description=description,
                amount=amount,
            ))

    def _parse_trade(self, row: List[str]):
        """Parse trade."""
        if len(row) < 10:
            return

        # Trades,Data,DataDiscriminator,Asset Category,Currency,Symbol,Date/Time,Quantity,T. Price,C. Price,Proceeds,Comm/Fee,Realized P/L,...
        symbol = row[5].strip() if len(row) > 5 else ""
        date_str = row[6].strip() if len(row) > 6 else ""
        quantity = self._parse_number(row[7]) if len(row) > 7 else 0
        price = self._parse_number(row[8]) if len(row) > 8 else 0
        proceeds = self._parse_number(row[10]) if len(row) > 10 else 0
        commission = self._parse_number(row[11]) if len(row) > 11 else 0
        realized_pnl = self._parse_number(row[12]) if len(row) > 12 else 0
        asset_class = row[3].strip() if len(row) > 3 else ""

        if symbol and quantity != 0:
            self.statement.trades.append(Trade(
                date=self._parse_date(date_str.split(",")[0] if "," in date_str else date_str),
                symbol=symbol,
                quantity=quantity,
                price=price,
                proceeds=proceeds,
                commission=commission,
                realized_pnl=realized_pnl,
                asset_class=asset_class,
            ))

    def _parse_position(self, row: List[str]):
        """Parse open position."""
        if len(row) < 10:
            return

        # Open Positions,Data,DataDiscriminator,Asset Category,Currency,Symbol,Quantity,Mult,Cost Price,Cost Basis,Close Price,Value,Unrealized P/L,...
        symbol = row[5].strip() if len(row) > 5 else ""
        quantity = self._parse_number(row[6]) if len(row) > 6 else 0
        cost_price = self._parse_number(row[8]) if len(row) > 8 else 0
        cost_basis = self._parse_number(row[9]) if len(row) > 9 else 0
        current_price = self._parse_number(row[10]) if len(row) > 10 else 0
        market_value = self._parse_number(row[11]) if len(row) > 11 else 0
        unrealized_pnl = self._parse_number(row[12]) if len(row) > 12 else 0
        asset_class = row[3].strip() if len(row) > 3 else ""

        if symbol and quantity != 0:
            pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis else 0
            self.statement.open_positions.append(Position(
                symbol=symbol,
                quantity=quantity,
                cost_price=cost_price,
                cost_basis=cost_basis,
                current_price=current_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=pnl_pct,
                asset_class=asset_class,
            ))


# =============================================================================
# MAIN
# =============================================================================

def import_csv(csv_path: str):
    """Import an IBKR CSV file into the database."""
    print(f"\n📂 Reading: {csv_path}")

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    print("📊 Parsing CSV...")
    parser = IBKRStatementParser()
    statement = parser.parse(content)

    print(f"\n📋 Account: {statement.account_info.account_id}")
    print(f"   Name: {statement.account_info.name}")
    print(f"   Period: {statement.period_start} → {statement.period_end}")
    print(f"   NAV: ${statement.nav.total:,.2f}")
    print(f"   Deposits/Withdrawals: {len(statement.deposits_withdrawals)}")
    print(f"   Dividends: {len(statement.dividends)}")
    print(f"   Interest: {len(statement.interest)}")
    print(f"   Trades: {len(statement.trades)}")
    print(f"   Positions: {len(statement.open_positions)}")

    print("\n💾 Importing to database...")
    repo = PortfolioRepository()
    results = repo.import_statement(statement, csv_path)

    print("\n✅ Import complete:")
    for key, value in results.items():
        if key != 'account_id' and isinstance(value, int):
            print(f"   {key}: {value} records")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
IBKR CSV Import Script
======================

Usage:
    python scripts/import_ibkr_csv.py <csv_file>

Examples:
    python scripts/import_ibkr_csv.py MULTI_20250321_20260123.csv
    python scripts/import_ibkr_csv.py U20993660_20250728_20260123.csv
        """)
        sys.exit(1)

    csv_file = sys.argv[1]

    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)

    import_csv(csv_file)