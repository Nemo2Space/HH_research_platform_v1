c = open('src/analytics/yf_subprocess.py','r',encoding='utf-8').read()

old_sig = 'def get_stock_history(ticker: str, period: str = "3mo") -> Optional[pd.DataFrame]:'
new_sig = 'def get_stock_history(ticker: str, period: str = "3mo", interval: str = None, timeout: int = 10) -> Optional[pd.DataFrame]:'

old_doc = '    """Get stock.history() via subprocess. Returns None on failure."""\n    script = f'
new_doc = '    """Get stock.history() via subprocess. Returns None on failure."""\n    interval_arg = f\', interval="{interval}"\' if interval else \'\'\n    script = f'

old_hist = 'hist = yf.Ticker("{ticker}").history(period="{period}")'
new_hist = 'hist = yf.Ticker("{ticker}").history(period="{period}"{interval_arg})'

c = c.replace(old_sig, new_sig)
c = c.replace(old_doc, new_doc)
c = c.replace(old_hist, new_hist)
open('src/analytics/yf_subprocess.py','w',encoding='utf-8').write(c)

import ast
ast.parse(open('src/analytics/yf_subprocess.py',encoding='utf-8').read())
print('yf_subprocess.py: interval support added, syntax OK')
