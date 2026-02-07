# Fix the NaN vs None check in backfill_returns.py
c = open('backfill_returns.py', 'r', encoding='utf-8').read()
c = c.replace(
    'if row[ret_col] is not None:',
    'if pd.notna(row[ret_col]):'
)
open('backfill_returns.py', 'w', encoding='utf-8').write(c)
print('Fixed. Re-running...')
