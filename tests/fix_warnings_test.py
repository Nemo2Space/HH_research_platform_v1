with open('test_full_flow.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add warning suppression at the top
new_header = '''import sys
sys.path.insert(0, 'dashboard')
import warnings
import logging

# Suppress Streamlit warnings when running outside Streamlit
warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')
logging.getLogger('streamlit').setLevel(logging.ERROR)

import pandas as pd
import psycopg2
'''

old_header = '''import sys
sys.path.insert(0, 'dashboard')
import pandas as pd
import psycopg2
import logging

# Enable logging to see what's happening
logging.basicConfig(level=logging.INFO)
'''

content = content.replace(old_header, new_header)

with open('test_full_flow.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Warnings suppressed in test file")
