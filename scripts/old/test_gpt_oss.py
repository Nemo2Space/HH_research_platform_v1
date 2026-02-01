"""Test GPT-OSS response format"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI(base_url='http://172.23.193.91:8091/v1', api_key='not-needed')

prompt = """Analyze these 5 news headlines about AAPL. Return indices of RELEVANT headlines.

1. [Reuters] Apple reports record Q4 earnings
2. [Yahoo] Top 10 stocks to watch  
3. [Bloomberg] Apple unveils new AI features
4. [Benzinga] Hedge fund adjusts AAPL position
5. [CNBC] Apple upgraded by Morgan Stanley

Return ONLY JSON like: {"relevant": [1, 3, 5]}"""

response = client.chat.completions.create(
    model='Qwen3-32B-Q6_K.gguf',
    messages=[{'role': 'user', 'content': prompt}],
    temperature=0.1,
    max_tokens=200
)

print('GPT-OSS Response:')
print(response.choices[0].message.content)