# Add this to scripts/debug_gpt_oss.py or create scripts/debug_gpt_oss2.py

"""Debug GPT-OSS news filtering"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI(base_url='http://172.23.193.91:8091/v1', api_key='not-needed')

# Test 6: News filtering WITHOUT system message
print("=" * 60)
print("Test 6: News filtering NO system message")
print("=" * 60)
response = client.chat.completions.create(
    model='Qwen3-32B-Q6_K.gguf',
    messages=[
        {'role': 'user', 'content': '''Which headlines are about AAPL earnings or products? Return their numbers.

1. Apple reports Q4 earnings beat
2. Top 10 stocks to watch
3. Apple launches new iPhone
4. Market closes higher
5. Apple CEO announces AI plans

Answer with just the numbers like: 1, 3, 5'''}
    ],
    temperature=0.3,
    max_tokens=100
)
print(f"Response: '{response.choices[0].message.content}'")
print()

# Test 7: Even simpler format
print("=" * 60)
print("Test 7: Simpler format")
print("=" * 60)
response = client.chat.completions.create(
    model='Qwen3-32B-Q6_K.gguf',
    messages=[
        {'role': 'user', 'content': '''Headlines about Apple stock:
1. Apple reports Q4 earnings beat
2. Top 10 stocks to watch
3. Apple launches new iPhone
4. Market closes higher
5. Apple CEO announces AI plans

Which are relevant to Apple? Reply with numbers only: '''}
    ],
    temperature=0.3,
    max_tokens=50
)
print(f"Response: '{response.choices[0].message.content}'")
print()

# Test 8: Yes/No format per headline
print("=" * 60)
print("Test 8: Numbered relevance")
print("=" * 60)
response = client.chat.completions.create(
    model='Qwen3-32B-Q6_K.gguf',
    messages=[
        {'role': 'user', 'content': '''Rate each headline as RELEVANT or NOT for Apple stock analysis:

1. Apple reports Q4 earnings beat - 
2. Top 10 stocks to watch - 
3. Apple launches new iPhone - 
4. Market closes higher - 
5. Apple CEO announces AI plans - '''}
    ],
    temperature=0.3,
    max_tokens=200
)
print(f"Response: '{response.choices[0].message.content}'")
print()

# Test 9: Simple numbered list response
print("=" * 60)
print("Test 9: Filter with simple instruction")
print("=" * 60)
response = client.chat.completions.create(
    model='Qwen3-32B-Q6_K.gguf',
    messages=[
        {'role': 'user', 'content': '''I have 5 news headlines. Tell me which ones are specifically about Apple company (not general market news).

1. Apple reports Q4 earnings beat
2. Top 10 stocks to watch  
3. Apple launches new iPhone
4. Market closes higher
5. Apple CEO announces AI plans

Reply with the headline numbers:'''}
    ],
    temperature=0.3,
    max_tokens=50
)
print(f"Response: '{response.choices[0].message.content}'")
print()

print("=" * 60)
print("Debug complete!")
print("=" * 60)