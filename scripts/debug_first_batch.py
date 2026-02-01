"""Debug why first batch returns empty"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from src.data.news import NewsCollector

# Collect real news
collector = NewsCollector()
articles = collector.collect_all_news('AAPL', days_back=7)
print(f"Collected {len(articles)} articles")

# Get first 10 articles (first batch)
batch = articles[:10]

print("\n" + "=" * 60)
print("First batch articles:")
print("=" * 60)
for i, article in enumerate(batch, 1):
    title = article.get('title', '')[:80]
    print(f"{i}. {title}")

# Build prompt exactly like sentiment.py
lines = []
for i, article in enumerate(batch, 1):
    title = article.get('title', '')[:80]
    lines.append(f"{i}. {title} - ")

articles_text = "\n".join(lines)

prompt = f"""Rate each headline as RELEVANT or NOT for AAPL stock analysis:

{articles_text}"""

print("\n" + "=" * 60)
print("Prompt being sent:")
print("=" * 60)
print(prompt)

# Send to GPT-OSS
client = OpenAI(base_url='http://172.23.193.91:8091/v1', api_key='not-needed')

print("\n" + "=" * 60)
print("GPT-OSS Response:")
print("=" * 60)

response = client.chat.completions.create(
    model='Qwen3-32B-Q6_K.gguf',
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2,
    max_tokens=1000
)

result = response.choices[0].message.content
print(f"Response: '{result}'")
print(f"Response length: {len(result) if result else 0}")