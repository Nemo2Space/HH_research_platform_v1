"""Debug GPT-OSS filter - see what's kept vs filtered"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

import re
from openai import OpenAI
from src.data.news import NewsCollector

# Collect real news
collector = NewsCollector()
articles = collector.collect_all_news('AAPL', days_back=7)
print(f"Collected {len(articles)} articles\n")

# Deduplicate
seen = set()
unique = []
for a in articles:
    title = a.get('title', '').lower().strip()
    normalized = re.sub(r'[^\w\s]', '', title)
    normalized = ' '.join(normalized.split()[:8])
    if normalized and normalized not in seen:
        seen.add(normalized)
        unique.append(a)

articles = unique
print(f"After dedup: {len(articles)} articles\n")

# Test filter on first 25 articles
batch = articles[:25]

print("=" * 80)
print("BATCH OF 25 ARTICLES:")
print("=" * 80)
for i, article in enumerate(batch, 1):
    title = article.get('title', '')[:70]
    source = article.get('source', '')[:15]
    print(f"{i:2}. [{source:<15}] {title}")

# Build prompt
lines = []
for i, article in enumerate(batch, 1):
    title = article.get('title', '')[:80]
    lines.append(f"{i}. {title} - ")

articles_text = "\n".join(lines)
prompt = f"""Rate each headline as RELEVANT or NOT for AAPL stock analysis:

{articles_text}"""

# Send to GPT-OSS
client = OpenAI(base_url='http://172.23.193.91:8091/v1', api_key='not-needed')

print("\n" + "=" * 80)
print("GPT-OSS RESPONSE:")
print("=" * 80)

response = client.chat.completions.create(
    model='Qwen3-32B-Q6_K.gguf',
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2,
    max_tokens=2000
)

result = response.choices[0].message.content
print(result)

# Parse what was marked relevant
print("\n" + "=" * 80)
print("PARSING RESULTS:")
print("=" * 80)

kept = []
filtered = []

for line in result.split('\n'):
    line_lower = line.lower()

    # Skip headers
    if '|---|' in line or ('headline' in line_lower and 'relevance' in line_lower):
        continue

    # Extract number
    num_match = re.search(r'(\d+)', line)
    if not num_match:
        continue

    idx = int(num_match.group(1))
    if idx < 1 or idx > len(batch):
        continue

    # Check relevant vs not
    if 'not' in line_lower:
        filtered.append(idx)
    elif 'relevant' in line_lower:
        kept.append(idx)

print(f"\nKEPT ({len(kept)} articles): {kept}")
print(f"FILTERED ({len(filtered)} articles): {filtered}")

print("\n" + "=" * 80)
print("KEPT ARTICLES:")
print("=" * 80)
for idx in kept:
    print(f"  {idx}. {batch[idx-1].get('title', '')[:70]}")

print("\n" + "=" * 80)
print("FILTERED OUT ARTICLES:")
print("=" * 80)
for idx in filtered:
    print(f"  {idx}. {batch[idx-1].get('title', '')[:70]}")