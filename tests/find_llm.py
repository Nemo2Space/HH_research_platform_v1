import os

# Find where LLM is called in portfolio_builder.py
with open('../dashboard/portfolio_builder.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

print("Searching for LLM/Qwen call in portfolio_builder.py...")
print("="*60)

keywords = ['llm', 'qwen', 'ollama', 'request', 'generate', 'chat', 'completion', 'api']

for i, line in enumerate(lines):
    line_lower = line.lower()
    if any(kw in line_lower for kw in keywords):
        print(f"Line {i+1}: {line[:100]}")
