"""Test the tool server search endpoint"""

import requests

response = requests.post(
    'http://localhost:7001/search',
    json={'query': 'AAPL Apple stock news', 'max_results': 5}
)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")