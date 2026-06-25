"""Quick test to check if Mistral API connection works."""
import os
import sys
from dotenv import load_dotenv

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Load API key from .env
load_dotenv()
api_key = os.getenv('MISTRAL_API_KEY')

if not api_key:
    print("FAIL: No MISTRAL_API_KEY found in .env file")
    exit(1)

print(f"  API key loaded: {api_key[:6]}...{api_key[-4:]}")

import requests

print("=" * 50)
print("Test: Live API call to Mistral")
print("=" * 50)

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}
payload = {
    "model": "mistral-small-latest",
    "messages": [
        {"role": "user", "content": "Say hello in one sentence."},
    ],
    "temperature": 0.3,
}

print("  Calling Mistral API...")
response = requests.post(
    "https://api.mistral.ai/v1/chat/completions",
    headers=headers, json=payload, timeout=60,
)
print(f"  Status code: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    content = data['choices'][0]['message']['content']
    print(f"  Response: {content}")
    print()
    print("  MISTRAL API IS WORKING!")
else:
    print(f"  Response: {response.text}")
