"""
Smoke test for /chat endpoint.

Run with: python -m tests.smoke_chat
Or:       python tests/smoke_chat.py

This test verifies that the chat endpoint:
1. Returns a 200 status code
2. Returns a response with expected fields
"""
from __future__ import annotations

import requests
import sys

BASE_URL = "http://127.0.0.1:8000"


def test_chat_endpoint():
    """Smoke test: call /chat and verify response structure."""
    url = f"{BASE_URL}/chat"
    payload = {
        "message": "Hello, this is a smoke test. Reply with 'OK' only.",
        "settings": {
            "session_id": "smoke_test_session",
            "memory": False  # Don't pollute memory with test
        }
    }
    
    print(f"[SMOKE] POST {url}")
    print(f"[SMOKE] Payload: {payload}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
    except requests.exceptions.ConnectionError:
        print("[FAIL] Could not connect to server. Is it running?")
        print(f"       Start with: uvicorn app:app --reload")
        return False
    
    print(f"[SMOKE] Status: {response.status_code}")
    
    # Check status code
    if response.status_code != 200:
        print(f"[FAIL] Expected 200, got {response.status_code}")
        print(f"       Response: {response.text}")
        return False
    
    # Parse JSON
    try:
        data = response.json()
    except Exception as e:
        print(f"[FAIL] Could not parse JSON: {e}")
        print(f"       Response: {response.text}")
        return False
    
    print(f"[SMOKE] Response: {data}")
    
    # Check required fields
    required_fields = ["type", "reply"]
    missing = [f for f in required_fields if f not in data]
    
    if missing:
        print(f"[FAIL] Missing required fields: {missing}")
        return False
    
    # Check that reply is non-empty string
    if not isinstance(data.get("reply"), str):
        print(f"[FAIL] 'reply' should be a string, got: {type(data.get('reply'))}")
        return False
    
    if not data.get("reply"):
        print(f"[WARN] 'reply' is empty (might be OK for some intents)")
    
    print("[PASS] Chat endpoint smoke test passed!")
    return True


def test_health_endpoint():
    """Verify health endpoint is responding."""
    url = f"{BASE_URL}/health"
    
    print(f"[SMOKE] GET {url}")
    
    try:
        response = requests.get(url, timeout=5)
    except requests.exceptions.ConnectionError:
        print("[FAIL] Could not connect to server.")
        return False
    
    if response.status_code != 200:
        print(f"[FAIL] Health check failed: {response.status_code}")
        return False
    
    print(f"[SMOKE] Health: {response.json()}")
    print("[PASS] Health endpoint OK!")
    return True


def main():
    """Run all smoke tests."""
    print("=" * 50)
    print("CLAWDBOT SMOKE TESTS")
    print("=" * 50)
    print()
    
    results = []
    
    # Test health first
    results.append(("Health Check", test_health_endpoint()))
    print()
    
    # Test chat
    results.append(("Chat Endpoint", test_chat_endpoint()))
    print()
    
    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("All smoke tests passed! ✓")
        return 0
    else:
        print("Some tests failed. ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
