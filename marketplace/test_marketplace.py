"""
test_marketplace.py
===================
Manual integration test to verify the new marketplace OOP logic.
Tests: Registration, Auth, Ordering, Matching, and Settlement.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_full_cycle():
    print("--- 1. Registering Two Nodes (Delhi_00, Delhi_01) ---")
    
    # Delhi_00 (Seller)
    resp_00 = requests.post(f"{BASE_URL}/nodes", json={"id": "Delhi_00", "city": "Delhi", "battery_cap_kwh": 10.0})
    if resp_00.status_code != 200:
        print(f"Node Delhi_00 registration failed: {resp_00.text}")
        return
    key_00 = resp_00.json()["api_key"]
    print(f"Registered Delhi_00 (Seller). Key: {key_00[:5]}...")

    # Delhi_01 (Buyer)
    resp_01 = requests.post(f"{BASE_URL}/nodes", json={"id": "Delhi_01", "city": "Delhi", "battery_cap_kwh": 10.0})
    if resp_01.status_code != 200:
        print(f"Node Delhi_01 registration failed: {resp_01.text}")
        return
    key_01 = resp_01.json()["api_key"]
    print(f"Registered Delhi_01 (Buyer). Key: {key_01[:5]}...")

    print("\n--- 2. Placing Sell Order (Delhi_00) ---")
    sell_payload = {
        "node_id": "Delhi_00",
        "order_type": "sell",
        "quantity_kwh": 2.0,
        "price_per_kwh": 6.0
    }
    headers_00 = {"X-API-Key": key_00}
    resp_sell = requests.post(f"{BASE_URL}/orders", json=sell_payload, headers=headers_00)
    print(f"Sell Order Result: {resp_sell.status_code}")
    print(json.dumps(resp_sell.json(), indent=2))

    print("\n--- 3. Placing Buy Order (Delhi_01) - Should Match! ---")
    buy_payload = {
        "node_id": "Delhi_01",
        "order_type": "buy",
        "quantity_kwh": 2.0,
        "price_per_kwh": 7.0
    }
    headers_01 = {"X-API-Key": key_01}
    resp_buy = requests.post(f"{BASE_URL}/orders", json=buy_payload, headers=headers_01)
    print(f"Buy Order Result: {resp_buy.status_code}")
    print(json.dumps(resp_buy.json(), indent=2))

    print("\n--- 4. Verifying Settlement & Wallets ---")
    wallet_00 = requests.get(f"{BASE_URL}/wallet/Delhi_00").json()
    wallet_01 = requests.get(f"{BASE_URL}/wallet/Delhi_01").json()
    
    # Trade was: 2kWh @ (6+7)/2 = 6.5 INR/kWh. Total = 13.0 INR.
    print(f"Seller (Delhi_00) Wallet Balance: {wallet_00['balance_inr']} INR")
    print(f"Buyer  (Delhi_01) Wallet Balance: {wallet_01['balance_inr']} INR")
    
    print("\n--- 5. Verifying Stats ---")
    stats = requests.get(f"{BASE_URL}/stats").json()
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    # Ensure uvicorn is running: uvicorn marketplace.main:app --host 0.0.0.0 --port 8000 --reload
    try:
        test_full_cycle()
    except Exception as e:
        print(f"Test failed: {e}")
