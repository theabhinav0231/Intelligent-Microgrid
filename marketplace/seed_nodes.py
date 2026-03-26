"""
seed_nodes.py
==========
Registers the 75 microgrid nodes in the marketplace and generates API keys.
Saves the API keys to a nodes_keys.json file for use by edge nodes and agents.
"""

import requests
import json
import os

BASE_URL = "http://localhost:8000"

CITIES = {
    "Delhi": 15,
    "Noida": 15,
    "Gurgaon": 15,
    "Chandigarh": 15,
    "Jaipur": 15
}

def seed():
    print("Starting node registration for 75 nodes...")
    node_keys = {}
    
    total = 0
    for city, count in CITIES.items():
        for i in range(count):
            node_id = f"{city}_{i:02d}"
            payload = {
                "id": node_id,
                "city": city,
                "battery_cap_kwh": 10.0
            }
            
            try:
                response = requests.post(f"{BASE_URL}/nodes", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    node_keys[node_id] = {
                        "api_key": data["api_key"],
                        "city": city
                    }
                    total += 1
                    print(f"Registered {node_id}")
                else:
                    print(f"Failed to register {node_id}: {response.text}")
            except Exception as e:
                print(f"Error registering {node_id}: {e}")
                
    # Save keys to a file for easy reference
    with open("node_keys.json", "w") as f:
        json.dump(node_keys, f, indent=4)
        
    print(f"\nSuccessfully registered {total} nodes.")
    print("API keys preserved in node_keys.json")

if __name__ == "__main__":
    # Ensure the server is running before seeding
    # This script assumes the marketplace is up at localhost:8000
    seed()
