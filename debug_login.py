#!/usr/bin/env python3
import requests
import json
import os

api_base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
user = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
passwd = os.environ.get("TENSOR_AGENT_PASS", "EcoTest!2026pass")

resp = requests.post(f"{api_base}/auth/login", json={"username": user, "password": passwd})
print(f"Status: {resp.status_code}")
print(f"Text: {resp.text}")
try:
    print(f"JSON: {resp.json()}")
except:
    pass
