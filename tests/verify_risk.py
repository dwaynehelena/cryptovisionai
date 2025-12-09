import sys
import os
import json
import urllib.request
import urllib.error
import time

BASE_URL = "http://localhost:8000/api/v1"

def make_request(endpoint, method="GET", data=None):
    url = f"{BASE_URL}{endpoint}"
    headers = {'Content-Type': 'application/json'}
    
    try:
        if data:
            data_bytes = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
        else:
            req = urllib.request.Request(url, headers=headers, method=method)
            
        with urllib.request.urlopen(req) as response:
            status_code = response.getcode()
            response_data = json.loads(response.read().decode('utf-8'))
            return status_code, response_data
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf-8')
    except urllib.error.URLError as e:
        return 0, str(e)

def test_risk_limits():
    print("\nTesting Risk Limits...")
    status, data = make_request("/risk/limits")
    if status == 200:
        print("✅ Get Limits: Success")
        print(json.dumps(data, indent=2))
    else:
        print(f"❌ Get Limits: Failed ({status})")
        print(data)

    # Update limits
    new_limits = {
        "max_position_size_pct": 15.0,
        "max_risk_per_trade_pct": 2.5,
        "max_drawdown_pct": 25.0,
        "daily_loss_limit_pct": 6.0,
        "auto_stop_loss_enabled": True,
        "atr_multiplier": 2.5,
        "max_correlation": 0.8,
        "trading_enabled": True
    }
    status, data = make_request("/risk/limits", method="POST", data=new_limits)
    if status == 200:
        print("✅ Update Limits: Success")
        print(json.dumps(data, indent=2))
    else:
        print(f"❌ Update Limits: Failed ({status})")
        print(data)

def test_risk_status():
    print("\nTesting Risk Status...")
    status, data = make_request("/risk/status")
    if status == 200:
        print("✅ Get Status: Success")
        print(json.dumps(data, indent=2))
    else:
        print(f"❌ Get Status: Failed ({status})")
        print(data)

def test_position_sizing():
    print("\nTesting Position Sizing...")
    params = {
        "entry_price": 50000.0,
        "stop_price": 49000.0,
        "volatility": 0.02
    }
    status, data = make_request("/risk/calculate-position-size", method="POST", data=params)
    if status == 200:
        print("✅ Calculate Position Size: Success")
        print(json.dumps(data, indent=2))
    else:
        print(f"❌ Calculate Position Size: Failed ({status})")
        print(data)

def test_emergency_controls():
    print("\nTesting Emergency Controls...")
    # Disable trading
    status, data = make_request("/risk/disable-trading", method="POST")
    if status == 200:
        print("✅ Disable Trading: Success")
    else:
        print(f"❌ Disable Trading: Failed ({status})")

    # Verify status
    status, data = make_request("/risk/limits")
    if status == 200 and data.get('trading_enabled') == False:
        print("✅ Trading Disabled Verified")
    else:
        print("❌ Trading Disabled Verification Failed")

    # Enable trading
    status, data = make_request("/risk/enable-trading", method="POST")
    if status == 200:
        print("✅ Enable Trading: Success")
    else:
        print(f"❌ Enable Trading: Failed ({status})")

if __name__ == "__main__":
    try:
        test_risk_limits()
        test_risk_status()
        test_position_sizing()
        test_emergency_controls()
    except Exception as e:
        print(f"❌ Error: {e}")
