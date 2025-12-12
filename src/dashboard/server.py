from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import sys
import requests
import time
import json

# Ensure import visibility
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.dashboard.log_parser import parse_logs

app = FastAPI()

# Mount static files (CSS/JS)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Cache for Mainnet Prices
mainnet_cache = {"last_updated": 0, "data": {}}

def get_mainnet_prices(symbols):
    global mainnet_cache
    if not symbols:
        return {}
        
    # Rate Limit Protection: Cache for 5 seconds
    if time.time() - mainnet_cache["last_updated"] < 5:
        return mainnet_cache["data"]
        
    try:
        # Construct URL param: ["BTCUSDT","ETHUSDT"]
        sym_list = json.dumps(symbols, separators=(',', ':')) # Compact JSON
        url = "https://api.binance.com/api/v3/ticker/price"
        
        resp = requests.get(url, params={"symbols": sym_list}, timeout=2)
        if resp.status_code == 200:
            raw_data = resp.json()
            # Convert to dict: {"BTCUSDT": 50000.0, ...}
            new_data = {item['symbol']: float(item['price']) for item in raw_data}
            
            # Merge with existing cache (to keep others if this partial update)
            mainnet_cache["data"].update(new_data)
            mainnet_cache["last_updated"] = time.time()
            return mainnet_cache["data"]
        else:
            print(f"Mainnet API Error: {resp.status_code} {resp.text}")
            # Likely invalid symbol in the batch. Fallback: try one by one?
            # Or just return empty?
            pass
            
    except Exception as e:
        print(f"Mainnet fetch error: {e}")
        
    return mainnet_cache["data"]

# Cache for Sparklines
sparkline_cache = {"last_updated": 0, "data": {}}

def get_sparklines(symbols):
    global sparkline_cache
    if not symbols:
        return {}
    
    # Cache for 60 seconds (Sparklines don't need real-time precision)
    if time.time() - sparkline_cache["last_updated"] < 60:
        return sparkline_cache["data"]
        
    new_data = {}
    # Fetch klines for each symbol
    # This is N requests. Be careful.
    for sym in symbols:
        # Check if we already have recent data for this symbol in cache to avoid re-fetching all?
        # For simplicity, just fetch all if cache expired.
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {"symbol": sym, "interval": "1h", "limit": 24}
            resp = requests.get(url, params=params, timeout=2)
            if resp.status_code == 200:
                klines = resp.json()
                # Extract Close prices
                closes = [float(k[4]) for k in klines]
                new_data[sym] = closes
            else:
                # Keep old data if fetch fails
                if sym in sparkline_cache["data"]:
                    new_data[sym] = sparkline_cache["data"][sym]
        except:
             if sym in sparkline_cache["data"]:
                new_data[sym] = sparkline_cache["data"][sym]
    
    sparkline_cache["data"].update(new_data)
    sparkline_cache["last_updated"] = time.time()
    return sparkline_cache["data"]

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(os.path.dirname(__file__), "static", "index.html"), "r") as f:
        return f.read()

@app.get("/api/stats")
async def get_stats():
    data = parse_logs()
    
    # Enrich with Mainnet Prices
    if "prices" in data and data["prices"]:
        active_symbols = [item['symbol'] for item in data['prices']]
        mainnet_prices = get_mainnet_prices(active_symbols)
        
        for item in data["prices"]:
            sym = item['symbol']
            if sym in mainnet_prices:
                item['mainnet_price'] = mainnet_prices[sym]

    # Enrich Positions with Sparklines
    if "positions" in data and data["positions"]:
        pos_symbols = [p['symbol'] for p in data['positions']]
        sparklines = get_sparklines(pos_symbols)
        
        for p in data['positions']:
            if p['symbol'] in sparklines:
                p['sparkline'] = sparklines[p['symbol']]
                
    return data

if __name__ == "__main__":
    import uvicorn
    # Use port 8090 to avoid conflict with standard 8000/8080 usually taken
    uvicorn.run(app, host="0.0.0.0", port=8090)
