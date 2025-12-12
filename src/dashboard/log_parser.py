import re
import os
from datetime import datetime

LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "autonomous_multicoin.log")

def parse_logs():
    data = {
        "pnl": 0.0,
        "equity": 0.0,
        "roi": 0.0,
        "positions": [],
        "prices": [],
        "signals": [],
        "last_updated": "Waiting for data...",
        "model_config": {"train_features": 0, "model_features": 0, "feature_set": "unknown"},
        "feature_list": []
    }
    
    if not os.path.exists(LOG_FILE):
        return data

    try:
        # Read ENTIRE file to capture startup ranking
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()

        # Regex Patterns
        # Regex Patterns
        pnl_pattern = re.compile(r"🚀 P/L: \$(?P<pnl>[+\-\d\.]+) \| Equity: \$(?P<equity>[+\-\d\.]+) \| ROI: (?P<roi>[+\-\d\.]+)%")
        pos_line_pattern = re.compile(r"\s+-\s+\[(?P<symbol>\w+)\] (?P<type>\w+) \| Amt: (?P<amt>[\d\.]+) \| Entry: \$(?P<entry>[\d\.]+) \| PnL: \$(?P<pnl>[+\-\d\.]+) \((?P<pct>[+\-\d\.]+)%\)")
        price_pattern = re.compile(r"💰 Prices: (?P<prices>.*)")
        signal_pattern = re.compile(r"\[(?P<symbol>\w+)\] 🧠 Signal: (?P<signal>\w+) \| Conf: (?P<conf>[\d\.]+)")
        # New Patterns
        config_pattern = re.compile(r"🧠 Model Config: Training Features: (?P<train>\d+) \| Model Features: (?P<model>\d+) \| Feature Set: (?P<set>\w+)")
        feature_pattern = re.compile(r"📜 Feature List: (?P<list>.*)")

        # ... (pnl parsing) ...
        
        # Scan for Model Config (Forward scan to find latest startup)
        # We can scan the whole file or just looking for the last occurrence
        for line in reversed(lines):
             cmatch = config_pattern.search(line)
             if cmatch:
                 data["model_config"] = {
                     "train_features": int(cmatch.group("train")),
                     "model_features": int(cmatch.group("model")),
                     "feature_set": cmatch.group("set")
                 }
                 break # Found latest
                 
        for line in reversed(lines):
             fmatch = feature_pattern.search(line)
             if fmatch:
                 raw_list = fmatch.group("list")
                 # Truncate if too long? No, UI handles it.
                 data["feature_list"] = raw_list.split(", ")
                 break
        # Match new format: "   1. SYMBOL: $1234.56M Vol"
        # We can support both with one regex or try multiple.
        # Let's support the new Volume format primarily.
        # Remove start anchors to matches anywhere in log line
        rank_pattern_vol = re.compile(r"\s+(?P<rank>\d+)\.\s+(?P<symbol>\w+):\s+\$(?P<perf>[-\d\.]+)M Vol")
        rank_pattern_pct = re.compile(r"\s+(?P<rank>\d+)\.\s+(?P<symbol>\w+):\s+(?P<perf>[-\d\.]+)%")
        
        # Define chunk for usage in reverse scans (Increased to 20k to catch older signals)
        chunk = lines[-20000:] if len(lines) > 20000 else lines

        # 1. Parsing P/L and Positions (Find LAST occurrence)
        last_pnl_idx = -1
        # Start searching from end to find P/L faster
        for i in range(len(lines) - 1, -1, -1):
           if "🚀 P/L:" in lines[i]:
               last_pnl_idx = i
               break # Found the latest one
               
        if last_pnl_idx != -1:
            # Parse P/L
            match = pnl_pattern.search(lines[last_pnl_idx])
            if match:
                data["pnl"] = float(match.group("pnl"))
                data["equity"] = float(match.group("equity"))
                data["roi"] = float(match.group("roi"))
                data["last_updated"] = lines[last_pnl_idx].split(" - ")[0]
            
            # Look ahead for positions
            for i in range(last_pnl_idx + 1, min(last_pnl_idx + 20, len(lines))):
                line = lines[i].strip()
                pos_match = pos_line_pattern.search(line)
                if pos_match:
                    data["positions"].append({
                        "symbol": pos_match.group("symbol"),
                        "type": pos_match.group("type"),
                        "amt": float(pos_match.group("amt")),
                        "entry": float(pos_match.group("entry")),
                        "pnl": float(pos_match.group("pnl")),
                        "pct": float(pos_match.group("pct"))
                    })
        # 2. Parse Prices (Reverse Scan for latest)
        for line in reversed(chunk):
            if "💰 Prices:" in line:
                pmatch = price_pattern.search(line)
                if pmatch:
                    raw = pmatch.group("prices")
                    parts = raw.split(" | ")
                    data["prices"] = []
                    for p in parts:
                        if ": $" in p:
                            try:
                                sym, rest = p.split(": $")
                                pr = rest.split(" [")[0]
                                data["prices"].append({"symbol": sym, "price": float(pr)})
                            except: pass
                    break

        # 3. Parse Top Performers (Find Latest Startup)
        # Scan reverse to find the last "Scanning for Top Volume Coins"
        start_scan_idx = -1
        for i in range(len(lines) - 1, -1, -1):
             if "Scanning for Top Volume Coins" in lines[i]:
                 start_scan_idx = i
                 break
        
        data["top_performers"] = []
        if start_scan_idx != -1:
            # Check next 50 lines for rankings
            temp_ranking = []
            for i in range(start_scan_idx, min(start_scan_idx + 50, len(lines))):
                line = lines[i]
                rmatch = rank_pattern_vol.search(line)
                is_vol = True
                if not rmatch:
                    rmatch = rank_pattern_pct.search(line)
                    is_vol = False
                    
                if rmatch:
                    rank = int(rmatch.group("rank"))
                    val = float(rmatch.group("perf"))
                    temp_ranking.append({
                        "rank": rank,
                        "symbol": rmatch.group("symbol"),
                        "perf": val, # Value only
                        "unit": "M" if is_vol else "%"
                    })
            if temp_ranking:
                 data["top_performers"] = sorted(temp_ranking, key=lambda x: x['rank'])

        # Signal Scan (keep reverse chunk)
        for line in reversed(chunk):
            signal_match = signal_pattern.search(line)
            if signal_match:
                data["signals"].append({
                    "symbol": signal_match.group("symbol"),
                    "signal": signal_match.group("signal"),
                    "conf": float(signal_match.group("conf")),
                    "time": line.split(" - ")[0]
                })
        
        # Limit signals
        data["signals"] = data["signals"][:10]
                                
        # 4. Calculate 'Days to Moonshot'
        # Estimate based on session performance
        try:
            start_time_dt = None
            initial_equity = None
            current_equity = data.get("equity", 0)
            
            # Find Start Time (First line of log or specific start marker)
            # Find Initial Equity (First P/L line)
            
            # We scan forward for the first P/L line to get initial equity
            for line in lines:
                if "🚀 P/L:" in line:
                     match = pnl_pattern.search(line)
                     if match:
                         initial_equity = float(match.group("equity")) - float(match.group("pnl")) # Approx initial
                         # Or just take the first equity if PnL was 0
                         # Ideally, Equity - PnL = Initial Capital + Deposits. 
                         # Let's assume Initial = Equity - PnL (Net Start)
                         
                         # Parse time
                         ts_str = line.split(" - ")[0]
                         try:
                             start_time_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
                         except:
                             start_time_dt = datetime.strptime(ts_str.split(",")[0], "%Y-%m-%d %H:%M:%S")
                         break
            
            if start_time_dt and current_equity > 0:
                now_str = data.get("last_updated", "") 
                if now_str:
                     try:
                         # 2025-12-12 21:49:55,598
                         now_dt = datetime.strptime(now_str, "%Y-%m-%d %H:%M:%S,%f")
                     except:
                         now_dt = datetime.now()
                else:
                    now_dt = datetime.now()
                    
                elapsed = (now_dt - start_time_dt).total_seconds() / 3600.0 # Hours
                
                total_profit = current_equity - initial_equity
                target_gain = 10000 - current_equity
                
                if elapsed > 0.01 and total_profit > 0:
                    hourly_rate = total_profit / elapsed
                    daily_rate = hourly_rate * 24
                    days_remaining = target_gain / daily_rate
                    data["days_to_moonshot"] = days_remaining
                elif total_profit <= 0:
                    data["days_to_moonshot"] = -1 # Infinite/Negative
                else:
                    data["days_to_moonshot"] = 0 # Already there?
                    
        except Exception as e:
            # print(f"Error calc moonshot: {e}")
            pass
            
    except Exception as e:
        print(f"Error parsing logs: {e}")
        
    return data
