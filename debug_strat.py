import os
import re
import traceback

LOG_FILE = "autonomous_multicoin.log"

def test_strat_parse():
    print(f"Testing Strat Parse on: {LOG_FILE}")
    
    with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
        
    print(f"Total lines: {len(lines)}")
    chunk = lines[-20000:]
    
    strat_pattern = re.compile(r"Strategy Config: Risk: (?P<risk>[\d\.]+)% \| SL: (?P<sl>[\d\.]+)% \| TP: (?P<tp>[\d\.]+)% \| MaxPos: (?P<maxpos>\d+)")
    
    found = False
    for line in reversed(chunk):
        if "Strategy Config:" in line:
            print(f"Found Line: {line.strip()}")
            smatch = strat_pattern.search(line)
            if smatch:
                print("Strategy Config Match!")
                print(f"Risk: {smatch.group('risk')}")
                print(f"SL: {smatch.group('sl')}")
                print(f"TP: {smatch.group('tp')}")
                print(f"MaxPos: {smatch.group('maxpos')}")
                found = True
                break
            else:
                print("Regex Failed to Match Line")
                
    if not found:
        print("Strategy Config NOT FOUND in chunk")

if __name__ == "__main__":
    test_strat_parse()
