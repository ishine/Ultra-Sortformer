#!/usr/bin/env python3
"""
duration=-1인 manifest 항목에 duration=180을 설정하여 180초 초과 로드를 방지.
"""
import json
import sys
from pathlib import Path

def main():
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else in_path.replace(".json", "_fixed.json")
    
    with open(in_path) as f:
        entries = [json.loads(line) for line in f]
    
    fixed = 0
    for e in entries:
        if e.get("duration") == -1 or e.get("duration") is None:
            e["duration"] = 180.0
            fixed += 1
    
    with open(out_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    
    print(f"Fixed {fixed} entries, wrote {out_path}")


if __name__ == "__main__":
    main()
