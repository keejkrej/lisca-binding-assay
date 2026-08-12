"""Extract MEDIAN_TRACE from the paper presentation kinetics TS → JSON.

Usage (from lisca-binding-assay root):
  .venv/bin/python scripts/_extract_median.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

_BA_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PRES = _BA_ROOT.parent / "lisca-paper" / "presentation"
PRES_ROOT = Path(
    os.environ.get("LISCA_PAPER_PRESENTATION", str(_DEFAULT_PRES))
).expanduser().resolve()

ts_path = PRES_ROOT / "data" / "kinetics-real.ts"
text = ts_path.read_text(encoding="utf-8")
m = re.search(r"export const MEDIAN_TRACE[^=]*=\s*(\[[\s\S]*?\n\]);", text)
if not m:
    raise SystemExit(f"MEDIAN_TRACE not found in {ts_path}")
raw = m.group(1)
# JS object-literal → JSON: quote bare keys
s = re.sub(r"([{\s,])(\w+)\s*:", r'\1"\2":', raw)
s = s.replace("None", "null")  # noop safety
data = json.loads(s)
print("n points", len(data))
print("keys", list(data[0].keys()))
print("first", data[0])
print("mid", data[len(data) // 2])
print("last", data[-1])
out = _BA_ROOT / "scripts" / "_median_trace.json"
out.write_text(json.dumps(data), encoding="utf-8")
print("wrote", out)
