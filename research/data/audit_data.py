#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_data.py — Audit roundtrip checkpoint and final training files."""
import sys, io, json, re
from pathlib import Path
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HTML_RE  = re.compile(r"<[^>]+>")
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")


def audit(path, label):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    html_in_inp, html_in_tgt, no_tamil = [], [], []
    for r in records:
        inp = r.get("input", "")
        tgt = r.get("target", "")
        if HTML_RE.search(inp): html_in_inp.append(r)
        if HTML_RE.search(tgt): html_in_tgt.append(r)
        if not TAMIL_RE.search(inp): no_tamil.append(r)

    print(f"\n=== {label} ({Path(path).name}) ===")
    print(f"  Total pairs       : {len(records)}")
    print(f"  HTML in input     : {len(html_in_inp)}")
    print(f"  HTML in target    : {len(html_in_tgt)}")
    print(f"  No Tamil in input : {len(no_tamil)}")

    if html_in_inp:
        print("\n  -- Sample HTML inputs (first 3) --")
        for r in html_in_inp[:3]:
            print("    IN : " + r["input"][:120])
            print("    TGT: " + r["target"][:120])
            print()

    if no_tamil:
        print("  -- Sample no-Tamil inputs (first 3) --")
        for r in no_tamil[:3]:
            print("    IN : " + r["input"][:120])
            print("    TGT: " + r["target"][:120])
            print()


base = Path(__file__).parent / "training"
audit(base / "roundtrip_checkpoint_42.jsonl", "RoundTrip Checkpoint")
audit(base / "final_train.jsonl", "Final Train")
audit(base / "final_val.jsonl",   "Final Val")
print("\nAudit complete.")
