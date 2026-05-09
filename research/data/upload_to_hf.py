#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
upload_to_hf.py
================
Run this ONCE on your local machine to push your Tamil OCR dataset
to a Hugging Face Dataset repository.

Usage:
    .venv\Scripts\python.exe OCR_Phase_2/data/upload_to_hf.py

Requirements:
    pip install huggingface_hub datasets
"""

import sys, io, json
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─── CONFIG ────────────────────────────────────────────────────────────────────
HF_TOKEN       = input("Paste your Hugging Face WRITE token and press Enter: ").strip()
HF_USERNAME    = input("Paste your Hugging Face username and press Enter: ").strip()
DATASET_REPO   = f"{HF_USERNAME}/tamil-ocr-byt5-dataset"
TRAINING_DIR   = Path(__file__).parent / "training"
# ───────────────────────────────────────────────────────────────────────────────

print(f"\n[1/4] Logging in to Hugging Face as '{HF_USERNAME}'...")
login(token=HF_TOKEN)

print("[2/4] Loading JSONL files...")

def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

train_records = load_jsonl(TRAINING_DIR / "final_train.jsonl")
val_records   = load_jsonl(TRAINING_DIR / "final_val.jsonl")

print(f"    Train: {len(train_records)} pairs")
print(f"    Val  : {len(val_records)} pairs")

print("[3/4] Converting to Hugging Face DatasetDict...")
ds = DatasetDict({
    "train": Dataset.from_list(train_records),
    "validation": Dataset.from_list(val_records),
})
print(ds)

print(f"[4/4] Pushing to Hub: '{DATASET_REPO}' (private=True)...")
ds.push_to_hub(
    DATASET_REPO,
    token=HF_TOKEN,
    private=True,
)

print(f"\nDone! Your dataset is now live at:")
print(f"  https://huggingface.co/datasets/{DATASET_REPO}")
print("\nNext step: attach this dataset to your Kaggle notebook.")
