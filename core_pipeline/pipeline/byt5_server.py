"""
byt5_server.py  — Minimal ByT5 correction micro-server
Runs as a subprocess spawned by main.py using a separate transformers==5.8.0 venv.
Accepts: JSON lines on stdin  {"text": "..."}
Returns: JSON lines on stdout {"corrected": "..."}
"""
import sys
import io
import json
import os

# Use binary stdout.buffer directly for all writes to avoid Windows cp1252 issues
_stdout = sys.stdout.buffer

# Silence HF warnings
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "Naren-hug/byt5-tamil-ocr-v1"
REVISION  = "b953fb849e8326ca8d103033c1937c8a6d8000b9"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, extra_special_tokens={})
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, revision=REVISION).to(DEVICE)
    model.eval()
    return tok, model

def correct(tok, model, text: str, max_len: int = 512) -> str:
    inputs = tok(
        text,
        return_tensors="pt",
        max_length=max_len,
        truncation=True,
    ).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_len,
            num_beams=4,
            repetition_penalty=1.5,
            no_repeat_ngram_size=15,  # Prevent looping of large chunks (15 chars)
            early_stopping=True
        )
    return tok.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    sys.stderr.write("[ByT5Server] Loading model...\n")
    sys.stderr.flush()
    tok, model = load_model()
    sys.stderr.write("[ByT5Server] Ready.\n")
    sys.stderr.flush()

    # Signal readiness to parent process (binary write)
    _stdout.write((json.dumps({"status": "ready"}) + "\n").encode("utf-8"))
    _stdout.flush()

    # Read stdin in binary mode to support Tamil UTF-8
    stdin_bin = sys.stdin.buffer
    for raw_line in stdin_bin:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            text = req.get("text", "")
            if not text:
                result = {"corrected": ""}
            else:
                corrected = correct(tok, model, text)
                result = {"corrected": corrected}
        except Exception as e:
            result = {"error": str(e), "corrected": req.get("text", "")}
        out = (json.dumps(result, ensure_ascii=False) + "\n").encode("utf-8")
        _stdout.write(out)
        _stdout.flush()
