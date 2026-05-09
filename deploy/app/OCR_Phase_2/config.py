"""
Central Configuration — OCR Phase 2
====================================
All thresholds, paths, language mappings, and engine settings.
Inherits Phase 1 defaults but adds engine comparison and LM correction config.
"""
import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent
DATA_DIR     = ROOT_DIR / "data"
OUTPUTS_DIR  = ROOT_DIR / "batch_evaluation_fixed_outputs"
MODELS_DIR   = ROOT_DIR / "models"

# ── Dataset paths (shared with Phase 1 — read-only access) ───────────────────
# Phase 1 lives at ../OCR/  — we reference its datasets but never modify them
PHASE1_DIR       = ROOT_DIR.parent / "OCR"
DATASET_DIR      = PHASE1_DIR / "OCR_dataset"
GROUND_TRUTH_DIR = PHASE1_DIR / "evaluation" / "ground_truth"

# ── PDF → Image ───────────────────────────────────────────────────────────────
PDF_DPI            = 300
PDF_FORMAT         = "png"

# ── Run Options ───────────────────────────────────────────────────────────────
PREPROCESS_ENABLED        = False # Surya expects raw RGB images; binarization harms accuracy

# ── OCR Engine ────────────────────────────────────────────────────────────────
DEFAULT_ENGINE     = "surya"       # "surya" | "paddle" | "best"
SURYA_DEVICE       = "cuda"
PADDLE_USE_GPU     = True
PADDLE_LANG        = "ta"          # PaddleOCR language code for Tamil
DEFAULT_LANGS      = ["ta", "en"]  # Languages to detect

TESSERACT_LANGS = {
    "ta": "tam", "hi": "hin", "te": "tel", "ml": "mal",
    "kn": "kan", "bn": "ben", "en": "eng",
}

# ── Quality Gate ──────────────────────────────────────────────────────────────
QUALITY_CONF_THRESHOLD    = 0.90   # Pages below this get LM correction
QUALITY_LOW_WORD_RATIO    = 0.20
WORD_LOW_CONF_THRESHOLD   = 0.70

# ── Engine Comparison ─────────────────────────────────────────────────────────
ENGINE_COMPARISON_REPORT  = OUTPUTS_DIR / "engine_comparison_report.json"
FALLBACK_CONF_THRESHOLD   = 0.70   # Use fallback engine if winner's conf < this

# ── Layout Reconstruction ─────────────────────────────────────────────────────
COLUMN_SPLIT_RATIO        = 0.5
HEADING_HEIGHT_MULTIPLIER = 1.5
MARGIN_FRACTION           = 0.05
PARAGRAPH_GAP_MULTIPLIER  = 1.5

# ── LM Correction (ByT5) ─────────────────────────────────────────────────────
BYT5_MODEL_NAME           = "google/byt5-small"
BYT5_FINETUNED_PATH       = MODELS_DIR / "byt5_finetuned"
LM_CORRECTION_ENABLED     = False  # Disabled for deployment; finetuning in progress
LM_MAX_INPUT_LENGTH       = 256    # Max chars per chunk sent to ByT5

# ── Synthetic Noise Generation ────────────────────────────────────────────────
NOISE_TRAIN_FILE          = DATA_DIR / "synthetic_noise" / "train.jsonl"
NOISE_VAL_FILE            = DATA_DIR / "synthetic_noise" / "val.jsonl"
NOISE_ERROR_RATE          = 0.15   # ~15% of chars corrupted per sentence

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_FORMATS     = ["txt", "md", "json"]
