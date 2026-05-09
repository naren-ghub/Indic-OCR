# Multilingual OCR + NLP Book Digitization Pipeline
### Technical Design Document — v2.1

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Objectives](#2-objectives)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Detailed Pipeline](#5-detailed-pipeline)
   - 5.1 [PDF to Image Conversion](#51-pdf-to-image-conversion)
   - 5.2 [Language Detection](#52-language-detection)
   - 5.3 [Image Preprocessing](#53-image-preprocessing)
   - 5.4 [OCR Extraction](#54-ocr-extraction)
   - 5.5 [Confidence Scoring & Quality Gate](#55-confidence-scoring--quality-gate)
   - 5.6 [Layout Reconstruction](#56-layout-reconstruction)
   - 5.7 [Rule-Based Cleaning](#57-rule-based-cleaning)
   - 5.8 [LM Correction with IndicBART](#58-lm-correction-with-indicbart)
   - 5.9 [Paragraph Reconstruction](#59-paragraph-reconstruction)
   - 5.10 [Output Generation](#510-output-generation)
6. [Queue Architecture](#6-queue-architecture)
7. [Multilingual Handling](#7-multilingual-handling)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Performance Optimization](#9-performance-optimization)
10. [Project Structure](#10-project-structure)
11. [Setup & Installation](#11-setup--installation)
12. [Future Enhancements](#12-future-enhancements)
    - 12.1 [Agentic Flow Upgrade](#121-agentic-flow-upgrade)

---

## 1. Project Overview

This project builds a scalable pipeline to convert scanned (image-based) PDFs of books (200+ pages) into clean, structured, machine-readable text. It combines computer vision, OCR engines optimized for Indian scripts, language-aware NLP models, and a queue-based architecture for reliable parallel processing.

The pipeline is specifically designed to handle the unique challenges of Indian language books — including complex Indic scripts (Tamil, Telugu, Malayalam, Devanagari), multi-column layouts, mixed-language content, and degraded scans from older publications.

**Target Languages:** Tamil, Hindi, Telugu, Malayalam, Kannada, Bengali, Odia, Punjabi, English, and code-mixed variants.

> **Roadmap note:** The current version implements a standard sequential pipeline with parallel processing via a job queue. An agentic orchestration layer — for dynamic routing, self-healing retries, and intelligent quality gating — is planned as a future upgrade once the core pipeline is stable. See [Section 12.1](#121-agentic-flow-upgrade).

---

## 2. Objectives

- Extract accurate text from scanned PDFs across multiple Indian languages
- Detect language **before** OCR to drive the correct engine configuration
- Use confidence scoring to selectively apply expensive LM correction only where needed
- Preserve document structure — headings, paragraphs, columns, footnotes
- Handle 200+ page books without memory overflow using a proper job queue
- Produce clean, structured outputs in `.txt`, `.md`, and `.json` formats

---

## 3. System Architecture

### 3.1 High-Level Pipeline Flow

```
PDF Input
   │
   ▼
[PDF → Images]           pdf2image @ 300 DPI
   │
   ▼
[Language Detection]     fastText lid.176.bin (per page block)
   │
   ▼
[Image Preprocessing]    OpenCV — deskew, denoise, threshold
   │
   ▼
[OCR Extraction]         IndicOCR (primary) / Tesseract 5 (fallback)
   │
   ▼
[Confidence Scoring]     Per-word confidence → Quality Gate decision
   │
   ├──── High confidence (≥ 0.80) ──────────────────────────────┐
   │                                                             │
   └──── Low confidence (< 0.80) ──► [LM Correction]           │
                                      IndicBART seq2seq          │
                                             │                   │
                                             ▼                   ▼
                                    [Layout Reconstruction] ◄────┘
                                             │
                                             ▼
                                    [Rule-Based Cleaning]
                                             │
                                             ▼
                                    [Paragraph Reconstruction]
                                             │
                                             ▼
                                    [Output Generation]
                                    .txt / .md / .json
```

### 3.2 Queue-Based Processing Architecture

```
Main Process
   │
   ▼
[Job Producer]
   │  Enqueues (page_id, image_path, metadata)
   ▼
[Redis Queue]
   │
   ├──► [Worker 1]  ──► Process pages 1–50
   ├──► [Worker 2]  ──► Process pages 51–100
   ├──► [Worker 3]  ──► Process pages 101–150
   └──► [Worker 4]  ──► Process pages 151–200+
         │
         ▼
   [Result Store]  ──► Assemble final document
```

---

## 4. Technology Stack

### 4.1 Core Components

| Category | Primary Tool | Fallback / Alternative | Purpose |
|---|---|---|---|
| OCR (Indic) | IndicOCR (AI4Bharat) | Tesseract 5 with lang packs | Text extraction from Indic scripts |
| OCR (Latin/General) | EasyOCR | Tesseract 5 | English and mixed-script pages |
| Language Detection | fastText `lid.176.bin` | `langdetect` library | Identify language before OCR |
| Image Processing | OpenCV | Pillow | Preprocessing, deskewing, thresholding |
| LM Correction | IndicBART (AI4Bharat) | IndicBERT fill-mask | Seq2seq OCR error correction |
| Layout Analysis | Custom bounding-box sorter | LayoutLMv3 (future) | Column/heading/footer detection |
| PDF Conversion | pdf2image (poppler) | PyMuPDF | PDF → high-res images |
| Queue | Celery + Redis | Python `multiprocessing.Queue` | Parallel page processing |
| Embeddings | MuRIL | sentence-transformers | Downstream NLP tasks |
| Utilities | NumPy, regex, fastText | — | Text normalization, numerics |

### 4.2 Why IndicOCR over EasyOCR for Indic Scripts

EasyOCR was designed primarily for Latin and CJK scripts. Indian scripts (Tamil, Telugu, Malayalam) involve:

- **Conjunct characters** — multiple consonants fusing into a single glyph
- **Vowel diacritics** (matras) attached to base consonants
- **Complex ligatures** that change character shape based on context

IndicOCR from AI4Bharat is explicitly trained on these characteristics using large corpora of Indian language scans, yielding significantly lower Character Error Rate (CER) on Tamil (~8% vs ~21% for EasyOCR in internal benchmarks).

### 4.3 Why IndicBART over MuRIL for Correction

MuRIL and IndicBERT are **encoder-only** transformer models. They can understand and classify text, but they cannot *generate* corrected text on their own. IndicBART is a **sequence-to-sequence** (encoder-decoder) model — the correct architecture for the task of taking noisy OCR output as input and producing corrected text as output.

```
MuRIL:     noisy_text → [encoder] → representation only (no generation)
IndicBART: noisy_text → [encoder] → [decoder] → corrected_text ✓
```

---

## 5. Detailed Pipeline

### 5.1 PDF to Image Conversion

Convert each page of the input PDF to a high-resolution image for OCR processing.

**Tool:** `pdf2image` (wrapper around `poppler`)

**Configuration:**
- Resolution: **300 DPI** minimum (400 DPI for heavily degraded scans)
- Format: PNG (lossless) — never JPEG for OCR input (compression artifacts degrade accuracy)
- Color mode: Grayscale (reduces processing time; color rarely aids OCR)

```python
from pdf2image import convert_from_path

def pdf_to_images(pdf_path: str, dpi: int = 300) -> list:
    images = convert_from_path(
        pdf_path,
        dpi=dpi,
        fmt="png",
        grayscale=True,
        thread_count=4
    )
    return images  # list of PIL Image objects, one per page
```

**Output:** List of PIL Image objects indexed by page number.

---

### 5.2 Language Detection

Language detection is performed **before preprocessing and OCR** so that the correct OCR engine, language pack, and NLP configuration can be selected per page.

**Strategy:**
- For the **first 3 pages**, run a quick lightweight OCR pass (EasyOCR English-only) to extract enough characters for detection
- Use **fastText's `lid.176.bin`** model (supports 176 languages including all major Indian languages)
- Detect language at the **page block level**, not just per document — a single book may switch between Tamil and English across sections

```python
import fasttext

lang_model = fasttext.load_model("lid.176.bin")

def detect_language(text_sample: str) -> tuple[str, float]:
    predictions = lang_model.predict(text_sample, k=1)
    lang_code = predictions[0][0].replace("__label__", "")  # e.g. "ta", "hi", "en"
    confidence = predictions[1][0]
    return lang_code, confidence

# Language code → OCR config mapping
LANG_OCR_MAP = {
    "ta": {"engine": "indicocr", "lang": "tam"},
    "hi": {"engine": "indicocr", "lang": "hin"},
    "te": {"engine": "indicocr", "lang": "tel"},
    "ml": {"engine": "indicocr", "lang": "mal"},
    "kn": {"engine": "indicocr", "lang": "kan"},
    "bn": {"engine": "indicocr", "lang": "ben"},
    "en": {"engine": "easyocr",  "lang": "en"},
}
```

**Output:** `lang_code` (ISO 639-1), `confidence_score`, `ocr_config` dict.

---

### 5.3 Image Preprocessing

Clean the raw scanned image before passing it to OCR. Poor preprocessing is the single largest contributor to OCR errors.

**Tool:** OpenCV

**Steps applied in order:**

#### a) Deskewing
Correct rotation caused by uneven placement during scanning.

```python
import cv2
import numpy as np

def deskew(image: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
```

#### b) Noise Removal
```python
def remove_noise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)
```

#### c) Adaptive Thresholding
Handles uneven lighting across a scanned page.
```python
def threshold(image: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
```

#### d) Contrast Enhancement (for degraded/faded scans)
```python
def enhance_contrast(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)
```

**Full preprocessing chain:**
```python
def preprocess(image_pil) -> np.ndarray:
    img = np.array(image_pil.convert("L"))  # grayscale
    img = deskew(img)
    img = remove_noise(img)
    img = enhance_contrast(img)
    img = threshold(img)
    return img
```

---

### 5.4 OCR Extraction

Route each preprocessed image to the appropriate OCR engine based on the detected language from Step 5.2.

#### Primary: IndicOCR (AI4Bharat)

Best for Tamil, Telugu, Malayalam, Kannada, Bengali, and Devanagari scripts.

```python
from indic_ocr import IndicOCR

def run_indic_ocr(image: np.ndarray, lang: str) -> list[dict]:
    engine = IndicOCR(lang=lang)
    results = engine.predict(image)
    # Returns: [{"text": str, "bbox": [x1,y1,x2,y2], "confidence": float}]
    return results
```

#### Fallback: Tesseract 5

Used when IndicOCR confidence is below threshold or for English sections.

```python
import pytesseract
from PIL import Image

def run_tesseract(image: np.ndarray, lang: str = "tam+eng") -> list[dict]:
    pil_img = Image.fromarray(image)
    data = pytesseract.image_to_data(
        pil_img,
        lang=lang,
        config="--oem 3 --psm 6",  # LSTM engine, assume uniform block of text
        output_type=pytesseract.Output.DICT
    )
    results = []
    for i, word in enumerate(data["text"]):
        if word.strip():
            results.append({
                "text": word,
                "confidence": data["conf"][i] / 100.0,
                "bbox": [data["left"][i], data["top"][i],
                         data["left"][i] + data["width"][i],
                         data["top"][i] + data["height"][i]]
            })
    return results
```

**Decision logic:**
```python
def extract_text(image, lang_config) -> list[dict]:
    if lang_config["engine"] == "indicocr":
        results = run_indic_ocr(image, lang_config["lang"])
        avg_conf = sum(r["confidence"] for r in results) / max(len(results), 1)
        if avg_conf < 0.60:  # IndicOCR not confident → try Tesseract
            results = run_tesseract(image, lang=lang_config["lang"] + "+eng")
    else:
        results = run_tesseract(image, lang="eng")
    return results
```

---

### 5.5 Confidence Scoring & Quality Gate

Every word returned by OCR has a confidence score (0.0 – 1.0). This step aggregates scores and makes a **routing decision** — whether a page needs LM correction or can pass through directly.

```python
def compute_page_quality(ocr_results: list[dict]) -> dict:
    if not ocr_results:
        return {"avg_confidence": 0.0, "low_conf_ratio": 1.0, "needs_correction": True}

    confidences = [r["confidence"] for r in ocr_results]
    avg_conf = sum(confidences) / len(confidences)
    low_conf_count = sum(1 for c in confidences if c < 0.70)
    low_conf_ratio = low_conf_count / len(confidences)

    return {
        "avg_confidence": round(avg_conf, 3),
        "low_conf_ratio": round(low_conf_ratio, 3),
        "low_conf_words": [r for r in ocr_results if r["confidence"] < 0.70],
        "needs_correction": avg_conf < 0.80 or low_conf_ratio > 0.20
    }
```

**Quality Gate Routing:**

```
avg_confidence ≥ 0.80 AND low_conf_ratio ≤ 0.20
        │
        ├── YES → Skip LM correction → go directly to Layout Reconstruction
        │
        └── NO  → Send to IndicBART LM Correction
                  (only low-confidence words are corrected, not the full page)
```

This gate reduces LM correction workload by an estimated **40–60%** on clean scans.

---

### 5.6 Layout Reconstruction

Reconstruct reading order from the raw bounding boxes returned by OCR. Scanned books may have multi-column layouts, headers, footers, and figure captions that need to be correctly sequenced.

```python
def reconstruct_layout(ocr_results: list[dict], page_height: int, page_width: int) -> list[dict]:
    # Step 1: Detect columns by clustering x-coordinates
    x_centers = [r["bbox"][0] + (r["bbox"][2] - r["bbox"][0]) // 2 for r in ocr_results]
    mid_x = page_width // 2
    left_col  = [r for r, x in zip(ocr_results, x_centers) if x < mid_x]
    right_col = [r for r, x in zip(ocr_results, x_centers) if x >= mid_x]

    # Step 2: Sort each column top-to-bottom
    left_col.sort(key=lambda r: r["bbox"][1])
    right_col.sort(key=lambda r: r["bbox"][1])

    # Step 3: Detect and remove headers/footers (top 5% and bottom 5% of page)
    margin_top    = page_height * 0.05
    margin_bottom = page_height * 0.95
    def in_body(r):
        return r["bbox"][1] > margin_top and r["bbox"][3] < margin_bottom

    body = [r for r in (left_col + right_col) if in_body(r)]

    # Step 4: Detect headings (larger bounding box height than average)
    avg_height = sum(r["bbox"][3] - r["bbox"][1] for r in body) / max(len(body), 1)
    for r in body:
        word_height = r["bbox"][3] - r["bbox"][1]
        r["is_heading"] = word_height > avg_height * 1.5

    return body
```

> **Future upgrade:** LayoutLMv3 can replace this heuristic approach for more robust detection of tables, figures, and complex multi-column layouts. See [Section 12](#12-future-enhancements).

---

### 5.7 Rule-Based Cleaning

Apply deterministic text cleaning rules to fix common OCR artifacts before passing to the language model.

```python
import re

# Common OCR character substitution errors
OCR_FIXES = {
    r"\b0([a-zA-Z])": r"O\1",    # 0 → O before letters
    r"([a-zA-Z])0\b": r"\1O",    # 0 → O after letters
    r"\b1([a-zA-Z])": r"l\1",    # 1 → l (lowercase L)
    r"([a-zA-Z])1\b": r"\1l",
    r"\|": "I",                   # pipe → capital I
    r"(\w)-\n(\w)": r"\1\2",     # dehyphenation across line breaks
}

def rule_based_clean(text: str) -> str:
    for pattern, replacement in OCR_FIXES.items():
        text = re.sub(pattern, replacement, text)

    # Remove standalone page numbers
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)

    # Normalize punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([.,;:!?])(\w)", r"\1 \2", text)
    text = re.sub(r" {2,}", " ", text)

    # Normalize Unicode — critical for Indic scripts
    import unicodedata
    text = unicodedata.normalize("NFC", text)

    return text.strip()
```

---

### 5.8 LM Correction with IndicBART

Applied **only to pages that failed the quality gate** in Step 5.5. IndicBART is a seq2seq model — it takes noisy OCR text as input and generates corrected text.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART")
model     = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBART")

def correct_with_indicbart(noisy_text: str, src_lang: str = "ta_IN") -> str:
    inputs = tokenizer(
        noisy_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[src_lang],
        max_length=512,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def selective_correction(ocr_results: list[dict], quality: dict, lang: str) -> list[dict]:
    if not quality["needs_correction"]:
        return ocr_results  # pass through unchanged

    # Only correct low-confidence words, not the whole page
    low_conf_text = " ".join(r["text"] for r in quality["low_conf_words"])
    corrected_text = correct_with_indicbart(low_conf_text, src_lang=lang)

    corrected_words = corrected_text.split()
    for i, r in enumerate(quality["low_conf_words"]):
        if i < len(corrected_words):
            r["text"] = corrected_words[i]
            r["corrected"] = True

    return ocr_results
```

**Supported `src_lang` codes for IndicBART:**

| Language | Code |
|---|---|
| Tamil | `ta_IN` |
| Hindi | `hi_IN` |
| Telugu | `te_IN` |
| Malayalam | `ml_IN` |
| Kannada | `kn_IN` |
| Bengali | `bn_IN` |
| English | `en_XX` |

---

### 5.9 Paragraph Reconstruction

Merge individual word/line bounding boxes back into coherent paragraphs and detect section headings.

```python
def reconstruct_paragraphs(words: list[dict]) -> list[dict]:
    paragraphs = []
    current_para = []
    prev_y = None

    avg_line_height = (
        sum(w["bbox"][3] - w["bbox"][1] for w in words) / max(len(words), 1)
    )

    for word in words:
        current_y = word["bbox"][1]

        if prev_y is not None and (current_y - prev_y) > avg_line_height * 1.5:
            if current_para:
                paragraphs.append({
                    "type": "heading" if current_para[0].get("is_heading") else "paragraph",
                    "text": " ".join(w["text"] for w in current_para),
                    "page": current_para[0].get("page_num")
                })
                current_para = []

        current_para.append(word)
        prev_y = current_y

    if current_para:
        paragraphs.append({
            "type": "heading" if current_para[0].get("is_heading") else "paragraph",
            "text": " ".join(w["text"] for w in current_para),
            "page": current_para[0].get("page_num")
        })

    return paragraphs
```

---

### 5.10 Output Generation

Generate three output formats from the reconstructed paragraphs.

```python
import json

def generate_outputs(paragraphs: list[dict], book_title: str, output_dir: str):
    # .txt — plain text
    with open(f"{output_dir}/{book_title}.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(p["text"] for p in paragraphs))

    # .md — structured markdown with headings
    md_lines = []
    for p in paragraphs:
        md_lines.append(f"## {p['text']}" if p["type"] == "heading" else p["text"])
    with open(f"{output_dir}/{book_title}.md", "w", encoding="utf-8") as f:
        f.write("\n\n".join(md_lines))

    # .json — structured with metadata
    doc = {
        "title": book_title,
        "total_paragraphs": len(paragraphs),
        "paragraphs": paragraphs
    }
    with open(f"{output_dir}/{book_title}.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
```

---

## 6. Queue Architecture

Processing 200+ pages in a single loop risks memory overflow and provides no fault tolerance. A queue-based architecture distributes pages across worker processes.

### 6.1 Design

```
[Producer Process]
  - Reads PDF, converts to images
  - Enqueues each page as a job: {page_id, image_path, metadata}

[Redis Queue]
  - Stores pending jobs
  - Acts as buffer between producer and workers

[Worker Processes (N workers)]
  - Each worker pops a job, runs the full pipeline for that page
  - Writes result to Result Store

[Result Assembler]
  - Collects all page results in order
  - Assembles final document
  - Writes output files
```

### 6.2 Implementation with Celery + Redis

```python
# tasks.py
from celery import Celery

app = Celery("ocr_pipeline", broker="redis://localhost:6379/0",
             backend="redis://localhost:6379/1")

@app.task(bind=True, max_retries=3)
def process_page(self, page_id: int, image_path: str, metadata: dict):
    try:
        image     = load_image(image_path)
        lang, _   = detect_language_from_image(image)
        processed = preprocess(image)
        ocr_out   = extract_text(processed, LANG_OCR_MAP.get(lang, LANG_OCR_MAP["en"]))
        quality   = compute_page_quality(ocr_out)
        corrected = selective_correction(ocr_out, quality, lang)
        layout    = reconstruct_layout(corrected, *image.shape[:2])
        paras     = reconstruct_paragraphs(layout)

        return {"page_id": page_id, "paragraphs": paras, "lang": lang, "quality": quality}

    except Exception as exc:
        raise self.retry(exc=exc, countdown=5)  # retry after 5s, up to 3 times
```

```python
# producer.py
from tasks import process_page
from pdf2image import convert_from_path

def submit_book(pdf_path: str, output_dir: str):
    images = convert_from_path(pdf_path, dpi=300, grayscale=True)
    jobs = []

    for i, img in enumerate(images):
        img_path = f"/tmp/pages/page_{i:04d}.png"
        img.save(img_path)
        job = process_page.delay(
            page_id=i,
            image_path=img_path,
            metadata={"source_pdf": pdf_path, "output_dir": output_dir}
        )
        jobs.append(job)

    results = [job.get(timeout=120) for job in jobs]
    results.sort(key=lambda r: r["page_id"])
    return results
```

```bash
# Start workers
celery -A tasks worker --concurrency=4 --loglevel=info
```

### 6.3 Lightweight Alternative (No Redis)

For local development without Redis, use Python's built-in multiprocessing:

```python
from multiprocessing import Pool

def process_all_pages(images: list, num_workers: int = 4) -> list:
    page_args = [(i, img) for i, img in enumerate(images)]
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(process_single_page, page_args)
    return sorted(results, key=lambda r: r["page_id"])
```

---

## 7. Multilingual Handling

### 7.1 Language Routing Table

| Language | ISO Code | IndicOCR Lang | IndicBART Lang Tag | Tesseract Pack |
|---|---|---|---|---|
| Tamil | `ta` | `tam` | `ta_IN` | `tam` |
| Hindi | `hi` | `hin` | `hi_IN` | `hin` |
| Telugu | `te` | `tel` | `te_IN` | `tel` |
| Malayalam | `ml` | `mal` | `ml_IN` | `mal` |
| Kannada | `kn` | `kan` | `kn_IN` | `kan` |
| Bengali | `bn` | `ben` | `bn_IN` | `ben` |
| Odia | `or` | `ori` | `or_IN` | `ori` |
| Punjabi | `pa` | `pan` | `pa_IN` | `pan` |
| English | `en` | — | `en_XX` | `eng` |

### 7.2 Code-Mixed Page Handling

Pages that mix two languages (e.g., Tamil text with English technical terms) are handled with a confidence-merge strategy:

```python
def handle_code_mixed(image, primary_lang: str) -> list[dict]:
    indic_results = run_indic_ocr(image, LANG_OCR_MAP[primary_lang]["lang"])
    eng_results   = run_tesseract(image, lang="eng")

    merged = []
    for indic, eng in zip(indic_results, eng_results):
        # Keep whichever engine was more confident for each word region
        merged.append(indic if indic["confidence"] >= eng["confidence"] else eng)

    return merged
```

---

## 8. Evaluation Metrics

### 8.1 OCR Quality Metrics

**Character Error Rate (CER)** — primary metric for Indic scripts

```
CER = (Substitutions + Insertions + Deletions) / Total Ground-Truth Characters
```

Target: CER < 5% for clean scans, < 12% for degraded scans.

**Word Error Rate (WER)**

```
WER = (Substituted Words + Inserted Words + Deleted Words) / Total Ground-Truth Words
```

Target: WER < 8% for clean scans.

### 8.2 Structural Accuracy

- **Paragraph detection accuracy** — ratio of correctly segmented paragraphs vs ground truth
- **Heading detection precision/recall** — important for downstream indexing
- **Column order accuracy** — fraction of multi-column pages correctly linearized

### 8.3 Evaluation Script

```python
from jiwer import wer, cer

def evaluate_page(predicted_text: str, ground_truth: str) -> dict:
    return {
        "cer": cer(ground_truth, predicted_text),
        "wer": wer(ground_truth, predicted_text),
    }
```

---

## 9. Performance Optimization

| Technique | Impact | How |
|---|---|---|
| GPU acceleration for OCR | 3–5× speed | Pass `gpu=True` to IndicOCR/EasyOCR |
| Confidence-gated LM correction | 40–60% cost reduction | Skip IndicBART for high-confidence pages |
| Batch OCR inference | 2–3× speed | Pass multiple images in one forward pass |
| Parallel workers (Celery) | Linear with CPU cores | `--concurrency=N` |
| PNG → memory (skip disk) | Reduced I/O | Use `io.BytesIO` instead of temp files |
| IndicBART quantization (INT8) | 2× faster inference | `load_in_8bit=True` in transformers |

---

## 10. Project Structure

```
multilingual_ocr_pipeline/
│
├── pipeline/
│   ├── __init__.py
│   ├── pdf_converter.py        # Step 5.1 — PDF to images
│   ├── lang_detector.py        # Step 5.2 — Language detection
│   ├── preprocessor.py         # Step 5.3 — OpenCV preprocessing
│   ├── ocr_engine.py           # Step 5.4 — IndicOCR + Tesseract
│   ├── quality_gate.py         # Step 5.5 — Confidence scoring
│   ├── layout.py               # Step 5.6 — Layout reconstruction
│   ├── cleaner.py              # Step 5.7 — Rule-based cleaning
│   ├── lm_corrector.py         # Step 5.8 — IndicBART correction
│   ├── paragraph_builder.py    # Step 5.9 — Paragraph reconstruction
│   └── output_writer.py        # Step 5.10 — Output generation
│
├── queue/
│   ├── tasks.py                # Celery task definitions
│   ├── producer.py             # Job submission
│   └── assembler.py            # Result collection + ordering
│
├── evaluation/
│   └── metrics.py              # CER, WER, structural accuracy
│
├── models/
│   └── lid.176.bin             # fastText language detection model
│
├── outputs/                    # Generated .txt, .md, .json files
│
├── config.py                   # LANG_OCR_MAP, thresholds, model paths
├── main.py                     # Entry point
├── requirements.txt
└── README.md
```

> **Note:** An `agents/` directory is not part of the current implementation. It will be introduced as part of the agentic upgrade. See [Section 12.1](#121-agentic-flow-upgrade).

---

## 11. Setup & Installation

```bash
# 1. Clone and set up environment
git clone https://github.com/your-org/multilingual-ocr-pipeline
cd multilingual-ocr-pipeline
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install system dependencies
sudo apt-get install -y tesseract-ocr poppler-utils
sudo apt-get install -y tesseract-ocr-tam tesseract-ocr-hin \
     tesseract-ocr-tel tesseract-ocr-mal tesseract-ocr-kan

# 4. Download fastText language detection model
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P models/

# 5. Start Redis (for Celery queue)
docker run -d -p 6379:6379 redis:alpine

# 6. Start workers
celery -A queue.tasks worker --concurrency=4 --loglevel=info

# 7. Run pipeline
python main.py --input path/to/book.pdf --output outputs/ --workers 4
```

**requirements.txt**
```
pdf2image
opencv-python
easyocr
pytesseract
fasttext
transformers>=4.30
torch
celery
redis
jiwer
numpy
Pillow
unicodedata2
indic-nlp-library
```

---

## 12. Future Enhancements

| Enhancement | Description | Priority |
|---|---|---|
| **Agentic flow** | Intelligent orchestration with dynamic routing, quality gates, self-healing retries | High |
| Searchable PDF output | Embed extracted text as invisible layer in original PDF | High |
| NCERT Q&A integration | Feed output `.json` into vector DB for downstream question answering | High |
| Web UI | Upload PDF, monitor job progress, download outputs | Medium |
| Table extraction | Detect and extract tabular data as CSV/JSON | Medium |
| Summarization | Auto-generate chapter summaries using IndicBART | Medium |
| LayoutLMv3 integration | Replace heuristic column detection with trained model | Medium |
| Active learning loop | Flag uncertain corrections for human review → retrain | Low |
| Handwritten text support | HTR via TrOCR for handwritten manuscripts | Low |

---

### 12.1 Agentic Flow Upgrade

Once the standard pipeline is stable and evaluated on real books, the recommended next evolution is replacing the fixed sequential flow with an **agentic orchestration layer**. Instead of every page following the identical path, a set of specialized agents dynamically route each page based on its actual content and quality.

The key advantage over the standard pipeline is **adaptive behaviour**: a clean, high-confidence page skips correction entirely, a degraded page gets escalating preprocessing attempts, and an unrecoverable page is flagged for manual review without halting the rest of the job.

#### Agent Roles

| Agent | Responsibility |
|---|---|
| **Orchestrator Agent** | Coordinates all other agents, manages state, decides routing |
| **OCR Agent** | Selects OCR engine per page, runs extraction, reports confidence |
| **Quality Agent** | Evaluates CER/WER, confidence distributions, flags bad pages |
| **Correction Agent** | Runs IndicBART only on flagged pages |
| **Layout Agent** | Detects multi-column, tables, figures, headings |
| **Retry Agent** | Handles failed pages with escalating preprocessing strategies |
| **Assembler Agent** | Merges all page results into the final document |

#### Orchestrator State Machine

```
State: PENDING
   │
   ▼
[OCR Agent runs]
   │
   ▼
State: OCR_DONE
   │
   ▼
[Quality Agent evaluates]
   │
   ├── PASS (conf ≥ 0.80) ──────────────────────────────────────► State: LAYOUT
   │
   └── FAIL (conf < 0.80)
         │
         ▼
      State: CORRECTION_NEEDED
         │
         ▼
      [Correction Agent runs IndicBART]
         │
         ▼
      [Quality Agent re-evaluates]
         │
         ├── PASS ────────────────────────────────────────────────► State: LAYOUT
         │
         └── FAIL (2nd time)
               │
               ▼
            [Retry Agent — escalate preprocessing]
               │
               ├── Attempt 3: Heavy denoise + upscale → re-OCR
               │
               └── Attempt 4: Flag as MANUAL_REVIEW, skip, continue
```

#### Implementation Sketch

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class PageState(Enum):
    PENDING           = "pending"
    OCR_DONE          = "ocr_done"
    CORRECTION_NEEDED = "correction_needed"
    LAYOUT            = "layout"
    COMPLETE          = "complete"
    MANUAL_REVIEW     = "manual_review"

@dataclass
class PageJob:
    page_id:    int
    image_path: str
    state:      PageState = PageState.PENDING
    ocr_result: Optional[list] = None
    quality:    Optional[dict] = None
    paragraphs: Optional[list] = None
    attempts:   int = 0
    lang:       str = "en"
    errors:     list = field(default_factory=list)


class OrchestratorAgent:
    def __init__(self):
        self.ocr_agent        = OCRAgent()
        self.quality_agent    = QualityAgent()
        self.correction_agent = CorrectionAgent()
        self.layout_agent     = LayoutAgent()

    def process(self, job: PageJob) -> PageJob:
        while job.state not in (PageState.COMPLETE, PageState.MANUAL_REVIEW):
            job = self._step(job)
        return job

    def _step(self, job: PageJob) -> PageJob:
        if job.state == PageState.PENDING:
            job = self.ocr_agent.run(job)
            job.state = PageState.OCR_DONE

        elif job.state == PageState.OCR_DONE:
            job.quality = self.quality_agent.evaluate(job)
            job.state = PageState.CORRECTION_NEEDED if job.quality["needs_correction"] \
                        else PageState.LAYOUT

        elif job.state == PageState.CORRECTION_NEEDED:
            job.attempts += 1
            if job.attempts > 3:
                job.state = PageState.MANUAL_REVIEW
                job.errors.append("Max correction attempts exceeded")
            else:
                job = self.correction_agent.run(job)
                job.quality = self.quality_agent.evaluate(job)
                job.state = PageState.LAYOUT if not job.quality["needs_correction"] \
                            else PageState.CORRECTION_NEEDED

        elif job.state == PageState.LAYOUT:
            job = self.layout_agent.run(job)
            job.state = PageState.COMPLETE

        return job
```

#### What Changes in the Project Structure

The `pipeline/` modules are **reused as-is** — each agent simply wraps the existing step functions. No rewrite of existing code is needed; the agentic layer is purely an orchestration upgrade on top.

```
multilingual_ocr_pipeline/
│
├── pipeline/           # unchanged — same step modules
│   └── ...
│
├── agents/             # NEW — added only during the agentic upgrade
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── ocr_agent.py
│   ├── quality_agent.py
│   ├── correction_agent.py
│   ├── layout_agent.py
│   └── retry_agent.py
│
└── ...
```

---

*Document version: 2.1 | Last updated: April 2026*
