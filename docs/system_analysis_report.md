# OCR System Analysis Report (Tamil + Indic Books)

Date: 2026-05-04  
Scope: `D:\Evolve_Robot_Lab\Project\NLP Projects\OCR` (plus the embedded pipeline under `OCR/multilingual_ocr_pipeline`)

## 1) What this system is (today)

This repository contains a **Phase 1 Tamil OCR pipeline** built around **Surya OCR** (GPU-first) with:

- PDF → image extraction/rendering using **PyMuPDF** (`fitz`)
- Optional **layout detection** using **Surya LayoutPredictor** (block-aware OCR)
- Rule-based **Tamil-focused text cleaning** (strip HTML tags, foreign-script hallucinations, inline page numbers)
- Output writers for per-page `.txt` plus document-level JSON summaries
- Ground-truth generation (Option B: OCR the “(verify)” PDFs) and **CER/WER evaluation** using `jiwer`
- A **Gradio** debug viewer that runs OCR on uploaded PDFs/images/DOCX (≤ 20 pages)

Important: the design doc describes a broader “Multilingual OCR + NLP pipeline” (language detection, multi-engine routing, LM correction, queues), but **the current code implements the Phase 1 Surya-based core only** (Tamil-first, sequential, no LM correction, no Redis/Celery).

## 2) Current project layout (key paths)

Top-level (`OCR/`):

- `OCR/gradio_app.py` — interactive debug viewer (single-engine Surya OCR, no layout reconstruction)
- `OCR/run_all.py` — batch runner for all PDFs in `OCR/OCR_dataset/`, then runs evaluation
- `OCR/analyze_pdfs.py` — diagnostics for PDFs (fonts/images/text extractability/render sanity)
- `OCR/debug_extraction.py` — compares PyMuPDF render (`get_pixmap`) vs embedded image extraction (`extract_image`)
- `OCR/generate_ground_truth_b.py` — generates ground truth by OCR-ing the “(verify)” PDFs
- `OCR/evaluation/evaluate.py` — CER/WER evaluation; writes `evaluation_report.json` under each story output directory
- `OCR/Doc/multilingual_ocr_nlp_pipeline_v1.md` — technical design document (aspirational architecture)
- `OCR/Doc/project_tracker.md` — what was built + Phase 1 results and issues

Pipeline package (`OCR/multilingual_ocr_pipeline/`):

- `OCR/multilingual_ocr_pipeline/main.py` — batch pipeline entry point (Phase 1)
- `OCR/multilingual_ocr_pipeline/config.py` — central thresholds/paths
- `OCR/multilingual_ocr_pipeline/pipeline/pdf_converter.py` — PDF → images (extract embedded image for pure scans)
- `OCR/multilingual_ocr_pipeline/pipeline/ocr_engine.py` — Surya OCR wrapper (recognition + detection)
- `OCR/multilingual_ocr_pipeline/pipeline/layout_engine.py` — Surya layout detection + block-aware OCR
- `OCR/multilingual_ocr_pipeline/pipeline/text_cleaner.py` — Tamil artifact cleaner (Unicode-range filtering + page-number stripping)
- `OCR/multilingual_ocr_pipeline/pipeline/preprocessor.py` — OpenCV preprocessing chain (**present but not wired into `main.py` yet**)
- `OCR/multilingual_ocr_pipeline/pipeline/postprocessor.py` — basic cleaning + confidence + a *different* `LayoutEngine` class (**name collision risk; not used by `main.py`**)

Data/artifacts:

- `OCR/OCR_dataset/` — input PDFs (source + `(verify)` PDFs)
- `OCR/multilingual_ocr_pipeline/outputs/<story>/` — pipeline outputs
- `OCR/multilingual_ocr_pipeline/data/page_images/<story>/` — saved page images for debugging/Gradio
- `OCR/evaluation/ground_truth/<story>/` — generated “ground truth” page text (from `(verify)` PDFs)

## 3) End-to-end pipeline flow (as implemented)

### 3.1 Batch pipeline (`OCR/multilingual_ocr_pipeline/main.py`)

High-level flow per PDF:

#### Diagram (layout blocks → OCR → assembly)

```mermaid
flowchart TD
  A[PDF Page] --> B[PDF→Image<br/>PyMuPDF: extract_image or render pixmap]
  B --> C[Layout Detection<br/>Surya LayoutPredictor]
  C --> D{Any usable blocks?<br/>(conf ≥ min_confidence)}
  D -- No --> E[Fallback: single full-page block<br/>label=Text, position=0]
  D -- Yes --> F[Layout Blocks<br/>label + bbox + position + confidence]

  E --> G
  F --> G[Filter blocks<br/>skip Picture/Figure/Table/Equation]

  G --> H[Crop each block bbox<br/>(+padding)]
  H --> I[Per-block OCR<br/>Surya RecognitionPredictor]
  I --> J[Clean block text<br/>strip HTML/foreign chars/page nums]
  J --> K[Attach text to block<br/>block.ocr_text]

  K --> L[Sort blocks by position]
  L --> M[Assemble page text<br/>headers spaced, body joined]
  M --> N[Write outputs<br/>page_###.txt + document_results.json]
```

1. **PDF → images**  
   `pipeline.pdf_converter.pdf_to_images()` loops pages and chooses one of:
   - Extract embedded image directly (`extract_image`) if page looks like a pure scan (exactly 1 image and no selectable text)
   - Otherwise render the page to a pixmap at `PDF_DPI` (default 300)

2. **Per-page layout analysis**  
   `pipeline.layout_engine.LayoutEngine.analyze_page()` runs Surya `LayoutPredictor` to return `PageLayout(blocks=[...])`.
   - Filters blocks by `min_confidence` (default 0.30)
   - If nothing usable is returned, falls back to one full-page `"Text"` block

3. **Block-aware OCR**  
   `layout_engine.fill_ocr_text(...)` crops each non-skipped block and calls `OCREngine.process_image(crop)`, then applies `text_cleaner.clean_text()`.
   - The assembled page body text is `layout.body_text` (reading order = `position`)

4. **Confidence reporting (current behavior)**  
   The code re-runs OCR over the **full raw page image** (`ocr_engine.process_image(raw_img)`) and averages line confidences via `Cleaner.calculate_confidence(...)`.
   - This is used only for logging + “quality gate warning”
   - No correction step is applied yet

5. **Final cleaning + outputs**
   - Saves `page_###.txt`
   - Saves `document_results.json` with block metadata + confidence + text per page
   - Saves `<story>_full.txt` concatenating pages with `---` separators

### 3.2 Gradio debug viewer (`OCR/gradio_app.py`)

This path is simpler than the batch pipeline:

- Loads pages from PDF/image/DOCX
- Runs `OCREngine.process_image(img)` on the whole page (no layout detection)
- Applies `text_cleaner.clean_text()` and writes a single downloadable `.txt`

It’s useful for rapid inspection but does not mirror the full layout-aware pipeline behavior.

## 4) OCR engine + layout engine internals

### 4.1 `OCREngine` (Surya)

File: `OCR/multilingual_ocr_pipeline/pipeline/ocr_engine.py`

- Loads (or reuses) `surya.foundation.FoundationPredictor`
- Uses `RecognitionPredictor(..., task_names=["ocr_without_boxes"], det_predictor=DetectionPredictor)`
- Returns:
  - `text` (joined line texts)
  - `lines` with `text`, `bbox` (if present), and `confidence` (if present)
  - `raw_result` (Surya result object)

### 4.2 Layout detection (`LayoutEngine` in `pipeline/layout_engine.py`)

- Wraps `surya.layout.LayoutPredictor`
- Produces `PageLayout` + `LayoutBlock` records
- Applies simple label filtering:
  - `SKIP_LABELS = {"Picture","Figure","Table","Equation"}`
  - `HEADER_LABELS = {"SectionHeader","PageHeader"}`
- Performs crop padding (4px) and per-block OCR

## 5) Text cleaning strategy (Tamil-first)

File: `OCR/multilingual_ocr_pipeline/pipeline/text_cleaner.py`

The cleaner is a practical response to real Surya output artifacts observed in the dataset:

- Removes HTML tags like `<b>`, `<i>`, `<u>`
- Drops characters from non-Tamil Unicode blocks (Devanagari/Malayalam/etc.) that appear as hallucinations
- Strips inline page numbers at line start when Tamil text follows (prevents deleting legitimate numbers embedded in content)
- Drops “garbage lines” that contain no Tamil characters and are longer than trivial punctuation

This cleaner is **Tamil-specific** right now (hard-coded Tamil Unicode block + “foreign” blocks list).

## 6) Evaluation + ground truth workflow

### 6.1 Ground truth generation (Option B)

File: `OCR/generate_ground_truth_b.py`

- Reads PDFs matching `*(verify).pdf`
- Renders each page at 300 DPI and runs Surya OCR
- Writes `OCR/evaluation/ground_truth/<story>/page_###.txt`

This is explicitly a pragmatic approach because some PDFs (esp. “verify” PDFs) may not yield clean text extraction.

### 6.2 CER/WER evaluation

File: `OCR/evaluation/evaluate.py`

- For each story that exists in both pipeline outputs and ground truth:
  - Concatenates all `page_*.txt` to compute **document-level CER/WER**
  - Also prints page-level metrics (informational; can be misaligned)
- Saves `evaluation_report.json` under `OCR/multilingual_ocr_pipeline/outputs/<story>/`

## 7) Environment assumptions / dependencies (inferred from imports)

There is currently **no `requirements.txt` / `pyproject.toml`** tracked under `OCR/`, so this is inferred from code imports:

Required:
- `surya-ocr` (plus its torch/cuda stack)
- `PyMuPDF` (`fitz`)
- `Pillow`
- `numpy`
- `opencv-python` (used by `preprocessor.py`, and some debug scripts)
- `jiwer` (evaluation)

Optional:
- `gradio` (debug UI)
- `python-docx` (DOCX upload support in Gradio)

Windows-specific handling:
- Many scripts call `sys.stdout.reconfigure(encoding="utf-8")` to avoid Unicode issues in terminals.

## 8) Gaps / tech debt (most important)

These are the main “drift points” between the design doc and the code, plus a few correctness/maintainability concerns:

1. **Multilingual routing not implemented yet**
   - No language detection
   - `config.py` includes `TESSERACT_LANGS`, but Tesseract/PaddleOCR integration isn’t wired

2. **No LM correction stage yet**
   - The “quality gate” only logs warnings; it doesn’t route to ByT5/IndicBART or other correction

3. **Preprocessing is unused**
   - `pipeline/preprocessor.py` exists but `main.py` does not call it before OCR

4. **Duplicate OCR pass per page in `main.py`**
   - Block-aware OCR runs on cropped blocks
   - Confidence is computed by running OCR again on the full page
   - This doubles inference cost per page (and can skew confidence vs block-level text)

5. **`LayoutEngine` name collision**
   - There is `pipeline/layout_engine.py::LayoutEngine` (Surya layout predictor)
   - And `pipeline/postprocessor.py::LayoutEngine` (paragraph reconstruction heuristic)
   - Only the former is used in `main.py`, but the shared name will cause confusion later

6. **Hard-coded absolute paths in some scripts**
   - `analyze_pdfs.py`, `debug_extraction.py`, `generate_ground_truth_b.py` embed absolute `d:\...` paths
   - This blocks portability and makes CI/testing harder

7. **No packaging / install story**
   - There’s no pinned dependency list or reproducible env description for the OCR folder

## 9) Recommended next updates (practical, high-leverage)

If you want, we can apply these in small PR-sized steps:

1. **Add a reproducible environment file**
   - Create `OCR/requirements.txt` (or `pyproject.toml`) with pinned versions known to work with your GPU + Surya

2. **Wire preprocessing into the pipeline**
   - Optional per-page toggle (on/off) + save preprocessed debug images

3. **Remove the duplicate full-page OCR pass**
   - Compute confidence from the same OCR results used for text assembly (block-level)
   - Or compute a weighted page confidence across blocks

4. **Resolve `LayoutEngine` naming collision**
   - Rename `pipeline/postprocessor.py::LayoutEngine` to something like `ParagraphReconstructor`

5. **Make scripts path-portable**
   - Replace absolute paths with `Path(__file__).parent`-relative paths and/or CLI args

6. **Lay groundwork for Phase 2**
   - Implement language detection (fastText or simple heuristics) and route to:
     - Surya (Tamil/Indic baseline)
     - PaddleOCR / Tesseract as fallback (per language)
   - Add a pluggable “correction” interface gated by confidence thresholds

## 10) Quick “how to run” (current)

Batch process one PDF:
- Run `OCR/multilingual_ocr_pipeline/main.py` with `--input <pdf>`

Process all source PDFs then evaluate:
- Run `OCR/run_all.py`

Start Gradio viewer:
- Run `OCR/gradio_app.py` (GPU recommended; PDFs limited to 20 pages)

---

If you tell me which direction you want next (Phase 2 multilingual routing vs LM correction vs layout improvements), I can start updating the codebase in that order.
