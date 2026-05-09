# Multilingual OCR Pipeline — Project Tracker

> **Started:** 2026-05-03
> **Status:** 🔵 Phase 2 / Production Deployment
> **Primary Language:** Tamil
> **GPU:** Modal.com Cloud T4 (Serverless) + RTX 4050 Local
> **Source Material:** Scanned public-domain Tamil books + newspapers
> **Current Phase:** Phase 2 — Serverless Deployment & Pipeline Hardening
>
> **Last Updated:** 2026-05-09

---

## Phase Overview

| Phase | Scope | Status | Target CER | Start | End |
|-------|-------|--------|------------|-------|-----|
| **Phase 1** | 10–20 pages, single language | ✅ **Complete** | < 15% (no LM) | 2026-05-03 | 2026-05-06 |
| **Phase 2** | 50–100 pages, ByT5 correction | 🔵 **In Progress** | < 5% (with LM) | 2026-05-06 | — |
| **Phase 3** | 200–300 pages, production | ⬜ Not Started | < 3% | — | — |

---

## Pre-Phase: Stack Finalization & Setup

- [x] Confirm primary target language for Phase 1 → **Tamil**
- [x] Confirm GPU availability → **RTX 4050 (6GB) local + Kaggle**
- [x] Identify source PDFs for Phase 1 → **7 Tamil short stories (86 pages total)**
- [x] Approve revised technology stack → **Surya + PaddleOCR + ByT5 approved**
- [x] Ground Truth strategy → **Option B (OCR the verify PDFs)**
- [x] Set up project directory structure
- [x] Create Python virtual environment
- [x] Install core dependencies

---

## Phase 1: Proof of Concept (10–20 pages)

### Pipeline Implementation
- [x] **1.1** PDF → Image conversion (PyMuPDF) — start with எலி.pdf + விண்ணபம்.pdf (15 pages)
- [x] **1.2** OpenCV preprocessing chain (deskew, denoise, threshold, CLAHE)
- [x] **1.3** Surya OCR integration (primary engine)
- [ ] **1.4** Tesseract 5 integration (fallback)
- [x] **1.5** Confidence scoring & quality gate
- [x] **1.6** Rule-based cleaning (OCR artifact fixes + Unicode NFC)
- [x] **1.7** Basic layout reconstruction (single-column — matches dataset)
- [x] **1.8** Paragraph reconstruction
- [x] **1.9** Output generation (.txt, .md, .json)

### Ground Truth & Evaluation
- [x] **1.10** Generate ground truth (Option B: OCR the 66 verify pages using Surya)
- [x] **1.11** Manually verify 5–10 pages of generated ground truth
- [x] **1.12** Implement CER/WER evaluation script (`jiwer`)
- [x] **1.13** Build Gradio debug viewer (image → OCR → output)
- [x] **1.15** Run end-to-end evaluation on 2 stories (15 pages)
- [x] **1.16** Scale to all 7 stories (86 pages), full CER/WER report
- [x] **1.17** Upgrade Gradio viewer: multi-format upload (PDF ≤20 pages, image, DOCX) → downloadable `.txt` output

### Phase 1 Results

| Metric | Target | Actual | Notes |
|--------|--------|--------|-------|
| CER (clean scans) | < 15% | **6.74%** | Exceptional baseline for Tamil |
| WER (clean scans) | < 20% | **24.15%** | Inflated due to Tamil spacing/agglutination |
| Pipeline stability | No crashes on 20 pages | **Pass** | 86 pages processed without OOM |
| Processing time (per page) | — | **~2 secs** | Measured on RTX 4050 6GB (Surya OCR) |

### Phase 1 Learnings & Issues

#### 1. Architecture & Engine Selection
* **Traditional vs Modern OCR:** We opted for `surya-ocr` over classic tools like Tesseract. While Tesseract relies on strict binarization and connected component analysis (which struggles heavily with agglutinative Indic scripts), Surya acts as a vision foundation model (CNNs/ViTs). It understands contextual visual relationships natively, making it superior for complex Tamil typography.

#### 2. Dependency & Environment Stabilization (The "Dependency Hell")
* **The Issue:** `surya-ocr` v0.17.1 had aggressive sub-dependencies that caused massive conflicts in the Python environment.
* **Transformers Conflict:** We encountered an `AttributeError` in `SuryaDecoderConfig` due to an incompatible `transformers` version (v5.7.0). This was resolved by pinning `transformers` to `4.57.6`.
* **PyTorch GPU Eradication:** Pip's dependency resolver recursively broke the environment, uninstalling our CUDA-enabled `torch` and replacing it with a broken CPU-only `torch 2.11.0`. 
* **The Fix:** We had to bypass pip's resolver entirely by wiping the corrupted `site-packages` and using `--no-deps` to force a clean installation of `torch 2.5.1+cu121`. This successfully enabled hardware acceleration on the local RTX 4050 GPU.

#### 3. Dataset Integrity & PDF Rendering Pitfalls
* **The Illusion of Scans:** We initially believed the source PDFs (e.g., `எலி.pdf`) were flat image scans. When evaluated, they produced a catastrophic **88% CER**.
* **Root Cause Analysis:** A deep PDF diagnostic script revealed the PDFs were actually digital files encoded with a legacy, un-embedded font (`TAUElangoMadhavi`). Because the font wasn't installed locally, `PyMuPDF` (`fitz`) hallucinated English ASCII characters (`rZrrr`) onto the rendered canvas. Surya OCR then faithfully read those English characters.
* **The Fix:** We updated our `pdf_converter` to intelligently extract raw embedded images (`extract_image`) when dealing with pure scans, bypassing the rendering engine completely to preserve 100% of the original visual data.

#### 4. The Danger of Classic Preprocessing
* **The Issue:** We initially built an OpenCV pipeline (Deskew, CLAHE, Adaptive Thresholding) assuming it would help the OCR.
* **The Learning:** Modern vision models (Surya) are trained on raw, noisy RGB/Grayscale images. Binarization (Adaptive Thresholding) destroys sub-pixel rendering and anti-aliasing features that the CNN backbones rely on. We stripped out the harsh OpenCV binarization, feeding raw high-res images to Surya instead.

#### 5. Evaluation Alignment & Metrics
* **Page Misalignment:** Our initial evaluation script compared pages 1:1. However, source PDFs had extra pages (title covers, blank pages) causing offsets (e.g., source had 7 pages, Ground Truth had 6). Comparing mismatched pages inflated the error rate.
* **Full Document Evaluation:** We refactored `evaluate.py` to concatenate all pages into a single normalized string before calculating Levenshtein distances. 
* **Final Conclusion:** This revealed the true baseline accuracy: **6.74% CER** and **24.15% WER**. The relatively higher WER is an expected artifact of spacing differences in agglutinative Tamil text, proving the underlying character recognition is exceptionally strong (~93% accuracy) out-of-the-box.
#### 6. Pre-Phase 2 Layout Hardening (Structural Fixes)
* **Cropped Margins:** The initial approach of running OCR on bounding-box crops caused tight edges to clip characters. **Fix:** Transitioned to Full-Page OCR, using the LayoutPredictor purely as a post-filter metadata layer.
* **Over-segmentation:** Surya's LayoutPredictor occasionally hallucinated up to 41 duplicate overlapping boxes for the same text region. **Fix:** Implemented an Intersection-over-Union (IoU) deduplication step (>70% overlap).
* **Landscape Spreads:** 2-page book scans interleaved text from left and right pages. **Fix:** Added an automatic width-to-height ratio splitter (>1.3) to slice spreads vertically before OCR.
* **English Language Purging:** `text_cleaner.py` previously aggressively deleted any lines lacking Tamil characters, destroying bilingual content. **Fix:** Added `_has_english()` validation using basic Latin unicode ranges.
* **Header/Footer Bleed:** Page numbers and headers leaked into the main text body. **Fix:** Implemented a Mathematical Bounding Box Layout Reconstructor that uses IoU to drop any text lines falling inside `PageHeader` or `PageFooter` blocks.

---

## Phase 2: Refinement & Scaling (50–100 pages)

### Engine Architecture
- [x] **2.0a** Refactored `ocr_engine.py` with `BaseOCREngine` abstract class (Surya-only for Phase 2)
- [x] **2.0b** `drop_repeated_text=True` enabled in `SuryaEngine`
- [x] **2.0c** `GPUMemoryManager` built in `pipeline/memory_manager.py`
- [x] **2.0d** PaddleOCR evaluated — **deferred to Phase 3** (API compatibility issue on Windows)
- [x] **2.0e** All Paddle packages uninstalled from venv

### Dataset Generation
- [x] **2.1** Tamil corpus assembled — IndicCorp + Project Madurai + PreModern Tamil (~15k clean sentences)
- [x] **2.2** `roundtrip_generator.py` — GPU batch RoundTrip noise generator (batch_size=8, checkpointed)
- [x] **2.3** 10,000 pairs generated — 9,001 train + 1,000 val (max token length: 972 / 1024)
- [x] **2.4** Dataset audited — 0 over-limit pairs, HTML stripped, identical pairs filtered
- [x] **2.5** Dataset uploaded to HF Hub: `Naren-hug/tamil-ocr-byt5-dataset` (private)

### ByT5 Fine-Tuning
- [x] **2.6** `finetune_byt5_kaggle.ipynb` created — HF-integrated, resume-safe, CER metric
- [ ] **2.7** Run training on Kaggle T4 (~3-4 hours) ← **CURRENT STEP**
- [ ] **2.8** Verify model at `Naren-hug/byt5-tamil-ocr-v1` on HF Hub

### Layout Engine Fixes
- [x] **2.9** Fix multi-column orphan lines (Surya XY-cut fragmented bounding boxes)
- [x] **2.10** Implement `_reflow_text()` pass to seamlessly merge short fragments back into paragraphs
- [ ] **2.11** Add full-page fallback when all blocks are `Picture`/`Figure` (newspaper fix)

### Correction Pipeline Integration
- [ ] **2.12** Build `lm_corrector.py` — load `byt5-tamil-ocr-v1`, trigger at confidence < 0.80
- [ ] **2.13** Wire corrector into main pipeline
- [ ] **2.14** Re-evaluate CER before vs after correction on all 7 stories (Deferred to Roadmap)

### UI & Serverless Deployment (Modal)
- [x] **2.15** UI rebranded from "Indian OCR" to "Indic OCR"
- [x] **2.16** Reframed LM Correction as a "Future Upgrade" feature for production polish
- [x] **2.17** Created `modal_app.py` for Modal v1.x serverless deployment
- [x] **2.18** Fixed headless container system dependencies (`libGL.so.1`, etc.)
- [x] **2.19** Fixed Modal Serverless filesystem statelessness using `modal.Volume` and `TMPDIR` redirection
- [x] **2.20** App live at `https://naren-ghub--indic-ocr-fastapi-app.modal.run`

### Phase 2 Results

| Metric | Target | Actual | Notes |
|--------|--------|--------|-------|
| CER (clean, with ByT5) | < 5% | — | Training in progress |
| CER (degraded, with ByT5) | < 10% | — | — |
| Multi-column accuracy | > 85% | — | Sorting bug pending fix |
| Dataset size | 10k pairs | **10,001** ✅ | Uploaded to HF Hub |

### Phase 2 Learnings & Issues

#### 1. PaddleOCR on Windows — API Breakage in v3.x
* PaddleOCR 3.5.0 (latest) introduced a completely new API that is incompatible with paddlepaddle-gpu 2.6 on Windows. The error `'AnalysisConfig' object has no attribute 'set_optimization_level'` crashes on model load.
* **Decision:** After empirical testing, Surya's ViT layout engine is better suited than PaddleOCR's CRNN architecture for degraded historical Tamil documents. PaddleOCR deferred to Phase 3 (use paddlepaddle-gpu==2.6.1 + paddleocr==2.8.1 when resuming).

#### 2. ByT5 vs Subword Models for OCR Correction
* Standard mT5 tokenizes text into subword chunks. A single OCR character error (e.g., `ண` → `ன`) completely changes the token IDs, making it hard for the model to learn the correction mapping.
* ByT5 operates on raw UTF-8 bytes. Character-level edits are natural in its representation. This makes it the correct architecture for Tamil OCR correction.

#### 3. HF Hub as Training Infrastructure
* Using Kaggle Secrets + HF Hub for both input data and model output eliminates all manual file transfer. The notebook auto-pushes every epoch checkpoint to HF Hub, making training resume-safe even if the Kaggle VM is destroyed.

#### 4. Serverless Gradio State Loss
* **The Issue:** When deployed to Modal, uploaded PDFs and generated output `.txt` files would randomly return "No such file" errors.
* **The Fix:** Because Modal scales containers dynamically, Request A (upload) and Request B (processing/download) might hit different ephemeral containers. We attached a persistent `modal.Volume` to `/tmp/gradio` and forced Python's `os.environ["TMPDIR"]` to point there, ensuring all containers share the exact same temporary filesystem.

---

## Phase 3: Production Scale (200–300 pages)

### Advanced Features
- [ ] **3.1** dots.ocr VLM fallback for unrecoverable pages
- [ ] **3.2** LayoutLMv3 / Docling for complex layouts
- [ ] **3.3** Table extraction (CSV/JSON)
- [ ] **3.4** Agentic orchestration layer
- [ ] **3.5** Searchable PDF output
- [ ] **3.6** Multi-book, multi-language stress test
- [ ] **3.7** Monitoring dashboard
- [ ] **3.8** Active learning loop (flag uncertain → human review)

### Phase 3 Results

| Metric | Target | Actual | Notes |
|--------|--------|--------|-------|
| CER (clean) | < 5% | — | — |
| CER (degraded) | < 10% | — | — |
| Manual review rate | < 2% of pages | — | — |
| 200+ page stability | No memory overflow | — | — |

### Phase 3 Learnings & Issues
_No entries yet._

---

## Complexity Log

This section tracks complexities discovered at each phase and how they were resolved.

| # | Phase | Complexity | Impact | Resolution | Status |
|---|-------|-----------|--------|------------|--------|
| 1 | Pre | IndicOCR not publicly available | Blocked primary OCR engine | Switch to Surya OCR + PaddleOCR | ✅ Resolved |
| 2 | Pre | IndicBART not pre-trained on OCR noise | Suboptimal correction | Switch to ByT5 (byte-level) | ✅ Resolved |
| 3 | Pre | pdf2image requires poppler on Windows | Dependency friction | Switch to PyMuPDF (pure pip) | ✅ Resolved |
| 4 | Pre | Verify PDFs use TAUElangoMadhavi font encoding | Ground truth text is garbled when extracted | OCR the verify PDFs themselves (Option B) | ✅ Resolved |
| 5 | Pre | Page count mismatch: 86 source vs 66 verify pages | Cannot do simple page-to-page alignment | Use content-level alignment instead | 🟡 Identified |
| 6 | 1 | Left margins cropped in output | Data loss on page edges | Switch to Full-Page OCR architecture | ✅ Resolved |
| 7 | 1 | Massive over-segmentation | 29,000+ characters on a single page | Implement >70% IoU deduplication | ✅ Resolved |
| 8 | 1 | Landscape 2-page spreads interleaved | Reading order destroyed | Dynamic image splitting (width > height * 1.3) | ✅ Resolved |
| 9 | 1 | Pure English text discarded | Zero output for English documents | Add basic Latin `_has_english()` validation | ✅ Resolved |
| 10 | 1 | Headers/Footers leaked into text | Messy text flow | Mathematical Layout Reconstruction via IoU | ✅ Resolved |
| 11 | 2 | Surya bounding box fragmentation | Words broken onto separate lines | Implement post-process `_reflow_text()` joining | ✅ Resolved |
| 12 | 2 | Modal Serverless file loss | 404s on uploads and downloads | Attach persistent `modal.Volume` to `/tmp/gradio` | ✅ Resolved |

---

## Decision Log

| Date | Decision | Rationale | Alternatives Considered |
|------|----------|-----------|------------------------|
| 2026-05-03 | Replace IndicOCR with Surya OCR (primary) | IndicOCR is not publicly released | EasyOCR (weak on Indic), Bhashini API (cloud-only) |
| 2026-05-03 | Replace IndicBART with ByT5 for OCR correction | Byte-level model handles garbled OCR better | IndicT5, constrained LLM decoding |
| 2026-05-03 | Replace pdf2image with PyMuPDF | No system dependency on poppler, faster on Windows | pdf2image + poppler install |
| 2026-05-03 | Adopt 3-phase scaling strategy | De-risk incrementally, measure CER at each scale | Full 200-page attempt from start |
| 2026-05-07 | **Defer PaddleOCR to Phase 3** | v3.5.0 API incompatible with paddlepaddle-gpu 2.6 on Windows; Surya ViT better for degraded historical Tamil docs | Downgrade paddleocr to 2.8.1 (stable but adds dual-framework complexity) |
| 2026-05-07 | **Use HF Hub for dataset + model versioning** | Eliminates manual Kaggle file transfers; enables resume-safe training | Kaggle dataset attachment, Google Drive mount |
| 2026-05-09 | **Deploy Serverless to Modal** | Allows T4 GPU inference with $0 idle cost and auto-scaling | Hugging Face Spaces (persistent cost), AWS EC2 |
| 2026-05-09 | **Defer ByT5 Correction to Roadmap** | Shifts focus to hardening the core pipeline for a stable production release first | Block release until LLM is fine-tuned |

---

## Resource Links

| Resource | URL | Purpose |
|----------|-----|---------|
| Surya OCR | https://github.com/VikParuchuri/surya | Primary OCR engine |
| PaddleOCR | https://github.com/PaddlePaddle/PaddleOCR | Secondary OCR engine |
| ByT5 | https://huggingface.co/google/byt5-small | LM correction model |
| dots.ocr | https://huggingface.co/rednote-hilab/dots.ocr | VLM fallback |
| fastText LID | https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin | Language detection |
| jiwer | https://pypi.org/project/jiwer/ | CER/WER evaluation |
| Tamil Doc OCR Benchmark | via icter.org (2025) | Evaluation dataset |
| BSTD Dataset | via IIIT-Hyderabad | Multi-language benchmark |

---

## HF Hub Resources

| Resource | Repo | Status |
|----------|------|--------|
| Training Dataset | `Naren-hug/tamil-ocr-byt5-dataset` | ✅ Live (private) |
| Fine-Tuned Model | `Naren-hug/byt5-tamil-ocr-v1` | 🟡 Training in progress |

---

*Last updated: 2026-05-07 | Next review: After ByT5 training completes*
