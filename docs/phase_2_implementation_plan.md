# Phase 2 — Implementation Plan

> **Goal:** Take CER from ~6.7% → <5%, harden remaining failure modes, add LM-based post-correction, and ship a production-quality Gradio UI.
>
> **Directory:** `d:\Evolve_Robot_Lab\Project\NLP Projects\OCR_Phase_2\` — fully separate from Phase 1. Phase 1 code in `OCR/` remains untouched as a stable baseline.

---

## Context: What Phase 1 Already Solved

| Problem | Status | How |
|---------|--------|-----|
| Left-edge text cropping | ✅ Fixed | Full-page OCR (no cropping) |
| 41x block over-segmentation | ✅ Fixed | IoU deduplication (>70%) |
| Landscape 2-page spreads | ✅ Fixed | Auto-split at width > height × 1.3 |
| English text deleted by cleaner | ✅ Fixed | `_has_english()` validator |
| HTML tag leaks (`<mark>`, `<sub>`) | ✅ Fixed | Broadened regex |
| Header/footer bleed into text | ✅ Fixed | BBox-intersection layout reconstruction |
| Language support | ✅ Fixed | `langs=["ta", "en"]` |

## Remaining Open Problems (from Weakness Analysis)

| Problem | Severity | Phase 2 Scope? |
|---------|----------|---------------|
| No empirical engine comparison | 🟠 High | ✅ Yes — Step 2.0 |
| Upside-down / rotated scans | 🔴 Critical | ✅ Yes — Step 2.1 |
| Newspaper dense layout misclassified as `Picture` | 🟠 High | ✅ Yes — Step 2.2 |
| TOC / index pages → number salad | 🟡 Medium | ⬜ Phase 3 (table structure) |
| Bamini legacy font encoding | 🟡 Low | ⬜ Phase 3 (edge case) |

---

## Phase 2 Steps

---

### 2.0 — Surya vs. PaddleOCR Engine Comparison (NEW)

**Goal:** Generate hard empirical data — CER, WER, speed, and failure-mode breakdown — to definitively determine which engine performs best on our specific Tamil documents, then wire the winner as the production default and the loser as the automatic fallback.

> [!IMPORTANT]
> This is a **data-first engineering decision**. We do not assume Surya is always better. We measure it. The output of this step directly informs every downstream architectural choice in Phase 2 and Phase 3.

---

#### 2.0a — PaddleOCR Installation & Dependency Isolation

**The challenge:** Surya runs on PyTorch. PaddleOCR runs on PaddlePaddle (Baidu's equivalent DL framework). Installing both into the same `venv` on Windows can cause CUDA runtime conflicts. We solve this with explicit version pinning.

**Installation strategy:**
```
# Step 1: Install PaddlePaddle GPU build that matches our CUDA version
pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html

# Step 2: Install PaddleOCR (the high-level OCR library on top of PaddlePaddle)
pip install paddleocr>=2.7.3

# Step 3: Verify both frameworks can see the GPU independently
python -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available())"
python -c "import paddle; print('Paddle CUDA:', paddle.device.is_compiled_with_cuda())"
```

**VRAM risk on RTX 4050 (6GB):**
- Surya (detection + recognition models): ~2.5–3.0 GB VRAM
- PaddleOCR (PP-OCRv4 server model): ~1.5–2.0 GB VRAM
- Running both simultaneously: ~4.5–5.0 GB — dangerously tight
- **Solution:** We never load both at the same time. The memory manager (see 2.0c) explicitly unloads one before loading the other.

#### [MODIFY] [requirements.txt](file:///d:/Evolve_Robot_Lab/Project/NLP%20Projects/OCR/multilingual_ocr_pipeline/requirements.txt)
- Add `paddleocr>=2.7.3` with a comment block explaining the CUDA version constraint
- Add `paddlepaddle-gpu==2.6.1.post120` with install URL note
- Add a warning comment: `# NOTE: do not pip install paddlepaddle and paddlepaddle-gpu simultaneously`

---

#### 2.0b — Abstract Engine Factory Pattern

**The problem with the current `ocr_engine.py`:** It is tightly coupled to Surya. Adding PaddleOCR as a second-class citizen would create a messy `if engine == "surya"` branching mess throughout the codebase.

**The solution:** Refactor `ocr_engine.py` around a clean abstract base class so that `main.py`, `gradio_app.py`, and `evaluate.py` are completely engine-agnostic — they just call `engine.process_image(img)` and receive identical output structures regardless of which engine is underneath.

#### [MODIFY] [pipeline/ocr_engine.py](file:///d:/Evolve_Robot_Lab/Project/NLP%20Projects/OCR/multilingual_ocr_pipeline/pipeline/ocr_engine.py)

Refactor into three classes:

**Class 1: `BaseOCREngine` (Abstract Base)**
```python
from abc import ABC, abstractmethod

class BaseOCREngine(ABC):
    """Engine-agnostic interface. All engines return the same output schema."""

    @abstractmethod
    def process_image(self, image: Image.Image) -> dict:
        """
        Returns:
          {
            "lines": [
              {
                "text": str,
                "bbox": [x1, y1, x2, y2],   # normalized 0–1
                "confidence": float           # 0.0 – 1.0
              }, ...
            ],
            "page_confidence": float,         # mean of all line confidences
            "engine": str                     # "surya" | "paddle"
          }
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """Explicitly load models into VRAM."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Explicitly release VRAM. Called by MemoryManager before switching engines."""
        pass
```

**Class 2: `SuryaEngine(BaseOCREngine)`**
- Wraps the existing Surya `OCRPredictor` logic
- `load()` — instantiates `FoundationPredictor` and moves models to `cuda`
- `unload()` — calls `del self._predictor`, then `torch.cuda.empty_cache()`
- `process_image()` — converts Surya's native output to the unified schema above
- Preserves `drop_repeated_text=True` flag (from Step 2.6)

**Class 3: `PaddleEngine(BaseOCREngine)`**
- Wraps PaddleOCR's `PaddleOCR()` API
- `load()` — instantiates `PaddleOCR(use_angle_cls=True, lang="ta", use_gpu=True)`
- `unload()` — calls `del self._ocr`, then `paddle.device.cuda.empty_cache()`
- `process_image()` — converts PaddleOCR's native list-of-tuples output `([[bbox], (text, conf)])` to the unified schema
- **Tamil language pack:** PaddleOCR uses `lang="ta"` for Tamil. We verify this works and falls back to `lang="latin"` if Tamil pack isn't available

**Factory function:**
```python
def create_engine(name: str, **kwargs) -> BaseOCREngine:
    if name == "surya":
        return SuryaEngine(**kwargs)
    elif name == "paddle":
        return PaddleEngine(**kwargs)
    else:
        raise ValueError(f"Unknown engine: {name}")
```

---

#### 2.0c — GPU Memory Manager

**Why this is needed:** Running back-to-back engine comparisons on a 6GB GPU without explicit VRAM management causes OOM crashes. We need a context manager that handles the load/unload cycle automatically.

#### [NEW] [pipeline/memory_manager.py](file:///d:/Evolve_Robot_Lab/Project/NLP%20Projects/OCR/multilingual_ocr_pipeline/pipeline/memory_manager.py)

```python
import gc, torch
from contextlib import contextmanager

class GPUMemoryManager:
    """
    Ensures only one OCR engine is resident in VRAM at a time.
    Usage:
        manager = GPUMemoryManager()
        with manager.engine_context(surya_engine):
            result = surya_engine.process_image(img)
        # surya_engine is now unloaded
        with manager.engine_context(paddle_engine):
            result = paddle_engine.process_image(img)
    """
    def __init__(self):
        self._active_engine = None

    @contextmanager
    def engine_context(self, engine: BaseOCREngine):
        if self._active_engine and self._active_engine is not engine:
            self._active_engine.unload()
            gc.collect()
            torch.cuda.empty_cache()
        engine.load()
        self._active_engine = engine
        try:
            yield engine
        finally:
            pass  # Keep loaded until next context call for single-engine runs

    def free_all(self):
        if self._active_engine:
            self._active_engine.unload()
        gc.collect()
        torch.cuda.empty_cache()
        self._active_engine = None
```

**Key behaviours:**
- In **normal single-engine mode** (`--engine surya`), the manager loads once and keeps the engine resident — zero overhead
- In **comparison mode** (`--compare-engines`), it automatically swaps engines between pages with full VRAM flush
- All existing `main.py` and `gradio_app.py` code continues to work unchanged — they just call `engine.process_image()` as before

---

#### 2.0d — Engine Comparison CLI & Report

#### [MODIFY] [main.py](file:///d:/Evolve_Robot_Lab/Project/NLP%20Projects/OCR/multilingual_ocr_pipeline/main.py)

Add a `--compare-engines` flag and `--engine [surya|paddle|best]` flag:

```
python main.py --input book.pdf --engine surya          # Normal run
python main.py --input book.pdf --engine paddle         # PaddleOCR run  
python main.py --input book.pdf --compare-engines       # A/B comparison
python main.py --input book.pdf --engine best           # Auto-select winner
```

**`--compare-engines` workflow per page:**
1. Load Surya → run `process_image(img)` → record output + time + confidence → unload
2. Load Paddle → run `process_image(img)` → record output + time + confidence → unload
3. If ground truth is available for this page: compute CER/WER for both engines
4. If no ground truth: use `page_confidence` as the proxy metric
5. Write per-page comparison to `outputs/engine_comparison_report.json`

**`--engine best` workflow:**
1. Reads `outputs/engine_comparison_report.json` from a previous comparison run
2. Selects the engine with lower median CER across all evaluated pages
3. Falls back to the losing engine only if the winner's `page_confidence < 0.70`

#### [NEW] [evaluation/engine_compare.py](file:///d:/Evolve_Robot_Lab/Project/NLP%20Projects/OCR/evaluation/engine_compare.py)

This script loads the `engine_comparison_report.json` and generates a human-readable summary:

```
=============================================================
ENGINE COMPARISON REPORT — Tamil OCR Benchmark
=============================================================
Documents evaluated : 7 stories, 66 pages
Ground truth pages  : 66

┌─────────────────────┬────────────┬────────────┬────────────┐
│ Metric              │ Surya OCR  │ PaddleOCR  │  Winner    │
├─────────────────────┼────────────┼────────────┼────────────┤
│ Mean CER            │   6.7%     │   TBD%     │   ???      │
│ Mean WER            │  18.3%     │   TBD%     │   ???      │
│ Speed (sec/page)    │   ~2.1s    │   TBD s    │   ???      │
│ Pages where engine  │            │            │            │
│   won individually  │   XX/66    │   YY/66    │            │
│ Blank output pages  │    0       │   TBD      │   ???      │
│ VRAM peak (GB)      │   ~2.8     │   ~1.7     │            │
└─────────────────────┴────────────┴────────────┴────────────┘

Document-level breakdown:
  ஜானாப்_பாட்டி   → Surya: 5.2% | Paddle: TBD% | Winner: ???
  புலிக்கலைஞன்    → Surya: 6.9% | Paddle: TBD% | Winner: ???
  ...

Recommendation: [engine_name] selected as production default.
Fallback engine: [engine_name] activated when confidence < 0.70.
=============================================================
```

**Failure-mode analysis columns added:**
- `hallucination_rate` — pages where OCR output is longer than 3× the expected page length
- `blank_rate` — pages returning fewer than 50 chars
- `tamil_char_ratio` — what fraction of output is valid Unicode Tamil codepoints

---

#### 2.0e — Gradio UI Integration for Engine Selection

#### [MODIFY] [gradio_app.py](file:///d:/Evolve_Robot_Lab/Project/NLP%20Projects/OCR/gradio_app.py)

Add an engine selector dropdown to the UI so users can interactively switch engines:

- **Dropdown:** `Engine: [Surya OCR ▼ | PaddleOCR | Auto (Best)]`
- **Auto mode** reads the winner from `engine_comparison_report.json` if it exists, otherwise defaults to Surya
- **Run report** shows which engine was used and the page-level confidence scores
- The `GPUMemoryManager` handles the VRAM swap transparently when the user switches engines between runs

---

### 2.1 — Orientation Detection & Auto-Rotation

**Problem:** The Taxi Driver PDF is 180° rotated — produces 42 chars of dots across 5 pages.

**Solution:** Use Surya's built-in orientation/rotation detection before OCR. Surya has `TextOrientationPredictor` that can detect if text is upside-down or sideways.

#### [MODIFY] [preprocessor.py](file:///d:/Evolve_Robot_Lab/Project/NLP%20Projects/OCR/multilingual_ocr_pipeline/pipeline/preprocessor.py)
- Add `detect_and_fix_orientation(image) -> Image` function
- Uses Surya's orientation predictor to detect 0°/90°/180°/270° rotation
- Auto-rotates the image before passing to OCR
- Shares the `FoundationPredictor` to avoid VRAM duplication

#### [MODIFY] [main.py](file:///d:/Evolve_Robot_Lab/Project/NLP%20Projects/OCR/multilingual_ocr_pipeline/main.py)
- Call `detect_and_fix_orientation()` between Step 1 (image extraction) and Step 2 (OCR)

#### [MODIFY] [gradio_app.py](file:///d:/Evolve_Robot_Lab/Project/NLP%20Projects/OCR/gradio_app.py)
- Same integration in the Gradio processing loop

---

### 2.2 — Dense Layout Fallback (Newspaper Fix)

**Problem:** காலைக்கதிர் newspaper classified as `Picture` → 0 chars extracted.

**Solution:** When layout returns only `Picture`/`Figure` blocks and zero `Text` blocks, treat the entire page as a `Text` fallback. The current `fallback` logic only triggers when *no blocks at all* are returned — we need to extend it.

#### [MODIFY] [layout_engine.py](file:///d:/Evolve_Robot_Lab/Project/NLP%20Projects/OCR/multilingual_ocr_pipeline/pipeline/layout_engine.py)
- In `analyze_page()`, after filtering usable blocks: if ALL blocks are `is_skip=True` (Picture/Figure), force a full-page `Text` fallback
- Log a warning: "All blocks classified as non-text — forcing full-page OCR fallback"

---

### 2.3 — ByT5 Post-OCR Correction (Core Phase 2 Feature)

**Problem:** Surya produces consistent character-level errors (~6.7% CER) — confusing visually similar Tamil characters like `ழ↔ள`, `ண↔ன`, dropping vowel marks (matras).

**Solution:** Fine-tune Google's `byt5-small` (300M params) on synthetic Tamil OCR noise, then integrate as a correction layer for low-confidence pages.

> [!IMPORTANT]
> This is the biggest item in Phase 2. It has 3 sub-steps:

#### 2.3a — Synthetic Noise Generator (RoundTrip OCR)

**Strategy:** Replace the heuristic text-based noise generator with a highly authentic "RoundTrip" pipeline. Instead of guessing how OCR fails, we will force the actual OCR engine to fail by feeding it deliberately degraded images of clean Tamil text.

#### [NEW] [pipeline/roundtrip_generator.py](file:///d:/Evolve_Robot_Lab/Project/NLP%20Projects/OCR_Phase_2/pipeline/roundtrip_generator.py)
This new module will contain the `RoundTripOCRGenerator` class.
*   **`render_text()`**: Uses `PIL.ImageDraw` to render a clean Tamil string onto a white background canvas using a standard Tamil TrueType font.
*   **`apply_degradation()`**: Uses `OpenCV` and `NumPy` to simulate bad scanned documents. It will randomly apply:
    *   **Morphological Erosion/Dilation**: Simulates ink bleed (letters merging) or faded print (letters breaking apart).
    *   **Gaussian & Motion Blur**: Simulates out-of-focus camera/scanner.
    *   **Salt & Pepper Noise**: Simulates dust, dirt, and paper grain.
    *   **Random Rotations**: Slight skewing (-2 to +2 degrees).
*   **`generate_pairs()`**: Takes the degraded image, passes it to the active OCR engine (Surya or Paddle), and pairs the resulting OCR text with the original clean text.

#### [MODIFY] [prepare_training_data.py](file:///d:/Evolve_Robot_Lab/Project/NLP%20Projects/OCR_Phase_2/prepare_training_data.py)
*   Add a new flag `--roundtrip` to the CLI.
*   Update the orchestrator to instantiate `RoundTripOCRGenerator` and `GPUMemoryManager`.
*   Pass the clean sentences from `data/corpus/combined.txt` through the RoundTrip generator to produce the noisy text, saving pairs to `data/synthetic_noise/train.jsonl`.

#### 2.3b — ByT5 Fine-Tuning (Kaggle notebook)

#### [NEW] [notebooks/byt5_finetuning.ipynb](file:///d:/Evolve_Robot_Lab/Project/NLP%20Projects/OCR/notebooks/byt5_finetuning.ipynb)
- Load `google/byt5-small` from HuggingFace
- Fine-tune on the synthetic noise pairs + real OCR pairs from ground truth
- Train on Kaggle T4 GPU (16GB VRAM — fits `byt5-small` comfortably)
- Save fine-tuned model to `models/byt5_finetuned/`
- Evaluate on held-out test set

#### 2.3c — LM Corrector Integration

#### [NEW] [pipeline/lm_corrector.py](file:///d:/Evolve_Robot_Lab/Project/NLP%20Projects/OCR/multilingual_ocr_pipeline/pipeline/lm_corrector.py)
- `LMCorrector` class that loads the fine-tuned ByT5 model
- `correct_text(text, confidence) -> str` — only runs when page confidence < `QUALITY_CONF_THRESHOLD` (0.80)
- Operates line-by-line to stay within ByT5's context window
- Falls back gracefully if model isn't available (just returns uncorrected text)

#### [MODIFY] [main.py](file:///d:/Evolve_Robot_Lab/Project/NLP%20Projects/OCR/multilingual_ocr_pipeline/main.py)
- Insert LM correction step between layout reconstruction and text cleaning
- Only trigger when `confidence < QUALITY_CONF_THRESHOLD`
- Log: "LM correction applied: X chars changed"

---

### 2.4 — Expand Ground Truth & Re-Evaluate

**Problem:** We only have ground truth for 7 stories (~66 pages). Need more diverse coverage.

#### [MODIFY] [evaluation/ground_truth/](file:///d:/Evolve_Robot_Lab/Project/NLP%20Projects/OCR/evaluation/ground_truth/)
- Add ground truth for 3-5 Test_data PDFs (the ones with clean single-column scans)
- Target: 50+ total pages of ground truth
- Use the OCR output from verify PDFs + manual spot-checks

#### [MODIFY] [evaluation/evaluate.py](file:///d:/Evolve_Robot_Lab/Project/NLP%20Projects/OCR/evaluation/evaluate.py)
- Add "before correction" vs "after correction" comparison mode
- Output a summary table showing CER improvement per story

---

### 2.5 — Gradio UI Overhaul

Based on the requirements in [ui_updates.txt](file:///d:/Evolve_Robot_Lab/Project/NLP%20Projects/OCR/Doc/ui_updates.txt):

#### [MODIFY] [gradio_app.py](file:///d:/Evolve_Robot_Lab/Project/NLP%20Projects/OCR/gradio_app.py)

1. **Real-time streaming status** — The current spinner implementation works but needs clearer backend stage messaging (already partially done with `OCR… → Layout filter… → Cleaning…`)

2. **Post-processing report with CER/WER** — Already implemented! The accordion reference text section and `_format_report_md()` are live. Verify they work end-to-end.

3. **Remove avg confidence from report** — Already addressed (report uses CER/WER, not raw confidence)

4. **Premium UI redesign** — Research modern OCR tool interfaces, then redesign with:
   - Dark mode / glassmorphism theme
   - Side-by-side image ↔ text view
   - Per-page navigation (click to jump to a page's output)
   - Progress bar instead of spinner
   - Responsive layout

> [!WARNING]
> The UI redesign is the most subjective item. I recommend we handle it as the final step after all backend improvements are validated. I'll research modern OCR tool UIs before designing.

---

### 2.6 — `drop_repeated_text` Flag

**Problem:** Surya sometimes repeats the same line multiple times in dense text regions.

**Solution:** Surya's `RecognitionPredictor.__call__` already has a `drop_repeated_text: bool = False` parameter. We just need to enable it.

#### [MODIFY] [pipeline/ocr_engine.py](file:///d:/Evolve_Robot_Lab/Project/NLP%20Projects/OCR/multilingual_ocr_pipeline/pipeline/ocr_engine.py)
- Pass `drop_repeated_text=True` in `process_image()` and `process_batch()`

---

## Execution Order

| Order | Step | Effort | Risk |
|-------|------|--------|------|
| 1 | 2.6 — Enable `drop_repeated_text` | 5 min | None |
| 2 | 2.2 — Dense layout fallback | 30 min | Low |
| 3 | 2.1 — Orientation detection | 1 hour | Medium (verify Surya API) |
| 4 | **2.0a** — PaddleOCR install & isolation | 30 min | Medium (CUDA conflict risk) |
| 5 | **2.0b** — Abstract engine factory refactor | 2 hours | Medium (touches ocr_engine.py) |
| 6 | **2.0c** — GPU memory manager | 1 hour | Low |
| 7 | **2.0d** — Comparison CLI + report | 2 hours | Low |
| 8 | **2.0e** — Run comparison on all 7 ground truth stories | 1 hour | Low |
| 9 | 2.3a — Synthetic noise generator | 2–3 hours | Low |
| 10 | 2.3b — ByT5 fine-tuning notebook | 3–4 hours | Medium (Kaggle session limits) |
| 11 | 2.3c — LM corrector integration | 1–2 hours | Low |
| 12 | 2.4 — Expand ground truth + evaluate | 2 hours | Low |
| 13 | 2.5 — Gradio UI overhaul | 3–4 hours | Low (cosmetic) |

**Total estimated effort: ~21 hours across multiple sessions**

---

## Decisions Made

| # | Question | Decision |
|---|----------|----------|
| Q1 | PaddleOCR — defer or include? | ✅ **Include in Phase 2** as a comparison engine (Step 2.0) |
| Q2 | Celery + Redis queue | ⬜ **Defer to Phase 3** — sequential pipeline handles 50–100 pages fine |

## Open Questions

## Decisions Made

| # | Question | Decision |
|---|----------|----------|
| Q1 | PaddleOCR — defer or include? | ✅ **Include in Phase 2** as a comparison engine (Step 2.0) |
| Q2 | Celery + Redis queue | ⬜ **Defer to Phase 3** — sequential pipeline handles 50–100 pages fine |
| Q3 | Tamil corpus source | ✅ **All sources** — Wikipedia + Project Madurai + Tamil news articles |
| Q4 | PaddleOCR Tamil lang pack | ✅ **Acknowledged** — will validate during Step 2.0a, fallback to detection-only if needed |
| Q5 | Phase 2 Gradio UI | ✅ **New UI** in `OCR_Phase_2/` — Phase 1 UI remains untouched |
| Q6 | Directory structure | ✅ **Separate directory** `OCR_Phase_2/` — Phase 1 code is stable baseline |

---

## Verification Plan

### Automated Tests
- Run `evaluate.py` on all 7 stories BEFORE and AFTER ByT5 correction
- CER target: < 5% on clean scans with correction
- Verify Taxi Driver PDF produces readable English output after orientation fix
- Verify காலைக்கதிர் newspaper produces text (not blank) after dense layout fallback

### Manual Verification
- Visual inspection of 5 corrected pages in Gradio UI
- Compare "raw OCR" vs "LM corrected" side-by-side for hallucination check
- Validate the new UI design is responsive and user-friendly

## Success Criteria

| Metric | Target |
|--------|--------|
| Engine comparison report | Generated for all 7 ground truth stories |
| Engine winner identified | Clear recommendation with CER data |
| Fallback routing working | Low-confidence pages auto-route to backup engine |
| CER (best engine, no LM) | Baseline measurement documented |
| CER (best engine, with LM) | < 5% |
| CER (degraded scans, with LM) | < 10% |
| Rotated scan detection | 100% for 90°/180°/270° |
| Dense page fallback | No blank outputs |
| UI engine selector | Surya / PaddleOCR / Auto dropdown working |
