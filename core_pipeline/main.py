"""
Multilingual OCR Pipeline — Phase 2 Entry Point
=================================================
Orchestrates: PDF → Images → Orientation Fix → OCR (engine-selectable)
            → Layout Analysis → LM Correction → Text Cleaning → Output

New in Phase 2:
  - --engine [surya|paddle|best]    Select OCR engine
  - --compare-engines               Run both engines and generate A/B report
  - Automatic orientation detection  (rotated/upside-down scan fix)
  - Dense layout fallback            (newspaper pages classified as Picture)
  - ByT5 LM correction              (low-confidence pages only)
"""
import sys
import json
import time
import os
from pathlib import Path

# ── Surya batch size guard (prevents GPU deadlock on RTX 4050 6GB) ────────────
# Default is 256 for CUDA; that OOMs / hangs on pages with 100+ lines.
os.environ.setdefault("RECOGNITION_BATCH_SIZE", "32")
os.environ.setdefault("DETECTOR_BATCH_SIZE", "4")

# Force UTF-8 encoding for Windows terminal output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


from pipeline.pdf_converter import pdf_to_images
from pipeline.ocr_engine import create_engine, SuryaEngine, BaseOCREngine
from pipeline.layout_engine import LayoutEngine
from pipeline.memory_manager import GPUMemoryManager
from pipeline.postprocessor import Cleaner
from pipeline.text_cleaner import clean_text as deep_clean
from pipeline.preprocessor import preprocess, preprocessed_to_pil
from pipeline.lm_corrector import LMCorrector
from pipeline.text_formatter import TextFormatter
from config import (
    OUTPUTS_DIR, DATA_DIR, QUALITY_CONF_THRESHOLD,
    DEFAULT_ENGINE, DEFAULT_LANGS, ENGINE_COMPARISON_REPORT,
    FALLBACK_CONF_THRESHOLD, PREPROCESS_ENABLED,
    LM_CORRECTION_ENABLED
)


def process_page(
    page_num: int,
    total_pages: int,
    raw_img,
    engine: BaseOCREngine,
    layout_engine: LayoutEngine,
    pdf_name: str,
    orientation_predictor=None,
    lm_corrector: LMCorrector = None,
) -> dict:
    """Process a single page through OCR + layout + cleaning. Returns page result dict."""
    # Preprocessing (including Orientation Fix)
    t_pre = time.perf_counter()
    if PREPROCESS_ENABLED:
        processed_arr = preprocess(raw_img, orientation_predictor=orientation_predictor)
        pre_img = preprocessed_to_pil(processed_arr)
    else:
        pre_img = raw_img
    pre_time = time.perf_counter() - t_pre
    
    # OCR
    t0 = time.perf_counter()
    raw_result = engine.process_image(pre_img)
    ocr_time = time.perf_counter() - t0

    page_text = raw_result.get("text", "")
    confidence = raw_result.get("page_confidence", 0.0)
    lines = raw_result.get("lines", [])
    print(f"     Pre: {pre_time:.1f}s | OCR: {len(lines)} lines, conf={confidence:.2f}, time={ocr_time:.1f}s")

    # Layout analysis
    layout = layout_engine.analyze_page(raw_img, page_num=page_num)
    if layout.fallback:
        print(f"     [Layout] No blocks detected — full-page mode")
    else:
        labels = [b.label for b in layout.blocks]
        print(f"     [Layout] {len(layout.blocks)} block(s): {labels}")

    # Reconstruct text with layout filtering (drops headers/footers)
    ocr_lines = raw_result.get("lines", [])

    # Surgical ByT5 correction (only low-confidence lines)
    lm_corrected = False
    if LM_CORRECTION_ENABLED and lm_corrector:
        t_lm = time.perf_counter()
        ocr_lines = lm_corrector.surgical_correct_lines(ocr_lines, threshold=QUALITY_CONF_THRESHOLD)
        lm_corrected = any(line.get("is_corrected", False) for line in ocr_lines)
        # Note: we update the lines in place before reconstruction
    
    page_text = LayoutEngine.reconstruct_page_text(ocr_lines, layout)

    # Text cleaning (Cleaner.clean_text internally calls deep_clean + whitespace normalization)
    final_text = Cleaner.clean_text(page_text)

    return {
        "page": page_num,
        "engine": engine.name,
        "confidence": round(confidence, 4),
        "ocr_time_s": round(ocr_time, 2),
        "num_lines": len(lines),
        "lm_corrected": lm_corrected,
        "fallback_layout": layout.fallback,
        "blocks": [
            {"label": b.label, "position": b.position,
             "bbox": [round(x) for x in b.bbox],
             "confidence": round(b.confidence, 4)}
            for b in layout.blocks
        ],
        "text": final_text,
    }


def process_pdf(
    pdf_path: Path,
    engine: BaseOCREngine,
    layout_engine: LayoutEngine,
    memory_manager: GPUMemoryManager,
    orientation_predictor=None,
    lm_corrector: LMCorrector = None,
):
    """Run the full OCR pipeline on a single PDF."""
    print(f"\n[{pdf_path.name}] Starting Phase 2 pipeline (engine: {engine.name})...")

    pages = pdf_to_images(pdf_path)
    print(f"[{pdf_path.name}]   → Extracted {len(pages)} pages")

    folder_name = pdf_path.stem.strip().replace(":", "_").replace("/", "_")
    story_output_dir = OUTPUTS_DIR / folder_name
    story_output_dir.mkdir(parents=True, exist_ok=True)

    doc_results = []
    full_text_parts = []

    with memory_manager.engine_context(engine):
        for page_num, raw_img in pages:
            # Save raw image for debugging
            debug_dir = DATA_DIR / "page_images" / pdf_path.stem
            debug_dir.mkdir(parents=True, exist_ok=True)
            raw_img.save(str(debug_dir / f"page_{page_num:03d}_raw.png"))

            result = process_page(
                page_num, len(pages), raw_img,
                engine, layout_engine, pdf_path.name,
                orientation_predictor=orientation_predictor,
                lm_corrector=lm_corrector,
            )
            doc_results.append(result)
            full_text_parts.append(result["text"])

    # ── Save raw per-page JSON ────────────────────────────────────────────────
    out_json = story_output_dir / "document_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(doc_results, f, ensure_ascii=False, indent=2)

    # Professional text formatting: strip page numbers, running headers,
    # merge fragmented lines, and join pages cleanly
    formatted_text = TextFormatter.format_document(full_text_parts, pdf_name=pdf_path.name)

    out_txt = story_output_dir / f"{pdf_path.stem}_full.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(formatted_text)

    print(f"\n[{pdf_path.name}] Pipeline complete! Saved to: {story_output_dir}")
    return doc_results


def compare_engines(
    pdf_path: Path,
    surya_engine: BaseOCREngine,
    paddle_engine: BaseOCREngine,
    layout_engine: LayoutEngine,
    memory_manager: GPUMemoryManager,
):
    """Run both engines on a PDF and generate comparison data."""
    print(f"\n{'='*60}")
    print(f"ENGINE COMPARISON: {pdf_path.name}")
    print(f"{'='*60}")

    pages = pdf_to_images(pdf_path)
    print(f"  Extracted {len(pages)} pages")

    comparison_pages = []

    for page_num, raw_img in pages:
        print(f"\n  --- Page {page_num}/{len(pages)} ---")

        # Run Surya
        with memory_manager.engine_context(surya_engine):
            surya_result = process_page(
                page_num, len(pages), raw_img,
                surya_engine, layout_engine, pdf_path.name,
            )

        # Run PaddleOCR
        with memory_manager.engine_context(paddle_engine):
            paddle_result = process_page(
                page_num, len(pages), raw_img,
                paddle_engine, layout_engine, pdf_path.name,
            )

        comparison_pages.append({
            "page": page_num,
            "surya": {
                "confidence": surya_result["confidence"],
                "num_lines": surya_result["num_lines"],
                "ocr_time_s": surya_result["ocr_time_s"],
                "text": surya_result["text"],
                "text_length": len(surya_result["text"]),
            },
            "paddle": {
                "confidence": paddle_result["confidence"],
                "num_lines": paddle_result["num_lines"],
                "ocr_time_s": paddle_result["ocr_time_s"],
                "text": paddle_result["text"],
                "text_length": len(paddle_result["text"]),
            },
        })

    return {
        "document": pdf_path.name,
        "total_pages": len(pages),
        "pages": comparison_pages,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2 OCR Pipeline")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input PDF or directory of PDFs")
    parser.add_argument("--engine", "-e", type=str, default=DEFAULT_ENGINE,
                        choices=["surya", "paddle", "best"],
                        help="OCR engine to use (default: surya)")
    parser.add_argument("--compare-engines", action="store_true",
                        help="Run both Surya and PaddleOCR, generate comparison report")
    args = parser.parse_args()

    input_path = Path(args.input)
    memory_manager = GPUMemoryManager()

    # Initialize predictors
    print("Loading layout engines...")
    from surya.foundation import FoundationPredictor
    
    fp = FoundationPredictor(device="cuda")
    layout_engine = LayoutEngine(foundation_predictor=fp)
    
    orientation_predictor = None # Disabled temporarily due to Surya API changes
    
    lm_corrector = None
    if LM_CORRECTION_ENABLED:
        print("Loading LM corrector (ByT5) on GPU...")
        lm_corrector = LMCorrector(device="cuda")

    if args.compare_engines:
        # ── Comparison mode ──
        surya_engine = create_engine("surya", foundation_predictor=fp, langs=DEFAULT_LANGS)
        paddle_engine = create_engine("paddle")

        pdfs = _collect_pdfs(input_path)
        all_comparisons = []

        for pdf in pdfs:
            result = compare_engines(
                pdf, surya_engine, paddle_engine,
                layout_engine, memory_manager,
            )
            all_comparisons.append(result)

        # Save comparison report
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(ENGINE_COMPARISON_REPORT, "w", encoding="utf-8") as f:
            json.dump(all_comparisons, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Comparison report saved: {ENGINE_COMPARISON_REPORT}")
        print(f"   Run `python evaluation/engine_compare.py` to see results.")

    else:
        # ── Single engine mode ──
        if args.engine == "best":
            engine_name = _load_best_engine_name()
        else:
            engine_name = args.engine

        if engine_name == "surya":
            engine = create_engine("surya", foundation_predictor=fp, langs=DEFAULT_LANGS)
        else:
            engine = create_engine(engine_name)

        pdfs = _collect_pdfs(input_path)
        for pdf in pdfs:
            process_pdf(
                pdf, engine, layout_engine, memory_manager,
                orientation_predictor=orientation_predictor,
                lm_corrector=lm_corrector,
            )

    memory_manager.free_all()


def _collect_pdfs(input_path: Path) -> list:
    """Collect PDF files from a path (single file or directory)."""
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]
    elif input_path.is_dir():
        pdfs = sorted([p for p in input_path.glob("*.pdf")
                       if "(verify)" not in p.name])
        print(f"Found {len(pdfs)} source PDFs to process.")
        return pdfs
    else:
        print("Invalid input. Must be a .pdf file or a directory.")
        return []


def _load_best_engine_name() -> str:
    """Load the best engine from a previous comparison report."""
    if ENGINE_COMPARISON_REPORT.exists():
        try:
            with open(ENGINE_COMPARISON_REPORT, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Simple heuristic: count page-level confidence wins
            surya_wins = 0
            paddle_wins = 0
            for doc in data:
                for page in doc.get("pages", []):
                    s_conf = page.get("surya", {}).get("confidence", 0)
                    p_conf = page.get("paddle", {}).get("confidence", 0)
                    if s_conf >= p_conf:
                        surya_wins += 1
                    else:
                        paddle_wins += 1
            winner = "surya" if surya_wins >= paddle_wins else "paddle"
            print(f"[Auto] Best engine: {winner} ({surya_wins} vs {paddle_wins} page wins)")
            return winner
        except Exception as e:
            print(f"[Auto] Could not read comparison report: {e}. Defaulting to surya.")
    else:
        print("[Auto] No comparison report found. Run --compare-engines first. Defaulting to surya.")
    return "surya"


if __name__ == "__main__":
    main()
