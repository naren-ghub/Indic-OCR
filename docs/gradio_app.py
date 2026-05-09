"""
Indic OCR — Multilingual Document Digitization
Supports: PDF (up to 100 pages), Image files (.png, .jpg, .tiff, .bmp), DOCX files.
Powered by Surya OCR with layout-aware text reconstruction.
"""
import re
import unicodedata
import sys
import io
import os
import tempfile
from pathlib import Path

import gradio as gr
from PIL import Image

# ── Project root on sys.path so pipeline imports work ──────────────────────
ROOT = Path(__file__).parent
# Phase 2 pipeline: check both local layout (ROOT.parent) and bundled layout (ROOT subdirectory)
# This allows the same code to run locally AND inside the Modal container.
_phase2_local  = ROOT.parent / "OCR_Phase_2"   # local: NLP Projects/OCR_Phase_2
_phase2_bundle = ROOT / "OCR_Phase_2"           # container: /root/deploy/OCR_Phase_2
PHASE2_ROOT = _phase2_bundle if _phase2_bundle.exists() else _phase2_local
sys.path.insert(0, str(PHASE2_ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Optional imports (metrics) ──────────────────────────────────────────────
try:
    import jiwer
except Exception:
    jiwer = None

# ── Lazy imports for optional formats ──────────────────────────────────────
def _import_fitz():
    import fitz
    return fitz

def _import_docx():
    import docx
    return docx

# ── Pipeline imports ───────────────────────────────────────────────────────
from pipeline.ocr_engine import SuryaEngine, create_engine
from pipeline.layout_engine import LayoutEngine
from pipeline.text_cleaner import clean_text as _deep_clean

MAX_PAGES = 100
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

# ── Short-line reflow ─────────────────────────────────────────────────────
# When the layout engine detects narrow text columns or fragmented bounding
# boxes, it may output very short lines that are really continuation fragments
# of the surrounding paragraph (e.g. a single word on its own line).
# This function joins such orphan lines back to the previous paragraph.
_REFLOW_THRESHOLD = 18  # lines shorter than this (in chars) get merged

def _reflow_text(text: str) -> str:
    """Merge short orphan lines into the previous paragraph."""
    lines = text.splitlines()
    merged: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Blank line — preserve as paragraph separator
            merged.append("")
        elif merged and stripped and len(stripped) < _REFLOW_THRESHOLD and not merged[-1] == "":
            # Short non-empty line following non-empty content → append to previous
            merged[-1] = merged[-1] + " " + stripped
        else:
            merged.append(stripped)
    # Collapse consecutive blank lines into one
    result: list[str] = []
    prev_blank = False
    for line in merged:
        if line == "":
            if not prev_blank:
                result.append("")
            prev_blank = True
        else:
            result.append(line)
            prev_blank = False
    return "\n".join(result).strip()

# ── Supported languages ───────────────────────────────────────────────────
LANGUAGE_OPTIONS = {
    "Tamil (தமிழ்)": "ta",
    "Hindi (हिन्दी)": "hi",
    "Telugu (తెలుగు)": "te",
    "Bengali (বাংলা)": "bn",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Malayalam (മലയാളം)": "ml",
    "Gujarati (ગુજરાતી)": "gu",
    "Marathi (मराठी)": "mr",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Odia (ଓଡ଼ିଆ)": "or",
    "English": "en",
}

# ── Load models once at startup ────────────────────────────────────────────
import torch as _torch
_DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
print(f"Loading Surya OCR models on {_DEVICE}… (first run takes ~10s)")
from surya.foundation import FoundationPredictor
fp = FoundationPredictor(device=_DEVICE)
# Engine will be re-created per request with the selected language
layout_engine = LayoutEngine(foundation_predictor=fp)
print("Models ready ✓")

# Logo path
LOGO_PATH = str(ROOT / "indian_ocr_logo.png")


# ── File → List[PIL.Image] converters ─────────────────────────────────────

def load_pdf(path: str) -> list[Image.Image]:
    fitz = _import_fitz()
    doc = fitz.open(path)
    total = len(doc)
    if total > MAX_PAGES:
        raise ValueError(
            f"PDF has {total} pages. Maximum allowed is {MAX_PAGES}. "
            f"Please upload a shorter document."
        )
    images = []
    for page_num in range(total):
        page = doc[page_num]
        embedded = page.get_images(full=True)
        text_len = len(page.get_text("text").strip())
        if len(embedded) == 1 and text_len == 0:
            # Pure scan → extract image directly (best quality)
            xref = embedded[0][0]
            base = doc.extract_image(xref)
            img = Image.open(io.BytesIO(base["image"])).convert("RGB")
        else:
            # Text-based or mixed → render at 300 DPI
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append((page_num + 1, img))
    doc.close()
    return images


def load_image_file(path: str) -> list[tuple[int, Image.Image]]:
    img = Image.open(path).convert("RGB")
    return [(1, img)]


def load_docx(path: str) -> list[tuple[int, Image.Image]]:
    """
    DOCX: extract embedded images in order and treat each as a page.
    """
    docx = _import_docx()
    doc = docx.Document(path)

    images = []
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            img_bytes = rel.target_part.blob
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append(img)
            except Exception:
                pass

    if not images:
        raise ValueError(
            "This DOCX contains no embedded images. "
            "For text-only DOCX files, please copy the text directly — "
            "OCR is only needed for scanned/image content."
        )

    if len(images) > MAX_PAGES:
        raise ValueError(
            f"DOCX contains {len(images)} images. Maximum allowed is {MAX_PAGES}."
        )

    return [(i + 1, img) for i, img in enumerate(images)]


# ── Metrics helpers ────────────────────────────────────────────────────────

def _normalize_for_metrics(text: str) -> str:
    """
    Strip structural OCR output artifacts before CER/WER comparison so that
    layout formatting does not inflate error rates against clean ground-truth text.
    Removes: OCR file header lines, ===separator lines, PAGE N headers,
    standalone page numbers, ### markdown headers, and single-char fragments.
    Also applies Unicode NFC normalization for consistent Indic script comparison.
    """
    if not text:
        return ""
    # Strip BOM if present
    text = text.lstrip('\ufeff')
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        s = line.strip()
        # Skip OCR output file header lines (e.g. 'Indic OCR Output — ...')
        if re.match(r'^Indic OCR Output', s): continue
        if re.match(r'^Languages:', s): continue
        if re.match(r'^Pages processed:', s): continue
        if re.match(r'^Tamil OCR Output', s): continue
        # Skip === separator lines
        if re.match(r'^=+$', s): continue
        # Skip PAGE N header lines
        if re.match(r'^PAGE\s+\d+', s): continue
        # Skip standalone page numbers
        if re.match(r'^\d{1,4}$', s): continue
        # Skip markdown headers that crept in
        if re.match(r'^#+\s', s): continue
        # Skip single/double-char layout fragment lines
        if len(s) <= 2 and s: continue
        cleaned.append(s)
    joined = ' '.join(' '.join(cleaned).split())
    # NFC normalize for consistent Indic script comparison
    return unicodedata.normalize('NFC', joined)


def _maybe_load_reference_text(
    doc_path: str,
    doc_ext: str,
    reference_text: str | None,
    reference_file_path: str | None,
) -> tuple[str | None, str]:
    """
    Returns: (reference_text_or_none, reference_source_label)
    """
    if reference_file_path:
        try:
            with open(reference_file_path, "r", encoding="utf-8") as f:
                txt = f.read()
            txt = txt.strip()
            if txt:
                return txt, "user-provided reference file"
        except Exception:
            pass

    if reference_text and reference_text.strip():
        return reference_text.strip(), "user-provided reference text"

    if doc_ext == ".pdf":
        # Best-effort: use embedded selectable text as reference, if present.
        try:
            fitz = _import_fitz()
            pdf = fitz.open(doc_path)
            parts: list[str] = []
            for page in pdf:
                parts.append(page.get_text("text", sort=True) or "")
            pdf.close()
            embedded = "\n".join(parts).strip()
            if embedded:
                return embedded, "embedded PDF text (best-effort)"
        except Exception:
            pass

    return None, "no reference available"


def _format_report_md(
    filename: str,
    total_pages: int,
    output_filename: str,
    cer: float | None,
    wer: float | None,
    reference_source: str,
    selected_langs: str,
    metrics_note: str | None = None,
) -> str:
    cer_str = f"{cer:.2%}" if cer is not None else "N/A"
    wer_str = f"{wer:.2%}" if wer is not None else "N/A"

    extra = f"\n\n{metrics_note}" if metrics_note else ""
    return (
        "### 📊 Run Report\n\n"
        f"| Property | Value |\n"
        f"|---|---|\n"
        f"| **File** | `{filename}` |\n"
        f"| **Pages processed** | {total_pages} |\n"
        f"| **Languages** | {selected_langs} |\n"
        f"| **Output** | `{output_filename}` |\n"
        f"| **Reference** | {reference_source} |\n"
        f"| **CER** | {cer_str} |\n"
        f"| **WER** | {wer_str} |\n"
        "\n"
        "> **CER** measures character-level accuracy. **WER** measures word-level accuracy. Lower is better.\n"
        f"{extra}"
    )


# ── Core processing function ───────────────────────────────────────────────

def run_ocr(file_obj, selected_languages, reference_text: str | None, reference_file_path: str | None):
    """
    Main Gradio callback.
    """
    if file_obj is None:
        yield "⚠️ Upload a file to begin.", "", None, ""
        return

    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
    if not path:
        yield "❌ Could not read uploaded file path.", "", None, ""
        return

    ext = Path(path).suffix.lower()

    # ── Resolve selected languages ──────────────────────────────────────
    if not selected_languages:
        selected_languages = ["Tamil (தமிழ்)"]
    lang_codes = [LANGUAGE_OPTIONS.get(l, "ta") for l in selected_languages]
    # Always include English for mixed-script documents
    if "en" not in lang_codes:
        lang_codes.append("en")
    lang_label = ", ".join(selected_languages)

    # Create OCR engine with selected languages
    engine = create_engine("surya", langs=lang_codes, device="cuda", foundation_predictor=fp)

    # ── Load pages ──────────────────────────────────────────────────────
    try:
        if ext == ".pdf":
            pages = load_pdf(path)
        elif ext in SUPPORTED_IMAGE_EXTS:
            pages = load_image_file(path)
        elif ext in {".docx", ".doc"}:
            pages = load_docx(path)
        else:
            yield (
                f"❌ Unsupported file type: '{ext}'. "
                "Supported: PDF, PNG, JPG, TIFF, BMP, WEBP, DOCX"
            ), "", None, ""
            return
    except ValueError as e:
        yield f"❌ {e}", "", None, ""
        return
    except Exception as e:
        yield f"❌ Error loading file: {e}", "", None, ""
        return

    total_pages = len(pages)
    log_lines = [f"📄 Loaded {total_pages} page(s) from '{Path(path).name}'"]
    log_lines.append(f"🌐 Languages: {lang_label}")
    log_lines.append("")
    report_md = ""
    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    spin_i = 0
    status = f"{spinner[spin_i]} Ready…"
    yield status, "\n".join(log_lines), None, report_md

    ref_text, ref_source = _maybe_load_reference_text(
        doc_path=path,
        doc_ext=ext,
        reference_text=reference_text,
        reference_file_path=reference_file_path,
    )

    all_text_parts = []
    pred_pages_for_metrics: list[str] = []

    # ── OCR and Layout processing ───────────────────────────────────────────
    for page_num, img in pages:
        spin_i = (spin_i + 1) % len(spinner)
        status = f"{spinner[spin_i]} Processing page {page_num}/{total_pages}…"
        log_lines.append(f"🔍 Page {page_num}/{total_pages}: OCR…")
        yield status, "\n".join(log_lines), None, report_md
        try:
            result = engine.process_image(img)
            ocr_lines = result.get("lines", [])
            lines_count = len(ocr_lines)

            spin_i = (spin_i + 1) % len(spinner)
            status = f"{spinner[spin_i]} Layout filtering…"
            log_lines.append(f"   🧩 Layout filter…")
            yield status, "\n".join(log_lines), None, report_md

            layout = layout_engine.analyze_page(img, page_num=page_num)
            text = LayoutEngine.reconstruct_page_text(ocr_lines, layout)

            spin_i = (spin_i + 1) % len(spinner)
            status = f"{spinner[spin_i]} Cleaning…"
            yield status, "\n".join(log_lines), None, report_md

            text = _deep_clean(text)
            text = _reflow_text(text)  # merge orphan short lines into paragraphs

            all_text_parts.append(
                f"{'='*60}\n"
                f"PAGE {page_num}\n"
                f"{'='*60}\n"
                f"{text}\n"
            )
            pred_pages_for_metrics.append(text)
            log_lines.append(
                f"   ✅ {lines_count} lines extracted"
            )
            spin_i = (spin_i + 1) % len(spinner)
            status = f"{spinner[spin_i]} Page {page_num} done"
            yield status, "\n".join(log_lines), None, report_md
        except Exception as e:
            log_lines.append(f"   ❌ Page {page_num} failed: {e}")
            all_text_parts.append(
                f"{'='*60}\nPAGE {page_num}\n{'='*60}\n[OCR ERROR: {e}]\n"
            )
            pred_pages_for_metrics.append("")
            spin_i = (spin_i + 1) % len(spinner)
            status = f"{spinner[spin_i]} Error on page {page_num}"
            yield status, "\n".join(log_lines), None, report_md

    # ── Write output .txt ───────────────────────────────────────────────
    output_filename = Path(path).stem + "_ocr_output.txt"
    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Indic OCR Output — {Path(path).name}\n")
        f.write(f"Languages: {lang_label}\n")
        f.write(f"Pages processed: {total_pages}\n")
        f.write("=" * 60 + "\n\n")
        f.write("\n".join(all_text_parts))

    # ── Compute CER/WER (best-effort) ───────────────────────────────────
    cer = None
    wer = None
    metrics_note = None

    pred_full = _normalize_for_metrics("\n".join(pred_pages_for_metrics))
    ref_full = _normalize_for_metrics(ref_text) if ref_text else ""

    if ref_full and jiwer is None:
        metrics_note = (
            "> Note: `jiwer` is not available in this environment, so CER/WER could not be computed.\n"
            "> Install it with `pip install jiwer`."
        )
    elif ref_full:
        try:
            cer = float(jiwer.cer(ref_full, pred_full))
            wer = float(jiwer.wer(ref_full, pred_full))
        except Exception as e:
            metrics_note = f"> Note: CER/WER computation failed: `{e}`"
    else:
        metrics_note = (
            "> Note: CER/WER require a reference text. Provide one (paste or upload a `.txt`),\n"
            "> or use a PDF that contains selectable text."
        )

    report_md = _format_report_md(
        filename=Path(path).name,
        total_pages=total_pages,
        output_filename=output_filename,
        cer=cer,
        wer=wer,
        reference_source=ref_source,
        selected_langs=lang_label,
        metrics_note=metrics_note,
    )

    log_lines.append("")
    log_lines.append(f"✅ Done! {total_pages} page(s) processed.")
    log_lines.append("📥 Download is ready.")

    status = "✅ Completed"
    yield status, "\n".join(log_lines), output_path, report_md


# ── Gradio UI ──────────────────────────────────────────────────────────────

# Demo file paths
DEMO_PDF_PATH = str(ROOT.parent / "OCR" / "OCR_dataset" / "எலி.pdf")
if not Path(DEMO_PDF_PATH).exists():
    DEMO_PDF_PATH = str(ROOT / "OCR_dataset" / "எலி.pdf")
DEMO_GT_PATH  = str(ROOT / "OCR_dataset" / "எலி_ground_truth.txt")


CUSTOM_CSS = """
/* ── Global ─────────────────────────────────────────────────────── */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    background: #0d1117 !important;
}
body { background: #0d1117 !important; }

/* ── Header ─────────────────────────────────────────────────────── */
#header-row {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0f1923 100%);
    border-radius: 16px;
    padding: 28px 36px !important;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
#header-row * { color: #e8e8e8 !important; }
#header-row h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #FF9933, #f0f0f0, #138808);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 !important;
    line-height: 1.3 !important;
}
#header-row p { font-size: 1rem !important; color: #8b949e !important; margin-top: 6px !important; }

/* ── Panel cards ─────────────────────────────────────────────────── */
.panel-card {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 14px;
    padding: 20px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
}
.panel-card h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #c9d1d9 !important;
    margin-bottom: 12px !important;
}

/* ── Scrollable log box ─────────────────────────────────────────── */
#log-box textarea {
    max-height: 340px !important;
    overflow-y: auto !important;
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace !important;
    font-size: 0.85rem !important;
    line-height: 1.55 !important;
    background: #0d1117 !important;
    color: #c9d1d9 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    border: 1px solid #21262d !important;
}

/* ── Run button ─────────────────────────────────────────────────── */
#run-btn {
    background: linear-gradient(135deg, #FF9933, #d97706) !important;
    color: #0d1117 !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(255,153,51,0.3) !important;
    transition: all 0.2s !important;
}
#run-btn:hover { box-shadow: 0 6px 20px rgba(255,153,51,0.45) !important; }

/* ── Download button ─────────────────────────────────────────────── */
#dl-btn {
    background: linear-gradient(135deg, #138808, #0a6b06) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
}

/* ── Demo + Instructions section ─────────────────────────────────── */
#demo-section {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 24px 28px !important;
    margin-top: 16px;
}
#demo-section h3 { color: #c9d1d9 !important; font-size: 1.1rem !important; font-weight: 600 !important; }
#demo-section p, #demo-section li { color: #8b949e !important; font-size: 0.92rem !important; line-height: 1.7 !important; }
#demo-section code {
    background: #21262d;
    color: #58a6ff;
    padding: 1px 6px;
    border-radius: 4px;
    font-family: monospace;
}

/* ── About section ───────────────────────────────────────────────── */
#about-section {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 28px 32px !important;
    margin-top: 16px;
}
#about-section h2 { font-size: 1.35rem !important; font-weight: 700 !important; color: #c9d1d9 !important; }
#about-section h3 { font-size: 1.05rem !important; font-weight: 600 !important; color: #8b949e !important; margin-top: 16px !important; }
#about-section p, #about-section li { font-size: 0.93rem !important; color: #6e7681 !important; line-height: 1.7 !important; }

/* ── Status badge ─────────────────────────────────────────────────── */
#status-box {
    padding: 10px 16px !important;
    border-radius: 8px;
    background: #161b22;
    border: 1px solid #30363d;
    font-weight: 500 !important;
    color: #c9d1d9 !important;
}

/* ── Language selector ───────────────────────────────────────────── */
#lang-select label { font-weight: 600 !important; color: #c9d1d9 !important; }
"""


def build_ui():
    with gr.Blocks(
        title="Indic OCR — Multilingual Document Digitization",
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────
        with gr.Row(elem_id="header-row"):
            with gr.Column(scale=1, min_width=80):
                gr.Image(
                    value=LOGO_PATH if Path(LOGO_PATH).exists() else None,
                    show_label=False,
                    container=False,
                    height=90,
                    width=90,
                )
            with gr.Column(scale=8):
                gr.Markdown(
                    """
# Indic OCR
**Multilingual Document Digitization** — Powered by Surya OCR Engine  
Upload scanned PDFs (up to 100 pages), images, or DOCX files across **11 Indic languages**.  
Formats: PDF · PNG · JPG · TIFF · BMP · WEBP · DOCX
"""
                )

        # ── Main I/O Area ───────────────────────────────────────────────
        with gr.Row():
            # ── LEFT: Input Panel ───────────────────────────────────────
            with gr.Column(scale=1, min_width=420, elem_classes="panel-card"):
                gr.Markdown("### 📤 Input")

                file_input = gr.File(
                    label="Upload Document",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg",
                                ".tiff", ".tif", ".bmp", ".webp",
                                ".docx", ".doc"],
                    type="filepath",
                )

                lang_select = gr.CheckboxGroup(
                    label="🌐 Select Languages",
                    choices=list(LANGUAGE_OPTIONS.keys()),
                    value=["Tamil (தமிழ்)"],
                    elem_id="lang-select",
                )

                with gr.Row():
                    run_btn = gr.Button("▶  Run OCR", variant="primary", size="lg", elem_id="run-btn")
                    gr.ClearButton(
                        [file_input],
                        value="Clear",
                    )

                with gr.Accordion("📏 Optional: CER / WER Reference", open=False):
                    gr.Markdown(
                        "Provide a ground-truth text to compute accuracy metrics.\n\n"
                        "- **CER** — Character Error Rate (character-level accuracy)\n"
                        "- **WER** — Word Error Rate (word-level accuracy)\n"
                    )
                    reference_text = gr.Textbox(
                        label="Reference text (paste)",
                        lines=4,
                        placeholder="Paste ground-truth text here (optional)…",
                    )
                    reference_file = gr.File(
                        label="Reference file (.txt)",
                        file_types=[".txt"],
                        type="filepath",
                    )

            # ── RIGHT: Output Panel ─────────────────────────────────────
            with gr.Column(scale=1, min_width=420, elem_classes="panel-card"):
                gr.Markdown("### 📥 Output")

                status_box = gr.Markdown(
                    value="**Status:** Upload a file and click **Run OCR** to begin.",
                    elem_id="status-box",
                )

                log_box = gr.Textbox(
                    label="Processing Log",
                    lines=15,
                    max_lines=15,
                    interactive=False,
                    placeholder="Processing log will appear here…",
                    elem_id="log-box",
                )

                download_btn = gr.DownloadButton(
                    label="⬇️  Download OCR Output (.txt)",
                    value=None,
                    variant="primary",
                    size="lg",
                    elem_id="dl-btn",
                )

                gr.Markdown("### 📊 Run Report")
                report_box = gr.Markdown(
                    value="Upload a file and click **Run OCR** to see a summary report here."
                )

        # ── Demo + Instructions Section ─────────────────────────────────
        with gr.Row():
            with gr.Column(elem_id="demo-section"):
                gr.Markdown(
                    """
### 🚀 Quick Start — Try the Demo

Download the sample Tamil story **எலி** (*The Mouse*, by Jeyamohan) and its reference ground-truth file below, then follow these steps:

1. **Upload** the `எலி.pdf` file using the **📂 Upload Document** button on the left panel.
2. **Select Language**: Make sure `Tamil (தமிழ்)` is checked in the language selector.
3. **Click ▶ Run OCR** to process the document.
4. **Evaluate Accuracy**: Once complete, click **▶ Optional: CER / WER Reference** and upload the `எலி_ground_truth.txt` file — or paste it in the text box — then click Run OCR again to get the Character Error Rate (CER) and Word Error Rate (WER) scores.
5. **Download** the extracted text using the `⬇️ Download OCR Output` button.

> **CER** (Character Error Rate) and **WER** (Word Error Rate) are automated metrics used to evaluate accuracy. Please note that while we normalize text, structural artifacts like page numbers, headers, and column splits can sometimes influence these scores. We recommend using them as a helpful guide alongside manual verification.
"""
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**📄 Sample PDF** — Tamil short story (6 pages)")
                        gr.DownloadButton(
                            label="⬇️ Download எலி.pdf",
                            value=DEMO_PDF_PATH if Path(DEMO_PDF_PATH).exists() else None,
                            variant="secondary",
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("**📋 Ground Truth Reference** — for CER/WER evaluation")
                        gr.DownloadButton(
                            label="⬇️ Download எலி_ground_truth.txt",
                            value=DEMO_GT_PATH if Path(DEMO_GT_PATH).exists() else None,
                            variant="secondary",
                        )

        # ── About Section ───────────────────────────────────────────────
        with gr.Row():
            with gr.Column(elem_id="about-section"):
                gr.Markdown(
                    """
## 🧠 Advanced AI & NLP Architecture

**Indic OCR** is a next-generation document digitization framework built on a **Multimodal AI Pipeline**. Unlike traditional rule-based systems, this project treats OCR as a complex Computer Vision and Natural Language Processing (NLP) problem, utilizing high-capacity Transformer models to handle the linguistic nuances of 11 Indic languages.

### 🧩 The AI Stack

The pipeline is orchestrated across four specialized neural layers:

1. **Neural Ingestion & Pre-processing**
   The system handles unstructured document formats (PDF/DOCX) by converting them into high-fidelity visual tensors. For scanned documents, we employ direct bitmap extraction to bypass compression artifacts, ensuring the downstream Vision models receive the highest possible signal-to-noise ratio.

2. **Vision-Transformer OCR Engine (Surya)**
   At the core of the recognition layer is the **Surya Engine**, which utilizes a **Vision Transformer (ViT)** backbone. 
   - **Bbox Detection**: A convolutional/transformer hybrid model identifies line-level regions of interest (RoI) with pixel-perfect precision.
   - **Sequence Modeling**: The text recognition phase treats each line as a sequence-to-sequence problem, mapping visual features to Unicode character sequences. This "neural reading" approach allows the model to generalize across thousands of fonts and even handwritten-style print, which traditional pattern-matching fails to capture.

3. **Cognitive Layout Analysis & XY-Cut Synthesis**
   Raw OCR output is often a "bag of words" without structural meaning. We apply a **Supervised Layout Model** to segment the page into semantic blocks (Body, Header, Footer, Caption, Table). 
   - **Semantic Filtering**: The model identifies and prunes "noise" blocks like running headers and page numbers.
   - **Geometric Reasoning**: We employ an **XY-Cut Algorithm** on the model's bounding boxes to reconstruct the human-intended reading order. This ensures that complex multi-column academic journals or newspaper layouts are digitized in a coherent, logical stream.

4. **NLP Refinement & Unicode Normalization**
   Post-recognition, the text undergoes a rigorous NLP cleaning phase. This involves regex-based noise suppression for common OCR hallucinations (like stray HTML tags or foreign script leakage) and strict **Unicode Normalization (NFKC)** to ensure that combined characters in Indic scripts (like vowel marks and consonants) are represented consistently for downstream search engines and LLMs.

### 🚀 Neural Language Correction *(Future Upgrade - Scope of this Project)*

The architectural roadmap includes a **Neural Correction Pass** utilizing **ByT5 (google/byt5-small)**—a character-level transformer model.
- **Character-Level Modeling**: Since OCR errors often occur at the sub-word level (e.g., a misread vowel mark), ByT5 is uniquely suited for Indic languages as it does not rely on a fixed vocabulary.
- **Surgical Gating**: Our "Surgical Pass" logic analyzes the confidence scores from the OCR engine and the noise density of the line. Only "suspicious" lines are routed through the ByT5 model for correction, maximizing throughput while significantly reducing the Character Error Rate (CER).
- **Fine-tuning Strategy**: The model fine-tuning leverages a custom synthetic-to-real corpus, designed to "hallucinate" the correct Tamil, Hindi, or Bengali morphology when the visual input is degraded.

### 📊 Performance Benchmarks

- **Throughput**: Optimized for GPU inference (NVIDIA T4/A100), processing ~2–5 seconds per page depending on text density.
- **Multilingual Support**: Native support for 11 Indic languages + English, including mixed-script detection.
- **Scalability**: Capable of batch processing documents up to 100 pages in a single session.
- **Accuracy**: Targets a 95%+ accuracy rate on modern printed text, with planned expansions focused on enhancing performance for 20th-century archived prints.
"""
                )

        # ── Wire up the callback ────────────────────────────────────────
        run_btn.click(
            fn=run_ocr,
            inputs=[file_input, lang_select, reference_text, reference_file],
            outputs=[status_box, log_box, download_btn, report_box],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7861")),
        inbrowser=True,
        theme=gr.themes.Default(),
        css=CUSTOM_CSS,
    )
