"""
Modal.com Deployment Script for Indic OCR — Microservice Architecture
=====================================================================
Split into two containers for cost-efficiency:

  1. CPU Container (cheap) — Serves the Gradio UI. No GPU attached.
     Users can browse the page, upload files, and read docs for free.

  2. GPU Container (T4) — Only cold-starts when user clicks "Run OCR".
     Loads Surya models, processes the document, yields streaming
     progress events back to the CPU container, then scales to zero.

Usage:
  1. Run bundle_deploy.py to populate app/ from source
  2. cd deploy/
  3. python -m modal deploy modal_app.py
"""
import modal
import sys
import os

# ── Modal App ──────────────────────────────────────────────────────────────
app = modal.App("indic-ocr")

# ── Shared image (both containers need basic Python deps) ──────────────────
_base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6"])
)

# GPU image: includes heavy AI dependencies + bundled pipeline code
gpu_image = (
    _base_image
    .pip_install(
        "surya-ocr==0.17.1",
        "transformers==4.57.6",
        "PyMuPDF>=1.23.0",
        "jiwer>=3.0.0",
        "pillow",
        "numpy",
        "opencv-python-headless",
        "python-docx",
        "torch>=2.2.0",
    )
    .add_local_dir("./app", remote_path="/root/deploy")
)

# CPU image: only needs Gradio + FastAPI (lightweight)
cpu_image = (
    _base_image
    .pip_install(
        "gradio>=6.14.0",
        "fastapi",
        "pillow",
        "jiwer>=3.0.0",
    )
    .add_local_dir("./app", remote_path="/root/deploy")
)

# Shared volume for output files (CPU container needs to serve the .txt download)
gradio_vol = modal.Volume.from_name("gradio-tmp", create_if_missing=True)


# ═══════════════════════════════════════════════════════════════════════════
# GPU FUNCTION — Only runs when "Run OCR" is clicked
# ═══════════════════════════════════════════════════════════════════════════
@app.function(
    image=gpu_image,
    volumes={"/tmp/gradio": gradio_vol},
    gpu="T4",
    min_containers=0,
    timeout=600,
)
def run_ocr_gpu(
    file_bytes: bytes,
    file_name: str,
    file_ext: str,
    lang_codes: list[str],
    lang_label: str,
    reference_text: str | None,
    reference_file_bytes: bytes | None,
):
    """
    Generator that runs the full OCR pipeline on a GPU container.
    Yields JSON-encoded progress/result events back to the CPU container.

    Event types:
      {"type": "progress", "status": "...", "log": "..."}
      {"type": "done", "status": "...", "log": "...", "output_path": "...", "report": "..."}
      {"type": "error", "message": "..."}
    """
    import io
    import re
    import json
    import unicodedata
    import tempfile
    from pathlib import Path
    from PIL import Image

    # ── Write uploaded bytes to a temp file so loaders can open it ──────
    tmp_input = tempfile.NamedTemporaryFile(
        delete=False, suffix=file_ext, dir="/tmp/gradio"
    )
    tmp_input.write(file_bytes)
    tmp_input.close()
    path = tmp_input.name

    # ── Write reference file bytes if provided ─────────────────────────
    ref_file_path = None
    if reference_file_bytes:
        tmp_ref = tempfile.NamedTemporaryFile(
            delete=False, suffix=".txt", dir="/tmp/gradio"
        )
        tmp_ref.write(reference_file_bytes)
        tmp_ref.close()
        ref_file_path = tmp_ref.name

    # ── Import pipeline (runs inside GPU container) ────────────────────
    sys.path.insert(0, "/root/deploy")
    sys.path.insert(0, "/root/deploy/OCR_Phase_2")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from surya.foundation import FoundationPredictor
    fp = FoundationPredictor(device=device)

    from pipeline.ocr_engine import create_engine
    from pipeline.layout_engine import LayoutEngine
    from pipeline.text_cleaner import clean_text as _deep_clean

    engine = create_engine("surya", langs=lang_codes, device=device, foundation_predictor=fp)
    layout_engine = LayoutEngine(foundation_predictor=fp)

    # ── Helpers (copied from gradio_app.py to keep GPU self-contained) ─
    MAX_PAGES = 100
    SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
    _REFLOW_THRESHOLD = 18

    def _reflow_text(text: str) -> str:
        lines = text.splitlines()
        merged: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                merged.append("")
            elif merged and stripped and len(stripped) < _REFLOW_THRESHOLD and not merged[-1] == "":
                merged[-1] = merged[-1] + " " + stripped
            else:
                merged.append(stripped)
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

    def _normalize_for_metrics(text: str) -> str:
        if not text:
            return ""
        text = text.lstrip('\ufeff')
        lines = text.splitlines()
        cleaned = []
        for line in lines:
            s = line.strip()
            if re.match(r'^Indic OCR Output', s): continue
            if re.match(r'^Languages:', s): continue
            if re.match(r'^Pages processed:', s): continue
            if re.match(r'^Tamil OCR Output', s): continue
            if re.match(r'^=+$', s): continue
            if re.match(r'^PAGE\s+\d+', s): continue
            if re.match(r'^\d{1,4}$', s): continue
            if re.match(r'^#+\s', s): continue
            if len(s) <= 2 and s: continue
            cleaned.append(s)
        joined = ' '.join(' '.join(cleaned).split())
        return unicodedata.normalize('NFC', joined)

    # ── File loaders ───────────────────────────────────────────────────
    def load_pdf(fpath):
        import fitz
        doc = fitz.open(fpath)
        total = len(doc)
        if total > MAX_PAGES:
            raise ValueError(f"PDF has {total} pages. Maximum allowed is {MAX_PAGES}.")
        images = []
        for page_num in range(total):
            page = doc[page_num]
            embedded = page.get_images(full=True)
            text_len = len(page.get_text("text").strip())
            if len(embedded) == 1 and text_len == 0:
                xref = embedded[0][0]
                base = doc.extract_image(xref)
                img = Image.open(io.BytesIO(base["image"])).convert("RGB")
            else:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            images.append((page_num + 1, img))
        doc.close()
        return images

    def load_image_file(fpath):
        img = Image.open(fpath).convert("RGB")
        return [(1, img)]

    def load_docx(fpath):
        import docx
        doc = docx.Document(fpath)
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
            raise ValueError("This DOCX contains no embedded images.")
        if len(images) > MAX_PAGES:
            raise ValueError(f"DOCX contains {len(images)} images. Maximum allowed is {MAX_PAGES}.")
        return [(i + 1, img) for i, img in enumerate(images)]

    # ── Reference text loader ──────────────────────────────────────────
    def _maybe_load_reference_text():
        if ref_file_path:
            try:
                with open(ref_file_path, "r", encoding="utf-8") as f:
                    txt = f.read().strip()
                if txt:
                    return txt, "user-provided reference file"
            except Exception:
                pass
        if reference_text and reference_text.strip():
            return reference_text.strip(), "user-provided reference text"
        if file_ext == ".pdf":
            try:
                import fitz
                pdf = fitz.open(path)
                parts = [page.get_text("text", sort=True) or "" for page in pdf]
                pdf.close()
                embedded = "\n".join(parts).strip()
                if embedded:
                    return embedded, "embedded PDF text (best-effort)"
            except Exception:
                pass
        return None, "no reference available"

    # ── Helper: yield a progress event ─────────────────────────────────
    def _progress(status, log_lines):
        return json.dumps({"type": "progress", "status": status, "log": "\n".join(log_lines)}, ensure_ascii=False)

    # ── MAIN PROCESSING ────────────────────────────────────────────────
    try:
        # Load pages
        if file_ext == ".pdf":
            pages = load_pdf(path)
        elif file_ext in SUPPORTED_IMAGE_EXTS:
            pages = load_image_file(path)
        elif file_ext in {".docx", ".doc"}:
            pages = load_docx(path)
        else:
            yield json.dumps({"type": "error", "message": f"Unsupported file type: '{file_ext}'"})
            return
    except (ValueError, Exception) as e:
        yield json.dumps({"type": "error", "message": str(e)})
        return

    total_pages = len(pages)
    log_lines = [
        f"📄 Loaded {total_pages} page(s) from '{file_name}'",
        f"🌐 Languages: {lang_label}",
        "",
    ]
    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    spin_i = 0

    yield _progress(f"{spinner[spin_i]} Processing…", log_lines)

    ref_text, ref_source = _maybe_load_reference_text()

    all_text_parts = []
    pred_pages_for_metrics: list[str] = []

    for page_num, img in pages:
        spin_i = (spin_i + 1) % len(spinner)
        log_lines.append(f"🔍 Page {page_num}/{total_pages}: OCR…")
        yield _progress(f"{spinner[spin_i]} Processing page {page_num}/{total_pages}…", log_lines)

        try:
            result = engine.process_image(img)
            ocr_lines = result.get("lines", [])
            lines_count = len(ocr_lines)

            spin_i = (spin_i + 1) % len(spinner)
            log_lines.append(f"   🧩 Layout filter…")
            yield _progress(f"{spinner[spin_i]} Layout filtering…", log_lines)

            layout = layout_engine.analyze_page(img, page_num=page_num)
            text = LayoutEngine.reconstruct_page_text(ocr_lines, layout)

            spin_i = (spin_i + 1) % len(spinner)
            yield _progress(f"{spinner[spin_i]} Cleaning…", log_lines)

            text = _deep_clean(text)
            text = _reflow_text(text)

            all_text_parts.append(
                f"{'='*60}\nPAGE {page_num}\n{'='*60}\n{text}\n"
            )
            pred_pages_for_metrics.append(text)
            log_lines.append(f"   ✅ {lines_count} lines extracted")

            spin_i = (spin_i + 1) % len(spinner)
            yield _progress(f"{spinner[spin_i]} Page {page_num} done", log_lines)

        except Exception as e:
            log_lines.append(f"   ❌ Page {page_num} failed: {e}")
            all_text_parts.append(f"{'='*60}\nPAGE {page_num}\n{'='*60}\n[OCR ERROR: {e}]\n")
            pred_pages_for_metrics.append("")
            yield _progress(f"{spinner[spin_i]} Error on page {page_num}", log_lines)

    # ── Write output .txt to shared volume ─────────────────────────────
    output_filename = Path(file_name).stem + "_ocr_output.txt"
    output_path = f"/tmp/gradio/{output_filename}"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Indic OCR Output — {file_name}\n")
        f.write(f"Languages: {lang_label}\n")
        f.write(f"Pages processed: {total_pages}\n")
        f.write("=" * 60 + "\n\n")
        f.write("\n".join(all_text_parts))

    # Commit the volume so the CPU container can read the file
    gradio_vol.commit()

    # ── Compute CER/WER ────────────────────────────────────────────────
    cer = None
    wer = None
    metrics_note = None

    pred_full = _normalize_for_metrics("\n".join(pred_pages_for_metrics))
    ref_full = _normalize_for_metrics(ref_text) if ref_text else ""

    try:
        import jiwer
        if ref_full:
            cer = float(jiwer.cer(ref_full, pred_full))
            wer = float(jiwer.wer(ref_full, pred_full))
        else:
            metrics_note = (
                "> Note: CER/WER require a reference text. Provide one (paste or upload a `.txt`),\n"
                "> or use a PDF that contains selectable text."
            )
    except ImportError:
        if ref_full:
            metrics_note = "> Note: `jiwer` is not available, so CER/WER could not be computed."
    except Exception as e:
        metrics_note = f"> Note: CER/WER computation failed: `{e}`"

    # ── Build report ───────────────────────────────────────────────────
    cer_str = f"{cer:.2%}" if cer is not None else "N/A"
    wer_str = f"{wer:.2%}" if wer is not None else "N/A"
    extra = f"\n\n{metrics_note}" if metrics_note else ""

    report_md = (
        "### 📊 Run Report\n\n"
        f"| Property | Value |\n"
        f"|---|---|\n"
        f"| **File** | `{file_name}` |\n"
        f"| **Pages processed** | {total_pages} |\n"
        f"| **Languages** | {lang_label} |\n"
        f"| **Output** | `{output_filename}` |\n"
        f"| **Reference** | {ref_source} |\n"
        f"| **CER** | {cer_str} |\n"
        f"| **WER** | {wer_str} |\n"
        "\n"
        "> **CER** measures character-level accuracy. **WER** measures word-level accuracy. Lower is better.\n"
        f"{extra}"
    )

    log_lines.append("")
    log_lines.append(f"✅ Done! {total_pages} page(s) processed.")
    log_lines.append("📥 Download is ready.")

    # ── Final event with output path ───────────────────────────────────
    yield json.dumps({
        "type": "done",
        "status": "✅ Completed",
        "log": "\n".join(log_lines),
        "output_path": output_path,
        "output_filename": output_filename,
        "report": report_md,
    }, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# CPU FUNCTION — Serves the Gradio UI (no GPU, cheap)
# ═══════════════════════════════════════════════════════════════════════════
@app.function(
    image=cpu_image,
    volumes={"/tmp/gradio": gradio_vol},
    min_containers=0,
    # No gpu= here! This is a CPU-only container.
)
@modal.asgi_app()
def fastapi_app():
    sys.path.insert(0, "/root/deploy")
    os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
    os.environ["TMPDIR"] = "/tmp/gradio"

    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from gradio_app import build_ui

    demo = build_ui()
    fastapi_obj = FastAPI()
    return mount_gradio_app(app=fastapi_obj, blocks=demo, path="/")
