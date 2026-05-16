"""
Indic OCR — Multilingual Document Digitization (Gradio UI)
==========================================================
This file contains ONLY the Gradio frontend UI.
All heavy AI model logic runs on a separate GPU container via Modal.

When deployed on Modal, this runs on a cheap CPU-only container.
When the user clicks "Run OCR", it calls the GPU function remotely.
"""
import json
import os
import sys
from pathlib import Path

import gradio as gr

# ── Project root on sys.path ──────────────────────────────────────────────
ROOT = Path(__file__).parent

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Supported languages (UI-only, no model imports needed) ────────────────
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

# Logo path
LOGO_PATH = str(ROOT / "indian_ocr_logo.png")

# Demo file paths
DEMO_PDF_PATH = str(ROOT.parent / "OCR" / "OCR_dataset" / "எலி.pdf")
if not Path(DEMO_PDF_PATH).exists():
    DEMO_PDF_PATH = str(ROOT / "OCR_dataset" / "எலி.pdf")
DEMO_GT_PATH = str(ROOT / "OCR_dataset" / "எலி_ground_truth.txt")


# ── Core processing function (thin wrapper → GPU remote call) ─────────────

def run_ocr(file_obj, selected_languages, reference_text: str | None, reference_file_path: str | None):
    """
    Main Gradio callback. Reads the file into bytes, calls the GPU
    container via Modal's remote_gen(), and streams progress back to the UI.
    """
    if file_obj is None:
        yield "⚠️ Upload a file to begin.", "", None, ""
        return

    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
    if not path:
        yield "❌ Could not read uploaded file path.", "", None, ""
        return

    file_name = Path(path).name
    file_ext = Path(path).suffix.lower()

    # ── Resolve selected languages ──────────────────────────────────────
    if not selected_languages:
        selected_languages = ["Tamil (தமிழ்)"]
    lang_codes = [LANGUAGE_OPTIONS.get(l, "ta") for l in selected_languages]
    if "en" not in lang_codes:
        lang_codes.append("en")
    lang_label = ", ".join(selected_languages)

    # ── Read file bytes to pass to GPU container ────────────────────────
    with open(path, "rb") as f:
        file_bytes = f.read()

    # ── Read reference file bytes if provided ───────────────────────────
    reference_file_bytes = None
    if reference_file_path:
        try:
            with open(reference_file_path, "rb") as f:
                reference_file_bytes = f.read()
        except Exception:
            pass

    # ── Show "Waking up GPU" status ─────────────────────────────────────
    yield "⏳ Waking up GPU… (~15 seconds on first use)", "Connecting to GPU server…", None, ""

    # ── Call the GPU function remotely ──────────────────────────────────
    import modal
    run_ocr_gpu = modal.Function.from_name("indic-ocr", "run_ocr_gpu")

    try:
        for event_json in run_ocr_gpu.remote_gen(
            file_bytes=file_bytes,
            file_name=file_name,
            file_ext=file_ext,
            lang_codes=lang_codes,
            lang_label=lang_label,
            reference_text=reference_text if reference_text else None,
            reference_file_bytes=reference_file_bytes,
        ):
            event = json.loads(event_json)

            if event["type"] == "progress":
                yield event["status"], event["log"], None, ""

            elif event["type"] == "done":
                # The output .txt is on the shared volume at /tmp/gradio/
                output_path = event.get("output_path")
                yield event["status"], event["log"], output_path, event["report"]

            elif event["type"] == "error":
                yield f"❌ {event['message']}", event.get("log", ""), None, ""

    except Exception as e:
        yield f"❌ GPU error: {e}", f"Error communicating with GPU server:\n{e}", None, ""


# ── Gradio UI ──────────────────────────────────────────────────────────────

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

The pipeline integrates a specialized Agent Workflow alongside neural layers:

1. **Router Agent (Vision LLM)**
   Analyzes the first page to detect document type, language, noise levels, and estimated column count.

2. **Neural Ingestion & Pre-processing**
   The system handles unstructured document formats (PDF/DOCX) by converting them into high-fidelity visual tensors without binarization.

3. **Vision-Transformer OCR Engine (Surya)**
   At the core of the recognition layer is the **Surya Engine**, which utilizes a **Vision Transformer (ViT)** backbone for pixel-perfect line detection and sequence modeling.

4. **Enhanced Context-Aware Layout Engine & XY-Cut Synthesis**
   Uses the Router's column estimation to look for narrow vertical gutters and cleanly separate text in dense historical newspapers using a modified XY-Cut Algorithm.

5. **Subprocess Isolation & Hybrid Local Correction (IndicBART)**
   To resolve CUDA deadlocks and LLM truncation issues, **IndicBART** runs as a dedicated background server. It surgically corrects archaic spellings while preserving 100% of the body text and page structure.

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
