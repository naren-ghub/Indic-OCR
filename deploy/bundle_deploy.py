"""
Deploy Bundler — Assembles the OCR_Deploy production package.
Copies all necessary source files into OCR_Deploy/ ready for:
  1. `modal deploy modal_app.py`   — serverless GPU deployment
  2. Drag `web/` to Netlify        — landing page hosting

Target layout:
  OCR_Deploy/
  ├── modal_app.py          ← Modal deployment driver (run from here)
  ├── requirements.txt      ← Python deps reference
  ├── README.md             ← Deployment instructions
  ├── app/                  ← Bundled Gradio app (mounted into cloud container)
  │   ├── gradio_app.py
  │   ├── indian_ocr_logo.png
  │   ├── OCR_dataset/
  │   │   ├── எலி.pdf
  │   │   └── எலி_ground_truth.txt
  │   └── OCR_Phase_2/
  │       ├── __init__.py
  │       ├── config.py
  │       └── pipeline/
  │           ├── __init__.py
  │           ├── ocr_engine.py
  │           ├── layout_engine.py
  │           ├── text_cleaner.py
  │           ├── pdf_converter.py
  │           ├── postprocessor.py
  │           └── preprocessor.py
  └── web/
      ├── index.html        ← Landing page (deploy to Netlify)
      └── logo.png

Usage:
    python bundle_deploy.py
"""
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

NLP_ROOT  = Path(r"d:\Evolve_Robot_Lab\Project\NLP Projects")
OCR_DIR   = NLP_ROOT / "OCR"
P2_DIR    = NLP_ROOT / "OCR_Phase_2"
DEPLOY    = NLP_ROOT / "OCR_Deploy"
APP       = DEPLOY / "app"
WEB       = DEPLOY / "web"

SRC_DIR   = OCR_DIR / "deploy"   # primary source: OCR/deploy/

print("=" * 60)
print("Indic OCR — Deployment Bundler")
print(f"Target: {DEPLOY}")
print("=" * 60)

# ── 1. Copy Gradio app ────────────────────────────────────────
shutil.copy2(SRC_DIR / "gradio_app.py", APP / "gradio_app.py")
print("[1] Copied gradio_app.py → app/")

# ── 2. Copy logo ──────────────────────────────────────────────
logo_src = SRC_DIR / "indian_ocr_logo.png"
if not logo_src.exists():
    logo_src = OCR_DIR / "indian_ocr_logo.png"
if logo_src.exists():
    shutil.copy2(logo_src, APP / "indian_ocr_logo.png")
    print("[2] Copied indian_ocr_logo.png → app/")

# ── 3. Copy Phase 2 pipeline ──────────────────────────────────
p2_dst = APP / "OCR_Phase_2"
pipeline_dst = p2_dst / "pipeline"
pipeline_dst.mkdir(parents=True, exist_ok=True)

shutil.copy2(P2_DIR / "config.py", p2_dst / "config.py")
(p2_dst / "__init__.py").write_text("")
(pipeline_dst / "__init__.py").write_text("")
print("[3] Copied OCR_Phase_2/config.py")

needed_modules = [
    "ocr_engine.py",
    "layout_engine.py",
    "text_cleaner.py",
    "pdf_converter.py",
    "postprocessor.py",
    "preprocessor.py",
]
for mod in needed_modules:
    src = P2_DIR / "pipeline" / mod
    if src.exists():
        shutil.copy2(src, pipeline_dst / mod)
        print(f"[3] Copied pipeline/{mod}")
    else:
        print(f"[3] WARNING: {mod} not found!")

# ── 4. Copy demo dataset ──────────────────────────────────────
demo_dst = APP / "OCR_dataset"
demo_dst.mkdir(parents=True, exist_ok=True)

demo_pdf = OCR_DIR / "OCR_dataset" / "எலி.pdf"
demo_gt  = OCR_DIR / "OCR_dataset" / "எலி_ground_truth.txt"

if demo_pdf.exists():
    shutil.copy2(demo_pdf, demo_dst / "எலி.pdf")
    print("[4] Bundled demo: எலி.pdf")
if demo_gt.exists():
    shutil.copy2(demo_gt, demo_dst / "எலி_ground_truth.txt")
    print("[4] Bundled demo: எலி_ground_truth.txt")

# ── 5. Confirm modal_app.py is in place ─────────────────────
if not (DEPLOY / "modal_app.py").exists():
    shutil.copy2(OCR_DIR / "deploy" / "modal_app.py", DEPLOY / "modal_app.py")
print("[5] modal_app.py ready at OCR_Deploy/")

# ── 6. Copy web landing page ──────────────────────────────────
WEB.mkdir(parents=True, exist_ok=True)
shutil.copy2(OCR_DIR / "index.html", WEB / "index.html")
logo_web = OCR_DIR / "indian_ocr_logo.png"
if logo_web.exists():
    shutil.copy2(logo_web, WEB / "logo.png")
print("[6] Copied index.html + logo → web/")

# ── Summary ───────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"Bundle complete → {DEPLOY}")
print(f"{'=' * 60}")
for p in sorted(DEPLOY.rglob("*")):
    if p.is_file() and "__pycache__" not in str(p):
        rel = p.relative_to(DEPLOY)
        size = p.stat().st_size
        print(f"  {rel}  ({size:,} bytes)")
