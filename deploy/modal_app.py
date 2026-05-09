"""
Modal.com Deployment Script for Indic OCR
=========================================
Compatible with Modal v1.x (uses .add_local_dir() instead of deprecated modal.Mount)

Fixes applied:
- transformers pinned to ==4.57.6 (local working version; 5.x is broken with surya 0.17.1)
- surya-ocr pinned to ==0.17.1 (exact version tested locally)
- Removed .run_commands() model pre-cache (crashes on CPU during build due to config bug)
  Models will instead download on the first GPU cold start (~60s one-time wait)
- Added all required system libraries (libGL etc.) for opencv in headless Linux

Folder layout (run from OCR_Deploy/):
  OCR_Deploy/
  ├── modal_app.py    ← this file — run: python -m modal deploy modal_app.py
  ├── requirements.txt
  ├── README.md
  ├── app/            ← bundled Gradio app (mounted into cloud container as /root/deploy)
  │   ├── gradio_app.py
  │   ├── OCR_dataset/
  │   └── OCR_Phase_2/pipeline/
  └── web/            ← landing page (deploy to Netlify separately)
      └── index.html

Usage:
1. Run bundle_deploy.py to populate app/ and web/ from source
2. cd OCR_Deploy
3. python -m modal deploy modal_app.py
"""
import modal
import sys
import os

# 1. Define the Modal App
app = modal.App("indic-ocr")

# 2. Define the container image, dependencies, and local files
image = (
    modal.Image.debian_slim(python_version="3.11")
    # System libraries required by opencv (libGL.so.1) — not present in slim containers
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6"])
    # Pin exact versions matching the tested local environment.
    # transformers 5.x is broken with surya-ocr 0.17.1 (SuryaDecoderConfig.pad_token_id bug)
    .pip_install(
        "surya-ocr==0.17.1",
        "transformers==4.57.6",   # pinned — 5.x breaks surya model loading
        "PyMuPDF>=1.23.0",
        "gradio>=6.14.0",
        "jiwer>=3.0.0",
        "pillow",
        "numpy",
        "opencv-python-headless", # headless variant — no display required in cloud
        "python-docx",
        "torch>=2.2.0",
        "fastapi",
        "requests",
    )
    # Bundle the app/ subdirectory into the container at /root/deploy
    # Run `bundle_deploy.py` first to populate app/ from source
    .add_local_dir("./app", remote_path="/root/deploy")
)

# 3. Create a shared volume for Gradio's temporary uploaded files.
# Since Modal is serverless, an upload might hit Container A and the OCR request might hit Container B.
# This shared volume ensures the uploaded PDF is available to all containers.
gradio_vol = modal.Volume.from_name("gradio-tmp", create_if_missing=True)

# 4. Define the serverless web endpoint
# T4 GPU, scales to 0 when idle (no charges while unused)
# NOTE: First cold start will take ~60s while Surya downloads model weights.
#       Subsequent starts reuse the cached weights from the container image layers.
@app.function(image=image, volumes={"/tmp/gradio": gradio_vol}, gpu="T4", min_containers=0)
@modal.asgi_app()
def fastapi_app():
    # Set up sys.path so gradio_app.py can find the bundled OCR_Phase_2 pipeline
    sys.path.insert(0, "/root/deploy")
    os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
    
    # Force Python's tempfile to use the shared volume!
    # This fixes the bug where output .txt files are lost between container scales.
    os.environ["TMPDIR"] = "/tmp/gradio"

    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from gradio_app import build_ui

    demo = build_ui()
    fastapi_obj = FastAPI()
    return mount_gradio_app(app=fastapi_obj, blocks=demo, path="/")
