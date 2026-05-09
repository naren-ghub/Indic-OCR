# Indic-OCR

An end-to-end Optical Character Recognition (OCR) pipeline explicitly optimized for complex Indic languages (focusing on Tamil). This project uses state-of-the-art vision models and language models to extract, structure, and correct text from difficult document layouts.

## Features

- **Surya Integration**: Robust line-level bounding box detection and text recognition without relying on destructive traditional binarization.
- **Recursive X-Y Cut Layout Analysis**: A deterministic, geometric approach to analyzing multi-column layouts, ensuring text is extracted in the correct human reading order.
- **Full-Page Fallback Mechanisms**: Automatically adapts to dense text layouts (like newspapers) where layout bounding boxes might misclassify text blocks as pictures.
- **ByT5 Error Correction Layer**: Incorporates a custom-trained `google/byt5-small` model (`Naren-hug/byt5-tamil-ocr-v1`) that naturally understands raw UTF-8 byte sequences to fix character-level OCR garbling.
- **Header & Footer Stripping**: Intelligent artifact stripping using `TextFormatter` to ensure seamless reading flow across page boundaries.

## Architecture

* `core_pipeline/`: The core logic, including the `SuryaEngine`, `layout_engine` (X-Y Cut), `text_formatter`, and `lm_corrector`.
* `deploy/`: Contains the Gradio web UI and Modal cloud serverless deployment scripts.
* `research/`: Notebooks and data processing scripts used to build the ByT5 finetuning dataset.
* `tools/`: Independent CLI utilities for bulk processing and PDF extraction.

---

## Local Setup

### Requirements

Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Indic-OCR.git
cd Indic-OCR

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

*(Note: PyTorch will be installed by the requirements file. If you have a GPU, ensure you install the CUDA-enabled version of PyTorch for optimal performance with Surya and ByT5).*

### Running the Pipeline Locally

You can use the command-line interface to run the OCR engine on a local PDF or image.

```bash
# Run the pipeline on a specific file
python core_pipeline/main.py --input "demo_data/எலி.pdf" --output_dir "output"
```

You can control various flags in `core_pipeline/config.py` (e.g., toggling the LM correction layer `LM_CORRECTION_ENABLED` or tweaking `QUALITY_CONF_THRESHOLD`).

---

## Cloud Deployment (Modal Serverless)

The application includes a fully containerized deployment script for [Modal](https://modal.com), allowing you to run the heavy AI models on serverless A10G/T4 GPUs.

1. **Install and authenticate with Modal:**
   ```bash
   pip install modal
   modal setup
   ```

2. **Deploy the Gradio Web App:**
   Navigate to the `deploy/` directory and run the deployment script. This will bundle the `core_pipeline` and deploy it as a serverless Gradio app.

   ```bash
   cd deploy
   modal deploy modal_app.py
   ```

3. **Access the Web App:**
   Once deployed, Modal will print a public URL (e.g., `https://your-workspace--indic-ocr-web.modal.run`) where you can interact with the OCR UI from any browser.
