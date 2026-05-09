---
title: Indic OCR
emoji: 🇮🇳
colorFrom: orange
colorTo: green
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: true
license: mit
hardware: zero-a10g
---
# 🇮🇳 Indic OCR — Multilingual Document Digitization

An end-to-end AI pipeline for extracting text from scanned documents in **11 Indic languages**.

## Features

- **Vision Transformer OCR** powered by Surya Engine
- **Layout-aware reconstruction** with XY-Cut algorithm
- **11 languages**: Tamil, Hindi, Telugu, Bengali, Kannada, Malayalam, Gujarati, Marathi, Punjabi, Odia, English
- **Supports**: PDF (up to 100 pages), PNG, JPG, TIFF, BMP, WEBP, DOCX
- **Built-in CER/WER evaluation** against ground truth references

## Architecture

1. **Neural Ingestion** — PDF/Image → high-fidelity visual tensors
2. **Vision-Transformer OCR (Surya)** — ViT backbone for character recognition
3. **Cognitive Layout Analysis** — Supervised layout model + XY-Cut synthesis
4. **NLP Refinement** — Unicode normalization + noise suppression

## Future Upgrade (Scope of this project)

- ByT5 neural language correction (character-level error repair)
