"""
router_agent.py — Node 1: Document Router Agent
================================================
Uses Llama-3.2-Vision via Groq to analyse the first page of the document
and decide which preprocessing path to take.

Returns:
    {
        "doc_type": "modern_print" | "historical_scan" | "newspaper",
        "detected_language": "ta" | "hi" | ... ,
        "noise_level": 0-10,
        "estimated_columns": 1-5
    }
"""
from __future__ import annotations

import base64
import json
import logging
from io import BytesIO
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# ── Supported languages exactly as defined in the OCR system ─────────────────
SUPPORTED_LANGUAGES = {
    "ta": "Tamil (தமிழ்)",
    "hi": "Hindi (हिन्दी)",
    "te": "Telugu (తెలుగు)",
    "bn": "Bengali (বাংলা)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "ml": "Malayalam (മലയാളം)",
    "gu": "Gujarati (ગુજરાતી)",
    "mr": "Marathi (मराठी)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
    "or": "Odia (ଓଡ଼ିଆ)",
    "en": "English",
}

_ROUTER_SYSTEM = """You are an expert OCR preprocessing classifier for Indic documents.

Supported language codes: ta (Tamil), hi (Hindi), te (Telugu), bn (Bengali),
kn (Kannada), ml (Malayalam), gu (Gujarati), mr (Marathi), pa (Punjabi),
or (Odia), en (English).

You will receive an image of the first page of a scanned document.
Analyse it and output a JSON object ONLY — no explanation, no markdown fences.

JSON schema:
{
  "doc_type": "modern_print" | "historical_scan" | "newspaper",
  "detected_language": "<iso_code>",
  "noise_level": <integer 0-10>,
  "estimated_columns": <integer 1-5>,
  "routing_reason": "<one short sentence>"
}

Definitions:
- modern_print  → clean digital-origin PDF, sharp fonts, good contrast
- historical_scan → degraded, noisy, low-contrast, often pre-1980 print
- newspaper → multi-column (3+ columns), dense, small font, mixed scripts
"""


def _image_to_b64(img: Image.Image, max_side: int = 1024) -> str:
    """Downsample and encode a PIL image as base64 JPEG."""
    img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_router_agent(
    page_image: Image.Image,
    groq_client: Any,
    model: str = "llama-3.2-11b-vision-preview",
) -> dict:
    """
    Calls the Groq Vision LLM with page_image and returns the routing decision.

    Args:
        page_image: PIL Image of the first page.
        groq_client: An initialised `groq.Groq` client instance.
        model: Groq vision model name.

    Returns:
        dict with keys: doc_type, detected_language, noise_level,
                        estimated_columns, routing_reason
    """
    b64 = _image_to_b64(page_image)

    logger.info("[RouterAgent] Sending page 1 to Groq Vision (%s)…", model)

    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": "Classify this document page and output the JSON.",
                        },
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()
        logger.debug("[RouterAgent] Raw response: %s", raw)

        # Strip markdown fences if the model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)
    except Exception as e:
        logger.warning("[RouterAgent] Vision API failed (%s). Defaulting to historical_scan/ta", e)
        result = {
            "doc_type": "historical_scan",
            "detected_language": "ta",
            "noise_level": 5,
            "estimated_columns": 1,
            "routing_reason": f"Fallback — Vision API error: {e}",
        }

    # Validate language code
    if result.get("detected_language") not in SUPPORTED_LANGUAGES:
        logger.warning(
            "[RouterAgent] Unknown language '%s', defaulting to Tamil.",
            result.get("detected_language"),
        )
        result["detected_language"] = "ta"

    logger.info(
        "[RouterAgent] Decision → type=%s | lang=%s | noise=%s | cols=%s",
        result.get("doc_type"),
        result.get("detected_language"),
        result.get("noise_level"),
        result.get("estimated_columns"),
    )
    return result
