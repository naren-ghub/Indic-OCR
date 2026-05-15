"""
qa_agent.py — Node 3: Quality Assurance Agent (Conditional)
============================================================
Triggered ONLY when average page confidence < 0.80 or when the Router Agent
detected a multi-column layout that X-Y Cut may have failed to separate.

Uses gpt-oss-120b (OpenAI-compatible API) to decide:
  - accept          → pass text to final processor
  - retry_single    → re-run Surya with forced single-column mode
  - retry_paddle    → run PaddleOCR as fallback
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    "ta": "Tamil (தமிழ்)",
    "hi": "Hindi (हिन्दी)",
    "te": "Telugu (తెలుగు)",
    "bn": "Bengali (বাংলা)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "ml": "Malayalam (മലയാளം)",
    "gu": "Gujarati (ગુજરાતી)",
    "mr": "Marathi (मराठी)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
    "or": "Odia (ଓଡ଼ିଆ)",
    "en": "English",
}

_QA_SYSTEM = """You are a multilingual Indic OCR quality inspector.
Supported language codes and names:
  ta=Tamil, hi=Hindi, te=Telugu, bn=Bengali, kn=Kannada,
  ml=Malayalam, gu=Gujarati, mr=Marathi, pa=Punjabi, or=Odia, en=English

You will receive:
1. The OCR-extracted text of one page.
2. A list of per-line confidence scores (float 0.0–1.0).
3. The detected document language code.

Tasks:
1. Does the text flow logically for this script? (yes/no)
2. Are there signs of column mixing or broken reading order? (yes/no)
3. Recommend one action:
   - "accept"         → text quality is acceptable
   - "retry_single"   → retry OCR forcing single-column mode
   - "retry_paddle"   → fallback to PaddleOCR engine

Output JSON ONLY. No explanation outside the JSON.
{
  "flow_ok": <bool>,
  "column_mixing": <bool>,
  "action": "accept" | "retry_single" | "retry_paddle",
  "reason": "<one sentence>"
}
"""

CONFIDENCE_THRESHOLD = 0.80


def should_run_qa(page_confidences: list[float], expected_columns: int, actual_text: str) -> bool:
    """
    Decide whether to invoke the QA LLM at all for this page.
    Returns True only if:
      - Average confidence is below threshold, OR
      - Multi-column layout was expected but text looks suspiciously sequential
    """
    if not page_confidences:
        return False
    avg_conf = sum(page_confidences) / len(page_confidences)
    if avg_conf < CONFIDENCE_THRESHOLD:
        logger.info(
            "[QAAgent] Triggering QA — avg_conf=%.3f < %.2f",
            avg_conf, CONFIDENCE_THRESHOLD,
        )
        return True
    if expected_columns >= 3 and len(actual_text.splitlines()) < 10:
        logger.info("[QAAgent] Triggering QA — multi-column doc but very few lines extracted.")
        return True
    return False


def run_qa_agent(
    page_text: str,
    page_confidences: list[float],
    detected_language: str,
    llm_client: Any,
    model: str,
) -> dict:
    """
    Calls the LLM to inspect one page's OCR output.

    Args:
        page_text: Raw OCR text for the page.
        page_confidences: List of per-line confidence floats.
        detected_language: ISO code (e.g. "ta").
        llm_client: OpenAI-compatible client.
        model: Model name string.

    Returns:
        dict with keys: flow_ok, column_mixing, action, reason
    """
    lang_label = SUPPORTED_LANGUAGES.get(detected_language, detected_language)
    avg_conf = sum(page_confidences) / len(page_confidences) if page_confidences else 0.0

    scores_preview = ", ".join(f"{c:.2f}" for c in page_confidences[:20])
    if len(page_confidences) > 20:
        scores_preview += f" … ({len(page_confidences)} lines total)"

    user_msg = (
        f"Document language: {lang_label} ({detected_language})\n"
        f"Average confidence: {avg_conf:.3f}\n"
        f"Per-line confidence scores: [{scores_preview}]\n\n"
        f"--- OCR TEXT ---\n{page_text[:3000]}\n--- END OCR TEXT ---"
    )

    logger.info("[QAAgent] Calling LLM for page QA (lang=%s, avg_conf=%.3f)…", detected_language, avg_conf)

    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _QA_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=200,
    )

    raw = response.choices[0].message.content.strip()
    logger.debug("[QAAgent] Raw: %s", raw)

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[QAAgent] JSON parse failed, defaulting to accept.")
        result = {"flow_ok": True, "column_mixing": False, "action": "accept", "reason": "Fallback"}

    logger.info("[QAAgent] Decision → action=%s | reason=%s", result.get("action"), result.get("reason"))
    return result
