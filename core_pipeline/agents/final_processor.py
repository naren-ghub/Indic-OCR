"""
final_processor.py — Node 4: Final Processing Agent (One Whole-Doc LLM Call)
=============================================================================
Receives the entire assembled raw OCR text and in a single LLM call:
  1. Proofreads OCR character errors using linguistic context
  2. Strips running headers, footers, and page numbers
  3. Formats the output as structured Markdown

Uses gpt-oss-120b via OpenAI-compatible API.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    "ta": "Tamil (தமிழ்)",
    "hi": "Hindi (हिन्दी)",
    "te": "Telugu (తెలుగు)",
    "bn": "Bengali (বাংলা)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "ml": "Malayalam (മലയാളം)",
    "gu": "Gujarati (ગુજரાતી)",
    "mr": "Marathi (मराठी)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
    "or": "Odia (ଓଡ଼ିଆ)",
    "en": "English",
}

_PASS1_SYSTEM = """You are an expert multilingual Indic document structurer.

Supported language codes and names:
  ta=Tamil, hi=Hindi, te=Telugu, bn=Bengali, kn=Kannada,
  ml=Malayalam, gu=Gujarati, mr=Marathi, pa=Punjabi, or=Odia, en=English

You will receive the complete OCR-extracted text of a document.
Perform ONLY the following structural tasks:

1. CLEAN — Remove the following:
   - Repeating running headers (same text appearing at the top of multiple pages)
   - Repeating running footers (same text at the bottom of multiple pages)
   - Standalone page numbers (bare numbers on their own line)
   - OCR noise artifacts: stray pipe characters |, long underscores ____, stray HTML tags

2. FORMAT — Structure the output as Markdown:
   - Chapter titles        → # Heading
   - Section headers       → ## Heading
   - Footnotes / endnotes  → > Blockquote (prefix with >)
   - Normal body text      → plain text with a blank line between paragraphs
   - Preserve the original script and language exactly as-is.

Output ONLY the cleaned, formatted Markdown text. Do NOT fix spelling yet. Do NOT add commentary.
"""

_PASS2_SYSTEM = """You are an expert multilingual Indic document proofreader.

You will receive structured Markdown text that has already been cleaned of headers and footers.
Your task is to carefully proofread the text for OCR errors:

1. PROOFREAD — Fix obvious OCR character errors using linguistic and contextual clues.
   - If unsure whether a word is an OCR error or archaic / dialect vocabulary, LEAVE IT unchanged.
   - NEVER add, hallucinate, or invent new content.
   - NEVER change proper nouns unless 100% certain.

2. PRESERVE MARKDOWN — Keep all Markdown tags (#, >, *, etc.) exactly where they are.
   - Output ONLY the proofread Markdown text. Do NOT add commentary.
"""

# How many characters to include per LLM call (context window guard)
# Chunk size reduced to avoid 429 Too Many Requests on Llama 3.3 70B
_MAX_CHARS = 5_000


def run_final_processor(
    assembled_text: str,
    detected_language: str,
    total_pages: int,
    llm_client: Any,
    model: str,
) -> str:
    """
    Sends the full assembled OCR text to the LLM for proofreading,
    cleaning, and Markdown formatting in one call.

    If the document is very long (> _MAX_CHARS), it splits into chunks
    of 50 pages and concatenates the outputs.

    Args:
        assembled_text: Full raw OCR text (all pages concatenated).
        detected_language: ISO code (e.g. "ta").
        total_pages: Total page count (for the prompt).
        llm_client: OpenAI-compatible client.
        model: Model name string.

    Returns:
        Formatted Markdown string.
    """
    lang_label = SUPPORTED_LANGUAGES.get(detected_language, detected_language)

    if len(assembled_text) <= _MAX_CHARS:
        logger.info(
            "[FinalProcessor] Single-call processing (%d chars, %d pages, lang=%s)…",
            len(assembled_text), total_pages, detected_language,
        )
        return _process_chunk(assembled_text, lang_label, detected_language, total_pages, llm_client, model)

    # ── Split into chunks for very long documents ─────────────────────────────
    logger.info(
        "[FinalProcessor] Document too long (%d chars). Splitting into chunks of %d chars…",
        len(assembled_text), _MAX_CHARS,
    )
    chunks = []
    start = 0
    chunk_idx = 1
    while start < len(assembled_text):
        end = start + _MAX_CHARS
        # Break at newline boundary if possible
        if end < len(assembled_text):
            nl = assembled_text.rfind("\n", start, end)
            if nl > start:
                end = nl + 1
        chunk = assembled_text[start:end]
        logger.info("[FinalProcessor] Processing chunk %d (%d chars)…", chunk_idx, len(chunk))
        chunks.append(_process_chunk(chunk, lang_label, detected_language, total_pages, llm_client, model))
        start = end
        chunk_idx += 1

    return "\n\n".join(chunks)


def _process_chunk(
    text: str,
    lang_label: str,
    lang_code: str,
    total_pages: int,
    llm_client: Any,
    model: str,
) -> str:
    """Internal helper — calls the LLM for one chunk of text."""
    user_msg = (
        f"Document language: {lang_label} ({lang_code})\n"
        f"Total pages in this document: {total_pages}\n\n"
        f"--- FULL OCR TEXT ---\n{text}\n--- END OCR TEXT ---"
    )

    # --- PASS 1: Structure & Format ---
    logger.info("[FinalProcessor] Running Pass 1 (Structure & Format)…")
    pass1_response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _PASS1_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=4000,
    )
    structured_text = pass1_response.choices[0].message.content.strip()
    
    # --- PASS 2: Contextual Proofreading ---
    logger.info("[FinalProcessor] Running Pass 2 (Contextual Proofreading)…")
    pass2_msg = f"--- STRUCTURED MARKDOWN ---\n{structured_text}\n--- END STRUCTURED MARKDOWN ---"
    
    pass2_response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _PASS2_SYSTEM},
            {"role": "user", "content": pass2_msg},
        ],
        temperature=0.1,
        max_tokens=4000,
    )

    final_result = pass2_response.choices[0].message.content.strip()
    logger.info("[FinalProcessor] Chunk complete (%d output chars).", len(final_result))
    return final_result
