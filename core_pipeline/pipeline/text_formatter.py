"""
Professional Text Formatter — Document-Level Post-Processing
=============================================================
Runs AFTER per-page OCR + layout reconstruction to produce clean,
publication-ready text output.

Handles:
  1. Page-number stripping     — standalone 1–3 digit lines at page boundaries
  2. Running-header detection  — repeated author names / story titles via
                                 frequency analysis across pages
  3. Short-line merging        — recombines fragmented OCR lines into paragraphs
  4. Document assembly         — joins pages with clean paragraph spacing

Design:
  - Operates on List[str] (one string per page), not on raw OCR data.
  - Non-destructive to body text: only removes structural artifacts.
  - Tamil-aware: uses Unicode properties for merge decisions.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import List, Optional, Set

# ── Tamil Unicode helpers ──────────────────────────────────────────────────────

TAMIL_RANGE = (0x0B80, 0x0BFF)

# Characters that typically end a Tamil sentence or a dialogue turn
# Includes straight and curly quote variants used in Tamil typography
SENTENCE_ENDERS = {'.', '!', '?', '।', '"', "'", "''", '"', '’’', '”', '’', '‘', '“', '…'}

# Regex for standalone page numbers (1–4 digits, possibly with surrounding whitespace)
PAGE_NUM_RE = re.compile(r'^\s*\d{1,4}\s*$')

# Roman numeral page numbers (i, ii, iii, iv, v, vi, ... xxviii, etc.)
ROMAN_NUM_RE = re.compile(r'^\s*[ivxlcdmIVXLCDM]{1,8}\s*$')

# Regex for <math> tag blocks that Surya OCR hallucinates from decorative elements
MATH_TAG_RE = re.compile(r'<math[^>]*>.*?</math>', re.DOTALL)

# URL watermark patterns (e.g., www.padippakam.com)
URL_RE = re.compile(r'\b(?:https?://|www\.)[^\s]+', re.IGNORECASE)

# Lines that are pure URL watermarks or site names
WATERMARK_RE = re.compile(r'^\s*(?:www\.[\w.]+|படிப்பகம்|padippakam)\s*$', re.IGNORECASE)

# Garbage patterns: standalone punctuation, math residue, Cyrillic residue
GARBAGE_LINE_RE = re.compile(
    r'^\s*'
    r'(?:'
    r'[()\[\]{}•\-_.,:;\\|/\s]{1,10}'  # Pure punctuation/brackets
    r'|[ÇЭЊ₩]+[\s\-.,]*'              # Cyrillic/currency residue
    r'|\d{1,2}\.\d{2}(?:<br>\d{1,2}\.\d{2})?'  # number patterns like 40.55<br>82.55
    r'|[oO0]'                           # Standalone o/O/0
    r')'
    r'\s*$'
)

# Regex for common OCR garbage lines (non-Tamil, non-English, very short)
GARBAGE_RE = re.compile(r'^[\s\W]{0,5}$')

# Regex matching lines that are ONLY English letters + spaces (author names like "GRAT GRATITION")
ENGLISH_ONLY_RE = re.compile(r'^[A-Za-z\s.,\'-]+$')


def _has_tamil(text: str) -> bool:
    """Check if a string contains any Tamil Unicode characters."""
    return any(TAMIL_RANGE[0] <= ord(c) <= TAMIL_RANGE[1] for c in text)


def _is_tamil_char(c: str) -> bool:
    """Check if a single character is in the Tamil Unicode block."""
    return TAMIL_RANGE[0] <= ord(c) <= TAMIL_RANGE[1]


def _ends_with_sentence_ender(line: str) -> bool:
    """Check if a line ends with a sentence-ending punctuation mark."""
    line = line.rstrip()
    if not line:
        return False
    # Check last 1–2 characters for sentence enders
    for ender in SENTENCE_ENDERS:
        if line.endswith(ender):
            return True
    return False


def _is_page_number(line: str) -> bool:
    """Check if a line is a standalone page number (Arabic or Roman)."""
    return bool(PAGE_NUM_RE.match(line)) or bool(ROMAN_NUM_RE.match(line))


def _is_short_structural_line(line: str) -> bool:
    """
    Check if a line is a short structural element (page header, title repeat, etc.)
    These are typically short (<40 chars), appear at page boundaries.
    """
    stripped = line.strip()
    if not stripped:
        return False
    # Very short Tamil-only lines (<15 chars) at page boundaries are likely headers
    if len(stripped) < 15 and _has_tamil(stripped) and not stripped.startswith("'"):
        return True
    return False


# ── Core Formatter Class ──────────────────────────────────────────────────────

class TextFormatter:
    """
    Document-level text formatter that produces professional output
    from raw per-page OCR text.
    """

    @staticmethod
    def strip_page_numbers(pages: List[str]) -> List[str]:
        """
        Remove standalone page number lines from the first/last 3 lines
        of each page's text.

        Page numbers in scanned books typically appear at the very top or
        very bottom of the page. We only strip lines that are PURELY numeric.
        """
        result = []
        for page_text in pages:
            lines = page_text.split('\n')
            cleaned = []
            for i, line in enumerate(lines):
                # Check first 3 and last 3 lines for page numbers
                is_boundary = (i < 3) or (i >= len(lines) - 3)
                if is_boundary and _is_page_number(line):
                    continue  # Skip page number
                cleaned.append(line)
            result.append('\n'.join(cleaned))
        return result

    @staticmethod
    def detect_running_headers(pages: List[str], threshold: float = 0.35) -> Set[str]:
        """
        Detect running headers by frequency analysis.

        A running header is a short line that appears in the first 3 lines
        of many pages (>threshold fraction). Common examples:
          - Author name: "அசோகமித்திரன்"
          - Story title: "கடன்", "எலி"
          - English title remnants: "GRAT GRATITION."

        Args:
            pages: List of page texts.
            threshold: Fraction of pages a line must appear in to be
                       classified as a running header.

        Returns:
            Set of header strings to strip.
        """
        if len(pages) < 3:
            return set()

        # Count how many pages each "edge line" appears in
        edge_line_counter = Counter()
        for page_text in pages:
            lines = [l.strip() for l in page_text.strip().split('\n') if l.strip()]
            if not lines:
                continue

            seen_on_page = set()
            # Look at first 3 and last 3 non-empty lines
            for line in lines[:3] + lines[-3:]:
                seen_on_page.add(line)

            for line in seen_on_page:
                edge_line_counter[line] += 1

        # A line is a running header if it appears in >threshold of pages
        # AND is relatively short (headers are typically short)
        min_appearances = max(2, int(len(pages) * threshold))
        headers = set()
        for line, count in edge_line_counter.items():
            if count >= min_appearances and len(line) < 50:
                # Don't strip lines that look like actual body text
                # (long sentences with Tamil content)
                if len(line) > 30 and _has_tamil(line):
                    continue
                headers.add(line)

        return headers

    @staticmethod
    def strip_running_headers(
        pages: List[str],
        headers: Set[str],
        page_metadata: Optional[List[dict]] = None,
    ) -> List[str]:
        """
        Remove identified running headers from the first 3 lines of each page.

        CRITICAL: Page 1 (index 0) is NEVER stripped — it contains the story/chapter
        title which is indistinguishable from a running header by frequency alone.
        Stripping it would silently delete the opening of the document.

        If page_metadata is provided, extracted page numbers and running headers
        are stored there for later use in page markers.
        """
        if not headers:
            return pages

        result = []
        for page_idx, page_text in enumerate(pages):
            # Never strip from the very first page — it contains the story title
            if page_idx == 0:
                result.append(page_text)
                continue

            lines = page_text.split('\n')
            
            # Find non-empty line indices
            non_empty_indices = [i for i, line in enumerate(lines) if line.strip()]
            top_indices = set(non_empty_indices[:4])
            bottom_indices = set(non_empty_indices[-4:])
            edge_indices = top_indices | bottom_indices

            cleaned = []
            stripped_meta = []
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Strip if it's an edge line and identified as a header/footer
                if i in edge_indices and stripped in headers:
                    if page_metadata is not None:
                        stripped_meta.append(stripped)
                    continue  # Skip running header/footer
                cleaned.append(line)

            if page_metadata is not None and page_idx < len(page_metadata):
                page_metadata[page_idx]['stripped_headers'] = stripped_meta

            result.append('\n'.join(cleaned))
        return result

    @staticmethod
    def merge_short_lines(text: str, min_length: int = 30) -> str:
        """
        Merge fragmented short OCR lines into proper paragraphs.

        Surya OCR sometimes fragments a single logical line into multiple
        short physical lines (e.g., when the book has narrow columns or
        justified text with large gaps). This function recombines them.

        Rules:
          - If a line is shorter than min_length characters AND
          - It does NOT end with sentence-ending punctuation AND
          - The next line exists and is not empty
          → Merge the two lines with a space.

        We DON'T merge:
          - Lines that end with '.', '!', '?', '"' (sentence boundary)
          - Lines followed by empty lines (paragraph boundary)
          - Lines that are dialogue (start with '' or ")
        """
        lines = text.split('\n')
        if len(lines) <= 1:
            return text

        merged = []
        i = 0
        while i < len(lines):
            current = lines[i]
            stripped = current.strip()

            # Empty line = paragraph separator, keep as-is
            if not stripped:
                merged.append(current)
                i += 1
                continue

            # Try to merge short lines with the next line
            while (i + 1 < len(lines) and
                   len(stripped) < min_length and
                   not _ends_with_sentence_ender(stripped) and
                   lines[i + 1].strip()):  # next line is not empty

                next_stripped = lines[i + 1].strip()

                # Don't merge if next line starts with any dialogue/quote marker
                # Covers both straight quotes and curly Tamil typography quotes
                dialogue_starters = ("'", '"', "\u2018", "\u2019", '\u201c', '\u201d',
                                     '\u2018\u2018', "''", '\xab', '\u2014')
                if next_stripped and (next_stripped[:2] in dialogue_starters or
                                      next_stripped[0] in dialogue_starters):
                    break

                # Don't merge if current line ENDS with a closing dialogue quote
                # ''...'' is a completed speaker turn — next line is a new turn
                if stripped.endswith("''") or stripped.endswith('\u2019\u2019') or \
                   stripped.endswith('"') or stripped.endswith('\u201d'):
                    break

                # Don't merge if current line looks like a title/header
                # (very short, all-caps, or standalone Tamil word)
                if len(stripped) < 8 and not _has_tamil(next_stripped[:1] if next_stripped else ''):
                    break

                # Merge
                stripped = stripped + ' ' + next_stripped
                i += 1

            merged.append(stripped)
            i += 1

        return '\n'.join(merged)

    @staticmethod
    def clean_empty_lines(text: str) -> str:
        """
        Normalize paragraph spacing:
        - Collapse 3+ consecutive blank lines into 1 blank line
        - Ensure no trailing whitespace on lines
        """
        lines = text.split('\n')
        cleaned = []
        blank_run = 0
        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                blank_run += 1
                if blank_run <= 1:
                    cleaned.append('')
            else:
                blank_run = 0
                cleaned.append(stripped)

        # Remove leading/trailing blank lines
        while cleaned and not cleaned[0]:
            cleaned.pop(0)
        while cleaned and not cleaned[-1]:
            cleaned.pop()

        return '\n'.join(cleaned)

    @classmethod
    def format_document(cls, pages: List[str], pdf_name: str = "") -> str:
        """
        Apply the full professional formatting pipeline to a list of
        per-page texts. This is the main entry point.

        Output structure mirrors Phase 1 Gardio format:
          - Document header block (PDF name + page count)
          - Per-page section markers with page number as metadata label
          - Clean body text with running headers and page numbers removed

        Args:
            pages: List of strings, one per page (from OCR + layout reconstruction).
            pdf_name: Optional PDF filename (used for running header detection).

        Returns:
            A single clean string — the full document text.
        """
        if not pages:
            return ""

        total_pages = len(pages)
        doc_name = pdf_name.replace('.pdf', '').replace('.PDF', '').strip() if pdf_name else 'Document'

        # Step 0: Strip inline artifacts (<math> tags, URLs, garbage lines)
        pages = cls.strip_artifacts(pages)

        # Step 1: Detect page numbers and running headers BEFORE stripping,
        # so we can use them as metadata labels in the page markers.
        # We collect per-page metadata (detected page number, detected headers).
        page_metadata = cls._extract_page_metadata(pages)

        # Step 2: Strip standalone page numbers from text
        pages = cls.strip_page_numbers(pages)

        # Step 3: Detect running headers for stripping
        headers = cls.detect_running_headers(pages)
        if pdf_name:
            stem = doc_name
            if stem:
                headers.add(stem)
        # strip_running_headers preserves page 0 (story title protection)
        pages = cls.strip_running_headers(pages, headers)

        # Step 4: Merge short lines within each page
        pages = [cls.merge_short_lines(page) for page in pages]

        # Step 5: Build professional document output
        DIVIDER = '=' * 60
        parts = []

        # Document-level header
        parts.append(DIVIDER)
        parts.append(f'Tamil OCR Output \u2014 {pdf_name if pdf_name else doc_name}')
        parts.append(f'Pages processed: {total_pages}')
        parts.append(DIVIDER)

        for idx, page_text in enumerate(pages):
            if not page_text.strip():
                continue

            meta = page_metadata[idx] if idx < len(page_metadata) else {}
            page_num_label = meta.get('page_number', '')
            header_label   = meta.get('running_header', '')

            # Build the page section marker
            parts.append('')
            parts.append(DIVIDER)
            page_label = f'PAGE {idx + 1}'
            if page_num_label:
                page_label += f'  [{page_num_label}]'
            parts.append(page_label)
            parts.append(DIVIDER)

            parts.append(page_text.strip())

        full_text = '\n'.join(parts)

        # Step 6: Final cleanup — normalize empty lines
        full_text = cls.clean_empty_lines(full_text)

        return full_text

    @staticmethod
    def _extract_page_metadata(pages: List[str]) -> List[dict]:
        """
        Before stripping, scan the first few lines of each page to capture:
          - 'page_number': a detected numeric or roman-numeral page number
          - 'running_header': the first non-numeric short line (author name / title)
        These are used as metadata labels in the page section markers.
        """
        metadata = []
        for page_text in pages:
            lines = [l.strip() for l in page_text.split('\n') if l.strip()]
            meta = {'page_number': '', 'running_header': ''}
            
            # Search top 5 and bottom 5 lines (reversed so extreme bottom is checked first)
            search_lines = lines[:5] + list(reversed(lines[-5:])) if len(lines) > 5 else lines
            
            for line in search_lines:
                if not meta['page_number'] and _is_page_number(line):
                    meta['page_number'] = line
                elif not meta['running_header'] and len(line) < 50 and _has_tamil(line):
                    # Short Tamil line after/before page number = running header
                    meta['running_header'] = line
                if meta['page_number'] and meta['running_header']:
                    break
            metadata.append(meta)
        return metadata

    @staticmethod
    def strip_artifacts(pages: List[str]) -> List[str]:
        """
        Remove inline OCR artifacts from each page:
          - <math> ... </math> tag blocks (Surya hallucinates these from decorative elements)
          - URL watermarks (www.padippakam.com, etc.)
          - Garbage lines (standalone punctuation, Cyrillic residue, etc.)
        """
        result = []
        for page_text in pages:
            # Strip <math> tags inline
            page_text = MATH_TAG_RE.sub('', page_text)
            # Strip inline URLs
            page_text = URL_RE.sub('', page_text)

            lines = page_text.split('\n')
            cleaned = []
            for line in lines:
                stripped = line.strip()
                # Skip watermark lines
                if WATERMARK_RE.match(stripped):
                    continue
                # Skip pure garbage lines
                if GARBAGE_LINE_RE.match(stripped):
                    continue
                # Skip lines that are only <br> tags or residual HTML
                if re.match(r'^\s*(<br\s*/?>\s*)*$', stripped, re.IGNORECASE):
                    continue
                cleaned.append(line)
            result.append('\n'.join(cleaned))
        return result


# ── CLI test ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # Simulate 3 pages of OCR output with typical artifacts
    test_pages = [
        "### இருந்தது. ஏனோ தினமும் அதைப் போட்டுக் கொள்ள\nமுடிவதில்லை.",
        "37\nஅசோகமித்திரன்\nமுடியுமோ அப்படியெல்லாம் செய்வேன்.\nஅப்போது\nஸ் டூடியோ\nகாண்டீனையும்\nநடத்திவந்தார்.",
        "38\nGRAT GRATITION.\nஅசோகமித்திரன்\nடாக்ஸி வேகமாகப் போய்க்கொண்டேயிருக்க.\nநிறைய வீடுகள் வந்தன.",
    ]

    formatter = TextFormatter()

    print("=== Input Pages ===")
    for i, p in enumerate(test_pages):
        print(f"\n--- Page {i+1} ---")
        print(p)

    result = formatter.format_document(test_pages, pdf_name="கண்ணாடி.pdf")
    print("\n\n=== Formatted Output ===")
    print(result)


