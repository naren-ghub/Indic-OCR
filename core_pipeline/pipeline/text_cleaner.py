"""
Post-OCR Text Cleaner for Tamil OCR Pipeline.

Identifies and fixes the following artifact classes observed in Surya OCR output
on Tamil scanned books:

  1. HTML markup tags  (<b>, <i>, <u> and closing variants)
  2. Foreign script hallucinations  (Malayalam, Devanagari, Thai, etc. characters
     that appear when Surya misreads decorative glyphs/page-number ornaments)
  3. Inline page numbers  (leading "30", "38", "101 AuthorName" patterns)
  4. Fully-garbage lines  (lines with zero Tamil content after stripping the above)

Strategy:
  - Rule-based Unicode-range filtering (fast, no model needed)
  - Non-destructive: logs every change so evaluation can track what was cleaned
  - Applied as a post-processing step in both the batch pipeline and Gradio app
"""

import re
import unicodedata
from typing import Tuple

# ── Unicode ranges ─────────────────────────────────────────────────────────

# Tamil Unicode block (the ONLY Indic script we expect here)
TAMIL_RANGE = (0x0B80, 0x0BFF)

# Scripts that should NOT appear in Tamil-only text but Surya sometimes
# hallucinates when it encounters decorative/drop-cap glyphs or page numbers
FOREIGN_SCRIPT_RANGES = [
    (0x0900, 0x097F),  # Devanagari  (Hindi, Sanskrit)
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0E00, 0x0E7F),  # Thai
    (0x0E80, 0x0EFF),  # Lao
    (0x0F00, 0x0FFF),  # Tibetan
    (0x1000, 0x109F),  # Myanmar
    (0x0600, 0x06FF),  # Arabic
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs (Chinese/Japanese/Korean)
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Odia
]

# HTML tag patterns Surya emits for bold/italic detected text
HTML_TAG_RE = re.compile(r'</?(?:b|i|u|mark|sub|sup|em|strong|span)\b[^>]*>', re.IGNORECASE)

# Page-number patterns observed in the dataset:
#   "30 " at start of text  (page num + space before story text)
#   "101 AuthorName" (page num + author header)
#   Standalone "38" on its own line
PAGE_NUM_LINE_RE = re.compile(
    r'^(\d{1,3})\s+([A-Z][a-zA-Z\s]{0,25})\s*$'  # "101 Asokamidhiran"
)
PAGE_NUM_INLINE_RE = re.compile(
    r'^(\d{1,3})\s+'  # "30 " at the very start of a line
)

# Minimum Tamil character ratio for a line to be considered "valid"
MIN_TAMIL_CHAR_RATIO = 0.15


# ── Helpers ────────────────────────────────────────────────────────────────

def _has_tamil(text: str) -> bool:
    return any(TAMIL_RANGE[0] <= ord(c) <= TAMIL_RANGE[1] for c in text)


def _has_english(text: str) -> bool:
    return any(0x0041 <= ord(c) <= 0x005A or 0x0061 <= ord(c) <= 0x007A for c in text)


def _count_tamil(text: str) -> int:
    return sum(1 for c in text if TAMIL_RANGE[0] <= ord(c) <= TAMIL_RANGE[1])


def _is_foreign_script_char(c: str) -> bool:
    cp = ord(c)
    return any(lo <= cp <= hi for lo, hi in FOREIGN_SCRIPT_RANGES)


def _strip_html_tags(line: str) -> Tuple[str, bool]:
    cleaned = HTML_TAG_RE.sub('', line)
    return cleaned, cleaned != line


def _strip_foreign_chars(line: str) -> Tuple[str, bool]:
    """Remove individual characters from foreign scripts."""
    cleaned = ''.join(c for c in line if not _is_foreign_script_char(c))
    # Collapse multiple spaces that might have been created
    cleaned = re.sub(r'  +', ' ', cleaned).strip()
    return cleaned, cleaned != line


def _strip_inline_page_number(line: str) -> Tuple[str, bool]:
    """
    Remove leading page-number prefixes like '30 ' or '101 AuthorName '.
    Only strips if Tamil or English text follows, so we don't accidentally eat content.
    """
    m = PAGE_NUM_INLINE_RE.match(line)
    if m:
        remainder = line[m.end():]
        if _has_tamil(remainder) or _has_english(remainder):
            return remainder.strip(), True
    return line, False


def _is_garbage_line(line: str) -> bool:
    """
    A line is garbage if it has no Tamil or English characters AND
    consists mostly of ASCII noise or foreign script.
    We keep lines that have ANY Tamil or English in them.
    """
    line = line.strip()
    if not line:
        return False  # empty lines are kept as paragraph separators
    if _has_tamil(line) or _has_english(line):
        return False  # Has valid text — keep it
    # No Tamil or English. Check if it's a page-number-only line or pure foreign script
    # Allow short pure-ASCII lines like numbers, quotation marks, etc. only
    # if they're very short (likely punctuation residue)
    if len(line) <= 2:
        return False  # e.g. "''", ".", "—"
    # Longer lines with no valid text → garbage
    return True


# ── Main clean function ────────────────────────────────────────────────────

def clean_line(line: str, log: list | None = None) -> str | None:
    """
    Clean a single line. Returns the cleaned string, or None if the
    line should be dropped entirely.

    Args:
        line: Raw OCR output line.
        log:  Optional list to append change records to.

    Returns:
        Cleaned line string, or None to drop the line.
    """
    original = line

    # 1. Strip HTML tags
    line, changed = _strip_html_tags(line)
    if changed and log is not None:
        log.append(('HTML_TAG', original, line))

    # 2. Strip foreign-script characters
    line, changed = _strip_foreign_chars(line)
    if changed and log is not None:
        log.append(('FOREIGN_CHAR', original, line))

    # 3. Strip leading page numbers (only if Tamil follows)
    line, changed = _strip_inline_page_number(line)
    if changed and log is not None:
        log.append(('PAGE_NUM', original, line))

    # 4. Drop fully-garbage lines (no Tamil, too long to be punctuation)
    if _is_garbage_line(line):
        if log is not None:
            log.append(('DROP', original, None))
        return None

    return line.strip() or None


def clean_text(text: str, log: list | None = None) -> str:
    """
    Clean a full block of OCR text (multi-line string).
    Returns the cleaned text.
    """
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        result = clean_line(line, log=log)
        if result is None:
            # Dropped line — don't add a blank line in its place
            # (avoids double-blank-line gaps)
            if cleaned_lines and cleaned_lines[-1] != '':
                pass  # just skip
        else:
            cleaned_lines.append(result)

    # Collapse more than 2 consecutive blank lines to at most 1
    final = []
    blank_run = 0
    for line in cleaned_lines:
        if line == '':
            blank_run += 1
            if blank_run <= 1:
                final.append(line)
        else:
            blank_run = 0
            final.append(line)

    return '\n'.join(final).strip()


# ── Standalone test ────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    samples = [
        '<i>ดู</i>ใช่ท<i>ี่ย</i>สเบเก',          # Thai hallucination (title area)
        '30 எலி நேரம் எல்லாம் அமைதியாக',           # inline page number
        '<b>கண்ண</b>ரடி டாக்ஸி',                   # HTML bold
        '101 அசோகமித்திரன் போனார்கள்.',             # page + author header
        '38',                                         # standalone page number
        '55 GRAT GRATITION.',                        # garbage line
        '<i>ராணுவ</i>',                              # HTML italic
        "''त का का",                                # Devanagari hallucination
        '<b>२</b>(/5',                               # Devanagari + HTML
        'ത്തി இரண்டாவது நாளாக',                      # Malayalam leading chars
        'அந்தப் பூதாகாரமான, பல மாடிக்கட்டிடத்தின்', # Normal Tamil — should pass unchanged
    ]

    print('=== Cleaner Test ===\n')
    log = []
    for s in samples:
        result = clean_line(s, log=log)
        status = 'DROP' if result is None else ('CLEAN' if result != s else 'OK')
        print(f'[{status}]')
        print(f'  IN : {repr(s)}')
        print(f'  OUT: {repr(result)}')
        print()
