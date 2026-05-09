#!/usr/bin/env python3
"""
corpus_cleaner.py — Source-Aware Tamil OCR Corpus Cleaner
=========================================================

Cleans each corpus source independently with tailored rules:
  • IndicCorp v2  → strip bracketed English, list markers, English-heavy lines
  • Project Madurai → HTML decode, split merged stanzas, strip numbers, drop commentary
  • Pre-Modern Tamil → (future) page numbers, headers, keep Sanskrit loanwords

Usage:
    python data/corpus_cleaner.py                    # Clean all sources
    python data/corpus_cleaner.py --report           # Clean + print detailed stats
    python data/corpus_cleaner.py --source indiccorp # Clean only IndicCorp
"""

import re
import html
import random
import argparse
import sys
import io
from pathlib import Path
from collections import Counter

# Fix Windows console encoding for Tamil output
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─── Paths ───────────────────────────────────────────────────────────────────
CORPUS_DIR = Path(__file__).parent / "corpus"
INDICCORP_PATH = CORPUS_DIR / "indiccorp.txt"
PROJECT_MADURAI_PATH = CORPUS_DIR / "project_madurai.txt"
PREMODERN_PATH = CORPUS_DIR / "premodern_tamil.txt"
CLEANED_IC_PATH = CORPUS_DIR / "cleaned_indiccorp.txt"
CLEANED_PM_PATH = CORPUS_DIR / "cleaned_project_madurai.txt"
CLEANED_PRE_PATH = CORPUS_DIR / "cleaned_premodern.txt"
CLEANED_COMBINED_PATH = CORPUS_DIR / "cleaned_combined.txt"


# ─── Shared Helpers ──────────────────────────────────────────────────────────

def _count_tamil_chars(text: str) -> int:
    """Count Tamil Unicode characters (U+0B80–U+0BFF)."""
    return sum(1 for ch in text if '\u0B80' <= ch <= '\u0BFF')


def _count_latin_chars(text: str) -> int:
    """Count Latin/English characters."""
    return sum(1 for ch in text if ch.isascii() and ch.isalpha())


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces to single, strip edges."""
    return re.sub(r' {2,}', ' ', text).strip()


# ─── IndicCorp v2 Cleaner ────────────────────────────────────────────────────

# Regex: bracketed English like (Anseriforms), ("Adam's Bridge"), (classical mechanics)
_RE_BRACKET_ENGLISH = re.compile(r'\(["\']?[a-zA-Z][a-zA-Z\s\-\',\.]{2,}["\']?\)')

# Regex: leading list markers like "8.புகை", "3) தமிழ்", "12- கிராமம்"
_RE_LIST_MARKER = re.compile(r'^\s*\d+[\.\-\)]\s*')

# Regex: lines that are mostly stray web/JS/HTML junk
_RE_WEB_JUNK = re.compile(r'(http[s]?://|www\.|\.com|\.org|javascript|function\(|var\s|<script)')


def clean_indiccorp(input_path: Path = INDICCORP_PATH,
                    output_path: Path = CLEANED_IC_PATH,
                    report: bool = False) -> dict:
    """
    Clean IndicCorp v2 Tamil corpus.

    Rules:
        1. html.unescape() any HTML entities
        2. Strip bracketed English inline — keep surrounding Tamil
        3. Strip leading list markers (1. 2. 3- etc.)
        4. Drop lines >30% Latin characters
        5. Drop web/JS junk lines
        6. Normalize whitespace
        7. Drop lines < 15 Tamil characters
    """
    if not input_path.exists():
        print(f"  ⚠️  {input_path.name} not found — skipping IndicCorp cleaning.")
        return {"source": "indiccorp", "input": 0, "output": 0, "dropped": {}}

    with open(input_path, "r", encoding="utf-8") as f:
        raw_lines = [l.strip() for l in f if l.strip()]

    stats = Counter()
    cleaned = []

    for line in raw_lines:
        original = line

        # Step 1: HTML unescape
        line = html.unescape(line)

        # Step 1b: Strip any stray HTML tags (<br>, <poem>, <mapframe>, etc.)
        stripped = _RE_HTML_TAG.sub('', line)
        if stripped != line:
            stats["html_tags_stripped"] += 1
            line = stripped

        # Step 2: Strip bracketed English
        stripped = _RE_BRACKET_ENGLISH.sub('', line)
        if stripped != line:
            stats["bracket_english_stripped"] += 1
            line = stripped

        # Step 3: Strip leading list markers
        stripped = _RE_LIST_MARKER.sub('', line)
        if stripped != line:
            stats["list_markers_stripped"] += 1
            line = stripped

        # Step 4: Drop English-heavy lines (>30% Latin)
        total_chars = len(line)
        if total_chars > 0 and _count_latin_chars(line) / total_chars > 0.3:
            stats["english_heavy_dropped"] += 1
            continue

        # Step 5: Drop web junk
        if _RE_WEB_JUNK.search(line):
            stats["web_junk_dropped"] += 1
            continue

        # Step 6: Normalize whitespace
        line = _normalize_whitespace(line)

        # Step 7: Quality gate — minimum Tamil content
        if _count_tamil_chars(line) < 15:
            stats["too_short_dropped"] += 1
            continue

        cleaned.append(line)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned))

    result = {
        "source": "indiccorp",
        "input": len(raw_lines),
        "output": len(cleaned),
        "dropped": dict(stats),
    }

    if report:
        _print_report(result)

    return result


# ─── Project Madurai Cleaner ─────────────────────────────────────────────────

# Regex: section headers like "1.2.3 புதல்வரைப் பெறுதல்", "அறத்துப்பால் 1.1 கடவுள் வாழ்த்து"
_RE_SECTION_HEADER = re.compile(r'^\d+\.\d+(\.\d+)?\s+')

# Regex: chapter end markers
_RE_CHAPTER_END = re.compile(
    r'(முற்றிற்று|இல்லறவியல்\s+முற்றிற்று|அரசியல்\s+முற்றிற்று|'
    r'அமைச்சியல்\s+முற்றிற்று|துறவறவியல்\s+முற்றிற்று|'
    r'ஊழியல்\s+முற்றிற்று|அறத்துப்பால்\s+முற்றிற்று|'
    r'பொருட்பால்\s+|காமத்துப்பால்\s+|அங்கவியல்\s+முற்றிற்று)',
    re.IGNORECASE,
)

# Regex: modern editorial markers at start of line
_RE_EDITORIAL = re.compile(
    r'^\s*(பொருள்\s*:|குறிப்பு\s*:|விளக்கம்\s*:|முன்னுரை|பின்னுரை|'
    r'ஆசிரியர்\s+குறிப்பு|ஆசிரியர்\s*:)',
    re.IGNORECASE,
)

# Regex: standalone stanza numbers (leading)
_RE_LEADING_NUM = re.compile(r'^\s*\d{1,4}\s+')

# Regex: trailing stanza numbers
_RE_TRAILING_NUM = re.compile(r'\s+\d{1,4}\s*$')

# Regex: multi-space gap indicating merged stanzas (3+ spaces followed by a digit)
_RE_STANZA_SPLIT = re.compile(r'\s{3,}(\d{1,4}\s+)')

# Regex: HTML <font> or other stray tags
_RE_HTML_TAG = re.compile(r'<[^>]+>')


def clean_project_madurai(input_path: Path = PROJECT_MADURAI_PATH,
                          output_path: Path = CLEANED_PM_PATH,
                          report: bool = False) -> dict:
    """
    Clean Project Madurai classical Tamil corpus.

    Rules:
        1. Decode HTML entities (&nbsp;, &amp;, etc.)
        2. Strip HTML tags (<font ...>)
        3. Split merged multi-stanza lines on whitespace gaps
        4. Strip leading/trailing stanza numbers
        5. Drop section headers (1.2.3 format)
        6. Drop chapter end markers ("முற்றிற்று")
        7. Drop modern editorial commentary
        8. Normalize whitespace
        9. Drop lines < 10 Tamil characters
    """
    if not input_path.exists():
        print(f"  ⚠️  {input_path.name} not found — skipping Project Madurai cleaning.")
        return {"source": "project_madurai", "input": 0, "output": 0, "dropped": {}}

    with open(input_path, "r", encoding="utf-8") as f:
        raw_lines = [l.strip() for l in f if l.strip()]

    stats = Counter()
    cleaned = []

    for line in raw_lines:
        # Step 1: HTML entity decode
        line = html.unescape(line)
        if '&nbsp;' in line or '&amp;' in line:
            stats["html_decoded"] += 1

        # Step 2: Strip HTML tags
        stripped = _RE_HTML_TAG.sub('', line)
        if stripped != line:
            stats["html_tags_stripped"] += 1
            line = stripped

        # Step 3: Split merged multi-stanza lines
        # Pattern: "...first stanza text.   863 second stanza text..."
        fragments = _RE_STANZA_SPLIT.split(line)
        if len(fragments) > 1:
            stats["stanzas_split"] += 1
            # Reassemble: fragments alternate between text and captured number groups
            # fragments = [text1, num1, text2, num2, text3, ...]
            sub_lines = []
            for i in range(0, len(fragments), 2):
                part = fragments[i].strip()
                if part:
                    sub_lines.append(part)
        else:
            sub_lines = [line]

        for sub_line in sub_lines:
            line = sub_line

            # Step 4: Strip leading stanza numbers
            stripped = _RE_LEADING_NUM.sub('', line)
            if stripped != line:
                stats["leading_nums_stripped"] += 1
                line = stripped

            # Strip trailing stanza numbers
            stripped = _RE_TRAILING_NUM.sub('', line)
            if stripped != line:
                stats["trailing_nums_stripped"] += 1
                line = stripped

            # Step 5: Drop section headers
            if _RE_SECTION_HEADER.match(line):
                stats["section_headers_dropped"] += 1
                continue

            # Also drop lines that are pure section title text (short, no punctuation)
            # e.g., "அறத்துப்பால் 1.1 கடவுள் வாழ்த்து"
            if re.match(r'^[\u0B80-\u0BFF\s]+\d+\.\d+', line):
                stats["section_headers_dropped"] += 1
                continue

            # Step 6: Drop chapter end markers
            if _RE_CHAPTER_END.search(line) and len(line) < 60:
                stats["chapter_ends_dropped"] += 1
                continue

            # Step 7: Drop modern editorial commentary
            if _RE_EDITORIAL.match(line):
                stats["editorial_dropped"] += 1
                continue

            # Step 8: Normalize whitespace
            line = _normalize_whitespace(line)

            # Step 9: Quality gate — minimum Tamil content
            if _count_tamil_chars(line) < 10:
                stats["too_short_dropped"] += 1
                continue

            cleaned.append(line)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned))

    result = {
        "source": "project_madurai",
        "input": len(raw_lines),
        "output": len(cleaned),
        "dropped": dict(stats),
    }

    if report:
        _print_report(result)

    return result


# ─── Pre-Modern Tamil Cleaner ────────────────────────────────────────────────

# Regex: standalone page numbers
_RE_PAGE_NUMBER = re.compile(r'^\s*\d{1,4}\s*$')

# Regex: footnote markers (superscript-like numbers at end)
_RE_FOOTNOTE = re.compile(r'\s*[\*†‡§]\d*\s*$')


def clean_premodern(input_path: Path = PREMODERN_PATH,
                    output_path: Path = CLEANED_PRE_PATH,
                    report: bool = False) -> dict:
    """
    Clean Pre-Modern Tamil (1800–1950) corpus.

    Rules:
        1. Drop standalone page numbers
        2. Drop short header-like lines (< 25 chars, no sentence punctuation)
        3. Keep Sanskrit loanwords INTACT — do NOT touch ஜ, ஷ, ஸ, ஹ
        4. Strip footnote markers
        5. Normalize whitespace
        6. Drop lines < 15 Tamil characters
    """
    if not input_path.exists():
        print(f"  ℹ️  {input_path.name} not found — skipping Pre-Modern cleaning.")
        print(f"      (This source will be available after Tamil Wikisource scraping.)")
        return {"source": "premodern", "input": 0, "output": 0, "dropped": {}}

    with open(input_path, "r", encoding="utf-8") as f:
        raw_lines = [l.strip() for l in f if l.strip()]

    stats = Counter()
    cleaned = []

    for line in raw_lines:
        # Step 1: Drop standalone page numbers
        if _RE_PAGE_NUMBER.match(line):
            stats["page_numbers_dropped"] += 1
            continue

        # Step 2: Drop short header-like lines
        if len(line) < 25 and not any(p in line for p in '.!?,;:'):
            stats["headers_dropped"] += 1
            continue

        # Step 3: Sanskrit loanwords — KEEP (no action needed)

        # Step 4: Strip footnote markers
        stripped = _RE_FOOTNOTE.sub('', line)
        if stripped != line:
            stats["footnotes_stripped"] += 1
            line = stripped

        # Step 5: Normalize whitespace
        line = _normalize_whitespace(line)

        # Step 6: Quality gate
        if _count_tamil_chars(line) < 15:
            stats["too_short_dropped"] += 1
            continue

        cleaned.append(line)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned))

    result = {
        "source": "premodern",
        "input": len(raw_lines),
        "output": len(cleaned),
        "dropped": dict(stats),
    }

    if report:
        _print_report(result)

    return result


# ─── Combiner ────────────────────────────────────────────────────────────────

def combine_cleaned(output_path: Path = CLEANED_COMBINED_PATH,
                    shuffle: bool = True,
                    report: bool = False) -> dict:
    """
    Merge all cleaned source files → deduplicate → shuffle → save.
    """
    all_lines = []
    source_counts = {}

    for name, path in [
        ("indiccorp", CLEANED_IC_PATH),
        ("project_madurai", CLEANED_PM_PATH),
        ("premodern", CLEANED_PRE_PATH),
    ]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            source_counts[name] = len(lines)
            all_lines.extend(lines)
        else:
            source_counts[name] = 0

    # Deduplicate
    before_dedup = len(all_lines)
    unique_lines = list(dict.fromkeys(all_lines))  # preserves order, removes dupes
    dupes_removed = before_dedup - len(unique_lines)

    # Shuffle
    if shuffle:
        random.seed(42)
        random.shuffle(unique_lines)

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(unique_lines))

    result = {
        "source": "combined",
        "sources": source_counts,
        "total_before_dedup": before_dedup,
        "duplicates_removed": dupes_removed,
        "output": len(unique_lines),
    }

    if report:
        print("\n" + "=" * 60)
        print("COMBINED CORPUS")
        print("=" * 60)
        for src, count in source_counts.items():
            print(f"  {src}: {count:,} lines")
        print(f"  Total before dedup: {before_dedup:,}")
        print(f"  Duplicates removed: {dupes_removed:,}")
        print(f"  Final output: {len(unique_lines):,} lines")
        print(f"  Saved to: {output_path}")

    return result


# ─── Reporting ───────────────────────────────────────────────────────────────

def _print_report(result: dict):
    """Pretty-print cleaning results."""
    print("\n" + "=" * 60)
    print(f"  {result['source'].upper()}")
    print("=" * 60)
    print(f"  Input lines:  {result['input']:,}")
    print(f"  Output lines: {result['output']:,}")
    dropped_total = result['input'] - result['output']
    print(f"  Dropped:      {dropped_total:,} ({dropped_total/max(result['input'],1)*100:.1f}%)")

    if result.get("dropped"):
        print("\n  --- Drop/Transform Breakdown ---")
        for key, count in sorted(result["dropped"].items(), key=lambda x: -x[1]):
            print(f"    {key}: {count:,}")


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_output(path: Path = CLEANED_COMBINED_PATH) -> bool:
    """Quick validation: check for common contaminants in the final output."""
    if not path.exists():
        print(f"❌ {path} not found!")
        return False

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    issues = []

    # Check for HTML entities
    html_count = text.count('&nbsp;') + text.count('&amp;') + text.count('&quot;')
    if html_count > 0:
        issues.append(f"HTML entities remaining: {html_count}")

    # Check for wiki markup
    wiki_count = text.count('{{') + text.count('}}')
    if wiki_count > 0:
        issues.append(f"Wiki markup remaining: {wiki_count}")

    # Check for HTML tags
    tag_count = len(re.findall(r'<[a-z]+[^>]*>', text, re.IGNORECASE))
    if tag_count > 0:
        issues.append(f"HTML tags remaining: {tag_count}")

    if issues:
        print("❌ Validation FAILED:")
        for issue in issues:
            print(f"   • {issue}")
        return False

    lines = text.strip().split('\n')
    print(f"✅ Validation PASSED: {len(lines):,} clean lines, 0 HTML/wiki artifacts.")
    return True


# ─── Entry Point ─────────────────────────────────────────────────────────────

def clean_all(report: bool = True) -> dict:
    """Run all cleaning pipelines and combine."""
    print("🧹 Tamil OCR Corpus Cleaner")
    print("=" * 60)

    results = {}

    # Clean each source
    print("\n[1/3] Cleaning IndicCorp v2...")
    results["indiccorp"] = clean_indiccorp(report=report)

    print("\n[2/3] Cleaning Project Madurai...")
    results["project_madurai"] = clean_project_madurai(report=report)

    print("\n[3/3] Cleaning Pre-Modern Tamil...")
    results["premodern"] = clean_premodern(report=report)

    # Combine
    print("\n[4/4] Combining all cleaned sources...")
    results["combined"] = combine_cleaned(report=report)

    # Validate
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    validate_output()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Tamil OCR corpus")
    parser.add_argument("--report", action="store_true", help="Print detailed stats")
    parser.add_argument(
        "--source",
        choices=["indiccorp", "project_madurai", "premodern", "all"],
        default="all",
        help="Which source to clean (default: all)"
    )
    args = parser.parse_args()

    if args.source == "all":
        clean_all(report=args.report or True)
    elif args.source == "indiccorp":
        clean_indiccorp(report=True)
    elif args.source == "project_madurai":
        clean_project_madurai(report=True)
    elif args.source == "premodern":
        clean_premodern(report=True)
