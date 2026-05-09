#!/usr/bin/env python3
"""
build_training_data.py — Balanced ByT5 Training Data Builder
=============================================================
Takes cleaned corpus files and produces a balanced, length-filtered
train.jsonl for ByT5 OCR correction fine-tuning.

Pipeline:
  1. Load cleaned sources (IndicCorp, Project Madurai, Pre-Modern)
  2. Concatenate short PM stanzas into 80–150 char groups
  3. Apply length filters (drop <20 chars, split >340 chars)
  4. Balance registers via oversampling
  5. Generate noisy→clean pairs (simulate OCR errors)
  6. Output train.jsonl

Usage:
    python data/build_training_data.py
    python data/build_training_data.py --report
"""

import re
import sys
import io
import json
import random
import statistics
import argparse
from pathlib import Path
from collections import Counter
import Levenshtein

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─── Paths ───────────────────────────────────────────────────────────────────
CORPUS_DIR = Path(__file__).parent / "corpus"
CLEANED_IC_PATH = CORPUS_DIR / "cleaned_indiccorp.txt"
CLEANED_PM_PATH = CORPUS_DIR / "cleaned_project_madurai.txt"
CLEANED_PRE_PATH = CORPUS_DIR / "cleaned_premodern.txt"
OUTPUT_JSONL = CORPUS_DIR / "train.jsonl"
OUTPUT_STATS = CORPUS_DIR / "training_data_stats.json"

# ─── ByT5 Constraints ───────────────────────────────────────────────────────
MIN_CHARS = 20       # Minimum character length
MAX_CHARS = 340      # Maximum (~1024 bytes for Tamil in UTF-8)
SWEET_SPOT_MIN = 50  # Ideal minimum
SWEET_SPOT_MAX = 200 # Ideal maximum

# ─── Register Balance Targets ───────────────────────────────────────────────
TARGET_MODERN_PCT = 0.40      # IndicCorp
TARGET_CLASSICAL_PCT = 0.33   # Project Madurai
TARGET_PREMODERN_PCT = 0.27   # Pre-Modern Tamil

# ─── Tamil OCR Error Simulation ──────────────────────────────────────────────
# Common OCR confusion pairs for Tamil script
# Format: (correct_char, [possible_OCR_errors])
TAMIL_OCR_CONFUSIONS = {
    # Visually similar consonants
    'க': ['ச', 'வ'],
    'ச': ['க', 'ஜ'],
    'ட': ['ண'],
    'ண': ['ட'],
    'த': ['ந'],
    'ந': ['த', 'ர'],
    'ப': ['வ'],
    'ம': ['ன'],
    'ன': ['ம', 'ண'],
    'ர': ['ற'],
    'ற': ['ர'],
    'ல': ['வ', 'ன'],
    'வ': ['ப', 'ல'],
    'ழ': ['ள'],
    'ள': ['ழ', 'ன'],
    # Vowel signs
    'ா': ['ி'],
    'ி': ['ா', 'ீ'],
    'ீ': ['ி'],
    'ு': ['ூ'],
    'ூ': ['ு'],
    'ே': ['ை', 'ொ'],
    'ை': ['ே'],
    'ொ': ['ோ'],
    'ோ': ['ொ'],
}


def load_lines(path: Path) -> list[str]:
    """Load non-empty lines from a text file."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def count_tamil(text: str) -> int:
    return sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')


# ─── Step 1: Stanza Concatenation (Project Madurai) ─────────────────────────

def concatenate_stanzas(lines: list[str], 
                        target_min: int = 80, 
                        target_max: int = 150) -> list[str]:
    """
    Group consecutive short stanzas into longer training examples.
    
    Strategy:
      - If a line is already >= target_min, keep it as-is
      - Otherwise, concatenate consecutive lines with ' ' until
        the combined length reaches target_min
      - Never exceed target_max
    """
    result = []
    buffer = []
    buffer_len = 0
    
    for line in lines:
        line_len = len(line)
        
        if line_len >= target_min:
            # Flush buffer if any
            if buffer:
                result.append(' '.join(buffer))
                buffer = []
                buffer_len = 0
            result.append(line)
        else:
            # Would adding this line exceed target_max?
            new_len = buffer_len + line_len + (1 if buffer else 0)
            
            if new_len > target_max and buffer:
                # Flush current buffer
                result.append(' '.join(buffer))
                buffer = [line]
                buffer_len = line_len
            else:
                buffer.append(line)
                buffer_len = new_len
                
                # If we've reached the sweet spot, flush
                if buffer_len >= target_min:
                    result.append(' '.join(buffer))
                    buffer = []
                    buffer_len = 0
    
    # Flush remaining
    if buffer:
        combined = ' '.join(buffer)
        if len(combined) >= MIN_CHARS:
            result.append(combined)
    
    return result


# ─── Step 2: Length Filtering ────────────────────────────────────────────────

def apply_length_filters(lines: list[str]) -> tuple[list[str], dict]:
    """
    Filter and split lines by character length.
    
    Rules:
      - Drop lines < MIN_CHARS (20)
      - Split lines > MAX_CHARS (340) at sentence boundaries
      - Keep everything else
    """
    stats = Counter()
    filtered = []
    
    def process_long_segment(segment: str):
        # Hard chunk any segment that is still too long
        chunks = []
        while len(segment) > MAX_CHARS:
            chunks.append(segment[:MAX_CHARS])
            segment = segment[MAX_CHARS:]
            stats["hard_truncated"] += 1
        if len(segment) >= MIN_CHARS:
            chunks.append(segment)
        elif segment:
            stats["dropped_too_short"] += 1
        return chunks

    for line in lines:
        char_len = len(line)
        
        if char_len < MIN_CHARS:
            stats["dropped_too_short"] += 1
            continue
        
        if char_len > MAX_CHARS:
            # Try to split at sentence punctuation
            parts = re.split(r'(?<=[.!?।])\s+', line)
            
            current = ""
            for part in parts:
                if len(current) + len(part) + 1 <= MAX_CHARS:
                    current = (current + " " + part).strip() if current else part
                else:
                    if current:
                        filtered.extend(process_long_segment(current))
                    current = part
            
            if current:
                filtered.extend(process_long_segment(current))
            
            stats["split_long"] += 1
        else:
            filtered.append(line)
            stats["kept"] += 1
    
    return filtered, dict(stats)


# ─── Step 3: Register Balancing ──────────────────────────────────────────────

def balance_registers(indiccorp: list[str], 
                      madurai: list[str], 
                      premodern: list[str],
                      total_target: int = 75000) -> tuple[list[str], list[str], list[str]]:
    """
    Balance the three registers to target proportions via oversampling/downsampling.
    
    Target: 40% modern, 33% classical, 27% pre-modern
    """
    target_ic = int(total_target * TARGET_MODERN_PCT)
    target_pm = int(total_target * TARGET_CLASSICAL_PCT)
    target_pre = int(total_target * TARGET_PREMODERN_PCT)
    
    print(f"\n  📊 Balancing registers:")
    print(f"     IndicCorp:  {len(indiccorp):,} → target {target_ic:,}")
    print(f"     Madurai:    {len(madurai):,} → target {target_pm:,}")
    print(f"     Pre-Modern: {len(premodern):,} → target {target_pre:,}")
    
    # IndicCorp: downsample if needed (we have plenty)
    if len(indiccorp) > target_ic:
        random.seed(42)
        balanced_ic = random.sample(indiccorp, target_ic)
    else:
        balanced_ic = indiccorp
    
    # Project Madurai: oversample
    if len(madurai) > 0:
        repeat_factor = max(1, target_pm // len(madurai))
        remainder = target_pm - (repeat_factor * len(madurai))
        balanced_pm = madurai * repeat_factor
        if remainder > 0:
            balanced_pm += random.sample(madurai, min(remainder, len(madurai)))
        balanced_pm = balanced_pm[:target_pm]
    else:
        balanced_pm = []
    
    # Pre-Modern: oversample if available, otherwise redistribute
    if len(premodern) > 0:
        repeat_factor = max(1, target_pre // len(premodern))
        remainder = target_pre - (repeat_factor * len(premodern))
        balanced_pre = premodern * repeat_factor
        if remainder > 0:
            balanced_pre += random.sample(premodern, min(remainder, len(premodern)))
        balanced_pre = balanced_pre[:target_pre]
    else:
        # No pre-modern data yet — redistribute to existing sources
        print(f"     ⚠️ No pre-modern data — redistributing {target_pre:,} lines")
        extra_ic = target_pre // 2
        extra_pm = target_pre - extra_ic
        
        if len(indiccorp) > len(balanced_ic) + extra_ic:
            remaining_ic = [l for l in indiccorp if l not in set(balanced_ic)]
            balanced_ic += random.sample(remaining_ic, min(extra_ic, len(remaining_ic)))
        
        if len(madurai) > 0:
            extra_pm_lines = random.choices(madurai, k=extra_pm)
            balanced_pm += extra_pm_lines
        
        balanced_pre = []
    
    print(f"\n  ✅ After balancing:")
    print(f"     IndicCorp:  {len(balanced_ic):,} ({len(balanced_ic)/(len(balanced_ic)+len(balanced_pm)+len(balanced_pre))*100:.1f}%)")
    print(f"     Madurai:    {len(balanced_pm):,} ({len(balanced_pm)/(len(balanced_ic)+len(balanced_pm)+len(balanced_pre))*100:.1f}%)")
    print(f"     Pre-Modern: {len(balanced_pre):,} ({len(balanced_pre)/(len(balanced_ic)+len(balanced_pm)+len(balanced_pre))*100:.1f}%)")
    
    return balanced_ic, balanced_pm, balanced_pre


# ─── Step 4: OCR Error Simulation ───────────────────────────────────────────

def simulate_ocr_error(clean_text: str, 
                       error_rate: float = 0.05,
                       seed: int = None) -> str:
    """
    Simulate realistic Tamil OCR errors on clean text.
    
    Error types (weighted):
      - Character substitution (visually similar): 75%
      - Character swap (adjacent): 12%
      - Character deletion: 8%
      - Character insertion (duplicate neighbor): 5%
    
    Key design choices:
      - n_errors is based on Tamil char count, not total length
      - Only chars in the confusion table get substituted (no random garbage)
      - Substitution-heavy to avoid length changes that cascade in ByT5
      - 30% of examples get a "density burst": a localized bad-patch window
        with 3x the error rate, simulating ink starvation on print scans
    """
    if seed is not None:
        random.seed(seed)
    
    chars = list(clean_text)
    
    # Count Tamil chars and compute errors from THAT count (not total length)
    tamil_positions = [i for i, c in enumerate(chars) if '\u0B80' <= c <= '\u0BFF']
    
    if not tamil_positions:
        return clean_text
    
    # ── Density Burst (30% of examples) ──────────────────────────────────────
    # Pick a random contiguous window of Tamil chars (15–25 chars) and
    # apply 3x the base error rate within that window only.
    # This mimics real print failure modes (ink starvation, compression artefacts)
    # where errors cluster locally rather than spread uniformly.
    burst_positions = set()
    if len(tamil_positions) >= 10 and random.random() < 0.30:
        window_size = random.randint(15, 25)
        start_idx = random.randint(0, max(0, len(tamil_positions) - window_size))
        burst_window = tamil_positions[start_idx: start_idx + window_size]
        n_burst = max(2, int(len(burst_window) * error_rate * 3))
        burst_positions = set(random.sample(burst_window, min(n_burst, len(burst_window))))

    # ── Uniform baseline errors ───────────────────────────────────────────────
    n_errors = max(1, int(len(tamil_positions) * error_rate))
    remaining = [p for p in tamil_positions if p not in burst_positions]
    uniform_positions = set(random.sample(remaining, min(n_errors, len(remaining)))) if remaining else set()

    error_positions = burst_positions | uniform_positions
    
    for pos in sorted(error_positions, reverse=True):
        error_type = random.random()
        
        if error_type < 0.75:
            # Substitution — ONLY use confusion table, skip if no match
            original_char = chars[pos]
            if original_char in TAMIL_OCR_CONFUSIONS:
                chars[pos] = random.choice(TAMIL_OCR_CONFUSIONS[original_char])
            # else: skip this position — no realistic confusion exists
        
        elif error_type < 0.87:
            # Swap adjacent (length-preserving)
            if pos + 1 < len(chars):
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        
        elif error_type < 0.95:
            # Deletion
            chars.pop(pos)
        
        else:
            # Insertion — duplicate the character at this position
            chars.insert(pos, chars[pos])
    
    return ''.join(chars)


# ─── Step 5: Generate train.jsonl ────────────────────────────────────────────

# ─── Weakness 8: HTML Tag Injection ─────────────────────────────────────────
# Surya OCR emits <mark>, <sub>, <sup> wrapping individual characters mid-word.
# Examples from test_data_weakness_analysis.md:
#   சுதந்<mark>தி</mark>ரம்   (mark splits a Tamil word)
#   பி<sub>ர</sub>             (sub wraps a single char)
#   <sup>ற</sup> இடங்களிற்     (sup at word start)

_HTML_LEAK_TAGS = ['mark', 'sub', 'sup']

# Injection rate per register — old/classical fonts produce more Surya artifacts
_HTML_INJECT_RATE = {
    'indiccorp':       0.07,   # 7%  of indiccorp pairs get HTML tags
    'project_madurai': 0.12,  # 12% of classical pairs
    'premodern':       0.15,  # 15% of pre-modern pairs (old letterpress)
}


def inject_html_tags(text: str, n_tags: int = None, seed: int = None) -> str:
    """
    Inject <mark>, <sub>, or <sup> tags around individual Tamil characters,
    simulating Surya OCR's HTML leak artifact (Weakness 8).

    Mimics real patterns:
      - Tags wrap 1 character (occasionally 2 adjacent chars)
      - Tags appear mid-word, never around spaces/punctuation
      - Each injection: <tag>char(s)</tag> inserted in place of the char(s)
    """
    if seed is not None:
        random.seed(seed)

    chars = list(text)
    # Find all Tamil character positions (mid-word candidates)
    tamil_positions = [
        i for i, c in enumerate(chars)
        if '\u0B80' <= c <= '\u0BFF'
    ]

    if not tamil_positions:
        return text

    # Decide how many tags to inject (1–3, scaled by string length)
    max_tags = max(1, len(tamil_positions) // 50)
    if n_tags is None:
        n_tags = random.randint(1, min(3, max_tags))
    n_tags = min(n_tags, len(tamil_positions))

    inject_positions = sorted(
        random.sample(tamil_positions, n_tags),
        reverse=True  # process right-to-left so earlier insertions don't shift indices
    )

    for pos in inject_positions:
        tag = random.choice(_HTML_LEAK_TAGS)
        # Occasionally wrap 2 adjacent Tamil chars (more realistic)
        if (
            random.random() < 0.3
            and pos + 1 < len(chars)
            and '\u0B80' <= chars[pos + 1] <= '\u0BFF'
        ):
            # Wrap 2 chars: replace chars[pos:pos+2] with <tag>cc</tag>
            wrapped = f'<{tag}>{chars[pos]}{chars[pos+1]}</{tag}>'
            chars[pos:pos + 2] = list(wrapped)
        else:
            # Wrap 1 char
            wrapped = f'<{tag}>{chars[pos]}</{tag}>'
            chars[pos:pos + 1] = list(wrapped)

    return ''.join(chars)


def generate_training_pairs(all_lines: list[str],
                            source_labels: list[str],
                            error_rates: dict = None) -> list[dict]:
    """
    Generate noisy→clean training pairs for ByT5 OCR correction.
    
    Each pair: {"input": "noisy Tamil text", "target": "clean Tamil text", "source": "..."}
    
    Error rates vary by register:
      - Modern (IndicCorp): lower error rate (modern fonts are cleaner OCR)
      - Classical (PM): medium error rate 
      - Pre-Modern: higher error rate (old fonts, poor print quality)
    """
    if error_rates is None:
        error_rates = {
            "indiccorp": 0.03,     # 3% — modern fonts, clean OCR
            "project_madurai": 0.05,  # 5% — classical texts
            "premodern": 0.08,     # 8% — old fonts, letterpress
        }
    
    pairs = []
    zero_edit_retries = 0
    html_injected = 0
    
    for i, (line, source) in enumerate(zip(all_lines, source_labels)):
        rate = error_rates.get(source, 0.05)
        
        # Pass 1: Character-level OCR noise (substitution, swap, deletion)
        # Try up to 5 seeds to avoid zero-edit identity pairs
        noisy = line
        for attempt in range(5):
            noisy = simulate_ocr_error(line, error_rate=rate, seed=i * 100 + attempt)
            if noisy != line:
                break
        else:
            zero_edit_retries += 1
        
        # Pass 2: HTML tag injection (Weakness 8 — Surya <mark>/<sub>/<sup> leaks)
        # Target stays CLEAN — ByT5 learns to strip these tags
        html_rate = _HTML_INJECT_RATE.get(source, 0.07)
        if random.random() < html_rate:
            noisy = inject_html_tags(noisy, seed=i)
            html_injected += 1
            
        # Pass 3: Strict 1024-byte limit enforcement
        # If byte length exceeds 1024, drop it to be 100% safe for ByT5
        if len(noisy.encode('utf-8')) > 1024 or len(line.encode('utf-8')) > 1024:
            continue
        
        pairs.append({
            "input": noisy,
            "target": line,
            "source": source,
        })
    
    if zero_edit_retries > 0:
        print(f"  ⚠️  {zero_edit_retries} pairs still identical after 5 retries (chars not in confusion table)")
    print(f"  🏷️  HTML tag injection applied to {html_injected:,} pairs ({html_injected/len(pairs)*100:.1f}%)")
    
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Build balanced ByT5 training data")
    parser.add_argument("--report", action="store_true", help="Print detailed stats")
    parser.add_argument("--total", type=int, default=75000, help="Target total training examples")
    parser.add_argument("--error-rate", type=float, default=None, help="Override OCR error rate")
    args = parser.parse_args()
    
    random.seed(42)
    
    print("=" * 60)
    print("  🔧 ByT5 Training Data Builder")
    print("=" * 60)
    
    # ─── Load cleaned sources ────────────────────────────────────────────
    print("\n[1/5] Loading cleaned corpus files...")
    
    indiccorp = load_lines(CLEANED_IC_PATH)
    madurai = load_lines(CLEANED_PM_PATH)
    premodern = load_lines(CLEANED_PRE_PATH)
    
    print(f"  IndicCorp:     {len(indiccorp):,} lines")
    print(f"  Project Madurai: {len(madurai):,} lines")
    print(f"  Pre-Modern:    {len(premodern):,} lines")
    
    # ─── Concatenate PM stanzas ──────────────────────────────────────────
    print("\n[2/5] Concatenating short Project Madurai stanzas...")
    
    pm_before = len(madurai)
    pm_short_before = sum(1 for l in madurai if len(l) < 50)
    
    madurai_concat = concatenate_stanzas(madurai, target_min=80, target_max=150)
    
    pm_short_after = sum(1 for l in madurai_concat if len(l) < 50)
    pm_mean_before = statistics.mean([len(l) for l in madurai]) if madurai else 0
    pm_mean_after = statistics.mean([len(l) for l in madurai_concat]) if madurai_concat else 0
    
    print(f"  Before: {pm_before:,} lines (mean {pm_mean_before:.0f} chars, {pm_short_before:,} short)")
    print(f"  After:  {len(madurai_concat):,} lines (mean {pm_mean_after:.0f} chars, {pm_short_after:,} short)")
    
    # ─── Length filtering ────────────────────────────────────────────────
    print("\n[3/5] Applying length filters...")
    
    indiccorp_filtered, ic_stats = apply_length_filters(indiccorp)
    madurai_filtered, pm_stats = apply_length_filters(madurai_concat)
    premodern_filtered, pre_stats = apply_length_filters(premodern)
    
    print(f"  IndicCorp:  {len(indiccorp):,} → {len(indiccorp_filtered):,} ({ic_stats})")
    print(f"  Madurai:    {len(madurai_concat):,} → {len(madurai_filtered):,} ({pm_stats})")
    print(f"  Pre-Modern: {len(premodern):,} → {len(premodern_filtered):,} ({pre_stats})")
    
    # ─── Balance registers ───────────────────────────────────────────────
    print("\n[4/5] Balancing registers...")
    
    bal_ic, bal_pm, bal_pre = balance_registers(
        indiccorp_filtered, madurai_filtered, premodern_filtered,
        total_target=args.total
    )
    
    # ─── Post-balance safety: re-enforce MAX_CHARS after oversampling ───
    def enforce_max_chars(lines: list[str]) -> list[str]:
        """Hard-split any line that still exceeds MAX_CHARS."""
        result = []
        for line in lines:
            if len(line) <= MAX_CHARS:
                result.append(line)
            else:
                # Split at Tamil sentence punctuation, then hard-chop
                parts = re.split(r'(?<=[.!?।])\s+', line)
                current = ""
                for part in parts:
                    if len(current) + len(part) + 1 <= MAX_CHARS:
                        current = (current + " " + part).strip() if current else part
                    else:
                        if current:
                            while len(current) > MAX_CHARS:
                                result.append(current[:MAX_CHARS])
                                current = current[MAX_CHARS:]
                            if len(current) >= MIN_CHARS:
                                result.append(current)
                        current = part
                if current:
                    while len(current) > MAX_CHARS:
                        result.append(current[:MAX_CHARS])
                        current = current[MAX_CHARS:]
                    if len(current) >= MIN_CHARS:
                        result.append(current)
        return result
    
    bal_ic = enforce_max_chars(bal_ic)
    bal_pm = enforce_max_chars(bal_pm)
    bal_pre = enforce_max_chars(bal_pre)
    
    print(f"\n  🔒 Post-balance length enforcement:")
    print(f"     IndicCorp:  {len(bal_ic):,} lines (max {max(len(l) for l in bal_ic) if bal_ic else 0} chars)")
    print(f"     Madurai:    {len(bal_pm):,} lines (max {max(len(l) for l in bal_pm) if bal_pm else 0} chars)")
    print(f"     Pre-Modern: {len(bal_pre):,} lines (max {max(len(l) for l in bal_pre) if bal_pre else 0} chars)")
    
    # Combine with source labels
    all_lines = []
    source_labels = []
    
    for line in bal_ic:
        all_lines.append(line)
        source_labels.append("indiccorp")
    for line in bal_pm:
        all_lines.append(line)
        source_labels.append("project_madurai")
    for line in bal_pre:
        all_lines.append(line)
        source_labels.append("premodern")
    
    # Shuffle together
    combined = list(zip(all_lines, source_labels))
    random.shuffle(combined)
    all_lines, source_labels = zip(*combined) if combined else ([], [])
    all_lines = list(all_lines)
    source_labels = list(source_labels)
    
    # ─── Generate noisy→clean pairs ──────────────────────────────────────
    print(f"\n[5/5] Generating {len(all_lines):,} noisy→clean training pairs...")
    
    error_rates = None
    if args.error_rate:
        error_rates = {
            "indiccorp": args.error_rate,
            "project_madurai": args.error_rate,
            "premodern": args.error_rate,
        }
    
    pairs = generate_training_pairs(all_lines, source_labels, error_rates)
    
    # ─── Save train.jsonl ────────────────────────────────────────────────
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    # ─── Stats ───────────────────────────────────────────────────────────
    source_counts = Counter(source_labels)
    char_lengths = [len(p["target"]) for p in pairs]
    byte_lengths = [len(p["target"].encode("utf-8")) for p in pairs]
    
    # Compute edit distances — count actual character differences
    n_edits = []
    
    # Strip HTML tags for realistic edit distance reporting
    def strip_html(text: str) -> str:
        for tag in _HTML_LEAK_TAGS:
            text = text.replace(f"<{tag}>", "").replace(f"</{tag}>", "")
        return text

    for p in pairs[:2000]:  # Sample for speed
        inp = strip_html(p["input"])
        tgt = strip_html(p["target"])
        # Use Levenshtein to properly account for insertions/deletions
        n_edits.append(Levenshtein.distance(inp, tgt))
    
    stats = {
        "total_pairs": len(pairs),
        "sources": dict(source_counts),
        "char_length": {
            "min": min(char_lengths),
            "max": max(char_lengths),
            "mean": round(statistics.mean(char_lengths), 1),
            "median": round(statistics.median(char_lengths), 1),
        },
        "byte_length": {
            "min": min(byte_lengths),
            "max": max(byte_lengths),
            "mean": round(statistics.mean(byte_lengths), 1),
            "median": round(statistics.median(byte_lengths), 1),
        },
        "avg_edits_per_example": round(statistics.mean(n_edits), 1) if n_edits else 0,
        "fits_in_512_bytes": sum(1 for b in byte_lengths if b <= 512),
        "fits_in_1024_bytes": sum(1 for b in byte_lengths if b <= 1024),
    }
    
    # Count HTML-injected pairs in final output
    html_pairs = sum(
        1 for p in pairs
        if any(f'<{t}>' in p['input'] for t in _HTML_LEAK_TAGS)
    )
    stats["html_tag_injected"] = html_pairs
    stats["html_tag_injected_pct"] = round(html_pairs / len(pairs) * 100, 1)
    
    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # ─── Report ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ✅ TRAINING DATA GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total pairs:  {len(pairs):,}")
    print(f"  Output file:  {OUTPUT_JSONL}")
    print(f"  Stats file:   {OUTPUT_STATS}")
    print(f"\n  📊 Source Distribution:")
    for src, count in source_counts.items():
        pct = count / len(pairs) * 100
        print(f"     {src:20s} {count:6,} ({pct:.1f}%)")
    
    print(f"\n  📏 Character Length:")
    print(f"     Min:    {stats['char_length']['min']}")
    print(f"     Max:    {stats['char_length']['max']}")
    print(f"     Mean:   {stats['char_length']['mean']}")
    print(f"     Median: {stats['char_length']['median']}")
    
    print(f"\n  📐 Byte Length (ByT5 input):")
    print(f"     Fits in 512 bytes:  {stats['fits_in_512_bytes']:,} ({stats['fits_in_512_bytes']/len(pairs)*100:.1f}%)")
    print(f"     Fits in 1024 bytes: {stats['fits_in_1024_bytes']:,} ({stats['fits_in_1024_bytes']/len(pairs)*100:.1f}%)")
    
    print(f"\n  🔧 OCR Error Simulation:")
    print(f"     Avg edits/example:    {stats['avg_edits_per_example']}")
    print(f"     HTML tag injections:  {stats['html_tag_injected']:,} ({stats['html_tag_injected_pct']}%) — <mark>/<sub>/<sup>")
    
    # Show sample pairs
    print(f"\n  📋 SAMPLE TRAINING PAIRS:")
    for i in range(min(5, len(pairs))):
        p = pairs[i]
        print(f"\n  [{p['source']}]")
        print(f"    Input:  {p['input'][:100]}{'...' if len(p['input']) > 100 else ''}")
        print(f"    Target: {p['target'][:100]}{'...' if len(p['target']) > 100 else ''}")


if __name__ == "__main__":
    main()
