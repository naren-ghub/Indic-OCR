"""
Tamil Corpus Downloader — Phase 2 / Step 2.3a
==============================================
Downloads clean Tamil text from three sources:

  1. Tamil Wikipedia — via HuggingFace `datasets` library (automated, ~50MB)
  2. Project Madurai — classic Tamil literature (public domain, HTTP download)
  3. IndicCorp / AI4Bharat Tamil — large news + web corpus (HuggingFace)

All text is cleaned, sentence-split, and saved to:
  data/corpus/wikipedia.txt
  data/corpus/project_madurai.txt
  data/corpus/indiccorp.txt
  data/corpus/combined.txt   ← merged deduplicated corpus for noise gen

Usage:
    python prepare_training_data.py --download-corpus
    python prepare_training_data.py --all
"""
from __future__ import annotations

import re
import sys
import time
import requests
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

CORPUS_DIR = Path(__file__).parent.parent / "data" / "corpus"

# ── Minimum sentence quality filters ──────────────────────────────────────────
MIN_SENTENCE_LENGTH  = 20   # chars
MAX_SENTENCE_LENGTH  = 500  # chars
MIN_TAMIL_CHAR_RATIO = 0.40 # at least 40% Tamil characters


def _has_min_tamil(text: str, ratio: float = MIN_TAMIL_CHAR_RATIO) -> bool:
    if not text:
        return False
    tamil_count = sum(1 for c in text if 0x0B80 <= ord(c) <= 0x0BFF)
    return (tamil_count / len(text)) >= ratio


def _clean_sentence(s: str) -> str:
    """Normalize whitespace and strip common Wikipedia/HTML artifacts."""
    s = re.sub(r'\[\d+\]', '', s)          # Remove citation markers [1], [23]
    s = re.sub(r'\[.*?\]', '', s)           # Remove other bracket content
    s = re.sub(r'http\S+', '', s)           # Remove URLs
    s = re.sub(r'[^\u0B80-\u0BFF\u0000-\u007F\s\.\!\?\,\;\:\-\"\'()]', '', s)
    s = re.sub(r'\s+', ' ', s)             # Collapse whitespace
    return s.strip()


def _is_valid_sentence(s: str) -> bool:
    s = s.strip()
    if len(s) < MIN_SENTENCE_LENGTH or len(s) > MAX_SENTENCE_LENGTH:
        return False
    return _has_min_tamil(s)


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences on Tamil/English punctuation."""
    parts = re.split(r'(?<=[.!?।\n])\s+', text)
    return [p.strip() for p in parts if p.strip()]


# ═══════════════════════════════════════════════════════════════════════════════
#  Source 1: Tamil Wikipedia (HuggingFace datasets)
# ═══════════════════════════════════════════════════════════════════════════════

def download_tamil_wikipedia(max_sentences: int = 50_000) -> Path:
    """
    Download Tamil Wikipedia via HuggingFace datasets.
    Requires: pip install datasets

    Returns path to saved corpus file.
    """
    print("\n[1/3] Downloading Tamil Wikipedia (HuggingFace datasets)...")
    out_path = CORPUS_DIR / "wikipedia.txt"

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  ⏭️ Skipping: {out_path.name} already exists.")
        return out_path

    try:
        from datasets import load_dataset
    except ImportError:
        print("  ❌ `datasets` not installed. Run: pip install datasets")
        return out_path

    print("  Loading wikimedia/wikipedia 20231101.ta — this may take a few minutes...")
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.ta",
        split="train",
    )

    sentences = []
    for article in dataset:
        text = article.get("text", "")
        for sent in _split_into_sentences(text):
            cleaned = _clean_sentence(sent)
            if _is_valid_sentence(cleaned):
                sentences.append(cleaned)
            if len(sentences) >= max_sentences:
                break
        if len(sentences) >= max_sentences:
            break

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences))

    print(f"  ✅ Wikipedia: {len(sentences):,} sentences → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Source 2: Project Madurai (Classic Tamil literature)
# ═══════════════════════════════════════════════════════════════════════════════

# Selected public-domain texts covering a range of literary styles
# These are direct text file URLs from Project Madurai's corpus
PROJECT_MADURAI_TEXTS = [
    ("thirukkural",    "https://www.projectmadurai.org/pm_etexts/utf8/pmuni0001.html"),
    ("purananuru",     "https://www.projectmadurai.org/pm_etexts/utf8/pmuni0002.html"),
    ("silappathikaram", "https://www.projectmadurai.org/pm_etexts/utf8/pmuni0046.html"),
    ("manimekalai",    "https://www.projectmadurai.org/pm_etexts/utf8/pmuni0141.html"),
    ("agananuru",      "https://www.projectmadurai.org/pm_etexts/utf8/pmuni0016.html"),
]


def download_project_madurai(max_sentences: int = 20_000) -> Path:
    """
    Download selected Project Madurai texts via HTTP.
    Returns path to saved corpus file.
    """
    print("\n[2/3] Downloading Project Madurai texts...")
    out_path = CORPUS_DIR / "project_madurai.txt"

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  ⏭️ Skipping: {out_path.name} already exists.")
        return out_path

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    all_sentences = []

    for name, url in PROJECT_MADURAI_TEXTS:
        print(f"  Fetching {name}...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            # Project Madurai serves UTF-8 encoded files (HTML or Text)
            text = response.content.decode("utf-8", errors="replace")
            
            # Strip HTML tags if present
            text = re.sub(r'<.*?>', ' ', text)

            for sent in _split_into_sentences(text):
                cleaned = _clean_sentence(sent)
                if _is_valid_sentence(cleaned):
                    all_sentences.append(cleaned)

            print(f"    ✅ {name}: {len(all_sentences):,} total sentences so far")
            time.sleep(1)  # Be polite to Project Madurai's server

        except requests.RequestException as e:
            print(f"    ⚠️  Could not download {name}: {e}")

        if len(all_sentences) >= max_sentences:
            break

    all_sentences = all_sentences[:max_sentences]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_sentences))

    print(f"  ✅ Project Madurai: {len(all_sentences):,} sentences → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Source 3: AI4Bharat IndicCorp (News + Web Tamil)
# ═══════════════════════════════════════════════════════════════════════════════

def download_indiccorp_tamil(max_sentences: int = 50_000) -> Path:
    """
    Download Tamil sentences from AI4Bharat's IndicCorp v2 via HuggingFace.
    This is a large news + web corpus — modern Tamil prose.

    Returns path to saved corpus file.
    """
    print("\n[3/3] Downloading AI4Bharat IndicCorp Tamil...")
    out_path = CORPUS_DIR / "indiccorp.txt"

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  ⏭️ Skipping: {out_path.name} already exists.")
        return out_path

    try:
        from datasets import load_dataset
    except ImportError:
        print("  ❌ `datasets` not installed. Run: pip install datasets")
        return out_path

    print("  Loading ai4bharat/IndicCorp (Tamil split)...")
    try:
        dataset = load_dataset(
            "ai4bharat/IndicCorpV2",
            "indiccorp_v2",
            split="tam_Taml",
            streaming=True,        # Stream — don't download the full 100GB dataset
        )

        sentences = []
        for example in dataset:
            text = example.get("text", "").strip()
            cleaned = _clean_sentence(text)
            if _is_valid_sentence(cleaned):
                sentences.append(cleaned)
            if len(sentences) >= max_sentences:
                break

    except Exception as e:
        print(f"  ⚠️  IndicCorp download failed: {e}")
        print("  Trying fallback: Oscar Tamil corpus...")
        sentences = _download_oscar_tamil(max_sentences)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences))

    print(f"  ✅ IndicCorp: {len(sentences):,} sentences → {out_path}")
    return out_path


def _download_oscar_tamil(max_sentences: int = 30_000) -> list[str]:
    """Fallback: OSCAR corpus Tamil subset (smaller, more accessible)."""
    try:
        from datasets import load_dataset
        print("  Loading oscar-corpus/OSCAR-2301 (ta)...")
        dataset = load_dataset(
            "oscar-corpus/OSCAR-2301",
            "ta",
            split="train",
            streaming=True,
            token=True,
        )
        sentences = []
        for example in dataset:
            text = example.get("content", "").strip()
            for sent in _split_into_sentences(text):
                cleaned = _clean_sentence(sent)
                if _is_valid_sentence(cleaned):
                    sentences.append(cleaned)
            if len(sentences) >= max_sentences:
                break
        return sentences[:max_sentences]
    except Exception as e:
        print(f"  ❌ OSCAR fallback also failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
#  Combine & Deduplicate
# ═══════════════════════════════════════════════════════════════════════════════

def combine_corpora(target_sentences: int = 100_000) -> Path:
    """
    Merge all downloaded corpus files, deduplicate, shuffle, and save.

    Returns path to combined corpus file.
    """
    print("\n[Combine] Merging all corpus sources...")
    out_path = CORPUS_DIR / "combined.txt"

    all_sentences = set()

    for corpus_file in CORPUS_DIR.glob("*.txt"):
        if corpus_file.name == "combined.txt":
            continue
        with open(corpus_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        before = len(all_sentences)
        all_sentences.update(lines)
        print(f"  {corpus_file.name}: added {len(all_sentences) - before:,} unique sentences")

    import random
    sentences_list = list(all_sentences)
    random.shuffle(sentences_list)
    sentences_list = sentences_list[:target_sentences]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences_list))

    print(f"\n  ✅ Combined corpus: {len(sentences_list):,} unique sentences → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def download_all_corpora():
    """Download all three sources and combine."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    download_tamil_wikipedia(max_sentences=50_000)
    download_project_madurai(max_sentences=20_000)
    download_indiccorp_tamil(max_sentences=50_000)
    combine_corpora(target_sentences=100_000)

    print("\n✅ All corpora downloaded and combined.")
    print(f"   Location: {CORPUS_DIR}")
