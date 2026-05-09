#!/usr/bin/env python3
"""
scrape_premodern.py — Pre-Modern Tamil Text Scraper (1800–1950)
================================================================
Scrapes Tamil Wikisource for pre-modern prose and verse texts.

Uses MediaWiki API with proper User-Agent headers.
Targets texts from the 1800–1950 era with Sanskrit/Grantha influence.

Usage:
    python data/scrape_premodern.py
"""

import re
import sys
import io
import time
import json
import requests
from pathlib import Path

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

CORPUS_DIR = Path(__file__).parent / "corpus"
OUTPUT_PATH = CORPUS_DIR / "premodern_tamil.txt"

# MediaWiki API endpoint
API_URL = "https://ta.wikisource.org/w/api.php"

# Proper headers — Wikisource blocks requests without User-Agent
HEADERS = {
    "User-Agent": "TamilOCRCorpusBot/1.0 (Academic research; Tamil OCR training data; contact: research@example.com)",
    "Accept": "application/json",
}

# ─── Known pre-modern Tamil text pages on ta.wikisource.org ──────────────────
# These are specific pages/books known to contain 1800-1950 prose/verse
# Curated manually from Tamil Wikisource's catalog

PREMODERN_PAGES = [
    # Prose & essays (1800-1950)
    "நன்னூல்",                           # Nannul - Tamil grammar treatise
    "தொல்காப்பியம்",                      # Tolkappiyam - ancient grammar (reprinted 1800s)
    "கம்பராமாயணம்",                       # Kamba Ramayanam
    "சிலப்பதிகாரம்/மதுரைக்காண்டம்",     # Silappathikaram - Madurai chapter
    "பெரியபுராணம்",                      # Periya Puranam - Sekkizhar
    "திருவாசகம்",                         # Thiruvasagam - Manikkavasagar
    "நாலடியார்",                          # Naladiyar
    "பழமொழி நானூறு",                     # Pazhamozhi Naanuru
    "ஆசாரக்கோவை",                        # Acharakkkovai
    "இன்னா நாற்பது",                     # Inna Narpathu
    "இனியவை நாற்பது",                    # Iniyavai Narpathu
    "திரிகடுகம்",                          # Trikadugam
    "ஏலாதி",                              # Elathi
    "சிறுபஞ்சமூலம்",                      # Sirupanchamoolam
    "முதுமொழிக்காஞ்சி",                  # Muthumozhikkanchi
    "கார் நாற்பது",                       # Kaar Narpathu
    "கைந்நிலை",                           # Kainnilai
    "அறநெறிச்சாரம்",                     # Aranerichcharam
]

# Categories that contain pre-modern texts
CATEGORIES_TO_SCRAPE = [
    "நூல்கள்",                            # Books
    "உரைநடை",                             # Prose
    "இலக்கணம்",                           # Grammar texts
]


def api_request(params: dict, max_retries: int = 3) -> dict:
    """Make a request to the Tamil Wikisource API with retry logic."""
    params["format"] = "json"
    
    for attempt in range(max_retries):
        try:
            resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"    ⚠️ API request failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return {}


def get_category_members(category: str, limit: int = 100) -> list[str]:
    """Get page titles from a Wikisource category."""
    print(f"  📂 Fetching category: {category}")
    
    titles = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": str(min(limit, 50)),
        "cmtype": "page",
    }
    
    while True:
        data = api_request(params)
        if not data:
            break
            
        members = data.get("query", {}).get("categorymembers", [])
        for m in members:
            titles.append(m["title"])
        
        # Handle continuation
        if "continue" in data and len(titles) < limit:
            params["cmcontinue"] = data["continue"]["cmcontinue"]
        else:
            break
        
        time.sleep(0.5)  # Rate limiting
    
    print(f"    Found {len(titles)} pages")
    return titles[:limit]


def get_page_text(title: str) -> str:
    """Get the plain text content of a Wikisource page."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "true",  # Plain text, no HTML
        "exlimit": "1",
    }
    
    data = api_request(params)
    if not data:
        return ""
    
    pages = data.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        if page_id == "-1":
            return ""
        return page_data.get("extract", "")
    
    return ""


def get_page_wikitext(title: str) -> str:
    """Get raw wikitext content and strip markup to extract clean text."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "rvlimit": "1",
    }
    
    data = api_request(params)
    if not data:
        return ""
    
    pages = data.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        if page_id == "-1":
            return ""
        revisions = page_data.get("revisions", [])
        if revisions:
            slots = revisions[0].get("slots", {})
            main_slot = slots.get("main", {})
            return main_slot.get("*", "")
    
    return ""


def strip_wikitext(wikitext: str) -> str:
    """Convert wikitext to plain Tamil text."""
    text = wikitext
    
    # Remove templates: {{...}}
    # Handle nested templates
    for _ in range(5):  # Multiple passes for nested
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)
    
    # Remove categories: [[Category:...]]
    text = re.sub(r'\[\[(?:Category|பகுப்பு):.*?\]\]', '', text, flags=re.IGNORECASE)
    
    # Convert wikilinks [[target|display]] → display, [[target]] → target
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)
    
    # Remove file/image links
    text = re.sub(r'\[\[(?:File|Image|படிமம்):.*?\]\]', '', text, flags=re.IGNORECASE)
    
    # Remove external links [url text] → text
    text = re.sub(r'\[https?://\S+\s+([^\]]*)\]', r'\1', text)
    text = re.sub(r'\[https?://\S+\]', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove wiki formatting
    text = re.sub(r"'{2,}", '', text)      # Bold/italic markers
    text = re.sub(r'^[=]+\s*', '', text, flags=re.MULTILINE)  # Section headers =
    text = re.sub(r'\s*[=]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\*+\s*', '', text, flags=re.MULTILINE)   # List markers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^:+\s*', '', text, flags=re.MULTILINE)
    
    # Remove table markup
    text = re.sub(r'^\{\|.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\|\}.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\|.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^!.*$', '', text, flags=re.MULTILINE)
    
    # Remove magic words
    text = re.sub(r'__[A-Z]+__', '', text)
    
    # Normalize whitespace
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def count_tamil_chars(text: str) -> int:
    """Count Tamil Unicode characters."""
    return sum(1 for ch in text if '\u0B80' <= ch <= '\u0BFF')


def clean_premodern_line(line: str) -> str:
    """Clean a single line of pre-modern Tamil text."""
    line = line.strip()
    
    # Remove standalone page numbers
    if re.match(r'^\s*\d{1,4}\s*$', line):
        return ""
    
    # Remove lines that are just punctuation or numbers
    if re.match(r'^[\s\d\.\-\*\(\)]+$', line):
        return ""
    
    # Remove footnote markers
    line = re.sub(r'\s*[\*†‡§]\d*\s*$', '', line)
    
    # Normalize whitespace
    line = re.sub(r' {2,}', ' ', line).strip()
    
    return line


def scrape_all_texts() -> list[str]:
    """Scrape all pre-modern Tamil texts from Wikisource."""
    all_lines = []
    
    print("=" * 60)
    print("  📚 Pre-Modern Tamil Text Scraper")
    print("=" * 60)
    
    # Step 1: Scrape known pages directly
    print("\n[1/2] Scraping known pre-modern text pages...")
    
    for title in PREMODERN_PAGES:
        print(f"\n  📖 {title}")
        
        # Try extracts first (cleaner)
        text = get_page_text(title)
        
        if not text or count_tamil_chars(text) < 100:
            # Fall back to wikitext parsing
            print(f"    → Trying wikitext fallback...")
            wikitext = get_page_wikitext(title)
            if wikitext:
                text = strip_wikitext(wikitext)
        
        if text and count_tamil_chars(text) >= 100:
            lines = text.split('\n')
            good_lines = []
            for line in lines:
                cleaned = clean_premodern_line(line)
                if cleaned and count_tamil_chars(cleaned) >= 10:
                    good_lines.append(cleaned)
            
            all_lines.extend(good_lines)
            print(f"    ✅ Extracted {len(good_lines)} lines ({count_tamil_chars(text)} Tamil chars)")
        else:
            print(f"    ⚠️ No usable content found")
        
        time.sleep(1)  # Rate limiting
    
    # Step 2: Scrape from categories
    print(f"\n[2/2] Scraping from categories...")
    
    seen_titles = set(PREMODERN_PAGES)
    
    for category in CATEGORIES_TO_SCRAPE:
        titles = get_category_members(category, limit=50)
        
        for title in titles:
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            print(f"\n  📖 {title}")
            
            text = get_page_text(title)
            if not text or count_tamil_chars(text) < 100:
                wikitext = get_page_wikitext(title)
                if wikitext:
                    text = strip_wikitext(wikitext)
            
            if text and count_tamil_chars(text) >= 100:
                lines = text.split('\n')
                good_lines = []
                for line in lines:
                    cleaned = clean_premodern_line(line)
                    if cleaned and count_tamil_chars(cleaned) >= 10:
                        good_lines.append(cleaned)
                
                all_lines.extend(good_lines)
                print(f"    ✅ Extracted {len(good_lines)} lines")
            else:
                print(f"    ⚠️ No usable content")
            
            time.sleep(1)
            
            # Stop if we have enough
            if len(all_lines) >= 20000:
                print(f"\n  🎯 Reached target: {len(all_lines)} lines")
                break
        
        if len(all_lines) >= 20000:
            break
    
    return all_lines


def main():
    print("\n" + "=" * 60)
    print("  🕰️  Pre-Modern Tamil (1800–1950) Scraper")
    print("=" * 60)
    
    lines = scrape_all_texts()
    
    if not lines:
        print("\n❌ No lines scraped! Check network connectivity.")
        return
    
    # Deduplicate
    unique_lines = list(dict.fromkeys(lines))
    
    # Save
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(unique_lines))
    
    print(f"\n{'='*60}")
    print(f"  ✅ SCRAPING COMPLETE")
    print(f"  Total lines:  {len(lines):,}")
    print(f"  Unique lines: {len(unique_lines):,}")
    print(f"  Saved to:     {OUTPUT_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
