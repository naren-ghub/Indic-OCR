#!/usr/bin/env python3
"""
scrape_premodern_v2.py — Expanded Pre-Modern Tamil Text Scraper
================================================================
Uses correct Wikisource category names (English) and expanded page list.
"""

import re
import sys
import io
import time
import requests
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

CORPUS_DIR = Path(__file__).parent / "corpus"
OUTPUT_PATH = CORPUS_DIR / "premodern_tamil.txt"

API_URL = "https://ta.wikisource.org/w/api.php"
HEADERS = {
    "User-Agent": "TamilOCRCorpusBot/1.0 (Academic research; Tamil OCR training data)",
    "Accept": "application/json",
}

# ─── Correct English category names from search results ─────────────────────
CATEGORIES = [
    "19th_century_works",
    "20th_century_works",
]

# ─── Expanded direct page list ──────────────────────────────────────────────
EXTRA_PAGES = [
    # Classical literature (reprinted in 1800s with commentaries)
    "நன்னூல்",
    "தொல்காப்பியம்",
    "கம்பராமாயணம்",
    "திருவாசகம்",
    "நாலடியார்",
    "பழமொழி நானூறு",
    "ஆசாரக்கோவை",
    "இன்னா நாற்பது",
    "திரிகடுகம்",
    "ஏலாதி",
    "கார் நாற்பது",
    "கைந்நிலை",
    # Additional Tamil literary works
    "சீவகசிந்தாமணி",
    "கலிங்கத்துப்பரணி",
    "கந்தபுராணம்",
    "வில்லிபாரதம்",
    "நந்திக்கலம்பகம்",
    "குறுந்தொகை",
    "அகநானூறு",
    "புறநானூறு",
    "பதிற்றுப்பத்து",
    "ஐங்குறுநூறு",
    "திருமுருகாற்றுப்படை",
    "மலைபடுகடாம்",
    "முல்லைப்பாட்டு",
    "நெடுநல்வாடை",
    "பட்டினப்பாலை",
    "பொருநராற்றுப்படை",
    "சிறுபாணாற்றுப்படை",
    "பெரும்பாணாற்றுப்படை",
    "மதுரைக்காஞ்சி",
    "திருக்குறள்/பொருட்பால்",
    "திருக்குறள்/இன்பத்துப்பால்",
    # 19th-20th century prose / essays / grammar
    "தமிழ் இலக்கணம்",
    "நன்னூல் விருத்தியுரை",
    "திருக்குறள் பரிமேலழகர் உரை",
    "அபிதான சிந்தாமணி",
    "தமிழ்நாட்டு வரலாறு",
    "சிலப்பதிகாரம் அரும்பதவுரை",
    # More prose texts
    "தமிழர் வரலாறு",
    "தமிழ் இலக்கிய வரலாறு",
    "தென்னிந்திய வரலாறு",
    "பாரதியார் கவிதைகள்",
    "பாரதிதாசன் கவிதைகள்",
]


def api_request(params, max_retries=3):
    params["format"] = "json"
    for attempt in range(max_retries):
        try:
            resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"    ⚠️ API error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return {}


def get_category_members(category, limit=200):
    """Fetch pages from a category using English category names."""
    print(f"\n  📂 Category: {category}")
    titles = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": "50",
        "cmtype": "page|subcat",
    }
    
    while len(titles) < limit:
        data = api_request(params)
        if not data:
            break
        members = data.get("query", {}).get("categorymembers", [])
        if not members:
            break
        for m in members:
            titles.append(m["title"])
        if "continue" in data:
            params["cmcontinue"] = data["continue"]["cmcontinue"]
        else:
            break
        time.sleep(0.5)
    
    print(f"    Found {len(titles)} pages/subcats")
    return titles[:limit]


def get_page_text(title):
    """Get plain text from a page using TextExtracts."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "true",
        "exlimit": "1",
    }
    data = api_request(params)
    if not data:
        return ""
    pages = data.get("query", {}).get("pages", {})
    for pid, pdata in pages.items():
        if pid == "-1":
            return ""
        return pdata.get("extract", "")
    return ""


def get_page_wikitext(title):
    """Get raw wikitext and clean it."""
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
    for pid, pdata in pages.items():
        if pid == "-1":
            return ""
        revs = pdata.get("revisions", [])
        if revs:
            return revs[0].get("slots", {}).get("main", {}).get("*", "")
    return ""


def strip_wikitext(wikitext):
    """Convert wikitext to plain Tamil text."""
    text = wikitext
    for _ in range(5):
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)
    text = re.sub(r'\[\[(?:Category|பகுப்பு):.*?\]\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'\[\[(?:File|Image|படிமம்):.*?\]\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[https?://\S+\s+([^\]]*)\]', r'\1', text)
    text = re.sub(r'\[https?://\S+\]', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r"'{2,}", '', text)
    text = re.sub(r'^[=]+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*[=]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\*+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^:+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\{\|.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\|\}.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\|.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^!.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'__[A-Z]+__', '', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def count_tamil(text):
    return sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')


def scrape_page(title):
    """Scrape a single page, return list of clean lines."""
    text = get_page_text(title)
    if not text or count_tamil(text) < 100:
        wikitext = get_page_wikitext(title)
        if wikitext:
            text = strip_wikitext(wikitext)
    
    if not text or count_tamil(text) < 100:
        return []
    
    lines = text.split('\n')
    good = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^\s*\d{1,4}\s*$', line):
            continue
        if re.match(r'^[\s\d\.\-\*\(\)]+$', line):
            continue
        line = re.sub(r'\s*[\*†‡§]\d*\s*$', '', line)
        line = re.sub(r' {2,}', ' ', line).strip()
        if count_tamil(line) >= 10:
            good.append(line)
    
    return good


def main():
    print("=" * 60)
    print("  🕰️  Pre-Modern Tamil Scraper v2 (Expanded)")
    print("=" * 60)
    
    all_lines = []
    seen_titles = set()
    
    # Load existing if any (append mode)
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = [l.strip() for l in f if l.strip()]
        print(f"\n  📄 Existing file has {len(existing):,} lines — will append new data")
        all_lines = existing
    
    # Step 1: Category scraping (19th & 20th century works)
    print("\n[1/2] Scraping from categories...")
    
    for cat in CATEGORIES:
        members = get_category_members(cat, limit=200)
        
        for title in members:
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            # Skip category/subcategory pages
            if title.startswith("Category:") or title.startswith("பகுப்பு:"):
                # Recurse into subcategory
                sub_members = get_category_members(title.replace("Category:", "").replace("பகுப்பு:", ""), limit=50)
                for sub_title in sub_members:
                    if sub_title in seen_titles:
                        continue
                    seen_titles.add(sub_title)
                    if sub_title.startswith("Category:") or sub_title.startswith("பகுப்பு:"):
                        continue
                    
                    print(f"  📖 {sub_title}", end="", flush=True)
                    lines = scrape_page(sub_title)
                    if lines:
                        all_lines.extend(lines)
                        print(f" → {len(lines)} lines ✅")
                    else:
                        print(f" → ⚠️ no content")
                    time.sleep(1)
                continue
            
            print(f"  📖 {title}", end="", flush=True)
            lines = scrape_page(title)
            if lines:
                all_lines.extend(lines)
                print(f" → {len(lines)} lines ✅")
            else:
                print(f" → ⚠️ no content")
            time.sleep(1)
            
            if len(all_lines) >= 25000:
                break
        
        if len(all_lines) >= 25000:
            break
    
    # Step 2: Direct page scraping (extra pages not in categories)
    print(f"\n[2/2] Scraping {len(EXTRA_PAGES)} known pages...")
    
    for title in EXTRA_PAGES:
        if title in seen_titles:
            continue
        seen_titles.add(title)
        
        print(f"  📖 {title}", end="", flush=True)
        lines = scrape_page(title)
        if lines:
            all_lines.extend(lines)
            print(f" → {len(lines)} lines ✅")
        else:
            print(f" → ⚠️ no content")
        time.sleep(1)
        
        if len(all_lines) >= 25000:
            break
    
    # Deduplicate
    unique = list(dict.fromkeys(all_lines))
    
    # Save
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(unique))
    
    print(f"\n{'='*60}")
    print(f"  ✅ SCRAPING COMPLETE")
    print(f"  Total lines:  {len(all_lines):,}")
    print(f"  Unique lines: {len(unique):,}")
    print(f"  Saved to:     {OUTPUT_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
