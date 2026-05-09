"""
Deep analysis of all PDFs in the OCR_dataset directory.
Checks: page count, embedded fonts, text extractability, image presence, rendering behavior.
"""
import sys
import io
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

DATASET = Path(r"d:\Evolve_Robot_Lab\Project\NLP Projects\OCR\OCR_dataset")

def analyze_pdf(pdf_path: Path):
    print(f"\n{'='*70}")
    print(f"FILE: {pdf_path.name}  ({pdf_path.stat().st_size / 1024:.1f} KB)")
    print(f"{'='*70}")
    
    doc = fitz.open(str(pdf_path))
    print(f"Pages: {len(doc)}")
    
    # --- Font analysis ---
    all_fonts = set()
    for page_num in range(len(doc)):
        page = doc[page_num]
        fonts = page.get_fonts(full=True)
        for f in fonts:
            # f = (xref, ext, type, basefont, name, encoding, is_embedded)
            all_fonts.add((f[3], f[4], f[5]))  # basefont, name, encoding
    
    print(f"\nFonts used ({len(all_fonts)}):")
    for basefont, name, encoding in sorted(all_fonts):
        print(f"  - basefont={basefont}, name={name}, encoding={encoding}")
    
    # --- Image analysis ---
    print(f"\nImage analysis (per page):")
    for page_num in range(min(3, len(doc))):  # Check first 3 pages
        page = doc[page_num]
        images = page.get_images(full=True)
        print(f"  Page {page_num+1}: {len(images)} embedded image(s)")
        for img_info in images[:3]:
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            print(f"    -> xref={xref}, size={base_image['width']}x{base_image['height']}, "
                  f"format={base_image['ext']}, colorspace={base_image.get('cs-name','?')}, "
                  f"bytes={len(base_image['image'])}")
    
    # --- Text extraction analysis ---
    print(f"\nRaw text extraction (first 3 pages):")
    for page_num in range(min(3, len(doc))):
        page = doc[page_num]
        text = page.get_text("text")
        # Show first 200 chars
        preview = text[:200].replace('\n', '\\n')
        print(f"  Page {page_num+1} ({len(text)} chars): {preview}")
    
    # --- Detailed text dict analysis (shows font info per span) ---
    print(f"\nDetailed text spans (page 1, first 10 spans):")
    page = doc[0]
    text_dict = page.get_text("dict")
    span_count = 0
    for block in text_dict.get("blocks", []):
        if block.get("type") == 0:  # text block
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span_count < 10:
                        print(f"  font='{span['font']}', size={span['size']:.1f}, "
                              f"flags={span['flags']}, "
                              f"text='{span['text'][:60]}'")
                        span_count += 1
    
    # --- Check if it's truly image-based by rendering and comparing ---
    page = doc[0]
    text_len = len(page.get_text("text").strip())
    images = page.get_images(full=True)
    
    print(f"\n--- VERDICT ---")
    if len(images) > 0 and text_len < 50:
        print(f"  TYPE: Pure Image-based scan (text_len={text_len}, images={len(images)})")
        print(f"  ACTION: Direct OCR on extracted images")
    elif len(images) == 0 and text_len > 50:
        print(f"  TYPE: Digital text PDF (text_len={text_len}, images={len(images)})")
        print(f"  ACTION: Extract text directly or render-then-OCR")
    elif len(images) > 0 and text_len > 50:
        print(f"  TYPE: Hybrid (both text and images, text_len={text_len}, images={len(images)})")
        print(f"  ACTION: Needs careful handling")
    else:
        print(f"  TYPE: Unknown (text_len={text_len}, images={len(images)})")
    
    # --- Render test: render page and check if it looks right ---
    pix = page.get_pixmap(dpi=150)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    # Check if the rendered image is mostly blank/white
    import numpy as np
    arr = np.array(img.convert("L"))
    dark_ratio = (arr < 128).sum() / arr.size
    print(f"  Rendered dark pixel ratio: {dark_ratio:.4f} "
          f"({'has content' if dark_ratio > 0.01 else 'MOSTLY BLANK'})")
    
    doc.close()


def main():
    # Separate source and verify PDFs
    source_pdfs = sorted([p for p in DATASET.glob("*.pdf") if "(verify)" not in p.name])
    verify_pdfs = sorted([p for p in DATASET.glob("*.pdf") if "(verify)" in p.name])
    
    print("=" * 70)
    print("ANALYZING SOURCE PDFs (the ones we need to OCR)")
    print("=" * 70)
    for pdf in source_pdfs:
        analyze_pdf(pdf)
    
    print("\n\n")
    print("=" * 70)
    print("ANALYZING VERIFY PDFs (the ground truth sources)")
    print("=" * 70)
    for pdf in verify_pdfs[:2]:  # Just check 2 verify PDFs for comparison
        analyze_pdf(pdf)


if __name__ == "__main__":
    main()
