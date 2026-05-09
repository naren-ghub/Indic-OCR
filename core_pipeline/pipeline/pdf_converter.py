"""
Step 1 — PDF to Image Conversion
Uses PyMuPDF (fitz) — no poppler dependency needed on Windows.
Extracts embedded images directly for image-based PDFs (better quality).
"""
import io
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
from typing import List, Tuple
from config import PDF_DPI


def pdf_to_images(pdf_path: str | Path, dpi: int = PDF_DPI) -> List[Tuple[int, Image.Image]]:
    """
    Convert every page of a PDF to a PIL Image.

    For image-based PDFs (scans): extracts the embedded image directly
    to avoid quality loss from re-rendering.

    For text-based PDFs: renders the page at the specified DPI.

    Args:
        pdf_path: Path to the input PDF.
        dpi:      Rendering resolution for text-based PDFs.

    Returns:
        List of (page_number, PIL.Image) tuples — 1-indexed page numbers.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    pages: List[Tuple[int, Image.Image]] = []
    page_idx = 0  # Sequential counter (handles spread splits)

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)
        text_len = len(page.get_text("text").strip())

        # Strategy: if the page has exactly 1 image and no selectable text,
        # it's a pure scan — extract the image directly for best quality.
        if len(images) == 1 and text_len == 0:
            xref = images[0][0]
            base_image = doc.extract_image(xref)
            img = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
        else:
            # Text-based or hybrid page — render at target DPI
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

        # Detect two-page landscape spreads and split them
        split_imgs = _maybe_split_spread(img)

        for sub_img in split_imgs:
            page_idx += 1
            pages.append((page_idx, sub_img))

    doc.close()
    return pages


def _maybe_split_spread(img: Image.Image, ratio_threshold: float = 1.3) -> list:
    """
    If the image is landscape (width > height * ratio_threshold), it is
    likely a two-page book spread scanned as a single image. Split it
    vertically down the middle into left-page and right-page.

    Returns a list of 1 or 2 PIL Images.
    """
    w, h = img.size
    if w > h * ratio_threshold:
        mid = w // 2
        left = img.crop((0, 0, mid, h))
        right = img.crop((mid, 0, w, h))
        return [left, right]
    return [img]


def pdf_to_images_save(
    pdf_path: str | Path,
    out_dir: str | Path,
    dpi: int = PDF_DPI,
) -> List[Path]:
    """
    Convert a PDF and save each page as a PNG file to disk.

    Returns:
        List of Paths to the saved PNG files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(pdf_path).stem
    saved_paths: List[Path] = []

    for page_num, img in pdf_to_images(pdf_path, dpi):
        out_path = out_dir / f"{stem}_page_{page_num:04d}.png"
        img.save(str(out_path), "PNG")
        saved_paths.append(out_path)

    return saved_paths


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_converter.py <path_to_pdf> [output_dir]")
        sys.exit(1)
    pdf = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "page_images"
    paths = pdf_to_images_save(pdf, out)
    print(f"Converted {len(paths)} pages → {out}")
