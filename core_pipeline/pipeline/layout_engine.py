"""
Layout Analysis Engine — Phase 1 (Single-Column Book Scans)
============================================================
Uses Surya's LayoutPredictor to detect the structure of each page
before OCR runs. This gives us:

  1. Block-level bounding boxes with semantic labels
     (Text, SectionHeader, PageHeader, PageFooter, Table, Picture …)
  2. Reading order (position index) so text is assembled correctly
  3. A structured page representation that separates headers from body text,
     enabling better paragraph reconstruction and downstream NLP

Design decisions
----------------
- LayoutPredictor is initialized once and reused (expensive GPU load).
- It shares the same FoundationPredictor backbone as OCREngine to avoid
  loading the same weights twice.
- For single-column books (our Phase 1 dataset) the predictor consistently
  returns Text + optional PageHeader blocks — we handle both.
- For blocks the model ignores (e.g. page number area), the fallback is
  to treat the whole page as a single Text region and run OCR on it.
- Block-aware OCR:  crop each detected block → run OCR on the crop → join
  results in reading order.

Supported layout label mapping (from LAYOUT_PRED_RELABEL):
  PageHeader, PageFooter, SectionHeader, Text, Caption, ListItem,
  Table, Picture, Figure, Equation, Code, Form, TableOfContents, Footnote
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from PIL import Image

# ── Label groupings ────────────────────────────────────────────────────────
# Labels whose text content we want to extract via OCR
TEXT_LABELS = {"Text", "SectionHeader", "Caption", "ListItem",
               "Footnote", "TableOfContents", "Form", "Code"}

# Labels whose text should be treated as a heading / separator
HEADER_LABELS = {"SectionHeader", "PageHeader"}

# Labels we deliberately skip for text extraction
SKIP_LABELS = {"Picture", "Figure", "Table", "Equation"}

# Header/footer labels that usually contain page numbers — skip or flag
STRUCTURAL_LABELS = {"PageHeader", "PageFooter"}


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class LayoutBlock:
    """A single detected layout block on the page."""
    label: str                    # Semantic label (Text, SectionHeader, …)
    position: int                 # Reading order index (0 = first)
    bbox: List[float]             # [x0, y0, x1, y1] in image pixels
    confidence: float             # Top confidence for this label
    is_header: bool = False
    is_skip: bool = False         # True for Pictures, Tables, etc.
    ocr_text: str = ""            # Filled in after OCR step


@dataclass
class PageLayout:
    """Complete layout analysis result for one page."""
    page_num: int
    image_size: tuple             # (width, height) of the source image
    blocks: List[LayoutBlock] = field(default_factory=list)
    fallback: bool = False        # True if layout detection found nothing useful

    # ── convenience accessors ──────────────────────────────────────────
    @property
    def text_blocks(self) -> List[LayoutBlock]:
        """Blocks that carry readable text, in correct reading order."""
        candidates = [b for b in self.blocks if not b.is_skip]
        # We use a robust column-aware sort instead of just trusting Surya's 'position'
        return LayoutEngine._sort_blocks_reading_order(candidates)

    @property
    def body_text(self) -> str:
        """
        Reconstruct the page text from all OCR-ed blocks in reading order.
        Headers get a blank line before them; body paragraphs are separated
        by a blank line to preserve the book's paragraph structure.
        """
        parts = []
        for block in self.text_blocks:
            text = block.ocr_text.strip()
            if not text:
                continue
            if block.is_header:
                # Header: blank line above + blank line below
                parts.append(f"\n{text}\n")
            else:
                parts.append(text)
        return "\n\n".join(parts).strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_num": self.page_num,
            "image_size": list(self.image_size),
            "fallback": self.fallback,
            "blocks": [
                {
                    "label": b.label,
                    "position": b.position,
                    "bbox": [round(x) for x in b.bbox],
                    "confidence": round(b.confidence, 4),
                    "is_header": b.is_header,
                    "is_skip": b.is_skip,
                    "ocr_text": b.ocr_text,
                }
                for b in self.blocks
            ],
        }


# ── Layout Engine ──────────────────────────────────────────────────────────

class LayoutEngine:
    """
    Wraps Surya's LayoutPredictor and adds block-aware OCR orchestration.

    Usage
    -----
        engine = LayoutEngine(foundation_predictor=fp)
        layout = engine.analyze_page(image, page_num=1)
        # layout.blocks  ← detected blocks
        # layout.body_text  ← assembled text (after OCR step fills ocr_text)
    """

    def __init__(self, foundation_predictor):
        from surya.layout import LayoutPredictor
        self._predictor = LayoutPredictor(foundation_predictor=foundation_predictor)
        self._initialized = True

    # ── Block deduplication ────────────────────────────────────────────

    @staticmethod
    def _iou(a: List[float], b: List[float]) -> float:
        """Intersection-over-Union of two [x0,y0,x1,y1] bboxes."""
        ix0 = max(a[0], b[0])
        iy0 = max(a[1], b[1])
        ix1 = min(a[2], b[2])
        iy1 = min(a[3], b[3])
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _is_page_number_text(text: str) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        if re.fullmatch(r"[0-9]+", text):
            return True
        if re.fullmatch(r"(?i)(page|p|பக்கம்|p\.)\s*[0-9]+", text):
            return True
        if re.fullmatch(r"[ivxlcdm]+", text.lower()):
            return True
        return False

    @staticmethod
    def _deduplicate_blocks(
        blocks: List[LayoutBlock],
        iou_threshold: float = 0.70,
    ) -> List[LayoutBlock]:
        """
        Remove near-duplicate blocks (same bbox, different label/conf).
        Keeps the highest-confidence block when IoU > threshold.
        """
        if len(blocks) <= 1:
            return blocks

        # Sort by confidence descending — best blocks get priority
        ranked = sorted(blocks, key=lambda b: b.confidence, reverse=True)
        accepted = []

        for candidate in ranked:
            is_dup = False
            for existing in accepted:
                if LayoutEngine._iou(candidate.bbox, existing.bbox) > iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                accepted.append(candidate)

        # Restore reading order by position
        accepted.sort(key=lambda b: b.position)
        return accepted

    @staticmethod
    def _sort_blocks_reading_order(blocks: List[LayoutBlock], estimated_columns: int = 1) -> List[LayoutBlock]:
        """
        Sort layout blocks in natural reading order using Recursive X-Y Cut.

        Algorithm:
          1. Project all blocks onto the X axis.
          2. Look for a "gutter" — a vertical gap spanning the full height
             of the region with NO block overlapping it.
          3. If a gutter exists → split into left column and right column.
             Recursively sort each column top-to-bottom, then concatenate
             (left column first, right column second).
          4. If no gutter → fall back to pure Y sort (single-column page).

        This correctly handles:
          - Single-column storybooks (falls through to Y-sort)
          - Two-column academic / newspaper layouts
          - Mixed pages (e.g., full-width header above two-column body)

        No ML required — purely geometric, zero GPU cost.
        """
        if len(blocks) <= 1:
            return blocks

        return LayoutEngine._xy_cut(blocks, estimated_columns=estimated_columns)

    @staticmethod
    def _xy_cut(blocks: List[LayoutBlock], estimated_columns: int = 1) -> List[LayoutBlock]:
        """
        Core recursive X-Y Cut implementation.

        Args:
            blocks: Blocks within a region to be sorted.

        Returns:
            Sorted list in reading order.
        """
        if len(blocks) <= 1:
            return blocks

        # ── Step 1: Try a vertical cut (find a column gutter) ──────────────
        # A valid vertical cut at x_cut means NO block has its x0 < x_cut < x1
        # AND the cut spans the full height of the region without interruption.

        x_coords = sorted({b.bbox[0] for b in blocks} | {b.bbox[2] for b in blocks})
        region_y0 = min(b.bbox[1] for b in blocks)
        region_y1 = max(b.bbox[3] for b in blocks)

        vertical_cut = None
        
        # If the Router tells us this is multi-column, we allow a small amount 
        # of horizontal bounding box overlap (noise tolerance) when looking for the gutter.
        overlap_tolerance = 15 if estimated_columns > 1 else 0
        
        for x in x_coords:
            # Check: does any block span across this x position?
            crosses = any(b.bbox[0] + overlap_tolerance < x < b.bbox[2] - overlap_tolerance for b in blocks)
            if crosses:
                continue

            # Check: are there blocks on BOTH sides of this x?
            left_blocks  = [b for b in blocks if b.bbox[2] <= x]
            right_blocks = [b for b in blocks if b.bbox[0] >= x]

            if not left_blocks or not right_blocks:
                continue

            # Valid gutter found — prefer the cut that gives the most
            # balanced split (near the horizontal centre of the region)
            region_x0 = min(b.bbox[0] for b in blocks)
            region_x1 = max(b.bbox[2] for b in blocks)
            centre    = (region_x0 + region_x1) / 2

            # Accept the first gutter that's reasonably near centre
            # (within the middle 70% of the page width)
            margin = (region_x1 - region_x0) * 0.15
            if region_x0 + margin < x < region_x1 - margin:
                vertical_cut = x
                break

        if vertical_cut is not None:
            left_blocks  = [b for b in blocks if b.bbox[2] <= vertical_cut]
            right_blocks = [b for b in blocks if b.bbox[0] >= vertical_cut]
            # Blocks that straddle the cut (e.g., full-width headers) go first
            straddle     = [b for b in blocks
                            if b not in left_blocks and b not in right_blocks]

            # Sort straddling blocks (full-width elements like chapter headings)
            # by Y position — they come before the columns they head
            straddle_sorted = sorted(straddle, key=lambda b: b.bbox[1])

            return (straddle_sorted +
                    LayoutEngine._xy_cut(sorted(left_blocks,  key=lambda b: b.bbox[1]), estimated_columns) +
                    LayoutEngine._xy_cut(sorted(right_blocks, key=lambda b: b.bbox[1]), estimated_columns))

        # ── Step 2: No vertical gutter found → try horizontal cut ──────────
        # A horizontal cut separates full-width bands (e.g., article headline
        # above a two-column body). This handles newspaper-style layouts.

        y_coords = sorted({b.bbox[1] for b in blocks} | {b.bbox[3] for b in blocks})
        horizontal_cut = None
        for y in y_coords:
            crosses = any(b.bbox[1] < y < b.bbox[3] for b in blocks)
            if crosses:
                continue
            top_blocks    = [b for b in blocks if b.bbox[3] <= y]
            bottom_blocks = [b for b in blocks if b.bbox[1] >= y]
            if top_blocks and bottom_blocks:
                horizontal_cut = y
                break  # Take the first (topmost) horizontal cut

        if horizontal_cut is not None:
            top_blocks    = [b for b in blocks if b.bbox[3] <= horizontal_cut]
            bottom_blocks = [b for b in blocks if b.bbox[1] >= horizontal_cut]
            return (LayoutEngine._xy_cut(top_blocks, estimated_columns) +
                    LayoutEngine._xy_cut(bottom_blocks, estimated_columns))

        # ── Step 3: No clean cut found → pure Y sort (single column) ───────
        return sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))

    # ── Core analysis ──────────────────────────────────────────────────

    def analyze_page(
        self,
        image: Image.Image,
        page_num: int = 0,
        estimated_columns: int = 1,
        min_confidence: float = 0.30,
    ) -> PageLayout:
        """
        Run layout detection on a single page image.

        Args:
            image:          Full-page PIL Image (RGB).
            page_num:       1-indexed page number (for reporting only).
            min_confidence: Blocks below this confidence are discarded.

        Returns:
            PageLayout with detected blocks (ocr_text fields are empty —
            call fill_ocr_text() next to populate them).
        """
        results = self._predictor([image])
        raw = results[0]

        layout = PageLayout(page_num=page_num, image_size=image.size)

        usable = []
        for box in raw.bboxes:
            conf = max(box.top_k.values()) if box.top_k else 0.0
            if conf < min_confidence:
                continue

            block = LayoutBlock(
                label=box.label,
                position=box.position,
                bbox=box.bbox,
                confidence=conf,
                is_header=(box.label in HEADER_LABELS),
                is_skip=(box.label in SKIP_LABELS),
            )
            usable.append(block)

        # Check if we have any actual text blocks.
        # If the model found blocks, but they are ALL 'Picture' or 'Figure', 
        # it probably misclassified a dense newspaper page. Trigger fallback.
        has_text = any(b.label in TEXT_LABELS for b in usable)

        if not usable or not has_text:
            if usable and not has_text:
                print(f"[LayoutEngine] Page {page_num}: Only non-text blocks found. Triggering full-page fallback.")
            
            # Fallback: treat entire page as a single Text block
            w, h = image.size
            fallback_block = LayoutBlock(
                label="Text",
                position=0,
                bbox=[0, 0, w, h],
                confidence=1.0,
                is_header=False,
                is_skip=False,
            )
            layout.blocks = [fallback_block]
            layout.fallback = True
        else:
            # Deduplicate overlapping blocks (collapses 41 dupes → 1)
            deduped = self._deduplicate_blocks(usable, iou_threshold=0.60 if estimated_columns > 1 else 0.70)
            # Apply X-Y Cut reading order: handles both single-column
            # (falls through to Y-sort) and two-column layouts correctly.
            layout.blocks = self._sort_blocks_reading_order(deduped, estimated_columns=estimated_columns)

        return layout

    def analyze_batch(
        self,
        images: List[Image.Image],
        start_page: int = 1,
        min_confidence: float = 0.30,
    ) -> List[PageLayout]:
        """Analyze a list of images in one GPU batch."""
        results = self._predictor(images)
        layouts = []
        for i, (raw, image) in enumerate(zip(results, images)):
            page_num = start_page + i
            layout = PageLayout(page_num=page_num, image_size=image.size)

            usable = []
            for box in raw.bboxes:
                conf = max(box.top_k.values()) if box.top_k else 0.0
                if conf < min_confidence:
                    continue
                block = LayoutBlock(
                    label=box.label,
                    position=box.position,
                    bbox=box.bbox,
                    confidence=conf,
                    is_header=(box.label in HEADER_LABELS),
                    is_skip=(box.label in SKIP_LABELS),
                )
                usable.append(block)

            if not usable:
                w, h = image.size
                layout.blocks = [LayoutBlock(
                    label="Text", position=0, bbox=[0, 0, w, h],
                    confidence=1.0
                )]
                layout.fallback = True
            else:
                layout.blocks = usable

            layouts.append(layout)
        return layouts

    # ── Block cropping ─────────────────────────────────────────────────

    @staticmethod
    def crop_block(image: Image.Image, block: LayoutBlock) -> Image.Image:
        """
        Crop the image to the block's bounding box.
        Adds a small padding to avoid clipping descenders / strokes.
        """
        PADDING = 4
        w, h = image.size
        x0 = max(0, block.bbox[0] - PADDING)
        y0 = max(0, block.bbox[1] - PADDING)
        x1 = min(w, block.bbox[2] + PADDING)
        y1 = min(h, block.bbox[3] + PADDING)
        return image.crop((x0, y0, x1, y1))

    # ── OCR integration ────────────────────────────────────────────────

    def fill_ocr_text(
        self,
        layout: PageLayout,
        image: Image.Image,
        ocr_engine,
        cleaner_fn=None,
    ) -> None:
        """
        For each non-skipped block in the layout, crop the block from the
        image and run OCR on it, storing the result in block.ocr_text.

        For single-column books where the predictor returns 1 full-page block,
        this is equivalent to running OCR on the whole page — no quality loss.

        Args:
            layout:     PageLayout from analyze_page().
            image:      Original full-page PIL Image.
            ocr_engine: Instance of OCREngine (has process_image()).
            cleaner_fn: Optional function(str) -> str for post-OCR cleaning.
        """
        for block in layout.text_blocks:
            if block.is_skip:
                continue

            crop = self.crop_block(image, block)
            result = ocr_engine.process_image(crop)
            text = result.get("text", "").strip()

    @staticmethod
    def reconstruct_page_text(ocr_lines: List[Dict[str, Any]], layout: PageLayout) -> str:
        """
        Maps OCR text lines back to their corresponding layout blocks using bounding box IoU.
        Filters out structural elements (headers/footers) and applies basic formatting.

        Args:
            ocr_lines: List of dicts, e.g. [{"text": "Hello", "bbox": [0,0,10,10], "confidence": 0.99}, ...]
            layout: PageLayout object containing detected blocks.

        Returns:
            A formatted string reconstructing the valid text on the page.
        """
        # For both fallback and non-fallback, we filter all text lines through
        # the same pipeline. This ensures page numbers and structural headers
        # are handled consistently regardless of whether layout detection worked.
        all_lines = [l.get("text", "").strip() for l in ocr_lines if l.get("text", "").strip()]

        if layout.fallback or not layout.blocks:
            # Fallback: join all lines (TextFormatter will handle page-number
            # and running-header stripping at the document level)
            return "\n".join(all_lines)

        # Map each text line to the layout block it overlaps with most
        assigned_blocks = {i: [] for i in range(len(layout.blocks))}
        unassigned_lines = []

        for line in ocr_lines:
            text = line.get("text", "").strip()
            bbox = line.get("bbox")
            
            if not text:
                continue

            if not bbox:
                # If no bbox, assign to the first available block to prevent dropping
                if len(layout.blocks) > 0:
                    assigned_blocks[0].append(text)
                continue

            best_iou = 0.0
            best_block_idx = -1

            for i, block in enumerate(layout.blocks):
                # Check for intersection ratio relative to the text line's area
                line_area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
                
                if line_area > 0:
                    ix0 = max(bbox[0], block.bbox[0])
                    iy0 = max(bbox[1], block.bbox[1])
                    ix1 = min(bbox[2], block.bbox[2])
                    iy1 = min(bbox[3], block.bbox[3])
                    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                    intersection_ratio = inter / line_area
                else:
                    intersection_ratio = 0.0

                if intersection_ratio > best_iou:
                    best_iou = intersection_ratio
                    best_block_idx = i

            # If the text line overlaps at least 50% with a layout block, assign it.
            if best_iou > 0.50 and best_block_idx != -1:
                assigned_blocks[best_block_idx].append(text)
            else:
                unassigned_lines.append({"text": text, "bbox": bbox})

        # We now have text for each block, and text for unassigned lines.
        # We need to sort ALL of them together to preserve natural reading order.
        chunks_to_sort = []

        for i, block in enumerate(layout.blocks):
            lines_in_block = assigned_blocks[i]
            if not lines_in_block:
                continue

            block_text = "\n".join(lines_in_block).strip()
            if not block_text:
                continue

            if block.label in STRUCTURAL_LABELS:
                # Keep page headers if they are real body text rather than simple page numbers.
                if block.label == "PageHeader" and not LayoutEngine._is_page_number_text(block_text):
                    chunks_to_sort.append({"text": block_text, "bbox": block.bbox})
                continue

            if block.is_skip:
                pass
            else:
                chunks_to_sort.append({"text": block_text, "bbox": block.bbox})

        # Include unassigned lines with their original bounding boxes
        for line in unassigned_lines:
            chunks_to_sort.append({"text": line["text"], "bbox": line["bbox"]})

        # To sort these chunks properly, we wrap them in dummy LayoutBlocks and use _xy_cut
        dummy_blocks = []
        text_map = {}
        for idx, chunk in enumerate(chunks_to_sort):
            b = LayoutBlock(label="Text", position=idx, bbox=chunk["bbox"], confidence=1.0)
            dummy_blocks.append(b)
            text_map[id(b)] = chunk["text"]

        # Run XY cut to determine reading order among these chunks
        sorted_dummy = LayoutEngine._xy_cut(dummy_blocks)
        
        parts = [text_map[id(b)] for b in sorted_dummy]
        return "\n\n".join(parts).strip()


# ── CLI smoke-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    sys.stdout.reconfigure(encoding="utf-8")

    from surya.foundation import FoundationPredictor

    TEST_IMAGE = (
        r"OCR/multilingual_ocr_pipeline/data/page_images/எலி/page_001_raw.png"
    )

    print("Loading models…")
    fp = FoundationPredictor(device="cuda")
    engine = LayoutEngine(foundation_predictor=fp)
    print("Ready.\n")

    img = Image.open(TEST_IMAGE).convert("RGB")
    layout = engine.analyze_page(img, page_num=1)

    print(f"Page 1  |  {len(layout.blocks)} block(s)  |  fallback={layout.fallback}")
    for b in layout.blocks:
        print(
            f"  [{b.label:>16}]  pos={b.position}  "
            f"conf={b.confidence:.2f}  "
            f"bbox={[int(x) for x in b.bbox]}"
        )

    print("\nJSON dump:")
    print(json.dumps(layout.to_dict(), ensure_ascii=False, indent=2))
