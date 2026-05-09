"""
Step 4 & 5 — Post-processing, Cleaning, and Layout Reconstruction
"""
import re
from typing import List, Dict, Any

# Import our artifact cleaner (strips HTML tags, foreign script, page numbers)
try:
    from pipeline.text_cleaner import clean_text as _deep_clean
except ImportError:
    from text_cleaner import clean_text as _deep_clean

class Cleaner:
    """Basic rule-based cleaning for OCR artifacts and Unicode normalization."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        # 1. Deep clean: strip HTML tags, foreign-script hallucinations, page numbers
        text = _deep_clean(text)
        # 2. Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        # 3. Fix common punctuation spacing
        text = re.sub(r' ([.,!?])', r'\1', text)
        return text.strip()

    @staticmethod
    def calculate_confidence(lines: List[Dict[str, Any]]) -> float:
        """Calculate average confidence for the page."""
        confs = [l["confidence"] for l in lines if l.get("confidence") is not None]
        if not confs:
            return 0.0
        return sum(confs) / len(confs)


class LayoutEngine:
    """
    Handles simple layout reconstruction.
    For Phase 1 (Single Column), this is mostly paragraph grouping.
    """
    
    @staticmethod
    def reconstruct_paragraphs(lines: List[Dict[str, Any]]) -> str:
        """
        Group lines into paragraphs.
        In a real scenario, this uses bounding box distances.
        For basic single-column, we look at line endings or indentation.
        Since Surya returns `ocr_without_boxes` in a readable order, we'll
        do a naive join for now, leaving room for bbox-based grouping later.
        """
        if not lines:
            return ""
            
        paragraphs = []
        current_para = []
        
        for line in lines:
            text = line["text"].strip()
            if not text:
                continue
                
            current_para.append(text)
            
            # Simple heuristic: if line ends with full stop (or Tamil equivalent ending)
            # consider it end of paragraph. (Tamil uses standard full stop '.')
            if text.endswith(('.', '!', '?', '။')):
                paragraphs.append(" ".join(current_para))
                current_para = []
                
        if current_para:
             paragraphs.append(" ".join(current_para))
             
        return "\n\n".join(paragraphs)
