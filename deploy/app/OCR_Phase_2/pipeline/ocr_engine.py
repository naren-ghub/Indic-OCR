"""
OCR Engine — Abstract Base + Surya + Factory
=============================================
Phase 2 engine architecture using abstract factory pattern.
All engines return a unified output schema so the rest of the pipeline
is completely engine-agnostic.

Note: PaddleOCR support is deferred to Phase 3.

Unified output schema:
    {
        "lines": [
            {"text": str, "bbox": [x1, y1, x2, y2], "confidence": float},
            ...
        ],
        "page_confidence": float,   # mean of all line confidences
        "engine": str               # "surya"
    }
"""
from __future__ import annotations

import gc
import sys
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from PIL import Image


# ═══════════════════════════════════════════════════════════════════════════════
#  Abstract Base Class
# ═══════════════════════════════════════════════════════════════════════════════

class BaseOCREngine(ABC):
    """Engine-agnostic interface. All engines return the same output schema."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier: 'surya' or 'paddle'."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether models are currently resident in VRAM."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Explicitly load models into VRAM."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Explicitly release VRAM. Called by MemoryManager before switching engines."""
        ...

    @abstractmethod
    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run OCR on a single PIL Image.

        Returns:
            {
                "lines": [
                    {"text": str, "bbox": [x1, y1, x2, y2], "confidence": float},
                    ...
                ],
                "page_confidence": float,
                "engine": str
            }
        """
        ...

    def process_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Run OCR on a batch of PIL Images. Subclasses can override for true batching."""
        return [self.process_image(img) for img in images]

    @staticmethod
    def _mean_confidence(lines: List[Dict[str, Any]]) -> float:
        """Compute mean confidence from a list of line dicts."""
        confs = [l["confidence"] for l in lines if l.get("confidence") is not None]
        return sum(confs) / len(confs) if confs else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Surya Engine
# ═══════════════════════════════════════════════════════════════════════════════

class SuryaEngine(BaseOCREngine):
    """
    Primary OCR engine using Surya OCR.
    Supports shared FoundationPredictor to avoid duplicate VRAM usage
    when LayoutEngine also needs the backbone.
    """

    def __init__(
        self,
        langs: List[str] | None = None,
        device: str = "cuda",
        foundation_predictor=None,
        drop_repeated_text: bool = True,
    ):
        self.langs = langs or ["ta", "en"]
        self.device = device
        self._shared_fp = foundation_predictor   # external FP, NOT owned by us
        self._foundation_predictor = None
        self._recognition_predictor = None
        self._detection_predictor = None
        self._drop_repeated_text = drop_repeated_text
        self._loaded = False

        # Auto-load if a shared FP was provided (backward compat with Phase 1)
        if self._shared_fp is not None:
            self.load()

    @property
    def name(self) -> str:
        return "surya"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return

        try:
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor
        except ImportError:
            raise RuntimeError("surya-ocr is not installed. Run: pip install surya-ocr")

        if self._shared_fp is not None:
            self._foundation_predictor = self._shared_fp
        else:
            print(f"[SuryaEngine] Loading FoundationPredictor on {self.device}...")
            self._foundation_predictor = FoundationPredictor(device=self.device)

        self._recognition_predictor = RecognitionPredictor(
            foundation_predictor=self._foundation_predictor
        )
        self._detection_predictor = DetectionPredictor(device=self.device)
        self._loaded = True
        print("[SuryaEngine] Models loaded.")

    def unload(self) -> None:
        if not self._loaded:
            return

        import torch

        # Only delete predictors we own (don't delete shared FP)
        del self._recognition_predictor
        del self._detection_predictor
        self._recognition_predictor = None
        self._detection_predictor = None

        if self._shared_fp is None and self._foundation_predictor is not None:
            del self._foundation_predictor
            self._foundation_predictor = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False
        print("[SuryaEngine] Models unloaded.")

    def process_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        if not self._loaded:
            self.load()
        if not self._loaded or self._recognition_predictor is None or self._detection_predictor is None:
            raise RuntimeError("SuryaEngine is not loaded; cannot run OCR.")

        predictions = self._recognition_predictor(
            images=images,
            langs=[self.langs] * len(images),
            task_names=["ocr_without_boxes"] * len(images),
            det_predictor=self._detection_predictor,
        )

        results = []
        for res in predictions:
            lines_data = []
            if res.text_lines:
                for line in res.text_lines:
                    lines_data.append({
                        "text": line.text,
                        "bbox": getattr(line, "bbox", None),
                        "confidence": getattr(line, "confidence", None),
                    })

            page_conf = self._mean_confidence(lines_data)
            results.append({
                "lines": lines_data,
                "page_confidence": round(page_conf, 4),
                "engine": "surya",
                "raw_result": res,
            })
        return results

    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        return self.process_batch([image])[0]


# ═══════════════════════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_engine(name: str, **kwargs) -> BaseOCREngine:
    """
    Factory to create an OCR engine by name.

    Args:
        name: "surya" (only supported engine in Phase 2)
        **kwargs: passed to the engine constructor

    Returns:
        An instance of BaseOCREngine
    """
    engines = {
        "surya": SuryaEngine,
    }
    if name not in engines:
        raise ValueError(
            f"Unknown engine '{name}'. Available: {list(engines.keys())}. "
            f"Note: PaddleOCR support is deferred to Phase 3."
        )
    return engines[name](**kwargs)
