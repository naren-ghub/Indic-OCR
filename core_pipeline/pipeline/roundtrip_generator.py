"""
RoundTrip OCR Generator — Phase 2
===================================
Generates highly authentic synthetic training pairs for ByT5 by rendering
clean text into an image, aggressively degrading the image (blur, noise,
erosion/dilation), and then running the actual OCR engine on it.

This forces the OCR engine to make natural mistakes (like hallucinating
spurious diacritics) based on real computer vision failures.

Usage:
    generator = RoundTripOCRGenerator(engine=surya_engine)
    noisy_text = generator.generate_single(clean_text)
"""
from __future__ import annotations

import os
import cv2
import numpy as np
import random
import requests
from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING, Iterable
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from pipeline.ocr_engine import BaseOCREngine

# Font download URL (Noto Sans Tamil)
FONT_URL = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansTamil/NotoSansTamil-Regular.ttf"
FONT_DIR = Path(__file__).parent.parent / "data" / "fonts"
FONT_PATH = FONT_DIR / "NotoSansTamil-Regular.ttf"


class RoundTripOCRGenerator:
    def __init__(
        self,
        engine: BaseOCREngine,
        font_size: int = 32,
        font_path: str | Path | None = None,
    ):
        self.engine = engine
        self.font_size = font_size
        self.font_path = self._resolve_font(font_path)
        self.font = ImageFont.truetype(str(self.font_path), self.font_size)

    @staticmethod
    def _iter_local_font_candidates() -> Iterable[Path]:
        """
        Local font fallbacks for offline/locked-down environments.
        Prefer Tamil-capable system fonts on Windows.
        """
        # 1) Anything already dropped into OCR_Phase_2/data/fonts
        if FONT_DIR.exists():
            for ext in ("*.ttf", "*.otf", "*.ttc"):
                for p in FONT_DIR.glob(ext):
                    yield p

        # 2) Common Windows Tamil fonts
        win_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        for name in (
            "latha.ttf",
            "lathab.ttf",
            "Nirmala.ttf",
            "NirmalaUI.ttf",
            "Nirmala.ttc",
        ):
            p = win_fonts / name
            if p.exists():
                yield p

    def _resolve_font(self, font_path: str | Path | None) -> Path:
        """
        Resolve a usable Tamil font path.
        Priority:
          1) explicit font_path
          2) local cached NotoSansTamil in data/fonts/
          3) download NotoSansTamil (if network available)
          4) Windows font fallbacks (Latha/Nirmala)
        """
        if font_path is not None:
            p = Path(font_path)
            if not p.exists():
                raise FileNotFoundError(f"RoundTrip font_path not found: {p}")
            return p

        if FONT_PATH.exists():
            return FONT_PATH

        # Attempt download, but gracefully fall back when offline.
        try:
            print(f"[RoundTrip] Downloading Tamil font to {FONT_PATH}...")
            FONT_DIR.mkdir(parents=True, exist_ok=True)
            response = requests.get(FONT_URL, timeout=20)
            response.raise_for_status()
            with open(FONT_PATH, "wb") as f:
                f.write(response.content)
            return FONT_PATH
        except Exception as e:
            print(f"[RoundTrip] Font download failed ({type(e).__name__}). Using local font fallback...")

        for candidate in self._iter_local_font_candidates():
            # Prefer .ttf/.otf over .ttc, unless that's all we have.
            return candidate

        raise RuntimeError(
            "No Tamil font available for RoundTrip generation.\n"
            f"- Tried download: {FONT_URL}\n"
            f"- Looked in: {FONT_DIR}\n"
            r"- Looked in: C:\Windows\Fonts (Latha/Nirmala)\n"
            "Fix: copy a Tamil-capable .ttf into OCR_Phase_2/data/fonts/ "
            "or pass --font-path to the dataset packer."
        )

    def render_text(self, text: str) -> Image.Image:
        """Render a single line of text onto a white PIL Image."""
        # Calculate bounding box
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), text, font=self.font)
        
        # Add padding (simulate margins)
        pad_x, pad_y = 40, 20
        width = bbox[2] - bbox[0] + (pad_x * 2)
        height = bbox[3] - bbox[1] + (pad_y * 2)

        # Create actual image
        img = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((pad_x, pad_y), text, font=self.font, fill="black")
        
        return img

    def apply_degradation(self, img: Image.Image) -> Image.Image:
        """
        Apply heavy computer vision degradation to mimic bad newspaper scans.
        """
        # Convert PIL to OpenCV format (BGR)
        cv_img = np.array(img)[:, :, ::-1].copy()
        
        # Random choice of degradations
        degradations = random.sample(["blur", "noise", "erosion", "dilation", "contrast"], k=random.randint(2, 4))

        if "blur" in degradations:
            # Gaussian or Motion blur
            if random.random() > 0.5:
                k = random.choice([3, 5])
                cv_img = cv2.GaussianBlur(cv_img, (k, k), 0)
            else:
                k = random.choice([3, 5])
                kernel = np.zeros((k, k))
                kernel[int((k-1)/2), :] = np.ones(k) / k
                cv_img = cv2.filter2D(cv_img, -1, kernel)

        if "erosion" in degradations:
            # Erosion makes text thinner/faded (ink starvation)
            kernel = np.ones((2, 2), np.uint8)
            cv_img = cv2.erode(cv_img, kernel, iterations=1)
            
        elif "dilation" in degradations:
            # Dilation makes text thicker/bleed together (ink bleed)
            kernel = np.ones((2, 2), np.uint8)
            cv_img = cv2.dilate(cv_img, kernel, iterations=1)

        if "noise" in degradations:
            # Salt and pepper noise (speckles/dust)
            row, col, ch = cv_img.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(cv_img)
            # Salt mode
            num_salt = np.ceil(amount * cv_img.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in cv_img.shape]
            out[tuple(coords)] = 255
            # Pepper mode
            num_pepper = np.ceil(amount * cv_img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in cv_img.shape]
            out[tuple(coords)] = 0
            cv_img = out

        if "contrast" in degradations:
            # Lower contrast (faded print)
            alpha = random.uniform(0.5, 0.8) # Contrast control (1.0-3.0)
            beta = random.randint(30, 80)    # Brightness control (0-100)
            cv_img = cv2.convertScaleAbs(cv_img, alpha=alpha, beta=beta)

        # Small rotation (-2 to +2 degrees)
        angle = random.uniform(-2, 2)
        h, w = cv_img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        # Use white border for rotation
        cv_img = cv2.warpAffine(cv_img, M, (w, h), borderValue=(255, 255, 255))

        # Convert back to PIL
        return Image.fromarray(cv_img[:, :, ::-1])

    def generate_single(self, text: str) -> str:
        """Run a single sentence through the full roundtrip."""
        return self.generate_batch([text])[0] if text.strip() else ""

    def generate_batch(self, texts: List[str]) -> List[str]:
        """Run a batch of sentences through the full roundtrip."""
        valid_texts = [t.strip() for t in texts if t.strip()]
        if not valid_texts:
            return []

        # 1. Render clean text to image
        imgs = [self.render_text(t) for t in valid_texts]

        # 2. Apply degradation
        degraded_imgs = [self.apply_degradation(img) for img in imgs]

        # 3. OCR the degraded images in batch
        results = self.engine.process_batch(degraded_imgs)
        
        noisy_texts = []
        for result in results:
            lines = result.get("lines", [])
            noisy_text = " ".join([l.get("text", "") for l in lines]).strip()
            noisy_texts.append(noisy_text)
            
        return noisy_texts

    def generate_pairs(
        self,
        clean_texts: List[str],
        max_pairs: int,
        *,
        min_edit_distance: int = 1,
        max_attempts: int | None = None,
        seed: int | None = None,
        checkpoint_file: str | Path | None = None,
    ) -> List[Tuple[str, str]]:
        """
        Generate a batch of (noisy, clean) pairs.

        Notes:
          - This method may need to attempt more than `max_pairs` sentences,
            because some roundtrips produce identical output.
          - `min_edit_distance` allows filtering out "too-clean" pairs.
        """
        pairs = []
        import tqdm
        import json

        completed_targets = set()
        if checkpoint_file is not None:
            checkpoint_path = Path(checkpoint_file)
            if checkpoint_path.exists():
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        obj = json.loads(line)
                        pairs.append((obj["input"], obj["target"]))
                        completed_targets.add(obj["target"])
                print(f"[RoundTrip] Resumed {len(pairs)} pairs from {checkpoint_path}")

        rng = random.Random(seed)
        texts = list(clean_texts)
        rng.shuffle(texts)

        if max_attempts is None:
            # Conservative cap to avoid infinite loops if OCR is too accurate.
            max_attempts = max_pairs * 25

        print(
            f"[RoundTrip] Target={max_pairs} pairs | min_edits>={min_edit_distance} | "
            f"engine={self.engine.name} | font={self.font_path.name}"
        )

        batch_size = 8
        attempts = 0
        batch_cleans = []

        def flush_batch():
            nonlocal attempts
            if not batch_cleans:
                return
            
            attempts += len(batch_cleans)
            noisy_results = self.generate_batch(batch_cleans)
            
            for noisy, clean in zip(noisy_results, batch_cleans):
                if not noisy or noisy == clean:
                    continue
                
                if self._edit_distance_at_least(noisy, clean, min_edit_distance):
                    pairs.append((noisy, clean))
                    completed_targets.add(clean)
                    if checkpoint_file is not None:
                        import json
                        with open(checkpoint_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"input": noisy, "target": clean}, ensure_ascii=False) + "\n")
            batch_cleans.clear()

        for text in tqdm.tqdm(texts):
            if len(pairs) >= max_pairs or attempts >= max_attempts:
                break

            clean = text.strip()
            if not clean or clean in completed_targets:
                continue

            batch_cleans.append(clean)
            if len(batch_cleans) >= batch_size:
                flush_batch()

        # Flush any remaining items in the final partial batch
        if len(pairs) < max_pairs and attempts < max_attempts:
            flush_batch()

        if len(pairs) < max_pairs:
            raise RuntimeError(
                f"[RoundTrip] Only generated {len(pairs)}/{max_pairs} pairs "
                f"after {attempts} attempt(s). "
                f"Try lowering min_edit_distance, increasing max_attempts, "
                f"or making degradation stronger."
            )

        return pairs

    @staticmethod
    def _edit_distance_at_least(a: str, b: str, threshold: int) -> bool:
        """
        Returns True iff Levenshtein distance(a,b) >= threshold.

        Optimized for small thresholds (e.g., 1–3) by using a banded DP;
        complexity is O(k * min(n, m)) rather than O(n*m).
        """
        if threshold <= 0:
            return True
        if a == b:
            return False
        if abs(len(a) - len(b)) >= threshold:
            return True

        # We only care whether distance >= threshold.
        # Compute bounded distance with max_dist = threshold-1.
        max_dist = threshold - 1

        # Ensure b is the shorter string to minimize work.
        if len(a) < len(b):
            a, b = b, a

        n = len(a)
        m = len(b)

        # If max_dist is 0, only exact matches have distance 0.
        if max_dist <= 0:
            return True

        # Banded DP (Ukkonen): only compute within +/- max_dist of diagonal.
        # If we can't keep distance <= max_dist, then distance >= threshold.
        INF = max_dist + 1
        prev = [INF] * (m + 1)
        for j in range(0, min(m, max_dist) + 1):
            prev[j] = j

        for i in range(1, n + 1):
            cur = [INF] * (m + 1)
            j_start = max(0, i - max_dist)
            j_end = min(m, i + max_dist)

            if j_start == 0:
                cur[0] = i

            row_min = INF
            ai = a[i - 1]
            for j in range(max(1, j_start), j_end + 1):
                cost = 0 if ai == b[j - 1] else 1
                cur[j] = min(
                    prev[j] + 1,        # deletion
                    cur[j - 1] + 1,     # insertion
                    prev[j - 1] + cost  # substitution
                )
                if cur[j] < row_min:
                    row_min = cur[j]

            if row_min > max_dist:
                return True
            prev = cur

        return prev[m] > max_dist
