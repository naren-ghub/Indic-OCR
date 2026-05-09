"""
Tamil OCR Noise Generator — Phase 2 / Step 2.3a
=================================================
Generates synthetic (noisy_ocr, clean_ground_truth) sentence pairs
for fine-tuning the ByT5 correction model.

Strategy:
  - Tamil-specific character confusion tables based on real OCR error analysis
  - Vowel mark (matra) drops — the most common single error in Tamil OCR
  - Consonant cluster confusion (visually similar strokes)
  - Word boundary errors (space insertion / deletion)
  - Pulli (virama) drop — changes consonant pronunciation entirely

All error patterns are derived from the Phase 1 weakness analysis
of Surya OCR output on scanned Tamil books.

Usage:
    from pipeline.noise_generator import TamilOCRNoiseGenerator
    gen = TamilOCRNoiseGenerator(error_rate=0.15)
    noisy = gen.corrupt(clean_sentence)
"""
from __future__ import annotations

import random
import unicodedata
from typing import List, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
#  Tamil Unicode Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Tamil vowel marks (matras / uyirmei) — U+0BBE to U+0BCD
TAMIL_VOWEL_MARKS = [
    "\u0BBE",  # ா  (aa)
    "\u0BBF",  # ி  (i)
    "\u0BC0",  # ீ  (ii)
    "\u0BC1",  # ு  (u)
    "\u0BC2",  # ூ  (uu)
    "\u0BC6",  # ெ  (e)
    "\u0BC7",  # ே  (ee)
    "\u0BC8",  # ை  (ai)
    "\u0BCA",  # ொ  (o)
    "\u0BCB",  # ோ  (oo)
    "\u0BCC",  # ௌ  (au)
    "\u0BCD",  # ்  (pulli / virama)
]

# Pulli (virama) — makes consonant unpronounced, critical diacritic
PULLI = "\u0BCD"  # ்

# Short vs long vowel mark confusions (Surya's #1 error class)
SHORT_LONG_CONFUSIONS = [
    ("\u0BBF", "\u0BC0"),   # ி ↔ ீ  (ki → kii)
    ("\u0BC1", "\u0BC2"),   # ு ↔ ூ  (ku → kuu)
    ("\u0BC6", "\u0BC7"),   # ெ ↔ ே  (ke → kee)
    ("\u0BCA", "\u0BCB"),   # ொ ↔ ோ  (ko → koo)
]

# Consonant visual confusion pairs — look very similar in degraded scans
# Based on Phase 1 Surya OCR error analysis and Industry Standards
CONSONANT_CONFUSIONS = [
    # Char A      Char B       Notes
    ("\u0BB4",  "\u0BB3"),   # ழ ↔ ள  (zha vs lla — most frequent Tamil OCR error)
    ("\u0BA3",  "\u0BA9"),   # ண ↔ ன  (retroflex n vs alveolar n)
    ("\u0BB1",  "\u0BB0"),   # ற ↔ ர  (rra vs ra)
    ("\u0BA8",  "\u0BAE"),   # ந ↔ ம  (na vs ma — stroke similarity)
    ("\u0B95",  "\u0B99"),   # க ↔ ங  (ka vs nga)
    ("\u0B9A",  "\u0B9E"),   # ச ↔ ஞ  (cha vs nya)
    ("\u0BB5",  "\u0BAA"),   # வ ↔ ப  (va vs pa — Box group)
    ("\u0BB5",  "\u0BB2"),   # வ ↔ ல  (va vs la — Box group)
    ("\u0BAA",  "\u0BB2"),   # ப ↔ ல  (pa vs la — Box group)
    ("\u0B90",  "\u0B9C"),   # ஐ ↔ ஜ  (ai vs ja — Loop group)
    ("\u0B95",  "\u0B9A"),   # க ↔ ச  (ka vs cha — Arch group)
    ("\u0BA4",  "\u0BA8"),   # த ↔ ந  (ta vs na — Arch group)
    ("\u0BA4",  "\u0BA4\u0BC1"),  # த ↔ து  (ta vs tu — vowel mark hallucination)
    ("\u0BB2",  "\u0BB3"),   # ல ↔ ள  (la vs lla)
]

# Spurious diacritics that OCR engines often hallucinate due to noise/dirt.
# Examples: "ுநலீன", "ிந்த", standalone "்" or "ு"
SPURIOUS_MARKS = [
    "\u0BC1",  # ு (u)
    "\u0BBF",  # ி (i)
    "\u0BCD",  # ் (pulli)
    "\u0BC2",  # ூ (uu)
]

# Common word-boundary errors seen in dense/newspaper scans
# e.g. "இந்த நாள்" → "இந்தநாள்" (space dropped)
# or "அவனுக்கு" → "அவனு க்கு" (space inserted mid-word)


# ═══════════════════════════════════════════════════════════════════════════════
#  Noise Generator Class
# ═══════════════════════════════════════════════════════════════════════════════

class TamilOCRNoiseGenerator:
    """
    Generates realistic Tamil OCR noise by applying a configurable mix of
    error types derived from Phase 1 Surya OCR failure analysis.
    """

    def __init__(
        self,
        error_rate: float = 0.15,
        seed: int | None = None,
    ):
        """
        Args:
            error_rate: Fraction of characters to corrupt (0.0–1.0).
                        0.15 = ~15% error rate, mimicking a mediocre scan.
            seed:       Random seed for reproducibility.
        """
        self.error_rate = error_rate
        if seed is not None:
            random.seed(seed)

        # Build fast-lookup dicts from confusion tables
        self._consonant_map: dict[str, str] = {}
        for a, b in CONSONANT_CONFUSIONS:
            self._consonant_map[a] = b
            self._consonant_map[b] = a

        self._short_long_map: dict[str, str] = {}
        for a, b in SHORT_LONG_CONFUSIONS:
            self._short_long_map[a] = b
            self._short_long_map[b] = a

    # ── Public API ──────────────────────────────────────────────────────────

    def corrupt(self, text: str) -> str:
        """
        Apply realistic OCR noise to clean Tamil text.

        Applies a weighted mix of:
          - Consonant visual confusion (ழ↔ள, ண↔ன, etc.)
          - Vowel mark short/long swap (ி↔ீ, ு↔ூ, etc.)
          - Pulli (virama) drop — changes word meaning entirely
          - Vowel mark deletion
          - Space insertion/deletion at word boundaries

        Args:
            text: Clean Tamil text string.

        Returns:
            Corrupted text simulating OCR errors.
        """
        if not text.strip():
            return text

        chars = list(text)
        result = []
        i = 0

        while i < len(chars):
            c = chars[i]

            if random.random() > self.error_rate:
                # No error this character — keep as-is
                result.append(c)
                i += 1
                continue

            # Choose error type with weights matching real OCR error frequencies
            error_type = random.choices(
                population=["consonant", "short_long", "pulli_drop",
                            "matra_drop", "space_insert", "space_delete",
                            "spurious_prefix", "solitary_mark"],
                weights=[25, 20, 15, 10, 5, 5, 10, 10],
                k=1,
            )[0]

            if error_type == "consonant":
                result.append(self._consonant_error(c))
                i += 1

            elif error_type == "short_long":
                result.append(self._short_long_error(c))
                i += 1

            elif error_type == "pulli_drop":
                result.append(self._pulli_drop_error(c, chars, i))
                i += 1

            elif error_type == "matra_drop":
                # Drop a vowel mark (simply skip it)
                if c in TAMIL_VOWEL_MARKS:
                    # Drop: don't append
                    i += 1
                else:
                    result.append(c)
                    i += 1

            elif error_type == "space_insert":
                # Insert a spurious space inside a Tamil word
                result.append(c)
                if self._is_tamil_char(c):
                    result.append(" ")
                i += 1

            elif error_type == "space_delete":
                # Delete a space between words
                if c == " " and i > 0 and i < len(chars) - 1:
                    # Skip the space (don't append)
                    i += 1
                else:
                    result.append(c)
                    i += 1

            elif error_type == "spurious_prefix":
                # Prepends a spurious matra to a word boundary, e.g. "ிந்த" or "ுநலீன"
                if i == 0 or chars[i-1] == " ":
                    result.append(random.choice(SPURIOUS_MARKS))
                result.append(c)
                i += 1

            elif error_type == "solitary_mark":
                # Injects a solitary matra/pulli surrounded by spaces, e.g. " ் "
                if c == " ":
                    result.append(" ")
                    result.append(random.choice(SPURIOUS_MARKS))
                    result.append(" ")
                else:
                    result.append(c)
                i += 1

            else:
                result.append(c)
                i += 1

        return "".join(result)

    def generate_pairs(
        self,
        clean_texts: List[str],
        num_pairs: int | None = None,
    ) -> List[Tuple[str, str]]:
        """
        Generate (noisy, clean) pairs from a list of clean sentences.

        Args:
            clean_texts: List of clean Tamil sentences.
            num_pairs:   If set, generate at most this many pairs
                         (resamples if needed). If None, one pair per sentence.

        Returns:
            List of (noisy_text, clean_text) tuples.
        """
        pairs = []

        if num_pairs is not None and num_pairs > len(clean_texts):
            # Resample to meet quota
            clean_texts = random.choices(clean_texts, k=num_pairs)

        for clean in clean_texts:
            clean = clean.strip()
            if not clean or not self._has_tamil(clean):
                continue
            noisy = self.corrupt(clean)
            # Only add pair if corruption actually changed something
            if noisy != clean:
                pairs.append((noisy, clean))

        return pairs

    # ── Private helpers ─────────────────────────────────────────────────────

    def _consonant_error(self, c: str) -> str:
        """Replace a consonant with its visual lookalike, or return unchanged."""
        return self._consonant_map.get(c, c)

    def _short_long_error(self, c: str) -> str:
        """Swap a short/long vowel mark, or return unchanged."""
        return self._short_long_map.get(c, c)

    def _pulli_drop_error(self, c: str, chars: list, i: int) -> str:
        """
        If this is a pulli (PULLI), drop it.
        If the next char is a pulli, consume both and return just the consonant.
        """
        if c == PULLI:
            return ""  # drop pulli
        if i + 1 < len(chars) and chars[i + 1] == PULLI:
            # Will be handled when we get to i+1 — just return c
            return c
        return c

    @staticmethod
    def _is_tamil_char(c: str) -> bool:
        return 0x0B80 <= ord(c) <= 0x0BFF

    @staticmethod
    def _has_tamil(text: str) -> bool:
        return any(0x0B80 <= ord(c) <= 0x0BFF for c in text)


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    gen = TamilOCRNoiseGenerator(error_rate=0.15, seed=42)

    test_sentences = [
        "தமிழ் மொழி உலகின் மிகப் பழமையான மொழிகளில் ஒன்று.",
        "அவன் அந்தப் பூதாகாரமான கட்டிடத்தை பார்த்தான்.",
        "கண்ணரடி டாக்ஸி ஓட்டுநர் நேரம் தாழ்த்தி வந்தான்.",
        "இந்த நாள் மிகவும் முக்கியமானது என்று அவள் சொன்னாள்.",
        "ழ ள ண ன ற ர — visually similar pairs test",
    ]

    print("=" * 60)
    print("TamilOCRNoiseGenerator — Sample Output")
    print("=" * 60)
    for clean in test_sentences:
        noisy = gen.corrupt(clean)
        print(f"\n CLEAN : {clean}")
        print(f" NOISY : {noisy}")
        # Show diff positions
        diffs = [(i, c, n) for i, (c, n) in enumerate(zip(clean, noisy)) if c != n]
        if diffs:
            print(f" DIFFS : {len(diffs)} char(s) changed")
