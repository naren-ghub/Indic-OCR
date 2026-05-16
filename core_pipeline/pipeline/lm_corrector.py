"""
Step 5 — Language Model Correction (IndicBART)
==============================================
Uses the ai4bharat/IndicBART model to correct
character and word-level OCR errors in the extracted text.

Implementation note:
  IndicBART is run in an isolated subprocess to prevent
  conflicts with the main environment's transformers version
  and to manage CUDA memory more effectively.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

_REPO_ROOT   = Path(__file__).resolve().parents[1]   # core_pipeline/
_NLP_ROOT    = Path(__file__).resolve().parents[2]   # Indic-OCR/
_SERVER      = _REPO_ROOT / "pipeline" / "indicbart_server.py"
_VENV_PYTHON = _NLP_ROOT.parent / ".venv_byt5" / "Scripts" / "python.exe"

class LMCorrector:
    """
    Wraps the IndicBART model for local OCR correction via subprocess.
    Uses binary I/O for the pipe to avoid Windows encoding issues with Tamil text.
    """

    def __init__(self, model_id="ai4bharat/IndicBART", device="cuda"):
        if not _VENV_PYTHON.exists():
            raise RuntimeError(
                f"[LMCorrector] Isolated venv not found at {_VENV_PYTHON}\n"
            )
        print(f"[LMCorrector] Spawning IndicBART server (device={device})...")
        self._proc = subprocess.Popen(
            [str(_VENV_PYTHON), str(_SERVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
        )
        ready_raw = self._proc.stdout.readline()
        ready_line = ready_raw.decode("utf-8", errors="replace")
        try:
            msg = json.loads(ready_line)
            if msg.get("status") == "ready":
                print("[LMCorrector] IndicBART server ready.")
            else:
                raise RuntimeError(f"[LMCorrector] Unexpected startup message: {ready_line!r}")
        except json.JSONDecodeError:
            raise RuntimeError(f"[LMCorrector] Failed to start server. Got: {ready_line!r}")

    def _call_server(self, text: str, lang_code: str) -> str:
        """Send one text to the server and get back the corrected version."""
        payload = json.dumps({"text": text, "lang_code": lang_code}, ensure_ascii=False) + "\n"
        self._proc.stdin.write(payload.encode("utf-8"))
        self._proc.stdin.flush()
        response_raw = self._proc.stdout.readline()
        response_line = response_raw.decode("utf-8", errors="replace")
        response = json.loads(response_line)
        if "error" in response:
            print(f"[LMCorrector] Server error: {response['error']}", file=sys.stderr)
        return response.get("corrected", text)

    def correct_text(self, text: str, lang_code: str = "ta") -> str:
        """Takes a string of messy OCR text and returns the corrected version."""
        if not text.strip():
            return ""
        return self._call_server(text, lang_code)

    def correct_page(self, text: str, lang_code: str = "ta") -> str:
        """
        Corrects an entire page by splitting into lines to ensure
        high-quality local context correction.
        """
        lines = text.split("\n")
        corrected_lines = []
        for line in lines:
            if len(line.strip()) < 3:  # Skip very short snippets
                corrected_lines.append(line)
                continue
            corrected_lines.append(self.correct_text(line, lang_code=lang_code))
        return "\n".join(corrected_lines)

    def surgical_correct_lines(self, lines: List[Dict[str, Any]], threshold: float = 0.90) -> List[Dict[str, Any]]:
        """
        Surgically corrects only the lines that fall below the confidence threshold
        OR contain known noise patterns (hallucinations).
        """
        import re
        # Patterns that strongly suggest OCR noise/hallucinations even if confidence is high
        NOISE_PATTERNS = [
            r"<br>", r"\|", r"\[\s*\]", r"\{.*\}", r"_{2,}", r"\.{3,}"
        ]
        noise_regex = re.compile("|".join(NOISE_PATTERNS))

        corrected_count = 0
        for line in lines:
            conf = line.get("confidence", 1.0)
            text = line.get("text", "")
            
            # Trigger correction if:
            # 1. Confidence is low
            # 2. OR it contains obvious noise symbols
            is_noisy = bool(noise_regex.search(text))
            
            if (conf < threshold or is_noisy) and len(text.strip()) > 3:
                # Correct only this line
                lang_code = "ta" # We can parameterize this later if needed
                corrected_text = self.correct_text(text, lang_code=lang_code)
                line["text"] = corrected_text
                line["is_corrected"] = True
                corrected_count += 1
        
        if corrected_count > 0:
            print(f"     [LM] Surgically corrected {corrected_count} lines (low-conf or noise detected).")
        return lines

    def __del__(self):
        """Cleanly terminate the subprocess when the corrector is garbage-collected."""
        try:
            if hasattr(self, '_proc') and self._proc and self._proc.poll() is None:
                self._proc.stdin.close()
                self._proc.wait(timeout=5)
        except Exception:
            pass

if __name__ == "__main__":
    # Smoke test
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    corrector = LMCorrector()
    test_text = "தமழ் ஒசிஆர் ததாழில்நந்பம்"
    result = corrector.correct_text(test_text)
    print(f"\nInput:  {test_text}")
    print(f"Output: {result}")
