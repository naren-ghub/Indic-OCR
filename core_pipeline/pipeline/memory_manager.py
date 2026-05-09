"""
GPU Memory Manager — Phase 2
==============================
Ensures only one OCR engine is resident in VRAM at a time.
Critical for the RTX 4050 (6GB) to avoid OOM crashes during
engine comparison mode.

In single-engine mode (normal operation), the manager loads once
and keeps the engine resident — zero overhead.

In comparison mode (--compare-engines), it automatically swaps
engines with a full VRAM flush between each switch.
"""
from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.ocr_engine import BaseOCREngine


class GPUMemoryManager:
    """
    Context manager for safe GPU memory swapping between OCR engines.

    Usage:
        manager = GPUMemoryManager()

        # Single engine (normal mode) — loads once, stays loaded
        with manager.engine_context(surya_engine) as eng:
            for img in images:
                result = eng.process_image(img)

        # Engine comparison — auto-swaps VRAM between engines
        for img in images:
            with manager.engine_context(surya_engine) as eng:
                surya_result = eng.process_image(img)
            with manager.engine_context(paddle_engine) as eng:
                paddle_result = eng.process_image(img)
    """

    def __init__(self):
        self._active_engine: BaseOCREngine | None = None

    @property
    def active_engine_name(self) -> str | None:
        """Name of the currently loaded engine, or None."""
        return self._active_engine.name if self._active_engine else None

    @contextmanager
    def engine_context(self, engine: BaseOCREngine):
        """
        Context manager that ensures `engine` is loaded before use.
        If a different engine was previously loaded, it is unloaded first
        with a full VRAM flush.

        The engine stays loaded after the context exits — it's only
        unloaded if a *different* engine is requested next, or if
        free_all() is called explicitly.
        """
        if self._active_engine is not None and self._active_engine is not engine:
            # Different engine was loaded — swap
            old_name = self._active_engine.name
            self._active_engine.unload()
            self._flush_gpu()
            print(f"[MemoryManager] Swapped: {old_name} → {engine.name}")

        if not engine.is_loaded:
            engine.load()
        if not engine.is_loaded:
            raise RuntimeError(f"Engine '{engine.name}' failed to load.")

        self._active_engine = engine

        try:
            yield engine
        finally:
            # Keep loaded — only unload on next context with a different engine
            pass

    def free_all(self) -> None:
        """Explicitly unload whatever is currently loaded and flush GPU."""
        if self._active_engine is not None:
            self._active_engine.unload()
            self._active_engine = None
        self._flush_gpu()
        print("[MemoryManager] All engines unloaded, GPU memory freed.")

    @staticmethod
    def _flush_gpu() -> None:
        """Force garbage collection and clear CUDA cache."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        try:
            import paddle
            paddle.device.cuda.empty_cache()
        except (ImportError, Exception):
            pass

    def get_vram_usage_mb(self) -> float | None:
        """Return current VRAM usage in MB (PyTorch only). Returns None if unavailable."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
        except ImportError:
            pass
        return None
