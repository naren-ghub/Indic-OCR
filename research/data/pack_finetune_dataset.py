#!/usr/bin/env python3
"""
pack_finetune_dataset.py — Phase 2 fine-tuning dataset packer
=============================================================
Creates a small, controlled training set for ByT5 OCR correction by:
  1) sampling N examples from the existing corpus JSONL
  2) generating M RoundTrip (image-degradation + real OCR) examples
  3) merging + shuffling deterministically
  4) writing train/val JSONL outputs

Output schema (same as data/corpus/train.jsonl):
  {"input": <noisy>, "target": <clean>, "source": <source>}

Example:
  python OCR_Phase_2/data/pack_finetune_dataset.py --corpus-pairs 8000 --roundtrip-pairs 2000
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_JSONL = ROOT_DIR / "data" / "corpus" / "train.jsonl"
DEFAULT_OUT_DIR = ROOT_DIR / "data" / "training"


@dataclass(frozen=True)
class Pair:
    input: str
    target: str
    source: str

    def to_json(self) -> str:
        return json.dumps(
            {"input": self.input, "target": self.target, "source": self.source},
            ensure_ascii=False,
        )


def read_pairs(jsonl_path: Path) -> list[Pair]:
    pairs: list[Pair] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pairs.append(
                Pair(
                    input=obj["input"],
                    target=obj["target"],
                    source=obj.get("source", "unknown"),
                )
            )
    return pairs


def write_pairs(jsonl_path: Path, pairs: list[Pair]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(p.to_json())
            f.write("\n")


def split_train_val(pairs: list[Pair], val_ratio: float, rng: random.Random) -> tuple[list[Pair], list[Pair]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1 (exclusive)")
    pairs = list(pairs)
    rng.shuffle(pairs)
    n_val = max(1, int(round(len(pairs) * val_ratio)))
    val = pairs[:n_val]
    train = pairs[n_val:]
    return train, val


def generate_roundtrip_pairs(
    *,
    clean_texts: list[str],
    roundtrip_pairs: int,
    min_edit_distance: int,
    seed: int,
    engine_name: str,
    font_path: str | None,
    checkpoint_file: Path | None = None,
) -> list[Pair]:
    # Local imports so this script can still do corpus-only packing without heavy deps.
    import sys
    sys.path.insert(0, str(ROOT_DIR))

    from pipeline.memory_manager import GPUMemoryManager
    from pipeline.ocr_engine import create_engine
    from pipeline.roundtrip_generator import RoundTripOCRGenerator

    memory_manager = GPUMemoryManager()
    if engine_name == "surya":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        engine = create_engine(engine_name, device=device)
    elif engine_name == "paddle":
        # PaddleOCR can be installed CPU-only or GPU-enabled; default to GPU when available.
        use_gpu = True
        try:
            import paddle
            use_gpu = bool(getattr(paddle.device, "is_compiled_with_cuda", lambda: False)())
        except Exception:
            use_gpu = False
        engine = create_engine(engine_name, use_gpu=use_gpu)
    else:
        engine = create_engine(engine_name)

    with memory_manager.engine_context(engine):
        generator = RoundTripOCRGenerator(engine=engine, font_size=36, font_path=font_path)
        pairs = generator.generate_pairs(
            clean_texts,
            roundtrip_pairs,
            min_edit_distance=min_edit_distance,
            seed=seed,
            checkpoint_file=checkpoint_file,
        )

    src = f"roundtrip_{engine_name}"
    return [Pair(input=noisy, target=clean, source=src) for (noisy, clean) in pairs]


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack corpus + roundtrip pairs into a finetuning dataset")
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS_JSONL, help="Path to corpus train.jsonl")
    parser.add_argument("--corpus-pairs", type=int, default=8000, help="Number of corpus pairs to sample")
    parser.add_argument("--roundtrip-pairs", type=int, default=2000, help="Number of roundtrip pairs to generate")
    parser.add_argument("--min-edit-distance", type=int, default=2, help="Min edit distance to accept roundtrip pair")
    parser.add_argument("--engine", type=str, default="surya", choices=["surya", "paddle"], help="OCR engine for roundtrip")
    parser.add_argument("--font-path", type=str, default=None, help="Optional explicit Tamil font path (.ttf/.ttc)")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation fraction (default 0.10)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.corpus_pairs <= 0:
        raise ValueError("--corpus-pairs must be > 0")
    if args.roundtrip_pairs < 0:
        raise ValueError("--roundtrip-pairs must be >= 0")
    if args.min_edit_distance < 0:
        raise ValueError("--min-edit-distance must be >= 0")

    if not args.corpus_jsonl.exists():
        raise FileNotFoundError(f"Corpus JSONL not found: {args.corpus_jsonl}")

    rng = random.Random(args.seed)

    print(f"[Pack] Reading corpus: {args.corpus_jsonl}")
    corpus_pairs = read_pairs(args.corpus_jsonl)
    if len(corpus_pairs) < args.corpus_pairs:
        raise RuntimeError(f"[Pack] Requested {args.corpus_pairs} corpus pairs, but only found {len(corpus_pairs)}")

    sampled_corpus = rng.sample(corpus_pairs, args.corpus_pairs)
    print(f"[Pack] Sampled corpus pairs: {len(sampled_corpus)}")

    roundtrip: list[Pair] = []
    if args.roundtrip_pairs > 0:
        clean_texts = [p.target for p in sampled_corpus]
        print(
            f"[Pack] Generating roundtrip pairs: {args.roundtrip_pairs} "
            f"(min_edits>={args.min_edit_distance}, engine={args.engine})"
        )
        checkpoint_path = args.out_dir / f"roundtrip_checkpoint_{args.seed}.jsonl"
        roundtrip = generate_roundtrip_pairs(
            clean_texts=clean_texts,
            roundtrip_pairs=args.roundtrip_pairs,
            min_edit_distance=args.min_edit_distance,
            seed=args.seed,
            engine_name=args.engine,
            font_path=args.font_path,
            checkpoint_file=checkpoint_path,
        )
        print(f"[Pack] Generated roundtrip pairs: {len(roundtrip)}")

    combined = list(sampled_corpus) + list(roundtrip)
    rng.shuffle(combined)

    train, val = split_train_val(combined, val_ratio=args.val_ratio, rng=rng)

    out_train = args.out_dir / "final_train.jsonl"
    out_val = args.out_dir / "final_val.jsonl"
    write_pairs(out_train, train)
    write_pairs(out_val, val)

    counts = {}
    for p in combined:
        counts[p.source] = counts.get(p.source, 0) + 1

    print(f"[Pack] Wrote: {out_train} ({len(train)} lines)")
    print(f"[Pack] Wrote: {out_val} ({len(val)} lines)")
    print(f"[Pack] Source mix: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
