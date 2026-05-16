"""
run_agentic.py — CLI runner for the Phase 3 Agentic OCR Pipeline
=================================================================
Usage:
  python run_agentic.py --input <path_to_pdf> [options]

Required env vars (or pass as CLI args):
  GROQ_API_KEY       — Groq API key for the Router Agent (Llama-3.2-Vision)
  OSS_LLM_API_KEY    — API key for the 120B LLM (e.g. Together.ai)
  OSS_LLM_BASE_URL   — OpenAI-compat endpoint (e.g. https://api.together.xyz/v1)
  OSS_LLM_MODEL      — Model name (e.g. meta-llama/Llama-3.3-70B-Instruct-Turbo)

Example:
  python run_agentic.py \\
      --input "demo_data/Test_data/Skipped/Page from காலைக்கதிர் 14.10.22.pdf" \\
      --langs ta \\
      --output outputs/agentic_output.txt \\
      --groq-key gsk_xxx \\
      --oss-key togetherai_xxx \\
      --oss-url https://api.together.xyz/v1 \\
      --oss-model meta-llama/Llama-3.3-70B-Instruct-Turbo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from graph import run_agentic_ocr


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 3 Agentic Indic OCR — run a document through the agent pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",     required=True, help="Path to PDF / image file to process")
    p.add_argument("--langs",     nargs="+", default=["ta"], help="Language hint(s) ISO codes, e.g. ta hi")
    p.add_argument("--output",    default=None, help="Path to write the output .txt (default: auto-named alongside input)")
    p.add_argument("--groq-key",  default=None, help="Groq API key (overrides GROQ_API_KEY env var)")
    p.add_argument("--oss-key",   default=None, help="OSS LLM API key (overrides OSS_LLM_API_KEY env var)")
    p.add_argument("--oss-url",   default=None, help="OSS LLM base URL (overrides OSS_LLM_BASE_URL env var)")
    p.add_argument("--oss-model", default=None, help="OSS LLM model name (overrides OSS_LLM_MODEL env var)")
    p.add_argument("--only-bart", action="store_true", help="Disable all LLM agents and use only IndicBART correction")
    p.add_argument("--print",     action="store_true", help="Also print final output to stdout")
    return p


def main() -> None:
    args = build_parser().parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Auto-name output if not specified
    output_path = args.output
    if output_path is None:
        output_path = str(input_path.parent / (input_path.stem + "_agentic_output.txt"))

    print(f"\n{'━' * 60}")
    print(f"  Agentic OCR — Phase 3")
    print(f"  Input   : {input_path.name}")
    print(f"  Langs   : {', '.join(args.langs)}")
    print(f"  Output  : {output_path}")
    print(f"{'━' * 60}\n")

    result = run_agentic_ocr(
        input_path=str(input_path),
        lang_hints=args.langs,
        output_txt=output_path,
        groq_api_key=args.groq_key,
        oss_api_key=args.oss_key,
        oss_base_url=args.oss_url,
        oss_model=args.oss_model,
        only_bart=args.only_bart,
    )

    # ── Print evaluation summary ──────────────────────────────────────────────
    print(f"\n{'━' * 60}")
    print("  EVALUATION SUMMARY")
    print(f"{'━' * 60}")
    print(f"  Detected language : {result['detected_language']}")
    print(f"  Document type     : {result['doc_type']}")
    print(f"  Routing reason    : {result['routing_reason']}")
    print(f"  Total pages       : {result['total_pages']}")
    print(f"  LLM calls made    : {result['llm_call_count']}")
    print(f"  OCR time          : {result['elapsed_ocr']:.1f}s")
    print(f"  Agent time        : {result['elapsed_agents']:.1f}s")
    print(f"  Total time        : {result['elapsed_total']:.1f}s")

    qa = result.get("qa_decisions", [])
    if qa:
        print(f"\n  QA Interventions  : {len(qa)} page(s) triggered")
        for d in qa:
            print(f"    Page {d['page']:>3} → action={d['action']}  reason={d.get('reason', '')}")
    else:
        print(f"  QA Interventions  : None (all pages passed confidence threshold)")

    print(f"\n  Output chars      : {len(result['final_markdown'])}")
    print(f"  Output file       : {Path(output_path).resolve()}")
    print(f"{'━' * 60}\n")

    # ── Optionally print to stdout ────────────────────────────────────────────
    if args.print:
        print("\n" + "═" * 60)
        print("  FINAL OUTPUT (Markdown)")
        print("═" * 60)
        print(result["final_markdown"])

    print("✅ Done.")


if __name__ == "__main__":
    main()
