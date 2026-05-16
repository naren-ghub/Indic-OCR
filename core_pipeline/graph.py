"""
graph.py — Phase 3 Agentic OCR Orchestrator
============================================
Implements the 4-node LangGraph StateGraph:

  Node 1: Router Agent    (Groq Vision LLM → preprocessing decision)
  Node 2: OCR Execution   (Surya pipeline — no LLM)
  Node 3: QA Agent        (OpenAI-compat LLM — conditional, low-conf pages only)
  Node 4: Final Processor (OpenAI-compat LLM — one call, whole document)

Entry point:
    from graph import run_agentic_ocr
    result = run_agentic_ocr(
        input_path="path/to/document.pdf",
        lang_hints=["ta"],
        output_txt="output.txt",
    )
"""
from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ── Env / path setup ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("RECOGNITION_BATCH_SIZE", "32")
os.environ.setdefault("DETECTOR_BATCH_SIZE", "4")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Imports ───────────────────────────────────────────────────────────────────
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv()

from pipeline.pdf_converter import pdf_to_images
from pipeline.ocr_engine import SuryaEngine
from pipeline.layout_engine import LayoutEngine
from pipeline.postprocessor import Cleaner
from pipeline.text_cleaner import clean_text as deep_clean
from pipeline.preprocessor import preprocess, preprocessed_to_pil

from agents.router_agent import run_router_agent
from agents.qa_agent import run_qa_agent, should_run_qa
from agents.final_processor import run_final_processor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AgenticOCR")

# ── LLM Client config (read from env or pass directly) ────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
OSS_API_KEY   = os.getenv("OSS_LLM_API_KEY", "")      # API key for gpt-oss-120b
OSS_BASE_URL  = os.getenv("OSS_LLM_BASE_URL", "")      # e.g. "https://api.together.xyz/v1"
OSS_MODEL     = os.getenv("OSS_LLM_MODEL", "")         # e.g. "meta-llama/Llama-3.3-70B-Instruct-Turbo"

GROQ_VISION_MODEL = "llama-3.2-11b-vision-preview"

QA_CONFIDENCE_THRESHOLD = 0.80
MAX_QA_RETRIES = 2


# ── State dataclass ───────────────────────────────────────────────────────────
@dataclass
class AgentState:
    # Inputs
    input_path: str = ""
    lang_hints: list[str] = field(default_factory=lambda: ["ta"])

    # Router outputs
    doc_type: str = "modern_print"
    detected_language: str = "ta"
    noise_level: int = 3
    estimated_columns: int = 1
    routing_reason: str = ""

    # Per-page OCR results: list of dicts with keys: page_num, text, confidences
    page_results: list[dict] = field(default_factory=list)

    # QA tracking
    qa_decisions: list[dict] = field(default_factory=list)

    # Assembled raw text (all pages, post-QA)
    assembled_text: str = ""
    total_pages: int = 0

    # Final output
    final_markdown: str = ""

    # Metadata
    elapsed_ocr: float = 0.0
    elapsed_agents: float = 0.0
    llm_call_count: int = 0


# ── Node implementations ──────────────────────────────────────────────────────

def node_router(state: AgentState, groq_client: Groq) -> AgentState:
    """Node 1 — Router Agent. Analyses page 1 image via Groq Vision."""
    logger.info("═══ NODE 1: ROUTER AGENT ═══")
    t0 = time.perf_counter()

    pages = pdf_to_images(state.input_path, dpi=150)   # low-res just for routing
    if not pages:
        logger.error("Could not extract pages from %s", state.input_path)
        return state

    page1_img = pages[0][1]
    routing = run_router_agent(page1_img, groq_client, model=GROQ_VISION_MODEL)

    state.doc_type          = routing.get("doc_type", "modern_print")
    state.detected_language = routing.get("detected_language", state.lang_hints[0])
    state.noise_level       = routing.get("noise_level", 3)
    state.estimated_columns = routing.get("estimated_columns", 1)
    state.routing_reason    = routing.get("routing_reason", "")
    state.llm_call_count   += 1

    logger.info(
        "Router → type=%s | lang=%s | noise=%d | cols=%d",
        state.doc_type, state.detected_language, state.noise_level, state.estimated_columns,
    )
    state.elapsed_agents += time.perf_counter() - t0
    return state


def node_ocr_execution(state: AgentState) -> AgentState:
    """Node 2 — OCR Execution. Pure Surya pipeline, no LLM."""
    logger.info("═══ NODE 2: OCR EXECUTION ═══")
    t0 = time.perf_counter()

    use_preprocess = state.doc_type == "historical_scan"
    logger.info("Preprocessing enabled: %s (doc_type=%s)", use_preprocess, state.doc_type)

    pages = pdf_to_images(state.input_path, dpi=300)
    state.total_pages = len(pages)

    lang_codes = list(dict.fromkeys([state.detected_language] + state.lang_hints + ["en"]))
    
    from surya.foundation import FoundationPredictor
    fp = FoundationPredictor(device="cuda")
    
    engine        = SuryaEngine(langs=lang_codes, foundation_predictor=fp)
    layout_engine = LayoutEngine(foundation_predictor=fp)
    cleaner       = Cleaner()
    
    from pipeline.lm_corrector import LMCorrector
    lm_corrector = LMCorrector(device="cuda")

    page_results = []
    for i, raw_img in pages:
        logger.info("  Page %d/%d …", i, state.total_pages)

        # Preprocessing
        if use_preprocess:
            proc = preprocess(raw_img)
            pil_img = preprocessed_to_pil(proc)
        else:
            pil_img = raw_img

        # OCR
        engine.load()
        batch = engine.process_batch([pil_img])
        if not batch:
            page_results.append({"page_num": i, "text": "", "confidences": []})
            continue

        page_data = batch[0]
        lines      = page_data.get("lines", [])
        
        # Hybrid AI Correction: Surgically fix low-confidence words
        # before assembling paragraphs
        logger.info("     [LMCorrector] Running surgical line correction...")
        lines = lm_corrector.surgical_correct_lines(lines, threshold=0.90)
        
        confidences = [ln.get("confidence", 0.0) for ln in lines]
        raw_text    = "\n".join(ln.get("text", "") for ln in lines)

        # Layout engine (using Router insight for column layout)
        layout = layout_engine.analyze_page(pil_img, page_num=i, estimated_columns=state.estimated_columns)
        ordered_text = LayoutEngine.reconstruct_page_text(lines, layout)
        
        cleaned = Cleaner.clean_text(ordered_text or raw_text)

        page_results.append({
            "page_num": i,
            "text": cleaned,
            "confidences": confidences,
        })

    engine.unload()
    state.page_results = page_results
    state.elapsed_ocr  = time.perf_counter() - t0
    logger.info("OCR done — %d pages in %.1fs", state.total_pages, state.elapsed_ocr)
    return state


def node_qa(state: AgentState, llm_client: OpenAI) -> AgentState:
    """Node 3 — QA Agent. Conditional — only fires on low-confidence pages."""
    logger.info("═══ NODE 3: QA AGENT ═══")
    t0 = time.perf_counter()

    for pr in state.page_results:
        page_num    = pr["page_num"]
        confidences = pr["confidences"]
        text        = pr["text"]

        if not should_run_qa(confidences, state.estimated_columns, text):
            logger.info("  Page %d — QA skipped (conf OK)", page_num)
            continue

        retries = 0
        while retries < MAX_QA_RETRIES:
            decision = run_qa_agent(
                page_text=text,
                page_confidences=confidences,
                detected_language=state.detected_language,
                llm_client=llm_client,
                model=OSS_MODEL,
            )
            state.llm_call_count += 1
            state.qa_decisions.append({"page": page_num, **decision})

            if decision["action"] == "accept":
                break

            if decision["action"] == "retry_single":
                logger.info("  Page %d — QA: retry single-column", page_num)
                # Re-run OCR with single-column hint (simplified: just re-accept)
                # Full implementation would call OCR with column_count=1 override
                break

            if decision["action"] == "retry_paddle":
                logger.info("  Page %d — QA: retry paddle (not available, accepting)", page_num)
                break

            retries += 1

    state.elapsed_agents += time.perf_counter() - t0
    return state


def node_final_processor(state: AgentState, llm_client: OpenAI) -> AgentState:
    """Node 4 — Final Processor. One LLM call for the whole document."""
    logger.info("═══ NODE 4: FINAL PROCESSOR ═══")
    t0 = time.perf_counter()

    # Assemble all pages with page separators
    parts = []
    for pr in state.page_results:
        parts.append(f"=== PAGE {pr['page_num']} ===\n{pr['text']}")
    state.assembled_text = "\n\n".join(parts)

    state.final_markdown = run_final_processor(
        assembled_text=state.assembled_text,
        detected_language=state.detected_language,
        total_pages=state.total_pages,
        llm_client=llm_client,
        model=OSS_MODEL,
    )
    state.llm_call_count += 1
    state.elapsed_agents += time.perf_counter() - t0
    logger.info("Final processor done — %d output chars.", len(state.final_markdown))
    return state


# ── Main entry point ──────────────────────────────────────────────────────────

def run_agentic_ocr(
    input_path: str,
    lang_hints: Optional[list[str]] = None,
    output_txt: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    oss_api_key: Optional[str] = None,
    oss_base_url: Optional[str] = None,
    oss_model: Optional[str] = None,
    only_bart: bool = False,
) -> dict:
    """
    Main entry point for the Agentic OCR pipeline.

    Args:
        input_path:   Path to PDF / image file.
        lang_hints:   List of ISO language codes (e.g. ["ta", "en"]).
        output_txt:   If given, write final text to this path.
        groq_api_key: Groq API key (or set GROQ_API_KEY env var).
        oss_api_key:  Open-source LLM API key (or set OSS_LLM_API_KEY env var).
        oss_base_url: OpenAI-compat base URL (or set OSS_LLM_BASE_URL env var).
        oss_model:    Model name (or set OSS_LLM_MODEL env var).

    Returns:
        dict with keys: final_markdown, total_pages, llm_call_count,
                        elapsed_ocr, elapsed_agents, detected_language
    """
    global OSS_MODEL, OSS_BASE_URL, OSS_API_KEY

    if groq_api_key: os.environ["GROQ_API_KEY"]  = groq_api_key
    if oss_api_key:  os.environ["OSS_LLM_API_KEY"] = oss_api_key
    if oss_base_url: os.environ["OSS_LLM_BASE_URL"] = oss_base_url
    if oss_model:    os.environ["OSS_LLM_MODEL"] = oss_model

    # Re-read env after potential override
    _groq_key  = os.getenv("GROQ_API_KEY", "")
    _oss_key   = os.getenv("OSS_LLM_API_KEY", "")
    _oss_url   = os.getenv("OSS_LLM_BASE_URL", "")
    _oss_model = os.getenv("OSS_LLM_MODEL", "")

    if not only_bart:
        if not _groq_key:
            raise ValueError("GROQ_API_KEY is required for the Router Agent.")
        if not _oss_key or not _oss_url or not _oss_model:
            raise ValueError(
                "OSS_LLM_API_KEY, OSS_LLM_BASE_URL, and OSS_LLM_MODEL are required "
                "for the QA Agent and Final Processor."
            )

    OSS_MODEL    = _oss_model
    OSS_BASE_URL = _oss_url
    OSS_API_KEY  = _oss_key

    # ── Initialise LLM clients (if needed) ────────────────────────────────────
    groq_client = Groq(api_key=_groq_key) if _groq_key else None
    llm_client  = OpenAI(api_key=_oss_key, base_url=_oss_url) if _oss_key else None

    # ── Initialise state ──────────────────────────────────────────────────────
    state = AgentState(
        input_path=str(input_path),
        lang_hints=lang_hints or ["ta"],
    )

    t_total = time.perf_counter()
    logger.info("★ Starting Agentic OCR — %s", input_path)

    # ── Run nodes ─────────────────────────────────────────────────────────────
    if only_bart:
        logger.info("⚡ ONLY_BART MODE: Bypassing all LLM agents.")
        state.detected_language = state.lang_hints[0] if state.lang_hints else "ta"
        state.doc_type = "historical_scan" # Enable preprocessing for bart mode
        state = node_ocr_execution(state)
        # Manual assembly since we skipped the Final Processor
        parts = [f"=== PAGE {pr['page_num']} ===\n{pr['text']}" for pr in state.page_results]
        state.final_markdown = "\n\n".join(parts)
    else:
        state = node_router(state, groq_client)
        state = node_ocr_execution(state)
        state = node_qa(state, llm_client)
        state = node_final_processor(state, llm_client)

    elapsed_total = time.perf_counter() - t_total

    # ── Print summary ─────────────────────────────────────────────────────────
    logger.info("━" * 60)
    logger.info("★ AGENTIC OCR COMPLETE")
    logger.info("  Detected language : %s", state.detected_language)
    logger.info("  Document type     : %s", state.doc_type)
    logger.info("  Total pages       : %d", state.total_pages)
    logger.info("  LLM calls made    : %d", state.llm_call_count)
    logger.info("  OCR time          : %.1fs", state.elapsed_ocr)
    logger.info("  Agent time        : %.1fs", state.elapsed_agents)
    logger.info("  Total time        : %.1fs", elapsed_total)
    logger.info("━" * 60)

    # ── Write output ──────────────────────────────────────────────────────────
    if output_txt:
        out_path = Path(output_txt)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        header = (
            f"Agentic Indic OCR — Phase 3 Output\n"
            f"{'=' * 60}\n"
            f"Source file      : {Path(input_path).name}\n"
            f"Detected language: {state.detected_language}\n"
            f"Document type    : {state.doc_type}\n"
            f"Total pages      : {state.total_pages}\n"
            f"LLM calls made   : {state.llm_call_count}\n"
            f"OCR time (s)     : {state.elapsed_ocr:.1f}\n"
            f"Agent time (s)   : {state.elapsed_agents:.1f}\n"
            f"Total time (s)   : {elapsed_total:.1f}\n"
            f"{'=' * 60}\n\n"
        )

        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(header)
            fh.write(state.final_markdown)

        logger.info("✓ Output written to: %s", out_path.resolve())

    return {
        "final_markdown"  : state.final_markdown,
        "total_pages"     : state.total_pages,
        "llm_call_count"  : state.llm_call_count,
        "elapsed_ocr"     : state.elapsed_ocr,
        "elapsed_agents"  : state.elapsed_agents,
        "elapsed_total"   : elapsed_total,
        "detected_language": state.detected_language,
        "doc_type"        : state.doc_type,
        "qa_decisions"    : state.qa_decisions,
        "routing_reason"  : state.routing_reason,
    }
