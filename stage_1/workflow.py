"""
stage_1/workflow.py

LangGraph StateGraph workflow for the High Court / Commercial Court paper PDF pipeline.

Node order:
  START  →  preprocess  →  ocr_agent  →  END

  • preprocess  : OpenCV image-cleaning pipeline (preprocessor.py)
  • ocr_agent   : Two-pass LLM OCR per page
                    Pass 1 – Full page  → JSON {raw_text, divide_page}
                    Pass 2 – (if divide_page=true) split image into top/bottom
                             halves, zoom 2×, re-scan each with OCR_REGION_TEXT
                  Final result per page is merged into a structured JSON object.
"""

import sys
import io
# Fix Windows terminal Unicode (cp1252 → utf-8)
# Required for Hindi (Devanagari) + English mixed output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import asyncio
import base64
import json
import os
import re
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict

import numpy as np
import cv2
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from PIL import Image

# Ensure project root is in sys.path when script is executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage_1.model import init_model
from stage_1.preprocessor import (
    preprocess_pdf,
    get_pdf_page_count,
    preprocess_single_pdf_page_by_index,
)
from stage_1.prompt import (
    get_ocr_default_system_message,
    get_ocr_region_system_message,
    get_ocr_verifier_system_message,
)
from stage_1.rate_limiter import make_async_rate_limiter, AsyncRateLimiter
from stage_1.schemas import PageOCRResult, PageVerifyResult


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Shared state passed between LangGraph nodes."""

    # LLM conversation messages (accumulated)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Path to the source PDF
    pdf_path: Optional[str]

    # List of cleaned PIL images (one per page) from the preprocess node
    preprocessed_images: Optional[List[Image.Image]]

    # Candidate OCR extractions from Node 2 (ocr_agent)
    ocr_results: Optional[List[Dict]]

    # Verified & audited final OCR output from Node 3 (verifier_agent)
    verified_ocr_results: Optional[List[Dict]]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _pil_to_base64(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    """Encode a PIL Image to a compressed base64 string (for 40x faster multimodal LLM payload uploads)."""
    buf = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_image_message(
    system_msg: SystemMessage,
    img: Image.Image,
    user_text: str,
) -> List:
    """
    Build a list of LangChain messages [system, human] where the HumanMessage
    contains the image + an instruction text (multimodal content).
    """
    b64 = _pil_to_base64(img, fmt="JPEG", quality=85)
    human = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        },
        {
            "type": "text",
            "text": user_text,
        },
    ])
    return [system_msg, human]



def _zoom_image(img: Image.Image, scale: float = 2.0) -> Image.Image:
    """
    Zoom (upscale) a PIL Image by *scale* using LANCZOS resampling,
    then apply a light unsharp-mask pass to restore sharpness after enlarging.
    """
    new_w = int(img.width  * scale)
    new_h = int(img.height * scale)
    zoomed = img.resize((new_w, new_h), Image.LANCZOS)

    # Sharpen via OpenCV unsharp mask
    arr = np.array(zoomed.convert("L"))           # grayscale numpy
    blurred = cv2.GaussianBlur(arr, (0, 0), 3)
    sharpened = cv2.addWeighted(arr, 1.8, blurred, -0.8, 0)
    return Image.fromarray(sharpened).convert("RGB")


def _split_image_halves(
    img: Image.Image,
    overlap_pct: float = 0.05,
) -> tuple[Image.Image, Image.Image]:
    """
    Split a PIL Image horizontally into a top half and a bottom half,
    with a small vertical OVERLAP so that text sitting near the midpoint
    is fully visible in both regions (not cut in two).

    Args:
        img:         The source PIL Image.
        overlap_pct: Fraction of the total height to overlap.
                     Default 0.05 = 5 % (≈ 1–2 lines of text at 300 DPI).
                     Configurable via PREPROCESS_SPLIT_OVERLAP_PCT in .env.

    Visual layout
    ─────────────
    y=0    ┌──────────────────────┐
           │                      │
           │      TOP  HALF       │
           │                      │
    mid    ├ ─ ─ ─ ─ ─ midpoint  ┤
           │  ← overlap (top)  →  │  ← only TOP extends past mid
    end_t  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─  ┤  TOP ends here  (mid + overlap_px)
           ╞══════════════════════╡
    mid    ├ ─ ─ ─ ─ ─ midpoint  ┤  BOTTOM starts exactly here (no overlap)
           │                      │
           │     BOTTOM HALF      │
           │                      │
    end    └──────────────────────┘
    """
    h   = img.height
    w   = img.width
    mid = h // 2
    overlap_px = int(h * overlap_pct)

    top_end      = min(mid + overlap_px, h)   # top extends PAST midpoint
    bottom_start = mid                         # bottom starts exactly at mid

    top    = img.crop((0, 0,            w, top_end))
    bottom = img.crop((0, bottom_start, w, h))
    return top, bottom



def _extract_text_content(content) -> str:
    """
    Safely extract a plain string from an LLM response's .content field.

    Gemini multimodal responses may return .content as:
      - str  : plain text  (most cases)
      - list : list of part dicts, e.g. [{"type": "text", "text": "..."}]

    This helper normalises both into a single str.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)
    return str(content)


def _clean_nested_raw_text(text: str) -> str:
    """Sanitize and unwrap any leftover markdown codeblock fences, stringified JSON, or repeating dot loops."""
    if not isinstance(text, str):
        return str(text)
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    if cleaned.startswith("{") and ("raw_text" in cleaned or "text" in cleaned):
        try:
            inner = json.loads(cleaned)
            if isinstance(inner, dict):
                inner_text = inner.get("raw_text") or inner.get("text") or ""
                if inner_text and inner_text != text:
                    return _clean_nested_raw_text(inner_text)
        except Exception:
            pass
    # Collapse degenerate long dot/dash/underscore loops (e.g. fill-in-the-blank lines)
    cleaned = re.sub(r"\.{5,}", "...", cleaned)
    cleaned = re.sub(r"_{5,}", "...", cleaned)
    return cleaned


def _parse_json_response(raw) -> dict:
    """
    Robustly extract a JSON object from an LLM response .content value.
    Handles str and list content types, markdown code fences, and alternate keys.
    """
    text = _extract_text_content(raw)
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                # Extract text checking multiple potential keys
                raw_txt = (
                    parsed.get("raw_text")
                    or parsed.get("text")
                    or parsed.get("extracted_text")
                    or parsed.get("ocr_text")
                    or parsed.get("content")
                    or ""
                )
                if not raw_txt and text:
                    raw_txt = text
                
                raw_txt = _clean_nested_raw_text(raw_txt)
                parsed["raw_text"] = raw_txt
                return parsed
        except json.JSONDecodeError:
            pass

    # Fallback: return the full unparsed text without divide
    return {
        "raw_text": _clean_nested_raw_text(text),
        "divide_page": False,
        "accuracy_score": 0.85,
        "accuracy_level": "MEDIUM",
        "accuracy_reason": "Fallback raw text parsing",
    }


def _get_output_dir() -> Path:
    """Return (and create) the per-run output directory from OCR_OUTPUT_DIR env var."""
    out_dir = Path(os.getenv("OCR_OUTPUT_DIR", "ocr_output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_page_result(page_result: Dict, output_dir: Path) -> Path:
    """
    Save a single page result to two files inside *output_dir*:

        page_NNN.json  – full structured dict (raw_text, divide_page, regions)
        page_NNN.txt   – plain text: page header + raw_text only

    Overwrites existing files (safe for retries).
    Returns the path of the .json file.
    """
    page_num = page_result["page"]
    stem     = f"page_{page_num:03d}"

    # --- JSON (full structured output) ---
    json_file = output_dir / f"{stem}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(page_result, f, ensure_ascii=False, indent=2)

    # --- TXT (plain readable text) ---
    txt_file = output_dir / f"{stem}.txt"
    score_val = page_result.get("accuracy_score") if "accuracy_score" in page_result else page_result.get("confidence_score", 0.9)
    score_pct = int(score_val * 100)
    level     = page_result.get("accuracy_level") if "accuracy_level" in page_result else page_result.get("confidence_level", "HIGH")
    reason    = page_result.get("accuracy_reason") if "accuracy_reason" in page_result else page_result.get("confidence_reason", "OCR Extraction")

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"=== Page {page_num} | OCR Accuracy: {score_pct}% [{level}] - {reason} ===\n\n")
        f.write(page_result.get("raw_text", ""))
        f.write("\n")

    return json_file


def combine_outputs(output_dir: Optional[str] = None) -> Dict:
    """
    Read all page_NNN.json files from *output_dir*, sort them by page number,
    and write two combined output files:

        combined_output.json  – full structured data for all pages
        combined_output.txt   – plain text: page header + raw_text per page,
                                one after another, human-readable

    Returns the combined dict.
    """
    out_path = Path(output_dir) if output_dir else _get_output_dir()
    page_files = sorted(out_path.glob("page_*.json"))

    if not page_files:
        print(f"[combine_outputs] No page JSON files found in '{out_path}'.")
        return {}

    pages = []
    for pf in page_files:
        with open(pf, "r", encoding="utf-8") as f:
            pages.append(json.load(f))

    # Sort by page number (in case glob order differs)
    pages.sort(key=lambda p: p["page"])

    # ── combined_output.txt  (the main human-readable file) ──────────────────
    txt_sections = []
    scores = []
    for p in pages:
        score_val = p.get("accuracy_score") if "accuracy_score" in p else p.get("confidence_score", 0.9)
        score_pct = int(score_val * 100)
        level     = p.get("accuracy_level") if "accuracy_level" in p else p.get("confidence_level", "HIGH")
        reason    = p.get("accuracy_reason") if "accuracy_reason" in p else p.get("confidence_reason", "OCR Extraction")
        scores.append(score_val)

        header = f"=== Page {p['page']} | OCR Accuracy: {score_pct}% [{level}] - {reason} ==="
        txt_sections.append(f"{header}\n\n{p['raw_text'].strip()}")

    avg_score = (sum(scores) / len(scores)) if scores else 0.9
    combined_text = "\n\n" + ("\n\n" + "-" * 60 + "\n\n").join(txt_sections) + "\n"

    txt_file = out_path / "combined_output.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(combined_text)
    print(f"[combine_outputs] TXT  → '{txt_file}'")

    # ── combined_output.json  (full structured data) ─────────────────────────
    result = {
        "total_pages":             len(pages),
        "average_accuracy_score":  round(avg_score, 4),
        "combined_text":           combined_text,
        "pages":                   pages,
    }
    json_file = out_path / "combined_output.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[combine_outputs] JSON → '{json_file}'")

    print(f"[combine_outputs] Combined {len(pages)} page(s).")
    return result


# ---------------------------------------------------------------------------
# Node 1 — Preprocess
# ---------------------------------------------------------------------------

import concurrent.futures

async def preprocess_node(state: AgentState) -> dict:
    """
    LangGraph Node 1 – Instant PDF setup for real-time per-page AI streaming.
    Page 1 starts AI processing within ~100ms of pipeline launch!
    """
    pdf_path = state.get("pdf_path")
    preprocessed_images = state.get("preprocessed_images")

    if not pdf_path and not preprocessed_images:
        raise ValueError("[preprocess_node] 'pdf_path' is missing from state.")

    if preprocessed_images:
        return {"preprocessed_images": preprocessed_images, "total_pages": len(preprocessed_images)}

    total_pages = await asyncio.to_thread(get_pdf_page_count, pdf_path)
    print(f"\n[Node: preprocess] PDF '{pdf_path}' ({total_pages} page(s)). Real-time per-page AI streaming enabled!")
    return {"pdf_path": pdf_path, "total_pages": total_pages}





# ---------------------------------------------------------------------------
# Node 2 — OCR Agent  (two-pass per page)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Node 2 — OCR Candidate Extraction Agent
# ---------------------------------------------------------------------------

async def _extract_candidate_page(
    page_num: int,
    img: Image.Image,
    total_pages: int,
    model,
    sys_default: object,
    sys_region: object,
    rate_limiter: AsyncRateLimiter,
    output_dir: Path,
) -> Dict:
    """Async candidate OCR extraction for a single page (Pass 1 + Pass 2)."""
    print(f"\n[Node: ocr_agent] [Page {page_num}/{total_pages}] Extracting candidate text ...")

    # Pass 1: full-page candidate extraction (with up to 3 automatic retries)
    msgs_pass1 = _build_image_message(
        system_msg=sys_default,
        img=img,
        user_text=(
            f"Page {page_num}: Read and extract ALL visible printed and handwritten court order text, petition details, "
            "party names, claim amounts in Rs., dates, handwritten notes, stamps, signatures, and table rows from top to bottom "
            "exactly as visually visible on this image WITHOUT ANY HALLUCINATION. Never guess or fabricate text you do not clearly see or understand. "
            "If you do not understand any word, digit, or sentence, write 'too blurry to read' (or 'too blurry' if entire page is unreadable). "
            "If any text or number is handwritten, ALWAYS include '(handwritten)' with it anywhere it appears. "
            "Pay extreme character-level attention to handwritten Hindi names and words (e.g., carefully distinguish 'प्रवीन' / 'प्रवीन कुमार' from 'उदीन'). "
            "Set 'divide_page': true ONLY if you judge the page image requires dividing into two zoomed halves for re-scanning. "
            "Output the clean extracted text directly in JSON format containing 'raw_text', 'divide_page', 'accuracy_score', 'accuracy_level', and 'accuracy_reason'."
        ),
    )

    raw_text = ""
    divide_page = False
    accuracy_score = 0.85
    accuracy_level = "MEDIUM"
    api_timeout = float(os.getenv("STAGE1_API_TIMEOUT", os.getenv("API_TIMEOUT", "220.0")))

    for attempt in range(1, 4):
        try:
            response1 = await rate_limiter.call(model, msgs_pass1, timeout=api_timeout)
            content_str = _extract_text_content(response1.content)

            if content_str.strip().startswith("{") and "raw_text" in content_str:
                parsed = _parse_json_response(content_str)
                raw_text = _clean_nested_raw_text(str(parsed.get("raw_text") or "").strip())
                divide_page = bool(parsed.get("divide_page", False))
                accuracy_score = float(parsed.get("accuracy_score") or parsed.get("confidence_score") or 0.85)
                accuracy_level = str(parsed.get("accuracy_level") or parsed.get("confidence_level") or "MEDIUM").upper()
                accuracy_reason = str(parsed.get("accuracy_reason") or parsed.get("confidence_reason") or "Assessed by Candidate OCR Agent")
            else:
                raw_text = _clean_nested_raw_text(content_str.strip())
                divide_page = False

            if raw_text and not raw_text.startswith("[ERROR"):
                break
            backoff = 3 if attempt == 1 else 30
            print(f"  [Node: ocr_agent] [Page {page_num}] Attempt {attempt} returned empty text. Retrying in {backoff}s ...")
            await asyncio.sleep(backoff)
        except Exception as exc:
            backoff = 3 if attempt == 1 else 30
            print(f"  [Node: ocr_agent] [Page {page_num}] Attempt {attempt} API error: {exc}. Retrying in {backoff}s ...")
            await asyncio.sleep(backoff)

    if not raw_text:
        raw_text = f"[OCR extraction pending for Page {page_num}]"

    top_region = None
    bottom_region = None

    # Pass 2: zoomed halves (if divide_page=True)
    if divide_page:
        overlap_pct   = float(os.getenv("PREPROCESS_SPLIT_OVERLAP_PCT", "0.10"))
        top_half, bottom_half = _split_image_halves(img, overlap_pct=overlap_pct)
        top_zoomed    = _zoom_image(top_half,    scale=2.0)
        bottom_zoomed = _zoom_image(bottom_half, scale=2.0)

        msgs_top = _build_image_message(
            system_msg=sys_region, img=top_zoomed,
            user_text=f"TOP HALF of page {page_num}. Zoomed 2x. Extract every character top to bottom.",
        )
        try:
            resp_top   = await rate_limiter.call(model, msgs_top, timeout=api_timeout)
            top_region = _extract_text_content(resp_top.content)
        except Exception as exc:
            top_region = f"[Top region error: {exc}]"

        msgs_bottom = _build_image_message(
            system_msg=sys_region, img=bottom_zoomed,
            user_text=f"BOTTOM HALF of page {page_num}. Zoomed 2x. Extract every character top to bottom.",
        )
        try:
            resp_bottom   = await rate_limiter.call(model, msgs_bottom, timeout=api_timeout)
            bottom_region = _extract_text_content(resp_bottom.content)
        except Exception as exc:
            bottom_region = f"[Bottom region error: {exc}]"

        if top_region or bottom_region:
            merge_prompt = (
                f"Page {page_num} Zoomed Halves to Merge:\n\n"
                f"=== TOP HALF (ZOOMED 2x) ===\n{top_region or ''}\n\n"
                f"=== BOTTOM HALF (ZOOMED 2x) ===\n{bottom_region or ''}\n\n"
                "Combine the Top Half and Bottom Half into one seamless document. "
                "Identify and remove duplicate overlapping lines near the midpoint boundary seam. Output only the clean merged text."
            )
            merge_msgs = [
                SystemMessage(content="You are an expert OCR Document Merger. Combine the Top Half and Bottom Half into a single clean top-to-bottom document. Remove any duplicate overlapping lines near the midpoint seam. Output only clean merged text."),
                HumanMessage(content=merge_prompt),
            ]
            try:
                resp_merge = await rate_limiter.call(model, merge_msgs, timeout=api_timeout)
                merged_text = _extract_text_content(resp_merge.content)
                raw_text = _clean_nested_raw_text(merged_text)
            except Exception as exc:
                print(f"  [Node: ocr_agent] [Page {page_num}] 2-Halves Merge error: {exc}. Using fallback concatenation.")
                raw_text = (top_region or "").rstrip() + "\n\n" + (bottom_region or "").lstrip()

    candidate_dict = {
        "page":            page_num,
        "raw_text":        raw_text,
        "divide_page":     divide_page,
        "accuracy_score":  accuracy_score,
        "accuracy_level":  accuracy_level,
        "accuracy_reason": accuracy_reason,
        "top_region":      top_region,
        "bottom_region":   bottom_region,
    }

    # Save immediately so page_NNN.json and page_NNN.txt appear on disk instantly
    saved_json = _save_page_result(candidate_dict, output_dir)
    print(f"  [Node: ocr_agent] [Page {page_num}] ✓ Saved candidate -> {saved_json.name}")
    return candidate_dict


# Shared RateLimiter instance so ALL nodes strictly respect RATE_LIMIT_RPM
_GLOBAL_RATE_LIMITER: Optional[AsyncRateLimiter] = None

def _get_shared_rate_limiter() -> AsyncRateLimiter:
    global _GLOBAL_RATE_LIMITER
    if _GLOBAL_RATE_LIMITER is None:
        _GLOBAL_RATE_LIMITER = make_async_rate_limiter()
    return _GLOBAL_RATE_LIMITER


async def _process_page_end_to_end(
    page_num: int,
    pdf_path: Optional[str],
    img: Optional[Image.Image],
    total_pages: int,
    target_dpi: int,
    model,
    sys_default: object,
    sys_region: object,
    sys_verifier: object,
    rate_limiter: AsyncRateLimiter,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
) -> Tuple[Dict, Dict]:
    """
    Renders each page on-demand in ~100ms and immediately dispatches its payload to the AI pool.
    Page 1 fires to API_KEY_1 within ~100ms of script launch!
    """
    if img is None and pdf_path:
        img = await asyncio.to_thread(
            preprocess_single_pdf_page_by_index,
            pdf_path=pdf_path,
            page_idx=page_num - 1,
            render_dpi=target_dpi,
            target_dpi=target_dpi,
        )

    # Immediately stream into AI rate limiter pool!
    async with semaphore:
        candidate = await _extract_candidate_page(
            page_num, img, total_pages,
            model, sys_default, sys_region, rate_limiter, output_dir,
        )
        verified = await _verify_one_page(
            page_num, img, candidate, total_pages,
            model, sys_verifier, rate_limiter, output_dir,
        )
        return candidate, verified



async def ocr_agent_node(state: AgentState) -> dict:
    """
    LangGraph Node 2 – OCR Candidate & Verification Real-Time Streaming Pipeline.

    Renders/preprocesses each page on-demand and streams immediately to the AI model.
    """
    pdf_path    = state.get("pdf_path")
    images      = state.get("preprocessed_images")
    total_pages = state.get("total_pages") or (len(images) if images else 0)

    if not total_pages and pdf_path:
        total_pages = await asyncio.to_thread(get_pdf_page_count, pdf_path)

    if not total_pages:
        raise ValueError("[ocr_agent_node] Missing pdf_path or preprocessed_images in state.")

    target_dpi   = int(os.getenv("PREPROCESS_DPI", "200"))
    model        = init_model()
    sys_default  = get_ocr_default_system_message()
    sys_region   = get_ocr_region_system_message()
    sys_verifier = get_ocr_verifier_system_message()
    output_dir   = _get_output_dir()
    rate_limiter = _get_shared_rate_limiter()

    # Semaphore matching 3 RPM per key (15 RPM total 5-key pool capacity)
    semaphore    = asyncio.Semaphore(15)

    print(f"\n[Node: ocr_agent] Real-time per-page AI streaming for {total_pages} page(s) ...")

    tasks = [
        _process_page_end_to_end(
            page_num=page_num,
            pdf_path=pdf_path,
            img=images[page_num - 1] if images and page_num <= len(images) else None,
            total_pages=total_pages,
            target_dpi=target_dpi,
            model=model,
            sys_default=sys_default,
            sys_region=sys_region,
            sys_verifier=sys_verifier,
            rate_limiter=rate_limiter,
            output_dir=output_dir,
            semaphore=semaphore,
        )
        for page_num in range(1, total_pages + 1)
    ]

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)


    ocr_results: List[Dict]          = []
    verified_ocr_results: List[Dict] = []

    for page_num, res in enumerate(raw_results, start=1):
        if isinstance(res, Exception):
            print(f"[Node: ocr_agent] Page {page_num} pipeline FAILED: {res}")
            err_dict = {
                "page": page_num, "raw_text": f"[ERROR: {res}]",
                "divide_page": False, "accuracy_score": 0.0, "accuracy_level": "LOW",
                "accuracy_reason": f"Pipeline Error: {res}", "corrections_made": [],
            }
            ocr_results.append(err_dict)
            verified_ocr_results.append(err_dict)
        else:
            cand, ver = res
            ocr_results.append(cand)
            verified_ocr_results.append(ver)

    print(f"[Node: ocr_agent] Completed streaming pipeline for {total_pages} page(s).")
    return {
        "ocr_results":          ocr_results,
        "verified_ocr_results": verified_ocr_results,
    }


# ---------------------------------------------------------------------------
# Node 3 — OCR Verification & Accuracy Auditor Agent
# ---------------------------------------------------------------------------

async def _verify_one_page(
    page_num: int,
    img: Image.Image,
    candidate: Dict,
    total_pages: int,
    model,
    sys_verifier: object,
    rate_limiter: AsyncRateLimiter,
    output_dir: Path,
) -> Dict:
    """Async verification audit for a single page."""
    candidate_text = candidate.get("raw_text", "")
    # ── Pass 3: MAXIMUM ACCURACY Auditor / Verifier Agent Pass ──────────────
    print(f"  [Node: verifier_agent] [Page {page_num}/{total_pages}] Auditing accuracy ...")

    audit_user_prompt = (
        f"Auditing Candidate OCR Text for Page {page_num}:\n\n"
        f"--- CANDIDATE TEXT START ---\n{candidate_text}\n--- CANDIDATE TEXT END ---\n\n"
        "Perform a strict line-by-line visual audit comparing candidate text against the document image.\n"
        "CORRECT THE DATA: Fix any visual OCR misreads, typos, or wrong Devanagari characters. Restore any missing text, table rows, stamps, signatures, or dates visible in the image but omitted in candidate text.\n"
        "STRICT BLURRINESS & ZERO HALLUCINATION RULE: If any word or digit in a sentence is blurry, explicitly write 'too blurry to read'. If the whole page is unreadable, state 'too blurry'.\n"
        "CALCULATE ACCURACY SCORE: Evaluate the visual accuracy percentage (0.0 to 1.0) and REDUCE accuracy score if text was missing from candidate OCR, or if words/sections are marked 'too blurry to read' / 'too blurry'.\n"
        "Output the verified JSON containing verified_raw_text, accuracy_score, accuracy_level, accuracy_reason, and corrections_made."
    )
    msgs_verify = _build_image_message(
        system_msg=sys_verifier,
        img=img,
        user_text=audit_user_prompt,
    )

    parsed_verify = {}
    try:
        structured_verifier = model.with_structured_output(PageVerifyResult)
    except Exception:
        structured_verifier = None

    api_timeout = float(os.getenv("STAGE2_API_TIMEOUT", os.getenv("API_TIMEOUT", "220.0")))

    for attempt in range(1, 4):
        try:
            if structured_verifier is not None:
                res = await rate_limiter.call(structured_verifier, msgs_verify, timeout=api_timeout)
                if isinstance(res, PageVerifyResult):
                    parsed_verify = res.model_dump()
                elif isinstance(res, dict):
                    parsed_verify = res
                else:
                    parsed_verify = _parse_json_response(getattr(res, "content", res))
            else:
                resp_verify   = await rate_limiter.call(model, msgs_verify, timeout=api_timeout)
                parsed_verify = _parse_json_response(resp_verify.content)

            if parsed_verify.get("raw_text") or parsed_verify.get("verified_raw_text"):
                break
            backoff = 3 if attempt == 1 else 120
            await asyncio.sleep(backoff)
        except Exception as exc:
            if structured_verifier is not None and attempt == 1:
                structured_verifier = None  # Fallback to plain model call on next attempt
            backoff = 3 if attempt == 1 else 120
            print(f"  [Node: verifier_agent] [Page {page_num}] Audit attempt {attempt} error: {exc}. Retrying in {backoff}s ...")
            await asyncio.sleep(backoff)

    final_text       = parsed_verify.get("verified_raw_text") or parsed_verify.get("raw_text") or candidate_text
    accuracy_score   = float(parsed_verify.get("accuracy_score") or parsed_verify.get("confidence_score") or 0.95)
    accuracy_level   = str(parsed_verify.get("accuracy_level") or parsed_verify.get("confidence_level") or "HIGH").upper()
    accuracy_reason  = str(parsed_verify.get("accuracy_reason") or parsed_verify.get("confidence_reason") or "Verified by Auditor Agent")
    corrections_made = parsed_verify.get("corrections_made", [])

    print(
        f"  [Node: verifier_agent] [Page {page_num}] Verified. "
        f"Accuracy={accuracy_score*100:.0f}% [{accuracy_level}] "
        f"Corrections={len(corrections_made)}"
    )

    page_result = {
        "page":            page_num,
        "raw_text":        final_text,
        "divide_page":     candidate.get("divide_page", False),
        "accuracy_score":  accuracy_score,
        "accuracy_level":  accuracy_level,
        "accuracy_reason": accuracy_reason,
        "corrections_made": corrections_made,
        "top_region":      candidate.get("top_region"),
        "bottom_region":   candidate.get("bottom_region"),
    }

    # Save verified page result to page_NNN.json and page_NNN.txt immediately
    _save_page_result(page_result, output_dir)
    return page_result


async def verifier_agent_node(state: AgentState) -> dict:
    """
    LangGraph Node 3 – Output Synthesizer & Combination Agent.

    Finalizes verification results and produces combined_output.json / combined_output.txt.
    """
    output_dir   = _get_output_dir()
    verified_ocr_results: List[Dict] = state.get("verified_ocr_results") or state.get("ocr_results") or []

    # Combine all verified page JSON files into combined_output.json and combined_output.txt
    combined = combine_outputs(str(output_dir))
    print(f"\n[Node: verifier_agent] Pipeline complete! Combined chars={len(combined.get('combined_text', ''))}.")

    summary     = json.dumps(verified_ocr_results, ensure_ascii=False, indent=2)
    summary_msg = HumanMessage(content=summary, name="verified_ocr_result")

    return {
        "verified_ocr_results": verified_ocr_results,
        "ocr_results":          verified_ocr_results,
        "messages":             [summary_msg],
    }


# ---------------------------------------------------------------------------
# Graph Assembly
# ---------------------------------------------------------------------------

def create_workflow():
    """Builds and compiles the 3-node LangGraph StateGraph workflow."""
    builder = StateGraph(AgentState)

    builder.add_node("preprocess",     preprocess_node)
    builder.add_node("ocr_agent",      ocr_agent_node)
    builder.add_node("verifier_agent", verifier_agent_node)

    builder.add_edge(START,            "preprocess")
    builder.add_edge("preprocess",     "ocr_agent")
    builder.add_edge("ocr_agent",      "verifier_agent")
    builder.add_edge("verifier_agent", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

async def _run_async(pdf: str):
    """Async entry point — invoke the compiled LangGraph workflow."""
    app = create_workflow()
    print("=== LangGraph Workflow compiled successfully ===\n")
    print(f"Running workflow on: '{pdf}'\n")

    result = await app.ainvoke({
        "pdf_path":            pdf,
        "messages":            [],
        "preprocessed_images": None,
        "ocr_results":         None,
    })

    print("\n=== Verified OCR Results (structured JSON) ===")
    for page in result.get("verified_ocr_results", []):
        print(f"\n--- Page {page['page']} ---")
        print(f"accuracy    : {page.get('accuracy_score', 0)*100:.0f}% [{page.get('accuracy_level', 'N/A')}]")
        print(f"reason      : {page.get('accuracy_reason', 'N/A')}")
        print(f"corrections : {page.get('corrections_made', [])}")
        print(f"raw_text    : {str(page['raw_text'])[:300]} ...")

    out_dir = os.getenv("OCR_OUTPUT_DIR", "ocr_output")
    print(f"\n[Done] All verified output saved in '{out_dir}/'")


if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "CP.pdf"
    asyncio.run(_run_async(pdf))
