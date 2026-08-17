"""
stage_2/workflow.py

Async Orchestrator for Stage 2: Page Classification and WHO Evidence Extraction.
Executes all pages concurrently using 15 async workers across the 5-key Rate Limiter pool.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path when script is executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple

from stage_2.schemas import PageClassificationResult, StructuredPageEvidence
from stage_2.classifier import classify_page_text_async
from stage_2.extractor import extract_page_evidence_async
from stage_1.model import init_model
from stage_1.rate_limiter import make_async_rate_limiter, AsyncRateLimiter


async def _process_single_page_stage2(
    page_item: Dict[str, Any],
    model,
    rate_limiter: AsyncRateLimiter,
    semaphore: asyncio.Semaphore
) -> Tuple[PageClassificationResult, StructuredPageEvidence]:
    """Processes a single page's classification and category evidence extraction asynchronously."""
    p_num = page_item.get("page", 1)
    raw_text = page_item.get("verified_raw_text") or page_item.get("raw_text", "")

    async with semaphore:
        cls_result = await classify_page_text_async(p_num, raw_text, model=model, rate_limiter=rate_limiter)
        who_str = f" | WHO: {cls_result.document_title}"
        print(f"[Stage 2] Page {p_num:03d} └─ Category: [{cls_result.category.value}]{who_str}")

        ev_result = await extract_page_evidence_async(p_num, raw_text, cls_result, model=model, rate_limiter=rate_limiter)
        if ev_result.identity_proofs:
            print(f"     ★ Page {p_num:03d}: Found {len(ev_result.identity_proofs)} Identity Proof(s) (Pehchan Patra)")
        if ev_result.vakalatnama:
            print(f"     ★ Page {p_num:03d}: Found Vakalatnama authorization details")

        return cls_result, ev_result


async def run_stage2_pipeline_async(
    ocr_results: List[Dict[str, Any]],
    output_dir: str = "ocr_output",
    model=None
) -> Dict[str, Any]:
    """
    Async Stage 2 Pipeline: Executes all page classifications & WHO evidence extractions concurrently.
    """
    if model is None:
        model = init_model()

    rate_limiter = make_async_rate_limiter()
    semaphore = asyncio.Semaphore(15)

    print(f"\n=== Starting Stage 2: Page Classification & Evidence Extraction ({len(ocr_results)} pages) ===")
    print(f"[Stage 2] Running 15 concurrent async workers across 5 API keys...")

    tasks = [
        _process_single_page_stage2(page_item, model, rate_limiter, semaphore)
        for page_item in ocr_results
    ]

    results = await asyncio.gather(*tasks)

    # Sort results by page number
    results.sort(key=lambda item: item[0].page_number)

    classifications = [r[0] for r in results]
    extracted_evidences = [r[1] for r in results]

    # Output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cls_json_path = out_path / "stage2_page_classification.json"
    ev_json_path = out_path / "stage2_extracted_evidence.json"

    cls_dicts = [c.model_dump() for c in classifications]
    ev_dicts = [e.model_dump() for e in extracted_evidences]

    with open(cls_json_path, "w", encoding="utf-8") as f:
        json.dump(cls_dicts, f, ensure_ascii=False, indent=2)

    with open(ev_json_path, "w", encoding="utf-8") as f:
        json.dump(ev_dicts, f, ensure_ascii=False, indent=2)

    print(f"\n[Stage 2 Complete] Saved classifications to '{cls_json_path}' and evidence to '{ev_json_path}'")

    return {
        "classifications": cls_dicts,
        "extracted_evidences": ev_dicts,
    }


def run_stage2_pipeline(
    ocr_results: List[Dict[str, Any]],
    output_dir: str = "ocr_output",
    model=None
) -> Dict[str, Any]:
    """Sync entry point for run_stage2_pipeline."""
    return asyncio.run(run_stage2_pipeline_async(ocr_results, output_dir=output_dir, model=model))

if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "ocr_output"
    out_path = Path(output_dir)
    combined_json = out_path / "combined_output.json"
    
    ocr_results = []
    if combined_json.exists():
        with open(combined_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                ocr_results = data.get("pages") or data.get("page_extractions") or data.get("ocr_results") or []
            elif isinstance(data, list):
                ocr_results = data

    if not ocr_results:
        # Fallback: load all individual page_*.json files
        page_files = sorted(out_path.glob("page_*.json"))
        for pf in page_files:
            if not pf.name.startswith("stage"):
                with open(pf, "r", encoding="utf-8") as f:
                    ocr_results.append(json.load(f))

    if not ocr_results:
        print(f"Error: No OCR result JSON files found in '{output_dir}/'. Run Stage 1 first!")
        sys.exit(1)

    asyncio.run(run_stage2_pipeline_async(ocr_results, output_dir=output_dir))
