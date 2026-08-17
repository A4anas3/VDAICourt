"""
main.py

End-to-End CLI Pipeline Orchestrator for High Court / Commercial Court Paper PDF Verification.
Executes the 3-Stage Pipeline:
  Stage 1: Multi-pass Vision OCR & Image Cleanup
  Stage 2: Page Classification & Category-Specific JSON Evidence Extraction (Pehchan Patra, Vakalatnama, Affidavits, Orders)
  Stage 3: Multi-Document Cross-Verification & Legitimate User Audit
"""

import sys
import io
# Fix Windows terminal encoding for Hindi Devanagari text
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any

from stage_1.workflow import create_workflow as create_stage1_workflow
from stage_1.preprocessor import get_pdf_page_count
from stage_2.workflow import run_stage2_pipeline
from stage_3.workflow import run_stage3_pipeline


def parse_page_range(pages_str: str, max_pages: int) -> List[int]:
    """Parses page selection string e.g. '1-10' or '1,3,5' or 'all'."""
    if not pages_str or pages_str.lower() == "all":
        return list(range(1, max_pages + 1))
    
    selected = set()
    parts = pages_str.split(",")
    for part in parts:
        part = part.strip()
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = max(1, int(start_str))
            end = min(max_pages, int(end_str))
            for p in range(start, end + 1):
                selected.add(p)
        else:
            p = int(part)
            if 1 <= p <= max_pages:
                selected.add(p)
    return sorted(list(selected))


async def run_pipeline(pdf_path: str, pages_arg: str = "1-10", output_dir: str = "ocr_output", skip_stage1: bool = False):
    """Executes the full 3-Stage Court Document Analysis & Verification Pipeline."""
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"Error: PDF file '{pdf_path}' not found!")
        sys.exit(1)

    total_pdf_pages = get_pdf_page_count(str(pdf_file))
    target_pages = parse_page_range(pages_arg, total_pdf_pages)

    print("===========================================================================")
    print("      COURT DOCUMENT MULTI-STAGE ANALYSIS & CROSS-VERIFICATION PIPELINE    ")
    print("===========================================================================")
    print(f"Target PDF         : {pdf_file.name}")
    print(f"Total Document Pages: {total_pdf_pages}")
    print(f"Target Processing Pages ({len(target_pages)}): {target_pages[:15]}{'...' if len(target_pages)>15 else ''}")
    print(f"Output Directory   : {output_dir}/")
    print("===========================================================================\n")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    combined_json_path = out_path / "combined_output.json"

    ocr_results: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------------------
    # Stage 1: Preprocessing & Vision OCR
    # ---------------------------------------------------------------------------
    if skip_stage1 and combined_json_path.exists():
        print(f"[Stage 1] Skipping OCR pass — loading cached output from '{combined_json_path}'")
        with open(combined_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            ocr_results = (data.get("pages") or data.get("page_extractions") or data.get("ocr_results") or []) if isinstance(data, dict) else data

    else:
        print("[Stage 1] Running Preprocessing & Multi-pass Vision OCR...")
        stage1_app = create_stage1_workflow()
        
        # Invoke Stage 1 workflow
        result = await stage1_app.ainvoke({
            "pdf_path": str(pdf_file),
            "messages": [],
            "preprocessed_images": None,
            "ocr_results": None,
        })
        
        ocr_results = result.get("verified_ocr_results") or result.get("ocr_results") or []

    # Filter OCR results to requested page range
    if target_pages and len(target_pages) < total_pdf_pages:
        target_set = set(target_pages)
        ocr_results = [p for p in ocr_results if p.get("page", 1) in target_set]

    if not ocr_results:
        print("Warning: No OCR results found for specified pages!")
        return

    # ---------------------------------------------------------------------------
    # Stage 2: Page Classification & JSON Evidence Extraction
    # ---------------------------------------------------------------------------
    stage2_output = run_stage2_pipeline(ocr_results, output_dir=output_dir)
    extracted_evidences = stage2_output.get("extracted_evidences", [])

    # ---------------------------------------------------------------------------
    # Stage 3: Multi-Document Cross-Verification & Legitimacy Audit Engine
    # ---------------------------------------------------------------------------
    audit_report = run_stage3_pipeline(
        extracted_evidences,
        output_dir=output_dir,
        pdf_filename=pdf_file.name
    )

    print("\n===========================================================================")
    print("                          FINAL VERIFICATION RESULTS                       ")
    print("===========================================================================")
    print(f"Legitimacy Score : {audit_report.overall_legitimacy_score*100:.0f}%")
    print(f"Risk Assessment  : {audit_report.risk_level.value}")
    print(f"Summary          : {audit_report.audit_summary}\n")
    print(f"Markdown Report  : {output_dir}/verification_summary.md")
    print(f"Audit JSON       : {output_dir}/stage3_verification_audit_report.json")
    print("===========================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="High Court / Commercial Court Paper PDF Verification Pipeline")
    parser.add_argument("--pdf", type=str, default="PO_R_LS_RAIR_SHRN_RUK1_37748.pdf", help="Path to court PDF file")
    parser.add_argument("--pages", type=str, default="1-10", help="Page range to analyze (e.g., '1-10', '1,2,5', or 'all')")
    parser.add_argument("--output-dir", type=str, default="ocr_output", help="Directory to save output files")
    parser.add_argument("--skip-stage1", action="store_true", help="Skip Stage 1 OCR if combined_output.json exists")

    args = parser.parse_args()
    asyncio.run(run_pipeline(args.pdf, pages_arg=args.pages, output_dir=args.output_dir, skip_stage1=args.skip_stage1))


if __name__ == "__main__":
    main()
