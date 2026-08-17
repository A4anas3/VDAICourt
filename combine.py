"""
combine.py

Full End-to-End Orchestrator for High Court & Commercial Court Document Verification.
Executes the entire pipeline sequentially from Page 1 to the End (Full Document):
  1. STAGE 1: Full-Page Preprocessing & Multi-Pass Vision OCR (Pages 1 to End)
  2. STAGE 2: Page Classification & Structured Evidence Extraction (Pages 1 to End)
  3. STAGE 3: Multi-Document Cross-Verification, Identity Matching & Legal Audit
"""

import sys
import io
import os
from pathlib import Path
from typing import List, Dict, Any
import asyncio
import json
import argparse

# Fix Windows console UTF-8 output encoding for Hindi Devanagari characters
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage_1.preprocessor import get_pdf_page_count
from stage_1.workflow import create_workflow as create_stage1_workflow, combine_outputs
from stage_2.workflow import run_stage2_pipeline_async
from stage_3.workflow import run_stage3_pipeline_async


async def run_full_pipeline(
    pdf_path: str,
    output_dir: str = "ocr_output",
    force_reocr: bool = False
):
    """
    Executes the complete 3-stage pipeline from Page 1 to the End of the document.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ Error: Target PDF file '{pdf_path}' not found!")
        sys.exit(1)

    total_pages = get_pdf_page_count(str(pdf_file))
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("      🏛️  HIGH COURT & COMMERCIAL COURT VERIFICATION PIPELINE  🏛️")
    print("=" * 80)
    print(f"📄 Target Document   : {pdf_file.name}")
    print(f"📑 Total Page Count  : {total_pages} Pages (Processing FULL from Page 1 to {total_pages})")
    print(f"📂 Output Directory  : {out_path.resolve()}")
    print("=" * 80 + "\n")

    # =========================================================================
    # PHASE 1: STAGE 1 — FULL DOCUMENT VISION OCR & VERIFICATION (PAGE 1 TO END)
    # =========================================================================
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║  PHASE 1: STAGE 1 — FULL VISION OCR & AUDITOR VERIFICATION (PAGES 1 TO END)  ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    
    combined_json_path = out_path / "combined_output.json"
    ocr_results: List[Dict[str, Any]] = []

    if not force_reocr and combined_json_path.exists():
        print(f"ℹ️  Existing OCR cache detected at '{combined_json_path}'. Checking page coverage...")
        try:
            with open(combined_json_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                cached_pages = (
                    cached_data.get("pages") 
                    or cached_data.get("page_extractions") 
                    or cached_data.get("ocr_results") 
                    or []
                ) if isinstance(cached_data, dict) else cached_data
                
                if len(cached_pages) >= total_pages:
                    print(f"✓ Found complete OCR results for all {len(cached_pages)} pages in cache.")
                    ocr_results = cached_pages
                else:
                    print(f"⚠️  Cache has {len(cached_pages)}/{total_pages} pages. Running Stage 1 OCR on full document...")
        except Exception as e:
            print(f"⚠️  Failed to read cache ({e}). Running Stage 1 OCR on full document...")

    if not ocr_results:
        print(f"🚀 Starting Stage 1 LangGraph OCR Workflow on all {total_pages} pages...")
        stage1_app = create_stage1_workflow()

        result = await stage1_app.ainvoke({
            "pdf_path": str(pdf_file),
            "messages": [],
            "preprocessed_images": None,
            "ocr_results": None,
        })

        ocr_results = result.get("verified_ocr_results") or result.get("ocr_results") or []
        
        # Ensure combined_output.json is fully updated and written
        combine_outputs(str(out_path))

    print(f"\n✅ [Stage 1 Complete] Verified OCR generated for {len(ocr_results)} pages.")

    # =========================================================================
    # PHASE 2: STAGE 2 — PAGE CLASSIFICATION & EVIDENCE EXTRACTION (PAGE 1 TO END)
    # =========================================================================
    print("\n╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║  PHASE 2: STAGE 2 — PAGE CLASSIFICATION & EVIDENCE EXTRACTION                ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print(f"🔍 Classifying document types and extracting WHO evidence across all {len(ocr_results)} pages...")

    stage2_output = await run_stage2_pipeline_async(
        ocr_results=ocr_results,
        output_dir=output_dir
    )

    extracted_evidences = stage2_output.get("extracted_evidences", [])
    print(f"\n✅ [Stage 2 Complete] Extracted legal facts & identities from {len(extracted_evidences)} pages.")

    # =========================================================================
    # PHASE 3: STAGE 3 — CROSS-DOCUMENT VERIFICATION & LEGAL INTEGRITY AUDIT
    # =========================================================================
    print("\n╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║  PHASE 3: STAGE 3 — CROSS-VERIFICATION & LEGAL INTEGRITY AUDIT               ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print("⚖️  Cross-checking Parties against Identity Cards, Vakalatnamas & Affidavits...")

    audit_report = await run_stage3_pipeline_async(
        extracted_evidences=extracted_evidences,
        output_dir=output_dir,
        pdf_filename=pdf_file.name
    )

    # =========================================================================
    # FINAL SUMMARY REPORT
    # =========================================================================
    print("\n" + "=" * 80)
    print("                          🏁 FINAL AUDIT SUMMARY REPORT 🏁                     ")
    print("=" * 80)
    print(f"📊 Overall Legitimacy Score : {audit_report.overall_legitimacy_score * 100:.1f}%")
    print(f"⚠️  Risk Assessment Level   : {audit_report.risk_level.value}")
    print(f"📝 Executive Summary        : {audit_report.audit_summary}\n")
    print(f"📑 Discrepancies Flagged    : {len(audit_report.discrepancies_detected)}")
    for d in audit_report.discrepancies_detected[:5]:
        print(f"   • [{d.severity.value}] {d.title}: {d.description}")
    if len(audit_report.discrepancies_detected) > 5:
        print(f"   • ... and {len(audit_report.discrepancies_detected) - 5} more.")

    print("\n📂 Output Files Generated:")
    print(f"   1. Markdown Audit Report : {output_dir}/verification_summary.md")
    print(f"   2. Stage 3 Audit JSON    : {output_dir}/stage3_verification_audit_report.json")
    print(f"   3. Stage 2 Evidence JSON : {output_dir}/stage2_extracted_evidence.json")
    print(f"   4. Stage 2 Classify JSON : {output_dir}/stage2_page_classification.json")
    print(f"   5. Stage 1 OCR Full JSON : {output_dir}/combined_output.json")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Full Document 3-Stage Court Verification Pipeline")
    parser.add_argument(
        "--pdf", 
        type=str, 
        default="PO_R_LS_RAIR_SHRN_RUK1_37748.pdf", 
        help="Path to the court PDF document (default: PO_R_LS_RAIR_SHRN_RUK1_37748.pdf)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="ocr_output", 
        help="Output folder to store all OCR and audit artifacts (default: ocr_output)"
    )
    parser.add_argument(
        "--force-reocr", 
        action="store_true", 
        help="Force re-running Stage 1 OCR even if cached combined_output.json exists"
    )

    args = parser.parse_args()
    asyncio.run(run_full_pipeline(
        pdf_path=args.pdf,
        output_dir=args.output_dir,
        force_reocr=args.force_reocr
    ))


if __name__ == "__main__":
    main()
