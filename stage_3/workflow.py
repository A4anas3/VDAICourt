"""
stage_3/workflow.py

Orchestrator for Stage 3: Multi-Document Cross-Verification & 100% Pure LLM Executive Audit Opinion.
Uses Gemini AI for cross-verification, entity resolution, and audit synthesis.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path when script is executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
from typing import List, Dict, Any, Optional

from stage_1.model import init_model
from stage_1.rate_limiter import make_async_rate_limiter
from stage_3.schemas import VerificationAuditReport
from stage_3.verifier import run_cross_verification_llm_async


def generate_verification_summary_md(report: VerificationAuditReport, extracted_evidences: Optional[List[Dict[str, Any]]] = None) -> str:
    """Generates an Executive Case Summary & Comprehensive Audit Report in Markdown format."""
    md = []
    md.append(f"# Comprehensive Court Document Verification & Executive Case Summary Report")
    md.append(f"**PDF File**: `{report.pdf_filename}`  ")
    md.append(f"**Total Pages Analyzed**: `{report.total_pages_analyzed}`  ")
    md.append(f"**Overall Legitimacy Score**: `{report.overall_legitimacy_score * 100:.0f}%`  ")
    md.append(f"**Risk Assessment Level**: `{report.risk_level.value}`  \n")

    md.append(f"---")
    md.append(f"## 🤖 Gemini LLM Executive Case Audit & Property Summary")
    md.append(f"{report.audit_summary}\n")

    md.append(f"---")
    md.append(f"### 1. Party Identity & Pehchan Patra Verification Roster")
    md.append(f"| Primary Party Name | Role | Pehchan Patra Found? | ID Type & Number | Match Status | Audit Notes |")
    md.append(f"|---|---|---|---|---|---|")
    for p in report.party_verifications:
        id_str = f"{p.matched_id_type or 'N/A'}: {p.matched_id_number or 'N/A'}" if p.pehchan_patra_found else "None"
        found_str = f"YES (Pg {p.pehchan_patra_page_no})" if p.pehchan_patra_found else "NO"
        md.append(f"| **{p.claimed_name}** | {p.role} | {found_str} | {id_str} | `{p.name_match_status.value}` | {p.audit_notes} |")
    md.append("\n")

    md.append(f"---")
    md.append(f"### 2. Document Ownership & WHO Mapping Matrix")
    md.append(f"| Page No | Document Category | Related Person (WHO) | Role | Summary & Key Evidence |")
    md.append(f"|---|---|---|---|---|")
    
    if extracted_evidences:
        for ev in extracted_evidences:
            p_num = ev.get("page_number", 1)
            cat = ev.get("category", "OTHER")
            who_name = ev.get("related_person_name") or (ev.get("all_extracted_names")[0] if ev.get("all_extracted_names") else "General Court Document")
            who_role = ev.get("related_person_role") or "Mentioned Party"
            summary = ev.get("document_summary") or ev.get("document_title") or "Court Paper"
            md.append(f"| Page {p_num:03d} | `{cat}` | **{who_name}** | {who_role} | {summary} |")
    else:
        md.append("| - | - | - | - | Evidence data pending |")
    md.append("\n")

    md.append(f"---")
    md.append(f"### 3. Discrepancies & Anomaly Audit Flags")
    if not report.discrepancies:
        md.append("✓ *No critical discrepancies or fraudulent identity flags detected.*\n")
    else:
        for idx, d in enumerate(report.discrepancies, 1):
            severity_icon = "🔴" if d.severity == "HIGH" else "🟡"
            md.append(f"#### {idx}. {severity_icon} [{d.severity}] {d.issue_category}")
            md.append(f"- **Description**: {d.description}")
            md.append(f"- **Action Recommendation**: {d.action_recommendation}")
            if d.page_numbers:
                md.append(f"- **Involved Pages**: Page(s) {', '.join(map(str, d.page_numbers))}")
            md.append("")

    md.append(f"---")
    md.append(f"### 4. Classified Page Manifest")
    for cat, pages in report.page_manifest.items():
        md.append(f"- **{cat}**: Pages `[{', '.join(map(str, pages))}]`")

    return "\n".join(md)


async def run_stage3_pipeline_async(
    extracted_evidences: List[Dict[str, Any]],
    output_dir: str = "ocr_output",
    pdf_filename: str = "PO_R_LS_RAIR_SHRN_RUK1_37748.pdf",
    model=None
) -> VerificationAuditReport:
    """
    Async Stage 3 Pipeline: Performs 100% Pure LLM Verification & Executive Audit Synthesis.
    """
    print(f"\n=== Starting Stage 3: Multi-Document Cross-Verification & Pure LLM Audit ===")
    
    rate_limiter = make_async_rate_limiter()
    report = await run_cross_verification_llm_async(
        extracted_evidences, pdf_filename=pdf_filename, model=model, rate_limiter=rate_limiter
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    report_json_path = out_path / "stage3_verification_audit_report.json"
    report_md_path = out_path / "verification_summary.md"

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, ensure_ascii=False, indent=2)

    md_content = generate_verification_summary_md(report, extracted_evidences=extracted_evidences)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n[Stage 3 Complete] Saved Audit JSON to '{report_json_path}' and Markdown summary to '{report_md_path}'")
    print(f"Legitimacy Score: {report.overall_legitimacy_score*100:.0f}% [{report.risk_level.value}]")

    return report


def run_stage3_pipeline(
    extracted_evidences: List[Dict[str, Any]],
    output_dir: str = "ocr_output",
    pdf_filename: str = "PO_R_LS_RAIR_SHRN_RUK1_37748.pdf",
    model=None
) -> VerificationAuditReport:
    """Sync wrapper for run_stage3_pipeline."""
    return asyncio.run(run_stage3_pipeline_async(extracted_evidences, output_dir=output_dir, pdf_filename=pdf_filename, model=model))


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "ocr_output"
    pdf_filename = sys.argv[2] if len(sys.argv) > 2 else "PO_R_LS_RAIR_SHRN_RUK1_37748.pdf"
    
    out_path = Path(output_dir)
    ev_json_path = out_path / "stage2_extracted_evidence.json"
    
    if not ev_json_path.exists():
        print(f"Error: '{ev_json_path}' not found! Run Stage 2 first using 'python stage_2/workflow.py'")
        sys.exit(1)

    with open(ev_json_path, "r", encoding="utf-8") as f:
        extracted_evidences = json.load(f)

    asyncio.run(run_stage3_pipeline_async(extracted_evidences, output_dir=output_dir, pdf_filename=pdf_filename))
