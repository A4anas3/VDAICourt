"""
stage_3/schemas.py

Pydantic schemas for Stage 3: Multi-Document Cross-Verification & Legitimacy Audit Report.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class MatchStatusEnum(str, Enum):
    """Name & Identity match status."""
    EXACT_MATCH = "EXACT_MATCH"
    PHONETIC_OR_TRANSLITERATION_MATCH = "PHONETIC_OR_TRANSLITERATION_MATCH"
    MINOR_SPELLING_VARIATION = "MINOR_SPELLING_VARIATION"
    MAJOR_NAME_MISMATCH = "MAJOR_NAME_MISMATCH"
    IDENTITY_PROOF_MISSING = "IDENTITY_PROOF_MISSING"
    UNVERIFIED = "UNVERIFIED"


class LegitimacyRiskLevelEnum(str, Enum):
    """Legitimacy audit risk levels."""
    LOW_RISK = "LOW_RISK"                 # Fully verified legitimate user & legal authorization
    MEDIUM_RISK = "MEDIUM_RISK"           # Minor spelling variations or unverified secondary parties
    HIGH_RISK = "HIGH_RISK"               # Unauthorized signatory, missing mandatory identity proof, or major name discrepancy


class PartyVerificationDetail(BaseModel):
    """Audit verification record for a specific party / person in the legal filing."""
    claimed_name: str = Field(description="Name of party listed in Memo of Parties / Petition")
    role: str = Field(description="Role: Petitioner / Respondent / Deponent")
    
    # Cross-document audit flags
    pehchan_patra_found: bool = Field(default=False, description="True if Pehchan Patra / Identity proof image/record exists in PDF")
    pehchan_patra_name: Optional[str] = Field(default=None, description="Exact name printed on Pehchan Patra card")
    matched_id_type: Optional[str] = Field(default=None, description="Type of ID card found (Voter ID / Aadhaar / PAN)")
    matched_id_number: Optional[str] = Field(default=None, description="Exact ID card number extracted")
    pehchan_patra_page_no: Optional[int] = Field(default=None, description="Page number where Pehchan Patra was found")
    
    vakalatnama_signed: bool = Field(default=False, description="True if client signed Vakalatnama authorizing counsel")
    vakalatnama_page_no: Optional[int] = Field(default=None, description="Page number of Vakalatnama")
    
    affidavit_signed: bool = Field(default=False, description="True if party signed sworn affidavit statement")
    affidavit_page_no: Optional[int] = Field(default=None, description="Page number of affidavit")
    
    name_match_status: MatchStatusEnum = Field(default=MatchStatusEnum.UNVERIFIED, description="Match status comparing Pehchan Patra to Petition/Memo")
    legitimacy_status: str = Field(default="UNVERIFIED", description="VERIFIED_LEGITIMATE / UNVERIFIED_MISSING_ID / SUSPICIOUS_MISMATCH")
    audit_notes: str = Field(default="", description="Detailed cross-verification explanation")


class DiscrepancyFlag(BaseModel):
    """Specific discrepancy or anomaly identified during audit."""
    issue_category: str = Field(description="CATEGORY: NAME_MISMATCH / MISSING_IDENTITY_PROOF / VAKALATNAMA_AUTHORIZATION_MISSING / DATE_ANOMALY")
    severity: str = Field(description="HIGH / MEDIUM / LOW")
    page_numbers: List[int] = Field(default_factory=list, description="Pages involved in discrepancy")
    description: str = Field(description="Detailed explanation of discrepancy")
    action_recommendation: str = Field(description="Recommended action or verification step")


class VerificationAuditReport(BaseModel):
    """Complete Stage 3 Cross-Verification Audit Report."""
    pdf_filename: str = Field(description="Name of source PDF file")
    total_pages_analyzed: int = Field(description="Total number of pages analyzed")
    
    overall_legitimacy_score: float = Field(description="Overall legitimacy confidence score (0.0 to 1.0 or 0-100%)")
    risk_level: LegitimacyRiskLevelEnum = Field(description="Overall risk level assessment")
    audit_summary: str = Field(description="Executive summary of verification findings")
    
    party_verifications: List[PartyVerificationDetail] = Field(default_factory=list, description="Detailed verification status per party")
    discrepancies: List[DiscrepancyFlag] = Field(default_factory=list, description="List of all flagged discrepancies")
    
    page_manifest: Dict[str, List[int]] = Field(
        default_factory=dict,
        description="Manifest mapping document categories (Pehchan Patra, Vakalatnama, Memo, Affidavit, Orders) to page numbers"
    )
