"""
stage_2/schemas.py

Pydantic schemas for Stage 2: Page Classification & Category-Specific JSON Evidence Extraction.
Handles bilingual (Hindi Devanagari + English) court transaction documents.
Extracts document type and links every document/evidence to WHO it relates to.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PageCategoryEnum(str, Enum):
    """Supported document page categories in court transaction filings."""
    PEHCHAN_PATRA_IDENTITY_PROOF = "PEHCHAN_PATRA_IDENTITY_PROOF"  # Voter ID, Aadhaar, PAN card, Driving License, Photo ID
    MEMO_OF_PARTIES = "MEMO_OF_PARTIES"                           # Cause title listing Petitioners vs Respondents
    VAKALATNAMA = "VAKALATNAMA"                                   # Power of Attorney appointing Advocates
    AFFIDAVIT = "AFFIDAVIT"                                       # Deposition / Oath verification statement with stamps
    PETITION_TEXT = "PETITION_TEXT"                               # Main body, facts, grounds, relief prayed
    COURT_ORDER_OR_JUDGMENT = "COURT_ORDER_OR_JUDGMENT"           # Order sheet / interim/final judgment by judge
    RECEIPT_OR_FEE_TRANSACTION = "RECEIPT_OR_FEE_TRANSACTION"     # Court fee receipt, treasury challan, stamp duty
    EVIDENCE_OR_ANNEXURE = "EVIDENCE_OR_ANNEXURE"                 # Annexure ('संलग्नक') / Exhibit ('प्रदर्श') copies
    OTHER_OFFICIAL_DOCUMENT = "OTHER_OFFICIAL_DOCUMENT"           # Registry memo, office diary, dispatch list


class PageClassificationResult(BaseModel):
    """Pydantic schema for Stage 2 Page Classifier."""
    page_number: int = Field(description="1-based page number")
    category: PageCategoryEnum = Field(description="Primary category classification of the page")
    document_title: str = Field(description="Extracted header/title of document on this page (Hindi or English)")
    detected_language: str = Field(default="Hindi/English", description="Language of text on page")
    contains_identity_proof: bool = Field(default=False, description="True if page has a Voter ID, Aadhaar, PAN card or Pehchan Patra image/text")
    contains_vakalatnama: bool = Field(default=False, description="True if page is a Vakalatnama or advocate power of attorney")
    contains_affidavit: bool = Field(default=False, description="True if page is a sworn affidavit with deponent details")
    classification_confidence: float = Field(default=0.9, description="Confidence score between 0.0 and 1.0")


class IdentityProofEvidence(BaseModel):
    """Extracted identity proof (Pehchan Patra) details and WHO it belongs to."""
    person_name: str = Field(description="Full name of person printed/written on identity document")
    related_party_role: Optional[str] = Field(default=None, description="Role of this person in suit (e.g. Petitioner 1, Respondent, Deponent)")
    father_or_husband_name: Optional[str] = Field(default=None, description="Father's or Husband's name if present")
    id_type: str = Field(description="Type of identity proof: Voter ID / Pehchan Patra / Aadhaar / PAN / Driving License")
    id_number: Optional[str] = Field(default=None, description="Exact ID card number (e.g. Voter ID card no, Aadhaar no, PAN no)")
    dob_or_age: Optional[str] = Field(default=None, description="Date of birth or age listed on ID")
    address: Optional[str] = Field(default=None, description="Address printed on ID card")
    photo_present: bool = Field(default=False, description="True if photo box / photograph is present on ID")
    issuing_authority: Optional[str] = Field(default=None, description="Issuing authority (e.g. Election Commission of India, UIDAI, Income Tax Dept)")


class PartyEvidence(BaseModel):
    """Extracted party (Petitioner / Respondent / Applicant / Defendant) detail."""
    role: str = Field(description="Role: Petitioner / Plaintiff / Appellant / Respondent / Defendant / Opposite Party")
    full_name: str = Field(description="Full name of the party")
    father_or_husband_name: Optional[str] = Field(default=None, description="Father's or Husband's name")
    age: Optional[str] = Field(default=None, description="Age if stated")
    address: Optional[str] = Field(default=None, description="Address of party")
    advocate_name: Optional[str] = Field(default=None, description="Representing advocate / counsel name")


class VakalatnamaEvidence(BaseModel):
    """Extracted Vakalatnama (Power of Attorney) details and WHO is involved."""
    authorizing_clients: List[str] = Field(default_factory=list, description="Names of client(s) authorizing the advocate")
    appointed_advocates: List[str] = Field(default_factory=list, description="Names of advocate(s) appointed")
    advocate_enrolment_numbers: List[str] = Field(default_factory=list, description="Enrolment / Registration numbers of advocates (e.g., UP/1234/2020)")
    client_signatures_present: bool = Field(default=False, description="True if client signature / thumb impression is present")
    advocate_signatures_present: bool = Field(default=False, description="True if advocate signature / acceptance is present")


class AffidavitEvidence(BaseModel):
    """Extracted Affidavit sworn statement details and WHO sworn it."""
    deponent_name: str = Field(description="Full name of the deponent taking oath")
    father_or_husband_name: Optional[str] = Field(default=None, description="Father's or Husband's name of deponent")
    age: Optional[str] = Field(default=None, description="Age of deponent")
    address: Optional[str] = Field(default=None, description="Address of deponent")
    verifier_name: Optional[str] = Field(default=None, description="Advocate or authority verifying deponent")
    notary_or_court_stamp_present: bool = Field(default=False, description="True if notary or court commissioner stamp is present")
    verification_date: Optional[str] = Field(default=None, description="Date of affidavit verification")


class TransactionFeeEvidence(BaseModel):
    """Extracted financial receipt / court fee transaction details and WHO paid/received."""
    receipt_or_challan_no: Optional[str] = Field(default=None, description="Receipt / Challan / Stamp serial number")
    amount_rs: Optional[str] = Field(default=None, description="Transaction amount in Rupees (Rs. / ₹)")
    transaction_date: Optional[str] = Field(default=None, description="Date of transaction / receipt")
    payer_name: Optional[str] = Field(default=None, description="Name of person or party paying")
    payee_or_authority: Optional[str] = Field(default=None, description="Name of authority / court receiving payment")


class StructuredPageEvidence(BaseModel):
    """Combined structured evidence extracted from a single classified page."""
    page_number: int = Field(description="1-based page number")
    category: PageCategoryEnum = Field(description="Category of document page")
    document_title: str = Field(description="Title of document on page")
    
    related_person_name: Optional[str] = Field(default=None, description="Primary person or party this document relates to (e.g., 'Devi Rai', 'Sharmila Devi')")
    related_person_role: Optional[str] = Field(default=None, description="Role of the related person (Petitioner / Respondent / Deponent / Advocate / Signatory / Payer)")
    document_summary: str = Field(default="", description="1-sentence explanation of what this page is and WHO it relates to")
    
    # Category-specific evidence fields
    identity_proofs: List[IdentityProofEvidence] = Field(default_factory=list)
    parties: List[PartyEvidence] = Field(default_factory=list)
    vakalatnama: Optional[VakalatnamaEvidence] = Field(default=None)
    affidavit: Optional[AffidavitEvidence] = Field(default=None)
    financial_transaction: Optional[TransactionFeeEvidence] = Field(default=None)
    
    # Generic entity summaries for quick cross-indexing
    all_extracted_names: List[str] = Field(default_factory=list, description="All person names mentioned on this page")
    all_extracted_id_numbers: List[str] = Field(default_factory=list, description="All ID numbers (Voter ID, Aadhaar, PAN) on page")
    key_dates: List[str] = Field(default_factory=list, description="Dates appearing on this page")
