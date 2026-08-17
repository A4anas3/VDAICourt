"""
stage_1/schemas.py

Pydantic data models for structured output generation in the High Court / Commercial Court paper PDF pipeline.
Used with LangChain's model.with_structured_output(...).
"""

from typing import List, Literal
from pydantic import BaseModel, Field


class PageOCRResult(BaseModel):
    """Pydantic schema for Pass 1 Candidate OCR Extraction."""
    raw_text: str = Field(
        description="ALL raw extracted Hindi and English text from top to bottom of the page image EXACTLY as printed"
    )
    divide_page: bool = Field(
        default=False,
        description="Set to true if page is blurry, too blurry, faint, or hard to read, to divide the image into two zoomed halves for re-scanning"
    )
    accuracy_score: float = Field(
        description="Assessed OCR accuracy score between 0.0 and 1.0 based on document clarity and legibility"
    )
    accuracy_level: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        default="MEDIUM",
        description="Accuracy level based on print quality and legibility"
    )
    accuracy_reason: str = Field(
        default="Candidate Extraction",
        description="Detailed explanation of OCR accuracy (assess print clarity, handwriting legibility, stamp sharpness)"
    )


class PageVerifyResult(BaseModel):
    """Pydantic schema for Pass 3 Verifier & Auditor Agent."""
    verified_raw_text: str = Field(
        description="Final audited and corrected Hindi and English text from top to bottom"
    )
    accuracy_score: float = Field(
        default=0.95,
        description="Audited accuracy score between 0.0 and 1.0 after verifying digits, dates, stamps, and correcting text"
    )
    accuracy_level: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        default="HIGH",
        description="Audited accuracy level"
    )
    accuracy_reason: str = Field(
        default="Audited by Verifier Agent",
        description="Detailed explanation of audit findings and accuracy calculation"
    )
    corrections_made: List[str] = Field(
        default_factory=list,
        description="List of specific corrections made during audit"
    )
