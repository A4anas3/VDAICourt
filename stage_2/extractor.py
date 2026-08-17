"""
stage_2/extractor.py

Async category-specific JSON evidence extractor for Stage 2.
Takes raw text and classification, and returns a StructuredPageEvidence object using the 5-key Rate Limiter pool.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from stage_1.model import init_model
from stage_1.rate_limiter import make_async_rate_limiter, AsyncRateLimiter
from stage_2.schemas import (
    StructuredPageEvidence,
    PageClassificationResult,
    PageCategoryEnum,
)
from stage_2.prompt import EVIDENCE_EXTRACTION_SYSTEM_PROMPT


async def extract_page_evidence_async(
    page_number: int,
    raw_text: str,
    classification: PageClassificationResult,
    model=None,
    rate_limiter: Optional[AsyncRateLimiter] = None
) -> StructuredPageEvidence:
    """Async category-specific evidence extractor using MultiKeyAsyncRateLimiter 5-key pool."""
    if not raw_text or not raw_text.strip() or "too blurry" in raw_text.lower():
        return StructuredPageEvidence(
            page_number=page_number,
            category=classification.category,
            document_title=classification.document_title,
        )

    if model is None:
        model = init_model()
    if rate_limiter is None:
        rate_limiter = make_async_rate_limiter()

    user_payload = (
        f"Page Number: {page_number}\n"
        f"Classification Category: {classification.category.value}\n"
        f"Document Title: {classification.document_title}\n\n"
        f"Page Text:\n{raw_text}"
    )

    messages = [
        SystemMessage(content=EVIDENCE_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=user_payload)
    ]

    try:
        evidence: StructuredPageEvidence = await rate_limiter.call(
            model, messages, schema=StructuredPageEvidence
        )
        evidence.page_number = page_number
        evidence.category = classification.category
        evidence.document_title = classification.document_title
        return evidence
    except Exception as e:
        print(f"[Extractor] LLM Extraction error on page {page_number}: {e}")
        return StructuredPageEvidence(
            page_number=page_number,
            category=classification.category,
            document_title=classification.document_title,
        )



def extract_page_evidence(
    page_number: int,
    raw_text: str,
    classification: PageClassificationResult,
    model=None
) -> StructuredPageEvidence:
    """Sync wrapper for evidence extraction."""
    import asyncio
    return asyncio.run(extract_page_evidence_async(page_number, raw_text, classification, model=model))
