"""
stage_2/classifier.py

Async page classifier for Stage 2.
Takes raw page OCR text and produces a PageClassificationResult using the 5-key Rate Limiter pool.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from stage_1.model import init_model
from stage_1.rate_limiter import make_async_rate_limiter, AsyncRateLimiter
from stage_2.schemas import PageClassificationResult, PageCategoryEnum
from stage_2.prompt import PAGE_CLASSIFIER_SYSTEM_PROMPT


def _fallback_classification(page_number: int, raw_text: str) -> PageClassificationResult:
    """Rule-based fallback classification if LLM structured call fails."""
    text_lower = raw_text.lower()
    cat = PageCategoryEnum.PETITION_TEXT
    title = "Court Paper"
    
    if "पहचान पत्र" in raw_text or "निर्वाचन" in raw_text or "elector" in text_lower or "voter" in text_lower or "identity card" in text_lower or "pan card" in text_lower or "aadhaar" in text_lower:
        cat = PageCategoryEnum.PEHCHAN_PATRA_IDENTITY_PROOF
        title = "Pehchan Patra / Identity Proof"
    elif "वकालतनामा" in raw_text or "vakalatnama" in text_lower:
        cat = PageCategoryEnum.VAKALATNAMA
        title = "Vakalatnama"
    elif "शपथ पत्र" in raw_text or "affidavit" in text_lower:
        cat = PageCategoryEnum.AFFIDAVIT
        title = "Affidavit / Oath Statement"
    elif "बनाम" in raw_text or "versus" in text_lower or "vs." in text_lower or "वादी" in raw_text:
        cat = PageCategoryEnum.MEMO_OF_PARTIES
        title = "Memo of Parties"
    elif "आदेश" in raw_text or "order" in text_lower or "judgement" in text_lower:
        cat = PageCategoryEnum.COURT_ORDER_OR_JUDGMENT
        title = "Court Order / Judgment"
    elif "संलग्नक" in raw_text or "annexure" in text_lower or "exhibit" in text_lower:
        cat = PageCategoryEnum.EVIDENCE_OR_ANNEXURE
        title = "Annexure / Evidence"

    return PageClassificationResult(
        page_number=page_number,
        category=cat,
        document_title=title,
        contains_identity_proof=(cat == PageCategoryEnum.PEHCHAN_PATRA_IDENTITY_PROOF),
        contains_vakalatnama=(cat == PageCategoryEnum.VAKALATNAMA),
        contains_affidavit=(cat == PageCategoryEnum.AFFIDAVIT),
        classification_confidence=0.7,
    )


async def classify_page_text_async(
    page_number: int,
    raw_text: str,
    model=None,
    rate_limiter: Optional[AsyncRateLimiter] = None
) -> PageClassificationResult:
    """Async page classifier using MultiKeyAsyncRateLimiter 5-key pool."""
    if not raw_text or not raw_text.strip() or "too blurry" in raw_text.lower():
        return PageClassificationResult(
            page_number=page_number,
            category=PageCategoryEnum.OTHER_OFFICIAL_DOCUMENT,
            document_title="Blurry / Unreadable Page",
            classification_confidence=0.5,
        )

    if model is None:
        model = init_model()
    if rate_limiter is None:
        rate_limiter = make_async_rate_limiter()

    messages = [
        SystemMessage(content=PAGE_CLASSIFIER_SYSTEM_PROMPT),
        HumanMessage(content=f"Page Number: {page_number}\n\nPage Raw Text:\n{raw_text}")
    ]

    try:
        result: PageClassificationResult = await rate_limiter.call(
            model, messages, schema=PageClassificationResult
        )
        result.page_number = page_number
        return result
    except Exception as e:
        print(f"[Classifier] Error on page {page_number}: {e}")
        return _fallback_classification(page_number, raw_text)



def classify_page_text(page_number: int, raw_text: str, model=None) -> PageClassificationResult:
    """Sync wrapper for classification."""
    import asyncio
    return asyncio.run(classify_page_text_async(page_number, raw_text, model=model))
