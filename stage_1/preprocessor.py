"""
stage_1/preprocessor.py

OpenCV-based image preprocessing pipeline for scanned black-and-white
handwritten Court and legal PDFs. Runs BEFORE the OCR / LLM step.

Pipeline steps (in order):
  1. Load PDF pages → numpy images via PyMuPDF
  2. Upscale to target DPI (default 300 DPI)
  3. Grayscale conversion
  4. Denoising  (cv2.fastNlMeansDenoising)
  5. Deskew     (rotate to correct slight page tilt)
  6. Binarise   (Sauvola / Otsu adaptive threshold)
  7. Sharpen    (unsharp mask)
  8. Morphology (remove isolated speckles)
  9. Return list of cleaned PIL Images ready for the OCR node
"""

import os
import math
import numpy as np
import cv2
import fitz                     # PyMuPDF  –  pip install pymupdf
from PIL import Image
from typing import List, Optional


# ---------------------------------------------------------------------------
# Tunable defaults  (override via kwargs or environment variables)
# ---------------------------------------------------------------------------
DEFAULT_TARGET_DPI    = int(os.getenv("PREPROCESS_DPI",    "240"))
DEFAULT_DENOISE_H     = int(os.getenv("PREPROCESS_DENOISE_H",  "10"))
DEFAULT_DENOISE_TEMPLATE = int(os.getenv("PREPROCESS_DENOISE_TEMPLATE", "7"))
DEFAULT_DENOISE_SEARCH   = int(os.getenv("PREPROCESS_DENOISE_SEARCH",   "21"))
DEFAULT_SHARPEN_AMOUNT   = float(os.getenv("PREPROCESS_SHARPEN", "1.5"))
DEFAULT_MORPH_KERNEL     = int(os.getenv("PREPROCESS_MORPH_K",   "2"))


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _upscale_to_dpi(img: np.ndarray, src_dpi: int, target_dpi: int) -> np.ndarray:
    """Rescale image so that physical resolution reaches *target_dpi*."""
    if src_dpi >= target_dpi or src_dpi == 0:
        return img
    scale = target_dpi / src_dpi
    new_w = int(img.shape[1] * scale)
    new_h = int(img.shape[0] * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR/BGRA image to grayscale. Pass-through if already gray."""
    if len(img.shape) == 2:
        return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _denoise(gray: np.ndarray, h: int, template_window: int, search_window: int) -> np.ndarray:
    """Apply Non-Local Means denoising (best for handwriting)."""
    return cv2.fastNlMeansDenoising(
        gray,
        h=h,
        templateWindowSize=template_window,
        searchWindowSize=search_window,
    )


def _deskew(gray: np.ndarray) -> np.ndarray:
    """
    Detect the dominant horizontal text angle and rotate the image to correct minor scan tilt.
    Uses Hough line transform with strict safety constraints (-10° to +10°).
    Prevents false rotations caused by vertical table lines or page boundaries.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=int(gray.shape[1] * 0.35))
    if lines is None:
        return gray

    angles = []
    for line in lines:
        rho, theta = line[0]
        # Convert theta to an angle in degrees relative to horizontal text lines
        angle = math.degrees(theta) - 90
        
        # STRICT DESKEW RULE: Only consider minor text tilt between -10.0° and +10.0°
        # Ignore vertical table lines, grid columns, and page border margins
        if -10.0 < angle < 10.0:
            angles.append(angle)

    if not angles:
        return gray

    median_angle = float(np.median(angles))
    
    # Do not rotate if angle is negligible (less than 0.5°)
    if abs(median_angle) < 0.5:
        return gray

    print(f"  [Deskew] Correcting minor text tilt: {median_angle:.2f}°")
    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated



def _binarise(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive binarisation using Gaussian thresholding.
    Produces a clean black-on-white binary image well-suited for OCR.
    """
    # Block size must be odd and > 1
    block_size = max(11, (min(gray.shape) // 30) | 1)   # ensure odd
    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=10,
    )
    return binary


def _sharpen(gray: np.ndarray, amount: float) -> np.ndarray:
    """
    Unsharp mask sharpening:
        sharpened = original + amount * (original − blurred)
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)
    return sharpened


def _morphology_cleanup(binary: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Remove isolated speckles (noise pixels) with morphological opening,
    then close tiny gaps in strokes with closing.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    opened  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    closed  = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_image(
    img_np: np.ndarray,
    src_dpi: int = 300,
    target_dpi: int = 300,
    denoise_h: int = 3,
    sharpen_amount: float = 1.2,
    save_debug: bool = False,
    debug_prefix: str = "debug_page",
) -> np.ndarray:
    """
    High-quality text-preserving preprocessing pipeline for scanned court papers and legal documents.

    Uses CLAHE contrast enhancement, light deskewing, and unsharp masking to preserve
    faint handwritten ink, decimal points, and fine Hindi/English text without destructive erosion.
    """
    # 1. Ensure 300 DPI resolution scale
    if src_dpi < target_dpi and src_dpi > 0:
        scale = target_dpi / src_dpi
        img_np = cv2.resize(img_np, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale conversion
    gray = _to_gray(img_np)

    # 3. Deskew (rotate if scanned at an angle)
    deskewed = _deskew(gray)

    # 4. CLAHE (Contrast Limited Adaptive Histogram Equalization) for faint text & ink
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(deskewed)

    # 5. Very mild non-local means denoise (preserve small dots and decimal points)
    if denoise_h > 0:
        denoised = cv2.fastNlMeansDenoising(enhanced, h=denoise_h, templateWindowSize=5, searchWindowSize=15)
    else:
        denoised = enhanced

    # 6. Razor-sharp unsharp masking
    sharpened = _sharpen(denoised, sharpen_amount)

    if save_debug:
        cv2.imwrite(f"{debug_prefix}_enhanced.png", sharpened)

    return sharpened


def _process_single_pdf_page(
    page_data: tuple[int, int, bytes, int, int, int, int, float, int, bool, dict]
) -> tuple[int, Image.Image]:
    """Helper function to process a single PDF page in parallel."""
    (page_num, total_pages, pix_samples, pix_height, pix_width, pix_n,
     render_dpi, target_dpi, save_debug, kwargs) = page_data

    img_np = np.frombuffer(pix_samples, dtype=np.uint8).reshape(
        pix_height, pix_width, pix_n
    )

    debug_prefix = f"debug_p{page_num:03d}" if save_debug else "debug"
    cleaned = preprocess_image(
        img_np,
        src_dpi=render_dpi,
        target_dpi=target_dpi,
        save_debug=save_debug,
        debug_prefix=debug_prefix,
        **kwargs,
    )
    print(f"  [Preprocessor] Page {page_num}/{total_pages} done ({cleaned.shape[1]}x{cleaned.shape[0]}px).")
    pil_img = Image.fromarray(cleaned).convert("RGB")
    return (page_num, pil_img)


def preprocess_pdf(
    pdf_path: str,
    render_dpi: int = 300,
    target_dpi: int = 300,
    save_debug: bool = False,
    **kwargs,
) -> List[Image.Image]:
    """
    Load a PDF with PyMuPDF, render and preprocess each page in parallel
    across CPU threads.

    Args:
        pdf_path:    Absolute or relative path to the PDF file.
        render_dpi:  DPI at which PyMuPDF renders each page.
        target_dpi:  Target DPI to upscale to in preprocessing.
        save_debug:  If True, saves debug images for each page.
        **kwargs:    Additional keyword arguments forwarded to preprocess_image().

    Returns:
        List of PIL Images (one per page), cleaned and ready for OCR.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"[Preprocessor] Loading PDF: '{pdf_path}'")
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    zoom = render_dpi / 72.0
    mat  = fitz.Matrix(zoom, zoom)

    # Render pixmaps synchronously (fast PyMuPDF C calls)
    page_inputs = []
    for page_num, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
        page_inputs.append((
            page_num, total_pages, pix.samples, pix.height, pix.width, pix.n,
            render_dpi, target_dpi, save_debug, kwargs
        ))

    # Parallelize heavy OpenCV preprocessing across CPU worker threads
    from concurrent.futures import ThreadPoolExecutor
    print(f"[Preprocessor] Preprocessing {total_pages} page(s) in parallel ...")

    results = [None] * total_pages
    with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, total_pages)) as executor:
        futures = executor.map(_process_single_pdf_page, page_inputs)
        for page_num, pil_img in futures:
            results[page_num - 1] = pil_img

    doc.close()
    print(f"[Preprocessor] Finished — {len(results)} page(s) preprocessed.")
    return results


def get_pdf_page_count(pdf_path: str) -> int:
    """Returns the total number of pages in a PDF file instantly."""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


def preprocess_single_pdf_page_by_index(
    pdf_path: str,
    page_idx: int,
    render_dpi: int = 300,
    target_dpi: int = 300,
    save_debug: bool = False,
    **kwargs,
) -> Image.Image:
    """
    Renders and cleans a single PDF page by 0-based page index.
    Enables immediate real-time streaming to the AI model per page.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_idx]
        zoom = render_dpi / 72.0
        mat  = fitz.Matrix(zoom, zoom)
        pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        debug_prefix = f"debug_p{page_idx+1:03d}" if save_debug else "debug"
        cleaned = preprocess_image(
            img_np,
            src_dpi=render_dpi,
            target_dpi=target_dpi,
            save_debug=save_debug,
            debug_prefix=debug_prefix,
            **kwargs,
        )
        return Image.fromarray(cleaned).convert("RGB")
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m stage_1.preprocessor <path/to/scanned.pdf>")
        sys.exit(1)

    pdf = sys.argv[1]
    pages = preprocess_pdf(pdf, render_dpi=150, target_dpi=300, save_debug=True)
    print(f"\nReturned {len(pages)} cleaned PIL image(s). Ready for OCR node.")
