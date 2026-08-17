# VDAICourt

High-performance, multi-stage AI pipeline for Indian High Court and Commercial Court legal document processing, OCR extraction, entity & evidence classification, and legal integrity verification.

## 🚀 Architecture Overview

- **Stage 1: Document Preprocessing & Multi-Pass Vision OCR**
  - High-resolution OpenCV image enhancement (adaptive thresholding, unsharp masking, bilateral denoising).
  - Dynamic page splitting for faint handwriting, high-contrast stamps, and complex legal petitions.
  - Multi-pass OCR with Auditor/Verifier Agent pass.
  - Round-robin Multi-Key Async Rate Limiter pool (5 API keys).
- **Stage 2: Page Classification & Evidence Extraction**
  - Categorizes pages: Petition Text, Pehchan Patra / Identity Proofs, Vakalatnama, Affidavits, Memo of Parties, Court Orders.
  - Extracts key legal entities, parties, advocate enrolments, claim amounts, and identity proofs.
- **Stage 3: Legal Integrity & Cross-Document Verification**
  - Cross-verifies party identities against Pehchan Patra (Voter ID, Aadhaar, PAN card).
  - Vakalatnama legal authorization checks & signatory roster.
  - Generates comprehensive markdown audit reports and discrepancy flags.

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/A4anas3/VDAICourt.git
   cd VDAICourt
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment:**
   ```bash
   cp .env.example .env
   # Edit .env and insert your API keys
   ```

4. **Run the Full Pipeline:**
   ```bash
   python main.py <path_to_court_pdf>
   ```

## 📜 License
MIT License
