# ⚡ SmartMatch: AI-Powered B2B Enquiry Parsing & Supplier Matching

> **One-Line Pitch**: An end-to-end AI intent parsing, quality gating, and vector supplier matching engine designed for IndiaMART's B2B marketplace.

---

## 📷 Demo Interface

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ ⚡ SmartMatch: AI Buyer Intent & Supplier Matching                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ Enter Unstructured Buyer Requirement Query:                                             │
│ [ need 500 kg SS 304 sheets delhi urgent delivery required                            ] │
│                                                                                         │
│ [ 🔍 Find Suppliers ]                                                                   │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  ✓ MATCHED    Quality Reason: Extracted product details, quantity, and delivery params. │
│                                                                                         │
│  📋 Extracted Intent:                                                                   │
│  • Product: SS 304 Sheets   • Qty: 500 kg   • Location: Delhi   • Urgency: HIGH           │
│                                                                                         │
│  🏆 Top Matched Suppliers:                                                              │
│  #1 Mahavir Iron & Steel Ind (Delhi, Delhi) — Relevance: 66%                            │
│     💡 Reason: Supplies matching products in Delhi with verified catalog specs.          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

- **Structured Entity Extraction**: Parses messy, shorthand, and Hinglish buyer queries into standardized JSON (`product`, `quantity`, `unit`, `location`, `urgency`).
- **3-Tier Quality Classifier Gate**: Classifies incoming enquiries into `clear`, `vague`, or `spam`.
- **Targeted Clarification Generator**: Dynamically prompts buyers with 1-2 missing specification questions for vague requests.
- **Semantic Vector Search**: Retrieves relevant suppliers from ChromaDB using `text-embedding-004` semantic vectors.
- **LLM Re-Ranking with No-Match Guardrail**: Re-ranks candidates, generates 1-line match justifications citing specific attributes, and strictly outputs `NO MATCH` for unserviceable products.
- **Zero Supplier Hallucinations**: Code-validated guardrails prevent inventing non-existent suppliers.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A["Buyer Raw Text Input"] --> B["src/extractor.py<br/>(Gemini 2.5 Flash / Fallback)"]
    B --> C{"Quality Gate"}
    
    C -- "spam" --> D["Status: REJECTED_SPAM<br/>(Immediate Exit)"]
    C -- "vague" --> E["Status: NEEDS_CLARIFICATION<br/>(1-2 Questions Generated)"]
    C -- "clear" --> F["src/indexer.py<br/>Vector Search (ChromaDB)"]
    
    F --> G{"Distance Threshold <= 0.55?"}
    G -- "No Candidates Pass" --> H["Status: NO_MATCH<br/>(Guardrail Triggered)"]
    G -- "Candidates Available" --> I["src/matcher.py<br/>LLM Candidate Re-Ranker"]
    
    I --> J{"Valid Match Found?"}
    J -- "No" --> H
    J -- "Yes (Top 5)" --> K["Status: MATCHED<br/>(Ranked Suppliers + Reasons)"]
```

---

## 🚀 Quickstart & Setup Instructions

### Prerequisites
- Python 3.10+
- Git

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/your-username/smartmatch.git
cd smartmatch

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the example environment file to `.env`:
```bash
cp .env.example .env
```
*(Optional: Add your `GEMINI_API_KEY` to `.env`. If unconfigured, SmartMatch runs seamlessly using its local fallback engines).*

### 4. Generate Synthetic Dataset
Generate 200 realistic Indian B2B suppliers and 60 ground-truth evaluation enquiries:
```bash
python src/generate_data.py
```

### 5. Build Vector Search Index
Embed suppliers and build the persistent ChromaDB collection:
```bash
python src/indexer.py
```

### 6. Run Interactive Streamlit Web App
Launch the web interface:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 7. Run Offline Benchmark Evaluation
Execute the evaluation suite across 60 ground-truth queries:
```bash
python eval/evaluate.py
```

---

## 📊 Offline Benchmark Evaluation Results

Evaluated over 60 ground-truth test enquiries (see [`eval/report.md`](file:///c:/Users/ASUS%20PC/Desktop/smartmatch/eval/report.md) and [`docs/failure_analysis.md`](file:///c:/Users/ASUS%20PC/Desktop/smartmatch/docs/failure_analysis.md)):

| Benchmark Metric | Verified Score | Target / Benchmark | Status |
| :--- | :---: | :---: | :---: |
| **Spam Precision** | **100.0%** | $\ge 98\%$ | ✅ PERFECT |
| **Spam Recall** | **100.0%** | $\ge 95\%$ | ✅ RESOLVED |
| **Location Extraction Accuracy** | **96.67%** | $\ge 90\%$ | ✅ EXCEEDED |
| **Product Extraction Accuracy** | **80.00%** | $\ge 90\%$ | 📈 IMPROVED |
| **Quantity Extraction Accuracy** | **76.67%** | $\ge 90\%$ | 📈 IMPROVED |
| **Out-of-Cat No-Match Precision**| **90.00%** | $\ge 90\%$ | ✅ PASS |
| **Supplier Hallucinations** | **0** | **0** | ✅ ZERO HALLUCINATIONS |
| **Average End-to-End Latency** | **166.76 ms** | $< 1500\text{ ms}$ | ⚡ ULTRA-FAST |
| **p95 End-to-End Latency** | **231.63 ms** | $< 2500\text{ ms}$ | ⚡ PREDICTABLE |

---

## 💡 Key Learnings, Limitations & Future Work

### Key Learnings
- **Strict Quality Gating Is Critical**: Filtering out spam and prompting for clarifications on vague queries drastically reduces lead decay and supplier churn.
- **Code-Enforced No-Match Guardrails**: Combining vector distance thresholds with explicit LLM prompt guardrails and candidate ID validation guarantees zero supplier hallucinations.

### System Limitations
- **Dense Vector Specificity**: Dense embeddings (`all-MiniLM-L6-v2` / `text-embedding-004`) cluster suppliers within the same category closely. Distinguishing between fine-grained technical grades (e.g. `SS 304` vs `Fe 500D`) requires keyword matching.
- **Single-Quantity Extractor**: Complex queries with dual nested quantities (e.g., *"50 boxes of 100 pcs"*) extract the primary box count but omit inner unit breakdown.

### Future Enhancements
- **Hybrid Dense-Sparse Retrieval**: Combine BM25 keyword indexing with ChromaDB vector search.
- **Voice Enquiry Support**: Integrate speech-to-text models for regional Indian audio lead submission.

---

## 📄 License
This project is licensed under the [MIT License](file:///c:/Users/ASUS%20PC/Desktop/smartmatch/LICENSE).
