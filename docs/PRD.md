# Product Requirement Document (PRD): SmartMatch Production Rollout

**Project Title**: SmartMatch - AI-Powered Buyer Enquiry Parsing & Supplier Match Engine  
**Target Platform**: IndiaMART InterMESH Ltd.  
**Author**: Assistant Product Manager (AI) candidate  
**Status**: Proposal for Production V1 Rollout  

---

## 1. Problem Statement
IndiaMART receives millions of B2B buyer enquiries daily. A significant portion of incoming queries are free-text, unstructured, shorthand, or Hinglish inputs (e.g., *"need 500 kg SS 304 sheets delhi urgent delivery required"*).

### Current Ecosystem Pain Points:
1. **Buyer Friction**: Traditional multi-step requirement forms cause drop-offs. Buyers prefer typing raw text requirements.
2. **Supplier Fatigue & Lead Decay**: Vague or irrelevant leads distributed to paid suppliers degrade trust, reduce lead acceptance rates, and hurt platform subscription renewals.
3. **Keyword Search Failures**: Traditional exact-keyword search engines fail on technical abbreviations ("SS 304" -> Stainless Steel Grade 304), missing units, or regional spelling variants.
4. **Hallucination Risk in AI Systems**: Unbounded generative AI models risk inventing non-existent suppliers or producing false matches when no relevant vendor exists in IndiaMART's catalog.

---

## 2. Target User Personas & Needs

| User Persona | Primary Need | Key Pain Point |
| :--- | :--- | :--- |
| **SME / Industrial Buyer** | Rapid procurement of specific raw materials or equipment without filling out long static forms. | Having to type detailed specifications repeatedly; waiting hours for non-relevant vendor quotes. |
| **Verified IndiaMART Supplier** | High-intent, qualified, actionable buyer leads matching their product category and regional serviceability. | Wasting paid lead credits on vague, out-of-location, or spam inquiries. |
| **IndiaMART Platform / Category Ops** | Higher lead-to-conversion rates, reduced spam distribution, and transparent match justifications. | Manual moderation latency and poor lead routing precision. |

---

## 3. Solution Overview & Core Architecture

SmartMatch is an end-to-end AI pipeline built using Gemini API (`gemini-2.5-flash`), `text-embedding-004` semantic vectors, and a local persistent ChromaDB store.

```
Buyer Raw Query ──► Intent Extractor ──► Quality Classifier Gate
                                                 │
                             ┌───────────────────┼───────────────────┐
                             ▼                   ▼                   ▼
                      [REJECTED_SPAM]  [NEEDS_CLARIFICATION]   [CLEAR QUERY]
                             │                   │                   │
                     Immediate Exit      1-2 Questions       Vector Retrieval
                                                                (ChromaDB)
                                                                     │
                                                                     ▼
                                                             LLM Re-Ranker &
                                                            No-Match Guardrail
                                                                     │
                                                                     ▼
                                                             [MATCHED] / [NO_MATCH]
```

### Core Features:
1. **Structured Entity Extraction**: Parses product, quantity, unit, location, and urgency from informal text.
2. **3-Tier Quality Classifier**: Categorizes enquiries into `clear`, `vague`, or `spam`.
3. **Targeted Clarification Generator**: Dynamically prompts buyers with 1-2 missing spec questions for vague queries.
4. **Semantic Vector Search**: Retrieves top candidate suppliers from ChromaDB using 768-dimensional embeddings.
5. **LLM Re-Ranker with No-Match Guardrail**: Re-ranks candidates, generates 1-line match justifications citing specific attributes, and outputs `NO MATCH` if no genuine supplier exists.

---

## 4. Success Metrics & Benchmark Results

The following metrics reflect actual offline benchmark evaluations conducted over 60 ground-truth buyer queries (see [`eval/report.md`](file:///c:/Users/ASUS%20PC/Desktop/smartmatch/eval/report.md)):

| Metric | Target | Verified Score | Business Impact |
| :--- | :---: | :---: | :--- |
| **Spam Precision** | $\ge 98\%$ | **100.0%** | Zero false positives; genuine buyer leads are never misclassified as spam. |
| **Spam Recall** | $\ge 95\%$ | **100.0%** | 100% of promotional scams and non-B2B chatter automatically filtered. |
| **Location Extraction Accuracy** | $\ge 90\%$ | **96.67%** | High precision in capturing delivery city (Delhi, Surat, Pune, Ludhiana). |
| **Product Extraction Accuracy** | $\ge 90\%$ | **80.00%** | Standardizes shorthand B2B product terms. |
| **Quantity Extraction Accuracy** | $\ge 90\%$ | **76.67%** | Parses numerical amounts and standardized measurement units. |
| **Out-of-Cat No-Match Precision**| $\ge 90\%$ | **90.00%** | Prevents matching unserviceable items (e.g. Alphonso mangoes). |
| **Supplier Hallucinations** | **0** | **0** | **Passed**: Zero hallucinated or invented supplier IDs returned. |
| **Average End-to-End Latency** | $< 1.5\text{s}$ | **166.76 ms** | Ultra-low latency for interactive web UI. |
| **p95 End-to-End Latency** | $< 2.5\text{s}$ | **231.63 ms** | Predictable performance under load. |

---

## 5. Scope for V1 & Explicit Non-Goals

### Included in V1 Scope:
- Text-based buyer enquiry intent extraction (English & Hinglish).
- 3-tier quality classification & automated clarifying questions.
- Vector retrieval over 6 industrial supplier categories (Steel, Packaging, Textiles, Machinery, Electricals, Chemicals).
- LLM candidate re-ranking with 1-line match justifications and strict `no_match` guardrail.
- Streamlit interactive demo app & automated evaluation pipeline.

### Explicitly Excluded from V1 (What We Chose NOT to Build & Why):

1. **Multi-Turn Conversational Chatbot**:
   - *Rationale*: B2B procurement buyers want instant lead submission, not long multi-turn chit-chat. Single-shot parsing with 1-step clarification minimizes friction.
2. **Real-Time Payment Escrow & Checkout**:
   - *Rationale*: IndiaMART operates primarily as a lead-generation marketplace. Transaction execution is managed out-of-band by buyers and sellers.
3. **Live ERP / Inventory Sync APIs**:
   - *Rationale*: Small Indian MSME suppliers rarely maintain real-time inventory APIs; metadata catalog updates via scheduled sync are sufficient.
4. **Multilingual Speech / Image OCR**:
   - *Rationale*: High complexity and latency for V1. Text/Hinglish parsing addresses >80% of digital enquiry volume.

---

## 6. Risk Analysis & Mitigation Strategies

| Risk Factor | Potential Impact | Severity | Mitigation Strategy |
| :--- | :--- | :---: | :--- |
| **LLM Supplier Hallucinated Matches** | High (Destroys platform credibility if non-existent vendors are recommended) | **Critical** | **Code-Level Validation**: Enforce strict distance thresholding in ChromaDB + hard code-level validation that re-ranker output contains only IDs present in candidate list. |
| **Bias Toward Verified Suppliers** | Medium (Unverified suppliers complain of zero lead allocation) | **Medium** | **Controlled Scoring Bonus**: Apply a max +0.15 relevance boost for verified status, ensuring unverified suppliers with superior product/location fit still rank #1. |
| **API Latency & Rate Limits at Scale** | High (Slow query response during peak daytime traffic) | **High** | **Two-Stage Architecture**: Fast local ChromaDB vector retrieval filters millions of records down to 8 candidates in <50ms before calling LLM re-ranker. |
| **LLM Token Cost at Scale** | Medium (High API costs at 10M daily queries) | **Medium** | Use lightweight `gemini-2.5-flash` model, JSON schema mode for tight outputs, and embedding caching. |
| **Ambiguous Hinglish Shorthand** | Medium (Misparsing complex technical terms) | **Medium** | Domain-specific prompt instruction + few-shot B2B examples in `src/prompts.py`. |

---

## 7. Phased Rollout Plan & A/B Experiment Design

```
Phase 1: Internal Alpha ──► Phase 2: 5% Category Beta ──► Phase 3: 50% A/B Test ──► Phase 4: 100% General Availability
  (Ops Team Validation)      (Steel & Packaging Categories)   (Controlled Experiment)     (Full Platform Rollout)
```

### A/B Test Design (Phase 3)
- **Control Group (50% Traffic)**: Existing keyword search + standard multi-field requirement form.
- **Treatment Group (50% Traffic)**: SmartMatch single-text box + AI intent parsing + vector re-ranking.

### Core Experiment Metrics:
1. **Primary Metric**: **Supplier Lead Acceptance Rate (%)** (Target: $+15\%$ increase in treatment group).
2. **Secondary Metric**: **Buyer Lead Completion Rate (%)** (Target: $+25\%$ increase due to reduced form friction).
3. **Guardrail Metric**: **Supplier Paid Account Unsubscribe Rate** (Must not increase by $>0.1\%$).

---

## 8. Open Questions & Future Roadmap

1. **Hybrid Dense-Sparse Vector Search (BM25 + ChromaDB)**:
   - *Question*: How to improve Top-1 exact rank matches for fine-grained technical specifications (e.g. `SS 304` vs `Fe 500D`)?
   - *Roadmap Item*: Integrate BM25 keyword index alongside ChromaDB vector embeddings.
2. **Voice-to-Text Enquiry Input**:
   - *Roadmap Item*: Support WhatsApp audio message transcription into SmartMatch text input for rural Indian buyers.
