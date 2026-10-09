# Product Specification: SmartMatch (IndiaMART APM Portfolio Project)

## 1. Problem Statement
IndiaMART is India's largest B2B marketplace, connecting millions of buyers with suppliers across diverse industrial and commercial categories. Every day, buyers submit unstructured, highly informal, shorthand, or incomplete requirement queries (e.g., *"need 500 kg SS 304 sheets delhi urgent"*).

### Key Challenges:
- **Unstructured Requirement Parsing**: Raw search engines struggle with implicit product specifications, ambiguous acronyms (e.g., "SS 304" = Stainless Steel Grade 304), mixed units, and location cues embedded in plain text.
- **Supplier Lead Quality**: Distributing vague or spam enquiries to paid suppliers degrades supplier trust, increases lead decay, and negatively impacts platform monetization.
- **Precision Matching & Hallucination Risks**: Traditional keyword match fails on context, while naive generative AI systems risk inventing non-existent suppliers or producing false matches when no relevant vendor exists in the marketplace catalogue.

**SmartMatch** addresses these challenges by delivering an end-to-end AI-powered pipeline that extracts structured entities, evaluates enquiry quality, retrieves candidate suppliers using vector embeddings, re-ranks candidate suppliers with actionable match justifications, and strictly enforces a "no-match" guardrail when relevant suppliers are unavailable.

---

## 2. Target Users

### Primary Users: B2B Buyers
- **Persona**: Purchasing Officers, Procurement Managers, SME Business Owners, Contractors.
- **Needs**: Fast, friction-free input without long structured forms; immediate confidence that their specific requirement will reach verified, capable suppliers.

### Secondary Users: B2B Suppliers / Sellers
- **Persona**: Manufacturers, Authorized Distributors, Wholesalers, Traders.
- **Needs**: High-intent, well-structured, non-spam leads tailored precisely to their product catalog, location serviceability, and order volume capabilities.

---

## 3. System Inputs & Outputs

### Inputs
- **Unstructured Buyer Query** (`text`, required): Free-form string containing buyer requirement (e.g., *"Looking for 200 meters 12mm TMT bars in Gurgaon for immediate delivery"*).
- **Buyer Location Context** (`string`, optional): Default or detected city/state of the buyer.

### Outputs
1. **Structured Entity Representation** (`JSON`):
   - `product_name`: Standardized product category/item.
   - `material_grade`: Specific grade/type if applicable (e.g., "SS 304", "Fe 500D").
   - `quantity`: Numerical value extracted.
   - `unit`: Standardized unit of measure (e.g., "kg", "meters", "tons", "pieces").
   - `location`: Delivery/target city or state.
   - `urgency`: Inferred urgency (`high`, `medium`, `low`).
   - `specifications`: Key technical parameters (e.g., thickness, diameter, color).

2. **Enquiry Quality Classification**:
   - `quality_label`: `clear` | `vague` | `spam`
   - `quality_reason`: Single-sentence rationale for the assigned quality grade.
   - `missing_fields`: Array of critical missing parameters (e.g., `["quantity", "location"]`).

3. **Supplier Match Results**:
   - `match_status`: `matched` | `no_match`
   - `suppliers` (List of Top 3-5 candidates if `match_status` == `matched`):
     - `supplier_id`: Unique identifier.
     - `company_name`: Vendor business name.
     - `location`: Operational city/state.
     - `matched_categories`: Relevant categories served.
     - `relevance_score`: Composite similarity score (0.0 to 1.0).
     - `match_reason`: 1-line plain-language justification explaining why this supplier was selected.

---

## 4. Success Metrics

| Metric | Target / Benchmark | Definition & Rationale |
| :--- | :--- | :--- |
| **Extraction Accuracy (F1)** | **≥ 90% F1 Score** | Evaluates precision and recall across extracted fields (`product_name`, `material_grade`, `quantity`, `unit`, `location`) against human-labeled evaluation benchmark dataset. |
| **Top-3 Match Hit Rate** | **≥ 85%** | Percentage of valid queries where at least 1 truly relevant and serviceable supplier is ranked in the top 3 results. |
| **Spam Precision** | **≥ 98%** | Ratio of true spam items among all queries flagged as spam. Prevents dropping genuine buyer leads. |
| **Spam Recall** | **≥ 95%** | Ratio of spam queries correctly identified and filtered out. |
| **Average Latency (p95)** | **< 1.5 seconds** | End-to-end processing time (Entity extraction + Vector retrieval + Re-ranking). Critical for interactive buyer UI. |
| **Average Cost per Query** | **< $0.002** | Total LLM and embedding API cost per enquiry execution, calculated using Gemini Flash token pricing. |

---

## 5. Non-Goals

- **Real-Time Payment / Escrow Integration**: Transaction completion and online checkout are outside the scope of lead parsing and matching.
- **Multi-Turn Conversational Chatbot**: SmartMatch operates as a single-shot intent parsing and supplier recommendation engine, not a multi-turn conversational bot.
- **Live Inventory / ERP Syncing**: System assumes static or periodically refreshed supplier catalog metadata rather than live real-time stock APIs.
- **Multilingual Speech / OCR**: Processing images, PDFs, or raw audio is excluded for v1 (text queries in English / Hinglish are supported).

---

## 6. Key Risks & Mitigation Strategies

| Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Hallucinated Suppliers** | High (Destroys platform credibility if non-existent vendors are displayed) | **Deterministic Guardrail**: Set a strict ChromaDB cosine distance threshold + explicit LLM system instructions to output `no_match` whenever top candidates fall below minimum similarity. |
| **High Re-Ranking Latency** | Medium (Degrades UI responsiveness) | **Two-Stage Retrieval Pipeline**: Stage 1 uses fast local ChromaDB vector search to fetch top 10 candidates; Stage 2 uses Gemini Flash to re-rank only those 10 candidates. |
| **Shorthand & Ambiguous B2B Terms** | Medium (Misinterpretation of technical specs like "SS 304" or "MS Angle") | **Domain-Aware Prompting**: Supply domain-specific JSON schema definitions and few-shot IndiaMART industrial query examples in prompt context. |
| **API Cost Volatility** | Low | Use `gemini-2.5-flash` for extraction/re-ranking and standard embedding models, strictly limiting prompt size and output token counts. |
