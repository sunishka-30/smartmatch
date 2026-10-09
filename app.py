"""
SmartMatch - Streamlit Web Application
AI-Powered B2B Buyer Enquiry Intent Parsing, Quality Gate, Vector Retrieval & Re-ranking for IndiaMART APM Portfolio.
"""

import sys
import os
import time
import pandas as pd
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.matcher import match_enquiry
from src.indexer import CHROMA_PATH, COLLECTION_NAME

# Streamlit Page Configuration
st.set_page_config(
    page_title="SmartMatch - AI Buyer Intent & Supplier Matching",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling cards, badges, and layout
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .supplier-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .verified-pill {
        background-color: #DCFCE7;
        color: #166534;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
        display: inline-block;
    }
    .unverified-pill {
        background-color: #F1F5F9;
        color: #64748B;
        font-size: 0.8rem;
        padding: 2px 8px;
        border-radius: 12px;
        display: inline-block;
    }
    .badge-matched {
        background-color: #DEF7EC;
        color: #03543F;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 1rem;
        display: inline-block;
    }
    .badge-clarification {
        background-color: #FEF08A;
        color: #713F12;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 1rem;
        display: inline-block;
    }
    .badge-spam {
        background-color: #FDE8E8;
        color: #9B1C1C;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 1rem;
        display: inline-block;
    }
    .badge-nomatch {
        background-color: #F3F4F6;
        color: #374151;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_catalogue_summary():
    """Cached helper to load supplier catalog count and details."""
    suppliers_path = os.path.join(os.path.dirname(__file__), "data", "suppliers.csv")
    if os.path.exists(suppliers_path):
        df = pd.read_csv(suppliers_path)
        return len(df), df['category'].nunique()
    return 200, 6


# Pre-set Examples
EXAMPLES = {
    "Select an example enquiry...": "",
    "1. Clear Catalogue Match": "need 500 kg SS 304 sheets delhi urgent delivery required",
    "2. Hinglish Inquiry": "bhai 1000 pcs 5 ply corrugated boxes Surat export pack fast",
    "3. Vague Inquiry": "need steel",
    "4. Spam Query": "Earn Rs 50000 monthly working online from home click http://bit.ly/fakejob",
    "5. Out-of-Catalogue Query": "require 50 crates organic Alphonso mangoes Pune",
    "6. Urgent Electrical Query": "urgent requirement 500 meters 4 core 25 sq mm copper cable Noida today"
}

# Sidebar Content
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=64)
    st.title("SmartMatch Control")
    st.caption("Assistant Product Manager AI Portfolio | IndiaMART")

    num_suppliers, num_categories = get_catalogue_summary()

    st.markdown("---")
    st.markdown("### 📊 Catalogue Overview")
    col_a, col_b = st.columns(2)
    col_a.metric("Suppliers", f"{num_suppliers}")
    col_b.metric("Categories", f"{num_categories}")

    st.markdown("---")
    st.markdown("### 🛠️ System Stack")
    st.markdown("""
    - **LLM Engine**: `gemini-2.5-flash`
    - **Embeddings**: `text-embedding-004`
    - **Vector DB**: ChromaDB (Cosine)
    - **Quality Gate**: 3-Tier Classifier
    - **Guardrail**: Strict No-Match Threshold
    """)

    st.markdown("---")
    st.markdown("### 💡 Preset Examples")
    selected_example_label = st.selectbox("Load Example Enquiry", list(EXAMPLES.keys()))
    if selected_example_label != "Select an example enquiry...":
        st.session_state["query_input"] = EXAMPLES[selected_example_label]

# Main Area
st.markdown('<div class="main-header">⚡ SmartMatch: AI Buyer Intent & Supplier Matching</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Instant intent parsing, quality classification, and vector supplier retrieval for IndiaMART B2B marketplace.</div>', unsafe_allow_html=True)

# 5-Step Pipeline Explander
with st.expander("ℹ️ How SmartMatch Works (5-Step AI Pipeline)"):
    st.markdown("""
    1. **Structured Intent Extraction**: Parses messy, shorthand, Hinglish text into clean JSON fields (`product`, `quantity`, `unit`, `location`, `urgency`).
    2. **Enquiry Quality Classification**: Gates incoming leads into `clear`, `vague`, or `spam` to protect supplier lead quality.
    3. **Spam & Vague Guardrails**: Rejects spam immediately and prompts buyers with targeted clarifying questions for vague requests.
    4. **Vector Retrieval**: Fetches top candidate suppliers from ChromaDB using Gemini `text-embedding-004` semantic vectors.
    5. **Re-Ranking & No-Match Guardrail**: Evaluates candidate fit, generates 1-line match justifications, and outputs `NO MATCH` if no genuine seller exists.
    """)

# User Input Form
default_text = st.session_state.get("query_input", "need 500 kg SS 304 sheets delhi urgent delivery required")

query_text = st.text_area(
    "Enter Unstructured Buyer Requirement Query:",
    value=default_text,
    height=90,
    placeholder="e.g. need 500 kg SS 304 sheets delhi urgent delivery required..."
)

col_btn, _ = st.columns([1, 4])
submit_button = col_btn.button("🔍 Find Suppliers", type="primary", use_container_width=True)

if submit_button or "last_query" in st.session_state:
    if submit_button:
        st.session_state["last_query"] = query_text

    current_query = st.session_state.get("last_query", query_text)

    if not current_query.strip():
        st.warning("Please enter a valid buyer requirement query.")
    else:
        with st.spinner("Processing intent parsing, vector retrieval, and supplier re-ranking..."):
            try:
                response = match_enquiry(current_query)
            except Exception as err:
                st.error(f"An unexpected error occurred while processing your request: {err}")
                st.stop()

        status = response.get("status")
        extracted = response.get("extracted", {})
        results = response.get("results", [])
        questions = response.get("clarifying_questions", [])
        latency = response.get("latency_ms", {})
        tokens = response.get("tokens_used", {})

        st.markdown("---")

        # 1. Status Banner
        st.markdown("### Pipeline Execution Result")
        status_col, reasoning_col = st.columns([1, 3])

        with status_col:
            if status == "matched":
                st.markdown('<div class="badge-matched">✓ MATCHED</div>', unsafe_allow_html=True)
            elif status == "needs_clarification":
                st.markdown('<div class="badge-clarification">⚠️ NEEDS CLARIFICATION</div>', unsafe_allow_html=True)
            elif status == "rejected_spam":
                st.markdown('<div class="badge-spam">🚫 REJECTED AS SPAM</div>', unsafe_allow_html=True)
            else: # no_match
                st.markdown('<div class="badge-nomatch">❌ NO MATCH FOUND</div>', unsafe_allow_html=True)

        with reasoning_col:
            st.info(f"**Quality Reason**: {extracted.get('reasoning', 'Processed successfully.')}")

        # 2. Extracted Requirements Card
        st.markdown("#### 📋 Extracted Intent & Structured Schema")
        ext_col1, ext_col2, ext_col3, ext_col4, ext_col5 = st.columns(5)

        ext_col1.metric("Product", extracted.get("product") or "None")
        ext_col2.metric("Quantity", f"{extracted.get('quantity') or 'N/A'} {extracted.get('unit') or ''}".strip())
        ext_col3.metric("Location", extracted.get("location") or "Unspecified")
        ext_col4.metric("Urgency", (extracted.get("urgency") or "unspecified").upper())
        ext_col5.metric("Quality Flag", (extracted.get("quality_flag") or "vague").upper())

        # 3. Render Results based on Status
        if status == "matched":
            st.markdown(f"#### 🏆 Top Matched Suppliers ({len(results)} Candidates)")

            for res in results:
                verified_html = '<span class="verified-pill">✓ Verified Seller</span>' if res.get('state') else '<span class="unverified-pill">Standard Supplier</span>'

                st.markdown(f"""
                <div class="supplier-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #1E3A8A;">#{res['rank']} {res['company_name']}</h4>
                        <span style="font-weight: 700; color: #0284C7; font-size: 1.1rem;">Relevance: {int(res['relevance_score']*100)}%</span>
                    </div>
                    <p style="margin: 4px 0 10px 0; color: #475569; font-size: 0.95rem;">
                        📍 <b>Location:</b> {res['city']}, {res['state']} | 🆔 <b>ID:</b> {res['supplier_id']}
                    </p>
                    <p style="margin: 0; background-color: #EFF6FF; padding: 8px 12px; border-radius: 6px; border-left: 4px solid #3B82F6; color: #1E40AF; font-size: 0.95rem;">
                        💡 <b>Match Reason:</b> {res['match_reason']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        elif status == "needs_clarification":
            st.markdown("#### ❓ Required Clarifications")
            st.warning("This enquiry is vague. Please clarify the following details to get matched suppliers:")
            for q in questions:
                st.markdown(f"- **{q}**")

        elif status == "rejected_spam":
            st.error("This enquiry was identified as promotional spam, job posting, or invalid text. No suppliers were notified.")

        elif status == "no_match":
            st.warning("🔒 **No-Match Guardrail Triggered**: No suppliers in our catalogue currently match this requirement. Our team will notify new suppliers upon onboarding.")

        # 4. Debug & Latency Breakdown Expander
        with st.expander("🛠️ Debug & Performance Telemetry"):
            d_col1, d_col2 = st.columns(2)

            with d_col1:
                st.markdown("##### ⏱️ Latency Breakdown (ms)")
                lat_df = pd.DataFrame([
                    {"Stage": "Intent Extraction", "Latency (ms)": latency.get("extraction_ms", 0.0)},
                    {"Stage": "Vector Retrieval", "Latency (ms)": latency.get("retrieval_ms", 0.0)},
                    {"Stage": "LLM Re-Ranking", "Latency (ms)": latency.get("rerank_ms", 0.0)},
                    {"Stage": "Total End-to-End", "Latency (ms)": latency.get("total_ms", 0.0)}
                ])
                st.dataframe(lat_df, use_container_width=True, hide_index=True)

            with d_col2:
                st.markdown("##### 🔢 Token Usage & Costs")
                tok_df = pd.DataFrame([
                    {"Metric": "Input Prompt Tokens", "Count": tokens.get("input_tokens", 0)},
                    {"Metric": "Output Completion Tokens", "Count": tokens.get("output_tokens", 0)},
                    {"Metric": "Total Tokens", "Count": tokens.get("total_tokens", 0)},
                    {"Metric": "Est. Cost (USD)", "Count": f"${(tokens.get('total_tokens', 0) * 0.0000003):.6f}"}
                ])
                st.dataframe(tok_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.caption("SmartMatch APM Portfolio Project | Built with Streamlit, Gemini API & ChromaDB for IndiaMART")
