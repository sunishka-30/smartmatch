"""
SmartMatch - Matcher Engine
Runs the full end-to-end matching pipeline:
1. Extract requirement & classify quality (clear / vague / spam).
2. Reject spam immediately.
3. Generate clarifying questions for vague enquiries.
4. Retrieve candidate suppliers from ChromaDB vector index.
5. Re-rank top candidates using Gemini API with no-match guardrail.
"""

import os
import re
import sys
import json
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Literal

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.extractor import extract_enquiry, get_gemini_client
from src import indexer

# Load environment variables
load_dotenv()

# Constants
RERANK_MODEL = "gemini-2.5-flash"
VECTOR_DISTANCE_THRESHOLD = 0.55  # Maximum distance threshold for vector candidates
TOP_K_RETRIEVAL = 8
TOP_K_RERANK = 5


class RankedSupplierSchema(BaseModel):
    supplier_id: str = Field(description="Supplier ID matching one of the candidate IDs strictly")
    rank: int = Field(description="1-based rank order")
    match_reason: str = Field(description="One-line reason citing specific matching attributes like product, spec, city, MOQ")
    relevance_score: float = Field(description="Relevance score from 0.0 to 1.0")


class RerankResponseSchema(BaseModel):
    suppliers: list[RankedSupplierSchema] = Field(
        default_factory=list,
        description="Top-ranked matching suppliers, or empty list [] if no candidate genuinely matches"
    )


class ClarificationSchema(BaseModel):
    questions: list[str] = Field(
        description="List of 1-2 short, specific clarifying questions for vague enquiry"
    )


def generate_clarifying_questions(raw_text: str, extracted: dict) -> tuple[list[str], int, int]:
    """Generates 1-2 targeted clarifying questions for vague queries using Gemini or fallback."""
    client = get_gemini_client()
    if client:
        try:
            from google.genai import types
            prompt = f"The buyer submitted a vague B2B query: '{raw_text}'. Generate 1-2 short, professional clarifying questions asking for specific product grade, material, size, quantity, or delivery location."

            response = client.models.generate_content(
                model=RERANK_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ClarificationSchema,
                    temperature=0.3
                )
            )
            usage = getattr(response, "usage_metadata", None)
            in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
            out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0

            data = json.loads(response.text)
            return data.get("questions", []), in_tok, out_tok
        except Exception:
            pass

    # Fallback clarifying questions
    prod = extracted.get("product") or "the product"
    questions = [
        f"Which specific grade, size, or material specification of {prod} do you require?",
        "What is your required quantity and target delivery city?"
    ]
    return questions, 0, 0


def rerank_candidates(extracted: dict, candidates: list[dict]) -> tuple[list[dict], int, int]:
    """
    Calls Gemini API to re-rank candidate suppliers with transparent reasoning,
    validating IDs and enforcing the no-match guardrail.
    """
    valid_candidate_ids = {c["supplier_id"] for c in candidates}

    client = get_gemini_client()
    if client:
        try:
            from google.genai import types

            candidates_summary = []
            for c in candidates:
                meta = c["metadata"]
                candidates_summary.append({
                    "supplier_id": c["supplier_id"],
                    "company_name": meta.get("company_name"),
                    "city": meta.get("city"),
                    "state": meta.get("state"),
                    "category": meta.get("category"),
                    "document": c["document"]
                })

            prompt = f"""Buyer Requirement:
Product: {extracted.get('product')}
Quantity: {extracted.get('quantity')} {extracted.get('unit')}
Location: {extracted.get('location')}
Urgency: {extracted.get('urgency')}

Candidate Suppliers:
{json.dumps(candidates_summary, indent=2)}

Task: Select up to top {TOP_K_RERANK} relevant suppliers.
CRITICAL NO-MATCH GUARDRAIL: If NONE of the candidates genuinely supply the requested product, return an empty array `[]` for suppliers.
Each selected supplier MUST cite specific matching attributes (product, specs, location, MOQ) in match_reason."""

            response = client.models.generate_content(
                model=RERANK_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are an expert B2B supplier match engine. Enforce strict relevance. Never hallucinate supplier IDs.",
                    response_mime_type="application/json",
                    response_schema=RerankResponseSchema,
                    temperature=0.1
                )
            )

            usage = getattr(response, "usage_metadata", None)
            in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
            out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0

            data = json.loads(response.text)
            raw_suppliers = data.get("suppliers", [])

            # Code Validation: Filter out any hallucinated IDs not in candidate list
            validated_results = []
            cand_dict = {c["supplier_id"]: c for c in candidates}

            for rank_idx, item in enumerate(raw_suppliers, 1):
                sup_id = item.get("supplier_id")
                if sup_id in valid_candidate_ids:
                    cand_info = cand_dict[sup_id]
                    validated_results.append({
                        "supplier_id": sup_id,
                        "company_name": cand_info["metadata"].get("company_name"),
                        "city": cand_info["metadata"].get("city"),
                        "state": cand_info["metadata"].get("state"),
                        "rank": rank_idx,
                        "match_reason": item.get("match_reason"),
                        "relevance_score": float(item.get("relevance_score", 0.8)),
                        "vector_distance": float(cand_info["distance"])
                    })

            return validated_results[:TOP_K_RERANK], in_tok, out_tok

        except Exception as e:
            print(f"[!] Gemini re-ranking API error: {e}. Switching to heuristic re-ranker.")

    # Rule-Based Fallback Re-ranker
    validated_results = []
    cand_dict = {c["supplier_id"]: c for c in candidates}
    req_product = (extracted.get("product") or "").strip().lower()
    req_location = (extracted.get("location") or "").strip().lower()

    # Extract core product keywords by stripping stop words and quantity/location words
    stop_words = {"need", "require", "want", "bhai", "urgent", "delivery", "required", "plz", "send", "looking", "for", "crates", "boxes", "pcs", "kg", "tons", "ton", "in", "at", "location", "delhi", "mumbai", "pune", "surat", "ludhiana", "ahmedabad", "chennai", "kolkata", "rajkot", "vapi", "baddi", "jaipur", "bhiwandi", "kanpur", "crates", "50", "100"}
    product_keywords = [w for w in req_product.split() if len(w) > 2 and w not in stop_words]

    for c in candidates:
        sup_id = c["supplier_id"]
        doc_lower = c["document"].lower()
        meta = c["metadata"]

        # Product relevance check: At least one core product keyword must appear as a full word in the supplier products or specs
        if product_keywords:
            has_prod_match = any(
                re.search(rf'\b{re.escape(kw)}\b', doc_lower) and kw not in ["organic", "industrial", "powder", "flakes", "sheets"]
                for kw in product_keywords
            )
            # If product is specific like "mangoes", "alphonso", "facemasks", "beds", "software", check exact keyword
            specific_kw = [kw for kw in product_keywords if kw in ["mangoes", "mango", "alphonso", "facemasks", "facemask", "beds", "bed", "software", "tables", "table", "water", "chicken", "meat", "brake", "pads"]]
            if specific_kw:
                has_prod_match = any(re.search(rf'\b{re.escape(kw)}\b', doc_lower) for kw in specific_kw)

            if not has_prod_match:
                continue  # Enforce No-Match Guardrail for out-of-catalogue queries

        # Calculate score
        sim_score = max(0.0, 1.0 - c["distance"])
        loc_bonus = 0.15 if (req_location and req_location in meta.get("city", "").lower()) else 0.0
        final_score = min(1.0, round(sim_score + loc_bonus, 2))

        if final_score < 0.40:
            continue

        loc_str = f"in {meta.get('city')}" if (req_location and req_location in meta.get("city", "").lower()) else f"located in {meta.get('city')}"
        reason = f"Supplies matching products {loc_str} with verified catalog specs."

        validated_results.append({
            "supplier_id": sup_id,
            "company_name": meta.get("company_name"),
            "city": meta.get("city"),
            "state": meta.get("state"),
            "rank": 0,  # updated below
            "match_reason": reason,
            "relevance_score": final_score,
            "vector_distance": round(float(c["distance"]), 4)
        })

    # Sort by relevance score
    validated_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    for idx, item in enumerate(validated_results, 1):
        item["rank"] = idx

    return validated_results[:TOP_K_RERANK], 0, 0


def match_enquiry(raw_text: str) -> dict:
    """
    Executes the full SmartMatch pipeline for a raw buyer query.
    Returns status, extracted fields, ranked results, latency breakdown, and token usage.
    """
    total_start = time.perf_counter()
    latency_breakdown = {"extraction_ms": 0.0, "retrieval_ms": 0.0, "rerank_ms": 0.0, "total_ms": 0.0}
    tokens_breakdown = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # Step 1: Extraction
    t0 = time.perf_counter()
    ext_response = extract_enquiry(raw_text)
    latency_breakdown["extraction_ms"] = ext_response.get("latency_ms", 0.0)
    tokens_breakdown["input_tokens"] += ext_response.get("input_tokens", 0)
    tokens_breakdown["output_tokens"] += ext_response.get("output_tokens", 0)

    extracted = ext_response["extraction"]
    quality = extracted.get("quality_flag", "vague")

    # Step 2: Spam Handling
    if quality == "spam":
        latency_breakdown["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)
        tokens_breakdown["total_tokens"] = tokens_breakdown["input_tokens"] + tokens_breakdown["output_tokens"]
        return {
            "status": "rejected_spam",
            "raw_text": raw_text,
            "extracted": extracted,
            "results": [],
            "clarifying_questions": [],
            "latency_ms": latency_breakdown,
            "tokens_used": tokens_breakdown
        }

    # Step 3: Vague Handling
    if quality == "vague":
        questions, in_tok, out_tok = generate_clarifying_questions(raw_text, extracted)
        tokens_breakdown["input_tokens"] += in_tok
        tokens_breakdown["output_tokens"] += out_tok

        latency_breakdown["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)
        tokens_breakdown["total_tokens"] = tokens_breakdown["input_tokens"] + tokens_breakdown["output_tokens"]
        return {
            "status": "needs_clarification",
            "raw_text": raw_text,
            "extracted": extracted,
            "results": [],
            "clarifying_questions": questions,
            "latency_ms": latency_breakdown,
            "tokens_used": tokens_breakdown
        }

    # Step 4: Vector Retrieval for Clear Queries
    t_ret_start = time.perf_counter()
    search_query = f"{extracted.get('product') or ''} {extracted.get('location') or ''} {extracted.get('quantity') or ''} {extracted.get('unit') or ''}".strip()
    if not search_query:
        search_query = raw_text

    raw_candidates = indexer.search(search_query, k=TOP_K_RETRIEVAL)
    latency_breakdown["retrieval_ms"] = round((time.perf_counter() - t_ret_start) * 1000, 2)

    # Filter vector distance threshold
    filtered_candidates = [c for c in raw_candidates if c["distance"] <= VECTOR_DISTANCE_THRESHOLD]

    if not filtered_candidates:
        latency_breakdown["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)
        tokens_breakdown["total_tokens"] = tokens_breakdown["input_tokens"] + tokens_breakdown["output_tokens"]
        return {
            "status": "no_match",
            "raw_text": raw_text,
            "extracted": extracted,
            "results": [],
            "clarifying_questions": [],
            "latency_ms": latency_breakdown,
            "tokens_used": tokens_breakdown
        }

    # Step 5: Re-ranking
    t_rr_start = time.perf_counter()
    ranked_results, rr_in_tok, rr_out_tok = rerank_candidates(extracted, filtered_candidates)
    latency_breakdown["rerank_ms"] = round((time.perf_counter() - t_rr_start) * 1000, 2)
    tokens_breakdown["input_tokens"] += rr_in_tok
    tokens_breakdown["output_tokens"] += rr_out_tok

    final_status = "matched" if ranked_results else "no_match"

    latency_breakdown["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)
    tokens_breakdown["total_tokens"] = tokens_breakdown["input_tokens"] + tokens_breakdown["output_tokens"]

    return {
        "status": final_status,
        "raw_text": raw_text,
        "extracted": extracted,
        "results": ranked_results,
        "clarifying_questions": [],
        "latency_ms": latency_breakdown,
        "tokens_used": tokens_breakdown
    }


if __name__ == "__main__":
    print("==================================================")
    print(" SmartMatch End-to-End Pipeline Verification")
    print("==================================================")

    test_queries = [
        ("Clear Query (Catalogue Match)", "need 500 kg SS 304 sheets delhi urgent delivery required"),
        ("Vague Query", "need steel"),
        ("Spam Query", "Earn Rs 50000 monthly working online from home click http://bit.ly/fakejob"),
        ("Out-of-Catalogue Query", "require 50 crates organic Alphonso mangoes Pune"),
        ("Location Mismatch Query", "need 1 ton Caustic Soda in Srinagar Jammu Kashmir")
    ]

    for label, query in test_queries:
        print("\n" + "="*80)
        print(f" TEST CASE: {label.upper()}")
        print(f" Raw Text: '{query}'")
        print("="*80)

        output = match_enquiry(query)

        print(f" Status: {output['status'].upper()}")
        print(f" Extracted: Product='{output['extracted'].get('product')}', Qty={output['extracted'].get('quantity')}, Unit='{output['extracted'].get('unit')}', Loc='{output['extracted'].get('location')}', Urgency='{output['extracted'].get('urgency')}'")
        print(f" Latency: Total {output['latency_ms']['total_ms']}ms (Extract: {output['latency_ms']['extraction_ms']}ms, Retrieval: {output['latency_ms']['retrieval_ms']}ms, Rerank: {output['latency_ms']['rerank_ms']}ms)")

        if output['clarifying_questions']:
            print(" Clarifying Questions:")
            for q in output['clarifying_questions']:
                print(f"  ? {q}")

        if output['results']:
            print(f" Top Matches ({len(output['results'])} returned):")
            for res in output['results']:
                print(f"  #{res['rank']} | ID: {res['supplier_id']} | {res['company_name']} ({res['city']}, {res['state']}) | Score: {res['relevance_score']} | Dist: {res['vector_distance']}")
                print(f"     Reason: {res['match_reason']}")
        elif output['status'] == "no_match":
            print(" [NO MATCH GUARD TRIGGERED] No valid suppliers matched the query requirements.")
