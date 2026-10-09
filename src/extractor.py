"""
SmartMatch - Buyer Enquiry Extractor & Quality Classifier
Parses raw, unstructured buyer queries into structured JSON fields using Gemini API (or rule-based fallback).
"""

import os
import re
import sys
import json
import time
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Literal

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prompts import EXTRACTION_SYSTEM_PROMPT, FEW_SHOT_EXAMPLES

# Load environment variables
load_dotenv()


class ExtractedEnquirySchema(BaseModel):
    product: Optional[str] = Field(default=None, description="Standardized product name or null if missing/unspecified")
    quantity: Optional[float] = Field(default=None, description="Numerical quantity required or null if unspecified")
    unit: Optional[str] = Field(default=None, description="Standardized unit of measure or null if unspecified")
    location: Optional[str] = Field(default=None, description="Delivery city/location name or null if unspecified")
    urgency: Literal["low", "medium", "high", "unspecified"] = Field(default="unspecified", description="Inferred requirement urgency level")
    quality_flag: Literal["clear", "vague", "spam"] = Field(default="vague", description="Enquiry quality classification")
    reasoning: str = Field(description="One short sentence explaining quality_flag rationale")


def get_gemini_client():
    """Initializes Gemini API client if API key is configured."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.startswith("your_"):
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def rule_based_fallback_extractor(raw_text: str, start_time: float) -> dict:
    """Rule-based heuristic extractor used when Gemini API is unconfigured or unavailable."""
    text_lower = raw_text.lower().strip()

    # 1. Check for Spam (Promotions, Non-B2B Chatter, Test Queries, Job Scams)
    spam_keywords = [
        "earn", "http", "www.", "click", "cibil", "job", "salary", "telecaller",
        "crypto", "asdfgh", "followers", "loan", "data entry", "payout",
        "testing query", "test test", "ignore", "call me back", "hello sir how are you"
    ]
    if any(kw in text_lower for kw in spam_keywords) or len(text_lower) < 6:
        extraction = {
            "product": None,
            "quantity": None,
            "unit": None,
            "location": None,
            "urgency": "unspecified",
            "quality_flag": "spam",
            "reasoning": "Query contains promotional links, conversational chatter, or non-B2B spam patterns."
        }
        return {
            "extraction": extraction,
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "input_tokens": 0,
            "output_tokens": 0
        }

    # 2. Check for Vague
    vague_phrases = ["need steel", "want packaging", "price list", "need machine", "wire required", "contact number", "buying raw material", "need plastic"]
    if any(text_lower.startswith(vp) or text_lower == vp for vp in vague_phrases) or (len(text_lower.split()) <= 3 and not re.search(r'\d+', text_lower)):
        extraction = {
            "product": raw_text.strip().title(),
            "quantity": None,
            "unit": None,
            "location": None,
            "urgency": "low" if "price" in text_lower or "rate" in text_lower else "unspecified",
            "quality_flag": "vague",
            "reasoning": "Query lacks specific technical parameters, grade, quantity, or location for matching."
        }
        return {
            "extraction": extraction,
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "input_tokens": 0,
            "output_tokens": 0
        }

    # 3. Clear Query Heuristic Extraction
    # Quantity & Unit
    qty_match = re.search(r'(\d+(?:\.\d+)?)\s*(kg|tons|ton|pcs|boxes|rolls|meters|units|litres|l|m|crates|sets|carboys|bags)?', text_lower)
    quantity = float(qty_match.group(1)) if qty_match else None
    unit = qty_match.group(2) if (qty_match and qty_match.group(2)) else None

    # Common Indian Cities
    cities = ["Delhi", "Mumbai", "Surat", "Pune", "Ludhiana", "Ahmedabad", "Chennai", "Kolkata", "Rajkot", "Vapi", "Bangalore", "Hyderabad", "Noida", "Gurgaon", "Jaipur", "Coimbatore", "Baddi", "Bhiwandi", "Kanpur", "Tirupur", "Ankleshwar", "Thane"]
    location = None
    for city in cities:
        if city.lower() in text_lower:
            location = city
            break

    # Urgency Classification
    if any(w in text_lower for w in ["urgent", "immediately", "asap", "chahiye urgent", "fast", "today"]):
        urgency = "high"
    elif any(w in text_lower for w in ["project", "construction", "this week", "within", "factory"]):
        urgency = "medium"
    elif any(w in text_lower for w in ["rate", "price", "quote", "price list", "low price"]):
        urgency = "low"
    else:
        urgency = "unspecified"

    # Clean Product Name: strip quantities, units, action verbs, and city names
    clean_product = raw_text
    words_to_strip = [
        "need", "require", "want", "bhai", "urgent", "delivery", "required", "plz", "send", "looking", "for",
        "rate", "quote", "price", "list", "construction", "factory", "project", "export", "pack", "fast", "today", "installation"
    ]
    for w in words_to_strip:
        clean_product = re.sub(rf'\b{w}\b', '', clean_product, flags=re.IGNORECASE)

    # Strip numbers and units from product string
    clean_product = re.sub(r'\b\d+(?:\.\d+)?\s*(kg|tons|ton|pcs|boxes|rolls|meters|units|litres|crates|sets|carboys|bags)?\b', '', clean_product, flags=re.IGNORECASE)

    # Strip city names from product string
    for city in cities:
        clean_product = re.sub(rf'\b{city}\b', '', clean_product, flags=re.IGNORECASE)

    # Clean punctuation and whitespace
    clean_product = re.sub(r'[^\w\s]', '', clean_product).strip().title()
    product = clean_product if clean_product else "Industrial Product"

    extraction = {
        "product": product,
        "quantity": quantity,
        "unit": unit,
        "location": location,
        "urgency": urgency,
        "quality_flag": "clear",
        "reasoning": "Extracted product details, quantity, and delivery parameters."
    }

    return {
        "extraction": extraction,
        "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
        "input_tokens": 0,
        "output_tokens": 0
    }


def extract_enquiry(raw_text: str) -> dict:
    """
    Parses a raw buyer text query into a structured dict with extraction fields, quality flag,
    reasoning, latency_ms, and token usage counts.
    """
    start_time = time.perf_counter()
    client = get_gemini_client()

    if client:
        try:
            from google.genai import types

            # Build prompt with system instruction and few-shot examples
            prompt_content = "Here are reference examples of query parsing:\n\n"
            for example in FEW_SHOT_EXAMPLES:
                prompt_content += f"Query: {example['input']}\nOutput JSON: {json.dumps(example['output'])}\n\n"
            prompt_content += f"Now parse this buyer query into JSON:\nQuery: {raw_text}\nOutput JSON:"

            # Primary Attempt
            for attempt in range(2):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt_content,
                        config=types.GenerateContentConfig(
                            system_instruction=EXTRACTION_SYSTEM_PROMPT,
                            response_mime_type="application/json",
                            response_schema=ExtractedEnquirySchema,
                            temperature=0.1
                        )
                    )

                    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    usage = getattr(response, "usage_metadata", None)
                    input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
                    output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

                    extracted_json = json.loads(response.text)

                    return {
                        "extraction": extracted_json,
                        "latency_ms": latency_ms,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens
                    }
                except Exception as attempt_err:
                    if attempt == 1:
                        raise attempt_err
                    time.sleep(0.5)

        except Exception as e:
            print(f"[!] Gemini extraction API call failed: {e}. Switching to rule-based fallback engine.")

    # Fallback to deterministic rule-based extractor
    return rule_based_fallback_extractor(raw_text, start_time)


if __name__ == "__main__":
    print("==================================================")
    print(" SmartMatch Buyer Enquiry Extractor Test")
    print("==================================================")

    sample_enquiries = [
        "need 500 kg SS 304 sheets delhi urgent delivery required",
        "bhai 1000 pcs 5 ply corrugated boxes Surat export pack fast",
        "req 2t MS angle Ludhiana construction",
        "want price list for cotton fabric rolls",
        "need steel",
        "Earn Rs 50000 monthly working online from home click http://bit.ly/fakejob",
        "asdfghjkl 12345 test query",
        "require 50 crates organic Alphonso mangoes Pune"
    ]

    results_list = []
    for enq in sample_enquiries:
        res = extract_enquiry(enq)
        ext = res["extraction"]
        results_list.append({
            "Raw Query": enq,
            "Quality": ext.get("quality_flag"),
            "Product": ext.get("product"),
            "Qty": ext.get("quantity"),
            "Unit": ext.get("unit"),
            "Location": ext.get("location"),
            "Urgency": ext.get("urgency"),
            "Latency (ms)": res.get("latency_ms"),
            "Tokens (In/Out)": f"{res.get('input_tokens')}/{res.get('output_tokens')}",
            "Reasoning": ext.get("reasoning")
        })

    df = pd.DataFrame(results_list)
    print("\n" + "="*80)
    print(" EXTRACTION RESULTS TABLE")
    print("="*80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df[['Raw Query', 'Quality', 'Product', 'Qty', 'Unit', 'Location', 'Urgency', 'Latency (ms)']].to_string(index=False))

    print("\n" + "="*80)
    print(" DETAILED REASONING BREAKDOWN")
    print("="*80)
    for i, row in enumerate(results_list, 1):
        print(f"{i}. [{row['Quality'].upper()}] {row['Raw Query']}")
        print(f"   Reasoning: {row['Reasoning']}\n")
