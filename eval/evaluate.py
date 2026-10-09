"""
SmartMatch - Offline Evaluation & Benchmarking Engine
Evaluates end-to-end performance across eval/enquiries.csv:
1. Field Extraction Accuracy (Product, Quantity, Location, Urgency).
2. Quality Classification Confusion Matrix & Spam Precision/Recall.
3. Top-1, Top-3 Match Hit Rates & Mean Reciprocal Rank (MRR).
4. No-Match Precision & False No-Match Rate.
5. Strict Supplier Hallucination Check.
6. Operational Telemetry (Avg/P95 Latency, Token Usage, Cost/Query).

Outputs eval/results.json and a detailed Markdown report at eval/report.md.
"""

import os
import sys
import json
import time
import re
import difflib
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.matcher import match_enquiry

ENQUIRIES_PATH = os.path.join(PROJECT_ROOT, "eval", "enquiries.csv")
SUPPLIERS_PATH = os.path.join(PROJECT_ROOT, "data", "suppliers.csv")
RESULTS_JSON_PATH = os.path.join(PROJECT_ROOT, "eval", "results.json")
REPORT_MD_PATH = os.path.join(PROJECT_ROOT, "eval", "report.md")


def normalize_quantity(val) -> float | None:
    """Extracts numerical quantity from string or float."""
    if pd.isna(val) or val is None:
        return None
    val_str = str(val).lower().strip()
    if val_str in ["none", "n/a", "unspecified", ""]:
        return None
    match = re.search(r'(\d+(?:\.\d+)?)', val_str)
    if match:
        return float(match.group(1))
    return None


def match_product(true_p: str, ext_p: str) -> bool:
    """Fuzzy product matching using string similarity and token set overlap."""
    if not true_p or str(true_p).lower().strip() in ["none", "n/a", "unspecified", ""]:
        return not ext_p or str(ext_p).lower().strip() in ["none", "n/a", "unspecified", ""]
    if not ext_p or str(ext_p).lower().strip() in ["none", "n/a", "unspecified", ""]:
        return False

    t_clean = re.sub(r'[^\w\s]', '', str(true_p).lower()).strip()
    e_clean = re.sub(r'[^\w\s]', '', str(ext_p).lower()).strip()

    if t_clean in e_clean or e_clean in t_clean:
        return True

    # Token overlap ratio
    t_tokens = set(t_clean.split())
    e_tokens = set(e_clean.split())
    if not t_tokens or not e_tokens:
        return False

    overlap = len(t_tokens.intersection(e_tokens)) / len(t_tokens)
    if overlap >= 0.50:
        return True

    # Difflib sequence similarity
    ratio = difflib.SequenceMatcher(None, t_clean, e_clean).ratio()
    return ratio >= 0.60


def match_location(true_loc: str, ext_loc: str) -> bool:
    """Normalised location matching."""
    t_str = str(true_loc).lower().strip() if true_loc and not pd.isna(true_loc) else ""
    e_str = str(ext_loc).lower().strip() if ext_loc and not pd.isna(ext_loc) else ""

    if t_str in ["none", "n/a", "unspecified", ""]:
        return e_str in ["none", "n/a", "unspecified", ""]

    if not e_str or e_str in ["none", "n/a", "unspecified", ""]:
        return False

    return t_str in e_str or e_str in t_str


def match_urgency(true_urg: str, ext_urg: str) -> bool:
    """Normalised urgency matching."""
    t_str = str(true_urg).lower().strip() if true_urg and not pd.isna(true_urg) else "unspecified"
    e_str = str(ext_urg).lower().strip() if ext_urg and not pd.isna(ext_urg) else "unspecified"
    return t_str == e_str


def run_evaluation():
    print("==================================================")
    print(" SmartMatch Offline Benchmark Evaluation Engine")
    print("==================================================")

    if not os.path.exists(ENQUIRIES_PATH):
        raise FileNotFoundError(f"Evaluation dataset not found at {ENQUIRIES_PATH}. Run src/generate_data.py first.")

    enquiries_df = pd.read_csv(ENQUIRIES_PATH)
    suppliers_df = pd.read_csv(SUPPLIERS_PATH)
    valid_supplier_ids = set(suppliers_df['supplier_id'].astype(str))

    print(f"[+] Loaded {len(enquiries_df)} ground-truth evaluation queries.")
    print(f"[+] Loaded {len(valid_supplier_ids)} catalogue suppliers for hallucination verification.")

    eval_results = []
    latencies = []
    input_tokens = []
    output_tokens = []
    hallucination_violations = []

    field_correct = {"product": 0, "quantity": 0, "location": 0, "urgency": 0}
    total_eval = len(enquiries_df)

    confusion_matrix = {"clear": {"clear": 0, "vague": 0, "spam": 0},
                        "vague": {"clear": 0, "vague": 0, "spam": 0},
                        "spam": {"clear": 0, "vague": 0, "spam": 0}}

    mrr_list = []
    hit1_count = 0
    hit3_count = 0
    in_cat_query_count = 0

    no_match_correct = 0
    out_of_cat_count = 0
    false_no_match_count = 0

    failures = []

    print("\n[+] Executing evaluation pipeline over dataset...")
    for idx, row in tqdm(enquiries_df.iterrows(), total=total_eval, desc="Evaluating Queries"):
        enq_id = str(row['enquiry_id'])
        raw_text = str(row['raw_text'])
        true_product = row['true_product']
        true_qty = row['true_quantity']
        true_loc = row['true_location']
        true_urgency = str(row['true_urgency'])
        true_quality = str(row['true_quality']).lower()

        # Parse ground-truth supplier IDs
        raw_expected_ids = str(row['expected_supplier_ids'])
        try:
            expected_supplier_ids = json.loads(raw_expected_ids) if raw_expected_ids else []
        except Exception:
            expected_supplier_ids = []

        # Run Pipeline
        pipeline_output = match_enquiry(raw_text)

        status = pipeline_output['status']
        extracted = pipeline_output['extracted']
        results = pipeline_output['results']
        lat = pipeline_output['latency_ms']
        tok = pipeline_output['tokens_used']

        latencies.append(lat.get("total_ms", 0.0))
        input_tokens.append(tok.get("input_tokens", 0))
        output_tokens.append(tok.get("output_tokens", 0))

        # 1. Extraction Accuracy Checks
        p_match = match_product(true_product, extracted.get("product"))
        q_match = (normalize_quantity(true_qty) == normalize_quantity(extracted.get("quantity")))
        l_match = match_location(true_loc, extracted.get("location"))
        u_match = match_urgency(true_urgency, extracted.get("urgency"))

        if p_match: field_correct["product"] += 1
        if q_match: field_correct["quantity"] += 1
        if l_match: field_correct["location"] += 1
        if u_match: field_correct["urgency"] += 1

        # 2. Quality Classification Confusion Matrix
        pred_quality = extracted.get("quality_flag", "vague").lower()
        if true_quality in confusion_matrix and pred_quality in confusion_matrix[true_quality]:
            confusion_matrix[true_quality][pred_quality] += 1

        # 3. Match Quality Checks (for queries with expected suppliers)
        returned_ids = [r['supplier_id'] for r in results]

        # Hallucination check
        for r_id in returned_ids:
            if r_id not in valid_supplier_ids:
                hallucination_violations.append({
                    "enquiry_id": enq_id,
                    "query": raw_text,
                    "invalid_id": r_id
                })

        if expected_supplier_ids:
            in_cat_query_count += 1
            # Check Hit@1 & Hit@3
            if returned_ids and returned_ids[0] in expected_supplier_ids:
                hit1_count += 1
            if any(r_id in expected_supplier_ids for r_id in returned_ids[:3]):
                hit3_count += 1

            # Reciprocal Rank
            rr = 0.0
            for r_idx, r_id in enumerate(returned_ids, 1):
                if r_id in expected_supplier_ids:
                    rr = 1.0 / r_idx
                    break
            mrr_list.append(rr)

            if status == "no_match":
                false_no_match_count += 1

        else: # Out-of-catalogue or spam/vague
            if true_quality == "clear":
                out_of_cat_count += 1
                if status == "no_match":
                    no_match_correct += 1

        # Failure Severity Scoring
        failure_reasons = []
        severity = 0

        if true_quality == "spam" and pred_quality != "spam":
            failure_reasons.append(f"Spam misclassified as {pred_quality}")
            severity += 3
        elif true_quality != "spam" and pred_quality == "spam":
            failure_reasons.append(f"Valid enquiry misclassified as spam")
            severity += 3

        if expected_supplier_ids and status == "no_match":
            failure_reasons.append("False No-Match on valid query")
            severity += 3

        if expected_supplier_ids and not any(r_id in expected_supplier_ids for r_id in returned_ids[:3]):
            failure_reasons.append("Missed Top-3 expected supplier match")
            severity += 2

        if not p_match:
            failure_reasons.append(f"Product mismatch (Exp: '{true_product}', Ext: '{extracted.get('product')}')")
            severity += 1

        if failure_reasons:
            failures.append({
                "severity": severity,
                "enquiry_id": enq_id,
                "raw_text": raw_text,
                "true_quality": true_quality,
                "expected": f"Product: {true_product} | Suppliers: {expected_supplier_ids}",
                "got": f"Status: {status} | Quality: {pred_quality} | Returned: {returned_ids[:3]}",
                "failure_reasons": "; ".join(failure_reasons)
            })

    # Metrics Calculation
    ext_acc = {field: round((count / total_eval) * 100, 2) for field, count in field_correct.items()}

    # Spam Precision & Recall
    tp_spam = confusion_matrix["spam"]["spam"]
    fp_spam = confusion_matrix["clear"]["spam"] + confusion_matrix["vague"]["spam"]
    fn_spam = confusion_matrix["spam"]["clear"] + confusion_matrix["spam"]["vague"]

    spam_precision = round(tp_spam / (tp_spam + fp_spam) * 100, 2) if (tp_spam + fp_spam) > 0 else 100.0
    spam_recall = round(tp_spam / (tp_spam + fn_spam) * 100, 2) if (tp_spam + fn_spam) > 0 else 100.0

    top1_hit_rate = round((hit1_count / in_cat_query_count) * 100, 2) if in_cat_query_count > 0 else 0.0
    top3_hit_rate = round((hit3_count / in_cat_query_count) * 100, 2) if in_cat_query_count > 0 else 0.0
    mrr = round(float(np.mean(mrr_list)), 4) if mrr_list else 0.0

    no_match_precision = round((no_match_correct / out_of_cat_count) * 100, 2) if out_of_cat_count > 0 else 100.0
    false_no_match_rate = round((false_no_match_count / in_cat_query_count) * 100, 2) if in_cat_query_count > 0 else 0.0

    avg_lat = round(float(np.mean(latencies)), 2)
    p95_lat = round(float(np.percentile(latencies, 95)), 2)
    avg_in_tok = round(float(np.mean(input_tokens)), 1)
    avg_out_tok = round(float(np.mean(output_tokens)), 1)
    avg_total_tok = round(avg_in_tok + avg_out_tok, 1)
    est_cost_per_query = round(avg_total_tok * 0.0000003, 6)

    # Save Results JSON
    summary_dict = {
        "extraction_accuracy_pct": ext_acc,
        "quality_confusion_matrix": confusion_matrix,
        "spam_metrics": {"precision_pct": spam_precision, "recall_pct": spam_recall},
        "match_metrics": {
            "top1_hit_rate_pct": top1_hit_rate,
            "top3_hit_rate_pct": top3_hit_rate,
            "mrr": mrr
        },
        "no_match_metrics": {
            "out_of_catalogue_precision_pct": no_match_precision,
            "false_no_match_rate_pct": false_no_match_rate
        },
        "hallucination_count": len(hallucination_violations),
        "telemetry": {
            "avg_latency_ms": avg_lat,
            "p95_latency_ms": p95_lat,
            "avg_input_tokens": avg_in_tok,
            "avg_output_tokens": avg_out_tok,
            "avg_total_tokens": avg_total_tok,
            "est_cost_per_query_usd": est_cost_per_query
        }
    }

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"\n[OK] Saved evaluation JSON metrics to {RESULTS_JSON_PATH}")

    # Generate Human-Readable Markdown Report
    failures.sort(key=lambda x: x['severity'], reverse=True)
    top_10_failures = failures[:10]

    report_md = f"""# SmartMatch Offline Evaluation Report

## 1. Executive Summary
- **Evaluation Dataset**: {total_eval} Ground-Truth Buyer Enquiries
- **Catalogue Size**: {len(valid_supplier_ids)} Verified Suppliers
- **Supplier Hallucinations**: **{len(hallucination_violations)}** {"(PASSED: 0 Hallucinations)" if len(hallucination_violations) == 0 else "🚨 FAILED: HALLUCINATED SUPPLIER DETECTED"}

---

## 2. Field Extraction Accuracy

| Field | Accuracy (%) | Benchmark Target | Status |
| :--- | :---: | :---: | :---: |
| **Product Extraction** | **{ext_acc['product']}%** | $\ge 90\%$ | {"✅ PASS" if ext_acc['product'] >= 90 else "⚠️ ATTENTION"} |
| **Quantity Extraction** | **{ext_acc['quantity']}%** | $\ge 90\%$ | {"✅ PASS" if ext_acc['quantity'] >= 90 else "⚠️ ATTENTION"} |
| **Location Extraction** | **{ext_acc['location']}%** | $\ge 90\%$ | {"✅ PASS" if ext_acc['location'] >= 90 else "⚠️ ATTENTION"} |
| **Urgency Extraction** | **{ext_acc['urgency']}%** | $\ge 85\%$ | {"✅ PASS" if ext_acc['urgency'] >= 85 else "⚠️ ATTENTION"} |

---

## 3. Quality Classification & Spam Detection

### Quality Confusion Matrix
| True \\ Predicted | Clear | Vague | Spam | Total True |
| :--- | :---: | :---: | :---: | :---: |
| **Clear** | {confusion_matrix['clear']['clear']} | {confusion_matrix['clear']['vague']} | {confusion_matrix['clear']['spam']} | {sum(confusion_matrix['clear'].values())} |
| **Vague** | {confusion_matrix['vague']['clear']} | {confusion_matrix['vague']['vague']} | {confusion_matrix['vague']['spam']} | {sum(confusion_matrix['vague'].values())} |
| **Spam** | {confusion_matrix['spam']['clear']} | {confusion_matrix['spam']['vague']} | {confusion_matrix['spam']['spam']} | {sum(confusion_matrix['spam'].values())} |

### Spam Metrics
- **Spam Precision**: **{spam_precision}%** (Target: $\ge 98\%$)
- **Spam Recall**: **{spam_recall}%** (Target: $\ge 95\%$)

---

## 4. Supplier Vector Retrieval & Match Performance

| Metric | Score | Target | Evaluation Notes |
| :--- | :---: | :---: | :--- |
| **Top-1 Hit Rate** | **{top1_hit_rate}%** | $\ge 70\%$ | Exact #1 rank match on in-catalogue queries |
| **Top-3 Hit Rate** | **{top3_hit_rate}%** | $\ge 85\%$ | Presence of expected supplier in top 3 |
| **Mean Reciprocal Rank (MRR)** | **{mrr}** | $\ge 0.80$ | Average inverse rank position of target supplier |
| **No-Match Precision (Out-of-Cat)** | **{no_match_precision}%** | $\ge 90\%$ | Correctly returning NO MATCH for unserviceable products |
| **False No-Match Rate** | **{false_no_match_rate}%** | $\le 5\%$ | Rate of falsely rejecting valid in-catalogue queries |

---

## 5. Operational Telemetry & Cost Analysis

| Operational Telemetry | Value |
| :--- | :--- |
| **Average End-to-End Latency** | **{avg_lat} ms** |
| **p95 End-to-End Latency** | **{p95_lat} ms** |
| **Average Input Tokens / Query** | **{avg_in_tok} tokens** |
| **Average Output Tokens / Query** | **{avg_out_tok} tokens** |
| **Estimated Cost / Query** | **${est_cost_per_query:.6f} USD** |

---

## 6. Top 10 Worst Failure Analysis

| # | Enquiry ID | Raw Query | Failure Reasons | Expected vs Actual |
| :---: | :---: | :--- | :--- | :--- |
"""

    for i, fail in enumerate(top_10_failures, 1):
        report_md += f"| **{i}** | `{fail['enquiry_id']}` | *{fail['raw_text']}* | {fail['failure_reasons']} | **Exp**: {fail['expected']}<br/>**Got**: {fail['got']} |\n"

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[OK] Generated Markdown evaluation report at {REPORT_MD_PATH}")

    # Print Terminal Summary
    print("\n" + "="*50)
    print(" EVALUATION SUMMARY RESULTS")
    print("="*50)
    print(f" • Product Extraction Accuracy : {ext_acc['product']}%")
    print(f" • Quantity Extraction Accuracy: {ext_acc['quantity']}%")
    print(f" • Location Extraction Accuracy: {ext_acc['location']}%")
    print(f" • Spam Precision / Recall     : {spam_precision}% / {spam_recall}%")
    print(f" • Top-1 / Top-3 Hit Rate      : {top1_hit_rate}% / {top3_hit_rate}%")
    print(f" • Mean Reciprocal Rank (MRR)  : {mrr}")
    print(f" • Out-of-Cat No-Match Precision: {no_match_precision}%")
    print(f" • False No-Match Rate         : {false_no_match_rate}%")
    print(f" • Supplier Hallucination Count: {len(hallucination_violations)} (ZERO REQUIREMENT)")
    print(f" • Avg / p95 Latency           : {avg_lat} ms / {p95_lat} ms")
    print(f" • Est. Cost / Query           : ${est_cost_per_query:.6f} USD")
    print("="*50)


if __name__ == "__main__":
    run_evaluation()
