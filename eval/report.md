# SmartMatch Offline Evaluation Report

## 1. Executive Summary
- **Evaluation Dataset**: 60 Ground-Truth Buyer Enquiries
- **Catalogue Size**: 200 Verified Suppliers
- **Supplier Hallucinations**: **0** (PASSED: 0 Hallucinations)

---

## 2. Field Extraction Accuracy

| Field | Accuracy (%) | Benchmark Target | Status |
| :--- | :---: | :---: | :---: |
| **Product Extraction** | **80.0%** | $\ge 90\%$ | ⚠️ ATTENTION |
| **Quantity Extraction** | **76.67%** | $\ge 90\%$ | ⚠️ ATTENTION |
| **Location Extraction** | **96.67%** | $\ge 90\%$ | ✅ PASS |
| **Urgency Extraction** | **41.67%** | $\ge 85\%$ | ⚠️ ATTENTION |

---

## 3. Quality Classification & Spam Detection

### Quality Confusion Matrix
| True \ Predicted | Clear | Vague | Spam | Total True |
| :--- | :---: | :---: | :---: | :---: |
| **Clear** | 40 | 0 | 0 | 40 |
| **Vague** | 7 | 3 | 0 | 10 |
| **Spam** | 0 | 0 | 10 | 10 |

### Spam Metrics
- **Spam Precision**: **100.0%** (Target: $\ge 98\%$)
- **Spam Recall**: **100.0%** (Target: $\ge 95\%$)

---

## 4. Supplier Vector Retrieval & Match Performance

| Metric | Score | Target | Evaluation Notes |
| :--- | :---: | :---: | :--- |
| **Top-1 Hit Rate** | **13.33%** | $\ge 70\%$ | Exact #1 rank match on in-catalogue queries |
| **Top-3 Hit Rate** | **46.67%** | $\ge 85\%$ | Presence of expected supplier in top 3 |
| **Mean Reciprocal Rank (MRR)** | **0.3067** | $\ge 0.80$ | Average inverse rank position of target supplier |
| **No-Match Precision (Out-of-Cat)** | **90.0%** | $\ge 90\%$ | Correctly returning NO MATCH for unserviceable products |
| **False No-Match Rate** | **10.0%** | $\le 5\%$ | Rate of falsely rejecting valid in-catalogue queries |

---

## 5. Operational Telemetry & Cost Analysis

| Operational Telemetry | Value |
| :--- | :--- |
| **Average End-to-End Latency** | **162.6 ms** |
| **p95 End-to-End Latency** | **227.87 ms** |
| **Average Input Tokens / Query** | **0.0 tokens** |
| **Average Output Tokens / Query** | **0.0 tokens** |
| **Estimated Cost / Query** | **$0.000000 USD** |

---

## 6. Top 10 Worst Failure Analysis

| # | Enquiry ID | Raw Query | Failure Reasons | Expected vs Actual |
| :---: | :---: | :--- | :--- | :--- |
| **1** | `ENQ-5001` | *need 500 kg SS 304 sheets delhi urgent delivery required* | False No-Match on valid query; Missed Top-3 expected supplier match | **Exp**: Product: SS 304 Sheets | Suppliers: ['SUP-1004', 'SUP-1006']<br/>**Got**: Status: no_match | Quality: clear | Returned: [] |
| **2** | `ENQ-5023` | *looking for 50W LED driver modules IP65 500 pcs Vadodara* | False No-Match on valid query; Missed Top-3 expected supplier match | **Exp**: Product: LED Driver Modules | Suppliers: ['SUP-1137', 'SUP-1138', 'SUP-1141']<br/>**Got**: Status: no_match | Quality: clear | Returned: [] |
| **3** | `ENQ-5025` | *want 7.5 HP three phase electric motor 1440 RPM Mumbai urgent* | False No-Match on valid query; Missed Top-3 expected supplier match | **Exp**: Product: Three Phase Electric Motor | Suppliers: ['SUP-1136', 'SUP-1137']<br/>**Got**: Status: no_match | Quality: clear | Returned: [] |
| **4** | `ENQ-5002` | *require 2 tons MS angles 50x50mm in Ludhiana for factory construction* | Missed Top-3 expected supplier match | **Exp**: Product: MS Angles | Suppliers: ['SUP-1001', 'SUP-1003']<br/>**Got**: Status: matched | Quality: clear | Returned: ['SUP-1023', 'SUP-1008', 'SUP-1016'] |
| **5** | `ENQ-5004` | *looking for 300 kg aluminum coils 1.5mm thick ahmedabad* | Missed Top-3 expected supplier match | **Exp**: Product: Aluminum Coils | Suppliers: ['SUP-1002', 'SUP-1009']<br/>**Got**: Status: matched | Quality: clear | Returned: ['SUP-1032', 'SUP-1030', 'SUP-1021'] |
| **6** | `ENQ-5005` | *urgent requirement copper pipes 1/4 inch 100 kg rajkot* | Missed Top-3 expected supplier match | **Exp**: Product: Copper Pipes | Suppliers: ['SUP-1002', 'SUP-1003']<br/>**Got**: Status: matched | Quality: clear | Returned: ['SUP-1010', 'SUP-1026', 'SUP-1014'] |
| **7** | `ENQ-5008` | *require 500 rolls brown BOPP tape 48mm delhi ncr* | Missed Top-3 expected supplier match | **Exp**: Product: BOPP Tapes | Suppliers: ['SUP-1035', 'SUP-1036']<br/>**Got**: Status: matched | Quality: clear | Returned: ['SUP-1052', 'SUP-1054', 'SUP-1041'] |
| **8** | `ENQ-5010` | *need 50 blue HDPE drums 200 Litre open top in Vapi* | Missed Top-3 expected supplier match | **Exp**: Product: HDPE Drums | Suppliers: ['SUP-1038', 'SUP-1040']<br/>**Got**: Status: matched | Quality: clear | Returned: ['SUP-1062', 'SUP-1063', 'SUP-1059'] |
| **9** | `ENQ-5012` | *plz give best rate for 1 ton polyester yarn 150D Surat bulk order* | Missed Top-3 expected supplier match | **Exp**: Product: Polyester Yarn | Suppliers: ['SUP-1069', 'SUP-1070']<br/>**Got**: Status: matched | Quality: clear | Returned: ['SUP-1098', 'SUP-1091', 'SUP-1072'] |
| **10** | `ENQ-5013` | *require indigo denim fabric 12 oz 1000 meters jaipur urgent* | Missed Top-3 expected supplier match | **Exp**: Product: Denim Fabric | Suppliers: ['SUP-1071', 'SUP-1074']<br/>**Got**: Status: matched | Quality: clear | Returned: ['SUP-1087', 'SUP-1095', 'SUP-1089'] |
