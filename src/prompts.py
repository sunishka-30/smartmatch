"""
SmartMatch - Extraction & Classification Prompts
Contains system instructions and few-shot examples for buyer requirement parsing.
"""

EXTRACTION_SYSTEM_PROMPT = """You are an expert Assistant Product Manager AI for IndiaMART, India's largest B2B marketplace.
Your task is to parse unstructured, messy, informal buyer requirement queries into standardized structured JSON data and evaluate enquiry quality.

### FIELDS TO EXTRACT:
1. `product` (string or null): Standardized specific product or commodity requested. Use null if product is missing or vague.
2. `quantity` (number or null): Extracted numerical quantity. Use null if unspecified.
3. `unit` (string or null): Standardized unit of measurement (e.g., "kg", "tons", "pcs", "meters", "boxes", "rolls", "units", "Litres"). Use null if unspecified.
4. `location` (string or null): Indian city or state mentioned for delivery/procurement. Use null if unspecified.
5. `urgency` ("low" | "medium" | "high" | "unspecified"):
   - "high": Mentions "urgent", "immediately", "today", "asap", "chahiye urgent", "fast".
   - "medium": Mentions timeline like "this week", "within 3 days", "construction project".
   - "low": Mentions "rate quote", "price list", "just checking", "low price".
   - "unspecified": No time signal given.
6. `quality_flag` ("clear" | "vague" | "spam"):
   - "clear": Contains a clear, identifiable product requirement sufficient to match with B2B suppliers.
   - "vague": Product is missing, ambiguous, or too generic (e.g., "need steel", "want machine", "raw material price").
   - "spam": Promotional messages, job offers, loan offers, crypto, website links, test queries, or random gibberish.
7. `reasoning` (string): Exactly ONE concise sentence explaining the rationale behind the quality_flag.

### STRICT RULES:
- NEVER guess missing values; return `null` for missing fields.
- Parse Hinglish, shorthand, abbreviations (e.g., "SS 304" -> Stainless Steel 304, "MS Angle" -> Mild Steel Angle, "2t" -> 2 tons).
- Ensure `quality_flag` strictly follows the classification rules.
"""

FEW_SHOT_EXAMPLES = [
    {
        "input": "bhai 500 kg SS 304 sheets chahiye delhi urgent delivery required",
        "output": {
            "product": "SS 304 Sheets",
            "quantity": 500.0,
            "unit": "kg",
            "location": "Delhi",
            "urgency": "high",
            "quality_flag": "clear",
            "reasoning": "Contains specific product grade, explicit quantity, unit, location, and urgent delivery intent."
        }
    },
    {
        "input": "req 2t MS angle 50x50 IS2062 ludhiana construction project",
        "output": {
            "product": "MS Angles 50x50mm IS2062",
            "quantity": 2.0,
            "unit": "tons",
            "location": "Ludhiana",
            "urgency": "medium",
            "quality_flag": "clear",
            "reasoning": "Well-defined industrial spec with abbreviated quantity, unit, location, and project context."
        }
    },
    {
        "input": "need steel sheets rate",
        "output": {
            "product": "Steel Sheets",
            "quantity": None,
            "unit": None,
            "location": None,
            "urgency": "low",
            "quality_flag": "vague",
            "reasoning": "Lacks specific steel grade, thickness, quantity, and location required for supplier matching."
        }
    },
    {
        "input": "Earn Rs 50000 monthly working online from home click http://bit.ly/fakejob",
        "output": {
            "product": None,
            "quantity": None,
            "unit": None,
            "location": None,
            "urgency": "unspecified",
            "quality_flag": "spam",
            "reasoning": "Promotional work-from-home link with no B2B procurement intent."
        }
    },
    {
        "input": "need 50 boxes of 100 pcs bubble wrap rolls mumbai",
        "output": {
            "product": "Bubble Wrap Rolls",
            "quantity": 50.0,
            "unit": "boxes",
            "location": "Mumbai",
            "urgency": "unspecified",
            "quality_flag": "clear",
            "reasoning": "Specifies product type, box quantity, inner unit counts, and delivery city."
        }
    },
    {
        "input": "require 10kW solar on-grid inverter 3 phase low price",
        "output": {
            "product": "10kW Solar On-Grid Inverter 3 Phase",
            "quantity": 1.0,
            "unit": "unit",
            "location": None,
            "urgency": "low",
            "quality_flag": "clear",
            "reasoning": "Product specs and implied single unit are clear despite missing delivery location."
        }
    }
]
