"""
SmartMatch - Data Generation Script
Generates synthetic dataset for IndiaMART APM portfolio project:
1. data/suppliers.csv - 200 realistic Indian B2B suppliers across 6 categories.
2. eval/enquiries.csv - 60 buyer enquiries with ground-truth labels and matching supplier IDs.

Supports generation via Gemini API with retry logic & fallback to deterministic Python generation.
"""

import os
import json
import random
import time
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set fixed random seeds for reproducibility
SEED = 42
random.seed(SEED)

# Output paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "eval")
SUPPLIERS_PATH = os.path.join(DATA_DIR, "suppliers.csv")
ENQUIRIES_PATH = os.path.join(EVAL_DIR, "enquiries.csv")

CATEGORIES = [
    "steel & metals",
    "packaging materials",
    "textiles & fabrics",
    "industrial machinery",
    "electrical components",
    "chemicals"
]

CITIES_STATES = [
    ("Delhi", "Delhi"),
    ("Mumbai", "Maharashtra"),
    ("Ludhiana", "Punjab"),
    ("Surat", "Gujarat"),
    ("Pune", "Maharashtra"),
    ("Chennai", "Tamil Nadu"),
    ("Ahmedabad", "Gujarat"),
    ("Kolkata", "West Bengal"),
    ("Rajkot", "Gujarat"),
    ("Kanpur", "Uttar Pradesh"),
    ("Jaipur", "Rajasthan"),
    ("Coimbatore", "Tamil Nadu"),
    ("Baddi", "Himachal Pradesh"),
    ("Ankleshwar", "Gujarat"),
    ("Faridabad", "Haryana"),
    ("Bangalore", "Karnataka"),
    ("Hyderabad", "Telangana"),
    ("Panipat", "Haryana"),
    ("Vapi", "Gujarat"),
    ("Bhiwandi", "Maharashtra")
]

# Fallback dataset definitions for realistic Indian B2B suppliers
SUPPLIER_TEMPLATES = {
    "steel & metals": {
        "prefixes": ["Jindal", "Apex", "Shree Ram", "Tata", "Sunflag", "Garg", "Mittal", "Vardhman", "Mahavir", "Kamdhenu", "National", "Bharat", "Swastik", "Reliable", "Ambica"],
        "suffixes": ["Steel & Alloys", "Metal Corp", "Tubes & Pipes", "Steels Pvt Ltd", "Iron & Steel Ind", "Metals & Tubes"],
        "products": [
            ("SS 304 Sheets", "Grade 304 2mm-5mm thickness, 2B Finish", "180 - 260 per kg", "500 kg"),
            ("MS Angles", "25x25mm to 100x100mm, IS 2062 Grade", "55,000 - 65,000 per ton", "1 ton"),
            ("TMT Rebars", "Fe 500D / Fe 550D, 8mm - 32mm dia", "52,000 - 60,000 per ton", "2 tons"),
            ("Aluminum Coils", "1050/1100 Grade, 0.5mm - 3mm thick", "220 - 310 per kg", "300 kg"),
            ("Copper Pipes", "ASTM B280, 1/4 to 2 inch outer dia", "750 - 920 per kg", "100 kg"),
            ("Brass Rods & Hex", "CuZn39Pb3 Grade, 6mm - 50mm dia", "480 - 580 per kg", "200 kg"),
            ("GI Wires", "Hot Dip Galvanized, 8 SWG to 20 SWG", "65 - 85 per kg", "1000 kg")
        ]
    },
    "packaging materials": {
        "prefixes": ["Prime", "EcoPack", "Universal", "Supreme", "PolyPack", "Royal", "Shrinath", "Century", "Canpac", "Avon", "Star", "Reliable", "Packwell"],
        "suffixes": ["Packagings", "Polymers", "Box Manufacturers", "Containers & Drums", "Flexipack Pvt Ltd", "Packaging Solutions"],
        "products": [
            ("Corrugated Boxes", "3-Ply / 5-Ply, 120-180 GSM Kraft Paper", "15 - 85 per box", "500 pcs"),
            ("Bubble Wrap Rolls", "50 Micron, 100m roll, 1m/1.2m width", "450 - 1200 per roll", "50 rolls"),
            ("BOPP Tapes", "48mm x 50m, 40-50 Micron, Transparent/Brown", "18 - 45 per roll", "200 rolls"),
            ("Stretch Film", "20-23 Micron, LLDPE Hand/Machine Roll", "140 - 190 per kg", "100 kg"),
            ("HDPE Drums", "200L Blue, Open Top / Narrow Mouth, UN Certified", "850 - 1400 per drum", "20 drums"),
            ("PP Woven Sacks", "50kg Capacity, Laminated, 60-90 GSM", "12 - 28 per bag", "1000 pcs")
        ]
    },
    "textiles & fabrics": {
        "prefixes": ["Vardhman", "Raymond", "Sangam", "Alok", "Welspun", "Arvind", "Bombay", "Laxmi", "Sitaram", "Sutlej", "Oswal", "Trident", "Grasim"],
        "suffixes": ["Fabrics", "Tex-Fab Ltd", "Textile Mills", "Synthetic & Yarns", "Weaving Works", "Textiles India"],
        "products": [
            ("Cotton Fabric Rolls", "100% Combed Cotton, 40s/60s count, 140 GSM", "95 - 220 per meter", "1000 meters"),
            ("Polyester Yarn", "150D/48F, FDY/DTY, Semi-Dull / Bright", "110 - 160 per kg", "500 kg"),
            ("Denim Fabric", "10 oz - 14 oz, Cotton Spandex Blend, Indigo", "140 - 260 per meter", "500 meters"),
            ("Rayon Printed Fabric", "30x30 count, 14kg quality, 44-58 inch width", "75 - 145 per meter", "1000 meters"),
            ("Jute Fabric & Bags", "14x15 Hessian Jute, 280-330 GSM", "45 - 90 per meter", "500 meters"),
            ("Non-Woven Fabric", "20-100 GSM Polypropylene Spunbond", "110 - 170 per kg", "300 kg")
        ]
    },
    "industrial machinery": {
        "prefixes": ["Kirloskar", "Godrej", "Ace Micromatic", "Batliboi", "L&T", "HMT", "Premier", "Electra", "Precision", "Apex", "Marshall", "Jyoti"],
        "suffixes": ["Machinery Works", "Engineering Solutions", "CNC Tools", "Automation Pvt Ltd", "Heavy Hydraulics", "Industrial Equipment"],
        "products": [
            ("CNC Lathe Machine", "3-Axis, Max turning length 500-1000mm, Fanuc/Siemens", "8,50,000 - 25,000,000 INR", "1 unit"),
            ("Hydraulic Press Machine", "20 Ton to 200 Ton Capacity, C-Frame / H-Frame", "1,80,000 - 12,000,000 INR", "1 unit"),
            ("Industrial Air Compressor", "10 HP - 50 HP Screw Type, 8-10 bar pressure", "1,20,000 - 6,50,000 INR", "1 unit"),
            ("Conveyor Belt System", "Roller / Belt type, SS 304 Frame, 0.5-3 HP Motor", "75,000 - 4,50,000 INR", "1 set"),
            ("Plastic Injection Molding Machine", "80 Ton - 350 Ton Clamping Force, Servo Motor", "12,00,000 - 45,00,000 INR", "1 unit"),
            ("Automatic Pouch Packaging Machine", "Form Fill Seal, 10-100g, 40-60 pouches/min", "1,40,000 - 5,50,000 INR", "1 unit")
        ]
    },
    "electrical components": {
        "prefixes": ["Havells", "Polycab", "Finolex", "Anchor", "Orient", "Crompton", "Schneider", "ABB", "Legrand", "Microtek", "V-Guard", "Luminous"],
        "suffixes": ["Electricals", "Switchgears Ltd", "Cable Industries", "Power Controls", "Electronics & Solar", "Power Systems"],
        "products": [
            ("MCB Circuit Breakers", "Single/Three Pole, 6A - 63A, 10kA breaking cap", "120 - 850 per unit", "50 pcs"),
            ("Armoured Copper Cables", "1.1kV, 4 Core 16 sq mm to 120 sq mm, XLPE", "280 - 1850 per meter", "100 meters"),
            ("LED Driver Modules", "12W - 100W, Constant Current IP65, 85-265V AC", "85 - 450 per unit", "100 pcs"),
            ("Solar On-Grid Inverter", "3kW - 50kW, 3-Phase MPPT, 98.4% efficiency", "28,000 - 2,50,000 INR", "2 units"),
            ("Three Phase Electric Motor", "1 HP to 50 HP, 1440 RPM / 2880 RPM, IE3", "4,500 - 85,000 INR", "5 units"),
            ("Industrial Control Panel", "PLC based, IP55 MS Powder Coated Enclosure", "35,000 - 3,50,000 INR", "1 unit")
        ]
    },
    "chemicals": {
        "prefixes": ["Gujarat Alkalies", "Aarti", "Atul", "Pidilite", "Tata Chemicals", "Deepak Nitrite", "Bodal", "Meghmani", "Chemplast", "Thirumalai", "SRF"],
        "suffixes": ["Chemical Industries", "Organics Ltd", "Fine Chem", "Solvents & Dyes", "Speciality Chemicals", "Chemicals India"],
        "products": [
            ("Industrial Caustic Soda", "Flakes / Lye 99% Purity, Technical Grade, 50kg bag", "38 - 62 per kg", "500 kg"),
            ("Industrial Solvents (Acetone/IPA)", "99.5% Purity, Drum / Tanker Supply, CAS 67-64-1", "75 - 140 per Litre", "200 Litres"),
            ("Organic Pigment Powder", "Phthalo Blue 15:3 / Yellow 12, High Heat Stability", "280 - 650 per kg", "100 kg"),
            ("Polyurethane Resins", "Liquid 2-Component, Shore A/D Hardness", "180 - 340 per kg", "200 kg"),
            ("Commercial Sulfuric Acid", "98% Concentration, Industrial Grade, HDPE Carboy", "12 - 25 per kg", "1000 kg"),
            ("Citric Acid Monohydrate", "Food / Pharma Grade, BP/USP Standard", "85 - 145 per kg", "500 kg")
        ]
    }
}


def try_gemini_generation():
    """Attempt to generate data using google-genai SDK if API key is present."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.startswith("your_"):
        print("[-] GEMINI_API_KEY not set or placeholder. Using Python deterministic generator.")
        return None, None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        print("[+] Attempting dataset generation via Gemini API...")

        # Batch generation logic for suppliers
        suppliers = []
        sup_id_counter = 1001

        for cat in CATEGORIES:
            prompt = f"""Generate 34 realistic Indian B2B suppliers in JSON format for the category '{cat}'.
Each supplier must have:
- company_name (realistic Indian business name)
- city (realistic Indian industrial city)
- state (Indian state)
- category: "{cat}"
- products (list of 2-3 specific product names)
- specifications (technical specs, grades, sizes)
- price_range_inr (e.g. "180 - 250 per kg")
- moq (e.g. "500 kg")
- years_in_business (integer between 2 and 35)
- verified (boolean true/false)

Return ONLY a valid JSON array of objects."""

            # Retry with exponential backoff
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.7,
                        )
                    )
                    data = json.loads(response.text)
                    for item in data:
                        item['supplier_id'] = f"SUP-{sup_id_counter}"
                        item['category'] = cat
                        if isinstance(item.get('products'), list):
                            item['products'] = ", ".join(item['products'])
                        suppliers.append(item)
                        sup_id_counter += 1
                        if len(suppliers) >= 200:
                            break
                    print(f"  [+] Gemini API: Generated {len(data)} suppliers for '{cat}'")
                    time.sleep(1.5)  # Respect free tier rate limits
                    break
                except Exception as e:
                    print(f"  [!] Attempt {attempt+1} failed for category '{cat}': {e}")
                    time.sleep(2 ** attempt)

            if len(suppliers) >= 200:
                break

        if len(suppliers) >= 200:
            suppliers = suppliers[:200]
            print(f"[+] Successfully generated {len(suppliers)} suppliers via Gemini API!")
            return suppliers, None

    except Exception as e:
        print(f"[!] Gemini API generation encountered an error: {e}. Falling back to Python generator.")

    return None, None


def generate_python_suppliers():
    """Generates 200 realistic Indian B2B suppliers using deterministic Python random generation."""
    suppliers = []
    sup_id = 1001

    # Distribute 200 suppliers across 6 categories (~33-34 per category)
    counts = [34, 34, 33, 33, 33, 33]

    for cat_idx, cat in enumerate(CATEGORIES):
        num_to_gen = counts[cat_idx]
        template = SUPPLIER_TEMPLATES[cat]

        for i in range(num_to_gen):
            prefix = random.choice(template["prefixes"])
            suffix = random.choice(template["suffixes"])
            city, state = random.choice(CITIES_STATES)

            # Ensure company name uniqueness
            company_name = f"{prefix} {suffix}"
            if any(s["company_name"] == company_name for s in suppliers):
                company_name = f"{prefix} {city} {suffix}"

            # Pick 2-3 products from the template
            prod_sample = random.sample(template["products"], random.randint(2, 3))
            product_names = [p[0] for p in prod_sample]
            specs_list = [f"{p[0]}: {p[1]}" for p in prod_sample]
            price_range = prod_sample[0][2]
            moq = prod_sample[0][3]

            supplier = {
                "supplier_id": f"SUP-{sup_id}",
                "company_name": company_name,
                "city": city,
                "state": state,
                "category": cat,
                "products": ", ".join(product_names),
                "specifications": "; ".join(specs_list),
                "price_range_inr": price_range,
                "moq": moq,
                "years_in_business": random.randint(3, 32),
                "verified": random.choice([True, True, True, False])  # 75% verified
            }
            suppliers.append(supplier)
            sup_id += 1

    return suppliers


def generate_python_enquiries(suppliers_df):
    """Generates 60 buyer enquiries with ground-truth labels and matching supplier IDs."""
    enquiries = []
    enq_id = 5001

    # 1. Clear enquiries (30 total - 5 per category)
    clear_templates = [
        # Steel & Metals
        ("need 500 kg SS 304 sheets delhi urgent delivery required", "SS 304 Sheets", "500 kg", "Delhi", "high", "steel & metals"),
        ("require 2 tons MS angles 50x50mm in Ludhiana for factory construction", "MS Angles", "2 tons", "Ludhiana", "medium", "steel & metals"),
        ("want rate quote for 10 tons TMT rebars fe500d pune location", "TMT Rebars", "10 tons", "Pune", "medium", "steel & metals"),
        ("looking for 300 kg aluminum coils 1.5mm thick ahmedabad", "Aluminum Coils", "300 kg", "Ahmedabad", "low", "steel & metals"),
        ("urgent requirement copper pipes 1/4 inch 100 kg rajkot", "Copper Pipes", "100 kg", "Rajkot", "high", "steel & metals"),

        # Packaging Materials
        ("need 1000 pcs 5 ply corrugated boxes Surat urgent export pack", "Corrugated Boxes", "1000 pcs", "Surat", "high", "packaging materials"),
        ("plz send price list for 100 rolls bubble wrap 50 micron mumbai", "Bubble Wrap Rolls", "100 rolls", "Mumbai", "medium", "packaging materials"),
        ("require 500 rolls brown BOPP tape 48mm delhi ncr", "BOPP Tapes", "500 rolls", "Delhi", "low", "packaging materials"),
        ("looking for LLDPE stretch film 20 micron 200 kg bhiwandi", "Stretch Film", "200 kg", "Bhiwandi", "medium", "packaging materials"),
        ("need 50 blue HDPE drums 200 Litre open top in Vapi", "HDPE Drums", "50 pcs", "Vapi", "high", "packaging materials"),

        # Textiles & Fabrics
        ("need 2000 meters 100% cotton fabric rolls 60s count tirupur", "Cotton Fabric Rolls", "2000 meters", "Tirupur", "high", "textiles & fabrics"),
        ("plz give best rate for 1 ton polyester yarn 150D Surat bulk order", "Polyester Yarn", "1 ton", "Surat", "medium", "textiles & fabrics"),
        ("require indigo denim fabric 12 oz 1000 meters jaipur urgent", "Denim Fabric", "1000 meters", "Jaipur", "high", "textiles & fabrics"),
        ("want 1500 meters rayon print fabric ahmedabad textile market", "Rayon Printed Fabric", "1500 meters", "Ahmedabad", "medium", "textiles & fabrics"),
        ("need 1000 pcs jute bags 50kg capacity ludhiana mill", "Jute Fabric & Bags", "1000 pcs", "Ludhiana", "low", "textiles & fabrics"),

        # Industrial Machinery
        ("looking for 3-axis CNC lathe machine Fanuc control Pune installation", "CNC Lathe Machine", "1 unit", "Pune", "medium", "industrial machinery"),
        ("require 50 ton hydraulic press machine C frame Coimbatore", "Hydraulic Press Machine", "1 unit", "Coimbatore", "medium", "industrial machinery"),
        ("need 15 HP screw industrial air compressor 10 bar Bangalore urgent", "Industrial Air Compressor", "1 unit", "Bangalore", "high", "industrial machinery"),
        ("want roller conveyor belt system SS 304 10 meters length Thane", "Conveyor Belt System", "1 set", "Thane", "medium", "industrial machinery"),
        ("requirement of automatic pouch packaging machine FFS Rajkot", "Automatic Pouch Packaging Machine", "1 unit", "Rajkot", "high", "industrial machinery"),

        # Electrical Components
        ("need 200 pcs 3 pole MCB 32A 10kA Delhi electrical market", "MCB Circuit Breakers", "200 pcs", "Delhi", "high", "electrical components"),
        ("require 500 meters 4 core 25 sq mm armoured copper cable Noida", "Armoured Copper Cables", "500 meters", "Noida", "medium", "electrical components"),
        ("looking for 50W LED driver modules IP65 500 pcs Vadodara", "LED Driver Modules", "500 pcs", "Vadodara", "medium", "electrical components"),
        ("need 10kW solar on-grid inverter 3 phase Jaipur installation", "Solar On-Grid Inverter", "1 unit", "Jaipur", "low", "electrical components"),
        ("want 7.5 HP three phase electric motor 1440 RPM Mumbai urgent", "Three Phase Electric Motor", "1 unit", "Mumbai", "high", "electrical components"),

        # Chemicals
        ("need 1 ton caustic soda flakes 99% purity Ankleshwar plant", "Industrial Caustic Soda", "1 ton", "Ankleshwar", "high", "chemicals"),
        ("require 1000 Litres Acetone solvent 99.5% drum packing Vapi", "Industrial Solvents (Acetone/IPA)", "1000 Litres", "Vapi", "medium", "chemicals"),
        ("looking for Phthalo Blue pigment powder 200 kg Ahmedabad", "Organic Pigment Powder", "200 kg", "Ahmedabad", "low", "chemicals"),
        ("need polyurethane resin liquid 2 component 500 kg Hyderabad", "Polyurethane Resins", "500 kg", "Hyderabad", "medium", "chemicals"),
        ("urgent requirement sulfuric acid 98% 5 tons carboys Chennai", "Commercial Sulfuric Acid", "5 tons", "Chennai", "high", "chemicals")
    ]

    for raw_text, prod_name, qty, loc, urgency, cat in clear_templates:
        # Find matching suppliers in suppliers_df
        matching_sups = suppliers_df[
            (suppliers_df['category'] == cat) &
            (suppliers_df['products'].str.contains(prod_name, case=False, regex=False))
        ]
        if matching_sups.empty:
            matching_sups = suppliers_df[suppliers_df['category'] == cat]

        # Pick 1-3 matching supplier IDs
        matched_ids = matching_sups['supplier_id'].head(random.randint(2, 3)).tolist()

        enquiries.append({
            "enquiry_id": f"ENQ-{enq_id}",
            "raw_text": raw_text,
            "true_product": prod_name,
            "true_quantity": qty,
            "true_location": loc,
            "true_urgency": urgency,
            "true_quality": "clear",
            "expected_supplier_ids": json.dumps(matched_ids)
        })
        enq_id += 1

    # 2. Vague enquiries (10 total)
    vague_templates = [
        "need steel sheets",
        "want packaging box urgently",
        "send fabric price list",
        "looking for machine for factory",
        "electrical wire required urgent",
        "chemical supplier contact number",
        "need plastic bags bulk",
        "want electric motor rate",
        "metal required in delhi location",
        "buying industrial raw material"
    ]
    for text in vague_templates:
        enquiries.append({
            "enquiry_id": f"ENQ-{enq_id}",
            "raw_text": text,
            "true_product": "unspecified",
            "true_quantity": "unspecified",
            "true_location": "unspecified",
            "true_urgency": "unspecified",
            "true_quality": "vague",
            "expected_supplier_ids": "[]"
        })
        enq_id += 1

    # 3. Spam/Irrelevant enquiries (10 total)
    spam_templates = [
        "Earn Rs 50000 monthly working online from home click http://bit.ly/fakejob",
        "Cheap personal loan approved in 5 mins no CIBIL check call 9876543210",
        "asdfghjkl zxcvbnm 123456 test query",
        "Required urgently telecallers for BPO in Noida salary 18k join immediately",
        "Buy Instagram followers and likes cheap rate 999 INR for 10k guaranteed",
        "Win iPhone 15 Pro Max free spin the wheel now http://spam-win.com",
        "Data entry work available daily payout contact on whatsapp 9123456789",
        "testing query 1234 test test test ignore",
        "Crypto trading signals 100x profit guaranteed join telegram channel @fakecrypto",
        "Hello sir how are you doing today please call me back"
    ]
    for text in spam_templates:
        enquiries.append({
            "enquiry_id": f"ENQ-{enq_id}",
            "raw_text": text,
            "true_product": "none",
            "true_quantity": "N/A",
            "true_location": "N/A",
            "true_urgency": "unspecified",
            "true_quality": "spam",
            "expected_supplier_ids": "[]"
        })
        enq_id += 1

    # 4. Out-of-catalogue enquiries (10 total)
    out_of_cat_templates = [
        ("need 500 boxes N95 surgical facemasks in Mumbai urgent", "N95 Surgical Masks", "500 boxes", "Mumbai", "high"),
        ("require 50 crates organic Alphonso mangoes for export in Pune", "Organic Alphonso Mangoes", "50 crates", "Pune", "medium"),
        ("looking for Oracle ERP Cloud software user license 25 seats Bangalore", "Oracle ERP Licenses", "25 seats", "Bangalore", "low"),
        ("need hydraulic brake pads for Hyundai Creta car 100 sets Delhi", "Automotive Brake Pads", "100 sets", "Delhi", "high"),
        ("requirement of 200 units wooden office dining tables in Hyderabad", "Wooden Office Tables", "200 units", "Hyderabad", "medium"),
        ("need 5000 litres packaged mineral drinking water 20L jars Gurgaon", "Packaged Mineral Water", "5000 Litres", "Gurgaon", "high"),
        ("want to buy 50 sets hospital ICU patient beds in Chennai", "Hospital ICU Beds", "50 sets", "Chennai", "high"),
        ("require solar rooftop panel installation 50kW residential Jaipur", "Solar Rooftop Panel Installation", "50 kW", "Jaipur", "medium"),
        ("looking for diesel generator set 125 kVA Silent Soundproof Kolkata", "Diesel Generator 125 kVA", "1 set", "Kolkata", "high"),
        ("need 1000 kg frozen chicken breast meat in Ahmedabad", "Frozen Chicken Meat", "1000 kg", "Ahmedabad", "medium")
    ]
    for text, prod, qty, loc, urgency in out_of_cat_templates:
        enquiries.append({
            "enquiry_id": f"ENQ-{enq_id}",
            "raw_text": text,
            "true_product": prod,
            "true_quantity": qty,
            "true_location": loc,
            "true_urgency": urgency,
            "true_quality": "clear",
            "expected_supplier_ids": "[]"
        })
        enq_id += 1

    return enquiries


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    print("==================================================")
    print(" SmartMatch Dataset Generation Pipeline")
    print("==================================================")

    # Step 1: Try Gemini API, fallback to Python generator
    suppliers_data, _ = try_gemini_generation()

    if not suppliers_data:
        print("[+] Generating suppliers via deterministic Python engine...")
        suppliers_data = generate_python_suppliers()

    suppliers_df = pd.DataFrame(suppliers_data)
    suppliers_df.to_csv(SUPPLIERS_PATH, index=False)
    print(f"[OK] Saved {len(suppliers_df)} suppliers to {os.path.abspath(SUPPLIERS_PATH)}")

    # Step 2: Generate enquiries dataset
    print("[+] Generating ground-truth buyer enquiries...")
    enquiries_data = generate_python_enquiries(suppliers_df)
    enquiries_df = pd.DataFrame(enquiries_data)
    enquiries_df.to_csv(ENQUIRIES_PATH, index=False)
    print(f"[OK] Saved {len(enquiries_df)} buyer enquiries to {os.path.abspath(ENQUIRIES_PATH)}")

    # Step 3: Print Data Summaries & Sample Rows
    print("\n" + "="*50)
    print(" DATASET SUMMARY STATISTICS")
    print("="*50)

    print("\n--- Suppliers per Category ---")
    cat_counts = suppliers_df['category'].value_counts()
    for cat, count in cat_counts.items():
        print(f"  * {cat.title():<25}: {count} suppliers")

    print("\n--- Enquiries per Quality Label ---")
    quality_counts = enquiries_df['true_quality'].value_counts()
    for q_label, count in quality_counts.items():
        print(f"  * {q_label.title():<15}: {count} enquiries")

    # Count out-of-catalogue queries
    out_of_cat_count = len(enquiries_df[(enquiries_df['true_quality'] == 'clear') & (enquiries_df['expected_supplier_ids'] == '[]')])
    print(f"  * Out-of-Catalogue (No Match): {out_of_cat_count} enquiries")

    print("\n" + "="*50)
    print(" 5 RANDOM SAMPLE ROWS: SUPPLIERS (data/suppliers.csv)")
    print("="*50)
    sample_suppliers = suppliers_df.sample(5, random_state=SEED)[
        ['supplier_id', 'company_name', 'city', 'category', 'products', 'moq', 'verified']
    ]
    print(sample_suppliers.to_string(index=False))

    print("\n" + "="*50)
    print(" 5 RANDOM SAMPLE ROWS: ENQUIRIES (eval/enquiries.csv)")
    print("="*50)
    sample_enquiries = enquiries_df.sample(5, random_state=SEED)[
        ['enquiry_id', 'raw_text', 'true_product', 'true_quality', 'expected_supplier_ids']
    ]
    print(sample_enquiries.to_string(index=False))
    print("\n[OK] Data generation completed successfully.")

if __name__ == "__main__":
    main()
