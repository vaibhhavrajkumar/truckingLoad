import datetime
import os
import re
from typing import Literal, Optional, Tuple
import fitz  # PyMuPDF
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

# ==========================================
# 1. Schema Definitions (Pydantic)
# ==========================================

class Location(BaseModel):
    city: str = Field(..., description="City name")
    state: str = Field(..., description="Two-letter state code (e.g., TX, CA)")
    zip: Optional[str] = Field(None, description="5-digit or 9-digit ZIP code")


class LLMRateConfirmationSchema(BaseModel):
    """Raw extraction model enforced directly during LLM generation."""
    load_id: Optional[str] = Field(None, description="Load number, reference, or order ID")
    origin: Location
    destination: Location
    pickup_date: Optional[str] = Field(
        None, description="Pickup date as extracted from document"
    )
    delivery_date: Optional[str] = Field(
        None, description="Delivery date as extracted from document"
    )
    equipment_type: Optional[Literal["van", "reefer", "flatbed", "other"]] = Field(
        None, description="Trailer type"
    )
    line_haul_rate: Optional[float] = Field(None, description="Base line haul rate amount")
    fuel_surcharge: Optional[float] = Field(None, description="Fuel surcharge amount (FSC)")
    total_rate: Optional[float] = Field(None, description="Total agreed gross rate")
    weight_lbs: Optional[float] = Field(None, description="Cargo weight in pounds")
    commodity: Optional[str] = Field(None, description="Freight or cargo description")


class FinalRateConfirmation(BaseModel):
    """Final output schema matching problem requirements."""
    load_id: Optional[str]
    origin: Location
    destination: Location
    pickup_date: Optional[str]
    delivery_date: Optional[str]
    equipment_type: Optional[Literal["van", "reefer", "flatbed", "other"]]
    line_haul_rate: Optional[float]
    fuel_surcharge: Optional[float]
    total_rate: Optional[float]
    weight_lbs: Optional[float]
    commodity: Optional[str]
    confidence: Literal["high", "medium", "low"]
    review_reasons: list[str] = Field(default_factory=list)


# ==========================================
# 2. PyMuPDF PDF Text Extractor
# ==========================================

def extract_text_with_pymupdf(pdf_path: str) -> Tuple[str, bool]:
    """
    Extracts text from a PDF file using PyMuPDF.
    Returns: (extracted_text, is_image_scan)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    doc = fitz.open(pdf_path)
    full_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 'blocks' mode extracts text chunks preserving visual reading order
        blocks = page.get_text("blocks")
        # Sort blocks by vertical position (Y-axis), then horizontal (X-axis)
        blocks.sort(key=lambda b: (b[1], b[0]))
        
        page_text = "\n".join([b[4].strip() for b in blocks if b[4].strip()])
        if page_text:
            full_text.append(f"--- PAGE {page_num + 1} ---\n" + page_text)

    doc.close()

    raw_text = "\n\n".join(full_text).strip()
    
    # Check if text layer is empty or extremely low (indicates flat image scan)
    is_image_scan = len(raw_text) < 30
    return raw_text, is_image_scan


# ==========================================
# 3. Validation Logic & Confidence Scoring
# ==========================================

def parse_and_validate_date(date_str: Optional[str]) -> Tuple[Optional[str], bool]:
    """Parses date string to YYYY-MM-DD and detects ambiguous numeric formats (e.g. 3/4/26)."""
    if not date_str:
        return None, False

    is_ambiguous = False
    cleaned = date_str.strip()

    # Regex check for formats like M/D/YY or D/M/YY where both numbers <= 12
    ambiguous_match = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$", cleaned)
    if ambiguous_match:
        p1, p2, year = map(int, ambiguous_match.groups())
        if p1 <= 12 and p2 <= 12 and p1 != p2:
            is_ambiguous = True
        
        # Standardize 2-digit year to 4-digit year assuming 2000s
        if year < 100:
            year += 2000
        
        # Default assumption: US Standard MM/DD/YYYY
        try:
            formatted_date = f"{year:04d}-{p1:02d}-{p2:02d}"
            return formatted_date, is_ambiguous
        except Exception:
            pass

    # Standard ISO parse attempt YYYY-MM-DD
    iso_match = re.match(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$", cleaned)
    if iso_match:
        y, m, d = map(int, iso_match.groups())
        return f"{y:04d}-{m:02d}-{d:02d}", False

    return cleaned, True


def run_pipeline(pdf_path: str, client: instructor.Instructor) -> FinalRateConfirmation:
    """Executes the extraction and validation pipeline for a PDF."""
    
    # Step 1: Extract Text using PyMuPDF
    raw_text, is_image_scan = extract_text_with_pymupdf(pdf_path)

    # Fast-fail if the PDF is an image scan without embedded text layer
    if is_image_scan:
        return FinalRateConfirmation(
            load_id=None,
            origin=Location(city="UNKNOWN", state="XX", zip=None),
            destination=Location(city="UNKNOWN", state="XX", zip=None),
            pickup_date=None,
            delivery_date=None,
            equipment_type=None,
            line_haul_rate=None,
            fuel_surcharge=None,
            total_rate=None,
            weight_lbs=None,
            commodity=None,
            confidence="low",
            review_reasons=["PDF contains no readable text layer (Scanned document/Image)."],
        )

    # Step 2: Instructor LLM Extraction
    system_prompt = """
    You are an expert freight logistics parser extracting structured data from Rate Confirmation documents.
    
    Extraction Guidelines:
    1. Extract numerical monetary values as clean floats (strip '$' and commas).
    2. Normalize equipment_type to: 'van', 'reefer', 'flatbed', or 'other'.
    3. Extract city and state explicitly for origin and destination.
    """

    try:
        extracted: LLMRateConfirmationSchema = client.chat.completions.create(
            model="gpt-4o",
            response_model=LLMRateConfirmationSchema,
            max_retries=3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract rate confirmation data:\n\n{raw_text}"},
            ],
        )
    except Exception as e:
        return FinalRateConfirmation(
            load_id=None,
            origin=Location(city="UNKNOWN", state="XX", zip=None),
            destination=Location(city="UNKNOWN", state="XX", zip=None),
            pickup_date=None,
            delivery_date=None,
            equipment_type=None,
            line_haul_rate=None,
            fuel_surcharge=None,
            total_rate=None,
            weight_lbs=None,
            commodity=None,
            confidence="low",
            review_reasons=[f"LLM Schema extraction failed: {str(e)}"],
        )

    # Step 3: Date Normalization & Ambiguity Detection
    pickup_date, p_ambiguous = parse_and_validate_date(extracted.pickup_date)
    delivery_date, d_ambiguous = parse_and_validate_date(extracted.delivery_date)
    date_ambiguous = p_ambiguous or d_ambiguous

    # Step 4: Line Haul + Fuel vs Total Math Validation
    line_haul = extracted.line_haul_rate or 0.0
    fuel = extracted.fuel_surcharge or 0.0
    total = extracted.total_rate

    math_mismatch = False
    if total is not None and (line_haul > 0 or fuel > 0):
        # 1-cent rounding threshold tolerance
        if abs((line_haul + fuel) - total) > 0.01:
            math_mismatch = True

    # Step 5: Real Confidence Logic Assignment
    review_reasons = []

    # Critical Field Check
    has_critical_missing = False
    if not extracted.load_id:
        review_reasons.append("Missing load_id")
        has_critical_missing = True
    if not extracted.origin.city or not extracted.origin.state:
        review_reasons.append("Incomplete origin address")
        has_critical_missing = True
    if not extracted.destination.city or not extracted.destination.state:
        review_reasons.append("Incomplete destination address")
        has_critical_missing = True
    if total is None:
        review_reasons.append("Missing total_rate")
        has_critical_missing = True

    if math_mismatch:
        review_reasons.append(f"Math Mismatch: line_haul ({line_haul}) + fuel ({fuel}) != total ({total})")
    if date_ambiguous:
        review_reasons.append("Ambiguous date format detected (e.g., 3/4/26)")

    # Assign Confidence Rating
    if has_critical_missing or math_mismatch:
        confidence = "low"
    elif date_ambiguous or extracted.equipment_type is None or extracted.weight_lbs is None:
        confidence = "medium"
    else:
        confidence = "high"

    return FinalRateConfirmation(
        load_id=extracted.load_id,
        origin=extracted.origin,
        destination=extracted.destination,
        pickup_date=pickup_date,
        delivery_date=delivery_date,
        equipment_type=extracted.equipment_type,
        line_haul_rate=extracted.line_haul_rate,
        fuel_surcharge=extracted.fuel_surcharge,
        total_rate=extracted.total_rate,
        weight_lbs=extracted.weight_lbs,
        commodity=extracted.commodity,
        confidence=confidence,
        review_reasons=review_reasons,
    )


# ==========================================
# 4. Example Pipeline Execution
# ==========================================

if __name__ == "__main__":
    # Initialize Instructor OpenAI Client
    oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    client = instructor.from_openai(oai_client)

    # Path to sample rate confirmation PDF
    #sample_pdf = "samples/sample_rate_confirmation.pdf"
    sample_pdf = "SkilltestAI.pdf"

    try:
        output = run_pipeline(sample_pdf, client)
        print(output.model_dump_json(indent=2))
    except Exception as err:
        print(f"Error running pipeline: {err}")
