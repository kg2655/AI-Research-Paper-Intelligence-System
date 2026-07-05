import re
import unicodedata
import pdfplumber

LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}

def clean_text(text: str) -> str:
    """Clean the extracted text by resolving ligatures and fixing spacing."""
    if not text:
        return ""
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKC", text)
    # Remove hyphenation at the end of lines
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse multiple newlines
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using pdfplumber for better layout retention."""
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text preserving layout reasonably well
            page_text = page.extract_text()
            if page_text:
                full_text.append(page_text)
    
    raw_text = "\n".join(full_text)
    return clean_text(raw_text)

HEADINGS = ["abstract", "introduction", "related work", "background", "methodology",
            "method", "methods", "approach", "dataset", "experiments", "results",
            "evaluation", "discussion", "conclusion", "references"]

HRE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?)?\s*(" + "|".join(HEADINGS) + r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)

def detect_sections(text: str) -> dict:
    """Extract sections from the document based on common headings."""
    ms = list(HRE.finditer(text))
    out = {}
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        body = text[m.end():end].strip()
        if body:
            out[m.group(1).lower()] = body
    return out

