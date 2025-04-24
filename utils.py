import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))

# Load stop words
stop_words = set(stopwords.words("english"))


def mask_pii(text: str) -> dict:
    """
    Detects and masks personally identifiable information (PII) and PCI data
    in the input text using context-aware rules and regex patterns.

    Args:
        text (str): The input email text.

    Returns:
        dict: A dictionary containing the masked text and the list of detected entities.
    """
    entities = []
    masked_text = text

    # 1. Context-aware name masking
    name_pattern = re.compile(
        r"(my name is|my full name is)\s+([A-Z][a-z]+(?: [A-Z][a-z]+){1,2})",
        flags=re.IGNORECASE
    )
    for match in name_pattern.finditer(masked_text):
        full_name = match.group(2)
        start = masked_text.find(full_name)
        end = start + len(full_name)
        entities.append({
            "position": [start, end],
            "classification": "full_name",
            "entity": full_name
        })
        masked_text = masked_text.replace(full_name, "[full_name]")

    # 2. Regex-based masking
    pii_patterns = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone_number": (
            r"\b(\+91[-\s]?|91[-\s]?|0)?[789]\d{9}\b|"                         # Indian numbers
            r"(\+?\d{1,3}[-.\s]?)?(\(?\d{1,4}\)?[-.\s]?){2,5}\d{2,4}"          # Global formats
        ),
        "dob": r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",
        "aadhar_num": r"\b\d{4} \d{4} \d{4}\b",
        "credit_debit_no": r"\b(?:\d[ -]*?){13,16}\b",
        "cvv_no": r"\b\d{3}\b",
        "expiry_no": r"\b(0[1-9]|1[0-2])/[0-9]{2}\b"
    }

    for label, pattern in pii_patterns.items():
        matches = list(re.finditer(pattern, masked_text))
        for match in matches:
            original = match.group()
            start, end = match.span()
            # Avoid double-masking or corrupt replacements
            if f"[{label}]" in original or "[" in original or "]" in original:
                continue
            entities.append({
                "position": [start, end],
                "classification": label,
                "entity": original
            })
            masked_text = masked_text.replace(original, f"[{label}]")

    return {"masked_text": masked_text, "entities": entities}


def preprocess_text(text: str) -> str:
    """
    Preprocesses text for classification.

    Steps:
    - Lowercasing
    - Tokenization
    - Removal of stopwords and non-alphanumeric tokens

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned and preprocessed text.
    """
    text = text.lower()
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(filtered_tokens)


def demask_pii(masked_text: str, entities: list) -> str:
    """
    Replaces PII placeholders with original values.

    Args:
        masked_text (str): The masked email.
        entities (list): List of entity dictionaries containing original values.

    Returns:
        str: The demasked text.
    """
    demasked_text = masked_text
    for entity in entities:
        tag = f"[{entity['classification']}]"
        demasked_text = demasked_text.replace(tag, entity["entity"], 1)
    return demasked_text