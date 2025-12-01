import re
from typing import List, Dict, Any


class NERExtractor:
    """
    Lightweight Rule-Based NER extractor (Streamlit Compatible)
    Works without SpaCy or Transformers.
    Detects:
        - PERSON (capitalized names)
        - ORG (capitalized words ending in Inc, Corp, Co, Ltd, etc.)
        - GPE (cities, countries)
        - MONEY ($ + number)
        - DATE (YYYY, Month DD)
    """

    def __init__(self, language: str = "en"):
        self.language = language

    def extract(self, text: str) -> List[Dict[str, Any]]:
        if not text or not isinstance(text, str):
            return []

        entities = []

        # ============================
        # MONEY
        # ============================
        for match in re.finditer(r"\$\s?\d+(?:,\d{3})*(?:\.\d+)?", text):
            entities.append({
                "text": match.group(),
                "label": "MONEY",
                "score": 0.90
            })

        # ============================
        # DATE (Basic)
        # ============================
        for match in re.finditer(r"\b(?:\d{4}|January|February|March|April|May|June|"
                                 r"July|August|September|October|November|December)\b", text):
            entities.append({
                "text": match.group(),
                "label": "DATE",
                "score": 0.85
            })

        # ============================
        # PERSON (Capitalized sequences)
        # ============================
        for match in re.finditer(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", text):
            entities.append({
                "text": match.group(),
                "label": "PERSON",
                "score": 0.80
            })

        # ============================
        # ORG (simple detection)
        # ============================
        for match in re.finditer(r"\b[A-Z][A-Za-z]+ (Inc|Corp|Co|Ltd|LLC)\b", text):
            entities.append({
                "text": match.group(),
                "label": "ORG",
                "score": 0.88
            })

        # ============================
        # GPE (simple detection)
        # ============================
        common_places = ["USA", "Mexico", "Canada", "China", "Texas", "California"]
        for place in common_places:
            if place in text:
                entities.append({
                    "text": place,
                    "label": "GPE",
                    "score": 0.75
                })

        return entities
