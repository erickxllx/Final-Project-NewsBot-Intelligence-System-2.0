from typing import Dict

class Translator:
    """
    Lightweight translator using simple dictionary rules.
    (Compatible with Streamlit Cloud, without external APIs.)
    """

    def __init__(self):
        pass

    # ==============================
    # Detect language
    # ==============================
    def detect_language(self, text: str) -> str:
        """
        VERY simple language detection.
        """
        spanish_tokens = ["el", "la", "que", "de", "y", "un", "una", "es", "pero", "para"]
        english_tokens = ["the", "and", "is", "of", "to", "in", "but", "with"]

        t = text.lower()

        score_es = sum(t.count(w) for w in spanish_tokens)
        score_en = sum(t.count(w) for w in english_tokens)

        if score_es > score_en:
            return "es"
        return "en"

    # ==============================
    # Translate Spanish → English
    # ==============================
    def translate_to_english(self, text: str) -> str:
        """
        Simplified translation (mock) for demo purposes.
        """
        if text is None or text.strip() == "":
            return "No text provided."

        translations = {
            "hola": "hello",
            "mundo": "world",
            "gracias": "thank you",
            "como estas": "how are you",
            "adios": "goodbye",
            "buenos dias": "good morning",
            "buenas noches": "good night",
        }

        output = text.lower()
        for es, en in translations.items():
            output = output.replace(es, en)

        return output.capitalize()

    # ==============================
    # Translate English → Spanish
    # ==============================
    def translate_to_spanish(self, text: str) -> str:
        if text is None or text.strip() == "":
            return "No text provided."

        translations = {
            "hello": "hola",
            "world": "mundo",
            "thank you": "gracias",
            "goodbye": "adios",
            "good night": "buenas noches",
            "good morning": "buenos dias",
        }

        output = text.lower()
        for en, es in translations.items():
            output = output.replace(en, es)

        return output.capitalize()
