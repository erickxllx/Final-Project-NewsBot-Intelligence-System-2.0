from transformers import pipeline
from langdetect import detect, LangDetectException


class Translator:
    def __init__(self):
        # Modelos MarianMT especializados en es<->en
        self.es_to_en = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-es-en"
        )
        self.en_to_es = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-en-es"
        )

    def detect_language(self, text: str) -> str:
        """
        Detecta idioma (es/en) con langdetect y un fallback sencillo.
        """
        text = text.strip()
        if not text:
            return "unknown"

        try:
            lang = detect(text)
        except LangDetectException:
            lang = "unknown"

        # Si langdetect duda, usamos un heurístico simple
        if lang not in ["es", "en"]:
            tl = text.lower()
            spanish_indicators = [" el ", " la ", " de ", " que ", " y ", " en ", " un ", " una ", "los ", "las "]
            score = sum(ind in tl for ind in spanish_indicators)
            return "es" if score >= 2 else "en"

        return lang

    def translate_to_english(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "Empty text, nothing to translate."

        result = self.es_to_en(text[:4000])
        return result[0]["translation_text"]

    def translate_to_spanish(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "Empty text, nothing to translate."

        result = self.en_to_es(text[:4000])
        return result[0]["translation_text"]
