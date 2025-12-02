from transformers import pipeline
from langdetect import detect, LangDetectException


class Translator:
    def __init__(self):
        # MarianMT models
        self.es_to_en = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-es-en"
        )
        self.en_to_es = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-en-es"
        )

    def detect_language(self, text: str) -> str:
        try:
            lang = detect(text)
            # langdetect devuelve 'es', 'en', etc.
            return lang
        except LangDetectException:
            return "unknown"

    def translate_to_english(self, text: str):
        if not text.strip():
            return "Empty text, nothing to translate."
        result = self.es_to_en(text[:4000])
        return result[0]["translation_text"]

    def translate_to_spanish(self, text: str):
        if not text.strip():
            return "Empty text, nothing to translate."
        result = self.en_to_es(text[:4000])
        return result[0]["translation_text"]
