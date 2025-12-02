from transformers import pipeline
from src.multilingual.translator import Translator


class Summarizer:
    def __init__(self):
        # Modelo fuerte para resumen (inglés)
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )
        self.translator = Translator()

    def summarize(self, text: str, max_length: int = 180, min_length: int = 60) -> str:
        text = text.strip()
        if not text:
            return "Empty text, nothing to summarize."

        # Detectar idioma original
        lang = self.translator.detect_language(text)
        original_lang = lang

        # Si el texto está en español, lo traducimos a inglés antes de resumir
        if lang == "es":
            text_en = self.translator.translate_to_english(text)
        else:
            text_en = text

        # Cortar por si es muy largo (límite de tokens)
        if len(text_en) > 4000:
            text_en = text_en[:4000]

        result = self.summarizer(
            text_en,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )

        summary_en = result[0]["summary_text"]

        # Si el original era en español, regresamos el resumen en español
        if original_lang == "es":
            summary = self.translator.translate_to_spanish(summary_en)
        else:
            summary = summary_en

        return summary
