import spacy


class NERExtractor:
    def __init__(self):
        # Cargamos modelos en y es
        self.nlp_en = spacy.load("en_core_web_sm")
        self.nlp_es = spacy.load("es_core_news_sm")

    def _detect_lang_simple(self, text: str) -> str:
        text_lower = text.lower()
        spanish_clues = [" el ", " la ", " de ", " que ", " y ", " en ", " un ", " una "]
        score = sum(1 for w in spanish_clues if w in text_lower)
        return "es" if score >= 2 else "en"

    def extract(self, text: str):
        """
        Returns list of entities: [{'text': 'USA', 'label': 'GPE'}, ...]
        """
        if not text.strip():
            return {"error": "Empty text for NER."}

        lang = self._detect_lang_simple(text)
        nlp = self.nlp_es if lang == "es" else self.nlp_en

        doc = nlp(text)
        entities = [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
        ]
        return {
            "language_detected": lang,
            "entities": entities
        }
