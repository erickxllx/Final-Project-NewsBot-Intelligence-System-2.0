from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


class NERExtractor:
    """
    Named Entity Recognition using a lightweight Transformer model.
    Works in Streamlit Cloud (does NOT require SpaCy).
    Supports English and Spanish.
    """

    def __init__(self, language: str = "en"):
        self.language = language
        
        # Multilingual NER model (works for EN + ES)
        self.model_name = "Davlan/xlm-roberta-base-ner-hrl"

        try:
            self.nlp = pipeline(
                "ner",
                model=AutoModelForTokenClassification.from_pretrained(self.model_name),
                tokenizer=AutoTokenizer.from_pretrained(self.model_name),
                aggregation_strategy="max"
            )
        except Exception as e:
            print("Error loading NER model:", e)
            raise e

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using a multilingual NER model.
        """

        if not text or not isinstance(text, str):
            return []

        results = self.nlp(text)

        entities = []
        for ent in results:
            entities.append(
                {
                    "text": ent["word"],
                    "label": ent["entity_group"],
                    "score": round(float(ent["score"]), 4),
                    "start": ent.get("start", None),
                    "end": ent.get("end", None)
                }
            )

        return entities
