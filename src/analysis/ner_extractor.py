from typing import List, Dict, Any
import spacy


class NERExtractor:
    """
    Named Entity Recognition using spaCy.
    Extracts key entities like PERSON, ORG, GPE, MONEY, DATE, etc.
    """

    def __init__(self, language: str = "en"):
        self.language = language
        self._load_spacy_model()

    def _load_spacy_model(self):
        try:
            if self.language == "en":
                self.nlp = spacy.load("en_core_web_sm")
            elif self.language == "es":
                self.nlp = spacy.load("es_core_news_sm")
            else:
                raise ValueError(f"Unsupported language: {self.language}")
        except OSError:
            raise OSError(
                f"Please install the SpaCy model:\n"
                f"python -m spacy download {self.language}_core_web_sm"
            )

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.

        Returns a list of:
        {
            "text": "Apple",
            "label": "ORG",
            "description": "Companies, agencies, institutions, etc."
        }
        """
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "description": spacy.explain(ent.label_) or "",
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                }
            )

        return entities
