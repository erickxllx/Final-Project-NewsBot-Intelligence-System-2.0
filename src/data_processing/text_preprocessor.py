import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

class TextPreprocessor:
    def __init__(self, language="en"):
        self.language = language
        self._load_spacy_model()

    def _load_spacy_model(self):
        """Load SpaCy model based on language."""
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

    def clean_text(self, text):
        """Remove noise, normalize spacing and punctuation."""
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s.,!?']", "", text)
        return text.strip()

    def remove_stopwords(self, text):
        """Remove stopwords using SpaCy."""
        doc = self.nlp(text)
        tokens = [token.text for token in doc if token.text not in STOP_WORDS]
        return " ".join(tokens)

    def lemmatize(self, text):
        """Reduce words to their base form."""
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])

    def preprocess(self, text):
        """Full preprocessing pipeline."""
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize(text)
        return text
