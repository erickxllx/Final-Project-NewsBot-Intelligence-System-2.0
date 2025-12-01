import re

class TextPreprocessor:
    """
    Lightweight text preprocessor for Streamlit.
    Avoids SpaCy model downloads (which Streamlit Cloud blocks).
    """

    def __init__(self, language="en"):
        self.language = language

    def clean_text(self, text: str) -> str:
        """
        Very lightweight text cleaning.
        """
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9áéíóúñü¿?¡!., ]", "", text)
        return text.strip()

    # Compatibility wrapper for old code
    def preprocess(self, text: str) -> str:
        return self.clean_text(text)
