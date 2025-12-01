import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TextGenerator:
    """
    Lightweight text generator + semantic similarity.
    TF-IDF based (Streamlit Cloud compatible).
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    # ======================================
    # Chat response
    # ======================================
    def generate_response(self, query: str) -> str:
        return f"I processed your request: '{query}'. Let me know what you want to analyze."

    # ======================================
    # Similarity
    # ======================================
    def compare_similarity(self, text1: str, text2: str):
        documents = [text1, text2]

        try:
            tfidf = self.vectorizer.fit_transform(documents)
            vec1 = tfidf[0].toarray()[0]
            vec2 = tfidf[1].toarray()[0]
        except:
            return {"similarity_score": 0.0, "interpretation": "Not enough text."}

        # cosine similarity
        dot = np.dot(vec1, vec2)
        norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        score = dot / norm if norm != 0 else 0
        score = float(round(score, 4))

        return {
            "similarity_score": score,
            "interpretation": self._interpret(score),
        }

    def _interpret(self, score):
        if score > 0.85:
            return "Highly similar"
        if score > 0.60:
            return "Moderately similar"
        if score > 0.40:
            return "Somewhat related"
        return "Not similar"
