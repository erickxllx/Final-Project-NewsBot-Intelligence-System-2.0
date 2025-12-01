from sentence_transformers import SentenceTransformer, util


class TextGenerator:
    """
    Handles:
    - Chat-style responses
    - Semantic similarity
    - Text rewriting / expansion
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    # ============================
    # Chat-style response generator
    # ============================
    def generate_response(self, user_query: str) -> str:
        """
        Basic conversational response.
        (You can expand this later for bonus points.)
        """
        return f"I processed your request: '{user_query}'. Please provide more details so I can help you better."

    # ============================
    # Semantic Similarity
    # ============================
    def compare_similarity(self, text_a: str, text_b: str) -> dict:
        """
        Computes similarity between two texts using cosine similarity.
        """
        embedding1 = self.model.encode(text_a, convert_to_tensor=True)
        embedding2 = self.model.encode(text_b, convert_to_tensor=True)

        similarity_score = util.cos_sim(embedding1, embedding2).item()
        similarity_score = round(float(similarity_score), 4)

        return {
            "similarity_score": similarity_score,
            "interpretation": self._interpret_similarity(similarity_score)
        }

    def _interpret_similarity(self, score: float) -> str:
        """
        Human-readable interpretation.
        """

        if score > 0.85:
            return "Highly similar — same topic or meaning."
        elif score > 0.60:
            return "Moderately similar — related but not identical."
        elif score > 0.40:
            return "Somewhat related — partially similar."
        else:
            return "Not similar — different meaning."

    # ============================
    # Basic rewriting (optional)
    # ============================
    def rewrite_text(self, text: str) -> str:
        return f"Rewritten version: {text}"

    # ============================
    # Basic expansion (optional)
    # ============================
    def expand_text(self, text: str) -> str:
        return f"Expanded explanation: {text}"
