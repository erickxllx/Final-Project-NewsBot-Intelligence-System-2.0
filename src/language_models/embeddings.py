# ============================================
# EmbeddingModel – SentenceTransformers Wrapper
# ============================================

from sentence_transformers import SentenceTransformer, util


class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Loads a lightweight embedding model suitable for web apps.
        """
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print("Error loading embedding model:", e)
            raise e

    # ---------------------------------------------------------
    # Encode single text → vector
    # ---------------------------------------------------------
    def embed(self, text: str):
        if not isinstance(text, str):
            return None
        return self.model.encode(text, convert_to_tensor=True)

    # ---------------------------------------------------------
    # Batch encoding for lists of texts
    # ---------------------------------------------------------
    def embed_batch(self, texts: list):
        return self.model.encode(texts, convert_to_tensor=True)

    # ---------------------------------------------------------
    # Semantic similarity between 2 texts
    # ---------------------------------------------------------
    def similarity(self, text_a: str, text_b: str) -> float:
        vec1 = self.embed(text_a)
        vec2 = self.embed(text_b)

        score = util.cos_sim(vec1, vec2).item()
        return round(float(score), 4)

    # ---------------------------------------------------------
    # Utility for vector distance (optional)
    # ---------------------------------------------------------
    def distance(self, text_a: str, text_b: str) -> float:
        vec1 = self.embed(text_a)
        vec2 = self.embed(text_b)

        score = util.pytorch_cos_sim(vec1, vec2).item()
        return 1 - score
