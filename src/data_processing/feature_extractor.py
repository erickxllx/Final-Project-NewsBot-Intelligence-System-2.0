from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

class FeatureExtractor:
    def __init__(self, tfidf_max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=tfidf_max_features,
            stop_words="english"
        )
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def fit_tfidf(self, corpus):
        """Fit the TF-IDF vectorizer to a corpus."""
        return self.vectorizer.fit_transform(corpus)

    def transform_tfidf(self, texts):
        """Transform new texts using the fitted TF-IDF model."""
        return self.vectorizer.transform(texts)

    def encode_embeddings(self, texts):
        """Generate dense embeddings using SentenceTransformers."""
        if isinstance(texts, str):
            texts = [texts]
        return self.embedding_model.encode(texts)

    def get_features(self, texts):
        """
        Generate both TF-IDF and embedding features.
        Perfect for classification, clustering, or semantic search.
        """
        tfidf_features = self.vectorizer.transform(texts)
        embedding_features = self.encode_embeddings(texts)

        return {
            "tfidf": tfidf_features,
            "embeddings": embedding_features
        }
