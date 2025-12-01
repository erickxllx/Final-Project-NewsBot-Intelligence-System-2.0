from typing import List, Dict, Any

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

from src.data_processing.text_preprocessor import TextPreprocessor


class TopicModeler:
    """
    Topic modeling component supporting both LDA and NMF.
    Uses bag-of-words representation via CountVectorizer.
    """

    def __init__(
        self,
        n_topics: int = 10,
        max_features: int = 5000,
        language: str = "en",
    ):
        self.n_topics = n_topics
        self.max_features = max_features
        self.language = language

        self.preprocessor = TextPreprocessor(language=language)
        self.vectorizer: CountVectorizer | None = None
        self.lda_model: LatentDirichletAllocation | None = None
        self.nmf_model: NMF | None = None

    def _build_vectorizer(self):
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            stop_words="english",
        )

    def fit(self, documents: List[str]):
        """
        Fit both LDA and NMF models on a corpus of documents.
        """
        if self.vectorizer is None:
            self._build_vectorizer()

        processed_docs = [self.preprocessor.preprocess(doc) for doc in documents]
        doc_term_matrix = self.vectorizer.fit_transform(processed_docs)

        # LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            learning_method="batch",
            random_state=42,
        )
        self.lda_model.fit(doc_term_matrix)

        # NMF model
        self.nmf_model = NMF(
            n_components=self.n_topics,
            random_state=42,
            init="nndsvd",
        )
        self.nmf_model.fit(doc_term_matrix)

    def _get_top_words(self, model, feature_names, n_top_words: int = 10) -> List[List[str]]:
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            top_features = [
                feature_names[i]
                for i in topic.argsort()[:-n_top_words - 1:-1]
            ]
            topics.append(top_features)
        return topics

    def get_topics(self, n_top_words: int = 10) -> Dict[str, Any]:
        """
        Return top words for each topic for both LDA and NMF.
        """
        if self.vectorizer is None or self.lda_model is None or self.nmf_model is None:
            raise ValueError("Models are not fitted. Call .fit(documents) first.")

        feature_names = self.vectorizer.get_feature_names_out()

        lda_topics = self._get_top_words(self.lda_model, feature_names, n_top_words)
        nmf_topics = self._get_top_words(self.nmf_model, feature_names, n_top_words)

        return {
            "lda_topics": lda_topics,
            "nmf_topics": nmf_topics,
        }

    def transform_document(self, document: str, method: str = "lda") -> List[float]:
        """
        Get topic distribution for a single document using LDA or NMF.
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted.")
        processed = self.preprocessor.preprocess(document)
        dtm = self.vectorizer.transform([processed])

        if method == "lda":
            if self.lda_model is None:
                raise ValueError("LDA model not fitted.")
            return self.lda_model.transform(dtm)[0].tolist()
        elif method == "nmf":
            if self.nmf_model is None:
                raise ValueError("NMF model not fitted.")
            return self.nmf_model.transform(dtm)[0].tolist()
        else:
            raise ValueError("method must be 'lda' or 'nmf'")
