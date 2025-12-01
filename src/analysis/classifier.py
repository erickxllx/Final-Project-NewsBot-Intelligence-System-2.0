from typing import List, Dict, Any

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_processing.text_preprocessor import TextPreprocessor


class NewsClassifier:
    """
    News article classifier using TF-IDF + Logistic Regression.
    This can be trained on your labeled news dataset and then
    used in the final NewsBot system for category prediction.
    """

    def __init__(self, language: str = "en"):
        self.language = language
        self.preprocessor = TextPreprocessor(language=language)
        self.pipeline: Pipeline | None = None
        self.label_mapping: Dict[int, str] | None = None

    def build_pipeline(self):
        """Initialize the sklearn pipeline."""
        self.pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(max_features=5000)),
                ("clf", LogisticRegression(max_iter=200)),
            ]
        )

    def fit(self, texts: List[str], labels: List[str]):
        """
        Train the classifier.

        :param texts: list of raw news texts
        :param labels: list of category labels (e.g., 'sports', 'politics')
        """
        if self.pipeline is None:
            self.build_pipeline()

        # Preprocess texts
        processed_texts = [self.preprocessor.preprocess(t) for t in texts]

        # Fit pipeline
        self.pipeline.fit(processed_texts, labels)

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict the category of a single news article.

        :return: dict with 'label' and 'confidence'
        """
        if self.pipeline is None:
            raise ValueError("Model is not trained. Call .fit() first.")

        processed = self.preprocessor.preprocess(text)
        proba = self.pipeline.predict_proba([processed])[0]
        classes = list(self.pipeline.classes_)

        # Get best label
        max_idx = proba.argmax()
        return {
            "label": classes[max_idx],
            "confidence": float(proba[max_idx]),
            "all_probs": {cls: float(p) for cls, p in zip(classes, proba)},
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict for a list of texts."""
        return [self.predict(t) for t in texts]
