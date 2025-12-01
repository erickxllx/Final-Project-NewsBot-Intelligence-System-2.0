from typing import Dict, Any

try:
    from transformers import pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


class SentimentAnalyzer:
    """
    Sentiment analysis using HuggingFace transformers if available.
    Falls back to a simple rule-based analyzer if transformers is not installed.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.model_name = model_name
        self._use_transformers = _TRANSFORMERS_AVAILABLE
        self._pipe = None

        if self._use_transformers:
            try:
                self._pipe = pipeline("sentiment-analysis", model=self.model_name)
            except Exception:
                # If model fails to load, use rule-based
                self._use_transformers = False

    def _rule_based(self, text: str) -> Dict[str, Any]:
        """
        Very simple rule-based sentiment as fallback.
        Not perfect, but enough to show functionality.
        """
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "positive", "success"]
        negative_words = ["bad", "terrible", "awful", "horrible", "negative", "failure", "crisis"]

        text_lower = text.lower()
        pos_count = sum(w in text_lower for w in positive_words)
        neg_count = sum(w in text_lower for w in negative_words)

        if pos_count > neg_count:
            label = "POSITIVE"
        elif neg_count > pos_count:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        score = (max(pos_count, neg_count) / (pos_count + neg_count)) if (pos_count + neg_count) > 0 else 0.5

        return {"label": label, "score": float(score), "engine": "rule-based"}

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a given text.

        Returns:
            {
                "label": "POSITIVE"/"NEGATIVE"/"NEUTRAL",
                "score": float,
                "engine": "transformers" or "rule-based"
            }
        """
        if self._use_transformers and self._pipe is not None:
            res = self._pipe(text)[0]
            return {
                "label": res["label"],
                "score": float(res["score"]),
                "engine": "transformers",
            }

        # Fallback
        return self._rule_based(text)
