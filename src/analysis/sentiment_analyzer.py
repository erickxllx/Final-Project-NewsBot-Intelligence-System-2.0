from transformers import pipeline


class SentimentAnalyzer:
    def __init__(self):
        # Modelo multilingüe (es/en)
        self.model = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )

    def analyze(self, text: str):
        """
        Returns sentiment label and score. Ej:
        {'label': '4 stars', 'score': 0.65}
        """
        if not text.strip():
            return {"error": "Empty text for sentiment analysis."}

        result = self.model(text[:4000])  # cortar por si es demasiado largo
        return result[0]
