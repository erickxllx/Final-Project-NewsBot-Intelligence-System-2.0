from transformers import pipeline


class NewsClassifier:
    def __init__(self):
        # Zero-shot classifier potente
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        # Labels típicos de noticias
        self.labels = [
            "politics",
            "economy",
            "business",
            "technology",
            "sports",
            "health",
            "entertainment",
            "science",
            "world",
            "environment"
        ]

    def predict(self, text: str):
        """
        Returns top label + scores for news topic.
        """
        if not text.strip():
            return {"error": "Empty text for classification."}

        result = self.classifier(
            text[:1500],
            candidate_labels=self.labels,
            multi_label=False
        )
        return {
            "predicted_topic": result["labels"][0],
            "scores": dict(zip(result["labels"], result["scores"]))
        }
