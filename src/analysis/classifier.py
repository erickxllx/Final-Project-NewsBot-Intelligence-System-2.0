from transformers import pipeline


class NewsClassifier:
    def __init__(self):
        # Modelo zero-shot MULTILINGÜE (sirve para español)
        self.classifier = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli"
        )

        # Etiquetas típicas de noticias
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
        text = text.strip()
        if not text:
            return {"error": "Empty text for classification."}

        result = self.classifier(
            text[:1500],
            candidate_labels=self.labels,
            multi_label=False
        )

        return {
            "label": result["labels"][0],
            "confidence": float(result["scores"][0]),
            "all_scores": dict(zip(result["labels"], map(float, result["scores"])))
        }
