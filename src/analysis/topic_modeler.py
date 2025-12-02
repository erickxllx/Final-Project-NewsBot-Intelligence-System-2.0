from transformers import pipeline


class TopicModeler:
    def __init__(self):
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
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

    def get_topics(self, text: str, top_k: int = 3):
        if not text.strip():
            return {"error": "Empty text for topic modeling."}

        result = self.classifier(
            text[:1500],
            candidate_labels=self.labels,
            multi_label=True
        )

        labels_scores = list(zip(result["labels"], result["scores"]))
        labels_scores.sort(key=lambda x: x[1], reverse=True)

        top = labels_scores[:top_k]

        return {
            "top_topics": [
                {"topic": label, "score": float(score)}
                for label, score in top
            ]
        }
