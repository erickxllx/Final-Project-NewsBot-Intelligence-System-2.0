class NewsClassifier:
    """
    Zero-shot topic classifier using keyword rules.
    Works without training and is 100% Streamlit-safe.
    """

    def __init__(self):
        # Keyword dictionaries for each topic
        self.topics = {
            "Politics": ["election", "president", "government", "senate", "law"],
            "Economy": ["market", "inflation", "stocks", "economy", "trade"],
            "Technology": ["ai", "software", "tech", "computer", "robot"],
            "Sports": ["match", "goal", "tournament", "team", "league"],
            "Health": ["covid", "vaccine", "health", "disease", "medical"],
            "Environment": ["climate", "pollution", "wildfire", "environment"]
        }

    def predict(self, text: str) -> dict:
        """
        Simple keyword-based classifier.
        Returns a dict with label + confidence score.
        """

        if not text or text.strip() == "":
            return {"label": "Unknown", "confidence": 0.0}

        t = text.lower()
        scores = {}

        # Count keywords per category
        for topic, keywords in self.topics.items():
            score = sum(t.count(k) for k in keywords)
            scores[topic] = score

        # Pick the best topic
        best_topic = max(scores, key=scores.get)
        best_score = scores[best_topic]

        # Confidence normalized (0–1)
        total = sum(scores.values())
        confidence = best_score / total if total > 0 else 0.0

        return {
            "label": best_topic,
            "confidence": round(confidence, 3)
        }
