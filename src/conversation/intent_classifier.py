class IntentClassifier:
    def __init__(self):
        self.intents = {
            "summarize": ["summarize", "summary", "resumen"],
            "sentiment": ["sentiment", "opinion", "feeling"],
            "ner": ["entities", "extract entities", "ner"],
            "translate": ["translate", "translation", "traducir"],
            "similarity": ["similarity", "compare"],
            "classify": ["classify", "category", "topic"],
            "chat": []
        }

    def predict_intent(self, query):
        query_lower = query.lower()

        for intent, keywords in self.intents.items():
            for kw in keywords:
                if kw in query_lower:
                    return intent

        return "chat"
