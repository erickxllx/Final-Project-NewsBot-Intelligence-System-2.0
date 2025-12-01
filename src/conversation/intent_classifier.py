class IntentClassifier:
    def __init__(self):
        self.intents = {
            "summarize": [
                "summarize", "summary", "summarize this news",
                "resume", "resumen"
            ],
            "sentiment": [
                "sentiment", "feeling", "opinion",
                "what is the sentiment", "analyze sentiment"
            ],
            "ner": [
                "entities", "extract entities", "named entity",
                "ner", "extract the entities"
            ],
            "translate": [
                "translate", "translation", "traducir",
                "translate this", "translate to english"
            ],
            "similarity": [
                "similarity", "compare", "are these two similar"
            ],
            "classify": [
                "classify", "topic", "category",
                "classify the topic"
            ],
            "chat": []  # fallback
        }

    def predict_intent(self, query):
        query_lower = query.lower()

        # buscar coincidencias exactas
        for intent, keywords in self.intents.items():
            for kw in keywords:
                if kw in query_lower:
                    return intent

        # fallback
        return "chat"
