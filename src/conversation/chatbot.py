from src.conversation.intent_classifier import IntentClassifier
from src.conversation.query_processor import QueryProcessor


class ChatBot:
    def __init__(self):
        self.ic = IntentClassifier()
        self.qp = QueryProcessor()

    def ask(self, query, context=None):
        if context is None:
            context = {}

        intent = self.ic.predict_intent(query)
        context["intent"] = intent

        return self.qp.process(query, context)
