from typing import Dict, Any

from src.conversation.query_processor import QueryProcessor
from src.conversation.response_generator import ResponseGenerator


class ChatBot:
    def __init__(self):
        from src.conversation.intent_classifier import IntentClassifier
        from src.conversation.query_processor import QueryProcessor

        self.ic = IntentClassifier()
        self.qp = QueryProcessor()

    def ask(self, query, context=None):
        if context is None:
            context = {}

        intent = self.ic.predict_intent(query)

        # FIX: call without keyword argument
        result = self.qp.process(query, context)

        return result

