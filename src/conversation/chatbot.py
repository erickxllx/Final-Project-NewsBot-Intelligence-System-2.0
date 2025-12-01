from typing import Dict, Any

from src.conversation.query_processor import QueryProcessor
from src.conversation.response_generator import ResponseGenerator


class ChatBot:
    """
    High-level chatbot interface for NewsBot.
    Handles:
      - intent detection
      - task execution
      - response formatting
    """

    def __init__(self):
        self.qp = QueryProcessor()
        self.rg = ResponseGenerator()

    def ask(self, query: str, context: Dict[str, Any] | None = None) -> str:
        result = self.qp.process(query, context=context)
        response = self.rg.generate(result)
        return response
