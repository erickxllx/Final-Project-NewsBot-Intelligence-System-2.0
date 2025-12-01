from src.conversation.intent_classifier import IntentClassifier
from src.conversation.chatbot import ChatBot


def test_intent_classifier():
    ic = IntentClassifier()
    assert ic.classify("Can you summarize this?") == "summarize"
    assert ic.classify("What is the sentiment?") == "sentiment"


def test_chatbot_summary():
    bot = ChatBot()
    response = bot.ask("Summarize this text: The economy is improving due to lower inflation.")

    assert isinstance(response, str)
    assert len(response) > 0
