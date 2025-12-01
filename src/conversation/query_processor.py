from src.text.summarizer import Summarizer
from src.text.sentiment import SentimentAnalyzer
from src.text.ner import EntityExtractor
from src.multilingual.translator import Translator
from src.semantic.similarity import SimilarityEngine
from src.classification.topic_classifier import TopicClassifier


class QueryProcessor:
    def __init__(self):
        self.summarizer = Summarizer()
        self.sentiment = SentimentAnalyzer()
        self.ner = EntityExtractor()
        self.translator = Translator()
        self.similarity = SimilarityEngine()
        self.classifier = TopicClassifier()

    def process(self, query, context):

        # Detect what the user wants based on "query"
        q = query.lower()

        # 1. SUMMARIZATION
        if "summarize" in q or "summary" in q:
            text = context.get("text", "")
            return self.summarizer.summarize(text)

        # 2. SENTIMENT ANALYSIS
        if "sentiment" in q or "feeling" in q:
            text = context.get("text", "")
            return self.sentiment.analyze(text)

        # 3. NAMED ENTITY RECOGNITION
        if "entities" in q or "ner" in q:
            text = context.get("text", "")
            return self.ner.extract(text)

        # 4. TRANSLATION
        if "translate" in q or "translation" in q:
            text = context.get("text", "")
            return self.translator.translate(text)

        # 5. SEMANTIC SIMILARITY
        if "similar" in q or "similarity" in q:
            text1 = context.get("text", "")
            text2 = context.get("reference", "")
            return self.similarity.compare(text1, text2)

        # 6. TOPIC CLASSIFICATION
        if "classify" in q or "topic" in q or "category" in q:
            text = context.get("text", "")
            return self.classifier.classify(text)

        # 7. DEFAULT CHAT MODE
        return f"🤖 I received your message: {query}"
