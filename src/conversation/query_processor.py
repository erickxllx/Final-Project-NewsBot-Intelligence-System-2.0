from src.analysis.classifier import NewsClassifier
from src.analysis.ner_extractor import NERExtractor
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.analysis.topic_modeler import TopicModeler

from src.language_models.summarizer import Summarizer
from src.language_models.embeddings import EmbeddingModel
from src.language_models.generator import TextGenerator

from src.multilingual.translator import Translator
from src.multilingual.language_detector import LanguageDetector
from src.multilingual.cross_lingual_analyzer import CrossLingualAnalyzer

class QueryProcessor:
    def __init__(self):
        self.summarizer = Summarizer()
        self.sentiment = SentimentAnalyzer()
        self.ner = NERExtractor()
        self.classifier = NewsClassifier()
        self.topics = TopicModeler()
        self.embed = EmbeddingModel()
        self.generator = TextGenerator()
        self.translator = Translator()
        self.lang_detect = LanguageDetector()
        self.cross = CrossLingualAnalyzer()

    def process(self, query, context):
        intent = context.get("intent", "")

        if intent == "summarize":
            return self.summarizer.summarize(context["text"])

        if intent == "sentiment":
            return self.sentiment.analyze(context["text"])

        if intent == "ner":
            return self.ner.extract(context["text"])

        if intent == "translate":
            return self.translator.translate(context["text"], "en")

        if intent == "similarity":
            return self.embed.similarity(context["text"], context["reference"])

        if intent == "classify":
            return self.classifier.predict(context["text"])

        return {"reply": "I did not understand the request."}
