# =========================================
# QueryProcessor – FIXED FOR YOUR PROJECT
# =========================================

# Data preprocessing
from src.data_processing.text_preprocessor import TextPreprocessor

# Analysis modules
from src.analysis.classifier import NewsClassifier
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.analysis.ner_extractor import NERExtractor
from src.analysis.topic_modeler import TopicModeler

# Language model tools
from src.language_models.summarizer import Summarizer
from src.language_models.embeddings import EmbeddingModel   
#from src.language_models.generator import TextGenerator

# Multilingual modules
from src.multilingual.translator import Translator
from src.multilingual.language_detector import LanguageDetector
from src.multilingual.cross_lingual_analyzer import CrossLingualAnalyzer


class QueryProcessor:

    def __init__(self):

        # Preprocessing
        self.pre = TextPreprocessor()

        # Analysis modules
        self.classifier = NewsClassifier()
        self.sentiment = SentimentAnalyzer()
        self.ner = NERExtractor()
        self.topic_modeler = TopicModeler()

        # Language models
        self.summarizer = Summarizer()
        self.embed = EmbeddingModel()
        #self.generator = TextGenerator()

        # Multilingual
        self.translator = Translator()
        self.lang_detector = LanguageDetector()
        self.cross = CrossLingualAnalyzer()

    def process(self, query, context):
        intent = context.get("intent", "")

        # =============================================
        # SUMMARIZATION
        # =============================================
        if intent == "summarize":
            text = context["text"]
            return self.summarizer.summarize(text)

        # =============================================
        # SENTIMENT
        # =============================================
        if intent == "sentiment":
            text = context["text"]
            return self.sentiment.analyze(text)

        # =============================================
        # NER
        # =============================================
        if intent == "ner":
            text = context["text"]
            return self.ner.extract(text)

        # =============================================
        # TRANSLATION
        # =============================================
        if intent == "translate":
            text = context["text"]
            return self.translator.translate(text, src_lang="auto", tgt_lang="en")

        # =============================================
        # SEMANTIC SIMILARITY
        # =============================================
        if intent == "similarity":
            text1 = context["text"]
            text2 = context["reference"]
            return self.embed.similarity(text1, text2)

        # =============================================
        # CLASSIFICATION
        # =============================================
        if intent == "classify":
            text = context["text"]
            return self.classifier.predict(text)

        # Default fallback
        return {"response": "I'm not sure what you want."}
