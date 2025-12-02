from src.data_processing.text_preprocessor import TextPreprocessor

# Analysis modules
from src.analysis.classifier import NewsClassifier
from src.analysis.topic_modeler import TopicModeler
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.analysis.ner_extractor import NERExtractor

# Language models
from src.language_models.summarizer import Summarizer
from src.language_models.generator import TextGenerator

# Multilingual
from src.multilingual.translator import Translator


class QueryProcessor:
    def __init__(self):
        # Preprocessing
        self.pre = TextPreprocessor()

        # Analysis modules (modelos potentes)
        self.classifier = NewsClassifier()
        self.sentiment = SentimentAnalyzer()
        self.ner = NERExtractor()
        self.topic_modeler = TopicModeler()

        # Language models
        self.summarizer = Summarizer()
        self.generator = TextGenerator()

        # Multilingual
        self.translator = Translator()

    def process(self, query, context=None):
        if context is None:
            context = {}

        query_lower = query.lower().strip()
        text = context.get("text", "")

        # ======================
        # Summarization
        # ======================
        if any(k in query_lower for k in ["summary", "summarize", "resumen", "resume"]):
            if not text.strip():
                return "No text provided to summarize."
            return self.summarizer.summarize(text)

        # ======================
        # Sentiment
        # ======================
        if any(k in query_lower for k in ["sentiment", "feeling", "opinion", "analyze sentiment"]):
            if not text.strip():
                return "No text provided for sentiment analysis."
            return self.sentiment.analyze(text)

        # ======================
        # NER
        # ======================
        if any(k in query_lower for k in ["entities", "ner", "extract entities", "named entity"]):
            if not text.strip():
                return "No text provided for entity extraction."
            return self.ner.extract(text)

        # ======================
        # Translation
        # ======================
        if any(k in query_lower for k in ["translate", "translation", "traducir"]):
            if not text.strip():
                return "No text provided to translate."
            lang = self.translator.detect_language(text)
            if lang == "es":
                return self.translator.translate_to_english(text)
            else:
                return self.translator.translate_to_spanish(text)

        # ======================
        # Similarity
        # (usa embeddings potentes en TextGenerator)
        # ======================
        if any(k in query_lower for k in ["similar", "similarity", "compare"]):
            text_a = context.get("text", "")
            text_b = context.get("reference", "")
            if not text_a.strip() or not text_b.strip():
                return "Two texts are required for similarity comparison."
            return self.generator.compare_similarity(text_a, text_b)

        # ======================
        # Classification (topic)
        # ======================
        if any(k in query_lower for k in ["classify", "topic", "category"]):
            if not text.strip():
                return "No text provided for topic classification."
            return self.classifier.predict(text)

        # ======================
        # Topic Modeling (más detallado)
        # ======================
        if "topics" in query_lower or "topic model" in query_lower:
            if not text.strip():
                return "No text provided for topic modeling."
            return self.topic_modeler.get_topics(text)

        # ======================
        # Default chat (respuesta generativa básica)
        # ======================
        return self.generator.generate_response(query)
