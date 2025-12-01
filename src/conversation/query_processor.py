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

        # Analysis modules
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

        query = query.lower().strip()

        # ======================
        # Summarization
        # ======================
        if "summary" in query or "summarize" in query or "resumen" in query:
            return self.summarizer.summarize(context.get("text", ""))

        # ======================
        # Sentiment
        # ======================
        if "sentiment" in query or "feeling" in query or "opinion" in query:
            return self.sentiment.analyze(context.get("text", ""))

        # ======================
        # NER
        # ======================
        if "entities" in query or "ner" in query:
            return self.ner.extract(context.get("text", ""))

        # ======================
        # Translation
        # ======================
        query_lower = query.lower()

                # ======================
                # Translation
                # ======================
        if "translate" in query_lower or "traducir" in query_lower:
                text = context.get("text", "")
                lang = self.translator.detect_language(text)

                if lang == "es":
                    return self.translator.translate_to_english(text)
                else:
                    return self.translator.translate_to_spanish(text)


        # ======================
        # Similarity
        # ======================
        if "similar" in query or "similarity" in query or "compare" in query:
            return self.generator.compare_similarity(
                context.get("text", ""),
                context.get("reference", "")
            )

        # ======================
        # Classification
        # ======================
        if "classify" in query or "topic" in query or "category" in query:
            return self.classifier.predict(context.get("text", ""))

        # ======================
        # Default chat
        # ======================
        return self.generator.generate_response(query)
