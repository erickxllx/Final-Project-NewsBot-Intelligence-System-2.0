from typing import Dict, Any

# Text processing
from src.data_processing.text_preprocessor import TextPreprocessor

# Analysis modules
from src.analysis.classifier import NewsClassifier
from src.analysis.topic_modeler import TopicModeler
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.analysis.ner_extractor import EntityExtractor

# Multilingual modules
from src.multilingual.translator import Translator
from src.multilingual.language_detector import LanguageDetector
from src.multilingual.cross_lingual_analyzer import CrossLingualAnalyzer

# Language models
from src.language_models.embeddings import EmbeddingModel
from src.language_models.generator import TextGenerator


class QueryProcessor:
    def __init__(self):
        self.pre = TextPreprocessor()

        # Analysis components
        self.classifier = NewsClassifier()
        self.topic_modeler = TopicModeler()
        self.sentiment = SentimentAnalyzer()
        self.ner = EntityExtractor()

        # Multilingual components
        self.translator = Translator()
        self.lang_detector = LanguageDetector()
        self.cross = CrossLingualAnalyzer()

        # Models
        self.embedder = EmbeddingModel()
        self.generator = TextGenerator()

    def process(self, query: str, context: Dict[str, Any]):
        intent = context.get("intent")
        text = context.get("text", "")

        # --- INTENT ROUTING ---
        if intent == "summarize":
            return self.generator.generate_summary(text)

        elif intent == "sentiment":
            return self.sentiment.analyze(text)

        elif intent == "ner":
            return self.ner.extract(text)

        elif intent == "translate":
            lang = self.lang_detector.detect(text)
            translation = self.translator.auto_translate(text)
            return {
                "detected_language": lang,
                "translation": translation
            }

        elif intent == "similarity":
            ref = context.get("reference", "")
            score = self.embedder.similarity(text, ref)
            return {"similarity": score}

        elif intent == "classify":
            return self.classifier.predict([text])[0]

        # DEFAULT → CHAT MODE (LLM)
        return self.generator.chat(query)
