from src.multilingual.language_detector import LanguageDetector
from src.multilingual.translator import Translator
from src.multilingual.cross_lingual_analyzer import CrossLingualAnalyzer


def test_language_detection():
    ld = LanguageDetector()
    assert ld.detect("Hola, cómo estás?") == "es"


def test_translation_es_to_en():
    tr = Translator()
    result = tr.translate("Hola mundo", src_lang="es", tgt_lang="en")
    assert isinstance(result, str)
    assert len(result) > 0


def test_cross_lingual_similarity():
    cla = CrossLingualAnalyzer()

    a = "El presidente habló sobre economía."
    b = "The president spoke about the economy."

    result = cla.compare_pair(a, b)
    assert "similarity" in result
    assert isinstance(result["similarity"], float)
