from src.analysis.ner_extractor import NERExtractor


def test_ner_extractor():
    ner = NERExtractor()
    text = "Barack Obama met with Tim Cook at Apple headquarters."

    entities = ner.extract(text)
    assert isinstance(entities, list)
    assert len(entities) > 0
