def test_classifier():
    from src.analysis.classifier import NewsClassifier
    clf = NewsClassifier()
    pred = clf.predict("Test")
    assert "label" in pred
