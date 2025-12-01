from src.analysis.classifier import NewsClassifier


def test_classifier_training_and_prediction():
    clf = NewsClassifier()

    texts = [
        "The president met with diplomats today.",
        "The soccer team won the championship.",
    ]
    labels = ["politics", "sports"]

    clf.fit(texts, labels)
    pred = clf.predict("The government announced new policies.")

    assert "label" in pred
    assert pred["label"] in ["politics", "sports"]
