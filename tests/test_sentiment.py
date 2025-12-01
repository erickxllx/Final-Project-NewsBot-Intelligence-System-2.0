from src.analysis.sentiment_analyzer import SentimentAnalyzer


def test_sentiment_analyzer():
    analyzer = SentimentAnalyzer()

    result = analyzer.analyze("I love this!")
    assert "label" in result
    assert result["label"] in ["POSITIVE", "NEGATIVE"]
