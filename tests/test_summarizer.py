from src.language_models.summarizer import Summarizer


def test_summarizer_returns_shorter_text():
    s = Summarizer()

    text = (
        "The European Union announced new sanctions against Russia today as part "
        "of a broader diplomatic effort involving energy and finance sectors."
    )

    summary = s.summarize(text)
    assert isinstance(summary, str)
    assert len(summary) < len(text)
