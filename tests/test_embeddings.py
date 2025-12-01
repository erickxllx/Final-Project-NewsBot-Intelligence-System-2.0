from src.language_models.embeddings import EmbeddingEngine


def test_embeddings_and_similarity():
    emb = EmbeddingEngine()

    v1 = emb.encode("The stock market crashed today.")
    v2 = emb.encode("Stocks fell sharply during economic uncertainty.")

    assert len(v1) == len(v2)

    score = emb.similarity("A", "B")
    assert isinstance(score, float)
