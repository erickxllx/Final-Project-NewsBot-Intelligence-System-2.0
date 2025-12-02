from transformers import pipeline


class Summarizer:
    def __init__(self):
        # Modelo fuerte para resumen de noticias
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )

    def summarize(self, text: str, max_length: int = 180, min_length: int = 60):
        if not text.strip():
            return "Empty text, nothing to summarize."

        # cortar por si el texto es gigante (limite de tokens)
        text = text.strip()
        if len(text) > 4000:
            text = text[:4000]

        result = self.summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )

        return result[0]["summary_text"]
