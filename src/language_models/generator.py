from sentence_transformers import SentenceTransformer, util


class TextGenerator:
    def __init__(self):
        # Modelo de embeddings potente
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def compare_similarity(self, text_a: str, text_b: str):
        if not text_a.strip() or not text_b.strip():
            return {"error": "Two non-empty texts are required."}

        embeddings = self.embedding_model.encode([text_a, text_b])
        score = float(util.cos_sim(embeddings[0], embeddings[1]))
        return {
            "similarity_score": score,
            "interpretation": (
                "Very similar" if score > 0.75 else
                "Somewhat similar" if score > 0.4 else
                "Not very similar"
            )
        }

    def generate_response(self, query: str):
        """
        Chat básico: no usa un LLM gigante para no matar Colab,
        pero responde de forma decente para fines del proyecto.
        """
        q = query.lower()
        if "hello" in q or "hi" in q or "hola" in q:
            return "Hola, soy NewsBot. Puedo ayudarte a analizar noticias: resumen, sentimiento, entidades, traducción y más."
        if "who are you" in q or "qué eres" in q:
            return "Soy el NewsBot Intelligence System, un asistente de NLP diseñado para analizar noticias y textos informativos."

        return (
            "He recibido tu mensaje, y puedo ayudarte a: \n"
            "- Resumir una noticia (Summarization)\n"
            "- Analizar el sentimiento (Sentiment)\n"
            "- Extraer entidades (NER)\n"
            "- Traducir entre inglés y español\n"
            "- Clasificar el tema de la noticia\n"
            "- Medir la similitud entre dos textos\n"
            "Usa el menú de la izquierda o envíame instrucciones más específicas. 🙂"
        )
