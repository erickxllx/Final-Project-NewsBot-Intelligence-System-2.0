from googletrans import Translator as GT

class Translator:
    def __init__(self):
        self.translator = GT()

    def detect_language(self, text):
        try:
            return self.translator.detect(text).lang
        except:
            return "unknown"

    def translate_to_english(self, text):
        if not text or not isinstance(text, str):
            return ""

        try:
            result = self.translator.translate(text, src="es", dest="en")
            return result.text
        except Exception as e:
            return f"[Translation error: {e}]"

    def translate_to_spanish(self, text):
        if not text or not isinstance(text, str):
            return ""

        try:
            result = self.translator.translate(text, src="en", dest="es")
            return result.text
        except Exception as e:
            return f"[Translation error: {e}]"
