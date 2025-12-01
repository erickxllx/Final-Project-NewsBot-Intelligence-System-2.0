class ResponseGenerator:
    """
    Optional utility to format responses in a clean and uniform way.
    The ChatBot and QueryProcessor work even without this file, but keeping it
    prevents missing-module errors and improves clarity.
    """

    @staticmethod
    def wrap(title: str, content):
        """
        Standard output wrapper for NLP results.
        """
        return {
            "title": title,
            "result": content
        }

    @staticmethod
    def error(message: str):
        return {
            "error": message
        }

    @staticmethod
    def plain(text: str):
        return {
            "response": text
        }
