import requests


class TranslateTool:
    """A tool for translating text using the LibreTranslate API."""

    def __init__(self, api_url="https://libretranslate.com"):
        self.api_url = api_url.rstrip("/")

    def translate(self, text, source, target):
        """Translate text from one language to another."""

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        response = requests.post(
            f"{self.api_url}/translate",
            data={
                "q": text,
                "source": source,
                "target": target,
                "format": "text",
            },
            timeout=10,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Translation API error: {response.status_code}"
            )

        data = response.json()

        return data["translatedText"]


if __name__ == "__main__":
    tool = TranslateTool()

    result = tool.translate(
        "Привет, как дела?",
        "ru",
        "en"
    )

    print(result)
