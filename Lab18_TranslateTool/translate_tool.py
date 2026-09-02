import requests


class TranslateTool:
    """Tool for translating text using MyMemory Translation API."""

    def __init__(self, api_url="https://api.mymemory.translated.net"):
        self.api_url = api_url.rstrip("/")

    def translate(self, text, source, target):
        """Translate text from source language to target language."""

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        response = requests.get(
            f"{self.api_url}/get",
            params={
                "q": text,
                "langpair": f"{source}|{target}",
            },
            timeout=10,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Translation API error: "
                f"{response.status_code}: {response.text}"
            )

        data = response.json()

        if data.get("responseStatus") != 200:
            raise RuntimeError(
                f"Translation API error: {data.get('responseDetails')}"
            )

        return data["responseData"]["translatedText"]


if __name__ == "__main__":
    tool = TranslateTool()

    result = tool.translate(
        "Привет как дела?",
        "ru",
        "en"
    )

    print(result)
