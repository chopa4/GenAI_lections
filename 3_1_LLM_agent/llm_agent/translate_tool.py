import requests
import json


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

    def use(self, tool_input):
        """
        Метод для вызова инструмента из LLMAgent.

        Ожидает JSON:
        {
            "text": "Привет",
            "source": "ru",
            "target": "en"
        }
        """

        if isinstance(tool_input, str):
            data = json.loads(tool_input)
        else:
            data = tool_input

        text = data.get("text")
        source = data.get("source")
        target = data.get("target")

        if not text:
            raise ValueError("Translation text is required")

        if not source:
            raise ValueError("Source language is required")

        if not target:
            raise ValueError("Target language is required")

        return self.translate(
            text,
            source,
            target
        )


if __name__ == "__main__":
    tool = TranslateTool()

    result = tool.translate(
        "Привет Кирилл из России",
        "ru",
        "en"
    )

    print(result)
