import pytest
from unittest.mock import Mock, patch

from llm_agent.translate_tool import TranslateTool


def test_translate_success():
    """Проверяет успешный перевод."""
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "responseStatus": 200,
        "responseData": {
            "translatedText": "Hello"
        }
    }

    with patch(
        "llm_agent.translate_tool.requests.get",
        return_value=response
    ):
        tool = TranslateTool()
        result = tool.translate("Привет", "ru", "en")

    assert result == "Hello"


def test_translate_empty_text():
    """Проверяет ошибку при пустом тексте."""
    tool = TranslateTool()

    with pytest.raises(ValueError):
        tool.translate("", "ru", "en")


def test_use():
    """Проверяет вызов инструмента через use."""
    tool = TranslateTool()

    with patch.object(
        tool,
        "translate",
        return_value="Hello"
    ):
        result = tool.use({
            "text": "Привет",
            "source": "ru",
            "target": "en"
        })

    assert result == "Hello"