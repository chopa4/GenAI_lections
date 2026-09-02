import pytest
from unittest.mock import Mock, patch

from translate_tool import TranslateTool


def test_translate_success():
    """Test successful translation."""
    tool = TranslateTool()

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "responseStatus": 200,
        "responseData": {
            "translatedText": "Hello, how are you?"
        }
    }

    with patch(
        "translate_tool.requests.get",
        return_value=mock_response
    ):
        result = tool.translate(
            "Привет, как дела?",
            "ru",
            "en"
        )

    assert result == "Hello, how are you?"


def test_empty_text():
    """Test empty text validation."""
    tool = TranslateTool()

    with pytest.raises(ValueError):
        tool.translate("", "ru", "en")


def test_whitespace_text():
    """Test whitespace-only text validation."""
    tool = TranslateTool()

    with pytest.raises(ValueError):
        tool.translate("   ", "ru", "en")


def test_http_error():
    """Test HTTP error handling."""
    tool = TranslateTool()

    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    with patch(
        "translate_tool.requests.get",
        return_value=mock_response
    ):
        with pytest.raises(RuntimeError):
            tool.translate("Привет", "ru", "en")


def test_api_error():
    """Test API error response handling."""
    tool = TranslateTool()

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "responseStatus": 403,
        "responseDetails": "Quota exceeded"
    }

    with patch(
        "translate_tool.requests.get",
        return_value=mock_response
    ):
        with pytest.raises(RuntimeError):
            tool.translate("Привет", "ru", "en")
