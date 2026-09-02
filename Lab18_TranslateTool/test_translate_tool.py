import pytest
from unittest.mock import Mock, patch

from translate_tool import TranslateTool


def test_translate_success():
    """Test successful translation."""
    tool = TranslateTool()

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "translatedText": "Hello, world!"
    }

    with patch(
        "translate_tool.requests.post",
        return_value=mock_response
    ):
        result = tool.translate("Привет, мир!", "ru", "en")

    assert result == "Hello, world!"


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


def test_api_error():
    """Test API error handling."""
    tool = TranslateTool()

    mock_response = Mock()
    mock_response.status_code = 500

    with patch(
        "translate_tool.requests.post",
        return_value=mock_response
    ):
        with pytest.raises(RuntimeError):
            tool.translate("Привет", "ru", "en")


def test_custom_api_url():
    """Test custom API URL configuration."""
    tool = TranslateTool("https://example.com/")

    assert tool.api_url == "https://example.com"
