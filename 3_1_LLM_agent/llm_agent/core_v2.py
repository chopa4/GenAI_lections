# llm_agent/core_v2.py

import requests
import json
import re
from typing import List, Dict, Optional
from decouple import config

from .tool_calculator import CalculatorTool
from .tool_websearch import WebSearchTool
from .translate_tool import TranslateTool


class LLMAgent:
    """
    LLM-агент, который планирует и выполняет задачи с помощью инструментов.

    Поддерживает:
    - OpenRouter API
    - локальный Ollama
    """

    def __init__(
        self,
        model: str = "tngtech/deepseek-r1t2-chimera",
        local: bool = False,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "qwen3.5:0.8b"
    ):
        """
        Инициализирует агента.

        Args:
            model (str): Название модели для OpenRouter.
            local (bool): Если True, используется локальный Ollama.
            ollama_base_url (str): URL локального Ollama.
            ollama_model (str): Название модели в Ollama.
        """

        self.local = local
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model

        # ---------------------------------------------------------
        # Настройка API
        # ---------------------------------------------------------

        if not self.local:
            self.api_key = config("OPENROUTER_API_KEY")
            self.url = "https://openrouter.ai/api/v1/chat/completions"
            self.model = model

        else:
            self.api_key = None
            self.url = f"{self.ollama_base_url}/api/chat"
            self.model = ollama_model

        # ---------------------------------------------------------
        # Инструменты
        # ---------------------------------------------------------

        self.tools = {
            "calculator": CalculatorTool(),
            "web_search": WebSearchTool(),
            "translator": TranslateTool(),
        }

        self.conversation_history = []

    # =========================================================
    # API REQUEST
    # =========================================================

    def _make_api_request(
        self,
        payload: Dict,
        headers: Optional[Dict] = None
    ) -> Dict:
        """
        Универсальный метод для отправки запросов к OpenRouter/Ollama.

        Для Ollama используется:
            POST http://localhost:11434/api/chat

        Для OpenRouter:
            POST https://openrouter.ai/api/v1/chat/completions
        """

        if headers is None:
            headers = {}

        headers["Content-Type"] = "application/json"

        try:

            # =================================================
            # LOCAL OLLAMA
            # =================================================

            if self.local:

                ollama_payload = {
                    "model": self.model,
                    "messages": payload["messages"],
                    "stream": False
                }

                response = requests.post(
                    self.url,
                    json=ollama_payload,
                    headers=headers,
                    timeout=120
                )

                response.raise_for_status()

                data = response.json()

                # Ollama возвращает:
                #
                # {
                #   "message": {
                #       "role": "assistant",
                #       "content": "..."
                #   }
                # }
                #
                # Преобразуем в формат OpenAI,
                # который ожидает остальной код.

                return {
                    "choices": [
                        {
                            "message": {
                                "content": data["message"]["content"]
                            }
                        }
                    ]
                }

            # =================================================
            # OPENROUTER
            # =================================================

            else:

                headers.update({
                    "Authorization": f"Bearer {self.api_key}"
                })

                response = requests.post(
                    self.url,
                    json=payload,
                    headers=headers,
                    timeout=120
                )

                response.raise_for_status()

                return response.json()

        except requests.exceptions.RequestException as e:

            raise Exception(
                f"Ошибка при запросе к API: {e}"
            )

        except (KeyError, json.JSONDecodeError) as e:

            raise Exception(
                f"Ошибка обработки ответа API: {e}"
            )

    # =========================================================
    # CREATE PLAN
    # =========================================================

    def _ask_llm_for_plan(self, query: str) -> List[Dict]:
        """
        Создает план действий, используя LLM.

        LLM самостоятельно определяет,
        какой инструмент необходимо использовать.
        """

        system_prompt = """
You are a helpful AI planning assistant.

Analyze the user's request and decide if you need to use any tools.

Available tools:

--------------------------------------------------
1. calculator
--------------------------------------------------

Use calculator for mathematical calculations.

Examples:

"Сколько будет 25 * 37?"

"Реши (5 + 3) * 2"

"Сколько будет 15 процентов от 300?"

Input should be the mathematical expression.

--------------------------------------------------
2. web_search
--------------------------------------------------

Use web_search when information from the real world
or current information is required.

Examples:

"Кто сейчас президент Франции?"

"Какая погода в Москве?"

"Кто выиграл последний матч?"

IMPORTANT:
Search queries MUST be written in Russian.

--------------------------------------------------
3. translator
--------------------------------------------------

Use translator when the user asks to translate text.

The translator input MUST be a JSON object:

{
    "text": "text to translate",
    "source": "source language code",
    "target": "target language code"
}

Examples:

Russian -> English:

{
    "text": "Привет, как дела?",
    "source": "ru",
    "target": "en"
}

English -> Russian:

{
    "text": "Hello, how are you?",
    "source": "en",
    "target": "ru"
}

German -> English:

{
    "text": "Guten Morgen",
    "source": "de",
    "target": "en"
}

--------------------------------------------------
IMPORTANT RULES
--------------------------------------------------

1. Decide yourself whether a tool is needed.

2. You can use multiple tools if necessary.

3. One action corresponds to one tool call.

4. Use calculator only for mathematics.

5. Use web_search only for information that requires search.

6. Use translator for translation requests.

7. Do not use web_search for simple translation.

8. Do not use calculator for translation.

9. If no tool is needed, return an empty plan.

10. Return ONLY valid JSON.

11. Do NOT use Markdown.

12. Do NOT explain your decision.

Required format:

{
    "plan": [
        {
            "action": "tool_name",
            "input": "tool input"
        }
    ]
}

If no tools are required:

{
    "plan": []
}
"""

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        }

        try:

            response_data = self._make_api_request(payload)

            # -------------------------------------------------
            # Получаем текст ответа LLM
            # -------------------------------------------------

            llm_text = response_data[
                "choices"
            ][0][
                "message"
            ][
                "content"
            ]

            print(
                f"> Ответ LLM для плана:\n"
                f"{llm_text}\n"
            )

            # -------------------------------------------------
            # Удаляем ```json ... ```
            # -------------------------------------------------

            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```",
                llm_text,
                re.DOTALL
            )

            if json_match:

                cleaned_json_text = json_match.group(1)

            else:

                cleaned_json_text = llm_text.strip()

            # -------------------------------------------------
            # Парсим JSON
            # -------------------------------------------------

            try:

                action_plan = json.loads(
                    cleaned_json_text
                )

            except json.JSONDecodeError:

                # Попытка найти JSON внутри ответа

                json_match = re.search(
                    r'\{.*?"plan"\s*:\s*\[.*?\]\s*\}',
                    cleaned_json_text,
                    re.DOTALL
                )

                if not json_match:

                    print(
                        "Не удалось найти JSON "
                        "в ответе LLM."
                    )

                    return []

                action_plan = json.loads(
                    json_match.group()
                )

            plan = action_plan.get(
                "plan",
                []
            )

            return plan

        except Exception as e:

            print(
                f"Произошла ошибка при создании плана: {e}"
            )

            return []

    # =========================================================
    # FINAL RESPONSE
    # =========================================================

    def _generate_final_response(
        self,
        user_query: str
    ) -> str:
        """
        Генерирует финальный ответ
        на основе результатов инструментов.
        """

        prompt = f"""
Based on the following conversation log,
provide a direct and helpful answer to the user's
original question.

Be concise and use the information from the tool
results to support your answer.

Answer in the same language as the user.

Original User Question:

{user_query}

Conversation Log:

{chr(10).join(
    [msg["content"] for msg in self.conversation_history]
)}
"""

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        try:

            response_data = self._make_api_request(
                payload
            )

            final_text = response_data[
                "choices"
            ][0][
                "message"
            ][
                "content"
            ]

            return final_text

        except Exception as e:

            return (
                "Ошибка при генерации "
                f"финального ответа. Детали: {e}"
            )

    # =========================================================
    # PROCESS QUERY
    # =========================================================

    def process_query(
        self,
        query: str
    ) -> str:
        """
        Основной метод обработки запроса пользователя.
        """

        print(
            f"Агент анализирует ваш запрос... "
            f"(Режим: "
            f"{'локальный Ollama' if self.local else 'OpenRouter'})"
        )

        # =====================================================
        # ШАГ 1. ПЛАНИРОВАНИЕ
        # =====================================================

        plan = self._ask_llm_for_plan(
            query
        )

        # =====================================================
        # Если инструменты не требуются
        # =====================================================

        if not plan:

            print(
                "Инструменты не требуются. "
                "Генерирую ответ напрямую."
            )

            direct_prompt = (
                "Ответьте на следующий вопрос "
                "кратко и информативно:\n\n"
                f"{query}"
            )

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": direct_prompt
                    }
                ]
            }

            try:

                response_data = self._make_api_request(
                    payload
                )

                return response_data[
                    "choices"
                ][0][
                    "message"
                ][
                    "content"
                ]

            except Exception as e:

                return (
                    "Извините, не удалось "
                    f"сгенерировать ответ: {e}"
                )

        # =====================================================
        # ШАГ 2. ИСПОЛНЕНИЕ ПЛАНА
        # =====================================================

        print(
            f"План действий: {plan}"
        )

        for step in plan:

            tool_name = step.get(
                "action"
            )

            tool_input = step.get(
                "input"
            )

            if tool_name in self.tools:

                print(
                    f"Выполняется инструмент: "
                    f"'{tool_name}'"
                )

                try:

                    result = self.tools[
                        tool_name
                    ].use(
                        tool_input
                    )

                    print(
                        f"Результат: {result}"
                    )

                    # Добавляем результат
                    # в историю

                    self.conversation_history.append({
                        "role": "system",
                        "content": (
                            f"Tool {tool_name} "
                            f"result: {result}"
                        )
                    })

                except Exception as e:

                    error_msg = (
                        f"Ошибка при выполнении "
                        f"инструмента "
                        f"'{tool_name}': {e}"
                    )

                    print(error_msg)

                    self.conversation_history.append({
                        "role": "system",
                        "content": error_msg
                    })

            else:

                error_msg = (
                    f"Ошибка: инструмент с именем "
                    f"'{tool_name}' не найден."
                )

                print(error_msg)

                self.conversation_history.append({
                    "role": "system",
                    "content": error_msg
                })

        # =====================================================
        # ШАГ 3. ФИНАЛЬНЫЙ ОТВЕТ
        # =====================================================

        print(
            "Составляю финальный ответ..."
        )

        final_response = (
            self._generate_final_response(
                query
            )
        )

        return final_response

    # =========================================================
    # TEST OLLAMA CONNECTION
    # =========================================================

    def test_ollama_connection(
        self
    ) -> bool:
        """
        Тестирует соединение с локальным Ollama сервером.

        Returns:
            bool: True если соединение успешно,
            иначе False.
        """

        if not self.local:

            return False

        try:

            test_url = (
                f"{self.ollama_base_url}/api/tags"
            )

            response = requests.get(
                test_url,
                timeout=10
            )

            return response.status_code == 200

        except:

            return False
