import os
import anthropic
from .client import InferenceClient


class ClaudeAPIClient(InferenceClient):

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = "claude-sonnet-4-6"
        self._max_tokens = 8192

    async def chat(self, messages: list[dict], system: str = "") -> str:
        kwargs = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return response.content[0].text
