from abc import ABC, abstractmethod


class InferenceClient(ABC):

    @abstractmethod
    async def chat(self, messages: list[dict], system: str = "") -> str:
        """
        Send a conversation to the model and return the reply as a string.

        messages: list of {"role": "user"|"assistant", "content": "..."}
        system:   system prompt, empty string means no system prompt
        """
        ...
