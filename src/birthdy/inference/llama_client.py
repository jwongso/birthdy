import aiohttp
from .client import InferenceClient


class LocalLlamaClient(InferenceClient):

    def __init__(self, base_url: str = "http://localhost:8080"):
        self._base_url = base_url.rstrip("/")

    async def chat(self, messages: list[dict], system: str = "", thinking: bool = False) -> str:
        if system:
            messages = [{"role": "system", "content": system}] + messages

        payload = {
            "model": "local",
            "messages": messages,
            "max_tokens": 8192,
        }

        if thinking:
            payload["thinking"] = True

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"llama-server {resp.status}: {body}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
