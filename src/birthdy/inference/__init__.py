import os
from .claude_client import ClaudeAPIClient
from .llama_client import LocalLlamaClient


def get_inference_client():
    backend = os.environ.get("INFERENCE_BACKEND", "claude").lower()

    if backend == "claude":
        return ClaudeAPIClient()
    elif backend == "llama":
        url = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080")
        return LocalLlamaClient(base_url=url)
    else:
        raise ValueError(f"Unknown INFERENCE_BACKEND: {backend!r}. Use 'claude' or 'llama'.")
