# Step 2 - Inference Layer

## Goal

Create the abstract `InferenceClient` interface and the `ClaudeAPIClient`
implementation. After this step you can call Claude from Python code and get
a response back.

---

## Why an abstract base class?

Today Birthdy uses the Claude API. In Jan/Feb 2027 it will use a local llama.cpp
server. If you write Claude API calls directly in the bot code, migrating means
finding and rewriting every call site.

With an abstract base class, you define one interface that both backends must
implement. The bot code calls `client.chat(messages, system)` and never knows
or cares whether the answer comes from Claude or llama.cpp. Swapping backends is
one line in `.env`.

This is the Dependency Inversion principle: high-level code (the bot) depends on
an abstraction, not a concrete implementation.

---

## 1. Create the abstract base class

Create `src/birthdy/inference/client.py`:

```python
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
```

### Why async?

python-telegram-bot is fully async (built on asyncio). If `chat()` were synchronous,
calling it from a Telegram handler would block the entire event loop - no other
messages could be processed while waiting for the LLM response. Making it async
allows the bot to handle other events while waiting for the API response.

### Why list[dict] for messages?

Both the Claude API and the OpenAI-compatible API that llama.cpp exposes use the
same message format: a list of `{"role": ..., "content": ...}` dicts. Using this
format in the interface means neither implementation needs to translate.

---

## 2. Create the Claude API client

Create `src/birthdy/inference/claude_client.py`:

```python
import os
import anthropic


class ClaudeAPIClient:

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
```

### Why inherit from InferenceClient?

Explicit inheritance means Python will raise a `TypeError` at startup if
`ClaudeAPIClient` ever fails to implement `chat()`. Without it, the error only
appears at runtime when `chat()` is actually called - harder to debug.

### Why claude-sonnet-4-6?

It is the best balance of speed, cost, and capability in the current Claude 4.x
family. Opus 4.7 is more capable but slower and more expensive - overkill for a
personal companion. Haiku 4.5 is cheaper but noticeably weaker at reasoning.
Sonnet is the right default.

### Why max_tokens 8192?

This is the maximum number of tokens in the *response*. 8192 gives room for long
answers without hitting a truncation limit. The context window (how much
conversation history fits) is separate and much larger.

### Why is the anthropic client synchronous inside an async method?

The `anthropic` SDK's default client is synchronous. Calling it from an async
function works but blocks the event loop during the API call. For a single-user
personal bot this is acceptable. If you need true async later, swap
`anthropic.Anthropic` for `anthropic.AsyncAnthropic` and add `await`.

---

## 3. Create a local llama.cpp stub

Create `src/birthdy/inference/llama_client.py`:

```python
import aiohttp


class LocalLlamaClient:

    def __init__(self, base_url: str = "http://localhost:8080"):
        self._base_url = base_url.rstrip("/")

    async def chat(self, messages: list[dict], system: str = "") -> str:
        if system:
            messages = [{"role": "system", "content": system}] + messages

        payload = {
            "model": "local",
            "messages": messages,
            "max_tokens": 8192,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
```

This calls the OpenAI-compatible endpoint that llama-server exposes. You already
verified this server works in Step 0. This stub is complete and functional - it
will work as-is when you point `LLAMA_SERVER_URL` at a running llama-server.

---

## 4. Create the client factory

Create `src/birthdy/inference/__init__.py` (replace the empty file):

```python
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
```

This is the only place that knows which backend is active. The bot imports
`get_inference_client()` and calls the returned object. Changing `INFERENCE_BACKEND`
in `.env` is all that is needed to switch.

---

## 5. Test it manually

Create a temporary test script at the project root (not inside src):

```python
# test_inference.py
import asyncio
from dotenv import load_dotenv

load_dotenv()

from birthdy.inference import get_inference_client

async def main():
    client = get_inference_client()
    messages = [{"role": "user", "content": "Say hello in one sentence."}]
    reply = await client.chat(messages, system="You are a friendly assistant.")
    print(reply)

asyncio.run(main())
```

Run it:

```bash
python test_inference.py
```

You should see a one-sentence hello from Claude. If you do, the inference layer
is working correctly.

Delete `test_inference.py` after the test - it is a scratch file, not part of
the project.

---

## What you have after this step

```
src/birthdy/inference/
  __init__.py       <- get_inference_client() factory
  client.py         <- InferenceClient abstract base class
  claude_client.py  <- ClaudeAPIClient (active now)
  llama_client.py   <- LocalLlamaClient (ready for the cluster)
```

Step 3 will add SQLite conversation state so the bot remembers what was said
earlier in the conversation - and across restarts.
