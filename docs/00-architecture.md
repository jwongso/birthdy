# Birthdy - Architecture Overview

## What is Birthdy?

Birthdy is a personal AI companion accessible via Telegram. It can hold long-running
conversations, remember things about you across sessions, and later run entirely on a
local hardware cluster (AMD Ryzen AI Max+ 395 nodes). For now it runs on a Gentoo
laptop using the Claude API as the inference backend.

---

## Core design principles

### 1. Swappable inference backend

The single most important architectural decision is to **never call the Claude API
directly from application code**. Instead, all inference goes through an abstract
`InferenceClient` interface with exactly one method:

```
chat(messages, system_prompt) -> str
```

Today that interface is backed by `ClaudeAPIClient`. When the Strix Halo cluster
arrives in Jan/Feb 2027, you replace it with `LocalLlamaClient` that points at a
llama.cpp server. The rest of the codebase does not change at all.

**Why this matters:** If you hardcode `anthropic.Anthropic().messages.create(...)` in
ten places, migrating to local inference means finding and rewriting all ten. With the
interface, it is one line in a config file.

### 2. Separation of concerns

The project is split into four distinct layers, each with a single responsibility:

```
bot/          - Telegram interface only (receive message, send reply)
inference/    - LLM calls only (no Telegram, no DB, no vector search)
memory/       - Qdrant vector search only (no LLM, no Telegram)
state/        - SQLite conversation history only (no LLM, no vector search)
```

Each layer can be tested, swapped, or debugged in isolation.

### 3. No 24/7 inference requirement

The laptop is not always on. Birthdy must handle this gracefully:
- Conversation state is persisted to SQLite so nothing is lost on shutdown
- The Telegram bot buffers messages; they are processed when the bot next starts
- Qdrant runs in-process from a local data directory, no separate daemon required

### 4. Progressive enhancement

Start with the simplest thing that works, then add layers:

```
Step 1: Telegram echo bot (no LLM)         <- verify bot token works
Step 2: Add ClaudeAPIClient                <- verify API key and inference work
Step 3: Add SQLite conversation history    <- persist context across restarts
Step 4: Add Qdrant + embedding             <- long-term semantic memory
Step 5: Add LoRA fine-tuning hooks         <- future, when cluster arrives
```

Each step produces something runnable. You never have a half-built system.

---

## Component map

```
Telegram
  |
  v
bot/telegram_bot.py          <- python-telegram-bot, async handlers
  |
  +-> state/db.py            <- SQLite: load/save conversation history
  |
  +-> memory/qdrant_client.py <- Qdrant: semantic search over past interactions
  |        |
  |        +-> memory/embedder.py  <- nomic-embed-text via Ollama HTTP API
  |
  +-> inference/client.py    <- InferenceClient abstract base class
           |
           +-> inference/claude_client.py   <- ClaudeAPIClient (today)
           +-> inference/llama_client.py    <- LocalLlamaClient (future)
```

---

## Technology choices and why

| Component | Choice | Reason |
|-----------|--------|--------|
| Telegram interface | python-telegram-bot | Mature, async, well-documented |
| Inference (now) | Claude API (claude-sonnet-4-6) | Best reasoning, available immediately |
| Inference (future) | llama.cpp server mode | Runs on Strix Halo GPUs, same HTTP API |
| Conversation state | SQLite | Zero-dependency, single file, survives restarts |
| Vector DB | Qdrant (local mode) | CPU-only ~500MB RAM, no Docker needed, Python client |
| Embeddings | nomic-embed-text via Ollama | Free, runs locally, good quality for retrieval |
| Language | Python 3.11+ | Best ecosystem for LLM tooling |
| Config | python-dotenv (.env file) | Simple, keeps secrets out of code |

---

## What we are NOT doing (and why)

- **No FastAPI/web layer** - Telegram is the only interface. Adding a web layer now
  is scope creep.
- **No Docker** - The laptop is a dev machine. Docker adds complexity with no benefit
  until the cluster stage.
- **No Kubernetes/Celery/task queues** - Single user, single bot instance. A queue
  is not needed.
- **No Redis** - SQLite handles all the state we need. Redis adds an extra process.

---

## File layout (target)

```
birthdy/
  docs/
    00-architecture.md      <- this file
    01-setup.md             <- environment setup steps
    02-telegram-bot.md      <- Step 1 walkthrough
    03-inference.md         <- Step 2 walkthrough
    04-state.md             <- Step 3 walkthrough
    05-memory.md            <- Step 4 walkthrough
  src/
    birthdy/
      __init__.py
      bot/
        telegram_bot.py
      inference/
        client.py           <- abstract base class
        claude_client.py
        llama_client.py     <- stub for future
      memory/
        embedder.py
        qdrant_mem.py
      state/
        db.py
  tests/
  .env.example
  requirements.txt
  README.md
```
