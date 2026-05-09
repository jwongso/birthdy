# Birthdy

A personal AI companion accessible via Telegram. Supports swappable inference
backends (Claude API or local llama.cpp), persistent conversation history, and
long-term semantic memory via Qdrant.

## Features

- Telegram bot interface with `/start` and `/clear` commands
- Pluggable inference backend: Claude API today, local llama.cpp tomorrow
- SQLite conversation history - persists across restarts
- Qdrant vector memory - recalls past conversations beyond the rolling window
- Nomic-embed-text embeddings via Ollama for semantic search
- Time-aware responses - knows the current date and time
- Runs as a systemd user service

## Stack

| Component | Technology |
|-----------|-----------|
| Interface | python-telegram-bot |
| Inference (cloud) | Anthropic Claude API |
| Inference (local) | llama.cpp server |
| Conversation state | SQLite |
| Vector memory | Qdrant |
| Embeddings | nomic-embed-text via Ollama |
| Config | python-dotenv |

## Project structure

```
src/birthdy/
  bot/           - Telegram handlers
  inference/     - InferenceClient interface + Claude and llama.cpp backends
  memory/        - Qdrant vector store and Ollama embedder
  state/         - SQLite conversation persistence
docs/            - Step-by-step build documentation
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
# edit .env with your keys
python3 -m birthdy
```

## Configuration

Copy `.env.example` to `.env` and fill in:

```
INFERENCE_BACKEND=claude        # or "llama"
ANTHROPIC_API_KEY=...
TELEGRAM_BOT_TOKEN=...
LLAMA_SERVER_URL=http://localhost:8080
QDRANT_URL=http://localhost:6333
OLLAMA_URL=http://localhost:11434
```

## Switching to local inference

Start a llama.cpp server:

```bash
./llama-server --hf-repo bartowski/Qwen_Qwen3-8B-GGUF \
  --hf-file Qwen_Qwen3-8B-Q4_K_M.gguf \
  --n-gpu-layers 999 --ctx-size 8192 --port 8080
```

Set in `.env`:

```
INFERENCE_BACKEND=llama
LLAMA_SERVER_URL=http://localhost:8080
```

Restart the bot. No code changes required.

## Documentation

Step-by-step build docs are in `docs/`:

- `00-architecture.md` - Design decisions and component overview
- `01-setup.md` - Python environment setup
- `02-inference.md` - Inference layer and backend abstraction
- `03-state.md` - SQLite conversation state
- `04-bot.md` - Telegram bot
- `05-memory.md` - Qdrant vector memory
- `06-daemon.md` - Running as a systemd user service
- `07-model-selection.md` - Choosing a base model for fine-tuning
- `08-rag-vs-finetune.md` - When to use RAG, when to fine-tune, when both
- `12-evaluation.md` - Benchmarking and evaluation framework

## License

MIT
