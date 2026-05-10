# Birthdy

A personal AI companion accessible via Telegram. Supports swappable inference
backends (Claude API or local llama.cpp), persistent conversation history,
long-term semantic memory via Qdrant, and a thinking mode for deep reasoning.

## Features

- Telegram bot interface with `/start`, `/clear`, and `/think` commands
- Pluggable inference backend: Claude API or local llama.cpp (Qwen3-8B)
- `/think` command enables chain-of-thought reasoning mode
- SQLite conversation history - persists across restarts
- Qdrant vector memory - recalls past conversations beyond the rolling window
- Nomic-embed-text embeddings via Ollama for semantic search
- Time-aware responses - knows the current date and time
- Long response splitting - automatically splits replies that exceed Telegram's limit
- Context overflow recovery - retries with fewer history messages if context is exceeded
- ASCII-safe output - normalises smart quotes, em dashes, and emoji
- Runs as a systemd user service

## Stack

| Component | Technology |
|-----------|-----------|
| Interface | python-telegram-bot |
| Inference (cloud) | Anthropic Claude API |
| Inference (local) | llama.cpp server (Qwen3-8B Q4_K_M) |
| Conversation state | SQLite |
| Vector memory | Qdrant |
| Embeddings | nomic-embed-text via Ollama |
| Config | python-dotenv |

## Project structure

```
src/birthdy/
  bot/           - Telegram handlers and message utilities
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
INFERENCE_BACKEND=llama         # or "claude"
TELEGRAM_BOT_TOKEN=...
LLAMA_SERVER_URL=http://localhost:8080
QDRANT_URL=http://localhost:6333
OLLAMA_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text

# Required only when INFERENCE_BACKEND=claude
ANTHROPIC_API_KEY=...
```

## Local inference setup

Download a GGUF model and start llama-server:

```bash
llama-server \
  --model Qwen_Qwen3-8B-Q4_K_M.gguf \
  --n-gpu-layers 999 \
  --ctx-size 12288 \
  --port 8080
```

Set `INFERENCE_BACKEND=llama` in `.env` and restart the bot. No code changes
required.

### Thinking mode

With a Qwen3 model loaded, send `/think <question>` in Telegram to enable
chain-of-thought reasoning. The model reasons internally before responding.
Useful for verification tasks, complex analysis, or anything where you want
more careful reasoning than a standard reply.

## Running as a service

Three systemd user services cover the full stack:

```
llama-server.service   - llama.cpp inference on port 8080
qdrant.service         - Qdrant vector database on port 6333
birthdy.service        - Telegram bot (requires both of the above)
```

```bash
systemctl --user enable --now llama-server.service
systemctl --user enable --now qdrant.service
systemctl --user enable --now birthdy.service
```

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
- `09-knowledge-distillation.md` - Teaching a small model with a large one
- `10-lora-finetuning.md` - LoRA fine-tuning pipeline
- `11-mcp-server.md` - MCP server for domain tools
- `12-evaluation.md` - Benchmarking and evaluation framework

## License

MIT
