# Step 1 - Python Environment Setup

## Goal

Create an isolated Python environment for Birthdy, install all dependencies, and
establish the config file pattern that keeps secrets out of the codebase.

---

## Why a virtualenv?

Your Gentoo system has Python 3.13 managed by Portage. Installing packages globally
with pip would conflict with Portage's package management and could break system tools.
A virtualenv creates a completely separate Python environment under the project
directory - its own pip, its own site-packages, no conflict with the system.

Rule: never run `pip install` outside a virtualenv on Gentoo.

---

## 1. Create the project directory structure

From `~/proj/priv/birthdy`, create the source layout:

```bash
mkdir -p src/birthdy/{bot,inference,memory,state}
mkdir -p tests
touch src/birthdy/__init__.py
touch src/birthdy/bot/__init__.py
touch src/birthdy/inference/__init__.py
touch src/birthdy/memory/__init__.py
touch src/birthdy/state/__init__.py
```

Why this layout? The `src/` layout is the modern Python packaging standard. It
prevents accidental imports of your source tree before it is properly installed,
which catches import errors early rather than at runtime in production.

---

## 2. Create the virtualenv

```bash
python3 -m venv .venv
```

This creates `.venv/` inside the birthdy project directory. The dot prefix keeps it
hidden from casual `ls` and signals it is not source code.

Activate it:

```bash
source .venv/bin/activate
```

Your prompt will change to show `(.venv)`. Every `python` and `pip` command now
refers to the virtualenv, not the system Python.

To deactivate later: just run `deactivate`.

---

## 3. Upgrade pip

```bash
pip install --upgrade pip
```

The pip bundled with the venv is often outdated. An old pip can fail to resolve
dependencies correctly or miss binary wheels. Always upgrade it first.

---

## 4. Install dependencies

```bash
pip install anthropic "python-telegram-bot[job-queue]" qdrant-client python-dotenv aiohttp
```

What each package does and why:

| Package | Purpose | Why this one |
|---------|---------|--------------|
| `anthropic` | Claude API client | Official SDK, handles auth, retries, streaming |
| `python-telegram-bot[job-queue]` | Telegram bot framework | Mature, async-first, well-documented. `job-queue` extra adds scheduled task support |
| `qdrant-client` | Qdrant vector DB client | Official client, supports local in-process mode (no server needed) |
| `python-dotenv` | Load `.env` files | Zero-dependency, keeps config out of code |
| `aiohttp` | Async HTTP client | Needed to call llama.cpp server API later; also used internally by python-telegram-bot |

---

## 5. Create requirements.txt

```bash
pip freeze > requirements.txt
```

`pip freeze` outputs every installed package with its exact pinned version. This
ensures that when you set up the cluster in Jan/Feb 2027, you get identical
dependency versions - no surprise breakage from a transitive dependency updating.

Check it was created:

```bash
cat requirements.txt
```

You should see ~20-30 lines of `package==version` entries.

---

## 6. Create pyproject.toml

Create the file `pyproject.toml` in the birthdy root with this content:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "birthdy"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]
```

Why bother with this? It lets you run `pip install -e .` (editable install), which
adds `src/birthdy` to the Python path so your imports work as `from birthdy.inference
import ...` rather than fiddling with `sys.path`. It also means tests can import
your code naturally.

Install it in editable mode now:

```bash
pip install -e .
```

---

## 7. Create .env.example

Create `.env.example` in the birthdy root:

```
# Inference backend: "claude" or "llama"
INFERENCE_BACKEND=claude

# Required when INFERENCE_BACKEND=claude
ANTHROPIC_API_KEY=your_anthropic_key_here

# Required when INFERENCE_BACKEND=llama
LLAMA_SERVER_URL=http://localhost:8080

# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token_here

# Qdrant (local file path for embedded mode)
QDRANT_PATH=./data/qdrant

# Ollama embedding server
OLLAMA_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text
```

Now create the actual `.env` by copying the example:

```bash
cp .env.example .env
```

The `.env` file will hold your real secrets. It must never be committed to git.

---

## 8. Create .gitignore

```bash
cat > .gitignore << 'EOF'
.venv/
.env
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
data/
*.db
EOF
```

Key entries:
- `.venv/` - never commit the virtualenv (it is OS and Python version specific)
- `.env` - never commit secrets
- `data/` - Qdrant storage and SQLite DB files are runtime state, not source

---

## 9. Verify the setup

Run this to confirm everything is wired up correctly:

```bash
python3 -c "import anthropic; import telegram; import qdrant_client; import dotenv; print('all imports ok')"
```

You should see: `all imports ok`

If any import fails, pip install that package again.

---

## What you have after this step

```
birthdy/
  .venv/              <- Python environment (not committed)
  .env                <- your secrets (not committed)
  .env.example        <- template (committed)
  .gitignore
  pyproject.toml
  requirements.txt
  docs/
    00-architecture.md
    01-setup.md       <- this file
  src/
    birthdy/
      __init__.py
      bot/__init__.py
      inference/__init__.py
      memory/__init__.py
      state/__init__.py
  tests/
```

No Python code yet - just the skeleton. Step 2 will add the first real file:
the abstract `InferenceClient` base class and the `ClaudeAPIClient` implementation.
