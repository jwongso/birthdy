# Step 5 - Vector Memory (Qdrant)

## Goal

Add long-term semantic memory using Qdrant so Birthdy can recall relevant
past conversations that have scrolled beyond the 40-message SQLite window.

---

## The problem with the 40-message limit

SQLite gives Birthdy a rolling window of the last 40 messages. This works well
for the current conversation but has a hard limit: if you talked about something
100 messages ago, Claude has no memory of it.

Vector memory solves this by storing every message as a semantic embedding -
a list of numbers that captures the *meaning* of the text. When you send a new
message, Birthdy searches the vector store for past messages with similar meaning
and injects the most relevant ones into the system prompt as context.

This gives Birthdy two layers of memory:

```
Layer 1 - Recent context (SQLite)
  The last 40 messages, verbatim, passed directly to Claude.
  Good for: following the current conversation thread.

Layer 2 - Long-term memory (Qdrant)
  Semantic search over all past messages.
  Good for: "remember when I told you about X?" three weeks ago.
```

---

## How embeddings work

An embedding model converts text into a fixed-size vector of numbers, e.g.:

```
"I work at Resideo as a Senior Software Engineer"
-> [0.023, -0.841, 0.312, 0.009, ..., -0.156]  (768 numbers)
```

Similar meanings produce vectors that are close together in space. "I am a
software engineer" and "I work as a developer" will be closer to each other
than either is to "I like pizza."

Qdrant stores these vectors and lets you search: "find the N past messages
most similar to this new message."

We use `nomic-embed-text` via Ollama to generate embeddings. It runs locally,
is free, and produces 768-dimensional vectors that work well for retrieval.

---

## 1. Start Ollama and pull the embedding model

If Ollama is not installed yet:

```bash
# Gentoo - check if available
which ollama
```

If not installed, download from ollama.com or:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Pull the embedding model:

```bash
ollama pull nomic-embed-text
```

Verify it works:

```bash
ollama list
```

You should see `nomic-embed-text` in the list.

---

## 2. Start Qdrant

Qdrant can run as a single binary with no Docker needed. Download it:

```bash
mkdir -p ~/tools/qdrant
cd ~/tools/qdrant
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-musl.tar.gz | tar xz
```

Run it (it will create a `./storage` directory for data):

```bash
cd ~/tools/qdrant
./qdrant
```

Qdrant runs on port 6333 by default. Leave it running in a separate terminal.
Verify it is up:

```bash
curl http://localhost:6333/healthz
```

Should return `{"title":"qdrant - vector search engine","version":"..."}`

---

## 3. Create the embedder module

Create `src/birthdy/memory/embedder.py`:

```python
import os
import aiohttp


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")


async def embed(text: str) -> list[float]:
    url = f"{OLLAMA_URL}/api/embed"
    payload = {"model": EMBED_MODEL, "input": text}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["embeddings"][0]
```

### Why Ollama for embeddings?

- Runs locally - no API cost, no rate limits, no internet required
- nomic-embed-text is specifically designed for retrieval tasks
- 768-dimensional vectors - small enough to be fast, large enough to be accurate
- The same Ollama instance can serve other models later

---

## 4. Create the Qdrant memory module

Create `src/birthdy/memory/qdrant_mem.py`:

```python
import os
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "birthdy_memory"
VECTOR_SIZE = 768


def _client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def init_collection() -> None:
    client = _client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


async def store_memory(
    session_id: str,
    role: str,
    content: str,
    vector: list[float],
    msg_id: int,
) -> None:
    client = _client()
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=msg_id,
                vector=vector,
                payload={
                    "session_id": session_id,
                    "role": role,
                    "content": content,
                },
            )
        ],
    )


async def search_memory(
    session_id: str,
    query_vector: list[float],
    limit: int = 5,
) -> list[dict]:
    client = _client()
    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="session_id",
                    match=MatchValue(value=session_id),
                )
            ]
        ),
        limit=limit,
        with_payload=True,
    )
    return [
        {"role": r.payload["role"], "content": r.payload["content"], "score": r.score}
        for r in results
    ]
```

### Why cosine distance?

Cosine similarity measures the angle between two vectors, ignoring their
magnitude. This is standard for text embeddings - two sentences about the same
topic will point in roughly the same direction in vector space regardless of
length.

### Why upsert with msg_id?

Using the SQLite message ID as the Qdrant point ID means:
- No duplicate embeddings if a message is stored twice
- You can correlate Qdrant results back to SQLite rows if needed
- Qdrant requires integer or UUID point IDs - SQLite autoincrement IDs fit perfectly

### Why filter by session_id?

So searches only return memories from the same chat. If you later add multiple
users, their memories stay separate.

---

## 5. Update memory __init__.py

Replace `src/birthdy/memory/__init__.py`:

```python
from .embedder import embed
from .qdrant_mem import init_collection, store_memory, search_memory
```

---

## 6. Update the state module to return message IDs

The Qdrant store needs the SQLite message ID. Update `src/birthdy/state/db.py`
to return the ID from `save_message`:

Change this function:

```python
def save_message(session_id: str, role: str, content: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, ts),
        )
```

To this:

```python
def save_message(session_id: str, role: str, content: str) -> int:
    ts = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, ts),
        )
        return cursor.lastrowid
```

---

## 7. Update the bot to use vector memory

Update `src/birthdy/bot/telegram_bot.py`:

Add the memory imports at the top (after the existing imports):

```python
from birthdy.memory import embed, init_collection, store_memory, search_memory
```

Add `init_collection()` to the `main()` function, after `init_db()`:

```python
def main() -> None:
    init_db()
    init_collection()
    ...
```

Replace `handle_message` with this updated version:

```python
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session_id = str(update.effective_chat.id)
    user_text = update.message.text

    msg_id = save_message(session_id, "user", user_text)

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing",
    )

    query_vector = await embed(user_text)
    await store_memory(session_id, "user", user_text, query_vector, msg_id)

    relevant = await search_memory(session_id, query_vector, limit=5)
    recent = load_history(session_id, limit=40)

    now = datetime.now(timezone.utc).strftime("%A, %d %B %Y %H:%M UTC")

    memory_block = ""
    if relevant:
        snippets = [f"- [{r['role']}]: {r['content']}" for r in relevant]
        memory_block = "\n\nRelevant past context:\n" + "\n".join(snippets)

    system = f"{SYSTEM_PROMPT}\n\nCurrent date and time: {now}{memory_block}"

    try:
        reply = await get_client().chat(recent, system=system)
    except Exception as e:
        logger.error("Inference error: %s", e)
        await update.message.reply_text("Sorry, something went wrong. Please try again.")
        return

    reply_id = save_message(session_id, "assistant", reply)
    reply_vector = await embed(reply)
    await store_memory(session_id, "assistant", reply, reply_vector, reply_id)

    await update.message.reply_text(reply)
```

---

## 8. Test it

Make sure both Ollama and Qdrant are running, then restart the bot:

```bash
python3 -m birthdy
```

Send a few messages, then use `/clear` to wipe the SQLite history. Then ask
about something you mentioned before the clear. Birthdy should still recall it
via Qdrant even though the recent history is empty.

Example:
1. Tell Birthdy: "My cat's name is Miso"
2. Chat for a while
3. `/clear`
4. Ask: "What is my cat's name?"

Birthdy should answer "Miso" from vector memory even after the clear.

---

## What you have after this step

```
src/birthdy/
  memory/
    __init__.py       <- exports embed, init_collection, store_memory, search_memory
    embedder.py       <- nomic-embed-text via Ollama
    qdrant_mem.py     <- Qdrant vector store
  bot/
    telegram_bot.py   <- updated with memory integration
  inference/          <- unchanged
  state/
    db.py             <- save_message now returns int ID
```

Birthdy now has two-layer memory: recent context from SQLite and long-term
semantic recall from Qdrant.
