# Step 3 - Conversation State (SQLite)

## Goal

Persist conversation history to SQLite so the bot remembers what was said
earlier in a session and across restarts.

---

## Why SQLite?

The simplest option that works. SQLite is:
- Built into Python's standard library - zero extra dependencies
- A single file on disk - easy to back up, inspect, or delete
- Durable - survives crashes and restarts
- Fast enough for a single-user bot with thousands of messages

Alternatives considered and rejected:
- **In-memory list** - lost on every restart, unacceptable
- **JSON file** - no concurrent write safety, messy to query
- **Redis** - requires a separate running process, overkill
- **PostgreSQL** - same problem, massive overkill for one user

---

## Data model

One table: `messages`

```
messages
  id         INTEGER PRIMARY KEY AUTOINCREMENT
  session_id TEXT    NOT NULL   -- groups messages into conversations
  role       TEXT    NOT NULL   -- "user" or "assistant"
  content    TEXT    NOT NULL   -- the message text
  created_at TEXT    NOT NULL   -- ISO8601 timestamp
```

### Why session_id?

A session groups messages that belong to one conversation. For now session_id
will be the Telegram chat_id (a number that uniquely identifies each chat).
This means each Telegram chat gets its own independent history. Later you could
add multiple sessions per user (e.g. "work mode" vs "personal mode") without
changing the schema.

### Why store role?

The Claude API requires messages in `[{"role": "user"|"assistant", "content":
"..."}]` format. Storing role directly means you can load history and pass it
straight to the API with no transformation.

---

## 1. Create the data directory

The database file lives outside `src/` because it is runtime data, not source
code. Create the directory:

```bash
mkdir -p data
```

It is already in `.gitignore` so the database will never be committed.

---

## 2. Create the state module

Create `src/birthdy/state/db.py`:

```python
import sqlite3
import os
from datetime import datetime, timezone
from pathlib import Path


DB_PATH = Path(os.environ.get("DB_PATH", "data/birthdy.db"))


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                role       TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                created_at TEXT    NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages (session_id, id)
        """)


def save_message(session_id: str, role: str, content: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, ts),
        )


def load_history(session_id: str, limit: int = 40) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def clear_history(session_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
```

---

## Why each function works the way it does

### `_connect()`

Opens a new connection each call. SQLite connections are cheap and this avoids
threading issues - each request gets its own connection, no shared state.

`conn.row_factory = sqlite3.Row` makes rows accessible by column name
(`r["role"]`) rather than index (`r[0]`), which is more readable.

### `init_db()`

`CREATE TABLE IF NOT EXISTS` is idempotent - safe to call on every startup.
The index on `(session_id, id)` makes `load_history` fast: SQLite can jump
directly to the right session's rows sorted by insertion order.

### `save_message()`

Uses `?` placeholders, never string formatting. This is mandatory - string
formatting SQL is vulnerable to SQL injection even in a personal bot.

Timestamps are stored as ISO8601 strings (`2026-05-09T10:30:00+00:00`).
SQLite has no native datetime type; ISO8601 strings sort correctly and are
human-readable when you inspect the DB directly.

### `load_history()`

The `ORDER BY id DESC LIMIT ?` trick: fetch the most recent N messages in
reverse order, then `reversed()` them back to chronological order. This is
more efficient than `ORDER BY id ASC` with an offset when the table is large.

`limit=40` gives Claude ~20 back-and-forth exchanges of context. More than
that and you risk hitting the context window limit for very long messages.
This is configurable per call.

### `clear_history()`

Lets the user start fresh without restarting the bot. The Telegram bot will
expose this as a `/clear` command later.

---

## 3. Update state __init__.py

Replace `src/birthdy/state/__init__.py` with:

```python
from .db import init_db, save_message, load_history, clear_history
```

---

## 4. Test it manually

Create `test_state.py` at the project root:

```python
from dotenv import load_dotenv
load_dotenv()

from birthdy.state import init_db, save_message, load_history, clear_history

init_db()

session = "test-session-1"

save_message(session, "user", "Hello, what is your name?")
save_message(session, "assistant", "My name is Birthdy, your personal AI companion.")
save_message(session, "user", "What can you help me with?")

history = load_history(session)
for msg in history:
    print(f"[{msg['role']}] {msg['content']}")

print("\nClearing history...")
clear_history(session)
print(f"After clear: {load_history(session)}")
```

Run it:

```bash
python3 test_state.py
```

Expected output:

```
[user] Hello, what is your name?
[assistant] My name is Birthdy, your personal AI companion.
[user] What can you help me with?

Clearing history...
After clear: []
```

Also verify the database file was created:

```bash
ls -lh data/birthdy.db
```

Delete the test script after:

```bash
rm test_state.py
```

---

## What you have after this step

```
src/birthdy/state/
  __init__.py   <- exports init_db, save_message, load_history, clear_history
  db.py         <- all SQLite logic

data/
  birthdy.db    <- created at runtime, not committed
```

Step 4 will wire inference + state together into a working Telegram bot.
