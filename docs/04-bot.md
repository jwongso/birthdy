# Step 4 - Telegram Bot

## Goal

Wire inference and state together into a running Telegram bot. After this step
you can open Telegram on your phone, send a message to your bot, and get a reply
from Claude that remembers the conversation.

---

## How python-telegram-bot works

The library uses an `Application` object that connects to Telegram's servers via
long polling - it repeatedly asks "any new messages?" and processes them as they
arrive. Each message type (text, commands, etc.) is handled by a registered
handler function.

The flow for each message:

```
User sends message on Telegram
  -> Telegram servers receive it
  -> Your bot polls and gets it
  -> Handler function is called
  -> Handler calls Claude via InferenceClient
  -> Handler sends reply back to Telegram
  -> User sees reply
```

---

## 1. Create the bot module

Create `src/birthdy/bot/telegram_bot.py`:

```python
import logging
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from birthdy.state import init_db, save_message, load_history, clear_history
from birthdy.inference import get_inference_client

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Birthdy, a personal AI companion for Juniarto.
You are helpful, honest, and direct. You remember the conversation history
provided to you. Keep responses concise unless asked for detail."""

_inference_client = None


def get_client():
    global _inference_client
    if _inference_client is None:
        _inference_client = get_inference_client()
    return _inference_client


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi Juniarto! I am Birthdy, your personal AI companion. "
        "Send me a message to start chatting. Use /clear to reset our conversation."
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session_id = str(update.effective_chat.id)
    clear_history(session_id)
    await update.message.reply_text("Conversation cleared. Starting fresh.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session_id = str(update.effective_chat.id)
    user_text = update.message.text

    save_message(session_id, "user", user_text)

    history = load_history(session_id, limit=40)

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing",
    )

    try:
        reply = await get_client().chat(history, system=SYSTEM_PROMPT)
    except Exception as e:
        logger.error("Inference error: %s", e)
        await update.message.reply_text("Sorry, something went wrong. Please try again.")
        return

    save_message(session_id, "assistant", reply)
    await update.message.reply_text(reply)


def main() -> None:
    init_db()

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Birthdy is running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
```

---

## Why each part works the way it does

### Lazy client initialization (`get_client`)

`_inference_client` is created once on the first message, not at import time.
This means if the API key is wrong you get a clear error on the first message,
not a cryptic import failure.

### `session_id = str(update.effective_chat.id)`

Telegram gives each chat a unique integer ID. Using it as `session_id` means:
- A private chat with your bot has one history
- If you add the bot to a group chat later, that group gets its own separate history
- Across restarts, the same chat resumes the same history automatically

### `save_message` before calling inference

The user message is saved to the DB before calling Claude. If Claude fails or
times out, the user message is still recorded. On the next message, history will
include the unanswered question, giving Claude context that something was asked.

### `send_chat_action("typing")`

Shows the "typing..." indicator in Telegram while Claude is thinking. Without
this, the user sees nothing for several seconds and may think the bot is broken.
It is a small touch that makes the bot feel responsive.

### `load_history` after `save_message`

History is loaded after saving the current message so the current message is
included in what gets sent to Claude. The order is:

1. Save user message
2. Load history (includes the message just saved)
3. Send full history to Claude
4. Save Claude's reply
5. Send reply to user

### Error handling in `handle_message`

If Claude throws (network error, rate limit, etc.), the user gets a friendly
message instead of silence. The user message is already saved, so they can
just try again.

### `filters.TEXT & ~filters.COMMAND`

This handler only fires for plain text messages. The `~filters.COMMAND` part
excludes messages starting with `/` so commands like `/clear` are not also
processed as chat messages.

---

## 2. Create the entry point

Create `src/birthdy/bot/__init__.py` (replace the empty file):

```python
from .telegram_bot import main
```

Create `src/birthdy/__main__.py`:

```python
from birthdy.bot import main

main()
```

This lets you run the bot with:

```bash
python3 -m birthdy
```

---

## 3. Run the bot

Make sure your `.env` has both keys set:

```
TELEGRAM_BOT_TOKEN=your_token_here
ANTHROPIC_API_KEY=your_key_here
INFERENCE_BACKEND=claude
```

Then run:

```bash
python3 -m birthdy
```

You should see:

```
2026-05-09 10:00:00 [INFO] birthdy.bot.telegram_bot: Birthdy is running...
```

---

## 4. Test it

Open Telegram on your phone or desktop, find your bot by its username, and:

1. Send `/start` - you should get the welcome message
2. Send `Hello, who are you?` - Claude should reply as Birthdy
3. Send a follow-up like `What did I just ask you?` - Claude should remember
4. Send `/clear` - history is wiped
5. Send `What did I just ask you?` again - Claude should not remember

If all five work, the bot is fully functional.

---

## 5. Stop the bot

Press `Ctrl+C` in the terminal. The bot will shut down cleanly.

---

## What you have after this step

```
src/birthdy/
  __main__.py         <- entry point: python3 -m birthdy
  bot/
    __init__.py
    telegram_bot.py   <- all Telegram logic
  inference/          <- from Step 2
  state/              <- from Step 3
```

Step 5 will add Qdrant vector memory so Birthdy can search across older
conversations that have scrolled out of the 40-message window.
