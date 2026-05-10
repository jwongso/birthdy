import logging
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from birthdy.state import init_db, save_message, load_history, clear_history
from birthdy.inference import get_inference_client
from birthdy.memory import embed, init_collection, store_memory, search_memory

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Birthdy, a personal AI companion for Juni.
You are helpful, honest, and direct. You remember the conversation history
provided to you.

Character rules:
- Use ASCII characters only. No smart quotes (" " ' '), em dashes (-), en dashes,
  ellipsis (...), or any other Unicode punctuation. Use straight quotes, hyphens,
  and three plain dots instead.

Response length rules:
- Keep responses under 800 words whenever possible.
- If the answer genuinely requires more than 800 words, split it into clearly
  labelled parts: start the first part with "Part 1/2:" and the second with
  "Part 2/2:". Never split into more than two parts.
- Never truncate mid-sentence. Always finish your thought."""

_inference_client = None
_processed_updates: set[int] = set()


def get_client():
    global _inference_client
    if _inference_client is None:
        _inference_client = get_inference_client()
    return _inference_client


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi Juni! I am Birthdy, your personal AI companion. "
        "Send me a message to start chatting. Use /clear to reset our conversation."
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session_id = str(update.effective_chat.id)
    clear_history(session_id)
    await update.message.reply_text("Conversation cleared. Starting fresh.")


TELEGRAM_MAX_LEN = 4096
HISTORY_MSG_MAX_CHARS = 1000

_ASCII_REPLACEMENTS = {
    "—": "-",   # em dash
    "–": "-",   # en dash
    "‘": "'",   # left single quote
    "’": "'",   # right single quote
    "“": '"',   # left double quote
    "”": '"',   # right double quote
    "…": "...", # ellipsis
    "•": "-",   # bullet
    "✓": "[ok]",  # checkmark
    "✔": "[ok]",
    "❌": "[x]",   # cross mark
    "⚠": "[!]",   # warning
}


def to_ascii(text: str) -> str:
    for char, replacement in _ASCII_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    # strip remaining non-ASCII, replace emoji blocks with empty string
    return text.encode("ascii", errors="ignore").decode("ascii")


async def send_long_message(update: Update, text: str) -> None:
    if len(text) <= TELEGRAM_MAX_LEN:
        await update.message.reply_text(text)
        return
    # split on paragraph boundaries where possible
    paragraphs = text.split("\n\n")
    chunk = ""
    for para in paragraphs:
        candidate = chunk + ("\n\n" if chunk else "") + para
        if len(candidate) <= TELEGRAM_MAX_LEN:
            chunk = candidate
        else:
            if chunk:
                await update.message.reply_text(chunk)
            # paragraph itself exceeds limit - split hard
            while len(para) > TELEGRAM_MAX_LEN:
                await update.message.reply_text(para[:TELEGRAM_MAX_LEN])
                para = para[TELEGRAM_MAX_LEN:]
            chunk = para
    if chunk:
        await update.message.reply_text(chunk)


async def think_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.update_id in _processed_updates:
        return
    _processed_updates.add(update.update_id)

    if not context.args:
        await update.message.reply_text("Usage: /think <your question>")
        return

    session_id = str(update.effective_chat.id)
    user_text = " ".join(context.args)
    user_text_short = user_text[:HISTORY_MSG_MAX_CHARS] + "...[truncated]" if len(user_text) > HISTORY_MSG_MAX_CHARS else user_text

    msg_id = save_message(session_id, "user", user_text_short)

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing",
    )

    query_vector = await embed(user_text)
    await store_memory(session_id, "user", user_text_short, query_vector, msg_id)

    relevant = await search_memory(session_id, query_vector, limit=3)
    recent = load_history(session_id, limit=10)

    now = datetime.now(timezone.utc).strftime("%A, %d %B %Y %H:%M UTC")

    memory_block = ""
    if relevant:
        snippets = [f"- [{r['role']}]: {r['content']}" for r in relevant]
        memory_block = "\n\nRelevant past context:\n" + "\n".join(snippets)

    system = f"{SYSTEM_PROMPT}\n\nCurrent date and time: {now}{memory_block}"

    reply = None
    for history_limit in [recent, recent[-5:], recent[-2:], []]:
        try:
            reply = await get_client().chat(history_limit, system=system, thinking=True)
            break
        except RuntimeError as e:
            if "exceed_context_size" in str(e):
                logger.warning("Context overflow, retrying with fewer messages")
                continue
            logger.error("Inference error: %s", e)
            await update.message.reply_text("Sorry, something went wrong. Please try again.")
            return
        except Exception as e:
            logger.error("Inference error: %s", e)
            await update.message.reply_text("Sorry, something went wrong. Please try again.")
            return

    if not reply:
        await update.message.reply_text("Sorry, the request is too large even with minimal history.")
        return

    reply = to_ascii(reply)
    reply_short = reply[:HISTORY_MSG_MAX_CHARS] + "...[truncated]" if len(reply) > HISTORY_MSG_MAX_CHARS else reply
    reply_id = save_message(session_id, "assistant", reply_short)
    reply_vector = await embed(reply_short)
    await store_memory(session_id, "assistant", reply_short, reply_vector, reply_id)

    await send_long_message(update, reply)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.update_id in _processed_updates:
        return
    _processed_updates.add(update.update_id)

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
    recent = load_history(session_id, limit=10)

    now = datetime.now(timezone.utc).strftime("%A, %d %B %Y %H:%M UTC")

    memory_block = ""
    if relevant:
        snippets = [f"- [{r['role']}]: {r['content']}" for r in relevant]
        memory_block = "\n\nRelevant past context:\n" + "\n".join(snippets)

    system = f"{SYSTEM_PROMPT}\n\nCurrent date and time: {now}{memory_block}"

    reply = None
    for history_limit in [recent, recent[-5:], recent[-2:], []]:
        try:
            reply = await get_client().chat(history_limit, system=system)
            break
        except RuntimeError as e:
            if "exceed_context_size" in str(e):
                logger.warning("Context overflow, retrying with fewer messages")
                continue
            logger.error("Inference error: %s", e)
            await update.message.reply_text("Sorry, something went wrong. Please try again.")
            return
        except Exception as e:
            logger.error("Inference error: %s", e)
            await update.message.reply_text("Sorry, something went wrong. Please try again.")
            return

    if not reply:
        await update.message.reply_text("Sorry, the request is too large even with minimal history.")
        return

    reply = to_ascii(reply)
    reply_id = save_message(session_id, "assistant", reply)
    reply_vector = await embed(reply)
    await store_memory(session_id, "assistant", reply, reply_vector, reply_id)

    await send_long_message(update, reply)


def main() -> None:
    init_db()
    init_collection()

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("think", think_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Birthdy is running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
