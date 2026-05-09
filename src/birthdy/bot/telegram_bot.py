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


def main() -> None:
    init_db()
    init_collection()

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
