# Step 6 - Running Birthdy as a Background Service

## Goal

Run Birthdy as a systemd user service so it starts automatically, survives
terminal closes, and restarts itself if it crashes.

---

## Why systemd user services?

systemd is already managing your Gentoo system. User services run under your
own account (no root needed) and behave exactly like system services:

- Start on login or boot
- Restart automatically on crash
- Logs go to journald - inspectable with `journalctl`
- Controlled with `systemctl --user start/stop/status/restart`

Alternatives considered:
- **tmux/screen** - manual, does not survive reboots, no auto-restart
- **nohup** - no auto-restart, hard to manage
- **Docker** - overkill, adds complexity, no benefit for a single process

---

## How the bot works as a daemon

The polling loop inside `app.run_polling()` runs forever. systemd simply keeps
that process alive. The flow is:

```
systemd starts birthdy.service on login
  -> python3 -m birthdy runs
  -> bot polls Telegram every few seconds
  -> you send a message from your phone
  -> bot picks it up, calls Claude, replies
  -> loop continues forever
  -> if the process crashes, systemd restarts it within 5 seconds
```

Qdrant and Ollama also need to be running. We will handle them the same way.

---

## 1. Create the systemd user service directory

```bash
mkdir -p ~/.config/systemd/user
```

---

## 2. Create the Birthdy service file

Create `~/.config/systemd/user/birthdy.service`:

```ini
[Unit]
Description=Birthdy AI companion bot
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/wdha/proj/priv/birthdy
ExecStart=/home/wdha/proj/priv/birthdy/.venv/bin/python3 -m birthdy
Restart=on-failure
RestartSec=5
EnvironmentFile=/home/wdha/proj/priv/birthdy/.env

[Install]
WantedBy=default.target
```

### Why each setting matters

**`WorkingDirectory`** - the bot uses relative paths like `data/birthdy.db`.
Setting WorkingDirectory ensures those paths resolve correctly regardless of
where systemd launches the process from.

**`ExecStart` uses the venv python directly** - this avoids needing to activate
the virtualenv. Calling `.venv/bin/python3` directly gives you the venv's Python
with all installed packages, identical to running with the venv activated.

**`Restart=on-failure`** - restarts only on non-zero exit. If you stop it with
`systemctl --user stop birthdy`, it does not restart. If it crashes, it does.

**`RestartSec=5`** - waits 5 seconds before restarting. Prevents a rapid
crash loop if something is fundamentally wrong (e.g. bad API key).

**`EnvironmentFile`** - loads your `.env` file as environment variables. This
is how the service gets `TELEGRAM_BOT_TOKEN`, `ANTHROPIC_API_KEY`, etc. without
baking secrets into the service file.

**`After=network-online.target`** - waits for network to be available before
starting. The bot needs internet to reach Telegram and the Claude API.

---

## 3. Create the Qdrant service file

Qdrant needs to be running before Birthdy starts. Create
`~/.config/systemd/user/qdrant.service`:

```ini
[Unit]
Description=Qdrant vector database
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/wdha/tools/qdrant
ExecStart=/home/wdha/tools/qdrant/qdrant
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

---

## 4. Create the Ollama service file

Create `~/.config/systemd/user/ollama.service`:

```ini
[Unit]
Description=Ollama local model server
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/ollama serve
Restart=on-failure
RestartSec=5
Environment=OLLAMA_HOST=127.0.0.1:11434

[Install]
WantedBy=default.target
```

---

## 5. Update the Birthdy service to depend on Qdrant and Ollama

Update `~/.config/systemd/user/birthdy.service` to add the dependencies:

```ini
[Unit]
Description=Birthdy AI companion bot
After=network-online.target qdrant.service ollama.service
Requires=qdrant.service ollama.service
```

The full updated file:

```ini
[Unit]
Description=Birthdy AI companion bot
After=network-online.target qdrant.service ollama.service
Requires=qdrant.service ollama.service

[Service]
Type=simple
WorkingDirectory=/home/wdha/proj/priv/birthdy
ExecStart=/home/wdha/proj/priv/birthdy/.venv/bin/python3 -m birthdy
Restart=on-failure
RestartSec=5
EnvironmentFile=/home/wdha/proj/priv/birthdy/.env

[Install]
WantedBy=default.target
```

---

## 6. Enable and start the services

Reload systemd to pick up the new files:

```bash
systemctl --user daemon-reload
```

Enable all three services (so they start automatically on login):

```bash
systemctl --user enable qdrant.service
systemctl --user enable ollama.service
systemctl --user enable birthdy.service
```

Start them now:

```bash
systemctl --user start qdrant.service
systemctl --user start ollama.service
systemctl --user start birthdy.service
```

---

## 7. Verify everything is running

```bash
systemctl --user status qdrant.service
systemctl --user status ollama.service
systemctl --user status birthdy.service
```

All three should show `Active: active (running)`.

Check Birthdy's logs:

```bash
journalctl --user -u birthdy.service -f
```

You should see:

```
... [INFO] birthdy.bot.telegram_bot: Birthdy is running...
```

Send a message to your bot on Telegram to confirm it responds.

---

## 8. Enable lingering (optional but recommended)

By default, user services only run while you are logged in. If you want Birthdy
to run even when you are not logged into the laptop (e.g. it is running headless),
enable lingering:

```bash
loginctl enable-linger wdha
```

With lingering enabled, your user services start at boot and keep running even
with no active login session.

---

## Common management commands

```bash
# Check status
systemctl --user status birthdy

# Stop the bot
systemctl --user stop birthdy

# Restart the bot (e.g. after code changes)
systemctl --user restart birthdy

# View live logs
journalctl --user -u birthdy -f

# View last 50 log lines
journalctl --user -u birthdy -n 50

# Disable autostart
systemctl --user disable birthdy
```

---

## What you have after this step

Birthdy runs automatically in the background. You can:
- Close the terminal - bot keeps running
- Reboot the laptop - bot restarts automatically on login
- Deploy code changes with `systemctl --user restart birthdy`
- Debug issues with `journalctl --user -u birthdy -f`
