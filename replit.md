# Project setup

This project is a standard-library-only Python Battlesnake webhook.

## Run on Replit

- Main workflow: `PORT=5000 python main.py`
- The server binds to `0.0.0.0` and uses Replit's required web preview port.
- No package installation is required; `requirements.txt` is intentionally empty.

## Optional configuration

The app works with built-in defaults. These environment variables can customize its Battlesnake identity:

- `FS_AVATAR_NAME`
- `FS_AVATAR_COLOR`
- `FS_AVATAR_AUTHOR`
- `FS_AVATAR_HEAD`
- `FS_AVATAR_TAIL`

## Webhook endpoints

- `GET /` — Battlesnake appearance and identity
- `POST /start` — game start
- `POST /move` — choose a move
- `POST /end` — game end
- `GET /dev/status` — development health and last decision

After verifying the development preview, publish the app and register its published URL at play.battlesnake.com.