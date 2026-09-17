# Walton-LoFly (Replit)

Battlesnake webhook for Funathon. **Full MaleCNS brain** (211k neurons, ~26M synapses) + dial gains. Deploy here, then register the public URL on play.battlesnake.com.

## What’s in this package

- `main.py` — Battlesnake webhook (`/move` blends policy dials with full-brain readout)
- `full_brain.py` + `full_brain_cache/` — fixed connectome (no subgraph)
- `ladder/dials/leader.json` — promoted dials (`wiring=full_malecns_v1`, `brain_blend≈0.55`)
- `cortex.py` / `dials.py` / `game_log.py`

## Deploy order

1. Open / pull this Replit App.
2. Secrets (optional): `FS_AVATAR_NAME=Walton-LoFly`, `FS_AVATAR_COLOR=#111111`, `FS_AVATAR_AUTHOR=Walton`.
3. Env should include `FS_AVATAR_BRAIN=1` (set in `.replit`). First boot loads ~100MB cache (~1–2s).
4. **Run**, then **Publish** so the URL stays awake.
5. Copy the public URL → [Battlesnake snakes](https://play.battlesnake.com/account/battlesnakes):
   - **Name:** `Walton-LoFly` (must include Walton)
   - **URL:** that Replit URL
6. Hit the URL in a browser before each tournament so Replit is awake.

## Sync from fly-brain (Cursor / local)

```bash
cd fly_console/activities/fs-avatar
bash scripts/deploy_replit.sh
# then on Replit: Pull / Republish
```

## Local check

```bash
pip install -r requirements.txt
FS_AVATAR_BRAIN=1 python main.py
curl -s http://127.0.0.1:${PORT:-8001}/dev/status
# expect "brain": true
```
