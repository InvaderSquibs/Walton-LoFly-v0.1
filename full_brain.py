"""
FS-Avatar full MaleCNS rate network — every annotated neuron, full synapse map.

No subgraph. Board signals drive annotation-defined channels; descending pools
vote on moves. Breedable dials are gains/biases on this fixed substrate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent


def _resolve_cache() -> Path:
    env = os.environ.get("FS_AVATAR_BRAIN_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    candidates = [
        HERE / "full_brain_cache",  # Replit / colocated with main.py
        HERE.parents[2] / "full_brain_cache",  # …/fly-brain from activities/fs-avatar
    ]
    for c in candidates:
        try:
            if (c / "W.npz").is_file() and (c / "channels.json").is_file():
                return c
        except OSError:
            continue
    return HERE / "full_brain_cache"


CACHE = _resolve_cache()

_state: Dict[str, Any] = {}


def cache_ready(path: Optional[Path] = None) -> bool:
    root = path or CACHE
    return (root / "W.npz").is_file() and (root / "channels.json").is_file()


def brain_enabled() -> bool:
    raw = (os.environ.get("FS_AVATAR_BRAIN") or "").strip().lower()
    if raw in ("0", "false", "off", "no", "-"):
        return False
    if raw in ("1", "true", "on", "yes", "full"):
        return cache_ready()
    # Auto: on when cache exists
    return cache_ready()


def load(force: bool = False) -> Dict[str, Any]:
    global _state, CACHE
    if _state and not force:
        return _state
    CACHE = _resolve_cache()
    if not cache_ready():
        raise FileNotFoundError(
            f"Full-brain cache missing at {CACHE}. Run: python3 build_full_brain_cache.py"
        )
    W = sparse.load_npz(CACHE / "W.npz")
    if not sparse.isspmatrix_csr(W):
        W = W.tocsr()
    # Row-normalize so |row| sums to ≤1 — keeps full-brain Euler stable at 26M edges
    abs_sum = np.asarray(np.abs(W).sum(axis=1)).ravel().astype(np.float32)
    scale = (1.0 / np.maximum(abs_sum, 1.0)).astype(np.float32)
    W = sparse.diags(scale) @ W
    W = W.tocsr().astype(np.float32)
    channels = json.loads((CACHE / "channels.json").read_text(encoding="utf-8"))
    move_pools = json.loads((CACHE / "move_pools.json").read_text(encoding="utf-8"))
    meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))
    ch_idx = {k: np.array(v, dtype=np.int32) for k, v in channels.items()}
    mv_idx = {k: np.array(v, dtype=np.int32) for k, v in move_pools.items()}
    _state = {
        "W": W,
        "N": int(meta["n_neurons"]),
        "channels": ch_idx,
        "move_pools": mv_idx,
        "meta": meta,
    }
    return _state


def _drive(I: np.ndarray, idxs: np.ndarray, rate: float) -> None:
    if idxs.size and rate != 0.0:
        I[idxs] += np.float32(rate)


def simulate(
    drives: Dict[str, float],
    dials: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Rate Euler on full W: r <- r + dt * (-r/tau + relu(W @ r + I)).

    drives: channel name → base Hz-ish input (before dial gains).
    Returns (move_rates, channel_mean_rates).
    """
    d = dials or {}
    st = load()
    W: sparse.csr_matrix = st["W"]
    N = st["N"]
    ch = st["channels"]
    pools = st["move_pools"]

    tau = float(d.get("brain_tau", 8.0))
    dt = float(d.get("brain_dt", 1.0))
    steps = int(d.get("brain_steps", 12))
    w_scale = float(d.get("brain_w_scale", 0.35))
    r_max = float(d.get("brain_r_max", 250.0))
    gain = {
        "ALPN": float(d.get("brain_gain_alpn", 1.0)),
        "ORN": float(d.get("brain_gain_orn", 1.0)),
        "MBON": float(d.get("brain_gain_mbon", 1.0)),
        "DAN": float(d.get("brain_gain_dan", 1.0)),
        "Kenyon_Cell": float(d.get("brain_gain_kc", 1.0)),
        "courtship_pC1": float(d.get("brain_gain_court", 1.0)),
        "courtship_vPR6": float(d.get("brain_gain_court", 1.0)),
        "courtship_TN1": float(d.get("brain_gain_court", 1.0)),
        "visual_L1": float(d.get("brain_gain_visual", 1.0)),
        "visual_L2": float(d.get("brain_gain_visual", 1.0)),
        "visual_L3": float(d.get("brain_gain_visual", 1.0)),
    }

    I = np.zeros(N, dtype=np.float32)
    for name, base in drives.items():
        idxs = ch.get(name)
        if idxs is None or not idxs.size:
            continue
        g = gain.get(name, 1.0)
        _drive(I, idxs, float(base) * g)

    r = np.zeros(N, dtype=np.float32)
    Ww = W * np.float32(w_scale)
    for _ in range(max(1, steps)):
        inp = Ww @ r + I
        np.maximum(inp, 0.0, out=inp)
        r += np.float32(dt) * (-r / np.float32(tau) + inp)
        np.clip(r, 0.0, r_max, out=r)

    def mean_of(idxs: np.ndarray) -> float:
        return float(r[idxs].mean()) if idxs.size else 0.0

    move_rates = {
        m: mean_of(pools[m]) * float(d.get(f"brain_readout_{m}", 1.0))
        for m in ("up", "down", "left", "right")
    }
    channel_rates = {name: mean_of(idxs) for name, idxs in ch.items()}
    return move_rates, channel_rates


def board_drives(
    *,
    danger: float,
    safety: float,
    hunger: float,
    food_near: float,
    courtship: float,
    size_advantage: float,
    panic: bool,
    visual_open: float,
) -> Dict[str, float]:
    """
    Fixed MaleCNS channel mapping from Battlesnake signals.
    Intensities are Hz-ish drives; dials scale them in simulate().
    """
    threat = max(0.0, 1.0 - min(1.0, danger / 4.0))
    return {
        "ALPN": 40.0 + 180.0 * threat + (80.0 if panic else 0.0),
        "ORN": 20.0 + 160.0 * hunger + 100.0 * food_near,
        "MBON": 30.0 + 150.0 * safety,
        "DAN": 25.0 + 120.0 * food_near + 90.0 * max(0.0, size_advantage - 0.45),
        "Kenyon_Cell": 15.0 + 100.0 * (1.0 - safety) + (60.0 if panic else 0.0),
        "courtship_pC1": 140.0 * courtship * safety,
        "courtship_vPR6": 120.0 * courtship * safety,
        "courtship_TN1": 100.0 * courtship * safety,
        "visual_L1": 30.0 + 140.0 * visual_open,
        "visual_L2": 30.0 + 140.0 * visual_open,
        "visual_L3": 20.0 + 100.0 * visual_open,
    }


def blend_move_scores(
    heuristic: Dict[str, float],
    brain_rates: Dict[str, float],
    dials: Dict[str, Any],
) -> Dict[str, float]:
    """Mix heuristic scores with full-brain readout. blend∈[0,1], 1=pure brain."""
    blend = float(dials.get("brain_blend", 0.55))
    blend = max(0.0, min(1.0, blend))
    # Normalize brain rates to heuristic scale
    vals = np.array([brain_rates.get(m, 0.0) for m in ("up", "down", "left", "right")], dtype=float)
    if vals.max() > vals.min():
        vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    else:
        vals = np.zeros(4)
    scale = float(dials.get("brain_score_scale", 8.0))
    out = {}
    for i, m in enumerate(("up", "down", "left", "right")):
        h = float(heuristic.get(m, 0.0))
        b = float(vals[i]) * scale
        out[m] = (1.0 - blend) * h + blend * b
    return out
