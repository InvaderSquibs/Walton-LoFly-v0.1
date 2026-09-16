"""
FS-Avatar cortex — short-term memory + budgeted multi-step planning.

Memory (per game_id):
  • last N rival head positions → inferred heading habits
  • scar cells where we nearly collided / took extreme danger
  • our recent moves (for soft anti-oscillation)

Planning:
  • Beam search over our moves, depth D, width W
  • Opponents advance on their last heading (or habit mode) — not full adversarial
  • Leaf eval = flood space after projected board (reuse server flood)
  • Hard wall-clock budget so /move stays well under Battlesnake timeouts

All knobs are dials so Fly99 can still mutate one gene at a time.
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict, deque
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

MOVES = ("up", "down", "left", "right")
DELTA = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}

FloodFn = Callable[[Dict[str, Any], Dict[str, int], set], int]
BlockedFn = Callable[[Dict[str, Any], Dict[str, Any]], set]


def _key(p: Dict[str, int]) -> Tuple[int, int]:
    return (int(p["x"]), int(p["y"]))


def _add(p: Dict[str, int], d: Tuple[int, int]) -> Dict[str, int]:
    return {"x": int(p["x"]) + d[0], "y": int(p["y"]) + d[1]}


def _in_bounds(board: Dict[str, Any], p: Dict[str, int]) -> bool:
    return 0 <= p["x"] < board["width"] and 0 <= p["y"] < board["height"]


def _heading(prev: Dict[str, int], cur: Dict[str, int]) -> Optional[str]:
    dx = int(cur["x"]) - int(prev["x"])
    dy = int(cur["y"]) - int(prev["y"])
    if dx == 1 and dy == 0:
        return "right"
    if dx == -1 and dy == 0:
        return "left"
    if dx == 0 and dy == 1:
        return "up"
    if dx == 0 and dy == -1:
        return "down"
    return None


class GameMemory:
    __slots__ = (
        "game_id",
        "turns",
        "rival_heads",
        "rival_headings",
        "rival_habits",
        "our_moves",
        "scars",
        "last_turn",
    )

    def __init__(self, game_id: str, memory_turns: int = 8) -> None:
        self.game_id = game_id
        self.turns = memory_turns
        self.rival_heads: Dict[str, Deque[Dict[str, int]]] = defaultdict(
            lambda: deque(maxlen=memory_turns)
        )
        self.rival_headings: Dict[str, Deque[str]] = defaultdict(
            lambda: deque(maxlen=memory_turns)
        )
        self.rival_habits: Dict[str, Counter] = defaultdict(Counter)
        self.our_moves: Deque[str] = deque(maxlen=memory_turns)
        self.scars: Dict[Tuple[int, int], float] = {}
        self.last_turn: Optional[int] = None


_GAMES: Dict[str, GameMemory] = {}


def clear(game_id: Optional[str] = None) -> None:
    if game_id is None:
        _GAMES.clear()
    else:
        _GAMES.pop(game_id, None)


def get_or_create(game_id: str, dials: Dict[str, Any]) -> GameMemory:
    n = int(dials.get("memory_turns", 8) or 8)
    mem = _GAMES.get(game_id)
    if mem is None or mem.turns != n:
        mem = GameMemory(game_id, memory_turns=max(2, n))
        _GAMES[game_id] = mem
    return mem


def remember(
    game_state: Dict[str, Any],
    dials: Dict[str, Any],
    last_move: Optional[str] = None,
    near_miss: bool = False,
) -> GameMemory:
    gid = str((game_state.get("game") or {}).get("id") or "anon")
    mem = get_or_create(gid, dials)
    turn = int(game_state.get("turn") or 0)
    board = game_state["board"]
    you = game_state["you"]
    you_id = you.get("id")

    for snake in board.get("snakes") or []:
        sid = str(snake.get("id") or "")
        if not sid or sid == you_id:
            continue
        head = snake.get("head") or (snake.get("body") or [{}])[0]
        if not head:
            continue
        hist = mem.rival_heads[sid]
        if hist:
            hdg = _heading(hist[-1], head)
            if hdg:
                mem.rival_headings[sid].append(hdg)
                mem.rival_habits[sid][hdg] += 1
        hist.append({"x": int(head["x"]), "y": int(head["y"])})

    if last_move:
        mem.our_moves.append(last_move)

    # Decay scars; stamp near-miss / head-adjacent cells
    decay = float(dials.get("memory_scar_decay", 0.85))
    for k in list(mem.scars.keys()):
        mem.scars[k] *= decay
        if mem.scars[k] < 0.05:
            del mem.scars[k]

    head = you.get("head") or you["body"][0]
    for snake in board.get("snakes") or []:
        if snake.get("id") == you_id:
            continue
        rh = snake.get("head") or (snake.get("body") or [None])[0]
        if not rh:
            continue
        if abs(int(rh["x"]) - int(head["x"])) + abs(int(rh["y"]) - int(head["y"])) <= 2:
            mem.scars[_key(rh)] = max(mem.scars.get(_key(rh), 0.0), 1.0)
            for m in MOVES:
                cell = _add(rh, DELTA[m])
                if _in_bounds(board, cell):
                    mem.scars[_key(cell)] = max(mem.scars.get(_key(cell), 0.0), 0.55)

    if near_miss:
        mem.scars[_key(head)] = max(mem.scars.get(_key(head), 0.0), 1.2)

    mem.last_turn = turn
    return mem


def predicted_rival_cells(
    mem: GameMemory,
    board: Dict[str, Any],
    you: Dict[str, Any],
    dials: Dict[str, Any],
) -> Set[Tuple[int, int]]:
    """Cells rivals are likely to occupy next turn (head step)."""
    out: Set[Tuple[int, int]] = set()
    you_id = you.get("id")
    for snake in board.get("snakes") or []:
        sid = str(snake.get("id") or "")
        if not sid or sid == you_id:
            continue
        head = snake.get("head") or (snake.get("body") or [None])[0]
        if not head:
            continue
        headings = list(mem.rival_headings.get(sid) or [])
        habit = mem.rival_habits.get(sid) or Counter()
        # Prefer last heading; else mode habit; else all 4 (cautious)
        cands: List[str] = []
        if headings:
            cands.append(headings[-1])
        if habit:
            cands.append(habit.most_common(1)[0][0])
        if not cands:
            cands = list(MOVES)
        # unique preserve order
        seen = set()
        for m in cands:
            if m in seen:
                continue
            seen.add(m)
            nxt = _add(head, DELTA[m])
            if _in_bounds(board, nxt):
                out.add(_key(nxt))
    return out


def scar_penalty(mem: GameMemory, cell: Dict[str, int], dials: Dict[str, Any]) -> float:
    w = float(dials.get("memory_scar_weight", 1.4))
    return w * float(mem.scars.get(_key(cell), 0.0))


def habit_threat(
    mem: GameMemory,
    cell: Dict[str, int],
    board: Dict[str, Any],
    you: Dict[str, Any],
    dials: Dict[str, Any],
) -> float:
    """Extra threat if cell matches a predicted rival head."""
    pred = predicted_rival_cells(mem, board, you, dials)
    if _key(cell) not in pred:
        return 0.0
    return float(dials.get("memory_predict_weight", 2.2))


def _body_after(you: Dict[str, Any], nxt: Dict[str, int], grow: bool) -> List[Dict[str, int]]:
    body = [dict(p) for p in (you.get("body") or [])]
    new_body = [dict(nxt)] + body
    if not grow and len(new_body) > 1:
        new_body = new_body[:-1]
    return new_body


def _advance_rivals_simple(
    board: Dict[str, Any],
    you_id: Any,
    mem: GameMemory,
) -> List[Dict[str, Any]]:
    """One-step greedy: each rival continues last heading if free, else stays."""
    snakes_out = []
    for snake in board.get("snakes") or []:
        if snake.get("id") == you_id:
            snakes_out.append(snake)
            continue
        sid = str(snake.get("id") or "")
        head = snake.get("head") or (snake.get("body") or [None])[0]
        body = [dict(p) for p in (snake.get("body") or [])]
        if not head or not body:
            snakes_out.append(snake)
            continue
        headings = list(mem.rival_headings.get(sid) or [])
        move = headings[-1] if headings else None
        nxt = _add(head, DELTA[move]) if move else None
        # occupied now
        occ = set()
        for s in board.get("snakes") or []:
            for p in s.get("body") or []:
                occ.add(_key(p))
        for h in board.get("hazards") or []:
            occ.add(_key(h))
        if nxt is None or not _in_bounds(board, nxt) or _key(nxt) in occ:
            snakes_out.append(snake)
            continue
        new_body = [dict(nxt)] + body[:-1]
        sn = dict(snake)
        sn["body"] = new_body
        sn["head"] = dict(nxt)
        sn["length"] = len(new_body)
        snakes_out.append(sn)
    return snakes_out


def plan_move_values(
    board: Dict[str, Any],
    you: Dict[str, Any],
    mem: GameMemory,
    dials: Dict[str, Any],
    flood: FloodFn,
    blocked_fn: BlockedFn,
    food_set: Optional[Set[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """
    Beam-search plan. Returns:
      values: {move: float}  — higher = better future space / survival
      meta: depth used, nodes, ms, best_line
    """
    if not bool(dials.get("plan_enabled", True)):
        return {"values": {}, "meta": {"enabled": False}}

    depth = max(1, int(dials.get("plan_depth", 3) or 3))
    beam = max(1, int(dials.get("plan_beam", 3) or 3))
    budget_ms = float(dials.get("plan_budget_ms", 45) or 45)
    t0 = time.perf_counter()
    food_set = food_set or {_key(f) for f in (board.get("food") or [])}
    you_id = you.get("id")
    neck = None
    body = you.get("body") or []
    if len(body) >= 2:
        neck = _heading(body[1], body[0])

    def timed_out() -> bool:
        return (time.perf_counter() - t0) * 1000.0 >= budget_ms

    def legal_moves(b: Dict[str, Any], y: Dict[str, Any]) -> List[str]:
        head = y.get("head") or y["body"][0]
        blk = blocked_fn(b, y)
        tip = _key(y["body"][-1]) if len(y.get("body") or []) > 1 else None
        n = None
        bod = y.get("body") or []
        if len(bod) >= 2:
            n = _heading(bod[1], bod[0])
        out = []
        for m in MOVES:
            if n and m == OPPOSITE.get(n):
                continue
            nxt = _add(head, DELTA[m])
            if not _in_bounds(b, nxt):
                continue
            k = _key(nxt)
            if k in blk and k != tip:
                continue
            out.append(m)
        return out

    def step(
        b: Dict[str, Any], y: Dict[str, Any], move: str
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], int]]:
        head = y.get("head") or y["body"][0]
        nxt = _add(head, DELTA[move])
        if not _in_bounds(b, nxt):
            return None
        grow = _key(nxt) in food_set
        new_body = _body_after(y, nxt, grow)
        you2 = {
            "id": y.get("id"),
            "body": new_body,
            "head": dict(nxt),
            "length": len(new_body),
            "health": y.get("health", 100),
        }
        snakes = []
        advanced = _advance_rivals_simple(b, you_id, mem)
        for s in advanced:
            snakes.append(you2 if s.get("id") == you_id else s)
        for s in snakes:
            if s.get("id") == you_id:
                continue
            rh = s.get("head") or (s.get("body") or [None])[0]
            if rh and _key(rh) == _key(nxt):
                their_len = s.get("length") or len(s.get("body") or [])
                if their_len >= len(new_body):
                    return None
        b2 = {
            "width": b["width"],
            "height": b["height"],
            "food": b.get("food") or [],
            "hazards": b.get("hazards") or [],
            "snakes": snakes,
        }
        blk = blocked_fn(b2, you2)
        space = flood(b2, nxt, blk)
        return you2, b2, space

    root_moves = legal_moves(board, you)

    # Root expansion — always keep a baseline score per root move
    root_base: Dict[str, float] = {}
    beams: List[Tuple[float, Dict[str, Any], Dict[str, Any], List[str]]] = []
    for m in root_moves:
        got = step(board, you, m)
        if got is None:
            continue
        y2, b2, space = got
        root_base[m] = float(space)
        beams.append((float(space), y2, b2, [m]))

    beams.sort(key=lambda t: t[0], reverse=True)
    beams = beams[:beam]
    nodes = len(beams)

    for _d in range(1, depth):
        if timed_out() or not beams:
            break
        nxt_beams: List[Tuple[float, Dict[str, Any], Dict[str, Any], List[str]]] = []
        for score, y, b, path in beams:
            if timed_out():
                break
            moves = legal_moves(b, y)
            if not moves:
                nxt_beams.append((score * 0.1, y, b, path))
                continue
            local: List[Tuple[float, Dict[str, Any], Dict[str, Any], List[str]]] = []
            for m in moves:
                if timed_out():
                    break
                got = step(b, y, m)
                nodes += 1
                if got is None:
                    continue
                y2, b2, space = got
                local.append((score * 0.55 + float(space), y2, b2, path + [m]))
            local.sort(key=lambda t: t[0], reverse=True)
            nxt_beams.extend(local[:beam])
        nxt_beams.sort(key=lambda t: t[0], reverse=True)
        beams = nxt_beams[: max(beam * 2, beam)]

    deep: Dict[str, float] = {}
    best_line: List[str] = []
    best_score = -1e18
    for score, _y, _b, path in beams:
        if not path:
            continue
        root = path[0]
        if score > deep.get(root, -1e9):
            deep[root] = score
        if score > best_score:
            best_score = score
            best_line = path

    # Prefer deep score when present; else 1-ply root flood
    values = dict(root_base)
    values.update(deep)

    # Normalize to ~0..1 relative (single survivor → 1.0 so plan_weight still fires)
    finite = [v for v in values.values() if v > -1e8]
    if len(finite) == 1:
        values = {m: (1.0 if v > -1e8 else 0.0) for m, v in values.items()}
    elif finite:
        lo, hi = min(finite), max(finite)
        span = max(1e-6, hi - lo)
        values = {m: ((v - lo) / span if v > -1e8 else 0.0) for m, v in values.items()}
    else:
        values = {}

    ms = (time.perf_counter() - t0) * 1000.0
    return {
        "values": values,
        "meta": {
            "enabled": True,
            "depth": depth,
            "beam": beam,
            "nodes": nodes,
            "ms": round(ms, 2),
            "best_line": best_line,
            "budget_ms": budget_ms,
        },
    }
