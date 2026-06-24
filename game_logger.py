"""
game_logger.py — logs every web-played game to disk in the exact JSON schema
that human_model.py consumes for behaviour cloning.

Each completed game becomes one runs/human_games/game_*.json file containing:
  - the full outcome (final points, bot margin, winner)
  - every decision (human and bot) with its encoded state / action features /
    history, so the file is a self-contained training record.

human_model.load_decisions() reads these with human_only=True, so only the
human's choices are used for BC; the bot decisions are kept for completeness
(full-game replay / future value training).

Storage notes:
  - GAME_LOG_DIR (env) overrides the output dir (default runs/human_games).
  - LOG_GAMES=0 (env) disables logging entirely.
  - On an ephemeral host (Render free tier) the disk is wiped on redeploy/
    restart, so pull data via the /api/games/export endpoint periodically or
    mount a persistent disk.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np

import encoder as enc
from model import encode_history_events


def _to_list(arr) -> list:
    """numpy -> json-serialisable nested list of python floats."""
    return np.asarray(arr, dtype=np.float32).tolist()


class GameLogger:
    def __init__(self, log_dir: Optional[str] = None, enabled: Optional[bool] = None):
        self.dir = Path(log_dir or os.getenv("GAME_LOG_DIR", "runs/human_games"))
        if enabled is None:
            enabled = os.getenv("LOG_GAMES", "1") != "0"
        self.enabled = enabled
        self._lock = threading.Lock()
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
        # Durable mirror (survives Render's ephemeral disk). No-op unless SUPABASE_* set.
        from supabase_store import SupabaseGameStore
        self.supabase = SupabaseGameStore()

    # -- per-decision encoding ------------------------------------------------
    def encode_decision(self, infoset, chosen_idx: int, is_human: bool) -> dict:
        legal = infoset["legal_actions"]
        state = enc.encode_state(infoset)
        action_feats = [enc.encode_move(a, infoset) for a in legal]
        history = encode_history_events(infoset)
        return {
            "is_human":    1 if is_human else 0,
            "actor_seat":  int(infoset["player_index"]),
            "chosen_idx":  int(chosen_idx),
            "n_legal":     len(legal),
            "deck_size":   int(infoset.get("deck_size", 0)),
            "state":       _to_list(state),
            "action_feats": [_to_list(af) for af in action_feats],
            "history":     _to_list(history) if len(history) else [],
        }

    # -- whole-game record ----------------------------------------------------
    def write_game(self, *, session_id: str, bot_first: bool,
                   human_seat: int, bot_seat: int,
                   human_points: int, bot_points: int,
                   decisions: List[dict]) -> Optional[str]:
        if not self.enabled:
            return None
        margin = int(bot_points - human_points)          # +ve => bot won by this much
        if bot_points > human_points:
            winner = "bot"
        elif human_points > bot_points:
            winner = "human"
        else:
            winner = "tie"

        record = {
            "schema":       "laoban_web_v1",
            "created":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id":   session_id,
            "bot_first":    bool(bot_first),
            "human_seat":   int(human_seat),
            "bot_seat":     int(bot_seat),
            "final_points": {"human": int(human_points), "bot": int(bot_points)},
            "bot_margin":   margin,
            "bot_won":      bot_points > human_points,
            "human_won":    human_points > bot_points,
            "winner":       winner,
            "n_decisions":  len(decisions),
            "decisions":    decisions,
        }

        fname = f"game_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}.json"
        path = self.dir / fname
        with self._lock:
            with open(path, "w") as f:
                json.dump(record, f)

        # Mirror to durable storage in a daemon thread so the HTTP POST never
        # blocks the API response (and a Supabase outage can't break gameplay).
        if self.supabase.enabled:
            threading.Thread(
                target=self.supabase.insert, args=(record,), daemon=True
            ).start()
        return str(path)

    # -- export helpers (for pulling data off an ephemeral host) --------------
    def list_games(self) -> List[dict]:
        if not self.dir.exists():
            return []
        out = []
        for gf in sorted(self.dir.glob("game_*.json")):
            try:
                with open(gf) as f:
                    rec = json.load(f)
                out.append({
                    "file":        gf.name,
                    "created":     rec.get("created"),
                    "winner":      rec.get("winner"),
                    "bot_margin":  rec.get("bot_margin"),
                    "n_decisions": rec.get("n_decisions"),
                })
            except Exception:
                continue
        return out

    def export_all(self) -> List[dict]:
        if not self.dir.exists():
            return []
        records = []
        for gf in sorted(self.dir.glob("game_*.json")):
            try:
                with open(gf) as f:
                    records.append(json.load(f))
            except Exception:
                continue
        return records
