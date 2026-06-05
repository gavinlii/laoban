"""
HistoryBeliefPVNet — the V5 policy/value network used by laoban.cards.

Extracted from train_ppo.py so the web backend doesn't pull in the
full training stack. Matches the architecture exactly so V5 checkpoints
load without any key renaming.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from game import RANKS

HISTORY_EVENT_DIM = 42
HISTORY_MAX_EVENTS = 48


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class HistoryBeliefPVNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, history_dim: int,
                 hidden: int = 256, hist_hidden: int = 160):
        super().__init__()
        self.hidden = hidden
        self.hist_hidden = hist_hidden

        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.history_in = nn.Sequential(nn.Linear(history_dim, hist_hidden), nn.ReLU())
        self.history_rnn = nn.GRU(hist_hidden, hist_hidden, batch_first=True)

        self.context_net = nn.Sequential(
            nn.Linear(hidden + hist_hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

        # Opponent-hand belief heads
        self.opp_rank_head  = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, len(RANKS)))
        self.opp_bomb_head   = nn.Linear(hidden, 1)
        self.opp_empty1_head = nn.Linear(hidden, 1)
        self.opp_empty2_head = nn.Linear(hidden, 1)
        self.opp_point_head  = nn.Linear(hidden, 1)

        self.belief_dim = len(RANKS) + 4
        self.belief_fuse = nn.Sequential(
            nn.Linear(hidden + self.belief_dim, hidden), nn.ReLU(),
        )

        self.action_net = nn.Sequential(
            nn.Linear(action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_net = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode_history(self, history_batch: torch.Tensor, history_lengths: torch.Tensor) -> torch.Tensor:
        if history_batch.size(1) == 0:
            return torch.zeros((history_batch.size(0), self.hist_hidden),
                               dtype=history_batch.dtype, device=history_batch.device)
        x = self.history_in(history_batch)
        safe_lengths = history_lengths.clamp(min=1).to(torch.int64).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, safe_lengths, batch_first=True, enforce_sorted=False)
        _, h = self.history_rnn(packed)
        hist = h[-1]
        zero_mask = (history_lengths == 0).unsqueeze(1)
        hist = hist.masked_fill(zero_mask, 0.0)
        return hist

    def compute_context(self, state_batch, history_batch, history_lengths):
        s   = self.state_net(state_batch)
        h   = self.encode_history(history_batch, history_lengths)
        ctx0 = self.context_net(torch.cat([s, h], dim=-1))
        aux = {
            "opp_rank":   self.opp_rank_head(ctx0),
            "opp_bomb":   self.opp_bomb_head(ctx0).squeeze(-1),
            "opp_empty1": self.opp_empty1_head(ctx0).squeeze(-1),
            "opp_empty2": self.opp_empty2_head(ctx0).squeeze(-1),
            "opp_points": self.opp_point_head(ctx0).squeeze(-1),
        }
        belief = torch.cat([
            torch.sigmoid(aux["opp_rank"]),
            torch.sigmoid(aux["opp_bomb"]).unsqueeze(-1),
            torch.sigmoid(aux["opp_empty1"]).unsqueeze(-1),
            torch.sigmoid(aux["opp_empty2"]).unsqueeze(-1),
            torch.sigmoid(aux["opp_points"]).unsqueeze(-1),
        ], dim=-1).detach()
        ctx = self.belief_fuse(torch.cat([ctx0, belief], dim=-1))
        return ctx, aux

    def score_actions(self, state_batch, history_batch, history_lengths, action_batch):
        ctx, aux = self.compute_context(state_batch, history_batch, history_lengths)
        a       = self.action_net(action_batch)
        ctx_exp = ctx.unsqueeze(1).expand(-1, action_batch.shape[1], -1)
        joint   = torch.cat([ctx_exp, a], dim=-1)
        logits  = self.policy_net(joint).squeeze(-1)
        values  = self.value_head(ctx).squeeze(-1)
        return logits, values, aux


# ---------------------------------------------------------------------------
# History encoding (mirrors train_ppo.encode_history_events exactly)
# ---------------------------------------------------------------------------

from game import POINT_VALUES, SMALL_JOKER, BIG_JOKER  # noqa: E402

_RANK_TO_IDX = {r: i for i, r in enumerate(RANKS)}


def encode_history_events(infoset, max_events: int = HISTORY_MAX_EVENTS) -> np.ndarray:
    """Encode the public_history deque into a (T, HISTORY_EVENT_DIM) float32 array."""
    history = list(infoset.get("public_history", []))[-max_events:]
    if not history:
        return np.zeros((0, HISTORY_EVENT_DIM), dtype=np.float32)

    out = np.zeros((len(history), HISTORY_EVENT_DIM), dtype=np.float32)
    for i, event in enumerate(history):
        offset = 0
        actor = event.get("actor", 0)
        out[i, offset] = float(actor); offset += 1                          # 1: who played

        is_pass = event.get("is_pass", False)
        out[i, offset] = float(is_pass); offset += 1                        # 1: was pass

        cards = event.get("cards", [])
        # rank one-hot (15)
        for c in cards:
            idx = _RANK_TO_IDX.get(getattr(c, "rank", c) if not isinstance(c, int) else c, -1)
            if idx >= 0:
                out[i, offset + idx] = 1.0
        offset += len(RANKS)                                                 # 15

        # point value of cards played
        pts = sum(POINT_VALUES.get(getattr(c, "rank", c) if not isinstance(c, int) else c, 0) for c in cards)
        out[i, offset] = pts / 20.0; offset += 1                            # 1

        # move type one-hot (6: none/single/pair/triple/straight/bomb)
        mtype = event.get("move_type", "")
        for j, t in enumerate(["single", "pair", "triple", "straight", "bomb"]):
            out[i, offset + j] = float(mtype == t)
        offset += 5                                                          # 5

        # strength (normalised)
        strength = event.get("strength", 0)
        out[i, offset] = min(float(strength) / 30.0, 1.0); offset += 1     # 1

        # pot before/after (normalised)
        out[i, offset]     = event.get("pot_before", 0) / 50.0; offset += 1  # 1
        out[i, offset]     = event.get("pot_after",  0) / 50.0; offset += 1  # 1

        # deck size at time of event
        out[i, offset] = event.get("deck_size", 0) / 54.0; offset += 1     # 1

        # number of cards played
        out[i, offset] = len(cards) / 5.0; offset += 1                     # 1

        # contains joker
        has_joker = any(
            (getattr(c, "rank", c) if not isinstance(c, int) else c) in (SMALL_JOKER, BIG_JOKER)
            for c in cards
        )
        out[i, offset] = float(has_joker); offset += 1                     # 1

        # pad remaining to HISTORY_EVENT_DIM (42)
        # offset should be 31 here; remaining 11 are zero-padded

    return out
